import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from utils import Tokenized_data
from models import TransformerSeq, TransformerBlock
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from utils import parse
from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from torch.distributed.fsdp.api import ShardingStrategy, CPUOffload
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType, FullStateDictConfig


def setup(args):
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend='nccl')
        args.device = torch.device("cuda", args.local_rank)
        # torch.cuda.set_device(args.device)
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
    else:
        args.local_rank = 0
        args.rank = 0
        args.world_size = 1
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup():
    dist.destroy_process_group()


def get_model(args):
    model = TransformerSeq(args)
    model = model.to(args.device).bfloat16()

    if args.rank == 0:
        params = sum(p.numel() for p in model.parameters())
        print(f'Before FSDP wrapping, model has {params/1e6:.2f}M params')

    transformer_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            TransformerBlock,
        },
    )

    auto_wrap_policy = partial(
        size_based_auto_wrap_policy, min_num_params=100_000_000
    )

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.float32,
    )

    model = FSDP(
        model,
        auto_wrap_policy=transformer_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=args.local_rank,
        cpu_offload=CPUOffload(offload_params=False),
        # mixed_precision=bf16_policy,
        use_orig_params=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE
    )

    # transformer_layer_cls = ( TransformerBlock )
    # check_fn = lambda m: isinstance(m, transformer_layer_cls)
    # apply_activation_checkpointing(
    #     model, 
    #     checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT), 
    #     check_fn=check_fn
    # )

    if args.rank == 0:
        params = sum(p.numel() for p in model.parameters())
        print(f'After FSDP wrapping, each rank has {params/1e6:.2f}M params')

    return model

def load_dataset(args):
    dataset = Tokenized_data(args)
    sampler = DistributedSampler(
        dataset,
        rank=args.rank,
        num_replicas=args.world_size,
        shuffle=args.shuffle)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        num_workers=args.dataset_workers,
        pin_memory=True
    )
    if args.rank == 0:
        print(f'Data ok. Total samples: {len(dataset)}')
    return dataloader

def train(args):
    setup(args)

    writer = SummaryWriter(f"{args.log_dir}/{args.tag}") if args.rank == 0 else None
    
    dataloader = load_dataset(args)
    model = get_model(args)

    if args.load_from:
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            if args.rank == 0:
                state_dict = torch.load(args.load_from)
                model.load_state_dict(state_dict)
                print(f"Model loaded from {args.load_from}")
            dist.barrier()

    model.train()

    loss_fn = nn.CrossEntropyLoss(ignore_index = args.vocab_size - 1)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(args.adam_beta1, args.adam_beta2), 
        eps=1e-8, 
        weight_decay=args.weight_decay
    )  
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.lr_warmup_steps, args.train_steps)

    train_steps = 0
    acc_steps = 1
    acc_loss = 0
    
    for epoch in range(args.max_epochs):
        dataloader.sampler.set_epoch(epoch)
                
        for batch, (source, target, _) in enumerate(dataloader):
            
            source, target = source.to(args.device), target.to(args.device)
            
            if acc_steps == 1:
                optimizer.zero_grad()

            logit = model(source)
            loss = loss_fn(logit.view(-1, args.vocab_size), target.view(-1)) / args.grad_acc
            acc_loss += loss.item()
        
            if acc_steps == args.grad_acc:
                if args.rank == 0:
                    writer.add_scalar("train_loss", acc_loss, train_steps)
                
                # regularization
                rloss = torch.zeros_like(loss)
                if args.reg_lambda > 0:
                    for name, p in model.module.decoders.named_parameters():
                        if ("ulinear" in name or "vlinear" in name or (not ("ln" in name))) and "weight" in name:
                            rloss += (torch.sum(p ** 2) * args.reg_alpha1 + \
                                    torch.sum(((p + 1e-6) ** -2) * args.reg_alpha2)) * \
                                    (1 / p.shape[0] / p.shape[1] * args.reg_lambda) 
                loss += rloss
                loss.backward()
                
            else:
                loss.backward()
                acc_steps += 1
                continue
            

            if args.rank == 0:
                print(f"rank: {args.rank}, "
                    f"epoch: {epoch}, "
                    f"batch: {train_steps}, "
                    f"loss: {acc_loss:.3f}, "
                    f"r-loss: {rloss.item() + acc_loss:.3f}"
                    )
            
            model.clip_grad_norm_(args.grad_clipping)
            optimizer.step()
            lr_scheduler.step()
            
            torch.cuda.synchronize() 

            # --- Checkpointing ---
            if batch % args.save_steps == 0 and batch > 0:
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                    cpu_state_dict = model.state_dict()
                
                if args.rank == 0:
                    save_path = f"{args.chkpt_dir}/{args.tag}/{epoch}_{batch}.pth"
                    torch.save(cpu_state_dict, save_path)
                    print(f"Model checkpoint saved at {save_path}")
                
                dist.barrier()
            
            acc_loss = 0
            acc_steps = 1
            train_steps += 1
            if args.train_steps == batch + 1:
                break
            
    cleanup()


if __name__ == "__main__":
    args = parse()
    train(args)
