
LOCAL_BATCH_SIZE=12
GRAD_ACC=1
# total batchsize = LOCAL_BATCH_SIZE * GRAD_ACC * NPROC

TAG=llama3-8b_nvfp4_with_2d_quantization
PORT=12345

python pp_main.py \
    --chkpt-dir checkpoints/ \
    --dataset-path ./dataset/DCLM-cleaned \
    --log-dir logs/ \
    --tokenizer-path r50k_base.tiktoken \
    --tag $TAG \
    --reg-lambda 0 \
    --layers 32 \
    --embed-dim 4096 \
    --max-epochs 3 \
    --heads 32 \
    --n-kv-heads 32 \
    --grad-clipping 1.0 \
    --win-size 1024 \
    --batch-size $LOCAL_BATCH_SIZE \
    --weight-decay 0.1 \
    --lr 1.5e-4 \
    --merged-lr 1.5e-4 \
    --grad-acc $GRAD_ACC \
    --lr-warmup-steps 20 \
    --use-nvfp4 \
    --nvfp4-with-2d-quantization 

# --nvfp4-with-rht






    
    
    
    
