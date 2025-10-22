
from transformer_engine.common import recipe


class GetRecipes:
    @staticmethod
    def nvfp4_vanilla():
        nvfp4_recipe = recipe.NVFP4BlockScaling()
        nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams()
        nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams()
        nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams()
        return nvfp4_recipe

    @staticmethod
    def nvfp4_rht_only():
        nvfp4_recipe = recipe.NVFP4BlockScaling()
        nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams(random_hadamard_transform=True)
        nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(random_hadamard_transform=False)
        nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams(random_hadamard_transform=True)
        return nvfp4_recipe

    @staticmethod
    def nvfp4_2d_quantization_only():
        nvfp4_recipe = recipe.NVFP4BlockScaling()
        nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams(fp4_2d_quantization=False)
        nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(fp4_2d_quantization=True)
        nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams(fp4_2d_quantization=False)
        return nvfp4_recipe

    @staticmethod
    def nvfp4_rht_and_2d_quantization():
        nvfp4_recipe = recipe.NVFP4BlockScaling()
        nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams(
            random_hadamard_transform=True, fp4_2d_quantization=False
        )
        nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(
            random_hadamard_transform=False, fp4_2d_quantization=True
        )
        nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams(
            random_hadamard_transform=True, fp4_2d_quantization=False
        )
        return nvfp4_recipe

    @staticmethod
    def nvfp4_recipe(with_rht: bool = False, with_2d_quantization: bool = False):
        if with_rht and with_2d_quantization:
            return GetRecipes.nvfp4_rht_and_2d_quantization()
        elif with_rht:
            return GetRecipes.nvfp4_rht_only()
        elif with_2d_quantization:
            return GetRecipes.nvfp4_2d_quantization_only()
        else:
            return GetRecipes.nvfp4_vanilla()
