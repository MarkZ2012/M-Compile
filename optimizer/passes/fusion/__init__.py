# optimizer/passes/fusion/__init__.py
"""
Fusion passes 自动注册。

在 compile.py 中执行：
    from my_ai_compiler.optimizer.passes import fusion

即可将 conv_bn_relu_fusion / gemm_fusion 注册到全局注册表，
之后用 pm.add_pass_by_name("conv_bn_relu_fusion") 启用。
"""
from ...pass_manager import FunctionPass, register_pass
from .conv_bn_fusion import conv_bn_relu_fusion
from .gemm_fusion    import gemm_fusion

register_pass("conv_bn_relu_fusion", FunctionPass("conv_bn_relu_fusion", conv_bn_relu_fusion))
register_pass("gemm_fusion",         FunctionPass("gemm_fusion",         gemm_fusion))
