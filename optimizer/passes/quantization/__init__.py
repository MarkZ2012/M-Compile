# optimizer/passes/quantization/__init__.py
"""
Quantization passes 自动注册。

使用方式（compile.py）：
    from my_ai_compiler.optimizer.passes import quantization
    pm.add_pass_by_name("post_training_quantize")   # PTQ
    # 或
    pm.add_pass_by_name("qat_fold")                  # QAT（二选一）

注意：量化 pass 必须在 fusion passes 之后注册 / 执行。
"""
from ...pass_manager import FunctionPass, register_pass
from .ptq import post_training_quantize
from .qat import qat_fold

register_pass("post_training_quantize", FunctionPass("post_training_quantize", post_training_quantize))
register_pass("qat_fold",               FunctionPass("qat_fold",               qat_fold))
