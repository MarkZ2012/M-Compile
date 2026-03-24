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
from .quant_config import QuantConfig, QuantMode, CalibrationMethod
from .quantizer import Quantizer, MinMaxCalibrator, MSECalibrator
from .quant_params_exporter import QuantParamsExporter, QuantParamsExport, LayerQuantParams

# 创建带配置的PTQ pass
def create_ptq_pass(config: QuantConfig = None):
    """创建带配置的PTQ pass"""
    def ptq_with_config(graph):
        return post_training_quantize(graph, config)
    return ptq_with_config

# 注册passes
register_pass("post_training_quantize", FunctionPass("post_training_quantize", post_training_quantize))
register_pass("qat_fold",               FunctionPass("qat_fold",               qat_fold))

# 导出符号
__all__ = [
    "QuantConfig",
    "QuantMode", 
    "CalibrationMethod",
    "Quantizer",
    "MinMaxCalibrator",
    "MSECalibrator",
    "QuantParamsExporter",
    "QuantParamsExport",
    "LayerQuantParams",
    "post_training_quantize",
    "create_ptq_pass",
]
