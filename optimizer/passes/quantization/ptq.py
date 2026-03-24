"""
Post-Training Quantization (PTQ) Pass

修复说明
--------
原实现硬编码 input_scale=1.0 / output_scale=1.0，导致激活量化精度只有约 2bit。
本版本通过 ActivationCalibrator 收集真实激活统计，计算正确的量化参数。

与 PassManager 的集成方式
-------------------------
PassManager 的 Pass.run(graph) 只接受一个参数，无法直接传 QuantConfig。
采用模块级变量注入：compile.py 在 pass_manager.run() 之前调用

    from my_ai_compiler.optimizer.passes.quantization.ptq import set_quant_config
    set_quant_config(cfg)

post_training_quantize(graph) 执行时自动读取该配置，无需修改 PassManager。
"""
import numpy as np
from typing import Optional, Dict, Tuple

from ....frontend.graph_ir import Graph, Node
from .quant_config import QuantConfig
from .quantizer import Quantizer
from .activation_calibrator import (
    ActivationCalibrator,
    get_tensors_to_calibrate,
    compute_scale_from_stats,
    ActivationStats,
)


# ── 模块级 QuantConfig（由 compile.py 在运行 pass 前注入）────────────
_MODULE_QUANT_CONFIG: Optional[QuantConfig] = None


def set_quant_config(config: QuantConfig) -> None:
    """
    在 PassManager 执行 PTQ pass 之前，由 compile.py 调用此函数注入配置。

    示例（compile.py 中）：
        from my_ai_compiler.optimizer.passes.quantization.ptq import set_quant_config
        set_quant_config(QuantConfig(calib_onnx_path=..., calib_inputs=[...]))
        pass_manager.add_pass_by_name("post_training_quantize")
        graph = pass_manager.run(graph)
    """
    global _MODULE_QUANT_CONFIG
    _MODULE_QUANT_CONFIG = config
    mode = "weight-only" if config.weight_only else "full PTQ"
    n = len(config.calib_inputs) if config.calib_inputs else 0
    print(f"  [PTQ] Config set: mode={mode}, calib_samples={n}, "
          f"symmetric={config.symmetric}, per_channel={config.per_channel}")


def get_quant_config() -> QuantConfig:
    """返回当前注入的 QuantConfig，未注入时返回默认配置。"""
    if _MODULE_QUANT_CONFIG is not None:
        return _MODULE_QUANT_CONFIG
    return QuantConfig()


# ── 可量化的算子类型 ────────────────────────────────────────────────────
_QUANTIZABLE_OPS = {
    "Conv", "ConvBn", "ConvBnRelu",
    "Gemm", "GemmRelu", "GemmBias", "Linear",
}

_OP_TYPE_MAP = {
    "Conv":       "quant_conv",
    "ConvBn":     "quant_conv",
    "ConvBnRelu": "quant_conv_relu",
    "Gemm":       "quant_gemm",
    "GemmBias":   "quant_gemm",
    "GemmRelu":   "quant_gemm_relu",
    "Linear":     "quant_gemm",
}


def post_training_quantize(graph: Graph, config: Optional[QuantConfig] = None) -> Graph:
    """
    PTQ pass 主入口。

    config 优先级：
        1. 显式传入的 config 参数
        2. set_quant_config() 注入的模块级变量
        3. 默认 QuantConfig()（自动退回 weight-only）
    """
    if config is None:
        config = get_quant_config()

    quantizer = Quantizer(config)

    # ------------------------------------------------------------------
    # Step 1: 激活校准
    # ------------------------------------------------------------------
    activation_stats: ActivationStats = {}
    use_calibration = (
        not config.weight_only
        and config.calib_inputs is not None
        and len(config.calib_inputs) > 0
        and config.calib_onnx_path is not None
    )

    if use_calibration:
        print("  [PTQ] Running activation calibration...")
        try:
            target_tensors = get_tensors_to_calibrate(graph.nodes)
            print(f"  [PTQ] Calibrating {len(target_tensors)} tensor(s) "
                  f"with {len(config.calib_inputs)} sample(s)...")
            calibrator = ActivationCalibrator(config.calib_onnx_path)
            activation_stats = calibrator.collect(
                config.calib_inputs,
                target_tensor_names=target_tensors,
                symmetric=config.symmetric,
            )
            print(f"  [PTQ] Calibration done: {len(activation_stats)} stats collected")
        except Exception as e:
            print(f"  [PTQ] WARNING: Calibration failed ({e}), falling back to weight-only")
            import traceback; traceback.print_exc()
            activation_stats = {}
    else:
        if not config.weight_only and config.calib_inputs is None:
            print("  [PTQ] WARNING: No calibration data provided. "
                  "Running weight-only quantization (input_scale=1.0). "
                  "Call set_quant_config() with calib_inputs for full PTQ.")

    # ------------------------------------------------------------------
    # Step 2: 量化权重 + 填写激活参数
    # ------------------------------------------------------------------
    quantized_count = 0

    for node in graph.nodes:
        if node.op_type not in _QUANTIZABLE_OPS:
            continue
        if not _op_is_allowed(node.op_type, config.quant_ops):
            continue
        if len(node.inputs) < 2:
            continue

        weight_name = node.inputs[1]
        weight_tensor = graph.tensors.get(weight_name)
        if weight_tensor is None or weight_tensor.data is None:
            continue
        if weight_tensor.data.dtype not in [np.float32, np.float64]:
            continue

        w = weight_tensor.data.astype(np.float32)
        _ensure_attrs(node)

        # 量化权重
        if config.per_channel and w.ndim >= 2:
            scales_arr, zps_arr = quantizer.compute_quant_params_per_channel(w, axis=0)
            w_q = quantizer.quantize_per_channel(w, scales_arr, zps_arr, axis=0)
            weight_tensor.data  = w_q
            weight_tensor.dtype = f"int{config.bit_width}"
            node.attrs.update({
                "quantized":          True,
                "per_channel":        True,
                "weight_scales":      scales_arr.tolist(),
                "weight_zero_points": zps_arr.tolist(),
                "weight_scale":       float(scales_arr[0]),
                "weight_zero_point":  0,
            })
        else:
            w_scale, w_zp = quantizer.compute_quant_params_per_tensor(w)
            w_q = quantizer.quantize(w, w_scale, w_zp)
            weight_tensor.data  = w_q
            weight_tensor.dtype = f"int{config.bit_width}"
            node.attrs.update({
                "quantized":         True,
                "per_channel":       False,
                "weight_scale":      float(w_scale),
                "weight_zero_point": int(w_zp),
            })

        node.attrs["bit_width"] = config.bit_width

        # 激活量化参数
        in_scale, in_zp   = _get_activation_params(node.inputs[0],  activation_stats, config)
        out_scale, out_zp = _get_activation_params(node.outputs[0], activation_stats, config)
        node.attrs.update({
            "input_scale":  in_scale,
            "input_zp":     in_zp,
            "output_scale": out_scale,
            "output_zp":    out_zp,
        })

        # 替换 op_type
        node.op_type = _OP_TYPE_MAP.get(node.op_type, "quant_conv")
        quantized_count += 1

    # ------------------------------------------------------------------
    # Step 3: 摘要
    # ------------------------------------------------------------------
    if quantized_count:
        mode_str  = "per-channel" if config.per_channel else "per-tensor"
        calib_str = "with activation calibration" if activation_stats else "weight-only"
        print(f"  [PTQ] Quantized {quantized_count} layer(s) to "
              f"int{config.bit_width} ({mode_str}, {calib_str})")
        if activation_stats:
            quant_nodes = [n for n in graph.nodes if n.attrs.get("quantized")][:3]
            print("  [PTQ] Sample layer scales (in / weight / out):")
            for n in quant_nodes:
                print(f"    {n.name:40s} "
                      f"in={n.attrs['input_scale']:.5f}  "
                      f"w={n.attrs.get('weight_scale', 0):.5f}  "
                      f"out={n.attrs['output_scale']:.5f}")
    else:
        print("  [PTQ] No quantizable weights found")

    return graph


# ── 工具函数 ────────────────────────────────────────────────────────────

def _op_is_allowed(op_type: str, quant_ops: list) -> bool:
    if op_type in quant_ops:
        return True
    for allowed in quant_ops:
        if op_type.startswith(allowed):
            return True
    return False


def _ensure_attrs(node: Node):
    if not hasattr(node, "attrs") or node.attrs is None:
        node.attrs = {}


def _get_activation_params(
    tensor_name: str,
    stats: ActivationStats,
    config: QuantConfig,
) -> Tuple[float, int]:
    if tensor_name in stats:
        mn, mx = stats[tensor_name]
        return compute_scale_from_stats(mn, mx, config.symmetric, config.bit_width)
    return 1.0, 0
