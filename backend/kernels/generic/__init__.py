"""
backend/kernels/generic/__init__.py

Generic C 平台的 KernelSpec 集合。
每个 KernelSpec 对应 runtime/ops/generic/ 下的一个或多个 .c 文件。

signature 占位符说明（和 generic_c.py 中的 emit_xxx 对应）：
  conv      : inp, N, C, H, W, weight, Co, kH, kW, bias, sH, sW, pH, pW, out
  relu      : buf, size
  maxpool   : inp, N, C, H, W, kH, kW, sH, sW, pH, pW, out
  add       : out, b, size
  reducemean: inp, N, C, H, W, out
  gemm      : inp, weight, bias, batch, in_feat, out_feat, out
  softmax   : buf, size
"""
from ..base import KernelSpec, KernelRegistry

registry = KernelRegistry(target_name="generic")

_HEADER   = '"ops/generic/ops.h"'
_CMAKE    = '"ops/generic/*.c"'

registry.register(KernelSpec(
    op_name    = "conv",
    c_func     = "op_conv2d",
    signature  = (
        "{inp}, {N},{C},{H},{W}, "
        "{weight}, {Co},{kH},{kW}, "
        "{bias}, {sH},{sW}, {pH},{pW}, "
        "{out}"
    ),
    headers    = [_HEADER],
    cmake_glob = [_CMAKE],
    notes      = "runtime/ops/generic/conv2d.c",
))

registry.register(KernelSpec(
    op_name    = "relu",
    c_func     = "op_relu",
    signature  = "{buf}, {size}",
    headers    = [_HEADER],
    cmake_glob = [_CMAKE],
    notes      = "runtime/ops/generic/activations.c",
))

registry.register(KernelSpec(
    op_name    = "maxpool",
    c_func     = "op_maxpool",
    signature  = (
        "{inp}, {N},{C},{H},{W}, "
        "{kH},{kW}, {sH},{sW}, {pH},{pW}, "
        "{out}"
    ),
    headers    = [_HEADER],
    cmake_glob = [_CMAKE],
    notes      = "runtime/ops/generic/pooling.c",
))

registry.register(KernelSpec(
    op_name    = "add",
    c_func     = "op_add",
    signature  = "{out}, {b}, {size}",
    headers    = [_HEADER],
    cmake_glob = [_CMAKE],
    notes      = "runtime/ops/generic/math_ops.c",
))

registry.register(KernelSpec(
    op_name    = "reducemean",
    c_func     = "op_avgpool_global",
    signature  = "{inp}, {N},{C},{H},{W}, {out}",
    headers    = [_HEADER],
    cmake_glob = [_CMAKE],
    notes      = "runtime/ops/generic/pooling.c — axes=[-1,-2] → GlobalAvgPool",
))

registry.register(KernelSpec(
    op_name    = "gemm",
    c_func     = "op_linear",
    signature  = "{inp}, {weight}, {bias}, {batch}, {in_feat}, {out_feat}, {out}",
    headers    = [_HEADER],
    cmake_glob = [_CMAKE],
    notes      = "runtime/ops/generic/linear.c",
))

registry.register(KernelSpec(
    op_name    = "softmax",
    c_func     = "op_softmax",
    signature  = "{buf}, {size}",
    headers    = [_HEADER],
    cmake_glob = [_CMAKE],
    notes      = "runtime/ops/generic/softmax.c",
))

# ============================================================================
# 量化算子 (Quantized Operations)
# ============================================================================

_QUANT_HEADER = '"ops/generic/quant_ops.h"'

registry.register(KernelSpec(
    op_name    = "quant_conv",
    c_func     = "op_quant_conv2d_int8",
    signature  = (
        "{inp}, {N},{C},{H},{W}, "
        "{weight}, {Co},{kH},{kW}, "
        "{bias}, {sH},{sW}, {pH},{pW}, "
        "{out}, "
        "{input_scale}, {input_zp}, "
        "{weight_scale}, {weight_zp}, "
        "{output_scale}, {output_zp}"
    ),
    headers    = [_QUANT_HEADER],
    cmake_glob = ['"ops/generic/quant_conv2d.c"'],
    notes      = "runtime/ops/generic/quant_conv2d.c — int8 quantized conv2d",
))

registry.register(KernelSpec(
    op_name    = "quant_conv_relu",
    c_func     = "op_quant_conv2d_relu_int8",
    signature  = (
        "{inp}, {N},{C},{H},{W}, "
        "{weight}, {Co},{kH},{kW}, "
        "{bias}, {sH},{sW}, {pH},{pW}, "
        "{out}, "
        "{input_scale}, {input_zp}, "
        "{weight_scale}, {weight_zp}, "
        "{output_scale}, {output_zp}"
    ),
    headers    = [_QUANT_HEADER],
    cmake_glob = ['"ops/generic/quant_conv2d.c"'],
    notes      = "runtime/ops/generic/quant_conv2d.c — int8 quantized conv2d + relu",
))

registry.register(KernelSpec(
    op_name    = "quant_gemm",
    c_func     = "op_quant_linear_int8",
    signature  = (
        "{inp}, {batch}, {in_feat}, "
        "{weight}, {out_feat}, "
        "{bias}, {out}, "
        "{input_scale}, {input_zp}, "
        "{weight_scale}, {weight_zp}, "
        "{output_scale}, {output_zp}"
    ),
    headers    = [_QUANT_HEADER],
    cmake_glob = ['"ops/generic/quant_linear.c"'],
    notes      = "runtime/ops/generic/quant_linear.c — int8 quantized linear/gemm",
))

registry.register(KernelSpec(
    op_name    = "quant_gemm_relu",
    c_func     = "op_quant_linear_relu_int8",
    signature  = (
        "{inp}, {batch}, {in_feat}, "
        "{weight}, {out_feat}, "
        "{bias}, {out}, "
        "{input_scale}, {input_zp}, "
        "{weight_scale}, {weight_zp}, "
        "{output_scale}, {output_zp}"
    ),
    headers    = [_QUANT_HEADER],
    cmake_glob = ['"ops/generic/quant_linear.c"'],
    notes      = "runtime/ops/generic/quant_linear.c — int8 quantized linear/gemm + relu",
))
