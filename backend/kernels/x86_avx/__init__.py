"""
backend/kernels/x86_avx/__init__.py

x86 AVX 平台的 KernelSpec 集合。
对于尚未有 AVX 实现的算子，直接 fallback 到 generic 的 spec。

使用方式
--------
在 backend/kernels/__init__.py 中注册：
    from .x86_avx import registry as _x86_registry
    _REGISTRIES["x86_avx"] = _x86_registry
"""
from ..base import KernelSpec, KernelRegistry
from ..generic import registry as _generic_reg   # fallback 来源

registry = KernelRegistry(target_name="x86_avx")

_H_AVX    = '"ops/x86_avx/ops_avx.h"'
_CMAKE_AVX = '"ops/x86_avx/*.c"'
_H_GEN    = '"ops/generic/ops.h"'
_CMAKE_GEN = '"ops/generic/*.c"'

# ---- 已有 AVX 优化的算子 ----
registry.register(KernelSpec(
    op_name    = "conv",
    c_func     = "op_conv2d_avx",
    signature  = (
        "{inp}, {N},{C},{H},{W}, "
        "{weight}, {Co},{kH},{kW}, "
        "{bias}, {sH},{sW}, {pH},{pW}, "
        "{out}"
    ),
    headers    = [_H_AVX],
    cmake_glob = [_CMAKE_AVX],
    notes      = "runtime/ops/x86_avx/conv2d_avx.c — AVX2 inner loop",
))

registry.register(KernelSpec(
    op_name    = "relu",
    c_func     = "op_relu_avx",
    signature  = "{buf}, {size}",
    headers    = [_H_AVX],
    cmake_glob = [_CMAKE_AVX],
    notes      = "runtime/ops/x86_avx/activations_avx.c",
))

registry.register(KernelSpec(
    op_name    = "gemm",
    c_func     = "op_linear_avx",
    signature  = "{inp}, {weight}, {bias}, {batch}, {in_feat}, {out_feat}, {out}",
    headers    = [_H_AVX],
    cmake_glob = [_CMAKE_AVX],
    notes      = "runtime/ops/x86_avx/linear_avx.c",
))

# ---- 尚无 AVX 实现：fallback 到 generic ----
for _op in ("maxpool", "add", "reducemean", "softmax"):
    _spec = _generic_reg.get(_op)
    if _spec is not None:
        # 复制一份，但保留 generic 的头文件和 cmake_glob
        registry.register(KernelSpec(
            op_name    = _spec.op_name,
            c_func     = _spec.c_func,
            signature  = _spec.signature,
            headers    = [_H_GEN],
            cmake_glob = [_CMAKE_GEN],
            notes      = f"[fallback] {_spec.notes}",
        ))
