"""
X86AvxTarget - x86 AVX/AVX2 专属 C 代码生成器。

当前为占位实现：所有算子回退到 generic 实现并添加 AVX 注释。
后续可在各 emit_xxx 方法中替换为调用 avx/ops_avx.h 中的 SIMD 内联函数。
"""
from typing import List
from .generic_c import GenericCTarget
from ...frontend.graph_ir import Node


class X86AvxTarget(GenericCTarget):

    @property
    def target_name(self) -> str:
        return "x86_avx"

    def get_includes(self) -> List[str]:
        return [
            "#include <immintrin.h>",
            '#include "avx/ops_avx.h"',
        ]

    def emit_cmake(self, model_name: str) -> str:
        return (
            f"cmake_minimum_required(VERSION 3.10)\n"
            f"project({model_name}_runtime C)\n"
            f"set(CMAKE_C_STANDARD 99)\n"
            f"# Enable AVX2 — adjust to AVX / AVX-512 as needed\n"
            f"add_compile_options(-mavx2 -mfma -O3)\n"
            f"file(GLOB OPS_SRCS \"avx/*.c\")\n"
            f"add_library({model_name} STATIC {model_name}.c ${{OPS_SRCS}})\n"
            f"add_executable({model_name}_test test.c)\n"
            f"target_link_libraries({model_name}_test {model_name} m)\n"
        )

    # ------------------------------------------------------------------
    # TODO: 用 AVX intrinsics 替换下列方法
    # 例如：emit_conv 调用 op_conv2d_avx，emit_relu 使用 _mm256_max_ps
    # 目前继承 GenericCTarget 的实现作为正确的回退。
    # ------------------------------------------------------------------

    def emit_relu(self, node: Node, alloc) -> List[str]:
        a        = alloc
        x, out   = node.inputs[0], node.outputs[0]
        size     = a.size(x)
        lines: List[str] = []
        if a.var(x) != a.var(out):
            lines.append(
                f"    memcpy({a.var(out)}, {a.var(x)}, {size}*sizeof(float));"
            )
        # TODO: 替换为 op_relu_avx 以利用 AVX 向量化
        lines.append(f"    op_relu({a.var(out)}, {size});  /* AVX fallback */")
        return lines
