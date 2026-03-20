"""
GenericCTarget - 通用 C 代码生成器（x64 兼容）。

实现 BaseTarget 定义的全部抽象算子接口，调用 generic/ops.h 中的
op_conv2d / op_relu / op_maxpool / op_add / op_avgpool_global /
op_linear / op_softmax 等运行时函数。
"""
from typing import List
from .base import BaseTarget
from ...frontend.graph_ir import Node


class GenericCTarget(BaseTarget):

    # ------------------------------------------------------------------
    # 平台标识 & 构建文件
    # ------------------------------------------------------------------

    @property
    def target_name(self) -> str:
        return "generic_c"

    def get_includes(self) -> List[str]:
        return ['#include "ops/generic/ops.h"']

    def emit_cmake(self, model_name: str) -> str:
        return (
            f"cmake_minimum_required(VERSION 3.10)\n"
            f"project({model_name}_runtime C)\n"
            f"set(CMAKE_C_STANDARD 99)\n"
            f"file(GLOB OPS_SRCS \"ops/generic/*.c\")\n"
            f"add_library({model_name} STATIC {model_name}.c ${{OPS_SRCS}})\n"
            f"add_executable({model_name}_test resnet18_test.c)\n"
            f"target_link_libraries({model_name}_test {model_name} m)\n"
        )

    # ------------------------------------------------------------------
    # 算子代码生成
    # ------------------------------------------------------------------

    def emit_conv(self, node: Node, alloc) -> List[str]:
        a      = alloc
        inp    = node.inputs[0]
        weight = node.inputs[1]
        bias   = node.inputs[2] if len(node.inputs) > 2 else None
        out    = node.outputs[0]

        s, ws = a.shape(inp), a.shape(weight)
        N  = s[0]  if len(s)  > 0 else 1
        C  = s[1]  if len(s)  > 1 else 1
        H  = s[2]  if len(s)  > 2 else 1
        W  = s[3]  if len(s)  > 3 else 1
        Co = ws[0] if len(ws) > 0 else 1
        kH = ws[2] if len(ws) > 2 else 1
        kW = ws[3] if len(ws) > 3 else 1

        strides  = node.attrs.get("strides", [1, 1])
        pads     = node.attrs.get("pads",    [0, 0, 0, 0])
        sH, sW   = strides[0], strides[1]
        pH       = pads[0]
        pW       = pads[2] if len(pads) > 2 else pads[0]
        bias_arg = a.var(bias) if bias else "NULL"

        return [
            f"    // Conv [{N},{C},{H},{W}]->[{N},{Co}] k={kH}x{kW} s={sH} p={pH}",
            f"    op_conv2d({a.var(inp)}, {N},{C},{H},{W},",
            f"              {a.var(weight)}, {Co},{kH},{kW},",
            f"              {bias_arg}, {sH},{sW}, {pH},{pW},",
            f"              {a.var(out)});",
        ]

    def emit_relu(self, node: Node, alloc) -> List[str]:
        a        = alloc
        x, out   = node.inputs[0], node.outputs[0]
        size     = a.size(x)
        lines: List[str] = []
        if a.var(x) != a.var(out):
            lines.append(
                f"    memcpy({a.var(out)}, {a.var(x)}, {size}*sizeof(float));"
            )
        lines.append(f"    op_relu({a.var(out)}, {size});")
        return lines

    def emit_maxpool(self, node: Node, alloc) -> List[str]:
        a      = alloc
        x, out = node.inputs[0], node.outputs[0]
        s      = a.shape(x)
        N = s[0] if len(s) > 0 else 1
        C = s[1] if len(s) > 1 else 1
        H = s[2] if len(s) > 2 else 1
        W = s[3] if len(s) > 3 else 1

        kernel  = node.attrs.get("kernel_shape", [3, 3])
        strides = node.attrs.get("strides",      [1, 1])
        pads    = node.attrs.get("pads",          [0, 0, 0, 0])
        kH, kW  = kernel[0], kernel[1]
        sH, sW  = strides[0], strides[1]
        pH      = pads[0]
        pW      = pads[2] if len(pads) > 2 else pads[0]

        return [
            f"    // MaxPool [{N},{C},{H},{W}] k={kH}x{kW} s={sH} p={pH}",
            f"    op_maxpool({a.var(x)}, {N},{C},{H},{W},",
            f"               {kH},{kW}, {sH},{sW}, {pH},{pW},",
            f"               {a.var(out)});",
        ]

    def emit_add(self, node: Node, alloc) -> List[str]:
        a              = alloc
        a_in, b_in, out = node.inputs[0], node.inputs[1], node.outputs[0]
        size           = a.size(a_in)
        lines: List[str] = []
        if a.var(a_in) != a.var(out):
            lines.append(
                f"    memcpy({a.var(out)}, {a.var(a_in)}, {size}*sizeof(float));"
            )
        lines.append(f"    // Add {a.var(a_in)} + {a.var(b_in)}")
        lines.append(f"    op_add({a.var(out)}, {a.var(b_in)}, {size});")
        return lines

    def emit_reducemean(self, node: Node, alloc) -> List[str]:
        # axes=[-1,-2] 等价于 GlobalAvgPool，输出 [N,C,1,1]
        a      = alloc
        x, out = node.inputs[0], node.outputs[0]
        s      = a.shape(x)
        N = s[0] if len(s) > 0 else 1
        C = s[1] if len(s) > 1 else 1
        H = s[2] if len(s) > 2 else 1
        W = s[3] if len(s) > 3 else 1
        return [
            f"    // ReduceMean -> GlobalAvgPool [{N},{C},{H},{W}]",
            f"    op_avgpool_global({a.var(x)}, {N},{C},{H},{W}, {a.var(out)});",
        ]

    def emit_gemm(self, node: Node, alloc) -> List[str]:
        a      = alloc
        x      = node.inputs[0]
        weight = node.inputs[1]
        bias   = node.inputs[2] if len(node.inputs) > 2 else None
        out    = node.outputs[0]

        xs, ws   = a.shape(x), a.shape(weight)
        batch    = xs[0] if len(xs) > 0 else 1
        in_feat  = xs[1] if len(xs) > 1 else (ws[1] if len(ws) > 1 else 512)
        out_feat = ws[0] if len(ws) > 0 else 1000
        bias_arg = a.var(bias) if bias else "NULL"

        return [
            f"    // Gemm [{batch},{in_feat}]->[{batch},{out_feat}]",
            f"    op_linear({a.var(x)}, {a.var(weight)}, {bias_arg},",
            f"              {batch}, {in_feat}, {out_feat}, {a.var(out)});",
            f"    op_softmax({a.var(out)}, {out_feat});",
        ]

    def emit_softmax(self, node: Node, alloc) -> List[str]:
        a        = alloc
        x, out   = node.inputs[0], node.outputs[0]
        size     = a.size(x)
        lines: List[str] = []
        if a.var(x) != a.var(out):
            lines.append(
                f"    memcpy({a.var(out)}, {a.var(x)}, {size}*sizeof(float));"
            )
        lines.append(f"    op_softmax({a.var(out)}, {size});")
        return lines
