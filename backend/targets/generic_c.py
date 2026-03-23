"""
GenericCTarget - 通用 C 代码生成器（x64 兼容）。

与原版的区别
-----------
* emit_xxx 方法不再硬编码函数名/头文件字符串；
  改为从 KernelRegistry 查找对应的 KernelSpec，
  再调用 spec.render_call(**kwargs) 生成调用语句。
* get_includes() / emit_cmake() 直接从 registry 聚合，
  新增算子只需在 kernels/generic/__init__.py 中注册即可。
"""
from typing import List

from .base import BaseTarget
from ...frontend.graph_ir import Node
from ..kernels import get_kernel_registry


class GenericCTarget(BaseTarget):

    def __init__(self):
        self._reg = get_kernel_registry("generic")

    # ------------------------------------------------------------------
    # 平台标识 & 构建文件
    # ------------------------------------------------------------------

    @property
    def target_name(self) -> str:
        return "generic_c"

    def get_includes(self) -> List[str]:
        return [f"#include {h}" for h in self._reg.all_headers()]

    def emit_cmake(self, model_name: str) -> str:
        globs = self._reg.all_cmake_globs()
        glob_lines = "\n".join(
            f'file(GLOB OPS_SRCS_{i} "{g}")' for i, g in enumerate(globs)
        )
        all_vars = " ".join(f"${{OPS_SRCS_{i}}}" for i in range(len(globs)))
        return (
            f"cmake_minimum_required(VERSION 3.10)\n"
            f"project({model_name}_runtime C)\n"
            f"set(CMAKE_C_STANDARD 99)\n"
            f"{glob_lines}\n"
            f"add_library({model_name} STATIC {model_name}.c {all_vars})\n"
            f"add_executable({model_name}_test resnet18_test.c)\n"
            f"target_link_libraries({model_name}_test {model_name} m)\n"
        )

    # ------------------------------------------------------------------
    # 算子代码生成（通过 KernelSpec.render_call）
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

        spec = self._reg.require("conv")
        call = spec.render_call(
            inp=a.var(inp), N=N, C=C, H=H, W=W,
            weight=a.var(weight), Co=Co, kH=kH, kW=kW,
            bias=bias_arg, sH=sH, sW=sW, pH=pH, pW=pW,
            out=a.var(out),
        )
        return [
            f"    // Conv [{N},{C},{H},{W}]->[{N},{Co}] k={kH}x{kW} s={sH} p={pH}",
            call,
        ]

    def emit_relu(self, node: Node, alloc) -> List[str]:
        a      = alloc
        x, out = node.inputs[0], node.outputs[0]
        size   = a.size(x)
        lines: List[str] = []
        if a.var(x) != a.var(out):
            lines.append(
                f"    memcpy({a.var(out)}, {a.var(x)}, {size}*sizeof(float));"
            )
        spec = self._reg.require("relu")
        lines.append(spec.render_call(buf=a.var(out), size=size))
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

        spec = self._reg.require("maxpool")
        call = spec.render_call(
            inp=a.var(x), N=N, C=C, H=H, W=W,
            kH=kH, kW=kW, sH=sH, sW=sW, pH=pH, pW=pW,
            out=a.var(out),
        )
        return [
            f"    // MaxPool [{N},{C},{H},{W}] k={kH}x{kW} s={sH} p={pH}",
            call,
        ]

    def emit_add(self, node: Node, alloc) -> List[str]:
        a               = alloc
        a_in, b_in, out = node.inputs[0], node.inputs[1], node.outputs[0]
        size            = a.size(a_in)
        lines: List[str] = []
        if a.var(a_in) != a.var(out):
            lines.append(
                f"    memcpy({a.var(out)}, {a.var(a_in)}, {size}*sizeof(float));"
            )
        lines.append(f"    // Add {a.var(a_in)} + {a.var(b_in)}")
        spec = self._reg.require("add")
        lines.append(spec.render_call(out=a.var(out), b=a.var(b_in), size=size))
        return lines

    def emit_reducemean(self, node: Node, alloc) -> List[str]:
        a      = alloc
        x, out = node.inputs[0], node.outputs[0]
        s      = a.shape(x)
        N = s[0] if len(s) > 0 else 1
        C = s[1] if len(s) > 1 else 1
        H = s[2] if len(s) > 2 else 1
        W = s[3] if len(s) > 3 else 1

        spec = self._reg.require("reducemean")
        call = spec.render_call(
            inp=a.var(x), N=N, C=C, H=H, W=W, out=a.var(out),
        )
        return [
            f"    // ReduceMean -> GlobalAvgPool [{N},{C},{H},{W}]",
            call,
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

        spec_linear  = self._reg.require("gemm")
        spec_softmax = self._reg.require("softmax")
        return [
            f"    // Gemm [{batch},{in_feat}]->[{batch},{out_feat}]",
            spec_linear.render_call(
                inp=a.var(x), weight=a.var(weight), bias=bias_arg,
                batch=batch, in_feat=in_feat, out_feat=out_feat,
                out=a.var(out),
            ),
            spec_softmax.render_call(buf=a.var(out), size=out_feat),
        ]

    def emit_softmax(self, node: Node, alloc) -> List[str]:
        a      = alloc
        x, out = node.inputs[0], node.outputs[0]
        size   = a.size(x)
        lines: List[str] = []
        if a.var(x) != a.var(out):
            lines.append(
                f"    memcpy({a.var(out)}, {a.var(x)}, {size}*sizeof(float));"
            )
        spec = self._reg.require("softmax")
        lines.append(spec.render_call(buf=a.var(out), size=size))
        return lines
