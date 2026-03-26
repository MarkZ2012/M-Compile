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
        groups   = node.attrs.get("group", node.attrs.get("groups", 1))
        sH, sW   = strides[0], strides[1]
        pH       = pads[0]
        pW       = pads[2] if len(pads) > 2 else pads[0]
        bias_arg = a.var(bias) if bias else "NULL"

        spec = self._reg.require("conv")
        call = spec.render_call(
            inp=a.var(inp), N=N, C=C, H=H, W=W,
            weight=a.var(weight), Co=Co, kH=kH, kW=kW,
            bias=bias_arg, sH=sH, sW=sW, pH=pH, pW=pW,
            groups=groups,
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

    def emit_clip(self, node: Node, alloc) -> List[str]:
        a = alloc
        x, min_v, max_v = node.inputs[0], node.inputs[1], node.inputs[2]
        out = node.outputs[0]
        size = a.size(x)
        spec = self._reg.require("clip")
        return [
            f"    // Clip {a.var(x)} -> {a.var(out)}",
            spec.render_call(x=a.var(x), min_v=a.var(min_v), max_v=a.var(max_v), size=size, out=a.var(out)),
        ]

    def emit_concat(self, node: Node, alloc) -> List[str]:
        """
        Concat (NCHW, rank4) - 支持 axis=1 的通道拼接。
        其它情况退化为 TODO。
        """
        a = alloc
        axis = node.attrs.get("axis", 1)
        x_out = node.outputs[0]
        in_names = node.inputs
        out_shape = a.shape(x_out)
        if len(out_shape) != 4 or axis != 1:
            return [f"    /* TODO: concat unsupported axis={axis} shape={out_shape} */"]
        N, outC, H, W = out_shape
        lines: List[str] = [f"    // Concat(axis={axis}) -> {a.var(x_out)}"]
        lines.append(f"    const int HW = {H} * {W};")

        chan_offset = 0
        for idx, x_in in enumerate(in_names):
            s = a.shape(x_in)
            if len(s) != 4:
                return [f"    /* TODO: concat input rank!=4 for {x_in} */"]
            Ci = s[1]
            in_ptr = a.var(x_in)
            out_ptr = a.var(x_out)
            # 每个输入在输出中的通道段：[chan_offset, chan_offset+Ci)
            lines.append(f"    for (int n = 0; n < {N}; n++) {{")
            lines.append(
                f"        memcpy({out_ptr} + (n*{outC} + {chan_offset})*HW, "
                f"{in_ptr} + (n*{Ci})*HW, {Ci}*HW*sizeof(float));"
            )
            lines.append("    }")
            chan_offset += Ci

        return lines

    def emit_transpose(self, node: Node, alloc) -> List[str]:
        """
        Transpose (rank4) - 依据 perm 做 NCHW 四维转置。
        """
        a = alloc
        x = node.inputs[0]
        out = node.outputs[0]
        x_shape = a.shape(x)
        out_shape = a.shape(out)
        perm = node.attrs.get("perm")
        if len(x_shape) != 4 or len(out_shape) != 4 or not perm or len(perm) != 4:
            return [f"    /* TODO: transpose unsupported perm={perm} */"]
        if perm == [0, 1, 2, 3]:
            size = a.size(x)
            if a.var(x) != a.var(out):
                return [f"    memcpy({a.var(out)}, {a.var(x)}, {size}*sizeof(float));"]
            return ["    // Transpose (identity, no-op)"]

        inN, inC, inH, inW = x_shape
        outN, outC, outH, outW = out_shape

        # inverse perm：inv[i] = output axis index for input axis i
        inv = [perm.index(i) for i in range(4)]

        def ovar(axis_idx: int) -> str:
            return f"o{axis_idx}"

        out_ptr = a.var(out)
        in_ptr = a.var(x)
        lines: List[str] = [f"    // Transpose perm={perm}"]
        lines.append(f"    for (int o0 = 0; o0 < {outN}; o0++) {{")
        lines.append(f"        for (int o1 = 0; o1 < {outC}; o1++) {{")
        lines.append(f"            for (int o2 = 0; o2 < {outH}; o2++) {{")
        lines.append(f"                for (int o3 = 0; o3 < {outW}; o3++) {{")

        # Map output indices back to input indices
        lines.append(f"                    int i0 = {ovar(inv[0])};")
        lines.append(f"                    int i1 = {ovar(inv[1])};")
        lines.append(f"                    int i2 = {ovar(inv[2])};")
        lines.append(f"                    int i3 = {ovar(inv[3])};")

        lines.append(
            f"                    {out_ptr}[((o0*{outC} + o1)*{outH} + o2)*{outW} + o3] = "
            f"{in_ptr}[((i0*{inC} + i1)*{inH} + i2)*{inW} + i3];"
        )

        lines.append("                }")
        lines.append("            }")
        lines.append("        }")
        lines.append("    }")
        return lines

    def emit_globalaveragepool(self, node: Node, alloc) -> List[str]:
        """
        GlobalAveragePool：依据 input 的 [N,C,H,W] 直接复用 ReduceMean 的实现内核。
        """
        a = alloc
        x, out = node.inputs[0], node.outputs[0]
        s = a.shape(x)
        N = s[0] if len(s) > 0 else 1
        C = s[1] if len(s) > 1 else 1
        H = s[2] if len(s) > 2 else 1
        W = s[3] if len(s) > 3 else 1
        spec = self._reg.require("reducemean")
        call = spec.render_call(inp=a.var(x), N=N, C=C, H=H, W=W, out=a.var(out))
        return [
            f"    // GlobalAveragePool [{N},{C},{H},{W}] -> {a.var(out)}",
            call,
        ]

    def emit_pad(self, node: Node, alloc) -> List[str]:
        """
        Pad (NCHW, rank4) - 仅支持常量 padding value（默认 0）。
        """
        a = alloc
        x = node.inputs[0]
        out = node.outputs[0]
        x_shape = a.shape(x)
        out_shape = a.shape(out)
        if len(x_shape) != 4 or len(out_shape) != 4:
            return [f"    /* TODO: pad unsupported */"]

        N, C, H, W = x_shape
        _, _, H_out, W_out = out_shape
        pads = node.attrs.get("pads", [0, 0, 0, 0])
        value = float(node.attrs.get("value", 0.0))
        if len(pads) == 8:
            pt, pl, pb, pr = pads[2], pads[3], pads[6], pads[7]
        elif len(pads) == 4:
            pt, pl, pb, pr = pads
        else:
            pt = pl = pb = pr = 0

        in_ptr = a.var(x)
        out_ptr = a.var(out)
        pad_val = f"{value:.8f}f"

        lines: List[str] = [f"    // Pad [{N},{C},{H},{W}] -> [{N},{C},{H_out},{W_out}]"]
        lines.append(f"    for (int n = 0; n < {N}; n++) {{")
        lines.append(f"        for (int c = 0; c < {C}; c++) {{")
        lines.append(f"            for (int oh = 0; oh < {H_out}; oh++) {{")
        lines.append(f"                for (int ow = 0; ow < {W_out}; ow++) {{")
        lines.append(f"                    int ih = oh - {pt};")
        lines.append(f"                    int iw = ow - {pl};")
        lines.append(f"                    float v = {pad_val};")
        lines.append(f"                    if (ih >= 0 && ih < {H} && iw >= 0 && iw < {W}) {{")
        lines.append(
            f"                        v = {in_ptr}[((n*{C} + c)*{H} + ih)*{W} + iw];"
        )
        lines.append("                    }")
        lines.append(
            f"                    {out_ptr}[((n*{C} + c)*{H_out} + oh)*{W_out} + ow] = v;"
        )
        lines.append("                }")
        lines.append("            }")
        lines.append("        }")
        lines.append("    }")
        return lines

    def emit_upsample(self, node: Node, alloc) -> List[str]:
        """
        Upsample (NCHW, rank4) - 最近邻插值。
        """
        a = alloc
        x = node.inputs[0]
        out = node.outputs[0]
        x_shape = a.shape(x)
        out_shape = a.shape(out)
        if len(x_shape) != 4 or len(out_shape) != 4:
            return [f"    /* TODO: upsample unsupported */"]

        N, C, H, W = x_shape
        _, _, H_out, W_out = out_shape
        in_ptr = a.var(x)
        out_ptr = a.var(out)

        lines: List[str] = [f"    // Upsample nearest [{N},{C},{H},{W}] -> [{N},{C},{H_out},{W_out}]"]
        lines.append(f"    for (int n = 0; n < {N}; n++) {{")
        lines.append(f"        for (int c = 0; c < {C}; c++) {{")
        lines.append(f"            for (int oh = 0; oh < {H_out}; oh++) {{")
        lines.append(f"                int ih = (oh * {H}) / {H_out};")
        lines.append(f"                for (int ow = 0; ow < {W_out}; ow++) {{")
        lines.append(f"                    int iw = (ow * {W}) / {W_out};")
        lines.append(
            f"                    {out_ptr}[((n*{C} + c)*{H_out} + oh)*{W_out} + ow] = "
            f"{in_ptr}[((n*{C} + c)*{H} + ih)*{W} + iw];"
        )
        lines.append("                }")
        lines.append("            }")
        lines.append("        }")
        lines.append("    }")
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

    # ========================================================================
    # 量化算子代码生成 (Quantized Operations)
    # ========================================================================

    def emit_quant_conv(self, node: Node, alloc) -> List[str]:
        """生成量化卷积代码"""
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

        # 获取量化参数
        input_scale = node.attrs.get("input_scale", 1.0)
        input_zp = node.attrs.get("input_zp", 0)
        weight_scale = node.attrs.get("weight_scale", 1.0)
        weight_zp = node.attrs.get("weight_zp", 0)
        output_scale = node.attrs.get("output_scale", 1.0)
        output_zp = node.attrs.get("output_zp", 0)

        spec = self._reg.require("quant_conv")
        call = spec.render_call(
            inp=a.var(inp), N=N, C=C, H=H, W=W,
            weight=a.var(weight), Co=Co, kH=kH, kW=kW,
            bias=bias_arg, sH=sH, sW=sW, pH=pH, pW=pW,
            out=a.var(out),
            input_scale=input_scale, input_zp=input_zp,
            weight_scale=weight_scale, weight_zp=weight_zp,
            output_scale=output_scale, output_zp=output_zp,
        )
        return [
            f"    // Quantized Conv [{N},{C},{H},{W}]->[{N},{Co}] k={kH}x{kW} s={sH} p={pH}",
            f"    // Quant params: in_scale={input_scale}, w_scale={weight_scale}, out_scale={output_scale}",
            call,
        ]

    def emit_quant_conv_relu(self, node: Node, alloc) -> List[str]:
        """生成量化卷积+ReLU代码"""
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

        # 获取量化参数
        input_scale = node.attrs.get("input_scale", 1.0)
        input_zp = node.attrs.get("input_zp", 0)
        weight_scale = node.attrs.get("weight_scale", 1.0)
        weight_zp = node.attrs.get("weight_zp", 0)
        output_scale = node.attrs.get("output_scale", 1.0)
        output_zp = node.attrs.get("output_zp", 0)

        spec = self._reg.require("quant_conv_relu")
        call = spec.render_call(
            inp=a.var(inp), N=N, C=C, H=H, W=W,
            weight=a.var(weight), Co=Co, kH=kH, kW=kW,
            bias=bias_arg, sH=sH, sW=sW, pH=pH, pW=pW,
            out=a.var(out),
            input_scale=input_scale, input_zp=input_zp,
            weight_scale=weight_scale, weight_zp=weight_zp,
            output_scale=output_scale, output_zp=output_zp,
        )
        return [
            f"    // Quantized Conv+ReLU [{N},{C},{H},{W}]->[{N},{Co}] k={kH}x{kW} s={sH} p={pH}",
            f"    // Quant params: in_scale={input_scale}, w_scale={weight_scale}, out_scale={output_scale}",
            call,
        ]

    def emit_quant_gemm(self, node: Node, alloc) -> List[str]:
        """生成量化全连接层代码"""
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

        # 获取量化参数
        input_scale = node.attrs.get("input_scale", 1.0)
        input_zp = node.attrs.get("input_zp", 0)
        weight_scale = node.attrs.get("weight_scale", 1.0)
        weight_zp = node.attrs.get("weight_zp", 0)
        output_scale = node.attrs.get("output_scale", 1.0)
        output_zp = node.attrs.get("output_zp", 0)

        spec = self._reg.require("quant_gemm")
        spec_softmax = self._reg.require("softmax")
        call = spec.render_call(
            inp=a.var(x), weight=a.var(weight), bias=bias_arg,
            batch=batch, in_feat=in_feat, out_feat=out_feat,
            out=a.var(out),
            input_scale=input_scale, input_zp=input_zp,
            weight_scale=weight_scale, weight_zp=weight_zp,
            output_scale=output_scale, output_zp=output_zp,
        )
        return [
            f"    // Quantized Gemm [{batch},{in_feat}]->[{batch},{out_feat}]",
            f"    // Quant params: in_scale={input_scale}, w_scale={weight_scale}, out_scale={output_scale}",
            call,
            spec_softmax.render_call(buf=a.var(out), size=out_feat),
        ]

    def emit_quant_gemm_relu(self, node: Node, alloc) -> List[str]:
        """生成量化全连接层+ReLU代码"""
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

        # 获取量化参数
        input_scale = node.attrs.get("input_scale", 1.0)
        input_zp = node.attrs.get("input_zp", 0)
        weight_scale = node.attrs.get("weight_scale", 1.0)
        weight_zp = node.attrs.get("weight_zp", 0)
        output_scale = node.attrs.get("output_scale", 1.0)
        output_zp = node.attrs.get("output_zp", 0)

        spec = self._reg.require("quant_gemm_relu")
        call = spec.render_call(
            inp=a.var(x), weight=a.var(weight), bias=bias_arg,
            batch=batch, in_feat=in_feat, out_feat=out_feat,
            out=a.var(out),
            input_scale=input_scale, input_zp=input_zp,
            weight_scale=weight_scale, weight_zp=weight_zp,
            output_scale=output_scale, output_zp=output_zp,
        )
        return [
            f"    // Quantized Gemm+ReLU [{batch},{in_feat}]->[{batch},{out_feat}]",
            f"    // Quant params: in_scale={input_scale}, w_scale={weight_scale}, out_scale={output_scale}",
            call,
        ]
