"""
Shape 推断器 - 通过节点拓扑顺序推断所有中间 tensor 的 shape。

支持的算子：Conv, Relu, Clip, Add, MaxPool, ReduceMean, Reshape,
           Gemm, Flatten, Shape, Concat, Transpose, GlobalAveragePool,
           Pad, Upsample, Sigmoid, Tanh
"""
from typing import Dict, List, Optional
from ...frontend.graph_ir import Graph, Node


class ShapeInferer:
    """
    从 Graph 中已知的 tensor shape（输入 + 权重）出发，
    按拓扑顺序迭代推断所有中间输出的 shape。
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        # 先用图里已知的 shape 初始化
        self.shapes: Dict[str, List[int]] = {}
        for name, t in graph.tensors.items():
            if t.shape:
                self.shapes[name] = list(t.shape)

        # 常量 int/value 推断（用于 Reshape 目标 shape 等）
        # 注意：Tensor.data 仅对 initializer/已折叠常量有效
        self._const_ints: Dict[str, List[int]] = {}
        for name, t in graph.tensors.items():
            if t.data is None:
                continue
            if getattr(t, "dtype", "") and (
                t.dtype.startswith("int") or t.dtype.startswith("uint")
            ):
                flat = t.data.reshape(-1).tolist()
                self._const_ints[name] = [int(v) for v in flat]
        self._infer(graph)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get(self, name: str) -> List[int]:
        """返回 tensor 的推断 shape，未知则返回空列表。"""
        return self.shapes.get(name, [])

    def _const_int_list(self, name: str) -> Optional[List[int]]:
        return self._const_ints.get(name)

    @staticmethod
    def _prod(shape: List[int]) -> int:
        r = 1
        for s in shape:
            r *= max(int(s), 1)
        return r

    # ------------------------------------------------------------------
    # 内部推断逻辑
    # ------------------------------------------------------------------

    def _infer(self, graph: Graph):
        """多轮迭代直到收敛（处理图中可能存在的反向依赖）。"""
        changed = True
        while changed:
            changed = False
            for node in graph.nodes:
                for name, shape in self._infer_node(node).items():
                    if name not in self.shapes or not self.shapes[name]:
                        self.shapes[name] = shape
                        changed = True

    def _infer_node(self, node: Node) -> Dict[str, List[int]]:
        """推断单个节点的输出 shape，返回 {tensor_name: shape} 字典。"""
        result: Dict[str, List[int]] = {}
        op = node.op_type

        def inp(idx: int = 0) -> List[int]:
            name = node.inputs[idx] if idx < len(node.inputs) else ""
            return self.shapes.get(name, [])

        # ---- Conv ----
        if op == "Conv":
            s, w = inp(0), inp(1)
            if len(s) == 4 and len(w) >= 1:
                pads    = node.attrs.get("pads",    [0, 0, 0, 0])
                strides = node.attrs.get("strides", [1, 1])
                N, C, H, W = s
                C_out = w[0]
                kH = w[2] if len(w) > 2 else 1
                kW = w[3] if len(w) > 3 else 1
                H_out = (H + pads[0] + pads[2] - kH) // strides[0] + 1
                W_out = (W + pads[1] + pads[3] - kW) // strides[1] + 1
                result[node.outputs[0]] = [N, C_out, H_out, W_out]
        
        # ---- 量化卷积算子（与 Conv 相同的形状推断）----
        elif op in ("quant_conv", "quant_conv_relu"):
            s, w = inp(0), inp(1)
            if len(s) == 4 and len(w) >= 1:
                pads    = node.attrs.get("pads",    [0, 0, 0, 0])
                strides = node.attrs.get("strides", [1, 1])
                N, C, H, W = s
                C_out = w[0]
                kH = w[2] if len(w) > 2 else 1
                kW = w[3] if len(w) > 3 else 1
                H_out = (H + pads[0] + pads[2] - kH) // strides[0] + 1
                W_out = (W + pads[1] + pads[3] - kW) // strides[1] + 1
                result[node.outputs[0]] = [N, C_out, H_out, W_out]

        # ---- 逐元素算子（shape 透传）----
        elif op in ("Relu", "Clip", "Add", "Sigmoid", "Tanh"):
            s = inp(0)
            if s:
                result[node.outputs[0]] = list(s)

        # ---- MaxPool ----
        elif op == "MaxPool":
            s = inp(0)
            if len(s) == 4:
                pads    = node.attrs.get("pads",         [0, 0, 0, 0])
                strides = node.attrs.get("strides",      [1, 1])
                kernel  = node.attrs.get("kernel_shape", [1, 1])
                N, C, H, W = s
                kH, kW = kernel[0], kernel[1]
                H_out = (H + pads[0] + pads[2] - kH) // strides[0] + 1
                W_out = (W + pads[1] + pads[3] - kW) // strides[1] + 1
                result[node.outputs[0]] = [N, C, H_out, W_out]

        # ---- ReduceMean（axes=[-1,-2] 等价于 GlobalAvgPool）----
        elif op == "ReduceMean":
            s = inp(0)
            if len(s) == 4:
                if node.attrs.get("keepdims", 1):
                    result[node.outputs[0]] = [s[0], s[1], 1, 1]
                else:
                    result[node.outputs[0]] = [s[0], s[1]]

        # ---- Reshape（[N,C,1,1] -> [N,C]，内存连续不变）----
        elif op == "Reshape":
            s = inp(0)
            if s:
                out_name = node.outputs[0]
                # ONNX Reshape 通常：inputs=[data, shape_tensor]
                # 若 shape_tensor 来自 initializer，可直接按其数值计算输出维度（支持 -1 / 0 语义）。
                if len(node.inputs) > 1:
                    target_name = node.inputs[1]
                    target = self._const_int_list(target_name)
                else:
                    target = None

                if target:
                    in_num = self._prod(s)
                    out_shape: List[int] = []
                    unknown_pos = -1
                    known_prod = 1
                    for i, d in enumerate(target):
                        if d == -1:
                            unknown_pos = i
                            out_shape.append(-1)
                        elif d == 0:
                            # 0 表示复制输入对应维度（ONNX 语义）
                            copied = s[i] if i < len(s) else 1
                            out_shape.append(int(copied))
                            known_prod *= int(copied)
                        else:
                            out_shape.append(int(d))
                            known_prod *= int(d)

                    if unknown_pos != -1:
                        inferred = in_num // max(known_prod, 1)
                        out_shape[unknown_pos] = int(inferred)

                    result[out_name] = out_shape
                else:
                    # 兜底：常见 squeeze: [N,C,1,1] -> [N,C]
                    if len(s) == 4 and s[2] == 1 and s[3] == 1:
                        result[out_name] = [s[0], s[1]]
                    else:
                        # 保守兜底：尽量维持 batch 和 channel
                        result[out_name] = s[:2] if len(s) >= 2 else list(s)

        # ---- Gemm ----
        elif op == "Gemm":
            s, w = inp(0), inp(1)
            if s and w:
                result[node.outputs[0]] = [s[0], w[0]]
        
        # ---- 量化 Gemm 算子（与 Gemm 相同的形状推断）----
        elif op in ("quant_gemm", "quant_gemm_relu"):
            s, w = inp(0), inp(1)
            if s and w:
                result[node.outputs[0]] = [s[0], w[0]]

        # ---- Flatten ----
        elif op == "Flatten":
            s = inp(0)
            if s:
                axis  = node.attrs.get("axis", 1)
                outer = 1
                for i in range(axis):
                    outer *= s[i]
                inner = 1
                for i in range(axis, len(s)):
                    inner *= s[i]
                result[node.outputs[0]] = [outer, inner]

        # ---- Shape / Concat（形状常量，运行期不生成数据）----
        elif op == "Shape":
            s = inp(0)
            result[node.outputs[0]] = [len(s)]

        elif op == "Concat":
            axis = node.attrs.get("axis", 0)
            shapes = [inp(i) for i in range(len(node.inputs))]
            shapes = [sh for sh in shapes if sh]
            if not shapes:
                return result
            rank = len(shapes[0])
            if rank == 1:
                axis = 0
            if axis < 0:
                axis += rank

            out_shape = list(shapes[0])
            out_shape[axis] = sum(sh[axis] for sh in shapes)
            result[node.outputs[0]] = out_shape

        # ---- Transpose（perm 只在 attrs 中出现的情况）----
        elif op == "Transpose":
            s = inp(0)
            if s:
                perm = node.attrs.get("perm")
                if perm and len(perm) == len(s):
                    result[node.outputs[0]] = [s[i] for i in perm]

        # ---- GlobalAveragePool ----
        elif op == "GlobalAveragePool":
            s = inp(0)
            if len(s) == 4:
                result[node.outputs[0]] = [s[0], s[1], 1, 1]
            elif len(s) == 2:
                result[node.outputs[0]] = list(s)

        # ---- Pad（仅支持 NCHW, constant pad value）----
        elif op == "Pad":
            s = inp(0)
            if len(s) == 4:
                pads = node.attrs.get("pads", [0, 0, 0, 0])
                if len(pads) == 8:
                    # [n0,c0,h0,w0,n1,c1,h1,w1]
                    ph0, pw0 = pads[2], pads[3]
                    ph1, pw1 = pads[6], pads[7]
                elif len(pads) == 4:
                    # [pt,pl,pb,pr]
                    ph0, pw0, ph1, pw1 = pads
                else:
                    ph0 = pw0 = ph1 = pw1 = 0
                result[node.outputs[0]] = [s[0], s[1], s[2] + ph0 + ph1, s[3] + pw0 + pw1]

        # ---- Upsample（常见 scales 属性）----
        elif op == "Upsample":
            s = inp(0)
            if s:
                scales = node.attrs.get("scales")
                if scales:
                    out_shape = []
                    if len(scales) == len(s):
                        for i, d in enumerate(s):
                            out_shape.append(int(round(d * float(scales[i]))))
                    elif len(scales) == 2 and len(s) == 4:
                        # 通常只缩放 H/W
                        out_shape = [s[0], s[1], int(round(s[2] * float(scales[0]))), int(round(s[3] * float(scales[1])))]
                    else:
                        out_shape = s
                    result[node.outputs[0]] = out_shape

        # 其他算子：默认不推断

        return result
