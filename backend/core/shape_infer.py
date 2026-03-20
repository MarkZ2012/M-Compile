"""
Shape 推断器 - 通过节点拓扑顺序推断所有中间 tensor 的 shape。

支持的算子：Conv, Relu, Add, MaxPool, ReduceMean, Reshape,
           Gemm, Flatten, Shape, Concat, Sigmoid, Tanh
"""
from typing import Dict, List
from ...frontend.graph_ir import Graph, Node


class ShapeInferer:
    """
    从 Graph 中已知的 tensor shape（输入 + 权重）出发，
    按拓扑顺序迭代推断所有中间输出的 shape。
    """

    def __init__(self, graph: Graph):
        # 先用图里已知的 shape 初始化
        self.shapes: Dict[str, List[int]] = {}
        for name, t in graph.tensors.items():
            if t.shape:
                self.shapes[name] = list(t.shape)
        self._infer(graph)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get(self, name: str) -> List[int]:
        """返回 tensor 的推断 shape，未知则返回空列表。"""
        return self.shapes.get(name, [])

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

        # ---- 逐元素一元算子（shape 透传）----
        elif op in ("Relu", "Add", "Sigmoid", "Tanh"):
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
                result[node.outputs[0]] = [s[0], s[1]]

        # ---- Gemm ----
        elif op == "Gemm":
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
            result[node.outputs[0]] = [1]

        return result
