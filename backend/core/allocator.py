"""
Buffer 分配器 - 为 Graph 中每个 tensor 分配 C 变量名和 buffer 大小。

分配策略：
  - 图输入  -> C 函数参数 "input"
  - 图输出  -> C 函数参数 "output"
  - 权重    -> 静态全局数组（变量名由 tensor 原始名称 safe 化得到）
  - 中间激活 -> buf_0, buf_1, buf_2, ... 静态全局数组
"""
from typing import Dict, List, Set, Tuple
from ...frontend.graph_ir import Graph
from .shape_infer import ShapeInferer


def _safe(name: str) -> str:
    """将 ONNX tensor 名称转换为合法的 C 标识符。"""
    return name.replace("/", "_").replace(":", "_").replace(".", "_").lstrip("_")


def _prod(shape: List[int]) -> int:
    """计算 shape 各维度乘积，每维至少为 1。"""
    r = 1
    for s in shape:
        r *= max(s, 1)
    return r


class BufferAllocator:
    """
    遍历 Graph，为每个 tensor 分配：
      - var  : C 变量名（字符串）
      - size : 元素个数（int）
      - shape: 维度列表（List[int]）

    使用方法::

        alloc = BufferAllocator(graph)
        var_name = alloc.var("tensor_name")
        elem_cnt = alloc.size("tensor_name")
        dims     = alloc.shape("tensor_name")
    """

    def __init__(self, graph: Graph):
        self.graph     = graph
        self._inferer  = ShapeInferer(graph)
        # {tensor_name: (c_var, elem_count, shape)}
        self._bufs: Dict[str, Tuple[str, int, List[int]]] = {}
        self._counter  = 0
        self._build(graph)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def var(self, name: str) -> str:
        """返回 tensor 对应的 C 变量名。"""
        return self._bufs[name][0] if name in self._bufs else "/*unknown*/"

    def size(self, name: str) -> int:
        """返回 tensor 的元素个数（float 数组长度）。"""
        return self._bufs[name][1] if name in self._bufs else 1

    def shape(self, name: str) -> List[int]:
        """返回 tensor 的维度列表。"""
        return self._bufs[name][2] if name in self._bufs else []

    def intermediate_bufs(self) -> List[Tuple[str, int, List[int]]]:
        """
        返回所有中间激活 buffer 的列表，按编号排序。
        每项为 (c_var_name, elem_count, shape)。
        """
        seen: Set[str] = set()
        result = []
        for _, (var, size, shape) in self._bufs.items():
            if var.startswith("buf_") and var not in seen:
                seen.add(var)
                result.append((var, size, shape))
        return sorted(result, key=lambda x: int(x[0].split("_")[1]))

    # ------------------------------------------------------------------
    # 内部构建逻辑
    # ------------------------------------------------------------------

    def _build(self, graph: Graph):
        # 1. 图输入 -> 映射为 C 函数参数 "input"
        for name in graph.inputs:
            shape = self._inferer.get(name)
            self._bufs[name] = ("input", _prod(shape), shape)

        # 2. 图输出 -> 映射为 C 函数参数 "output"
        for name in graph.outputs:
            shape = self._inferer.get(name)
            self._bufs[name] = ("output", _prod(shape) or 1000, shape)

        # 3. 权重 tensor -> 静态全局数组，变量名 safe 化
        for name, tensor in graph.tensors.items():
            if name in self._bufs:
                continue
            if tensor.data is not None:
                self._bufs[name] = (_safe(name), _prod(tensor.shape), tensor.shape)

        # 4. 中间激活（按节点拓扑顺序）-> buf_0, buf_1, ...
        for node in graph.nodes:
            for out_name in node.outputs:
                if out_name in self._bufs:
                    continue
                shape   = self._inferer.get(out_name)
                size    = _prod(shape) if shape else 1
                varname = f"buf_{self._counter}"
                self._counter += 1
                self._bufs[out_name] = (varname, size, shape)
