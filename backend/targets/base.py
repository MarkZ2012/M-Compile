"""
BaseTarget - 所有硬件平台代码生成器必须实现的抽象接口。

每个 emit_xxx 方法接收：
  - node  : 当前算子节点（含 inputs / outputs / attrs）
  - alloc : BufferAllocator 实例，用于查询 C 变量名、tensor 尺寸和 shape

返回值均为 List[str]，表示要写入 .c 文件的若干行 C 代码（含前导空格缩进）。
"""
from abc import ABC, abstractmethod
from typing import List
from ...frontend.graph_ir import Node


class BaseTarget(ABC):
    """定义所有硬件平台必须实现的代码生成接口。"""

    # ------------------------------------------------------------------
    # 平台标识
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def target_name(self) -> str:
        """返回平台名称字符串，例如 'generic_c'、'x86_avx'、'arm_neon'。"""

    # ------------------------------------------------------------------
    # 必须实现的算子（ResNet18 完整路径覆盖）
    # ------------------------------------------------------------------

    @abstractmethod
    def emit_conv(self, node: Node, alloc) -> List[str]:
        """生成 Conv2d 算子 C 代码。"""

    @abstractmethod
    def emit_relu(self, node: Node, alloc) -> List[str]:
        """生成 ReLU 激活 C 代码。"""

    @abstractmethod
    def emit_maxpool(self, node: Node, alloc) -> List[str]:
        """生成 MaxPool2d 算子 C 代码。"""

    @abstractmethod
    def emit_add(self, node: Node, alloc) -> List[str]:
        """生成逐元素 Add（残差连接）C 代码。"""

    @abstractmethod
    def emit_reducemean(self, node: Node, alloc) -> List[str]:
        """生成 ReduceMean（等价于 GlobalAvgPool）C 代码。"""

    @abstractmethod
    def emit_gemm(self, node: Node, alloc) -> List[str]:
        """生成全连接层（Gemm / Linear）C 代码。"""

    @abstractmethod
    def emit_softmax(self, node: Node, alloc) -> List[str]:
        """生成 Softmax C 代码。"""

    # ------------------------------------------------------------------
    # 可选覆盖的算子（BaseTarget 提供通用默认实现）
    # ------------------------------------------------------------------

    def emit_shape(self, node: Node, alloc) -> List[str]:
        """Shape 算子：运行期为常量，通常直接跳过。"""
        return ["    // Shape (batch dim constant, skip)"]

    def emit_concat(self, node: Node, alloc) -> List[str]:
        """Concat 算子：通常为 shape 拼接常量，跳过。"""
        return ["    // Concat (shape constant, skip)"]

    def emit_reshape(self, node: Node, alloc) -> List[str]:
        """Reshape：内存布局不变时为 no-op，必要时插入 memcpy。"""
        a = alloc
        x, out = node.inputs[0], node.outputs[0]
        size = a.size(x)
        lines = []
        if a.var(x) != a.var(out):
            lines.append(
                f"    memcpy({a.var(out)}, {a.var(x)}, {size}*sizeof(float));"
            )
        lines.append("    // Reshape (no-op)")
        return lines

    def emit_flatten(self, node: Node, alloc) -> List[str]:
        """Flatten：内存连续时为 no-op，必要时插入 memcpy。"""
        a = alloc
        x, out = node.inputs[0], node.outputs[0]
        size = a.size(x)
        lines = []
        if a.var(x) != a.var(out):
            lines.append(
                f"    memcpy({a.var(out)}, {a.var(x)}, {size}*sizeof(float));"
            )
        lines.append("    // Flatten (no-op)")
        return lines

    def emit_unknown(self, node: Node, alloc) -> List[str]:
        """未知/未支持算子：插入 TODO 注释，不中断编译。"""
        return [
            f"    /* TODO: unsupported op [{node.op_type}] "
            f"inputs={node.inputs} outputs={node.outputs} on {self.target_name} */"
        ]

    # ------------------------------------------------------------------
    # 平台专属构建文件
    # ------------------------------------------------------------------

    @abstractmethod
    def get_includes(self) -> List[str]:
        """返回该平台所需的 #include 指令列表。"""

    @abstractmethod
    def emit_cmake(self, model_name: str) -> str:
        """返回平台专属的 CMakeLists.txt 文件内容。"""
