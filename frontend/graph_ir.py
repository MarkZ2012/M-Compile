"""
Graph IR - 统一的中间表示定义
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class Tensor:
    """张量定义"""
    name: str
    shape: List[int]
    dtype: str  # float32, int8, etc.
    data: Optional[np.ndarray] = None  # 权重常量
    
    def __repr__(self):
        return f"Tensor(name={self.name}, shape={self.shape}, dtype={self.dtype})"


@dataclass
class Node:
    """计算节点"""
    op_type: str        # "Conv", "Relu", "Gemm", etc.
    name: str           # 节点名称
    inputs: List[str]   # 输入tensor names
    outputs: List[str]  # 输出tensor names
    attrs: Dict[str, Any] = field(default_factory=dict)  # 属性，如kernel_size, strides等
    
    def __repr__(self):
        return f"Node(op={self.op_type}, name={self.name})"


@dataclass
class Graph:
    """计算图"""
    name: str
    nodes: List[Node] = field(default_factory=list)
    tensors: Dict[str, Tensor] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)   # 输入tensor names
    outputs: List[str] = field(default_factory=list)  # 输出tensor names
    
    def add_node(self, node: Node):
        """添加节点"""
        self.nodes.append(node)
    
    def add_tensor(self, tensor: Tensor):
        """添加张量"""
        self.tensors[tensor.name] = tensor
    
    def get_tensor(self, name: str) -> Optional[Tensor]:
        """获取张量"""
        return self.tensors.get(name)
    
    def topological_sort(self) -> List[Node]:
        """拓扑排序"""
        # 构建依赖图
        in_degree = {node.name: 0 for node in self.nodes}
        dependents = {node.name: [] for node in self.nodes}
        
        # 建立节点输出到节点的映射
        output_to_node = {}
        for node in self.nodes:
            for output in node.outputs:
                output_to_node[output] = node.name
        
        # 计算入度
        for node in self.nodes:
            for input_name in node.inputs:
                if input_name in output_to_node:
                    producer = output_to_node[input_name]
                    if producer != node.name:
                        in_degree[node.name] += 1
                        dependents[producer].append(node.name)
        
        # 拓扑排序
        queue = [name for name, deg in in_degree.items() if deg == 0]
        sorted_nodes = []
        node_map = {node.name: node for node in self.nodes}
        
        while queue:
            current = queue.pop(0)
            sorted_nodes.append(node_map[current])
            
            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return sorted_nodes
    
    def __repr__(self):
        return f"Graph(name={self.name}, nodes={len(self.nodes)}, tensors={len(self.tensors)})"