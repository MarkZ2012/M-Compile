"""
Constant Folding Pass - 常量折叠优化
"""
import numpy as np
from typing import Dict, Set
from ...frontend.graph_ir import Graph, Node, Tensor


def constant_fold(graph: Graph) -> Graph:
    """
    常量折叠优化
    识别可以预先计算的常量表达式，直接计算结果
    """
    # 找出所有常量张量
    constants: Set[str] = set()
    for name, tensor in graph.tensors.items():
        if tensor.data is not None:
            constants.add(name)
    
    # 找出可以折叠的节点
    nodes_to_remove = []
    new_constants = {}
    
    for node in graph.nodes:
        # 检查是否所有输入都是常量
        all_inputs_constant = all(inp in constants for inp in node.inputs)
        
        if all_inputs_constant and node.op_type in ["Add", "Mul", "Sub", "Div"]:
            # 可以折叠的算术运算
            try:
                result = _evaluate_constant_node(node, graph.tensors)
                if result is not None:
                    # 创建新的常量张量
                    output_name = node.outputs[0]
                    new_tensor = Tensor(
                        name=output_name,
                        shape=list(result.shape),
                        dtype="float32",
                        data=result
                    )
                    new_constants[output_name] = new_tensor
                    constants.add(output_name)
                    nodes_to_remove.append(node)
            except Exception:
                pass
    
    # 移除被折叠的节点
    if nodes_to_remove:
        new_nodes = [n for n in graph.nodes if n not in nodes_to_remove]
        graph.nodes = new_nodes
        
        # 添加新的常量张量
        for name, tensor in new_constants.items():
            graph.tensors[name] = tensor
        
        print(f"  [ConstantFold] Folded {len(nodes_to_remove)} constant expressions")
    
    return graph


def _evaluate_constant_node(node: Node, tensors: Dict[str, Tensor]) -> np.ndarray:
    """计算常量节点的值"""
    inputs = [tensors[inp].data for inp in node.inputs]
    
    if node.op_type == "Add":
        return inputs[0] + inputs[1]
    elif node.op_type == "Mul":
        return inputs[0] * inputs[1]
    elif node.op_type == "Sub":
        return inputs[0] - inputs[1]
    elif node.op_type == "Div":
        return inputs[0] / inputs[1]
    
    return None