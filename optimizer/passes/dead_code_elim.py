"""
Dead Code Elimination Pass - 死代码消除
"""
from typing import Set
from ...frontend.graph_ir import Graph


def dead_code_elim(graph: Graph) -> Graph:
    """
    死代码消除
    移除对输出没有贡献的节点
    """
    # 从输出开始，反向追踪所有需要的节点
    needed_nodes: Set[str] = set()
    needed_tensors: Set[str] = set()
    
    # 初始化：所有输出都是需要的
    queue = list(graph.outputs)
    needed_tensors.update(graph.outputs)
    
    # 构建输出到节点的映射
    output_to_node = {}
    for node in graph.nodes:
        for output in node.outputs:
            output_to_node[output] = node.name
    
    # 构建节点名称到节点的映射
    node_map = {node.name: node for node in graph.nodes}
    
    # BFS遍历
    while queue:
        tensor_name = queue.pop(0)
        
        if tensor_name in output_to_node:
            node_name = output_to_node[tensor_name]
            if node_name not in needed_nodes:
                needed_nodes.add(node_name)
                node = node_map[node_name]
                
                # 将所有输入加入队列
                for inp in node.inputs:
                    if inp not in needed_tensors:
                        needed_tensors.add(inp)
                        queue.append(inp)
    
    # 移除不需要的节点
    original_count = len(graph.nodes)
    graph.nodes = [node for node in graph.nodes if node.name in needed_nodes]
    removed_count = original_count - len(graph.nodes)
    
    if removed_count > 0:
        print(f"  [DeadCodeElim] Removed {removed_count} dead nodes")
    
    return graph