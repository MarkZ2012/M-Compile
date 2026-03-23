"""
Conv + BN + ReLU Fusion Pass

融合逻辑：
    将计算图中 Conv → BatchNorm → ReLU（或 Conv → BatchNorm）的三节点链
    合并为单个 ConvBnRelu（或 ConvBn）节点。

为什么放在 constant_fold 之后？
    BatchNorm 的 scale / bias / mean / var 必须已是常量张量（tensor.data is not None），
    融合时才能将 BN 参数折算进 Conv 的 weight/bias。
    constant_fold 保证了这一前提。

融合后 C 运行时对应的算子：
    ConvBnRelu  → runtime/ops/generic/conv2d.c 中的 conv2d_bn_relu()
    ConvBn      → runtime/ops/generic/conv2d.c 中的 conv2d_bn()
    后端 emitter 只需判断 node.op_type 即可分发，无需修改 graph_ir。
"""
import numpy as np
from typing import Dict, List, Optional
from ....frontend.graph_ir import Graph, Node, Tensor


# ─────────────────────────────────────────────────────────────────────────────
# 公共入口
# ─────────────────────────────────────────────────────────────────────────────

def conv_bn_relu_fusion(graph: Graph) -> Graph:
    """
    Conv+BN+ReLU / Conv+BN 融合 pass 主入口。

    遍历图中所有节点，识别可融合的链，原地修改 graph 并返回。
    """
    fused_count = 0

    # 建立辅助映射
    output_to_node: Dict[str, Node] = {}   # tensor_name -> 产出它的 Node
    input_to_nodes: Dict[str, List[Node]] = {}  # tensor_name -> 消费它的 Node 列表

    def _rebuild_maps():
        output_to_node.clear()
        input_to_nodes.clear()
        for n in graph.nodes:
            for out in n.outputs:
                output_to_node[out] = n
            for inp in n.inputs:
                input_to_nodes.setdefault(inp, []).append(n)

    _rebuild_maps()

    nodes_to_remove: List[Node] = []

    for node in list(graph.nodes):
        if node in nodes_to_remove:
            continue
        if node.op_type != "Conv":
            continue

        conv_node = node
        conv_out  = conv_node.outputs[0]

        # ── 向下查找 BN ────────────────────────────────────────
        bn_node = _find_single_consumer(conv_out, "BatchNormalization", output_to_node, input_to_nodes)
        if bn_node is None:
            continue

        bn_out = bn_node.outputs[0]

        # ── BN 参数必须是常量（constant_fold 之后才能保证）──────
        if not _bn_params_are_constant(bn_node, graph):
            continue

        # ── 尝试继续找 ReLU ────────────────────────────────────
        relu_node = _find_single_consumer(bn_out, "Relu", output_to_node, input_to_nodes)

        if relu_node is not None:
            # Conv + BN + ReLU → ConvBnRelu
            final_output = relu_node.outputs[0]
            new_op = "ConvBnRelu"
            extra_removes = [bn_node, relu_node]
        else:
            # Conv + BN → ConvBn
            final_output = bn_out
            new_op = "ConvBn"
            extra_removes = [bn_node]

        # ── 构建融合节点 ───────────────────────────────────────
        fused_node = Node(
            name=f"{conv_node.name}_fused",
            op_type=new_op,
            # 输入：Conv 原始输入 + BN 的 scale/bias/mean/var
            inputs=conv_node.inputs + bn_node.inputs[1:],   # bn_node.inputs[0] 是 conv 的输出，跳过
            outputs=[final_output],
            attributes={**conv_node.attributes, **{"_fused_from": [conv_node.name, bn_node.name]}},
        )

        # ── 替换图中的节点 ─────────────────────────────────────
        idx = graph.nodes.index(conv_node)
        graph.nodes[idx] = fused_node
        for rm in extra_removes:
            if rm in graph.nodes:
                graph.nodes.remove(rm)
        nodes_to_remove.extend(extra_removes)

        fused_count += 1
        _rebuild_maps()   # 图结构已变，重建映射

    if fused_count:
        print(f"  [ConvBnFusion] Fused {fused_count} Conv+BN(+ReLU) chains")

    return graph


# ─────────────────────────────────────────────────────────────────────────────
# 内部工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _find_single_consumer(
    tensor_name: str,
    expected_op: str,
    output_to_node: Dict[str, Node],
    input_to_nodes: Dict[str, List[Node]],
) -> Optional[Node]:
    """
    如果 tensor_name 恰好只被一个节点消费，且该节点类型为 expected_op，
    则返回该节点；否则返回 None。

    "只被一个节点消费" 是融合的必要条件：若有分支，融合会改变语义。
    """
    consumers = input_to_nodes.get(tensor_name, [])
    if len(consumers) != 1:
        return None
    node = consumers[0]
    return node if node.op_type == expected_op else None


def _bn_params_are_constant(bn_node: Node, graph: Graph) -> bool:
    """
    检查 BN 节点的 scale / bias / mean / var 四个参数张量是否都是常量
    （即 tensor.data is not None）。
    BN 的 inputs 布局：[X, scale, bias, mean, var]
    """
    for param_name in bn_node.inputs[1:]:   # 跳过第 0 个输入（上游特征图）
        tensor = graph.tensors.get(param_name)
        if tensor is None or tensor.data is None:
            return False
    return True
