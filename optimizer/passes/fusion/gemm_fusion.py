"""
GEMM Fusion Pass

融合目标：
    Gemm → Add（bias）→ Relu/Sigmoid/Tanh
    合并为 GemmBiasRelu / GemmBiasSigmoid / GemmBias

使用场景：
    ResNet18 的全连接层（fc）在 ONNX 中通常被表示为 Gemm，
    融合后减少一次 bias add 的单独 kernel 调用开销。

注意：
    Gemm 节点本身带 transB/alpha/beta 属性，融合时原样保留。
    若 Gemm 已包含 bias 输入（inputs[2]），则不需要额外的 Add 节点融合，
    本 pass 只处理 bias 以独立 Add 节点形式出现的情况。
"""
from typing import Dict, List, Optional
from ....frontend.graph_ir import Graph, Node


_SUPPORTED_ACTIVATIONS = {"Relu", "Sigmoid", "Tanh"}


def gemm_fusion(graph: Graph) -> Graph:
    """
    GEMM + Add(bias) + Activation 融合 pass 主入口。
    """
    fused_count = 0

    output_to_node: Dict[str, Node] = {}
    input_to_nodes: Dict[str, List[Node]] = {}

    def _rebuild():
        output_to_node.clear()
        input_to_nodes.clear()
        for n in graph.nodes:
            for out in n.outputs:
                output_to_node[out] = n
            for inp in n.inputs:
                input_to_nodes.setdefault(inp, []).append(n)

    _rebuild()

    for node in list(graph.nodes):
        if node.op_type != "Gemm":
            continue

        gemm_out = node.outputs[0]

        # ── 若 Gemm 已包含 bias（inputs 长度 == 3），跳过 Add 融合 ──
        has_builtin_bias = len(node.inputs) == 3

        if not has_builtin_bias:
            # 尝试找后接的 Add 节点（bias 以独立节点存在）
            add_node = _find_single_consumer_of_type(
                gemm_out, "Add", output_to_node, input_to_nodes
            )
            if add_node is None:
                continue
            chain_out = add_node.outputs[0]
            extra_inputs = [_other_input(add_node, gemm_out)]  # bias tensor
            removes = [add_node]
        else:
            chain_out = gemm_out
            extra_inputs = []
            removes = []

        # ── 尝试继续找激活函数 ──────────────────────────────────
        act_node = None
        for act_op in _SUPPORTED_ACTIVATIONS:
            act_node = _find_single_consumer_of_type(
                chain_out, act_op, output_to_node, input_to_nodes
            )
            if act_node is not None:
                break

        if act_node is not None:
            final_out = act_node.outputs[0]
            new_op    = f"Gemm{act_node.op_type}"   # e.g. GemmRelu
            removes.append(act_node)
        else:
            if not removes:
                # 没有 Add 也没有激活，无需融合
                continue
            final_out = chain_out
            new_op    = "GemmBias"

        fused_node = Node(
            name=f"{node.name}_fused",
            op_type=new_op,
            inputs=node.inputs + extra_inputs,
            outputs=[final_out],
            attributes={**node.attributes, "_fused_from": [node.name]},
        )

        idx = graph.nodes.index(node)
        graph.nodes[idx] = fused_node
        for rm in removes:
            if rm in graph.nodes:
                graph.nodes.remove(rm)

        fused_count += 1
        _rebuild()

    if fused_count:
        print(f"  [GemmFusion] Fused {fused_count} Gemm chain(s)")

    return graph


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _find_single_consumer_of_type(
    tensor: str,
    op_type: str,
    output_to_node: Dict[str, Node],
    input_to_nodes: Dict[str, List[Node]],
) -> Optional[Node]:
    consumers = input_to_nodes.get(tensor, [])
    if len(consumers) != 1:
        return None
    n = consumers[0]
    return n if n.op_type == op_type else None


def _other_input(node: Node, exclude: str) -> str:
    """返回 Add 节点中非 exclude 的那个输入（即 bias tensor 名）。"""
    for inp in node.inputs:
        if inp != exclude:
            return inp
    raise ValueError(f"Add node '{node.name}' has no input other than '{exclude}'")
