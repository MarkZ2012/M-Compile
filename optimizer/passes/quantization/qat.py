"""
Quantization-Aware Training (QAT) Pass

读取 QAT 训练后导出的 ONNX 模型中的 FakeQuant 节点，
提取量化参数（scale / zero_point），折叠为真实的 int8 权重。

执行时机：与 PTQ 互斥，二者选其一；同样需在 fusion 之后。

当前实现状态：
    ☑ 识别 FakeQuantWithMinMaxVars 节点并提取参数（骨架）
    ☐ 真实折叠逻辑（TODO，依赖具体训练框架的 QAT 导出格式）
"""
from ....frontend.graph_ir import Graph


def qat_fold(graph: Graph) -> Graph:
    """
    QAT fold pass 主入口：将 FakeQuant 节点折叠为真实 int8 权重。

    TODO:
        实现依赖训练框架（PyTorch FX / TensorFlow QAT）导出的具体 ONNX 结构，
        待确认格式后补充。
    """
    fake_quant_nodes = [n for n in graph.nodes if "FakeQuant" in n.op_type]

    if not fake_quant_nodes:
        # 模型中无 FakeQuant 节点，可能不是 QAT 模型，直接跳过
        return graph

    # TODO: 提取 scale / zero_point → 折叠权重 → 移除 FakeQuant 节点
    print(f"  [QAT] Found {len(fake_quant_nodes)} FakeQuant node(s) — fold not yet implemented")
    return graph
