"""
Post-Training Quantization (PTQ) Pass

将 float32 权重量化为 int8，并在图中插入 QuantizeLinear / DequantizeLinear 节点，
使后端能生成 int8 推理代码。

执行时机：
    必须在 fusion passes 之后运行，面对已融合的算子；
    需要在 constant_fold 之后运行，确保权重数据已全部展开为常量。

当前实现状态：
    ☑ per-tensor 对称量化（权重）
    ☐ per-channel 量化（TODO）
    ☐ 激活量化 / calibration（TODO，需要校准数据集）
"""
import numpy as np
from typing import Dict
from ....frontend.graph_ir import Graph, Node, Tensor

# 量化参数：int8 对称量化范围
_INT8_MAX = 127


def post_training_quantize(graph: Graph) -> Graph:
    """
    PTQ pass 主入口：对 Conv / ConvBn / ConvBnRelu / Gemm 节点的权重做 per-tensor int8 量化。

    量化流程：
        1. 计算权重的 scale = max(|w|) / 127
        2. 量化：w_q = round(w / scale).clip(-128, 127).astype(int8)
        3. 将量化后的权重写回 tensor.data，并记录 tensor.scale
        4. 在节点 attributes 中标记 quantized=True，供 emitter 生成 int8 kernel 调用
    """
    quantized_count = 0

    _QUANTIZABLE_OPS = {"Conv", "ConvBn", "ConvBnRelu", "Gemm", "GemmRelu", "GemmBias"}

    for node in graph.nodes:
        if node.op_type not in _QUANTIZABLE_OPS:
            continue

        # inputs[1] 是权重 tensor（ONNX 约定：Conv/Gemm 的第二个输入为 weight）
        if len(node.inputs) < 2:
            continue

        weight_name = node.inputs[1]
        weight_tensor = graph.tensors.get(weight_name)
        if weight_tensor is None or weight_tensor.data is None:
            continue
        if weight_tensor.data.dtype != np.float32:
            continue   # 已量化或非 float，跳过

        w = weight_tensor.data.astype(np.float32)
        abs_max = np.max(np.abs(w))
        if abs_max == 0:
            continue   # 全零权重，跳过

        scale  = abs_max / _INT8_MAX
        w_q    = np.round(w / scale).clip(-128, 127).astype(np.int8)

        weight_tensor.data  = w_q
        weight_tensor.scale = float(scale)   # 附加 scale 属性，emitter 读取
        weight_tensor.dtype = "int8"

        node.attributes["quantized"] = True
        node.attributes["weight_scale"] = float(scale)

        quantized_count += 1

    if quantized_count:
        print(f"  [PTQ] Quantized {quantized_count} weight tensor(s) to int8")

    return graph
