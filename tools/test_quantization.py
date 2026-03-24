"""
量化功能集成测试
测试 PTQ pass 对 ResNet18 ONNX 模型的量化效果
"""
import sys
import os
import numpy as np
from pathlib import Path

# 添加项目根路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from my_ai_compiler.frontend.onnx_parser import ONNXParser
from my_ai_compiler.optimizer.pass_manager import PassManager
from my_ai_compiler.optimizer.passes.quantization.ptq import post_training_quantize


ONNX_PATH = Path(__file__).parent / "model" / "resnet18.onnx"


# ── 1. 解析 ONNX ──────────────────────────────────────────────
def load_graph():
    print(f"[1/4] 解析 ONNX: {ONNX_PATH}")
    parser = ONNXParser()
    graph = parser.parse(str(ONNX_PATH))
    print(f"      节点数: {len(graph.nodes)},  tensor 数: {len(graph.tensors)}")
    return graph


# ── 2. 量化前统计 ─────────────────────────────────────────────
def stats_before(graph):
    quantizable = {"Conv", "ConvBn", "ConvBnRelu", "Gemm", "GemmRelu", "GemmBias"}
    nodes = [n for n in graph.nodes if n.op_type in quantizable]
    print(f"\n[2/4] 量化前统计")
    print(f"      可量化算子数: {len(nodes)}")

    float32_params = 0
    for n in nodes:
        if len(n.inputs) >= 2:
            t = graph.tensors.get(n.inputs[1])
            if t is not None and t.data is not None:
                float32_params += t.data.size
    print(f"      float32 权重参数量: {float32_params:,}")
    print(f"      预计 float32 占用: {float32_params * 4 / 1024 / 1024:.2f} MB")
    return nodes


# ── 3. 运行 PTQ pass ──────────────────────────────────────────
def run_ptq(graph):
    print(f"\n[3/4] 运行 PTQ pass ...")
    graph_q = post_training_quantize(graph)
    return graph_q


# ── 4. 量化后验证 ─────────────────────────────────────────────
def verify_after(graph_q, nodes_before):
    quantizable = {"Conv", "ConvBn", "ConvBnRelu", "Gemm", "GemmRelu", "GemmBias"}
    print(f"\n[4/4] 量化后验证")

    ok_count = 0
    fail_list = []

    for n in graph_q.nodes:
        if n.op_type not in quantizable:
            continue
        if len(n.inputs) < 2:
            continue

        t = graph_q.tensors.get(n.inputs[1])
        if t is None or t.data is None:
            continue

        # ① dtype 检查
        dtype_ok = t.data.dtype == np.int8 or getattr(t, "dtype", None) == "int8"
        # ② scale 检查
        scale_ok = hasattr(t, "scale") and isinstance(t.scale, float) and t.scale > 0
        # ③ attributes 检查
        attr_ok = n.attributes.get("quantized") is True

        if dtype_ok and scale_ok and attr_ok:
            ok_count += 1
        else:
            fail_list.append({
                "node": n.name or n.op_type,
                "dtype_ok": dtype_ok,
                "scale_ok": scale_ok,
                "attr_ok": attr_ok,
            })

    int8_params = 0
    for n in graph_q.nodes:
        if n.op_type not in quantizable:
            continue
        if len(n.inputs) >= 2:
            t = graph_q.tensors.get(n.inputs[1])
            if t is not None and t.data is not None and t.data.dtype == np.int8:
                int8_params += t.data.size

    print(f"      量化成功算子数: {ok_count} / {len(nodes_before)}")
    print(f"      int8 权重参数量: {int8_params:,}")
    print(f"      量化后权重占用:  {int8_params / 1024 / 1024:.2f} MB  "
          f"(理论压缩比 4x)")

    # 抽样打印几个 scale 值
    sample_count = 0
    for n in graph_q.nodes:
        if n.op_type not in quantizable or sample_count >= 3:
            break
        t = graph_q.tensors.get(n.inputs[1]) if len(n.inputs) >= 2 else None
        if t and hasattr(t, "scale"):
            print(f"      {n.op_type:12s}  weight_scale = {t.scale:.6f}")
            sample_count += 1

    if fail_list:
        print(f"\n  ⚠ 以下节点量化不完整:")
        for f in fail_list:
            print(f"    {f}")
    else:
        print(f"\n  ✅ 所有可量化节点已正确标记 int8")


# ── main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    graph = load_graph()
    nodes_before = stats_before(graph)
    graph_q = run_ptq(graph)
    verify_after(graph_q, nodes_before)