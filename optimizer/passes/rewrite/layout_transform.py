"""
Layout Transformation Pass

将计算图的 tensor 内存布局从 NCHW（PyTorch/ONNX 默认）
转换为 NHWC（ARM NEON / TFLite 友好格式）。

执行时机：
    必须在所有其他 pass 之前执行，确保后续 pass
    看到的图已是统一布局，无需各自处理两种格式。

当前支持：
    NCHW → NHWC   （面向 arm_neon target）

不支持（暂不处理，直接跳过）：
    - 非 4-D tensor（1-D bias、2-D weight 等）
    - 已是目标布局的 tensor

后端感知：
    layout_transform 会在 Graph 上设置 graph.layout 属性，
    backend/emitter.py 读取该属性决定生成代码的内存访问顺序，
    generic / x86_avx target 忽略该属性（保持 NCHW）。
"""
from typing import Optional
import numpy as np
from ....frontend.graph_ir import Graph, Tensor


def layout_nchw_to_nhwc(graph: Graph) -> Graph:
    """
    NCHW → NHWC Layout Transformation pass 主入口。

    对所有形状为 (N, C, H, W) 的常量张量（权重）做 transpose，
    并更新 tensor.shape；运行时输入的 transpose 由 emitter 生成前处理代码。
    """
    converted = 0
    skipped   = 0

    for name, tensor in graph.tensors.items():
        if len(tensor.shape) != 4:
            skipped += 1
            continue
        if getattr(tensor, "layout", "NCHW") == "NHWC":
            skipped += 1
            continue

        # 只对已有数据的常量权重做静态 transpose；
        # 运行时激活 tensor（data is None）的 transpose 由 emitter 插入
        if tensor.data is not None:
            # numpy transpose: (N,C,H,W) → (N,H,W,C)
            tensor.data  = np.transpose(tensor.data, (0, 2, 3, 1))

        # 更新 shape 记录
        n, c, h, w   = tensor.shape
        tensor.shape = [n, h, w, c]
        tensor.layout = "NHWC"   # 标记已转换
        converted += 1

    # 在 graph 上记录目标 layout，供 emitter 读取
    graph.layout = "NHWC"

    print(f"  [LayoutTransform] NCHW→NHWC: {converted} tensors converted, {skipped} skipped")
    return graph
