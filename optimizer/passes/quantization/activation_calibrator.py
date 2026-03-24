"""
Activation Calibrator — 激活值校准模块

使用 ONNX Runtime 对每一个可量化节点的输入/输出进行统计，
收集 (min, max) 用于计算真实的 input_scale / output_scale。

设计原则
--------
* 不依赖任何训练框架，只需要 onnxruntime + numpy。
* 通过在 ONNX 图中插入中间输出（"截面"）来提取每层激活，
  不使用任何 hook 机制，兼容所有 onnxruntime 版本。
* 输出的 stats dict 的 key 与 Graph IR 中的 tensor name 对齐，
  ptq.py 可以直接通过 node.inputs[0] / node.outputs[0] 查找。

用法
----
    from .activation_calibrator import ActivationCalibrator

    calibrator = ActivationCalibrator(onnx_model_path)
    stats = calibrator.collect(calib_inputs)
    # stats: dict[tensor_name, (min_val, max_val)]
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import copy


ActivationStats = Dict[str, Tuple[float, float]]  # tensor_name -> (min, max)


class ActivationCalibrator:
    """
    基于 ONNX Runtime 的激活校准器。

    Parameters
    ----------
    onnx_model_path : str
        原始浮点 ONNX 模型路径（必须是 float32 模型，不是量化后的）。
    """

    def __init__(self, onnx_model_path: str):
        self.onnx_model_path = onnx_model_path
        self._sess = None          # 带所有中间输出的 session
        self._intermediate_names: List[str] = []  # 需要观测的 tensor name

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def collect(
        self,
        calib_inputs: List[np.ndarray],
        target_tensor_names: Optional[List[str]] = None,
        symmetric: bool = True,
    ) -> ActivationStats:
        """
        运行校准数据集，收集每个 tensor 的激活统计。

        Parameters
        ----------
        calib_inputs :
            预处理后的输入样本列表，每个元素 shape = (1, C, H, W)，dtype = float32。
        target_tensor_names :
            只统计这些 tensor 的激活。如果为 None，则统计所有中间张量。
        symmetric :
            True  → 用 abs_max 作为范围（对称量化，zero_point=0）
            False → 用 (min, max) 范围（非对称量化）

        Returns
        -------
        dict : tensor_name -> (min_val, max_val)
            当 symmetric=True 时，返回 (-abs_max, +abs_max)。
        """
        try:
            import onnxruntime as ort
            import onnx
        except ImportError as e:
            raise ImportError(
                "activation_calibrator 需要 onnxruntime 和 onnx：\n"
                "  pip install onnxruntime onnx"
            ) from e

        # 1. 构建带所有中间输出的扩展 session
        sess, observable_names = self._build_extended_session(
            ort, onnx, target_tensor_names
        )

        if not observable_names:
            print("  [Calibrator] WARNING: no observable tensors found, returning empty stats")
            return {}

        # 2. 逐样本推理，累计 min/max
        running_min: Dict[str, float] = {}
        running_max: Dict[str, float] = {}

        print(f"  [Calibrator] Running {len(calib_inputs)} calibration sample(s) "
              f"over {len(observable_names)} tensor(s)...")

        input_name = sess.get_inputs()[0].name

        for idx, inp in enumerate(calib_inputs):
            if inp.ndim == 3:
                inp = inp[np.newaxis, ...]  # 自动补 batch 维度

            outputs = sess.run(observable_names, {input_name: inp})

            for name, arr in zip(observable_names, outputs):
                arr = arr.astype(np.float32)
                cur_min = float(np.min(arr))
                cur_max = float(np.max(arr))

                if name not in running_min:
                    running_min[name] = cur_min
                    running_max[name] = cur_max
                else:
                    running_min[name] = min(running_min[name], cur_min)
                    running_max[name] = max(running_max[name], cur_max)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(calib_inputs):
                print(f"  [Calibrator]   {idx + 1}/{len(calib_inputs)} done")

        # 3. 整理结果
        stats: ActivationStats = {}
        for name in observable_names:
            mn = running_min.get(name, 0.0)
            mx = running_max.get(name, 0.0)
            if symmetric:
                abs_max = max(abs(mn), abs(mx))
                stats[name] = (-abs_max, abs_max)
            else:
                stats[name] = (mn, mx)

        print(f"  [Calibrator] Collected stats for {len(stats)} tensors")
        return stats

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _build_extended_session(self, ort, onnx, target_names):
        """
        在原始 ONNX 模型上添加所有中间 tensor 作为额外输出，
        返回 (session, list_of_observable_tensor_names)。

        这是最兼容的做法——不依赖 onnxruntime 的 intermediate_outputs
        接口（该接口在低版本中不存在）。
        """
        model = onnx.load(self.onnx_model_path)
        graph = model.graph

        # 收集图中所有 value_info（中间张量的类型信息）
        existing_output_names = {o.name for o in graph.output}
        value_info_map = {vi.name: vi for vi in graph.value_info}
        # 也包含已有的输出（它们有完整类型信息）
        for o in graph.output:
            value_info_map[o.name] = o

        # 决定要观测哪些 tensor
        if target_names is not None:
            observable = [n for n in target_names if n in value_info_map]
        else:
            # 默认：所有有类型信息且不是最终输出的中间张量
            observable = [
                name for name, vi in value_info_map.items()
                if name not in existing_output_names
            ]

        if not observable:
            # fallback：即使没有 value_info，也把 node outputs 都加进去
            for node in graph.node:
                for out_name in node.output:
                    if out_name and out_name not in existing_output_names:
                        observable.append(out_name)

        # 把要观测的 tensor 全部添加为图输出
        model_copy = copy.deepcopy(model)
        graph_copy = model_copy.graph

        added = set(o.name for o in graph_copy.output)
        for name in observable:
            if name not in added:
                # 如果有 value_info，使用它；否则创建一个空的（ort 会推断类型）
                if name in value_info_map:
                    graph_copy.output.append(value_info_map[name])
                else:
                    import onnx.helper as helper
                    vi = helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
                    graph_copy.output.append(vi)
                added.add(name)

        # 序列化为字节并创建 session（避免写临时文件）
        model_bytes = model_copy.SerializeToString()
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # 只显示错误
        sess = ort.InferenceSession(model_bytes, sess_options=sess_options)

        # 返回实际可用的输出名（session 能成功运行的那些）
        session_output_names = {o.name for o in sess.get_outputs()}
        valid_observable = [n for n in observable if n in session_output_names]

        return sess, valid_observable


# ------------------------------------------------------------------
# 便捷函数：根据 Graph IR 中的节点列表决定要观测哪些 tensor
# ------------------------------------------------------------------

def get_tensors_to_calibrate(graph_nodes) -> List[str]:
    """
    从 Graph IR 的节点列表中提取所有量化节点的 input[0] 和 output[0]，
    作为校准目标。

    Parameters
    ----------
    graph_nodes : list[Node]
        Graph IR 的节点列表（在 PTQ pass 之前，所以 op_type 还是原始名）

    Returns
    -------
    list[str] : tensor names to observe
    """
    _QUANTIZABLE_OPS = {"Conv", "ConvBn", "ConvBnRelu", "Gemm", "GemmRelu", "GemmBias", "Linear"}
    targets = []
    seen = set()

    for node in graph_nodes:
        if node.op_type not in _QUANTIZABLE_OPS:
            continue
        # input[0] → 这一层的输入激活
        if node.inputs and node.inputs[0] not in seen:
            targets.append(node.inputs[0])
            seen.add(node.inputs[0])
        # output[0] → 这一层的输出激活（下一层的输入 scale 来源）
        if node.outputs and node.outputs[0] not in seen:
            targets.append(node.outputs[0])
            seen.add(node.outputs[0])

    return targets


def compute_scale_from_stats(
    min_val: float,
    max_val: float,
    symmetric: bool = True,
    bits: int = 8,
) -> Tuple[float, int]:
    """
    根据激活统计值计算量化 scale 和 zero_point。

    Parameters
    ----------
    min_val, max_val : float
        该 tensor 在校准集上的最小/最大值
    symmetric : bool
        True → 对称量化（zero_point = 0）
    bits : int
        量化位宽

    Returns
    -------
    (scale, zero_point)
    """
    qmax = 2 ** (bits - 1) - 1  # 127 for int8

    if symmetric:
        abs_max = max(abs(min_val), abs(max_val))
        if abs_max == 0.0:
            return 1.0, 0
        scale = abs_max / qmax
        zero_point = 0
    else:
        qmin_unsigned = 0
        qmax_unsigned = 2 ** bits - 1  # 255 for uint8
        if max_val == min_val:
            return 1.0, 0
        scale = (max_val - min_val) / (qmax_unsigned - qmin_unsigned)
        zero_point = int(round(-min_val / scale))
        zero_point = int(np.clip(zero_point, qmin_unsigned, qmax_unsigned))

    return float(scale), int(zero_point)
