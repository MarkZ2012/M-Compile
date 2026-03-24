"""
量化配置管理模块
"""
from dataclasses import dataclass, field
from typing import List, Optional, Any
from enum import Enum


class QuantMode(Enum):
    PTQ = "ptq"
    QAT = "qat"


class CalibrationMethod(Enum):
    MIN_MAX = "min_max"
    MSE     = "mse"
    KL      = "kl"


@dataclass
class QuantConfig:
    """
    量化配置类

    新增字段
    --------
    calib_onnx_path : str | None
        原始浮点 ONNX 模型路径，供 ActivationCalibrator 使用。
        通常与编译器入口传入的 model_path 相同。
    calib_inputs : list[np.ndarray] | None
        预处理后的校准样本，每个元素 shape=(1,C,H,W), dtype=float32。
        当 weight_only=False 时必须提供，否则 PTQ pass 会退回到
        weight-only 模式并打印警告。
    """
    quant_mode: QuantMode = QuantMode.PTQ
    bit_width: int = 8
    symmetric: bool = True
    per_channel: bool = False
    quant_ops: List[str] = field(default_factory=lambda: ["Conv", "Gemm", "Linear"])
    calibration_method: CalibrationMethod = CalibrationMethod.MIN_MAX
    calibration_samples: int = 100
    weight_only: bool = False          # 改为 False：默认启用激活量化

    # ── 校准数据（由 compile.py 填充）──────────────────────────────────
    calib_onnx_path: Optional[str] = None          # 浮点 ONNX 路径
    calib_inputs: Optional[List[Any]] = None        # list[np.ndarray]

    # ── 量化范围 ────────────────────────────────────────────────────────
    @property
    def quant_min(self) -> int:
        if self.symmetric:
            return -(2 ** (self.bit_width - 1))
        return 0

    @property
    def quant_max(self) -> int:
        if self.symmetric:
            return 2 ** (self.bit_width - 1) - 1
        return 2 ** self.bit_width - 1

    def __post_init__(self):
        if self.bit_width not in [8, 16]:
            raise ValueError(f"Unsupported bit_width: {self.bit_width}")
        if isinstance(self.quant_ops, str):
            self.quant_ops = [self.quant_ops]

    def to_dict(self) -> dict:
        return {
            "quant_mode": self.quant_mode.value,
            "bit_width": self.bit_width,
            "symmetric": self.symmetric,
            "per_channel": self.per_channel,
            "quant_ops": self.quant_ops,
            "calibration_method": self.calibration_method.value,
            "calibration_samples": self.calibration_samples,
            "weight_only": self.weight_only,
            # calib_inputs 不序列化（太大）
            "calib_onnx_path": self.calib_onnx_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QuantConfig":
        cfg = cls()
        if "quant_mode" in d:
            cfg.quant_mode = QuantMode(d["quant_mode"])
        if "bit_width" in d:
            cfg.bit_width = d["bit_width"]
        if "symmetric" in d:
            cfg.symmetric = d["symmetric"]
        if "per_channel" in d:
            cfg.per_channel = d["per_channel"]
        if "quant_ops" in d:
            cfg.quant_ops = d["quant_ops"]
        if "calibration_method" in d:
            cfg.calibration_method = CalibrationMethod(d["calibration_method"])
        if "calibration_samples" in d:
            cfg.calibration_samples = d["calibration_samples"]
        if "weight_only" in d:
            cfg.weight_only = d["weight_only"]
        if "calib_onnx_path" in d:
            cfg.calib_onnx_path = d["calib_onnx_path"]
        return cfg


DEFAULT_QUANT_CONFIG = QuantConfig()
