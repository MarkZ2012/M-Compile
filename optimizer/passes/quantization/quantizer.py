"""
量化器核心模块

实现量化/反量化的核心逻辑，包括：
- scale 和 zero_point 的计算
- per-tensor 和 per-channel 量化
- 支持多种校准方法
"""
import numpy as np
from typing import Tuple, List, Optional
from .quant_config import QuantConfig, CalibrationMethod


class Quantizer:
    """
    量化器类
    
    提供量化相关的核心功能
    """
    
    def __init__(self, config: Optional[QuantConfig] = None):
        self.config = config or QuantConfig()
    
    def compute_quant_params_per_tensor(
        self, 
        tensor: np.ndarray
    ) -> Tuple[float, int]:
        """
        计算 per-tensor 量化参数
        
        Args:
            tensor: 输入张量 (float32)
            
        Returns:
            scale: 缩放因子
            zero_point: 零点
        """
        tensor = tensor.astype(np.float32)
        
        if self.config.symmetric:
            # 对称量化
            abs_max = np.max(np.abs(tensor))
            if abs_max == 0:
                return 1.0, 0
            scale = abs_max / abs(self.config.quant_max)
            zero_point = 0
        else:
            # 非对称量化
            t_min = np.min(tensor)
            t_max = np.max(tensor)
            if t_max == t_min:
                return 1.0, 0
            scale = (t_max - t_min) / (self.config.quant_max - self.config.quant_min)
            zero_point = int(np.round(-t_min / scale))
            zero_point = np.clip(zero_point, self.config.quant_min, self.config.quant_max)
        
        return float(scale), int(zero_point)
    
    def compute_quant_params_per_channel(
        self, 
        tensor: np.ndarray, 
        axis: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 per-channel 量化参数
        
        Args:
            tensor: 输入张量 (float32)，通常是权重 [out_ch, in_ch, ...]
            axis: 通道轴，默认为0（输出通道）
            
        Returns:
            scales: 每个通道的缩放因子
            zero_points: 每个通道的零点
        """
        tensor = tensor.astype(np.float32)
        num_channels = tensor.shape[axis]
        
        scales = np.zeros(num_channels, dtype=np.float32)
        zero_points = np.zeros(num_channels, dtype=np.int32)
        
        for i in range(num_channels):
            # 提取第i个通道的数据
            slices = [slice(None)] * tensor.ndim
            slices[axis] = i
            channel_data = tensor[tuple(slices)]
            
            scale, zp = self.compute_quant_params_per_tensor(channel_data)
            scales[i] = scale
            zero_points[i] = zp
        
        return scales, zero_points
    
    def quantize(
        self, 
        tensor: np.ndarray, 
        scale: float, 
        zero_point: int
    ) -> np.ndarray:
        """
        量化浮点张量为整数张量
        
        Args:
            tensor: 浮点张量
            scale: 缩放因子
            zero_point: 零点
            
        Returns:
            量化后的整数张量
        """
        tensor = tensor.astype(np.float32)
        
        if self.config.bit_width == 8:
            dtype = np.int8
        elif self.config.bit_width == 16:
            dtype = np.int16
        else:
            raise ValueError(f"Unsupported bit_width: {self.config.bit_width}")
        
        # 量化公式: q = round(x / scale) + zero_point
        q = np.round(tensor / scale + zero_point)
        q = np.clip(q, self.config.quant_min, self.config.quant_max)
        
        return q.astype(dtype)
    
    def quantize_per_channel(
        self, 
        tensor: np.ndarray, 
        scales: np.ndarray, 
        zero_points: np.ndarray,
        axis: int = 0
    ) -> np.ndarray:
        """
        Per-channel 量化
        
        Args:
            tensor: 输入张量
            scales: 每个通道的缩放因子
            zero_points: 每个通道的零点
            axis: 通道轴
            
        Returns:
            量化后的整数张量
        """
        tensor = tensor.astype(np.float32)
        num_channels = tensor.shape[axis]
        
        if self.config.bit_width == 8:
            dtype = np.int8
        elif self.config.bit_width == 16:
            dtype = np.int16
        else:
            raise ValueError(f"Unsupported bit_width: {self.config.bit_width}")
        
        result = np.zeros_like(tensor, dtype=dtype)
        
        for i in range(num_channels):
            slices = [slice(None)] * tensor.ndim
            slices[axis] = i
            channel_data = tensor[tuple(slices)]
            result[tuple(slices)] = self.quantize(channel_data, scales[i], zero_points[i])
        
        return result
    
    def dequantize(
        self, 
        quantized: np.ndarray, 
        scale: float, 
        zero_point: int
    ) -> np.ndarray:
        """
        反量化整数张量为浮点张量
        
        Args:
            quantized: 量化后的整数张量
            scale: 缩放因子
            zero_point: 零点
            
        Returns:
            反量化后的浮点张量
        """
        # 反量化公式: x = (q - zero_point) * scale
        return (quantized.astype(np.float32) - zero_point) * scale
    
    def dequantize_per_channel(
        self, 
        quantized: np.ndarray, 
        scales: np.ndarray, 
        zero_points: np.ndarray,
        axis: int = 0
    ) -> np.ndarray:
        """
        Per-channel 反量化
        
        Args:
            quantized: 量化后的整数张量
            scales: 每个通道的缩放因子
            zero_points: 每个通道的零点
            axis: 通道轴
            
        Returns:
            反量化后的浮点张量
        """
        num_channels = quantized.shape[axis]
        result = np.zeros_like(quantized, dtype=np.float32)
        
        for i in range(num_channels):
            slices = [slice(None)] * quantized.ndim
            slices[axis] = i
            channel_data = quantized[tuple(slices)]
            result[tuple(slices)] = self.dequantize(channel_data, scales[i], zero_points[i])
        
        return result


class MinMaxCalibrator:
    """
    Min-Max 校准器
    
    收集激活值的最小/最大值，用于计算量化参数
    """
    
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.count = 0
    
    def collect(self, tensor: np.ndarray):
        """收集统计数据"""
        self.min_val = min(self.min_val, float(np.min(tensor)))
        self.max_val = max(self.max_val, float(np.max(tensor)))
        self.count += 1
    
    def get_quant_params(self, symmetric: bool = True) -> Tuple[float, int]:
        """获取量化参数"""
        if symmetric:
            abs_max = max(abs(self.min_val), abs(self.max_val))
            if abs_max == 0:
                return 1.0, 0
            scale = abs_max / 127
            zero_point = 0
        else:
            if self.max_val == self.min_val:
                return 1.0, 0
            scale = (self.max_val - self.min_val) / 255
            zero_point = int(np.round(-self.min_val / scale))
            zero_point = np.clip(zero_point, 0, 255)
        
        return float(scale), int(zero_point)
    
    def reset(self):
        """重置校准器"""
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.count = 0


class MSECalibrator:
    """
    MSE (Mean Squared Error) 校准器
    
    通过最小化量化误差来选择最优的量化参数
    """
    
    def __init__(self, num_bins: int = 2048):
        self.histogram = np.zeros(num_bins, dtype=np.int64)
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.num_bins = num_bins
        self.count = 0
    
    def collect(self, tensor: np.ndarray):
        """收集统计数据"""
        tensor = tensor.astype(np.float32)
        self.min_val = min(self.min_val, float(np.min(tensor)))
        self.max_val = max(self.max_val, float(np.max(tensor)))
        
        # 构建直方图
        if self.max_val > self.min_val:
            normalized = (tensor - self.min_val) / (self.max_val - self.min_val)
            indices = np.clip((normalized * (self.num_bins - 1)).astype(int), 0, self.num_bins - 1)
            for idx in indices.flatten():
                self.histogram[idx] += 1
        
        self.count += 1
    
    def get_quant_params(self, symmetric: bool = True) -> Tuple[float, int]:
        """获取最优量化参数"""
        if self.min_val >= self.max_val:
            return 1.0, 0
        
        # 简化版MSE：使用99.9%分位数
        total = np.sum(self.histogram)
        cumsum = np.cumsum(self.histogram)
        threshold_idx = np.searchsorted(cumsum, total * 0.999)
        threshold_idx = min(threshold_idx, self.num_bins - 1)
        
        threshold_val = self.min_val + (self.max_val - self.min_val) * threshold_idx / self.num_bins
        
        if symmetric:
            abs_max = max(abs(self.min_val), abs(threshold_val))
            scale = abs_max / 127
            zero_point = 0
        else:
            scale = (threshold_val - self.min_val) / 255
            zero_point = int(np.round(-self.min_val / scale))
            zero_point = np.clip(zero_point, 0, 255)
        
        return float(scale), int(zero_point)
    
    def reset(self):
        """重置校准器"""
        self.histogram = np.zeros(self.num_bins, dtype=np.int64)
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.count = 0


def quantize_weight(
    weight: np.ndarray, 
    config: Optional[QuantConfig] = None
) -> Tuple[np.ndarray, float, int]:
    """
    便捷函数：量化权重
    
    Args:
        weight: 权重张量
        config: 量化配置
        
    Returns:
        quantized_weight: 量化后的权重
        scale: 缩放因子
        zero_point: 零点
    """
    quantizer = Quantizer(config)
    scale, zero_point = quantizer.compute_quant_params_per_tensor(weight)
    quantized = quantizer.quantize(weight, scale, zero_point)
    return quantized, scale, zero_point


def quantize_weight_per_channel(
    weight: np.ndarray, 
    axis: int = 0,
    config: Optional[QuantConfig] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    便捷函数：Per-channel 量化权重
    
    Args:
        weight: 权重张量
        axis: 通道轴
        config: 量化配置
        
    Returns:
        quantized_weight: 量化后的权重
        scales: 每个通道的缩放因子
        zero_points: 每个通道的零点
    """
    quantizer = Quantizer(config)
    scales, zero_points = quantizer.compute_quant_params_per_channel(weight, axis)
    quantized = quantizer.quantize_per_channel(weight, scales, zero_points, axis)
    return quantized, scales, zero_points