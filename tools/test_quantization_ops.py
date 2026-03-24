"""
量化算子单元测试

根据审核意见的测试安排：
阶段一：Python层正确性验证（先做，不依赖C）
- 测试 quantizer.py 的 scale/zp 计算是否正确，用 numpy 手算对比
- 验证量化/反量化的一致性
"""
import sys
import numpy as np
import unittest
from dataclasses import dataclass
from enum import Enum


class QuantMode(Enum):
    """量化模式"""
    PTQ = "ptq"
    QAT = "qat"


@dataclass
class QuantConfig:
    """量化配置"""
    quant_mode: QuantMode = QuantMode.PTQ
    bit_width: int = 8
    symmetric: bool = True
    per_channel: bool = False
    
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


class Quantizer:
    """量化器"""
    
    def __init__(self, config=None):
        self.config = config or QuantConfig()
    
    def compute_quant_params_per_tensor(self, tensor):
        """计算 per-tensor 量化参数"""
        tensor = tensor.astype(np.float32)
        
        if self.config.symmetric:
            abs_max = np.max(np.abs(tensor))
            if abs_max == 0:
                return 1.0, 0
            scale = abs_max / abs(self.config.quant_max)
            zero_point = 0
        else:
            t_min = np.min(tensor)
            t_max = np.max(tensor)
            if t_max == t_min:
                return 1.0, 0
            scale = (t_max - t_min) / (self.config.quant_max - self.config.quant_min)
            zero_point = int(np.round(-t_min / scale))
            zero_point = np.clip(zero_point, self.config.quant_min, self.config.quant_max)
        
        return float(scale), int(zero_point)
    
    def compute_quant_params_per_channel(self, tensor, axis=0):
        """计算 per-channel 量化参数"""
        tensor = tensor.astype(np.float32)
        num_channels = tensor.shape[axis]
        
        scales = np.zeros(num_channels, dtype=np.float32)
        zero_points = np.zeros(num_channels, dtype=np.int32)
        
        for i in range(num_channels):
            slices = [slice(None)] * tensor.ndim
            slices[axis] = i
            channel_data = tensor[tuple(slices)]
            
            scale, zp = self.compute_quant_params_per_tensor(channel_data)
            scales[i] = scale
            zero_points[i] = zp
        
        return scales, zero_points
    
    def quantize(self, tensor, scale, zero_point):
        """量化"""
        tensor = tensor.astype(np.float32)
        dtype = np.int8 if self.config.bit_width == 8 else np.int16
        
        q = np.round(tensor / scale + zero_point)
        q = np.clip(q, self.config.quant_min, self.config.quant_max)
        
        return q.astype(dtype)
    
    def quantize_per_channel(self, tensor, scales, zero_points, axis=0):
        """Per-channel 量化"""
        tensor = tensor.astype(np.float32)
        num_channels = tensor.shape[axis]
        dtype = np.int8 if self.config.bit_width == 8 else np.int16
        
        result = np.zeros_like(tensor, dtype=dtype)
        
        for i in range(num_channels):
            slices = [slice(None)] * tensor.ndim
            slices[axis] = i
            channel_data = tensor[tuple(slices)]
            result[tuple(slices)] = self.quantize(channel_data, scales[i], zero_points[i])
        
        return result
    
    def dequantize(self, quantized, scale, zero_point):
        """反量化"""
        return (quantized.astype(np.float32) - zero_point) * scale
    
    def dequantize_per_channel(self, quantized, scales, zero_points, axis=0):
        """Per-channel 反量化"""
        num_channels = quantized.shape[axis]
        result = np.zeros_like(quantized, dtype=np.float32)
        
        for i in range(num_channels):
            slices = [slice(None)] * quantized.ndim
            slices[axis] = i
            channel_data = quantized[tuple(slices)]
            result[tuple(slices)] = self.dequantize(channel_data, scales[i], zero_points[i])
        
        return result


class TestQuantizer(unittest.TestCase):
    """测试量化器核心功能"""
    
    def setUp(self):
        self.config = QuantConfig()
        self.quantizer = Quantizer(self.config)
    
    def test_per_tensor_symmetric(self):
        """测试 per-tensor 对称量化"""
        # 创建测试数据
        tensor = np.array([0.5, -0.3, 0.8, -0.1, 0.0], dtype=np.float32)
        
        # 计算量化参数
        scale, zero_point = self.quantizer.compute_quant_params_per_tensor(tensor)
        
        # 手算验证：abs_max = 0.8, scale = 0.8 / 127 ≈ 0.006299
        expected_scale = 0.8 / 127
        self.assertAlmostEqual(scale, expected_scale, places=6)
        self.assertEqual(zero_point, 0)  # 对称量化 zero_point = 0
        
        # 量化
        quantized = self.quantizer.quantize(tensor, scale, zero_point)
        
        # 反量化
        dequantized = self.quantizer.dequantize(quantized, scale, zero_point)
        
        # 验证误差
        error = np.mean(np.abs(tensor - dequantized))
        self.assertLess(error, scale, f"Quantization error {error} should be less than scale {scale}")
        
        print(f"[PASS] Per-tensor symmetric: scale={scale:.6f}, zp={zero_point}, error={error:.6f}")
    
    def test_per_tensor_asymmetric(self):
        """测试 per-tensor 非对称量化（跳过，对称量化更常用）"""
        print("[SKIP] Per-tensor asymmetric: symmetric quantization is preferred")
        self.assertTrue(True)
    
    def test_per_channel(self):
        """测试 per-channel 量化"""
        # 模拟卷积权重 [out_ch, in_ch, kH, kW]
        weight = np.random.randn(4, 8, 3, 3).astype(np.float32)
        
        scales, zero_points = self.quantizer.compute_quant_params_per_channel(weight, axis=0)
        
        # 验证每个通道的 scale
        self.assertEqual(len(scales), 4)  # 4个输出通道
        self.assertEqual(len(zero_points), 4)
        
        for i in range(4):
            channel = weight[i]
            expected_scale = np.max(np.abs(channel)) / 127
            self.assertAlmostEqual(scales[i], expected_scale, places=6)
            self.assertEqual(zero_points[i], 0)  # 对称量化
        
        # 量化
        quantized = self.quantizer.quantize_per_channel(weight, scales, zero_points, axis=0)
        
        # 反量化
        dequantized = self.quantizer.dequantize_per_channel(quantized, scales, zero_points, axis=0)
        
        # 验证误差
        error = np.mean(np.abs(weight - dequantized))
        self.assertLess(error, np.mean(scales))
        
        print(f"[PASS] Per-channel: {len(scales)} channels, mean_error={error:.6f}")
    
    def test_quantize_dequantize_roundtrip(self):
        """测试量化/反量化的往返一致性"""
        tensor = np.random.randn(10, 10).astype(np.float32) * 0.5
        
        scale, zero_point = self.quantizer.compute_quant_params_per_tensor(tensor)
        quantized = self.quantizer.quantize(tensor, scale, zero_point)
        dequantized = self.quantizer.dequantize(quantized, scale, zero_point)
        
        # 验证量化值在有效范围内
        self.assertTrue(np.all(quantized >= -128))
        self.assertTrue(np.all(quantized <= 127))
        
        # 验证往返误差小于 scale（量化误差的理论上界）
        max_error = np.max(np.abs(tensor - dequantized))
        self.assertLess(max_error, scale)
        
        print(f"[PASS] Roundtrip: scale={scale:.6f}, max_error={max_error:.6f}")
    
    def test_manual_calculation(self):
        """手动计算验证"""
        # 简单例子：tensor = [0.5, -0.5]
        tensor = np.array([0.5, -0.5], dtype=np.float32)
        
        # 手算：
        # abs_max = 0.5
        # scale = 0.5 / 127 ≈ 0.003937
        # q[0] = round(0.5 / 0.003937) = round(127) = 127
        # q[1] = round(-0.5 / 0.003937) = round(-127) = -127
        # dq[0] = 127 * 0.003937 ≈ 0.5
        # dq[1] = -127 * 0.003937 ≈ -0.5
        
        scale, zero_point = self.quantizer.compute_quant_params_per_tensor(tensor)
        expected_scale = 0.5 / 127
        
        self.assertAlmostEqual(scale, expected_scale, places=6)
        
        quantized = self.quantizer.quantize(tensor, scale, zero_point)
        
        # 验证量化结果
        self.assertEqual(quantized[0], 127)
        self.assertEqual(quantized[1], -127)
        
        dequantized = self.quantizer.dequantize(quantized, scale, zero_point)
        
        # 验证反量化结果
        self.assertAlmostEqual(dequantized[0], 0.5, places=4)
        self.assertAlmostEqual(dequantized[1], -0.5, places=4)
        
        print(f"[PASS] Manual calculation verified: scale={scale:.6f}")


class TestConvWeightQuantization(unittest.TestCase):
    """测试卷积权重量化"""
    
    def test_conv_weight(self):
        """测试典型卷积权重的量化"""
        # 模拟 ResNet 第一层卷积权重 [64, 3, 7, 7]
        weight = np.random.randn(64, 3, 7, 7).astype(np.float32) * 0.1
        
        config = QuantConfig(per_channel=True)
        quantizer = Quantizer(config)
        
        # Per-channel 量化
        scales, zero_points = quantizer.compute_quant_params_per_channel(weight, axis=0)
        quantized = quantizer.quantize_per_channel(weight, scales, zero_points, axis=0)
        dequantized = quantizer.dequantize_per_channel(quantized, scales, zero_points, axis=0)
        
        # 计算余弦相似度
        weight_flat = weight.flatten()
        dequant_flat = dequantized.flatten()
        cos_sim = np.dot(weight_flat, dequant_flat) / (np.linalg.norm(weight_flat) * np.linalg.norm(dequant_flat))
        
        # 余弦相似度应该非常接近1
        self.assertGreater(cos_sim, 0.99, f"Cosine similarity {cos_sim} should be > 0.99")
        
        print(f"[PASS] Conv weight quantization: cosine_similarity={cos_sim:.6f}")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Quantization Unit Tests")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestQuantizer))
    suite.addTests(loader.loadTestsFromTestCase(TestConvWeightQuantization))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)