"""
量化 ResNet18 测试脚本

用于测试量化模型的编译和推理，对比浮点和量化模型的结果。

用法：
    python test_quantized_resnet18.py
"""
import numpy as np
import onnxruntime as ort
import subprocess
import os
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class ResNet18QuantizationTest:
    """ResNet18 量化测试类"""
    
    def __init__(self):
        self.image_path = project_root / 'cat.png'
        self.onnx_model = project_root / 'resnet18.onnx'
        self.build_dir = project_root / 'build' / 'generic'
        
    def preprocess_image(self, image_path):
        """预处理图像"""
        from PIL import Image
        
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        
        # Resize 到 256
        img = img.resize((256, 256), Image.BILINEAR)
        
        # Center crop 到 224
        left = (256 - 224) // 2
        top = (256 - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))
        
        # 转换为 numpy 数组
        img_np = np.array(img, dtype=np.float32)
        
        # 归一化到 [0, 1]
        img_np = img_np / 255.0
        
        # ImageNet 标准化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        
        # 转换为 NCHW 格式
        img_np = np.transpose(img_np, (2, 0, 1))
        
        return img_np.astype(np.float32)
    
    def run_float_model_onnx(self):
        """运行浮点模型（ONNX Runtime）"""
        print("=== Running Float Model (ONNX Runtime) ===")
        
        # 预处理图像
        input_data = self.preprocess_image(self.image_path)
        input_data = np.expand_dims(input_data, axis=0)
        
        # 运行 ONNX Runtime
        sess = ort.InferenceSession(str(self.onnx_model))
        outputs = sess.run(None, {'input': input_data})
        
        # 获取 top-5 预测
        probs = outputs[0][0]
        top5_idx = np.argsort(probs)[-5:][::-1]
        
        print("Top-5 predictions:")
        for idx in top5_idx:
            print(f"  Class {idx}: {probs[idx]:.4f}")
        
        return probs
    
    def compile_model(self, quantize=False):
        """编译模型"""
        mode = "quantized" if quantize else "float"
        print(f"\n=== Compiling {mode.upper()} Model ===")
        
        # 编译命令
        cmd = [
            sys.executable, str(project_root / 'tools' / 'compile.py'),
            str(self.onnx_model),
            '-o', str(self.build_dir),
            '-n', 'resnet18',
            '-t', 'generic'
        ]
        
        if quantize:
            cmd.append('-q')
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
        
        if result.returncode != 0:
            print(f"Compilation failed:")
            print(result.stderr)
            return False
        
        print(f"{mode.upper()} model compiled successfully")
        return True
    
    def build_c_executable(self):
        """编译C可执行文件"""
        print(f"\n=== Building C Executable ===")
        
        # 运行 build.py
        build_script = project_root / 'tools' / 'test' / 'example' / 'resnet18' / 'build.py'
        
        if not build_script.exists():
            print(f"Build script not found: {build_script}")
            return False
        
        cmd = [
            sys.executable, str(build_script),
            '--target', 'generic'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
        
        if result.returncode != 0:
            print(f"Build failed:")
            print(result.stderr)
            return False
        
        print("Build completed successfully")
        return True
    
    def run_compiled_model(self):
        """运行编译后的C模型"""
        exe_path = self.build_dir / 'resnet18_test.exe'
        
        if not exe_path.exists():
            print(f"Executable not found: {exe_path}")
            return None
        
        # 运行可执行文件
        result = subprocess.run([str(exe_path)], capture_output=True, text=True, cwd=str(self.build_dir))
        
        if result.returncode != 0:
            print(f"Execution failed:")
            print(result.stderr)
            return None
        
        # 解析输出
        return self.parse_output(result.stdout)
    
    def parse_output(self, output_str):
        """解析C程序的输出"""
        predictions = {}
        
        # 查找 "Class XXX: YY.YY%" 格式
        for line in output_str.split('\n'):
            if 'Class' in line and '%' in line:
                try:
                    # 提取 class id 和概率
                    parts = line.strip().split()
                    for i, part in enumerate(parts):
                        if part == 'Class':
                            class_id = int(parts[i+1].rstrip(':'))
                            # 查找百分比
                            for j in range(i+1, len(parts)):
                                if '%' in parts[j]:
                                    prob = float(parts[j].rstrip('%')) / 100.0
                                    predictions[class_id] = prob
                                    break
                except:
                    continue
        
        return predictions
    
    def compare_results(self, float_probs, c_preds, model_name):
        """比较浮点和C模型结果"""
        print(f"\n=== Comparison: ONNX Runtime vs {model_name} ===")
        
        # 获取浮点模型的top-5
        float_top5_idx = np.argsort(float_probs)[-5:][::-1]
        float_top5 = {idx: float_probs[idx] for idx in float_top5_idx}
        
        # 获取C模型的top-5
        if c_preds:
            c_top5 = dict(sorted(c_preds.items(), key=lambda x: x[1], reverse=True)[:5])
        else:
            c_top5 = {}
        
        print(f"\nONNX Runtime Top-5:")
        for idx, prob in float_top5.items():
            print(f"  Class {idx}: {prob:.4f}")
        
        print(f"\n{model_name} Top-5:")
        for idx, prob in c_top5.items():
            print(f"  Class {idx}: {prob:.4f}")
        
        # 计算分类一致性和精度损失
        if c_top5:
            # Top-1 是否一致
            float_top1 = float_top5_idx[0]
            c_top1 = list(c_top5.keys())[0] if c_top5 else -1
            
            print(f"\nTop-1 Prediction:")
            print(f"  ONNX: Class {float_top1} ({float_top5[float_top1]:.4f})")
            print(f"  {model_name}: Class {c_top1} ({c_top5.get(c_top1, 0):.4f})")
            print(f"  Match: {'✓' if float_top1 == c_top1 else '✗'}")
            
            # 计算Top-5准确率
            top5_match = set(float_top5.keys()) & set(c_top5.keys())
            print(f"  Top-5 Match: {len(top5_match)}/5 classes match")


def main():
    """主函数"""
    print("=" * 60)
    print("ResNet18 Quantization Test")
    print("=" * 60)
    
    tester = ResNet18QuantizationTest()
    
    # 检查必要文件
    if not tester.onnx_model.exists():
        print(f"Error: ONNX model not found: {tester.onnx_model}")
        return 1
    
    if not tester.image_path.exists():
        print(f"Error: Test image not found: {tester.image_path}")
        return 1
    
    # 1. 获取浮点模型的基准结果
    float_probs = tester.run_float_model_onnx()
    
    # 2. 编译和测试浮点模型
    print("\n" + "=" * 60)
    print("Testing Float Model (C)")
    print("=" * 60)
    
    if tester.compile_model(quantize=False):
        if tester.build_c_executable():
            float_preds = tester.run_compiled_model()
            if float_preds:
                tester.compare_results(float_probs, float_preds, "Float C")
    
    # 3. 编译和测试量化模型
    print("\n" + "=" * 60)
    print("Testing Quantized Model (C)")
    print("=" * 60)
    
    if tester.compile_model(quantize=True):
        if tester.build_c_executable():
            quant_preds = tester.run_compiled_model()
            if quant_preds:
                tester.compare_results(float_probs, quant_preds, "Quantized C")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())