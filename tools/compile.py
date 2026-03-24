"""
AI Model Compiler - 主编译入口 CLI

新增 --calib-dir 参数：指向包含校准图片的目录（jpg/png），
编译器自动预处理图片并传给 PTQ pass 做激活校准。

用法示例
--------
# 全精度（不变）
python compile.py resnet18.onnx -t generic -o build/generic

# 量化 + 激活校准（推荐）
python compile.py resnet18.onnx -t generic -o build/quantized -q --calib-dir ./calib_images

# 量化（weight-only 退化模式，兼容旧行为）
python compile.py resnet18.onnx -t generic -o build/quantized -q
"""
import sys
import os
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from my_ai_compiler.frontend.onnx_parser import parse_onnx
from my_ai_compiler.optimizer.pass_manager import PassManager

import my_ai_compiler.optimizer.passes
import my_ai_compiler.optimizer.passes.fusion
import my_ai_compiler.optimizer.passes.rewrite
import my_ai_compiler.optimizer.passes.quantization

from my_ai_compiler.backend.emitter import emit_c_code
from my_ai_compiler.backend.targets import get_target


# ── ImageNet 标准预处理 ─────────────────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_path: str, input_size: int = 224) -> np.ndarray:
    """
    标准 ImageNet 预处理：resize(256) → center_crop(224) → normalize → NCHW。
    返回 shape=(1,3,224,224), dtype=float32。
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("需要 Pillow：pip install pillow")

    img = Image.open(image_path).convert("RGB")

    # Resize 短边到 256
    scale = 256 / min(img.width, img.height)
    new_w, new_h = int(round(img.width * scale)), int(round(img.height * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop
    left = (new_w - input_size) // 2
    top  = (new_h - input_size) // 2
    img  = img.crop((left, top, left + input_size, top + input_size))

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]   # (1,3,H,W)
    return arr.astype(np.float32)


def load_calib_images(calib_dir: str, max_samples: int = 32) -> List[np.ndarray]:
    """从目录加载校准图片，返回预处理后的列表。"""
    calib_path = Path(calib_dir)
    if not calib_path.exists():
        print(f"  WARNING: calib_dir '{calib_dir}' not found, skipping calibration")
        return []

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in calib_path.iterdir() if p.suffix.lower() in exts)[:max_samples]

    if not files:
        print(f"  WARNING: No images found in '{calib_dir}'")
        return []

    print(f"  [Calib] Loading {len(files)} image(s) from '{calib_dir}'...")
    inputs = []
    for p in files:
        try:
            inputs.append(preprocess_image(str(p)))
        except Exception as e:
            print(f"  [Calib] Skipping {p.name}: {e}")

    print(f"  [Calib] {len(inputs)} calibration sample(s) ready")
    return inputs


# ── 主编译函数 ──────────────────────────────────────────────────────────

def compile_model(
    model_path: str,
    output_dir: str,
    model_name: str = "model",
    target_name: str = "generic",
    enable_quantize: bool = False,
    calib_image_dir: Optional[str] = None,
    calib_max_samples: int = 32,
) -> bool:
    if output_dir == "output":
        output_dir = f"build/{target_name}"

    print("=" * 60)
    print("AI Model Compiler")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 解析 ONNX
    # ------------------------------------------------------------------
    print(f"\n[1/4] Parsing ONNX model: {model_path}")
    try:
        graph = parse_onnx(model_path)
        print(f"  Nodes: {len(graph.nodes)}, Tensors: {len(graph.tensors)}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    # ------------------------------------------------------------------
    # 2. 加载校准数据
    # ------------------------------------------------------------------
    calib_inputs: List[np.ndarray] = []
    if enable_quantize:
        if calib_image_dir:
            print(f"\n[2/4] Loading calibration images from: {calib_image_dir}")
            calib_inputs = load_calib_images(calib_image_dir, max_samples=calib_max_samples)
        else:
            print("\n[2/4] No --calib-dir provided → weight-only quantization mode")
            print("      For better accuracy, provide --calib-dir with representative images.")
    else:
        print("\n[2/4] Calibration: skipped (not quantizing)")

    # ------------------------------------------------------------------
    # 3. 注入 QuantConfig（在 PassManager 执行前！）
    # ------------------------------------------------------------------
    if enable_quantize:
        from my_ai_compiler.optimizer.passes.quantization.quant_config import QuantConfig
        from my_ai_compiler.optimizer.passes.quantization.ptq import set_quant_config

        quant_cfg = QuantConfig(
            weight_only      = (len(calib_inputs) == 0),
            calib_onnx_path  = model_path,
            calib_inputs     = calib_inputs if calib_inputs else None,
        )
        set_quant_config(quant_cfg)

    # ------------------------------------------------------------------
    # 4. 优化（含 PTQ）
    # ------------------------------------------------------------------
    print(f"\n[3/4] Optimizing graph...")
    pass_manager = PassManager()

    if target_name == "arm_neon":
        pass_manager.add_pass_by_name("layout_nchw_to_nhwc")

    pass_manager.add_pass_by_name("constant_fold")
    pass_manager.add_pass_by_name("dead_code_elim")
    pass_manager.add_pass_by_name("conv_bn_relu_fusion")
    pass_manager.add_pass_by_name("gemm_fusion")

    if enable_quantize:
        pass_manager.add_pass_by_name("post_training_quantize")

    try:
        graph = pass_manager.run(graph)
        print(f"  Optimized nodes: {len(graph.nodes)}")
    except Exception as e:
        print(f"  WARNING: Optimization failed: {e}")
        import traceback; traceback.print_exc()
        print("  Continuing without optimization...")

    # ------------------------------------------------------------------
    # 5. 生成 C 代码
    # ------------------------------------------------------------------
    print(f"\n[4/4] Generating C code → '{output_dir}' (target: {target_name})")
    try:
        target = get_target(target_name)
        emit_c_code(graph, output_dir, model_name, target)
        print("  C code generation completed!")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("Compilation completed successfully!")
    print("=" * 60)
    return True


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Model Compiler - ONNX to C",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("model", help="Path to ONNX model file")
    parser.add_argument("-o", "--output",  default="output")
    parser.add_argument("-n", "--name",    default="model")
    parser.add_argument("-t", "--target",  default="generic",
                        help="generic / x86_avx / arm_neon")
    parser.add_argument("-q", "--quantize", action="store_true", default=False,
                        help="Enable PTQ int8 quantization")
    parser.add_argument(
        "--calib-dir", default=None,
        help=(
            "Directory of calibration images (jpg/png).\n"
            "Enables full activation quantization (recommended).\n"
            "Without this, falls back to weight-only quantization."
        ),
    )
    parser.add_argument(
        "--calib-samples", type=int, default=32,
        help="Max calibration images to load (default: 32)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        sys.exit(1)

    success = compile_model(
        model_path       = args.model,
        output_dir       = args.output,
        model_name       = args.name,
        target_name      = args.target,
        enable_quantize  = args.quantize,
        calib_image_dir  = args.calib_dir,
        calib_max_samples= args.calib_samples,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
