"""
图片预处理脚本 - 为 MobileNetV2 C 推理准备输入数据
输出：同目录下（或运行目录）的 `input_data.bin`

说明：
- 尽量复用现有 ResNet18 的预处理逻辑（ImageNet mean/std）。
- 不强依赖本目录下存在 cat.png：如果本目录没有 cat.png，会从
  `my_ai_compiler/tools/test/example/resnet18/cat.png` 取一份。
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def _guess_project_root() -> Path:
    # 兼容两种运行方式：
    # 1) 在示例目录运行：.../tools/test/example/MobileNetV2/preprocess_image.py
    # 2) 被 build.py 复制到 build/generic 运行：.../build/generic/preprocess_image.py
    p = Path(__file__).resolve()
    for up in range(0, 8):
        cand = p.parents[up] if up < len(p.parents) else None
        if cand is not None and (cand / "my_ai_compiler").exists():
            return cand
    return p.parents[0]


def preprocess_image(image_path: str) -> np.ndarray:
    """ImageNet 预处理：resize(256) -> centerCrop(224) -> normalize -> NCHW"""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image)  # (3,224,224)
    return input_tensor.numpy()


def main() -> int:
    project_root = _guess_project_root()
    default_cat = Path(__file__).resolve().parent / "cat.png"
    fallback_cat = project_root / "my_ai_compiler" / "tools" / "test" / "example" / "resnet18" / "cat.png"

    if default_cat.exists():
        image_path = str(default_cat)
    elif fallback_cat.exists():
        image_path = str(fallback_cat)
    else:
        print(f"Error: cannot find cat.png. Tried:\n  {default_cat}\n  {fallback_cat}")
        return 1

    output_path = "input_data.bin"
    print(f"Processing image: {image_path}")
    input_data = preprocess_image(image_path).astype(np.float32)
    input_data.astype(np.float32).tofile(output_path)
    print(f"Input data saved to: {output_path}")
    print(f"Shape: {input_data.shape}")
    print(f"Size: {input_data.size} floats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

