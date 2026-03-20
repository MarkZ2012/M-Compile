"""
图片预处理脚本 - 为C推理准备输入数据
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def preprocess_image(image_path):
    """预处理图片"""
    # ImageNet预处理参数
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    return input_tensor.numpy()


def main():
    # 设置路径
    # cat.png 和本脚本在同一目录下
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cat.png")
    output_path = "input_data.bin"
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return 1
    
    # 预处理图片
    print(f"Processing image: {image_path}")
    input_data = preprocess_image(image_path)
    
    # 保存为二进制文件
    input_data.astype(np.float32).tofile(output_path)
    
    print(f"Input data saved to: {output_path}")
    print(f"Shape: {input_data.shape}")
    print(f"Size: {input_data.size} floats")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())