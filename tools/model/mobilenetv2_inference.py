"""
PyTorch 推理脚本 - 使用预训练 MobileNetV2 导出 ONNX
=====================================================

仿照项目内 `resnet18_inference.py`：
  1) 加载 torchvision 官方预训练 MobileNetV2（默认）
  2) 导出到 `my_ai_compiler/mobilenetv2.onnx`
  3) 对 `cat.png` 做 top-k 推理并保存 `pytorch_result.json`

注意：
- 如果你的本地 torchvision/网络环境无法下载预训练权重，会自动退回到随机初始化（weights=None / pretrained=False），保证脚本仍可跑通。
"""

import json
import os
import pathlib
import sys

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def load_model():
    """加载 torchvision 官方预训练 MobileNetV2（失败则退回随机初始化）。"""
    print("Loading MobileNetV2 model...")
    try:
        # torchvision >= 0.13
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    except Exception as e:
        print(f"  WARNING: failed to load pretrained weights: {e}")
        print("  Fallback to untrained MobileNetV2 (weights=None / pretrained=False).")
        try:
            model = models.mobilenet_v2(weights=None)
        except TypeError:
            model = models.mobilenet_v2(pretrained=False)

    model.eval()
    return model


def preprocess_image(image_path):
    """ImageNet 预处理：resize(256) → center_crop(224) → normalize → NCHW"""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def get_imagenet_labels():
    """获取 ImageNet 中与猫相关类别（复用 resnet18 的简化映射）。"""
    cat_labels = {
        281: "tabby, tabby cat",
        282: "tiger cat",
        283: "Persian cat",
        284: "Siamese cat, Siamese",
        285: "Egyptian cat",
        286: "cougar, puma, catamount, mountain lion, painter, panther",
        287: "lynx, catamount",
        288: "leopard, Panthera pardus",
        289: "snow leopard, ounce, Panthera uncia",
        290: "jaguar, panther, Panthera onca, Felis onca",
        291: "lion, king of beasts, Panthera leo",
        292: "tiger, Panthera tigris",
        293: "cheetah, cheetah, Acinonyx jubatus",
    }
    return cat_labels


def predict(model, input_batch, top_k=10):
    """进行预测（softmax 后 top-k）。"""
    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_indices = torch.topk(probabilities, top_k)
    return top_indices.numpy(), top_probs.numpy()


def export_to_onnx(model: torch.nn.Module, output_path: str) -> None:
    """导出模型为 ONNX。"""
    print(f"Exporting model to ONNX: {output_path}")
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose=False,
    )
    print("ONNX export completed!")


def main():
    # 某些 Windows/Python 环境下 torch.onnx 导出日志可能含非 GBK 字符，
    # 这里强制 stdout/stderr 使用 utf-8，避免 UnicodeEncodeError。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    project_root = pathlib.Path(__file__).parent.parent.parent
    image_path = project_root / "cat.png"
    onnx_path = project_root / "mobilenetv2.onnx"

    if not os.path.exists(str(image_path)):
        print(f"Error: Image file '{image_path}' not found!")
        return

    model = load_model()
    export_to_onnx(model, str(onnx_path))

    print(f"\nProcessing image: {image_path}")
    input_batch = preprocess_image(str(image_path))

    print("Running inference...")
    top_indices, top_probs = predict(model, input_batch, top_k=10)

    cat_labels = get_imagenet_labels()
    print("\n" + "=" * 60)
    print("PyTorch MobileNetV2 Inference Results:")
    print("=" * 60)

    found_cat = False
    for idx, prob in zip(top_indices, top_probs):
        if idx in cat_labels:
            found_cat = True
            print(f"  [{idx}] {cat_labels[idx]}: {prob * 100:.2f}%")

    if not found_cat:
        print("  No cat detected in top predictions.")
        print("\n  Top predictions:")
        for idx, prob in zip(top_indices[:5], top_probs[:5]):
            print(f"    Class {idx}: {prob * 100:.2f}%")

    print("=" * 60)

    result = {
        "top_indices": top_indices.tolist(),
        "top_probs": top_probs.tolist(),
        "cat_detected": found_cat,
    }
    with open("pytorch_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nResults saved to pytorch_result.json")
    print(f"ONNX model saved to {onnx_path}")


if __name__ == "__main__":
    main()

