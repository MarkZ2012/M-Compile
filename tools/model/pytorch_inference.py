"""
PyTorch推理脚本 - 使用预训练ResNet18进行猫咪种类识别
"""
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os


def load_model():
    """加载预训练的ResNet18模型"""
    print("Loading ResNet18 model...")
    model = models.resnet18(pretrained=True)
    model.eval()
    return model


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
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def get_imagenet_labels():
    """获取ImageNet标签（简化的猫咪相关类别）"""
    # ImageNet中与猫相关的类别 (索引281-285是不同种类的猫)
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
        293: "cheetah, chetah, Acinonyx jubatus",
    }
    return cat_labels


def predict(model, input_batch, top_k=5):
    """进行预测"""
    with torch.no_grad():
        output = model(input_batch)
    
    # 获取top-k预测结果
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    return top_indices.numpy(), top_probs.numpy()


def export_to_onnx(model, output_path):
    """导出模型为ONNX格式"""
    print(f"Exporting model to ONNX: {output_path}")
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("ONNX export completed!")


def main():
    # 设置路径 - 从项目根目录查找cat.png
    import pathlib
    project_root = pathlib.Path(__file__).parent.parent.parent
    image_path = str(project_root / "cat.png")
    onnx_path = str(project_root / "resnet18.onnx")
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # 加载模型
    model = load_model()
    
    # 导出ONNX模型
    export_to_onnx(model, onnx_path)
    
    # 预处理图片
    print(f"\nProcessing image: {image_path}")
    input_batch = preprocess_image(image_path)
    
    # 进行预测
    print("Running inference...")
    top_indices, top_probs = predict(model, input_batch, top_k=10)
    
    # 获取标签
    cat_labels = get_imagenet_labels()
    
    # 显示结果
    print("\n" + "="*60)
    print("PyTorch ResNet18 Inference Results:")
    print("="*60)
    
    found_cat = False
    for idx, prob in zip(top_indices, top_probs):
        if idx in cat_labels:
            found_cat = True
            print(f"  [{idx}] {cat_labels[idx]}: {prob*100:.2f}%")
    
    if not found_cat:
        print("  No cat detected in top predictions.")
        print("\n  Top predictions:")
        for idx, prob in zip(top_indices[:5], top_probs[:5]):
            print(f"    Class {idx}: {prob*100:.2f}%")
    
    print("="*60)
    
    # 保存结果供后续比较
    result = {
        "top_indices": top_indices.tolist(),
        "top_probs": top_probs.tolist(),
        "cat_detected": found_cat
    }
    
    with open("pytorch_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("\nResults saved to pytorch_result.json")
    print(f"ONNX model saved to {onnx_path}")


if __name__ == "__main__":
    main()