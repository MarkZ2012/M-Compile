"""
精度对比测试：float32 vs PTQ int8（PyTorch端验证）
在编译器 emitter 支持 int8 之前，用 PyTorch 量化作为精度基线参考
"""
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path

# ImageNet 猫类标签
CAT_LABELS = {
    281: "tabby cat", 282: "tiger cat", 283: "Persian cat",
    284: "Siamese cat", 285: "Egyptian cat",
}

def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

def topk_result(model, inp, k=5):
    with torch.no_grad():
        out = model(inp)
    probs = torch.softmax(out[0], dim=0)
    vals, idxs = torch.topk(probs, k)
    return [(int(i), float(v)) for i, v in zip(idxs, vals)]

def print_results(label, results):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for idx, prob in results:
        tag = " ← 🐱" if idx in CAT_LABELS else ""
        name = CAT_LABELS.get(idx, f"class_{idx}")
        print(f"  [{idx:4d}] {name:30s} {prob*100:6.2f}%{tag}")

# ── float32 baseline ──────────────────────────────────────────
print("加载 float32 ResNet18 ...")
model_fp32 = models.resnet18(pretrained=True).eval()

project_root = Path(__file__).parent.parent.parent
image_path = project_root / "build" / "generic" / "cat.png"
if not image_path.exists():
    # fallback
    image_path = Path(__file__).parent / "test" / "example" / "resnet18" / "cat.png"

inp = preprocess(str(image_path))
results_fp32 = topk_result(model_fp32, inp, k=10)
print_results("Float32 推理结果", results_fp32)

top1_fp32 = results_fp32[0][0]

# ── PTQ int8（PyTorch 动态量化）─────────────────────────────
print("\n\n对 float32 模型做 PTQ 动态量化 ...")
model_ptq = models.resnet18(pretrained=True).eval()
model_ptq_int8 = torch.quantization.quantize_dynamic(
    model_ptq,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

results_ptq = topk_result(model_ptq_int8, inp, k=10)
print_results("PTQ int8 推理结果", results_ptq)

top1_ptq = results_ptq[0][0]

# ── 精度对比 ──────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  精度对比摘要")
print(f"{'='*50}")
print(f"  Float32 Top-1: [{top1_fp32}] {CAT_LABELS.get(top1_fp32, top1_fp32)}")
print(f"  PTQ int8 Top-1: [{top1_ptq}] {CAT_LABELS.get(top1_ptq, top1_ptq)}")
print(f"  Top-1 一致: {'✅ 是' if top1_fp32 == top1_ptq else '❌ 不同'}")

# top-5 重叠率
top5_fp32 = {r[0] for r in results_fp32[:5]}
top5_ptq  = {r[0] for r in results_ptq[:5]}
overlap = len(top5_fp32 & top5_ptq)
print(f"  Top-5 重叠: {overlap}/5")

# 概率分布差异
probs_fp32 = np.array([r[1] for r in results_fp32[:5]])
probs_ptq  = np.array([r[1] for r in results_ptq[:5]])
print(f"  Top-5 概率 L1 误差: {np.abs(probs_fp32 - probs_ptq).mean()*100:.3f}%")

# 模型体积对比
fp32_size = sum(p.numel() * 4 for p in model_fp32.parameters()) / 1024 / 1024
int8_size = sum(
    p.numel() * 1 if p.dtype == torch.qint8 else p.numel() * 4
    for p in model_ptq_int8.parameters()
) / 1024 / 1024
print(f"\n  Float32 权重大小: {fp32_size:.1f} MB")
print(f"  PTQ int8 权重大小: ~{fp32_size/4:.1f} MB (理论)")
print(f"{'='*50}")

# 保存结果
result = {
    "fp32_top5": results_fp32[:5],
    "ptq_top5": results_ptq[:5],
    "top1_match": top1_fp32 == top1_ptq,
    "top5_overlap": overlap,
}
with open("accuracy_compare_result.json", "w") as f:
    json.dump(result, f, indent=2)
print("\n结果已保存到 accuracy_compare_result.json")