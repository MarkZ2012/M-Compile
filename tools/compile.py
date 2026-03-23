"""
AI Model Compiler - 主编译入口 CLI

调用流程：
    compile.py
        └─> compile_model()
                ├─ frontend/onnx_parser.py   (parse_onnx)
                ├─ optimizer/pass_manager.py (PassManager + PassRegistry)
                └─ backend/emitter.py        (emit_c_code)
                        └─ backend/targets/  (GenericCTarget / X86AvxTarget / ArmNeonTarget)

Pass 执行顺序（目标平台决定是否启用各组）：
    [rewrite]       layout_nchw_to_nhwc      （仅 arm_neon）
    [builtin]       constant_fold
    [builtin]       dead_code_elim
    [fusion]        conv_bn_relu_fusion
    [fusion]        gemm_fusion
    [quantization]  post_training_quantize   （仅传入 --quantize 时）
"""
import sys
import os
import argparse

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from my_ai_compiler.frontend.onnx_parser import parse_onnx
from my_ai_compiler.optimizer.pass_manager import PassManager

# ── 导入各 pass 子包：import 动作本身即触发自动注册 ──────────────────────────
import my_ai_compiler.optimizer.passes              # 注册 constant_fold, dead_code_elim
import my_ai_compiler.optimizer.passes.fusion       # 注册 conv_bn_relu_fusion, gemm_fusion
import my_ai_compiler.optimizer.passes.rewrite      # 注册 layout_nchw_to_nhwc
import my_ai_compiler.optimizer.passes.quantization # 注册 post_training_quantize, qat_fold

from my_ai_compiler.backend.emitter import emit_c_code
from my_ai_compiler.backend.targets import get_target


def compile_model(
    model_path: str,
    output_dir: str,
    model_name: str = "model",
    target_name: str = "generic",
    enable_quantize: bool = False,
) -> bool:
    """
    编译 ONNX 模型为 C 代码。

    Args:
        model_path       : ONNX 模型文件路径
        output_dir       : 输出目录（自动创建）
        model_name       : 生成的 C 文件前缀
        target_name      : 目标平台，支持 generic / x86_avx / arm_neon
        enable_quantize  : 是否启用 PTQ int8 量化（默认关闭）

    Returns:
        True 表示编译成功，False 表示出错。
    """
    if output_dir == "output":
        output_dir = f"build/{target_name}"

    print("=" * 60)
    print("AI Model Compiler")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 解析 ONNX 模型
    # ------------------------------------------------------------------
    print(f"\n[1/3] Parsing ONNX model: {model_path}")
    try:
        graph = parse_onnx(model_path)
        print(f"  Graph  : {graph}")
        print(f"  Nodes  : {len(graph.nodes)}")
        print(f"  Tensors: {len(graph.tensors)}")
        print(f"  Inputs : {graph.inputs}")
        print(f"  Outputs: {graph.outputs}")
    except Exception as e:
        print(f"  ERROR: Failed to parse ONNX model: {e}")
        return False

    # ------------------------------------------------------------------
    # 2. 优化
    # ------------------------------------------------------------------
    print(f"\n[2/3] Optimizing graph...")
    pass_manager = PassManager()

    # ── rewrite：layout 变换必须第一个跑，且仅 arm_neon 需要 ──────────
    if target_name == "arm_neon":
        pass_manager.add_pass_by_name("layout_nchw_to_nhwc")

    # ── 内置 pass（原有逻辑，顺序不变）──────────────────────────────────
    pass_manager.add_pass_by_name("constant_fold")
    pass_manager.add_pass_by_name("dead_code_elim")

    # ── fusion（constant_fold 之后，BN 参数已为常量）──────────────────
    pass_manager.add_pass_by_name("conv_bn_relu_fusion")
    pass_manager.add_pass_by_name("gemm_fusion")

    # ── 量化（可选，fusion 之后面对已融合的算子）──────────────────────
    if enable_quantize:
        pass_manager.add_pass_by_name("post_training_quantize")

    try:
        graph = pass_manager.run(graph)
        print(f"  Optimized nodes: {len(graph.nodes)}")
    except Exception as e:
        print(f"  WARNING: Optimization failed: {e}")
        print("  Continuing without optimization...")

    # ------------------------------------------------------------------
    # 3. 生成 C 代码
    # ------------------------------------------------------------------
    print(f"\n[3/3] Generating C code to '{output_dir}' (target: {target_name})")
    try:
        target = get_target(target_name)
        emit_c_code(graph, output_dir, model_name, target)
        print("  C code generation completed!")
    except Exception as e:
        print(f"  ERROR: Failed to generate C code: {e}")
        return False

    print("\n" + "=" * 60)
    print("Compilation completed successfully!")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="AI Model Compiler - ONNX to C",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("model", help="Path to ONNX model file")
    parser.add_argument("-o", "--output",   default="output", help="Output directory (default: output)")
    parser.add_argument("-n", "--name",     default="model",  help="Model name (default: model)")
    parser.add_argument(
        "-t", "--target",
        default="generic",
        help=(
            "Target platform (default: generic)\n"
            "  generic   -> generic C, works on any x64\n"
            "  x86_avx   -> x86 with AVX/AVX2 SIMD\n"
            "  arm_neon  -> ARM with NEON SIMD"
        ),
    )
    parser.add_argument(
        "-q", "--quantize",
        action="store_true",
        default=False,
        help="Enable Post-Training Quantization (int8 weights)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        sys.exit(1)

    success = compile_model(
        args.model,
        args.output,
        args.name,
        args.target,
        enable_quantize=args.quantize,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
