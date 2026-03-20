"""
AI Model Compiler - 主编译入口 CLI

调用流程：
    compile.py
        └─> compile_model()
                ├─ frontend/onnx_parser.py   (parse_onnx)
                ├─ optimizer/pass_manager.py (PassManager)
                └─ backend/emitter.py        (emit_c_code)
                        └─ backend/targets/  (GenericCTarget / X86AvxTarget / ArmNeonTarget)
"""
import sys
import os
import argparse

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from my_ai_compiler.frontend.onnx_parser import parse_onnx
from my_ai_compiler.optimizer.pass_manager import PassManager
from my_ai_compiler.optimizer.passes.constant_fold import constant_fold
from my_ai_compiler.optimizer.passes.dead_code_elim import dead_code_elim

# ★ 新入口：backend/emitter.py（替代原 backend/codegen/c_emitter.py）
from my_ai_compiler.backend.emitter import emit_c_code
from my_ai_compiler.backend.targets import get_target


def compile_model(
    model_path: str,
    output_dir: str,
    model_name: str = "model",
    target_name: str = "generic",
) -> bool:
    """
    编译 ONNX 模型为 C 代码。

    Args:
        model_path  : ONNX 模型文件路径
        output_dir  : 输出目录（自动创建），默认为 build/{target_name}
        model_name  : 生成的 C 文件前缀
        target_name : 目标平台，支持 generic / x86_avx / arm_neon

    Returns:
        True 表示编译成功，False 表示出错。
    """
    # 如果未指定输出目录，使用 build/{target_name}
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
    pass_manager.add_function_pass("constant_fold", constant_fold)
    pass_manager.add_function_pass("dead_code_elim", dead_code_elim)

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
    parser.add_argument("-o", "--output",  default="output",  help="Output directory (default: output)")
    parser.add_argument("-n", "--name",    default="model",   help="Model name (default: model)")
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

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        sys.exit(1)

    success = compile_model(args.model, args.output, args.name, args.target)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
