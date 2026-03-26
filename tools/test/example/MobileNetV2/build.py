"""
MobileNetV2 Test Compilation Script
=================================
用于编译/运行 `mobilenetv2_test.c`

依赖：
  - build/mobilenetv2_generic/ （由 my_ai_compiler/tools/compile.py 生成）
  - runtime/ops/generic/ （拷贝到 build 目录下的 ops/generic）
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path


script_dir = Path(__file__).parent
# 需要向上 5 级到项目根目录 M-Compile
project_root = script_dir.parent.parent.parent.parent.parent

build_dir = project_root / "build" / "mobilenetv2_generic"
runtime_ops_dir = project_root / "my_ai_compiler" / "runtime" / "ops" / "generic"
test_file = script_dir / "mobilenetv2_test.c"
output_exe = script_dir / "mobilenetv2_test.exe"


def copy_files_for_target(source_dir=None):
    global build_dir

    if source_dir is None:
        source_dir = project_root / "output"
    elif isinstance(source_dir, str):
        if source_dir in ["float", "generic", "quantized"]:
            source_dir = project_root / "build" / source_dir
        else:
            source_dir = Path(source_dir)

    build_dir = source_dir

    # 如果源目录不存在，尝试重新编译
    if not source_dir.exists():
        print(f"源目录不存在: {source_dir}")
        print("尝试重新编译模型...")
        compile_script = project_root / "my_ai_compiler" / "tools" / "compile.py"
        cmd = [
            sys.executable,
            str(compile_script),
            "my_ai_compiler/mobilenetv2.onnx",
            "--target",
            "generic",
            "--output",
            str(build_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
        if result.returncode != 0:
            print(f"编译失败: {result.stderr}")
            return False
        print("编译成功!")

    print("=" * 60)
    print(f"拷贝文件到 {build_dir} 目录")
    print(f"源目录: {source_dir}")
    print("=" * 60)

    build_dir.mkdir(parents=True, exist_ok=True)

    # 1) 拷贝测试文件到 build_dir
    files_to_copy = ["mobilenetv2_test.c", "preprocess_image.py"]
    for filename in files_to_copy:
        src_file = script_dir / filename
        dst_file = build_dir / filename
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"  已拷贝: {filename}")
        else:
            print(f"  跳过（不存在）: {filename}")

    # 2) 拷贝运行时 ops
    dst_ops_dir = build_dir / "ops" / "generic"
    if runtime_ops_dir.exists():
        if dst_ops_dir.exists():
            shutil.rmtree(dst_ops_dir)
        dst_ops_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(runtime_ops_dir, dst_ops_dir)
        print(f"  已拷贝 runtime/ops/generic -> {dst_ops_dir}")
    else:
        print(f"  错误: 源目录不存在 {runtime_ops_dir}")
        return False

    # 3) 检查 model.c/model.h + weights 目录
    has_model_files = (source_dir / "model.c").exists() and (source_dir / "model.h").exists()
    if not has_model_files and source_dir != build_dir:
        # 如果 compile 产物在其它目录，交由 compile.py 逻辑处理
        print("  警告: model.c/model.h 不在 source_dir 中，可能会导致编译失败。")

    return True


def check_dependencies(use_build_dir=False):
    errors = []
    if use_build_dir:
        current_build_dir = build_dir
        current_runtime_ops_dir = build_dir / "ops" / "generic"
        current_test_file = build_dir / "mobilenetv2_test.c"
    else:
        current_build_dir = build_dir
        current_runtime_ops_dir = runtime_ops_dir
        current_test_file = test_file

    if not current_test_file.exists():
        errors.append(f"测试文件不存在: {current_test_file}")

    model_c = current_build_dir / "model.c"
    model_h = current_build_dir / "model.h"
    if not model_c.exists():
        errors.append(f"模型代码不存在: {model_c}")
    if not model_h.exists():
        errors.append(f"模型头文件不存在: {model_h}")

    weights_dir = current_build_dir / "weights"
    if not weights_dir.exists():
        errors.append(f"权重目录不存在: {weights_dir}")

    if not current_runtime_ops_dir.exists():
        errors.append(f"算子目录不存在: {current_runtime_ops_dir}")

    ops_files = [
        "activations.c",
        "conv2d.c",
        "linear.c",
        "pooling.c",
        "softmax.c",
        "math_ops.c",
        "batchnorm.c",
    ]
    for ops_file in ops_files:
        ops_path = current_runtime_ops_dir / ops_file
        if not ops_path.exists():
            errors.append(f"算子文件不存在: {ops_path}")

    return errors


def compile_test(use_build_dir=False):
    print("=" * 60)
    print("MobileNetV2 Test Compilation Script")
    print("=" * 60)

    if use_build_dir:
        current_build_dir = build_dir
        current_runtime_ops_dir = build_dir / "ops" / "generic"
        current_test_file = build_dir / "mobilenetv2_test.c"
        current_output_exe = build_dir / "mobilenetv2_test.exe"
    else:
        current_build_dir = build_dir
        current_runtime_ops_dir = runtime_ops_dir
        current_test_file = test_file
        current_output_exe = output_exe

    print("\n[1/4] 检查依赖文件...")
    errors = check_dependencies(use_build_dir)
    if errors:
        print("错误: 缺少以下依赖文件:")
        for e in errors:
            print(f"  - {e}")
        return False
    print("  所有依赖文件检查通过!")

    print("\n[2/4] 收集源文件...")
    ops_files = [
        "activations.c",
        "conv2d.c",
        "linear.c",
        "pooling.c",
        "softmax.c",
        "math_ops.c",
        "batchnorm.c",
    ]
    source_files = [str(current_test_file), str(current_build_dir / "model.c")]
    for ops_file in ops_files:
        source_files.append(str(current_runtime_ops_dir / ops_file))

    print("\n[3/4] 配置编译参数...")
    include_dirs = [
        f"-I{current_build_dir}",          # so "ops/generic/ops.h" resolves
        f"-I{current_runtime_ops_dir}",  # so "ops.h" resolves for runtime .c compilation units
    ]

    compile_flags = ["-o", str(current_output_exe), "-lm"]
    compile_cmd = ["gcc"] + source_files + include_dirs + compile_flags
    print(f"  编译命令: {' '.join(compile_cmd)}")

    print("\n[4/4] 执行编译...")
    result = subprocess.run(
        compile_cmd,
        capture_output=True,
        text=True,
        cwd=str(project_root),
    )
    if result.returncode == 0:
        print("  编译成功!")
        print(f"  输出文件: {current_output_exe}")
        return True

    print("  编译失败!")
    print(result.stderr)
    return False


def run_test(use_build_dir=False):
    print("\n" + "=" * 60)
    print("运行 MobileNetV2 测试程序")
    print("=" * 60)

    if use_build_dir:
        exe_path = build_dir / "mobilenetv2_test.exe"
        run_dir = build_dir
    else:
        exe_path = output_exe
        run_dir = script_dir

    if not exe_path.exists():
        print(f"错误: 可执行文件不存在 {exe_path}")
        return False

    input_file = run_dir / "input_data.bin"
    if not input_file.exists():
        print(f"警告: 输入数据文件不存在 {input_file}")
        print("  请先运行 'python preprocess_image.py' 生成输入数据")
        return True

    result = subprocess.run(
        [str(exe_path)],
        capture_output=True,
        text=True,
        cwd=str(run_dir),
        timeout=120,
    )
    print(result.stdout)
    if result.returncode == 0:
        print("测试程序执行成功!")
        return True

    print("测试程序执行失败!")
    if result.stderr:
        print(result.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(description="MobileNetV2 Test Compilation Script")
    parser.add_argument("--target", type=str, help="目标平台 (例如: generic)")
    parser.add_argument("--source", type=str, help="源目录 (例如: float, generic, quantized)")
    parser.add_argument("--no-run", action="store_true", help="编译后不运行测试")
    args = parser.parse_args()

    use_build_dir = False
    if args.target:
        if args.target == "generic":
            if not copy_files_for_target(source_dir=args.source):
                print("\n文件拷贝失败!")
                sys.exit(1)
            use_build_dir = True
        else:
            print(f"不支持的目标平台: {args.target}")
            sys.exit(1)

    compile_success = compile_test(use_build_dir=use_build_dir)
    if not compile_success:
        print("\n编译失败!")
        sys.exit(1)

    test_success = True
    if not args.no_run:
        test_success = run_test(use_build_dir=use_build_dir)

    sys.exit(0 if (compile_success and test_success) else 1)


if __name__ == "__main__":
    main()

