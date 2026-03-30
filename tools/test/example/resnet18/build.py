"""
ResNet18 Test Compilation Script
=================================
用于编译 resnet18_test.c 的 Python 脚本
依赖：
  - build/generic/resnet18.c (模型代码)
  - build/generic/weights/ (权重文件)
  - my_ai_compiler/runtime/ops/generic/ (算子实现)

用法：
  python build.py                    # 直接编译（依赖已有文件）
  python build.py --target generic   # 拷贝文件并编译

平台兼容性
----------
本脚本支持 Windows 和 Linux 平台：
  - Windows: 生成 resnet18_test.exe
  - Linux:   生成 resnet18_test
"""

import os
import sys
import subprocess
import shutil
import argparse
import platform
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"
EXE_SUFFIX = ".exe" if IS_WINDOWS else ""

script_dir = Path(__file__).parent
# script_dir: my_ai_compiler/tools/test/example/resnet18
# 需要向上5级到项目根目录 (M-Compile)
# resnet18 -> example -> test -> tools -> my_ai_compiler -> M-Compile
project_root = script_dir.parent.parent.parent.parent.parent

# 定义路径
# build目录在项目根目录下
build_dir = project_root / "build" / "generic"
# runtime在my_ai_compiler目录下
runtime_ops_dir = project_root / "my_ai_compiler" / "runtime" / "ops" / "generic"
test_file = script_dir / "resnet18_test.c"
output_exe = script_dir / f"resnet18_test{EXE_SUFFIX}"


def copy_files_for_target(source_dir=None):
    """拷贝文件到build目录
    
    Args:
        source_dir: 源目录，可以是 'float', 'generic', 'quantized' 或自定义路径
    """
    global build_dir  # 修改全局的 build_dir
    
    # 确定源目录
    if source_dir is None:
        source_dir = project_root / "output"
    elif isinstance(source_dir, str):
        if source_dir in ['float', 'generic', 'quantized']:
            source_dir = project_root / "build" / source_dir
        else:
            source_dir = Path(source_dir)
    
    # 使用与源目录相同的目标目录
    build_dir = source_dir
    
    # 如果源目录不存在，尝试重新编译
    if not source_dir.exists():
        print(f"源目录不存在: {source_dir}")
        print("尝试重新编译模型...")
        # compile.py 在 my_ai_compiler 目录下
        compile_script = project_root / "my_ai_compiler" / "tools" / "compile.py"
        if source_dir.name == "quantized":
            # 运行量化编译
            cmd = ["python", str(compile_script), "my_ai_compiler/resnet18.onnx", 
                   "--target", "generic", "--quantize", "--output", "build/quantized"]
            # 更新 source_dir 到新创建的目录
            source_dir = project_root / "build" / "quantized"
        else:
            # 运行普通编译
            cmd = ["python", str(compile_script), "my_ai_compiler/resnet18.onnx", 
                   "--target", "generic", "--output", "build/generic"]
            # 更新 source_dir 到新创建的目录
            source_dir = project_root / "build" / "generic"
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
        if result.returncode != 0:
            print(f"编译失败: {result.stderr}")
            return False
        print("编译成功!")
    
    print("=" * 60)
    print(f"拷贝文件到 {build_dir} 目录")
    print(f"源目录: {source_dir}")
    print("=" * 60)
    
    # 确保build目录存在
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 拷贝resnet18目录的文件到build_dir
    print(f"\n[1/4] 拷贝 {script_dir} 的文件到 {build_dir}...")
    files_to_copy = ["resnet18_test.c", "cat.png", "preprocess_image.py"]
    for filename in files_to_copy:
        src_file = script_dir / filename
        dst_file = build_dir / filename
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"  已拷贝: {filename}")
        else:
            print(f"  跳过（不存在）: {filename}")
    
    # 2. 拷贝runtime/ops/generic目录到build_dir/ops/generic
    print(f"\n[2/4] 拷贝 {runtime_ops_dir} 到 {build_dir / 'ops' / 'generic'}...")
    dst_ops_dir = build_dir / "ops" / "generic"
    if runtime_ops_dir.exists():
        if dst_ops_dir.exists():
            shutil.rmtree(dst_ops_dir)
        # 创建ops目录
        dst_ops_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(runtime_ops_dir, dst_ops_dir)
        print(f"  已拷贝整个目录: runtime/ops/generic -> {build_dir}/ops/generic")
    else:
        print(f"  错误: 源目录不存在 {runtime_ops_dir}")
        return False
    
    # 3. 检查并从源目录拷贝模型文件
    print(f"\n[3/4] 检查模型文件...")
    # 检查模型文件是否存在为 model.c/model.h
    has_model_files = (source_dir / "model.c").exists() and (source_dir / "model.h").exists()
    
    # 使用 model.c/model.h 直接
    model_files = ["model.c", "model.h"]
    target_model_files = ["model.c", "model.h"]
    
    # 如果源目录和目标目录相同，则跳过拷贝，只检查
    if source_dir == build_dir:
        print(f"  源目录和目标目录相同，检查模型文件...")
        for src_name in model_files:
            src_file = source_dir / src_name
            if src_file.exists():
                print(f"  模型文件存在: {src_name}")
            else:
                print(f"  警告: 模型文件不存在 {src_file}")
    else:
        # 从源目录拷贝模型文件
        for src_name, dst_name in zip(model_files, target_model_files):
            src_file = source_dir / src_name
            dst_file = build_dir / dst_name
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                print(f"  从{source_dir.name}拷贝: {src_name} -> {dst_name}")
            else:
                print(f"  警告: 模型文件不存在 {src_file}")
    
    # 4. 检查并从源目录拷贝权重目录
    src_weights_dir = source_dir / "weights"
    dst_weights_dir = build_dir / "weights"
    if source_dir == build_dir:
        print(f"  源目录和目标目录相同，跳过权重目录拷贝")
        if src_weights_dir.exists():
            print(f"  权重目录存在: {src_weights_dir}")
        else:
            print(f"  警告: 权重目录不存在 {src_weights_dir}")
    else:
        if src_weights_dir.exists():
            if dst_weights_dir.exists():
                shutil.rmtree(dst_weights_dir)
            shutil.copytree(src_weights_dir, dst_weights_dir)
            print(f"  从{source_dir.name}拷贝: weights/ 目录")
        else:
            print(f"  警告: 权重目录不存在 {src_weights_dir}")
    
    print("\n[4/4] 文件拷贝完成!")
    return True


def check_dependencies(use_build_dir=False):
    """检查所有依赖文件是否存在"""
    errors = []
    
    # 根据模式确定使用的目录
    if use_build_dir:
        current_build_dir = build_dir
        current_runtime_ops_dir = build_dir / "ops" / "generic"
        current_test_file = build_dir / "resnet18_test.c"
    else:
        current_build_dir = build_dir
        current_runtime_ops_dir = runtime_ops_dir
        current_test_file = test_file
    
    # 检查测试文件
    if not current_test_file.exists():
        errors.append(f"测试文件不存在: {current_test_file}")
    
    # 检查build目录下的模型代码
    model_c = current_build_dir / "model.c"
    model_h = current_build_dir / "model.h"
    if not model_c.exists():
        errors.append(f"模型代码不存在: {model_c}")
    if not model_h.exists():
        errors.append(f"模型头文件不存在: {model_h}")
    
    # 检查权重目录
    weights_dir = current_build_dir / "weights"
    if not weights_dir.exists():
        errors.append(f"权重目录不存在: {weights_dir}")
    
    # 检查ops目录
    if not current_runtime_ops_dir.exists():
        errors.append(f"算子目录不存在: {current_runtime_ops_dir}")
    
    # 检查算子文件
    ops_files = ["activations.c", "conv2d.c", "linear.c", "pooling.c", "softmax.c", "math_ops.c", "batchnorm.c"]
    for ops_file in ops_files:
        ops_path = current_runtime_ops_dir / ops_file
        if not ops_path.exists():
            errors.append(f"算子文件不存在: {ops_path}")
    
    # 检查量化算子文件（可选）
    quant_ops_files = ["quant_conv2d.c", "quant_linear.c"]
    for ops_file in quant_ops_files:
        ops_path = current_runtime_ops_dir / ops_file
        if not ops_path.exists():
            print(f"  警告: 量化算子文件不存在（如果需要量化请检查）: {ops_path}")
    
    return errors


def compile_test(use_build_dir=False):
    """编译测试程序
    
    Args:
        use_build_dir: 是否使用build/generic目录下的文件进行编译
    """
    print("=" * 60)
    print("ResNet18 Test Compilation Script")
    print("=" * 60)
    
    # 根据参数确定使用的目录
    if use_build_dir:
        current_build_dir = build_dir
        current_runtime_ops_dir = build_dir / "ops" / "generic"
        current_test_file = build_dir / "resnet18_test.c"
        current_output_exe = build_dir / f"resnet18_test{EXE_SUFFIX}"
    else:
        current_build_dir = build_dir
        current_runtime_ops_dir = runtime_ops_dir
        current_test_file = test_file
        current_output_exe = output_exe
    
    # 检查依赖
    print("\n[1/4] 检查依赖文件...")
    errors = check_dependencies(use_build_dir)
    
    if errors:
        print("错误: 缺少以下依赖文件:")
        for error in errors:
            print(f"  - {error}")
        return False
    print("  所有依赖文件检查通过!")
    
    # 收集源文件
    print("\n[2/4] 收集源文件...")
    ops_files = ["activations.c", "conv2d.c", "linear.c", "pooling.c", "softmax.c", "math_ops.c", "batchnorm.c"]
    quant_ops_files = ["quant_conv2d.c", "quant_linear.c"]
    
    source_files = [
        str(current_test_file),
        str(current_build_dir / "model.c"),
    ]
    
    # 添加算子文件
    for ops_file in ops_files:
        source_files.append(str(current_runtime_ops_dir / ops_file))
    
    # 添加量化算子文件（如果存在）
    quant_ops_found = 0
    for ops_file in quant_ops_files:
        ops_path = current_runtime_ops_dir / ops_file
        if ops_path.exists():
            source_files.append(str(ops_path))
            quant_ops_found += 1
    
    print(f"  找到 {len(source_files)} 个源文件")
    if quant_ops_found > 0:
        print(f"  包含 {quant_ops_found} 个量化算子文件")
    
    # 设置编译参数
    print("\n[3/4] 配置编译参数...")
    if use_build_dir:
        # 当使用build目录时，include路径需要包含：
        # 1. build/generic - 包含 resnet18.h
        # 2. build/generic/ops/generic - 包含 ops.h
        include_dirs = [
            f"-I{current_build_dir}",
            f"-I{current_runtime_ops_dir}",
        ]
    else:
        include_dirs = [
            f"-I{current_build_dir}",
            f"-I{current_runtime_ops_dir}",
        ]
    
    compile_flags = [
        "-o", str(current_output_exe),
        "-lm",  # 链接数学库
    ]
    
    # 构建编译命令
    compile_cmd = ["gcc"] + source_files + include_dirs + compile_flags
    print(f"  编译命令: {' '.join(compile_cmd)}")
    
    # 执行编译
    print("\n[4/4] 执行编译...")
    try:
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )
        
        if result.returncode == 0:
            print("  编译成功!")
            print(f"  输出文件: {current_output_exe}")
            return True
        else:
            print("  编译失败!")
            print("  错误信息:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"  编译过程中发生错误: {e}")
        return False


def run_test(use_build_dir=False):
    """运行测试程序"""
    print("\n" + "=" * 60)
    print("运行测试程序")
    print("=" * 60)
    
    # 根据模式确定可执行文件和运行目录
    if use_build_dir:
        exe_path = build_dir / f"resnet18_test{EXE_SUFFIX}"
        run_dir = build_dir
    else:
        exe_path = output_exe
        run_dir = script_dir
    
    if not exe_path.exists():
        print(f"错误: 可执行文件不存在 {exe_path}")
        return False
    
    # 检查输入数据文件
    input_file = run_dir / "input_data.bin"
    if not input_file.exists():
        print(f"警告: 输入数据文件不存在 {input_file}")
        print("  请先运行 'python preprocess_image.py' 生成输入数据")
        print("  跳过测试执行")
        return True  # 不算作失败，只是跳过
    
    print(f"\n执行: {exe_path}")
    print(f"工作目录: {run_dir}")
    print()
    
    try:
        result = subprocess.run(
            [str(exe_path)],
            capture_output=True,
            text=True,
            cwd=str(run_dir),
            timeout=60  # 60秒超时
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("\n测试程序执行成功!")
            return True
        else:
            print("\n测试程序执行失败!")
            if result.stderr:
                print("错误信息:")
                print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("\n测试程序执行超时!")
        return False
    except Exception as e:
        print(f"\n执行测试程序时发生错误: {e}")
        return False


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="ResNet18 Test Compilation Script")
    parser.add_argument("--target", type=str, help="目标平台 (例如: generic)")
    parser.add_argument("--source", type=str, help="源目录 (例如: float, generic, quantized)")
    parser.add_argument("--no-run", action="store_true", help="编译后不运行测试")
    args = parser.parse_args()
    
    # 确定是否使用build目录
    use_build_dir = False
    if args.target:
        if args.target == "generic":
            if not copy_files_for_target(source_dir=args.source):
                print("\n文件拷贝失败!")
                sys.exit(1)
            print()
            use_build_dir = True
        else:
            print(f"不支持的目标平台: {args.target}")
            sys.exit(1)
    
    # 执行编译
    compile_success = compile_test(use_build_dir=use_build_dir)
    
    if not compile_success:
        print("\n" + "=" * 60)
        print("编译失败!")
        print("=" * 60)
        sys.exit(1)
    
    # 运行测试（除非指定 --no-run）
    test_success = True
    if not args.no_run:
        test_success = run_test(use_build_dir=use_build_dir)
    
    # 根据模式确定输出路径
    if use_build_dir:
        final_output_exe = build_dir / f"resnet18_test{EXE_SUFFIX}"
        final_run_dir = build_dir
    else:
        final_output_exe = output_exe
        final_run_dir = script_dir
    
    # 根据平台确定运行命令格式
    run_cmd_prefix = ".\\" if IS_WINDOWS else "./"
    exe_name = f"resnet18_test{EXE_SUFFIX}"
    
    print("\n" + "=" * 60)
    if compile_success and test_success:
        print("编译和测试完成!")
        print(f"可执行文件: {final_output_exe}")
        print("\n手动运行测试:")
        print(f"  cd {final_run_dir}")
        print(f"  {run_cmd_prefix}{exe_name}")
    elif compile_success:
        print("编译完成!")
        print(f"可执行文件: {final_output_exe}")
        print("\n手动运行测试:")
        print(f"  cd {final_run_dir}")
        print(f"  {run_cmd_prefix}{exe_name}")
    print("=" * 60)
    
    sys.exit(0 if (compile_success and test_success) else 1)


if __name__ == "__main__":
    main()