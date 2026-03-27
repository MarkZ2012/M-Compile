# AI Model Compiler - 工具脚本使用指南

## 目录

1. [compile.py - 主编译工具](#1-compilepy---主编译工具)
2. [analyze.py - 统一分析工具](#2-analyzepy---统一分析工具)
3. [model_visualizer.py - 模型结构可视化](#3-model_visualizerpy---模型结构可视化)
4. [memory_analyzer.py - 内存分析工具](#4-memory_analyzerpy---内存分析工具)
5. [performance_profiler.py - 性能分析工具](#5-performance_profilerpy---性能分析工具)
6. [快速参考](#快速参考)

---

## 1. compile.py - 主编译工具

**位置**: `my_ai_compiler/tools/compile.py`

**功能**: 将ONNX模型编译为C代码，支持量化、优化和多种目标平台

### 基础用法

```bash
python compile.py <model.onnx> [选项]
```

### 命令行参数

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | - | str | - | **必填**，ONNX模型文件路径 |
| `--output` | `-o` | str | `output` | 输出目录，编译后的C代码保存位置 |
| `--name` | `-n` | str | `model` | 模型名称，用于生成源代码文件名 |
| `--target` | `-t` | str | `generic` | 目标平台，选项: `generic` / `x86_avx` / `arm_neon` |
| `--quantize` | `-q` | bool | False | 启用PTQ int8量化 |
| `--calib-dir` | - | str | None | 校准图片目录（jpg/png），用于激活量化 |
| `--calib-samples` | - | int | 32 | 最多加载的校准图片数量 |

### 使用示例

#### 1. 全精度编译（基础编译）
```bash
# 编译ResNet18，使用generic目标
python compile.py resnet18.onnx -t generic -o build/generic

# 自定义模型名称
python compile.py mobilenetv2.onnx -n mobilenet -t generic -o build/mobilenet
```

#### 2. 量化编译（Weight-only量化）
```bash
# 启用量化，不提供校准图片（权重量化）
python compile.py resnet18.onnx -t generic -o build/quantized -q
```

#### 3. 量化编译（推荐方式，带激活量化）
```bash
# 提供校准图片目录，启用完整PTQ量化（最推荐）
python compile.py resnet18.onnx -t generic -o build/quantized -q --calib-dir ./calib_images

# 指定校准样本数量
python compile.py resnet18.onnx -o build/quantized -q --calib-dir ./calib_images --calib-samples 64
```

#### 4. 针对不同目标平台编译
```bash
# ARM NEON优化（移动设备）
python compile.py mobilenetv2.onnx -t arm_neon -o build/arm

# x86 AVX优化（PC）
python compile.py resnet18.onnx -t x86_avx -o build/x86
```

#### 5. 完整示例
```bash
# 针对ARM平台的完整量化编译
python compile.py mobilenetv2.onnx \
  -t arm_neon \
  -o build/mobilenetv2_arm_quantized \
  -n mobilenetv2 \
  -q \
  --calib-dir ./calib_images \
  --calib-samples 32
```

### 编译流程

编译分为4个阶段：
1. **[1/4] 解析ONNX** - 加载模型，构建计算图
2. **[2/4] 加载校准数据** - 如果指定-q和--calib-dir，加载校准图片
3. **[3/4] 图优化** - 常量折叠、融合、死码消除，可选量化
4. **[4/4] 生成C代码** - 根据目标平台生成优化的C代码

### 输出文件

编译成功后，在输出目录生成：
- `model.c` - 模型主文件
- `model.h` - 模型头文件
- `preprocess_image.py` - 图片预处理脚本
- `CMakeLists.txt` - 构建配置
- `{model}_test.c` - 测试程序

---

## 2. analyze.py - 统一分析工具

**位置**: `my_ai_compiler/tools/analyze.py`

**功能**: 整合所有分析功能（结构、性能、内存），生成综合报告

### 基础用法

```bash
python analyze.py <model.onnx> [选项]
```

### 命令行参数

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | - | str | - | **必填**，ONNX模型文件路径 |
| `--output` | `-o` | str | None | 输出HTML报告路径 |
| `--structure` | - | bool | False | 仅分析模型结构 |
| `--performance` | - | bool | False | 仅分析性能 |
| `--memory` | - | bool | False | 仅分析内存 |
| `--all` | - | bool | False | 运行所有分析（默认行为） |
| `--summary` | - | bool | False | 仅打印摘要（不生成HTML） |
| `--compare` | - | str | None | 与另一个模型对比（如量化后的模型） |

### 使用示例

#### 1. 生成完整分析报告
```bash
# 分析模型并生成综合HTML报告
python analyze.py resnet18.onnx -o report.html

# 分析MobileNetV2
python analyze.py mobilenetv2.onnx -o analysis/mobilenetv2_report.html
```

#### 2. 分析特定方面
```bash
# 仅分析结构（算子分布、层信息）
python analyze.py resnet18.onnx --structure -o structure_report.html

# 仅分析性能（FLOPs、内存带宽、瓶颈）
python analyze.py resnet18.onnx --performance -o performance_report.html

# 仅分析内存占用
python analyze.py resnet18.onnx --memory -o memory_report.html

# 运行所有分析
python analyze.py resnet18.onnx --all -o full_report.html
```

#### 3. 仅输出摘要（控制台）
```bash
# 打印分析摘要到控制台，不生成HTML
python analyze.py resnet18.onnx --summary

# 同时输出摘要和HTML
python analyze.py resnet18.onnx --summary -o report.html
```

#### 4. 对比分析
```bash
# 对比全精度模型和量化模型
python analyze.py resnet18.onnx --compare resnet18_quantized.onnx -o comparison.html
```

### 报告内容

生成的HTML报告包含4个标签页：

| 标签页 | 内容 |
|--------|------|
| **Overview** | 总层数、算子类型应、FLOPs、参数量、峰值内存 |
| **Structure** | 层详情表、张量信息、图拓扑 |
| **Performance** | FLOPs分布、瓶颈层、算子性能对比 |
| **Memory** | 权重/激活内存、峰值内存、优化建议 |

---

## 3. model_visualizer.py - 模型结构可视化

**位置**: `my_ai_compiler/tools/model_visualizer.py`

**功能**: 生成交互式计算图可视化，展示网络拓扑和算子分布

### 基础用法

```bash
python model_visualizer.py <model.onnx> [选项]
```

### 命令行参数

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | - | str | - | **必填**，ONNX模型文件路径 |
| `--output` | `-o` | str | None | 输出HTML文件路径 |
| `--title` | `-t` | str | `Model Visualization` | 报告标题 |
| `--summary` | - | bool | False | 仅打印摘要，不生成HTML |

### 使用示例

#### 1. 生成可视化报告
```bash
# 基础可视化
python model_visualizer.py resnet18.onnx -o model_viz.html

# 自定义标题
python model_visualizer.py mobilenetv2.onnx \
  -o visualization.html \
  -t "MobileNetV2 Architecture"
```

#### 2. 仅输出摘要
```bash
# 打印模型摘要到控制台
python model_visualizer.py resnet18.onnx --summary

# 同时生成HTML和打印摘要
python model_visualizer.py resnet18.onnx -o viz.html --summary
```

### 报告功能

生成的HTML包含：

- **交互式计算图**
  - 缩放（鼠标滚轮）
  - 拖动平移（鼠标拖拽）
  - 悬停查看节点详情
  
- **算子统计**
  - 各算子类型出现次数
  - 条形图展示
  
- **层信息表**
  - 操作索引、名称、类型
  - 输入/输出形状
  - 属性参数

---

## 4. memory_analyzer.py - 内存分析工具

**位置**: `my_ai_compiler/tools/memory_analyzer.py`

**功能**: 分析权重内存、激活内存、峰值内存，提供优化建议

### 基础用法

```bash
python memory_analyzer.py <model.onnx> [选项]
```

### 命令行参数

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | - | str | - | **必填**，ONNX模型文件路径 |
| `--output` | `-o` | str | None | 输出HTML文件路径 |
| `--title` | `-t` | str | `Memory Analysis` | 报告标题 |
| `--summary` | - | bool | False | 仅打印摘要 |
| `--compare-quantized` | - | str | None | 与量化模型对比 |

### 使用示例

#### 1. 分析内存占用
```bash
# 生成内存分析报告
python memory_analyzer.py resnet18.onnx -o memory_report.html

# 自定义报告标题
python memory_analyzer.py mobilenetv2.onnx \
  -o memory_analysis.html \
  -t "MobileNetV2 Memory Profile"
```

#### 2. 仅输出摘要
```bash
# 控制台打印内存信息
python memory_analyzer.py resnet18.onnx --summary
```

#### 3. 对比量化前后的内存
```bash
# 对比全精度和量化模型的内存占用
python memory_analyzer.py resnet18.onnx \
  --compare-quantized resnet18_quantized.onnx

# 生成对比报告
python memory_analyzer.py resnet18.onnx \
  -o memory_comparison.html \
  --compare-quantized resnet18_quantized.onnx
```

### 报告内容

分析的内存类型：

| 内存类型 | 说明 |
|---------|------|
| **权重内存** | 模型参数占用的内存 |
| **激活内存** | 推理过程中中间特征图占用的内存 |
| **峰值内存** | 推理过程中最大的内存占用 |

### 优化建议示例

工具会自动生成优化建议，如：
- "发现3个大权重张量(>10MB)，考虑使用INT8量化减少75%内存"
- "峰值内存超过500MB，考虑梯度检查点或模型并行"
- "考虑使用FP16混合精度，可减少约50%的激活内存"

---

## 5. performance_profiler.py - 性能分析工具

**位置**: `my_ai_compiler/tools/performance_profiler.py`

**功能**: 计算FLOPs、内存带宽、算术强度，识别性能瓶颈

### 基础用法

```bash
python performance_profiler.py <model.onnx> [选项]
```

### 命令行参数

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | - | str | - | **必填**，ONNX模型文件路径 |
| `--output` | `-o` | str | None | 输出HTML文件路径 |
| `--title` | `-t` | str | `Performance Profile` | 报告标题 |
| `--summary` | - | bool | False | 仅打印摘要 |
| `--compare-quantized` | - | str | None | 与量化模型对比 |

### 使用示例

#### 1. 生成性能分析报告
```bash
# 基础性能分析
python performance_profiler.py resnet18.onnx -o perf_profile.html

# 自定义标题
python performance_profiler.py mobilenetv2.onnx \
  -o performance.html \
  -t "MobileNetV2 Performance Analysis"
```

#### 2. 仅输出摘要
```bash
# 控制台打印性能信息
python performance_profiler.py resnet18.onnx --summary

# 同时生成报告和摘要
python performance_profiler.py resnet18.onnx -o perf.html --summary
```

#### 3. 对比量化前后的性能
```bash
# 对比全精度和量化模型的性能
python performance_profiler.py resnet18.onnx \
  --compare-quantized resnet18_quantized.onnx
```

### 关键性能指标

| 指标 | 单位 | 说明 |
|------|------|------|
| **FLOPs** | GigaFLOPs | 浮点运算次数 |
| **内存访问** | MB | 读写内存大小 |
| **算术强度** | FLOPs/字节 | FLOPs/内存访问量 |
| **参数量** | M | 模型参数数量 |
| **瓶颈层** | - | FLOPs最多的Top 5层 |

### 内存绑定比例

- **低强度** (< 10): 内存绑定，优化空间在内存带宽
- **中等强度** (10-50): 混合绑定
- **高强度** (> 50): 计算绑定，优化空间在计算

---

## 快速参考

### typical工作流

```bash
# 1. 分析原始模型
python analyze.py resnet18.onnx -o analysis/resnet18_fp32.html

# 2. 编译模型（带量化）
python compile.py resnet18.onnx \
  -o build/quantized \
  -q \
  --calib-dir ./calib_images

# 3. 分析量化后的模型
python analyze.py resnet18_quantized.onnx -o analysis/resnet18_quant.html

# 4. 性能对比
python performance_profiler.py resnet18.onnx \
  --compare-quantized resnet18_quantized.onnx

# 5. 内存对比
python memory_analyzer.py resnet18.onnx \
  --compare-quantized resnet18_quantized.onnx
```

### 快速命令

```bash
# 快速分析一个模型
python analyze.py model.onnx -o report.html

# 快速可视化
python model_visualizer.py model.onnx -o viz.html

# 编译到默认位置
python compile.py model.onnx

# 量化编译
python compile.py model.onnx -q --calib-dir ./images

# 针对ARM平台的量化编译
python compile.py model.onnx -t arm_neon -q --calib-dir ./images
```

### 校准图片准备

```bash
# 校准图片的推荐做法
mkdir calib_images

# 从ImageNet或自己的数据集中选择32-256张代表性图片
# 放入calib_images目录，支持格式：jpg, png, jpeg, bmp, webp

# 使用校准
python compile.py model.onnx -q --calib-dir ./calib_images --calib-samples 100
```

### 输出文件说明

```
├── build/
│   ├── generic/              # 通用目标平台
│   │   ├── model.c
│   │   ├── model.h
│   │   ├── CMakeLists.txt
│   │   └── resnet18_test.c
│   ├── arm_neon/             # ARM NEON优化
│   │   └── ...
│   └── x86_avx/              # x86 AVX优化
│       └── ...
├── analysis/
│   ├── model_analysis.html   # 综合分析
│   ├── model_viz.html        # 结构可视化
│   ├── memory_report.html    # 内存分析
│   └── perf_profile.html     # 性能分析
└── calib_images/             # 校准图片
    ├── img1.jpg
    ├── img2.png
    └── ...
```

---

## 常见问题

### Q: 编译时出现 "No images found in calib_dir"
**A**: 检查校准图片目录是否存在且包含jpg/png文件。如不需要激活量化，省略--calib-dir参数。

### Q: 量化后内存占用没有减少
**A**: 确保--calib-dir有效且图片质量好。可尝试--compare-quantized对比。

### Q: 如何选择合适的目标平台？
**A**: 
- `generic`: 通用CPU，最兼容
- `arm_neon`: ARM处理器（手机、树莓派等）
- `x86_avx`: Intel/AMD CPU（PC）

### Q: 分析报告中"Memory Bound Ratio"是什么意思？
**A**: 内存绑定的层占比。数值越高，优化潜力越大。

### Q: 如何对比量化前后的差异？
**A**: 使用--compare或--compare-quantized参数：
```bash
python analyze.py fp32_model.onnx --compare quantized_model.onnx
```

---

## 相关资源

- ONNX模型下载: https://github.com/onnx/models
- 校准数据集: ImageNet ILSVRC2012
- C代码编译: 见build目录下的CMakeLists.txt

---

**最后更新**: 2026-03-27  
**版本**: 1.0  
**编译器**: AI Model Compiler v1.0
