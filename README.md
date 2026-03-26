# M-Compile: AI 模型编译器

M-Compile 是一个将 ONNX 模型编译为高性能 C 代码的 AI 编译器工具链。它支持多种硬件目标平台，可以将深度学习模型转换为可在嵌入式设备、服务器或桌面系统上高效运行的本地代码。

## 🚀 特性

- **ONNX 模型支持**: 直接解析 ONNX 格式的深度学习模型
- **多目标平台**: 
  - `generic`: 通用 C 代码，兼容所有 x64 平台
  - `x86_avx`: 针对 x86 处理器优化的 AVX/AVX2 SIMD 指令
  - `arm_neon`: 针对 ARM 处理器优化的 NEON SIMD 指令
- **自动优化**: 内置常量折叠、死代码消除等优化 pass
- **内存管理**: 智能的 buffer 分配和复用策略
- **运行时库**: 提供完整的卷积、池化、激活等算子实现
- **CMake 构建**: 自动生成 CMakeLists.txt，支持跨平台编译

## 📁 项目结构

目录树: M-Compile
============================================================
M-Compile/
├── build/
│   ├── generic/
│   │   ├── ops/
│   │   │   └── generic/
│   │   │       ├── __init__.py
│   │   │       ├── activations.c
│   │   │       ├── batchnorm.c
│   │   │       ├── conv2d.c
│   │   │       ├── linear.c
│   │   │       ├── math_ops.c
│   │   │       ├── ops.h
│   │   │       ├── pooling.c
│   │   │       ├── quant_conv2d.c
│   │   │       ├── quant_linear.c
│   │   │       ├── quant_ops.h
│   │   │       └── softmax.c
│   │   ├── weights/
│   │   │   ├── conv1_weight.bin
│   │   │   ├── conv1_weight_bias.bin
│   │   │   ├── fc_bias.bin
│   │   │   ├── fc_weight.bin
│   │   │   ├── layer1_0_conv1_weight.bin
│   │   │   ├── layer1_0_conv1_weight_bias.bin
│   │   │   ├── layer1_0_conv2_weight.bin
│   │   │   ├── layer1_0_conv2_weight_bias.bin
│   │   │   ├── layer1_1_conv1_weight.bin
│   │   │   ├── layer1_1_conv1_weight_bias.bin
│   │   │   ├── layer1_1_conv2_weight.bin
│   │   │   ├── layer1_1_conv2_weight_bias.bin
│   │   │   ├── layer2_0_conv1_weight.bin
│   │   │   ├── layer2_0_conv1_weight_bias.bin
│   │   │   ├── layer2_0_conv2_weight.bin
│   │   │   ├── layer2_0_conv2_weight_bias.bin
│   │   │   ├── layer2_0_downsample_0_weight.bin
│   │   │   ├── layer2_0_downsample_0_weight_bias.bin
│   │   │   ├── layer2_1_conv1_weight.bin
│   │   │   ├── layer2_1_conv1_weight_bias.bin
│   │   │   ├── layer2_1_conv2_weight.bin
│   │   │   ├── layer2_1_conv2_weight_bias.bin
│   │   │   ├── layer3_0_conv1_weight.bin
│   │   │   ├── layer3_0_conv1_weight_bias.bin
│   │   │   ├── layer3_0_conv2_weight.bin
│   │   │   ├── layer3_0_conv2_weight_bias.bin
│   │   │   ├── layer3_0_downsample_0_weight.bin
│   │   │   ├── layer3_0_downsample_0_weight_bias.bin
│   │   │   ├── layer3_1_conv1_weight.bin
│   │   │   ├── layer3_1_conv1_weight_bias.bin
│   │   │   ├── layer3_1_conv2_weight.bin
│   │   │   ├── layer3_1_conv2_weight_bias.bin
│   │   │   ├── layer4_0_conv1_weight.bin
│   │   │   ├── layer4_0_conv1_weight_bias.bin
│   │   │   ├── layer4_0_conv2_weight.bin
│   │   │   ├── layer4_0_conv2_weight_bias.bin
│   │   │   ├── layer4_0_downsample_0_weight.bin
│   │   │   ├── layer4_0_downsample_0_weight_bias.bin
│   │   │   ├── layer4_1_conv1_weight.bin
│   │   │   ├── layer4_1_conv1_weight_bias.bin
│   │   │   ├── layer4_1_conv2_weight.bin
│   │   │   ├── layer4_1_conv2_weight_bias.bin
│   │   │   ├── val_227.bin
│   │   │   └── val_230.bin
│   │   ├── CMakeLists.txt
│   │   ├── cat.png
│   │   ├── input_data.bin
│   │   ├── model.c
│   │   ├── model.h
│   │   ├── preprocess_image.py
│   │   ├── resnet18_test.c
│   │   └── resnet18_test.exe
│   ├── mobilenetv2_generic/
│   │   ├── ops/
│   │   │   └── generic/
│   │   │       ├── __init__.py
│   │   │       ├── activations.c
│   │   │       ├── batchnorm.c
│   │   │       ├── conv2d.c
│   │   │       ├── linear.c
│   │   │       ├── math_ops.c
│   │   │       ├── ops.h
│   │   │       ├── pooling.c
│   │   │       ├── quant_conv2d.c
│   │   │       ├── quant_linear.c
│   │   │       ├── quant_ops.h
│   │   │       └── softmax.c
│   │   ├── weights/
│   │   │   ├── classifier_1_bias.bin
│   │   │   ├── classifier_1_weight.bin
│   │   │   ├── features_0_0_weight.bin
│   │   │   ├── features_0_0_weight_bias.bin
│   │   │   ├── features_10_conv_0_0_weight.bin
│   │   │   ├── features_10_conv_0_0_weight_bias.bin
│   │   │   ├── features_10_conv_1_0_weight.bin
│   │   │   ├── features_10_conv_1_0_weight_bias.bin
│   │   │   ├── features_10_conv_2_weight.bin
│   │   │   ├── features_10_conv_2_weight_bias.bin
│   │   │   ├── features_11_conv_0_0_weight.bin
│   │   │   ├── features_11_conv_0_0_weight_bias.bin
│   │   │   ├── features_11_conv_1_0_weight.bin
│   │   │   ├── features_11_conv_1_0_weight_bias.bin
│   │   │   ├── features_11_conv_2_weight.bin
│   │   │   ├── features_11_conv_2_weight_bias.bin
│   │   │   ├── features_12_conv_0_0_weight.bin
│   │   │   ├── features_12_conv_0_0_weight_bias.bin
│   │   │   ├── features_12_conv_1_0_weight.bin
│   │   │   ├── features_12_conv_1_0_weight_bias.bin
│   │   │   ├── features_12_conv_2_weight.bin
│   │   │   ├── features_12_conv_2_weight_bias.bin
│   │   │   ├── features_13_conv_0_0_weight.bin
│   │   │   ├── features_13_conv_0_0_weight_bias.bin
│   │   │   ├── features_13_conv_1_0_weight.bin
│   │   │   ├── features_13_conv_1_0_weight_bias.bin
│   │   │   ├── features_13_conv_2_weight.bin
│   │   │   ├── features_13_conv_2_weight_bias.bin
│   │   │   ├── features_14_conv_0_0_weight.bin
│   │   │   ├── features_14_conv_0_0_weight_bias.bin
│   │   │   ├── features_14_conv_1_0_weight.bin
│   │   │   ├── features_14_conv_1_0_weight_bias.bin
│   │   │   ├── features_14_conv_2_weight.bin
│   │   │   ├── features_14_conv_2_weight_bias.bin
│   │   │   ├── features_15_conv_0_0_weight.bin
│   │   │   ├── features_15_conv_0_0_weight_bias.bin
│   │   │   ├── features_15_conv_1_0_weight.bin
│   │   │   ├── features_15_conv_1_0_weight_bias.bin
│   │   │   ├── features_15_conv_2_weight.bin
│   │   │   ├── features_15_conv_2_weight_bias.bin
│   │   │   ├── features_16_conv_0_0_weight.bin
│   │   │   ├── features_16_conv_0_0_weight_bias.bin
│   │   │   ├── features_16_conv_1_0_weight.bin
│   │   │   ├── features_16_conv_1_0_weight_bias.bin
│   │   │   ├── features_16_conv_2_weight.bin
│   │   │   ├── features_16_conv_2_weight_bias.bin
│   │   │   ├── features_17_conv_0_0_weight.bin
│   │   │   ├── features_17_conv_0_0_weight_bias.bin
│   │   │   ├── features_17_conv_1_0_weight.bin
│   │   │   ├── features_17_conv_1_0_weight_bias.bin
│   │   │   ├── features_17_conv_2_weight.bin
│   │   │   ├── features_17_conv_2_weight_bias.bin
│   │   │   ├── features_18_0_weight.bin
│   │   │   ├── features_18_0_weight_bias.bin
│   │   │   ├── features_1_conv_0_0_weight.bin
│   │   │   ├── features_1_conv_0_0_weight_bias.bin
│   │   │   ├── features_1_conv_1_weight.bin
│   │   │   ├── features_1_conv_1_weight_bias.bin
│   │   │   ├── features_2_conv_0_0_weight.bin
│   │   │   ├── features_2_conv_0_0_weight_bias.bin
│   │   │   ├── features_2_conv_1_0_weight.bin
│   │   │   ├── features_2_conv_1_0_weight_bias.bin
│   │   │   ├── features_2_conv_2_weight.bin
│   │   │   ├── features_2_conv_2_weight_bias.bin
│   │   │   ├── features_3_conv_0_0_weight.bin
│   │   │   ├── features_3_conv_0_0_weight_bias.bin
│   │   │   ├── features_3_conv_1_0_weight.bin
│   │   │   ├── features_3_conv_1_0_weight_bias.bin
│   │   │   ├── features_3_conv_2_weight.bin
│   │   │   ├── features_3_conv_2_weight_bias.bin
│   │   │   ├── features_4_conv_0_0_weight.bin
│   │   │   ├── features_4_conv_0_0_weight_bias.bin
│   │   │   ├── features_4_conv_1_0_weight.bin
│   │   │   ├── features_4_conv_1_0_weight_bias.bin
│   │   │   ├── features_4_conv_2_weight.bin
│   │   │   ├── features_4_conv_2_weight_bias.bin
│   │   │   ├── features_5_conv_0_0_weight.bin
│   │   │   ├── features_5_conv_0_0_weight_bias.bin
│   │   │   ├── features_5_conv_1_0_weight.bin
│   │   │   ├── features_5_conv_1_0_weight_bias.bin
│   │   │   ├── features_5_conv_2_weight.bin
│   │   │   ├── features_5_conv_2_weight_bias.bin
│   │   │   ├── features_6_conv_0_0_weight.bin
│   │   │   ├── features_6_conv_0_0_weight_bias.bin
│   │   │   ├── features_6_conv_1_0_weight.bin
│   │   │   ├── features_6_conv_1_0_weight_bias.bin
│   │   │   ├── features_6_conv_2_weight.bin
│   │   │   ├── features_6_conv_2_weight_bias.bin
│   │   │   ├── features_7_conv_0_0_weight.bin
│   │   │   ├── features_7_conv_0_0_weight_bias.bin
│   │   │   ├── features_7_conv_1_0_weight.bin
│   │   │   ├── features_7_conv_1_0_weight_bias.bin
│   │   │   ├── features_7_conv_2_weight.bin
│   │   │   ├── features_7_conv_2_weight_bias.bin
│   │   │   ├── features_8_conv_0_0_weight.bin
│   │   │   ├── features_8_conv_0_0_weight_bias.bin
│   │   │   ├── features_8_conv_1_0_weight.bin
│   │   │   ├── features_8_conv_1_0_weight_bias.bin
│   │   │   ├── features_8_conv_2_weight.bin
│   │   │   ├── features_8_conv_2_weight_bias.bin
│   │   │   ├── features_9_conv_0_0_weight.bin
│   │   │   ├── features_9_conv_0_0_weight_bias.bin
│   │   │   ├── features_9_conv_1_0_weight.bin
│   │   │   ├── features_9_conv_1_0_weight_bias.bin
│   │   │   ├── features_9_conv_2_weight.bin
│   │   │   ├── features_9_conv_2_weight_bias.bin
│   │   │   ├── max_val_cast.bin
│   │   │   ├── min_val_cast.bin
│   │   │   ├── val_577.bin
│   │   │   ├── val_578.bin
│   │   │   └── val_581.bin
│   │   ├── CMakeLists.txt
│   │   ├── input_data.bin
│   │   ├── mobilenetv2_test.c
│   │   ├── mobilenetv2_test.exe
│   │   ├── model.c
│   │   ├── model.h
│   │   └── preprocess_image.py
│   ├── mobilenetv2_quantized/
│   │   ├── weights/
│   │   │   ├── classifier_1_weight.bin
│   │   │   ├── features_0_0_weight.bin
│   │   │   ├── features_10_conv_0_0_weight.bin
│   │   │   ├── features_10_conv_1_0_weight.bin
│   │   │   ├── features_10_conv_2_weight.bin
│   │   │   ├── features_11_conv_0_0_weight.bin
│   │   │   ├── features_11_conv_1_0_weight.bin
│   │   │   ├── features_11_conv_2_weight.bin
│   │   │   ├── features_12_conv_0_0_weight.bin
│   │   │   ├── features_12_conv_1_0_weight.bin
│   │   │   ├── features_12_conv_2_weight.bin
│   │   │   ├── features_13_conv_0_0_weight.bin
│   │   │   ├── features_13_conv_1_0_weight.bin
│   │   │   ├── features_13_conv_2_weight.bin
│   │   │   ├── features_14_conv_0_0_weight.bin
│   │   │   ├── features_14_conv_1_0_weight.bin
│   │   │   ├── features_14_conv_2_weight.bin
│   │   │   ├── features_15_conv_0_0_weight.bin
│   │   │   ├── features_15_conv_1_0_weight.bin
│   │   │   ├── features_15_conv_2_weight.bin
│   │   │   ├── features_16_conv_0_0_weight.bin
│   │   │   ├── features_16_conv_1_0_weight.bin
│   │   │   ├── features_16_conv_2_weight.bin
│   │   │   ├── features_17_conv_0_0_weight.bin
│   │   │   ├── features_17_conv_1_0_weight.bin
│   │   │   ├── features_17_conv_2_weight.bin
│   │   │   ├── features_18_0_weight.bin
│   │   │   ├── features_1_conv_0_0_weight.bin
│   │   │   ├── features_1_conv_1_weight.bin
│   │   │   ├── features_2_conv_0_0_weight.bin
│   │   │   ├── features_2_conv_1_0_weight.bin
│   │   │   ├── features_2_conv_2_weight.bin
│   │   │   ├── features_3_conv_0_0_weight.bin
│   │   │   ├── features_3_conv_1_0_weight.bin
│   │   │   ├── features_3_conv_2_weight.bin
│   │   │   ├── features_4_conv_0_0_weight.bin
│   │   │   ├── features_4_conv_1_0_weight.bin
│   │   │   ├── features_4_conv_2_weight.bin
│   │   │   ├── features_5_conv_0_0_weight.bin
│   │   │   ├── features_5_conv_1_0_weight.bin
│   │   │   ├── features_5_conv_2_weight.bin
│   │   │   ├── features_6_conv_0_0_weight.bin
│   │   │   ├── features_6_conv_1_0_weight.bin
│   │   │   ├── features_6_conv_2_weight.bin
│   │   │   ├── features_7_conv_0_0_weight.bin
│   │   │   ├── features_7_conv_1_0_weight.bin
│   │   │   ├── features_7_conv_2_weight.bin
│   │   │   ├── features_8_conv_0_0_weight.bin
│   │   │   ├── features_8_conv_1_0_weight.bin
│   │   │   ├── features_8_conv_2_weight.bin
│   │   │   ├── features_9_conv_0_0_weight.bin
│   │   │   ├── features_9_conv_1_0_weight.bin
│   │   │   ├── features_9_conv_2_weight.bin
│   │   │   ├── max_val_cast.bin
│   │   │   ├── min_val_cast.bin
│   │   │   ├── val_577.bin
│   │   │   └── val_581.bin
│   │   ├── CMakeLists.txt
│   │   ├── model.c
│   │   └── model.h
│   └── quantized/
│       ├── ops/
│       │   └── generic/
│       │       ├── __init__.py
│       │       ├── activations.c
│       │       ├── batchnorm.c
│       │       ├── conv2d.c
│       │       ├── linear.c
│       │       ├── math_ops.c
│       │       ├── ops.h
│       │       ├── pooling.c
│       │       ├── quant_conv2d.c
│       │       ├── quant_linear.c
│       │       ├── quant_ops.h
│       │       └── softmax.c
│       ├── weights/
│       │   ├── conv1_weight.bin
│       │   ├── conv1_weight_bias.bin
│       │   ├── fc_bias.bin
│       │   ├── fc_weight.bin
│       │   ├── layer1_0_conv1_weight.bin
│       │   ├── layer1_0_conv1_weight_bias.bin
│       │   ├── layer1_0_conv2_weight.bin
│       │   ├── layer1_0_conv2_weight_bias.bin
│       │   ├── layer1_1_conv1_weight.bin
│       │   ├── layer1_1_conv1_weight_bias.bin
│       │   ├── layer1_1_conv2_weight.bin
│       │   ├── layer1_1_conv2_weight_bias.bin
│       │   ├── layer2_0_conv1_weight.bin
│       │   ├── layer2_0_conv1_weight_bias.bin
│       │   ├── layer2_0_conv2_weight.bin
│       │   ├── layer2_0_conv2_weight_bias.bin
│       │   ├── layer2_0_downsample_0_weight.bin
│       │   ├── layer2_0_downsample_0_weight_bias.bin
│       │   ├── layer2_1_conv1_weight.bin
│       │   ├── layer2_1_conv1_weight_bias.bin
│       │   ├── layer2_1_conv2_weight.bin
│       │   ├── layer2_1_conv2_weight_bias.bin
│       │   ├── layer3_0_conv1_weight.bin
│       │   ├── layer3_0_conv1_weight_bias.bin
│       │   ├── layer3_0_conv2_weight.bin
│       │   ├── layer3_0_conv2_weight_bias.bin
│       │   ├── layer3_0_downsample_0_weight.bin
│       │   ├── layer3_0_downsample_0_weight_bias.bin
│       │   ├── layer3_1_conv1_weight.bin
│       │   ├── layer3_1_conv1_weight_bias.bin
│       │   ├── layer3_1_conv2_weight.bin
│       │   ├── layer3_1_conv2_weight_bias.bin
│       │   ├── layer4_0_conv1_weight.bin
│       │   ├── layer4_0_conv1_weight_bias.bin
│       │   ├── layer4_0_conv2_weight.bin
│       │   ├── layer4_0_conv2_weight_bias.bin
│       │   ├── layer4_0_downsample_0_weight.bin
│       │   ├── layer4_0_downsample_0_weight_bias.bin
│       │   ├── layer4_1_conv1_weight.bin
│       │   ├── layer4_1_conv1_weight_bias.bin
│       │   ├── layer4_1_conv2_weight.bin
│       │   ├── layer4_1_conv2_weight_bias.bin
│       │   ├── val_227.bin
│       │   └── val_230.bin
│       ├── CMakeLists.txt
│       ├── cat.png
│       ├── input_data.bin
│       ├── model.c
│       ├── model.h
│       ├── preprocess_image.py
│       ├── resnet18_test.c
│       └── resnet18_test.exe
├── calib_images/
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   ├── cat.10.jpg
│   ├── cat.11.jpg
│   ├── cat.12.jpg
│   ├── cat.13.jpg
│   ├── cat.14.jpg
│   ├── cat.15.jpg
│   ├── cat.16.jpg
│   ├── cat.17.jpg
│   ├── cat.18.jpg
│   ├── cat.19.jpg
│   ├── cat.2.jpg
│   ├── cat.20.jpg
│   ├── cat.21.jpg
│   ├── cat.22.jpg
│   ├── cat.23.jpg
│   ├── cat.24.jpg
│   ├── cat.25.jpg
│   ├── cat.26.jpg
│   ├── cat.27.jpg
│   ├── cat.28.jpg
│   ├── cat.29.jpg
│   ├── cat.3.jpg
│   ├── cat.30.jpg
│   ├── cat.4.jpg
│   ├── cat.5.jpg
│   ├── cat.6.jpg
│   ├── cat.7.jpg
│   ├── cat.8.jpg
│   ├── cat.9.jpg
│   └── cat.png
├── my_ai_compiler/
│   ├── backend/
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── allocator.py
│   │   │   └── shape_infer.py
│   │   ├── kernels/
│   │   │   ├── generic/
│   │   │   │   └── __init__.py
│   │   │   ├── x86_avx/
│   │   │   │   └── __init__.py
│   │   │   ├── __init__.py
│   │   │   └── base.py
│   │   ├── targets/
│   │   │   ├── __init__.py
│   │   │   ├── arm_neon.py
│   │   │   ├── base.py
│   │   │   ├── generic_c.py
│   │   │   └── x86_avx.py
│   │   ├── __init__.py
│   │   └── emitter.py
│   ├── frontend/
│   │   ├── __init__.py
│   │   ├── graph_ir.py
│   │   └── onnx_parser.py
│   ├── optimizer/
│   │   ├── passes/
│   │   │   ├── fusion/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── conv_bn_fusion.py
│   │   │   │   └── gemm_fusion.py
│   │   │   ├── quantization/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── activation_calibrator.py
│   │   │   │   ├── ptq.py
│   │   │   │   ├── qat.py
│   │   │   │   ├── quant_config.py
│   │   │   │   ├── quant_params_exporter.py
│   │   │   │   └── quantizer.py
│   │   │   ├── rewrite/
│   │   │   │   ├── __init__.py
│   │   │   │   └── layout_transform.py
│   │   │   ├── __init__.py
│   │   │   ├── constant_fold.py
│   │   │   └── dead_code_elim.py
│   │   ├── __init__.py
│   │   └── pass_manager.py
│   ├── runtime/
│   │   ├── include/
│   │   │   └── myrt.h
│   │   ├── ops/
│   │   │   ├── arm_neon/
│   │   │   ├── generic/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── activations.c
│   │   │   │   ├── batchnorm.c
│   │   │   │   ├── conv2d.c
│   │   │   │   ├── linear.c
│   │   │   │   ├── math_ops.c
│   │   │   │   ├── ops.h
│   │   │   │   ├── pooling.c
│   │   │   │   ├── quant_conv2d.c
│   │   │   │   ├── quant_linear.c
│   │   │   │   ├── quant_ops.h
│   │   │   │   └── softmax.c
│   │   │   └── x86_avx/
│   │   ├── src/
│   │   │   └── loader.c
│   │   └── __init__.py
│   ├── tools/
│   │   ├── model/
│   │   │   ├── mobilenetv2_inference.py
│   │   │   ├── resnet18.onnx
│   │   │   ├── resnet18.onnx.data
│   │   │   └── resnet18_inference.py
│   │   ├── test/
│   │   │   └── example/
│   │   │       ├── MobileNetV2/
│   │   │       │   ├── build.py
│   │   │       │   ├── mobilenetv2_test.c
│   │   │       │   └── preprocess_image.py
│   │   │       └── resnet18/
│   │   │           ├── build.py
│   │   │           ├── cat.png
│   │   │           ├── preprocess_image.py
│   │   │           └── resnet18_test.c
│   │   ├── compile.py
│   │   ├── test_accuracy_compare.py
│   │   ├── test_quantization.py
│   │   └── test_quantization_ops.py
│   ├── README.md
│   ├── __init__.py
│   ├── cat.png
│   ├── compiler_optimization_roadmap.html
│   ├── mobilenetv2.onnx
│   ├── mobilenetv2.onnx.data
│   ├── resnet18.onnx
│   ├── resnet18.onnx.data
│   ├── task_progress.md
│   └── test_quantized_resnet18.py
├── MR.txt
├── Quantizer.docx
├── QuantizerPlan.md
├── README.md
├── TEST_MOBILE.txt
├── TEST_QUANTIZER.txt
├── TestRESNET18.md
├── accuracy_compare_result.json
├── build error.txt
├── debug_conv.c
├── directory_tree.txt
├── input_data.bin
├── list_tree.py
├── mobilenetv2.onnx
├── mobilenetv2.onnx.data
├── pytorch_result.json
└── quantization_bug_analysis.md
============================================================
生成时间: 2026/03/26 周四 15:42


```

## 🛠️ 快速开始

### 环境要求

- Python 3.7+
- ONNX 库 (`pip install onnx numpy`)
- GCC 编译器（用于编译生成的 C 代码）
- CMake（可选，用于复杂项目构建）

### 安装依赖

```bash
pip install onnx numpy
```

### 基本用法

1. **编译 ONNX 模型为 C 代码**

```bash
# 使用通用 C 目标
python -m my_ai_compiler.tools.compile your_model.onnx -o output -n model_name -t generic

# 使用 x86 AVX 优化
python -m my_ai_compiler.tools.compile your_model.onnx -o output -n model_name -t x86_avx

# 使用 ARM NEON 优化
python -m my_ai_compiler.tools.compile your_model.onnx -o output -n model_name -t arm_neon
```

参数说明：
- `your_model.onnx`: 输入的 ONNX 模型文件
- `-o output`: 输出目录（默认为 `output`）
- `-n model_name`: 生成的 C 文件前缀（默认为 `model`）
- `-t target`: 目标平台（`generic`/`x86_avx`/`arm_neon`）

2. **编译生成的 C 代码**

```bash
cd output
mkdir build && cd build
cmake ..
make
```

3. **运行测试**

```bash
# 使用 ResNet18 示例
cd my_ai_compiler/tools/test/example/resnet18
python build.py --target generic
```

## 📊 示例：编译 ResNet18

项目提供了完整的 ResNet18 编译示例：

```bash
# 1. 生成 ResNet18 的 C 代码
#python -m my_ai_compiler.tools.compile my_ai_compiler/tools/model/resnet18.onnx -o build/generic -n resnet18 -t generic
PS C:\Users\aistar\Desktop\M-Compile> python my_ai_compiler/tools/compile.py my_ai_compiler/tools/model/resnet18.onnx -o build/generic -n resnet18 -t generic

# 2. 编译测试程序
PS C:\Users\aistar\Desktop\M-Compile> python .\my_ai_compiler\tools\test\example\resnet18\build.py --target generic
#cd my_ai_compiler/tools/test/example/resnet18
#python build.py --target generic

# 3. 运行推理测试
cd ../../../../build/generic
./resnet18_test.exe
```
20260326

PS C:\Users\aistar\Desktop\M-Compile> python .\my_ai_compiler\tools\compile.py .\my_ai_compiler\resnet18.onnx --target generic --quantize --output build/quantized
PS C:\Users\aistar\Desktop\M-Compile> python .\my_ai_compiler\tools\compile.py .\my_ai_compiler\resnet18.onnx --target generic --output build/generic

PS C:\Users\aistar\Desktop\M-Compile> python .\my_ai_compiler\tools\test\example\resnet18\build.py --target generic --source quantized
PS C:\Users\aistar\Desktop\M-Compile> python .\my_ai_compiler\tools\test\example\resnet18\build.py --target generic --source generic

python .\my_ai_compiler\tools\compile.py .\my_ai_compiler\resnet18.onnx --target generic --quantize --output build/quantized --calib-dir .\calib_images --calib-samples 32

C:\Users\aistar\Desktop\M-Compile> python my_ai_compiler/tools/model/mobilenetv2_inference.py 
C:\Users\aistar\Desktop\M-Compile> python .\my_ai_compiler\tools\model\resnet18_inference.py

- `python my_ai_compiler/tools/compile.py my_ai_compiler/mobilenetv2.onnx -t generic -o build/mobilenetv2_generic`
- `python my_ai_compiler/tools/test/example/MobileNetV2/build.py --target generic --source build/mobilenetv2_generic`

 `build/mobilenetv2_generic/mobilenetv2_test.exe` 

## 🔧 编译器架构

### 编译流程

```
ONNX 模型
    ↓
[前端] ONNX 解析器
    ↓
Graph IR (中间表示)
    ↓
[优化器] 优化 Pass (常量折叠、死代码消除等)
    ↓
[后端] 目标代码生成
    ↓
C 源码 + 权重二进制文件
```

### 支持的主要算子

- **卷积**: Conv2d
- **激活函数**: ReLU, Softmax
- **池化**: MaxPool, Global Average Pooling
- **线性层**: Fully Connected (Gemm)
- **批归一化**: BatchNorm
- **张量操作**: Add, Reshape 等

## 🎯 目标平台优化

### 通用 C (`generic`)
- 兼容所有支持 C99 的编译器
- 适合快速原型开发和跨平台部署
- 性能基准实现

### x86 AVX (`x86_avx`)
- 利用 AVX/AVX2 SIMD 指令集
- 适合 Intel/AMD 处理器
- 提供 2-8 倍的性能提升

### ARM NEON (`arm_neon`)
- 利用 ARM NEON SIMD 指令集
- 适合移动设备和嵌入式 ARM 处理器
- 优化功耗和性能平衡

## 📝 开发指南

### 添加新的目标平台

1. 在 `my_ai_compiler/backend/targets/` 中创建新的目标类
2. 继承 `BaseTarget` 基类
3. 实现所有算子的代码生成方法
4. 在 `my_ai_compiler/backend/targets/__init__.py` 中注册新目标

### 添加新的优化 Pass

1. 在 `my_ai_compiler/optimizer/passes/` 中创建新的优化文件
2. 实现优化函数
3. 在 `my_ai_compiler/optimizer/pass_manager.py` 中注册优化

### 扩展算子支持

1. 在运行时库中实现新的算子函数
2. 在目标平台类中添加对应的代码生成方法
3. 更新 ONNX 解析器以支持新的算子类型

## 🐛 故障排除

### 常见问题

1. **ONNX 解析失败**
   - 确保 ONNX 模型格式正确
   - 检查模型是否包含不支持的算子

2. **编译错误**
   - 确保已安装 GCC 编译器
   - 检查生成的 C 代码语法是否正确

3. **运行时错误**
   - 确保权重文件路径正确
   - 检查输入数据格式是否符合模型要求

### 调试技巧

- 使用 `--no-run` 参数只编译不运行测试
- 查看生成的 C 代码了解具体实现
- 检查 `build/weights/` 目录下的权重文件

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- ONNX 社区提供的模型格式标准
- 开源编译器项目的灵感和参考
- 所有贡献者的支持和帮助

## 📞 联系方式

- 项目主页: [GitHub Repository]
- 问题反馈: [GitHub Issues]
- 邮箱: [zhangtaozt6@gmail.com]

---

**M-Compile** - 让 AI 模型在任何设备上高效运行 🚀
