# AI Model Compiler - 算子库手册

> 本文档描述 `runtime/ops/generic/` 目录下所有算子的设计原理、接口规范和使用方法。

---

## 目录

1. [概述](#概述)
2. [数据布局](#数据布局)
3. [浮点算子](#浮点算子)
   - [卷积类](#卷积类)
     - [op_conv2d](#op_conv2d)
     - [op_depthwise_conv2d](#op_depthwise_conv2d)
   - [池化类](#池化类)
     - [op_maxpool](#op_maxpool)
     - [op_avgpool_global](#op_avgpool_global)
   - [全连接类](#全连接类)
     - [op_linear](#op_linear)
   - [归一化类](#归一化类)
     - [op_batchnorm](#op_batchnorm)
     - [op_layernorm](#op_layernorm)
   - [激活类](#激活类)
     - [op_relu](#op_relu)
     - [op_clip](#op_clip)
     - [op_softmax](#op_softmax)
   - [数学运算类](#数学运算类)
     - [op_add](#op_add)
   - [张量变换类](#张量变换类)
     - [op_concat](#op_concat)
     - [op_split](#op_split)
     - [op_resize_bilinear](#op_resize_bilinear)
     - [op_resize_nearest](#op_resize_nearest)
   - [注意力机制](#注意力机制)
     - [op_attention](#op_attention)
4. [量化算子](#量化算子)
   - [量化原理](#量化原理)
   - [量化卷积类](#量化卷积类)
     - [op_quant_conv2d_int8](#op_quant_conv2d_int8)
     - [op_quant_conv2d_relu_int8](#op_quant_conv2d_relu_int8)
     - [op_quant_depthwise_conv2d_int8](#op_quant_depthwise_conv2d_int8)
   - [量化全连接类](#量化全连接类)
     - [op_quant_linear_int8](#op_quant_linear_int8)
     - [op_quant_linear_relu_int8](#op_quant_linear_relu_int8)
   - [量化加法类](#量化加法类)
     - [op_quant_add_int8](#op_quant_add_int8)
     - [op_quant_add_relu_int8](#op_quant_add_relu_int8)

---

## 概述

本算子库为 AI Model Compiler 提供运行时推理支持，采用纯 C 语言实现，具有以下特点：

- **零依赖**：仅依赖标准 C 库（`<math.h>`, `<stdint.h>`, `<string.h>`）
- **跨平台**：支持 Windows、Linux、嵌入式平台
- **NCHW 布局**：采用 NCHW 内存布局，与 ONNX/PyTorch 默认格式一致
- **量化支持**：提供完整的 INT8 量化算子实现

---

## 数据布局

所有 4D 张量采用 **NCHW** 布局：

```
张量形状: [N, C, H, W]
内存索引: ((n * C + c) * H + h) * W + w
```

| 符号 | 含义 |
|------|------|
| N | Batch size（批次大小）|
| C | Channels（通道数）|
| H | Height（高度）|
| W | Width（宽度）|

---

## 浮点算子

### 卷积类

#### op_conv2d

2D 卷积算子，支持标准卷积和分组卷积。

**函数原型：**

```c
void op_conv2d(
    const float* input,   // 输入张量 [N, C_in, H, W]
    int N,                // Batch size
    int C_in,             // 输入通道数
    int H,                // 输入高度
    int W,                // 输入宽度
    const float* weight,  // 权重张量 [C_out, C_in/groups, kH, kW]
    int C_out,            // 输出通道数
    int kH,               // 卷积核高度
    int kW,               // 卷积核宽度
    const float* bias,    // 偏置 [C_out]，可为 NULL
    int stride_h,         // 高度方向步长
    int stride_w,         // 宽度方向步长
    int pad_h,            // 高度方向填充
    int pad_w,            // 宽度方向填充
    int groups,           // 分组数（1=标准卷积，C_in=深度可分离卷积）
    float* output         // 输出张量 [N, C_out, H_out, W_out]
);
```

**输出尺寸计算：**

```
H_out = (H + 2 * pad_h - kH) / stride_h + 1
W_out = (W + 2 * pad_w - kW) / stride_w + 1
```

**设计说明：**

- 支持 Zero Padding（超出边界的输入视为 0）
- 支持分组卷积（Grouped Convolution）
- 当 `groups = C_in` 时，等价于深度可分离卷积

---

#### op_depthwise_conv2d

深度可分离卷积（Depthwise Convolution），是分组卷积的特例。

**函数原型：**

```c
void op_depthwise_conv2d(
    const float* input,   // 输入张量 [N, C_in, H, W]
    int N,                // Batch size
    int C_in,             // 输入通道数（= 输出通道数）
    int H,                // 输入高度
    int W,                // 输入宽度
    const float* weight,  // 权重张量 [C_in, 1, kH, kW]
    int kH,               // 卷积核高度
    int kW,               // 卷积核宽度
    const float* bias,    // 偏置 [C_in]，可为 NULL
    int stride_h,         // 高度方向步长
    int stride_w,         // 宽度方向步长
    int pad_h,            // 高度方向填充
    int pad_w,            // 宽度方向填充
    float* output         // 输出张量 [N, C_in, H_out, W_out]
);
```

**设计说明：**

- 内部调用 `op_conv2d`，设置 `groups = C_in`
- 每个通道独立进行卷积操作
- 常用于 MobileNet 等轻量级网络

---

### 池化类

#### op_maxpool

2D 最大池化。

**函数原型：**

```c
void op_maxpool(
    const float* input,   // 输入张量 [N, C, H, W]
    int N,                // Batch size
    int C,                // 通道数
    int H,                // 输入高度
    int W,                // 输入宽度
    int kH,               // 池化核高度
    int kW,               // 池化核宽度
    int sH,               // 高度方向步长
    int sW,               // 宽度方向步长
    int pH,               // 高度方向填充
    int pW,               // 宽度方向填充
    float* output         // 输出张量 [N, C, H_out, W_out]
);
```

**输出尺寸计算：**

```
H_out = (H + 2 * pH - kH) / sH + 1
W_out = (W + 2 * pW - kW) / sW + 1
```

**设计说明：**

- 超出边界的区域不参与最大值计算
- 初始最大值设为 `-1e30f`

---

#### op_avgpool_global

全局平均池化。

**函数原型：**

```c
void op_avgpool_global(
    const float* x,       // 输入张量 [N, C, H, W]
    int N,                // Batch size
    int C,                // 通道数
    int H,                // 高度
    int W,                // 宽度
    float* out            // 输出张量 [N, C]
);
```

**计算公式：**

```
out[n, c] = mean(x[n, c, :, :]) = sum(x[n, c, h, w]) / (H * W)
```

**设计说明：**

- 常用于 ResNet、MobileNet 等网络的分类头
- 将空间维度 (H, W) 压缩为单个值

---

### 全连接类

#### op_linear

全连接层（矩阵乘法 + 偏置）。

**函数原型：**

```c
void op_linear(
    const float* x,       // 输入张量 [batch, in_feat]
    const float* w,       // 权重张量 [out_feat, in_feat]
    const float* b,       // 偏置 [out_feat]，可为 NULL
    int batch,            // Batch size
    int in_feat,          // 输入特征数
    int out_feat,         // 输出特征数
    float* out            // 输出张量 [batch, out_feat]
);
```

**计算公式：**

```
out[n, o] = bias[o] + sum(x[n, i] * w[o, i])  for i in [0, in_feat)
```

**设计说明：**

- 权重布局为 `[out_feat, in_feat]`（行主序）
- 等价于 ONNX 的 Gemm 算子

---

### 归一化类

#### op_batchnorm

批归一化（Batch Normalization）- 推理模式。

**函数原型：**

```c
void op_batchnorm(
    float* x,             // 输入/输出张量 [N, C, H, W]（原地操作）
    int N,                // Batch size
    int C,                // 通道数
    int H,                // 高度
    int W,                // 宽度
    const float* gamma,   // 缩放参数 [C]
    const float* beta,    // 平移参数 [C]
    const float* mean,    // 运行均值 [C]
    const float* var,     // 运行方差 [C]
    float eps             // 数值稳定性参数
);
```

**计算公式：**

```
scale = gamma[c] / sqrt(var[c] + eps)
shift = beta[c] - mean[c] * scale
x[n, c, h, w] = x[n, c, h, w] * scale + shift
```

**设计说明：**

- 原地操作（In-place），节省内存
- 仅用于推理，使用运行时统计量

---

#### op_layernorm

层归一化（Layer Normalization）。

**函数原型：**

```c
void op_layernorm(
    const float* x,       // 输入张量 [N, norm_size]
    int N,                // Batch size
    int norm_size,        // 归一化维度大小
    const float* gamma,   // 缩放参数 [norm_size]
    const float* beta,    // 平移参数 [norm_size]
    float eps,            // 数值稳定性参数
    float* output         // 输出张量 [N, norm_size]
);
```

**计算公式：**

```
mean = sum(x[i]) / norm_size
var = sum((x[i] - mean)^2) / norm_size
output[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
```

**设计说明：**

- 常用于 Transformer、BERT 等模型
- 在最后一个维度上进行归一化

---

### 激活类

#### op_relu

ReLU 激活函数。

**函数原型：**

```c
void op_relu(
    float* x,             // 输入/输出张量（原地操作）
    int n                 // 元素数量
);
```

**计算公式：**

```
x[i] = max(0, x[i])
```

**设计说明：**

- 原地操作，高效节省内存
- 最常用的激活函数

---

#### op_clip

裁剪函数（通常用于 ReLU6）。

**函数原型：**

```c
void op_clip(
    const float* x,       // 输入张量
    const float* min_v,   // 最小值（标量指针）
    const float* max_v,   // 最大值（标量指针）
    int n,                // 元素数量
    float* out            // 输出张量
);
```

**计算公式：**

```
out[i] = min(max(x[i], min_v[0]), max_v[0])
```

**设计说明：**

- `min_v` 和 `max_v` 为标量指针（单元素数组）
- 当 `min_v=0, max_v=6` 时，等价于 ReLU6

---

#### op_softmax

Softmax 激活函数。

**函数原型：**

```c
void op_softmax(
    float* x,             // 输入/输出张量（原地操作）
    int n                 // 元素数量
);
```

**计算公式：**

```
max_val = max(x[i])
x[i] = exp(x[i] - max_val)
sum = sum(x[i])
x[i] = x[i] / sum
```

**设计说明：**

- 原地操作
- 使用数值稳定技巧（减去最大值）
- 常用于分类网络的输出层

---

### 数学运算类

#### op_add

逐元素加法。

**函数原型：**

```c
void op_add(
    float* a,             // 输入/输出张量 [n]（原地操作）
    const float* b,       // 输入张量 [n]
    int n                 // 元素数量
);
```

**计算公式：**

```
a[i] = a[i] + b[i]
```

**设计说明：**

- 原地操作，结果存入 `a`
- 常用于残差连接（ResNet）

---

### 张量变换类

#### op_concat

沿通道维度拼接张量。

**函数原型：**

```c
void op_concat(
    const float** inputs,     // 输入张量指针数组
    int num_inputs,           // 输入张量数量
    int N,                    // Batch size
    const int* C_per_input,   // 每个输入的通道数数组
    int H,                    // 高度
    int W,                    // 宽度
    float* output             // 输出张量 [N, sum(C_i), H, W]
);
```

**设计说明：**

- 沿 NCHW 的 C 维度（axis=1）拼接
- 所有输入张量的 N、H、W 必须相同

---

#### op_split

沿通道维度分割张量。

**函数原型：**

```c
void op_split(
    const float* input,       // 输入张量 [N, C_total, H, W]
    int N,                    // Batch size
    int C_total,              // 总通道数
    int H,                    // 高度
    int W,                    // 宽度
    float** outputs,          // 输出张量指针数组
    const int* C_per_output,  // 每个输出的通道数数组
    int num_outputs           // 输出张量数量
);
```

**设计说明：**

- `op_concat` 的逆操作
- 沿 NCHW 的 C 维度（axis=1）分割

---

#### op_resize_bilinear

双线性插值缩放。

**函数原型：**

```c
void op_resize_bilinear(
    const float* input,   // 输入张量 [N, C, H_in, W_in]
    int N,                // Batch size
    int C,                // 通道数
    int H_in,             // 输入高度
    int W_in,             // 输入宽度
    float* output,        // 输出张量 [N, C, H_out, W_out]
    int H_out,            // 输出高度
    int W_out,            // 输出宽度
    int align_corners     // 是否对齐角点
);
```

**设计说明：**

- 双线性插值，输出更平滑
- `align_corners` 影响缩放比例计算：
  - `align_corners=1`: `scale = (in_size - 1) / (out_size - 1)`
  - `align_corners=0`: `scale = in_size / out_size`

---

#### op_resize_nearest

最近邻插值缩放。

**函数原型：**

```c
void op_resize_nearest(
    const float* input,   // 输入张量 [N, C, H_in, W_in]
    int N,                // Batch size
    int C,                // 通道数
    int H_in,             // 输入高度
    int W_in,             // 输入宽度
    float* output,        // 输出张量 [N, C, H_out, W_out]
    int H_out,            // 输出高度
    int W_out,            // 输出宽度
    int align_corners     // 是否对齐角点
);
```

**设计说明：**

- 最近邻插值，计算速度快
- 适用于分割任务的上采样

---

### 注意力机制

#### op_attention

缩放点积注意力（单头）。

**函数原型：**

```c
void op_attention(
    const float* Q,       // Query 张量 [batch, seq_q, head_dim]
    const float* K,       // Key 张量 [batch, seq_k, head_dim]
    const float* V,       // Value 张量 [batch, seq_k, head_dim]
    int batch,            // Batch size
    int seq_q,            // Query 序列长度
    int seq_k,            // Key/Value 序列长度
    int head_dim,         // 头维度
    float* output         // 输出张量 [batch, seq_q, head_dim]
);
```

**计算公式：**

```
scores = Q @ K^T / sqrt(head_dim)
attention = softmax(scores)
output = attention @ V
```

**设计说明：**

- 单头注意力实现
- 内部动态分配 scores 矩阵内存
- 用于 Transformer 模型

---

## 量化算子

### 量化原理

本算子库采用 **对称量化**（Symmetric Quantization）方案：

**量化公式（Float → INT8）：**

```
q = round(x / scale) + zp
q = clamp(q, -128, 127)
```

**反量化公式（INT8 → Float）：**

```
x = (q - zp) * scale
```

**量化卷积计算流程：**

```
1. 输入量化：x_int8 = round(x_float / input_scale + input_zp)
2. INT8 累加：acc_int32 = sum((x_int8 - input_zp) * (w_int8 - weight_zp))
3. 反量化：acc_float = acc_int32 * input_scale * weight_scale
4. 加偏置：acc_float += bias
5. 激活（可选）：ReLU/Clip
```

---

### 量化卷积类

#### op_quant_conv2d_int8

INT8 量化卷积。

**函数原型：**

```c
void op_quant_conv2d_int8(
    const float* input,       // 输入张量 [N, C_in, H, W]（float 格式）
    int N,                    // Batch size
    int C_in,                 // 输入通道数
    int H,                    // 输入高度
    int W,                    // 输入宽度
    const int8_t* weight,     // INT8 权重 [C_out, C_in, kH, kW]
    int C_out,                // 输出通道数
    int kH,                   // 卷积核高度
    int kW,                   // 卷积核宽度
    const float* bias,        // FP32 偏置 [C_out]，可为 NULL
    int stride_h,             // 高度方向步长
    int stride_w,             // 宽度方向步长
    int pad_h,                // 高度方向填充
    int pad_w,                // 宽度方向填充
    float* output,            // 输出张量 [N, C_out, H_out, W_out]（float 格式）
    float input_scale,        // 输入量化 scale
    int32_t input_zp,         // 输入量化零点
    float weight_scale,       // 权重量化 scale
    int32_t weight_zp,        // 权重量化零点
    float output_scale,       // 输出量化 scale（当前未使用）
    int32_t output_zp         // 输出量化零点（当前未使用）
);
```

**设计说明：**

- 输入为 float 格式，内部动态量化为 INT8
- 权重预先量化为 INT8
- 输出为 float 格式（反量化后）
- 使用 INT32 累加器避免溢出

---

#### op_quant_conv2d_relu_int8

INT8 量化卷积 + ReLU 激活。

**函数原型：**

```c
void op_quant_conv2d_relu_int8(
    // 参数与 op_quant_conv2d_int8 相同
    ...
);
```

**设计说明：**

- 在反量化后应用 ReLU：`output = max(0, acc_float)`
- 用于 Conv + ReLU 融合场景

---

#### op_quant_depthwise_conv2d_int8

INT8 量化深度可分离卷积。

**函数原型：**

```c
void op_quant_depthwise_conv2d_int8(
    const float* input,       // 输入张量 [N, C_in, H, W]
    int N,                    // Batch size
    int C_in,                 // 输入通道数
    int H,                    // 输入高度
    int W,                    // 输入宽度
    const int8_t* weight,     // INT8 权重 [C_in, kH, kW]
    int kH,                   // 卷积核高度
    int kW,                   // 卷积核宽度
    const float* bias,        // FP32 偏置 [C_in]
    int stride_h,             // 高度方向步长
    int stride_w,             // 宽度方向步长
    int pad_h,                // 高度方向填充
    int pad_w,                // 宽度方向填充
    float* output,            // 输出张量 [N, C_in, H_out, W_out]
    float input_scale,        // 输入量化 scale
    int32_t input_zp,         // 输入量化零点
    float weight_scale,       // 权重量化 scale
    int32_t weight_zp,        // 权重量化零点
    float output_scale,       // 输出量化 scale
    int32_t output_zp         // 输出量化零点
);
```

**设计说明：**

- 每个通道独立进行卷积
- 权重布局为 `[C_in, kH, kW]`
- 常用于 MobileNet 量化推理

---

### 量化全连接类

#### op_quant_linear_int8

INT8 量化全连接层。

**函数原型：**

```c
void op_quant_linear_int8(
    const float* input,       // 输入张量 [batch, in_features]
    int batch,                // Batch size
    int in_features,          // 输入特征数
    const int8_t* weight,     // INT8 权重 [out_features, in_features]
    int out_features,         // 输出特征数
    const float* bias,        // FP32 偏置 [out_features]
    float* output,            // 输出张量 [batch, out_features]
    float input_scale,        // 输入量化 scale
    int32_t input_zp,         // 输入量化零点
    float weight_scale,       // 权重量化 scale
    int32_t weight_zp,        // 权重量化零点
    float output_scale,       // 输出量化 scale
    int32_t output_zp         // 输出量化零点
);
```

**设计说明：**

- 输入动态量化，权重预量化
- 输出为 float 格式
- 常用于分类层量化

---

#### op_quant_linear_relu_int8

INT8 量化全连接层 + ReLU。

**函数原型：**

```c
void op_quant_linear_relu_int8(
    // 参数与 op_quant_linear_int8 相同
    ...
);
```

---

### 量化加法类

#### op_quant_add_int8

量化张量加法。

**函数原型：**

```c
void op_quant_add_int8(
    const float* a,           // 输入张量 A [n]
    const float* b,           // 输入张量 B [n]
    int n,                    // 元素数量
    float a_scale,            // A 的量化 scale
    int32_t a_zp,             // A 的量化零点
    float b_scale,            // B 的量化 scale
    int32_t b_zp,             // B 的量化零点
    float output_scale,       // 输出量化 scale
    int32_t output_zp,        // 输出量化零点
    float* output             // 输出张量 [n]
);
```

**设计说明：**

- 处理两个不同量化参数的张量相加
- 先在浮点域相加，再量化到输出 scale
- 常用于 ResNet 残差连接的量化

---

#### op_quant_add_relu_int8

量化张量加法 + ReLU。

**函数原型：**

```c
void op_quant_add_relu_int8(
    // 参数与 op_quant_add_int8 相同
    ...
);
```

---

## 附录

### A. 文件结构

```
runtime/ops/generic/
├── ops.h              # 浮点算子头文件
├── quant_ops.h        # 量化算子头文件
├── conv2d.c           # 卷积实现
├── depthwise_conv2d.c # 深度可分离卷积
├── linear.c           # 全连接层
├── pooling.c          # 池化层
├── activations.c      # 激活函数
├── batchnorm.c        # 批归一化
├── layernorm.c        # 层归一化
├── softmax.c          # Softmax
├── math_ops.c         # 数学运算
├── concat.c           # 张量拼接
├── split.c            # 张量分割
├── resize.c           # 缩放操作
├── attention.c        # 注意力机制
├── quant_conv2d.c     # 量化卷积
├── quant_linear.c     # 量化全连接
├── quant_depthwise_conv2d.c # 量化深度可分离卷积
└── quant_add.c        # 量化加法
```

### B. 编译依赖

- C99 标准编译器
- 标准库：`<math.h>`, `<stdint.h>`, `<string.h>`, `<stdlib.h>`, `<float.h>`

### C. 性能优化建议

1. **内存访问优化**：使用连续内存布局，提高缓存命中率
2. **循环展开**：对小卷积核进行手动展开
3. **SIMD 优化**：可使用 SSE/AVX/NEON 指令集加速
4. **多线程**：对 batch 维度进行并行化

---

*文档版本: 1.0*  
*最后更新: 2024*
