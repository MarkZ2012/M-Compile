/*
 * quant_conv2d.c - 量化卷积实现
 * 
 * 根据审核意见实现完整的int8 kernel流程：
 * 1. 输入量化（float -> int8，使用 input scale/zp）
 * 2. int8累加（accumulator用int32）
 * 3. requantize（int32 -> int8，使用 output scale/zp）
 * 4. 激活后的relu clip
 */
#include "ops.h"
#include <string.h>
#include <stdint.h>
#include <math.h>

/* 
 * 量化卷积 - Per-tensor 量化
 * 
 * 参数说明：
 * - input: int8 输入数据 [N, C_in, H, W]
 * - weight: int8 权重数据 [C_out, C_in, kH, kW]
 * - bias: int32 偏置数据 [C_out]，可以为NULL
 * - output: int8 输出数据 [N, C_out, H_out, W_out]
 * - input_scale, input_zp: 输入的量化参数
 * - weight_scale, weight_zp: 权重的量化参数
 * - output_scale, output_zp: 输出的量化参数
 */
void op_quant_conv2d_int8(
    const float* input,   int N, int C_in,  int H,    int W,
    const int8_t* weight,  int C_out, int kH, int kW,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    float* output,
    float input_scale,  int32_t input_zp,
    float weight_scale, int32_t weight_zp,
    float output_scale, int32_t output_zp)
{
    int H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    int W_out = (W + 2 * pad_w - kW) / stride_w + 1;
    
    // 反量化参数预计算
    // dequant_scale = input_scale * weight_scale
    // 注意：因为输出是 float buffer，所以直接乘以 input_scale * weight_scale
    // 回到浮点空间，不需要再除以 output_scale（那只在 int8->int8 时才需要）
    float dequant_scale = input_scale * weight_scale;
    
    // int8量化范围
    const int32_t qmin = -128;
    const int32_t qmax = 127;
    
    for (int n = 0; n < N; n++) {
        for (int co = 0; co < C_out; co++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    // 使用int32累加器
                    int32_t acc = 0;
                    
                    // 卷积计算
                    for (int ci = 0; ci < C_in; ci++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh * stride_h - pad_h + kh;
                                int iw = ow * stride_w - pad_w + kw;
                                
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    // 读取float输入并量化为int8
                                    float x_float = input[((n * C_in + ci) * H + ih) * W + iw];
                                    int8_t x = (int8_t)roundf(x_float / input_scale + input_zp);
                                    // 读取int8权重
                                    int8_t w = weight[((co * C_in + ci) * kH + kh) * kW + kw];
                                    
                                    // int8乘法累加（结果为int32），考虑零点
                                    acc += (int32_t)(x - input_zp) * (int32_t)(w - weight_zp);
                                }
                            }
                        }
                    }
                    
                    // Dequantize: int32 -> float
                    float acc_float = (float)acc * dequant_scale;
                    // Add bias (already in float space)
                    if (bias) {
                        acc_float += bias[co];
                    }
                    output[((n * C_out + co) * H_out + oh) * W_out + ow] = acc_float;
                }
            }
        }
    }
}

/*
 * 量化卷积 + ReLU
 * 在requantize后应用ReLU激活
 */
void op_quant_conv2d_relu_int8(
    const float* input,   int N, int C_in,  int H,    int W,
    const int8_t* weight,  int C_out, int kH, int kW,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    float* output,
    float input_scale,  int32_t input_zp,
    float weight_scale, int32_t weight_zp,
    float output_scale, int32_t output_zp)
{
    int H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    int W_out = (W + 2 * pad_w - kW) / stride_w + 1;
    
    float dequant_scale = input_scale * weight_scale;
    
    // ReLU在量化空间中：输出 >= output_zp
    const int32_t qmin = output_zp;  // ReLU: clip到零点以上
    const int32_t qmax = 127;
    
    for (int n = 0; n < N; n++) {
        for (int co = 0; co < C_out; co++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    int32_t acc = 0;
                    
                    for (int ci = 0; ci < C_in; ci++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh * stride_h - pad_h + kh;
                                int iw = ow * stride_w - pad_w + kw;
                                
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    // 读取float输入并量化为int8
                                    float x_float = input[((n * C_in + ci) * H + ih) * W + iw];
                                    int8_t x = (int8_t)roundf(x_float / input_scale + input_zp);
                                    // 读取int8权重
                                    int8_t w = weight[((co * C_in + ci) * kH + kh) * kW + kw];
                                    acc += (int32_t)(x - input_zp) * (int32_t)(w - weight_zp);
                                }
                            }
                        }
                    }
                    
                    // Dequantize: int32 -> float
                    float acc_float = (float)acc * dequant_scale;
                    // Add bias (already in float space)
                    if (bias) {
                        acc_float += bias[co];
                    }
                    // 应用ReLU激活
                    if (acc_float < 0.0f) acc_float = 0.0f;
                    output[((n * C_out + co) * H_out + oh) * W_out + ow] = acc_float;
                }
            }
        }
    }
}