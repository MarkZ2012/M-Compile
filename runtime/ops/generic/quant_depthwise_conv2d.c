// runtime/ops/generic/quant_depthwise_conv2d.c
#include "ops.h"
#include <string.h>
#include <stdint.h>
#include <math.h>

/*
 * 量化深度可分离卷积 (Depthwise Conv2D) - Per-tensor 量化
 * 
 * 参数说明：
 * - input: float 输入数据 [N, C_in, H, W] (实际是量化后的float，但我们将量化为int8)
 * - weight: int8 权重数据 [C_in, 1, kH, kW] 或 [C_in, kH, kW] (depthwise)
 * - bias: float 偏置数据 [C_in]，可以为NULL
 * - output: float 输出数据 [N, C_in, H_out, W_out]
 * - input_scale, input_zp: 输入的量化参数
 * - weight_scale, weight_zp: 权重的量化参数
 * - output_scale, output_zp: 输出的量化参数
 * 
 * 注意：此实现假设权重是 depthwise 的，即 groups = C_in，每个通道独立卷积。
 */
void op_quant_depthwise_conv2d_int8(
    const float* input,   int N, int C_in,  int H,    int W,
    const int8_t* weight, int kH, int kW,
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
    float dequant_scale = input_scale * weight_scale;
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C_in; c++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    int32_t acc = 0;
                    
                    // 卷积计算，只在当前通道c上进行
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int ih = oh * stride_h - pad_h + kh;
                            int iw = ow * stride_w - pad_w + kw;
                            
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                // 读取float输入并量化为int8
                                float x_float = input[((n * C_in + c) * H + ih) * W + iw];
                                int8_t x = (int8_t)roundf(x_float / input_scale + input_zp);
                                // 读取int8权重，权重布局假设为 [C_in, 1, kH, kW] 或 [C_in, kH, kW]
                                // 这里假设权重索引为 ((c * 1 + 0) * kH + kh) * kW + kw，但为了简单，假设权重是 [C_in, kH, kW]
                                int8_t w = weight[(c * kH + kh) * kW + kw];
                                
                                acc += (int32_t)(x - input_zp) * (int32_t)(w - weight_zp);
                            }
                        }
                    }
                    
                    // Dequantize: int32 -> float
                    float acc_float = (float)acc * dequant_scale;
                    // Add bias
                    if (bias) {
                        acc_float += bias[c];
                    }
                    output[((n * C_in + c) * H_out + oh) * W_out + ow] = acc_float;
                }
            }
        }
    }
}

/*
 * 量化深度可分离卷积 + ReLU
 */
void op_quant_depthwise_conv2d_relu_int8(
    const float* input,   int N, int C_in,  int H,    int W,
    const int8_t* weight, int kH, int kW,
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
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C_in; c++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    int32_t acc = 0;
                    
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int ih = oh * stride_h - pad_h + kh;
                            int iw = ow * stride_w - pad_w + kw;
                            
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                float x_float = input[((n * C_in + c) * H + ih) * W + iw];
                                int8_t x = (int8_t)roundf(x_float / input_scale + input_zp);
                                int8_t w = weight[(c * kH + kh) * kW + kw];
                                acc += (int32_t)(x - input_zp) * (int32_t)(w - weight_zp);
                            }
                        }
                    }
                    
                    float acc_float = (float)acc * dequant_scale;
                    if (bias) {
                        acc_float += bias[c];
                    }
                    // 应用ReLU激活
                    if (acc_float < 0.0f) acc_float = 0.0f;
                    output[((n * C_in + c) * H_out + oh) * W_out + ow] = acc_float;
                }
            }
        }
    }
}