/*
 * quant_linear.c - 量化全连接层实现
 * 
 * 与量化卷积类似的int8 kernel流程：
 * 1. 输入量化（float -> int8）
 * 2. int8累加（accumulator用int32）
 * 3. requantize（int32 -> int8）
 * 4. 激活后的relu clip
 */
#include "ops.h"
#include <string.h>
#include <stdint.h>
#include <math.h>

/*
 * 量化全连接层 - Per-tensor 量化
 * 
 * 参数说明：
 * - input: int8 输入数据 [batch, in_features]
 * - weight: int8 权重数据 [out_features, in_features]
 * - bias: int32 偏置数据 [out_features]，可以为NULL
 * - output: int8 输出数据 [batch, out_features]
 */
void op_quant_linear_int8(
    const float* input,  int batch, int in_features,
    const int8_t* weight, int out_features,
    const float* bias,
    float* output,
    float input_scale,  int32_t input_zp,
    float weight_scale, int32_t weight_zp,
    float output_scale, int32_t output_zp)
{
    // dequant_scale = input_scale * weight_scale
    // 输出是 float buffer，直接反量化到浮点空间
    float dequant_scale = input_scale * weight_scale;
    
    const int32_t qmin = -128;
    const int32_t qmax = 127;
    
    for (int b = 0; b < batch; b++) {
        for (int o = 0; o < out_features; o++) {
            // 使用int32累加器
            int32_t acc = 0;
            
            // 矩阵乘法
            for (int i = 0; i < in_features; i++) {
                // 读取float输入并量化为int8
                float x_float = input[b * in_features + i];
                int8_t x = (int8_t)roundf(x_float / input_scale + input_zp);
                // 读取int8权重
                int8_t w = weight[o * in_features + i];
                acc += (int32_t)(x - input_zp) * (int32_t)(w - weight_zp);
            }
            
            // Dequantize: int32 -> float
            float acc_float = (float)acc * dequant_scale;
            // Add bias (already in float space)
            if (bias) {
                acc_float += bias[o];
            }
            output[b * out_features + o] = acc_float;
        }
    }
}

/*
 * 量化全连接层 + ReLU
 */
void op_quant_linear_relu_int8(
    const float* input,  int batch, int in_features,
    const int8_t* weight, int out_features,
    const float* bias,
    float* output,
    float input_scale,  int32_t input_zp,
    float weight_scale, int32_t weight_zp,
    float output_scale, int32_t output_zp)
{
    float dequant_scale = input_scale * weight_scale;
    
    // ReLU在量化空间中：输出 >= output_zp
    const int32_t qmin = output_zp;
    const int32_t qmax = 127;
    
    for (int b = 0; b < batch; b++) {
        for (int o = 0; o < out_features; o++) {
            int32_t acc = 0;
            
            for (int i = 0; i < in_features; i++) {
                // 读取float输入并量化为int8
                float x_float = input[b * in_features + i];
                int8_t x = (int8_t)roundf(x_float / input_scale + input_zp);
                // 读取int8权重
                int8_t w = weight[o * in_features + i];
                acc += (int32_t)(x - input_zp) * (int32_t)(w - weight_zp);
            }
            
            // Dequantize: int32 -> float
            float acc_float = (float)acc * dequant_scale;
            // Add bias (already in float space)
            if (bias) {
                acc_float += bias[o];
            }
            // 应用ReLU激活
            if (acc_float < 0.0f) acc_float = 0.0f;
            output[b * out_features + o] = acc_float;
        }
    }
}