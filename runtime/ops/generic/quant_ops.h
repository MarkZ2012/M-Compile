/*
 * quant_ops.h - 量化算子头文件
 */
#ifndef QUANT_OPS_H
#define QUANT_OPS_H

#include <stdint.h>

/* 量化卷积 - Per-tensor */
void op_quant_conv2d_int8(
    const float* input,   int N, int C_in,  int H,    int W,
    const int8_t* weight,  int C_out, int kH, int kW,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    float* output,
    float input_scale,  int32_t input_zp,
    float weight_scale, int32_t weight_zp,
    float output_scale, int32_t output_zp);

/* 量化卷积 + ReLU */
void op_quant_conv2d_relu_int8(
    const float* input,   int N, int C_in,  int H,    int W,
    const int8_t* weight,  int C_out, int kH, int kW,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    float* output,
    float input_scale,  int32_t input_zp,
    float weight_scale, int32_t weight_zp,
    float output_scale, int32_t output_zp);

/* 量化全连接层 - Per-tensor */
void op_quant_linear_int8(
    const float* input,  int batch, int in_features,
    const int8_t* weight, int out_features,
    const float* bias,
    float* output,
    float input_scale,  int32_t input_zp,
    float weight_scale, int32_t weight_zp,
    float output_scale, int32_t output_zp);

/* 量化全连接层 + ReLU */
void op_quant_linear_relu_int8(
    const float* input,  int batch, int in_features,
    const int8_t* weight, int out_features,
    const float* bias,
    float* output,
    float input_scale,  int32_t input_zp,
    float weight_scale, int32_t weight_zp,
    float output_scale, int32_t output_zp);

/* 量化深度可分离卷积 (Depthwise Conv2D) */
void op_quant_depthwise_conv2d_int8(
    const float* input,   int N, int C_in,  int H,    int W,
    const int8_t* weight, int kH, int kW,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    float* output,
    float input_scale,  int32_t input_zp,
    float weight_scale, int32_t weight_zp,
    float output_scale, int32_t output_zp);

/* 量化深度可分离卷积 + ReLU */
void op_quant_depthwise_conv2d_relu_int8(
    const float* input,   int N, int C_in,  int H,    int W,
    const int8_t* weight, int kH, int kW,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    float* output,
    float input_scale,  int32_t input_zp,
    float weight_scale, int32_t weight_zp,
    float output_scale, int32_t output_zp);

/* 量化加法 */
void op_quant_add_int8(
    const float* a, const float* b,
    int n,
    float a_scale, int32_t a_zp,
    float b_scale, int32_t b_zp,
    float output_scale, int32_t output_zp,
    float* output);

/* 量化加法 + ReLU */
void op_quant_add_relu_int8(
    const float* a, const float* b,
    int n,
    float a_scale, int32_t a_zp,
    float b_scale, int32_t b_zp,
    float output_scale, int32_t output_zp,
    float* output);

#endif /* QUANT_OPS_H */
