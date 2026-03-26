// runtime/ops/generic/depthwise_conv2d.c
// Depthwise Conv2D wrapper: groups = C_in
#include "ops.h"

void op_depthwise_conv2d(
    const float* input,  int N, int C_in,  int H,    int W,
    const float* weight, int kH, int kW,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    float* output)
{
    // Depthwise conv: groups = C_in, weight shape [C_in, 1, kH, kW] (or [C_in, kH, kW] for convenience)
    // We assume weight is laid out as [C_in, 1, kH, kW] (C_out = C_in)
    // Call generic conv2d with groups = C_in
    op_conv2d(input, N, C_in, H, W,
              weight, C_in, kH, kW,
              bias,
              stride_h, stride_w,
              pad_h, pad_w,
              C_in, // groups = C_in for depthwise
              output);
}