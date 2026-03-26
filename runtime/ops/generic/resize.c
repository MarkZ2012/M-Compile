// runtime/ops/generic/resize.c
#include "ops.h"
#include <math.h>

// 双线性插值 resize
void op_resize_bilinear(
    const float* input, int N, int C, int H_in, int W_in,
    float* output, int H_out, int W_out,
    int align_corners) 
{
    float scale_h = align_corners ? (float)(H_in - 1) / (H_out - 1) : (float)H_in / H_out;
    float scale_w = align_corners ? (float)(W_in - 1) / (W_out - 1) : (float)W_in / W_out;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float ih_float = align_corners ? oh * scale_h : (oh + 0.5f) * scale_h - 0.5f;
                    float iw_float = align_corners ? ow * scale_w : (ow + 0.5f) * scale_w - 0.5f;
                    
                    int ih0 = (int)floorf(ih_float);
                    int iw0 = (int)floorf(iw_float);
                    int ih1 = ih0 + 1;
                    int iw1 = iw0 + 1;
                    
                    float w0 = (ih_float - ih0);
                    float w1 = (iw_float - iw0);
                    
                    // clamp to valid range
                    if (ih0 < 0) ih0 = 0;
                    if (iw0 < 0) iw0 = 0;
                    if (ih1 >= H_in) ih1 = H_in - 1;
                    if (iw1 >= W_in) iw1 = W_in - 1;
                    
                    float val = 0.0f;
                    val += input[((n * C + c) * H_in + ih0) * W_in + iw0] * (1 - w0) * (1 - w1);
                    val += input[((n * C + c) * H_in + ih1) * W_in + iw0] * w0 * (1 - w1);
                    val += input[((n * C + c) * H_in + ih0) * W_in + iw1] * (1 - w0) * w1;
                    val += input[((n * C + c) * H_in + ih1) * W_in + iw1] * w0 * w1;
                    
                    output[((n * C + c) * H_out + oh) * W_out + ow] = val;
                }
            }
        }
    }
}

// 最近邻 resize
void op_resize_nearest(
    const float* input, int N, int C, int H_in, int W_in,
    float* output, int H_out, int W_out,
    int align_corners) 
{
    float scale_h = align_corners ? (float)(H_in - 1) / (H_out - 1) : (float)H_in / H_out;
    float scale_w = align_corners ? (float)(W_in - 1) / (W_out - 1) : (float)W_in / W_out;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float ih_float = align_corners ? oh * scale_h : (oh + 0.5f) * scale_h - 0.5f;
                    float iw_float = align_corners ? ow * scale_w : (ow + 0.5f) * scale_w - 0.5f;
                    
                    int ih = (int)roundf(ih_float);
                    int iw = (int)roundf(iw_float);
                    
                    // clamp
                    if (ih < 0) ih = 0;
                    if (iw < 0) iw = 0;
                    if (ih >= H_in) ih = H_in - 1;
                    if (iw >= W_in) iw = W_in - 1;
                    
                    output[((n * C + c) * H_out + oh) * W_out + ow] = 
                        input[((n * C + c) * H_in + ih) * W_in + iw];
                }
            }
        }
    }
}