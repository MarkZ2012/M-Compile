// runtime/ops/conv2d.c
#include "ops.h"
#include <string.h>

void op_conv2d(
    const float* input,  int N, int C_in,  int H,    int W,
    const float* weight, int C_out, int kH, int kW,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    float* output)
{
    int H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    int W_out = (W + 2 * pad_w - kW) / stride_w + 1;

    for (int n = 0; n < N; n++) {
        for (int co = 0; co < C_out; co++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float acc = bias ? bias[co] : 0.0f;

                    for (int ci = 0; ci < C_in; ci++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh * stride_h - pad_h + kh;
                                int iw = ow * stride_w - pad_w + kw;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                                    continue; // zero padding

                                float x = input[((n * C_in + ci) * H + ih) * W + iw];
                                float w = weight[((co * C_in + ci) * kH + kh) * kW + kw];
                                acc += x * w;
                            }
                        }
                    }
                    output[((n * C_out + co) * H_out + oh) * W_out + ow] = acc;
                }
            }
        }
    }
}