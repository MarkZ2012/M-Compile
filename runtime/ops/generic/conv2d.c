// runtime/ops/conv2d.c
#include "ops.h"
#include <string.h>

void op_conv2d(
    const float* input,  int N, int C_in,  int H,    int W,
    const float* weight, int C_out, int kH, int kW,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    int groups,
    float* output)
{
    int H_out = (H + 2 * pad_h - kH) / stride_h + 1;
    int W_out = (W + 2 * pad_w - kW) / stride_w + 1;

    // Grouped conv (including depthwise when groups == C_in):
    // weight layout: [C_out, C_in/groups, kH, kW]
    int C_in_per_group  = (groups > 0) ? (C_in / groups) : C_in;
    int C_out_per_group = (groups > 0) ? (C_out / groups) : C_out;

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups; g++) {
            for (int co_local = 0; co_local < C_out_per_group; co_local++) {
                int co = g * C_out_per_group + co_local;
                for (int oh = 0; oh < H_out; oh++) {
                    for (int ow = 0; ow < W_out; ow++) {
                        float acc = bias ? bias[co] : 0.0f;

                        for (int ci_local = 0; ci_local < C_in_per_group; ci_local++) {
                            int ci = g * C_in_per_group + ci_local;
                            for (int kh = 0; kh < kH; kh++) {
                                for (int kw = 0; kw < kW; kw++) {
                                    int ih = oh * stride_h - pad_h + kh;
                                    int iw = ow * stride_w - pad_w + kw;
                                    if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                                        continue; // zero padding

                                    float x = input[((n * C_in + ci) * H + ih) * W + iw];
                                    float w = weight[((co * C_in_per_group + ci_local) * kH + kh) * kW + kw];
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
}