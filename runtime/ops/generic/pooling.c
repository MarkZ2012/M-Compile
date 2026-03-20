// ops/pooling.c  — 替换原文件，补充 op_maxpool
#include "ops.h"

// Global average pool: [N,C,H,W] -> [N,C,1,1] (output as [N,C])
void op_avgpool_global(const float* x, int N, int C, int H, int W, float* out) {
    int HW = H * W;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            const float* ptr = x + (n * C + c) * HW;
            for (int i = 0; i < HW; i++) sum += ptr[i];
            out[n * C + c] = sum / (float)HW;
        }
    }
}

// MaxPool 2D (NCHW)
// kH/kW: kernel size, sH/sW: stride, pH/pW: padding
void op_maxpool(const float* input, int N, int C, int H, int W,
                int kH, int kW, int sH, int sW, int pH, int pW,
                float* output)
{
    int H_out = (H + 2 * pH - kH) / sH + 1;
    int W_out = (W + 2 * pW - kW) / sW + 1;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float mx = -1e30f;
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int ih = oh * sH - pH + kh;
                            int iw = ow * sW - pW + kw;
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                                continue;
                            float v = input[((n * C + c) * H + ih) * W + iw];
                            if (v > mx) mx = v;
                        }
                    }
                    output[((n * C + c) * H_out + oh) * W_out + ow] = mx;
                }
            }
        }
    }
}
