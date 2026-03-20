// ops/ops.h
#pragma once
#include <stdint.h>

// Conv2d NCHW
// input:  [N, C_in,  H,    W   ]
// weight: [C_out, C_in, kH, kW ]
// bias:   [C_out] or NULL
// output: [N, C_out, H_out, W_out]
void op_conv2d(
    const float* input,  int N, int C_in, int H, int W,
    const float* weight, int C_out, int kH, int kW,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    float* output
);

// BatchNorm inference (in-place)
// x shape: [N, C, H, W]
void op_batchnorm(
    float* x,
    int N, int C, int H, int W,
    const float* gamma,   // scale (BN weight)
    const float* beta,    // shift (BN bias)
    const float* mean,    // running_mean
    const float* var,     // running_var
    float eps
);

// ReLU in-place
void op_relu(float* x, int n);

// Element-wise add: a += b  (in-place on a)
void op_add(float* a, const float* b, int n);

// Global average pool: [N,C,H,W] -> [N,C]
void op_avgpool_global(const float* x, int N, int C, int H, int W, float* out);

// MaxPool 2D (NCHW)
void op_maxpool(const float* input, int N, int C, int H, int W,
                int kH, int kW, int sH, int sW, int pH, int pW,
                float* output);

// Fully connected: out[batch, out_feat] = x[batch, in_feat] * w[out_feat, in_feat]^T + b
void op_linear(const float* x, const float* w, const float* b,
               int batch, int in_feat, int out_feat, float* out);

// Softmax in-place
void op_softmax(float* x, int n);
