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
    int groups,
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

// Clip (typically ReLU6): y = min(max(x, min_v), max_v)
// Note: in current compiler usage, min_v/max_v are scalar constants.
void op_clip(
    const float* x,
    const float* min_v,
    const float* max_v,
    int n,
    float* out
);

// Depthwise Conv2D (groups = C_in)
void op_depthwise_conv2d(
    const float* input,  int N, int C_in,  int H,    int W,
    const float* weight, int kH, int kW,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h,    int pad_w,
    float* output);

// Resize bilinear
void op_resize_bilinear(
    const float* input, int N, int C, int H_in, int W_in,
    float* output, int H_out, int W_out,
    int align_corners);

// Resize nearest
void op_resize_nearest(
    const float* input, int N, int C, int H_in, int W_in,
    float* output, int H_out, int W_out,
    int align_corners);

// Concat along channel dimension (axis=1 for NCHW)
void op_concat(
    const float** inputs, int num_inputs,
    int N, const int* C_per_input, int H, int W,
    float* output);

// Split along channel dimension (axis=1 for NCHW)
void op_split(
    const float* input,
    int N, int C_total, int H, int W,
    float** outputs, const int* C_per_output, int num_outputs);

// Layer Normalization
void op_layernorm(
    const float* x,
    int N, int norm_size,
    const float* gamma,
    const float* beta,
    float eps,
    float* output);

// Scaled dot-product attention (single head)
void op_attention(
    const float* Q, const float* K, const float* V,
    int batch, int seq_q, int seq_k, int head_dim,
    float* output);
