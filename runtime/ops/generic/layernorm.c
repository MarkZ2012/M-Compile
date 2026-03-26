// runtime/ops/generic/layernorm.c
#include "ops.h"
#include <math.h>

// Layer Normalization
// x shape: [N, ...] where the last dimension of size norm_size is normalized
// output shape same as x
// gamma and beta shape: [norm_size]
void op_layernorm(
    const float* x,
    int N, int norm_size,
    const float* gamma,
    const float* beta,
    float eps,
    float* output)
{
    int total = N * norm_size;
    for (int i = 0; i < N; i++) {
        const float* x_ptr = x + i * norm_size;
        float* out_ptr = output + i * norm_size;
        
        // compute mean
        float mean = 0.0f;
        for (int j = 0; j < norm_size; j++) {
            mean += x_ptr[j];
        }
        mean /= norm_size;
        
        // compute variance
        float var = 0.0f;
        for (int j = 0; j < norm_size; j++) {
            float diff = x_ptr[j] - mean;
            var += diff * diff;
        }
        var /= norm_size;
        
        // normalize
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < norm_size; j++) {
            out_ptr[j] = (x_ptr[j] - mean) * inv_std * gamma[j] + beta[j];
        }
    }
}