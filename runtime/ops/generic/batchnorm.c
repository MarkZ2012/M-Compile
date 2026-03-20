// runtime/ops/batchnorm.c
#include "ops.h"
#include <math.h>

void op_batchnorm(
    float* x,
    int N, int C, int H, int W,
    const float* gamma, const float* beta,
    const float* mean,  const float* var,
    float eps)
{
    int HW = H * W;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float scale = gamma[c] / sqrtf(var[c] + eps);
            float shift = beta[c] - mean[c] * scale;
            float* ptr = x + (n * C + c) * HW;
            for (int i = 0; i < HW; i++) {
                ptr[i] = ptr[i] * scale + shift;
            }
        }
    }
}