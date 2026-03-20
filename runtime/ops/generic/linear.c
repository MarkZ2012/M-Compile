// runtime/ops/linear.c
#include "ops.h"

void op_linear(const float* x, const float* w, const float* b,
               int batch, int in_feat, int out_feat, float* out) {
    for (int n = 0; n < batch; n++) {
        for (int o = 0; o < out_feat; o++) {
            float acc = b ? b[o] : 0.0f;
            for (int i = 0; i < in_feat; i++)
                acc += x[n * in_feat + i] * w[o * in_feat + i];
            out[n * out_feat + o] = acc;
        }
    }
}