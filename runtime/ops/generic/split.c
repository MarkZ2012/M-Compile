// runtime/ops/generic/split.c
#include "ops.h"
#include <string.h>

// Split tensor along channel dimension (axis=1 for NCHW)
// input: shape [N, C_total, H, W]
// outputs: array of pointers to output tensors, each of shape [N, C_i, H, W]
void op_split(
    const float* input,
    int N, int C_total, int H, int W,
    float** outputs, const int* C_per_output, int num_outputs)
{
    int in_idx = 0;
    for (int n = 0; n < N; n++) {
        for (int i = 0; i < num_outputs; i++) {
            int C_i = C_per_output[i];
            int copy_size = C_i * H * W;
            memcpy(outputs[i] + n * copy_size, input + in_idx, copy_size * sizeof(float));
            in_idx += copy_size;
        }
    }
}