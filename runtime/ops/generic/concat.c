// runtime/ops/generic/concat.c
#include "ops.h"
#include <string.h>

// Concatenate tensors along channel dimension (axis=1 for NCHW)
// inputs: array of pointers to input tensors, each of shape [N, C_i, H, W]
// output: shape [N, sum(C_i), H, W]
void op_concat(
    const float** inputs, int num_inputs,
    int N, const int* C_per_input, int H, int W,
    float* output)
{
    int total_C = 0;
    for (int i = 0; i < num_inputs; i++) {
        total_C += C_per_input[i];
    }
    
    int out_idx = 0;
    for (int n = 0; n < N; n++) {
        for (int i = 0; i < num_inputs; i++) {
            int C_i = C_per_input[i];
            int copy_size = C_i * H * W;
            memcpy(output + out_idx, inputs[i] + n * copy_size, copy_size * sizeof(float));
            out_idx += copy_size;
        }
    }
}