// runtime/ops/math_ops.c
#include "ops.h"

void op_add(float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

// Clip: y = min(max(x, min_v), max_v)
// In our current ONNX export (MobileNetV2), min_v/max_v are scalar constants.
void op_clip(
    const float* x,
    const float* min_v,
    const float* max_v,
    int n,
    float* out
) {
    float mn = min_v[0];
    float mx = max_v[0];
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v < mn) v = mn;
        if (v > mx) v = mx;
        out[i] = v;
    }
}