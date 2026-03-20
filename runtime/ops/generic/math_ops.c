// runtime/ops/math_ops.c
#include "ops.h"

void op_add(float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}