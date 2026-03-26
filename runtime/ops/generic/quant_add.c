// runtime/ops/generic/quant_add.c
#include "ops.h"
#include <stdint.h>
#include <math.h>

/*
 * 量化加法 - 两个量化张量相加
 * 处理两路输入的 scale 对齐问题。
 * 假设输入 a 和 b 都是 float 类型（实际上是量化后的浮点表示），
 * 但我们需要将它们量化到统一的 scale，然后相加，再反量化。
 * 
 * 参数说明：
 * - a, b: 输入张量，float 类型，形状相同
 * - n: 元素数量
 * - a_scale, a_zp: 张量 a 的量化参数
 * - b_scale, b_zp: 张量 b 的量化参数
 * - output_scale, output_zp: 输出的量化参数
 * - output: 输出张量，float 类型
 */
void op_quant_add_int8(
    const float* a, const float* b,
    int n,
    float a_scale, int32_t a_zp,
    float b_scale, int32_t b_zp,
    float output_scale, int32_t output_zp,
    float* output)
{
    // 将 a 和 b 量化到 int8，使用各自的 scale 和 zp
    // 然后相加，再反量化到 output scale
    // 或者直接在浮点域相加，但需要统一 scale
    // 简单方法：将 a 和 b 反量化到浮点，然后相加，再量化到 output scale
    // 但这里我们直接在浮点域相加，因为输入已经是浮点（量化后的浮点表示）
    // 实际上，输入 a 和 b 是量化后的浮点值，即已经乘以 scale 并加上 zp？
    // 根据量化公式：float_val = (int8_val - zp) * scale
    // 所以如果输入是 float，那么它们已经是反量化后的值，可以直接相加。
    // 但输出也需要量化，所以我们需要将相加后的浮点结果量化到 output scale。
    
    for (int i = 0; i < n; i++) {
        // 直接相加（因为已经是浮点值）
        float sum = a[i] + b[i];
        // 量化到 output scale
        int32_t quantized = (int32_t)roundf(sum / output_scale + output_zp);
        // 裁剪到 int8 范围
        if (quantized < -128) quantized = -128;
        if (quantized > 127) quantized = 127;
        // 反量化回浮点（保持一致性）
        output[i] = (quantized - output_zp) * output_scale;
    }
}

/*
 * 量化加法 + ReLU
 */
void op_quant_add_relu_int8(
    const float* a, const float* b,
    int n,
    float a_scale, int32_t a_zp,
    float b_scale, int32_t b_zp,
    float output_scale, int32_t output_zp,
    float* output)
{
    for (int i = 0; i < n; i++) {
        float sum = a[i] + b[i];
        // 应用ReLU
        if (sum < 0.0f) sum = 0.0f;
        // 量化到 output scale
        int32_t quantized = (int32_t)roundf(sum / output_scale + output_zp);
        if (quantized < -128) quantized = -128;
        if (quantized > 127) quantized = 127;
        output[i] = (quantized - output_zp) * output_scale;
    }
}