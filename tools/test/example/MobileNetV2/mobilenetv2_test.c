/**
 * MobileNetV2 C 推理测试程序
 * 用于测试编译生成的模型（generic 目标）
 */
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
// 切换到 exe 所在目录，保证 weights/ 相对路径正确
static void cd_to_exe_dir(void) {
    char path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);
    char *last = strrchr(path, '\\');
    if (last) { *last = '\0'; SetCurrentDirectoryA(path); }
}
#endif

// 声明 forward 函数
void model_forward(const float* input, float* output);

static float* load_input_data(const char* path, int size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Error: Cannot open input file: %s\n", path);
        return NULL;
    }

    float* data = (float*)malloc(size * sizeof(float));
    if (!data) {
        fclose(f);
        return NULL;
    }

    fread(data, sizeof(float), size, f);
    fclose(f);
    return data;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    cd_to_exe_dir();
#endif

    printf("============================================================\n");
    printf("MobileNetV2 C Inference Test\n");
    printf("============================================================\n\n");

    // 1) 初始化模型
    printf("[1/4] Initializing model...\n");
    if (model_init() != 0) {
        printf("Error: Failed to initialize model\n");
        return 1;
    }
    printf("  Model initialized successfully\n\n");

    // 2) 加载输入
    printf("[2/4] Loading input data...\n");
    const char* input_file = "input_data.bin";
    const int input_size = 3 * 224 * 224;

    float* input_data = load_input_data(input_file, input_size);
    if (!input_data) {
        printf("Error: Failed to load input data\n");
        printf("  Please run 'python preprocess_image.py' first\n");
        model_cleanup();
        return 1;
    }
    printf("  Input data loaded: %d floats\n\n", input_size);

    // 3) 推理
    printf("[3/4] Running inference...\n");
    float output[1000];
    model_forward(input_data, output);

    // 诊断：NaN/Inf + min/max
    printf("\n[DIAG] Output array check:\n");
    int nan_count = 0, inf_count = 0;
    float min_val = output[0], max_val = output[0], sum = 0.0f;
    for (int i = 0; i < 1000; i++) {
        if (isnan(output[i])) nan_count++;
        if (isinf(output[i])) inf_count++;
        if (output[i] < min_val) min_val = output[i];
        if (output[i] > max_val) max_val = output[i];
        sum += output[i];
    }
    printf("  NaN count: %d\n", nan_count);
    printf("  Inf count: %d\n", inf_count);
    printf("  Min value: %f\n", min_val);
    printf("  Max value: %f\n", max_val);
    printf("  Sum:       %f\n", sum);

    printf("\n[4/4] Top-5 Predictions:\n");
    int top_indices[5];
    float top_probs[5];
    for (int k = 0; k < 5; k++) {
        int max_idx = -1;
        float max_prob = -1.0f;
        for (int i = 0; i < 1000; i++) {
            int already_selected = 0;
            for (int j = 0; j < k; j++) {
                if (top_indices[j] == i) { already_selected = 1; break; }
            }
            if (!already_selected && output[i] > max_prob) {
                max_prob = output[i];
                max_idx = i;
            }
        }
        top_indices[k] = max_idx;
        top_probs[k] = max_prob;
    }

    for (int k = 0; k < 5; k++) {
        printf("  [%d] Class %d: %.4f%%\n", k, top_indices[k], top_probs[k] * 100.0f);
    }

    free(input_data);
    model_cleanup();

    printf("\nTest completed successfully!\n");
    return 0;
}

