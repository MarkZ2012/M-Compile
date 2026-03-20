/**
 * ResNet18 C推理测试程序
 * 用于测试编译生成的模型
 */
#include "resnet18.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
// 切换到 exe 所在目录，这样 weights/ 的相对路径永远正确
static void cd_to_exe_dir(void) {
    char path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);
    // 截掉文件名，只保留目录部分
    char *last = strrchr(path, '\\');
    if (last) { *last = '\0'; SetCurrentDirectoryA(path); }
}
#endif

// 声明forward函数
void resnet18_forward(const float* input, float* output);

// ImageNet中与猫相关的类别
const char* cat_labels[] = {
    [281] = "tabby, tabby cat",
    [282] = "tiger cat",
    [283] = "Persian cat",
    [284] = "Siamese cat, Siamese",
    [285] = "Egyptian cat",
    [286] = "cougar, puma, catamount, mountain lion, painter, panther",
    [287] = "lynx, catamount",
    [288] = "leopard, Panthera pardus",
    [289] = "snow leopard, ounce, Panthera uncia",
    [290] = "jaguar, panther, Panthera onca, Felis onca",
    [291] = "lion, king of beasts, Panthera leo",
    [292] = "tiger, Panthera tigris",
    [293] = "cheetah, chetah, Acinonyx jubatus",
};

/**
 * 加载预处理的输入数据
 */
float* load_input_data(const char* path, int size) {
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
    printf("ResNet18 C Inference Test (Full Implementation)\n");
    printf("============================================================\n\n");
    
    // 1. 初始化模型
    printf("[1/4] Initializing model...\n");
    if (resnet18_init() != 0) {
        printf("Error: Failed to initialize model\n");
        return 1;
    }
    printf("  Model initialized successfully\n\n");
    
    // 2. 加载输入数据
    printf("[2/4] Loading input data...\n");
    const char* input_file = "input_data.bin";
    int input_size = 3 * 224 * 224;  // RGB image, 224x224
    float* input_data = load_input_data(input_file, input_size);
    if (!input_data) {
        printf("Error: Failed to load input data\n");
        printf("  Please run 'python preprocess_image.py' first\n");
        resnet18_cleanup();
        return 1;
    }
    printf("  Input data loaded: %d floats\n\n", input_size);
    
    // 3. 运行推理
    printf("[3/4] Running inference...\n");
    printf("  This may take a few seconds...\n");
    
    // 创建输出
    float output[1000];
    
    // 调用forward函数
    resnet18_forward(input_data, output);

    // === 加入以下诊断代码 ===
    printf("\n[DIAG] Output array check:\n");
    int nan_count = 0, inf_count = 0, neg_count = 0;
    float min_val = output[0], max_val = output[0], sum = 0;
    for (int i = 0; i < 1000; i++) {
        if (isnan(output[i]))  nan_count++;
        if (isinf(output[i]))  inf_count++;
        if (output[i] < 0)     neg_count++;
        if (output[i] < min_val) min_val = output[i];
        if (output[i] > max_val) max_val = output[i];
        sum += output[i];
    }
    printf("  NaN count:      %d\n", nan_count);
    printf("  Inf count:      %d\n", inf_count);
    printf("  Negative count: %d\n", neg_count);
    printf("  Min value:      %f\n", min_val);
    printf("  Max value:      %f\n", max_val);
    printf("  Sum:            %f\n", sum);
    // ======================
    
    printf("  Inference completed\n\n");
    
    // 4. 显示结果
    printf("[4/4] Results:\n");
    printf("============================================================\n");
    printf("C Inference Results:\n");
    printf("============================================================\n");
    
    // 找到top-5预测
    int top_indices[5];
    float top_probs[5];
    
    for (int k = 0; k < 5; k++) {
        int max_idx = -1;
        float max_prob = -1;
        
        for (int i = 0; i < 1000; i++) {
            int already_selected = 0;
            for (int j = 0; j < k; j++) {
                if (top_indices[j] == i) {
                    already_selected = 1;
                    break;
                }
            }
            
            if (!already_selected && output[i] > max_prob) {
                max_prob = output[i];
                max_idx = i;
            }
        }
        
        top_indices[k] = max_idx;
        top_probs[k] = max_prob;
    }
    
    // 显示猫相关的预测
    int found_cat = 0;
    for (int k = 0; k < 5; k++) {
        int idx = top_indices[k];
        if (idx >= 281 && idx <= 293 && cat_labels[idx] != NULL) {
            found_cat = 1;
            printf("  [%d] %s: %.2f%%\n", idx, cat_labels[idx], top_probs[k] * 100);
        }
    }
    
    if (!found_cat) {
        printf("  No cat detected in top predictions.\n");
        printf("\n  Top predictions:\n");
        for (int k = 0; k < 5; k++) {
            printf("    Class %d: %.2f%%\n", top_indices[k], top_probs[k] * 100);
        }
    }
    
    printf("============================================================\n\n");
    
    // 清理
    free(input_data);
    resnet18_cleanup();
    
    printf("Test completed successfully!\n");
    
    return 0;
}
