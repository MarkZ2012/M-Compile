/**
 * 模型加载器实现
 */
#include "myrt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char name[256];
    int input_count;
    int output_count;
    int32_t input_shape[4];
    int32_t output_shape[4];
} model_info_t;

struct myrt_model {
    model_info_t info;
    float* weights;
    int32_t weight_count;
    myrt_tensor_t** inputs;
    myrt_tensor_t** outputs;
};

static char g_error_msg[512] = {0};

myrt_model_t* myrt_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        snprintf(g_error_msg, sizeof(g_error_msg), "Failed to open model file: %s", path);
        return NULL;
    }
    
    myrt_model_t* model = (myrt_model_t*)calloc(1, sizeof(myrt_model_t));
    if (!model) {
        fclose(f);
        snprintf(g_error_msg, sizeof(g_error_msg), "Memory allocation failed");
        return NULL;
    }
    
    if (fread(&model->info, sizeof(model_info_t), 1, f) != 1) {
        free(model);
        fclose(f);
        snprintf(g_error_msg, sizeof(g_error_msg), "Failed to read model info");
        return NULL;
    }
    
    if (fread(&model->weight_count, sizeof(int32_t), 1, f) != 1) {
        free(model);
        fclose(f);
        snprintf(g_error_msg, sizeof(g_error_msg), "Failed to read weight count");
        return NULL;
    }
    
    model->weights = (float*)malloc(model->weight_count * sizeof(float));
    if (!model->weights) {
        free(model);
        fclose(f);
        snprintf(g_error_msg, sizeof(g_error_msg), "Memory allocation failed for weights");
        return NULL;
    }
    
    if (fread(model->weights, sizeof(float), model->weight_count, f) != model->weight_count) {
        free(model->weights);
        free(model);
        fclose(f);
        snprintf(g_error_msg, sizeof(g_error_msg), "Failed to read weights");
        return NULL;
    }
    
    fclose(f);
    return model;
}

int myrt_get_input_count(myrt_model_t* model) {
    return model ? model->info.input_count : 0;
}

int myrt_get_output_count(myrt_model_t* model) {
    return model ? model->info.output_count : 0;
}

myrt_tensor_t* myrt_create_input_tensor(myrt_model_t* model, int index) {
    if (!model || index < 0 || index >= model->info.input_count) {
        snprintf(g_error_msg, sizeof(g_error_msg), "Invalid input index");
        return NULL;
    }
    
    myrt_tensor_t* tensor = (myrt_tensor_t*)calloc(1, sizeof(myrt_tensor_t));
    if (!tensor) {
        snprintf(g_error_msg, sizeof(g_error_msg), "Memory allocation failed");
        return NULL;
    }
    
    tensor->ndim = 4;
    tensor->shape = (int32_t*)malloc(4 * sizeof(int32_t));
    memcpy(tensor->shape, model->info.input_shape, 4 * sizeof(int32_t));
    
    tensor->total_size = 1;
    for (int i = 0; i < 4; i++) {
        tensor->total_size *= tensor->shape[i];
    }
    
    tensor->data = (float*)calloc(tensor->total_size, sizeof(float));
    tensor->dtype = MYRT_FLOAT32;
    
    return tensor;
}

int myrt_set_input(myrt_model_t* model, int index, myrt_tensor_t* tensor) {
    if (!model || !tensor || index < 0) {
        snprintf(g_error_msg, sizeof(g_error_msg), "Invalid arguments");
        return -1;
    }
    
    if (!model->inputs) {
        model->inputs = (myrt_tensor_t**)calloc(model->info.input_count, sizeof(myrt_tensor_t*));
    }
    
    model->inputs[index] = tensor;
    return 0;
}

int myrt_run(myrt_model_t* model) {
    if (!model) {
        snprintf(g_error_msg, sizeof(g_error_msg), "Model is NULL");
        return -1;
    }
    
    if (!model->outputs) {
        model->outputs = (myrt_tensor_t**)calloc(model->info.output_count, sizeof(myrt_tensor_t*));
        for (int i = 0; i < model->info.output_count; i++) {
            model->outputs[i] = (myrt_tensor_t*)calloc(1, sizeof(myrt_tensor_t));
            model->outputs[i]->ndim = 4;
            model->outputs[i]->shape = (int32_t*)malloc(4 * sizeof(int32_t));
            memcpy(model->outputs[i]->shape, model->info.output_shape, 4 * sizeof(int32_t));
            model->outputs[i]->total_size = 1;
            for (int j = 0; j < 4; j++) {
                model->outputs[i]->total_size *= model->outputs[i]->shape[j];
            }
            model->outputs[i]->data = (float*)calloc(model->outputs[i]->total_size, sizeof(float));
            model->outputs[i]->dtype = MYRT_FLOAT32;
        }
    }
    
    return 0;
}

myrt_tensor_t* myrt_get_output(myrt_model_t* model, int index) {
    if (!model || !model->outputs || index < 0 || index >= model->info.output_count) {
        snprintf(g_error_msg, sizeof(g_error_msg), "Invalid output index");
        return NULL;
    }
    return model->outputs[index];
}

void myrt_free_tensor(myrt_tensor_t* tensor) {
    if (tensor) {
        free(tensor->data);
        free(tensor->shape);
        free(tensor);
    }
}

void myrt_free(myrt_model_t* model) {
    if (model) {
        free(model->weights);
        if (model->inputs) {
            for (int i = 0; i < model->info.input_count; i++) {
                myrt_free_tensor(model->inputs[i]);
            }
            free(model->inputs);
        }
        if (model->outputs) {
            for (int i = 0; i < model->info.output_count; i++) {
                myrt_free_tensor(model->outputs[i]);
            }
            free(model->outputs);
        }
        free(model);
    }
}

const char* myrt_get_error(void) {
    return g_error_msg;
}