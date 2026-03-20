/**
 * MyRT - AI Model Runtime
 * 轻量级神经网络推理运行时
 */

#ifndef MYRT_H
#define MYRT_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// 张量数据类型
typedef enum {
    MYRT_FLOAT32 = 0,
    MYRT_INT8 = 1,
    MYRT_INT32 = 2,
    MYRT_FLOAT16 = 3
} myrt_dtype_t;

// 张量结构
typedef struct {
    float* data;        // 数据指针
    int32_t* shape;     // 形状数组
    int32_t ndim;       // 维度数
    int32_t total_size; // 总元素数
    myrt_dtype_t dtype; // 数据类型
} myrt_tensor_t;

// 模型结构
typedef struct myrt_model myrt_model_t;

/**
 * 加载模型
 * @param path 模型文件路径
 * @return 模型指针，失败返回NULL
 */
myrt_model_t* myrt_load(const char* path);

/**
 * 获取模型输入数量
 */
int myrt_get_input_count(myrt_model_t* model);

/**
 * 获取模型输出数量
 */
int myrt_get_output_count(myrt_model_t* model);

/**
 * 创建输入张量
 */
myrt_tensor_t* myrt_create_input_tensor(myrt_model_t* model, int index);

/**
 * 设置输入数据
 */
int myrt_set_input(myrt_model_t* model, int index, myrt_tensor_t* tensor);

/**
 * 运行推理
 */
int myrt_run(myrt_model_t* model);

/**
 * 获取输出张量
 */
myrt_tensor_t* myrt_get_output(myrt_model_t* model, int index);

/**
 * 释放张量
 */
void myrt_free_tensor(myrt_tensor_t* tensor);

/**
 * 释放模型
 */
void myrt_free(myrt_model_t* model);

/**
 * 获取错误信息
 */
const char* myrt_get_error(void);

#ifdef __cplusplus
}
#endif

#endif // MYRT_H