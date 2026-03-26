// runtime/ops/generic/attention.c
#include "ops.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>

// Scaled dot-product attention (single head)
// Q shape: [batch, seq_q, head_dim]
// K shape: [batch, seq_k, head_dim]
// V shape: [batch, seq_k, head_dim]
// output shape: [batch, seq_q, head_dim]
// Note: This is a naive implementation for simplicity.
void op_attention(
    const float* Q, const float* K, const float* V,
    int batch, int seq_q, int seq_k, int head_dim,
    float* output)
{
    // Compute attention scores: scores = Q * K^T / sqrt(head_dim)
    // Then softmax over last dimension (seq_k)
    // Then output = scores * V
    
    int q_size = seq_q * head_dim;
    int k_size = seq_k * head_dim;
    int scores_size = seq_q * seq_k;
    
    for (int b = 0; b < batch; b++) {
        const float* Q_b = Q + b * q_size;
        const float* K_b = K + b * k_size;
        const float* V_b = V + b * k_size;
        float* out_b = output + b * q_size;
        
        // Compute scores matrix (seq_q x seq_k)
        float* scores = (float*)malloc(scores_size * sizeof(float));
        float scale = 1.0f / sqrtf((float)head_dim);
        
        for (int i = 0; i < seq_q; i++) {
            for (int j = 0; j < seq_k; j++) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += Q_b[i * head_dim + d] * K_b[j * head_dim + d];
                }
                scores[i * seq_k + j] = dot * scale;
            }
        }
        
        // Softmax per row (over seq_k)
        for (int i = 0; i < seq_q; i++) {
            float* row = scores + i * seq_k;
            float max_val = -FLT_MAX;
            for (int j = 0; j < seq_k; j++) {
                if (row[j] > max_val) max_val = row[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < seq_k; j++) {
                row[j] = expf(row[j] - max_val);
                sum += row[j];
            }
            for (int j = 0; j < seq_k; j++) {
                row[j] /= sum;
            }
        }
        
        // Compute output: scores * V
        for (int i = 0; i < seq_q; i++) {
            for (int d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (int j = 0; j < seq_k; j++) {
                    acc += scores[i * seq_k + j] * V_b[j * head_dim + d];
                }
                out_b[i * head_dim + d] = acc;
            }
        }
        
        free(scores);
    }
}