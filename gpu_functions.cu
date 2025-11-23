#include "gpu_functions.h"

__global__ void multiply_matrix(float* A, float* B, float* res, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float res_val = 0;
        for (int i = 0; i < n; i++) {
            res_val += A[row*n + i] * B[i*n + col];
        }
        res[row*n + col] = res_val;
    }
}
