#include "matrix_lib.h"
#include <omp.h>
#include <iostream>

void generate_matrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 10);
    }
}

void print_matrix(float* matrix, int n) {
    std::cout << "----------------------" << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            std::cout << matrix[idx] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "--------------------" << std::endl;
}

void multiply_matrix_on_CPU(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float C_val = 0;
            for (int k = 0; k < n; k++) {
                C_val += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = C_val;
        }
    }
}

void multiply_matrix_on_parallel_CPU(float* A, float* B, float* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float C_val = 0;
            for (int k = 0; k < n; k++) {
                C_val += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = C_val;
        }
    }
}


bool verify_multiplication(float* A, float* B, int n) {
    int num_el = n*n;
    for (int i = 0; i < num_el; i++) {
        if (A[i] - B[i] > 1e-5) {
            return false;
        }
    }
    return true;
}
