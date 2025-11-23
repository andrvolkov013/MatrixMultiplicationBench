#pragma once

void generate_matrix(float* matrix, int size);

void print_matrix(float* matrix, int n);

void multiply_matrix_on_CPU(float* A, float* B, float* C, int n);

void multiply_matrix_on_parallel_CPU(float* A, float* B, float* C, int n);

bool verify_multiplication(float* A, float* B, int n);
