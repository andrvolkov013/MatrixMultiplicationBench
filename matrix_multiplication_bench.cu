#include "matrix_multiplication_bench.h"
#include <iostream>
#include <ctime>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include "matrix_lib.h"
#include "gpu_functions.h"


std::vector<std::vector<float>> matrix_multiplication_bench() {
    srand(time(0));
    std::vector<int> sizes = {8, 16, 32, 64, 128, 256, 512, 1024};
    std::vector<std::vector<float>> results;
    // Прогрев GPU
    std::cout << "Warming up GPU..." << std::endl;
    {
        float *d_dummy;
        cudaMalloc(&d_dummy, 1024);
        // Запускаем "холостое" ядро
        multiply_matrix<<<1, 1>>>(d_dummy, d_dummy, d_dummy, 1);
        cudaDeviceSynchronize(); // Ждем завершения
        cudaFree(d_dummy);
    }
    std::cout << "Warm-up complete.\n" << std::endl;

    for (int n : sizes) {
        int num_el = n*n;
        int size = num_el*sizeof(float);
        float *h_matrix_1, *h_matrix_2, *h_matrix_res, *h_matrix_res_CPU, *d_matrix_1, *d_matrix_2, *d_matrix_res;
        h_matrix_1 = new float[num_el];
        h_matrix_2 = new float[num_el];
        h_matrix_res = new float[num_el];
        h_matrix_res_CPU = new float[num_el];
        //генерируем матрицы
        generate_matrix(h_matrix_1, num_el);
        generate_matrix(h_matrix_2, num_el);
        // создаем объекты событий CUDA
        cudaEvent_t start_kernel, stop_kernel, start_h2d, stop_h2d, start_d2h, stop_d2h;
        cudaEventCreate(&start_kernel); cudaEventCreate(&start_h2d); cudaEventCreate(&start_d2h);
        cudaEventCreate(&stop_kernel); cudaEventCreate(&stop_h2d); cudaEventCreate(&stop_d2h);
        // выделяем память на GPU
        cudaMalloc(&d_matrix_1, size);
        cudaMalloc(&d_matrix_2, size);
        cudaMalloc(&d_matrix_res, size);
        cudaEventRecord(start_h2d); // замеряем время передачи данных на GPU
        cudaMemcpy(d_matrix_1, h_matrix_1, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrix_2, h_matrix_2, size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop_h2d);
        cudaEventSynchronize(stop_h2d);
        // считаем время передачи
        float time_h2d = 0;
        cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d);
        // выделяем потоки
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid;
        blocksPerGrid.x = (n + threadsPerBlock.x - 1) / threadsPerBlock.x;
        blocksPerGrid.y = (n + threadsPerBlock.y - 1) / threadsPerBlock.y;
        // работа ядра
        cudaEventRecord(start_kernel); // Записываем начальное событие в поток команд для GPU
        multiply_matrix<<<blocksPerGrid, threadsPerBlock>>>(d_matrix_1, d_matrix_2, d_matrix_res, n);
        cudaEventRecord(stop_kernel); // Записываем конечное событие в поток команд для GPU
        cudaEventSynchronize(stop_kernel); // CPU "ждет", пока закончит GPU
        float time_kernel = 0;
        cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel); // Считаем время в миллисекундах
        // копирование результата в CPU
        cudaEventRecord(start_d2h);
        cudaMemcpy(h_matrix_res, d_matrix_res, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop_d2h);
        cudaEventSynchronize(stop_d2h);
        float time_d2h = 0;
        cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);
        // Удаляем события
        cudaEventDestroy(start_kernel); cudaEventDestroy(stop_kernel);
        cudaEventDestroy(start_h2d);    cudaEventDestroy(stop_h2d);
        cudaEventDestroy(start_d2h);    cudaEventDestroy(stop_d2h);

        //Вычисление на однопоточном CPU
        auto start_time = std::chrono::high_resolution_clock::now(); // стартовая отсечка времени
        multiply_matrix_on_CPU(h_matrix_1, h_matrix_2, h_matrix_res_CPU, n);
        auto end_time = std::chrono::high_resolution_clock::now(); // Конечная отсечка времени
        auto time_CPU = end_time - start_time;
        float time_CPU_ms = std::chrono::duration<float, std::milli>(time_CPU).count(); // перевод в миллисекунды
        //Вычисление на параллельном CPU
        auto start_time_p = std::chrono::high_resolution_clock::now(); // стартовая отсечка времени
        multiply_matrix_on_parallel_CPU(h_matrix_1, h_matrix_2, h_matrix_res_CPU, n);
        auto end_time_p = std::chrono::high_resolution_clock::now(); // Конечная отсечка времени
        auto time_parallel_CPU = end_time_p - start_time_p;
        float time_parallel_CPU_ms = std::chrono::duration<float, std::milli>(time_parallel_CPU).count(); // перевод в миллисекунды
        // проверим результат на корректность
        bool verification = verify_multiplication(h_matrix_res, h_matrix_res_CPU, n); // проверка вычислений
        std::cout << "Multiplication " << n << "x" << n << " completed" << std::endl;
        if (!verification) {
            std::cout << "Multiplication was not correct" << std::endl;
        }
        // собираем результат
        std::vector<float> res = {
            static_cast<float>(n),
            time_kernel,
            time_h2d,
            time_d2h,
            (time_kernel + time_h2d + time_d2h),
            time_CPU_ms,
            time_parallel_CPU_ms,
        };
        results.push_back(res);
        cudaFree(d_matrix_1);
        cudaFree(d_matrix_2);
        cudaFree(d_matrix_res);
        delete[] h_matrix_1;
        delete[] h_matrix_2;
        delete[] h_matrix_res;
    }
    return results;
}