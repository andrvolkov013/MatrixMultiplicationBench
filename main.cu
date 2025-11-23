#include "matrix_multiplication_bench.h"
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>

void printBenchmarkResults(const std::vector<std::vector<float>>& results);


int main() {
    std::vector<std::vector<float>> matrix_perm_res = matrix_multiplication_bench();
    printBenchmarkResults(matrix_perm_res);
}

void printBenchmarkResults(const std::vector<std::vector<float>>& results) {
    // настройки ширины колонок
    const int w_n = 8;
    const int w_time = 12;
    const int w_speedup = 10;

    // вспомогательная функция для рисования горизонтальной линии
    auto print_line = [&](char corner = '+', char horizontal = '-') {
        std::cout << corner << std::string(w_n + 2, horizontal)        << corner
                  << std::string(w_time + 2, horizontal) << corner     // H2D
                  << std::string(w_time + 2, horizontal) << corner     // Kernel
                  << std::string(w_time + 2, horizontal) << corner     // D2H
                  << std::string(w_time + 2, horizontal) << corner     // GPU Tot
                  << std::string(w_time + 2, horizontal) << corner     // CPU
                  << std::string(w_time + 2, horizontal) << corner     // Par. CPU
                  << std::string(w_speedup + 2, horizontal) << corner  // Speedup
                  << std::endl;
    };

    std::cout << "\n=== Heterogeneous Benchmark Results (Time in ms) ===\n" << std::endl;

    print_line();
    std::cout << "| " << std::setw(w_n) << "N"
              << " | " << std::setw(w_time) << "H2D"
              << " | " << std::setw(w_time) << "Kernel"
              << " | " << std::setw(w_time) << "D2H"
              << " | " << std::setw(w_time) << "GPU Tot"
              << " | " << std::setw(w_time) << "CPU"
              << " | " << std::setw(w_time) << "CPU (Par)"
              << " | " << std::setw(w_speedup) << "Speedup"
              << " |" << std::endl;
    print_line();

    for (const auto& row : results) {
        int n = static_cast<int>(row[0]);
        float t_kernel = row[1];
        float t_h2d = row[2];
        float t_d2h = row[3];
        float t_gpu = row[4];
        float t_cpu = row[5];
        float t_p_cpu = row[6];

        // формируем строку ускорения
        std::string speedup_str;
        if (t_cpu > 0 && t_gpu > 0) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << (t_p_cpu / t_gpu) << "x";
            speedup_str = ss.str();
        } else if (t_cpu == 0) {
            speedup_str = "-";
        } else {
            speedup_str = "Err";
        }

        std::cout << "| " << std::setw(w_n) << n
                  << std::fixed << std::setprecision(3) // применяем формат ко всем float
                  << " | " << std::setw(w_time) << t_h2d
                  << " | " << std::setw(w_time) << t_kernel
                  << " | " << std::setw(w_time) << t_d2h
                  << " | " << std::setw(w_time) << t_gpu
                  << " | " << std::setw(w_time) << t_cpu
                  << " | " << std::setw(w_time) << t_p_cpu
                  << " | " << std::setw(w_speedup) << speedup_str
                  << " |" << std::endl;
    }
    print_line();
}