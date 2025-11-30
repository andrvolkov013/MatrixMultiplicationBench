# Matrix multiplication benchmark
This repository contains the source code for a simple CUDA benchmark.
It demonstrates the performance differences between CPU (Single-threaded), CPU (Multi-threaded with OpenMP), and GPU (CUDA) implementations of matrix multiplication.
## How to build and run?
### Requirements
* NVIDIA GPU with CUDA support
* NVCC compiler (CUDA toolkit)
* C++ compiler
* CMake (version 3.20 or higher)
### Build instructions
In this project CMake is used for automatic setup for your GPU.
#### 1. Clone repository
```
git clone https://github.com/andrvolkov013/MatrixMultiplicationBench.git
cd MatrixMultiplicationBench
```
#### 2. Create directory for build
```
mkdir build
cd build
```
#### 3. Configuration and compilation
For Windows:
```
cmake ..
cmake --build . --config Release
```
For Linux/MacOS:
```
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```
#### 4. Run benchmark
