#include <cuda_runtime.h>
#include <format>
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#include <chrono>

void check(cudaError_t err, const std::string &msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "\n========== CUDA ERROR ==========\n";
        std::cerr << "Message     : " << msg << '\n';
        std::cerr << "Error Code  : " << static_cast<int>(err) << '\n';
        std::cerr << "CUDA Error  : " << cudaGetErrorString(err) << '\n';
        std::cerr << "File        : " << __FILE__ << '\n';
        std::cerr << "Line        : " << __LINE__ << '\n';

        // Check for async kernel errors too
        cudaError_t asyncErr = cudaDeviceSynchronize();
        if (asyncErr != cudaSuccess)
        {
            std::cerr << "Async Error : "
                      << cudaGetErrorString(asyncErr) << '\n';
        }

        std::cerr << "================================\n";

        std::exit(EXIT_FAILURE);
    }
}

void init(int32_t numElements, int32_t *vec_a, int32_t *vec_b, int32_t *mat)
{
    // std::random_device dev;
    std::mt19937 prng(2024);
    std::uniform_int_distribution<int32_t> distrib(1, 4);

    for (auto i = 0; i < numElements; i++)
    {
        vec_a[i] = distrib(prng);
        vec_b[i] = distrib(prng);
    }

    for (auto i = 0; i < numElements * numElements; i++)
        mat[i] = distrib(prng);
}

void pretty_print(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat)
{
    std::cout << "Vec A:" << std::endl;
    for (auto i = 0; i < size; i++)
        std::cout << vec_a[i] << std::endl;

    std::cout << "Vec B:" << std::endl;
    for (auto i = 0; i < size; i++)
        std::cout << vec_b[i] << std::endl;

    std::cout << "Matrix:" << std::endl;
    for (auto i = 0; i < size; i++)
    {
        for (auto j = 0; j < size; j++)
            std::cout << mat[i * size + j] << " ";

        std::cout << std::endl;
    }
}

__global__ void compute_kernel(int32_t numElements, int32_t *vec_a, int32_t *vec_b, int32_t *mat, int32_t *out)
{
    int global_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (global_index >= numElements)
    {
        return;
    }

    int32_t sum = 0;
    for (auto j = 0; j < numElements; j++)
    {
        sum += (vec_a[j] + vec_b[j]) * mat[global_index * numElements + j];
    }
    out[global_index] = sum;
}

void compute(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat, int32_t *out)
{
    // BLOCK vec add a + b
    auto tmp = (int32_t *)malloc(sizeof(int32_t) * size);
    for (auto i = 0; i < size; i++)
        tmp[i] = vec_a[i] + vec_b[i];
    // BLOCK

    // BLOCK (a + b)^T @ M
    for (auto i = 0; i < size; i++)
    {
        out[i] = 0;
        for (auto j = 0; j < size; j++)
            out[i] += tmp[j] * mat[i * size + j];
    }
    // BLOCK
    free(tmp);
}

void allocateDeviceMemory(int32_t **d_a, int32_t *h_a, size_t size, cudaError_t &err)
{
    err = cudaMalloc((void **)d_a, size);
    check(err, "Failed to allocate device vector");

    err = cudaMemcpy(*d_a, h_a, size, cudaMemcpyHostToDevice);
    check(err, "Failed to copy vector from host to device");
}

int main()
{
    cudaError_t err = cudaSuccess;

    int32_t numElements = 32768;
    // int32_t numElements = 3;

    auto h_a = (int32_t *)malloc(sizeof(int32_t) * numElements);
    auto h_b = (int32_t *)malloc(sizeof(int32_t) * numElements);
    // Flat Buffer for matrix
    auto h_mat = (int32_t *)malloc(sizeof(int32_t) * numElements * numElements);
    auto h_out = (int32_t *)malloc(sizeof(int32_t) * numElements);

    init(numElements, h_a, h_b, h_mat);

    size_t vec_size = numElements * sizeof(int32_t);
    size_t mat_size = numElements * numElements * sizeof(int32_t);

    // pretty_print(numElements, h_a, h_b, h_mat);

    int32_t *d_a = NULL;
    int32_t *d_b = NULL;
    int32_t *d_mat = NULL;
    int32_t *d_out = NULL;

    allocateDeviceMemory(&d_a, h_a, vec_size, err);
    allocateDeviceMemory(&d_b, h_b, vec_size, err);
    allocateDeviceMemory(&d_mat, h_mat, mat_size, err);
    err = cudaMalloc((void **)&d_out, vec_size);
    check(err, "Failed to allocate device vector out");

    //-------------------- GPU 512 blocksize ----------------------

    int threadsPerBlock = 512;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    compute_kernel<<<blocksPerGrid, threadsPerBlock>>>(numElements, d_a, d_b, d_mat, d_out);
    check(cudaGetLastError(), "Failed to launch compute kernel");
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "-------------------- GPU 512 blocksize ---------------------- " << std::endl;
    std::cout << "Elapsed time: " << milliseconds << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // -------------------- GPU 512 blocksize ----------------------

    err = cudaMemcpy(h_out, d_out, vec_size, cudaMemcpyDeviceToHost);
    check(err, "Failed to copy vector C from device to host");

    // -------------------- GPU 256 blocksize ----------------------

    threadsPerBlock = 256;
    blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    compute_kernel<<<blocksPerGrid, threadsPerBlock>>>(numElements, d_a, d_b, d_mat, d_out);
    check(cudaGetLastError(), "Failed to launch compute kernel");
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "-------------------- GPU 256 block size ---------------------- " << std::endl;
    std::cout << "Elapsed time: " << milliseconds << " ms\n";

    err = cudaMemcpy(h_out, d_out, vec_size, cudaMemcpyDeviceToHost);
    check(err, "Failed to copy vector C from device to host");

    // -------------------- GPU 256 blocksize ----------------------

    // -------------------- GPU 128 blocksize ----------------------

    threadsPerBlock = 128;
    blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    compute_kernel<<<blocksPerGrid, threadsPerBlock>>>(numElements, d_a, d_b, d_mat, d_out);
    check(cudaGetLastError(), "Failed to launch compute kernel");
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "-------------------- GPU 128 blocksize ---------------------- " << std::endl;
    std::cout << "Elapsed time: " << milliseconds << " ms\n";

    err = cudaMemcpy(h_out, d_out, vec_size, cudaMemcpyDeviceToHost);
    check(err, "Failed to copy vector C from device to host");

    // -------------------- GPU 128 blocksize ----------------------

    // -------------------- GPU 32 blocksize ----------------------

    threadsPerBlock = 32;
    blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    compute_kernel<<<blocksPerGrid, threadsPerBlock>>>(numElements, d_a, d_b, d_mat, d_out);
    check(cudaGetLastError(), "Failed to launch compute kernel");
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "-------------------- GPU 32 blocksize ---------------------- " << std::endl;
    std::cout << "Elapsed time: " << milliseconds << " ms\n";

    err = cudaMemcpy(h_out, d_out, vec_size, cudaMemcpyDeviceToHost);
    check(err, "Failed to copy vector C from device to host");

    // -------------------- GPU 32 blocksize ----------------------

    // -------------------- GPU 16 blocksize ----------------------

    threadsPerBlock = 16;
    blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    compute_kernel<<<blocksPerGrid, threadsPerBlock>>>(numElements, d_a, d_b, d_mat, d_out);
    check(cudaGetLastError(), "Failed to launch compute kernel");
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "-------------------- GPU 16 blocksize ---------------------- " << std::endl;
    std::cout << "Elapsed time: " << milliseconds << " ms\n";

    err = cudaMemcpy(h_out, d_out, vec_size, cudaMemcpyDeviceToHost);
    check(err, "Failed to copy vector C from device to host");

    // -------------------- GPU 16 blocksize ----------------------

    // -------------------- GPU 8 blocksize ----------------------

    threadsPerBlock = 8;
    blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    compute_kernel<<<blocksPerGrid, threadsPerBlock>>>(numElements, d_a, d_b, d_mat, d_out);
    check(cudaGetLastError(), "Failed to launch compute kernel");
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "-------------------- GPU 8 blocksize ---------------------- " << std::endl;
    std::cout << "Elapsed time: " << milliseconds << " ms\n";

    err = cudaMemcpy(h_out, d_out, vec_size, cudaMemcpyDeviceToHost);
    check(err, "Failed to copy vector C from device to host");

    // -------------------- GPU 8 blocksize ----------------------

    // ------------------- CPU -----------------------
    auto cpustart = std::chrono::system_clock::now();
    int32_t *cpu_out = (int32_t *)malloc(sizeof(int32_t) * numElements);
    compute(numElements, h_a, h_b, h_mat, cpu_out);
    auto cpuend = std::chrono::system_clock::now();

    auto elapsed_seconds = cpuend - cpustart;
    std::cout << "-------------------- CPU ---------------------- " << std::endl;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    // ------------------- CPU -----------------------

    for (auto i = 0; i < numElements; i++)
    {
        if (h_out[i] - cpu_out[i] != 0)
            std::cout << "false calculation" << std::endl;
    }
    std::cout << "correct calculation" << std::endl;

    err = cudaFree(d_a);
    check(err, "Failed to free device vector A");
    err = cudaFree(d_b);
    check(err, "Failed to free device vector B");
    err = cudaFree(d_mat);
    check(err, "Failed to free device mat");
    err = cudaFree(d_out);
    check(err, "Failed to free device vector out");

    free(h_a);
    free(h_b);
    free(h_mat);
    free(h_out);
}