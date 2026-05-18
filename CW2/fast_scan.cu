#include <chrono>
#include <curand.h>
#include <iostream>
#include <stdlib.h>

#include "helper.cu"

#define BLOCK_DIM 1024
#define COARSE_FACTOR 32

__global__ void block_scan(float *input, float *output)
{
    __shared__ float2 products_s[BLOCK_DIM];

    auto segment = 2 * COARSE_FACTOR * blockDim.x * blockIdx.x;
    //auto segment = 2 * 2 * blockDim.x * blockIdx.x;
    
    auto g_id = segment + threadIdx.x * 2;
    auto l_id = threadIdx.x;
    //This serves the purpose to index every second element. 

    // ------------ THREAD COARSENING ------------
    float a = input[g_id];
    float b = input[g_id + 1];

    for (auto tile = 1; tile < COARSE_FACTOR; ++tile)
    {
        auto a_temp = a;
        auto b_temp = b;
        auto c = input[g_id + tile * 2 * BLOCK_DIM];
        auto d = input[g_id + 1 + tile * 2 * BLOCK_DIM];
        a = a_temp * c - b_temp * d;
        b = a_temp * d + b_temp * c;
    }
    products_s[l_id].x = a;
    products_s[l_id].y = b;

    // ------------ THREAD COARSENING ------------

    for (auto stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (l_id < stride)
        {                                                                                
            auto a = products_s[l_id].x;
            auto b = products_s[l_id].y;
            auto c = products_s[l_id + stride].x;
            auto d = products_s[l_id + stride].y;

            products_s[l_id].x = a * c - b * d;
            products_s[l_id].y = a * d + b * c;
        }
    }

    // save output to global memory
    if (l_id == 0)
    {
        output[blockIdx.x * 2] = products_s[0].x;
        output[blockIdx.x * 2 + 1] = products_s[0].y;
    }
}

void sequential_scan(size_t size, float *in_h, float *out_h)
{
    out_h[0] = in_h[0];
    out_h[1] = in_h[1];
    for (auto i = 2; i < size; i += 2)
    {
        float real_prev = out_h[i - 2];
        float im_prev = out_h[i - 1];
        float real_cur = in_h[i];
        float im_cur = in_h[i + 1];

        out_h[i] = real_prev * real_cur - im_prev * im_cur;
        out_h[i + 1] = real_prev * im_cur + real_cur * im_prev;
    }
}

std::pair<float, float> sequential_scan_last_value(size_t size, float *input)
{
    float *out_h = (float *)calloc(size, sizeof(float));

    out_h[0] = input[0];
    out_h[1] = input[1];
    for (auto i = 2; i < size; i += 2)
    {
        float real_prev = out_h[i - 2];
        float im_prev = out_h[i - 1];
        float real_cur = input[i];
        float im_cur = input[i + 1];

        out_h[i] = real_prev * real_cur - im_prev * im_cur;
        out_h[i + 1] = real_prev * im_cur + real_cur * im_prev;
    }

    return {out_h[size - 2], out_h[size - 1]};
}

int main()
{
    size_t size = 33554432 * 2;
    float *in_d, *in_h, *out_h;

    // Kernel hyperparameters
    int blocks = size / (2 * BLOCK_DIM * COARSE_FACTOR);
    int threads_pb = BLOCK_DIM;

    // Allocate on host
    in_h = (float *)calloc(size, sizeof(float));
    CHECK_ALLOC(in_h);
    out_h = (float *)calloc(size, sizeof(float));
    CHECK_ALLOC(out_h);

    // Allocate on device
    CUDA_CALL(cudaMalloc((void **)&in_d, size * sizeof(float)));

    // Allocate 2 floats per block
    float *block_results_d, *block_results_h;

    CUDA_CALL(cudaMalloc((void **)&block_results_d, blocks * 2 * sizeof(float)));
    block_results_h = (float *)calloc(blocks * 2, sizeof(float));

    // Initialize
    // int e = random_init(size, in_d, in_h);
    int e = init_unit_circle(size, in_d, in_h);
    if (e == EXIT_FAILURE)
        return EXIT_FAILURE;

    auto start = std::chrono::system_clock::now();
    sequential_scan(size, in_h, out_h);
    auto end = std::chrono::system_clock::now();

    std::cout << "------------ CPU ------------" << std::endl;
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
    std::cout << "------------ CPU ------------" << std::endl;

    start = std::chrono::system_clock::now();
    block_scan<<<blocks, threads_pb>>>(in_d, block_results_d);
    cudaMemcpy(block_results_h, block_results_d, blocks * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    // Possible rerun of block_scan with in_d = block_results_d and block_results_d = new_block_results_d.
    // And blocks /= BLOCK_DIM
    std::pair<float, float> result = sequential_scan_last_value(blocks * 2, block_results_h);
    end = std::chrono::system_clock::now();

    std::cout << "------------ GPU ------------" << std::endl;
    elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
    std::cout << "------------ GPU ------------" << std::endl;

    // ------------ CHECK CORRECTNESS ------------
    std::cout << "final value cpu" << std::endl;
    std::cout << "a: " << out_h[size - 2] << " b: " << out_h[size - 1] << std::endl;

    std::cout << "final value gpu" << std::endl;
    std::cout << "a: " << result.first << " b: " << result.second << std::endl;

    // for (int32_t i = 0; i < 10 * 2; i += 2)
    //     std::cout << block_results_h[i] << " + " << block_results_h[i + 1] << std::endl;

    // ------------ CHECK CORRECTNESS ------------

    CUDA_CALL(cudaFree(in_d));
    free(in_h);
    free(out_h);
    return EXIT_SUCCESS;
}
