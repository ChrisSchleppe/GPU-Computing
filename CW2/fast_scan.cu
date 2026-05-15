#include <chrono>
#include <curand.h>
#include <iostream>
#include <stdlib.h>

#include "helper.cu"

#define BLOCK_DIM 512

__global__ void segment_mult(float* input, float* output)
{
    // shared structure for block product 
    __shared__ float products_s[BLOCK_DIM];

    // Segment should be 4 times blocksize because it combines 2 complex numbers that occupy 4 floats in input.
    auto segment = 2 * blockDim.x * blockIdx.x;
    auto g_id = segment + threadIdx.x;
    auto l_id = threadIdx.x;

    if (l_id % 2 == 0) 
    {
        // output[i] = ac - bd 
        products_s[l_id] = input[g_id] * input[g_id + BLOCK_DIM] - input[g_id + 1] * input[g_id + BLOCK_DIM + 1];
        // output[i+1] = ad + bc
        products_s[l_id + 1] = input[g_id] * input[g_id + BLOCK_DIM + 1] + input[g_id + 1] * input[g_id + BLOCK_DIM];
    }

    // //stride doesn't make sense 
    for (auto stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if(l_id < stride && l_id % 2 == 0)
        {
            auto a = products_s[l_id];
            auto b = products_s[l_id + 1 ];
            auto c = products_s[l_id + stride];            
            auto d = products_s[l_id + stride + 1];

            products_s[l_id] = a * c - b * d;
            products_s[l_id + 1] = a * d + b * c;
        }
    }

    if (threadIdx.x == 0) 
    {
        output[blockIdx.x] = products_s[0];
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

int main()
{
    size_t size = 33554432 * 2;
    float *in_d, *out_d, *in_h, *out_h;

    // Allocate on host
    in_h = (float *)calloc(size, sizeof(float));
    CHECK_ALLOC(in_h);
    out_h = (float *)calloc(size, sizeof(float));
    CHECK_ALLOC(out_h);
    
    // Allocate on device
    CUDA_CALL(cudaMalloc((void **)&in_d, size * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&out_d, size * sizeof(float)));

    // Kernel hyperparameters
    int blocks = size / (2 * BLOCK_DIM);
    int threads_pb = BLOCK_DIM;
    
    // Initialize
    int e = random_init(size, in_d, in_h);
    if (e == EXIT_FAILURE)
        return EXIT_FAILURE;

    auto start = std::chrono::system_clock::now();
    sequential_scan(size, in_h, out_h);
    auto end = std::chrono::system_clock::now();

    // std::cout << "First 3 entries of In Vec:" << std::endl;
    // for (int32_t i = 0; i < 5 * 2; i += 2)
    //     std::cout << in_h[i] << "," << in_h[i + 1] << std::endl;
    // std::cout << "First 3 entries of Out Vec:" << std::endl;
    // for (int32_t i = 0; i < 5 * 2; i += 2)
    //     std::cout << out_h[i] << " + " << out_h[i + 1] << std::endl;
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
    

    start = std::chrono::system_clock::now();
    segment_mult<<<blocks, threads_pb>>>(in_d, out_d);
    end = std::chrono::system_clock::now();
    
    // ------------ CHECK CORRECTNESS ------------
    std::cout << "final value cpu" << std::endl;
    std::cout << "a: " << out_h[size - 2] << " b: " << out_h[size - 1] << std::endl;
    


    // ------------ CHECK CORRECTNESS ------------
    
    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));
    free(in_h);
    free(out_h);
    return EXIT_SUCCESS;
}
