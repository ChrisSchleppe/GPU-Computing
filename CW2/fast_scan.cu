#include <chrono>
#include <curand.h>
#include <iostream>
#include <stdlib.h>

#include "helper.cu"

#define BLOCK_DIM 512

__global__ void scan_kernel(float* input, float* output)
{
    __shared__ float input_s[BLOCK_DIM];
    // shared structure for block product 
    __shared__ float output_s[BLOCK_DIM / 2];

    // Segment should be 4 times blocksize because it combines 2 complex numbers that occupy 4 floats in input.
    auto segment = 4 * blockDim.x * blockIdx.x;
    auto g_id = segment + threadIdx.x;
    auto l_id = threadIdx.x;

    for (auto stride = blockDim.x / 2; stride >= 1; stride /= 4)
    {

        if(threadIdx.x < stride)
        {
            // output[i] = ac - bd 
            output_s[g_id] = input_s[g_id] * input[g_id + stride] - input[g_id + 1] * input[g_id + stride + 1];
            // outputp[i+1] = ad + bd            
            output_s[g_id + 1] = input_s[g_id] * input[g_id + stride + 1] + input[g_id + 1] * input[g_id + stride + 1];
        }
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

    // Initialize
    int e = random_init(size, in_d, in_h);
    if (e == EXIT_FAILURE)
        return EXIT_FAILURE;

    auto start = std::chrono::system_clock::now();
    sequential_scan(size, in_h, out_h);
    auto end = std::chrono::system_clock::now();

    std::cout << "First 3 entries of In Vec:" << std::endl;
    for (int32_t i = 0; i < 5 * 2; i += 2)
        std::cout << in_h[i] << "," << in_h[i + 1] << std::endl;
    std::cout << "First 3 entries of Out Vec:" << std::endl;
    for (int32_t i = 0; i < 5 * 2; i += 2)
        std::cout << out_h[i] << " + " << out_h[i + 1] << std::endl;

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    int blocks = size / (2 * BLOCK_DIM);
    int threads_pb = BLOCK_DIM;

    scan_kernel<<<blocks, threads_pb>>>(in_d, out_d);

    CUDA_CALL(cudaFree(in_d));
    CUDA_CALL(cudaFree(out_d));
    free(in_h);
    free(out_h);
    return EXIT_SUCCESS;
}
