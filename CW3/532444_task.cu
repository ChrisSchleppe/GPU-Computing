#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#define CHECK_CUDA(call)                                        \
    if ((call) != cudaSuccess)                                  \
    {                                                           \
        std::cerr << "CUDA error at " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                     \
    }

const int NUM_MATRICES = 10; // Number of matrix multiplications
const int MATRIX_SIZE = 4096;
const int TILE_SIZE = 32;

// Simple kernel for matrix multiplication
__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C, int n)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Tiled kernel for matrix multiplication
__global__ void matrixMultiplyKernelTiled(const float *A, const float *B, float *C, int n)
{
    // TODO: allocate shared memory for two tiles (one for A and one for B)
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    // TODO: iterate over tiles

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    int l_col = threadIdx.x;
    int l_row = threadIdx.y;

    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        // Column index for A
        int tile_col = t * TILE_SIZE + l_col;
        
        // Row index for B
        int tile_row = t * TILE_SIZE + l_row;
        
        // TODO: copy tiles from global memory into shared memory

        // Load tile from A into shared memory
        if (row < n && tile_col < n)
            tile_A[l_row][l_col] = A[row * n + tile_col];
        else
            tile_A[l_row][l_col] = 0.0f;

        // Load tile from B into shared memory
        if (tile_row < n && col < n)
            tile_B[l_row][l_col] = B[tile_row * n + col];
        else
            tile_B[l_row][l_col] = 0.0f;

        __syncthreads();

        // TODO: compute the matrix multiplication of the two tiles
        // Outer Product
        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum += tile_A[l_row][k] *
                   tile_B[k][l_col];
        }

        __syncthreads();

        // TODO: write back the results into the matrix C
        if (row < n && col < n)
            C[row * n + col] = sum;
    }
}

void matrixMultiplyNoStreams()
{
    // Host and device pointers
    float *h_A[NUM_MATRICES], *h_B[NUM_MATRICES], *h_C[NUM_MATRICES];
    float *d_A[NUM_MATRICES], *d_B[NUM_MATRICES], *d_C[NUM_MATRICES];

    for (int i = 0; i < NUM_MATRICES; i++)
    {
        h_A[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        h_B[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
        h_C[i] = (float *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

        // Initialize example matrices with random numbers
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++)
        {
            // pick testing values, that allow us to compute the expected result on the CPU cheaply
            h_A[i][j] = 1.0f;
            h_B[i][j] = 0.01f;
            h_C[i][j] = 0.0f;
        }

        CHECK_CUDA(cudaMalloc(&d_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));

        // Copy matrices A and B to the device
        CHECK_CUDA(cudaMemcpy(d_A[i], h_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B[i], h_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        // Launch matrix multiplication kernel
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 blocksPerGrid(MATRIX_SIZE / TILE_SIZE, MATRIX_SIZE / TILE_SIZE);

        std::cout << "Launch kernel with " << blocksPerGrid.x * blocksPerGrid.y << " blocks each with " << threadsPerBlock.x * threadsPerBlock.y << " threads\n";
        // matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        matrixMultiplyKernelTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        CHECK_CUDA(cudaGetLastError());

        // Copy results back to the host
        CHECK_CUDA(cudaMemcpy(h_C[i], d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        // Verify results
        double eps = 1.e-6; // machine zero
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++)
        {
            double abs_err = fabs(h_C[i][j] - (MATRIX_SIZE * 0.01f));
            double dot_length = MATRIX_SIZE;
            double abs_val = fabs(h_C[i][j]);
            double rel_err = abs_err / abs_val / dot_length;

            if (rel_err > eps)
            {
                printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                       j, h_C[i][j], MATRIX_SIZE * 0.01f, eps);
            }
        }

        // Cleanup
        free(h_A[i]);
        free(h_B[i]);
        free(h_C[i]);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
}

void matrixMultiplyWithStreams()
{
    // Host and device pointers
    float *h_A[NUM_MATRICES], *h_B[NUM_MATRICES], *h_C[NUM_MATRICES], *check_C[NUM_MATRICES];
    float *d_A[NUM_MATRICES], *d_B[NUM_MATRICES], *d_C[NUM_MATRICES];

    cudaStream_t streams[NUM_MATRICES];

    // TODO: Allocate memory, initialize data, create streams and copy data asynchronously
    for (size_t i = 0; i < NUM_MATRICES; ++i)
    {
        CHECK_CUDA(cudaMallocHost(&h_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&h_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&h_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&check_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float)));

        // Initialize example matrices with random numbers
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE; j++)
        {
            // pick testing values, that allow us to compute the expected result on the CPU cheaply
            h_A[i][j] = 1.0f;
            h_B[i][j] = 0.01f;
            h_C[i][j] = 0.0f;
        }

        CHECK_CUDA(cudaStreamCreate(&streams[i]));

        CHECK_CUDA(cudaMallocAsync(&d_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), streams[i]));
        CHECK_CUDA(cudaMallocAsync(&d_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), streams[i]));
        CHECK_CUDA(cudaMallocAsync(&d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), streams[i]));

        // Copy matrices A and B to the device
        CHECK_CUDA(cudaMemcpyAsync(d_A[i], h_A[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_B[i], h_B[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice, streams[i]));

        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 blocksPerGrid(MATRIX_SIZE / TILE_SIZE, MATRIX_SIZE / TILE_SIZE);
        int shared_mem = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

        // TODO: Launch matrix multiplication kernel for each streams
        std::cout << "Launch tile kernel with " << blocksPerGrid.x * blocksPerGrid.y << " blocks each with " << threadsPerBlock.x * threadsPerBlock.y << " threads\n";
        matrixMultiplyKernelTiled<<<blocksPerGrid, threadsPerBlock, shared_mem, streams[i]>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        CHECK_CUDA(cudaGetLastError());

        // TODO: Copy results back to the host asynchronously
        CHECK_CUDA(cudaMemcpyAsync(h_C[i], d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
    }

    cudaDeviceSynchronize();

    // Verify results (slow! use only for debugging)
    // Calculate check_C
    for (int i = 0; i < NUM_MATRICES; i++)
    {

        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 blocksPerGrid(MATRIX_SIZE / TILE_SIZE, MATRIX_SIZE / TILE_SIZE);
        
        std::cout << "Launch kernel with " << blocksPerGrid.x * blocksPerGrid.y << " blocks each with " << threadsPerBlock.x * threadsPerBlock.y << " threads\n";
        matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A[i], d_B[i], d_C[i], MATRIX_SIZE);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemcpy(check_C[i], d_C[i], MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    }

    double eps = 1.e-6; // machine zero
    for (int i = 0; i < NUM_MATRICES; i++)
    {
        for (int j = 0; j < MATRIX_SIZE * MATRIX_SIZE;  ++j)
        {
            float err = fabs(h_C[i][j] - check_C[i][j]);
            if (err > eps)
            {
                std::cout << " is incorrect ";
                break;
            }
        }
    }

    // TODO: Cleanup
    for (size_t i = 0; i < NUM_MATRICES; ++i)
    {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));

        cudaFreeHost(h_A[i]);
        cudaFreeHost(h_B[i]);
        cudaFreeHost(h_C[i]);

        CHECK_CUDA(cudaFreeAsync(d_A[i], streams[i]));
        CHECK_CUDA(cudaFreeAsync(d_B[i], streams[i]));
        CHECK_CUDA(cudaFreeAsync(d_C[i], streams[i]));

        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
}

int main()
{
    matrixMultiplyWithStreams();
    // matrixMultiplyNoStreams();
    return EXIT_SUCCESS;
}
