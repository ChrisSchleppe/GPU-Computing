#include <random>
#include <iostream>
#include <mma.h>
#include <utility>
#include <chrono>

using namespace nvcuda;

#define NUM_KERNELS 16 // How many kernels we have to convolve against the input
#define KERNEL_SIZE 16 // The length of each kernel vector, and equivalently the size of each window
#define INPUT_SIZE (32000-1) // The length of the input vector.
#define OUTPUT_SIZE (INPUT_SIZE-KERNEL_SIZE+1) // The length of the output vector, or equivalently the number of windows.
#define CHUNK_SIZE 16 // we group CHUNK_SIZE-many consecutive windows into a chunk.
#define NUM_CHUNKS (OUTPUT_SIZE / CHUNK_SIZE) // How many chunks we have to compute.

// Write A[idx(i, j, N, K)] to index A_i,j into an NxK matrix.
#define idx(i, j, N, K) (i*K + j)

// Simplifying assumption: the produced output size is divisible by our chunk size.
static_assert(OUTPUT_SIZE%CHUNK_SIZE == 0);

#define CUDA_CALL(x)                                                                                          \
    do                                                                                                        \
    {                                                                                                         \
        cudaError_t error = x;                                                                                \
        if (error != cudaSuccess)                                                                             \
        {                                                                                                     \
            const char *cuda_err_str = cudaGetErrorString(error);                                             \
            std::cerr << "Cuda Error at" << __FILE__ << ":" << __LINE__ << ": " << cuda_err_str << std::endl; \
            return EXIT_FAILURE;                                                                              \
        }                                                                                                     \
    } while (0)

// Produces a random vector, and copies it over to the GPU.
std::pair<half*, half*> random_vec(size_t n) {
    static std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    half* cpu_out = (half*) malloc(sizeof(half)*n);
    for (int i = 0; i < n; i++) {
        cpu_out[i] = (half) dist(rng);
    }

    half* gpu_out;
    cudaError_t error;
    error = cudaMalloc(&gpu_out, sizeof(half)*n);
    if (error != cudaSuccess) {
        std::cerr << "Failed to cudaMalloc in random_vec" << cudaGetErrorString(error) << std::endl;
        return {nullptr, nullptr};
    }
    error = cudaMemcpy(gpu_out, cpu_out, sizeof(half)*n, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to cudaMemcpy in random_vec" << cudaGetErrorString(error) << std::endl;
        return {nullptr, nullptr};
    }
    return {cpu_out, gpu_out};
}

bool almost_equal(float a, float b, float eps = 1e-2f) {
    float diff = fabsf(a - b);
    return diff <= eps * fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
}

///////////////////////////////////////////////////////////////
// CPU 1 - A naive convolution
///////////////////////////////////////////////////////////////
void convolve_cpu(half* out, half* input, half* kernels) {
    for (int p = 0; p < OUTPUT_SIZE; p++) {
        for (int k = 0; k < NUM_KERNELS; k++) {
            half acc = 0.0;
            for (int o = 0; o < KERNEL_SIZE; o++) {
                acc += kernels[idx(o, k, KERNEL_SIZE, NUM_KERNELS)] * input[p + o];
            }
            out[idx(p, k, OUTPUT_SIZE, NUM_KERNELS)] = acc;
        }
    }
}

///////////////////////////////////////////////////////////////
// CPU 2 - A convolution using matrix multiplication
///////////////////////////////////////////////////////////////

void matmul(half *a, half *b, half *c) {
    // a :: N x K
    // b :: K x M
    // c :: N x M

    // We fix N, M, K, as tensor cores work with fixed size matrices.
    const int N = 16;
    const int M = 16;
    const int K = 16;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            half acc = 0.0;
            for (int k = 0; k < K; k++) {
                acc += a[idx(i, k, N, K)] * b[idx(k, j, K, M)];
            }
            c[idx(i, j, N, M)] = acc;
        }
    }
}

// already closer to a GPU implementation.
void convolve_cpu2(half* out, half* input, half* kernels) {
    for (int s = 0; s < NUM_CHUNKS; s++) {
        half a[CHUNK_SIZE*KERNEL_SIZE];
        half b[KERNEL_SIZE*NUM_KERNELS];
        half c[CHUNK_SIZE*NUM_KERNELS];

        // init A:
        for (int w=0; w<CHUNK_SIZE; w++) {
            auto p = s*CHUNK_SIZE + w;
            for (int o=0; o<KERNEL_SIZE; o++) {
                a[idx(w, o, CHUNK_SIZE, KERNEL_SIZE)] = input[p + o];
            }
        }

        // init B:
        for (int k=0; k<NUM_KERNELS; k++) {
            for (int o=0; o<KERNEL_SIZE; o++) {
                b[idx(o, k, KERNEL_SIZE, NUM_KERNELS)] = kernels[idx(o, k, KERNEL_SIZE, NUM_KERNELS)];
            }
        }

        matmul(a, b, c);

        for (int k=0; k<NUM_KERNELS; k++) {
            for (int w=0; w<CHUNK_SIZE; w++) {
                auto p = s*CHUNK_SIZE + w;
                out[idx(p, k, OUTPUT_SIZE, NUM_KERNELS)] = c[idx(w, k, CHUNK_SIZE, NUM_KERNELS)];
            }
        }
    }
}

/////////////////////
// GPU
/////////////////////
// So input is INPUT_SIZE big and kernels is KERNEL_SIZE * NUM_KERNELS.
__global__ void convolve_gpu(half* out, half* input, half* kernels) {
    // steps:
    // - load input: the CHUNK_SIZE-many windows, each of size KERNEL_SIZE (A : CHUNK_SIZE x KERNEL_SIZE)
    __shared__ half A[CHUNK_SIZE*KERNEL_SIZE]; 

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    for (int i = tid; i < CHUNK_SIZE * KERNEL_SIZE; i+= blockDim.x) {
        int w = i / KERNEL_SIZE;
        int o = i % KERNEL_SIZE;
        int p = bid * CHUNK_SIZE + w;

        A[idx(w, o, CHUNK_SIZE, KERNEL_SIZE)] = input[p + o];
    }


    // - load kernels: the NUM_KERNELS-many kernels, each of size KERNEL_SIZE (B : KERNEL_SIZE x NUM_KERNELS)
    __shared__ half B[KERNEL_SIZE*NUM_KERNELS]; 
    
    for (int i = tid; i < KERNEL_SIZE * NUM_KERNELS; i+= blockDim.x) {
        int k = i / KERNEL_SIZE;
        int o = i % KERNEL_SIZE;

        B[idx(k, o, KERNEL_SIZE, NUM_KERNELS)] = kernels[idx(k, o, KERNEL_SIZE, NUM_KERNELS)];
    }

    __syncthreads();

    wmma::fragment<wmma::matrix_a, CHUNK_SIZE, NUM_KERNELS, KERNEL_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, CHUNK_SIZE, NUM_KERNELS, KERNEL_SIZE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, CHUNK_SIZE, NUM_KERNELS, KERNEL_SIZE, half> c_frag;

    wmma::fill_fragment(c_frag, __float2half(0.0f));

    wmma::load_matrix_sync(a_frag, A, KERNEL_SIZE);
    wmma::load_matrix_sync(b_frag, B, NUM_KERNELS);

    // - compute C := matmul(A, B): (C : CHUNK_SIZE x NUM_KERNELS)
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // - write C back to out.
    wmma::store_matrix_sync(out + idx(bid * CHUNK_SIZE, 0, OUTPUT_SIZE, NUM_KERNELS), c_frag, NUM_KERNELS, wmma::mem_row_major);
}

int main() {
    ///////////////////
    // INIT
    ///////////////////

    // input :: INPUT_SIZE
    auto [cpu_input, gpu_input] = random_vec(INPUT_SIZE);


    // kernels :: KERNEL_SIZE x NUM_KERNELS
    auto [cpu_kernels, gpu_kernels] = random_vec(KERNEL_SIZE*NUM_KERNELS);

    // out :: OUTPUT_SIZE x NUM_KERNELS
    auto [cpu_out, gpu_out] = random_vec(OUTPUT_SIZE*NUM_KERNELS);
    half *cpu_out2 = (half*) malloc(sizeof(half)*OUTPUT_SIZE*OUTPUT_SIZE);

    ///////////////////
    // CALL CPU
    ///////////////////
    auto start = std::chrono::high_resolution_clock::now();
    convolve_cpu(cpu_out, cpu_input, cpu_kernels);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "-------------------- CPU 1 ---------------------- " << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << "ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    convolve_cpu2(cpu_out2, cpu_input, cpu_kernels);
    end = std::chrono::high_resolution_clock::now();

    std::cout << "-------------------- CPU 2 ---------------------- " << std::endl;
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count() << "ms" << std::endl;

    ///////////////////
    // CALL GPU
    ///////////////////

    int threads_per_block = 32;
    int blocks_per_grid = NUM_CHUNKS;


    cudaEvent_t cstart, cstop;
    cudaEventCreate(&cstart);
    cudaEventCreate(&cstop);

    cudaEventRecord(cstart);
    // TODO: choose correct kernel launch parameters, and implement convolve_gpu.
    convolve_gpu<<<blocks_per_grid, threads_per_block>>>(gpu_out, gpu_input, gpu_kernels);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    cudaEventRecord(cstop);

    cudaEventSynchronize(cstop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, cstart, cstop);

    std::cout << "-------------------- GPU ---------------------- " << std::endl;
    std::cout << "Elapsed time: " << milliseconds << " ms\n";

    ///////////////////
    // COMPARE
    ///////////////////
    half* cpu_out_from_gpu = (half*) malloc(sizeof(half)*OUTPUT_SIZE*NUM_KERNELS);
    CUDA_CALL(cudaMemcpy(cpu_out_from_gpu, gpu_out, sizeof(half)*OUTPUT_SIZE*NUM_KERNELS, cudaMemcpyDeviceToHost));

    for (int k = 0; k < NUM_KERNELS; k++) {
        for (int p = 0; p < OUTPUT_SIZE; p++) {
            float cpu1 = __half2float(cpu_out[idx(p, k, OUTPUT_SIZE, NUM_KERNELS)]);
            float cpu2 = __half2float(cpu_out2[idx(p, k, OUTPUT_SIZE, NUM_KERNELS)]);
            float gpu = __half2float(cpu_out_from_gpu[idx(p, k, OUTPUT_SIZE, NUM_KERNELS)]);

            if (!almost_equal(cpu1, cpu2) || !almost_equal(cpu1, gpu)) {
                std::cout << "INCORRECT: " << cpu1 << " ~ " << cpu2 << " ~ " << gpu << std::endl;
            }
        }
    }

    free(cpu_input);
    free(cpu_kernels);
    free(cpu_out);
    free(cpu_out2);
    free(cpu_out_from_gpu);
    CUDA_CALL(cudaFree(gpu_kernels));
    CUDA_CALL(cudaFree(gpu_input));
    CUDA_CALL(cudaFree(gpu_out));
}
