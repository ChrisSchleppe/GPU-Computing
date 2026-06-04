#include <random>
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#pragma region stuff
#define CUDA_CHECK(err) _cuda_check(err, __FILE__, __LINE__)
void _cuda_check(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr
        << "\n\nASSERTION FAILED(" << static_cast<int>(err) << ") in "
        << file << ":" << line << ": " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}
#define OOM_CHECK(ptr) _oom_check(ptr, __FILE__, __LINE__)
void *_oom_check(void *ptr, const char *file, int line) {
    if (ptr == NULL) {
        std::cerr
        << "\n\nOOM in "
        << file << ":" << line << "\n";
        exit(EXIT_FAILURE);
    }
    return ptr;
}
#pragma endregion


#define INCLUDE_SLOW


// Fill vec_a(s), vec_b(s) and mat(s,s) with random data
void init(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat)
{
    // std::random_device dev;
    std::mt19937 prng(2024); // Code from 2024?
    std::uniform_int_distribution<int32_t> distrib(-16, 16);

    for (auto i = 0; i < size; i++)
    {
        vec_a[i] = distrib(prng);
        vec_b[i] = distrib(prng);
    }

    for (auto i = 0; i < size * size; i++)
        mat[i] = distrib(prng);
}

void compute(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat, int32_t *out)
{
    auto tmp = (int32_t *)OOM_CHECK(malloc(sizeof(int32_t) * size)); // alloc t : vec<size>

    // t = a + b
    for (auto i = 0; i < size; i++)
        tmp[i] = vec_a[i] + vec_b[i];

    // o : vec<size>
    // o = t * m
    for (auto i = 0; i < size; i++)
    {
        out[i] = 0;

        for (auto j = 0; j < size; j++)
            out[i] += tmp[j] * mat[i * size + j];
    }

    free(tmp);
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

// Single kernel version (but recomputes a+b each time, since we cannot sync threads between blocks and wait for t to finish computing)
__global__ void kernel(const int32_t *a, const int32_t *b, const int32_t *m, int32_t *out, int len) {
    int th_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (th_idx >= len) {
        return;
    }
    
    int32_t sum = 0;
    for (int j = 0; j < len; j++) {
        sum += (a[j] + b[j]) * m[th_idx * len + j];
    }
    out[th_idx] = sum;
}        



__global__ void kernel_vecAdd(const int32_t *a, const int32_t *b, int32_t *out, int len) {
    // To what does int translate? Is it always assumed int32_t or can it also map to others?
    // What length is blockIdx, is it constant between devices?
    int th_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (th_idx < len) {
        out[th_idx] = a[th_idx] + b[th_idx];
    }
}

__global__ void kernel_vecMatMul(const int32_t *a, const int32_t *m, int32_t *out, int len) {
    int th_idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (th_idx >= len) {
        return;
    }

    // Funnily enough this seems as fast or faster then using cudaMemset beforehand
    out[th_idx] = 0;

    // Can we parallelize this?
    // How to prevent race conditions when writing to out?
    // -> Chunk into smaller pieces and write into another accumulator array[len(chunks)]
    // Should be `n/chunks+chunks` instead of `n` steps?
    // Optimal with chunks=sqrt(n)?
    // This would probably also decrease memory stalling, since each thread only needs to load `n/chunk`
    // pieces of data instead of all `n`.
    // But that wouldn't be equal to the original Cpp program anymore
    for (int j = 0; j < len; j++) {
        out[th_idx] += a[j] * m[th_idx * len + j];
    }
}



void dev_compute(int threadsPerBlock, size_t len, int32_t *a, int32_t *b, int32_t *m, int32_t *o) {
    size_t vecSize = len * sizeof(int32_t);
    size_t matSize = len * len * sizeof(int32_t);

    // Allocate memory on device
    int32_t *dev_A, *dev_B, *dev_T, *dev_O;
    int32_t *dev_M;

    CUDA_CHECK(cudaMalloc((void **)&dev_A, vecSize));
    CUDA_CHECK(cudaMalloc((void **)&dev_B, vecSize));
    CUDA_CHECK(cudaMalloc((void **)&dev_T, vecSize));
    CUDA_CHECK(cudaMalloc((void **)&dev_O, vecSize));

    CUDA_CHECK(cudaMalloc((void **)&dev_M, matSize));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(dev_A, a, vecSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_B, b, vecSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_M, m, matSize, cudaMemcpyHostToDevice));

    
    // Run kernel(s)

    // According to the AI overlords a 'grid' is all the threads launched by a kernel invocation
    // !! We must schedule enough total threads or the data will not be fully processed.
    // -> block size depends on thread size. (If threadsPerBlock decreases, blocksPerGrid must increase.)
    int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);


    // Even though we recompute the a+b, performance wise it doesn't seem to make a measurable difference,
    // going from 1.6797s (single kernel) to 1.68738s (double kernel) at 32 threads per block
    // kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_M, dev_O, len);
    // CUDA_CHECK(cudaGetLastError());


    // Can we just run them after each other, or do we need some sync?
    kernel_vecAdd<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_T, len);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize1();
    CUDA_CHECK(cudaGetLastError());
    kernel_vecMatMul<<<blocksPerGrid, threadsPerBlock>>>(dev_T, dev_M, dev_O, len);
    CUDA_CHECK(cudaGetLastError());


    // Load results from device
    CUDA_CHECK(cudaMemcpy(o, dev_O, vecSize, cudaMemcpyDeviceToHost));


    // Free device memory
    CUDA_CHECK(cudaFree(dev_A));
    CUDA_CHECK(cudaFree(dev_B));
    CUDA_CHECK(cudaFree(dev_T));
    CUDA_CHECK(cudaFree(dev_O));
    CUDA_CHECK(cudaFree(dev_M));
}


int main(int argc, char *argv[])
{
    int threadsPerBlock = 32;
    if (argc > 1) {
        try {
            threadsPerBlock = std::stoi(argv[1]);
        } catch (...) { }
    }

    /**
    Combinations tried:
    Blocks/Threads
    128/256
    64/512
    32/1024
    8/4096
    129/255
    256/128
    1024/32
    16384/2
    2048/16
    131072/256
    4096/32
    8192/32
    512/32
    512/32
    128/32
    16/32
 
    *Although benchmarking is hard when performance numbers change noticeably during reruns
    */


    std::cout << "Starting..." << std::endl;
    // int32_t size = 3;
    int32_t size = 32768;

    auto vec_a = (int32_t *)OOM_CHECK(malloc(sizeof(int32_t) * size));
    auto vec_b = (int32_t *)OOM_CHECK(malloc(sizeof(int32_t) * size));
    // Flat Buffer for matrix
    auto mat = (int32_t *)OOM_CHECK(malloc(sizeof(int32_t) * size * size)); // This was sizeof(int32_t *) but it's not pointer, is it?
#ifdef INCLUDE_SLOW
    auto out1 = (int32_t *)OOM_CHECK(malloc(sizeof(int32_t) * size));
#endif
    auto out2 = (int32_t *)OOM_CHECK(malloc(sizeof(int32_t) * size));

    init(size, vec_a, vec_b, mat);

    std::cout << "Init done" << std::endl;

    // pretty_print(size, vec_a, vec_b, mat);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;

#ifdef INCLUDE_SLOW
    std::cout << "Running slow" << std::endl;
    start = std::chrono::system_clock::now();
    compute(size, vec_a, vec_b, mat, out1);
    end = std::chrono::system_clock::now();

    std::cout << "First 3 entries of Out Vec:" << std::endl;
    for (int32_t i = 0; i < 3; i++)
        std::cout << out1[i] << std::endl;

    elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
#endif


    std::cout << "Running fast" << std::endl;
    start = std::chrono::system_clock::now();
    dev_compute(threadsPerBlock, size, vec_a, vec_b, mat, out2);
    end = std::chrono::system_clock::now();

    std::cout << "First 3 entries of Out Vec:" << std::endl;
    for (int32_t i = 0; i < 3; i++)
        std::cout << out2[i] << std::endl;

    elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

#ifdef INCLUDE_SLOW
    std::cout << "Diff: " << std::memcmp(out1, out2, sizeof(int32_t) * size) << std::endl;
#endif


    free(vec_a);
    free(vec_b);
    free(mat);
#ifdef INCLUDE_SLOW
    free(out1);
#endif
    free(out2);

    return 0;
}

