// This program computes matrix multiplication using shared memory tiling
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <chrono>
#include <string>

typedef std::chrono::high_resolution_clock Clock;

using std::cout;
using std::generate;
using std::vector;

// Matrix dimensions
constexpr int M = 13;
constexpr int N = 13;
constexpr int K = 13;

int32_t matrixAGlobal[N * N] = {
        14, 39, 117, 89, 111, 73, 79, 102, 52, 81, 123, 70, 39,
        82, 29, 125, 85, 51, 60, 102, 39, 120, 106, 19, 15, 58,
        124, 31, 32, 23, 19, 69, 60, 61, 10, 33, 72, 1, 91,
        96, 112, 32, 111, 90, 12, 63, 77, 47, 105, 115, 38, 90,
        13, 35, 23, 78, 57, 109, 122, 89, 21, 116, 86, 123, 113,
        27, 14, 80, 69, 9, 23, 106, 26, 115, 31, 6, 73, 112,
        53, 70, 64, 118, 121, 17, 6, 113, 30, 8, 5, 116, 66,
        12, 113, 71, 94, 98, 116, 2, 95, 66, 107, 54, 11, 34,
        90, 36, 81, 124, 73, 41, 105, 14, 127, 109, 87, 29, 2,
        84, 77, 56, 81, 21, 81, 110, 110, 123, 104, 113, 39, 54,
        75, 102, 44, 79, 61, 55, 90, 125, 52, 45, 4, 120, 12,
        20, 20, 105, 41, 20, 44, 108, 74, 72, 62, 76, 34, 111,
        38, 97, 124, 5, 97, 87, 85, 106, 12, 31, 87, 6, 77
};

int32_t matrixBGlobal[N * N] = {
        69, 96, 71, 89, 127, 108, 96, 121, 64, 65, 62, 91, 73,
        9, 67, 113, 48, 47, 53, 96, 66, 7, 63, 17, 9, 8,
        107, 45, 112, 33, 114, 48, 102, 70, 52, 47, 34, 81, 17,
        38, 15, 61, 1, 104, 82, 68, 53, 69, 110, 12, 25, 46,
        111, 89, 54, 0, 107, 81, 127, 124, 36, 17, 99, 117, 75,
        125, 72, 48, 67, 31, 104, 64, 98, 94, 57, 81, 15, 16,
        111, 16, 127, 119, 88, 41, 75, 125, 22, 50, 120, 6, 81,
        75, 7, 78, 38, 35, 115, 114, 37, 66, 106, 64, 91, 97,
        75, 102, 84, 112, 65, 76, 87, 22, 45, 100, 19, 18, 89,
        27, 25, 109, 18, 116, 19, 116, 33, 103, 31, 29, 78, 8,
        24, 12, 86, 20, 32, 53, 31, 13, 51, 36, 100, 56, 44,
        13, 8, 54, 24, 101, 73, 115, 120, 56, 23, 63, 39, 93,
        77, 50, 108, 56, 106, 58, 121, 74, 70, 88, 19, 49, 83
};

// Threads per CTA dimension
constexpr int THREADS = 16;

// Padded matrix dimensions
constexpr int M_padded = M + THREADS - M % THREADS;
constexpr int N_padded = N + THREADS - N % THREADS;
constexpr int K_padded = K + THREADS - K % THREADS;

// Size of shared memory per TB
constexpr int SHMEM_SIZE = THREADS * THREADS;

__global__ void matrixMul(const int* a, const int* b, int* c) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("Dim: %d \n", blockIdx.x);

    // Statically allocated shared memory
    __shared__ int s_a[SHMEM_SIZE];
    __shared__ int s_b[SHMEM_SIZE];

    // Accumulate in temporary variable
    int tmp = 0;

    // Sweep tile across matrix
    for (int i = 0; i < K_padded; i += blockDim.x) {
        // Load in elements for this tile
        s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K + i + threadIdx.x];
        s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];

        //printf("A: %d B: %d \n", row * K + i + threadIdx.x, i * N + threadIdx.y * N + col);
        

        // Wait for both tiles to be loaded in before doing computation
        __syncthreads();

        // The C = A + B part
        tmp += s_a[threadIdx.y * blockDim.x + threadIdx.x] + s_b[threadIdx.y * blockDim.x + threadIdx.x];

        // Do matrix multiplication on the small matrix
        for (int j = 0; j < blockDim.x; j++) {
            tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
        }

        // Wait for all threads to finish using current tiles before loading in new
        // ones
        __syncthreads();
    }

    // Write back results
    if (row < M && col < N) c[row * N + col] = tmp;
}

// Calculates AB + A + B
__global__ void
gpuMatrixComputation(const int32_t* matrixA, const int32_t* matrixB, int32_t* matrixC) {
    int8_t c, d, k;
    for (c = 0; c < N; ++c) {
        for (d = 0; d < N; ++d) {
            matrixC[c * N + d] = 0;
            for (k = 0; k < N; ++k) {
                matrixC[c * N + d] += matrixA[c * N + k] * matrixB[k * N + d];
            }
            matrixC[c * N + d] += matrixA[c * N + d] + matrixB[c * N + d];
        }
    }
}

// Calculates AB + A + B
__global__ void
gpuParallelMatrixComputation(const int32_t* matrixA, const int32_t* matrixB, int32_t* matrixC, const int size) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;

#pragma unroll
    for (int k = 0; k < size; ++k) {
#ifdef TRANSPOSED
        sum += matrixA[x * size + k] * matrixB[y * size + k];
#else
        sum += matrixA[x * size + k] * matrixB[k * size + y];
#endif
    }
    sum += matrixA[x * size + y] + matrixB[x * size + y];
    if (x < M && y < N) matrixC[x * size + y] = sum;
}

// Check result on the CPU
// MxN = MxK * KxN
void verify_result(vector<int>& a, vector<int>& b, vector<int>& c) {
    // For every row...
    for (int row = 0; row < M_padded; row++) {
        if (row >= M) continue;
        // For every column...
        for (int col = 0; col < N_padded; col++) {
            if (col >= N) continue;
            // For every element in the row-column pair
            int tmp = 0;
            for (int i = 0; i < K_padded; i++) {
                // Accumulate the partial results
                tmp += a[row * K + i] * b[i * N + col];
            }
            tmp += a[row * N + col] + b[row * N + col];
            
            // Check against the CPU result
            // printf("tmp: %d c: %d\n", tmp, c[row * N + col]);
           // assert(tmp == c[row * N + col]);
        }
    }
}

int main() {
    // Size (in bytes) of matrix
    // MxN = MxK * KxN
    size_t bytes_a = M_padded * K_padded * sizeof(int);
    size_t bytes_b = K_padded * N_padded * sizeof(int);
    size_t bytes_c = M * N * sizeof(int);

    // Host vectors
    vector<int> h_a(M_padded * K_padded);
    vector<int> h_b(K_padded * N_padded);
    vector<int> h_c(M * N);

    // Initialize matrices
    // Padded matrix A
    for (int i = 0; i < M_padded; i++) {
        for (int j = 0; j < K_padded; j++) {
            if (i < M && j < K) h_a[i * K + j] = matrixAGlobal[i * K + j];
        }
    }
    // Padded matrix B
    for (int i = 0; i < K_padded; i++) {
        for (int j = 0; j < N_padded; j++) {
            if (i < K && j < N) h_b[i * N + j] = matrixBGlobal[i * K + j];
        }
    }

    // Allocate device memory
    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice);

    // Blocks per grid dimension (assumes THREADS divides M and N evenly)
    int BLOCKS_X = N_padded / THREADS;
    int BLOCKS_Y = M_padded / THREADS;

    printf("x: %d y: %d\n", BLOCKS_X, BLOCKS_Y);

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS_X, BLOCKS_Y);

    unsigned long time_taken = 0;
    for (uint16_t i = 0; i < 1e3; i++) {
        auto cpu_start = Clock::now();

        // Launch kernel
        // Shared memory
        matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

        // Global memory
        // gpuParallelMatrixComputation <<<blocks, threads >>> (d_a, d_b, d_c, N);

        // Non-parallized
        // gpuMatrixComputation <<<1, 1>>> (d_a, d_b, d_c);

        auto cpu_end = Clock::now();

        time_taken += std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count();
    }


    // Copy back to the host
    cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);

    // Check result
    verify_result(h_a, h_b, h_c);

    for (int c = 0; c < N; c++) {
        for (int d = 0; d < N; d++) {
            std::cout << std::to_string((int32_t)h_c[c * N + d]) + "\t";
        }
        std::cout << "\n";
    }
    std::cout << "Calculation took: " << std::to_string(time_taken / 1000) << " nanoseconds" << std::endl;
    cout << "COMPLETED SUCCESSFULLY\n";

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}