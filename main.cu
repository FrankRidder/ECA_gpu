#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 13

typedef std::chrono::high_resolution_clock Clock;

int32_t matrixCGlobal[N][N];

int8_t matrixAGlobal[N][N] = {
        {14,  39,  117, 89,  111, 73,  79,  102, 52,  81,  123, 70,  39},
        {82,  29,  125, 85,  51,  60,  102, 39,  120, 106, 19,  15,  58},
        {124, 31,  32,  23,  19,  69,  60,  61,  10,  33,  72,  1,   91},
        {96,  112, 32,  111, 90,  12,  63,  77,  47,  105, 115, 38,  90},
        {13,  35,  23,  78,  57,  109, 122, 89,  21,  116, 86,  123, 113},
        {27,  14,  80,  69,  9,   23,  106, 26,  115, 31,  6,   73,  112},
        {53,  70,  64,  118, 121, 17,  6,   113, 30,  8,   5,   116, 66},
        {12,  113, 71,  94,  98,  116, 2,   95,  66,  107, 54,  11,  34},
        {90,  36,  81,  124, 73,  41,  105, 14,  127, 109, 87,  29,  2},
        {84,  77,  56,  81,  21,  81,  110, 110, 123, 104, 113, 39,  54},
        {75,  102, 44,  79,  61,  55,  90,  125, 52,  45,  4,   120, 12},
        {20,  20,  105, 41,  20,  44,  108, 74,  72,  62,  76,  34,  111},
        {38,  97,  124, 5,   97,  87,  85,  106, 12,  31,  87,  6,   77}
};

int8_t matrixBGlobal[N][N] = {
        {69,  96,  71,  89,  127, 108, 96,  121, 64,  65,  62,  91,  73},
        {9,   67,  113, 48,  47,  53,  96,  66,  7,   63,  17,  9,   8},
        {107, 45,  112, 33,  114, 48,  102, 70,  52,  47,  34,  81,  17},
        {38,  15,  61,  1,   104, 82,  68,  53,  69,  110, 12,  25,  46},
        {111, 89,  54,  0,   107, 81,  127, 124, 36,  17,  99,  117, 75},
        {125, 72,  48,  67,  31,  104, 64,  98,  94,  57,  81,  15,  16},
        {111, 16,  127, 119, 88,  41,  75,  125, 22,  50,  120, 6,   81},
        {75,  7,   78,  38,  35,  115, 114, 37,  66,  106, 64,  91,  97},
        {75,  102, 84,  112, 65,  76,  87,  22,  45,  100, 19,  18,  89},
        {27,  25,  109, 18,  116, 19,  116, 33,  103, 31,  29,  78,  8},
        {24,  12,  86,  20,  32,  53,  31,  13,  51,  36,  100, 56,  44},
        {13,  8,   54,  24,  101, 73,  115, 120, 56,  23,  63,  39,  93},
        {77,  50,  108, 56,  106, 58,  121, 74,  70,  88,  19,  49,  83}
};

// Calculates AB + A + B
void naiveMatrixComputation() {
    int8_t c, d, k;
    int32_t sum;
    for (c = 0; c < N; c++) {
        for (d = 0; d < N; d++) {
            sum = 0;
            for (k = 0; k < N; ++k) {
                sum += matrixAGlobal[c][k] * matrixBGlobal[k][d];
            }
            sum += matrixAGlobal[c][d] + matrixBGlobal[c][d];
            matrixCGlobal[c][d] = sum;
        }
    }
}

// Calculates AB + A + B
__global__ void
transposedMatrixComputation(int8_t **matrixA, int8_t **matrixB, int8_t **matrixB_Transposed, int32_t **matrixC) {
    int8_t c, d, k;
    for (c = 0; c < N; ++c)
        for (d = 0; d < N; ++d)
            matrixB_Transposed[c][d] = matrixB[d][c];
    for (c = 0; c < N; ++c) {
        for (d = 0; d < N; ++d) {
            matrixC[c][d] = 0;
            for (k = 0; k < N; ++k) {
                matrixC[c][d] += matrixA[c][k] * matrixB_Transposed[d][k];
            }
            matrixC[c][d] += matrixA[c][d] + matrixB[c][d];
        }
    }
}

int main() {
    /*
     * Measure the time it takes to execute 1000 times
     */

    std::cout << "Malloc device mem" << std::endl;

    int8_t **gpu_a, **gpu_b, **gpu_transpose;
    int32_t **gpu_c;

    int32_t **c_out;
    c_out = (int32_t **) malloc(N * N * sizeof(int32_t));

    // We need variables accessible to the GPU,
    // so cudaMallocManaged provides these
    if (cudaMallocManaged(&gpu_a, N * N * sizeof(int8_t)) != 0) {
        std::cout << "malloc failed" << std::endl;
    }
    if (cudaMallocManaged(&gpu_b, N * N * sizeof(int8_t)) != 0) {
        std::cout << "malloc failed" << std::endl;
    }

    if (cudaMallocManaged(&gpu_transpose, N * sizeof(int8_t)) != 0) {
        std::cout << "malloc failed" << std::endl;
    }
    if (cudaMallocManaged(&gpu_c, N * N * sizeof(int32_t)) != 0) {
        std::cout << "malloc failed" << std::endl;
    }

    std::cout << "Move to device mem" << std::endl;

    if (cudaMemcpy(gpu_a, matrixAGlobal, (N * N * sizeof(int8_t)), cudaMemcpyHostToDevice) != 0) {
        std::cout << "memcpy failed" << std::endl;
    }
    if (cudaMemcpy(gpu_b, matrixBGlobal, (N * N * sizeof(int8_t)), cudaMemcpyHostToDevice) != 0) {
        std::cout << "memcpy failed" << std::endl;
    }

    std::cout << "Computation" << std::endl;

    unsigned long time_taken = 0;
    for (uint16_t i = 0; i < 1000; i++) {
        auto cpu_start = Clock::now();
        // naiveMatrixComputation();
        transposedMatrixComputation<<<1, 1>>>(gpu_a, gpu_b, gpu_transpose, gpu_c);
        cudaDeviceSynchronize();
        auto cpu_end = Clock::now();

        time_taken += std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count();
    }
    /*
     * Print the resultating matrix
     */
    int8_t c, d;

    std::cout << "Copy from device to host" << std::endl;
    int cudaError = cudaMemcpy(c_out, gpu_c, (N * N * sizeof(int32_t)), cudaMemcpyDeviceToHost);
    if (cudaError != 0) {
        std::cout << "memcpy output failed: " << cudaError << std::endl;
    }

    std::cout << "Printing mem" << std::endl;

    for (c = 0; c < N; c++) {
        for (d = 0; d < N; d++) {
            std::cout << std::to_string((int32_t) c_out[c][d]) + "\t";
        }
        std::cout << "\n";
    }

    std::cout << "Freeing mem" << std::endl;

    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_transpose);
    cudaFree(gpu_c);

    std::cout << "Calculation took: " << std::to_string(time_taken / 1000) << " nanoseconds" << std::endl;

    return 0;
}
