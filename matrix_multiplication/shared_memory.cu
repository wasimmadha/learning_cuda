#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// Define block size
#define BLOCK_SIZE 32

// Helper macro for ceiling division
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__global__ void sgemm_shared(int M, int N, int K, float alpha, const float *A, 
                             const float *B, float beta, float *C) {
    // Shared memory arrays for blocks of A and B
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    // Thread identifiers
    int threadRow = threadIdx.y;  // Row within the block
    int threadCol = threadIdx.x;  // Column within the block

    // Block identifiers
    int cRow = blockIdx.y;  // Row of the block in C
    int cCol = blockIdx.x;  // Column of the block in C

    // Advance pointers to starting positions
    const float *A_start = A + cRow * BLOCK_SIZE * K;
    const float *B_start = B + cCol * BLOCK_SIZE;
    float *C_start = C + cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE;

    // Initialize the result for this thread
    float tmp = 0.0f;

    // Loop over all chunks of A and B
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE) {
        // Each thread loads its corresponding elements from A and B into shared memory
        As[threadRow * BLOCK_SIZE + threadCol] = 
            A_start[threadRow * K + (threadCol + bkIdx)];
        Bs[threadRow * BLOCK_SIZE + threadCol] = 
            B_start[(threadRow + bkIdx) * N + threadCol];

        // Synchronize to make sure shared memory is fully populated
        __syncthreads();

        // Perform dot product
        for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCK_SIZE + dotIdx] * 
                   Bs[dotIdx * BLOCK_SIZE + threadCol];
        }

        // Synchronize to prevent overwriting shared memory prematurely
        __syncthreads();
    }

    // Write the computed value to the C matrix
    C_start[threadRow * N + threadCol] = alpha * tmp + 
                                         beta * C_start[threadRow * N + threadCol];
}

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // Define matrix dimensions
    const int M = 4092;  // rows of A and C
    const int N = 4092;  // columns of B and C
    const int K = 4092;  // columns of A and rows of B

    // Define SGEMM parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices with random values
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    init_matrix(h_C, M, N);  // Initialize C even though beta is 0 in this example

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE), 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Benchmark kernel execution
    printf("Running SGEMM with matrix dimensions M=%d, N=%d, K=%d\n", M, N, K);
    printf("Grid dimensions: %d x %d\n", gridDim.x, gridDim.y);
    
    double start_time = get_time();
    
    // Launch kernel
    sgemm_shared<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    
    // Synchronize and check timing
    cudaDeviceSynchronize();
    double end_time = get_time();
    
    printf("Kernel execution time: %f ms\n", (end_time - start_time) * 1000.0);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate and print GFLOPS
    double gflops = (2.0 * M * N * K) / ((end_time - start_time) * 1e9);
    printf("Performance: %.2f GFLOPS\n", gflops);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}