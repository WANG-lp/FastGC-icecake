#include "cuda.cuh"
#include <cuda_runtime.h>

__global__ void _cusub(int *a, int *b, int *c) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *c = *a - *b;
    }
}

int sub_cuda(int a, int b) {
    int *d_a, *d_b, *d_c;
    int out = 0;

    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMalloc(&d_c, sizeof(int));

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    _cusub<<<1, 1>>>(d_a, d_b, d_c);

    cudaMemcpy(&out, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    return out;
}