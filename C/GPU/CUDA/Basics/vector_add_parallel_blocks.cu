#include <stdlib.h>
#include <cuda_profiler_api.h>

#define N 1000000

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < n; i += stride){
        out[i] = a[i] + b[i];
    }
}

int main(){
    cudaProfilerStart();

    float *a, *b, *out; 

    // Allocate memory
    cudaMallocManaged(&a, sizeof(float) * N);
    cudaMallocManaged(&b, sizeof(float) * N);
    cudaMallocManaged(&out, sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // Main function
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vector_add<<<numBlocks, blockSize>>>(out, a, b, N);

    cudaDeviceSynchronize();

    cudaFree(a);
    cudaFree(b);
    cudaFree(out);

    cudaDeviceReset();
    cudaProfilerStop();
}
