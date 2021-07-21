#include <stdlib.h>
#include <cuda_profiler_api.h>

#define N 1000000

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int index = threadIdx.x;
    int stride = blockDim.x;

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
    vector_add<<<1,256>>>(out, a, b, N);

    cudaDeviceSynchronize();

    cudaFree(a);
    cudaFree(b);
    cudaFree(out);

    cudaDeviceReset();
    cudaProfilerStop();
}
