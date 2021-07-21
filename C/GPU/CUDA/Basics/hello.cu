#include <stdio.h>
#include <cuda_profiler_api.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cudaProfilerStart();

    cuda_hello<<<1,1>>>(); 

    cudaDeviceSynchronize();
    cudaDeviceReset();
    cudaProfilerStop();

    return 0;
}

