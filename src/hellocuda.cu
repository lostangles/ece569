
#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cudaError_t cudaerr;
    
    cuda_hello<<<1,1>>>();

    cudaerr = cudaDeviceSynchronize();

    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error %d \"%s\".\n",
	       cudaerr,
               cudaGetErrorString(cudaerr));
    return 0;
}

