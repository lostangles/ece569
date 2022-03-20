
#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cudaError_t cudaerr;
    float* deviceA;
    int result = 0;

    size_t available, total;
    cudaMemGetInfo(&available, &total);


    if(result = (cudaMalloc((void **) &deviceA, 10 * sizeof(float)) != cudaSuccess))
	printf("Failed to malloc deviceA, errorCode:%d\n", result);

    cuda_hello<<<1,1>>>();

    cudaerr = cudaDeviceSynchronize();

    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error %d \"%s\".\n",
	       cudaerr,
               cudaGetErrorString(cudaerr));
    return 0;
}

