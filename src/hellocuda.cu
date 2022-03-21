
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

enum allocation_type {
    NO_ALLOCATION, SELF_ALLOCATED, STB_ALLOCATED
};

typedef struct {
    int width;
    int height;
    int channels;
    size_t size;
    uint8_t *data;
    enum allocation_type allocation_;
} Image;

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cudaError_t cudaerr;
    float* deviceA;
    int result = 0;
    Image* img = (Image*)malloc(sizeof(Image));
    char* fname = "../images/image1.png";
    if((img->data = stbi_load(fname, &img->width, &img->height, &img->channels, 0)) != NULL) 
    {
        img->size = img->width * img->height * img->channels;
        img->allocation_ = STB_ALLOCATED;
    }

    printf("img->size:  %d img->width: %d image->height: %d img->channels: %d\n", img->size, img->width, img->height, img->channels);

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

