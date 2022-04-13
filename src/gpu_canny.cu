#include <vector>
#include <math.h>
#include <chrono> // time/clocks
#include <iostream> // cout, cerr

#include "canny.h"

#define _USE_MATH_DEFINES
#define RGB2GRAY_CONST_ARR_SIZE 3
#define STRONG_EDGE 0xFF
#define NON_EDGE 0x0

#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//*****************************************************************************************
// CUDA Gaussian Filter Implementation
//*****************************************************************************************

///
/// \brief Apply gaussian filter. This is the CUDA kernel for applying a gaussian blur to an image.
///
__global__
void cu_apply_gaussian_filter(pixel_t *in_pixels, pixel_t *out_pixels, int rows, int cols, double *in_kernel)
{
    //copy kernel array from global memory to a shared array
    __shared__ double kernel[KERNEL_SIZE][KERNEL_SIZE];
    extern __shared__ pixel_t shared_pixels[];

    int pixNum = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int pix = row * cols + col;

    int b_row = threadIdx.y;
    int b_col = threadIdx.x;
    int b_width = blockDim.x + (KERNEL_SIZE/2)*2;
    int b_height = blockDim.y + (KERNEL_SIZE/2)*2;
    int b_p = (b_row * b_width + b_width * (KERNEL_SIZE/2)) + b_col+(KERNEL_SIZE/2); // block pixel = b_p


    for (int i = 0; i < KERNEL_SIZE; ++i) {
        for (int j = 0; j < KERNEL_SIZE; ++j) {
            kernel[i][j] = in_kernel[i * KERNEL_SIZE + j];
        }
    }

    pixel_t temp;
    temp.red = 0;
    temp.green = 0;
    temp.blue = 0;
    //Load global pixels into shared memory pixels
    if (row < rows && col < cols)
	    shared_pixels[b_p] = in_pixels[pix];
    //Boundaries on top and left side also load right and bottom - speedup if this part splits?
    if (b_col < KERNEL_SIZE/2)
    {
       //Handle left edge
       if( ((pix-KERNEL_SIZE/2) >= row*cols )
         && ((pix-KERNEL_SIZE/2) < cols*rows) )
       {
          shared_pixels[b_p - KERNEL_SIZE/2] = in_pixels[pix - KERNEL_SIZE/2];
       }
       //Handle right edge
       if( ((pix+blockDim.x) < (row+1)*cols)
         && (row < rows))
       {
          shared_pixels[b_p+blockDim.x] = in_pixels[pix+blockDim.x]; 
       }
    }
    
    if (b_row < KERNEL_SIZE/2)
    {
       if( ((pix - ( blockDim.x * KERNEL_SIZE/2 )) >= 0)
        && ((pix - ( blockDim.x * KERNEL_SIZE/2 )) < rows*cols) )
       {
           shared_pixels[b_p - blockDim.x * KERNEL_SIZE/2] = in_pixels[pix - (blockDim.x*KERNEL_SIZE/2)];
       }
       if( ((pix + (blockDim.x * KERNEL_SIZE/2 )) <= rows*cols) )
       {
          shared_pixels[b_p + blockDim.x * KERNEL_SIZE/2] = in_pixels[pix + (blockDim.x*KERNEL_SIZE/2)];
       }
    }
    //End loading shared memory

    __syncthreads();

    //determine id of thread which corresponds to an individual pixel
    double kernelSum;
    double redPixelVal;
    double greenPixelVal;
    double bluePixelVal;

    if (pix >= 0 && pix < rows * cols) {

        //Apply Kernel to each pixel of image
        for (int i = 0; i < KERNEL_SIZE; ++i) {
            for (int j = 0; j < KERNEL_SIZE; ++j) {
                    redPixelVal += kernel[i][j] * shared_pixels[b_p + ((i - ((KERNEL_SIZE - 1) / 2))) + j - ((KERNEL_SIZE - 1) / 2)].red;
                    greenPixelVal += kernel[i][j] * shared_pixels[b_p + ((i - ((KERNEL_SIZE - 1) / 2))) + j - ((KERNEL_SIZE - 1) / 2)].green;
                    bluePixelVal += kernel[i][j] * shared_pixels[b_p + ((i - ((KERNEL_SIZE - 1) / 2))) + j - ((KERNEL_SIZE - 1) / 2)].blue;
                    kernelSum += kernel[i][j];
            }
        }
        //update output image
    }
    __syncthreads();
        out_pixels[pix].red = redPixelVal / kernelSum;
        out_pixels[pix].green = greenPixelVal / kernelSum;
        out_pixels[pix].blue = bluePixelVal / kernelSum;
    
}

//*****************************************************************************************
// CUDA Intensity Gradient Implementation
//*****************************************************************************************

///
/// \brief Compute gradient (first order derivative x and y). This is the CUDA kernel for taking the derivative of color contrasts in adjacent images.
///
__global__
void cu_compute_intensity_gradient(pixel_t *in_pixels, pixel_channel_t_signed *deltaX_channel, pixel_channel_t_signed *deltaY_channel, unsigned parser_length, unsigned offset)
{
    // compute delta X ***************************
    // deltaX = f(x+1) - f(x-1)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* condition here skips first and last row */
    if ((idx > offset) && (idx < (parser_length * offset) - offset))
    {
        int16_t deltaXred = 0;
        int16_t deltaYred = 0;
        int16_t deltaXgreen = 0;
        int16_t deltaYgreen = 0;
        int16_t deltaXblue = 0;
        int16_t deltaYblue = 0;

        /* first column */
        if((idx % offset) == 0)
        {
            // gradient at the first pixel of each line
            // note: at the edge pix[idx-1] does NOT exist
            deltaXred = (int16_t)(in_pixels[idx+1].red - in_pixels[idx].red);
            deltaXgreen = (int16_t)(in_pixels[idx+1].green - in_pixels[idx].green);
            deltaXblue = (int16_t)(in_pixels[idx+1].blue - in_pixels[idx].blue);
            // gradient at the first pixel of each line
            // note: at the edge pix[idx-1] does NOT exist
            deltaYred = (int16_t)(in_pixels[idx+offset].red - in_pixels[idx].red);
            deltaYgreen = (int16_t)(in_pixels[idx+offset].green - in_pixels[idx].green);
            deltaYblue = (int16_t)(in_pixels[idx+offset].blue - in_pixels[idx].blue);
        }
        /* last column */
        else if((idx % offset) == (offset - 1))
        {
            deltaXred = (int16_t)(in_pixels[idx].red - in_pixels[idx-1].red);
            deltaXgreen = (int16_t)(in_pixels[idx].green - in_pixels[idx-1].green);
            deltaXblue = (int16_t)(in_pixels[idx].blue - in_pixels[idx-1].blue);
            deltaYred = (int16_t)(in_pixels[idx].red - in_pixels[idx-offset].red);
            deltaYgreen = (int16_t)(in_pixels[idx].green - in_pixels[idx-offset].green);
            deltaYblue = (int16_t)(in_pixels[idx].blue - in_pixels[idx-offset].blue);
        }
        /* gradients where NOT edge */
        else
        {
            deltaXred = (int16_t)(in_pixels[idx+1].red - in_pixels[idx-1].red);
            deltaXgreen = (int16_t)(in_pixels[idx+1].green - in_pixels[idx-1].green);
            deltaXblue = (int16_t)(in_pixels[idx+1].blue - in_pixels[idx-1].blue);
            deltaYred = (int16_t)(in_pixels[idx+offset].red - in_pixels[idx-offset].red);
            deltaYgreen = (int16_t)(in_pixels[idx+offset].green - in_pixels[idx-offset].green);
            deltaYblue = (int16_t)(in_pixels[idx+offset].blue - in_pixels[idx-offset].blue);
        }
        deltaX_channel[idx] = (int16_t)(0.2989 * deltaXred + 0.5870 * deltaXgreen + 0.1140 * deltaXblue);
        deltaY_channel[idx] = (int16_t)(0.2989 * deltaYred + 0.5870 * deltaYgreen + 0.1140 * deltaYblue);
    }
}

//*****************************************************************************************
// CUDA Gradient Magnitude Implementation
//*****************************************************************************************

///
/// \brief Compute magnitude of gradient(deltaX & deltaY) per pixel.
///
__global__
void cu_magnitude(pixel_channel_t_signed *deltaX, pixel_channel_t_signed *deltaY, pixel_channel_t *out_pixel, unsigned parser_length, unsigned offset)
{
    //computation
    //Assigned a thread to each pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < parser_length * offset) {
            out_pixel[idx] =  (pixel_channel_t)(sqrt((double)deltaX[idx]*deltaX[idx] +
                            (double)deltaY[idx]*deltaY[idx]) + 0.5);
        }
}

//*****************************************************************************************
// CUDA Non Maximal Suppression Implementation
//*****************************************************************************************

///
/// \brief Non Maximal Suppression
/// If the centre pixel is not greater than neighboured pixels in the direction,
/// then the center pixel is set to zero.
/// This process results in one pixel wide ridges.
///
__global__
void cu_suppress_non_max(pixel_channel_t *mag, pixel_channel_t_signed *deltaX, pixel_channel_t_signed *deltaY, pixel_channel_t *nms, unsigned parser_length, unsigned offset)
{

    const pixel_channel_t SUPPRESSED = 0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < parser_length * offset)
    {
        float alpha;
        float mag1, mag2;
        // put zero all boundaries of image
        // TOP edge line of the image
        if((idx >= 0) && (idx <offset))
            nms[idx] = 0;

        // BOTTOM edge line of image
        else if((idx >= (parser_length-1)*offset) && (idx < (offset * parser_length)))
            nms[idx] = 0;

        // LEFT & RIGHT edge line
        else if(((idx % offset)==0) || ((idx % offset)==(offset - 1)))
        {
            nms[idx] = 0;
        }

        else // not the boundaries
        {
            // if magnitude = 0, no edge
            if(mag[idx] == 0)
                nms[idx] = SUPPRESSED;
                else{
                    if(deltaX[idx] >= 0)
                    {
                        if(deltaY[idx] >= 0)  // dx >= 0, dy >= 0
                        {
                            if((deltaX[idx] - deltaY[idx]) >= 0)       // direction 1 (SEE, South-East-East)
                            {
                                alpha = (float)deltaY[idx] / deltaX[idx];
                                mag1 = (1-alpha)*mag[idx+1] + alpha*mag[idx+offset+1];
                                mag2 = (1-alpha)*mag[idx-1] + alpha*mag[idx-offset-1];
                            }
                            else                                // direction 2 (SSE)
                            {
                                alpha = (float)deltaX[idx] / deltaY[idx];
                                mag1 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset+1];
                                mag2 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset-1];
                            }
                        }
                        else  // dx >= 0, dy < 0
                        {
                            if((deltaX[idx] + deltaY[idx]) >= 0)    // direction 8 (NEE)
                            {
                                alpha = (float)-deltaY[idx] / deltaX[idx];
                                mag1 = (1-alpha)*mag[idx+1] + alpha*mag[idx-offset+1];
                                mag2 = (1-alpha)*mag[idx-1] + alpha*mag[idx+offset-1];
                            }
                            else                                // direction 7 (NNE)
                            {
                                alpha = (float)deltaX[idx] / -deltaY[idx];
                                mag1 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset-1];
                                mag2 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset+1];
                            }
                        }
                    }

                    else
                    {
                        if(deltaY[idx] >= 0) // dx < 0, dy >= 0
                        {
                            if((deltaX[idx] + deltaY[idx]) >= 0)    // direction 3 (SSW)
                            {
                                alpha = (float)-deltaX[idx] / deltaY[idx];
                                mag1 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset-1];
                                mag2 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset+1];
                            }
                            else                                // direction 4 (SWW)
                            {
                                alpha = (float)deltaY[idx] / -deltaX[idx];
                                mag1 = (1-alpha)*mag[idx-1] + alpha*mag[idx+offset-1];
                                mag2 = (1-alpha)*mag[idx+1] + alpha*mag[idx-offset+1];
                            }
                        }

                        else // dx < 0, dy < 0
                        {
                             if((-deltaX[idx] + deltaY[idx]) >= 0)   // direction 5 (NWW)
                             {
                                 alpha = (float)deltaY[idx] / deltaX[idx];
                                 mag1 = (1-alpha)*mag[idx-1] + alpha*mag[idx-offset-1];
                                 mag2 = (1-alpha)*mag[idx+1] + alpha*mag[idx+offset+1];
                             }
                             else                                // direction 6 (NNW)
                             {
                                 alpha = (float)deltaX[idx] / deltaY[idx];
                                 mag1 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset-1];
                                 mag2 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset+1];
                             }
                        }
                    }

                    // non-maximal suppression
                    // compare mag1, mag2 and mag[t]
                    // if mag[t] is smaller than one of the neighbours then suppress it
                    if((mag[idx] < mag1) || (mag[idx] < mag2))
                         nms[idx] = SUPPRESSED;
                    else
                    {
                         nms[idx] = mag[idx];
                    }

            } // END OF ELSE (mag != 0)
        } // END OF FOR(j)
    } // END OF FOR(i)
}

//*****************************************************************************************
// CUDA Hysteresis Implementation
//*****************************************************************************************

///
/// \brief This is a helper function that runs on the GPU.
///
/// It checks if the eight immediate neighbors of a pixel at a given index are above
/// a low threshold, and if they are, sets them to strong edges. This effectively
/// connects the edges.
///
__device__
void trace_immed_neighbors(pixel_channel_t *out_pixels, pixel_channel_t *in_pixels,
                            unsigned idx, pixel_channel_t t_low, unsigned img_width)
{
    /* directions representing indices of neighbors */
    unsigned n, s, e, w;
    unsigned nw, ne, sw, se;

    /* get indices */
    n = idx - img_width;
    nw = n - 1;
    ne = n + 1;
    s = idx + img_width;
    sw = s - 1;
    se = s + 1;
    w = idx - 1;
    e = idx + 1;

    if (in_pixels[nw] >= t_low) {
        out_pixels[nw] = STRONG_EDGE;
    }
    if (in_pixels[n] >= t_low) {
        out_pixels[n] = STRONG_EDGE;
    }
    if (in_pixels[ne] >= t_low) {
        out_pixels[ne] = STRONG_EDGE;
    }
    if (in_pixels[w] >= t_low) {
        out_pixels[w] = STRONG_EDGE;
    }
    if (in_pixels[e] >= t_low) {
        out_pixels[e] = STRONG_EDGE;
    }
    if (in_pixels[sw] >= t_low) {
        out_pixels[sw] = STRONG_EDGE;
    }
    if (in_pixels[s] >= t_low) {
        out_pixels[s] = STRONG_EDGE;
    }
    if (in_pixels[se] >= t_low) {
        out_pixels[se] = STRONG_EDGE;
    }
}

///
/// \brief CUDA implementation of Canny hysteresis high thresholding.
///
/// This kernel is the first pass in the parallel hysteresis step.
/// It launches a thread for every pixel and checks if the value of that pixel
/// is above a high threshold. If it is, the thread marks it as a strong edge (set to 1)
/// in a pixel map and sets the value to the channel max. If it is not, the thread sets
/// the pixel map at the index to 0 and zeros the output buffer space at that index.
///
/// The output of this step is a mask of strong edges and an output buffer with white values
/// at the mask indices which are set.
///
__global__
void cu_hysteresis_high(pixel_channel_t *out_pixels, pixel_channel_t *in_pixels, unsigned *strong_edge_mask,
                        pixel_channel_t t_high, unsigned img_height, unsigned img_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (img_height * img_width)) {
        /* apply high threshold */
        if (in_pixels[idx] > t_high) {
            strong_edge_mask[idx] = 1;
            out_pixels[idx] = STRONG_EDGE;
        } else {
            strong_edge_mask[idx] = 0;
            out_pixels[idx] = NON_EDGE;
        }
    }
}

///
/// \brief CUDA implementation of Canny hysteresis low thresholding.
///
/// This kernel is the second pass in the parallel hysteresis step.
/// It launches a thread for every pixel, but skips the first and last rows and columns.
/// For surviving threads, the pixel at the thread ID index is checked to see if it was
/// previously marked as a strong edge in the first pass. If it was, the thread checks
/// their eight immediate neighbors and connects them (marks them as strong edges)
/// if the neighbor is above the low threshold.
///
/// The output of this step is an output buffer with both "strong" and "connected" edges
/// set to whtie values. This is the final edge detected image.
///
__global__
void cu_hysteresis_low(pixel_channel_t *out_pixels, pixel_channel_t *in_pixels, unsigned *strong_edge_mask,
                        unsigned t_low, unsigned img_height, unsigned img_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((idx > img_width)                               /* skip first row */
        && (idx < (img_height * img_width) - img_width) /* skip last row */
        && ((idx % img_width) < (img_width - 1))        /* skip last column */
        && ((idx % img_width) > (0)) )                  /* skip first column */
    {
        if (1 == strong_edge_mask[idx]) { /* if this pixel was previously found to be a strong edge */
            trace_immed_neighbors(out_pixels, in_pixels, idx, t_low, img_width);
        }
    }
}

void cu_detect_edges(pixel_channel_t *final_pixels, pixel_t *orig_pixels, int rows, int cols, double kernel[KERNEL_SIZE][KERNEL_SIZE])
{
    /* kernel execution configuration parameters */
    int num_blks = (rows * cols) / 1024;
    int thd_per_blk = 1024;
    int grid = 0;
    int thr = 32;
    dim3 b(thr,thr);
    dim3 g((cols+b.x-1)/b.x,(rows+b.y-1)/b.y);
    pixel_channel_t t_high = 0x0A;
    pixel_channel_t t_low = 0x00;

    /* device buffers */
    pixel_t *in, *out;
    pixel_channel_t *single_channel_buf0;
    pixel_channel_t *single_channel_buf1;
    pixel_channel_t_signed *deltaX;
    pixel_channel_t_signed *deltaY;
    double *d_blur_kernel;
    unsigned *idx_map;

    /* allocate device memory */
    cudaMalloc((void**) &in, sizeof(pixel_t)*rows*cols);
    cudaMalloc((void**) &out, sizeof(pixel_t)*rows*cols);
    cudaMalloc((void**) &single_channel_buf0, sizeof(pixel_channel_t)*rows*cols);
    cudaMalloc((void**) &single_channel_buf1, sizeof(pixel_channel_t)*rows*cols);
    cudaMalloc((void**) &deltaX, sizeof(pixel_channel_t_signed)*rows*cols);
    cudaMalloc((void**) &deltaY, sizeof(pixel_channel_t_signed)*rows*cols);
    cudaMalloc((void**) &idx_map, sizeof(idx_map[0])*rows*cols);
    cudaMalloc((void**) &d_blur_kernel, sizeof(d_blur_kernel[0])*KERNEL_SIZE*KERNEL_SIZE);

    /* data transfer image pixels to device */
    cudaMemcpy(in, orig_pixels, rows*cols*sizeof(pixel_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blur_kernel, kernel, sizeof(d_blur_kernel[0])*KERNEL_SIZE*KERNEL_SIZE, cudaMemcpyHostToDevice);

    /* run canny edge detection core - CUDA kernels */
    /* use streams to ensure the kernels are in the same task */
    cudaStream_t stream;
    cudaStreamCreate(&stream);


    std::chrono::high_resolution_clock::time_point start1 = std::chrono::high_resolution_clock::now();
    cu_apply_gaussian_filter<<<g, b, (thr + KERNEL_SIZE*2) * (thr + KERNEL_SIZE*2) * sizeof(pixel_t) , stream>>>(in, out, rows, cols, d_blur_kernel);
    cudaDeviceSynchronize();
    std::chrono::high_resolution_clock::time_point end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
    std::cout << "Elapsed time Gaussian: " << duration1 << " us" << std::endl;
    
    std::chrono::high_resolution_clock::time_point start2 = std::chrono::high_resolution_clock::now();
    cu_compute_intensity_gradient<<<num_blks, thd_per_blk, grid, stream>>>(out, deltaX, deltaY, rows, cols);
    cudaDeviceSynchronize();
    std::chrono::high_resolution_clock::time_point end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
    std::cout << "Elapsed time Intensity Gradient: " << duration2 << " us" << std::endl;
    
    std::chrono::high_resolution_clock::time_point start3 = std::chrono::high_resolution_clock::now();
    cu_magnitude<<<num_blks, thd_per_blk, grid, stream>>>(deltaX, deltaY, single_channel_buf0, rows, cols);
    cudaDeviceSynchronize();
    std::chrono::high_resolution_clock::time_point end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3).count();
    std::cout << "Elapsed time magnitude: " << duration3 << " us" << std::endl;

    std::chrono::high_resolution_clock::time_point start4 = std::chrono::high_resolution_clock::now();
    cu_suppress_non_max<<<num_blks, thd_per_blk, grid, stream>>>(single_channel_buf0, deltaX, deltaY, single_channel_buf1, rows, cols);    
    cudaDeviceSynchronize();
    std::chrono::high_resolution_clock::time_point end4 = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(end4 - start4).count();
    std::cout << "Elapsed time supression: " << duration4 << " us" << std::endl;

    std::chrono::high_resolution_clock::time_point start5 = std::chrono::high_resolution_clock::now();
    cu_hysteresis_high<<<num_blks, thd_per_blk, grid, stream>>>(single_channel_buf0, single_channel_buf1, idx_map, t_high, rows, cols);
    cudaDeviceSynchronize();
    std::chrono::high_resolution_clock::time_point end5 = std::chrono::high_resolution_clock::now();
    auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(end5 - start5).count();
    std::cout << "Elapsed time hysterisis high: " << duration5 << " us" << std::endl;

    std::chrono::high_resolution_clock::time_point start6 = std::chrono::high_resolution_clock::now();
    cu_hysteresis_low<<<num_blks, thd_per_blk, grid, stream>>>(single_channel_buf0, single_channel_buf1, idx_map, t_low, rows, cols);
    cudaDeviceSynchronize();
    std::chrono::high_resolution_clock::time_point end6 = std::chrono::high_resolution_clock::now();
    auto duration6 = std::chrono::duration_cast<std::chrono::microseconds>(end6 - start6).count();
    std::cout << "Elapsed time hysterisis low: " << duration6 << " us" << std::endl;




    /* wait for everything to finish */
    cudaDeviceSynchronize();

    /* copy result back to the host */
    cudaMemcpy(final_pixels, single_channel_buf0, rows*cols*sizeof(pixel_channel_t), cudaMemcpyDeviceToHost);

    /* cleanup */
    cudaFree(in);
    cudaFree(out);
    cudaFree(single_channel_buf0);
    cudaFree(single_channel_buf1);
    cudaFree(deltaX);
    cudaFree(deltaY);
    cudaFree(idx_map);
    cudaFree(d_blur_kernel);
}

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

void populate_blur_kernel(double out_kernel[KERNEL_SIZE][KERNEL_SIZE])
{
    double scaleVal = 1;
    double stDev = (double)KERNEL_SIZE/3;

    for (int i = 0; i < KERNEL_SIZE; ++i) {
        for (int j = 0; j < KERNEL_SIZE; ++j) {
            double xComp = pow((i - KERNEL_SIZE/2), 2);
            double yComp = pow((j - KERNEL_SIZE/2), 2);

            double stDevSq = pow(stDev, 2);
            double pi = M_PI;

            //calculate the value at each index of the Kernel
            double kernelVal = exp(-(((xComp) + (yComp)) / (2 * stDevSq)));
            kernelVal = (1 / (sqrt(2 * pi)*stDev)) * kernelVal;

            //populate Kernel
            out_kernel[i][j] = kernelVal;

            if (i==0 && j==0) 
            {
                scaleVal = out_kernel[0][0];
            }

            //normalize Kernel
            out_kernel[i][j] = out_kernel[i][j] / scaleVal;
        }
    }
}

int main(int argc, char *argv[]) {
    cudaError_t cudaerr;
    float* deviceA;
    int result = 0;
    Image* img = (Image*)malloc(sizeof(Image));
    char* defaultFname = "../images/image1.png";
    char* defaultOutName = "output.jpg";
    char* fname;
    char* outFname;
    if (argc != 5)
    {
       fname = defaultFname;
       outFname = defaultOutName;
    }
    else
    {
       fname = argv[2];
       outFname = argv[4];
    }
    if((img->data = stbi_load(fname, &img->width, &img->height, &img->channels, STBI_rgb)) != NULL) 
    {
        img->size = img->width * img->height * img->channels;
        img->allocation_ = STB_ALLOCATED;
    }

    stbi_write_jpg("Original_Image.jpg", img->width, img->height, 3, img->data, 100);

    pixel_t* inputImage = (pixel_t*)malloc( (img->width * img->height)*sizeof(pixel_t) );
    pixel_channel_t* outputImage = (pixel_channel_t*)malloc( (img->width * img->height)*sizeof(pixel_channel_t) * img->channels);

    for(int i=0; i<img->height; i++)
    {
       for(int k=0; k<img->width; k++)
       {
	  size_t base = (k + img->width * i);
          inputImage[base].red    = img->data[ base*3 ];     //Red
          inputImage[base].green  = img->data[ base*3 + 1]; //Green
	  inputImage[base].blue   = img->data[ base*3 + 2]; //blue
       }
    }

    //Intermediat debug output
    /*
    char* stbiDebug = (char*)malloc( (img->height * img->width) * 3  );
    for(int i=0; i<img->height; i++)
    {
       for(int k=0; k<img->width; k++)
       {
	  size_t base = (k + img->width * i);
          stbiDebug[base*3    ] = inputImage[base].red;
	  stbiDebug[base*3 + 1] = inputImage[base].green;
	  stbiDebug[base*3 + 2] = inputImage[base].blue;
       }
    }
    

    stbi_write_jpg("debug.jpg", img->width, img->height, 3, stbiDebug, 100);
    */

    double kernel[KERNEL_SIZE][KERNEL_SIZE];
    populate_blur_kernel(kernel);
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    cu_detect_edges(outputImage, inputImage, img->height, img->width, kernel); 
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Elapsed time total: " << duration << " us" << std::endl;

    //Convert 16 bit pixel_type_t back into 8 bytes for stbi_write_image
    char* stbiOut = (char*)malloc( (img->height * img->width)  );
    for(int i=0; i<img->height*img->width; i++)
    {
          stbiOut[i] = outputImage[i];
	  if (outputImage[i] > 255)
		  printf("outputImage[%d]: %d\n", i, outputImage[i]);
    }



    stbi_write_jpg(outFname, img->width, img->height, 1, stbiOut, 100);

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

    free(img);
    free (inputImage);
    free (outputImage);
    return 0;
}

