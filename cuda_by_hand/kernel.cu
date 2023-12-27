#include <stdlib.h>
#include <stdio.h> 
#include "kernel.hu"
#define ORDER 1
#define MAX_BRIGHTNESS 255
#define GetValue(a) ((a)&0xff)
#define ITERATION_NUM 10

__device__ void cuda_image_set_pixel( int x, int y, unsigned clr, int width, int height, unsigned *data ) {
	if ( x < 0 || x >= width || y < 0 || y >= height ) return;
	data[y*width+x] = clr;
}

__device__ double cuda_image_get_pixeld( int x, int y, int width, int height, unsigned *data) {
	if ( x < 0 || x >= width || y < 0 || y >= height ) return 0.;
	return (double) GetValue(data[y*width+x]);
}

__global__ void stencilCUDA(int width, int height, unsigned *d_in, unsigned *d_out, unsigned *d_buffer){
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x < 1 || x >= width - 2 || y < 1 || y >= height - 2 ) 
        return;
    // printf("x: %d, y: %d\n", x, y);

    double val = abs(
			8 * cuda_image_get_pixeld(x, y, width, height, d_in)  -(
			cuda_image_get_pixeld(x-1, y-1 , width, height, d_in) +
			cuda_image_get_pixeld(x  , y-1 , width, height, d_in) +
			cuda_image_get_pixeld(x+1, y-1 , width, height, d_in) +
			cuda_image_get_pixeld(x-1, y   , width, height, d_in) +
			cuda_image_get_pixeld(x+1, y   , width, height, d_in) +
			cuda_image_get_pixeld(x-1, y+1 , width, height, d_in) +
			cuda_image_get_pixeld(x  , y+1 , width, height, d_in) +
			cuda_image_get_pixeld(x+1, y+1 , width, height, d_in)));
	
    // buffer = val;
    // if ( val > max ) max = val;
    // if ( val < min ) min = val;
    // val = MAX_BRIGHTNESS * (buffer - min) / (max-min);
    // cuda_image_set_pixel(x, y, val , width, height, d_out);
    d_buffer[y*width+x] = val;
}

__global__ void produceOutputCUDA(int width, int height, int min, int max, unsigned *d_out, unsigned *d_buf){
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    double val = MAX_BRIGHTNESS * (cuda_image_get_pixeld(x, y, width, height, d_buf) - min) / (max-min);
    if (val > 15)
        val = 255;
    cuda_image_set_pixel(x, y, val , width, height, d_out);
}