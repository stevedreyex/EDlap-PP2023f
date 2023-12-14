#include <stdlib.h>
#include "kernel.hu"
#define ORDER 1
#define MAX_BRIGHTNESS 255
#define GetValue(a) ((a)&0xff)
#define ITERATION_NUM 10

__device__ void cuda_image_set_pixel( Image* image, int x, int y, unsigned clr ) {
	if ( x < 0 || x >= image->width || y < 0 || y >= image->height ) return;
	image->data[y*image->width+x] = clr;
}

__device__ double cuda_image_get_pixeld( Image* image, int x, int y ) {
	if ( x < 0 || x >= image->width || y < 0 || y >= image->height ) return 0.;
	return (double)GetValue(image->data[y*image->width+x]);
}

__global__ void stencilCUDA(Image* in, Image* out){
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( x < ORDER || x >= in->width-ORDER || y < ORDER || y >= in->height-ORDER ) return;
        return;

    double min = 1.0, max = 0.0;
    double buffer = 0.0;

    double val = abs(
			8 * cuda_image_get_pixeld(in, x, y)  -(
			cuda_image_get_pixeld(in, x-1, y-1 ) +
			cuda_image_get_pixeld(in, x  , y-1 ) +
			cuda_image_get_pixeld(in, x+1, y-1 ) +
			cuda_image_get_pixeld(in, x-1, y   ) +
			cuda_image_get_pixeld(in, x+1, y   ) +
			cuda_image_get_pixeld(in, x-1, y+1 ) +
			cuda_image_get_pixeld(in, x  , y+1 ) +
			cuda_image_get_pixeld(in, x+1, y+1 )));
	
    buffer = val;
    if ( val > max ) max = val;
    if ( val < min ) min = val;
    val = MAX_BRIGHTNESS * (buffer - min) / (max-min);
    cuda_image_set_pixel( out, x, y, val );
}
