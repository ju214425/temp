#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define ERROR_CHECK_RATE 1.0e-5f
#define EPSILON 1.0e-5

typedef struct _tensor
{
	cl_float *data;     //c * h * w
	cl_int c,h,w;

}tensor;

typedef struct _conv2d_filter
{
	cl_float *window;   //out_channels * in_channels * filter_size * filter*size
	cl_float *bias; // out_chaneels
	cl_int size,input_ch,output_ch;

}conv2d_filter;


