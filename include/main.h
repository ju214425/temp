#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <CL/cl.h>
#include "types.h"

// ANSI TYPE COLOR DEFINE //

#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_BLACK "\xlb[30m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_BLUE "\x1b[34m"
#define ANSI_COLOR_RESET "\x1b[0m"

//////////// Error Message ///////////

#define CHECK_ERROR(err) \
	if ( err != CL_SUCCESS ) { \
		printf(ANSI_COLOR_RED "[ERR][%s:%d]\t\tOpenCL error %d\n" ANSI_COLOR_RESET, __FILE__, __LINE__, err); \
		exit(EXIT_FAILURE); \
	}

//////////////////////////////////////

/////////// message Part /////////////

#define INITMSG(msg) \
	printf(ANSI_COLOR_GREEN \
		   "[INIT]\t\t%s\n" \
		   ANSI_COLOR_RESET, msg);

#define NOTICE(msg) \
	printf(ANSI_COLOR_BLUE \
		   "[NOTICE]\t%s\n" \
		   ANSI_COLOR_RESET, msg);

//////////////////////////////////////

///////// Constant Variable //////////

extern cl_platform_id platform;
extern cl_device_id device;
extern cl_context context;
extern cl_command_queue queue;
extern cl_program program;

extern double layer1_0_weight[144];
extern double layer1_0_bias[16];
extern double layer1_1_weight[16];
extern double layer1_1_bias[16];
extern double layer2_0_weight[4608];
extern double layer2_0_bias[32];
extern double layer2_1_weight[32];
extern double layer2_1_bias[32];
extern double layer3_0_weight[18432];
extern double layer3_0_bias[64];
extern double layer3_1_weight[64];
extern double layer3_1_bias[64];
extern double layer4_0_weight[73728];
extern double layer4_0_bias[128];
extern double layer4_1_weight[128];
extern double layer4_1_bias[128];
extern double layer5_0_weight[294912];
extern double layer5_0_bias[256];
extern double layer5_1_weight[256];
extern double layer5_1_bias[256];
extern double layer6_0_weight[207360];
extern double layer6_0_bias[90];
extern double layer6_1_weight[90];
extern double layer6_1_bias[90];
extern double layer1_1_running_mean[16];
extern double layer1_1_running_var[16];
extern double layer2_1_running_mean[32];
extern double layer2_1_running_var[32];
extern double layer3_1_running_mean[64];
extern double layer3_1_running_var[64];
extern double layer4_1_running_mean[128];
extern double layer4_1_running_var[128];
extern double layer5_1_running_mean[256];
extern double layer5_1_running_var[256];
extern double layer6_1_running_mean[90];
extern double layer6_1_running_var[90];

extern const char *class_names[90];
//////////////////////////////////////

/////////// Function List ////////////

// layer.c
tensor* get_tensor( int h, int w, int c );
tensor* init_tensor( int h, int w, int c );
conv2d_filter* get_filter(int filter_size, int input_ch, int output_ch);
conv2d_filter* load_filter(int filter_size, int input_ch, int output_ch, double *load_filter, double *load_bias);
tensor * convolution_2d(tensor* in_fmap, conv2d_filter* filter,int padding ,int stride);
void relu(tensor* in_fmap);
tensor *max_pool(tensor* in_fmap, int filter_size, int stride);
tensor *max_pool_gpu(tensor* in_fmap, int filter_size, int stride);
tensor *avgPoolToSingleLayer(tensor *in_fmap);
void tensor_details(tensor * fmap, const char * str);
tensor * convolution_2d_gpu(tensor* in_fmap, conv2d_filter* filter,int padding ,int stride);
void  verify_conv2d(tensor *out1, tensor *out2);
tensor *fully_connected(tensor *in_fmap, int layer_size);
tensor *fully_connected_gpu(tensor *in_fmap, int layer_size);

double get_mean(tensor *in_fmap);
double get_variance(tensor *in_fmap, int mean);
tensor *normalization(tensor *in_fmap, tensor *beta, tensor *gamma);
tensor *load_normalized(tensor *in_fmap, double *beta, double *gamma, double *mean, double *variance);

// model.c
tensor* forward(tensor * in);

// gpu.c
char *get_source_code(const char *file_name, size_t *len);
void gpu_connect(void);

// util.c
double get_time();
tensor *loadImage(const char *filename, const char *mode);
void show_result(tensor *result);

////////////////////////////////////////





