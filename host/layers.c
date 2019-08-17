#include "include/main.h"

#define leakyRelu(x,y) ((x>y)?x:x*0.1)
#define max(x,y) ((x>y)?x:y)

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;

//////////////////////////////
// Setting Data/tensor Part //
//////////////////////////////

tensor* get_tensor(int h, int w, int c)
{
	tensor * tmp = (tensor *) malloc(sizeof(tensor));
	tmp->h = h;
	tmp->w = w;
	tmp->c = c;

	tmp->data = (cl_double *)malloc(sizeof(cl_double) *c *h *w);
	for(int i = 0; i < c; i++)
	{
		for(int j = 0; j < h; j++)
		{
			for(int k = 0; k < w; k++){
				int idx = i *h *w + j *w + k;
				tmp->data[idx]=(cl_double)(0.00050 - (rand()%100)/100000.0);
			}
		}

	}

	return tmp;
}

tensor* init_tensor(int h, int w, int c)
{
	tensor *tmp = (tensor *)malloc(sizeof(tensor));
	tmp->h = h;
	tmp->w = w;
	tmp->c = c;

	tmp->data =  (cl_double *)malloc(sizeof(cl_double) *c *h *w);
	for(int i = 0; i < c; i++)
	{
		for(int j = 0; j < h; j++)
		{
			for(int k = 0; k < w; k++){
				int idx = i *h *w + j *w + k;
				tmp->data[idx]=(0.0);
			}
		}

	}

	return tmp;
}



conv2d_filter* get_filter(int size, int input_ch, int output_ch)
{
	conv2d_filter* tmp = (conv2d_filter *)malloc(sizeof(conv2d_filter));
	tmp->size = size;
	tmp->input_ch = input_ch;
	tmp->output_ch = output_ch;

	int window_size = sizeof(cl_double) *output_ch *input_ch *size *size;
	tmp->window = (cl_double *)malloc(window_size);
	tmp->bias = (cl_double *)malloc(sizeof(cl_double) *output_ch);
	for(int i = 0;i < output_ch; i++)
	{
		for(int j = 0; j < input_ch; j++)
		{
			for(int k = 0; k < size; k++)
			{
				for(int l = 0; l < size; l++)
				{
					int idx = i * input_ch *size *size+ j *size *size + k *size + l;
					tmp->window[idx] = (cl_double)(0.00050 - (rand()%100)/100000.0);
				}
			}
		}
		tmp->bias[i] = (cl_double) (0.000050 - (rand()%10)/100000.0);
	}


	return tmp;

}

conv2d_filter* load_filter(int size, int input_ch, int output_ch, double *load_filter, double *load_bias)
{
	conv2d_filter* tmp = (conv2d_filter *)malloc(sizeof(conv2d_filter));
	tmp->size = size;
	tmp->input_ch = input_ch;
	tmp->output_ch = output_ch;

	int window_size = sizeof(cl_double) *output_ch *input_ch *size *size;
	tmp->window = (cl_double *)malloc(window_size);
	tmp->bias = (cl_double *)malloc(sizeof(cl_double) *output_ch);
	for(int i = 0;i < output_ch; i++)
	{
		for(int j = 0; j < input_ch; j++)
		{
			for(int k = 0; k < size; k++)
			{
				for(int l = 0; l < size; l++)
				{
					int idx = i * input_ch *size *size+ j *size *size + k *size + l;
					tmp->window[idx] = (cl_double)(load_filter[idx]);
				}
			}
		}
		tmp->bias[i] = (cl_double) (load_bias[i]);
	}


	return tmp;
}


/////////////////////////////////
// Convolution inmplementation //
/////////////////////////////////

tensor * convolution_2d(tensor* in_fmap, conv2d_filter* filter, int padding, int stride)
{
	int out_h = (in_fmap->h - filter->size + 2*padding)/stride + 1;
	int out_w = (in_fmap->w - filter->size + 2*padding)/stride + 1;
	tensor * out_fmap = init_tensor(out_h, out_w, filter->output_ch);
	for(int c_it=0; c_it < filter->output_ch; c_it++)
	{
		int h_pos = padding;
		for(int h_it = 0; h_it < out_h - padding; h_it += stride)
		{
			int w_pos = padding;
			for(int w_it = 0; w_it < out_w - padding; w_it += stride)
			{
				double dot=0.0;
				for(int i = 0; i < filter->input_ch; i++)
				{
					for(int j = 0; j < filter->size; j++)
					{
						for(int k = 0; k < filter->size; k++)
						{
							int idx = c_it *(filter->input_ch) *(filter->size) *(filter->size) + i *(filter->size) *(filter->size) + j *(filter->size) + k;
							int idy = i *filter->size *filter->size + (j+h_it)*filter->size + (k+w_it);
							dot += filter->window[idx]*in_fmap->data[idy];
						}
					}
				}
				dot += filter->bias[c_it];
				int idz = c_it *out_h *out_w + h_pos *out_w + w_pos;
				out_fmap->data[idz] = dot;
				w_pos++;
			}
			h_pos++;
		}
	}

	return out_fmap;
}


tensor * convolution_2d_gpu(tensor* in_fmap, conv2d_filter* filter,int padding ,int stride)
{
	cl_kernel kernel;
	cl_int err;
	int out_h = (in_fmap->h - filter->size + 2*padding)/stride + 1;
	int out_w = (in_fmap->w - filter->size + 2*padding)/stride + 1;
	tensor *out_fmap = init_tensor(out_h, out_w, filter->output_ch);
	
	kernel = clCreateKernel(program, "conv_layer", &err);
	CHECK_ERROR(err);
	
	cl_mem buf_in, buf_filter, buf_bias, buf_out;
	int input_size = sizeof(cl_double) *in_fmap->c *in_fmap->h *in_fmap->w;
	buf_in = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size, NULL, &err);
	CHECK_ERROR(err);

	int filter_size = sizeof(cl_double) *filter->output_ch *filter->input_ch *filter->size *filter->size;	
	buf_filter = clCreateBuffer(context, CL_MEM_READ_ONLY, filter_size, NULL, &err);
	CHECK_ERROR(err);
	
	int bias_size = sizeof(cl_double) *filter->output_ch;
	buf_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, bias_size, NULL, &err);

	int output_size = sizeof(cl_double) *out_fmap->c *out_fmap->h *out_fmap->w;
	buf_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size, NULL, &err);
	CHECK_ERROR(err);

	err	 = clEnqueueWriteBuffer(queue, buf_in, CL_FALSE, 0, input_size, in_fmap->data, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, buf_filter, CL_FALSE, 0, filter_size, filter->window, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, buf_bias, CL_FALSE, 0, bias_size, filter->bias, 0, NULL, NULL);
	CHECK_ERROR(err);
	
	int size = filter->size;
	int input_ch = filter->input_ch;
	int output_ch = filter->output_ch;

	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_filter);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_out);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_bias);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_int), &size);
	err |= clSetKernelArg(kernel, 5, sizeof(cl_int), &input_ch);
	err |= clSetKernelArg(kernel, 6, sizeof(cl_int), &out_h);
	err |= clSetKernelArg(kernel, 7, sizeof(cl_int), &out_w);
	err |= clSetKernelArg(kernel, 8, sizeof(cl_int), &stride);
	err |= clSetKernelArg(kernel, 9, sizeof(cl_int), &padding);
	CHECK_ERROR(err);

	size_t global_size[3] = { out_w, out_h, output_ch };
	size_t local_size[3] = { 1 , 1 , 1 };
	
	err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size, local_size, 0, NULL, NULL);
	CHECK_ERROR(err);
	
	err = clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, output_size, out_fmap->data, 0, NULL, NULL);
	CHECK_ERROR(err);

	clReleaseMemObject(buf_in);
	clReleaseMemObject(buf_filter);
	clReleaseMemObject(buf_out);
	clReleaseKernel(kernel);

	return out_fmap;
}

// Verify CPU & GPU processing CONV2D data
void verify_conv2d(tensor *out1, tensor *out2)
{
	int errors = 0;
	for(int i=0;i<out1->c;i++){
		for(int j=0;j<out1->h;j++){
			for(int k=0;k<out1->w;k++){
				int idx1 = i *out1->h *out1->w + j *out1->w + k;
				int idx2 = i *out2->h *out2->w + j *out2->w + k;
				double result = out1->data[idx1] - out2->data[idx2];
				if(fabsf(result) > ERROR_CHECK_RATE){
					errors++;
				}
			}
		}
	}
	if(errors != 0){
		printf("[WARN]\t\tError counts : %d/%d\n", errors, out2->c *out2->h *out2->w);
	}
}


void relu(tensor* in_fmap)
{
	for(int i=0;i<in_fmap->c;i++){
		for(int j=0;j<in_fmap->h;j++){
			for(int k=0;k<in_fmap->w;k++){
				int idx = i *in_fmap->h *in_fmap->w + j *in_fmap->w + k;
				in_fmap->data[idx] = leakyRelu( in_fmap->data[idx], 0.0 );
			}
		}
	}
}

/////////////////////////////////
// Pooling Data with max value //
/////////////////////////////////

tensor *max_pool(tensor* in_fmap, int filter_size, int stride)
{
	int out_h = (in_fmap->h - filter_size)/stride + 1;
	int out_w = (in_fmap->w - filter_size)/stride + 1;
	tensor *out_fmap = get_tensor(out_h, out_w, in_fmap->c);

	for(int c_it = 0; c_it < in_fmap->c; c_it++)
	{
		int h_pos = 0;
		for(int h_it=0; h_it < out_h; h_it += stride){
			int w_pos = 0;
			for(int w_it = 0; w_it < out_w; w_it += stride){
				for(int i = 0; i < in_fmap->c; i++)
				{
					cl_double max_value = 0.0;
					for(int j = 0; j < filter_size; j++)
					{
						for(int k = 0; k < filter_size; k++)
						{
							int idx = i *filter_size *filter_size + (j+h_it) *filter_size + (k+w_it);
							max_value = max(in_fmap->data[idx], max_value);
						}
					}
					int idy = c_it *out_h *out_w + h_pos *out_w + w_pos;
					out_fmap->data[idy] = max_value;
				}
				w_pos++;
			}
			h_pos++;
		}
	}

	return out_fmap;
}

tensor *max_pool_gpu(tensor* in_fmap, int filter_size, int stride)
{
	cl_kernel kernel;
	cl_int err;
	int out_h = (in_fmap->h - filter_size)/stride + 1;
	int out_w = (in_fmap->w - filter_size)/stride + 1;
	tensor *out_fmap = get_tensor(out_h, out_w, in_fmap->c);

	kernel = clCreateKernel(program, "max_pool", &err);
	CHECK_ERROR(err);

	cl_mem buf_in, buf_out;
	int input_size = sizeof(double) *in_fmap->c *in_fmap->h *in_fmap->w;
	buf_in = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size, NULL, &err);
	CHECK_ERROR(err);

	int output_size = sizeof(double) *out_fmap->c *out_fmap->h *out_fmap->w;
	buf_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size, NULL, &err);
	CHECK_ERROR(err);

	err	 = clEnqueueWriteBuffer(queue, buf_in, CL_FALSE, 0, input_size, in_fmap->data, 0, NULL, NULL);
	CHECK_ERROR(err);

	int channel = out_fmap->c;

	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
	err	|= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_int), &filter_size);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &channel);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_int), &out_h);
	err |= clSetKernelArg(kernel, 5, sizeof(cl_int), &out_w);
	err |= clSetKernelArg(kernel, 6, sizeof(cl_int), &stride);
	CHECK_ERROR(err);

	size_t global_size[3] = { out_w, out_h, channel };
	size_t local_size[3] = { 1, 1, 1 };
	
	err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size, local_size, 0, NULL, NULL);
	CHECK_ERROR(err);
		
	err = clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, output_size, out_fmap->data, 0, NULL, NULL);
	CHECK_ERROR(err);
	
	clReleaseMemObject(buf_in);
	clReleaseMemObject(buf_out);
	clReleaseKernel(kernel);
	return out_fmap;
}

tensor *avgPoolToSingleLayer(tensor *in_fmap)
{
	tensor *out_fmap = init_tensor(1, 1, in_fmap->c);
	int i, j, k;
	
	for(i = 0; i < in_fmap->c; i++)
	{
		double sum = 0.0;
		double avg = 0.0;
		for(j = 0; j < in_fmap->h; j++)
		{
			for(k = 0; k < in_fmap->w; k++)
			{
				int idx = i *(in_fmap->h) *(in_fmap->w) + j *(in_fmap->w) + k;
				sum += in_fmap->data[idx];
			}
		}
		avg = (cl_double)(sum / ((in_fmap->h) *(in_fmap->w)));
		out_fmap->data[i] = avg;
	}
	return out_fmap;
}

//////////////////////////
// Fully Connected Part //
//////////////////////////

tensor *fully_connected(tensor *in_fmap, int layer_size)
{
	tensor *out_fmap = get_tensor(1, 1, layer_size);
	tensor *weight = get_tensor(1, 1, layer_size);
	int ch, i, j, k;
	
	for(ch = 0; ch < layer_size; ch++){
		double dot = 0.0;
		for(i = 0; i < in_fmap->c; i++){
			for(j = 0; j < in_fmap->h; j++){
				for(k = 0; k < in_fmap->w; k++){
					int idx = i *(in_fmap->h) *(in_fmap->w) + j *(in_fmap->w) + k;
					dot += in_fmap->data[idx] * weight->data[ch];
				}
			}
		}
		out_fmap->data[ch] = dot;
	}
	return out_fmap;
}

tensor *fully_connected_gpu(tensor *in_fmap, int layer_size)
{
	cl_kernel kernel;
	cl_int err;
	tensor *out_fmap = get_tensor(1, 1, layer_size);
	tensor *weight = get_tensor(1, 1, layer_size);
	int ch, i, j, k;

	kernel = clCreateKernel(program, "fully_connect", &err);
	CHECK_ERROR(err);

	cl_mem buf_in, buf_weight, buf_out;
	int input_size = sizeof(double) *in_fmap->c *in_fmap->h *in_fmap->w;
	buf_in = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size, NULL, &err);
	CHECK_ERROR(err);

	int filter_size = sizeof(double) *layer_size;
	buf_weight = clCreateBuffer(context, CL_MEM_READ_ONLY, filter_size, NULL, &err);
	CHECK_ERROR(err);

	int output_size = sizeof(double) *layer_size;
	buf_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size, NULL, &err);
	CHECK_ERROR(err);
	
	err  = clEnqueueWriteBuffer(queue, buf_in, CL_FALSE, 0, input_size, in_fmap->data, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, buf_weight, CL_FALSE, 0, filter_size, weight->data, 0, NULL, NULL);
	CHECK_ERROR(err);

	int channel = in_fmap->c;
	int width = in_fmap->w;
	int height = in_fmap->h;

	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_weight);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_out);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &width);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_int), &height);
	err |= clSetKernelArg(kernel, 5, sizeof(cl_int), &channel);
	CHECK_ERROR(err);
	
	size_t global_size[3] = { width, height, channel };
	size_t local_size[3] = { 1 , 1 , 1 };
	
	err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size, local_size, 0, NULL, NULL);
	CHECK_ERROR(err);
	
	err = clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, output_size, out_fmap->data, 0, NULL, NULL);
	CHECK_ERROR(err);

	return out_fmap;
}

//////////////////////////////////
// Data Normalization Part with //
// mean, variance, beta, gamma  //
//////////////////////////////////

double get_mean(tensor *in_fmap)
{
	double sum = 0.0;
	double mean = 0.0;
	int i;

	int total_size = (in_fmap->c) *(in_fmap->h) *(in_fmap->w);
	for(i = 0; i < total_size; i++)
	{
		sum += in_fmap->data[i];
	}
	mean = (double)(sum / total_size);

	return mean;
}

double get_variance(tensor *in_fmap, int mean)
{
	double deviation = 0.0;
	double var = 0.0;
	int i, j, k;

	int total_size = (in_fmap->c) *(in_fmap->h) *(in_fmap->w);
	for(i = 0; i < total_size; i++){
		deviation += (in_fmap->data[i] - mean) *(in_fmap->data[i] - mean);
	}
	var = (double)(deviation / total_size);
	
	return var;
}

tensor *normalization(tensor *in_fmap, tensor *beta, tensor *gamma)
{
	tensor *out_fmap = get_tensor(in_fmap->h, in_fmap->w, in_fmap->c);
	double mean = get_mean(in_fmap);
	double variance = get_variance(in_fmap, mean);
	int i, j, k;
	
	for(i = 0; i < in_fmap->c; i++)
	{
		for(j = 0; j < in_fmap->h; j++)
		{
			for(k = 0; k < in_fmap->w; k++)
			{
				double dot = 0.0;
				int idx = i *(in_fmap->h) *(in_fmap->w) + j *(in_fmap->w) + k;
				dot = (gamma->data[i] * ((in_fmap->data[idx] - mean) / sqrt(variance + EPSILON) ) + beta->data[i]);
				out_fmap->data[idx] = dot;
			}
		}
	}
	return out_fmap;
}

tensor *load_normalized(tensor *in_fmap, double *beta, double *gamma, double *mean, double *variance)
{
	tensor *out_fmap = get_tensor(in_fmap->h, in_fmap->w, in_fmap->c);
	int i, j, k;
	
	for(i = 0; i < in_fmap->c; i++)
	{
		for(j = 0; j < in_fmap->h; j++)
		{
			for(k = 0; k < in_fmap->w; k++)
			{
				double dot = 0.0;
				int idx = i *(in_fmap->h) *(in_fmap->w) + j *(in_fmap->w) + k;
				dot = (gamma[i] * ((in_fmap->data[idx] - mean[i]) / sqrt(variance[i] + EPSILON) ) + beta[i]);
				out_fmap->data[idx] = dot;
			}
		}
	}
	return out_fmap;
}

//////////////////////////////////////////
// Show result of processed tensor Data //
//////////////////////////////////////////

void tensor_details(tensor * fmap, const char * str)
{
	printf("%s tensor (channels, height, width) = %d,%d,%d\n",str,fmap->c,fmap->h,fmap->w);
}
