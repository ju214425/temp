
__kernel 
void conv_layer(__global double *input, __global double *filter, 
				__global double *output, __global double *bias,
				int filter_size, int input_ch,
				int out_h, int out_w, int stride, int padding) {
	int c_it = get_global_id(2); // output channel
	int h_it = get_global_id(1) *stride; // output height
	int w_it = get_global_id(0) *stride;// output width
	int i, j, k;
	
	double dot = 0.0;
	int h_pos = h_it / stride + padding;
	int w_pos = w_it / stride + padding;
	if( h_it < out_h - padding && w_it < out_w - padding){
		//printf("%d, %d, %d\n", c_it, h_it, w_it);
		for (i = 0; i < input_ch; i++){
			for (j = 0; j < filter_size; j++){
				for (k = 0; k < filter_size; k++){
					int idx = c_it *input_ch *filter_size *filter_size + i *filter_size *filter_size + j *filter_size + k;
					int idy = i *filter_size *filter_size + (j+h_it) *filter_size + (k+w_it);
					dot += filter[idx] *input[idy];
					//printf("[%d:%d] %f\n", idx, idy, dot);
				}
			}
		}
		dot += bias[c_it];
		int idz = c_it *out_h *out_w + h_pos *out_w + w_pos;
		//printf("[%d] %f\n", idz, dot);
		output[idz] = dot;
	}
}


__kernel
void max_pool(__global double *input, __global double *output,
			  int filter_size, int channel,
			  int out_h, int out_w, int stride)
{
	int c_it = get_global_id(2);
	int h_it = get_global_id(1) *stride;
	int w_it = get_global_id(0) *stride;
	int i, j, k;
	
	int h_pos = get_global_id(1);
	int w_pos = get_global_id(0);
	if(h_it < out_h && w_it < out_w){
		for(i = 0; i < channel; i++)
		{
			double max_value = input[i *filter_size *filter_size];
			for(j = 0; j < filter_size; j++){
				for(k = 0; k < filter_size; k++){
					int idx = i *filter_size *filter_size + (j+h_it) *filter_size + (k+w_it);
					if(max_value <= input[idx])
					{
						max_value = input[idx];
					}
				}
			}
			int idy = c_it *out_h *out_w + h_pos *out_w + w_pos;
			output[idy] = max_value;
		}
	}
}


__kernel
void fully_connect(__global double *input, __global double *weight,
				   __global double *output, int width, int height, int channel)
{
	int ch = get_global_id(0);
	int i, j, k;

	double dot = 0.0;
	for(i = 0; i < channel; i++)
	{
		for(j = 0; j < height; j++)
		{
			for(k = 0; k < width; k++)
			{
				int idx = i *height *width + j *width + k;
				dot += input[idx] *weight[ch];
			}
		}
	}
	output[ch] = dot;
}



