#include "include/main.h"


tensor *convolution_layer(tensor *in, conv2d_filter *layer, double *beta, double *gamma, double *mean, double *variance)
{
	int pool_size = 2;
	tensor *out = convolution_2d_gpu(in, layer, 1, 1);
	tensor *out_cpu = convolution_2d(in, layer, 1, 1);
	verify_conv2d(out, out_cpu);
	out = load_normalized(out, beta, gamma, mean, variance);
	relu(out);
	out = max_pool_gpu(out, pool_size, 2);

	return out;
}



tensor* forward(tensor * in)
{
	conv2d_filter *layer1 = load_filter(3, 1, 16, layer1_0_weight, layer1_0_bias);
	conv2d_filter *layer2 = load_filter(3, 16, 32, layer2_0_weight, layer2_0_bias);
	conv2d_filter *layer3 = load_filter(3, 32, 64, layer3_0_weight, layer3_0_bias);
	conv2d_filter *layer4 = load_filter(3, 64, 128, layer4_0_weight, layer4_0_bias);
	conv2d_filter *layer5 = load_filter(3, 128, 256, layer5_0_weight, layer5_0_bias);
	conv2d_filter *layer6 = load_filter(3, 256, 90, layer6_0_weight, layer6_0_bias);
	double start_time, end_time, total_time;

	start_time = get_time();

	// LAYER 1 CONVOLUTION, ReLU
	tensor_details(in,"input");
	tensor* out1 = convolution_layer(in, layer1, layer1_1_weight, layer1_1_bias, layer1_1_running_mean, layer1_1_running_var);
	tensor_details(out1, "layer1 out");
	free(layer1);

	// LAYER2 CONVOLUTION, ReLU
	tensor* out2 = convolution_layer(out1, layer2, layer2_1_weight, layer2_1_bias, layer2_1_running_mean, layer2_1_running_var);
	tensor_details(out2, "layer2 out");
	free(layer2);
	
	// LAYER3 CONVOLUTION, ReLU
	tensor* out3 = convolution_layer(out2, layer3, layer3_1_weight, layer3_1_bias, layer3_1_running_mean, layer3_1_running_var);
	tensor_details(out3, "layer3 out");
	free(layer3);

	// LAYER4 CONVOLUTION, ReLU
	tensor* out4 = convolution_layer(out3, layer4, layer4_1_weight, layer4_1_bias, layer4_1_running_mean, layer4_1_running_var);
	tensor_details(out4, "layer4 out");
	free(layer4);

	// LAYER5 CONVOLUTION, ReLU
	tensor *out5 = convolution_layer(out4, layer5, layer5_1_weight, layer5_1_bias, layer5_1_running_mean, layer5_1_running_var);
	tensor_details(out5, "layer5 out");
	free(layer5);

	// LAYER6 fully connected
	tensor *out6 = convolution_2d_gpu(out5, layer6, 1, 1);
	tensor *out6_cpu = convolution_2d(out5, layer6, 1, 1);
	verify_conv2d(out6, out6_cpu);
	out6 = load_normalized(out6, layer6_1_weight, layer6_1_bias, layer6_1_running_mean, layer6_1_running_var);
	relu(out6);
	out6 = avgPoolToSingleLayer(out6); 
	tensor_details(out6, "classification layer out");

	end_time = get_time();
	total_time = end_time - start_time;

	printf("total processing time : %f sec\n", total_time);

	free(out5);
	free(out4);
	free(out3);
	free(out2);
	free(out1);

	return out6;
}
