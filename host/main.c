#include "include/main.h"

int main(void)
{
	// open image
	tensor *imgBuf1 = loadImage("image/image_1.bmp", "r");
	tensor *imgBuf2 = loadImage("image/image_2.bmp", "r");
	tensor *imgBuf3 = loadImage("image/image_3.bmp", "r");
	tensor *imgBuf4 = loadImage("image/image_4.bmp", "r");
	gpu_connect();
	tensor * output1 = forward(imgBuf1);	
	show_result(output1);
	tensor * output2 = forward(imgBuf2);	
	show_result(output2);
	tensor * output3 = forward(imgBuf3);	
	show_result(output3);
	tensor * output4 = forward(imgBuf4);	
	show_result(output4);

	

	free(output4);
	free(output3);
	free(output2);
	free(output1);
	free(imgBuf4);
	free(imgBuf3);
	free(imgBuf2);
	free(imgBuf1);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	return EXIT_SUCCESS;
}
