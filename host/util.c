#include "include/main.h"

double get_time(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double) tv.tv_sec + (double)1e-6 *tv.tv_usec;	
}

tensor *loadImage(const char *filename, const char *mode)
{
	FILE *fimage = fopen(filename, mode);
	unsigned char header[60];
	unsigned char buf[4];
	tensor *imgBuf;
	int i, j;
	// Read image header
	fread(header, 54, 1, fimage);
	
	// width & height
	int width = *(int *)&header[18];
	int height = *(int *)&header[22];
	
	printf("[IMGINFO] width : %d, height : %d, channel : 3\n", width, height);
	// read all pixel
	imgBuf = (tensor *)malloc(sizeof(tensor));
	imgBuf->h = height;
	imgBuf->w = width;
	imgBuf->c = 1;
	imgBuf->data = (cl_double *)malloc(sizeof(cl_double) *height *width *1);

	//INITMSG("GET IMAGE RGB PIXEL VALUE");

	for( i =0; i < height; i++ )
	{
		for( j = 0; j < width; j++ )
		{				
			if( fread(buf, 4, 1, fimage) != 0 ) {
				int idx = i *width + j;
				double value = (((cl_double)buf[2]) + ((cl_double)buf[1]) + ((cl_double)buf[0]))/3.0;
				imgBuf->data[idx] = value;
			}
		}
	}

	return imgBuf;
}

void show_result(tensor *result)
{
	int i;
	double max = result->data[0];
	int max_index = 0;
	for(i = 0; i < result->c; i++){
		printf("%s\tprobability : %f\n", class_names[i], result->data[i]);
		if( max < result->data[i])
		{
			max = result->data[i];
			max_index = i;
		}
	}
	printf("[MAX](%d)\t%s : %f\n", max_index, class_names[max_index], max);
}

