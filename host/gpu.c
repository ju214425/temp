#include "./include/main.h"

void gpu_connect(void){
	char *kernel_source;
	size_t kernel_source_size;
	size_t header_size = 0, kernel_size = 0;
	cl_int err;

	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err)

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	queue = clCreateCommandQueue(context, device, 0, &err);
	CHECK_ERROR(err);

	kernel_source = get_source_code("device/kernel.cl", &kernel_source_size);
	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
	// CHECK kernel initial program build OK.
	if( err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char *log;
		
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		CHECK_ERROR(err);

		log = (char *)malloc(log_size + 1);
		memset(log, 0, log_size+1);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);	
		CHECK_ERROR(err);

		log[log_size] = '\0';
		printf("[LOG]\t\tCompiler error: \n%s\n", log);
		free(log);
		exit(0);
	}
	CHECK_ERROR(err);
}


char *get_source_code(const char *file_name, size_t *len){
	char *source_code;
	size_t length;
	FILE *file = fopen(file_name, "rb");
	
	if(file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}
	
	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);
	
	source_code = (char *)malloc(length +1);
	fread(source_code, length, 1, file);
	source_code[length] = '\0';

	fclose(file);
	
	*len = length;
	return source_code;
}


