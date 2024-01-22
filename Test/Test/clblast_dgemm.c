#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings

// Includes the CLBlast library (C interface)
#include <clblast_c.h>

// =================================================================================================


int main(int argc,char **argv) 
{	
	const size_t m = atoi(argv[1]);
	const size_t n = m;
	const size_t k = m;
	const double alpha = 0.7f;
	const double beta = 1.0f;
	const size_t a_ld = k;
	const size_t b_ld = n;
	const size_t c_ld = n;

	double* host_a = (double*)malloc(sizeof(double)*m*k);
	double* host_b = (double*)malloc(sizeof(double)*n*k);
	double* host_c = (double*)malloc(sizeof(double)*m*n);
	for (size_t i=0; i<m*k; ++i) { host_a[i] = 12.193f; }
	for (size_t i=0; i<n*k; ++i) { host_b[i] = -8.199f; }
	for (size_t i=0; i<m*n; ++i) { host_c[i] = 0.0f; }


	struct timeval start, end1, end2, end3, end4;
	gettimeofday(&start,NULL);


  	const size_t platform_id = 0;
  	const size_t device_id = 0;


	cl_uint num_platforms;
	clGetPlatformIDs(0, NULL, &num_platforms);
	cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms*sizeof(cl_platform_id));
	clGetPlatformIDs(num_platforms, platforms, NULL);
	cl_platform_id platform = platforms[platform_id];

	cl_uint num_devices;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	cl_device_id* devices = (cl_device_id*)malloc(num_devices*sizeof(cl_device_id));
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	cl_device_id device = devices[device_id];

	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

	cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);


	cl_event event = NULL;

	cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, m*k*sizeof(double), NULL, NULL);
	cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, n*k*sizeof(double), NULL, NULL);
	cl_mem device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, m*n*sizeof(double), NULL, NULL);
	clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, m*k*sizeof(double), host_a, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, n*k*sizeof(double), host_b, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, m*n*sizeof(double), host_c, 0, NULL, NULL);


	CLBlastStatusCode status;
	gettimeofday(&end1,NULL);

	status = CLBlastDgemm(CLBlastLayoutRowMajor,
						CLBlastTransposeNo, CLBlastTransposeNo,
						m, n, k,
						alpha,
						device_a, 0, a_ld,
						device_b, 0, b_ld,
						beta,
						device_c, 0, c_ld,
						&queue, &event);
		

	gettimeofday(&end2,NULL);
	

	if (status == CLBlastSuccess) {
		clWaitForEvents(1, &event);
		clReleaseEvent(event);
	}

	clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, m*n*sizeof(double), host_c, 0, NULL, NULL);


    clReleaseMemObject(device_a);
	clReleaseMemObject(device_b);
	clReleaseMemObject(device_c);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	free(platforms);
	free(devices);

    gettimeofday(&end3,NULL);

	free(host_a);
	free(host_b);
	free(host_c);

	gettimeofday(&end4,NULL);
	double data_migration_time1		=	(end1.tv_sec-start.tv_sec)+(end1.tv_usec-start.tv_usec)/1000000.0;
	double compute_time 			=	(end2.tv_sec-end1.tv_sec)+(end2.tv_usec-end1.tv_usec)/1000000.0;
	double data_migration_time2 	=	(end3.tv_sec-end2.tv_sec)+(end3.tv_usec-end2.tv_usec)/1000000.0;
	double release_time 			=	(end4.tv_sec-end3.tv_sec)+(end4.tv_usec-end3.tv_usec)/1000000.0;
	double total_time				=	(end4.tv_sec-start.tv_sec)+(end4.tv_usec-start.tv_usec)/1000000.0;

	printf("m=%zu GFLOPS = %lf\n",(double)2*m*n*k/(compute_time+data_migration_time1+data_migration_time2)/1000000000);

    return 0;
}

// =================================================================================================
