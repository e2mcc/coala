
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

// Includes the CLBlast library (C interface)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// =================================================================================================


int main(int argc,char **argv) 
{	

	const size_t m = atoi(argv[1]);
	const size_t n = m;
	const size_t k = m;
	const float alpha = 0.7f;
	const float beta = 1.0f;
	const size_t a_ld = k;
	const size_t b_ld = n;
	const size_t c_ld = n;


	float* host_a = (float*)malloc(sizeof(float)*m*k);
	float* host_b = (float*)malloc(sizeof(float)*n*k);
	float* host_c = (float*)malloc(sizeof(float)*m*n);
	for (size_t i=0; i<m*k; ++i) { host_a[i] = 12.193f; }
	for (size_t i=0; i<n*k; ++i) { host_b[i] = -8.199f; }
	for (size_t i=0; i<m*n; ++i) { host_c[i] = 0.0f; }



	struct timeval start, end1, end2, end3, end4;
	gettimeofday(&start,NULL);
	

	cublasHandle_t handle;

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        return EXIT_FAILURE;
    }

	float *device_a,*device_b,*device_c;   
	cudaMalloc((void**)&device_a,sizeof(float)*m*k);
  	cudaMalloc((void**)&device_b,sizeof(float)*k*n);
  	cudaMalloc((void**)&device_c,sizeof(float)*m*n);

	cudaMemcpy(device_a,host_a,sizeof(float)*m*k,cudaMemcpyHostToDevice);
  	cudaMemcpy(device_b,host_b,sizeof(float)*k*n,cudaMemcpyHostToDevice);
	cudaMemcpy(device_c,host_c,sizeof(float)*m*n,cudaMemcpyHostToDevice);
	
	gettimeofday(&end1,NULL);
	
	cublasSgemm(
		  handle,
          CUBLAS_OP_N,   
          CUBLAS_OP_N,  
          m,        
          n,       
          k,         
          &alpha,             
          device_a,           
          a_ld,         
          device_b,          
          b_ld,          
          &beta,            
          device_c,           
          c_ld        
  	);
	cudaDeviceSynchronize();

	gettimeofday(&end2,NULL);
	
	
	cudaMemcpy(host_c,device_c,sizeof(float)*m*n,cudaMemcpyDeviceToHost);

	cublasDestroy(handle);
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);

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
