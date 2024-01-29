#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

// Includes the OpenBlas library (C interface)
#include <cblas.h>

// =================================================================================================


float * matGen(int const m, int const n, float const val)
{	
	float * mat = (float*)malloc(sizeof(float)*m*n);
	for (size_t i=0; i<m*n; ++i) { mat[i] = val; }
	return mat;
}





// Example use of the single-precision routine SGEMM
int main(int argc,char **argv) 
{	
	//初始化
	// Example SGEMM arguments
	// const size_t m = atoi(argv[1]);
	// const size_t n = atoi(argv[2]);
	// const size_t k = atoi(argv[3]);
	FILE * fp = fopen("./test_data.csv","w");
	
	//1000GFLOPS
	double const peak = CPU_FP32_PEAK;
	if(peak==0||peak<0)
	{
		fprintf(stderr,"peak's settings are wrong\n");
		return -1;
	}

	struct timeval start, end;
	

	unsigned int const steps = 100;
	unsigned int const iterations = 1;
	size_t m = 0;
	size_t n = 0;
	size_t k = 0;
	float const alpha = 0.7f;
	float const beta = 1.0f;

	fprintf(fp,"m,n,k,dimension,compute-time(s),GFlops,UR\n");
	fflush(fp);

	for(int i=0;i<steps;++i)
	{
		
		m = (rand() % 9901)+100;
		n = (rand() % 9901)+100;
		k = (rand() % 9901)+100;

		size_t a_ld = k;
		size_t b_ld = n;
		size_t c_ld = n;

		float * host_a = matGen(m,k,12.193f);
		float * host_b = matGen(k,n,-8.199f);
		float * host_c = matGen(m,n,0.0f);

		gettimeofday(&start,NULL);
		for(unsigned int i=0; i<iterations; i++)
		{
			// 计算部分 Call the SGEMM routine.
			cblas_sgemm(CblasRowMajor,
						CblasNoTrans, CblasNoTrans,
						m, n, k,
						alpha,
						host_a, a_ld,
						host_b, b_ld,
						beta,
						host_c, c_ld);
		}
		gettimeofday(&end,NULL);
		// Clean-up
		free(host_a);
		free(host_b);
		// printf("hostc[1]=%lf\n",host_c[1]);
		free(host_c);

		double compute_time = ((end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec)/1000000.0)/iterations;
		double GFlops = 2*m*n*k/compute_time/1000000000;
		double UR = GFlops/peak;
		fprintf(fp,"%zu,%zu,%zu,%zu,%lf,%lf,%lf\n",m,n,k,m*n*k,compute_time,GFlops,UR);
		fflush(fp);

	}
	fclose(fp);
  	return 0;
}

// =================================================================================================
