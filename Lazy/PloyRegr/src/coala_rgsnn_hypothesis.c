#include "coala_rgsnn_hypothesis.h"

#ifdef _OPENMP
#	include <omp.h>
#else
#	pragma message "Compiler did NOT support OPENMP"
#endif


COALA_RGSNN_HYPOTHESIS_t * defineHypothesis
(
	unsigned int const degree
)
{	
	if(degree>9)
	{
		fprintf(stderr,"WRONG! 函数 %s : degree不可以超过9",__FUNCTION__);
		exit(0);
	}

	COALA_RGSNN_HYPOTHESIS_t * f = malloc(sizeof(*f));
	f->degree = degree;
	f->params = malloc((degree+1)*sizeof(*f->params));
	
	//初始化hypothesis参数
	for(unsigned int i = 0; i<=degree; i++)
	{
		f->params[i] = ((double)(rand() % 10))/10;
	}
	
	f->params[degree] = -5;
	
	return f;
}


//计算假设函数f(x)的值
double computeHypothesis
(	
	COALA_RGSNN_HYPOTHESIS_t * const f,
	double const x
) 
{	
	double y = 0;
	unsigned int const degree = f->degree;

	#pragma omp parallel for num_threads(degree+1) reduction(+:y)
	for(unsigned int i=0;i<=degree;i++)
	{
		float powx = pow(x,i);
		y += powx*f->params[i];
	}
	return y;
}