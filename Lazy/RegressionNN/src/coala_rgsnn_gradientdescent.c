#include "coala_rgsnn_gradientdescent.h"

#ifdef _OPENMP
#	include <omp.h>
#else
#	pragma message "Compiler did NOT support OPENMP"
#endif

//梯度下降
void gradient_descent
(
	COALA_RGSNN_TRANING_DATA_t * const data_training,
	COALA_RGSNN_HYPOTHESIS_t * const hypothesis,
	double  const learning_rate,
	unsigned int const iterations
)
{	
	//读取
	unsigned int const degree =  hypothesis->degree;
	unsigned int const m = data_training->shape;
	

	double * gradient_params = malloc(degree * sizeof(*gradient_params));
	memset(gradient_params,0,degree * sizeof(*gradient_params));

	//训练（迭代n次）
	for (unsigned int i = 0; i < iterations; i++)
	{
		// for(unsigned int j=0; j<m; j++)
		// {	
		// 	double prediction = computeHypothesis(hypothesis,data_training->points[j].feature);
		// 	for(unsigned int k=0; k<=degree; k++)
		// 	{
		// 		gradient_params[k] += (prediction - data_training->points[j].target) * pow(data_training->points[j].feature,k);
		// 	}
		// }
		
		#pragma omp parallel  for num_threads(degree+1)
		for(unsigned int j=0; j<=degree; j++)
		{	
			double temp_sum = 0;
			for(unsigned int k=0; k<m; k++)
			{
				double prediction = computeHypothesis(hypothesis,data_training->points[k].feature);
				temp_sum += (prediction - data_training->points[k].target) * pow(data_training->points[k].feature,j);
			}
			gradient_params[j] = temp_sum/m;
		}

		//更新Hypothesis函数参数（更新权重）
		#pragma omp parallel  for num_threads(degree+1)
		for(unsigned int i = 0; i<=degree; i++)
		{
			hypothesis->params[i] = hypothesis->params[i] - learning_rate * gradient_params[i];
		}

		// //输出每500步的loss
		// if(i%2000==0)
		// {
		// 	printf("iteration [%d/%d], loss = %lf\n",i,iterations,cost_function(data_training,hypothesis));
		// }
		
	}

	free(gradient_params);
}