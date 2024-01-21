#include "coala_rgsnn_loss.h"



//损失函数J(x,y)=MSE=(E(y_pred-y)^2)/2m
double cost_function
(
	COALA_RGSNN_TRANING_DATA_t * const data_training,
	COALA_RGSNN_HYPOTHESIS_t * const hypothesis
)
{
	double sum = 0;
	unsigned int const m = data_training->shape;
	for (unsigned int i = 0; i < m; i++)
	{
		double prediction = computeHypothesis(hypothesis,data_training->points[i].feature);
		sum += pow((prediction - data_training->points[i].target),2);
	}
	return sum / (2 * m);
	
}


//决定系数（R-Square)
double RSquare
(
	COALA_RGSNN_TRANING_DATA_t * const data_training,
	COALA_RGSNN_HYPOTHESIS_t * const hypothesis
)
{	
	double numerator = 0.0;
	unsigned int const m = data_training->shape;
	for(unsigned int i = 0; i < m; i++)
	{
		double prediction = computeHypothesis(hypothesis,data_training->points[i].feature);
		numerator += pow((prediction - data_training->points[i].target),2);
	}

	double mean_val = 0.0;
	double temp_sum = 0.0;
	for(unsigned int i = 0; i < m; i++)
	{
		temp_sum += data_training->points[i].target;
	}
	mean_val = temp_sum/m;

	double denominator = 0.0;
	for(unsigned int i = 0; i < m; i++)
	{
		denominator += pow((mean_val - data_training->points[i].target),2);
	}

	double R_Square = 1-(numerator/denominator);

	return R_Square;

}


//校正决定系数（Adjusted R-Square)
double AdjustedRSquare
(
	COALA_RGSNN_TRANING_DATA_t * const data_training,
	COALA_RGSNN_HYPOTHESIS_t * const hypothesis
)
{
	unsigned int const m = data_training->shape;
	double numerator = (1-RSquare(data_training,hypothesis))*(m-1);
	double denominator = m-1-1;
	double ARS = 1-(numerator/denominator);
	return ARS;
}