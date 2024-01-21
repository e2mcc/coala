#ifndef _COALA_REGRESSION_NN_LOSS_H
#define _COALA_REGRESSION_NN_LOSS_H

#include "coala_rgsnn_structure.h"
#include "coala_rgsnn_hypothesis.h"

double cost_function
(
	COALA_RGSNN_TRANING_DATA_t * const data_training,
	COALA_RGSNN_HYPOTHESIS_t * const hypothesis
);

//决定系数（R-Square)
double RSquare
(
	COALA_RGSNN_TRANING_DATA_t * const data_training,
	COALA_RGSNN_HYPOTHESIS_t * const hypothesis
);

//校正决定系数（Adjusted R-Square)
double AdjustedRSquare
(
	COALA_RGSNN_TRANING_DATA_t * const data_training,
	COALA_RGSNN_HYPOTHESIS_t * const hypothesis
);
#endif