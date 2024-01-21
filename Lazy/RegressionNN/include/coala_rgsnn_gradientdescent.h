#ifndef _COALA_REGRESSION_NN_GRADIENTDESCENT_H
#define _COALA_REGRESSION_NN_GRADIENTDESCENT_H

#include "coala_rgsnn_structure.h"
#include "coala_rgsnn_hypothesis.h"
#include "coala_rgsnn_loss.h"
#include <stdio.h>
#include <string.h>

void gradient_descent
(
	COALA_RGSNN_TRANING_DATA_t * const data_training,
	COALA_RGSNN_HYPOTHESIS_t * const hypothesis,
	double  const learning_rate,
	unsigned int const iterations
);




#endif