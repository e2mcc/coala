#ifndef _COALA_REGRESSION_NN_HYPOTHESIS
#define _COALA_REGRESSION_NN_HYPOTHESIS

#include "coala_rgsnn_structure.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

COALA_RGSNN_HYPOTHESIS_t * defineHypothesis
(
	unsigned int const degree
);


double computeHypothesis
(	
	COALA_RGSNN_HYPOTHESIS_t * const f,
	double const x
);

#endif




