#ifndef _COALA_REGRESSION_NN_STRUCTURE_H
#define _COALA_REGRESSION_NN_STRUCTURE_H


typedef struct 
{
	double feature;
	double target;
}COALA_RGSNN_POINT_t;


typedef struct 
{	
	COALA_RGSNN_POINT_t * points;
	unsigned int shape; 
}COALA_RGSNN_TRANING_DATA_t;


typedef struct 
{
	unsigned int degree;
	double * params;
}COALA_RGSNN_HYPOTHESIS_t;

#endif