#ifndef COALA_MLP_INITIALIZATION_H
#define COALA_MLP_INITIALIZATION_H



//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------------
// Enums
//----------------------------------------------------------------------------------------------

typedef enum
{
    COALA_MLP_INITIALIZATION_NONE       = 0,
    COALA_MLP_INITIALIZATION_RANDOM     = 1,
    COALA_MLP_INITIALIZATION_XAVIER     = 2,
    COALA_MLP_INITIALIZATION_HE         = 3
}COALA_MLP_INITIALIZATION;

int coala_mlp_srandom(float * output, int size, int seed);
int coala_mlp_drandom(double * output, int size, int seed);
int coala_mlp_sxavier(float * mat, int rows, int cols, int seed);
int coala_mlp_dxavier(double * mat, int rows, int cols, int seed);
int coala_mlp_she(float * weights, int input_size, int seed);
int coala_mlp_dhe(double * weights, int input_size, int seed);

#endif