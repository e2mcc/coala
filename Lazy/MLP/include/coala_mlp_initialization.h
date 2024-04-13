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
    COALA_MLP_INITIALIZATION_ZERO       = 1,
    COALA_MLP_INITIALIZATION_ONES       = 2,
    COALA_MLP_INITIALIZATION_RANDOM     = 3,
    COALA_MLP_INITIALIZATION_XAVIER     = 4,
    COALA_MLP_INITIALIZATION_HE         = 5
}COALA_MLP_INITIALIZATION;

int coala_mlp_szero(float * vec, int size);
int coala_mlp_dzero(double * vec, int size);
int coala_mlp_sones(float * vec, int size);
int coala_mlp_dones(double * vec, int size);
int coala_mlp_srandom(float * vec, int size, int seed);
int coala_mlp_drandom(double * vec, int size, int seed);
int coala_mlp_sxavier(float * mat, int rows, int cols, int seed);
int coala_mlp_dxavier(double * mat, int rows, int cols, int seed);
int coala_mlp_she(float * weights, int input_size, int seed);
int coala_mlp_dhe(double * weights, int input_size, int seed);

#endif