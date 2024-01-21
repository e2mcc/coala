#ifndef COALA_MLOP_LOSS_H
#define COALA_MLOP_LOSS_H


//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------------
// Enums
//----------------------------------------------------------------------------------------------

typedef enum
{
    COALA_MLP_LOSS_NONE             = 0,
    COALA_MLP_LOSS_MSE              = 1,
    COALA_MLP_LOSS_CROSS_ENTROPY    = 2
}COALA_MLP_LOSS;



float coala_mlp_smse(float * VecPred, float * VecReal, int size);
double coala_mlp_dmse(double * VecPred, double * VecReal, int size);

#endif // COALA_MLOP_LOSS_H