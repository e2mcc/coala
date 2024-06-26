#ifndef COALA_MLOP_COST_H
#define COALA_MLOP_COST_H


//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------------
// Enums
//----------------------------------------------------------------------------------------------

typedef enum
{
    COALA_MLP_COST_NONE             = 0,
    COALA_MLP_COST_MSE              = 1,
    COALA_MLP_COST_CROSS_ENTROPY    = 2
}COALA_MLP_COST;



float coala_mlp_smse(float * MatPred, float * MatReal, int RowDim, int ColDim);
void coala_mlp_smse_grad(float * MatGrad, float * MatPred, float * MatReal, int RowDim, int ColDim);

float coala_mlp_cost(COALA_MLP_COST cost_func_rank, float * MatPred, float * MatReal, int RowDim, int ColDim);
void coala_mlp_costGrad(COALA_MLP_COST cost_func_rank, float * MatGrad, float * MatPred, float * MatReal, int RowDim, int ColDim);

#endif // COALA_MLOP_COST_H