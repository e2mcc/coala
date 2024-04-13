//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------

#include "coala_mlp_loss.h"
#include "coala_mlp_blas.h"
#include <math.h>
#include <string.h>

float coala_mlp_smse(float * MatPred, float * MatReal, int RowDim, int ColDim)
{
    float * temp = (float *)malloc(sizeof(float) * RowDim * ColDim);
    memcpy(temp, MatPred, sizeof(float) * RowDim * ColDim);
    coala_mlp_saxpy(RowDim * ColDim, -1.0, MatReal, 1, temp, 1);
    float loss = 0.0f;
    loss = coala_mlp_sdot(RowDim * ColDim, temp, 1, temp, 1);
    free(temp);
    return loss / (float)(2 * RowDim * ColDim);
}

void coala_mlp_smse_grad(float * MatGrad, float * MatPred, float * MatReal, int RowDim, int ColDim)
{
    memcpy(MatGrad, MatPred, sizeof(float) * RowDim * ColDim);
    
    coala_mlp_saxpy(RowDim * ColDim, -1.0, MatReal, 1, MatGrad, 1);
    
    float size = (float)(RowDim * ColDim);

    coala_mlp_saxpy(RowDim * ColDim, (1.0-size) / size, MatGrad, 1, MatGrad, 1);

    return;
}

float coala_mlp_cost(COALA_MLP_LOSS losstype, float * MatPred, float * MatReal, int RowDim, int ColDim)
{
    switch (losstype)
    {
        case COALA_MLP_LOSS_MSE:
            return coala_mlp_smse(MatPred, MatReal, RowDim, ColDim);
        default:
            return 0;
    }
    return 0;
}

void coala_mlp_costGrad(COALA_MLP_LOSS losstype, float * MatGrad, float * MatPred, float * MatReal, int RowDim, int ColDim)
{
    switch (losstype)
    {
        case COALA_MLP_LOSS_MSE:
            coala_mlp_smse_grad(MatGrad, MatPred, MatReal, RowDim, ColDim);
            break;
        default:
            break;
    }
    return;
}