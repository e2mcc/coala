//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------

#include "coala_mlp_loss.h"
#include <math.h>

float coala_mlp_smse(float * VecPred, float * VecReal, int size)
{
   float loss = 0.0;
    for(int i = 0; i < size; ++i) 
    {
        loss += pow((VecPred[i] - VecReal[i]), 2);
    }
    loss /= size;
    return loss;
}

double coala_mlp_dmse(double * VecPred, double * VecReal, int size)
{
    double loss = 0.0;
    for(int i = 0; i < size; ++i) 
    {
        loss += pow((VecPred[i] - VecReal[i]), 2);
    }
    loss /= size;
    
    return loss;
}
