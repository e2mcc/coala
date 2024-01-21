//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_activation.h"
#include <cmath>


int coala_mlp_ssigmoid(float * output, float * input, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = 1.0 / (1.0 + exp(-input[i]));
    }
    return 0;
}

int coala_mlp_dsigmoid(double * output, double * input, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = 1.0 / (1.0 + exp(-input[i]));
    }
    return 0;
}

int coala_mlp_stanh(float * output, float * input, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = tanh(input[i]);
    }
    return 0;
}

int coala_mlp_dtanh(double * output, double * input, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = tanh(input[i]);
    }
    return 0;
}

int coala_mlp_srelu(float * output, float * input, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
    return 0;
}

int coala_mlp_drelu(double * output, double * input, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
    return 0;
}

int coala_mlp_sleakyrelu(float * output, float * input, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = input[i] > 0 ? input[i] : 0.01 * input[i];
    }
    return 0;
}

int coala_mlp_dleakyrelu(double * output, double * input, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = input[i] > 0 ? input[i] : 0.01 * input[i];
    }
    return 0;
}

int coala_mlp_ssoftmax(float * output, float * input, int m, int n)
{
    float sum = 0.0;
    for(int i = 0; i < m; i++)
    {
        sum = 0.0;
        for(int j = 0; j < n; j++)
        {
            sum += exp(input[i * n + j]);
        }
        for(int j = 0; j < n; j++)
        {
            output[i * n + j] = exp(input[i * n + j]) / sum;
        }
    }
    return 0;
}

int coala_mlp_dsoftmax(double * output, double * input, int m, int n)
{
    double sum = 0.0;
    for(int i = 0; i < m; i++)
    {
        sum = 0.0;
        for(int j = 0; j < n; j++)
        {
            sum += exp(input[i * n + j]);
        }
        for(int j = 0; j < n; j++)
        {
            output[i * n + j] = exp(input[i * n + j]) / sum;
        }
    }
    return 0;
}
