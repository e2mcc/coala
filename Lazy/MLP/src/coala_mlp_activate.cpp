//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_activate.h"
#include <cmath>


int coala_mlp_activation(COALA_MLP_ACTIVATE_FUNC const activation_rank, float * output, float * input, int const rows, int const cols)
{
    switch(activation_rank)
    {
        case COALA_MLP_ACTIVATE_SIGMOID:
            coala_mlp_ssigmoid(output, input, rows * cols);
            break;
        case COALA_MLP_ACTIVATE_TANH:
            coala_mlp_stanh(output, input, rows * cols);
            break;
        case COALA_MLP_ACTIVATE_RELU:
            coala_mlp_srelu(output, input, rows * cols);
            break;
        case COALA_MLP_ACTIVATE_LEAKY_RELU:
            coala_mlp_sleakyrelu(output, input, rows * cols);
            break;
        case COALA_MLP_ACTIVATE_SOFTMAX:
            coala_mlp_ssoftmax(output, input, rows, cols);
            break;
        default:
            break;
    }
    return 0;
}

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


int coala_mlp_ssigmoid_gradient(float * output, float * input, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = input[i] * (1.0 - input[i]);
    }
    return 0;
}



int coala_mlp_dsigmoid_gradient(double * output, double * input, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = input[i] * (1.0 - input[i]);
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

int coala_mlp_stanh_gradient(float * output, float * input, int size)
{
    return 0;
}

int coala_mlp_dtanh_gradient(double * output, double * input, int size)
{
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

int coala_mlp_srelu_gradient(float * output, float * input, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = input[i] > 0 ? 1 : 0;
    }
    return 0;
}

int coala_mlp_drelu_gradient(double * output, double * input, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = input[i] > 0 ? 1 : 0;
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

int coala_mlp_sleakyrelu_gradient(float * output, float * input, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = input[i] > 0 ? 1 : 0.01;
    }
    return 0;
}

int coala_mlp_dleakyrelu_gradient(double * output, double * input, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = input[i] > 0 ? 1 : 0.01;
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

int coala_mlp_ssoftmax_gradient(float * output, float * input, int m, int n)
{
    return 0;
}

int coala_mlp_dsoftmax_gradient(double * output, double * input, int m, int n)
{
    return 0;
}