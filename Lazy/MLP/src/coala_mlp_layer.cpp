//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_layer.h"
#include "coala_mlp_blas.h"
#include <cstdlib>
#include <cstring>


//----------------------------------------------------------------------------------------------
// CoalaMlpInputLayer
//----------------------------------------------------------------------------------------------
CoalaMlpInputLayer::CoalaMlpInputLayer(int input_output_size)
{
    this->input_output_size = input_output_size;
}

void CoalaMlpInputLayer::forward(float * input, int examples)
{
    this->input = input;
    this->output = this->input;
    return;
}

float * CoalaMlpInputLayer::getOutput()
{
    return this->output;
}

//----------------------------------------------------------------------------------------------
// CoalaMlpHiddenLayer
//----------------------------------------------------------------------------------------------
CoalaMlpHiddenLayer::CoalaMlpHiddenLayer(int input_size, int output_size)
{
    this->input_size = input_size;
    this->output_size = output_size;
    this->activation_function = COALA_MLP_ACTIVATION_NONE;

    this->weights = (float*)malloc(sizeof(float) * input_size * output_size);
    std::memset(this->weights, 0, sizeof(float) * input_size * output_size);
    this->biases = (float*)malloc(sizeof(float) * output_size);
    std::memset(this->biases, 0, sizeof(float) * output_size);
    
    this->weights_gradient = (float*)malloc(sizeof(float) * input_size * output_size);
    std::memset(this->weights_gradient, 0, sizeof(float) * input_size * output_size);
    this->biases_gradient = (float*)malloc(sizeof(float) * output_size);
    std::memset(this->biases_gradient, 0, sizeof(float) * output_size);
}


int CoalaMlpHiddenLayer::setInputSize(int input_size)
{
    this->input_size = input_size;
    return 0;
}
int CoalaMlpHiddenLayer::setOutputSize(int output_size)
{
    this->output_size = output_size;
    return 0;
}
int CoalaMlpHiddenLayer::getOutputSize()
{
    return this->output_size;
}

int CoalaMlpHiddenLayer::setActivation(COALA_MLP_ACTIVATION activation_rank)
{
    this->activation_function = activation_rank;
    return 0;
}

float * CoalaMlpHiddenLayer::getOutput()
{
    return this->output;
}

void CoalaMlpHiddenLayer::initializeWeights(COALA_MLP_INITIALIZATION initialization_rank)
{   
    int seed = 0;
    switch (initialization_rank)
    {
        case COALA_MLP_INITIALIZATION_NONE:
            break;
        case COALA_MLP_INITIALIZATION_RANDOM:
            coala_mlp_srandom(this->weights, this->input_size * this->output_size, seed);
            coala_mlp_srandom(this->biases, this->output_size, seed);
            break;
        case COALA_MLP_INITIALIZATION_XAVIER:
            coala_mlp_sxavier(this->weights, this->input_size, this->output_size, seed);
            coala_mlp_sxavier(this->biases, 1, this->output_size, seed);
            break;
        case COALA_MLP_INITIALIZATION_HE:
            coala_mlp_she(this->weights, this->input_size, seed);
            coala_mlp_she(this->biases, 1, seed);
            break;
        default:
            break;
    }
    return;
}


void CoalaMlpHiddenLayer::forward(float * input, int examples)
{
    this->input = input;
    if(trained_times==0)
    {
        this->output = (float*)malloc(sizeof(float) * examples * this->output_size);
        memset(this->output, 0, sizeof(float) * examples * this->output_size);
    }

    //------------------------------------------------------------------------------------------------
    // 计算隐藏层 z = wx+b
    //------------------------------------------------------------------------------------------------
    // 计算 Z[i][j] = X[i][k] x W[k][j] + b[j]
    // i 表示样本编号, k 表示特征编号, j 表示神经元编号
    //          特征1 特征2 特征3             神经元1 神经元2  神经元3              神经元1 神经元2 神经元3
    // 样本1->  | x11  x12  x13  |         |  w11    w12    w13  |  特征1       |  b1    b2    b3  |  样本1
    // 样本2->  | x21  x22  x23  |         |  w21    w12    w13  |  特征2       |  b1    b2    b3  |  样本2
    // 样本3->  | x31  x32  x33  |  times  |  w31    w12    w13  |  特征3  plus |  b1    b2    b3  |  样本3
    // 样本4->  | x41  x42  x43  |                                              |  b1    b2    b3  |  样本4
    // 样本5->  | x51  x52  x53  |                                              |  b1    b2    b3  |  样本5
    size_t m = examples;
    size_t n = this->output_size;
    size_t k = this->input_size;
    for(size_t i=0; i<n; i++)
    {
        coala_mlp_saxpy(m, 1.0f, &this->biases[i], 0, &this->output[i*m], 1);
    }
    coala_mlp_sgemm(1, 0, 0, m, n, k, 1.0f, this->input, m, this->weights, k, 1.0f, this->output, m);

    //------------------------------------------------------------------------------------------------
    // 计算隐藏层激活
    //------------------------------------------------------------------------------------------------
    // 计算 A[i][j] = f(Z[i][j])
    switch (this->activation_function)
    {
        case COALA_MLP_ACTIVATION_NONE:
            break;
        case COALA_MLP_ACTIVATION_SIGMOID:
            coala_mlp_ssigmoid(this->output, this->output, m*n);
            break;
        case COALA_MLP_ACTIVATION_TANH:
            coala_mlp_stanh(this->output, this->output, m*n);
            break;
        case COALA_MLP_ACTIVATION_RELU:
            coala_mlp_srelu(this->output, this->output, m*n);
            break;
        case COALA_MLP_ACTIVATION_LEAKY_RELU:
            coala_mlp_sleakyrelu(this->output, this->output, m*n);
            break;
        case COALA_MLP_ACTIVATION_SOFTMAX:
            coala_mlp_ssoftmax(this->output, this->output, m, n);
            break;
        default:
            break;
    }

    this->trained_times++;

    return;
}


void CoalaMlpHiddenLayer::backward(float* input, float* output)
{
    return;
}

//----------------------------------------------------------------------------------------------
// CoalaMlpOutputLayer
//----------------------------------------------------------------------------------------------
CoalaMlpOutputLayer::CoalaMlpOutputLayer(int input_size, int output_size)
{
    this->input_size = input_size;
    this->output_size = output_size;
    this->activation_function = COALA_MLP_ACTIVATION_NONE;

    this->weights = (float*)malloc(sizeof(float) * input_size * output_size);
    std::memset(this->weights, 0, sizeof(float) * input_size * output_size);
    this->biases = (float*)malloc(sizeof(float) * output_size);
    std::memset(this->biases, 0, sizeof(float) * output_size);
    
    this->weights_gradient = (float*)malloc(sizeof(float) * input_size * output_size);
    std::memset(this->weights_gradient, 0, sizeof(float) * input_size * output_size);
    this->biases_gradient = (float*)malloc(sizeof(float) * output_size);
    std::memset(this->biases_gradient, 0, sizeof(float) * output_size);
}


int CoalaMlpOutputLayer::setInputSize(int input_size)
{
    this->input_size = input_size;
    return 0;
}

int CoalaMlpOutputLayer::setActivation(COALA_MLP_ACTIVATION activation_rank)
{
    this->activation_function = activation_rank;
    return 0;
}

void CoalaMlpOutputLayer::initializeWeights(COALA_MLP_INITIALIZATION initialization_rank)
{
    int seed = 0;
    switch (initialization_rank)
    {
        case COALA_MLP_INITIALIZATION_NONE:
            break;
        case COALA_MLP_INITIALIZATION_RANDOM:
            coala_mlp_srandom(this->weights, this->input_size * this->output_size, seed);
            coala_mlp_srandom(this->biases, this->output_size, seed);
            break;
        case COALA_MLP_INITIALIZATION_XAVIER:
            coala_mlp_sxavier(this->weights, this->input_size, this->output_size, seed);
            coala_mlp_sxavier(this->biases, 1, this->output_size, seed);
            break;
        case COALA_MLP_INITIALIZATION_HE:
            coala_mlp_she(this->weights, this->input_size, seed);
            coala_mlp_she(this->biases, 1, seed);
            break;
        default:
            break;
    }
    return;
}



void CoalaMlpOutputLayer::forward(float * input, int examples)
{
    this->input = input;
    if(trained_times==0)
    {
        this->output = (float*)malloc(sizeof(float) * examples * this->output_size);
        memset(this->output, 0, sizeof(float) * examples * this->output_size);
    }

    //------------------------------------------------------------------------------------------------
    // 计算隐藏层 z = wx+b
    //------------------------------------------------------------------------------------------------
    // 计算 Z[i][j] = X[i][k] x W[k][j] + b[j]
    // i 表示样本编号, k 表示特征编号, j 表示神经元编号
    //          特征1 特征2 特征3             神经元1 神经元2  神经元3              神经元1 神经元2 神经元3
    // 样本1->  | x11  x12  x13  |         |  w11    w12    w13  |  特征1       |  b1    b2    b3  |  样本1
    // 样本2->  | x21  x22  x23  |         |  w21    w12    w13  |  特征2       |  b1    b2    b3  |  样本2
    // 样本3->  | x31  x32  x33  |  times  |  w31    w12    w13  |  特征3  plus |  b1    b2    b3  |  样本3
    // 样本4->  | x41  x42  x43  |                                              |  b1    b2    b3  |  样本4
    // 样本5->  | x51  x52  x53  |                                              |  b1    b2    b3  |  样本5
    size_t m = examples;
    size_t n = this->output_size;
    size_t k = this->input_size;
    for(size_t i=0; i<n; i++)
    {
        coala_mlp_saxpy(m, 1.0f, &this->biases[i], 0, &this->output[i*m], 1);
    }
    
    if( n == 1 )
        coala_mlp_sgemv(1, 0, m, k, 1.0f, this->input, m, this->weights, 1, 1.0f, this->output, 1);
    else    
        coala_mlp_sgemm(1, 0, 0, m, n, k, 1.0f, this->input, m, this->weights, k, 1.0f, this->output, m);


    //------------------------------------------------------------------------------------------------
    // 计算隐藏层激活
    //------------------------------------------------------------------------------------------------
    // 计算 A[i][j] = f(Z[i][j])
    switch (this->activation_function)
    {
        case COALA_MLP_ACTIVATION_NONE:
            break;
        case COALA_MLP_ACTIVATION_SIGMOID:
            coala_mlp_ssigmoid(this->output, this->output, m*n);
            break;
        case COALA_MLP_ACTIVATION_TANH:
            coala_mlp_stanh(this->output, this->output, m*n);
            break;
        case COALA_MLP_ACTIVATION_RELU:
            coala_mlp_srelu(this->output, this->output, m*n);
            break;
        case COALA_MLP_ACTIVATION_LEAKY_RELU:
            coala_mlp_sleakyrelu(this->output, this->output, m*n);
            break;
        case COALA_MLP_ACTIVATION_SOFTMAX:
            coala_mlp_ssoftmax(this->output, this->output, m, n);
            break;
        default:
            break;
    }

    this->trained_times++;
}

float *  CoalaMlpOutputLayer::getOutput()
{
    return this->output;
}

void CoalaMlpOutputLayer::backward(float * real, int examples)
{
    // i is the example index
    // MSE = 1/N * sum{ (y_pred_i - y_real_i)^2 }
    // dMSE/dy_pred_i = 2*(y_pred_i - y_real_i)/N

    // Sigmoid:
    // y_pred_i = 1/( 1+e^(-z_i) )
    // dSigmoid/dz = Sigmoid * (1-Sigmoid) = y_pred_i * (1-y_pred_i)

    // z = X_i * W_i + B_i
    // dz/dW = X
    // dz/dB = 1

    // dMSE/dW = sum{ dMSE/dy_pred_i * dy_pred_i/dz_i * dz_i/dW } / N

    return;
}
