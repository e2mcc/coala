//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_layer.h"
#include "coala_mlp_blas.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

//----------------------------------------------------------------------------------------------
// CoalaMlpInputLayer
//----------------------------------------------------------------------------------------------
CoalaMlpInputLayer::CoalaMlpInputLayer(int neurons)
{
    this->neurons = neurons;
}

void CoalaMlpInputLayer::forward(float * mat, int examples, int features)
{
    this->input = mat;
    this->output = this->input;
    return;
}

int CoalaMlpInputLayer::getNeuronsNum()
{
    return this->neurons;
}

float * CoalaMlpInputLayer::getOutput()
{
    return this->output;
}

//----------------------------------------------------------------------------------------------
// CoalaMlpHiddenLayer
//----------------------------------------------------------------------------------------------
CoalaMlpHiddenLayer::CoalaMlpHiddenLayer(int features, int neurons)
{
    this->features = features;
    this->neurons = neurons;
    this->activation_function = COALA_MLP_ACTIVATION_NONE;

    this->weights = (float*)malloc(sizeof(float) * features * neurons);
    std::memset(this->weights, 0, sizeof(float) * features * neurons);
    this->biases = (float*)malloc(sizeof(float) * neurons);
    std::memset(this->biases, 0, sizeof(float) * neurons);
    
    this->weights_gradient = (float*)malloc(sizeof(float) * features * neurons);
    std::memset(this->weights_gradient, 0, sizeof(float) * features * neurons);
    this->biases_gradient = (float*)malloc(sizeof(float) * neurons);
    std::memset(this->biases_gradient, 0, sizeof(float) * neurons);
}


int CoalaMlpHiddenLayer::setInputSize(int input_size)
{
    this->features = input_size;
    return 0;
}

int CoalaMlpHiddenLayer::getInputSize()
{
    return this->features;
}

int CoalaMlpHiddenLayer::setOutputSize(int output_size)
{
    this->neurons = output_size;
    return 0;
}
int CoalaMlpHiddenLayer::getOutputSize()
{
    return this->neurons;
}

int CoalaMlpHiddenLayer::setActivation(COALA_MLP_ACTIVATION activation_rank)
{
    this->activation_function = activation_rank;
    return 0;
}

int CoalaMlpHiddenLayer::getActivation()
{
    return this->activation_function;
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
            coala_mlp_srandom(this->weights, this->features * this->neurons, seed);
            coala_mlp_srandom(this->biases, this->neurons, seed);
            break;
        case COALA_MLP_INITIALIZATION_XAVIER:
            coala_mlp_sxavier(this->weights, this->features, this->neurons, seed);
            coala_mlp_sxavier(this->biases, 1, this->neurons, seed);
            break;
        case COALA_MLP_INITIALIZATION_HE:
            coala_mlp_she(this->weights, this->features, seed);
            coala_mlp_she(this->biases, 1, seed);
            break;
        default:
            break;
    }
    return;
}

float * CoalaMlpHiddenLayer::getWeights()
{
    return this->weights;
}

float * CoalaMlpHiddenLayer::getBiases()
{
    return this->biases;
}


void CoalaMlpHiddenLayer::forward(float * input, int examples)
{
    this->input = input;
    if(trained_times==0)
    {
        this->output = (float*)malloc(sizeof(float) * examples * this->neurons);
        memset(this->output, 0, sizeof(float) * examples * this->neurons);
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
    size_t n = this->neurons;
    size_t k = this->features;
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


void CoalaMlpHiddenLayer::backward()
{
    return;
}

//----------------------------------------------------------------------------------------------
// CoalaMlpOutputLayer
//----------------------------------------------------------------------------------------------
CoalaMlpOutputLayer::CoalaMlpOutputLayer(int features, int neurons)
{
    this->features = features;
    this->neurons = neurons;
    this->activation_function = COALA_MLP_ACTIVATION_NONE;

    this->weights = (float*)malloc(sizeof(float) * features * neurons);
    std::memset(this->weights, 0, sizeof(float) * features * neurons);
    this->biases = (float*)malloc(sizeof(float) * neurons);
    std::memset(this->biases, 0, sizeof(float) * neurons);
    
    this->weights_gradient = (float*)malloc(sizeof(float) * features * neurons);
    std::memset(this->weights_gradient, 0, sizeof(float) * features * neurons);
    this->biases_gradient = (float*)malloc(sizeof(float) * neurons);
    std::memset(this->biases_gradient, 0, sizeof(float) * neurons);
}


int CoalaMlpOutputLayer::setInputSize(int input_size)
{
    this->features = input_size;
    return 0;
}

int CoalaMlpOutputLayer::getFeaturesNum()
{
    return this->features;
}

int CoalaMlpOutputLayer::setActivation(COALA_MLP_ACTIVATION activation_rank)
{
    this->activation_function = activation_rank;
    return 0;
}

int CoalaMlpOutputLayer::getActivation()
{
    return this->activation_function;
}

int CoalaMlpOutputLayer::getNeuronsNum()
{
    return this->neurons;
}

float * CoalaMlpOutputLayer::getWeights()
{
    return this->weights;
}

float * CoalaMlpOutputLayer::getBiases()
{
    return this->biases;
}

void CoalaMlpOutputLayer::initializeWeights(COALA_MLP_INITIALIZATION initialization_rank)
{
    int seed = 0;
    switch (initialization_rank)
    {
        case COALA_MLP_INITIALIZATION_NONE:
            break;
        case COALA_MLP_INITIALIZATION_RANDOM:
            coala_mlp_srandom(this->weights, this->features * this->neurons, seed);
            coala_mlp_srandom(this->biases, this->neurons, seed);
            break;
        case COALA_MLP_INITIALIZATION_XAVIER:
            coala_mlp_sxavier(this->weights, this->features, this->neurons, seed);
            coala_mlp_sxavier(this->biases, 1, this->neurons, seed);
            break;
        case COALA_MLP_INITIALIZATION_HE:
            coala_mlp_she(this->weights, this->features, seed);
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
        this->output_z = (float*)malloc(sizeof(float) * examples * this->neurons);
        memset(this->output_z, 0, sizeof(float) * examples * this->neurons);
        this->output_y = (float*)malloc(sizeof(float) * examples * this->neurons);
        memset(this->output_y, 0, sizeof(float) * examples * this->neurons);
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
    size_t n = this->neurons;
    size_t k = this->features;
    for(size_t i=0; i<n; i++)
    {
        coala_mlp_saxpy(m, 1.0f, &this->biases[i], 0, &this->output_z[i*m], 1);
    }
    
    if( n == 1 )
        coala_mlp_sgemv(1, 0, m, k, 1.0f, this->input, m, this->weights, 1, 1.0f, this->output_z, 1);
    else    
        coala_mlp_sgemm(1, 0, 0, m, n, k, 1.0f, this->input, m, this->weights, k, 1.0f, this->output_z, m);


    //------------------------------------------------------------------------------------------------
    // 计算隐藏层激活
    //------------------------------------------------------------------------------------------------
    // 计算 A[i][j] = f(Z[i][j])
    switch (this->activation_function)
    {
        case COALA_MLP_ACTIVATION_NONE:
            break;
        case COALA_MLP_ACTIVATION_SIGMOID:
            coala_mlp_ssigmoid(this->output_y, this->output_z, m*n);
            break;
        case COALA_MLP_ACTIVATION_TANH:
            coala_mlp_stanh(this->output_y, this->output_z, m*n);
            break;
        case COALA_MLP_ACTIVATION_RELU:
            coala_mlp_srelu(this->output_y, this->output_z, m*n);
            break;
        case COALA_MLP_ACTIVATION_LEAKY_RELU:
            coala_mlp_sleakyrelu(this->output_y, this->output_z, m*n);
            break;
        case COALA_MLP_ACTIVATION_SOFTMAX:
            coala_mlp_ssoftmax(this->output_y, this->output_z, m, n);
            break;
        default:
            break;
    }

    this->trained_times++;
}

float *  CoalaMlpOutputLayer::getOutput()
{
    return this->output_y;
}


// real_mat 列优先存储
void CoalaMlpOutputLayer::backward(float * real_mat, int examples, int real_dim)
{   
    if(real_dim != this->neurons)
    {
        printf("Error: CoalaMlpOutputLayer::backward() real_dim != this->output_size\n");
        return;
    }

    // dMSE/dyij = 2/m * (yij - rij)
    this->dloss2dy = (float*)malloc(sizeof(float) * examples * this->neurons);
    for(int i=0; i<examples; i++)
    {
        for(int j=0; j<real_dim; j++)
        {
            this->dloss2dy[i+j*examples] = 2*(this->output_y[i+j*examples] - real_mat[i+j*examples])/examples;
        }
    }

    //dyij/dzij
    this->dy2dz = (float*)malloc(sizeof(float) * examples * this->neurons);
    switch(this->activation_function)
    {
        case COALA_MLP_ACTIVATION_NONE:
            break;
        case COALA_MLP_ACTIVATION_SIGMOID:
            coala_mlp_ssigmoid_gradient(this->dy2dz, this->output_z, examples*real_dim);
            break;
        case COALA_MLP_ACTIVATION_TANH:
            coala_mlp_stanh_gradient(this->dy2dz, this->output_z, examples*real_dim);
            break;
        case COALA_MLP_ACTIVATION_RELU:
            coala_mlp_srelu_gradient(this->dy2dz, this->output_z, examples*real_dim);
            break;
        case COALA_MLP_ACTIVATION_LEAKY_RELU:
            coala_mlp_sleakyrelu_gradient(this->dy2dz, this->output_z, examples*real_dim);
            break;
        case COALA_MLP_ACTIVATION_SOFTMAX:
            coala_mlp_ssoftmax_gradient(this->dy2dz, this->output_z, examples, real_dim);
            break;
        default:
            break;
    }

    // dzij/dwpj = hip ; hip 是本层的输入，也是隐藏层的输出
    // dmse/dwpj = 1/m sum_i{ dMSE/dyij * dyij/dzij * hip }
    for( int p=0; p<this->features; p++)
    {
        for(int j=0;j<this->neurons; j++)
        {
            float sum = 0.0;
            for(int i=0;i<examples;i++)
            {
                sum += this->dloss2dy[i+j*examples] * this->dy2dz[i+j*examples] * this->input[i+p*examples];
                
            }
            this->weights_gradient[p+j*this->features] = sum/examples;
        }
    }

    // dzij/dbj = 1 ;
    // dmse/dbj = 1/m sum_i{ dMSE/dyij * dyij/dzij * 1}
    for(int j=0;j<this->neurons; j++)
    {
        float sum = 0.0;
        for(int i=0;i<examples;i++)
        {
            sum += this->dloss2dy[i+j*examples] * this->dy2dz[i+j*examples];
            
        }
        this->biases_gradient[j] = sum/examples;
    }

    return;
}


void CoalaMlpOutputLayer::update(float learning_rate)
{
    // w = w - learning_rate * dw
    coala_mlp_saxpy(this->features*this->neurons, -learning_rate, this->weights_gradient, 1, this->weights, 1);
    // b = b - learning_rate * db
    coala_mlp_saxpy(this->neurons, -learning_rate, this->biases_gradient, 1, this->biases, 1);
    return;
}

