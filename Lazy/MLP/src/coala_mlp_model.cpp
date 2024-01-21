#include "coala_mlp_model.h"
#include "coala_mlp_loss.h"


CoalaMlpModel::CoalaMlpModel(int input_size, int hidden_layers_count=1, int hidden_layers_output_size=5, int output_size=1, float learning_rate=0.01f)
{
    this->input_size = input_size;
    this->output_size = output_size;
    this->hidden_layers_count = hidden_layers_count;
    this->learning_rate = learning_rate;

    this->input_layer = std::make_shared<CoalaMlpInputLayer>(input_size);
   
    this->hidden_layers = std::vector<std::shared_ptr<CoalaMlpHiddenLayer>>(hidden_layers_count);
    for(int i=0; i<hidden_layers_count; i++)
    {
        if(i==0)
        {
            this->hidden_layers[i] = std::make_shared<CoalaMlpHiddenLayer>(input_size, hidden_layers_output_size);
        }
        else
        {
            this->hidden_layers[i] = std::make_shared<CoalaMlpHiddenLayer>(hidden_layers_output_size, hidden_layers_output_size);
        }
        this->hidden_layers[i]->setActivation(COALA_MLP_ACTIVATION_RELU);
    }

    this->output_layer = std::make_shared<CoalaMlpOutputLayer>(hidden_layers_output_size, output_size);
    this->output_layer->setActivation(COALA_MLP_ACTIVATION_SIGMOID);
}



int CoalaMlpModel::setHiddenLayer(int hidden_layer_rank, COALA_MLP_ACTIVATION activation_rank, int output_size)
{
    if(hidden_layer_rank < 0 || hidden_layer_rank >= this->hidden_layers_count) return -1;
    
    this->hidden_layers[hidden_layer_rank]->setActivation(activation_rank);
    
    this->hidden_layers[hidden_layer_rank]->setOutputSize(output_size);

    int nextlayer = hidden_layer_rank+1;
    if( nextlayer < this->hidden_layers_count)
        this->hidden_layers[nextlayer]->setInputSize(output_size);
    else
        this->output_layer->setInputSize(output_size);
    return 0;
}


int CoalaMlpModel::setOutputLayerActivation(COALA_MLP_ACTIVATION activation_rank)
{
    this->output_layer->setActivation(activation_rank);
    return 0;
}



void CoalaMlpModel::initializeWeights(COALA_MLP_INITIALIZATION initialization_rank)
{
    for(int i=0; i<this->hidden_layers_count; i++)
    {
        this->hidden_layers[i]->initializeWeights(initialization_rank);
    }
    this->output_layer->initializeWeights(initialization_rank);
    return;
}


void CoalaMlpModel::forward(float * input, int examples)
{
    this->input_layer->forward(input, examples);
    for(int i=0; i<this->hidden_layers_count; i++)
    {
        if(i==0)
        {
            this->hidden_layers[i]->forward(this->input_layer->getOutput(), examples);
        }
        else
        {
            this->hidden_layers[i]->forward(this->hidden_layers[i-1]->getOutput(), examples);
        }
    }
    this->output_layer->forward(this->hidden_layers[this->hidden_layers_count-1]->getOutput(), examples);
    this->trained_times++;
    return;
}

float CoalaMlpModel::cost(float * VecPred, float * VecReal, int dim)
{
    return coala_mlp_smse(VecPred, VecReal, dim);
}


void CoalaMlpModel::backward(float * input, float * output)
{
    // this->output_layer->backward(this->hidden_layers[this->hidden_layers_count-1]->getOutput(), output);
    // for(int i=this->hidden_layers_count-1; i>=0; i--)
    // {
    //     if(i==0)
    //     {
    //         this->hidden_layers[i]->backward(this->input_layer->getOutput(), this->output_layer->getInputGradient());
    //     }
    //     else
    //     {
    //         this->hidden_layers[i]->backward(this->hidden_layers[i-1]->getOutput(), this->hidden_layers[i+1]->getInputGradient());
    //     }
    // }
    // this->input_layer->backward(input, this->hidden_layers[0]->getInputGradient());
    return;
}