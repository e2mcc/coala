#include "coala_mlp_model.h"
#include "coala_mlp_loss.h"
#include <string>

CoalaMlpModel::CoalaMlpModel(int input_layer_neurons, int hidden_layers_count=1, int hidden_layers_neurons=5, int output_layer_neurons=1, float learning_rate=0.01f)
{
    this->hidden_layers_count = hidden_layers_count;
    this->learning_rate = learning_rate;

    this->input_layer = std::make_shared<CoalaMlpInputLayer>(input_layer_neurons);
   
    this->hidden_layers = std::vector<std::shared_ptr<CoalaMlpHiddenLayer>>(hidden_layers_count);
    for(int i=0; i<hidden_layers_count; i++)
    {
        if(i==0)
        {
            this->hidden_layers[i] = std::make_shared<CoalaMlpHiddenLayer>(input_layer_neurons,hidden_layers_neurons);
        }
        else
        {
            this->hidden_layers[i] = std::make_shared<CoalaMlpHiddenLayer>(output_layer_neurons,hidden_layers_neurons);
        }
        this->hidden_layers[i]->setActivation(COALA_MLP_ACTIVATION_RELU);
    }

    this->output_layer = std::make_shared<CoalaMlpOutputLayer>(hidden_layers_neurons, output_layer_neurons);
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


void CoalaMlpModel::forward(float* mat, int examples, int features)
{
    this->input_layer->forward(mat, examples, features);
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

float CoalaMlpModel::cost(float * MatPred, float * MatReal, int examples, int output_layer_neurons)
{
    return coala_mlp_smse(MatPred,  MatReal, examples, output_layer_neurons);
}


void CoalaMlpModel::backward(float * real_mat, int examples, int real_dim)
{
    this->output_layer->backward(real_mat, examples, real_dim);
    for(int i=this->hidden_layers_count-1; i>=0; i--)
    {
       
        this->hidden_layers[i]->backward();
    }
    this->input_layer->backward();
    return;
}


void CoalaMlpModel::update(float learning_rate)
{
    this->input_layer->update(learning_rate);
    for(int i=0; i<this->hidden_layers_count; i++)
    {
        this->hidden_layers[i]->update(learning_rate);
    }
    this->output_layer->update(learning_rate);
    return;
}

void CoalaMlpModel::saveToFile(std::string filename)
{
    //打开文件
    FILE *fp = fopen(filename.c_str(), "w");
    
    //文件首先保存模型层数数据
    fprintf(fp, "1 %d 1\n", this->hidden_layers_count);
    //每层用7个%分割
    fprintf(fp, "%%%%%%%%");
    //文件然后保存模型输入层神经元数
    fprintf(fp, "%d\n", this->input_layer->getNeuronsNum());
    //每层用7个%分割
    fprintf(fp, "%%%%%%%%\n");
    //文件其次开始保存模型隐藏层数据
    for(int i=0; i<this->hidden_layers_count; i++)
    {
        fprintf(fp, "%d %d %d\n", this->hidden_layers[i]->getInputSize(), this->hidden_layers[i]->getOutputSize(), this->hidden_layers[i]->getActivation());
        for(int j=0; j<this->hidden_layers[i]->getOutputSize(); j++)
        {
            for(int k=0; k<this->hidden_layers[i]->getInputSize(); k++)
            {
                fprintf(fp, "%f ", this->hidden_layers[i]->getWeights()[j*this->hidden_layers[i]->getInputSize()+k]);
            }
            fprintf(fp, "\n");
        }
        for(int j=0; j<this->hidden_layers[i]->getOutputSize(); j++)
        {
            fprintf(fp, "%f ", this->hidden_layers[i]->getBiases()[j]);
        }
        fprintf(fp, "\n");
        //每层用7个%分割
        fprintf(fp, "%%%%%%%%\n");
    }

    //文件最后开始保存模型输出层数据
    fprintf(fp, "%d %d %d\n", this->output_layer->getFeaturesNum(), this->output_layer->getNeuronsNum(),this->output_layer->getActivation());
    for(int j=0; j<this->output_layer->getNeuronsNum(); j++)
    {
        for(int k=0; k<this->output_layer->getFeaturesNum(); k++)
        {
            fprintf(fp, "%f ", this->output_layer->getWeights()[j*this->output_layer->getFeaturesNum()+k]);
        }
        fprintf(fp, "\n");
    }
    for(int j=0; j<this->output_layer->getNeuronsNum(); j++)
    {
        fprintf(fp, "%f ", this->output_layer->getBiases()[j]);
    }
    fprintf(fp, "\n");


    //关闭文件
    fclose(fp);
    
    return;
}
