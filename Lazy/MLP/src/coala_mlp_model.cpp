#include "coala_mlp_model.h"

using namespace coala::mlp;


CoalaMLP::CoalaMLP(int const input_layer_neurons, int const hidden_layers_count=1, int const output_layer_neurons=1)
{
    //默认参数设置
    this->input_layer_neurons = input_layer_neurons;
    this->hidden_layers_count = hidden_layers_count;
    this->output_layer_neurons = output_layer_neurons;
    this->batch_size = 1;
    for(int i=0; i<hidden_layers_count; i++)
    {
        this->hidden_layers_neurons.push_back(5*this->input_layer_neurons);
        this->hidden_layer_activate_funcs.push_back(COALA_MLP_ACTIVATE_NONE);
    }
    this->output_layer_activate_func = COALA_MLP_ACTIVATE_NONE;
    this->initialize_func = COALA_MLP_INITIALIZE_ZERO;
    this->cost_func = COALA_MLP_COST_MSE;


    //---------------------------------------------------------------------
    // 构建计算图
    //---------------------------------------------------------------------
    //构建计算图
    this->graph = std::make_shared<CoalaMlpGraph>();
    
    //加入输入节点
    this->graph->addForwardNodeVa(COALA_MLP_GRAPH_VARIABLE_INPUT, 0, this->batch_size, this->input_layer_neurons);
    

    //加入隐藏层的相关节点
    for(int i=0; i<this->hidden_layers_count; i++)
    {
        //加入隐藏层计算节点 matmul
        this->graph->addForwardNodeOp(COALA_MLP_GRAPH_OPERATOR_MATMUL, i+1);
        this->graph->addForwardEdge(i, i+1);

        //加入隐藏层权重W节点
        this->graph->addForwardNodeVa(COALA_MLP_GRAPH_VARIABLE_WEIGHT, i+2, this->input_layer_neurons, this->hidden_layers_neurons[i], this->initialize_func);
        this->graph->addForwardEdge(i+2, i+1);

        //加入ans节点
        this->graph->addForwardNodeVa(COALA_MLP_GRAPH_VARIABLE_ANS, i+3, this->batch_size, this->hidden_layers_neurons[i]);
        this->graph->addForwardEdge(i+1, i+3);

        //加入隐藏层计算节点 plus
        this->graph->addForwardNodeOp(COALA_MLP_GRAPH_OPERATOR_PLUS, i+4);
        this->graph->addForwardEdge(i+3, i+4);

        //加入隐藏层权重B节点
        this->graph->addForwardNodeVa(COALA_MLP_GRAPH_VARIABLE_WEIGHT, i+5, this->batch_size, this->hidden_layers_neurons[i],this->initialize_func);
        this->graph->addForwardEdge(i+5, i+4);
        
        //加入ans节点
        this->graph->addForwardNodeVa(COALA_MLP_GRAPH_VARIABLE_ANS, i+6, this->batch_size, this->hidden_layers_neurons[i]);
        this->graph->addForwardEdge(i+4, i+6);

        //加入隐藏层激活函数节点
        this->graph->addForwardNodeOp(COALA_MLP_GRAPH_OPERATOR_ACTIVATE, i+7, this->hidden_layer_activate_funcs[i]);
        this->graph->addForwardEdge(i+6, i+7);

        //加入ans节点
        this->graph->addForwardNodeVa(COALA_MLP_GRAPH_VARIABLE_ANS, i+8, this->batch_size, this->hidden_layers_neurons[i]);
        this->graph->addForwardEdge(i+7, i+8);
    }

    //加入输出层的相关节点
    //加入输出层计算节点 matmul
    this->graph->addForwardNodeOp(COALA_MLP_GRAPH_OPERATOR_MATMUL, 8*this->hidden_layers_count+1);
    this->graph->addForwardEdge(8*this->hidden_layers_count, 8*this->hidden_layers_count+1);

    //加入输出层权重W节点
    this->graph->addForwardNodeVa(COALA_MLP_GRAPH_VARIABLE_WEIGHT, 8*this->hidden_layers_count+2, this->hidden_layers_neurons[this->hidden_layers_count-1], this->output_layer_neurons, this->initialize_func);
    this->graph->addForwardEdge(8*this->hidden_layers_count+2, 8*this->hidden_layers_count+1);

    //加入ans节点
    this->graph->addForwardNodeVa(COALA_MLP_GRAPH_VARIABLE_ANS, 8*this->hidden_layers_count+3, this->batch_size, this->output_layer_neurons);
    this->graph->addForwardEdge(8*this->hidden_layers_count+1, 8*this->hidden_layers_count+3);

    //加入输出层计算节点 plus
    this->graph->addForwardNodeOp(COALA_MLP_GRAPH_OPERATOR_PLUS, 8*this->hidden_layers_count+4);
    this->graph->addForwardEdge(8*this->hidden_layers_count+3, 8*this->hidden_layers_count+4);

    //加入输出层权重B节点
    this->graph->addForwardNodeVa(COALA_MLP_GRAPH_VARIABLE_WEIGHT, 8*this->hidden_layers_count+5, this->batch_size, this->output_layer_neurons, this->initialize_func);
    this->graph->addForwardEdge(8*this->hidden_layers_count+5, 8*this->hidden_layers_count+4);

    //加入ans节点
    this->graph->addForwardNodeVa(COALA_MLP_GRAPH_VARIABLE_ANS, 8*this->hidden_layers_count+6, this->batch_size, this->output_layer_neurons);
    this->graph->addForwardEdge(8*this->hidden_layers_count+4, 8*this->hidden_layers_count+6);

    //加入输出层激活函数节点
    this->graph->addForwardNodeOp(COALA_MLP_GRAPH_OPERATOR_ACTIVATE, 8*this->hidden_layers_count+7,this->output_layer_activate_func);
    this->graph->addForwardEdge(8*this->hidden_layers_count+6, 8*this->hidden_layers_count+7);

    //加入ans节点
    this->graph->addForwardNodeVa(COALA_MLP_GRAPH_VARIABLE_ANS, 8*this->hidden_layers_count+8, this->batch_size, this->output_layer_neurons);
    this->graph->addForwardEdge(8*this->hidden_layers_count+7, 8*this->hidden_layers_count+8);


    //加入输出层的损失计算的相关节点
    //加入输出层的损失计算的计算节点
    this->graph->addForwardNodeOp(COALA_MLP_GRAPH_OPERATOR_COST, 8*this->hidden_layers_count+9, this->cost_func);
    this->graph->addForwardEdge(8*this->hidden_layers_count+8, 8*this->hidden_layers_count+9);

    //加入输出层的损失计算的真实值节点
    this->graph->addForwardNodeVa(COALA_MLP_GRAPH_VARIABLE_INPUT, 8*this->hidden_layers_count+10, this->batch_size, this->output_layer_neurons);
    this->graph->addForwardEdge(8*this->hidden_layers_count+10, 8*this->hidden_layers_count+9);

    //加入输出层的损失计算的ans节点
    this->graph->addForwardNodeVa(COALA_MLP_GRAPH_VARIABLE_ANS, 8*this->hidden_layers_count+11, 1,1);
    this->graph->addForwardEdge(8*this->hidden_layers_count+9, 8*this->hidden_layers_count+11);


    //---------------------------------------------------------------------
    //标记输入节点与输出节点
    //---------------------------------------------------------------------
    this->graph->setInputX(0);
    this->graph->setInputY(8*this->hidden_layers_count+10);

    return;
}


int CoalaMLP::setTraningBatch(int const batch_size)
{
    this->batch_size = batch_size;
    
    //更新输入节点
    this->graph->updateForwardNodeVa(0, COALA_MLP_GRAPH_VARIABLE_INPUT, this->batch_size, this->input_layer_neurons);
    

    //更新隐藏层的相关节点
    for(int i=0; i<this->hidden_layers_count; i++)
    {
        //更新隐藏层矩阵乘法ans节点
        this->graph->updateForwardNodeVa(i+3, COALA_MLP_GRAPH_VARIABLE_ANS,  this->batch_size, this->hidden_layers_neurons[i]);


        //更新隐藏层权重B节点
        this->graph->updateForwardNodeVa(i+5, COALA_MLP_GRAPH_VARIABLE_WEIGHT,  this->batch_size, this->hidden_layers_neurons[i],this->initialize_func);

        
        //更新隐藏层加法ans节点
        this->graph->updateForwardNodeVa(i+6, COALA_MLP_GRAPH_VARIABLE_ANS,  this->batch_size, this->hidden_layers_neurons[i]);


        //更新隐藏层激活ans节点
        this->graph->updateForwardNodeVa(i+8, COALA_MLP_GRAPH_VARIABLE_ANS,  this->batch_size, this->hidden_layers_neurons[i]);

    }

    //更新输出层的相关节点

    //更新输出层矩阵乘法ans节点
    this->graph->updateForwardNodeVa(8*this->hidden_layers_count+3, COALA_MLP_GRAPH_VARIABLE_ANS,  this->batch_size, this->output_layer_neurons);


    //更新输出层权重B节点
    this->graph->updateForwardNodeVa(8*this->hidden_layers_count+5, COALA_MLP_GRAPH_VARIABLE_WEIGHT,  this->batch_size, this->output_layer_neurons,this->initialize_func);


    //更新输出层加法ans节点
    this->graph->updateForwardNodeVa(8*this->hidden_layers_count+6, COALA_MLP_GRAPH_VARIABLE_ANS,  this->batch_size, this->output_layer_neurons);


    //更新输出层激活ans节点
    this->graph->updateForwardNodeVa(8*this->hidden_layers_count+8, COALA_MLP_GRAPH_VARIABLE_ANS,  this->batch_size, this->output_layer_neurons);


    //更新输出层的损失计算的相关节点
    //更新输出层的损失计算的真实值节点
    this->graph->updateForwardNodeVa(8*this->hidden_layers_count+10, COALA_MLP_GRAPH_VARIABLE_INPUT,  this->batch_size, this->output_layer_neurons);

    return 0;
}

int CoalaMLP::setHiddenLayersNeurons(int const layer, int const neurons)
{
    if(layer < 0 || layer >= this->hidden_layers_count) return 1;
    this->hidden_layers_neurons[layer] = neurons;


    //更新隐藏层权重W节点
    this->graph->updateForwardNodeVa(layer+2, COALA_MLP_GRAPH_VARIABLE_WEIGHT,  this->input_layer_neurons, this->hidden_layers_neurons[layer], this->initialize_func);

    //更新隐藏层矩阵乘法ans节点
    this->graph->updateForwardNodeVa(layer+3, COALA_MLP_GRAPH_VARIABLE_ANS,  this->batch_size, this->hidden_layers_neurons[layer]);


    //更新隐藏层权重B节点
    this->graph->updateForwardNodeVa(layer+5, COALA_MLP_GRAPH_VARIABLE_WEIGHT,  this->batch_size, this->hidden_layers_neurons[layer], this->initialize_func);

    
    //更新隐藏层加法ans节点
    this->graph->updateForwardNodeVa(layer+6, COALA_MLP_GRAPH_VARIABLE_ANS,  this->batch_size, this->hidden_layers_neurons[layer]);


    //更新隐藏层激活ans节点
    this->graph->updateForwardNodeVa(layer+8, COALA_MLP_GRAPH_VARIABLE_ANS,  this->batch_size, this->hidden_layers_neurons[layer]);


    //更新输出层的相关节点
    //更新输出层权重W节点
    if(layer == this->hidden_layers_count-1)
    {
        this->graph->updateForwardNodeVa( 8*this->hidden_layers_count+2, COALA_MLP_GRAPH_VARIABLE_WEIGHT, this->hidden_layers_neurons[this->hidden_layers_count-1], this->output_layer_neurons, this->initialize_func);
    }
    
    return 0;
}

int CoalaMLP::setHiddenLayerActivateFunc(int const layer, COALA_MLP_ACTIVATE_FUNC const activation_func)
{
    this->hidden_layer_activate_funcs[layer] = activation_func;

    this->graph->updateForwardNodeOp( layer+7, COALA_MLP_GRAPH_OPERATOR_ACTIVATE , activation_func);
    return 0;
}

int CoalaMLP::setOutputLayerActivateFunc(COALA_MLP_ACTIVATE_FUNC const activate_func)
{
    this->output_layer_activate_func = activate_func;
    this->graph->updateForwardNodeOp( 8*this->hidden_layers_count+7, COALA_MLP_GRAPH_OPERATOR_ACTIVATE , activate_func);
    return 0;
}

int CoalaMLP::setCostFunc(COALA_MLP_COST const cost_func)
{
    this->cost_func = cost_func;
    this->graph->updateForwardNodeOp( 8*this->hidden_layers_count+9, COALA_MLP_GRAPH_OPERATOR_COST, cost_func );
    return 0;
}


int CoalaMLP::setInitializeFunc(COALA_MLP_INITIALIZE_FUNC const initialize_func)
{
    this->initialize_func = initialize_func;

    for(int i=0;i<this->hidden_layers_count; i++)
    {
        //更新隐藏层权重W节点
        this->graph->updateForwardNodeVa(i+2, COALA_MLP_GRAPH_VARIABLE_WEIGHT,  this->input_layer_neurons, this->hidden_layers_neurons[i], this->initialize_func);

        //更新隐藏层权重B节点
        this->graph->updateForwardNodeVa(i+5, COALA_MLP_GRAPH_VARIABLE_WEIGHT,  this->batch_size, this->hidden_layers_neurons[i], this->initialize_func);

    }

    //更新输出层的相关节点
    //更新输出层权重W节点
    this->graph->updateForwardNodeVa( 8*this->hidden_layers_count+2, COALA_MLP_GRAPH_VARIABLE_WEIGHT, this->hidden_layers_neurons[this->hidden_layers_count-1], this->output_layer_neurons, this->initialize_func);
    //更新输出层权重B节点
    this->graph->updateForwardNodeVa(8*this->hidden_layers_count+5, COALA_MLP_GRAPH_VARIABLE_WEIGHT,  this->batch_size, this->output_layer_neurons,this->initialize_func);

    return 0;
}


int CoalaMLP::readyForTraining(void)
{
    this->graph->activating();
    return 0;
}