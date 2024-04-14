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
        this->hidden_layer_activate_funcs.push_back(COALA_MLP_ACTIVATION_NONE);
    }
    this->output_layer_activate_func = COALA_MLP_ACTIVATION_NONE;
    this->initialize_func = COALA_MLP_INITIALIZATION_NONE;
    this->cost_func = COALA_MLP_LOSS_MSE;


    //---------------------------------------------------------------------
    // 构建计算图
    //---------------------------------------------------------------------
    //构建计算图
    this->graph = std::make_shared<CoalaMlpGraph>();
    
    //加入输入节点
    this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_VARIABLE_INPUT, 0);
    

    //加入隐藏层的相关节点
    for(int i=0; i<this->hidden_layers_count; i++)
    {
        //加入隐藏层计算节点 matmul
        this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_OPERATOR_MATMUL, i+1);
        this->graph->addPlanningForwardEdge(i, i+1);

        //加入隐藏层权重W节点
        this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_VARIABLE_WEIGHT, i+2);
        this->graph->addPlanningForwardEdge(i+2, i+1);

        //加入ans节点
        this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_VARIABLE_ANS, i+3);
        this->graph->addPlanningForwardEdge(i+1, i+3);

        //加入隐藏层计算节点 plus
        this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_OPERATOR_PLUS, i+4);
        this->graph->addPlanningForwardEdge(i+3, i+4);

        //加入隐藏层权重B节点
        this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_VARIABLE_WEIGHT, i+5);
        this->graph->addPlanningForwardEdge(i+5, i+4);
        
        //加入ans节点
        this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_VARIABLE_ANS, i+6);
        this->graph->addPlanningForwardEdge(i+4, i+6);

        //加入隐藏层激活函数节点
        this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_OPERATOR_ACTIVATE, i+7);
        this->graph->addPlanningForwardEdge(i+6, i+7);

        //加入ans节点
        this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_VARIABLE_ANS, i+8);
        this->graph->addPlanningForwardEdge(i+7, i+8);
    }

    //加入输出层的相关节点
    //加入输出层计算节点 matmul
    this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_OPERATOR_MATMUL, 8*this->hidden_layers_count+1);
    this->graph->addPlanningForwardEdge(8*this->hidden_layers_count, 8*this->hidden_layers_count+1);

    //加入输出层权重W节点
    this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_VARIABLE_WEIGHT, 8*this->hidden_layers_count+2);
    this->graph->addPlanningForwardEdge(8*this->hidden_layers_count+2, 8*this->hidden_layers_count+1);

    //加入ans节点
    this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_VARIABLE_ANS, 8*this->hidden_layers_count+3);
    this->graph->addPlanningForwardEdge(8*this->hidden_layers_count+1, 8*this->hidden_layers_count+3);

    //加入输出层计算节点 plus
    this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_OPERATOR_PLUS, 8*this->hidden_layers_count+4);
    this->graph->addPlanningForwardEdge(8*this->hidden_layers_count+3, 8*this->hidden_layers_count+4);

    //加入输出层权重B节点
    this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_VARIABLE_WEIGHT, 8*this->hidden_layers_count+5);
    this->graph->addPlanningForwardEdge(8*this->hidden_layers_count+5, 8*this->hidden_layers_count+4);

    //加入ans节点
    this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_VARIABLE_ANS, 8*this->hidden_layers_count+6);
    this->graph->addPlanningForwardEdge(8*this->hidden_layers_count+4, 8*this->hidden_layers_count+6);

    //加入输出层激活函数节点
    this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_OPERATOR_ACTIVATE, 8*this->hidden_layers_count+7);
    this->graph->addPlanningForwardEdge(8*this->hidden_layers_count+6, 8*this->hidden_layers_count+7);

    //加入ans节点
    this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_VARIABLE_ANS, 8*this->hidden_layers_count+8);
    this->graph->addPlanningForwardEdge(8*this->hidden_layers_count+7, 8*this->hidden_layers_count+8);


    //加入输出层的损失计算的相关节点
    //加入输出层的损失计算的计算节点
    this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_OPERATOR_COST, 8*this->hidden_layers_count+9);
    this->graph->addPlanningForwardEdge(8*this->hidden_layers_count+8, 8*this->hidden_layers_count+9);

    //加入输出层的损失计算的真实值节点
    this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_VARIABLE_INPUT, 8*this->hidden_layers_count+10);
    this->graph->addPlanningForwardEdge(8*this->hidden_layers_count+10, 8*this->hidden_layers_count+9);

    //加入输出层的损失计算的ans节点
    this->graph->addPlanningForwardNode(COALA_MLP_GRAPH_VARIABLE_ANS, 8*this->hidden_layers_count+11);
    this->graph->addPlanningForwardEdge(8*this->hidden_layers_count+9, 8*this->hidden_layers_count+11);

    return;
}


int CoalaMLP::setTraningBatch(int const batch_size)
{
    this->batch_size = batch_size;
    return 0;
}

int CoalaMLP::setHiddenLayersNeurons(int const layer, int const neurons)
{
    if(layer < 0 || layer >= this->hidden_layers_count) return 1;
    this->hidden_layers_neurons[layer] = neurons;
    return 0;
}

int CoalaMLP::setHiddenLayerActivateFunc(int const layer, COALA_MLP_ACTIVATION const activation_func)
{
    this->hidden_layer_activate_funcs[layer] = activation_func;
    return 0;
}

int CoalaMLP::setOutputLayerActivateFunc(COALA_MLP_ACTIVATION const activate_func)
{
    this->output_layer_activate_func = activate_func;
    return 0;
}

int CoalaMLP::setCostFunc(COALA_MLP_COST const cost_func)
{
    this->cost_func = cost_func;
    return 0;
}


int CoalaMLP::setInitializeFunc(COALA_MLP_INITIALIZATION const initialize_func)
{
    this->initialize_func = initialize_func;
    return 0;
}


int CoalaMLP::readyForTraining(void)
{
    this->graph->activating();
    return 0;
}