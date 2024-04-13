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
        this->hidden_layer_activation_funcs.push_back(COALA_MLP_ACTIVATION_NONE);
    }
    this->output_layer_activation_func = COALA_MLP_ACTIVATION_NONE;
    this->initialization_func = COALA_MLP_INITIALIZATION_NONE;
    this->cost_func = COALA_MLP_LOSS_MSE;

    //---------------------------------------------------------------------
    // 构建计算图
    //---------------------------------------------------------------------
    //计算总节点数
    int total_nodes_count = 1+hidden_layers_count*8+8+3;

    //构建计算图
    this->graph = std::make_shared<CoalaMlpGraph>(total_nodes_count);
    
    //输入层：输入数据的节点
    this->graph->addNode(std::make_shared<Variable>(0));


    //隐含层
    for(int i=0;i<this->hidden_layers_count;i++)
    {
        this->graph->addNode(std::make_shared<OperatorDot>(i*8+1)); // 乘
        this->graph->setForwardEdge(i*8,i*8+1);
        this->graph->setBackwardEdge(i*8+1,i*8);

        this->graph->addNode(std::make_shared<Variable>(i*8+2)); // W
        this->graph->setForwardEdge(i*8+2,i*8+1);
        this->graph->setBackwardEdge(i*8+1,i*8+2);


        this->graph->addNode(std::make_shared<Variable>(i*8+3)); // ans
        this->graph->setForwardEdge(i*8+1,i*8+3);
        this->graph->setBackwardEdge(i*8+3,i*8+1);

        this->graph->addNode(std::make_shared<OperatorAdd>(i*8+4)); // 加
        this->graph->setForwardEdge(i*8+3,i*8+4);
        this->graph->setBackwardEdge(i*8+4,i*8+3);

        this->graph->addNode(std::make_shared<Variable>(i*8+5)); // B
        this->graph->setForwardEdge(i*8+5,i*8+4);
        this->graph->setBackwardEdge(i*8+4,i*8+5);

        this->graph->addNode(std::make_shared<Variable>(i*8+6)); // ans
        this->graph->setForwardEdge(i*8+4,i*8+6);
        this->graph->setBackwardEdge(i*8+6,i*8+4);

        this->graph->addNode(std::make_shared<OperatorActivation>(i*8+7)); // 激活函数
        this->graph->setForwardEdge(i*8+6,i*8+7);
        this->graph->setBackwardEdge(i*8+7,i*8+6);

        this->graph->addNode(std::make_shared<Variable>(i*8+8)); // ans
        this->graph->setForwardEdge(i*8+7,i*8+8);
        this->graph->setBackwardEdge(i*8+8,i*8+7);
    }

    //输出层
    this->graph->addNode(std::make_shared<OperatorDot>(hidden_layers_count*8+1)); // 乘
    this->graph->setForwardEdge(hidden_layers_count*8,hidden_layers_count*8+1);
    this->graph->setBackwardEdge(hidden_layers_count*8+1,hidden_layers_count*8);

    this->graph->addNode(std::make_shared<Variable>(hidden_layers_count*8+2)); // W
    this->graph->setForwardEdge(hidden_layers_count*8+2,hidden_layers_count*8+1);
    this->graph->setBackwardEdge(hidden_layers_count*8+1,hidden_layers_count*8+2);

    this->graph->addNode(std::make_shared<Variable>(hidden_layers_count*8+3)); // ans
    this->graph->setForwardEdge(hidden_layers_count*8+1,hidden_layers_count*8+3);
    this->graph->setBackwardEdge(hidden_layers_count*8+3,hidden_layers_count*8+1);

    this->graph->addNode(std::make_shared<OperatorAdd>(hidden_layers_count*8+4)); // 加
    this->graph->setForwardEdge(hidden_layers_count*8+3,hidden_layers_count*8+4);
    this->graph->setBackwardEdge(hidden_layers_count*8+4,hidden_layers_count*8+3);

    this->graph->addNode(std::make_shared<Variable>(hidden_layers_count*8+5)); // B
    this->graph->setForwardEdge(hidden_layers_count*8+5,hidden_layers_count*8+4);
    this->graph->setBackwardEdge(hidden_layers_count*8+4,hidden_layers_count*8+5);

    this->graph->addNode(std::make_shared<Variable>(hidden_layers_count*8+6)); // ans
    this->graph->setForwardEdge(hidden_layers_count*8+4,hidden_layers_count*8+6);
    this->graph->setBackwardEdge(hidden_layers_count*8+6,hidden_layers_count*8+4);

    this->graph->addNode(std::make_shared<OperatorActivation>(hidden_layers_count*8+7)); // 激活函数
    this->graph->setForwardEdge(hidden_layers_count*8+6,hidden_layers_count*8+7);
    this->graph->setBackwardEdge(hidden_layers_count*8+7,hidden_layers_count*8+6);

    this->graph->addNode(std::make_shared<Variable>(hidden_layers_count*8+8)); // 预测值
    this->graph->setForwardEdge(hidden_layers_count*8+7,hidden_layers_count*8+8);
    this->graph->setBackwardEdge(hidden_layers_count*8+8,hidden_layers_count*8+7);


    //损失
    this->graph->addNode(std::make_shared<OperatorCost>(hidden_layers_count*8+8+1)); // 损失函数
    this->graph->setForwardEdge(hidden_layers_count*8+8,hidden_layers_count*8+8+1);
    this->graph->setBackwardEdge(hidden_layers_count*8+8+1,hidden_layers_count*8+8);

    this->graph->addNode(std::make_shared<Variable>(hidden_layers_count*8+8+2)); // 真实值
    this->graph->setForwardEdge(hidden_layers_count*8+8+2,hidden_layers_count*8+8+1);
    this->graph->setBackwardEdge(hidden_layers_count*8+8+1,hidden_layers_count*8+8+2);
    
    this->graph->addNode(std::make_shared<Variable>(hidden_layers_count*8+8+3)); // 损失
    this->graph->setForwardEdge(hidden_layers_count*8+8+1,hidden_layers_count*8+8+3);
    this->graph->setBackwardEdge(hidden_layers_count*8+8+3,hidden_layers_count*8+8+1);

   
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

int CoalaMLP::setHiddenLayerActivation(int const layer, COALA_MLP_ACTIVATION const activation_func)
{
    this->hidden_layer_activation_funcs[layer] = activation_func;
    return 0;
}

int CoalaMLP::setOutputLayerActivation(COALA_MLP_ACTIVATION const activation_func)
{
    this->output_layer_activation_func = activation_func;
    return 0;
}

int CoalaMLP::setCostFunction(COALA_MLP_LOSS const cost_func)
{
    this->cost_func = cost_func;
    return 0;
}


int CoalaMLP::setInitializationFunction(COALA_MLP_INITIALIZATION const initialization_func)
{
    this->initialization_func = initialization_func;
    return 0;
}


int CoalaMLP::readyForTraining(void)
{
    Variable * vnode;
    
    //输入层：输入数据的节点
    if(vnode = dynamic_cast<Variable*>(this->graph->getNode(0).get()))
    {
        vnode->setDataSize(this->batch_size,this->input_layer_neurons);
        vnode->setInitializationFunction(COALA_MLP_INITIALIZATION_ZERO);
    }
       
    //---------------------------------------------------------------------
    // Variable 设置
    //---------------------------------------------------------------------
    //隐含层
    for(int i=0;i<this->hidden_layers_count;i++)
    {
        if(vnode = dynamic_cast<Variable*>(this->graph->getNode(i*8+2).get()))
        {
            vnode->setDataSize( dynamic_cast<Variable*>(this->graph->getNode(i*8).get())->getDataCols(), this->hidden_layers_neurons[i] ); // W
            vnode->setInitializationFunction(this->initialization_func);
        }
            
        
        if(vnode = dynamic_cast<Variable*>(this->graph->getNode(i*8+3).get()))
        {
            vnode->setDataSize( this->batch_size, this->hidden_layers_neurons[i] ); // ans
            vnode->setInitializationFunction(COALA_MLP_INITIALIZATION_ZERO);
        }
            
        
        if(vnode = dynamic_cast<Variable*>(this->graph->getNode(i*8+5).get()))
        {
            vnode->setDataSize( this->batch_size, this->hidden_layers_neurons[i] ); // B
            vnode->setInitializationFunction(this->initialization_func);
        }
        
        if(vnode = dynamic_cast<Variable*>(this->graph->getNode(i*8+6).get()))
        {
            vnode->setDataSize( this->batch_size, this->hidden_layers_neurons[i] ); // ans
            vnode->setInitializationFunction(COALA_MLP_INITIALIZATION_ZERO);
        }
        

        
        if(vnode = dynamic_cast<Variable*>(this->graph->getNode(i*8+8).get()))
        {
            vnode->setDataSize( this->batch_size, this->hidden_layers_neurons[i] ); // ans
            vnode->setInitializationFunction(COALA_MLP_INITIALIZATION_ZERO);
        }
            
    }

    //输出层
    if(vnode = dynamic_cast<Variable*>(this->graph->getNode(this->hidden_layers_count*8+2).get()))
    {
        vnode->setDataSize( dynamic_cast<Variable*>(this->graph->getNode(this->hidden_layers_count*8).get())->getDataCols(), this->output_layer_neurons); // W
        vnode->setInitializationFunction(this->initialization_func);
    }
        

    if(vnode = dynamic_cast<Variable*>(this->graph->getNode(this->hidden_layers_count*8+3).get()))
    {
        vnode->setDataSize( this->batch_size, this->output_layer_neurons); // ans
        vnode->setInitializationFunction(COALA_MLP_INITIALIZATION_ZERO);
    }
    
    if(vnode = dynamic_cast<Variable*>(this->graph->getNode(this->hidden_layers_count*8+5).get()))
    {
        vnode->setDataSize( this->batch_size, this->output_layer_neurons); // B
        vnode->setInitializationFunction(this->initialization_func);
    }
        

    if(vnode = dynamic_cast<Variable*>(this->graph->getNode(this->hidden_layers_count*8+6).get()))
    {
        vnode->setDataSize( this->batch_size, this->output_layer_neurons); // ans
        vnode->setInitializationFunction(COALA_MLP_INITIALIZATION_ZERO);
    }
    
    if(vnode = dynamic_cast<Variable*>(this->graph->getNode(this->hidden_layers_count*8+8).get()))
    {
        vnode->setDataSize( this->batch_size, this->output_layer_neurons); // ans
        vnode->setInitializationFunction(COALA_MLP_INITIALIZATION_ZERO);
    }

    //损失
    if(vnode = dynamic_cast<Variable*>(this->graph->getNode(this->hidden_layers_count*8+8+2).get()))
    {
        vnode->setDataSize( this->batch_size, this->output_layer_neurons); // 真实值
        vnode->setInitializationFunction(COALA_MLP_INITIALIZATION_ZERO);
    }
        

    if(vnode = dynamic_cast<Variable*>(this->graph->getNode(this->hidden_layers_count*8+8+3).get()))
    {
        switch(this->cost_func)
        {
            case COALA_MLP_LOSS_MSE:
                vnode->setDataSize(1,1); // 损失
                vnode->setInitializationFunction(COALA_MLP_INITIALIZATION_ZERO);
                break;
            case COALA_MLP_LOSS_CROSS_ENTROPY:
                vnode->setDataSize(1,1); // 损失
                vnode->setInitializationFunction(COALA_MLP_INITIALIZATION_ZERO);
                break;
            default:
                break;
        }
    }
    


    //---------------------------------------------------------------------
    // OperatorActivation 设置
    //---------------------------------------------------------------------
    OperatorActivation * anode;

    //隐含层
    for(int i=0;i<this->hidden_layers_count;i++)
    {
        if(anode = dynamic_cast<OperatorActivation*>(this->graph->getNode(i*8+7).get()))
        {
            anode->setActivationFunc(this->hidden_layer_activation_funcs[i]);
        }
    }
    
    if(anode = dynamic_cast<OperatorActivation*>(this->graph->getNode(this->hidden_layers_count*8+7).get()))
    {
        anode->setActivationFunc(this->output_layer_activation_func);
    }

    this->graph->activating();
    return 0;
}