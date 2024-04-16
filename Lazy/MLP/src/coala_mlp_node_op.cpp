#include "coala_mlp_node_op.h"



using namespace coala::mlp;

//----------------------------------------------------------------------------------------------
// Node Operator
//----------------------------------------------------------------------------------------------
void Operator::setOpFunc(int const op_func)
{
    this->op_func = op_func;
    return;
}

int Operator::getOpFunc(void)
{
    return this->op_func;
}

//----------------------------------------------------------------------------------------------
// Node Operator Cost
//----------------------------------------------------------------------------------------------
OperatorCostCompute::OperatorCostCompute(COALA_MLP_COST const costfunc)
{
    setOpFunc(op_func);
}



int OperatorCostCompute::lockin()
{
    //检查输入输出节点的数量是否正确
    if(this->input_nodes.size() != 2)
    {
        return 1;
    }
    if(this->output_nodes.size() != 1)
    {
        return 2;
    }

    //检查输入节点的类型是否正确
    if(!isVariable(this->input_nodes[0]->getNodeType()))
    {
        return 3;
    }
    if(!isVariable(this->input_nodes[1]->getNodeType()))
    {
        return 3;
    }
    //检查输出节点的类型是否正确
    if(this->output_nodes[0]->getNodeType() != COALA_MLP_GRAPH_VARIABLE_ANS)
    {
        return 4;
    }

    // 把输入节点放第一个输入节点
    if(this->input_nodes[1]->getNodeType() == COALA_MLP_GRAPH_VARIABLE_INPUT)
    {
        std::shared_ptr<Node> temp = this->input_nodes[0];
        this->input_nodes[0] = this->input_nodes[1];
        this->input_nodes[1] = temp;
    }

    return 0;
}



OperatorCostGrad::OperatorCostGrad(COALA_MLP_COST const costfunc)
{
    setOpFunc(costfunc);
}



//----------------------------------------------------------------------------------------------
// Node Operator Activate
//----------------------------------------------------------------------------------------------
OperatorActivateCompute::OperatorActivateCompute(COALA_MLP_ACTIVATE_FUNC const activatefunc)
{
    setOpFunc(activatefunc);
}



int OperatorActivateCompute::lockin()
{
    //检查输入输出节点的数量是否正确
    if(this->input_nodes.size() != 1)
    {
        return 1;
    }
    if(this->output_nodes.size() != 1)
    {
        return 2;
    }

    //检查输入节点的类型是否正确
    if(!isVariable(this->input_nodes[0]->getNodeType()))
    {
        return 3;
    }
    //检查输出节点的类型是否正确
    if(this->output_nodes[0]->getNodeType() != COALA_MLP_GRAPH_VARIABLE_ANS)
    {
        return 4;
    }

    return 0;
}

OperatorActivateGrad::OperatorActivateGrad(COALA_MLP_ACTIVATE_FUNC const activatefunc)
{
    setOpFunc(activatefunc);
}



//----------------------------------------------------------------------------------------------
// Node Operator Plus
//----------------------------------------------------------------------------------------------
int OperatorPlusCompute::lockin()
{
    //检查输入输出节点的数量是否正确
    if(this->input_nodes.size() != 2)
    {
        return 1;
    }
    if(this->output_nodes.size() != 1)
    {
        return 2;
    }

    //检查输入节点的类型是否正确
    if(!isVariable(this->input_nodes[0]->getNodeType()))
    {
        return 3;
    }
    if(!isVariable(this->input_nodes[1]->getNodeType()))
    {
        return 3;
    }
    //检查输出节点的类型是否正确
    if(this->output_nodes[0]->getNodeType() != COALA_MLP_GRAPH_VARIABLE_ANS)
    {
        return 4;
    }

    // 把权重节点放第二个输入节点
    if(this->input_nodes[0]->getNodeType() == COALA_MLP_GRAPH_VARIABLE_WEIGHT)
    {
        std::shared_ptr<Node> temp = this->input_nodes[0];
        this->input_nodes[0] = this->input_nodes[1];
        this->input_nodes[1] = temp;
    }

    return 0;
}


//----------------------------------------------------------------------------------------------
// Node Operator Matmul
//----------------------------------------------------------------------------------------------
int OperatorMatmulCompute::lockin()
{
    //检查输入输出节点的数量是否正确
    if(this->input_nodes.size() != 2)
    {
        return 1;
    }
    if(this->output_nodes.size() != 1)
    {
        return 2;
    }

    //检查输入节点的类型是否正确
    if(!isVariable(this->input_nodes[0]->getNodeType()))
    {
        return 3;
    }
    if(!isVariable(this->input_nodes[1]->getNodeType()))
    {
        return 3;
    }
    //检查输出节点的类型是否正确
    if(this->output_nodes[0]->getNodeType() != COALA_MLP_GRAPH_VARIABLE_ANS)
    {
        return 4;
    }

    // 把权重节点放第二个输入节点
    if(this->input_nodes[0]->getNodeType() == COALA_MLP_GRAPH_VARIABLE_WEIGHT)
    {
        std::shared_ptr<Node> temp = this->input_nodes[0];
        this->input_nodes[0] = this->input_nodes[1];
        this->input_nodes[1] = temp;
    }

    return 0;
}