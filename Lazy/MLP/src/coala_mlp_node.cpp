#include "coala_mlp_node.h"
#include "coala_mlp_node_op.h"
#include "coala_mlp_node_va.h"


using namespace coala::mlp;
bool coala::mlp::isOperator(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code)
{
    //获得最高位置的数字
    int type = node_type_code;
    while(type >= 10)
    {
        type /= 10;
    }

    if(type == 1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool coala::mlp::isVariable(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code)
{
    //获得最高位置的数字
    int type = node_type_code;
    while(type >= 10)
    {
        type /= 10;
    }

    if(type == 2)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool coala::mlp::isCompute(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code)
{
    //获取个位数字
    int type = node_type_code;
    type %= 10;
    if(type == 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}



/*====================================================================
| Node
======================================================================*/
int Node::setInputNode(std::shared_ptr<Node> const node)
{
    this->input_nodes.push_back(node);
    return 0;
}

int Node::setOutputNode(std::shared_ptr<Node> const node)
{
    return 0;
}

std::shared_ptr<Node> Node::getInputNode(int const index)
{   
    if(index >= this->input_nodes.size())
    {
        return nullptr;
    }

    return this->input_nodes[index];
}

std::shared_ptr<Node> Node::getOutputNode(int const index)
{
    if(index >= this->output_nodes.size())
    {
        return nullptr;
    }

    return this->output_nodes[index];
}

COALA_MLP_GRAPH_NODE_TYPE_CODE Node::getNodeType(void)
{
    return this->type;
}

int Node::getRank(void)
{
    return this->rank;

}

/*====================================================================
| CoalaMlpGraphNodeFactory
======================================================================*/
std::shared_ptr<Node> CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeOp( COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code,int const op_func)
{
    if(!isOperator(node_type_code))
    {
        return nullptr;
    }
    switch(node_type_code)
    {
        //Operator Cost
        case COALA_MLP_GRAPH_OPERATOR_COST_COMPUTE:
            return std::make_shared<OperatorCostCompute>(op_func);
        case COALA_MLP_GRAPH_OPERATOR_COST_GRAD:
            return std::make_shared<OperatorCostGrad>();
        
        //Operator Activate
        case COALA_MLP_GRAPH_OPERATOR_ACTIVATE_COMPUTE:
            return std::make_shared<OperatorActivateCompute>(op_func);
        case COALA_MLP_GRAPH_OPERATOR_ACTIVATE_GRAD:
            return std::make_shared<OperatorActivateGrad>();

        //Operator Plus
        case COALA_MLP_GRAPH_OPERATOR_PLUS_COMPUTE:
            return std::make_shared<OperatorPlusCompute>();
        case COALA_MLP_GRAPH_OPERATOR_PLUS_GRAD1ST:
            return std::make_shared<OperatorPlusGrad1st>();
        case COALA_MLP_GRAPH_OPERATOR_PLUS_GRAD2ND:
            return std::make_shared<OperatorPlusGrad2nd>();

        //Operator Matmul
        case COALA_MLP_GRAPH_OPERATOR_MATMUL_COMPUTE:
            return std::make_shared<OperatorMatmulCompute>();
        case COALA_MLP_GRAPH_OPERATOR_MATMUL_GRAD1ST:
            return std::make_shared<OperatorMatmulGrad1st>();
        case COALA_MLP_GRAPH_OPERATOR_MATMUL_GRAD2ND:   
            return std::make_shared<OperatorMatmulGrad2nd>();
        
        default:
            return nullptr;
    }
    return nullptr;
}



std::shared_ptr<Node> CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeVa( COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func)
{
    if(!isVariable(node_type_code))
    {
        return nullptr;
    }
    switch(node_type_code)
    {   
        //Variable Weight
        case COALA_MLP_GRAPH_VARIABLE_WEIGHT:
            return std::make_shared<VariableWeight>(rows,cols,init_func);

        //Variable Ans
        case COALA_MLP_GRAPH_VARIABLE_ANS:
            return std::make_shared<VariableAns>(rows,cols,init_func);

        //Variable Input
        case COALA_MLP_GRAPH_VARIABLE_INPUT:
            return std::make_shared<VariableInput>(rows,cols,init_func);
        default:
            return nullptr;
    }
    return nullptr;
}