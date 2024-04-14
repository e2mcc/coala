#include "coala_mlp_node.h"
#include "coala_mlp_node_op.h"
#include "coala_mlp_node_var.h"


using namespace coala::mlp;
bool isOperator(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code)
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

bool isVariable(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code)
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

/*====================================================================
| CoalaMlpGraphNodeFactory
======================================================================*/
std::shared_ptr<Node> CoalaMlpGraphNodeFactory::createACoalaMlpGraphNode( COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code)
{
    switch(node_type_code)
    {
        //Operator Cost
        case COALA_MLP_GRAPH_OPERATOR_COST_COMPUTE:
            return std::make_shared<OperatorCostCompute>();
        case COALA_MLP_GRAPH_OPERATOR_COST_GRAD:
            return std::make_shared<OperatorCostGrad>();
        
        //Operator Activate
        case COALA_MLP_GRAPH_OPERATOR_ACTIVATE_COMPUTE:
            return std::make_shared<OperatorActivateCompute>();
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
        
        //Variable Weight
        case COALA_MLP_GRAPH_VARIABLE_WEIGHT:
            return std::make_shared<VariableWeight>();

        //Variable Ans
        case COALA_MLP_GRAPH_VARIABLE_ANS:
            return std::make_shared<VariableAns>();

        //Variable Input
        case COALA_MLP_GRAPH_VARIABLE_INPUT:
            return std::make_shared<VariableInput>();
        default:
            return nullptr;
    }
    return nullptr;
}