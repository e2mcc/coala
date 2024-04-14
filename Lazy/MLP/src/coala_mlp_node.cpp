#include "coala_mlp_node.h"


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

