#ifndef COALA_MLP_NODE_H
#define COALA_MLP_NODE_H

//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_tensor.h"
#include "coala_mlp_initialization.h"
#include "coala_mlp_activation.h"
#include "coala_mlp_cost.h"
#include <string>
#include <vector>


//----------------------------------------------------------------------------------------------
// Namespace
//----------------------------------------------------------------------------------------------
namespace coala {
namespace mlp {

//----------------------------------------------------------------------------------------------
// Node Type Code
//   _____________________________________________
//   |   x   |   x   |   x   |    x    |    x    |
//   ---------------------------------------------
//   |  OP   |     Type      | Compute and Grad  |
//   |  VAR  |     Type      |                   |
//   ---------------------------------------------
//
//----------------------------------------------------------------------------------------------
typedef enum
{   
    //Node
    COALA_MLP_GRAPH_NODE = 0,
    COALA_MLP_GRAPH_NODE_OP = 1,
    COALA_MLP_GRAPH_NODE_VARIABLE = 2,
    

    //Operator Cost
    COALA_MLP_GRAPH_OPERATOR_COST           = 101,
    COALA_MLP_GRAPH_OPERATOR_COST_COMPUTE   = 10100,
    COALA_MLP_GRAPH_OPERATOR_COST_GRAD      = 10101,
    

    //Operator Activate
    COALA_MLP_GRAPH_OPERATOR_ACTIVATE           = 102,
    COALA_MLP_GRAPH_OPERATOR_ACTIVATE_COMPUTE   = 10200,
    COALA_MLP_GRAPH_OPERATOR_ACTIVATE_GRAD      = 10201,
    
    // Operator Plus
    COALA_MLP_GRAPH_OPERATOR_PLUS              = 103,
    COALA_MLP_GRAPH_OPERATOR_PLUS_COMPUTE      = 10300,
    COALA_MLP_GRAPH_OPERATOR_PLUS_GRAD1ST      = 10301,
    COALA_MLP_GRAPH_OPERATOR_PLUS_GRAD2ND      = 10302,
    

    // Operator MATMUL
    COALA_MLP_GRAPH_OPERATOR_MATMUL              = 104,
    COALA_MLP_GRAPH_OPERATOR_MATMUL_COMPUTE      = 10400,
    COALA_MLP_GRAPH_OPERATOR_MATMUL_GRAD1ST      = 10401,
    COALA_MLP_GRAPH_OPERATOR_MATMUL_GRAD2ND      = 10402,

}COALA_MLP_GRAPH_NODE_TYPE_CODE;

bool isOperator(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code);
bool isVariable(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code);

//----------------------------------------------------------------------------------------------
// Node
//----------------------------------------------------------------------------------------------
class Node
{

};




}//end of namespace mlp
}//end of namespace coala
#endif