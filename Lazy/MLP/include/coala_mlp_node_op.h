#ifndef COALA_MLP_NODE_OP_H
#define COALA_MLP_NODE_OP_H

//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_node.h"
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
// Node Operator: TypeCode is 1
//----------------------------------------------------------------------------------------------
class Operator : public coala::mlp::Node
{
    protected:
    Operator(){}

};




//----------------------------------------------------------------------------------------------
// Node Operator Cost: TypeCode is 101
//----------------------------------------------------------------------------------------------
class OperatorCost : public Operator
{
    public:
    OperatorCost();
   
};

// Node Operator Cost Compute: TypeCode is 10100
class OperatorCostCompute : public OperatorCost
{
    public:
    OperatorCostCompute();
};

// Node Operator Cost Grad: TypeCode is 10101
class OperatorCostGrad : public OperatorCost
{
    public:
    OperatorCostGrad();
};

//----------------------------------------------------------------------------------------------
// Node Operator Activate: TypeCode is 102
//----------------------------------------------------------------------------------------------
class OperatorActivate : public Operator
{
    public:
    OperatorActivate();
   
};

// Node Operator Activate Compute: TypeCode is 10200
class OperatorActivateCompute : public OperatorActivate
{
    public:
    OperatorActivateCompute();
};

// Node Operator Activate Grad: TypeCode is 10201
class OperatorActivateGrad : public OperatorActivate
{
    public:
    OperatorActivateGrad();
};

//----------------------------------------------------------------------------------------------
// Node Operator Plus: TypeCode is 103
//----------------------------------------------------------------------------------------------
class OperatorPlus : public Operator
{
    public:
    OperatorPlus();
   
};

// Node Operator Plus Compute: TypeCode is 10301
class OperatorPlusCompute : public OperatorPlus
{
    public:
    OperatorPlusCompute();
};

// Node Operator Plus Grad1st: TypeCode is 10302
class OperatorPlusGrad1st : public OperatorPlus
{
    public:
    OperatorPlusGrad1st();
};

// Node Operator Plus Grad2nd: TypeCode is 10303
class OperatorPlusGrad2nd : public OperatorPlus
{
    public:
    OperatorPlusGrad2nd();
};

//----------------------------------------------------------------------------------------------
// Node Operator Matmul: TypeCode is 104
//----------------------------------------------------------------------------------------------
class OperatorMatmul : public Operator
{
    public:
    OperatorMatmul();

};


// Node Operator Matmul Compute: TypeCode is 10400
class OperatorMatmulCompute : public OperatorMatmul
{
    public:
    OperatorMatmulCompute();
};

// Node Operator Matmul Grad1st: TypeCode is 10401
class OperatorMatmulGrad1st : public OperatorMatmul
{
    public:
    OperatorMatmulGrad1st();
};

// Node Operator Matmul Grad2nd: TypeCode is 10402
class OperatorMatmulGrad2nd : public OperatorMatmul
{
    public:
    OperatorMatmulGrad2nd();
};








}//end of namespace mlp
}//end of namespace coala
#endif