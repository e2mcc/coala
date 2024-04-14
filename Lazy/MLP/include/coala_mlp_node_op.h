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
// Node Operator
//----------------------------------------------------------------------------------------------



class Operator : public coala::mlp::Node
{
    protected:
    Operator(){}

};




//----------------------------------------------------------------------------------------------
// Node Operator Cost: TypeCode is 100
//----------------------------------------------------------------------------------------------
class OperatorCost : public Operator
{
    public:
    OperatorCost();
    void forward();
    void backward();
};

// Node Operator Cost Compute: TypeCode is 1000
class OperatorCostCompute : public OperatorCost
{

};

// Node Operator Cost Grad: TypeCode is 1001
class OperatorCostGrad : public OperatorCost
{

};

//----------------------------------------------------------------------------------------------
// Node Operator Activate: TypeCode is 110
//----------------------------------------------------------------------------------------------
class OperatorActivate : public Operator
{
    public:
    OperatorActivate();
    void forward();
    void backward();
};

// Node Operator Activate Compute: TypeCode is 1100
class OperatorActivateCompute : public OperatorActivate
{

};

// Node Operator Activate Grad: TypeCode is 1101
class OperatorActivateGrad : public OperatorActivate
{

};

//----------------------------------------------------------------------------------------------
// Node Operator Plus: TypeCode is 120
//----------------------------------------------------------------------------------------------
class OperatorPlus : public Operator
{
    public:
    OperatorPlus();
    void forward();
    void backward();
};

// Node Operator Plus Compute: TypeCode is 1200
class OperatorPlusCompute : public OperatorPlus
{

};

// Node Operator Plus Grad1st: TypeCode is 1201
class OperatorPlusGrad1st : public OperatorPlus
{

};

// Node Operator Plus Grad2nd: TypeCode is 1202
class OperatorPlusGrad2nd : public OperatorPlus
{

};

//----------------------------------------------------------------------------------------------
// Node Operator Matmul: TypeCode is 130
//----------------------------------------------------------------------------------------------
class OperatorMatmul : public Operator
{
    public:
    OperatorMatmul();
    void forward();
    void backward();
};


// Node Operator Matmul Compute: TypeCode is 1300
class OperatorMatmulCompute : public OperatorMatmul
{

};
// Node Operator Matmul Grad1st: TypeCode is 1301
class OperatorMatmulGrad1st : public OperatorMatmul
{

};

// Node Operator Matmul Grad2nd: TypeCode is 1302
class OperatorMatmulGrad2nd : public OperatorMatmul
{

};








}//end of namespace mlp
}//end of namespace coala
#endif