#ifndef COALA_MLP_NODE_OP_H
#define COALA_MLP_NODE_OP_H

//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_node.h"
#include "coala_mlp_tensor.h"
#include "coala_mlp_activate.h"
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

    public:
    virtual int lockin() = 0;

    virtual int setOpFunc(int const op_func) = 0;

    
};




//----------------------------------------------------------------------------------------------
// Node Operator Cost: TypeCode is 101
//----------------------------------------------------------------------------------------------
class OperatorCost : public Operator
{
    protected:
    OperatorCost(){}

    public:
    virtual int lockin() = 0;
    virtual int setOpFunc(int const op_func) = 0;
};

// Node Operator Cost Compute: TypeCode is 10100
class OperatorCostCompute : public OperatorCost
{
    protected:
    OperatorCostCompute(){}

    public:
    OperatorCostCompute(COALA_MLP_COST const costfunc);
    int setOpFunc(int const op_func) override;
    int lockin() override;
    
    private:
    int costfunc;
};

// Node Operator Cost Grad: TypeCode is 10101
class OperatorCostGrad : public OperatorCost
{
    protected:
    OperatorCostGrad(){}

    public:
    OperatorCostGrad(COALA_MLP_COST const costfunc);
    int setOpFunc(int const op_func) override;
    int lockin() override;

    private:
    int costfunc;
};

//----------------------------------------------------------------------------------------------
// Node Operator Activate: TypeCode is 102
//----------------------------------------------------------------------------------------------
class OperatorActivate : public Operator
{
    protected:
    OperatorActivate(){}
    
    public:
    virtual int lockin() = 0;
    virtual int setOpFunc(int const op_func) = 0;
    
   
};

// Node Operator Activate Compute: TypeCode is 10200
class OperatorActivateCompute : public OperatorActivate
{
    protected:
    OperatorActivateCompute(){}

    public:
    OperatorActivateCompute(COALA_MLP_ACTIVATE_FUNC const activatefunc);
    int setOpFunc(int const op_func) override;
    int lockin() override;

    private:
    int activatefunc;
};

// Node Operator Activate Grad: TypeCode is 10201
class OperatorActivateGrad : public OperatorActivate
{
    protected:
    OperatorActivateGrad(){}

    public:
    OperatorActivateGrad(COALA_MLP_ACTIVATE_FUNC const activatefunc);
    int setOpFunc(int const op_func) override;
    int lockin() override;

    private:
    int activatefunc;
};

//----------------------------------------------------------------------------------------------
// Node Operator Plus: TypeCode is 103
//----------------------------------------------------------------------------------------------
class OperatorPlus : public Operator
{
    protected:
    OperatorPlus(){}

    public:
    virtual int lockin() = 0;
    virtual int setOpFunc(int const op_func) = 0;
   
};

// Node Operator Plus Compute: TypeCode is 10301
class OperatorPlusCompute : public OperatorPlus
{
    public:
    OperatorPlusCompute(){}
    int setOpFunc(int const op_func) override {return 0;}
    int lockin() override;

};

// Node Operator Plus Grad1st: TypeCode is 10302
class OperatorPlusGrad1st : public OperatorPlus
{
    public:
    OperatorPlusGrad1st(){}
    int setOpFunc(int const op_func) override {return 0;}
    int lockin() override;
};

// Node Operator Plus Grad2nd: TypeCode is 10303
class OperatorPlusGrad2nd : public OperatorPlus
{
    public:
    OperatorPlusGrad2nd(){}
    int setOpFunc(int const op_func) override {return 0;}
    int lockin() override;

};

//----------------------------------------------------------------------------------------------
// Node Operator Matmul: TypeCode is 104
//----------------------------------------------------------------------------------------------
class OperatorMatmul : public Operator
{
    protected:
    OperatorMatmul(){}

    public:
    virtual int lockin() = 0;
    virtual int setOpFunc(int const op_func) = 0;
    
};


// Node Operator Matmul Compute: TypeCode is 10400
class OperatorMatmulCompute : public OperatorMatmul
{
    public:
    OperatorMatmulCompute(){}
    int setOpFunc(int const op_func) override {return 0;}
    int lockin() override;
};

// Node Operator Matmul Grad1st: TypeCode is 10401
class OperatorMatmulGrad1st : public OperatorMatmul
{
    public:
    OperatorMatmulGrad1st(){}
    int setOpFunc(int const op_func) override {return 0;}
    int lockin() override;
};

// Node Operator Matmul Grad2nd: TypeCode is 10402
class OperatorMatmulGrad2nd : public OperatorMatmul
{
    public:
    OperatorMatmulGrad2nd(){}
    int setOpFunc(int const op_func) override {return 0;}
    int lockin() override;
};








}//end of namespace mlp
}//end of namespace coala
#endif