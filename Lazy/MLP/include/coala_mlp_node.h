#ifndef COALA_MLP_NODE_H
#define COALA_MLP_NODE_H

//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_tensor.h"
#include "coala_mlp_initialization.h"
#include "coala_mlp_activation.h"
#include "coala_mlp_loss.h"
#include <string>
#include <vector>

//----------------------------------------------------------------------------------------------
// Namespace
//----------------------------------------------------------------------------------------------
namespace coala {
namespace mlp {


//----------------------------------------------------------------------------------------------
// Node
//----------------------------------------------------------------------------------------------
class Node
{
    public:
    Node() {}
    virtual ~Node() {}
    int id;
    std::vector<std::shared_ptr<Node>> backward_nodes;
    std::vector<std::shared_ptr<Node>> forward_nodes;

    int addForwardNode(std::shared_ptr<Node> node);
    int addBackwardNode(std::shared_ptr<Node> node);
    virtual void activating(void) = 0;
};


//----------------------------------------------------------------------------------------------
// Operator
//----------------------------------------------------------------------------------------------
class Operator : public Node
{   
    protected:
    Operator() {}

    public:
    virtual ~Operator() {}
    Operator(int const id) {this->id = id;} 
    
    //前向传播时的输入输出
    std::vector<std::shared_ptr<Node>> forward_input_nodes;
    std::vector<std::shared_ptr<Node>> forward_output_nodes;

    virtual void activating(void) = 0;
    virtual void forward(void) = 0;
    virtual void backward(void) = 0;
};

// 加法操作
class OperatorAdd : public Operator
{
    protected:
    OperatorAdd() {}

    public:
    virtual ~OperatorAdd() {}
    OperatorAdd(int const id) {this->id = id;}

    void activating(void) override;

    void forward(void) override;
    void backward(void) override;
};

// 点乘操作
class OperatorDot : public Operator
{
    protected:
    OperatorDot() {}
    
    public:
    virtual ~OperatorDot() {}
    OperatorDot(int const id) {this->id = id;}

    void activating(void) override;

    void forward(void) override;
    void backward(void) override;

};

//  激活函数操作
class OperatorActivation : public Operator
{
    protected:
    OperatorActivation() {}

    private:
    COALA_MLP_ACTIVATION activation_func;

    public:
    virtual ~OperatorActivation() {}
    OperatorActivation(int const id) {this->id = id;}
    void setActivationFunc(COALA_MLP_ACTIVATION const activation_rank);

    void activating(void) override;
    void forward(void) override;
    void backward(void) override;
};

// 损失函数操作
class OperatorCost: public Operator
{
    protected:
    OperatorCost() {}

    private:
    COALA_MLP_LOSS cost_func;

    public:
    virtual ~OperatorCost() {}
    OperatorCost(int const id) {this->id = id;}
    void setCostFunc(COALA_MLP_LOSS const cost_rank);

    void activating(void) override;
    void forward(void) override;
    void backward(void) override;
};

//----------------------------------------------------------------------------------------------
// Variable
//----------------------------------------------------------------------------------------------
class Variable : public Node
{
    protected:
    Variable() {}

    private:
    sMATRIX_t data;
    COALA_MLP_INITIALIZATION initialization_func;

    public:
    virtual ~Variable() {}
    Variable(int const id);

    int setDataSize(int const rows, int const cols);
    int setInitializationFunction(COALA_MLP_INITIALIZATION const initialization_func);
    int getDataRows();
    int getDataCols();
    float * getMetaData();
    void activating(void) override;
};


}//end of namespace mlp
}//end of namespace coala
#endif