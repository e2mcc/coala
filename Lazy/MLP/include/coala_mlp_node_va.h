#ifndef COALA_MLP_NODE_VA_H
#define COALA_MLP_NODE_VA_H

//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_node.h"
#include "coala_mlp_tensor.h"
#include "coala_mlp_initialize.h"
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
// Variable: TypeCode is 2
//----------------------------------------------------------------------------------------------
class Variable : public coala::mlp::Node
{
    protected:
    Variable(){}

    public:
    int data_rows;
    int data_cols;

    std::shared_ptr<sMATRIX_t> metadata;
    COALA_MLP_INITIALIZE_FUNC init_func;
    virtual int setInitFunc(COALA_MLP_INITIALIZE_FUNC const init_func) = 0;
    std::shared_ptr<sMATRIX_t> getMetaData(void){return this->metadata;}

    bool grad_required = true;
    void setGradRequired(bool yes_or_no) {this->grad_required= yes_or_no;}
    std::shared_ptr<sMATRIX_t> graddata;
    std::shared_ptr<sMATRIX_t> getGradData(void){return this->graddata;}

   

    int getDataRows(void){return this->data_rows;}
    int getDataCols(void){return this->data_cols;}
    virtual int setDataShape(int const rows, int const cols) = 0;
   
   
    virtual int lockin() = 0;
    

};

//----------------------------------------------------------------------------------------------
// VariableWeight: TypeCode is 201
//----------------------------------------------------------------------------------------------
class VariableWeight : public Variable
{
    protected:
    VariableWeight(){}

    private:
    int setDataRows(int const rows);
    int setDataCols(int const cols);

    public:
    VariableWeight(int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func);
    int setInitFunc(COALA_MLP_INITIALIZE_FUNC const init_func) override;
    int setDataShape(int const rows, int const cols) override;
    int lockin() override;
};


//----------------------------------------------------------------------------------------------
// VariableAns: TypeCode is 202
//----------------------------------------------------------------------------------------------
class VariableAns : public Variable
{
    protected:
    VariableAns(){}

    private:
    int setDataRows(int const rows);
    int setDataCols(int const cols);
    
    public:
    VariableAns(int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func);
    int setInitFunc(COALA_MLP_INITIALIZE_FUNC const init_func) override;
    int setDataShape(int const rows, int const cols) override;
    int lockin() override;
    
};

//----------------------------------------------------------------------------------------------
// VariableInput: TypeCode is 203
//----------------------------------------------------------------------------------------------
class VariableInput : public Variable
{
    protected:
    VariableInput(){}

    private:
    int setDataRows(int const rows);
    int setDataCols(int const cols);

    public:
    VariableInput(int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func);
    int setInitFunc(COALA_MLP_INITIALIZE_FUNC const init_func) override;
    int setDataShape(int const rows, int const cols) override;
    int lockin() override;
};




}//end of namespace mlp
}//end of namespace coala
#endif