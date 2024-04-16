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
    sMATRIX_t metadata;
    int metadata_rows;
    int metadata_cols;
    COALA_MLP_INITIALIZE_FUNC init_func;

    virtual int lockin() = 0;

    virtual int setInitFunc(COALA_MLP_INITIALIZE_FUNC const init_func) = 0;
    virtual int setShape(int const rows, int const cols) = 0;
    
};

//----------------------------------------------------------------------------------------------
// VariableWeight: TypeCode is 201
//----------------------------------------------------------------------------------------------
class VariableWeight : public Variable
{
    protected:
    VariableWeight(){}

    private:
    int setMetaDataRows(int const rows);
    int setMetaDataCols(int const cols);

    public:
    VariableWeight(int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func);
    int setInitFunc(COALA_MLP_INITIALIZE_FUNC const init_func) override;
    int setShape(int const rows, int const cols) override;
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
    int setMetaDataRows(int const rows);
    int setMetaDataCols(int const cols);

    public:
    VariableAns(int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func);
    int setInitFunc(COALA_MLP_INITIALIZE_FUNC const init_func) override;
    int setShape(int const rows, int const cols) override;
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
    int setMetaDataRows(int const rows);
    int setMetaDataCols(int const cols);

    public:
    VariableInput(int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func);
    int setInitFunc(COALA_MLP_INITIALIZE_FUNC const init_func) override;
    int setShape(int const rows, int const cols) override;
    int lockin() override;
};




}//end of namespace mlp
}//end of namespace coala
#endif