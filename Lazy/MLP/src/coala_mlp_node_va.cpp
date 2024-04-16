#include "coala_mlp_node_va.h"



using namespace coala::mlp;

//----------------------------------------------------------------------------------------------
// Node Variable Weight
//----------------------------------------------------------------------------------------------
VariableWeight::VariableWeight(int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func)
{
    this->data_rows = rows;
    this->data_cols = cols;
    this->init_func = init_func;
    this->metadata = std::make_shared<sMATRIX_t>();
    this->metadata->rows = this->data_rows;
    this->metadata->cols = this->data_cols;
    this->graddata = std::make_shared<sMATRIX_t>();
    this->graddata->rows = this->data_rows;
    this->graddata->cols = this->data_cols;
}

int VariableWeight::setInitFunc(COALA_MLP_INITIALIZE_FUNC const init_func)
{
    this->init_func = init_func;
    return 0;
}

int VariableWeight::setDataRows(int const rows)
{
    this->data_rows = rows;
}

int VariableWeight::setDataCols(int const cols)
{
    this->data_cols = cols;
    return 0;
}

int VariableWeight::setDataShape(int const rows, int const cols)
{
    VariableWeight::setDataRows(rows);
    VariableWeight::setDataCols(cols);
    return 0;
}

int VariableWeight::lockin(void)
{
    this->metadata->data = new float[this->data_rows * this->data_cols];

    switch(this->init_func)
    {
        case COALA_MLP_INITIALIZE_ZERO:
            coala_mlp_szero(this->metadata->data, this->data_rows*this->data_cols);
            break;

        case COALA_MLP_INITIALIZE_ONES:
            coala_mlp_sones(this->metadata->data, this->data_rows*this->data_cols);
            break;
        
        case COALA_MLP_INITIALIZE_RANDOM:
            coala_mlp_srandom(this->metadata->data, this->data_rows*this->data_cols, 0);
            break;
        
        case COALA_MLP_INITIALIZE_XAVIER:
            coala_mlp_sxavier(this->metadata->data, this->data_rows, this->data_cols, 0);
            break;
        
        case COALA_MLP_INITIALIZE_HE:
            coala_mlp_she(this->metadata->data, this->data_cols, 0);
            break;
        default:
            break;
    }

    // Weight 节点的 grad_required 必然是 true
    if(this->grad_required)
    {
        this->graddata->data = new float[this->graddata->rows * this->graddata->cols];
        coala_mlp_szero(this->graddata->data, this->graddata->rows*this->graddata->cols);
    }

    return 0;
}

//----------------------------------------------------------------------------------------------
// Node Variable Ans
//----------------------------------------------------------------------------------------------
VariableAns::VariableAns(int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func)
{
    this->data_rows = rows;
    this->data_cols = cols;
    this->init_func = init_func;
    this->metadata = std::make_shared<sMATRIX_t>();
    this->metadata->rows = this->data_rows;
    this->metadata->cols = this->data_cols;
    this->graddata = std::make_shared<sMATRIX_t>();
    this->graddata->rows = this->data_rows;
    this->graddata->cols = this->data_cols;
}

int VariableAns::setInitFunc(COALA_MLP_INITIALIZE_FUNC const init_func)
{
    this->init_func = init_func;
    return 0;
}

int VariableAns::setDataRows(int const rows)
{
    this->data_rows = rows;
    return 0;
}

int VariableAns::setDataCols(int const cols)
{
    this->data_cols = cols;
    return 0;
}

int VariableAns::setDataShape(int const rows, int const cols)
{
    VariableAns::setDataRows(rows);
    VariableAns::setDataCols(cols);
    return 0;
}

int VariableAns::lockin(void)
{
    this->metadata->data = new float[this->data_rows * this->data_cols];

    switch(this->init_func)
    {
        case COALA_MLP_INITIALIZE_ZERO:
            coala_mlp_szero(this->metadata->data, this->data_rows*this->data_cols);
            break;

        case COALA_MLP_INITIALIZE_ONES:
            coala_mlp_sones(this->metadata->data, this->data_rows*this->data_cols);
            break;
        
        case COALA_MLP_INITIALIZE_RANDOM:
            coala_mlp_srandom(this->metadata->data, this->data_rows*this->data_cols, 0);
            break;
        
        case COALA_MLP_INITIALIZE_XAVIER:
            coala_mlp_sxavier(this->metadata->data, this->data_rows, this->data_cols, 0);
            break;
        
        case COALA_MLP_INITIALIZE_HE:
            coala_mlp_she(this->metadata->data, this->data_cols, 0);
            break;
        default:
            break;
    }

    //如果是叶子节点,则肯定不需要梯度
    if(this->output_nodes.size() == 0)
    {
        this->grad_required = false;
    }

    if(this->grad_required)//如果需要求梯度
    {
        this->graddata->data = new float[this->graddata->rows * this->graddata->cols];
        coala_mlp_szero(this->graddata->data, this->graddata->rows*this->graddata->cols);
    }
    else //如果不需要求梯度，要么是最终正向计算结果，要么是求导结果
    {
        // 如果不需要求梯度,且是最终正向计算结果
        if( isCompute( this->getInputNode(0)->getNodeType()) ) return 0;

        // 如果不需要求梯度,且是求导结果
        switch( this->getInputNode(0)->getNodeType() )
        {
            case COALA_MLP_GRAPH_OPERATOR_COST_GRAD:
            case COALA_MLP_GRAPH_OPERATOR_ACTIVATE_GRAD:
            case COALA_MLP_GRAPH_OPERATOR_MATMUL_GRAD1ST:
            case COALA_MLP_GRAPH_OPERATOR_PLUS_GRAD1ST:
                // 把本节点的结果数据发给需求梯度的节点
                this->metadata = dynamic_cast<Variable*>(this->getInputNode(0)->getInputNode(0).get())->getGradData();
                break;

            case COALA_MLP_GRAPH_OPERATOR_MATMUL_GRAD2ND:
            case COALA_MLP_GRAPH_OPERATOR_PLUS_GRAD2ND:
                // 把本节点的结果数据发给需求梯度的节点
                this->metadata = dynamic_cast<Variable*>(this->getInputNode(0)->getInputNode(1).get())->getGradData();
                break;

            default:
                break;
        }

    }
    return 0;
}

//----------------------------------------------------------------------------------------------
// Node Variable Input
//----------------------------------------------------------------------------------------------
VariableInput::VariableInput(int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func)
{
    this->data_rows = rows;
    this->data_cols = cols;
    this->init_func = init_func;
    this->metadata = std::make_shared<sMATRIX_t>();
    this->metadata->rows = this->data_rows;
    this->metadata->cols = this->data_cols;
    this->graddata = nullptr;
}

int VariableInput::setInitFunc(COALA_MLP_INITIALIZE_FUNC const init_func)
{
    this->init_func = init_func;
    return 0;
}

int VariableInput::setDataRows(int const rows)
{
    this->data_rows = rows;
    return 0;
}

int VariableInput::setDataCols(int const cols)
{
    this->data_cols = cols;
    return 0;
}

int VariableInput::setDataShape(int const rows, int const cols)
{
    VariableInput::setDataRows(rows);
    VariableInput::setDataCols(cols);
    return 0;
}

int VariableInput::lockin(void)
{
    return 0;
}