#include "coala_mlp_node_va.h"



using namespace coala::mlp;

//----------------------------------------------------------------------------------------------
// Node Variable Weight
//----------------------------------------------------------------------------------------------
VariableWeight::VariableWeight(int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func)
{
    this->metadata_rows = rows;
    this->metadata_cols = cols;
    this->init_func = init_func;
}

int VariableWeight::setInitFunc(COALA_MLP_INITIALIZE_FUNC const init_func)
{
    this->init_func = init_func;
    return 0;
}

int VariableWeight::setMetaDataRows(int const rows)
{
    this->metadata_rows = rows;
}

int VariableWeight::setMetaDataCols(int const cols)
{
    this->metadata_cols = cols;
    return 0;
}

int VariableWeight::setShape(int const rows, int const cols)
{
    VariableWeight::setMetaDataRows(rows);
    VariableWeight::setMetaDataCols(cols);
    return 0;
}

int VariableWeight::lockin(void)
{
    this->metadata.rows = this->metadata_rows;
    this->metadata.cols = this->metadata_cols;
    this->metadata.data = new float[this->metadata_rows * this->metadata_cols];

    switch(this->init_func)
    {
        case COALA_MLP_INITIALIZE_ZERO:
            coala_mlp_szero(this->metadata.data, this->metadata_rows*this->metadata_cols);
            break;

        case COALA_MLP_INITIALIZE_ONES:
            coala_mlp_sones(this->metadata.data, this->metadata_rows*this->metadata_cols);
            break;
        
        case COALA_MLP_INITIALIZE_RANDOM:
            coala_mlp_srandom(this->metadata.data, this->metadata_rows*this->metadata_cols, 0);
            break;
        
        case COALA_MLP_INITIALIZE_XAVIER:
            coala_mlp_sxavier(this->metadata.data, this->metadata_rows, this->metadata_cols, 0);
            break;
        
        case COALA_MLP_INITIALIZE_HE:
            coala_mlp_she(this->metadata.data, this->metadata_cols, 0);
            break;
        default:
            break;
    }
    return 0;
}

//----------------------------------------------------------------------------------------------
// Node Variable Ans
//----------------------------------------------------------------------------------------------
VariableAns::VariableAns(int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func)
{
    this->metadata_rows = rows;
    this->metadata_cols = cols;
    this->init_func = init_func;
}

int VariableAns::setInitFunc(COALA_MLP_INITIALIZE_FUNC const init_func)
{
    this->init_func = init_func;
    return 0;
}

int VariableAns::setMetaDataRows(int const rows)
{
    this->metadata_rows = rows;
    return 0;
}

int VariableAns::setMetaDataCols(int const cols)
{
    this->metadata_cols = cols;
    return 0;
}

int VariableAns::setShape(int const rows, int const cols)
{
    VariableAns::setMetaDataRows(rows);
    VariableAns::setMetaDataCols(cols);
    return 0;
}

int VariableAns::lockin(void)
{
    this->metadata.rows = this->metadata_rows;
    this->metadata.cols = this->metadata_cols;
    this->metadata.data = new float[this->metadata_rows * this->metadata_cols];

    switch(this->init_func)
    {
        case COALA_MLP_INITIALIZE_ZERO:
            coala_mlp_szero(this->metadata.data, this->metadata_rows*this->metadata_cols);
            break;

        case COALA_MLP_INITIALIZE_ONES:
            coala_mlp_sones(this->metadata.data, this->metadata_rows*this->metadata_cols);
            break;
        
        case COALA_MLP_INITIALIZE_RANDOM:
            coala_mlp_srandom(this->metadata.data, this->metadata_rows*this->metadata_cols, 0);
            break;
        
        case COALA_MLP_INITIALIZE_XAVIER:
            coala_mlp_sxavier(this->metadata.data, this->metadata_rows, this->metadata_cols, 0);
            break;
        
        case COALA_MLP_INITIALIZE_HE:
            coala_mlp_she(this->metadata.data, this->metadata_cols, 0);
            break;
        default:
            break;
    }
    return 0;
}

//----------------------------------------------------------------------------------------------
// Node Variable Input
//----------------------------------------------------------------------------------------------
VariableInput::VariableInput(int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func)
{
    this->metadata_rows = rows;
    this->metadata_cols = cols;
    this->init_func = init_func;
}

int VariableInput::setInitFunc(COALA_MLP_INITIALIZE_FUNC const init_func)
{
    this->init_func = init_func;
    return 0;
}

int VariableInput::setMetaDataRows(int const rows)
{
    this->metadata_rows = rows;
    return 0;
}

int VariableInput::setMetaDataCols(int const cols)
{
    this->metadata_cols = cols;
    return 0;
}

int VariableInput::setShape(int const rows, int const cols)
{
    VariableInput::setMetaDataRows(rows);
    VariableInput::setMetaDataCols(cols);
    return 0;
}

int VariableInput::lockin(void)
{
    this->metadata.rows = this->metadata_rows;
    this->metadata.cols = this->metadata_cols;
    this->metadata.data = nullptr;
    return 0;
}