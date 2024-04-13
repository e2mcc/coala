#ifndef COALA_MLP_TENSOR_H
#define COALA_MLP_TENSOR_H


//----------------------------------------------------------------------------------------------
// Namespace
//----------------------------------------------------------------------------------------------
namespace coala {
namespace mlp {

struct sMATRIX_t
{
    int rows;
    int cols;
    float * data;
};

struct dMATRIX_t
{
    int rows;
    int cols;
    double * data;
};


}//end of namespace mlp
}//end of namespace coala
#endif