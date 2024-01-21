#ifndef _COALA_BLAS_BASE_H
#define _COALA_BLAS_BASE_H

typedef enum
{
    COALA_MATRIX_ROW_MAJOR = 101,
    COALA_MATRIX_COL_MAJOR = 102,

    CblasRowMajor=101,
    CblasColMajor=102,

    ClblastRowMajor = 101,
    ClblastColMajor = 102
}COALA_BLAS_MATRIX_LAYOUT;


typedef enum
{
    COALA_MATRIX_NOTRANS = 111,
    COALA_MATRIX_TRANS = 112,
    COALA_MATRIX_CONJTRANS = 113,
    
    CblasNoTrans=111,
    CblasTrans=112,
    CblasConjTrans=113,

    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,

    ClblastNoTrans=111,
    ClblastTrans=112,
    ClblastConjTrans=113
}COALA_BLAS_MATRIX_TRANSPOSE;

#endif