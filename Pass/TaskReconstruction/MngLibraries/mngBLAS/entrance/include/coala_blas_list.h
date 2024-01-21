#ifndef _COALA_BLAS_LIST_H
#define _COALA_BLAS_LIST_H

#include <string>
#include <unordered_map>

//-----------------------------------------
// BLAS = 1x
// => CBLAS = 11
// => CUBLAS = 12
// => CLBALST = 13
//-----------------------------------------
typedef enum
{
    //==============================================
    // 未找到
    //----------------------------------------------
    NOT_FOUND = 0,

    //==============================================
    // Level 3 => xx3x
    //==============================================
    // GEMM => xx31
    // s = 1, d = 2
    //----------------------------------------------
    COALA_BLAS_CBLAS_SGEMM = 11311, COALA_BLAS_CBLAS_DGEMM = 11312,
    COALA_BLAS_CUBLAS_SGEMM = 12311, COALA_BLAS_CUBLAS_DGEMM = 12312,
    COALA_BLAS_CLBLAST_SGEMM = 13311, COALA_BLAS_CLBLAST_DGEMM = 13312,

    //==============================================
    // Level 2 => xx2x
    //==============================================
    // GEMV => xx21
    // s = 1, d = 2
    //----------------------------------------------
    COALA_BLAS_CBLAS_SGEMV = 11211, COALA_BLAS_CBLAS_DGEMV = 11212,
    COALA_BLAS_CUBLAS_SGEMV = 12211, COALA_BLAS_CUBLAS_DGEMV = 12212,
    COALA_BLAS_CLBLAST_SGEMV = 13211, COALA_BLAS_CLBLAST_DGEMV = 13212
}COALA_BLAS_ROUTINES_CODE;

//与主流CBLAS一致
typedef enum
{
    COALA_MATRIX_ROW_MAJOR = 101,
    COALA_MATRIX_COL_MAJOR = 102
}COALA_BLAS_MATRIX_LAYOUT;

//与主流CBLAS一致
typedef enum
{
    COALA_MATRIX_NOTRANS = 111,
    COALA_MATRIX_TRANS = 112,
    COALA_MATRIX_CONJTRANS = 113
}COALA_BLAS_MATRIX_TRANSPOSE;

extern std::unordered_map<std::string, COALA_BLAS_ROUTINES_CODE> COALA_BLAS_ROUTINES_NAMELIST;

#endif
