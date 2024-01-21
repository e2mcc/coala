#include "coala_blas_list.h"

//-----------------------------------------
// BLAS = 1x
// => CBLAS = 11
// => CUBLAS = 12
// => CLBALST = 13
//-----------------------------------------
std::unordered_map<std::string, COALA_BLAS_ROUTINES_CODE> COALA_BLAS_ROUTINES_NAMELIST = {
    //==============================================
    // Level 3 
    //==============================================
    // GEMM 
    {"cblas_sgemm",COALA_BLAS_CBLAS_SGEMM},{"cblas_dgemm",COALA_BLAS_CBLAS_DGEMM},    //cblas
    {"cublasSgemm",COALA_BLAS_CUBLAS_SGEMM},{"cublasDgemm",COALA_BLAS_CUBLAS_DGEMM},    //cublas
    {"CLBlastSgemm",COALA_BLAS_CLBLAST_SGEMM},{"CLBlastDgemm",COALA_BLAS_CLBLAST_DGEMM},  //CLBlast

    //==============================================
    // Level 2 
    //==============================================
    // GEMV 
    {"cblas_sgemv",COALA_BLAS_CBLAS_SGEMV},{"cblas_dgemv",COALA_BLAS_CBLAS_DGEMV},    //cblas
    {"cublasSgemv",COALA_BLAS_CUBLAS_SGEMV},{"cublasDgemv",COALA_BLAS_CUBLAS_DGEMV},    //cublas
    {"CLBlastSgemv",COALA_BLAS_CLBLAST_SGEMV},{"CLBlastDgemv",COALA_BLAS_CLBLAST_DGEMV}  //CLBlast
};