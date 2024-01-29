#include "coala_blas_gemm.h"
#include <stdio.h>

#ifdef COALA_ENABLE_CUBLAS
#include <cublas_v2.h>
#endif

#ifdef COALA_ENABLE_CLBLAST
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 
#include <clblast_c.h>
#endif

int coala_blas_dgemm
(   
    coala_probelist_t * probelist,
    size_t const taskid,
    COALA_BLAS_MATRIX_LAYOUT layout,
    COALA_BLAS_MATRIX_TRANSPOSE transa, COALA_BLAS_MATRIX_TRANSPOSE transb,
    int m, int n, int k,
    double * alpha,
    double * A,
    int lda,
    double * B,
    int ldb,
    double * beta,
    double * C,
    int ldc
)
{
    printf("Here in coala_blas_dgemm\n");

    #ifdef COALA_ENABLE_CUBLAS
    cublasHandle_t handle;
    #endif

    #ifdef COALA_MLP_PREDICT
    probelist->probes[taskid].optimalR = modelPredictDGemm(m,n,k);
    #endif
    
    //TODO: 这里还需要考虑转置的问题
    int errcode;
    switch (probelist->probes[taskid].optimalR)
    {
        #ifdef COALA_ENABLE_OPENBLAS
        case 0:
            errcode = cblas_dgemm(layout, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            return errcode;
        #endif

        #ifdef COALA_ENABLE_CUBLAS
        case 1:
            errcode = cublasDgemm(probelist->handle, tansa, tansb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            cudaDeviceSynchronize();
            return errcode;
        #endif

        #ifdef COALA_ENABLE_CLBLAST
        case 2:
            errcode = CLBlastDgemm(layout, transa, transb, m, n, k, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc, probelist->queue, probelist->event);
            return errcode;
        #endif

        default:
            break;
    }

    return 0;
}