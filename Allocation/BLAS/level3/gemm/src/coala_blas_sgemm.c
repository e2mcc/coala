#include "coala_blas_gemm.h"
#include <stdio.h>

#ifdef COALA_ENABLE_CUBLAS
#include <cublas_v2.h>
#endif

#ifdef COALA_ENABLE_CLBLAST
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 
#include <clblast_c.h>
#endif

int coala_blas_sgemm
(   
    coala_probelist_t * probelist,
    size_t const taskid,
    COALA_BLAS_MATRIX_LAYOUT layout,
    COALA_BLAS_MATRIX_TRANSPOSE transa, COALA_BLAS_MATRIX_TRANSPOSE transb,
    int m, int n, int k,
    float * alpha,
    float * A,
    int lda,
    float * B,
    int ldb,
    float * beta,
    float * C,
    int ldc
)
{   
    printf("Here in coala_blas_sgemm\n");

    #ifdef COALA_ENABLE_CUBLAS
    cublasHandle_t handle;
    #endif
    
    #ifdef COALA_MLP_PREDICT
    probelist->probes[taskid].optimalR = modelPredictSGemm(m,n,k);
    #endif

    //TODO: 这里还需要考虑到多种转置的问题
    int errcode;
    switch (probelist->probes[taskid].optimalR)
    {
        #ifdef COALA_ENABLE_OPENBLAS
        case 0:
            errcode = cblas_sgemm(layout, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            return errcode;
        #endif

        #ifdef COALA_ENABLE_CUBLAS
        case 1:
            errcode = cublasSgemm(probelist->probes[taskid].handle, tansa, tansb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            cudaDeviceSynchronize();
            return errcode;
        #endif

        #ifdef COALA_ENABLE_CLBLAST
        case 2:
            errcode = CLBlastSgemm(layout, transa, transb, m, n, k, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc, probelist->probes[taskid].queue, probelist->probes[taskid].event);
            return errcode;
        #endif

        default:
            break;
    }

    return 0;
}