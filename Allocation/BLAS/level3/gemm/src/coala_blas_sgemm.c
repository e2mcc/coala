#include "coala_blas_gemm.h"
#include <stdio.h>

#ifdef COALA_ENABLE_CUBLAS
#include <cublas_v2.h>
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
    

    //TODO: 这里还需要考虑转置的问题
    int errcode;
    switch (probelist->probes[taskid].optimalR)
    {
        case 0:
            return 0;

        #ifdef COALA_ENABLE_CUBLAS
        case 1:
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return CUBLAS_STATUS_FAILURE;
            errcode = cublasSgemm(handle, tansa, tansb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            cudaDeviceSynchronize();
            return errcode;
        #endif

        #ifdef COALA_ENABLE_OPENCL
        case 2:
            return 0;
        #endif

        default:
            break;
    }

    return 0;
}