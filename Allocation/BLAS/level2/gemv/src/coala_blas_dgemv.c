#include "coala_blas_gemv.h"
#include <stdio.h>

#ifdef COALA_ENABLE_CUBLAS
#include <cublas_v2.h>
#endif

#ifdef COALA_ENABLE_CLBLAST
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 
#include <clblast_c.h>
#endif


int coala_blas_dgemv
(
    coala_probelist_t * probelist,
    size_t const taskid,
    COALA_BLAS_MATRIX_LAYOUT layout,
    COALA_BLAS_MATRIX_TRANSPOSE transa,
    int m, int n,
    double * alpha,
    double * A,
    int lda,
    double * X,
    int incx,
    double * beta,
    double * Y,
    int incy
)
{
    printf("Here in coala_blas_dgemv\n");

    #ifdef COALA_ENABLE_CUBLAS
    cublasHandle_t handle;
    #endif

    #ifdef COALA_MLP_PREDICT
    probelist->probes[taskid].optimalR = modelPredictDGemv(m,n);
    #endif
    
    //TODO: 这里还需要考虑cublas转置的问题
    int errcode;
    switch (probelist->probes[taskid].optimalR)
    {
        #ifdef COALA_ENABLE_OPENBLAS
        case 0:
            errcode = cblas_dgemv(layout, transa, m, n, alpha, A, lda, X, incx,  beta, Y, incy);
            return errcode;
        #endif

        #ifdef COALA_ENABLE_CUBLAS
        case 1:
            errcode = cublasDgemv(probelist->handle, tansa, m, n, alpha, A, lda, X, incx, beta, Y, incy);
            cudaDeviceSynchronize();
            return errcode;
        #endif

        #ifdef COALA_ENABLE_CLBLAST
        case 2:
            errcode = CLBlastDgemv(layout, transa, m, n, alpha, A, 0, lda, X, 0, incx, beta, Y, 0, incy, probelist->queue, probelist->event);
            return errcode;
        #endif

        default:
            break;
    }

    return 0;
}