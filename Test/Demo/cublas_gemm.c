#include "cublas_gemm.h"
#include <stdlib.h>

cublasStatus_t cublasCreate(cublasHandle_t * handle)
{
    // printf("Here in cublasCreate\n");
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMatrix (int M, int N, int size, float * A, int ldA, float * devPtrA, int ldPtrA)
{
    // printf("Here in cublasSetMatrix\n"); 
    return CUBLAS_STATUS_SUCCESS;
}


cublasStatus_t cublasSgemm
(   cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float           *alpha,
    const float           *A, int lda,
    const float           *B, int ldb,
    const float           *beta,
    float                 *C, int ldc
)
{
    printf("Here in cublasSgemm\n");
    return 0;
}


cublasStatus_t cublasGetMatrix (int M, int N, int size, float * devPtrA, int ldDevA, float * A, int ldA)
{
    // printf("Here in cublasGetMatrix\n");
    return CUBLAS_STATUS_SUCCESS;
}


void cublasDestroy(cublasHandle_t handle)
{
    // printf("Here in cublasDestroy\n"); 
    return;
}

cudaError_t cudaMalloc(void ** ptr, int size)
{   
    *ptr = (void*)malloc(size);
    printf("Here in cudaMalloc\n");
    return  cudaSuccess;
}

void cudaFree(void * ptr)
{
    free(ptr);
    printf("Here in cudaFree\n");
    return;
}