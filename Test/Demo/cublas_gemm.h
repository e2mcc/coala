#include <stdio.h>

//--------------------------------------
// CUBLAS API
//--------------------------------------
typedef enum
{
    CUBLAS_OP_N = 101,
    CUBLAS_OP_T = 102,
    CUBLAS_OP_C = 103
}cublasOperation_t;


typedef struct 
{
    int size;
}cublasHandle_t;


typedef enum{
    CUBLAS_STATUS_SUCCESS         =0,
    CUBLAS_STATUS_NOT_INITIALIZED =1,
    CUBLAS_STATUS_ALLOC_FAILED    =3,
    CUBLAS_STATUS_INVALID_VALUE   =7,
    CUBLAS_STATUS_ARCH_MISMATCH   =8,
    CUBLAS_STATUS_MAPPING_ERROR   =11,
    CUBLAS_STATUS_EXECUTION_FAILED=13,
    CUBLAS_STATUS_INTERNAL_ERROR  =14,
    CUBLAS_STATUS_NOT_SUPPORTED   =15,
    CUBLAS_STATUS_LICENSE_ERROR   =16
}cublasStatus_t;


cublasStatus_t cublasCreate(cublasHandle_t * handle);

cublasStatus_t cublasSetMatrix (int M, int N, int size, float * A, int ldA, float * devPtrA, int ldPtrA);

cublasStatus_t cublasSgemm
(   cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float           *alpha,
    const float           *A, int lda,
    const float           *B, int ldb,
    const float           *beta,
    float                 *C, int ldc
);

cublasStatus_t cublasGetMatrix (int M, int N, int size, float * devPtrA, int ldPtrA, float * A, int ldA);


void cublasDestroy(cublasHandle_t handle);


//--------------------------------------
// CUDA API
//--------------------------------------

typedef enum
{
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1
}cudaError_t;


cudaError_t cudaMalloc(void ** ptr, int size);


void cudaFree(void * ptr);