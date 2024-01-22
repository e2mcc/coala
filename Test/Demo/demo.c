#include "cublas_gemm.h"
#include <stdio.h>
#include <stdlib.h>

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))




int main (int argc,char **argv)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    float * devPtrA = NULL;
    float * devPtrB = NULL;
    float * devPtrC = NULL;
    devPtrA = (float *)malloc(M * K * sizeof (*devPtrA));
    devPtrB = (float *)malloc(K * N * sizeof (*devPtrB));
    devPtrC = (float *)malloc(M * N * sizeof (*devPtrC));
    
    float alpha = 1.0;
    float beta = 0.7;

    float * A = NULL;
    float * B = NULL;
    float * C = NULL;

    A = (float *)malloc(M * K * sizeof (*A));
    B = (float *)malloc(K * N * sizeof (*B));
    C = (float *)malloc(M * N * sizeof (*C));
    
    if (!A||!B||!C) 
    {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    for (int j = 1; j <= K; j++) {
        for (int i = 1; i <= M; i++) {
            A[IDX2F(i,j,M)] = (float)((i-1) * K + j);
        }
    }

    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= K; i++) {
            B[IDX2F(i,j,K)] = (float)((i-1) * N + j);
        }
    }

    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= M; i++) {
            C[IDX2F(i,j,M)] = (float)((i-1) * N + j);
        }
    }

    cudaStat = cudaMalloc((void**)&devPtrA, M*K*sizeof(*A));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }

    cudaStat = cudaMalloc((void**)&devPtrB, K*N*sizeof(*B));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        cudaFree (devPtrA);
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc((void**)&devPtrC, M*N*sizeof(*C));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        cudaFree (devPtrA);
        cudaFree (devPtrB);
        return EXIT_FAILURE;
    }


    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        cudaFree (devPtrA);
        cudaFree (devPtrB);
        cudaFree (devPtrC);
        return EXIT_FAILURE;
    }


    stat = cublasSetMatrix (M, K, sizeof(*A), A, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cudaFree (devPtrB);
        cudaFree (devPtrC);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (K, N, sizeof(*B), C, K, devPtrB, K);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cudaFree (devPtrB);
        cudaFree (devPtrC);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (M, N, sizeof(*C), C, M, devPtrC, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cudaFree (devPtrB);
        cudaFree (devPtrC);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }


    stat = cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M,N,K,
        &alpha,
        devPtrA, M,
        devPtrB, K,
        &beta,
        devPtrC, M
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("cublasSgemm failed");
        cudaFree (devPtrA);
        cudaFree (devPtrB);
        cudaFree (devPtrC);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }


    stat = cublasGetMatrix (M, N, sizeof(*C), devPtrC, M, C, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cudaFree (devPtrB);
        cudaFree (devPtrC);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaFree (devPtrA);
    cudaFree (devPtrB);
    cudaFree (devPtrC);
    cublasDestroy(handle);

    free(A);
    free(B);

    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= M; i++) {
            printf ("%7.0f", C[IDX2F(i,j,M)]);
        }
        printf ("\n");
    }

    free(C);

    return EXIT_SUCCESS;
}