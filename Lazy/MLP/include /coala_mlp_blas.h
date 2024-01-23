#ifndef COALA_MLP_BLAS_H
#define COALA_MLP_BLAS_H


//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include <cstddef>



int coala_mlp_sgemm
(
    size_t layout,
    size_t transa,
    size_t transb,
    size_t m,
    size_t n,
    size_t k,
    float alpha,
    float *A,
    size_t lda,
    float *B,
    size_t ldb,
    float beta,
    float *C,
    size_t ldc
);


int coala_mlp_sgemv
(
    size_t layout,
    size_t transa,
    size_t m,
    size_t n,
    float alpha,
    float *A,
    size_t lda,
    float *X,
    size_t incx,
    float beta,
    float *Y,
    size_t incy
);


int coala_mlp_saxpy
(
    size_t m, //向量的元素数量
    float alpha, 
    float * X,
    size_t incx, 
    float * Y, 
    size_t incy
);

float coala_mlp_sdot
(
    size_t m,
    float * X,
    size_t incx,
    float * Y,
    size_t incy
);


//两个矩阵的对应元素进行相乘的运算被称为哈达玛积(Hadamard)
int coala_mlp_shadamm
(
    size_t m,
    size_t n,
    float * A,
    size_t lda,
    float * B,
    size_t ldb,
    float * C,
    size_t ldc
);
#endif