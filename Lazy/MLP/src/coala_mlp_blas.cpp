//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_blas.h"
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
)
{
    return 0;
}




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
)
{
    return 0;
}



// Y = alpha*X+Y
int coala_mlp_saxpy
(
    size_t m, //向量的元素数量
    float alpha, 
    float * X,
    size_t incx, 
    float * Y, 
    size_t incy
)
{
    size_t xidx = 0;
    size_t yidx = 0;
    // 只做m次计算
    for(size_t count = 0; count<m; count++)
    {
        Y[yidx] += alpha*X[xidx];
        yidx += incy;
        xidx += incx;
    }
    return 0;
}