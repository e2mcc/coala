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





float coala_mlp_sdot
(
    size_t m,
    float * X,
    size_t incx,
    float * Y,
    size_t incy
)
{
    float result = 0.0f;
    size_t xidx = 0;
    size_t yidx = 0;
    // 只做m次计算
    for(size_t count = 0; count<m; count++)
    {
        result += X[xidx]*Y[yidx];
        yidx += incy;
        xidx += incx;
    }
    return result;
}


//列优先存储
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
)
{
    for (size_t i = 0; i < m; i++) 
    {
        for (size_t j = 0; j < n; j++) 
        {
            C[i + j*ldc] = A[i + j*lda] * B[i + j*ldb];
        }
    }
    return 0;
}
