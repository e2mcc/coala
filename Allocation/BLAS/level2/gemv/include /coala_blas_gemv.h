#ifndef _COALA_BLAS_GEMV_H
#define _COALA_BLAS_GEMV_H

#include "coala_blas_base.h"
#include "coala_probes_structure.h"
#include <stddef.h> // for size_t

int coala_blas_sgemv
(   
    coala_probelist_t * probelist,
    size_t const taskid,
    COALA_BLAS_MATRIX_LAYOUT layout,
    COALA_BLAS_MATRIX_TRANSPOSE transa,
    int m, int n,
    float * alpha,
    float * A,
    int lda,
    float * X,
    int incx,
    float * beta,
    float * Y,
    int incy
);


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
);

#endif