#ifndef _COALA_BLAS_GEMM_H
#define _COALA_BLAS_GEMM_H

#include "coala_blas_base.h"
#include "coala_probes_structure.h"
#include <stddef.h> // for size_t

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
);


int coala_blas_dgemm
(
    coala_probelist_t * probelist,
    size_t const taskid,
    COALA_BLAS_MATRIX_LAYOUT layout,
    COALA_BLAS_MATRIX_TRANSPOSE transa, COALA_BLAS_MATRIX_TRANSPOSE transb,
    int m, int n, int k,
    double * alpha,
    double * A,
    int lda,
    double * B,
    int ldb,
    double * beta,
    double * C,
    int ldc
);

#endif