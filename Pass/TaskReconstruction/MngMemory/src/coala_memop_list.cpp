#include "coala_memop_list.h"

//-----------------------------------------
// Memory Allocation = 1
// Memory Host to Device = 2
// Memory Device to Host = 3
// Memory Free = 4
//-----------------------------------------
std::unordered_map<COALA_MEMOP_CODE, std::string> COALA_MEMOP_NAMELIST ={
    //==============================================
    // Memory Allocation = 1x
    //==============================================
    // ==> COALA = 1
    // ==> Linux = 2
    // ==> CUDA = 3
    // ==> OpenCL = 4
    //----------------------------------------------
    {COALA_MEMOP_HOSTMALC,      "coala_memop_hostmalc"},
    {COALA_MEMOP_DEVMALC,       "coala_memop_devmalc"},
    {COALA_MEMOP_LINUX_MALC,    "malloc"},
    {COALA_MEMOP_CUDA_MALC,     "cudaMalloc"},
    {COALA_MEMOP_OPENCL_MALC,   "clCreateBuffer"},

    //==============================================
    // Memory Host to Device = 2x
    //==============================================
    // ==> COALA = 1
    // ==> Linux = 2
    // ==> CUDA = 3
    // ==> OpenCL = 4
    //----------------------------------------------
    {COALA_MEMOP_H2D,           "coala_memop_host2dev"},
    {COALA_MEMOP_CUDA_H2D,      "cudaMemcpy"},
    {COALA_MEMOP_CUBLAS_H2D,    "cublasSetMatrix"},
    {COALA_MEMOP_OPENCL_H2D,    "clEnqueueWriteBuffer"},
    
    //==============================================
    // Memory Device to Host = 3x
    //==============================================
    // ==> COALA = 1
    // ==> Linux = 2
    // ==> CUDA = 3
    // ==> OpenCL = 4
    //----------------------------------------------
    {COALA_MEMOP_D2H,           "coala_memop_dev2host"},
    {COALA_MEMOP_CUDA_D2H,      "cudaMemcpy"},
    {COALA_MEMOP_CUBLAS_D2H,    "cublasGetMatrix"},
    {COALA_MEMOP_OPENCL_D2H,    "clEnqueueReadBuffer"},


    //==============================================
    // Memory Free = 4x
    //==============================================
    // ==> COALA = 1
    // ==> Linux = 2
    // ==> CUDA = 3
    // ==> OpenCL = 4
    //----------------------------------------------
    {COALA_MEMOP_HOSTFREE,      "coala_memop_hostfree"},
    {COALA_MEMOP_DEVFREE,       "coala_memop_devfree"},
    {COALA_MEMOP_LINUX_FREE,    "free"},
    {COALA_MEMOP_CUDA_FREE,     "cudaFree"},
    {COALA_MEMOP_OPENCL_FREE,   "clReleaseMemObject"}
};