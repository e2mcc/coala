#ifndef _COALA_MEMOP_LIST_H
#define _COALA_MEMOP_LIST_H

#include <string>
#include <unordered_map>

//-----------------------------------------
// Memory Allocation = 1
// Memory Host to Device = 2
// Memory Device to Host = 3
// Memory Free = 4
//-----------------------------------------
typedef enum
{
    //==============================================
    // 未找到
    //----------------------------------------------
    COALA_MEMOP_NOT_FOUND = 0,

    //==============================================
    // Memory Allocation = 1x
    //==============================================
    // ==> COALA = 1
    // ==> Linux = 2
    // ==> CUDA = 3
    // ==> OpenCL = 4
    //----------------------------------------------
    COALA_MEMOP_HOSTMALC = 111, COALA_MEMOP_DEVMALC = 112,
    COALA_MEMOP_LINUX_MALC = 120,
    COALA_MEMOP_CUDA_MALC = 130,
    COALA_MEMOP_OPENCL_MALC = 140,

    //==============================================
    // Memory Host to Device = 2x
    //==============================================
    // ==> COALA = 1
    // ==> Linux = 2
    // ==> CUDA = 3
    // ==> OpenCL = 4
    //----------------------------------------------
    COALA_MEMOP_H2D = 210,
    COALA_MEMOP_LINUX_H2D = 220,
    COALA_MEMOP_CUDA_H2D = 231, COALA_MEMOP_CUBLAS_H2D = 232,
    COALA_MEMOP_OPENCL_H2D = 240,
    
    //==============================================
    // Memory Device to Host = 3x
    //==============================================
    // ==> COALA = 1
    // ==> Linux = 2
    // ==> CUDA = 3
    // ==> OpenCL = 4
    //----------------------------------------------
    COALA_MEMOP_D2H = 310,
    COALA_MEMOP_LINUX_D2H = 320,
    COALA_MEMOP_CUDA_D2H = 331, COALA_MEMOP_CUBLAS_D2H = 332,
    COALA_MEMOP_OPENCL_D2H = 340,


    //==============================================
    // Memory Free = 4x
    //==============================================
    // ==> COALA = 1
    // ==> Linux = 2
    // ==> CUDA = 3
    // ==> OpenCL = 4
    //----------------------------------------------
    COALA_MEMOP_HOSTFREE = 411, COALA_MEMOP_DEVFREE = 412,
    COALA_MEMOP_LINUX_FREE = 420,
    COALA_MEMOP_CUDA_FREE = 430,
    COALA_MEMOP_OPENCL_FREE = 440

}COALA_MEMOP_CODE;


extern std::unordered_map<COALA_MEMOP_CODE, std::string> COALA_MEMOP_NAMELIST;

#endif
