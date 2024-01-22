#ifndef COALA_RUNTIME_PROBES_STRUCTURE_H
#define COALA_RUNTIME_PROBES_STRUCTURE_H

/*********************************************
* Include
**********************************************/
#include <stddef.h> // for size_t

#ifdef COALA_ENABLE_CUBLAS
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#ifdef COALA_ENABLE_CLBLAST
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 
#include <clblast_c.h>
#endif

/*********************************************
* Structs
**********************************************/
typedef struct
{
    size_t taskid;
    size_t taskcode;
    size_t datasizenum;
    size_t * datasizes;
    size_t optimalM;
    size_t optimalR;
}coala_probe_t;


typedef struct
{
   coala_probe_t * probes;
   size_t length;
   size_t size; 
   
   #ifdef COALA_ENABLE_CUBLAS
   cublasHandle_t handle;
   #endif

   #ifdef COALA_ENABLE_CLBLAST
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   #endif
}coala_probelist_t;

#endif