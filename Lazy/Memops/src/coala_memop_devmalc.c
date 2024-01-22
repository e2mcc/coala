#include "coala_memop_devmalc.h"
#include <stdio.h>
#include <stdlib.h>

#define COALA_DEBUG //For debug

#ifdef COALA_ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef COALA_ENABLE_CLBLAST
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 
#include <clblast_c.h>
#endif


int coala_memop_devmalc
(
    coala_probelist_t * probelist,
    size_t const taskid,
    void ** devptr,
    int const datasize
)
{
    //  Check for NULL pointers
    if (probelist == NULL)
    {
        fprintf(stderr,"Invalid pointer in coala_memop_devmalc(): probelist is NULL\n");
        exit(EXIT_FAILURE);
    }

    //  Check for valid taskid
    if( taskid >= probelist->length )
    {
        fprintf(stderr,"Invalid taskid in coala_memop_devmalc(): taskid is out of range\n");
    }

    #ifdef COALA_DEBUG
    printf("coala_memop_devmalc\n");
    printf("--taskid=%ld\n",taskid);
    printf("--probelist->probes[0]->taskcode=%ld\n",probelist->probes[0].taskcode);
    printf("--probelist->probes[0].datasizenum=%ld\n",probelist->probes[0].datasizenum);
    for(size_t i = 0; i<probelist->probes[0].datasizenum; i++)
    {
        printf("--probelist->probes[0].datasizes[%ld]=%ld\n",i,probelist->probes[0].datasizes[i]);
    }
    #endif

    switch (probelist->probes[taskid].optimalM)
    {
        case 0:
            return 0;

        #ifdef COALA_ENABLE_CUDA
        case 1:
            return cudaMalloc((void**)devptr,datasize);
        #endif

        #ifdef COALA_ENABLE_CLBLAST
        case 2:
            *devptr = clCreateBuffer(probelist->context, CL_MEM_READ_WRITE, datasize, NULL, NULL);
            return 0;
        #endif

        default:
            break;
    }
    return 0;
}