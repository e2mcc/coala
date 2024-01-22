#include "coala_memop_host2dev.h"
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

int coala_memop_host2dev
(
    coala_probelist_t * probelist,
    size_t const taskid,
    void * hostptr,
    void * devptr,
    int const datasize
)
{
    //  Check for NULL pointers
    if (probelist == NULL)
    {
        fprintf(stderr,"Invalid pointer in coala_memop_host2dev(): probelist is NULL\n");
        exit(EXIT_FAILURE);
    }

    //  Check for NULL pointers
    if (hostptr == NULL || devptr == NULL)
    {
        fprintf(stderr,"Invalid pointer in coala_memop_host2dev(): hostptr or devptr is NULL\n");
        // exit(EXIT_FAILURE);
    }

    //  Check for invalid taskid
    if (taskid >= probelist->length)
    {
        fprintf(stderr,"Invalid taskid in coala_memop_host2dev(): taskid is out of range\n");
        exit(EXIT_FAILURE);
    }


    #ifdef COALA_DEBUG
    printf("Here in coala_memop_host2dev\n");
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
            return cudaMemcpy(devptr,hostptr,datasize,cudaMemcpyHostToDevice);
        #endif

        #ifdef COALA_ENABLE_CLBLAST
        case 2:
            clEnqueueWriteBuffer(probelist->queue, devptr, CL_TRUE, 0, datasize, hostptr, 0, NULL, NULL);
            return 0;
        #endif

        default:
            break;
    }


    return 0;
}