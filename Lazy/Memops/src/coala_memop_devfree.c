#include "coala_memop_devfree.h"
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

void coala_memop_devfree
(
    coala_probelist_t * probelist,
    size_t const taskid,
    void * devptr
)
{
    //  Check for NULL pointers
    if( NULL == probelist )
    {
        fprintf(stderr,"Invalid pointer in coala_memop_devfree(): probelist\n");
        exit(EXIT_FAILURE);
    }
    
    //  Check for NULL pointers
    if( NULL == devptr )
    {
        fprintf(stderr,"Invalid pointer in coala_memop_devfree(): devptr is NULL\n");
        // exit(EXIT_FAILURE);
    }

    //  Check for valid taskid
    if( taskid >= probelist->length )
    {
        fprintf(stderr,"Invalid taskid in coala_memop_devfree(): taskid is out of range\n");
        exit(EXIT_FAILURE);
    } 
    
    #ifdef COALA_DEBUG
    printf("Here in coala_memop_devfree\n");
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
            return;

        #ifdef COALA_ENABLE_CUDA
        case 1:
            cudaFree(devptr);
            return;
        #endif

        #ifdef COALA_ENABLE_CLBLAST
        case 2:
            clReleaseMemObject(devptr);
            return;
        #endif

        default:
            break;
    }

    return;
}