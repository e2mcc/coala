#ifndef _COALA_MEMOPRT_DEV2HOST_H
#define _COALA_MEMOPRT_DEV2HOST_H

#include "coala_probes_structure.h"

int coala_memop_dev2host
(   
    coala_probelist_t * probelist,
    size_t const taskid,
    void * devptr,
    void * hostptr,
    int const datasize
);


#endif