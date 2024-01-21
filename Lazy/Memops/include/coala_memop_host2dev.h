#ifndef _COALA_MEMOPRT_HOST2DEV_H
#define _COALA_MEMOPRT_HOST2DEV_H

#include "coala_probes_structure.h"

int coala_memop_host2dev
(
    coala_probelist_t * probelist,
    size_t const taskid,
    void * hostptr,
    void * devptr,
    int const datasize
);

#endif