#ifndef _COALA_MEMOPRT_DEVMALC_H
#define _COALA_MEMOPRT_DEVMALC_H

#include "coala_probes_structure.h"

int coala_memop_devmalc
(
    coala_probelist_t * probelist,
    size_t const taskid,
    void ** devptr,
    int const datasize
);

#endif