#ifndef _COALA_MEMOPRT_DEVFREE_H
#define _COALA_MEMOPRT_DEVFREE_H

#include "coala_probes_structure.h"

void coala_memop_devfree
(
    coala_probelist_t * probelist, 
    size_t const taskid, 
    void * devptr
);

#endif