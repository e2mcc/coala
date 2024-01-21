#ifndef COALA_RUNTIME_PROBES_PROBE_H
#define COALA_RUNTIME_PROBES_PROBE_H

/*********************************************
* Include
**********************************************/
#include <stddef.h> // for size_t

#include "coala_probes_structure.h"


/*********************************************
* Function
**********************************************/
coala_probelist_t * coala_probelist_getOrInit(coala_probelist_t * probelist);
void coala_probe(coala_probelist_t * probelist, size_t const taskid, size_t const taskcode, size_t const dynnum, ...);

#endif