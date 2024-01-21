#ifndef COALA_RUNTIME_PROBES_PROBELIST_H
#define COALA_RUNTIME_PROBES_PROBELIST_H

/*********************************************
* Include
**********************************************/
#include <stddef.h> // for size_t

#include "coala_probes_structure.h"


/*********************************************
* Function
**********************************************/
coala_probelist_t * coala_probelist_getOrInit(coala_probelist_t * probelist);

#endif