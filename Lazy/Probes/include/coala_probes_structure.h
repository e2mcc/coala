#ifndef COALA_RUNTIME_PROBES_STRUCTURE_H
#define COALA_RUNTIME_PROBES_STRUCTURE_H

/*********************************************
* Include
**********************************************/
#include <stddef.h> // for size_t

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
}coala_probelist_t;

#endif