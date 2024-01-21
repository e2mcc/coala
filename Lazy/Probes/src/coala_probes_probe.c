/*********************************************
* Include
**********************************************/
#include "coala_probes_probe.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#define COALA_DEBUG //For Debug

/*********************************************
* Define
**********************************************/
// Increamental size of the "probes" array
#define COALA_PROBES_INCR_SIZE (5)

/*********************************************
* Function
**********************************************/
void coala_probe(coala_probelist_t * probelist, size_t const taskid, size_t const taskcode, size_t const dynnum, ...)
{
    if(probelist == NULL)
    {
        fprintf(stderr, "Invalid pointer in coala_probe(): probelist is NULL\n");
        exit(EXIT_FAILURE);
    }

    // 输入判断(probes满了)
    if(probelist->length==probelist->size)
    {
        //调整大小
        probelist->probes = realloc(probelist->probes,(COALA_PROBES_INCR_SIZE+probelist->size)*sizeof(*probelist->probes));
        if( probelist->probes == NULL ) 
        {
            fprintf(stderr, "Error in coala_probe(): probelist->probes connot realloc\n");
            exit(EXIT_FAILURE);
        }
        probelist->size += COALA_PROBES_INCR_SIZE;
    }

    //输入结构体存储
    probelist->probes[probelist->length].taskid = taskid;
    probelist->probes[probelist->length].taskcode = taskcode;
    probelist->probes[probelist->length].datasizenum = dynnum; 
    probelist->probes[probelist->length].datasizes = malloc(dynnum*sizeof(*(probelist->probes[probelist->length].datasizes)));
    
    if( probelist->probes[probelist->length].datasizes == NULL)
    {
        fprintf(stderr, "Invalid pointer in coala_probe(): datasizes malloc was failure\n");
        exit(EXIT_FAILURE);
    }

    va_list valist;
    // 为 num 个参数初始化 valist
    va_start(valist, dynnum);

    for(size_t i=0;i<dynnum;i++)
    {
        (probelist->probes[probelist->length].datasizes)[i] = va_arg(valist,int);
    }

    // 清理为 valist 保留的内存
    va_end(valist);

    probelist->length++;


    #ifdef COALA_DEBUG
    //Debug
    printf("coala_probe\n");
    printf("--probelist->probes[0].taskid=%ld\n",probelist->probes[0].taskid);
    printf("--probelist->probes[0].taskcode=%ld\n",probelist->probes[0].taskcode);
    printf("--probelist->probes[0].datasizenum=%ld\n",probelist->probes[0].datasizenum);
    for(size_t i = 0; i<probelist->probes[0].datasizenum; i++)
    {
        printf("--probelist->probes[0].datasizes[%ld]=%ld\n",i,probelist->probes[0].datasizes[i]);
    }
    #endif

    return;
}