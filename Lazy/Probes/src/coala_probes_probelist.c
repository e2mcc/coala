/*********************************************
* Include
**********************************************/
#include "coala_probes_probelist.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define COALA_DEBUG //For Debug

/*********************************************
* Define
**********************************************/
// Initializing size of the "probes" array
#define COALA_PROBES_INIT_SIZE (10)


/*********************************************
* Function
**********************************************/
coala_probelist_t * coala_probelist_getOrInit(coala_probelist_t * probelist)
{   
    //判断是否已经初始化
    if( probelist != NULL )  return probelist;

    //创建结构体
    coala_probelist_t * _probelist = malloc(sizeof(*_probelist));

    //设置初始结构体数组大小
    _probelist->size = COALA_PROBES_INIT_SIZE;

    //设置当前长度
    _probelist->length = 0;

    //初始化
    _probelist->probes = malloc(_probelist->size*sizeof(*_probelist->probes));

    if( _probelist->probes == NULL)
    {
        fprintf(stderr, "Invalid pointer in coala_probelist_getOrInit(): probes malloc was failure\n");
        exit(EXIT_FAILURE);
    }

    //初始化
    for(size_t i=0;i<_probelist->size;i++)
    {   
        _probelist->probes[i].taskid=0;
        _probelist->probes[i].taskcode=0;
        _probelist->probes[i].datasizenum=0;
        _probelist->probes[i].datasizes=NULL;
        _probelist->probes[i].optimalM = 0;
        _probelist->probes[i].optimalR = 0;
    }
    
    return  _probelist;
}