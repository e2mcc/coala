#ifndef _COALA_MEMOP_DATA_MIGRATION_D2D_H
#define _COALA_MEMOP_DATA_MIGRATION_D2D_H

// COALA-TODO:整体待验证


/*********************************************
* Include
**********************************************/
#include "coala_memop_datamigr.h"
#include <map>
#include <llvm/IR/Module.h>//for Module class
#include <llvm/IR/Function.h>//for Function class
#include <llvm/IR/Instructions.h>

/************************************************
* Namespace
*************************************************/
namespace llvm{


/************************************************
* Derived Class
* Data Transmission Device To Device
*************************************************/
class CoalaMemopDataMigrationD2DCallee : public CoalaMemopDataMigrationCallee
{
    public:
    CoalaMemopDataMigrationD2DCallee(){}
    
    //父类继承
    void transform2coala(GlobalValue * probelist, size_t const taskid) override;
};


/************************************************
* Derived Class
* Data Transmission Device To Device For xxx
*************************************************/





}//end of namespace
#endif