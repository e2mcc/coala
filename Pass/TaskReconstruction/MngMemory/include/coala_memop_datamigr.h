#ifndef _COALA_MEMOP_DATA_MIGRATION_H
#define _COALA_MEMOP_DATA_MIGRATION_H

// COALA-TODO:整体待验证

/*********************************************
* Include
**********************************************/
#include <map>
#include <llvm/IR/Module.h>//for Module class
#include <llvm/IR/Function.h>//for Function class
#include <llvm/IR/Instructions.h>
#include "coala_memop_list.h"

/************************************************
* Namespace
*************************************************/
namespace llvm{


/************************************************
* Base Class
* Data Transmission
*************************************************/
class CoalaMemopDataMigrationCallee
{
    protected:
    CoalaMemopDataMigrationCallee(){}

    //是否已被转化为 coala_memop_devmalc
    bool hasbeencoala;

    //----------------------------------------------------------------------------------
    // MEMOP 的基本信息 
    //----------------------------------------------------------------------------------
    //函数调用
	CallInst * memop_callee;
	//函数
	COALA_MEMOP_CODE memop_code;
    //函数参数
    std::unordered_map<std::string, Value*> memop_callee_params;

    public:
    //----------------------------------------------------------------------------------
    // MEMOP 的基本信息的 get 方法
    //----------------------------------------------------------------------------------
    CallInst * getMemopCallee();
	std::string getMemopName();
    Value * getMemopParam(std::string param_name);


    //转换为 coala_host_malloc 函数
    virtual void transform2coala(GlobalValue * probelist, size_t const taskid) = 0;
};



}//end of namespace

#endif