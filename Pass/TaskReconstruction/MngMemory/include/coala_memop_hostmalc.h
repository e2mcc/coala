#ifndef _COALA_MEMOP_HOSTMALC_H
#define _COALA_MEMOP_HOSTMALC_H

/*********************************************
* Include
**********************************************/
#include <unordered_map>
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
* Host Memory Allocation
*************************************************/
class CoalaMemopHostMalcCallee
{
    protected:
    CoalaMemopHostMalcCallee(){}

    //是否已被转化为 coala_memop_hostmalc
    bool hasbeencoala;

    Function * coala_memop_hostmalc;
    void _create_coala_memop_hostmalc_FuncDeclaration(Module & M);


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


    //转换为 coala_memop_hostmalc 函数
    virtual void transform2coala(size_t const taskid) = 0;
};



/************************************************
* Derived Class
* Host Memory Allocation For Linux: malloc
*************************************************/
class CoalaMemopHostMalcCallee4Linux : public CoalaMemopHostMalcCallee
{
    protected:
    CoalaMemopHostMalcCallee4Linux(){}

    private:
    //set function
    Value * setSize();


    public:
    CoalaMemopHostMalcCallee4Linux(CallInst * _callee, COALA_MEMOP_CODE _code)
    {
        memop_callee = _callee;
        memop_code = _code;
        if(memop_code == COALA_MEMOP_HOSTMALC)
        {
            hasbeencoala = true;
        }
        else
        {   
            hasbeencoala = false;
            memop_callee_params.emplace("Size",setSize());
        }
    }

    Value * getSize();

    //转换为 coala_host_malloc 函数
    void transform2coala(size_t const taskid) override;
};



}//end of namespace



#endif