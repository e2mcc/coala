#ifndef _COALA_MEMOP_DEVMALC_H
#define _COALA_MEMOP_DEVMALC_H

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
* Device Memory Allocation
*************************************************/
class CoalaMemopDevMalcCallee
{
    protected:
    CoalaMemopDevMalcCallee(){}

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

    Function * coala_memop_devmalc;
    Function * _create_coala_memop_devmalc_FuncDeclaration(Module & M);

    public:
    //----------------------------------------------------------------------------------
    // MEMOP 的基本信息的 get 方法
    //----------------------------------------------------------------------------------
    CallInst * getMemopCallee();
	std::string getMemopName();
    Value * getMemopParam(std::string param_name);


    //转换为 coala_memop_devmalc 函数
    virtual void transform2coala(GlobalValue * probelist, size_t const taskid) = 0;
};



/************************************************
* Derived Class
* Device Memory Allocation For Cuda
* cudaMalloc(void ** ,i64);
*************************************************/
class CoalaMemopDevMalcCallee4Cuda: public CoalaMemopDevMalcCallee
{
    protected:
    CoalaMemopDevMalcCallee4Cuda(){}

    private:
    //set function
    Value * setMemHandlePtrPtr();
    Value * setSize();   

    public:
    CoalaMemopDevMalcCallee4Cuda(CallInst * _callee, COALA_MEMOP_CODE _code)
    {
        memop_callee = _callee;
        memop_code = _code;
        if(memop_code == COALA_MEMOP_DEVMALC)
        {
            hasbeencoala = true;
        }
        else
        {   
            hasbeencoala = false;
            memop_callee_params.emplace("MemHandlePtrPtr",setMemHandlePtrPtr());
            memop_callee_params.emplace("Size",setSize());
        }
    }

    Value * getMemHandlePtrPtr();
    Value * getSize();

    void transform2coala(GlobalValue * probelist, size_t const taskid) override;
};



/************************************************
* Derived Class
* Device Memory Allocation For OpenCL
*************************************************/
class CoalaMemopDevMalcCallee4OpenCL: public CoalaMemopDevMalcCallee
{
    protected:
    CoalaMemopDevMalcCallee4OpenCL(){}

    private:
    //set function
    Value * setOpenclContext();
    Value * setOpenclMemOpFlag();
    Value * setSize();
    Value * setOpenclHostPtr();
    Value * setOpenclErrCodePtr();   

    public:
    CoalaMemopDevMalcCallee4OpenCL(CallInst * _callee, COALA_MEMOP_CODE _code)
    {
        memop_callee = _callee;
        memop_code = _code;
        if(memop_code == COALA_MEMOP_DEVMALC)
        {
            hasbeencoala = true;
        }
        else
        {   
            hasbeencoala = false;
            memop_callee_params.emplace("OpenclContext",setOpenclContext());
            memop_callee_params.emplace("OpenclMemOpFlag",setOpenclMemOpFlag());
            memop_callee_params.emplace("Size",setSize());
            memop_callee_params.emplace("OpenclHostPtr",setOpenclHostPtr());
            memop_callee_params.emplace("OpenclErrCodePtr",setOpenclErrCodePtr());
        }
    }

    Value * getOpenclContext();
    Value * getOpenclMemOpFlag();
    Value * getSize();
    Value * getOpenclHostPtr();
    Value * getOpenclErrCodePtr();  

    void transform2coala(GlobalValue * probelist, size_t const taskid) override;
};


}//end of namespce
#endif