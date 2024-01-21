#ifndef _COALA_MEMOP_DEVFREE_H
#define _COALA_MEMOP_DEVFREE_H

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
* Device Memory Free: CoalaMemopDevFreeCallee
*************************************************/
class CoalaMemopDevFreeCallee
{   
    protected:
    CoalaMemopDevFreeCallee(){}


    //是否已被转化为 coala_memop_hostfree
    bool hasbeencoala;

    Function * coala_memop_devfree;
    Function * _create_coala_memop_devfree_FuncDeclaration(Module & M);

    
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
    virtual void transform2coala(GlobalValue * probelist, size_t const taskid) = 0;
};



/************************************************
* Derived Class
* Device Memory Free For Cuda
*************************************************/
class CoalaMemopDevFreeCallee4Cuda : public CoalaMemopDevFreeCallee
{
    protected:
    CoalaMemopDevFreeCallee4Cuda(){}

    private:
    //set function
    Value * setMemPtr();


    public:
    CoalaMemopDevFreeCallee4Cuda(CallInst * _callee, COALA_MEMOP_CODE _code)
    {
        memop_callee = _callee;
        memop_code = _code;
        if(memop_code == COALA_MEMOP_DEVFREE)
        {
            hasbeencoala = true;
        }
        else
        {   
            hasbeencoala = false;
            memop_callee_params.emplace("MemPtr",setMemPtr());
        }
    }

    Value * getMemPtr();

    void transform2coala(GlobalValue * probelist, size_t const taskid) override;
};


/************************************************
* Derived Class
* Device Memory Free For OpenCL
*************************************************/
class CoalaMemopDevFreeCallee4OpenCL : public CoalaMemopDevFreeCallee
{
    protected:
    CoalaMemopDevFreeCallee4OpenCL(){}

    private:
    //set function
    Value * setMemPtr();


    public:
    CoalaMemopDevFreeCallee4OpenCL(CallInst * _callee, COALA_MEMOP_CODE _code)
    {
        memop_callee = _callee;
        memop_code = _code;
        if(memop_code == COALA_MEMOP_DEVFREE)
        {
            hasbeencoala = true;
        }
        else
        {   
            hasbeencoala = false;
            memop_callee_params.emplace("MemPtr",setMemPtr());
        }
    }

    Value * getMemPtr();

    void transform2coala(GlobalValue * probelist, size_t const taskid) override;
};


}//end of namespce
#endif