#ifndef _COALA_PROBES_H
#define _COALA_PROBES_H

/*********************************************
* Include
**********************************************/
#include <llvm/IR/Module.h>



/*********************************************
* Namespace
**********************************************/
namespace llvm{


/*********************************************
* Class
**********************************************/
class CoalaProbes
{
    private:
    CoalaProbes(){}

    //-------------------------------------------------------
    //在 Module 中的全局变量:coala_probelist
    //-------------------------------------------------------
    GlobalVariable * coala_probelist;
    GlobalVariable * _create_coala_probelist_GlobalV(Module & M);

    //-------------------------------------------------------
    // coala_probelist * coala_probe(void)
    //-------------------------------------------------------
    //在 Module 中的声明 coala_probelist_getOrInit 函数
    Function * coala_probelist_getOrInit;
    Function * _create_coala_probelist_getOrInit_FuncDeclaration(Module & M);

    //在 Function 中插入的 coala_probelist_getOrInit 函数调用并存储到全局变量 coala_probelist 中
    CallInst * coala_probelist_getOrInit_callee;
    CallInst * _insert_coala_probelist_getOrInit_FuncCallee(Function & F);

    //-------------------------------------------------------
    // void coala_probe()
    //-------------------------------------------------------
    //在 Module 中的声明 coala_probe 函数
    Function * coala_probe;
    Function * _create_coala_probe_FuncDeclaration(Module & M);
    CallInst * coala_probelist_callee;
    

    public:
    CoalaProbes(Function & F);

    void insert_coala_probe_FuncCallee
    (
        CallInst * insertpoint,
        size_t const taskid,
        size_t const taskcode,
        size_t const dnum,
        ...
    );

    GlobalVariable * getCoalaProbelistGV();
};





}//end of namespace

#endif
