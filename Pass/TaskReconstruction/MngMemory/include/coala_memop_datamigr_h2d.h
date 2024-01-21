#ifndef _COALA_MEMOP_DATA_MIGRATION_H2D_H
#define _COALA_MEMOP_DATA_MIGRATION_H2D_H

// COALA-TODO:整体待验证

/************************************************
* Include
*************************************************/
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
* Derived Subbase Class
* Data Transmission Host To Device
*************************************************/
class CoalaMemopDataMigrationH2DCallee : public CoalaMemopDataMigrationCallee
{
    protected:
    CoalaMemopDataMigrationH2DCallee(){}
    
    //本类声明
    Function * coala_memop_host2dev;
    Function * _create_coala_memop_host2dev_FuncDeclaration(Module & M);

    public:
    //父类继承
    virtual void transform2coala(GlobalValue * probelist, size_t const taskid) = 0;
};


/************************************************
* Derived Class
* Data Transmission Host To Device For Cuda
* cudaMemcpy(void * dest, void* source, size_t size, flag)
*************************************************/
class CoalaMemopDataMigrationH2DCallee4Cuda : public CoalaMemopDataMigrationH2DCallee 
{
    private:
    CoalaMemopDataMigrationH2DCallee4Cuda(){}

    //set function
    std::string setName();
    //cudaMemcpy参数
    Value * setDevPtr();
    Value * setHostPtr();
    Value * setSize();
    Value * setDTFlag();

    public:
    CoalaMemopDataMigrationH2DCallee4Cuda(CallInst * _callee, COALA_MEMOP_CODE _code)
    {
        memop_callee = _callee;
        memop_code = _code;
        memop_callee_params.emplace("DevPtr",setDevPtr());
        memop_callee_params.emplace("HostPtr",setHostPtr());
        memop_callee_params.emplace("Size",setSize());
        memop_callee_params.emplace("DTFlag",setDTFlag());
    }

    //get function
    Value * getDevPtr();
    Value * getHostPtr();
    Value * getSize();
    Value * getDTFlag();

    //转换为 coala_host_malloc 函数
    void transform2coala(GlobalValue * probelist, size_t const taskid) override;
};



class CoalaMemopDataMigrationH2DCallee4Cublas : public CoalaMemopDataMigrationH2DCallee 
{
    private:
    CoalaMemopDataMigrationH2DCallee4Cublas(){}

    //set function param
    Value * setRowDimension();
    Value * setColDimension();
    Value * setTypeSize();
    Value * setHostMatPtr();
    Value * setHostMatLd();
    Value * setDevMatPtr();
    Value * setDevMatLd();

    Value * _getLoadRowInstruction(CallInst * insertpoint);
    Value * _getLoadColInstruction(CallInst * insertpoint);
    // 插入计算size的指令块
    Value * _insert_datasizecalculation_instructions(CallInst * insertpoint);

    public:
    CoalaMemopDataMigrationH2DCallee4Cublas(CallInst * _callee, COALA_MEMOP_CODE _code)
    {
        memop_callee = _callee;
        memop_code = _code;
        if(memop_code == COALA_MEMOP_H2D)
        {
            hasbeencoala = true;
        }
        else
        {
            hasbeencoala = false;
            memop_callee_params.emplace("RowDimension",setRowDimension());
            memop_callee_params.emplace("ColDimension",setColDimension());
            memop_callee_params.emplace("TypeSize",setTypeSize());
            memop_callee_params.emplace("HostMatPtr",setHostMatPtr());
            memop_callee_params.emplace("HostMatLd",setHostMatLd());
            memop_callee_params.emplace("DevMatPtr",setDevMatPtr());
            memop_callee_params.emplace("DevMatLd",setDevMatLd());
        }
    }

    //get function
    Value * getRowDimension();
    Value * getColDimension();
    Value * getTypeSize();
    Value * getHostMatPtr();
    Value * getHostMatLd();
    Value * getDevMatPtr();
    Value * getDevMatLd();

    //转换为 coala_host_malloc 函数
    void transform2coala(GlobalValue * probelist, size_t const taskid) override;
};


}//end of namespace





#endif