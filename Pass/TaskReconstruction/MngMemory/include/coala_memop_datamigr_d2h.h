#ifndef _COALA_MEMOP_DATA_MIGRATION_D2H_H
#define _COALA_MEMOP_DATA_MIGRATION_D2H_H

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
* Data Transmission Device To Host
*************************************************/
class CoalaMemopDataMigrationD2HCallee : public CoalaMemopDataMigrationCallee
{
    public:
    CoalaMemopDataMigrationD2HCallee(){}

    //本类声明
    Function * coala_memop_dev2host;
    Function * _create_coala_memop_dev2host_FuncDeclaration(Module & M);


    //父类继承
    virtual void transform2coala(GlobalValue * probelist, size_t const taskid) = 0 ;
};


/************************************************
* Derived Class
* Data Transmission Device To Host For Cuda
*************************************************/
class CoalaMemopDataMigrationD2HCallee4Cuda : public CoalaMemopDataMigrationD2HCallee 
{
    private:
    CoalaMemopDataMigrationD2HCallee4Cuda(){}

    //set function
    //cudaMemcpy参数
    Value * setHostPtr();
    Value * setDevPtr();
    Value * setSize();
    Value * setDTFlag();

    public:
    CoalaMemopDataMigrationD2HCallee4Cuda(CallInst * _callee, COALA_MEMOP_CODE _code)
    {
        memop_callee = _callee;
        memop_code = _code;
        memop_callee_params.emplace("HostPtr",setHostPtr());
        memop_callee_params.emplace("DevPtr",setDevPtr());  
        memop_callee_params.emplace("Size",setSize());
        memop_callee_params.emplace("DTFlag",setDTFlag());
    }

    //get function
    Value * getHostPtr();
    Value * getDevPtr();
    Value * getSize();
    Value * getDTFlag();

    //转换为 coala_host_malloc 函数
    void transform2coala(GlobalValue * probelist, size_t const taskid) override;
};





//cublasStatus_t cublasGetMatrix (int M, int N, int size, float * devPtrA, int ldDevA, float * A, int ldA)
class CoalaMemopDataMigrationD2HCallee4Cublas : public CoalaMemopDataMigrationD2HCallee 
{
    private:
    CoalaMemopDataMigrationD2HCallee4Cublas(){}

    //set function param
    Value * setRowDimension();
    Value * setColDimension();
    Value * setTypeSize();
    Value * setDevMatPtr();
    Value * setDevMatLd();
    Value * setHostMatPtr();
    Value * setHostMatLd();

    Value * _getLoadRowInstruction(CallInst * insertpoint);
    Value * _getLoadColInstruction(CallInst * insertpoint);
    // 插入计算size的指令块
    Value * _insert_datasizecalculation_instructions(CallInst * insertpoint);

    public:
    CoalaMemopDataMigrationD2HCallee4Cublas(CallInst * _callee, COALA_MEMOP_CODE _code)
    {
        memop_callee = _callee;
        memop_code = _code;
        if(memop_code == COALA_MEMOP_D2H)
        {
            hasbeencoala = true;
        }
        else
        {   
            hasbeencoala = false;
            memop_callee_params.emplace("RowDimension",setRowDimension());
            memop_callee_params.emplace("ColDimension",setColDimension());
            memop_callee_params.emplace("TypeSize",setTypeSize());
            memop_callee_params.emplace("DevMatPtr",setDevMatPtr());
            memop_callee_params.emplace("DevMatLd",setDevMatLd());
            memop_callee_params.emplace("HostMatPtr",setHostMatPtr());
            memop_callee_params.emplace("HostMatLd",setHostMatLd());
        }   
    }

    //get function
    Value * getRowDimension();
    Value * getColDimension();
    Value * getTypeSize();
    Value * getDevMatPtr();
    Value * getDevMatLd();
    Value * getHostMatPtr();
    Value * getHostMatLd();
    

    //转换为 coala_host_malloc 函数
    void transform2coala(GlobalValue * probelist, size_t const taskid) override;
};

}//end of namespace
#endif