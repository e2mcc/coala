#ifndef _COALA_BLAS_GEMM_TASK_H
#define _COALA_BLAS_GEMM_TASK_H

/************************************************
* Include
*************************************************/
#include "coala_blas_task.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <llvm/IR/Instructions.h>

/************************************************
* Name Space
*************************************************/
namespace llvm{



/************************************************
* CoalaBlasGemmTask
*************************************************/
class CoalaBlasGemmTask : public CoalaBlasTask
{
    protected:
    CoalaBlasGemmTask(){}

    //----------------------------------------------------------------------------------
    // 计算例程 
    //----------------------------------------------------------------------------------
    private:
    // coala 例程声明
	Function * coala_blas_gemm_funcdecl;
    Function * _create_coala_blas_sgemm_FuncDeclaration(Module & M);
	Function * _create_coala_blas_dgemm_FuncDeclaration(Module & M);

	// coala 例程插入
	void _insert_coala_blas_sgemm_FuncCallee(GlobalValue * probelist, size_t const taskid);
	void _insert_coala_blas_dgemm_FuncCallee(GlobalValue * probelist, size_t const taskid);

    // 参数set方法(必要)
	virtual Value * setLayout() 	= 0;//order
  	virtual Value * setTransA()		= 0;//transa
	virtual Value * setTransB()		= 0;//transb
	virtual Value * setM()			= 0;//m
	virtual Value * setN()			= 0;//n
	virtual Value * setK()			= 0;//k
  	virtual Value * setAlpha() 		= 0;//alpha
	virtual Value * setMatA() 		= 0;//HostA
	virtual Value * setLDA() 		= 0;//lda
	virtual Value * setMatB() 		= 0;//HostB
	virtual Value * setLDB() 		= 0;//ldb
	virtual Value * setBeta() 		= 0;//beta
	virtual Value * setMatC() 		= 0;//HostC
	virtual Value * setLDC() 		= 0;//ldc

	//----------------------------------------------------------------------------------
    // 相关内存操作
    //----------------------------------------------------------------------------------
	virtual void _setDevMalcCalleeInfo(Value* param) 	= 0;
    virtual void _setHost2DevCalleeInfo(Value * param) 	= 0;
	virtual void _setDev2HostCalleeInfo(Value * param) 	= 0;
    virtual void _setDevFreeCalleeInfo(Value* param) 	= 0;

	Value * _insertLoadMInstruction(CallInst * insertpoint);
	Value * _insertLoadNInstruction(CallInst * insertpoint);
	Value * _insertLoadKInstruction(CallInst * insertpoint);

    public:
	// 参数get方法(必要)
	virtual Value * getLayout() 	= 0;//order
  	virtual Value * getTransA()		= 0;//transa
	virtual Value * getTransB()		= 0;//transb
	virtual Value * getM()			= 0;//m
	virtual Value * getN()			= 0;//n
	virtual Value * getK()			= 0;//k
  	virtual Value * getAlpha() 		= 0;//alpha
	virtual Value * getMatA() 		= 0;//HostA
	virtual Value * getLDA() 		= 0;//lda
	virtual Value * getMatB() 		= 0;//HostB
	virtual Value * getLDB() 		= 0;//ldb
	virtual Value * getBeta() 		= 0;//beta
	virtual Value * getMatC() 		= 0;//HostC
	virtual Value * getLDC() 		= 0;//ldc

	// 重构
	void reconstrcuting() override;

	

    // dump指一股脑输出所有信息，用于debug
	virtual void dump() =0;

	

};


/************************************************
* 工厂类
*************************************************/
class CoalaBlasGemmTaskFactory
{
	public:
    static std::shared_ptr<CoalaBlasGemmTask> createACoalaBlasGemmTask(CallInst * CI, COALA_BLAS_ROUTINES_CODE const namecode, size_t const taskid);
};


}//end of namespace
#endif
