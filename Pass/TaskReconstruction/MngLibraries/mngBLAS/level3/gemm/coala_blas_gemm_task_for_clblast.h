#ifndef _COALA_BLAS_GEMM_TASK_FOR_CLBLAST_H
#define _COALA_BLAS_GEMM_TASK_FOR_CLBLAST_H

/************************************************
* Include
*************************************************/
#include "coala_blas_gemm_task.h"


/************************************************
* Namespace
*************************************************/
namespace llvm{



/************************************************
* CoalaBlasGemmTask4Cublas
*************************************************/
class CoalaBlasGemmTask4Clblast: public  CoalaBlasGemmTask
{
    private:
	//构造函数(不可公开访问)
	CoalaBlasGemmTask4Clblast(){}

    //----------------------------------------------------------------------------------
    // 计算例程 
    //----------------------------------------------------------------------------------
    //参数set方法(私有)
	Value * setLayout() 	override;//order
  	Value * setTransA() 	override;//transa
	Value * setTransB() 	override;//transb
	Value * setM() 			override;//m
	Value * setN() 			override;//n
	Value * setK() 			override;//k
  	Value * setAlpha() 		override;//alpha
	Value * setMatA() 		override;//DeviceA
	Value * setOffA();				 //offsetA
	Value * setLDA() 		override;//lda
	Value * setMatB() 		override;//DeviceB
	Value * setOffB(); 				 //offsetB
	Value * setLDB() 		override;//ldb
	Value * setBeta() 		override;//beta
	Value * setMatC() 		override;//DeviceC
	Value * setOffC();      		 //offsetC
	Value * setLDC() 		override;//ldc	
	Value * setCommandQueues();
	Value * setEvents();

	//----------------------------------------------------------------------------------
    // 相关内存操作
    //----------------------------------------------------------------------------------
	void _setDevMalcCalleeInfo(Value* param) 	override;
    void _setHost2DevCalleeInfo(Value * param) 	override;
	void _setDev2HostCalleeInfo(Value * param) 	override;
    void _setDevFreeCalleeInfo(Value* param) 	override;

    public:
	//构造函数（重载）
   	CoalaBlasGemmTask4Clblast(CallInst * _callee, COALA_BLAS_ROUTINES_CODE const _code, size_t const _taskid)
	{	
		routine_callee 	= _callee;
		routine_code 	= _code;
		taskid 			= _taskid;
		routine_callee_params.emplace("Layout",setLayout());
		routine_callee_params.emplace("TransA",setTransA());
		routine_callee_params.emplace("TransB",setTransB());
		routine_callee_params.emplace("M",setM());
		routine_callee_params.emplace("N",setN());
		routine_callee_params.emplace("K",setK());
		routine_callee_params.emplace("Alpha",setAlpha());
		routine_callee_params.emplace("MatA",setMatA());
		routine_callee_params.emplace("OffA",setOffA());
		routine_callee_params.emplace("LDA",setLDA());
		routine_callee_params.emplace("MatB",setMatB());
		routine_callee_params.emplace("OffB",setOffB());
		routine_callee_params.emplace("LDB",setLDB());
		routine_callee_params.emplace("Beta",setBeta());
		routine_callee_params.emplace("MatC",setMatC());
		routine_callee_params.emplace("OffC",setOffC());
		routine_callee_params.emplace("LDC",setLDC());
		routine_callee_params.emplace("CommandQueues",setCommandQueues());
		routine_callee_params.emplace("Events",setEvents());
	}

	// 参数get方法(必要)
	Value * getLayout() 	override;//order
  	Value * getTransA()		override;//transa
	Value * getTransB()		override;//transb
	Value * getM()			override;//m
	Value * getN()			override;//n
	Value * getK()			override;//k
  	Value * getAlpha() 		override;//alpha
	Value * getMatA() 		override;//HostA
	Value * getLDA() 		override;//lda
	Value * getMatB() 		override;//HostB
	Value * getLDB() 		override;//ldb
	Value * getBeta() 		override;//beta
	Value * getMatC() 		override;//HostC
	Value * getLDC() 		override;//ldc
	
	//dump指一股脑输出所有信息，用于debug
	void dump() override;
};



}//end of namespace
#endif