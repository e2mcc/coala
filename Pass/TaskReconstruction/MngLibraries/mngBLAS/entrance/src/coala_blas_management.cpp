/************************************************
* Include
*************************************************/
#include "coala_blas_management.h"
#include "coala_cornerstone_utils.h"

//llvm
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>

/************************************************
* Name Space
*************************************************/
using namespace llvm;


/************************************************
* Static Function Only In This File
*************************************************/
/**
 * @brief 识别一个 Callee 是不是 blas routine
 * @param CI
 * @return CODE
*/
static COALA_BLAS_ROUTINES_CODE _findBlasRoutineCallee(CallInst * CI)
{
	Function * callee = CI->getCalledFunction();
	std::string name = getDemangledName(*callee);
	// 遍历 COALA_BLAS_ROUTINES_NAMELIST 中所有的元素并与 name 比较
	for (auto & pair : COALA_BLAS_ROUTINES_NAMELIST) 
	{	
		if( name == pair.first ) 
		{	
			return pair.second;
		}
    }
	return NOT_FOUND;
}

/************************************************
* Private Function
*************************************************/
/**
 * @brief 扫描整个 Module
 * 找出所有的 blas routine callee 出来
 * 并且设置对应的标识，组成<CI,tag>二元组
 * 将这个二元组存入队列中
 * @param M
 * @return void
*/
void BlasManagement::_scanning(Module & M)
{   
    //扫描 M 中的每个 Function
    for(auto M_iterator = M.getFunctionList().begin(); M_iterator != M.getFunctionList().end(); M_iterator++)
	{	
		//声明一个F
		Function & F = * M_iterator;
		// 遍历整个F
	 	for(inst_iterator I = inst_begin(F); I != inst_end(F); ++I) 
		{
			Instruction & Inst = *I;
			//判断是否是一个函数调用指令
			if (isa<CallInst>(Inst))
			{	
				CallInst * CI = dyn_cast<CallInst>(&Inst);
				COALA_BLAS_ROUTINES_CODE namecode = _findBlasRoutineCallee(CI);
				if(namecode!=NOT_FOUND)
				{
					BlasManagement::blas_routine_callees.emplace(CI,namecode);
				}
			}
		}
	}
	return;
}



/**
 * @brief 
 * 对每一个 blas routine 的 callee 进行分析
 * 记录所有了依赖
 * 最后生成对应的BALS任务
 * @param M LLVM-IR Module
 * @return void
*/
void BlasManagement::_analyzing(Module & M)
{
	//至少要有一个元素
	if(BlasManagement::blas_routine_callees.empty())
	{
		return;
	}

	size_t taskid = 0;
	//对BlasManagement::blas_routines队列中的任务进行逐一处理
	for (auto & pair : BlasManagement::blas_routine_callees)
	{
		//根据callee的name创建task
		outs()<<"Locating "<<*pair.first<<"\n";
		std::shared_ptr<CoalaBlasTask> cbt = CoalaBlasTaskFactory::createACoalaBlasTask(pair.first, pair.second, taskid);
		// outs()<<"BlasManagement::blas_tasks.size()="<<BlasManagement::blas_tasks.size()<<"\n";
		if(cbt!=nullptr)
		{
			BlasManagement::blas_tasks.push_back(cbt);
			// outs()<<"BlasManagement::blas_tasks.size()="<<BlasManagement::blas_tasks.size()<<"\n";
			// BlasManagement::blas_tasks[0]->dump();
			BlasManagement::blas_tasks[0]->reconstrcuting();
		}

		taskid++;
	}

	return;
}



//COALA-TODO
/**
 * @brief 对单个计算任务进行重构 
 * @param M 
 * @param CI 
 */
// void BlasManagement::_taskReconstrcuting(Module & M)
// {	
// 	return;
// }




/************************************************
* Public Function
*************************************************/
//注意：构造函数不能 return
BlasManagement::BlasManagement(Module & M)
{
	//扫描整个Module，然后把所有callee存储起来
	BlasManagement::_scanning(M);

	//对所有已检测出的任务进行分析
	BlasManagement::_analyzing(M);

}

//COALA-TODO
BlasManagement::BlasManagement(Module & M, CoalaProbes & CPs)
{
	//扫描整个Module，然后把所有callee存储起来
	BlasManagement::_scanning(M);

	//对所有已检测出的任务进行分析
	BlasManagement::_analyzing(M);

}

//COALA-TODO
BlasManagement::BlasManagement(Module & M, CoalaProbes & CPs, CoalaTaskGraph & CTG)
{
	//扫描整个Module，然后把所有callee存储起来
	BlasManagement::_scanning(M);

	//对所有已检测出的任务进行分析
	BlasManagement::_analyzing(M);
}





