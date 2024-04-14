/************************************************
* Include
*************************************************/
#include "coala_blas_task.h"
#include "coala_cornerstone_utils.h"
//GEMM
#include "coala_blas_gemm_task.h"

/************************************************
* Name Space
*************************************************/
using namespace llvm;



/************************************************
* Class: CoalaBlasTask
*************************************************/
size_t CoalaBlasTask::_get_first_devmalc_callee_info_idx()
{    
    size_t min = 99999;
    size_t minidx =0;
    for(size_t i = 0; i < CoalaBlasTask::devmalc_callee_infos.size(); ++i)
    {
        size_t rank = getInstructionSequenceNumber(CoalaBlasTask::devmalc_callee_infos[i]->getMemopCallee());
        if(rank<min) 
        {
            min = rank;
            minidx = i;
        }
    }
    return minidx;
}



CallInst * CoalaBlasTask::getRoutineCallee()
{
    return CoalaBlasTask::routine_callee;
}

std::string CoalaBlasTask::getRoutineName()
{
    for (auto & pair : COALA_BLAS_ROUTINES_NAMELIST) 
	{	
		if( CoalaBlasTask::routine_code == pair.second ) 
		{	
			return pair.first;
		}
    }
    return "Not Found";
}

size_t CoalaBlasTask::getTaskCode()
{
    return CoalaBlasTask::routine_code;
}

Value * CoalaBlasTask::getRoutineParam(std::string param_name)
{
    return CoalaBlasTask::routine_callee_params[param_name];
}


/************************************************
* Class: CoalaBlasTaskFactory
*************************************************/

std::shared_ptr<CoalaBlasTask> CoalaBlasTaskFactory::createACoalaBlasTask(CallInst * CI, COALA_BLAS_ROUTINES_CODE const namecode, size_t const taskid)
{
    switch(namecode)
    {
        case COALA_BLAS_CBLAS_SGEMM:
        case COALA_BLAS_CBLAS_DGEMM:
        case COALA_BLAS_CUBLAS_SGEMM:
        case COALA_BLAS_CUBLAS_DGEMM:
        case COALA_BLAS_CLBLAST_SGEMM:
        case COALA_BLAS_CLBLAST_DGEMM:
            return CoalaBlasGemmTaskFactory::createACoalaBlasGemmTask(CI, namecode, taskid); 
        default:
            return  nullptr;
    }
    return  nullptr;
}



