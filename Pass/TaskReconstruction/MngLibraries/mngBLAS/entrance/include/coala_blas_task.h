#ifndef _COALA_BLAS_TASK_H
#define _COALA_BLAS_TASK_H

/*********************************************
* Include
**********************************************/
#include "coala_cornerstone_task.h"
#include "coala_blas_list.h"
#include "coala_memop_hostmalc.h"
#include "coala_memop_hostfree.h"
#include "coala_memop_devmalc.h"
#include "coala_memop_devfree.h"
#include "coala_memop_datamigr_h2d.h"
#include "coala_memop_datamigr_d2h.h"
#include <string>
#include <vector>
#include <unordered_map>

/*************************************************
* Name Space
**************************************************/
/**
 * @brief llvm 命名空间
 */
namespace llvm{



/*************************************************
* Class: CoalaBlasTask
**************************************************/
class CoalaBlasTask : public CoalaTask
{
    protected:
    CoalaBlasTask(){}

    //----------------------------------------------------------------------------------
    // 前置内存操作
    //----------------------------------------------------------------------------------
    std::vector<std::shared_ptr<CoalaMemopDevMalcCallee>>           devmalc_callee_infos;
    std::vector<std::shared_ptr<CoalaMemopDataMigrationH2DCallee>>    host2dev_callee_infos;
    virtual void _setDevMalcCalleeInfo(Value* param) = 0;
    virtual void _setHost2DevCalleeInfo(Value * param) = 0;

    //----------------------------------------------------------------------------------
    // 后置内存操作
    //----------------------------------------------------------------------------------
    std::vector<std::shared_ptr<CoalaMemopDataMigrationD2HCallee>>    dev2host_callee_infos;
    std::vector<std::shared_ptr<CoalaMemopDevFreeCallee>>             devfree_callee_infos;
    virtual void _setDev2HostCalleeInfo(Value * param) = 0;
    virtual void _setDevFreeCalleeInfo(Value* param) = 0;

    size_t _get_first_devmalc_callee_info_idx();
    
    //----------------------------------------------------------------------------------
    // BLAS 计算例程的基本信息 
    //----------------------------------------------------------------------------------
    //例程
    CallInst * routine_callee;
	//例程码(含有精度信息)
	COALA_BLAS_ROUTINES_CODE routine_code;
    //例程参数
    std::unordered_map<std::string, Value*> routine_callee_params;

    public:
    //get 方法
    CallInst * getRoutineCallee();
	std::string getRoutineName();
    size_t getTaskCode();
    Value * getRoutineParam(std::string param_name);

    

    //任务重构
    virtual void reconstrcuting() = 0;
    //打印输出
    virtual void dump() =0;
};



class CoalaBlasTaskFactory
{   
    public:
    static std::shared_ptr<CoalaBlasTask> createACoalaBlasTask(CallInst * CI, COALA_BLAS_ROUTINES_CODE const namecode, size_t const taskid);
};


}//end of namespace

#endif