#ifndef _COALA_BLAS_MANAGEMENT_H
#define _COALA_BLAS_MANAGEMENT_H
/************************************************
* Include
*************************************************/
#include "coala_cornerstone_taskgraph.h"
#include "coala_cornerstone_probes.h"
#include "coala_blas_list.h"
#include "coala_blas_task.h"
#include <llvm/IR/Module.h>
#include <vector>
#include <unordered_map>
/************************************************
* Name Space
*************************************************/
/**
 * @brief llvm 命名空间
 */
namespace llvm{

/************************************************
* Class
*************************************************/

class BlasManagement
{
    private:
    BlasManagement(){}

    std::vector<std::shared_ptr<CoalaBlasTask>> blas_tasks;
    std::unordered_map<CallInst*, COALA_BLAS_ROUTINES_CODE> blas_routine_callees;
    
    // 扫描整个 Module
    void _scanning(Module & M);

    // 依赖分析
    void _analyzing(Module & M);
   

    //对每一个callee开始重构task
    // void _taskReconstrcuting(Module & M);

    public:
    BlasManagement(Module & M);
    BlasManagement(Module & M, CoalaProbes & CPs);
    BlasManagement(Module & M, CoalaProbes & CPs, CoalaTaskGraph & CTG);
    
};

}//end of llvm
#endif