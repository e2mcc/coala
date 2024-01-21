#ifndef _COALA_TASK_GRAPH_H
#define _COALA_TASK_GRAPH_H

/*********************************************
* Include
**********************************************/
#include <llvm/IR/Module.h>


namespace llvm{

/*********************************************
* Class
**********************************************/
class CoalaTaskGraph
{   
    private:
    CoalaTaskGraph(){}//不可公开调用
    GlobalVariable *  graph_mat;
    unsigned int m;//行维度
    unsigned int n;//列维度
    GlobalVariable *  _graphInitialize(Module & M);
    
    void _addMatrixArcToConstantAggregateZero(Module & M,unsigned int const task_a_id, unsigned int const task_b_id);
    void _addMatrixArcToConstantDataSequential(Module & M,unsigned int const task_a_id, unsigned int const task_b_id);


    public:
    CoalaTaskGraph(Module & M);
    void setArcA2B(Module & M,unsigned int const task_a_id, unsigned int const task_b_id);
    int getArcA2B(unsigned int const task_a_id, unsigned int const task_b_id);

    void dump();
};

}//end of namespace
#endif