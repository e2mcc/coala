#ifndef _COALA_TASK_H
#define _COALA_TASK_H

/*********************************************
* Include
**********************************************/
#include <vector>
#include <string>
#include <llvm/IR/Instructions.h>

/*********************************************
* Namespace
**********************************************/

namespace llvm{

/*********************************************
* Class
**********************************************/
class CoalaTask
{
    protected:
    CoalaTask(){}

    size_t taskid;

    public:
    virtual void reconstrcuting() = 0;
    
};

}//end of namespace
#endif