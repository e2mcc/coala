/*====================================================================
| INCLUDE
======================================================================*/
#include "coala_memop_hostmalc.h"
#include "coala_cornerstone_utils.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>

/*====================================================================
| NameSpace
======================================================================*/
using namespace llvm;



/*====================================================================
| Base Class : CoalaMemopHostMalcCallee
======================================================================*/
// private function : _create_coala_host_malc_declaration
// int coala_host_malc(coala_probes_t *, int, void**, size_t, ...)
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CoalaMemopHostMalcCallee::_create_coala_memop_hostmalc_FuncDeclaration(Module & M)
{	
	return;
}


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Get Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CallInst * CoalaMemopHostMalcCallee::getMemopCallee()
{
	return CoalaMemopHostMalcCallee::memop_callee;
}

std::string CoalaMemopHostMalcCallee::getMemopName()
{	
	if (COALA_MEMOP_NAMELIST.find(CoalaMemopHostMalcCallee::memop_code) != COALA_MEMOP_NAMELIST.end()) 
	{
		return COALA_MEMOP_NAMELIST[CoalaMemopHostMalcCallee::memop_code];
	}
    return "Not Found";
}


Value * CoalaMemopHostMalcCallee::getMemopParam(std::string param_name)
{
	return CoalaMemopHostMalcCallee::memop_callee_params[param_name];
}



/*====================================================================
| Derived Class : CoalaMemopHostMalcCallee4Linux
======================================================================*/
// Set Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopHostMalcCallee4Linux::setSize()
{
	if( CoalaMemopHostMalcCallee4Linux::memop_callee->arg_size()!=1 ) return nullptr;
	Value * _size = CoalaMemopHostMalcCallee4Linux::memop_callee->getArgOperand(0);
	return _size;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Get Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopHostMalcCallee4Linux::getSize()
{
	return CoalaMemopHostMalcCallee4Linux::getMemopParam("Size");
}


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// transform2coala Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/**
 * @brief 转化为coala_memop_hostmlac(probelist,taskID,memptrptr,size)
 * @param task_id 任务ID
 */
void CoalaMemopHostMalcCallee4Linux::transform2coala(size_t const taskid)
{	
	return;
}