/*====================================================================
| INCLUDE
======================================================================*/
#include "coala_memop_hostfree.h"
#include "coala_cornerstone_utils.h"
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IRBuilder.h>

/*====================================================================
| NameSpace
======================================================================*/
using namespace llvm;



/*====================================================================
| Base Class : CoalaMemopHostFreeCallee
======================================================================*/
// private function : _create_coalahostfree_declaration
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CoalaMemopHostFreeCallee::_create_coala_memop_hostfree_FuncDeclaration(Module & M)
{
	return;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Get Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CallInst * CoalaMemopHostFreeCallee::getMemopCallee()
{
	return CoalaMemopHostFreeCallee::memop_callee;
}

std::string CoalaMemopHostFreeCallee::getMemopName()
{	
	if (COALA_MEMOP_NAMELIST.find(CoalaMemopHostFreeCallee::memop_code) != COALA_MEMOP_NAMELIST.end()) 
	{
		return COALA_MEMOP_NAMELIST[CoalaMemopHostFreeCallee::memop_code];
	}
    return "Not Found";
}


Value * CoalaMemopHostFreeCallee::getMemopParam(std::string param_name)
{
	return CoalaMemopHostFreeCallee::memop_callee_params[param_name];
}




/*====================================================================
| Derived Class : CoalaMemopHostFreeCallee4Linux
======================================================================*/
// Set Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopHostFreeCallee4Linux::setMemPtr()
{
	if( CoalaMemopHostFreeCallee4Linux::memop_callee->arg_size()!=1 ) return nullptr;
	Value * _memptr = CoalaMemopHostFreeCallee4Linux::memop_callee->getArgOperand(0);
	return _memptr;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Get Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopHostFreeCallee4Linux::getMemPtr()
{
	return CoalaMemopHostFreeCallee4Linux::getMemopParam("MemPtr");
}



//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// transform2coala Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CoalaMemopHostFreeCallee4Linux::transform2coala(size_t const taskid)
{
    return;
}