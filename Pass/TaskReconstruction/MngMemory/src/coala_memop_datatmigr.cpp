/*====================================================================
| INCLUDE
======================================================================*/
#include "coala_memop_datamigr.h"
#include "coala_cornerstone_utils.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>

/*====================================================================
| NameSpace
======================================================================*/
using namespace llvm;



/*====================================================================
| Derived Base Class : CoalaDataTransmissionCallee
======================================================================*/
// Get Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CallInst * CoalaMemopDataMigrationCallee::getMemopCallee()
{
	return CoalaMemopDataMigrationCallee::memop_callee;
}

std::string CoalaMemopDataMigrationCallee::getMemopName()
{	
	if (COALA_MEMOP_NAMELIST.find(CoalaMemopDataMigrationCallee::memop_code) != COALA_MEMOP_NAMELIST.end()) 
	{
		return COALA_MEMOP_NAMELIST[CoalaMemopDataMigrationCallee::memop_code];
	}
    return "Not Found";
}


Value * CoalaMemopDataMigrationCallee::getMemopParam(std::string param_name)
{
	return CoalaMemopDataMigrationCallee::memop_callee_params[param_name];
}
