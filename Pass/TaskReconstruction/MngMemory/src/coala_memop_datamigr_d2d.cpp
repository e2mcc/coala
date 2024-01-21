/*====================================================================
| INCLUDE
======================================================================*/
#include "coala_memop_datamigr_d2d.h"
#include "coala_cornerstone_utils.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>

/*====================================================================
| NameSpace
======================================================================*/
using namespace llvm;


/*====================================================================
| Derived Subbase Class : CoalaMemopDataMigrationD2DCallee
======================================================================*/
void CoalaMemopDataMigrationD2DCallee::transform2coala(GlobalValue * probelist, size_t const taskid)
{
	return;
}