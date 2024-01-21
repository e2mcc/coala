#ifndef _COALA_TRAVERSE_ENTRANCE_H
#define _COALA_TRAVERSE_ENTRANCE_H
/*********************************************
* Include
**********************************************/
#include <llvm/IR/PassManager.h>

#include "coala_traverse_implement.h"

/*********************************************
* NameSpace
**********************************************/
namespace llvm {

/*============================================
| Pass 主入口（必须条件1/5）
| 自定义 PASS 须继承自 PassInfoMixin<Tamplet> 模板
==============================================*/
class TraversePass : public PassInfoMixin<TraversePass>
{
	public:
	//重载run方法（必须条件2/5）
	PreservedAnalyses run(Module & M, ModuleAnalysisManager &MAM);

	private:
	TraverseImp TI;
};

}//namespace llvm

#endif
