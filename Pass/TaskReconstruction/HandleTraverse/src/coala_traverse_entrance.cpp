/************************************************
* Include
*************************************************/
#include "coala_traverse_entrance.h"
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>
#include <string>
/***********************************************
* NameSpace
************************************************/

using namespace llvm;

/*==================================================
| Pass 主入口（必须条件3/5）
| 实现重载 run 方法，需要特定的 return 返回值
===================================================*/
PreservedAnalyses TraversePass::run(Module & M, ModuleAnalysisManager &MAM)
{
	outs()<<"This is the main TraversePass\n";

	TI.traverseModule(M);

	
	PreservedAnalyses PA;
	return PA.none();
}