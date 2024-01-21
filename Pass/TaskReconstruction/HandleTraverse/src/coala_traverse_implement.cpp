/************************************************
* Include
*************************************************/
#include "coala_traverse_implement.h"
#include <llvm/IR/Metadata.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/Support/raw_ostream.h>

/************************************************
* NameSpace
*************************************************/

using namespace llvm;

/************************************************
* 函数实现
*************************************************/

void TraverseImp::traverseModule(Module & M)
{
	for(auto M_iterator = M.getFunctionList().begin(); 
		M_iterator != M.getFunctionList().end(); M_iterator++)
	{	
		//声明一个F
		Function & F = *M_iterator;
		TraverseImp::traverseFunction(F);
	}
}


void TraverseImp::traverseFunction(Function & F)
{
	for(BasicBlock & B : F)
	{
		TraverseImp::traverseBasicBlock(B);
	}
}


void TraverseImp::traverseBasicBlock(BasicBlock & B)
{
	//访问BasicBlock中的每个指令
	for(Instruction & I : B)
	{
		TraverseImp::visitInstruction(I);
	}

	outs()<<"\n";
}


// PRIVATE FUNCTION
void TraverseImp::visitInstruction(Instruction & I)
{	
	unsigned line = 0;
	//看看该IR指令是否存在metadata
	if(I.hasMetadata())
	{	
		MDNode * N = I.getMetadata("dbg");
		if(N!=NULL)
		{
			if(isa<DILocation>(N))
			{
				DILocation * DILoc = dyn_cast<DILocation>(N);
				line = DILoc->getLine();
				// StringRef File = DILoc->getFilename();
			}
		}		
	}

	outs()<<"line: "<<line<<" "<<I<<"\n";
}