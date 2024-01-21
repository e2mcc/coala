#ifndef _COALA_TRAVERSE_IMPLEMENT_H
#define _COALA_TRAVERSE_IMPLEMENT_H
/*********************************************
* Include
**********************************************/
#include <llvm/IR/Module.h>//for Module class
#include <llvm/IR/Function.h>//for Function class
#include <llvm/IR/Instructions.h> 

/*********************************************
* NameSpace
**********************************************/
namespace llvm {

/*============================================
| Pass 主入口（必须条件1/5）
| 自定义 PASS 须继承自 PassInfoMixin<Tamplet> 模板
==============================================*/
class TraverseImp
{
	public:
	TraverseImp(){}
	void traverseModule(Module & M);
	void traverseFunction(Function & F);
	void traverseBasicBlock(BasicBlock & B);

	private:
	void visitInstruction(Instruction & I);
};

}//namespace llvm

#endif