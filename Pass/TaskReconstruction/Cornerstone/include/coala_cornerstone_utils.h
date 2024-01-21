#ifndef _COALA_PASS_UTILS_H
#define _COALA_PASS_UTILS_H

/************************************************************
* Include
*************************************************************/
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>



/************************************************************
* NameSpace
*************************************************************/

using namespace llvm;


/************************************************************
* Function Denote
*************************************************************/
std::string getDemangledName(const Function & F);
std::string getDemangledName(const Function * F);
std::string getDemangledName(std::string mangledName);

void unreachable(std::string header, Value * V);

int getInstructionSequenceNumber(Instruction * inst);
int getSequnentialDistanceBetweenInstructions(Instruction *S, Instruction *E);

/************************************************************
* Tools Regarding Define-Use Chain
*************************************************************/
void eraseInstUseChain(Instruction * inst);

void eraseInstUseChainExcept(Instruction * target_inst, Instruction * except_inst);




/************************************************************
* Tools For Creating A Global Variable
*************************************************************/
GlobalVariable * createGlobalInt32ConstElement
(
	Module & M,
	std::string const name,
	int const val
);


GlobalVariable * createGlobalInt32ConstVector
(
	Module & M,
	std::string const name,
	unsigned int const dimension,
	int const val
);

GlobalVariable *  createGlobalInt32ConstMatrix
(
	Module & M,
	std::string const name,
	unsigned int const m,
	unsigned int const n,
	int const val
);

GlobalVariable * createGlobalStructPtr
(
	Module & M,
	std::string const name
);

/************************************************************
* Tools For Finding A Specific Global Variable By The Name
*************************************************************/
GlobalVariable * findNamedGlobalVariableInModule(Module & M, std::string name);

/************************************************************
* Tools For Finding A Specific Callee By The Name
*************************************************************/
CallInst * findFirstNamedCalleeInFunction(Function & F, std::string name);
CallInst * findFirstNamedCalleeInModule(Module & M, std::string name);


Function * findNamedFunctionDeclaration(Module & M,  std::string name);



#endif