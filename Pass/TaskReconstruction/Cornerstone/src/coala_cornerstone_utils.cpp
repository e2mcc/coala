/************************************************
* Include
*************************************************/
#include "coala_cornerstone_utils.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>

/************************************************
* Function
*************************************************/

std::string getDemangledName(const Function &F)
{
	ItaniumPartialDemangler IPD;
	std::string name = F.getName().str();
	if (IPD.partialDemangle(name.c_str())) return name;
	if (IPD.isFunction())
		return IPD.getFunctionBaseName(nullptr, nullptr);
	else 
		return IPD.finishDemangle(nullptr, nullptr);
}

std::string getDemangledName(const Function *F) 
{ 
	return getDemangledName(*F); 
}

std::string getDemangledName(std::string mangledName) 
{
	ItaniumPartialDemangler IPD;
	if (IPD.partialDemangle(mangledName.c_str())) return mangledName;

	if (IPD.isFunction())
		return IPD.getFunctionBaseName(nullptr, nullptr);
	else
		return IPD.finishDemangle(nullptr, nullptr);
}

void unreachable(std::string header, Value *V) 
{
	std::string msg;
	raw_string_ostream rso(msg);
	rso << header << ": " << *V << "\n";
	llvm_unreachable(rso.str().c_str());
}

int getInstructionSequenceNumber(Instruction * inst)
{
	Function * Func = inst->getFunction();
	int counter = 0;
	for (inst_iterator it = inst_begin(*Func); it != inst_end(*Func); it++) 
	{
		Instruction * I = &*it;
		if (I == inst) break;
		counter++;
	}
	return counter;
}

int getSequnentialDistanceBetweenInstructions(Instruction *S, Instruction *E) 
{
	// 输入判断：两条指令必须在同一个 function 中
	if(S->getFunction() != E->getFunction()) 
	{
		outs()<<"WRONG: 两条指令不在同一个FUNCTION中, 无法直接比较"<<"\n";
		exit(0);
	}
	// 获取指令序列号
	int S_seq = getInstructionSequenceNumber(S);
	int E_seq = getInstructionSequenceNumber(E);
	return E_seq - S_seq;
}

//递归删除use树
void eraseInstUseChain(Instruction * inst)
{	
	//这个make_early_inc_range非常重要
	//因为删除迭代器的内容会使得迭代器越界
	for(auto U : make_early_inc_range(inst->users()))
	{	
		if(isa<Instruction>(U))
		{
			Instruction * I = dyn_cast<Instruction>(U);
			eraseInstUseChain(I);
			outs()<<"将删除: "<<*I<<"\n";

			//如果要删除的这个指令是一个分支指令
			//那可推出：此指令必然是个分支条件跳转指令，且其使用到的操作数必定是0号
			//所以无脑走第一个可选分支( getOperand(2) )
			if(isa<BranchInst>(I))
			{	
				BranchInst * Br = dyn_cast<BranchInst>(I);
				if(Br->isConditional())//为了保险起见还是判断一下是不是分支跳转指令
				{
					IRBuilder<> builder(Br);
					BasicBlock * BB = dyn_cast<BasicBlock>(Br->getOperand(2));
					builder.CreateBr(BB);
				}
				else
				{
					//报错
					outs()<<"Wrong:该分支指令:"<<*Br<<"不是条件跳转指令！无法删除\n";
					return;
				}
			}

			I->eraseFromParent();
		}
	}
	return;
}


void eraseInstUseChainExcept(Instruction * target_inst, Instruction * except_inst)
{
	//这个make_early_inc_range非常重要
	//因为删除迭代器的内容会使得迭代器越界
	for(auto U : make_early_inc_range(target_inst->users()))
	{	
		if(isa<Instruction>(U))
		{
			Instruction * I = dyn_cast<Instruction>(U);
			
			if(I==except_inst) continue;//与eraseInstUseChain()不同的地方在这儿

			eraseInstUseChain(I);
			
			outs()<<"将删除: "<<*I<<"\n";

			//如果要删除的这个指令是一个分支指令
			//那可推出：此指令必然是个分支条件跳转指令，且其使用到的操作数必定是0号
			//所以无脑走第一个可选分支( getOperand(2) )
			if(isa<BranchInst>(I))
			{	
				BranchInst * Br = dyn_cast<BranchInst>(I);
				if(Br->isConditional())//为了保险起见还是判断一下是不是分支跳转指令
				{
					IRBuilder<> builder(Br);
					BasicBlock * BB = dyn_cast<BasicBlock>(Br->getOperand(2));
					builder.CreateBr(BB);
				}
				else
				{
					//报错
					outs()<<"Wrong:该分支指令:"<<*Br<<"不是条件跳转指令！无法删除\n";
					return;
				}
			}

			I->eraseFromParent();
		}
	}
	return;
}


//在Module中创建全局常量: @name = constant i32 val
GlobalVariable * createGlobalInt32ConstElement
(
	Module & M,
	std::string const name,
	int const val
) 
{	
	//获得Module环境
	LLVMContext & Ctx = M.getContext();

	//----------------------------------------
	// 1.创建元素
	//----------------------------------------
	// 创建数组元素类型
    Type * element_type = Type::getInt32Ty(Ctx);
	Constant* element = ConstantInt::get(element_type, val);

	//----------------------------------------
	// 2.声明全局变量
	//----------------------------------------
	// 在 M 中插入全局变量
	M.getOrInsertGlobal(name, element_type);

	//获取全局变量
	GlobalVariable * global_element = M.getNamedGlobal(name);

	//设置 linkage 属性为 dso_local
    global_element->setDSOLocal(true);

	// 将常量数组赋值给 global_arr
    global_element->setInitializer(element);

	return global_element;
}




GlobalVariable * createGlobalInt32ConstVector
(
	Module & M,
	std::string const name,
	unsigned int const dimension,
	int const val
)
{
	//获得Module环境
	LLVMContext & Ctx = M.getContext();

	//----------------------------------------
	// 1.创建元素
	//----------------------------------------
	// 创建数组元素类型
    Type * element_type = Type::getInt32Ty(Ctx);
	Constant * element = ConstantInt::get(element_type, val);

	//----------------------------------------
	// 2.创建单行
	//----------------------------------------
	// 创建数组类型
    ArrayType * row_type = ArrayType::get(element_type, dimension);
	std::vector<Constant*> element_vec;
	for(unsigned int i=0; i<dimension; i++)
	{
		element_vec.push_back(element);
	}
	ArrayRef<Constant*> row_ref(element_vec);
	Constant * row = ConstantArray::get(row_type, row_ref);

	//----------------------------------------
	// 3.声明全局变量
	//----------------------------------------
	// 在 M 中插入全局变量
	M.getOrInsertGlobal(name, row_type);

	//获取全局变量
	GlobalVariable * global_vec = M.getNamedGlobal(name);

	//设置 linkage 属性为 dso_local
	global_vec->setDSOLocal(true);

	// 将常量数组赋值给 global_arr
    global_vec->setInitializer(row);

	return global_vec;
}


GlobalVariable * createGlobalInt32ConstMatrix
(
	Module & M,
	std::string const name,
	unsigned int const m, //rows_dimension
	unsigned int const n, //columns_dimension
	int const val
)
{
	// 获得Module环境
	LLVMContext & Ctx = M.getContext();

	//----------------------------------------
	// 1.创建元素
	//----------------------------------------
	// 创建数组元素类型
    Type * element_type = Type::getInt32Ty(Ctx);
	Constant* element = ConstantInt::get(element_type, val);
    
	//----------------------------------------
	// 2.创建单行
	//----------------------------------------
	// 创建数组类型
    ArrayType * row_type = ArrayType::get(element_type, n);
	std::vector<Constant*> element_vec;
	for(unsigned int i=0; i<n; i++)
	{
		element_vec.push_back(element);
	}
	ArrayRef<Constant*> row_ref(element_vec);
	Constant * row = ConstantArray::get(row_type, row_ref);


	//----------------------------------------
	// 3.创建矩阵
	//----------------------------------------
    // 创建矩阵类型
	ArrayType * matrix_type = ArrayType::get(row_type, m);
	std::vector<Constant*> row_vec;
	for(unsigned int i=0; i<m; i++)
	{
		row_vec.push_back(row);
	}
	ArrayRef<Constant*> matrix_ref(row_vec);
	Constant * matrix = ConstantArray::get(matrix_type, row_vec);

	//----------------------------------------
	// 4.声明全局变量
	//----------------------------------------
	// 在 M 中插入全局变量
	M.getOrInsertGlobal(name, matrix_type);

	//获取全局变量
	GlobalVariable * global_mat = M.getNamedGlobal(name);

	//设置 linkage 属性为 dso_local
    global_mat->setDSOLocal(true);

	// 将常量数组赋值给 global_arr
    global_mat->setInitializer(matrix);

	return global_mat;
}


GlobalVariable * createGlobalStructPtr
(
	Module & M,
	std::string const name
)
{	

	// 获得Module环境
	LLVMContext & Mcontext = M.getContext();

	//----------------------------------------
	// 1.创建指针类型
	//----------------------------------------
	PointerType * PtrTy = Type::getInt8PtrTy(Mcontext);

	//----------------------------------------
	// 2.声明全局变量
	//----------------------------------------
	// 在 M 中插入全局变量
	M.getOrInsertGlobal(name, PtrTy);

	//----------------------------------------
	// 3.声明全局变量
	//----------------------------------------
	// 获取全局变量
	GlobalVariable * global_ptr = M.getNamedGlobal(name);

	// 设置初始值
	global_ptr->setInitializer(Constant::getNullValue(PtrTy));

	//设置 linkage 属性为 dso_local
    global_ptr->setDSOLocal(true);

	//设置对齐
    global_ptr->setAlignment(Align(8));

	return global_ptr;
}


/************************************************************
* Tools For Finding A Specific Global Variable By The Name
*************************************************************/
GlobalVariable * findNamedGlobalVariableInModule(Module & M, std::string name)
{	
	return M.getNamedGlobal(name);
}


CallInst * findFirstNamedCalleeInFunction(Function & F, std::string name)
{
	//遍历整个Function的每一条指令
	for (inst_iterator I = inst_begin(F); I != inst_end(F); ++I)
	{	
		//用一个Instruction基类对象的引用来接收迭代器I
		Instruction & Inst = *I;
		//判断是否是一个函数调用指令
		if (isa<CallInst>(Inst))
		{	
			CallInst * CI = dyn_cast<CallInst>(&Inst);
			Function * callee = CI->getCalledFunction();
			auto _name = getDemangledName(*callee);
			if(_name == name) 
				return CI;
		}
	}
	return nullptr;
}

CallInst * findFirstNamedCalleeInModule(Module & M, std::string name)
{
	//遍历整个Function的每一条指令
	for(auto M_iterator = M.getFunctionList().begin(); M_iterator != M.getFunctionList().end(); M_iterator++)
	{	
		//声明一个F
		Function & F = * M_iterator;
		CallInst * CI = findFirstNamedCalleeInFunction(F, name);
		return CI;
	}
	return nullptr;
}



Function * findNamedFunctionDeclaration(Module & M,  std::string name)
{
	for(auto M_iterator = M.getFunctionList().begin(); 
		M_iterator != M.getFunctionList().end(); M_iterator++)
	{	
		//声明一个F
		Function & F = *M_iterator;
		std::string _name = getDemangledName(F);
		if(name == _name)
		{
			return &F;
		}
	}
	return nullptr;
}