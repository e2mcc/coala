/*********************************************
* Include
**********************************************/
#include "coala_memop_devfree.h"
#include "coala_cornerstone_utils.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>


/*====================================================================
| NameSpace
======================================================================*/
using namespace llvm;


/*====================================================================
| Base Class : CoalaMemopDevFreeCallee
======================================================================*/
// private function : _create_coala_dev_free_declaration
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Function * CoalaMemopDevFreeCallee::_create_coala_memop_devfree_FuncDeclaration(Module & M)
{
	// 目标: 在 Module 中生成
	// 源码形式：
    // 
	
    //判定 Module 中是否已经存在 coala_memop_devfree 函数声明
    Function * _func = findNamedFunctionDeclaration(M, "coala_memop_devfree");
    if(_func != nullptr)
    {
        return _func;
    }
    
    // 获得 Module 环境
	LLVMContext & Mcontext = M.getContext();

    // 1. 声明返回值类型: void
    Type * ret_type = Type::getVoidTy(Mcontext);

    // 2. 声明函数参数类型:
    std::vector<Type *> param_types = 
	{
		Type::getInt8PtrTy(Mcontext), // ptr noundef
        Type::getInt32Ty(Mcontext),  // i32 noundef
        Type::getInt8PtrTy(Mcontext), // ptr noundef
	};

    // 3. 声明返回值类型+函数参数类型=一个函数类型：
    FunctionType * func_type = FunctionType::get(ret_type, param_types, false);

    // 4. 声明函数(如果M中有函数声明就获取，没有就插入这个函数声明)
	_func = Function::Create(func_type, Function::ExternalLinkage, "coala_memop_devfree", &M);

    // 5. 设置参数属性
    _func->addParamAttr(0, Attribute::NoUndef);
    _func->addParamAttr(1, Attribute::NoUndef);
    _func->addParamAttr(2, Attribute::NoUndef);

    return _func;
}


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Get Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CallInst * CoalaMemopDevFreeCallee::getMemopCallee()
{
	return CoalaMemopDevFreeCallee::memop_callee;
}

std::string CoalaMemopDevFreeCallee::getMemopName()
{	
	if (COALA_MEMOP_NAMELIST.find(CoalaMemopDevFreeCallee::memop_code) != COALA_MEMOP_NAMELIST.end()) 
	{
		return COALA_MEMOP_NAMELIST[CoalaMemopDevFreeCallee::memop_code];
	}
    return "Not Found";
}


Value * CoalaMemopDevFreeCallee::getMemopParam(std::string param_name)
{
	return CoalaMemopDevFreeCallee::memop_callee_params[param_name];
}



/*====================================================================
| Derived Class : CoalaMemopDevFreeCallee4Cuda
======================================================================*/
// Set Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDevFreeCallee4Cuda::setMemPtr()
{
	if( CoalaMemopDevFreeCallee4Cuda::memop_callee->arg_size()!=1 ) return nullptr;
	Value * _memptr = CoalaMemopDevFreeCallee4Cuda::memop_callee->getArgOperand(0);
	return _memptr;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Get Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDevFreeCallee4Cuda::getMemPtr()
{
	return CoalaMemopDevFreeCallee4Cuda::getMemopParam("MemPtr");
}



//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// transform2coala Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CoalaMemopDevFreeCallee4Cuda::transform2coala(GlobalValue * probelist, size_t const taskid)
{
	outs()<<"----Transforming-->"<<*CoalaMemopDevFreeCallee4Cuda::memop_callee<<"\n";

	// 判断是否已被转化
	if( CoalaMemopDevFreeCallee4Cuda::hasbeencoala )
	{
		outs()<<"------已被转化为-->"<<*CoalaMemopDevFreeCallee4Cuda::memop_callee<<"\n";
		return;
	}

	//获得Module
	Module * M_ptr = CoalaMemopDevFreeCallee4Cuda::memop_callee->getModule();

	// 创建函数声明
	CoalaMemopDevFreeCallee4Cuda::coala_memop_devfree
		 = CoalaMemopDevFreeCallee4Cuda::_create_coala_memop_devfree_FuncDeclaration(*M_ptr);
	
	//获得Module环境
	LLVMContext & Mcontext = M_ptr->getContext();

	//创建IRBuilder对象
	IRBuilder<> IRB(Mcontext);

	//获取插入点:在callee指令前
	IRB.SetInsertPoint(CoalaMemopDevFreeCallee4Cuda::memop_callee);

	//创建一个constant
    Type * element_type = Type::getInt32Ty(Mcontext);
	Constant* element = ConstantInt::get(element_type, taskid);


	// 7. 创建 %3 = load ptr, ptr @probelist, align 8 指令
    LoadInst * retload = IRB.CreateLoad(Type::getInt8PtrTy(Mcontext), probelist);

	//插入参数设置
	Value * args[] = 
	{
		//插入probelist
		retload,
		//插入task_id
		element,
		CoalaMemopDevFreeCallee4Cuda::getMemPtr(),
	};

	//创建callee
	CallInst * ret = IRB.CreateCall(CoalaMemopDevFreeCallee4Cuda::coala_memop_devfree,args);

	if(ret==nullptr)
	{
		errs()<<"WRONG in CoalaMemopDevFreeCallee4Cuda::transform2coala()\n";
		return;
	}
	// outs()<<"------已成功插入"<<*ret<<"\n";

	//替换原指令返回值影响
	CoalaMemopDevFreeCallee4Cuda::memop_callee->replaceAllUsesWith(ret);

	// outs()<<"------已成功替换"<<*CoalaMemopDevFreeCallee4Cuda::memop_callee<<"\n";

	//删除原指令
	CoalaMemopDevFreeCallee4Cuda::memop_callee->eraseFromParent();

    return;
}




/*====================================================================
| Derived Class : CoalaMemopDevFreeCallee4OpenCL
======================================================================*/
// Set Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDevFreeCallee4OpenCL::setMemPtr()
{
	if( CoalaMemopDevFreeCallee4OpenCL::memop_callee->arg_size()!=1 ) return nullptr;
	Value * _memptr = CoalaMemopDevFreeCallee4OpenCL::memop_callee->getArgOperand(0);
	return _memptr;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Get Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDevFreeCallee4OpenCL::getMemPtr()
{
	return CoalaMemopDevFreeCallee4OpenCL::getMemopParam("MemPtr");
}



//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// transform2coala Function
// COALA-TODO:待验证
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CoalaMemopDevFreeCallee4OpenCL::transform2coala(GlobalValue * probelist, size_t const taskid)
{
    return;
}