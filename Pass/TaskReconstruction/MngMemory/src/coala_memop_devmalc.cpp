/*====================================================================
| INCLUDE
======================================================================*/
#include "coala_memop_devmalc.h"
#include "coala_cornerstone_utils.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>

/*====================================================================
| NameSpace
======================================================================*/
using namespace llvm;



/*====================================================================
| Base Class : CoalaMemopDevMalcCallee
======================================================================*/
// private function : _create_coala_memop_devmalc_declaration
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Function * CoalaMemopDevMalcCallee::_create_coala_memop_devmalc_FuncDeclaration(Module & M)
{	
	// 目标: 在 Module 中生成
	// 源码形式：void coala_memop_devmalc(coala_problist * probes,size_t taskid, double * ptr, size_t size );
    // declare void @coala_memop_devmalc(ptr noundef, i32 noundef, ptr noundef, i32 noundef) 语句
	
    //判定 Module 中是否已经存在 coala_probe 函数声明
    Function * _func = findNamedFunctionDeclaration(M, "coala_memop_devmalc");
    if(_func != nullptr)
    {
        return _func;
    }
    
    // 获得 Module 环境
	LLVMContext & Mcontext = M.getContext();

    // 1. 声明返回值类型: i32
    Type * ret_type = Type::getInt32Ty(Mcontext);

    // 2. 声明函数参数类型:
    std::vector<Type *> param_types = 
	{
		Type::getInt8PtrTy(Mcontext), // ptr noundef
        Type::getInt32Ty(Mcontext),  // i32 noundef
        Type::getInt8PtrTy(Mcontext), // ptr noundef
        Type::getInt32Ty(Mcontext)  // i32 noundef
	};

    // 3. 声明返回值类型+函数参数类型=一个函数类型：
    FunctionType * func_type = FunctionType::get(ret_type, param_types, false);

    // 4. 声明函数(如果M中有函数声明就获取，没有就插入这个函数声明)
	_func = Function::Create(func_type, Function::ExternalLinkage, "coala_memop_devmalc", &M);

    // 5. 设置参数属性
    _func->addParamAttr(0, Attribute::NoUndef);
    _func->addParamAttr(1, Attribute::NoUndef);
    _func->addParamAttr(2, Attribute::NoUndef);
    _func->addParamAttr(3, Attribute::NoUndef);


    return _func;
}


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Get Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CallInst * CoalaMemopDevMalcCallee::getMemopCallee()
{
	return CoalaMemopDevMalcCallee::memop_callee;
}

std::string CoalaMemopDevMalcCallee::getMemopName()
{	
	if (COALA_MEMOP_NAMELIST.find(CoalaMemopDevMalcCallee::memop_code) != COALA_MEMOP_NAMELIST.end()) 
	{
		return COALA_MEMOP_NAMELIST[CoalaMemopDevMalcCallee::memop_code];
	}
    return "Not Found";
}


Value * CoalaMemopDevMalcCallee::getMemopParam(std::string param_name)
{
	return CoalaMemopDevMalcCallee::memop_callee_params[param_name];
}



/*====================================================================
| Derived Class : CoalaMemopDevMalcCallee4Cuda
======================================================================*/
// Set Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDevMalcCallee4Cuda::setMemHandlePtrPtr()
{
	if( CoalaMemopDevMalcCallee4Cuda::memop_callee->arg_size()!=2 ) return nullptr;
	Value * _mem_handle_ptr_ptr = CoalaMemopDevMalcCallee4Cuda::memop_callee->getArgOperand(0);
	return _mem_handle_ptr_ptr;
}


Value * CoalaMemopDevMalcCallee4Cuda::setSize()
{
	if( CoalaMemopDevMalcCallee4Cuda::memop_callee->arg_size()!=2 ) return nullptr;
	Value * _size = CoalaMemopDevMalcCallee4Cuda::memop_callee->getArgOperand(1);
	return _size;
}


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Get Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDevMalcCallee4Cuda::getMemHandlePtrPtr()
{
	return CoalaMemopDevMalcCallee4Cuda::getMemopParam("MemHandlePtrPtr");
}

Value * CoalaMemopDevMalcCallee4Cuda::getSize()
{
	return CoalaMemopDevMalcCallee4Cuda::getMemopParam("Size");
}



//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// transform2coala Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/**
 * @brief 转化为coala_memop_devmlac(probelist,taskID,target_ptr,size,...)
 * @param taskid 任务ID
 */
void CoalaMemopDevMalcCallee4Cuda::transform2coala(GlobalValue * probelist, size_t const taskid)
{	
	outs()<<"----Transforming-->"<<*CoalaMemopDevMalcCallee4Cuda::memop_callee<<"\n";
	
	// 判断是否已被转化
	if( CoalaMemopDevMalcCallee4Cuda::hasbeencoala )
	{
		outs()<<"------已被转化为-->"<<*CoalaMemopDevMalcCallee4Cuda::memop_callee<<"\n";
		return;
	}

	//获得Module
	Module * M_ptr = CoalaMemopDevMalcCallee4Cuda::memop_callee->getModule();

	// 创建函数声明
	CoalaMemopDevMalcCallee4Cuda::coala_memop_devmalc
		 = CoalaMemopDevMalcCallee4Cuda::_create_coala_memop_devmalc_FuncDeclaration(*M_ptr);
	
	//获得Module环境
	LLVMContext & Mcontext = M_ptr->getContext();

	//创建IRBuilder对象
	IRBuilder<> IRB(Mcontext);

	//获取插入点:在callee指令前
	IRB.SetInsertPoint(CoalaMemopDevMalcCallee4Cuda::memop_callee);

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
		CoalaMemopDevMalcCallee4Cuda::getMemHandlePtrPtr(),
		//插入size
		CoalaMemopDevMalcCallee4Cuda::getSize()
	};

	//创建callee
	CallInst * ret = IRB.CreateCall(CoalaMemopDevMalcCallee4Cuda::coala_memop_devmalc,args);

	if(ret==nullptr)
	{
		errs()<<"WRONG in CoalaMemopDevMalcCallee4Cuda::transform2coala()\n";
		return;
	}
	// outs()<<"------已成功插入"<<*ret<<"\n";

	//替换原指令返回值影响
	CoalaMemopDevMalcCallee4Cuda::memop_callee->replaceAllUsesWith(ret);

	// outs()<<"------已成功替换"<<*CoalaMemopDevMalcCallee4Cuda::memop_callee<<"\n";

	//删除原指令
	CoalaMemopDevMalcCallee4Cuda::memop_callee->eraseFromParent();

	return;
}



/*====================================================================
| Derived Class : CoalaMemopDevMalcCallee4OpenCL
======================================================================*/
// Set Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDevMalcCallee4OpenCL::setOpenclContext()
{
	if( CoalaMemopDevMalcCallee4OpenCL::memop_callee->arg_size()!=5 ) return nullptr;
	Value * _opencl_context = CoalaMemopDevMalcCallee4OpenCL::memop_callee->getArgOperand(0);
	return _opencl_context;
}

Value * CoalaMemopDevMalcCallee4OpenCL::setOpenclMemOpFlag()
{
	if( CoalaMemopDevMalcCallee4OpenCL::memop_callee->arg_size()!=5 ) return nullptr;
	Value * _opencl_mem_op_flag = CoalaMemopDevMalcCallee4OpenCL::memop_callee->getArgOperand(1);
	return _opencl_mem_op_flag;
}

Value * CoalaMemopDevMalcCallee4OpenCL::setSize()
{
	if( CoalaMemopDevMalcCallee4OpenCL::memop_callee->arg_size()!=5 ) return nullptr;
	Value * _size = CoalaMemopDevMalcCallee4OpenCL::memop_callee->getArgOperand(2);
	return _size;
}

Value * CoalaMemopDevMalcCallee4OpenCL::setOpenclHostPtr()
{
	if( CoalaMemopDevMalcCallee4OpenCL::memop_callee->arg_size()!=5 ) return nullptr;
	Value * _opencl_host_ptr = CoalaMemopDevMalcCallee4OpenCL::memop_callee->getArgOperand(3);
	return _opencl_host_ptr;
}


Value * CoalaMemopDevMalcCallee4OpenCL::setOpenclErrCodePtr()
{
	if( CoalaMemopDevMalcCallee4OpenCL::memop_callee->arg_size()!=5 ) return nullptr;
	Value * _opencl_err_code_ptr = CoalaMemopDevMalcCallee4OpenCL::memop_callee->getArgOperand(4);
	return _opencl_err_code_ptr;
}



//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Get Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDevMalcCallee4OpenCL::getOpenclContext()
{
	return CoalaMemopDevMalcCallee4OpenCL::getMemopParam("OpenclContext");
}

Value * CoalaMemopDevMalcCallee4OpenCL::getOpenclMemOpFlag()
{
	return CoalaMemopDevMalcCallee4OpenCL::getMemopParam("OpenclMemOpFlag");
}


Value * CoalaMemopDevMalcCallee4OpenCL::getSize()
{
	return CoalaMemopDevMalcCallee4OpenCL::getMemopParam("Size");
}

Value * CoalaMemopDevMalcCallee4OpenCL::getOpenclHostPtr()
{
	return CoalaMemopDevMalcCallee4OpenCL::getMemopParam("OpenclHostPtr");
}

Value * CoalaMemopDevMalcCallee4OpenCL::getOpenclErrCodePtr()
{
	return CoalaMemopDevMalcCallee4OpenCL::getMemopParam("OpenclErrCodePtr");
}


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// transform2coala Function
// COALA-TODO:待验证
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/**
 * @brief 转化为coala_memop_devmlac(probelist,taskID, target_ptr, size,...)
 * @param task_id 任务ID
 */
void CoalaMemopDevMalcCallee4OpenCL::transform2coala(GlobalValue * probelist, size_t const taskid)
{	
	// 创建函数声明

	return;
}