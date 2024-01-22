/*====================================================================
| INCLUDE
======================================================================*/
#include "coala_blas_gemm_task.h"
#include "coala_cornerstone_utils.h"
#include "coala_cornerstone_probes.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>

//gemm 
#include "coala_blas_gemm_task_for_cblas.h"
#include "coala_blas_gemm_task_for_cublas.h"
#include "coala_blas_gemm_task_for_clblast.h"

/*====================================================================
| NameSpace
======================================================================*/
using namespace llvm;

/*====================================================================
| CoalaBlasGemmTask
======================================================================*/
Function * CoalaBlasGemmTask::_create_coala_blas_sgemm_FuncDeclaration(Module & M)
{
	//判定 Module 中是否已经存在 coala_probe 函数声明
    Function * _func = findNamedFunctionDeclaration(M, "coala_blas_sgemm");
    if(_func != nullptr)
    {
        return _func;
    }

	//获取 Module 环境
	LLVMContext & Mcontext = M.getContext();

	// 1. 设置返回值类型:int
	Type * ret_type = Type::getInt32Ty(Mcontext);

	// 2. 声明函数参数类型:
	std::vector<Type *> param_types = 
	{
		Type::getInt8PtrTy(Mcontext), // ptr noundef probelist
        Type::getInt64Ty(Mcontext),  // i64 noundef	taskid
		//sgemm
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, order
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, transeA
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, transeB
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, M
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, N
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, K
		Type::getFloatPtrTy(Mcontext),	 		//	float* 类型, alpha
		Type::getFloatPtrTy(Mcontext),			//	float* 类型, A
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, lda
		Type::getFloatPtrTy(Mcontext),			//	float* 类型, B
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, ldb
		Type::getFloatPtrTy(Mcontext),	 		//	float* 类型, beta
		Type::getFloatPtrTy(Mcontext),			//	float* 类型, C
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, ldc
	};

	// 声明返回值类型+函数参数类型=一个函数类型：
	FunctionType * func_type = FunctionType::get(ret_type, param_types, false);

	 // 4. 声明函数(如果M中有函数声明就获取，没有就插入这个函数声明)
	_func = Function::Create(func_type, Function::ExternalLinkage, "coala_blas_sgemm", &M);

    // 5. 设置参数属性
    _func->addParamAttr(0, Attribute::NoUndef);
    _func->addParamAttr(1, Attribute::NoUndef);
    _func->addParamAttr(2, Attribute::NoUndef);
    _func->addParamAttr(3, Attribute::NoUndef);
	_func->addParamAttr(4, Attribute::NoUndef);
    _func->addParamAttr(5, Attribute::NoUndef);
    _func->addParamAttr(6, Attribute::NoUndef);
    _func->addParamAttr(7, Attribute::NoUndef);
	_func->addParamAttr(8, Attribute::NoUndef);
	_func->addParamAttr(9, Attribute::NoUndef);
    _func->addParamAttr(10, Attribute::NoUndef);
    _func->addParamAttr(11, Attribute::NoUndef);
    _func->addParamAttr(12, Attribute::NoUndef);
	_func->addParamAttr(13, Attribute::NoUndef);
    _func->addParamAttr(14, Attribute::NoUndef);
    _func->addParamAttr(15, Attribute::NoUndef);
	return _func;
}


Function * CoalaBlasGemmTask::_create_coala_blas_dgemm_FuncDeclaration(Module & M)
{
	//判定 Module 中是否已经存在 coala_probe 函数声明
    Function * _func = findNamedFunctionDeclaration(M, "coala_blas_dgemm");
    if(_func != nullptr)
    {
        return _func;
    }

	//获取 Module 环境
	LLVMContext & Mcontext = M.getContext();

	// 1. 设置返回值类型:int
	Type * ret_type = Type::getInt32Ty(Mcontext);

	// 2. 声明函数参数类型:
	std::vector<Type *> param_types = 
	{
		Type::getInt8PtrTy(Mcontext), // ptr noundef probelist
        Type::getInt64Ty(Mcontext),  // i64 noundef	taskid
		//sgemm
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, order
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, transeA
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, transeB
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, M
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, N
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, K
		Type::getFloatTy(Mcontext),	 			//	float 类型, alpha
		Type::getFloatPtrTy(Mcontext),			//	float* 类型, A
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, lda
		Type::getFloatPtrTy(Mcontext),			//	float* 类型, B
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, ldb
		Type::getFloatTy(Mcontext),	 			//	float 类型, beta
		Type::getFloatPtrTy(Mcontext),			//	float* 类型, C
		Type::getInt32Ty(Mcontext),   			// 	int32 类型, ldc
	};

	// 声明返回值类型+函数参数类型=一个函数类型：
	FunctionType * func_type = FunctionType::get(ret_type, param_types, false);

	 // 4. 声明函数(如果M中有函数声明就获取，没有就插入这个函数声明)
	_func = Function::Create(func_type, Function::ExternalLinkage, "coala_blas_dgemm", &M);

    // 5. 设置参数属性
    _func->addParamAttr(0, Attribute::NoUndef);
    _func->addParamAttr(1, Attribute::NoUndef);
    _func->addParamAttr(2, Attribute::NoUndef);
    _func->addParamAttr(3, Attribute::NoUndef);
	_func->addParamAttr(4, Attribute::NoUndef);
    _func->addParamAttr(5, Attribute::NoUndef);
    _func->addParamAttr(6, Attribute::NoUndef);
    _func->addParamAttr(7, Attribute::NoUndef);
	_func->addParamAttr(8, Attribute::NoUndef);
	_func->addParamAttr(9, Attribute::NoUndef);
    _func->addParamAttr(10, Attribute::NoUndef);
    _func->addParamAttr(11, Attribute::NoUndef);
    _func->addParamAttr(12, Attribute::NoUndef);
	_func->addParamAttr(13, Attribute::NoUndef);
    _func->addParamAttr(14, Attribute::NoUndef);
    _func->addParamAttr(15, Attribute::NoUndef);
	return _func;
}



Value * CoalaBlasGemmTask::_insertLoadMInstruction(CallInst * insertpoint)
{
	Value * _inst = CoalaBlasGemmTask::getRoutineParam("M");
	
	if(_inst==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaBlasGemmTask::_insertLoadMInstruction()\n";
		exit(0);
	}

	// 目标：识别 
	// 源码形式：int M
	// LLVM-IR形式：%9 = alloca i32, align 4

	//-------------------------------------------------------------------------------------
	// 首先向上找到 %9 = alloca i32, align 4
	//-------------------------------------------------------------------------------------

	//设置最大查找层数
	for(size_t i = 0; i<5; i++)
	{
		if(isa<LoadInst>(*_inst))
		{	
			LoadInst * _LI = dyn_cast<LoadInst>(_inst);
			_inst = _LI->getOperand(0);
		}
		else
		{
			break;
		}
	}
	
	//如果不是AllocaInst
	if(!isa<AllocaInst>(*_inst))
	{
		outs()<<"WRONG: Cannot find the alloca inst for this value:\n\t"<<*CoalaBlasGemmTask::getRoutineParam("M")<<"\n";
		exit(0);
	}

	AllocaInst * _alloca = dyn_cast<AllocaInst>(_inst);

	//-------------------------------------------------------------------------------------
	// 插入loadInst指令
	//-------------------------------------------------------------------------------------
	//获取 Module 环境
	LLVMContext & Mcontext = insertpoint->getModule()->getContext();
	
	// 创键一个指令创键环境
	IRBuilder<> builder(Mcontext);
    builder.SetInsertPoint(insertpoint);

	// 7. 创建 %3 = load ptr, ptr @probelist, align 8 指令
    LoadInst * retload = builder.CreateLoad(Type::getInt32Ty(Mcontext),_alloca);
	outs()<<"----Inserted--> "<<*retload<<"\n";
	return retload;
}

Value * CoalaBlasGemmTask::_insertLoadNInstruction(CallInst * insertpoint)
{
	Value * _inst = CoalaBlasGemmTask::getRoutineParam("N");
	
	if(_inst==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaBlasGemmTask::_insertLoadMInstruction()\n";
		exit(0);
	}

	// 目标：识别 
	// 源码形式：int M
	// LLVM-IR形式：%9 = alloca i32, align 4

	//-------------------------------------------------------------------------------------
	// 首先向上找到 %9 = alloca i32, align 4
	//-------------------------------------------------------------------------------------

	//设置最大查找层数
	for(size_t i = 0; i<5; i++)
	{
		if(isa<LoadInst>(*_inst))
		{	
			LoadInst * _LI = dyn_cast<LoadInst>(_inst);
			_inst = _LI->getOperand(0);
		}
		else
		{
			break;
		}
	}
	
	//如果不是AllocaInst
	if(!isa<AllocaInst>(*_inst))
	{
		outs()<<"WRONG: Cannot find the alloca inst for this value:\n\t"<<*CoalaBlasGemmTask::getRoutineParam("N")<<"\n";
		exit(0);
	}

	AllocaInst * _alloca = dyn_cast<AllocaInst>(_inst);


	//-------------------------------------------------------------------------------------
	// 插入loadInst指令
	//-------------------------------------------------------------------------------------
	//获取 Module 环境
	LLVMContext & Mcontext = insertpoint->getModule()->getContext();
	
	// 创键一个指令创键环境
	IRBuilder<> builder(Mcontext);
    builder.SetInsertPoint(insertpoint);

	// 7. 创建 %3 = load ptr, ptr @probelist, align 8 指令
    LoadInst * retload = builder.CreateLoad(Type::getInt32Ty(Mcontext),_alloca);
	outs()<<"----Inserted--> "<<*retload<<"\n";
	return retload;
}

Value * CoalaBlasGemmTask::_insertLoadKInstruction(CallInst * insertpoint)
{
	Value * _inst = CoalaBlasGemmTask::getRoutineParam("K");
	
	if(_inst==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaBlasGemmTask::_insertLoadMInstruction()\n";
		exit(0);
	}

	// 目标：识别 
	// 源码形式：int M
	// LLVM-IR形式：%9 = alloca i32, align 4

	//-------------------------------------------------------------------------------------
	// 首先向上找到 %9 = alloca i32, align 4
	//-------------------------------------------------------------------------------------

	//设置最大查找层数
	for(size_t i = 0; i<5; i++)
	{
		if(isa<LoadInst>(*_inst))
		{	
			LoadInst * _LI = dyn_cast<LoadInst>(_inst);
			_inst = _LI->getOperand(0);
		}
		else
		{
			break;
		}
	}
	
	//如果不是AllocaInst
	if(!isa<AllocaInst>(*_inst))
	{
		outs()<<"WRONG: Cannot find the alloca inst for this value:\n\t"<<*CoalaBlasGemmTask::getRoutineParam("K")<<"\n";
		exit(0);
	}

	AllocaInst * _alloca = dyn_cast<AllocaInst>(_inst);

	//-------------------------------------------------------------------------------------
	// 插入loadInst指令
	//-------------------------------------------------------------------------------------
	//获取 Module 环境
	LLVMContext & Mcontext = insertpoint->getModule()->getContext();
	
	// 创键一个指令创键环境
	IRBuilder<> builder(Mcontext);
    builder.SetInsertPoint(insertpoint);

	// 7. 创建 %3 = load ptr, ptr @probelist, align 8 指令
    LoadInst * retload = builder.CreateLoad(Type::getInt32Ty(Mcontext),_alloca);

	outs()<<"----Inserted--> "<<*retload<<"\n";
	return retload;
}

void CoalaBlasGemmTask::_insert_coala_blas_sgemm_FuncCallee(GlobalValue * probelist, size_t const taskid)
{
	//获得Module
	Module * M_ptr = CoalaBlasGemmTask::routine_callee->getModule();
	
	//获得Module环境
	LLVMContext & Mcontext = M_ptr->getContext();

	//创建IRBuilder对象
	IRBuilder<> IRB(Mcontext);

	//获取插入点:在callee指令前
	IRB.SetInsertPoint(CoalaBlasGemmTask::routine_callee);


	// 7. 创建 %3 = load ptr, ptr @probelist, align 8 指令
    LoadInst * retload = IRB.CreateLoad(Type::getInt8PtrTy(Mcontext), probelist);

	
	//创建一个存taskid的constant
    Type * element_i64type = Type::getInt64Ty(Mcontext);
	Constant* element = ConstantInt::get(element_i64type, taskid);

	//插入参数设置
	Value * args[] = 
	{
		//插入probelist
		retload,
		//插入task_id
		element,
		CoalaBlasGemmTask::getRoutineParam("Layout"),
		CoalaBlasGemmTask::getRoutineParam("TransA"),
		CoalaBlasGemmTask::getRoutineParam("TransB"),
		CoalaBlasGemmTask::getRoutineParam("M"),
		CoalaBlasGemmTask::getRoutineParam("N"),
		CoalaBlasGemmTask::getRoutineParam("K"),
		CoalaBlasGemmTask::getRoutineParam("Alpha"),
		CoalaBlasGemmTask::getRoutineParam("MatA"),
		CoalaBlasGemmTask::getRoutineParam("LDA"),
		CoalaBlasGemmTask::getRoutineParam("MatB"),
		CoalaBlasGemmTask::getRoutineParam("LDB"),
		CoalaBlasGemmTask::getRoutineParam("Beta"),
		CoalaBlasGemmTask::getRoutineParam("MatC"),
		CoalaBlasGemmTask::getRoutineParam("LDC")
	};

	//创建callee
	CallInst * ret = IRB.CreateCall(CoalaBlasGemmTask::coala_blas_gemm_funcdecl,args);
	
	if(ret==nullptr)
	{
		outs()<<"WRONG in CoalaBlasGemmTask::_insert_coala_blas_sgemm_FuncCallee()\n";
		return;
	}

	// outs()<<"------已成功插入"<<*ret<<"\n";

	//替换原指令返回值影响
	CoalaBlasGemmTask::routine_callee->replaceAllUsesWith(ret);

	// outs()<<"------已成功替换"<<*CoalaBlasGemmTask::routine_callee<<"\n";

	//删除原指令
	CoalaBlasGemmTask::routine_callee->eraseFromParent();
	return;
}

void CoalaBlasGemmTask::_insert_coala_blas_dgemm_FuncCallee(GlobalValue * probelist, size_t const taskid)
{
	//获得Module
	Module * M_ptr = CoalaBlasGemmTask::routine_callee->getModule();
	
	//获得Module环境
	LLVMContext & Mcontext = M_ptr->getContext();

	//创建IRBuilder对象
	IRBuilder<> IRB(Mcontext);

	//获取插入点:在callee指令前
	IRB.SetInsertPoint(CoalaBlasGemmTask::routine_callee);


	// 7. 创建 %3 = load ptr, ptr @probelist, align 8 指令
    LoadInst * retload = IRB.CreateLoad(Type::getInt8PtrTy(Mcontext), probelist);

	
	//创建一个存taskid的constant
    Type * element_i64type = Type::getInt64Ty(Mcontext);
	Constant* element = ConstantInt::get(element_i64type, taskid);

	//插入参数设置
	Value * args[] = 
	{
		//插入probelist
		retload,
		//插入task_id
		element,
		CoalaBlasGemmTask::getRoutineParam("Layout"),
		CoalaBlasGemmTask::getRoutineParam("TransA"),
		CoalaBlasGemmTask::getRoutineParam("TransB"),
		CoalaBlasGemmTask::getRoutineParam("M"),
		CoalaBlasGemmTask::getRoutineParam("N"),
		CoalaBlasGemmTask::getRoutineParam("K"),
		CoalaBlasGemmTask::getRoutineParam("Alpha"),
		CoalaBlasGemmTask::getRoutineParam("MatA"),
		CoalaBlasGemmTask::getRoutineParam("LDA"),
		CoalaBlasGemmTask::getRoutineParam("MatB"),
		CoalaBlasGemmTask::getRoutineParam("LDB"),
		CoalaBlasGemmTask::getRoutineParam("Beta"),
		CoalaBlasGemmTask::getRoutineParam("MatC"),
		CoalaBlasGemmTask::getRoutineParam("LDC")
	};

	//创建callee
	CallInst * ret = IRB.CreateCall(CoalaBlasGemmTask::coala_blas_gemm_funcdecl,args);
	
	if(ret==nullptr)
	{
		outs()<<"WRONG in CoalaBlasGemmTask::_insert_coala_blas_sgemm_FuncCallee()\n";
		return;
	}

	// outs()<<"------已成功插入"<<*ret<<"\n";

	//替换原指令返回值影响
	CoalaBlasGemmTask::routine_callee->replaceAllUsesWith(ret);

	// outs()<<"------已成功替换"<<*CoalaBlasGemmTask::routine_callee<<"\n";

	//删除原指令
	CoalaBlasGemmTask::routine_callee->eraseFromParent();
	return;
}

void CoalaBlasGemmTask::reconstrcuting()
{
	//输入判断
	if(CoalaBlasGemmTask::getRoutineCallee()==nullptr)
	{
		outs()<<"ERROR: coala_blas_gemm_task::reconstrcuting() : getRoutineCallee()==nullptr\n";
		exit(0);
	}

	outs()<<"Starting task reconstructing:\n";
	outs()<<"--Stage 1: Probe Insert\n";
	// CMakeList.txt 中要依赖 cornerstone 库文件
	CoalaProbes probes = CoalaProbes(*CoalaBlasGemmTask::getRoutineCallee()->getCaller());
	
	//插入点
	CallInst * coala_probe_FuncCallee_insert_point =  CoalaBlasGemmTask::devmalc_callee_infos[CoalaBlasGemmTask::_get_first_devmalc_callee_info_idx()]->getMemopCallee();
	
	Value * loadm = CoalaBlasGemmTask::_insertLoadMInstruction(coala_probe_FuncCallee_insert_point);
	Value * loadn = CoalaBlasGemmTask::_insertLoadNInstruction(coala_probe_FuncCallee_insert_point);
	Value * loadk = CoalaBlasGemmTask::_insertLoadKInstruction(coala_probe_FuncCallee_insert_point);
	probes.insert_coala_probe_FuncCallee(
			coala_probe_FuncCallee_insert_point, 
			CoalaBlasGemmTask::taskid,
			CoalaBlasGemmTask::routine_code,
			3, loadm, loadn, loadk);

	outs()<<"--Stage 1: complete\n\n";

	outs()<<"--Stage 2: devmalc reconstructing\n";
	//transform memop to coala
	for (size_t i = 0; i < CoalaBlasGemmTask::devmalc_callee_infos.size(); ++i) 
	{
    	auto & callee = CoalaBlasGemmTask::devmalc_callee_infos[i];
		callee->transform2coala(probes.getCoalaProbelistGV(), CoalaBlasGemmTask::taskid);
	}
	outs()<<"--Stage 2: complete\n\n";

	outs()<<"--Stage 3: host2dev reconstructing\n";
	for (size_t i = 0; i < CoalaBlasGemmTask::host2dev_callee_infos.size(); ++i) 
	{
    	auto & callee = CoalaBlasGemmTask::host2dev_callee_infos[i];
		callee->transform2coala(probes.getCoalaProbelistGV(), CoalaBlasGemmTask::taskid);
	}
	outs()<<"--Stage 3: complete\n\n";

	outs()<<"--Stage 4: dev2host reconstructing\n";
	for (size_t i = 0; i < CoalaBlasGemmTask::dev2host_callee_infos.size(); ++i) 
	{
    	auto & callee = CoalaBlasGemmTask::dev2host_callee_infos[i];
		callee->transform2coala(probes.getCoalaProbelistGV(), CoalaBlasGemmTask::taskid);
	}
	outs()<<"--Stage 4: complete\n\n";

	outs()<<"--Stage 5: devfree reconstructing\n";
	for (size_t i = 0; i < CoalaBlasGemmTask::devfree_callee_infos.size(); ++i) 
	{
    	auto & callee = CoalaBlasGemmTask::devfree_callee_infos[i];
		callee->transform2coala(probes.getCoalaProbelistGV(), CoalaBlasGemmTask::taskid);
	}
	outs()<<"--Stage 5: complete\n\n";

	outs()<<"--Stage 6: blas_gemm reconstructing\n";
	switch (CoalaBlasGemmTask::routine_code)
	{
		case COALA_BLAS_CUBLAS_SGEMM:
			CoalaBlasGemmTask::coala_blas_gemm_funcdecl = 
				CoalaBlasGemmTask::_create_coala_blas_sgemm_FuncDeclaration(*CoalaBlasGemmTask::getRoutineCallee()->getModule());
			CoalaBlasGemmTask::_insert_coala_blas_sgemm_FuncCallee(probes.getCoalaProbelistGV(), CoalaBlasGemmTask::taskid);
			break;
		case COALA_BLAS_CUBLAS_DGEMM:
			CoalaBlasGemmTask::coala_blas_gemm_funcdecl = 
				CoalaBlasGemmTask::_create_coala_blas_dgemm_FuncDeclaration(*CoalaBlasGemmTask::getRoutineCallee()->getModule());
			CoalaBlasGemmTask::_insert_coala_blas_dgemm_FuncCallee(probes.getCoalaProbelistGV(), CoalaBlasGemmTask::taskid);
			break;
		default:
			outs()<<"WRONG: coala_blas_gemm_task::reconstrcuting(): routine_code is not supported\n";
			exit(0);
	}
	outs()<<"--Stage 6: complete\n\n";
	outs()<<"--Task reconstruction done\n\n";
	return;
}


/*====================================================================
| CoalaBlasGemmTaskFactory
======================================================================*/
std::shared_ptr<CoalaBlasGemmTask> CoalaBlasGemmTaskFactory::createACoalaBlasGemmTask(CallInst * CI, COALA_BLAS_ROUTINES_CODE const namecode, size_t const taskid)
{
	switch(namecode)
    {
        case COALA_BLAS_CBLAS_SGEMM:
        case COALA_BLAS_CBLAS_DGEMM:
            return std::make_shared<CoalaBlasGemmTask4Cblas>(CI,namecode,taskid);
        case COALA_BLAS_CUBLAS_SGEMM:
        case COALA_BLAS_CUBLAS_DGEMM:
            return std::make_shared<CoalaBlasGemmTask4Cublas>(CI,namecode,taskid);
        case COALA_BLAS_CLBLAST_SGEMM:
        case COALA_BLAS_CLBLAST_DGEMM:
            return std::make_shared<CoalaBlasGemmTask4Clblast>(CI,namecode,taskid);    
        default:
            return  nullptr;
    }
    return  nullptr;
}