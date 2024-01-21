/*====================================================================
| INCLUDE
======================================================================*/
#include "coala_blas_gemm_task_for_cublas.h"
#include "coala_cornerstone_utils.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include "coala_memop_list.h"

/*====================================================================
| NameSpace
======================================================================*/
using namespace llvm;


/*====================================================================
| CBLAS GEMM INFO: CoalaBlasGemmTask4Cublas
======================================================================*/
// CoalaBlasGemmTask4Cublas set Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//order:cublas gemm 仅支持 column major
Value * CoalaBlasGemmTask4Cublas::setLayout() 
{	
	Module * M_ptr = CoalaBlasGemmTask4Cublas::routine_callee->getModule();
	LLVMContext & Mcontext = M_ptr->getContext();
	//创建一个存Layout的constant
    Type * element_i32type = Type::getInt32Ty(Mcontext);
	Constant * _layout = ConstantInt::get(element_i32type, COALA_MATRIX_COL_MAJOR);
	return _layout;
}

//order
Value * CoalaBlasGemmTask4Cublas::setHandle()
{
	if( CoalaBlasGemmTask4Cublas::routine_callee->arg_size()!=14 ) return nullptr;
	Value * _handle = CoalaBlasGemmTask4Cublas::routine_callee->getArgOperand(0);
	return _handle;
}

//transa
Value * CoalaBlasGemmTask4Cublas::setTransA() 
{
	if( CoalaBlasGemmTask4Cublas::routine_callee->arg_size()!=14 ) return nullptr;
	Value * _transA = CoalaBlasGemmTask4Cublas::routine_callee->getArgOperand(1);
	return _transA;
}

//transb
Value * CoalaBlasGemmTask4Cublas::setTransB() 
{
	if( CoalaBlasGemmTask4Cublas::routine_callee->arg_size()!=14 ) return nullptr;
	Value * _transB = CoalaBlasGemmTask4Cublas::routine_callee->getArgOperand(2);
	return _transB;
}

//m
Value * CoalaBlasGemmTask4Cublas::setM() 
{
	if( CoalaBlasGemmTask4Cublas::routine_callee->arg_size()!=14 ) return nullptr;
	Value * _M = CoalaBlasGemmTask4Cublas::routine_callee->getArgOperand(3);
	return _M;
}

//n
Value * CoalaBlasGemmTask4Cublas::setN() 
{
	if( CoalaBlasGemmTask4Cublas::routine_callee->arg_size()!=14 ) return nullptr;
	Value * _N = CoalaBlasGemmTask4Cublas::routine_callee->getArgOperand(4);
	return _N;
}

Value * CoalaBlasGemmTask4Cublas::setK() 
{
	if( CoalaBlasGemmTask4Cublas::routine_callee->arg_size()!=14 ) return nullptr;
	Value * _K = CoalaBlasGemmTask4Cublas::routine_callee->getArgOperand(5);
	return _K;
}//k

Value * CoalaBlasGemmTask4Cublas::setAlpha() 
{
	if( CoalaBlasGemmTask4Cublas::routine_callee->arg_size()!=14 ) return nullptr;
	Value * _alpha = CoalaBlasGemmTask4Cublas::routine_callee->getArgOperand(6);
	return _alpha;
}//alpha

Value * CoalaBlasGemmTask4Cublas::setMatA() 
{
	if( CoalaBlasGemmTask4Cublas::routine_callee->arg_size()!=14 ) return nullptr;
	Value * _A = CoalaBlasGemmTask4Cublas::routine_callee->getArgOperand(7);
	return _A;
}//A

Value * CoalaBlasGemmTask4Cublas::setLDA() 
{
	if( CoalaBlasGemmTask4Cublas::routine_callee->arg_size()!=14 ) return nullptr;
	Value * _lda = CoalaBlasGemmTask4Cublas::routine_callee->getArgOperand(8);
	return _lda;
}//lda

Value * CoalaBlasGemmTask4Cublas::setMatB() 
{
	if( CoalaBlasGemmTask4Cublas::routine_callee->arg_size()!=14 ) return nullptr;
	Value * _B = CoalaBlasGemmTask4Cublas::routine_callee->getArgOperand(9);
	return _B;
}//B

Value * CoalaBlasGemmTask4Cublas::setLDB() 
{
	if( CoalaBlasGemmTask4Cublas::routine_callee->arg_size()!=14 ) return nullptr;
	Value * _ldb = CoalaBlasGemmTask4Cublas::routine_callee->getArgOperand(10);
	return _ldb;
}//ldb

Value * CoalaBlasGemmTask4Cublas::setBeta() 
{
	if( CoalaBlasGemmTask4Cublas::routine_callee->arg_size()!=14 ) return nullptr;
	Value * _beta = CoalaBlasGemmTask4Cublas::routine_callee->getArgOperand(11);
	return _beta;
}//beta

Value * CoalaBlasGemmTask4Cublas::setMatC() 
{
	if( CoalaBlasGemmTask4Cublas::routine_callee->arg_size()!=14 ) return nullptr;
	Value * _C = CoalaBlasGemmTask4Cublas::routine_callee->getArgOperand(12);
	return _C;
}//C

Value * CoalaBlasGemmTask4Cublas::setLDC() 
{
	if( CoalaBlasGemmTask4Cublas::routine_callee->arg_size()!=14 ) return nullptr;
	Value * _ldc = CoalaBlasGemmTask4Cublas::routine_callee->getArgOperand(13);
	return _ldc;
}//ldc



/*====================================================================
| 相关内存操作
======================================================================*/
void CoalaBlasGemmTask4Cublas::_setDevMalcCalleeInfo(Value * param)
{	
	if(CoalaBlasGemmTask4Cublas::routine_callee==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaBlasGemmTask4Cublas::setDevMalcCalleeInfo\n";
		exit(0);
	}

	// 目标：识别 
	// 源码形式：cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
	// LLVM-IR形式：%192 = call i32 @cudaMalloc(ptr noundef %13, i32 noundef %191)
	
	//-------------------------------------------------------------------------------------
	// 首先向上找到 %12 = alloca ptr, align 8
	//-------------------------------------------------------------------------------------
	
	Value * _inst = param;

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
		outs()<<"WRONG: Cannot find the alloca inst for this value:\n\t"<<*param<<"\n";
		exit(0);
	}

	AllocaInst * _alloca = dyn_cast<AllocaInst>(_inst);


	//向下找use
	for(auto U = _alloca->use_begin(); U!=_alloca->use_end(); ++U)
	{	
		Use & UU = *U;
		User * user = UU.getUser();
		if(isa<CallInst>(user))
		{	
			CallInst * _CI = dyn_cast<CallInst>(user);
			Function * callee = _CI->getCalledFunction();
			auto _name = getDemangledName(*callee);
			if( _name == COALA_MEMOP_NAMELIST[COALA_MEMOP_CUDA_MALC])
			{
				outs()<<"终于找到你了："<<* _CI<<"\n";
				CoalaBlasGemmTask4Cublas::devmalc_callee_infos.push_back(
					std::make_shared<CoalaMemopDevMalcCallee4Cuda>(_CI,COALA_MEMOP_CUDA_MALC));
				return;
			}
			if( _name == COALA_MEMOP_NAMELIST[COALA_MEMOP_DEVMALC])
			{
				outs()<<"终于找到你了："<<* _CI<<"\n";
				CoalaBlasGemmTask4Cublas::devmalc_callee_infos.push_back(
					std::make_shared<CoalaMemopDevMalcCallee4Cuda>(_CI,COALA_MEMOP_DEVMALC));
				return;
			}
		}
	}
	outs()<<"Wrong:setDevMalcCalleeInfo 没找到 CallInst\n";
	exit(0);
	return;
}



void CoalaBlasGemmTask4Cublas::_setHost2DevCalleeInfo(Value * param)
{
	if(CoalaBlasGemmTask4Cublas::routine_callee==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaBlasGemmTask4Cublas::setHost2DevCalleeInfo\n";
		exit(0);
	}

	// 目标：识别 
	// 源码形式：stat = cublasSetMatrix (M, K, sizeof(*A), A, M, devPtrA, M);
	// LLVM-IR形式： %228 = call i32 @cublasSetMatrix(i32 noundef %222, i32 noundef %223, i32 noundef 4,
	//									ptr noundef %224, i32 noundef %225, ptr noundef %226, i32 noundef %227)							
	
	//-------------------------------------------------------------------------------------
	// 首先向上找到 %12 = alloca ptr, align 8
	//-------------------------------------------------------------------------------------
	Value * _inst = param;

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
		outs()<<"WRONG: Cannot find the alloca inst for this value:\n\t"<<*param<<"\n";
		exit(0);
	}

	AllocaInst * _alloca = dyn_cast<AllocaInst>(_inst);


	//向下找use
	for(auto U1 = _alloca->use_begin(); U1!=_alloca->use_end(); ++U1)
	{	
		Use & UU1 = *U1;
		User * user1 = UU1.getUser();
		if(isa<LoadInst>(user1))
		{
			LoadInst * _LI = dyn_cast<LoadInst>(user1);
			for(auto U2 = _LI->use_begin(); U2!=_LI->use_end(); ++U2)
			{
				Use & UU2 = *U2;
				User * user2 = UU2.getUser();
				if(isa<CallInst>(user2))
				{	
					CallInst * _CI = dyn_cast<CallInst>(user2);
					Function * callee = _CI->getCalledFunction();
					auto _name = getDemangledName(*callee);
					if( _name == COALA_MEMOP_NAMELIST[COALA_MEMOP_CUBLAS_H2D])
					{
						outs()<<"终于找到你了："<<* _CI<<"\n";
						CoalaBlasGemmTask4Cublas::host2dev_callee_infos.push_back(
							std::make_shared<CoalaMemopDataMigrationH2DCallee4Cublas>(_CI,COALA_MEMOP_CUBLAS_H2D));
						return;
					}
					if( _name == COALA_MEMOP_NAMELIST[COALA_MEMOP_H2D])
					{
						outs()<<"终于找到你了："<<* _CI<<"\n";
						CoalaBlasGemmTask4Cublas::host2dev_callee_infos.push_back(
							std::make_shared<CoalaMemopDataMigrationH2DCallee4Cublas>(_CI,COALA_MEMOP_H2D));
						return;
					}
				}
			}
			
		}
		
	}
	outs()<<"Wrong:setHost2DevCalleeInfo 没找到 CallInst\n";
	return;
}



void CoalaBlasGemmTask4Cublas::_setDev2HostCalleeInfo(Value * param)
{	
	if(CoalaBlasGemmTask4Cublas::routine_callee==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaBlasGemmTask4Cublas::setDev2HostCalleeInfo\n";
		exit(0);
	}

	// 目标：识别 
	// 源码形式：stat = cublasGetMatrix (M, K, sizeof(*A), A, M, devPtrA, M);
	// LLVM-IR形式： %228 = call i32 @cublasGetMatrix(i32 noundef %222, i32 noundef %223, i32 noundef 4,
	//									ptr noundef %224, i32 noundef %225, ptr noundef %226, i32 noundef %227)							
	
	//-------------------------------------------------------------------------------------
	// 首先向上找到 %12 = alloca ptr, align 8
	//-------------------------------------------------------------------------------------
	Value * _inst = param;

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
		outs()<<"WRONG: Cannot find the alloca inst for this value:\n\t"<<*param<<"\n";
		exit(0);
	}

	AllocaInst * _alloca = dyn_cast<AllocaInst>(_inst);
	outs()<<"CoalaBlasGemmTask4Cublas::_setDev2HostCalleeInfo 找到 AllocaInst:\n\t"<<*_alloca<<"\n";


	//向下找use
	for(auto U1 = _alloca->use_begin(); U1!=_alloca->use_end(); ++U1)
	{	
		Use & UU1 = *U1;
		User * user1 = UU1.getUser();
		if(isa<LoadInst>(user1))
		{
			LoadInst * _LI = dyn_cast<LoadInst>(user1);
			for(auto U2 = _LI->use_begin(); U2!=_LI->use_end(); ++U2)
			{
				Use & UU2 = *U2;
				User * user2 = UU2.getUser();
				if(isa<CallInst>(user2))
				{	
					CallInst * _CI = dyn_cast<CallInst>(user2);
					Function * callee = _CI->getCalledFunction();
					auto _name = getDemangledName(*callee);
					if( _name == COALA_MEMOP_NAMELIST[COALA_MEMOP_CUBLAS_D2H])
					{
						outs()<<"COALA_MEMOP_CUBLAS_D2H终于找到你了："<<* _CI<<"\n";
						CoalaBlasGemmTask4Cublas::dev2host_callee_infos.push_back(
							std::make_shared<CoalaMemopDataMigrationD2HCallee4Cublas>(_CI,COALA_MEMOP_CUBLAS_D2H));
						return;
					}
					if( _name == COALA_MEMOP_NAMELIST[COALA_MEMOP_D2H])
					{
						outs()<<"COALA_MEMOP_D2H终于找到你了："<<* _CI<<"\n";
						CoalaBlasGemmTask4Cublas::dev2host_callee_infos.push_back(
							std::make_shared<CoalaMemopDataMigrationD2HCallee4Cublas>(_CI,COALA_MEMOP_D2H));
						return;
					}
				}
			}
			
		}
		
	}
	outs()<<"Wrong:setDevt2HostCalleeInfo 没找到 CallInst\n";
	return;
}



void CoalaBlasGemmTask4Cublas::_setDevFreeCalleeInfo(Value* param)
{	
	if(CoalaBlasGemmTask4Cublas::routine_callee==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaBlasGemmTask4Cublas::setDevFreeCalleeInfo\n";
		exit(0);
	}

	// 目标：识别 
	// 源码形式：cudaFree (devPtrA);
	// LLVM-IR形式：
	
	//-------------------------------------------------------------------------------------
	// 首先向上找到 %12 = alloca ptr, align 8
	//-------------------------------------------------------------------------------------
	Value * _inst = param;

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
		outs()<<"WRONG: Cannot find the alloca inst for this value:\n\t"<<*param<<"\n";
		exit(0);
	}

	AllocaInst * _alloca = dyn_cast<AllocaInst>(_inst);


	//向下找use
	for(auto U1 = _alloca->use_begin(); U1!=_alloca->use_end(); ++U1)
	{	
		Use & UU1 = *U1;
		User * user1 = UU1.getUser();
		if(isa<LoadInst>(user1))
		{
			LoadInst * _LI = dyn_cast<LoadInst>(user1);
			for(auto U2 = _LI->use_begin(); U2!=_LI->use_end(); ++U2)
			{
				Use & UU2 = *U2;
				User * user2 = UU2.getUser();
				if(isa<CallInst>(user2))
				{	
					CallInst * _CI = dyn_cast<CallInst>(user2);
					Function * callee = _CI->getCalledFunction();
					auto _name = getDemangledName(*callee);
					if( _name == COALA_MEMOP_NAMELIST[COALA_MEMOP_CUDA_FREE])
					{
						outs()<<"终于找到你了："<<* _CI<<"\n";
						CoalaBlasGemmTask4Cublas::devfree_callee_infos.push_back(
							std::make_shared<CoalaMemopDevFreeCallee4Cuda>(_CI,COALA_MEMOP_CUDA_FREE));
						break;
					}
					if( _name == COALA_MEMOP_NAMELIST[COALA_MEMOP_DEVFREE])
					{
						outs()<<"终于找到你了："<<* _CI<<"\n";
						CoalaBlasGemmTask4Cublas::devfree_callee_infos.push_back(
							std::make_shared<CoalaMemopDevFreeCallee4Cuda>(_CI,COALA_MEMOP_DEVFREE));
						break;
					}
				}
			}
			
		}
		
	}
	
	return;
}


Value * CoalaBlasGemmTask4Cublas::getLayout()
{
	return CoalaBlasGemmTask4Cublas::getRoutineParam("Layout");
}
Value * CoalaBlasGemmTask4Cublas::getTransA()
{
	return CoalaBlasGemmTask4Cublas::getRoutineParam("TransA");
}
Value * CoalaBlasGemmTask4Cublas::getTransB()
{
	return CoalaBlasGemmTask4Cublas::getRoutineParam("TransB");
}
Value * CoalaBlasGemmTask4Cublas::getM()
{
	return CoalaBlasGemmTask4Cublas::getRoutineParam("M");
}
Value * CoalaBlasGemmTask4Cublas::getN()
{
	return CoalaBlasGemmTask4Cublas::getRoutineParam("N");
}
Value * CoalaBlasGemmTask4Cublas::getK()
{
	return CoalaBlasGemmTask4Cublas::getRoutineParam("K");
}
Value * CoalaBlasGemmTask4Cublas::getAlpha()
{
	return CoalaBlasGemmTask4Cublas::getRoutineParam("Alpha");
}
Value * CoalaBlasGemmTask4Cublas::getMatA()
{
	return CoalaBlasGemmTask4Cublas::getRoutineParam("MatA");
}
Value * CoalaBlasGemmTask4Cublas::getLDA()
{
	return CoalaBlasGemmTask4Cublas::getRoutineParam("LDA");
}
Value * CoalaBlasGemmTask4Cublas::getMatB()
{
	return CoalaBlasGemmTask4Cublas::getRoutineParam("MatB");
}
Value * CoalaBlasGemmTask4Cublas::getLDB()
{
	return CoalaBlasGemmTask4Cublas::getRoutineParam("LDB");
}
Value * CoalaBlasGemmTask4Cublas::getBeta()
{
	return CoalaBlasGemmTask4Cublas::getRoutineParam("Beta");
}
Value * CoalaBlasGemmTask4Cublas::getMatC()
{
	return CoalaBlasGemmTask4Cublas::getRoutineParam("MatC");
}
Value * CoalaBlasGemmTask4Cublas::getLDC()
{
	return CoalaBlasGemmTask4Cublas::getRoutineParam("LDC");
}


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// CuBLAS GEMM INFO dump Function for debug
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CoalaBlasGemmTask4Cublas::dump() 
{
	outs()<<"名字: "<<CoalaBlasGemmTask4Cublas::getRoutineName()<<"\n"
		<<*CoalaBlasGemmTask4Cublas::getRoutineParam("Handle")<<"\n"
		<<*CoalaBlasGemmTask4Cublas::getRoutineParam("TransA")<<"\n"
		<<*CoalaBlasGemmTask4Cublas::getRoutineParam("TransB")<<"\n"
		<<*CoalaBlasGemmTask4Cublas::getRoutineParam("M")<<"\n"
		<<*CoalaBlasGemmTask4Cublas::getRoutineParam("N")<<"\n"
		<<*CoalaBlasGemmTask4Cublas::getRoutineParam("K")<<"\n"
		<<*CoalaBlasGemmTask4Cublas::getRoutineParam("Alpha")<<"\n"
		<<*CoalaBlasGemmTask4Cublas::getRoutineParam("MatA")<<"\n"
		<<*CoalaBlasGemmTask4Cublas::getRoutineParam("LDA")<<"\n"
		<<*CoalaBlasGemmTask4Cublas::getRoutineParam("MatB")<<"\n"
		<<*CoalaBlasGemmTask4Cublas::getRoutineParam("LDB")<<"\n"
		<<*CoalaBlasGemmTask4Cublas::getRoutineParam("Beta")<<"\n"
		<<*CoalaBlasGemmTask4Cublas::getRoutineParam("MatC")<<"\n"
		<<*CoalaBlasGemmTask4Cublas::getRoutineParam("LDC")<<"\n";
}