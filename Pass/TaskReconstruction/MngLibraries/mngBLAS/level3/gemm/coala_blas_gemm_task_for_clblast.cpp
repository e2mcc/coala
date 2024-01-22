/*====================================================================
| INCLUDE
======================================================================*/
#include "coala_blas_gemm_task_for_clblast.h"
#include "coala_cornerstone_utils.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>


/*====================================================================
| NameSpace
======================================================================*/
using namespace llvm;


/*====================================================================
| CBLAS GEMM INFO: CoalaBlasGemmTask4Clblast
======================================================================*/
// CoalaBlasGemmTask4Clblast set Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//order
Value * CoalaBlasGemmTask4Clblast::setLayout() 
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _layout = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(0);
	return _layout;
}

//transa
Value * CoalaBlasGemmTask4Clblast::setTransA() 
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _transA = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(1);
	return _transA;
}

//transb
Value * CoalaBlasGemmTask4Clblast::setTransB() 
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _transB = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(2);
	return _transB;
}

//m
Value * CoalaBlasGemmTask4Clblast::setM() 
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _M = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(3);
	return _M;
}

//n
Value * CoalaBlasGemmTask4Clblast::setN() 
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _N = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(4);
	return _N;
}

Value * CoalaBlasGemmTask4Clblast::setK() 
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _K = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(5);
	return _K;
}//k

Value * CoalaBlasGemmTask4Clblast::setAlpha() 
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _alpha = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(6);
	return _alpha;
}//alpha

Value * CoalaBlasGemmTask4Clblast::setMatA() 
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _A = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(7);
	return _A;
}//A

Value * CoalaBlasGemmTask4Clblast::setOffA()
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _offa = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(8);
	return _offa;
}//offA

Value * CoalaBlasGemmTask4Clblast::setLDA() 
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _lda = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(9);
	return _lda;
}//lda

Value * CoalaBlasGemmTask4Clblast::setMatB() 
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _B = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(10);
	return _B;
}//B

Value * CoalaBlasGemmTask4Clblast::setOffB()
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _offb = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(11);
	return _offb;
}//offB

Value * CoalaBlasGemmTask4Clblast::setLDB() 
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _ldb = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(12);
	return _ldb;
}//ldb

Value * CoalaBlasGemmTask4Clblast::setBeta() 
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _beta = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(13);
	return _beta;
}//beta

Value * CoalaBlasGemmTask4Clblast::setMatC() 
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _C = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(14);
	return _C;
}//C

Value * CoalaBlasGemmTask4Clblast::setOffC()
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _offc = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(15);
	return _offc;
}//offC

Value * CoalaBlasGemmTask4Clblast::setLDC() 
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _ldc = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(16);
	return _ldc;
}//ldc


Value * CoalaBlasGemmTask4Clblast::setCommandQueues()
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _commandqueues = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(17);
	return _commandqueues;
}


Value * CoalaBlasGemmTask4Clblast::setEvents()
{
	if( CoalaBlasGemmTask4Clblast::routine_callee->arg_size()!=19 ) return nullptr;
	Value * _events = CoalaBlasGemmTask4Clblast::routine_callee->getArgOperand(18);
	return _events;
}


/*====================================================================
| 相关内存操作
======================================================================*/
void CoalaBlasGemmTask4Clblast::_setDevMalcCalleeInfo(Value* param)
{	
	if(CoalaBlasGemmTask4Clblast::routine_callee==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaBlasGemmTask4Clblast::setDevMalcCalleeInfo\n";
		exit(0);
	}



	return;
}

void CoalaBlasGemmTask4Clblast::_setHost2DevCalleeInfo(Value * param)
{
	if(CoalaBlasGemmTask4Clblast::routine_callee==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaBlasGemmTask4Clblast::setHost2DevCalleeInfo\n";
		exit(0);
	}

	return;
}

void CoalaBlasGemmTask4Clblast::_setDev2HostCalleeInfo(Value * param)
{	
	if(CoalaBlasGemmTask4Clblast::routine_callee==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaBlasGemmTask4Clblast::setDev2HostCalleeInfo\n";
		exit(0);
	}

	return;
}

void CoalaBlasGemmTask4Clblast::_setDevFreeCalleeInfo(Value* param)
{	
	if(CoalaBlasGemmTask4Clblast::routine_callee==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaBlasGemmTask4Clblast::setDevFreeCalleeInfo\n";
		exit(0);
	}

	return;
}

Value * CoalaBlasGemmTask4Clblast::getLayout()
{
	return CoalaBlasGemmTask4Clblast::getRoutineParam("Layout");
}
Value * CoalaBlasGemmTask4Clblast::getTransA()
{
	return CoalaBlasGemmTask4Clblast::getRoutineParam("TransA");
}
Value * CoalaBlasGemmTask4Clblast::getTransB()
{
	return CoalaBlasGemmTask4Clblast::getRoutineParam("TransB");
}
Value * CoalaBlasGemmTask4Clblast::getM()
{
	return CoalaBlasGemmTask4Clblast::getRoutineParam("M");
}
Value * CoalaBlasGemmTask4Clblast::getN()
{
	return CoalaBlasGemmTask4Clblast::getRoutineParam("N");
}
Value * CoalaBlasGemmTask4Clblast::getK()
{
	return CoalaBlasGemmTask4Clblast::getRoutineParam("K");
}
Value * CoalaBlasGemmTask4Clblast::getAlpha()
{
	return CoalaBlasGemmTask4Clblast::getRoutineParam("Alpha");
}
Value * CoalaBlasGemmTask4Clblast::getMatA()
{
	return CoalaBlasGemmTask4Clblast::getRoutineParam("MatA");
}
Value * CoalaBlasGemmTask4Clblast::getLDA()
{
	return CoalaBlasGemmTask4Clblast::getRoutineParam("LDA");
}
Value * CoalaBlasGemmTask4Clblast::getMatB()
{
	return CoalaBlasGemmTask4Clblast::getRoutineParam("MatB");
}
Value * CoalaBlasGemmTask4Clblast::getLDB()
{
	return CoalaBlasGemmTask4Clblast::getRoutineParam("LDB");
}
Value * CoalaBlasGemmTask4Clblast::getBeta()
{
	return CoalaBlasGemmTask4Clblast::getRoutineParam("Beta");
}
Value * CoalaBlasGemmTask4Clblast::getMatC()
{
	return CoalaBlasGemmTask4Clblast::getRoutineParam("MatC");
}
Value * CoalaBlasGemmTask4Clblast::getLDC()
{
	return CoalaBlasGemmTask4Clblast::getRoutineParam("LDC");
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// CLBlast GEMM INFO dump Function for debug
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CoalaBlasGemmTask4Clblast::dump() 
{
	outs()<<"Task Name: "<<CoalaBlasGemmTask4Clblast::getRoutineName()<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("Layout")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("TransA")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("TransB")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("M")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("N")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("K")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("Alpha")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("MatA")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("OffA")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("LDA")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("MatB")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("OffB")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("LDB")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("Beta")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("MatC")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("OffC")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("LDC")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("CommandQueues")<<"\n"
		<<*CoalaBlasGemmTask4Clblast::getRoutineParam("Events")<<"\n";
}