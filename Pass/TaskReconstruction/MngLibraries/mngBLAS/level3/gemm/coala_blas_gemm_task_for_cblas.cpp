/*====================================================================
| INCLUDE
======================================================================*/
#include "coala_blas_gemm_task_for_cblas.h"
#include "coala_cornerstone_utils.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>


/*====================================================================
| NameSpace
======================================================================*/
using namespace llvm;


/*====================================================================
| CBLAS GEMM INFO: CoalaBlasGemmTask4Cblas
======================================================================*/
// CoalaBlasGemmTask4Cblas set Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//order
Value * CoalaBlasGemmTask4Cblas::setLayout() 
{
	if( CoalaBlasGemmTask4Cblas::routine_callee->arg_size()!=14 ) return nullptr;
	Value * _layout = CoalaBlasGemmTask4Cblas::routine_callee->getArgOperand(0);
	return _layout;
}

//transa
Value * CoalaBlasGemmTask4Cblas::setTransA() 
{
	if(CoalaBlasGemmTask4Cblas::routine_callee->arg_size()!=14) return nullptr;

	Value * _transa = CoalaBlasGemmTask4Cblas::routine_callee->getArgOperand(1);
	
    return _transa;
}

//transb
Value * CoalaBlasGemmTask4Cblas::setTransB() 
{
	if( CoalaBlasGemmTask4Cblas::routine_callee->arg_size()!=14 ) return nullptr;
	
    Value * _transb =  CoalaBlasGemmTask4Cblas::routine_callee->getArgOperand(2);
	
    return _transb;
}

//m
Value * CoalaBlasGemmTask4Cblas::setM() 
{
	if( CoalaBlasGemmTask4Cblas::routine_callee->arg_size()!=14 ) return nullptr;

	Value * _m = CoalaBlasGemmTask4Cblas::routine_callee->getArgOperand(3);
	
    return _m;
}

//n
Value * CoalaBlasGemmTask4Cblas::setN() 
{
	if( CoalaBlasGemmTask4Cblas::routine_callee->arg_size()!=14 ) return nullptr;

	Value * _n = CoalaBlasGemmTask4Cblas::routine_callee->getArgOperand(4);

	return _n;
}

Value * CoalaBlasGemmTask4Cblas::setK() 
{
	if( CoalaBlasGemmTask4Cblas::routine_callee->arg_size()!=14 ) return nullptr;

	Value * _k = CoalaBlasGemmTask4Cblas::routine_callee->getArgOperand(5);
	
    return _k;
}//k

Value * CoalaBlasGemmTask4Cblas::setAlpha()
{
	if( CoalaBlasGemmTask4Cblas::routine_callee->arg_size()!=14 ) return nullptr;
    
	Value * _alpha = CoalaBlasGemmTask4Cblas::routine_callee->getArgOperand(6);

	return _alpha;
}//alpha

Value * CoalaBlasGemmTask4Cblas::setMatA()
{
	if( CoalaBlasGemmTask4Cblas::routine_callee->arg_size()!=14 ) return nullptr;
    
	Value * _A = CoalaBlasGemmTask4Cblas::routine_callee->getArgOperand(7);
	return _A;
}

Value * CoalaBlasGemmTask4Cblas::setLDA() 
{
	if( CoalaBlasGemmTask4Cblas::routine_callee->arg_size()!=14 ) return nullptr;
    
	Value * _lda = CoalaBlasGemmTask4Cblas::routine_callee->getArgOperand(8);
	return _lda;
}//lda

Value * CoalaBlasGemmTask4Cblas::setMatB() 
{
	if( CoalaBlasGemmTask4Cblas::routine_callee->arg_size()!=14 ) return nullptr;
    
	Value * _B = CoalaBlasGemmTask4Cblas::routine_callee->getArgOperand(9);
	return _B;
}

Value * CoalaBlasGemmTask4Cblas::setLDB()
{
	if( CoalaBlasGemmTask4Cblas::routine_callee->arg_size()!=14 ) return nullptr;
    
	Value * _ldb = CoalaBlasGemmTask4Cblas::routine_callee->getArgOperand(10);
	return _ldb;
}//ldb

Value * CoalaBlasGemmTask4Cblas::setBeta()
{
	if( CoalaBlasGemmTask4Cblas::routine_callee->arg_size()!=14 ) return nullptr;
    
	Value * _beta = CoalaBlasGemmTask4Cblas::routine_callee->getArgOperand(11);
	return _beta;
}//beta

Value * CoalaBlasGemmTask4Cblas::setMatC() 
{
	if( CoalaBlasGemmTask4Cblas::routine_callee->arg_size()!=14 ) return nullptr;
    
	Value * _C = CoalaBlasGemmTask4Cblas::routine_callee->getArgOperand(12);
	return _C;
}

Value * CoalaBlasGemmTask4Cblas::setLDC() 
{
	if( CoalaBlasGemmTask4Cblas::routine_callee->arg_size()!=14 ) return nullptr;
    
	Value * _ldc = CoalaBlasGemmTask4Cblas::routine_callee->getArgOperand(13);
	return _ldc;
}//ldc



/*====================================================================
| 相关内存操作
======================================================================*/
void CoalaBlasGemmTask4Cblas::_setDevMalcCalleeInfo(Value* param)
{	
	if(CoalaBlasGemmTask4Cblas::routine_callee==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaBlasGemmTask4Cblas::setDevMalcCalleeInfo\n";
		exit(0);
	}



	return;
}

void CoalaBlasGemmTask4Cblas::_setHost2DevCalleeInfo(Value * param)
{
	if(CoalaBlasGemmTask4Cblas::routine_callee==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaBlasGemmTask4Cblas::setHost2DevCalleeInfo\n";
		exit(0);
	}

	return;
}

void CoalaBlasGemmTask4Cblas::_setDev2HostCalleeInfo(Value * param)
{	
	if(CoalaBlasGemmTask4Cblas::routine_callee==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaBlasGemmTask4Cblas::setDev2HostCalleeInfo\n";
		exit(0);
	}

	return;
}

void CoalaBlasGemmTask4Cblas::_setDevFreeCalleeInfo(Value* param)
{	
	if(CoalaBlasGemmTask4Cblas::routine_callee==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaBlasGemmTask4Cblas::setDevFreeCalleeInfo\n";
		exit(0);
	}

	return;
}


Value * CoalaBlasGemmTask4Cblas::getLayout()
{
	return CoalaBlasGemmTask4Cblas::getRoutineParam("Layout");
}
Value * CoalaBlasGemmTask4Cblas::getTransA()
{
	return CoalaBlasGemmTask4Cblas::getRoutineParam("TransA");
}
Value * CoalaBlasGemmTask4Cblas::getTransB()
{
	return CoalaBlasGemmTask4Cblas::getRoutineParam("TransB");
}
Value * CoalaBlasGemmTask4Cblas::getM()
{
	return CoalaBlasGemmTask4Cblas::getRoutineParam("M");
}
Value * CoalaBlasGemmTask4Cblas::getN()
{
	return CoalaBlasGemmTask4Cblas::getRoutineParam("N");
}
Value * CoalaBlasGemmTask4Cblas::getK()
{
	return CoalaBlasGemmTask4Cblas::getRoutineParam("K");
}
Value * CoalaBlasGemmTask4Cblas::getAlpha()
{
	return CoalaBlasGemmTask4Cblas::getRoutineParam("Alpha");
}
Value * CoalaBlasGemmTask4Cblas::getMatA()
{
	return CoalaBlasGemmTask4Cblas::getRoutineParam("MatA");
}
Value * CoalaBlasGemmTask4Cblas::getLDA()
{
	return CoalaBlasGemmTask4Cblas::getRoutineParam("LDA");
}
Value * CoalaBlasGemmTask4Cblas::getMatB()
{
	return CoalaBlasGemmTask4Cblas::getRoutineParam("MatB");
}
Value * CoalaBlasGemmTask4Cblas::getLDB()
{
	return CoalaBlasGemmTask4Cblas::getRoutineParam("LDB");
}
Value * CoalaBlasGemmTask4Cblas::getBeta()
{
	return CoalaBlasGemmTask4Cblas::getRoutineParam("Beta");
}
Value * CoalaBlasGemmTask4Cblas::getMatC()
{
	return CoalaBlasGemmTask4Cblas::getRoutineParam("MatC");
}
Value * CoalaBlasGemmTask4Cblas::getLDC()
{
	return CoalaBlasGemmTask4Cblas::getRoutineParam("LDC");
}
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// dump for debug
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void CoalaBlasGemmTask4Cblas::dump() 
{
	outs()<<"Task Name: "<<CoalaBlasGemmTask4Cblas::getRoutineName()<<"\n"
		<<"Layout:\t"<<*CoalaBlasGemmTask4Cblas::getRoutineParam("Layout")<<"\n"
		<<"TransA:\t"<<*CoalaBlasGemmTask4Cblas::getRoutineParam("TransA")<<"\n"
		<<"TransB:\t"<<*CoalaBlasGemmTask4Cblas::getRoutineParam("TransB")<<"\n"
		<<"M:\t"<<*CoalaBlasGemmTask4Cblas::getRoutineParam("M")<<"\n"
		<<"N:\t"<<*CoalaBlasGemmTask4Cblas::getRoutineParam("N")<<"\n"
		<<"K:\t"<<*CoalaBlasGemmTask4Cblas::getRoutineParam("K")<<"\n"
		<<"Alpha:\t"<<*CoalaBlasGemmTask4Cblas::getRoutineParam("Alpha")<<"\n"
		<<"MatA:\t"<<*CoalaBlasGemmTask4Cblas::getRoutineParam("MatA")<<"\n"
		<<"LDA:\t"<<*CoalaBlasGemmTask4Cblas::getRoutineParam("LDA")<<"\n"
		<<"MatB:\t"<<*CoalaBlasGemmTask4Cblas::getRoutineParam("MatB")<<"\n"
		<<"LDB:\t"<<*CoalaBlasGemmTask4Cblas::getRoutineParam("LDB")<<"\n"
		<<"Beta:\t"<<*CoalaBlasGemmTask4Cblas::getRoutineParam("Beta")<<"\n"
		<<"MatC:\t"<<*CoalaBlasGemmTask4Cblas::getRoutineParam("MatC")<<"\n"
		<<"LDC:\t"<<*CoalaBlasGemmTask4Cblas::getRoutineParam("LDC")<<"\n";
}