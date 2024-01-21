/*====================================================================
| INCLUDE
======================================================================*/
#include "coala_memop_datamigr_h2d.h"
#include "coala_cornerstone_utils.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>

/*====================================================================
| NameSpace
======================================================================*/
using namespace llvm;



/*====================================================================
| Derived Subbase Class : CoalaMemopDataMigrationH2DCallee
======================================================================*/
// private function : _create_coala_data_transmission_h2d_declaration
// int coala_host2dev(coala_probes_t *, int, void*, void*, size_t, ...)
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Function * CoalaMemopDataMigrationH2DCallee::_create_coala_memop_host2dev_FuncDeclaration(Module & M)
{
    //判定 Module 中是否已经存在 coala_probe 函数声明
    Function * _func = findNamedFunctionDeclaration(M, "coala_memop_host2dev");
    if(_func != nullptr)
    {
		outs()<<"----Already Exist--> "<<*_func<<"\n";
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
        Type::getInt64Ty(Mcontext),  // i64 noundef
        Type::getInt8PtrTy(Mcontext), // ptr noundef
		Type::getInt8PtrTy(Mcontext), // ptr noundef
        Type::getInt32Ty(Mcontext)  // i32 noundef
	};

    // 3. 声明返回值类型+函数参数类型=一个函数类型：
    FunctionType * func_type = FunctionType::get(ret_type, param_types, false);

    // 4. 声明函数(如果M中有函数声明就获取，没有就插入这个函数声明)
	_func = Function::Create(func_type, Function::ExternalLinkage, "coala_memop_host2dev", &M);

    // 5. 设置参数属性
    _func->addParamAttr(0, Attribute::NoUndef);
    _func->addParamAttr(1, Attribute::NoUndef);
    _func->addParamAttr(2, Attribute::NoUndef);
    _func->addParamAttr(3, Attribute::NoUndef);
	_func->addParamAttr(4, Attribute::NoUndef);

	outs()<<"----Creating-----> "<<*_func<<"\n";
    return _func;
}



/*====================================================================
| Derived Class : CoalaMemopDataMigrationH2DCallee4Cuda
======================================================================*/
// Set Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDataMigrationH2DCallee4Cuda::setDevPtr()
{
	if( CoalaMemopDataMigrationH2DCallee4Cuda::memop_callee->arg_size()!=4 ) return nullptr;
	Value * _device_ptr = CoalaMemopDataMigrationH2DCallee4Cuda::memop_callee->getArgOperand(0);
	return _device_ptr;
}

Value * CoalaMemopDataMigrationH2DCallee4Cuda::setHostPtr()
{
	if( CoalaMemopDataMigrationH2DCallee4Cuda::memop_callee->arg_size()!=4 ) return nullptr;
	Value * _host_ptr = CoalaMemopDataMigrationH2DCallee4Cuda::memop_callee->getArgOperand(1);
	return _host_ptr;
}


Value * CoalaMemopDataMigrationH2DCallee4Cuda::setSize()
{
	if( CoalaMemopDataMigrationH2DCallee4Cuda::memop_callee->arg_size()!=4 ) return nullptr;
	Value * _size = CoalaMemopDataMigrationH2DCallee4Cuda::memop_callee->getArgOperand(2);
	return _size;
}

Value * CoalaMemopDataMigrationH2DCallee4Cuda::setDTFlag()
{
	if( CoalaMemopDataMigrationH2DCallee4Cuda::memop_callee->arg_size()!=4 ) return nullptr;
	Value * _data_transmission_flag = CoalaMemopDataMigrationH2DCallee4Cuda::memop_callee->getArgOperand(3);
	return _data_transmission_flag;
}



//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Get Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDataMigrationH2DCallee4Cuda::getDevPtr()
{
	return CoalaMemopDataMigrationH2DCallee4Cuda::getMemopParam("DevPtr");
}

Value * CoalaMemopDataMigrationH2DCallee4Cuda::getHostPtr()
{
	return CoalaMemopDataMigrationH2DCallee4Cuda::getMemopParam("HostPtr");
}

Value * CoalaMemopDataMigrationH2DCallee4Cuda::getSize()
{
	return CoalaMemopDataMigrationH2DCallee4Cuda::getMemopParam("Size");
}

Value * CoalaMemopDataMigrationH2DCallee4Cuda::getDTFlag()
{
	return CoalaMemopDataMigrationH2DCallee4Cuda::getMemopParam("DTFlag");
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// transform2coala Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/**
 * @brief 转化为coala_host2dev(probelist,taskID,dest_ptr,source_pte,size,...)
 * @param task_id 任务ID
 */
void CoalaMemopDataMigrationH2DCallee4Cuda::transform2coala(GlobalValue * probelist, size_t const taskid)
{	
	return;
}



/*====================================================================
| Derived Class : CoalaMemopDataMigrationH2DCallee4Cublas
======================================================================*/
// Set Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDataMigrationH2DCallee4Cublas::setRowDimension()
{
	if( CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->arg_size()!=7 ) return nullptr;
	Value * _row_dimension = CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->getArgOperand(0);
	return _row_dimension;
}

Value * CoalaMemopDataMigrationH2DCallee4Cublas::setColDimension()
{
	if( CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->arg_size()!=7 ) return nullptr;
	Value * _col_dimension = CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->getArgOperand(1);
	return _col_dimension;
}

Value * CoalaMemopDataMigrationH2DCallee4Cublas::setTypeSize()
{
	if( CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->arg_size()!=7 ) return nullptr;
	Value * _size = CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->getArgOperand(2);
	return _size;
}

Value * CoalaMemopDataMigrationH2DCallee4Cublas::setHostMatPtr()
{
	if( CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->arg_size()!=7 ) return nullptr;
	Value * _host_mat_ptr = CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->getArgOperand(3);
	return _host_mat_ptr;
}

Value * CoalaMemopDataMigrationH2DCallee4Cublas::setHostMatLd()
{
	if( CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->arg_size()!=7 ) return nullptr;
	Value * _host_mat_ld = CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->getArgOperand(4);
	return _host_mat_ld;
}

Value * CoalaMemopDataMigrationH2DCallee4Cublas::setDevMatPtr()
{
	if( CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->arg_size()!=7 ) return nullptr;
	Value * _dev_mat_ptr = CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->getArgOperand(5);
	return _dev_mat_ptr;
}

Value * CoalaMemopDataMigrationH2DCallee4Cublas::setDevMatLd()
{
	if( CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->arg_size()!=7 ) return nullptr;
	Value * _dev_mat_ld = CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->getArgOperand(6);
	return _dev_mat_ld;
}


Value * CoalaMemopDataMigrationH2DCallee4Cublas::_getLoadRowInstruction(CallInst * insertpoint)
{
	Value * _inst = CoalaMemopDataMigrationH2DCallee4Cublas::getMemopParam("RowDimension");
	
	if(_inst==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaMemopDataMigrationH2DCallee4Cublas::_getLoadRowInstruction()\n";
		exit(0);
	}

	if(!isa<LoadInst>(*_inst))
	{	
		outs()<<"WRONG in CoalaMemopDataMigrationH2DCallee4Cublas::_getLoadRowInstruction():can not get LoadRowInstruction\n";
		exit(0);
	}

	return _inst;
}


Value * CoalaMemopDataMigrationH2DCallee4Cublas::_getLoadColInstruction(CallInst * insertpoint)
{
	Value * _inst = CoalaMemopDataMigrationH2DCallee4Cublas::getMemopParam("ColDimension");
	
	if(_inst==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaMemopDataMigrationH2DCallee4Cublas::_getLoadColInstruction\n";
		exit(0);
	}

	if(!isa<LoadInst>(*_inst))
	{	
		outs()<<"WRONG in CoalaMemopDataMigrationH2DCallee4Cublas::_getLoadColInstruction():can not get LoadColInstruction\n";
		exit(0);
	}

	return _inst;
}



Value * CoalaMemopDataMigrationH2DCallee4Cublas::_insert_datasizecalculation_instructions(CallInst * insertpoint)
{
	Module & M  = *insertpoint->getModule();
	
	// 获得 Module 环境
	LLVMContext & Mcontext = M.getContext();

	// 创键一个指令创键环境
	IRBuilder<> builder(Mcontext);
    builder.SetInsertPoint(insertpoint);

	//Insert Load Row Instruction
	Value * loadRow = CoalaMemopDataMigrationH2DCallee4Cublas::_getLoadRowInstruction(insertpoint);
	//Insert Load Col Instruction
	Value * loadCol = CoalaMemopDataMigrationH2DCallee4Cublas::_getLoadColInstruction(insertpoint);
	//Insert %retmul = mul nsw i32 %loadCol, %loadRow
	Value * retmul1 = builder.CreateMul(loadCol, loadRow, "",false, true);
	//Insert %retsext1 = %sext i32 %retmul to i64
	Value * retsext1 = builder.CreateSExt(retmul1, Type::getInt64Ty(Mcontext));
	//Insert %retsext2 = %sext i32 %typesize to i64
	Value * retsext2 = builder.CreateSExt(CoalaMemopDataMigrationH2DCallee4Cublas::getMemopParam("TypeSize"), Type::getInt64Ty(Mcontext));
	//Insert %retmul = mul i64 %retsext1, %retsext2
	Value * retmul2 = builder.CreateMul(retsext1,retsext2);
	//Insert  %228 = trunc i64 retmul2 to i32
	Value * rettrunc = builder.CreateTrunc(retmul2, Type::getInt32Ty(Mcontext));
	return rettrunc;
}


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Get Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDataMigrationH2DCallee4Cublas::getRowDimension()
{
	return CoalaMemopDataMigrationH2DCallee4Cublas::getMemopParam("RowDimension");
}

Value * CoalaMemopDataMigrationH2DCallee4Cublas::getColDimension()
{
	return CoalaMemopDataMigrationH2DCallee4Cublas::getMemopParam("ColDimension");
}

Value * CoalaMemopDataMigrationH2DCallee4Cublas::getTypeSize()
{
	return CoalaMemopDataMigrationH2DCallee4Cublas::getMemopParam("TypeSize");
}

Value * CoalaMemopDataMigrationH2DCallee4Cublas::getHostMatPtr()
{
	return CoalaMemopDataMigrationH2DCallee4Cublas::getMemopParam("HostMatPtr");
}

Value * CoalaMemopDataMigrationH2DCallee4Cublas::getHostMatLd()
{
	return CoalaMemopDataMigrationH2DCallee4Cublas::getMemopParam("HostMatLd");
}

Value * CoalaMemopDataMigrationH2DCallee4Cublas::getDevMatPtr()
{
	return CoalaMemopDataMigrationH2DCallee4Cublas::getMemopParam("DevMatPtr");
}

Value * CoalaMemopDataMigrationH2DCallee4Cublas::getDevMatLd()
{
	return CoalaMemopDataMigrationH2DCallee4Cublas::getMemopParam("DevMatLd");
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// transform2coala Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/**
 * @brief 转化为coala_host2dev(probelist,taskID,dest_ptr,source_pte,size,...)
 * @param task_id 任务ID
 */
void CoalaMemopDataMigrationH2DCallee4Cublas::transform2coala(GlobalValue * probelist, size_t const taskid)
{	
	outs()<<"----Transforming-->"<<*CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee<<"\n";
	// 判断是否已被转化
	if( CoalaMemopDataMigrationH2DCallee4Cublas::hasbeencoala )
	{	
		outs()<<"------已被转化为-->"<<*CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee<<"\n";
		return;
	}

	//获得Module
	Module * M_ptr = CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->getModule();

	// 创建函数声明
	CoalaMemopDataMigrationH2DCallee4Cublas::coala_memop_host2dev
		 = CoalaMemopDataMigrationH2DCallee4Cublas::_create_coala_memop_host2dev_FuncDeclaration(*M_ptr);
	
	//获得Module环境
	LLVMContext & Mcontext = M_ptr->getContext();

	//创建IRBuilder对象
	IRBuilder<> IRB(Mcontext);

	//获取插入点:在callee指令前
	IRB.SetInsertPoint(CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee);

	Value * datasize = _insert_datasizecalculation_instructions(CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee);


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
		CoalaMemopDataMigrationH2DCallee4Cublas::getHostMatPtr(),
		CoalaMemopDataMigrationH2DCallee4Cublas::getDevMatPtr(),
		//插入size
		datasize
	};

	//创建callee
	CallInst * ret = IRB.CreateCall(CoalaMemopDataMigrationH2DCallee4Cublas::coala_memop_host2dev,args);
	
	
	if(ret==nullptr)
	{
		outs()<<"WRONG in CoalaMemopDataMigrationH2DCallee4Cublas::transform2coala()\n";
		return;
	}

	// outs()<<"------已成功插入"<<*ret<<"\n";

	//替换原指令返回值影响
	CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->replaceAllUsesWith(ret);

	// outs()<<"------已成功替换"<<*CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee<<"\n";

	//删除原指令
	CoalaMemopDataMigrationH2DCallee4Cublas::memop_callee->eraseFromParent();
	
	return;
}