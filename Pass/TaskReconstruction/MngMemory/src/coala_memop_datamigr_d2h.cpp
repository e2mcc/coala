/*====================================================================
| INCLUDE
======================================================================*/
#include "coala_memop_datamigr_d2h.h"
#include "coala_cornerstone_utils.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>

/*====================================================================
| NameSpace
======================================================================*/
using namespace llvm;


/*====================================================================
| Derived Subbase Class : CoalaMemopDataMigrationD2HCallee
======================================================================*/
// private function : _create_coala_data_transmission_h2d_declaration
// int coala_dev2host(coala_probes_t *, int, void*, void*, size_t, ...)
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Function * CoalaMemopDataMigrationD2HCallee::_create_coala_memop_dev2host_FuncDeclaration(Module & M)
{
	//判定 Module 中是否已经存在 coala_probe 函数声明
    Function * _func = findNamedFunctionDeclaration(M, "coala_memop_dev2host");
    if(_func != nullptr)
    {	
		outs()<<"----coala_memop_dev2host function Declaration already exists.\n";
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
	_func = Function::Create(func_type, Function::ExternalLinkage, "coala_memop_dev2host", &M);

    // 5. 设置参数属性
    _func->addParamAttr(0, Attribute::NoUndef);
    _func->addParamAttr(1, Attribute::NoUndef);
    _func->addParamAttr(2, Attribute::NoUndef);
    _func->addParamAttr(3, Attribute::NoUndef);
	_func->addParamAttr(4, Attribute::NoUndef);

    return _func;
}



/*====================================================================
| Derived Class : CoalaMemopDataMigrationD2HCallee4Cuda
======================================================================*/
// Set Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDataMigrationD2HCallee4Cuda::setHostPtr()
{
	if( CoalaMemopDataMigrationD2HCallee4Cuda::memop_callee->arg_size()!=4 ) return nullptr;
	Value * _host_ptr = CoalaMemopDataMigrationD2HCallee4Cuda::memop_callee->getArgOperand(0);
	return _host_ptr;
}

Value * CoalaMemopDataMigrationD2HCallee4Cuda::setDevPtr()
{
	if( CoalaMemopDataMigrationD2HCallee4Cuda::memop_callee->arg_size()!=4 ) return nullptr;
	Value * _device_ptr = CoalaMemopDataMigrationD2HCallee4Cuda::memop_callee->getArgOperand(1);
	return _device_ptr;
}


Value * CoalaMemopDataMigrationD2HCallee4Cuda::setSize()
{
	if( CoalaMemopDataMigrationD2HCallee4Cuda::memop_callee->arg_size()!=4 ) return nullptr;
	Value * _size = CoalaMemopDataMigrationD2HCallee4Cuda::memop_callee->getArgOperand(2);
	return _size;
}

Value * CoalaMemopDataMigrationD2HCallee4Cuda::setDTFlag()
{
	if( CoalaMemopDataMigrationD2HCallee4Cuda::memop_callee->arg_size()!=4 ) return nullptr;
	Value * _data_transmission_flag = CoalaMemopDataMigrationD2HCallee4Cuda::memop_callee->getArgOperand(3);
	return _data_transmission_flag;
}


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Get Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDataMigrationD2HCallee4Cuda::getHostPtr()
{
	return CoalaMemopDataMigrationD2HCallee4Cuda::getMemopParam("HostPtr");
}

Value * CoalaMemopDataMigrationD2HCallee4Cuda::getDevPtr()
{
	return CoalaMemopDataMigrationD2HCallee4Cuda::getMemopParam("DevPtr");
}

Value * CoalaMemopDataMigrationD2HCallee4Cuda::getSize()
{
	return CoalaMemopDataMigrationD2HCallee4Cuda::getMemopParam("Size");
}

Value * CoalaMemopDataMigrationD2HCallee4Cuda::getDTFlag()
{
	return CoalaMemopDataMigrationD2HCallee4Cuda::getMemopParam("DTFlag");
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// transform2coala Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/**
 * @brief 转化为coala_host2dev(probelist,taskID,dest_ptr,source_pte,size,...)
 * @param task_id 任务ID
 */
void CoalaMemopDataMigrationD2HCallee4Cuda::transform2coala(GlobalValue * probelist, size_t const taskid)
{	
	return;
}




/*====================================================================
| Derived Class : CoalaMemopDataMigrationD2HCallee4Cublas
======================================================================*/
// Set Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDataMigrationD2HCallee4Cublas::setRowDimension()
{
	if( CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->arg_size()!=7 ) return nullptr;
	Value * _row_dimension = CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->getArgOperand(0);
	return _row_dimension;
}

Value * CoalaMemopDataMigrationD2HCallee4Cublas::setColDimension()
{
	if( CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->arg_size()!=7 ) return nullptr;
	Value * _col_dimension = CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->getArgOperand(1);
	return _col_dimension;
}

Value * CoalaMemopDataMigrationD2HCallee4Cublas::setTypeSize()
{
	if( CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->arg_size()!=7 ) return nullptr;
	Value * _size = CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->getArgOperand(2);
	return _size;
}

Value * CoalaMemopDataMigrationD2HCallee4Cublas::setDevMatPtr()
{
	if( CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->arg_size()!=7 ) return nullptr;
	Value * _dev_mat_ptr = CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->getArgOperand(3);
	return _dev_mat_ptr;
}

Value * CoalaMemopDataMigrationD2HCallee4Cublas::setDevMatLd()
{
	if( CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->arg_size()!=7 ) return nullptr;
	Value * _dev_mat_ld = CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->getArgOperand(4);
	return _dev_mat_ld;
}

Value * CoalaMemopDataMigrationD2HCallee4Cublas::setHostMatPtr()
{
	if( CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->arg_size()!=7 ) return nullptr;
	Value * _host_mat_ptr = CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->getArgOperand(5);
	return _host_mat_ptr;
}

Value * CoalaMemopDataMigrationD2HCallee4Cublas::setHostMatLd()
{
	if( CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->arg_size()!=7 ) return nullptr;
	Value * _host_mat_ld = CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->getArgOperand(6);
	return _host_mat_ld;
}



Value * CoalaMemopDataMigrationD2HCallee4Cublas::_getLoadRowInstruction(CallInst * insertpoint)
{
	Value * _inst = CoalaMemopDataMigrationD2HCallee4Cublas::getMemopParam("RowDimension");
	
	if(_inst==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaMemopDataMigrationD2HCallee4Cublas::_getLoadRowInstruction()\n";
		exit(0);
	}

	if(!isa<LoadInst>(*_inst))
	{	
		outs()<<"WRONG in CoalaMemopDataMigrationD2HCallee4Cublas::_getLoadRowInstruction():can not get LoadRowInstruction\n";
		exit(0);
	}

	return _inst;
}


Value * CoalaMemopDataMigrationD2HCallee4Cublas::_getLoadColInstruction(CallInst * insertpoint)
{
	Value * _inst = CoalaMemopDataMigrationD2HCallee4Cublas::getMemopParam("ColDimension");
	
	if(_inst==nullptr)
	{
		outs()<<"WRONG: Invalid Call of CoalaMemopDataMigrationD2HCallee4Cublas::_getLoadColInstruction()\n";
		exit(0);
	}

	if(!isa<LoadInst>(*_inst))
	{	
		outs()<<"WRONG in CoalaMemopDataMigrationD2HCallee4Cublas::_getLoadColInstruction():can not get LoadColInstruction\n";
		exit(0);
	}

	return _inst;
}



Value * CoalaMemopDataMigrationD2HCallee4Cublas::_insert_datasizecalculation_instructions(CallInst * insertpoint)
{
	Module & M  = *insertpoint->getModule();
	
	// 获得 Module 环境
	LLVMContext & Mcontext = M.getContext();

	// 创键一个指令创键环境
	IRBuilder<> builder(Mcontext);
    builder.SetInsertPoint(insertpoint);

	//Insert Load Row Instruction
	Value * loadRow = CoalaMemopDataMigrationD2HCallee4Cublas::_getLoadRowInstruction(insertpoint);
	//Insert Load Col Instruction
	Value * loadCol = CoalaMemopDataMigrationD2HCallee4Cublas::_getLoadColInstruction(insertpoint);
	//Insert %retmul = mul nsw i32 %loadCol, %loadRow
	Value * retmul1 = builder.CreateMul(loadCol, loadRow, "",false, true);
	//Insert %retsext1 = %sext i32 %retmul to i64
	Value * retsext1 = builder.CreateSExt(retmul1, Type::getInt64Ty(Mcontext));
	//Insert %retsext2 = %sext i32 %typesize to i64
	Value * retsext2 = builder.CreateSExt(CoalaMemopDataMigrationD2HCallee4Cublas::getMemopParam("TypeSize"), Type::getInt64Ty(Mcontext));
	//Insert %retmul = mul i64 %retsext1, %retsext2
	Value * retmul2 = builder.CreateMul(retsext1,retsext2);
	//Insert  %228 = trunc i64 retmul2 to i32
	Value * rettrunc = builder.CreateTrunc(retmul2, Type::getInt32Ty(Mcontext));
	return rettrunc;
}





//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Get Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Value * CoalaMemopDataMigrationD2HCallee4Cublas::getRowDimension()
{
	return CoalaMemopDataMigrationD2HCallee4Cublas::getMemopParam("RowDimension");
}

Value * CoalaMemopDataMigrationD2HCallee4Cublas::getColDimension()
{
	return CoalaMemopDataMigrationD2HCallee4Cublas::getMemopParam("ColDimension");
}

Value * CoalaMemopDataMigrationD2HCallee4Cublas::getTypeSize()
{
	return CoalaMemopDataMigrationD2HCallee4Cublas::getMemopParam("TypeSize");
}

Value * CoalaMemopDataMigrationD2HCallee4Cublas::getDevMatPtr()
{
	return CoalaMemopDataMigrationD2HCallee4Cublas::getMemopParam("DevMatPtr");
}

Value * CoalaMemopDataMigrationD2HCallee4Cublas::getDevMatLd()
{
	return CoalaMemopDataMigrationD2HCallee4Cublas::getMemopParam("DevMatLd");
}

Value * CoalaMemopDataMigrationD2HCallee4Cublas::getHostMatPtr()
{
	return CoalaMemopDataMigrationD2HCallee4Cublas::getMemopParam("HostMatPtr");
}

Value * CoalaMemopDataMigrationD2HCallee4Cublas::getHostMatLd()
{
	return CoalaMemopDataMigrationD2HCallee4Cublas::getMemopParam("HostMatLd");
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// transform2coala Function
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/**
 * @brief 转化为coala_host2dev(probelist,taskID,dest_ptr,source_pte,size,...)
 * @param task_id 任务ID
 */
void CoalaMemopDataMigrationD2HCallee4Cublas::transform2coala(GlobalValue * probelist, size_t const taskid)
{	
	outs()<<"----transforming-->"<<*CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee<<"\n";
	// 判断是否已被转化
	if( CoalaMemopDataMigrationD2HCallee4Cublas::hasbeencoala )
	{	
		outs()<<"------已被转化为-->"<<*CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee<<"\n";
		return;
	}

	//获得Module
	Module * M_ptr = CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->getModule();

	// 创建函数声明
	CoalaMemopDataMigrationD2HCallee4Cublas::coala_memop_dev2host
		 = CoalaMemopDataMigrationD2HCallee4Cublas::_create_coala_memop_dev2host_FuncDeclaration(*M_ptr);
	
	//获得Module环境
	LLVMContext & Mcontext = M_ptr->getContext();

	//创建IRBuilder对象
	IRBuilder<> IRB(Mcontext);

	//获取插入点:在callee指令前
	IRB.SetInsertPoint(CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee);

	Value * datasize = _insert_datasizecalculation_instructions(CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee);

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
		CoalaMemopDataMigrationD2HCallee4Cublas::getDevMatPtr(),
		CoalaMemopDataMigrationD2HCallee4Cublas::getHostMatPtr(),
		//插入size
		datasize
	};


	//创建callee
	CallInst * ret = IRB.CreateCall(CoalaMemopDataMigrationD2HCallee4Cublas::coala_memop_dev2host,args);
	if(ret==nullptr)
	{
		outs()<<"WRONG in CoalaMemopDataMigrationD2HCallee4Cublas::transform2coala()\n";
		return;
	}
	// outs()<<"------已成功插入"<<*ret<<"\n";

	//替换原指令返回值影响
	CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->replaceAllUsesWith(ret);

	// outs()<<"------已成功替换"<<*CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee<<"\n";

	//删除原指令
	CoalaMemopDataMigrationD2HCallee4Cublas::memop_callee->eraseFromParent();

	return;
}