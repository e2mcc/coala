/*********************************************
* Include
**********************************************/
#include "coala_cornerstone_probes.h"
#include "coala_cornerstone_utils.h"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Attributes.h>
#include <string>
#include <vector>
#include <cstdarg>
/*********************************************
* Namespace
**********************************************/
using namespace llvm;

/*********************************************
* Private Function
**********************************************/
GlobalVariable * CoalaProbes::_create_coala_probelist_GlobalV(Module & M)
{
    outs()<<"----creating--> coala_probelist global variable\n";
    // 目标: 在 Module 中的全局变量区域中生成
    // @coala_probelist = dso_local global ptr null, align 8 语句
    return createGlobalStructPtr(M,"coala_probelist");
}

Function * CoalaProbes::_create_coala_probelist_getOrInit_FuncDeclaration(Module & M)
{
    // 目标: 在 Module 中生成 
    // declare void @coala_probelist_getOrInit(ptr noundef) #1 语句

    //判定 Module 中是否已经存在 coala_probelist_getOrInit 函数声明
    Function * _func = findNamedFunctionDeclaration(M, "coala_probelist_getOrInit");
    if(_func != nullptr)
    {
        return _func;
    }

	// 获得 Module 环境
	LLVMContext & Mcontext = M.getContext();

    // 1. 声明返回值类型:ptr
	Type * ret_type = Type::getInt8PtrTy(Mcontext);

    // 2. 声明函数参数类型:
    std::vector<Type *> param_types = 
	{
		Type::getInt8PtrTy(Mcontext) // ptr noundef
	};

    // 3. 声明返回值类型+函数参数类型=一个函数类型：
    FunctionType * func_type = FunctionType::get(ret_type, param_types, false);

    // 4. 声明函数(如果M中有函数声明就获取，没有就插入这个函数声明)
	_func = Function::Create(func_type, Function::ExternalLinkage, "coala_probelist_getOrInit", &M);

    _func->addParamAttr(0, Attribute::NoUndef);

    return _func;
}

/**
 * @brief 插入 coala_probelist_getOrInit 函数调用并存储到全局变量 coala_probelist 中
 *
 * 在给定的 Function 中的第一条语句位置插入 coala_probelist_getOrInit 的函数调用。
 * coala_probelist_getOrInit的作用是:
 * 1. 获取内部已经初始化好了的 coala_probelist_t 的全局变量指针
 * 或2. 获取内部未初始化好的 coala_probelist_t 的全局变量指针并对它进行初始化
 * 
 * @param F Function
 *
 * @return 返回 CallInst 指针，表示创建的函数调用指令
 */
CallInst * CoalaProbes::_insert_coala_probelist_getOrInit_FuncCallee(Function & F)
{
    // 目标：生成如下语句
    // %3 = load ptr, ptr @coala_probelist, align 8
    // call void @coala_probelist_getOrInit(ptr noundef %3)

    // 获得 Module
    Module & M = *F.getParent();

	// 获得 Module 环境
	LLVMContext & Mcontext = M.getContext();

    // 5. 找插入点: 这个 Function 中的的第一个 basicblock 里的第一条指令
    BasicBlock & first_basicblock_in_function = F.getEntryBlock();
    Instruction & first_instruction_in_basicblock = *first_basicblock_in_function.getFirstInsertionPt();
    
    // 6. 创键一个指令创键环境
	IRBuilder<> builder(Mcontext);
    builder.SetInsertPoint(&first_instruction_in_basicblock);

	// 7. 创建 %3 = load ptr, ptr @probelist, align 8 指令
    LoadInst * retload = builder.CreateLoad(Type::getInt8PtrTy(Mcontext), CoalaProbes::coala_probelist);

    // 8. 在插入点位置创键函数调用
    CallInst * ret = builder.CreateCall(CoalaProbes::coala_probelist_getOrInit,retload);
    
    ret->addParamAttr(0, Attribute::NoUndef);

    // 7. 创建 store ptr %177, ptr @probelist, align 8 指令
    builder.CreateStore(ret, CoalaProbes::coala_probelist);

    outs()<<"----Inserted--> "<<*ret<<"\n";

    return ret;
}


Function * CoalaProbes::_create_coala_probe_FuncDeclaration(Module & M)
{   
    // 目标: 在 Module 中生成
    // declare void @coala_probe(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ...) 语句

    //判定 Module 中是否已经存在 coala_probe 函数声明
    Function * _func = findNamedFunctionDeclaration(M, "coala_probe");
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
        Type::getInt64Ty(Mcontext),  // i64 noundef
        Type::getInt64Ty(Mcontext),  // i64 noundef
        Type::getInt64Ty(Mcontext)  // i64 noundef
	};

    // 3. 声明返回值类型+函数参数类型=一个函数类型：
    FunctionType * func_type = FunctionType::get(ret_type, param_types, true);

    // 4. 声明函数(如果M中有函数声明就获取，没有就插入这个函数声明)
	_func = Function::Create(func_type, Function::ExternalLinkage, "coala_probe", &M);

    // 5. 设置参数属性
    _func->addParamAttr(0, Attribute::NoUndef);
    _func->addParamAttr(1, Attribute::NoUndef);
    _func->addParamAttr(2, Attribute::NoUndef);
    _func->addParamAttr(3, Attribute::NoUndef);


    return _func;
}



void CoalaProbes::insert_coala_probe_FuncCallee
(
    CallInst * insertpoint,
    size_t const taskid,
    size_t const taskcode,
    size_t const dnum,
    ...
)
{
    // 目标：生成如下语句
    // %3 = load ptr, ptr @coala_probelist, align 8
    // call void @coala_probe(ptr noundef %3)

    // 获得 Module
   
    Module & M = *insertpoint->getModule();

	// 获得 Module 环境
	LLVMContext & Mcontext = M.getContext();
    
    // 6. 创键一个指令创键环境
	IRBuilder<> builder(Mcontext);
    builder.SetInsertPoint(insertpoint);

	// 7. 创建 %3 = load ptr, ptr @probelist, align 8 指令
    LoadInst * retload = builder.CreateLoad(Type::getInt8PtrTy(Mcontext), CoalaProbes::coala_probelist);

    //创建一个taskid 的 constant
    Type * element_i64type = Type::getInt64Ty(Mcontext);
	Constant * taskid_element = ConstantInt::get(element_i64type, taskid);
    Constant * taskcode_element = ConstantInt::get(element_i64type, taskcode);
    Constant * dnum_element = ConstantInt::get(element_i64type, dnum);

    std::vector<Value*> callee_args;

    callee_args.push_back(retload);
    callee_args.push_back(taskid_element);
    callee_args.push_back(taskcode_element);
    callee_args.push_back(dnum_element);

    va_list args;
    va_start(args, dnum);
    for (size_t i = 0; i < dnum; ++i) {
        Value * dimension = va_arg(args, Value*);
        callee_args.push_back(dimension);
    }
    va_end(args);
    
    // 8. 在插入点位置创键函数调用
    CallInst * ret = builder.CreateCall(CoalaProbes::coala_probe,callee_args);

    ret->addParamAttr(0, Attribute::NoUndef);
    ret->addParamAttr(1, Attribute::NoUndef);
    ret->addParamAttr(2, Attribute::NoUndef);
    ret->addParamAttr(3, Attribute::NoUndef);

    outs()<<"----Inserted--> "<<*ret<<"\n";
    return;
}




/*********************************************
* Public Function
**********************************************/
/**
 * @brief CoalaProbes 构造函数
 *
 * CoalaProbes 类的构造函数，用于初始化 CoalaProbes 对象。
 *
 * @param M 模块引用
 */
CoalaProbes::CoalaProbes(Function & F)
{   
    //设置 coala_probelist
    GlobalVariable * _probes_t = findNamedGlobalVariableInModule(*F.getParent(), "coala_probelist");
    if(_probes_t==nullptr)
    {
       CoalaProbes::coala_probelist = CoalaProbes::_create_coala_probelist_GlobalV(*F.getParent());
    }
    else
    {
        CoalaProbes::coala_probelist = _probes_t;
        outs()<<"Found 'coala_probelist' in this LLVM Module:\n"<<*CoalaProbes::coala_probelist<<"\n";
    }

    //设置 coala_probelist_getOrInit
    CoalaProbes::coala_probelist_getOrInit = CoalaProbes::_create_coala_probelist_getOrInit_FuncDeclaration(*F.getParent());

    //设置 coala_probelist_getOrInit_callee
    CallInst * _callee = findFirstNamedCalleeInFunction(F, "coala_probelist_getOrInit");
    if(_callee==nullptr)
    {
        CoalaProbes::coala_probelist_getOrInit_callee = CoalaProbes::_insert_coala_probelist_getOrInit_FuncCallee(F);
    }
    else
    {
        CoalaProbes::coala_probelist_getOrInit_callee = _callee;
        outs()<<"Found 'coala_probelist_getOrInit' in this LLVM Function:\n"<<*CoalaProbes::coala_probelist_getOrInit_callee<<"\n";
    }

    //设置 coala_probe
    CoalaProbes::coala_probe = CoalaProbes::_create_coala_probe_FuncDeclaration(*F.getParent());
}


GlobalVariable * CoalaProbes::getCoalaProbelistGV()
{
    return CoalaProbes::coala_probelist;
}
