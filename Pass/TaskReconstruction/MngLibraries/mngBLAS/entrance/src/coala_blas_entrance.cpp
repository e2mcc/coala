/************************************************
* Include
*************************************************/
#include "coala_blas_entrance.h"
#include "coala_cornerstone_taskgraph.h"
#include "coala_cornerstone_probes.h"
#include "coala_blas_management.h"
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/PassManager.h>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Function.h>//for Function class

//test
// #include "coala_cornerstone_utils.h"
// #include "coala_host_malloc.h"
// #include "coala_host_free.h"
// #include "coala_device_malloc.h"
// #include "coala_device_free.h"
/*********************************************
* NameSpace
**********************************************/
using namespace llvm;

/*==================================================
| Pass 主入口（必须条件3/5）
| 实现重载 run 方法，需要特定的 return 返回值
===================================================*/
PreservedAnalyses BlasPass::run(Module & M, ModuleAnalysisManager &MAM)
{
	outs()<<"This is the BlasPass\n";
	
	// --------------------------
	// 1.获取 Module 全局 task graph 对象
	// ---------------------------
	// CoalaTaskGraph CTG(M);
	
	// --------------------------
	// 2.初始化probes
	// ---------------------------
	// CoalaProbes CPs(M);

	// --------------------------
	// 3.扫描整个 Module 获取 task
	// ---------------------------
	//TODO: debug：这里只要用了 BlasManagement 类的实例，就会在使用的时候找不到 pass
	BlasManagement BM(M);
	// BlasManagement BM(M,CPs);
	
	
	//保存
	PreservedAnalyses PA;
	return PA.all();
}

