/************************************************
* Include
*************************************************/
#include "coala_traverse_entrance.h"
#include "coala_blas_entrance.h"
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>

/*********************************************
* NameSpace
**********************************************/
using namespace llvm;

/*====================================================
| Pass 主入口
| 使用 llvmGetPassPluginInfo 注册到新的 Pass Manager 里
======================================================*/
extern "C" PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK llvmGetPassPluginInfo()
{
	PassPluginLibraryInfo PPLI;
	PPLI.APIVersion = LLVM_PLUGIN_API_VERSION;
	PPLI.PluginName = "coala";
	PPLI.PluginVersion = "v0.1";
	PPLI.RegisterPassBuilderCallbacks = [](PassBuilder& passBuilder)
	{
		/*====================================================
		| Pass 主入口（必须条件5/5）
		| 使用相关的 passBuilder.register.. 开头的函数控制注册行为
		=====================================================*/
		passBuilder.registerPipelineParsingCallback //可选
		(
			[]( StringRef name, 
				ModulePassManager& MPM, //须与run方法参数表对应
				ArrayRef<PassBuilder::PipelineElement>)
			{
				int flag = 0;

				if (name == "coala-traverse")
				{
					MPM.addPass(TraversePass());
					flag++;
				}
				if (name == "coala-blas")
				{
					MPM.addPass(BlasPass());
					flag++;
				}
				if(flag>0)
					return true;
				
				return false;
			}
		); 
	};
	return PPLI;
}