/*********************************************
* Include
**********************************************/
#include "coala_cornerstone_taskgraph.h"
#include "coala_cornerstone_utils.h"
#include <llvm/IR/Constants.h>
#include <llvm/Support/raw_ostream.h>

/*********************************************
* Namespace
**********************************************/

using namespace llvm;

/*********************************************
* Function
**********************************************/

//任务有向无环图的初始化
GlobalVariable * CoalaTaskGraph::_graphInitialize(Module & M)
{
    //Assume here, in this Module, has 10 tasks at most.
    unsigned int matrix_dimension = 10;

    CoalaTaskGraph::m = matrix_dimension;
    CoalaTaskGraph::n = matrix_dimension;
	
    //Assume the init value is 1;
    int initval = 0;

    GlobalVariable * _graph_mat = createGlobalInt32ConstMatrix(M, "coalataskgraphmatrix", m, n, initval);
    return _graph_mat;
}






void CoalaTaskGraph::_addMatrixArcToConstantAggregateZero
(
    Module & M, 
    unsigned int const task_a_id, 
    unsigned int const task_b_id
)
{
    //获得Module环境
	LLVMContext & Ctx = M.getContext();

    //------------------------------------------------------------------
    // 1. 输入判断
    //------------------------------------------------------------------
    if(task_a_id >= CoalaTaskGraph::m || task_b_id >= CoalaTaskGraph::n)
    {
        outs()<<"WRONG: task_a_id 或 task_b_id 越界了\n";
        return;
    }

    //------------------------------------------------------------------
    // 2. 创建新元素
    //------------------------------------------------------------------
    // 创建数组元素类型
    Type * element_type = Type::getInt32Ty(Ctx);
    Constant * new_element = ConstantInt::get(element_type, 1);
    // 创建Zero元素类型
    Constant * zero_element = ConstantInt::get(element_type, 0);

    //------------------------------------------------------------------
    // 3. 创建新行数组
    //------------------------------------------------------------------
    std::vector<Constant*> new_element_vec;
    for(unsigned int i=0; i<CoalaTaskGraph::n; i++)
    {   

        if(task_b_id == i)
        {
            new_element_vec.push_back(new_element);
        }
        else
        {
            new_element_vec.push_back(zero_element);
        }
    }
    //转换为ArrayRef
    ArrayRef<Constant*> new_element_vec_ref(new_element_vec);
    ArrayType * row_type = ArrayType::get(element_type, CoalaTaskGraph::n);
    Constant * new_row = ConstantArray::get(row_type, new_element_vec_ref);

    //------------------------------------------------------------------
    // 4. 创建0行数组
    //------------------------------------------------------------------
    //创建0元素行
    std::vector<Constant*> zero_element_vec(CoalaTaskGraph::n,zero_element);
    ArrayRef<Constant*> zero_element_vec_ref(zero_element_vec);
    Constant * zero_row = ConstantArray::get(row_type, zero_element_vec_ref);

    //------------------------------------------------------------------
    // 5. 创建新矩阵
    //------------------------------------------------------------------
    //设置矩阵类型
    std::vector<Constant*> new_row_vec;
    for(unsigned int i=0; i<CoalaTaskGraph::m; i++)
    {   
        if(task_a_id==i)
        {
            new_row_vec.push_back(new_row);
        }
        else
        {
            new_row_vec.push_back(zero_row);
        }
    }
    //转换为ArrayRef
    ArrayRef<Constant*> new_row_vec_ref(new_row_vec);
    ArrayType * matrix_type = ArrayType::get(row_type, CoalaTaskGraph::m); 
    Constant * new_matrix = ConstantArray::get(matrix_type, new_row_vec_ref);
    

    //------------------------------------------------------------------
    // 6. 覆盖
    //------------------------------------------------------------------
    CoalaTaskGraph::graph_mat->setInitializer(new_matrix);

    return;
}


void CoalaTaskGraph::_addMatrixArcToConstantDataSequential
(
    Module & M, 
    unsigned int const task_a_id, 
    unsigned int const task_b_id
)
{   
    //获得Module环境
	LLVMContext & Ctx = M.getContext();

    //------------------------------------------------------------------
    // 1. 输入判断
    //------------------------------------------------------------------
    if(task_a_id >= CoalaTaskGraph::m || task_b_id >= CoalaTaskGraph::n)
    {
        outs()<<"WRONG: task_a_id 或 task_b_id 越界了\n";
        return;
    }


    //------------------------------------------------------------------
    // 2. 获取原始数据
    //------------------------------------------------------------------
    Constant * original_initializer = CoalaTaskGraph::graph_mat->getInitializer();
    
    //转化为ConstantArray数据类型处理
    ConstantArray * original_matrix = dyn_cast<ConstantArray>(original_initializer);
    Constant * original_row = original_matrix->getOperand(task_a_id);


    //------------------------------------------------------------------
    // 3. 创建新元素
    //------------------------------------------------------------------
    // 创建数组元素类型
    Type * element_type = Type::getInt32Ty(Ctx);
    Constant * new_element = ConstantInt::get(element_type, 1);
    // 创建Zero元素类型
    Constant * zero_element = ConstantInt::get(element_type, 0);


    //------------------------------------------------------------------
    // 4. 创建新行数组
    //------------------------------------------------------------------
    std::vector<Constant*> new_elements_vec;
    for(unsigned int i=0; i<CoalaTaskGraph::n; i++)
    {   

        if(task_b_id == i)
        {
            new_elements_vec.push_back(new_element);
        }
        else
        {   
            if(dyn_cast<ConstantAggregateZero>(original_row))
            {
                new_elements_vec.push_back(zero_element);
            }
            else
            {
                new_elements_vec.push_back(dyn_cast<ConstantDataArray>(original_row)->getElementAsConstant(i));
            }
        }
    }
    //转换为ArrayRef
    ArrayRef<Constant*> new_elements_vec_ref(new_elements_vec);
    ArrayType * row_type = ArrayType::get(element_type, CoalaTaskGraph::n);
    Constant * new_row = ConstantArray::get(row_type, new_elements_vec_ref);

    outs()<<"_addMatrixArcToConstantDataSequential():\n\t*new_row ="<<*new_row<<"\n";

    //------------------------------------------------------------------
    // 4. 创建新矩阵
    //------------------------------------------------------------------

    std::vector<Constant*> new_rows_vec;
    for(unsigned int i=0;i<CoalaTaskGraph::m;i++)
    {   
        if(i==task_a_id)
        {
            new_rows_vec.push_back(new_row);
        }
        else
        {
            new_rows_vec.push_back(original_matrix->getOperand(i));
        }
    }
    //转换为ArrayRef
    ArrayRef<Constant*> new_rows_vec_ref(new_rows_vec);
    Constant * new_matrix = ConstantArray::get(original_matrix->getType(), new_rows_vec_ref);

    //------------------------------------------------------------------
    // 6. 覆盖
    //------------------------------------------------------------------
    CoalaTaskGraph::graph_mat->setInitializer(new_matrix);

    return;
}





//设置task A 到 task B 的一条有向边
void CoalaTaskGraph::setArcA2B
(
    Module & M, 
    unsigned int const task_a_id, 
    unsigned int const task_b_id
)
{   

    //------------------------------------------------------------------
    // 1. 获取该全局变量的原始值与原始类型
    //------------------------------------------------------------------
    Constant * original_initializer = CoalaTaskGraph::graph_mat->getInitializer();

    //------------------------------------------------------------------
    // 2. 判断类型，是否为ConstantAggregateZero
    //------------------------------------------------------------------
    if(dyn_cast<ConstantAggregateZero>(original_initializer))
    {
        CoalaTaskGraph::_addMatrixArcToConstantAggregateZero(M,task_a_id,task_b_id);
    }
    else
    {
        CoalaTaskGraph::_addMatrixArcToConstantDataSequential(M,task_a_id,task_b_id);
    }
    return;
}   



int CoalaTaskGraph::getArcA2B(unsigned int const task_a_id, unsigned int const task_b_id)
{
    Constant * initializer = CoalaTaskGraph::graph_mat->getInitializer();

    ConstantArray * matrix = dyn_cast<ConstantArray>(initializer);

    unsigned int matrix_row_dimension = matrix->getNumOperands();

    //输入判断
    if(task_a_id>=matrix_row_dimension)
    {
        outs()<<"task_a_id 越界\n";
        return 0;
    }
    
    Constant * row = matrix->getOperand(task_a_id);
    
    ConstantDataArray * the_row = dyn_cast<ConstantDataArray>(row);
    
    unsigned int matrix_column_dimension = the_row->getNumElements();

    //输入判断
    if(task_b_id>=matrix_column_dimension)
    {
        outs()<<"task_b_id 越界\n";
        return 0;
    }

    Constant * element = the_row->getElementAsConstant(task_b_id);

    ConstantInt * the_element = dyn_cast<ConstantInt>(element);

    return the_element->getLimitedValue();
}

//构造函数
CoalaTaskGraph::CoalaTaskGraph(Module & M)
{
    
    //全局搜索Module中名为coalataskgraphmatrix的全局二维数组
    GlobalVariable * GV = M.getNamedGlobal("coalataskgraphmatrix");
    //如果没找到，则调用graphInitialize去初始化
    if(GV==nullptr)
    {
        CoalaTaskGraph::graph_mat = CoalaTaskGraph::_graphInitialize(M);
    }
    else
    {
        CoalaTaskGraph::graph_mat = GV;
    }


    //test
    CoalaTaskGraph::setArcA2B(M,1,2);
    CoalaTaskGraph::setArcA2B(M,0,1);
}


