#ifndef COALA_MLP_GRAPH_H
#define COALA_MLP_GRAPH_H

//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_node.h"
#include "coala_mlp_node_op.h"
#include "coala_mlp_node_var.h"
#include "coala_mlp_initialization.h"
#include <string>
#include <vector>

//----------------------------------------------------------------------------------------------
// Namespace
//----------------------------------------------------------------------------------------------
namespace coala {
namespace mlp {


//----------------------------------------------------------------------------------------------
// CLASS CoalaMlpGraph
//----------------------------------------------------------------------------------------------
class CoalaMlpGraph
{
    public:
    CoalaMlpGraph();

    //--------------------------------------------------------------------
    // 用户: 计划构建
    //--------------------------------------------------------------------
    private:
    std::vector<COALA_MLP_GRAPH_NODE_TYPE_CODE> planning_forward_nodes;
    
    // 用户自定义命名的节点编号
    std::vector<int> user_named_forward_node_ids;
    
    // 用户自定义的变量节点的行列数
    std::vector<int> planning_forward_nodeva_rows;
    std::vector<int> planning_forward_nodeva_cols;
    // 用户自定义的变量节点的初始化函数
    std::vector<COALA_MLP_INITIALIZATION> planning_forward_nodeva_initfuncs;

    // 用户自定义的操作节点的操作函数
    std::vector<int> planning_forward_nodeop_funcs;

    // 用户自定义的节点关系
    std::vector<int> planning_forward_graphmat;

    public:
    /// @brief 添加计划构建的正向传播图的操作节点
    int addPlanningForwardNodeOp(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const user_named_node_id, int const op_func=-1);
    /// @brief 添加计划构建的正向传播图的变量节点
    int addPlanningForwardNodeVa(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const user_named_node_id, int const rows, int const cols, COALA_MLP_INITIALIZATION const init_func=COALA_MLP_INITIALIZATION_ZERO);
    /// @brief 添加计划构建的正向传播图的节点关系
    int addPlanningForwardEdge(int const source_id, int const dest_id);
    
    /// @brief 更新计划构建的正向传播图的操作节点
    int updatePlanningForwardNodeOp(int const user_named_node_id, COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const op_func=-1);
    /// @brief 更新计划构建的正向传播图的变量节点
    int updatePlanningForwardNodeVa(int const user_named_node_id, COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const rows, int const cols, COALA_MLP_INITIALIZATION const init_func=COALA_MLP_INITIALIZATION_ZERO);


    //--------------------------------------------------------------------
    // 本图根据用户计划进行正式构建( 通过 activating() 启动 )
    //--------------------------------------------------------------------
    private:
    std::vector<std::shared_ptr<Node>> nodes;
    
    int constructing(void);

    public:

    void activating(void);
    
    
    




    //--------------------------------------------------------------------
    // 执行
    //--------------------------------------------------------------------
    public:
    /// @brief 激活计算图(激活后可进行正反传播计算)
    

};



} // end of namespace mlp
} // end of namespace coala

#endif