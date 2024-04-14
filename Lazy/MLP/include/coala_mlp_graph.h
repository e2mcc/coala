#ifndef COALA_MLP_GRAPH_H
#define COALA_MLP_GRAPH_H

//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_node.h"
#include "coala_mlp_node_op.h"
#include "coala_mlp_node_var.h"
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
    
    std::vector<int> planning_forward_graphmat;

    public:
    /// @brief 添加计划构建的正向传播图的节点
    int addPlanningForwardNode(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const user_named_node_id);
    /// @brief 添加计划构建的正向传播图的节点关系
    int addPlanningForwardEdge(int const source_id, int const dest_id);
    


    //--------------------------------------------------------------------
    // 本图根据用户计划进行正式构建( 通过 activating() 启动 )
    //--------------------------------------------------------------------
    private:
    std::vector<std::shared_ptr<Node>> nodes;
    
    int Constructing(void);
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