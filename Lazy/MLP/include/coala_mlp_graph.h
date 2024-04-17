#ifndef COALA_MLP_GRAPH_H
#define COALA_MLP_GRAPH_H

//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_node.h"
#include "coala_mlp_node_op.h"
#include "coala_mlp_node_va.h"
#include "coala_mlp_initialize.h"
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

    private:
    std::vector<std::shared_ptr<Node>> nodes;

    // 节点关系
    std::vector<int> forward_graphmat;

    // 节点类型
    std::vector<COALA_MLP_GRAPH_NODE_TYPE_CODE> node_types;

    int inputX_id;
    int inputY_id;
    int loss_id;

    //--------------------------------------------------------------------
    // 用户: 计划构建
    //--------------------------------------------------------------------
    private:
    // 用户自定义命名的节点编号
    std::vector<int> user_named_forward_node_ids;

    public:
    /// @brief 添加计划构建的正向传播图的操作节点
    int addForwardNodeOp(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const user_named_node_id, int const op_func=0);
    /// @brief 添加计划构建的正向传播图的变量节点
    int addForwardNodeVa(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const user_named_node_id, int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func=COALA_MLP_INITIALIZE_ZERO);
    /// @brief 添加计划构建的正向传播图的节点关系
    int addForwardEdge(int const source_id, int const dest_id);
    
    /// @brief 更新计划构建的正向传播图的操作节点
    int updateForwardNodeOp(int const user_named_node_id, COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const op_func=0);
    /// @brief 更新计划构建的正向传播图的变量节点
    int updateForwardNodeVa(int const user_named_node_id, COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func=COALA_MLP_INITIALIZE_ZERO);

    ///
    int setInputXId(int const user_named_node_id);
    int setInputYId(int const user_named_node_id);
    int setLossId(int const user_named_node_id);

    //--------------------------------------------------------------------
    // 本图根据正向传播图进行正式扩展为完整的计算图 (通过 activating 启动 constructing 和 parallelAnalyzing)
    //--------------------------------------------------------------------
    private:
    int constructing(void);

    std::vector<std::vector<int>> rowparallel_matrix;
    int parallelAnalyzing(void);

    public:
    int activating(void);

    //--------------------------------------------------------------------
    // 执行
    //--------------------------------------------------------------------
    public:
    /// @brief 激活计算图(激活后可进行正反传播计算)
    void forward(void);
    void backward(void);

};



} // end of namespace mlp
} // end of namespace coala

#endif