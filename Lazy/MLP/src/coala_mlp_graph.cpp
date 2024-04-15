#include "coala_mlp_graph.h"
#include <typeinfo>

using namespace coala::mlp;

CoalaMlpGraph::CoalaMlpGraph()
{
    planning_forward_graphmat.resize(400, 0);
}

//=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
// 用户: 计划构建
//=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
int CoalaMlpGraph::addPlanningForwardNodeOp(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const user_named_node_id, int const op_func=-1)
{   
    if(user_named_node_id<0) return 1;
    if(!isOperator(node_type_code)) return 2;

    // 检查用户是否已经添加了这个id名
    for(int i=0; i<this->user_named_forward_node_ids.size(); i++)
    {
        if(this->user_named_forward_node_ids[i] == user_named_node_id) return 1;
    }

    this->planning_forward_nodes.push_back(node_type_code);
    this->user_named_forward_node_ids.push_back(user_named_node_id);

    this->planning_forward_nodeop_funcs.push_back(op_func);
    this->planning_forward_nodeva_rows.push_back(0);
    this->planning_forward_nodeva_cols.push_back(0);
    this->planning_forward_nodeva_initfuncs.push_back(COALA_MLP_INITIALIZATION_NONE);
    return 0;
}


int CoalaMlpGraph::addPlanningForwardNodeVa(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const user_named_node_id, int const rows, int const cols, COALA_MLP_INITIALIZATION const init_func=COALA_MLP_INITIALIZATION_ZERO)
{
    if(user_named_node_id<0) return 1;
    if(!isVariable(node_type_code)) return 2;
    if(rows<=0 || cols<=0) return 3;

    // 检查用户是否已经添加了这个id名
    for(int i=0; i<this->user_named_forward_node_ids.size(); i++)
    {
        if(this->user_named_forward_node_ids[i] == user_named_node_id) return 1;
    }

    this->planning_forward_nodes.push_back(node_type_code);
    this->user_named_forward_node_ids.push_back(user_named_node_id);

    this->planning_forward_nodeva_rows.push_back(rows);
    this->planning_forward_nodeva_cols.push_back(cols);
    this->planning_forward_nodeva_initfuncs.push_back(init_func);

    this->planning_forward_nodeop_funcs.push_back(-1);

    return 0;
}
int CoalaMlpGraph::addPlanningForwardEdge(int const source_id, int const dest_id)
{
    // 输入检查
    if(source_id<0 || dest_id<0) return 1;
    if(source_id == dest_id) return 1;

    int sourceidx = -1;
    int destidx  = -1;

    // 检查source_id和dest_id是否在user_named_forward_node_ids中
    for(int i=0; i<this->user_named_forward_node_ids.size(); i++)
    {
        if(user_named_forward_node_ids[i]==source_id) sourceidx = i;
        if(user_named_forward_node_ids[i]==dest_id) destidx = i; 
    }
     
    if(sourceidx == -1 || destidx == -1) return 2;


    // 检查图矩阵是不是有这么大
    int disparity =  this->planning_forward_nodes.size() * this->planning_forward_nodes.size() - this->planning_forward_graphmat.size();

    if(disparity > 0)
    {
        planning_forward_graphmat.resize(this->planning_forward_nodes.size() * this->planning_forward_nodes.size(), 0);
    }

    // source和dest必须一个是OP，一个是VAR
    if(isOperator(this->planning_forward_nodes[sourceidx]))
    {
        if(isVariable(this->planning_forward_nodes[destidx]))
        {
            // pass
        }
        else
        {
            return 3;
        }
    }
    else
    {
        if(isOperator(this->planning_forward_nodes[destidx]))
        {
            // pass
        }
        else
        {
            return 3;
        }
    }


    // 行优先存储
    this->planning_forward_graphmat[sourceidx*this->planning_forward_nodes.size() + destidx] = 1;

    return 0;
}


int CoalaMlpGraph::updatePlanningForwardNodeOp(int const user_named_node_id, COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const op_func=-1)
{
    if(user_named_node_id<0) return 1;
    if(!isOperator(node_type_code)) return 2;

    // 检查用户是否已经添加了这个id名
    int idx = -1;
    for(int i=0; i<this->user_named_forward_node_ids.size(); i++)
    {
        if(this->user_named_forward_node_ids[i] == user_named_node_id) idx = i;
    }
    if(idx<0) return 3;

    if(!isOperator(this->planning_forward_nodes[idx])) return 4;

    //更新
    this->planning_forward_nodes[idx] = node_type_code;
    this->planning_forward_nodeop_funcs[idx] = op_func;
    
    return 0;
}

int CoalaMlpGraph::updatePlanningForwardNodeVa(int const user_named_node_id, COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const rows, int const cols, COALA_MLP_INITIALIZATION const init_func=COALA_MLP_INITIALIZATION_ZERO)
{
    if(user_named_node_id<0) return 1;
    if(!isVariable(node_type_code)) return 2;

    // 检查用户是否已经添加了这个id名
    int idx = -1;
    for(int i=0; i<this->user_named_forward_node_ids.size(); i++)
    {
        if(this->user_named_forward_node_ids[i] == user_named_node_id) idx = i;
    }
    if(idx<0) return 3;

    if(!isVariable(this->planning_forward_nodes[idx])) return 4;

    //更新
    this->planning_forward_nodes[idx] = node_type_code;
    this->planning_forward_nodeva_rows[idx] = rows;
    this->planning_forward_nodeva_cols[idx] = cols;
    this->planning_forward_nodeva_initfuncs[idx] = init_func;

    return 0;
}



//=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
// 正式构建
//=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

//TODO:
int CoalaMlpGraph::constructing(void)
{
    //---------------------------------------------------------------
    // 先构造一遍计算节点
    //---------------------------------------------------------------
    for(int i=0; i<this->planning_forward_nodes.size(); i++)
    {
        // 创建节点
        switch (this->planning_forward_nodes[i])
        {
            case COALA_MLP_GRAPH_OPERATOR_COST:
                this->nodes.push_back(CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeOp(COALA_MLP_GRAPH_OPERATOR_COST_COMPUTE, this->planning_forward_nodeop_funcs[i]));
                break;
            
            case COALA_MLP_GRAPH_OPERATOR_ACTIVATE:
                this->nodes.push_back(CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeOp(COALA_MLP_GRAPH_OPERATOR_ACTIVATE_COMPUTE, this->planning_forward_nodeop_funcs[i]));
                break;
            
            case COALA_MLP_GRAPH_OPERATOR_MATMUL:
                this->nodes.push_back(CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeOp(COALA_MLP_GRAPH_OPERATOR_MATMUL_COMPUTE, this->planning_forward_nodeop_funcs[i]));
                break;

            case COALA_MLP_GRAPH_OPERATOR_PLUS:
                this->nodes.push_back(CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeOp(COALA_MLP_GRAPH_OPERATOR_PLUS_COMPUTE, this->planning_forward_nodeop_funcs[i]));
                break;
            
            case COALA_MLP_GRAPH_VARIABLE_INPUT:
                this->nodes.push_back(CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeVa(COALA_MLP_GRAPH_VARIABLE_INPUT, 
                                        this->planning_forward_nodeva_rows[i], this->planning_forward_nodeva_cols[i], this->planning_forward_nodeva_initfuncs[i]));
                break;

            case COALA_MLP_GRAPH_VARIABLE_WEIGHT:
                this->nodes.push_back(CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeVa(COALA_MLP_GRAPH_VARIABLE_WEIGHT,
                                        this->planning_forward_nodeva_rows[i], this->planning_forward_nodeva_cols[i], this->planning_forward_nodeva_initfuncs[i]));
                break;

            case COALA_MLP_GRAPH_VARIABLE_ANS:
                this->nodes.push_back(CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeVa(COALA_MLP_GRAPH_VARIABLE_ANS,
                                        this->planning_forward_nodeva_rows[i], this->planning_forward_nodeva_cols[i], this->planning_forward_nodeva_initfuncs[i]));
                break;

            default:
                break;
        }
    }
    //建立节点关系


    return 0;
}




void CoalaMlpGraph::activating()
{
    for(int i=0; i<this->nodes.size(); i++)
    {
        // this->nodes[i]->activating();
    }
    return;
}