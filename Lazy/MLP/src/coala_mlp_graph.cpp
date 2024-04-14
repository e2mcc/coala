#include "coala_mlp_graph.h"
#include <typeinfo>

using namespace coala::mlp;

CoalaMlpGraph::CoalaMlpGraph()
{
    planning_forward_graphmat.resize(400, 0);
}


int CoalaMlpGraph::addPlanningForwardNode(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const user_named_node_id, bool is_inputX=false, bool is_inputY=false)
{   
    if(user_named_node_id<0) return 1;
    if(is_inputX && is_inputY) return 2;
    if(is_inputX && isVariable(node_type_code)==false) return 3;
    if(is_inputY && isVariable(node_type_code)==false) return 3;

    // 检查用户是否已经添加了这个id名
    for(int i=0; i<this->user_named_forward_node_ids.size(); i++)
    {
        if(this->user_named_forward_node_ids[i] == user_named_node_id) return 1;
    }

    this->planning_forward_nodes.push_back(node_type_code);
    this->user_named_forward_node_ids.push_back(user_named_node_id);
    
    if(is_inputX)
    {
        this->inputX_id = this->planning_forward_nodes.size() - 1;
    }

    if(is_inputY)
    {
        this->inputY_id = this->planning_forward_nodes.size() - 1;
    }

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



void CoalaMlpGraph::activating()
{
    for(int i=0; i<this->nodes.size(); i++)
    {
        // this->nodes[i]->activating();
    }
    return;
}