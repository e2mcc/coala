#include "coala_mlp_graph.h"
#include <typeinfo>

using namespace coala::mlp;

CoalaMlpGraph::CoalaMlpGraph()
{
    this->forward_graphmat.resize(400, 0);
}

//=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
// 用户: 计划构建
//=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
int CoalaMlpGraph::addForwardNodeOp(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const user_named_node_id, int const op_func=0)
{   
    if(user_named_node_id<0) return 1;
    if(!isOperator(node_type_code)) return 2;

    // 检查用户是否已经添加了这个id名
    for(int i=0; i<this->user_named_forward_node_ids.size(); i++)
    {
        if(this->user_named_forward_node_ids[i] == user_named_node_id) return 1;
    }

    this->node_types.push_back(node_type_code);

    switch(node_type_code)
    {
        case COALA_MLP_GRAPH_OPERATOR_COST:
        case COALA_MLP_GRAPH_OPERATOR_COST_COMPUTE:
            this->nodes.push_back( CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeOp(COALA_MLP_GRAPH_OPERATOR_COST_COMPUTE, op_func) );
            return 0;

        case COALA_MLP_GRAPH_OPERATOR_ACTIVATE:
        case COALA_MLP_GRAPH_OPERATOR_ACTIVATE_COMPUTE:
            this->nodes.push_back( CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeOp(COALA_MLP_GRAPH_OPERATOR_ACTIVATE_COMPUTE, op_func) );
            return 0;

        case COALA_MLP_GRAPH_OPERATOR_MATMUL:
        case COALA_MLP_GRAPH_OPERATOR_MATMUL_COMPUTE:
            this->nodes.push_back( CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeOp(COALA_MLP_GRAPH_OPERATOR_MATMUL_COMPUTE, op_func) );
            return 0;

        case COALA_MLP_GRAPH_OPERATOR_PLUS:
        case COALA_MLP_GRAPH_OPERATOR_PLUS_COMPUTE:
            this->nodes.push_back( CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeOp(COALA_MLP_GRAPH_OPERATOR_PLUS_COMPUTE, op_func) );
            return 0;
        default:
            return 1;
    }

    return 1;
}


int CoalaMlpGraph::addForwardNodeVa(COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const user_named_node_id, int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func=COALA_MLP_INITIALIZE_ZERO)
{
    if(user_named_node_id<0) return 1;
    if(!isVariable(node_type_code)) return 2;
    if(rows<=0 || cols<=0) return 3;

    // 检查用户是否已经添加了这个id名
    for(int i=0; i<this->user_named_forward_node_ids.size(); i++)
    {
        if(this->user_named_forward_node_ids[i] == user_named_node_id) return 1;
    }

    this->node_types.push_back(node_type_code);

    switch(node_type_code)
    {
        case COALA_MLP_GRAPH_VARIABLE_INPUT:
            this->nodes.push_back( CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeVa(COALA_MLP_GRAPH_VARIABLE_INPUT, rows, cols, init_func) );
            return 0;
        case COALA_MLP_GRAPH_VARIABLE_WEIGHT:
            this->nodes.push_back( CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeVa(COALA_MLP_GRAPH_VARIABLE_WEIGHT, rows, cols, init_func) );
            return 0;
        case COALA_MLP_GRAPH_VARIABLE_ANS:
            this->nodes.push_back( CoalaMlpGraphNodeFactory::createACoalaMlpGraphNodeVa(COALA_MLP_GRAPH_VARIABLE_ANS, rows, cols, init_func) );
            return 0;
        default:
            return 1;
    }

    return 1;
}

int CoalaMlpGraph::addForwardEdge(int const source_id, int const dest_id)
{
    // 输入检查
    if(source_id<0 || dest_id<0) return 1;
    if(source_id == dest_id) return 1;

    int sourceidx = -1;
    int destidx  = -1;

    // 检查source_id和dest_id是否在user_named_forward_node_ids中
    for(int i=0; i<this->user_named_forward_node_ids.size(); i++)
    {
        if(this->user_named_forward_node_ids[i]==source_id) sourceidx = i;
        if(this->user_named_forward_node_ids[i]==dest_id) destidx = i; 
    }
     
    if(sourceidx == -1 || destidx == -1) return 2;


    // 检查图矩阵是不是有这么大
    int disparity =  this->nodes.size() * this->nodes.size() - this->forward_graphmat.size();

    if(disparity > 0)
    {
        this->forward_graphmat.resize(this->nodes.size() * this->nodes.size(), 0);
    }

    // source和dest必须一个是OP，一个是VAR
    if(isOperator(this->node_types[sourceidx]))
    {
        if(isVariable(this->node_types[destidx]))
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
        if(isOperator(this->node_types[destidx]))
        {
            // pass
        }
        else
        {
            return 3;
        }
    }

    // 行优先存储
    this->forward_graphmat[sourceidx*this->nodes.size() + destidx] = 1;


    // 节点连接
    this->nodes[sourceidx]->setOutputNode(this->nodes[destidx]);
    this->nodes[destidx]->setInputNode(this->nodes[sourceidx]);

    return 0;
}


int CoalaMlpGraph::updateForwardNodeOp(int const user_named_node_id, COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const op_func=0)
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

    //要修改的节点类型必须和原来的节点类型相同
    if(this->node_types[idx] != node_type_code) return 4;

    //更新
    static_cast<Operator*>(this->nodes[idx].get())->setOpFunc(op_func);
   
    return 0;
}

int CoalaMlpGraph::updateForwardNodeVa(int const user_named_node_id, COALA_MLP_GRAPH_NODE_TYPE_CODE const node_type_code, int const rows, int const cols, COALA_MLP_INITIALIZE_FUNC const init_func=COALA_MLP_INITIALIZE_ZERO)
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

    //要修改的节点类型必须和原来的节点类型相同
    if(this->node_types[idx] != node_type_code) return 4;

    //更新
    static_cast<Variable*>(this->nodes[idx].get())->setShape(rows, cols);
    static_cast<Variable*>(this->nodes[idx].get())->setInitFunc(init_func);
    return 0;
}



//=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
// 正式构建
//=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

//TODO:
int CoalaMlpGraph::constructing(void)
{   
    //将当前所有节点锁定
    for(int i=0;i<this->nodes.size();i++)
    {
        this->nodes[i]->lockin();
    }

    //构建求导节点

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