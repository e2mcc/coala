#include "coala_mlp_graph.h"
#include <typeinfo>

using namespace coala::mlp;


CoalaMlpGraph::CoalaMlpGraph(int const nodes_count)
{
    this->nodetypes = std::vector<int>(nodes_count, 0);
    this->forward_mat_dimension = nodes_count;
    this->forward_mat = std::vector<int>(this->forward_mat_dimension*this->forward_mat_dimension, 0);
    for(int i=0; i<this->forward_mat_dimension; i++)
    {
        this->forward_mat[i+i*this->forward_mat_dimension] = 1;
    }
    this->backward_mat_dimension = nodes_count;
    this->backward_mat = std::vector<int>(this->backward_mat_dimension*this->backward_mat_dimension, 0);
    for(int i=0; i<this->backward_mat_dimension; i++)
    {
        this->backward_mat[i+i*this->backward_mat_dimension] = 1;
    }
}


std::shared_ptr<Node> CoalaMlpGraph::getNode(int const id)
{
    return this->nodes[id];
}


int CoalaMlpGraph::getNodesSize()
{
    return this->nodes.size();
}

void CoalaMlpGraph::addNode(std::shared_ptr<Node> node, bool isXinput=false, bool isYinput=false)
{   
    //输入正确性检查
    if(isXinput && isYinput) return;
    if( (isXinput || isYinput) && typeid(node)!=typeid(Variable) ) return;

    this->nodes.push_back(node);
    
    if(isXinput)
    {
        this->input_features_nodeid = this->nodes.size()-1;
    }
    else if(isYinput)
    {
        this->input_lables_nodeid = this->nodes.size()-1;
    }

    if(typeid(node)==typeid(Variable))
    {
        this->nodetypes[this->nodes.size()-1] = 0;
    }
    else if(typeid(node)==typeid(Operator))
    {
        this->nodetypes[this->nodes.size()-1] = 1;
    }
    else
    {
        return;
    }

    return;
}


void CoalaMlpGraph::setForwardEdge(int const source, int const dest)
{
    if(source < 0 || source >= this->nodes.size()) return;
    if(dest < 0 || dest >= this->nodes.size()) return;

    //this->forward_mat is row-major
    this->forward_mat[source*this->forward_mat_dimension+dest] = 1;

    this->nodes[source]->addForwardNode(this->nodes[dest]);
    return;
}



void CoalaMlpGraph::setBackwardEdge(int const source, int const dest)
{
    if(source < 0 || source >= this->nodes.size()) return;
    if(dest < 0 || dest >= this->nodes.size()) return;

    //this->backward_mat is row-major
    this->backward_mat[source*this->backward_mat_dimension+dest] = 1;

    this->nodes[source]->addBackwardNode(this->nodes[dest]);
    return;
}



void CoalaMlpGraph::activating()
{
    for(int i=0; i<this->nodes.size(); i++)
    {
        this->nodes[i]->activating();
    }
    return;
}