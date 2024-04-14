#ifndef COALA_MLP_GRAPH_H
#define COALA_MLP_GRAPH_H

//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_node.h"
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
    protected:
    CoalaMlpGraph() {}


    private:
    std::vector<std::shared_ptr<Node>> nodes;
    int input_lables_nodeid;
    int input_features_nodeid;
    std::vector<int> nodetypes;
    int forward_mat_dimension;
    std::vector<int> forward_mat; //row-major
    int backward_mat_dimension;
    std::vector<int> backward_mat; //row-major


    public:
    virtual ~CoalaMlpGraph() {}

    //构造函数
    CoalaMlpGraph(int const nodes_count);
    
    std::shared_ptr<Node> getNode(int const id);
    int getNodesSize();
    void addNode(std::shared_ptr<Node> node, bool isXinput=false, bool isYinput=false);
    void setForwardEdge(int const source, int const dest);
    void setBackwardEdge(int const source, int const dest);
    void activating();
    
    void setInputExampleFeatures(float * data);
    void setInputExampleLables(float * data);

    void forward();
    void backward();

};



} // end of namespace mlp
} // end of namespace coala

#endif