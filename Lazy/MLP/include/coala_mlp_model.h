#ifndef COALA_MLP_MODEL_H
#define COALA_MLP_MODEL_H

//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include <string>
#include <vector>
#include "coala_mlp_graph.h"
#include "coala_mlp_loss.h"
#include "coala_mlp_activation.h"
#include "coala_mlp_initialization.h"
//----------------------------------------------------------------------------------------------
// Namespace
//----------------------------------------------------------------------------------------------
namespace coala {
namespace mlp {


//----------------------------------------------------------------------------------------------
// CLASS CoalaMLP
//----------------------------------------------------------------------------------------------
class CoalaMLP
{
    protected:
    CoalaMLP(){}

    private:
    int input_layer_neurons;
    int hidden_layers_count;
    std::vector<int> hidden_layers_neurons;
    int output_layer_neurons;
    int batch_size;


    COALA_MLP_LOSS cost_func;
    std::vector<COALA_MLP_ACTIVATION> hidden_layer_activation_funcs;
    COALA_MLP_ACTIVATION output_layer_activation_func;
    COALA_MLP_INITIALIZATION initialization_func;


    //计算图
    std::shared_ptr<CoalaMlpGraph> graph;


    public:
    CoalaMLP(int const input_layer_neurons, int const hidden_layers_count=1, int const output_layer_neurons=1);
    int setHiddenLayersNeurons(int const layer, int const neurons);
    int setTraningBatch(int const batch_size);
    int setCostFunction(COALA_MLP_LOSS const cost_func);
    int setInitializationFunction(COALA_MLP_INITIALIZATION const initialization_func);
    int setHiddenLayerActivation(int const layer, COALA_MLP_ACTIVATION const activation_func);
    int setOutputLayerActivation(COALA_MLP_ACTIVATION const activation_func);
    int readyForTraining(void);


};


}//end of namespace mlp
}//end of namespace coala

#endif
