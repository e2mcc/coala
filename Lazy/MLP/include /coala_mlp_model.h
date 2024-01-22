#ifndef COALA_MLP_MODEL_H
#define COALA_MLP_MODEL_H

//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_layer.h"
#include "coala_mlp_activation.h"
#include "coala_mlp_initialization.h"
#include <cstdlib>
#include <cstring>
#include <vector>

//----------------------------------------------------------------------------------------------
// Class
//----------------------------------------------------------------------------------------------
class CoalaMlpModel
{
    protected:
    CoalaMlpModel(){}

    private:
    //Input Layer
    std::shared_ptr<CoalaMlpInputLayer>  input_layer;

    //Hidden Layer
    int hidden_layers_count;
    std::vector<std::shared_ptr<CoalaMlpHiddenLayer>> hidden_layers;

    //Output Layer
    std::shared_ptr<CoalaMlpOutputLayer>  output_layer;
    
    
    float learning_rate;
    int trained_times;
    float loss;

    public:
    CoalaMlpModel(int input_layer_neurons, int hidden_layers_count=1, int hidden_layers_neurons=5, int output_layer_neurons=1, float learning_rate=0.01f);
    
    int setHiddenLayer(int hidden_layer_rank, COALA_MLP_ACTIVATION activation_rank, int output_size);
    int setOutputLayerActivation(COALA_MLP_ACTIVATION activation_rank);

    void initializeWeights(COALA_MLP_INITIALIZATION initialization_rank);

    void forward(float* Mat, int examples, int features);
    
    float cost(float * MatPred, float * MatReal, int examples, int output_layer_neurons);

    void backward(float * real_mat, int examples, int real_dim);

    void update(float learning_rate);

    void saveToFile(std::string filename);
};

#endif // COALA_MLP_MODEL_H