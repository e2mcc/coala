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
// Classes
//----------------------------------------------------------------------------------------------
class CoalaMlpModel
{
    protected:
    CoalaMlpModel(){}

    private:
    int input_size;                             //< The size of the input to the model.
    int hidden_layers_count;                    //< The number of hidden layers in the model.
    int output_size;                            //< The size of the output from the model.
    std::shared_ptr<CoalaMlpInputLayer>  input_layer;                   //< The input layer of the model.
    std::vector<std::shared_ptr<CoalaMlpHiddenLayer>> hidden_layers;    //< The hidden layers of the model.
    std::shared_ptr<CoalaMlpOutputLayer>  output_layer;                 //< The output layer of the model.
    float learning_rate;                           //< The learning rate for weight updates.
    int trained_times;                          //< The number of times the model has been trained.
    
    public:
    CoalaMlpModel(int input_size, int hidden_layers_count=1, int hidden_layers_output_size=5, int output_size=1, float learning_rate=0.01f);
    
    int setHiddenLayer(int hidden_layer_rank, COALA_MLP_ACTIVATION activation_rank, int output_size);
    int setOutputLayerActivation(COALA_MLP_ACTIVATION activation_rank);


    void initializeWeights(COALA_MLP_INITIALIZATION initialization_rank);

    /**
     * @brief Performs forward propagation for the model.
     * 
     * @param input The input to the model.
     * @param output The output of the model.
     */
    void forward(float* input, int examples);
    
    /**
     * @brief Performs backward propagation for the model.
     * 
     * @param input The input to the model.
     * @param output The output of the model.
     */
    void backward(float* input, float* output);

    /**
     * @brief Updates the weights and biases of the model using the specified learning rate.
     * 
     * @param learning_rate The learning rate for weight updates.
     */
    void update(float learning_rate);
};


#endif // COALA_MLP_MODEL_H