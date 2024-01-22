#ifndef COALA_MLP_LAYER_H
#define COALA_MLP_LAYER_H

//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include <cstdlib>
#include "coala_mlp_activation.h"
#include "coala_mlp_initialization.h"


//----------------------------------------------------------------------------------------------
// Classes
//----------------------------------------------------------------------------------------------
class CoalaMlpInputLayer
{
    protected:
    CoalaMlpInputLayer(){}

    private:
    float * input;
    float * output;                 //< The input features to the layer.
    int neurons;         //< The size of the input.

    public:
    /**
     * @brief Constructs a CoalaMlpInputLayer object with the specified input and output sizes.
     * 
     * @param neurons The number of the features withnin the input.
     */
    CoalaMlpInputLayer(int neurons);
    
    /**
     * @brief Performs forward propagation for the layer.
     * 
     * @param input The input to the layer.
     * @param output The output of the layer.
     */
    void forward(float * mat, int examples, int features);
    int getNeuronsNum();
    float * getOutput();
    
    /**
     * @brief Performs backward propagation for the layer.
     * 
     * @param input The input to the layer.
     * @param output The output of the layer.
     */
    void backward();

    /**
     * @brief Updates the weights and biases of the layer using the specified learning rate.
     * 
     * @param learning_rate The learning rate for weight updates.
     */
    void update(float learning_rate);
};



/**
 * @brief Represents a hidden layer in a Coala Multi-Layer Perceptron (MLP).
 * 
 * This class encapsulates the functionality of a hidden layer in a Coala MLP.
 * It contains weights, biases, gradients, and other variables necessary for forward and backward propagation.
 * The layer can perform forward propagation, backward propagation, and weight updates.
 */
class CoalaMlpHiddenLayer
{
    protected:
    CoalaMlpHiddenLayer(){}

    private:
    int features;               //< The size of the input.
    int neurons;                //< The size of the output.
    float * weights;             //< The weights of the layer.
    float * biases;              //< The biases of the layer.
    float * weights_gradient;    //< The gradients of the weights.
    float * biases_gradient;     //< The gradients of the biases.
    float * input;               //< The input to the layer.
    float * output;              //< The output of the layer.
    float * input_gradient;      //< The gradients of the input.
    float * output_gradient;     //< The gradients of the output.
    int activation_function;    //< The activation function of the layer.
    int trained_times;          //< The number of times the layer has been trained.

    public:
    /**
     * @brief Constructs a CoalaMlpHiddenLayer object with the specified input and output sizes.
     * 
     * @param input_size The size of the input to the layer.
     * @param output_size The size of the output from the layer.
     */
    CoalaMlpHiddenLayer(int features, int neurons);
    int setInputSize(int input_size);
    int getInputSize();
    int setOutputSize(int output_size);
    int getOutputSize();
    int setActivation(COALA_MLP_ACTIVATION activation_rank);
    int getActivation();
    float * getWeights();
    float * getBiases();

    void initializeWeights(COALA_MLP_INITIALIZATION initialization_rank);

    void forward(float* input, int examples);
    
    float * getOutput();

    void backward(float* input, float* output);

    void update(float learning_rate);
};







class CoalaMlpOutputLayer
{
    protected:
    CoalaMlpOutputLayer(){}

    private:
    float * weights;             //< The weights of the layer.
    float * biases;              //< The biases of the layer.
    float * weights_gradient;    //< The gradients of the weights.
    float * biases_gradient;     //< The gradients of the biases.
    float * input;               //< The input to the layer.
    float * output_z;              //the output of the layer before activation
    float * output_y;              //< The output of the layer.
    float * dloss2dy;      //< The gradients of the input.
    float * dy2dz;     //< The gradients of the output.
    int features;           //< The size of the input.
    int neurons;          //< The size of the output.
    int activation_function;    //< The activation function of the layer.
    int trained_times;          //< The number of times the layer has been trained.
    
    public:
    /**
     * @brief Constructs a CoalaMlpOutputLayer object with the specified input and output sizes.
     * 
     * @param features The size of the input to the layer.
     * @param output_size The size of the output from the layer.
     */
    CoalaMlpOutputLayer(int features, int neurons);
    int setInputSize(int input_size);
    int getFeaturesNum();
    int setActivation(COALA_MLP_ACTIVATION activation_rank);
    int getActivation();
    float * getOutput();
    int getNeuronsNum();
    float * getWeights();
    float * getBiases();
    void initializeWeights(COALA_MLP_INITIALIZATION initialization_rank);

    /**
     * @brief Performs forward propagation for the layer.
     * 
     * @param input The input to the layer.
     * @param output The output of the layer.
     */
    void forward(float * input, int examples);
    
    /**
     * @brief Performs backward propagation for the layer.
     * 
     * @param input The input to the layer.
     * @param output The output of the layer.
     */
    void backward(float * real_mat, int examples, int real_dim);

    /**
     * @brief Updates the weights and biases of the layer using the specified learning rate.
     * 
     * @param learning_rate The learning rate for weight updates.
     */
    void update(float learning_rate);
};


#endif // COALA_MLP_LAYER_H