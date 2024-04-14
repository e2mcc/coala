#ifndef COALA_MLOP_ACTIVATION_H
#define COALA_MLOP_ACTIVATION_H


//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------------
// Enums
//----------------------------------------------------------------------------------------------

typedef enum
{
    COALA_MLP_ACTIVATION_NONE       = 0,
    COALA_MLP_ACTIVATION_SIGMOID    = 1,
    COALA_MLP_ACTIVATION_TANH       = 2,
    COALA_MLP_ACTIVATION_RELU       = 3,
    COALA_MLP_ACTIVATION_LEAKY_RELU = 4,
    COALA_MLP_ACTIVATION_SOFTMAX    = 5
}COALA_MLP_ACTIVATION;


//----------------------------------------------------------------------------------------------
// Functions
//----------------------------------------------------------------------------------------------
int coala_mlp_activation(COALA_MLP_ACTIVATION const activation_rank, float * output, float * input, int const rows, int const cols);



int coala_mlp_ssigmoid(float * output, float * input, int size);
int coala_mlp_dsigmoid(double * output, double * input, int size);
int coala_mlp_ssigmoid_gradient(float * output, float * input, int size);
int coala_mlp_dsigmoid_gradient(double * output, double * input, int size);



int coala_mlp_stanh(float * output, float * input, int size);
int coala_mlp_dtanh(double * output, double * input, int size);
int coala_mlp_stanh_gradient(float * output, float * input, int size);
int coala_mlp_dtanh_gradient(double * output, double * input, int size);


int coala_mlp_srelu(float * output, float * input, int size);
int coala_mlp_drelu(double * output, double * input, int size);
int coala_mlp_srelu_gradient(float * output, float * input, int size);
int coala_mlp_drelu_gradient(double * output, double * input, int size);


int coala_mlp_sleakyrelu(float * output, float * input, int size);
int coala_mlp_dleakyrelu(double * output, double * input, int size);
int coala_mlp_sleakyrelu_gradient(float * output, float * input, int size);
int coala_mlp_dleakyrelu_gradient(double * output, double * input, int size);


int coala_mlp_ssoftmax(float * output, float * input, int m, int n);
int coala_mlp_dsoftmax(double * output, double * input, int m, int n);
int coala_mlp_ssoftmax_gradient(float * output, float * input, int m, int n);
int coala_mlp_dsoftmax_gradient(double * output, double * input, int m, int n);

#endif // COALA_MLOP_ACTIVATION_H
