#ifndef COALA_MLOP_ACTIVATE_H
#define COALA_MLOP_ACTIVATE_H


//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------------
// Enums
//----------------------------------------------------------------------------------------------

typedef enum
{
    COALA_MLP_ACTIVATE_NONE       = 0,
    COALA_MLP_ACTIVATE_SIGMOID    = 1,
    COALA_MLP_ACTIVATE_TANH       = 2,
    COALA_MLP_ACTIVATE_RELU       = 3,
    COALA_MLP_ACTIVATE_LEAKY_RELU = 4,
    COALA_MLP_ACTIVATE_SOFTMAX    = 5
}COALA_MLP_ACTIVATE_FUNC;


//----------------------------------------------------------------------------------------------
// Functions
//----------------------------------------------------------------------------------------------
int coala_mlp_activation(COALA_MLP_ACTIVATE_FUNC const activation_rank, float * output, float * input, int const rows, int const cols);



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

#endif // COALA_MLOP_ACTIVATE_H
