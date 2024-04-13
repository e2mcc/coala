#include "coala_mlp_node.h"
#include "coala_mlp_blas.h"
#include "coala_mlp_loss.h"



using namespace coala::mlp;

//-----------------------------------------------------------------------
// Node
//-----------------------------------------------------------------------
int Node::addForwardNode(std::shared_ptr<Node> node)
{
    this->forward_nodes.push_back(node);
    return 0;
}

int  Node::addBackwardNode(std::shared_ptr<Node> node)
{
    this->backward_nodes.push_back(node);
    return 0;
}

//-----------------------------------------------------------------------
// Variable
//-----------------------------------------------------------------------
Variable::Variable(int const id) 
{
    this->id = id;
}


int Variable::setDataSize(int const rows, int const cols)
{
    this->data.rows = rows;
    this->data.cols = cols;
}


int Variable::setInitializationFunction(COALA_MLP_INITIALIZATION const initialization_func)
{
    this->initialization_func = initialization_func;
    return 0;
}

int Variable::getDataRows()
{
    return this->data.rows;
}

int Variable::getDataCols()
{
    return this->data.cols;
}

float * Variable::getMetaData()
{
    return this->data.data;
}

void Variable::activating(void)
{
    this->data.data = (float*)malloc(this->data.rows*this->data.cols*sizeof(float));
    switch(this->initialization_func)
    {
        case COALA_MLP_INITIALIZATION_NONE:
            break;
        case COALA_MLP_INITIALIZATION_ZERO:
            coala_mlp_szero(this->data.data, this->data.rows*this->data.cols);
            break;
        case COALA_MLP_INITIALIZATION_ONES:
            coala_mlp_sones(this->data.data, this->data.rows*this->data.cols);
            break;
        case COALA_MLP_INITIALIZATION_RANDOM:
            coala_mlp_srandom(this->data.data, this->data.rows*this->data.cols, 0);
            break;
        case COALA_MLP_INITIALIZATION_XAVIER:
            coala_mlp_sxavier(this->data.data, this->data.rows, this->data.cols, 0);
            break;
        default:
            break;
    }
}


//-----------------------------------------------------------------------
// OperatorAdd
//-----------------------------------------------------------------------
void OperatorAdd::activating(void)
{
    this->forward_input_nodes.push_back(this->backward_nodes[0]);
    this->forward_input_nodes.push_back(this->backward_nodes[1]);
    this->forward_output_nodes.push_back(this->forward_nodes[0]);
}

void OperatorAdd::forward(void)
{
    Variable * vnodeA = dynamic_cast<Variable*>(this->forward_input_nodes[0].get());
    Variable * vnodeB = dynamic_cast<Variable*>(this->forward_input_nodes[1].get());
    Variable * vnodeC = dynamic_cast<Variable*>(this->forward_output_nodes[0].get());
    coala_mlp_saxpy(vnodeC->getDataRows()*vnodeC->getDataCols(), 1.0, vnodeA->getMetaData(), 1, vnodeC->getMetaData(), 1);
    coala_mlp_saxpy(vnodeC->getDataRows()*vnodeC->getDataCols(), 1.0, vnodeB->getMetaData(), 1, vnodeC->getMetaData(), 1);
}

//-----------------------------------------------------------------------
// OperatorDot
//-----------------------------------------------------------------------
void OperatorDot::activating(void)
{
    this->forward_input_nodes.push_back(this->backward_nodes[0]);
    this->forward_input_nodes.push_back(this->backward_nodes[1]);
    this->forward_output_nodes.push_back(this->forward_nodes[0]);
}

void OperatorDot::forward(void)
{
    Variable * vnodeA = dynamic_cast<Variable*>(this->forward_input_nodes[0].get());
    Variable * vnodeB = dynamic_cast<Variable*>(this->forward_input_nodes[1].get());
    Variable * vnodeC = dynamic_cast<Variable*>(this->forward_output_nodes[0].get());
    
    coala_mlp_sgemm(1,0,0, 
        vnodeA->getDataRows(), vnodeB->getDataCols(), vnodeA->getDataCols(), 
        1.0f, vnodeA->getMetaData(), vnodeA->getDataRows(), 
        vnodeB->getMetaData(), vnodeB->getDataRows(),
        0.0f, vnodeC->getMetaData(), vnodeC->getDataRows());
}

//-----------------------------------------------------------------------
// OperatorActivation
//-----------------------------------------------------------------------
void OperatorActivation::setActivationFunc(COALA_MLP_ACTIVATION const activation_rank)
{
    this->activation_func = activation_rank;
}

void OperatorActivation::activating(void)
{
    this->forward_input_nodes.push_back(this->backward_nodes[0]);
    this->forward_output_nodes.push_back(this->forward_nodes[0]);
}

void OperatorActivation::forward(void)
{
    Variable * vnodeA = dynamic_cast<Variable*>(this->forward_input_nodes[0].get());
    Variable * vnodeB = dynamic_cast<Variable*>(this->forward_output_nodes[0].get());
    switch(this->activation_func)
    {
        case COALA_MLP_ACTIVATION_NONE:
            coala_mlp_scopy(vnodeA->getDataRows()*vnodeA->getDataCols(), vnodeA->getMetaData(), 1, vnodeB->getMetaData(), 1);
            break;
        
        case COALA_MLP_ACTIVATION_SIGMOID:
            coala_mlp_ssigmoid(vnodeB->getMetaData(), vnodeA->getMetaData(), vnodeA->getDataRows()*vnodeA->getDataCols());
            break;

        case COALA_MLP_ACTIVATION_TANH:
            coala_mlp_stanh(vnodeB->getMetaData(), vnodeA->getMetaData(), vnodeA->getDataRows()*vnodeA->getDataCols());
            break;

        case COALA_MLP_ACTIVATION_RELU:
            coala_mlp_srelu(vnodeB->getMetaData(), vnodeA->getMetaData(), vnodeA->getDataRows()*vnodeA->getDataCols());
            break;

        case COALA_MLP_ACTIVATION_LEAKY_RELU:
            coala_mlp_sleakyrelu(vnodeB->getMetaData(), vnodeA->getMetaData(), vnodeA->getDataRows()*vnodeA->getDataCols());
            break;

        case COALA_MLP_ACTIVATION_SOFTMAX:
            coala_mlp_ssoftmax(vnodeB->getMetaData(), vnodeA->getMetaData(), vnodeA->getDataRows(), vnodeA->getDataCols());
            break;
        default:
            return;
    }
}

//-----------------------------------------------------------------------
// OperatorCost
//-----------------------------------------------------------------------
void OperatorCost::setCostFunc(COALA_MLP_LOSS const cost_rank)
{
    this->cost_func = cost_rank;
}


void OperatorCost::activating(void)
{
    this->forward_input_nodes.push_back(this->backward_nodes[0]);
    this->forward_input_nodes.push_back(this->backward_nodes[1]);
    this->forward_output_nodes.push_back(this->forward_nodes[0]);
}

void OperatorCost::forward(void)
{
    Variable * vnodeA = dynamic_cast<Variable*>(this->forward_input_nodes[0].get());
    Variable * vnodeB = dynamic_cast<Variable*>(this->forward_input_nodes[1].get());
    Variable * vnodeC = dynamic_cast<Variable*>(this->forward_output_nodes[0].get());
    (vnodeC->getMetaData())[0] = (this->cost_func, vnodeA->getMetaData(), vnodeB->getMetaData(), vnodeA->getDataRows(), vnodeA->getDataCols());
}