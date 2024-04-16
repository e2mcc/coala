#ifndef COALA_MLP_MODEL_H
#define COALA_MLP_MODEL_H

//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include <string>
#include <vector>
#include "coala_mlp_graph.h"
#include "coala_mlp_cost.h"
#include "coala_mlp_activate.h"
#include "coala_mlp_initialize.h"
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
    /// @brief 输入层神经元数(即输入特征数)
    int input_layer_neurons;

    /// @brief 隐藏层层数
    int hidden_layers_count;

    /// @brief 隐藏层神经元数
    std::vector<int> hidden_layers_neurons;

    /// @brief 输出层神经元数
    int output_layer_neurons;

    /// @brief 训练批次大小(即每次输入进行训练的样本数，每批次训练完后更新一次权重)
    int batch_size;

    /// @brief 损失函数
    COALA_MLP_COST cost_func;

    /// @brief 隐藏层激活函数
    std::vector<COALA_MLP_ACTIVATE_FUNC> hidden_layer_activate_funcs;

    /// @brief 输出层激活函数
    COALA_MLP_ACTIVATE_FUNC output_layer_activate_func;

    /// @brief 内部权重初始化函数
    COALA_MLP_INITIALIZE_FUNC initialize_func;

    /// @brief 计算图
    std::shared_ptr<CoalaMlpGraph> graph;


    public:
    CoalaMLP(int const input_layer_neurons, int const hidden_layers_count=1, int const output_layer_neurons=1);
    int setHiddenLayersNeurons(int const layer, int const neurons);
    int setTraningBatch(int const batch_size);
    int setCostFunc(COALA_MLP_COST const cost_func);
    int setInitializeFunc(COALA_MLP_INITIALIZE_FUNC const initialize_func);
    int setHiddenLayerActivateFunc(int const hidden_layer, COALA_MLP_ACTIVATE_FUNC const activate_func);
    int setOutputLayerActivateFunc(COALA_MLP_ACTIVATE_FUNC const activate_func);
    int readyForTraining(void);


};


}//end of namespace mlp
}//end of namespace coala

#endif
