//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_initialization.h"
#include <cstdlib>
#include <cmath>


int coala_mlp_srandom(float * output, int size, int seed)
{
    srand(seed);

    for(int i = 0; i < size; i++)
    {
        output[i] = (float)rand() / (float)RAND_MAX;
    }
    return 0;
}

int coala_mlp_drandom(double * output, int size, int seed)
{
    srand(seed);
    
    for(int i = 0; i < size; i++)
    {
        output[i] = (double)rand() / (double)RAND_MAX;
    }
    return 0;
}


int coala_mlp_sxavier(float * weights, int input_size, int output_size, int seed)
{
    float limit = sqrt(6.0 / (input_size + output_size));

    // 初始化随机数生成器
    srand(seed);
    
    for (int i = 0; i < input_size * output_size; ++i) 
    {
        // 生成 [-limit, limit] 范围内的随机数
        float randNum = (float)rand() / RAND_MAX; // 转换为 [0, 1]
        weights[i] = randNum * 2 * limit - limit; // 转换为 [-limit, limit]
    }

    return 0;
}

int coala_mlp_dxavier(double * weights, int input_size, int output_size, int seed)
{
    double limit = sqrt(6.0 / (input_size + output_size));

    // 初始化随机数生成器
    srand(seed);
    
    for (int i = 0; i < input_size * output_size; ++i) 
    {
        // 生成 [-limit, limit] 范围内的随机数
        double randNum = (double)rand() / RAND_MAX; // 转换为 [0, 1]
        weights[i] = randNum * 2 * limit - limit; // 转换为 [-limit, limit]
    }

    return 0;
}

int coala_mlp_she(float * weights, int input_size, int seed)
{
    // 初始化随机数生成器
    srand(seed);

    float stddev = sqrt(2.0 / input_size);

    for (int i = 0; i < input_size; ++i) {
        float randNum = (float)rand() / RAND_MAX; // 转换为 [0, 1]
        randNum = randNum * 2 - 1; // 转换为 [-1, 1]
        weights[i] = randNum * stddev;
    }

    return 0;
}


int coala_mlp_dhe(double * weights, int input_size, int seed)
{
    // 初始化随机数生成器
    srand(seed);

    double stddev = sqrt(2.0 / input_size);

    for (int i = 0; i < input_size; ++i) {
        double randNum = (double)rand() / RAND_MAX; // 转换为 [0, 1]
        randNum = randNum * 2 - 1; // 转换为 [-1, 1]
        weights[i] = randNum * stddev;
    }

    return 0;
}