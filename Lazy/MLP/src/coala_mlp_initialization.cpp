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


int coala_mlp_sxavier(float * mat, int rows, int cols, int seed)
{
   float limit = sqrt(6.0 / (rows + cols));

    // 初始化随机数生成器
    srand(seed);
    
    for (int i = 0; i < rows * cols; ++i) 
    {
        // 生成 [-limit, limit] 范围内的随机数
        float randNum = (float)rand() / RAND_MAX; // 转换为 [0, 1]
        mat[i] = randNum * 2 * limit - limit; // 转换为 [-limit, limit]
    }

    return 0;
}

int coala_mlp_dxavier(double * mat, int rows, int cols, int seed)
{
    double limit = sqrt(6.0 / (rows + cols));

    // 初始化随机数生成器
    srand(seed);
    
    for (int i = 0; i < rows * cols; ++i) 
    {
        // 生成 [-limit, limit] 范围内的随机数
        double randNum = (double)rand() / RAND_MAX; // 转换为 [0, 1]
        mat[i] = randNum * 2 * limit - limit; // 转换为 [-limit, limit]
    }

    return 0;
}

int coala_mlp_she(float * weights, int input_size, int seed)
{
    return 0;
}


int coala_mlp_dhe(double * weights, int input_size, int seed)
{
    return 0;
}