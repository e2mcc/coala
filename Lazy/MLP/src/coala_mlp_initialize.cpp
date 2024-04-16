//----------------------------------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------------------------------
#include "coala_mlp_initialize.h"
#include <cstdlib>
#include <cmath>
#include <string.h>

int coala_mlp_szero(float * vec, int size)
{
    memset(vec, 0, sizeof(float) * size);
    return 0;
}

int coala_mlp_dzero(double * vec, int size)
{
    memset(vec, 0, sizeof(double) * size);
    return 0;
}

int coala_mlp_sones(float * vec, int size)
{
    for(int i = 0; i < size; i++)
    {
        vec[i] = 1.0;
    }
    return 0;
}

int coala_mlp_dones(double * vec, int size)
{
    for(int i = 0; i < size; i++)
    {
        vec[i] = 1.0;
    }
    return 0;
}

int coala_mlp_srandom(float * vec, int size, int seed)
{
    srand(seed);

    for(int i = 0; i < size; i++)
    {
        vec[i] = (float)rand() / (float)RAND_MAX;
    }
    return 0;
}

int coala_mlp_drandom(double * vec, int size, int seed)
{
    srand(seed);
    
    for(int i = 0; i < size; i++)
    {
        vec[i] = (double)rand() / (double)RAND_MAX;
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