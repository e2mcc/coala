#include "coala_rgsnn_data.h"


char * poniterToPostfix(char * const string)
{
	if(string==NULL) 
	{
		fprintf(stderr,"WRONG! 函数 %s: 输入参数为非发输入\n",__FUNCTION__);
		exit(0);
	}
	//找到最后一次“.”出现的位置
	char *  dot = strrchr(string, '.');
	//如果没找到
	if(!dot||dot == string)
	{
		return NULL;
	}
	else
	{
		
		dot++;
	}
	return dot;
}


int isCorrectPostfix(char * const string, char * const postfix)
{
	if(string==NULL||postfix==NULL)
	{
		fprintf(stderr,"WRONG! 函数 %s: 输入参数为非发输入\n",__FUNCTION__);
		exit(0);
	}

	if(strcmp(poniterToPostfix(string), postfix)!=0) return 1;
	return 0;
}



COALA_RGSNN_TRANING_DATA_t * loadDataFromCSV(char * const file_path)
{	
	//--------------------------------
	//输入检测
	//--------------------------------
	if( file_path == NULL )
	{
		fprintf(stderr,"WRONG! 函数 %s: 输入参数为非发输入\n",__FUNCTION__);
		exit(0);
	}
	
	if(isCorrectPostfix(file_path,"csv")!=0)
	{
		fprintf(stderr,"WRONG! 函数 %s: 输入文件不是csv格式的文件(不以csv为后缀名)\n",__FUNCTION__);
		exit(0);
	}

	//--------------------------------
	// 读取csv数据
	//--------------------------------
	//以只读模式打开
	FILE * fp1 = fopen(file_path,"r");
	if( fp1 == NULL ) return NULL;

	unsigned int line_count = 0;
	char c;

	//获取行总数
    while ((c = fgetc(fp1)) != EOF) 
	{
        if (c == '\n') 
		{
            line_count++;
        }
    }

	// printf("行总数:%d\n",line_count);

	fclose(fp1);
	FILE * fp = fopen(file_path,"r");

	COALA_RGSNN_TRANING_DATA_t * data = malloc(sizeof(*data));
	data->shape = line_count;
	data->points = malloc(data->shape*sizeof(*data->points));


	char line[1024];
	char * token;
	int i = 0;
	while(fgets(line, sizeof(line), fp))
	{
		//删除行尾的换行符（如果存在）
		line[strcspn(line, "\n")] = 0;
		//使用逗号作为分隔符标记行
		token = strtok(line, ",");
		// printf("token = %s\n",token);
		if(token != NULL)
		{
			data->points[i].feature = strtold(token,NULL);
			// data->points[i].feature /= 1000;
		}
		token = strtok(NULL, ",");
		// printf("token = %s\n",token);
		if(token != NULL)
		{
			data->points[i].target = strtold(token,NULL);
		}
		i++;
	}

	fclose(fp);
	
	return data;

}

int dataNormalize(COALA_RGSNN_TRANING_DATA_t * data)
{
	unsigned int m = data->shape;
	//找到min和max
	double min = data->points[0].feature;
	double max = data->points[0].feature;
	for(int i=0;i<m;i++)
	{
		if(data->points[i].feature<min)
		{
			min = data->points[i].feature;
		}
		if(data->points[i].feature>max)
		{
			max = data->points[i].feature;
		}
	}
	
	for(int i=0;i<m;i++)
	{
		data->points[i].feature = (data->points[i].feature-min)/(max-min);
	}
	printf("min=%lf,max=%lf\n",min,max);
	return 0;

}

int dataNormalize_Specific(COALA_RGSNN_TRANING_DATA_t * data, double const min, double const max)
{
	unsigned int m = data->shape;
	for(int i=0;i<m;i++)
	{
		data->points[i].feature = (data->points[i].feature-min)/(max-min);
	}
	return 0;
}