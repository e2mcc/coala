#include <stdio.h>
#include <string.h>
#include "coala_rgsnn_data.h"
#include "coala_rgsnn_hypothesis.h"
#include "coala_rgsnn_gradientdescent.h"
#include "coala_rgsnn_loss.h"
#include <sys/time.h>

//设置数据文件存放路径
#define TRAINING_DATA_CSV ("/home/cqy/Regression/test/train_data.csv")
#define TESTING_DATA_CSV ("/home/cqy/Regression/test/test_data.csv")
#define PREDICTING_DATA_CSV ("/home/cqy/Regression/test/predict_data.csv")

int main(int argc,char ** argv)
{
	char * source_file_path = TRAINING_DATA_CSV;
	double lr = 0.04;
    int iterations = 200000;
	int degree = atoi(argv[1]);

	//读取数据

	COALA_RGSNN_TRANING_DATA_t* train_data = loadDataFromCSV(source_file_path);
	
	dataNormalize(train_data);

	// for(int i=0;i<data->shape;i++)
	// {
	// 	printf("(%d/%d):\t%lf\t---\t%lf\n",i,data->shape,data->points[i].feature,data->points[i].target);
	// }

	COALA_RGSNN_HYPOTHESIS_t * f = defineHypothesis(degree);
	
	//开始计时
	struct timeval start, end;
	gettimeofday(&start,NULL);

	gradient_descent(train_data,f,lr,iterations);

	gettimeofday(&end,NULL);
	double total_time	=	(end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec)/1000000.0;
	
	free(train_data);

	//测试
	char * test_file_path = TESTING_DATA_CSV;
	COALA_RGSNN_TRANING_DATA_t* test_data = loadDataFromCSV(test_file_path);
	dataNormalize_Specific(test_data,3*100*100,3*10000*10000);

	char * predict_file_path = PREDICTING_DATA_CSV;
	FILE * fp = fopen(predict_file_path,"w");
	fprintf(fp,"total time,RS,ARS\n");
	fprintf(fp,"%lf,%lf,%lf\n",total_time,RSquare(test_data,f),AdjustedRSquare(test_data,f));
	fprintf(fp,"feature,real,predict\n");

	int const m =test_data->shape;

	for(int i=0;i<m;i++)
	{
		fprintf(fp,"%lf,%lf,%lf\n",test_data->points[i].feature,test_data->points[i].target,computeHypothesis(f,test_data->points[i].feature));
	}
	fclose(fp);



	//计算一下推理时间
	struct timeval start2, end2;
	gettimeofday(&start2,NULL);
	double temp=0.0;
	for(int i=0;i<m;i++)
	{
		temp = computeHypothesis(f,test_data->points[i].feature);
	}
	gettimeofday(&end2,NULL);
	double total_time_2	=	(end2.tv_sec-start2.tv_sec)+(end2.tv_usec-start2.tv_usec)/1000000.0;
	printf("推理总时间(degree=%d):\n%lf\n", degree,total_time_2);
	free(f);
	free(test_data);
	return 0;
}