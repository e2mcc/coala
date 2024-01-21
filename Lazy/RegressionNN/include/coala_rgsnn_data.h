#ifndef _COALA_REGRESSION_NN_DATA_H
#define _COALA_REGRESSION_NN_DATA_H

/****************************************
 * Include
*****************************************/
#include "coala_rgsnn_structure.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char * poniterToPostfix(char * const string);
int isCorrectPostfix(char * const string, char * const postfix);
COALA_RGSNN_TRANING_DATA_t * loadDataFromCSV(char * const file_path);
int dataNormalize(COALA_RGSNN_TRANING_DATA_t * data);

int dataNormalize_Specific(COALA_RGSNN_TRANING_DATA_t * data, double const min, double const max);
#endif