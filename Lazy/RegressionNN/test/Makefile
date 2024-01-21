CC = gcc


RressionNNDir = $(CURDIR)/..


xmain:./main.c
	$(CC) -I$(RressionNNDir)/include -O2 $< -o $@ $(RressionNNDir)/libcoalargsnn.a -lm -fopenmp


run:xmain
	./xmain 2
	mv ./predicting_data.csv predicting_data_2d.csv
	./xmain 3
	mv ./predicting_data.csv predicting_data_3d.csv
	./xmain 4
	mv ./predicting_data.csv predicting_data_4d.csv
	./xmain 5
	mv ./predicting_data.csv predicting_data_5d.csv
	./xmain 6
	mv ./predicting_data.csv predicting_data_6d.csv
	./xmain 7
	mv ./predicting_data.csv predicting_data_7d.csv
	./xmain 8
	mv ./predicting_data.csv predicting_data_8d.csv
	./xmain 9
	mv ./predicting_data.csv predicting_data_9d.csv