CC  = gcc

RressionNNDir = $(CURDIR)
INC = -I$(RressionNNDir)/include

CFLAGS = -O2 -fopenmp


libcoalargsnn.a: data.o hypothesis.o loss.o gradientdescent.o
	ar rcs $@ $^
	rm -rf $^



data.o: ./src/coala_rgsnn_data.c
	$(CC) $(INC) $(CFLAGS) -c $< -o $@

hypothesis.o: ./src/coala_rgsnn_hypothesis.c
	$(CC) $(INC) $(CFLAGS) -c $< -o $@

loss.o: ./src/coala_rgsnn_loss.c
	$(CC) $(INC) $(CFLAGS) -c $< -o $@

gradientdescent.o: ./src/coala_rgsnn_gradientdescent.c
	$(CC) $(INC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -rf libcoalargsnn.a
