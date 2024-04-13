CXX = g++

MLPDir = $(CURDIR)
I = -I$(MLPDir)/include

CFLAGS = -O2 -fopenmp

activation.o: $(MLPDir)/src/coala_mlp_activation.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@

blas.o: $(MLPDir)/src/coala_mlp_blas.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@

graph.o: $(MLPDir)/src/coala_mlp_graph.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@

initialization.o: $(MLPDir)/src/coala_mlp_initialization.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@

loss.o: $(MLPDir)/src/coala_mlp_loss.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@

model.o: $(MLPDir)/src/coala_mlp_model.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@

node.o: $(MLPDir)/src/coala_mlp_node.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@

libmlp.a: activation.o blas.o graph.o initialization.o loss.o model.o node.o
	ar rcs $@ $^
	rm -rf $^

.PHONY: clean
clean:
	rm -rf activation.o blas.o graph.o initialization.o loss.o model.o node.o