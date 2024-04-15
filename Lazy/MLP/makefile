CXX = g++

MLPDir = $(CURDIR)
I = -I$(MLPDir)/include

CFLAGS = -O2 -fopenmp

libmlp.a: activation.o graph.o initialization.o cost.o model.o node.o nodeop.o nodeva.o blas.o
	ar rcs $@ $^
	rm -rf $^

activation.o: $(MLPDir)/src/coala_mlp_activation.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@

blas.o: $(MLPDir)/src/coala_mlp_blas.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@

graph.o: $(MLPDir)/src/coala_mlp_graph.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@

initialization.o: $(MLPDir)/src/coala_mlp_initialization.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@

cost.o: $(MLPDir)/src/coala_mlp_cost.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@

model.o: $(MLPDir)/src/coala_mlp_model.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@

node.o: $(MLPDir)/src/coala_mlp_node.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@

nodeop.o: $(MLPDir)/src/coala_mlp_node_op.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@

nodeva.o: $(MLPDir)/src/coala_mlp_node_var.cpp
	$(CXX) $(I) $(CFLAGS) -c $< -o $@



.PHONY: clean
clean:
	rm -rf activation.o graph.o initialization.o cost.o model.o node.o nodeop.o nodeva.o blas.o