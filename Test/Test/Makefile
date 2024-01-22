# Compiler
CLANG = clang
OPT = opt
GCC = gcc
NVCC = nvcc

OPENCL = -lOpenCL

# CFLAGS 
CFLAGS = -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC


# OpenCL headfile
ClBlastDir = 
ClBlastInc = $(ClBlastDir)/include
ClBlastLib = -Wl,-rpath,$(ClBlastDir)/build -L$(ClBlastDir)/build  $(ClBlastDir)/build/libclblast.so

# OpenBlas
OpenBlasDir = 
OpenBlasInc = $(OpenBlasDir)
OpenBlasLib = $(OpenBlasDir)/libopenblas.a


#cublas in A100 server
CuBlasDir = 
CuBlasInc = $(CuBlasDir)/include
CuBlasLib = $(CuBlasDir)/lib/libcublas_static.a

#coala
COALA-PASS = ../../build/lib/libcoala.so
COALA-PROBE = ../../build/lib/libcoalartprobes.a
COALA-MEMOP = ../../build/lib/libcoalartmemops.a
COALA-BLAS = ../../build/lib/libcoalablasl3.a
# ------------------------------------
# Original
# ------------------------------------
xsgemm_clblast_ori: ./clblast_sgemm.c
	$(GCC) -O2 -I$(ClBlastInc) $< -o $@ $(ClBlastLib) $(OPENCL)
xdgemm_clblast_ori: ./clblast_dgemm.c
	$(GCC) -O2 -I$(ClBlastInc) $< -o $@ $(ClBlastLib) $(OPENCL)


xsgemm_openblas_ori: ./openblas_sgemm.c
	$(GCC) -O2 -I$(OpenBlasInc) $< -o $@ $(OpenBlasLib) -lpthread
xdgemm_openblas_ori: ./openblas_dgemm.c
	$(GCC) -O2 -I$(OpenBlasInc) $< -o $@ $(OpenBlasLib) -lpthread

xsgemm_cublas_ori: ./cublas_sgemm.c
	$(NVCC) -gencode arch=compute_70,code=sm_70  -I$(CuBlasInc) $< -o $@ -lcublas
xdgemm_cublas_ori: ./cublas_dgemm.c
	$(NVCC) -gencode arch=compute_70,code=sm_70  -I$(CuBlasInc) $< -o $@ -lcublas

# ------------------------------------
# COALA
# ------------------------------------
sgemm_clblast_coala.o:./clblast_sgemm.c
	$(CLANG) -I. $< -emit-llvm -S -c -o xsgemm_clblast_coala1.ll
	$(OPT) -load-pass-plugin=$(COALA-PASS) -passes="coala-blas" xsgemm_clblast_coala1.ll -S -o xsgemm_clblast_coala2.ll
	$(CLANG) xsgemm_clblast_coala2.ll -c -o $@
	rm -rf xsgemm_clblast_coala1.ll xsgemm_clblast_coala2.ll

dgemm_clblast_coala.o:./clblast_dgemm.c
	$(CLANG) -I. $< -emit-llvm -S -c -o xdgemm_clblast_coala1.ll
	$(OPT) -load-pass-plugin=$(COALA-PASS) -passes="coala-blas" xdgemm_clblast_coala1.ll -S -o xdgemm_clblast_coala2.ll
	$(CLANG) xdgemm_clblast_coala2.ll -c -o $@
	rm -rf xdgemm_clblast_coala1.ll xdgemm_clblast_coala2.ll

sgemm_cublas_coala.o:./cublas_sgemm.c
	$(CLANG) -I. $< -emit-llvm -S -c -o xsgemm_cublas_coala1.ll
	$(OPT) -load-pass-plugin=$(COALA-PASS) -passes="coala-blas" xsgemm_cublas_coala1.ll -S -o xsgemm_cublas_coala2.ll
	$(CLANG) xsgemm_cublas_coala2.ll -c -o $@
	rm -rf xsgemm_cublas_coala1.ll xsgemm_cublas_coala2.ll

dgemm_cublas_coala.o:./cublas_dgemm.c
	$(CLANG) -I. $< -emit-llvm -S -c -o xdgemm_cublas_coala1.ll
	$(OPT) -load-pass-plugin=$(COALA-PASS) -passes="coala-blas" xdgemm_cublas_coala1.ll -S -o xdgemm_cublas_coala2.ll
	$(CLANG) xdgemm_cublas_coala2.ll -c -o $@
	rm -rf xdgemm_cublas_coala1.ll xdgemm_cublas_coala2.ll

sgemm_openblas_coala.o:./openblas_sgemm.c
	$(CLANG) -I. $< -emit-llvm -S -c -o xsgemm_openblas_coala1.ll
	$(OPT) -load-pass-plugin=$(COALA-PASS) -passes="coala-blas" xsgemm_openblas_coala1.ll -S -o xsgemm_openblas_coala2.ll
	$(CLANG) xsgemm_openblas_coala2.ll -c -o $@
	rm -rf xsgemm_openblas_coala1.ll xsgemm_openblas_coala2.ll

dgemm_openblas_coala.o:./openblas_dgemm.c
	$(CLANG) -I. $< -emit-llvm -S -c -o xdgemm_openblas_coala1.ll
	$(OPT) -load-pass-plugin=$(COALA-PASS) -passes="coala-blas" xdgemm_openblas_coala1.ll -S -o xdgemm_openblas_coala2.ll
	$(CLANG) xdgemm_openblas_coala2.ll -c -o $@
	rm -rf xdgemm_openblas_coala1.ll xdgemm_openblas_coala2.ll

# Link
xsgemm_clblast_coala: sgemm_clblast_coala.o
	$(CLANG) -O2 -I$(ClBlastInc) $< -o $@ $(COALA-PROBE) $(COALA-MEMOP) $(COALA-BLAS) $(ClBlastLib) $(OPENCL)
xdgemm_clblast_coala: dgemm_clblast_coala.o
	$(CLANG) -O2 -I$(ClBlastInc) $< -o $@ $(COALA-PROBE) $(COALA-MEMOP) $(COALA-BLAS) $(ClBlastLib) $(OPENCL)

xsgemm_cublas_coala: sgemm_cublas_coala.o
	$(NVCC) -gencode arch=compute_70,code=sm_70 -I$(CuBlasInc) $< -o $@ $(COALA-PROBE) $(COALA-MEMOP) $(COALA-BLAS) -lcublas
xdgemm_cublas_coala: dgemm_cublas_coala.o
	$(NVCC) -gencode arch=compute_70,code=sm_70 -I$(CuBlasInc) $< -o $@ $(COALA-PROBE) $(COALA-MEMOP) $(COALA-BLAS) -lcublas

xsgemm_openblas_coala: sgemm_openblas_coala.o
	$(CLANG) -O2 -I$(OpenBlasInc) $< -o $@ $(COALA-PROBE) $(COALA-MEMOP) $(COALA-BLAS) $(OpenBlasLib) -lpthread
xdgemm_openblas_coala: dgemm_openblas_coala.o
	$(CLANG) -O2 -I$(OpenBlasInc) $< -o $@ $(COALA-PROBE) $(COALA-MEMOP) $(COALA-BLAS) $(OpenBlasLib) -lpthread


# clean
.PHONY : clean
clean:
	rm -rf xsgemm_clblast_ori xdgemm_clblast_ori xsgemm_openblas_ori xdgemm_openblas_ori xsgemm_cublas_ori xdgemm_cublas_ori
	rm -rf xsgemm_clblast_coala xdgemm_clblast_coala xsgemm_cublas_coala xdgemm_cublas_coala xsgemm_openblas_coala xdgemm_openblas_coala
	rm -rf sgemm_clblast_coala.o dgemm_clblast_coala.o sgemm_cublas_coala.o dgemm_cublas_coala.o sgemm_openblas_coala.o dgemm_openblas_coala.o
