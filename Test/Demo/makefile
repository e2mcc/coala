CLANG = clang
OPT = opt

COALA-TRPass = ../../build/lib/libcoala.so
COALA-PROBE = ../../build/lib/libcoalartprobes.a
COALA-MEMOP = ../../build/lib/libcoalartmemops.a
COALA-BLAS = ../../build/lib/libcoalablasl3.a

demo1.ll: ./demo.c
	$(CLANG) -I. $< -emit-llvm -S -c -o $@

demo2.ll: demo1.ll
	$(OPT) -load-pass-plugin=$(COALA-TRPass) -passes="coala-blas" $< -S -o $@

demo.o: demo2.ll
	$(CLANG) $< -c -o $@

gemm.o:./cublas_gemm.c
	$(CLANG) -I. -c -O2 $< -o $@

probe.o:./probe.c
	$(CLANG) -I. -c -O2 $< -o $@

xdemo: demo.o gemm.o
	$(CLANG) $^ -o $@ $(COALA-PROBE) $(COALA-MEMOP) $(COALA-BLAS)


.PHONY: clean
clean:
	rm -rf xdemo gemm.o demo.o  probe.o demo1.ll demo2.ll