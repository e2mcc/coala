GCC = /usr/bin/gcc
CC = $(GCC)

SamplingDir = $(CURDIR)/../..
SysInfoInc = $(SamplingDir)/sampling_sysinfo_settings

# OpenBlas
# OpenBlasDir = /home/user/lsl/OpenBLAS-develop
OpenBlasDir = /data/Sources/OpenBLAS-0.3.21
OpenBlasInc = $(OpenBlasDir)/lapack-netlib/CBLAS/include/
OpenBlasLib = $(OpenBlasDir)/build/lib/libopenblas.a

xtrain: ./openblas_gemm_trainingdata.c
	$(CC) -O2 -I$(SysInfoInc) -I$(OpenBlasInc) $< -o $@ $(OpenBlasLib) -lpthread

xtest: ./openblas_gemm_testdata.c
	$(CC) -O2 -I$(SysInfoInc) -I$(OpenBlasInc) $< -o $@ $(OpenBlasLib) -lpthread