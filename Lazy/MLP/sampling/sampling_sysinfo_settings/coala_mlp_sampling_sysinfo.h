#ifndef COALA_MLP_SAMPLING_SYSINFO_H
#define COALA_MLP_SAMPLING_SYSINFO_H

#ifdef INTEL_CORE_I5_10400
#define CPU_SOCKETS (1f)
#define CPU_CORES   (6f)
#define CPU_FREQ    (2.9f) // (*GHz)
#define CPU_FP32_PER_CYCLE (32f)
#define CPU_FP64_PER_CYCLE (16f)
#define CPU_FP32_PEAK    (CPU_SOCKETS * CPU_CORES * CPU_FREQ * CPU_FP32_PER_CYCLE) // (GFLOPS)
#define CPU_FP64_PEAK    (CPU_SOCKETS * CPU_CORES * CPU_FREQ * CPU_FP64_PER_CYCLE) // (GFLOPS)
#elif INTEL_XEON_GOLD_5120
#define CPU_SOCKETS (2f)
#define CPU_CORES   (14f)
#define CPU_FREQ    (2.2f) // (*GHz)
#define CPU_FP32_PER_CYCLE (32f)
#define CPU_FP64_PER_CYCLE (16f)
#define CPU_FP32_PEAK    (CPU_SOCKETS * CPU_CORES * CPU_FREQ * CPU_FP32_PER_CYCLE) // (GFLOPS)
#define CPU_FP64_PEAK    (CPU_SOCKETS * CPU_CORES * CPU_FREQ * CPU_FP64_PER_CYCLE) // (GFLOPS)
#else
#define CPU_FP32_PEAK    (1000f) // (GFLOPS)
#define CPU_FP64_PEAK    (500f) // (GFLOPS)
#endif


#ifdef AMD_RADEON_RX550
#define GPU_F32_PEAK    (1211f) // (GPLOPS)
#define GPU_F64_PEAK    (75.71f) // (GPLOPS)
#elif NVIDIA_A100
#define GPU_F32_PEAK    (19500f) // (GPLOPS)
#define GPU_F64_PEAK    (9700f) // (GPLOPS)     
#else
#define GPU_F32_PEAK    (10000f) // (GPLOPS)
#define GPU_F64_PEAK    (5000f) // (GPLOPS)
#endif

#endif