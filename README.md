# COALA 
<img src="Doc/Image/The koala maintains its whimsic-DALL·E.png" alt="Alt text" width="400"/>

**COALA**: A **Co**mpiler-Assisted **A**daptive **L**ibrary Routines **A**llocation Framework for Heterogeneous Systems

DOI Bookmark: [10.1109/TC.2024.3385269](https://doi.ieeecomputersociety.org/10.1109/TC.2024.3385269)

## Requirements
### Platform

| CPU | GPU |
|-----|-----|
| Intel Xeon Gold 5120| Nvidia A100 |
| Intel Core i5-10400| AMD RX550 |


### OS
| OS | version |
|-----|-----|
| Ubuntu| 22.04 |


### Environment
| Environment | version |
|-----|-----|
| LLVM| 17.0.6 |
| GCC | >=12.0.0 |
| CMake | >=3.27.0 |
| CUDA | >=6.0 |
| OpenCL | >=3.0 |

### Library

| Library | version |
|-----|-----|
| OpenBlas|0.3.26 |
| Cublas | 12.3 |
| ClBlast | 1.6.1 |




## Installation

Before installing COALA, please make sure that the requirements are met.



COALA can be compiled in the usual CMake-way, e.g.:
```bash
mkdir build && cd build
cmake ..
make -j 4
```

## Testing
The test examples are in the 'Test' folder.

In the event that certain environments may not be ready in time, we have prepared a functional verification DEMO for convenient and easy testing.
```bash
cd Test && cd Demo
make xdemo
./xdemo 3 4 5
```

## MLP is being reconstructed
In order to make it more convenient for lab partners to custom-build lightweight neural network models that can be embedded into compilers, and to serve as basic learning material for neural network frameworks, the MLP subproject is being reconstructed with a focus on the computational graph concept.
