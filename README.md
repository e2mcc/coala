# COALA 
<img src="Doc/Image/The koala maintains its whimsic-DALL·E.png" alt="Alt text" width="400"/>

**COALA**: A **Co**mpiler-Assisted **A**daptive **L**ibrary Routines **A**llocation Framework for Heterogeneous Systems


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

Just in case some environments from being configured in time, here is a functional verification DEMO that can be used for convenient testing
```bash
cd Test && cd Demo
make xdemo
./xdemo 3 4 5
```

