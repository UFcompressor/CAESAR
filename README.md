# CAESAR_C

This repository contains my work on converting [CAESAR](https://github.com/Shaw-git/CAESAR) into C++ code using **LibTorch**.  
The goal is to provide a C++ implementation of the CAESAR foundation model.  

---

## Build Instructions

```bash

mkdir build && cd build


cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..


make


../download_models.sh
