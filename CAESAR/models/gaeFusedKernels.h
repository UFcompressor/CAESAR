#pragma once

#ifdef USE_CUDA
#include <cstdint>

bool caesarGaeFusedKernelSmoke();

bool caesarGaeBuildFinalMask(
    const float* allCoeff,
    const int32_t* sortIndex,
    bool* finalMask,
    int64_t rows,
    int64_t cols,
    double quanBin,
    double errorBound);

#else
inline bool caesarGaeFusedKernelSmoke() { return false; }
#endif
