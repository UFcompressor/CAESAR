#include "gaeFusedKernels.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <math.h>

__global__ void caesarGaeSmokeKernel(int* out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = 1234;
    }
}

bool caesarGaeFusedKernelSmoke()
{
    int* d_out = nullptr;
    int h_out = 0;

    cudaError_t err = cudaMalloc(&d_out, sizeof(int));
    if (err != cudaSuccess) return false;

    caesarGaeSmokeKernel<<<1, 32>>>(d_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_out);
        return false;
    }

    err = cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    return err == cudaSuccess && h_out == 1234;
}

__global__ void buildFinalMaskKernel(
    const float* __restrict__ allCoeff,
    const int32_t* __restrict__ sortIndex,
    bool* __restrict__ finalMask,
    int64_t rows,
    int64_t cols,
    float quanBin,
    float errorBoundSq)
{
    int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) return;

    extern __shared__ unsigned char smem[];
    float* keepPower = reinterpret_cast<float*>(smem);

    float totalPower = 0.0f;

    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float coeff = allCoeff[row * cols + j];
        totalPower += coeff * coeff;
        finalMask[row * cols + j] = false;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        totalPower += __shfl_down_sync(0xffffffff, totalPower, offset);
    }

    __shared__ float rowTotalPower;
    if (threadIdx.x == 0) rowTotalPower = totalPower;
    __syncthreads();

    if (threadIdx.x == 0) {
        double remaining = static_cast<double>(rowTotalPower);
        int64_t forceIdx = 0;
        bool foundFalse = false;

        for (int64_t rank = 0; rank < cols; ++rank) {
            int64_t idx = static_cast<int64_t>(sortIndex[row * cols + rank]);
            double coeff = static_cast<double>(allCoeff[row * cols + idx]);
            double q = nearbyint(coeff / static_cast<double>(quanBin));
            double diff = coeff - q * static_cast<double>(quanBin);
            double retained = coeff * coeff - diff * diff;

            remaining -= retained;
            bool keep = remaining > static_cast<double>(errorBoundSq);
            if (!keep) {
                forceIdx = rank;
                foundFalse = true;
                break;
            }
        }

        int64_t lastRank = foundFalse ? forceIdx : (cols - 1);
        for (int64_t rank = 0; rank <= lastRank; ++rank) {
            int64_t idx = static_cast<int64_t>(sortIndex[row * cols + rank]);
            float coeff = allCoeff[row * cols + idx];
            float q = nearbyintf(coeff / quanBin);
            finalMask[row * cols + idx] = (q != 0.0f);
        }

        if (foundFalse) {
            int64_t forcedIdx = static_cast<int64_t>(sortIndex[row * cols + forceIdx]);
            float forcedCoeff = allCoeff[row * cols + forcedIdx];
            float forcedQ = nearbyintf(forcedCoeff / quanBin);
            finalMask[row * cols + forcedIdx] = (forcedQ != 0.0f);
        }
    }
}

bool caesarGaeBuildFinalMask(
    const float* allCoeff,
    const int32_t* sortIndex,
    bool* finalMask,
    int64_t rows,
    int64_t cols,
    double quanBin,
    double errorBound)
{
    if (!allCoeff || !sortIndex || !finalMask || rows <= 0 || cols <= 0) return false;

    dim3 block(32);
    dim3 grid(static_cast<unsigned>(rows));
    size_t shmem = static_cast<size_t>(cols) * sizeof(float);

    buildFinalMaskKernel<<<grid, block, shmem>>>(
        allCoeff,
        sortIndex,
        finalMask,
        rows,
        cols,
        static_cast<float>(quanBin),
        static_cast<float>(errorBound * errorBound));

    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess;
}
#endif
