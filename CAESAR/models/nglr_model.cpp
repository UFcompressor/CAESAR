#include "nglr_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <zstd.h>

#if defined(USE_CUDA) && !defined(USE_ROCM) && !defined(__HIP_PLATFORM_AMD__)
#include <cuda_runtime.h>
#ifdef ENABLE_NVCOMP
#include <nvcomp/zstd.h>
#endif
#endif

#if defined(USE_CUDA) && defined(ENABLE_NVCOMP) && !defined(USE_ROCM) && !defined(__HIP_PLATFORM_AMD__)
#define NGLR_CHECK_CUDA(cmd) do {                                      \
    cudaError_t e = (cmd);                                             \
    if (e != cudaSuccess) {                                            \
        throw std::runtime_error(std::string("CUDA error in " #cmd ": ") + \
                                 cudaGetErrorString(e));               \
    }                                                                  \
} while (0)

#define NGLR_CHECK_NVCOMP(cmd) do {                                    \
    nvcompStatus_t s = (cmd);                                          \
    if (s != nvcompSuccess) {                                          \
        throw std::runtime_error("nvCOMP error in " #cmd);             \
    }                                                                  \
} while (0)
#endif


namespace caesar::nglr {

torch::Tensor recons_features(const torch::Tensor& x) {
    torch::Tensor dt = torch::zeros_like(x);
    torch::Tensor dh = torch::zeros_like(x);
    torch::Tensor dw = torch::zeros_like(x);

    dt.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(),
         torch::indexing::Slice(1, torch::indexing::None),
         torch::indexing::Slice(), torch::indexing::Slice()},
        x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                 torch::indexing::Slice(1, torch::indexing::None),
                 torch::indexing::Slice(), torch::indexing::Slice()}) -
        x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                 torch::indexing::Slice(0, -1),
                 torch::indexing::Slice(), torch::indexing::Slice()})
    );

    dh.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(),
         torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None),
         torch::indexing::Slice()},
        x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                 torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None),
                 torch::indexing::Slice()}) -
        x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                 torch::indexing::Slice(), torch::indexing::Slice(0, -1),
                 torch::indexing::Slice()})
    );

    dw.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(),
         torch::indexing::Slice(), torch::indexing::Slice(),
         torch::indexing::Slice(1, torch::indexing::None)},
        x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                 torch::indexing::Slice(), torch::indexing::Slice(),
                 torch::indexing::Slice(1, torch::indexing::None)}) -
        x.index({torch::indexing::Slice(), torch::indexing::Slice(),
                 torch::indexing::Slice(), torch::indexing::Slice(),
                 torch::indexing::Slice(0, -1)})
    );

    return torch::cat(
        {x, dt, dh, dw, dt.abs(), dh.abs(), dw.abs()},
        1
    );
}

ResBlock3DImpl::ResBlock3DImpl(int64_t ch) {
    c1 = register_module(
        "c1",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(ch, ch, 3).padding(1)
        )
    );

    c2 = register_module(
        "c2",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(ch, ch, 3).padding(1)
        )
    );

    const int64_t groups = std::min<int64_t>(4, ch);

    n1 = register_module(
        "n1",
        torch::nn::GroupNorm(
            torch::nn::GroupNormOptions(groups, ch)
        )
    );

    n2 = register_module(
        "n2",
        torch::nn::GroupNorm(
            torch::nn::GroupNormOptions(groups, ch)
        )
    );
}

torch::Tensor ResBlock3DImpl::forward(const torch::Tensor& x) {
    torch::Tensor y =
        torch::gelu(n1(c1(x)));

    y = n2(c2(y));

    return torch::gelu(x + y);
}

CausalNeuralLorenzoNetImpl::CausalNeuralLorenzoNetImpl(
    int64_t hidden,
    int64_t q_hidden,
    int64_t blocks
) {
    const int64_t groups = std::min<int64_t>(4, hidden);

    recons_in = register_module(
        "recons_in",
        torch::nn::Sequential(
            torch::nn::Conv3d(
                torch::nn::Conv3dOptions(7, hidden, 3).padding(1)
            ),
            torch::nn::GroupNorm(
                torch::nn::GroupNormOptions(groups, hidden)
            ),
            torch::nn::GELU()
        )
    );

    recons_blocks = register_module(
        "recons_blocks",
        torch::nn::Sequential()
    );

    for (int64_t i = 0; i < blocks; ++i) {
        recons_blocks->push_back(
            register_module(
                "recons_block_" + std::to_string(i),
                ResBlock3D(hidden)
            )
        );
    }

    q_branch = register_module(
        "q_branch",
        torch::nn::Sequential(
            torch::nn::Conv3d(
                torch::nn::Conv3dOptions(8, q_hidden, 1)
            ),
            torch::nn::GELU(),
            torch::nn::Conv3d(
                torch::nn::Conv3dOptions(q_hidden, q_hidden, 1)
            ),
            torch::nn::GELU()
        )
    );

    fusion = register_module(
        "fusion",
        torch::nn::Sequential(
            torch::nn::Conv3d(
                torch::nn::Conv3dOptions(hidden + q_hidden, hidden, 1)
            ),
            torch::nn::GELU(),
            torch::nn::Conv3d(
                torch::nn::Conv3dOptions(hidden, hidden, 1)
            ),
            torch::nn::GELU(),
            torch::nn::Conv3d(
                torch::nn::Conv3dOptions(hidden, 1, 1)
            )
        )
    );

    {
        torch::NoGradGuard no_grad;
        auto& final_conv = fusion->at<torch::nn::Conv3dImpl>(4);
        final_conv.weight.zero_();
        if (final_conv.bias.defined()) {
            final_conv.bias.zero_();
        }
    }
}

torch::Tensor CausalNeuralLorenzoNetImpl::encode_recons(
    const torch::Tensor& r
) {
    return recons_blocks->forward(
        recons_in->forward(recons_features(r))
    );
}

torch::Tensor CausalNeuralLorenzoNetImpl::forward_from_feature(
    const torch::Tensor& rf,
    const torch::Tensor& qctx
) {
    torch::Tensor qf =
        q_branch->forward(qctx);

    return fusion->forward(
        torch::cat({rf, qf}, 1)
    );
}

torch::Tensor CausalNeuralLorenzoNetImpl::forward(
    const torch::Tensor& r,
    const torch::Tensor& qctx
) {
    return forward_from_feature(
        encode_recons(r),
        qctx
    );
}

namespace {

int get_env_int_or_default(const char* name, int default_value);

torch::Device resolve_nglr_device(torch::Device requested) {
    if (requested.is_cpu()) {
        return torch::Device(torch::kCPU);
    }

#if defined(USE_CUDA)
    if (requested.is_cuda()) {
        return requested;
    }

    if (torch::cuda::is_available()) {
        int device_index = get_env_int_or_default("CAESAR_NGLR_CUDA_DEVICE", 0);
        return torch::Device(torch::kCUDA, device_index);
    }
#endif

    // Follow GAE's device policy: allow non-CUDA Torch backends such as Apple MPS
    // for model/tensor work, but keep the byte-stream codec off nvCOMP unless the
    // active device is CUDA.
    return requested;
}

int get_env_int_or_default(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || std::string(value).empty()) {
        return default_value;
    }
    return std::stoi(value);
}

torch::Tensor reference_from_pred_bias(
    const torch::Tensor& pred,
    const torch::Tensor& bias_norm,
    double delta_scale,
    torch::Device device
) {
    if (device.type() == torch::kMPS) {
        return torch::round(
            pred.to(torch::kFloat32) +
            bias_norm.to(torch::kFloat32) * static_cast<float>(delta_scale)
        ).to(torch::kInt32);
    }

    return torch::round(
        pred.to(torch::kFloat64) +
        bias_norm.to(torch::kFloat64) * delta_scale
    ).to(torch::kInt32);
}

constexpr uint32_t kNGLRModelTensorMagic = 0x57544D4E; // NMTW
constexpr uint32_t kNGLRModelTensorVersion = 1;

std::string get_nglr_model_path() {
    const char* value = std::getenv("CAESAR_NGLR_MODEL_PATH");
    if (value == nullptr || std::string(value).empty()) {
        return "nglr_model.pt";
    }
    return std::string(value);
}

std::vector<NGLRModelTensor> export_model_tensors(CausalNeuralLorenzoNet model) {
    std::vector<NGLRModelTensor> tensors;

    torch::NoGradGuard no_grad;
    for (const auto& item : model->named_parameters(/*recurse=*/true)) {
        torch::Tensor cpu =
            item.value()
                .detach()
                .to(torch::kCPU)
                .to(torch::kFloat32)
                .contiguous();

        NGLRModelTensor tensor;
        tensor.name = item.key();
        tensor.shape.assign(cpu.sizes().begin(), cpu.sizes().end());
        tensor.values.resize(static_cast<size_t>(cpu.numel()));
        std::memcpy(
            tensor.values.data(),
            cpu.data_ptr<float>(),
            tensor.values.size() * sizeof(float)
        );

        tensors.push_back(std::move(tensor));
    }

    return tensors;
}

void load_model_tensors(
    CausalNeuralLorenzoNet model,
    const std::vector<NGLRModelTensor>& tensors
) {
    torch::NoGradGuard no_grad;
    auto params = model->named_parameters(/*recurse=*/true);

    for (const auto& tensor : tensors) {
        bool found = false;

        for (auto& param : params) {
            if (param.key() != tensor.name) {
                continue;
            }

            torch::Tensor value =
                torch::from_blob(
                    const_cast<float*>(tensor.values.data()),
                    tensor.shape,
                    torch::TensorOptions().dtype(torch::kFloat32)
                ).clone();

            param.value().copy_(
                value.to(param.value().device(), param.value().scalar_type())
            );
            found = true;
            break;
        }

        if (!found) {
            throw std::runtime_error(
                "NGLR metadata contains unknown model tensor: " + tensor.name
            );
        }
    }
}

CausalNeuralLorenzoNet load_nglr_model(
    const NGLRMetaData& meta,
    torch::Device device
) {
    CausalNeuralLorenzoNet model =
        CausalNeuralLorenzoNet(
            meta.hidden,
            meta.q_hidden,
            meta.model_blocks
        );

    if (!meta.model_tensors.empty()) {
        load_model_tensors(model, meta.model_tensors);
    } else {
        torch::load(model, get_nglr_model_path());
    }

    model->to(device);
    model->eval();
    return model;
}

struct BlockSlice {
    int64_t b = 0;
    int64_t c = 0;
    int64_t t0 = 0;
    int64_t t1 = 0;
    int64_t h0 = 0;
    int64_t h1 = 0;
    int64_t w0 = 0;
    int64_t w1 = 0;
};

struct EncodedDeltaBlock {
    int bit_count = 0;
    int64_t T = 0;
    int64_t H = 0;
    int64_t W = 0;
    std::vector<std::vector<uint8_t>> streams;
};

std::vector<BlockSlice> iter_block_slices(
    const std::vector<int64_t>& shape,
    int64_t block_t,
    int64_t block_h,
    int64_t block_w
) {
    if (shape.size() != 5) {
        throw std::runtime_error("NGLR expects 5D tensor shape [B,C,T,H,W]");
    }

    std::vector<BlockSlice> slices;

    const int64_t B = shape[0];
    const int64_t C = shape[1];
    const int64_t T = shape[2];
    const int64_t H = shape[3];
    const int64_t W = shape[4];

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t t0 = 0; t0 < T; t0 += block_t) {
                int64_t t1 = std::min(t0 + block_t, T);

                for (int64_t h0 = 0; h0 < H; h0 += block_h) {
                    int64_t h1 = std::min(h0 + block_h, H);

                    for (int64_t w0 = 0; w0 < W; w0 += block_w) {
                        int64_t w1 = std::min(w0 + block_w, W);

                        slices.push_back({b, c, t0, t1, h0, h1, w0, w1});
                    }
                }
            }
        }
    }

    return slices;
}

torch::Tensor slice_5d_to_3d(const torch::Tensor& x, const BlockSlice& sl) {
    return x.index({
        sl.b,
        sl.c,
        torch::indexing::Slice(sl.t0, sl.t1),
        torch::indexing::Slice(sl.h0, sl.h1),
        torch::indexing::Slice(sl.w0, sl.w1)
    }).contiguous();
}

std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>>
diagonal_indices(int64_t T, int64_t H, int64_t W) {
    std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>> out;

    for (int64_t s = 0; s <= T + H + W - 3; ++s) {
        std::vector<int64_t> ts;
        std::vector<int64_t> hs;
        std::vector<int64_t> ws;

        int64_t t_min = std::max<int64_t>(0, s - (H - 1) - (W - 1));
        int64_t t_max = std::min<int64_t>(T - 1, s);

        for (int64_t t = t_min; t <= t_max; ++t) {
            int64_t h_min = std::max<int64_t>(0, s - t - (W - 1));
            int64_t h_max = std::min<int64_t>(H - 1, s - t);

            for (int64_t h = h_min; h <= h_max; ++h) {
                int64_t w = s - t - h;

                ts.push_back(t);
                hs.push_back(h);
                ws.push_back(w);
            }
        }

        if (!ts.empty()) {
            out.emplace_back(std::move(ts), std::move(hs), std::move(ws));
        }
    }

    return out;
}

std::vector<uint32_t> zigzag_encode_int32_vector(const torch::Tensor& delta) {
    torch::Tensor d =
        delta.to(torch::kCPU).to(torch::kInt32).contiguous();

    auto acc = d.accessor<int32_t, 3>();

    const int64_t T = d.size(0);
    const int64_t H = d.size(1);
    const int64_t W = d.size(2);

    std::vector<uint32_t> out;
    out.reserve(static_cast<size_t>(T * H * W));

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                int64_t v = acc[t][h][w];

                uint64_t zz =
                    (v >= 0)
                        ? static_cast<uint64_t>(v * 2)
                        : static_cast<uint64_t>(-2 * v - 1);

                out.push_back(static_cast<uint32_t>(zz));
            }
        }
    }

    return out;
}

torch::Tensor zigzag_decode_uint32_vector_to_tensor(
    const std::vector<uint32_t>& zz,
    int64_t T,
    int64_t H,
    int64_t W
) {
    torch::Tensor out =
        torch::empty({T, H, W}, torch::kInt32);

    auto acc = out.accessor<int32_t, 3>();

    int64_t idx = 0;

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                uint32_t u = zz[static_cast<size_t>(idx++)];

                int32_t v =
                    (u & 1u) == 0
                        ? static_cast<int32_t>(u >> 1)
                        : -static_cast<int32_t>((u + 1u) >> 1);

                acc[t][h][w] = v;
            }
        }
    }

    return out;
}

std::vector<uint8_t> pack_bitplane_little(
    const std::vector<uint32_t>& flat,
    int bit
) {
    size_t n = flat.size();

    std::vector<uint8_t> packed((n + 7) / 8, 0);

    for (size_t i = 0; i < n; ++i) {
        if (((flat[i] >> bit) & 1u) != 0) {
            packed[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
        }
    }

    return packed;
}

std::vector<uint32_t> unpack_bitplanes_little(
    const std::vector<std::vector<uint8_t>>& packed_streams,
    int bit_count,
    int64_t num_values
) {
    std::vector<uint32_t> flat(static_cast<size_t>(num_values), 0);

    for (int b = 0; b < bit_count; ++b) {
        const auto& packed = packed_streams[static_cast<size_t>(b)];

        for (int64_t i = 0; i < num_values; ++i) {
            uint8_t byte = packed[static_cast<size_t>(i / 8)];
            uint8_t bit = (byte >> (i % 8)) & 1u;

            if (bit) {
                flat[static_cast<size_t>(i)] |= (1u << b);
            }
        }
    }

    return flat;
}

int compute_bit_count(const std::vector<uint32_t>& values) {
    uint32_t max_val = 0;

    for (uint32_t v : values) {
        max_val = std::max(max_val, v);
    }

    if (max_val == 0) {
        return 1;
    }

    int bit_count = 0;
    while (max_val != 0 && bit_count < 32) {
        ++bit_count;
        max_val >>= 1;
    }

    return bit_count;
}

// Store integer residuals as compressed bitplanes. Even if every residual is zero, the
// block still matters because the decoder must replay the same prediction steps.
std::vector<uint8_t> zstd_compress_cpu(
    const std::vector<uint8_t>& input,
    int zstd_level
) {
    const int level = std::max(1, zstd_level);
    const size_t bound = ZSTD_compressBound(input.size());

    std::vector<uint8_t> output(bound);
    const size_t written =
        ZSTD_compress(output.data(), output.size(), input.data(), input.size(), level);

    if (ZSTD_isError(written)) {
        throw std::runtime_error(
            std::string("NGLR CPU zstd compress failed: ") +
            ZSTD_getErrorName(written)
        );
    }

    output.resize(written);
    return output;
}

std::vector<uint8_t> zstd_decompress_cpu(
    const std::vector<uint8_t>& input,
    size_t expected_size
) {
    std::vector<uint8_t> output(expected_size);

    const size_t written =
        ZSTD_decompress(output.data(), output.size(), input.data(), input.size());

    if (ZSTD_isError(written)) {
        throw std::runtime_error(
            std::string("NGLR CPU zstd decompress failed: ") +
            ZSTD_getErrorName(written)
        );
    }

    if (written != expected_size) {
        throw std::runtime_error(
            "NGLR CPU zstd decompress size mismatch: expected " +
            std::to_string(expected_size) +
            " got " +
            std::to_string(written)
        );
    }

    return output;
}

std::vector<std::vector<uint8_t>> compress_bitplanes_batch(
    const std::vector<std::vector<uint8_t>>& inputs,
    int zstd_level,
    bool use_cpu_zstd
) {
    if (use_cpu_zstd) {
        std::vector<std::vector<uint8_t>> outputs;
        outputs.reserve(inputs.size());

        for (const auto& input : inputs) {
            outputs.push_back(zstd_compress_cpu(input, zstd_level));
        }

        return outputs;
    }
#if defined(USE_CUDA) && defined(ENABLE_NVCOMP) && !defined(USE_ROCM) && !defined(__HIP_PLATFORM_AMD__)
    const size_t N = inputs.size();

    std::vector<std::vector<uint8_t>> outputs(N);
    if (N == 0) {
        return outputs;
    }

    size_t total_input_bytes = 0;
    size_t max_input_bytes = 0;

    for (const auto& input : inputs) {
        total_input_bytes += input.size();
        max_input_bytes = std::max(max_input_bytes, input.size());
    }

    if (total_input_bytes == 0 || max_input_bytes == 0) {
        return outputs;
    }

    nvcompBatchedZstdCompressOpts_t comp_opts =
        nvcompBatchedZstdCompressDefaultOpts;

    size_t max_output_bytes = 0;
    NGLR_CHECK_NVCOMP(
        nvcompBatchedZstdCompressGetMaxOutputChunkSize(
            max_input_bytes,
            comp_opts,
            &max_output_bytes
        )
    );

    size_t temp_bytes = 0;
    NGLR_CHECK_NVCOMP(
        nvcompBatchedZstdCompressGetTempSizeAsync(
            N,
            max_input_bytes,
            comp_opts,
            &temp_bytes,
            total_input_bytes
        )
    );

    cudaStream_t stream;
    NGLR_CHECK_CUDA(cudaStreamCreate(&stream));

    uint8_t* d_input_pool = nullptr;
    uint8_t* d_output_pool = nullptr;
    void* d_temp = nullptr;
    void** d_input_ptrs = nullptr;
    void** d_output_ptrs = nullptr;
    size_t* d_input_sizes = nullptr;
    size_t* d_output_sizes = nullptr;
    nvcompStatus_t* d_statuses = nullptr;

    NGLR_CHECK_CUDA(cudaMalloc(&d_input_pool, total_input_bytes));
    NGLR_CHECK_CUDA(cudaMalloc(&d_output_pool, N * max_output_bytes));
    if (temp_bytes > 0) {
        NGLR_CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
    }

    NGLR_CHECK_CUDA(cudaMalloc(&d_input_ptrs, N * sizeof(void*)));
    NGLR_CHECK_CUDA(cudaMalloc(&d_output_ptrs, N * sizeof(void*)));
    NGLR_CHECK_CUDA(cudaMalloc(&d_input_sizes, N * sizeof(size_t)));
    NGLR_CHECK_CUDA(cudaMalloc(&d_output_sizes, N * sizeof(size_t)));
    NGLR_CHECK_CUDA(cudaMalloc(&d_statuses, N * sizeof(nvcompStatus_t)));

    std::vector<void*> h_input_ptrs(N);
    std::vector<void*> h_output_ptrs(N);
    std::vector<size_t> h_input_sizes(N);

    size_t offset = 0;
    for (size_t i = 0; i < N; ++i) {
        const size_t nbytes = inputs[i].size();

        uint8_t* d_in = d_input_pool + offset;
        uint8_t* d_out = d_output_pool + i * max_output_bytes;

        NGLR_CHECK_CUDA(
            cudaMemcpyAsync(
                d_in,
                inputs[i].data(),
                nbytes,
                cudaMemcpyHostToDevice,
                stream
            )
        );

        h_input_ptrs[i] = d_in;
        h_output_ptrs[i] = d_out;
        h_input_sizes[i] = nbytes;

        offset += nbytes;
    }

    NGLR_CHECK_CUDA(cudaMemcpyAsync(
        d_input_ptrs,
        h_input_ptrs.data(),
        N * sizeof(void*),
        cudaMemcpyHostToDevice,
        stream
    ));

    NGLR_CHECK_CUDA(cudaMemcpyAsync(
        d_output_ptrs,
        h_output_ptrs.data(),
        N * sizeof(void*),
        cudaMemcpyHostToDevice,
        stream
    ));

    NGLR_CHECK_CUDA(cudaMemcpyAsync(
        d_input_sizes,
        h_input_sizes.data(),
        N * sizeof(size_t),
        cudaMemcpyHostToDevice,
        stream
    ));

    NGLR_CHECK_NVCOMP(
        nvcompBatchedZstdCompressAsync(
            reinterpret_cast<const void* const*>(d_input_ptrs),
            d_input_sizes,
            max_input_bytes,
            N,
            d_temp,
            temp_bytes,
            reinterpret_cast<void* const*>(d_output_ptrs),
            d_output_sizes,
            comp_opts,
            d_statuses,
            stream
        )
    );

    std::vector<size_t> h_output_sizes(N);
    std::vector<nvcompStatus_t> h_statuses(N);

    NGLR_CHECK_CUDA(cudaMemcpyAsync(
        h_output_sizes.data(),
        d_output_sizes,
        N * sizeof(size_t),
        cudaMemcpyDeviceToHost,
        stream
    ));

    NGLR_CHECK_CUDA(cudaMemcpyAsync(
        h_statuses.data(),
        d_statuses,
        N * sizeof(nvcompStatus_t),
        cudaMemcpyDeviceToHost,
        stream
    ));

    NGLR_CHECK_CUDA(cudaStreamSynchronize(stream));

    for (size_t i = 0; i < N; ++i) {
        if (h_statuses[i] != nvcompSuccess) {
            throw std::runtime_error(
                "NGLR nvCOMP Zstd compress failed on bitplane " +
                std::to_string(i)
            );
        }

        outputs[i].resize(h_output_sizes[i]);

        NGLR_CHECK_CUDA(
            cudaMemcpy(
                outputs[i].data(),
                d_output_pool + i * max_output_bytes,
                h_output_sizes[i],
                cudaMemcpyDeviceToHost
            )
        );
    }

    cudaFree(d_input_pool);
    cudaFree(d_output_pool);
    if (d_temp) {
        cudaFree(d_temp);
    }
    cudaFree(d_input_ptrs);
    cudaFree(d_output_ptrs);
    cudaFree(d_input_sizes);
    cudaFree(d_output_sizes);
    cudaFree(d_statuses);
    cudaStreamDestroy(stream);

    return outputs;
#else
    (void)inputs;
    (void)zstd_level;
    throw std::runtime_error(
        "NGLR requires nvCOMP. Reconfigure CAESAR with ENABLE_NVCOMP."
    );
#endif
}

// Read the bitplanes back into integer residuals. This must match the writer exactly:
/// same number of streams, same order, and same block shape.
std::vector<std::vector<uint8_t>> decompress_bitplanes_batch(
    const std::vector<std::vector<uint8_t>>& compressed_inputs,
    size_t expected_size,
    bool use_cpu_zstd
) {
    if (use_cpu_zstd) {
        std::vector<std::vector<uint8_t>> outputs;
        outputs.reserve(compressed_inputs.size());

        for (const auto& input : compressed_inputs) {
            outputs.push_back(zstd_decompress_cpu(input, expected_size));
        }

        return outputs;
    }

#if defined(USE_CUDA) && defined(ENABLE_NVCOMP) && !defined(USE_ROCM) && !defined(__HIP_PLATFORM_AMD__)
    const size_t N = compressed_inputs.size();

    std::vector<std::vector<uint8_t>> outputs(N);
    if (N == 0) {
        return outputs;
    }

    size_t total_comp_bytes = 0;
    size_t max_comp_bytes = 0;

    for (const auto& input : compressed_inputs) {
        total_comp_bytes += input.size();
        max_comp_bytes = std::max(max_comp_bytes, input.size());
    }

    if (total_comp_bytes == 0 || max_comp_bytes == 0 || expected_size == 0) {
        return outputs;
    }

    nvcompBatchedZstdDecompressOpts_t decomp_opts =
        nvcompBatchedZstdDecompressDefaultOpts;

    const size_t total_decomp_bytes = N * expected_size;

    size_t temp_bytes = 0;
    NGLR_CHECK_NVCOMP(
        nvcompBatchedZstdDecompressGetTempSizeAsync(
            N,
            expected_size,
            decomp_opts,
            &temp_bytes,
            total_decomp_bytes
        )
    );

    cudaStream_t stream;
    NGLR_CHECK_CUDA(cudaStreamCreate(&stream));

    uint8_t* d_comp_pool = nullptr;
    uint8_t* d_decomp_pool = nullptr;
    void* d_temp = nullptr;
    void** d_input_ptrs = nullptr;
    void** d_output_ptrs = nullptr;
    size_t* d_input_sizes = nullptr;
    size_t* d_output_sizes = nullptr;
    nvcompStatus_t* d_statuses = nullptr;

    NGLR_CHECK_CUDA(cudaMalloc(&d_comp_pool, N * max_comp_bytes));
    NGLR_CHECK_CUDA(cudaMalloc(&d_decomp_pool, N * expected_size));
    if (temp_bytes > 0) {
        NGLR_CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
    }

    NGLR_CHECK_CUDA(cudaMalloc(&d_input_ptrs, N * sizeof(void*)));
    NGLR_CHECK_CUDA(cudaMalloc(&d_output_ptrs, N * sizeof(void*)));
    NGLR_CHECK_CUDA(cudaMalloc(&d_input_sizes, N * sizeof(size_t)));
    NGLR_CHECK_CUDA(cudaMalloc(&d_output_sizes, N * sizeof(size_t)));
    NGLR_CHECK_CUDA(cudaMalloc(&d_statuses, N * sizeof(nvcompStatus_t)));

    std::vector<void*> h_input_ptrs(N);
    std::vector<void*> h_output_ptrs(N);
    std::vector<size_t> h_input_sizes(N);
    std::vector<size_t> h_output_sizes(N, expected_size);

    for (size_t i = 0; i < N; ++i) {
        uint8_t* d_in = d_comp_pool + i * max_comp_bytes;
        uint8_t* d_out = d_decomp_pool + i * expected_size;

        NGLR_CHECK_CUDA(
            cudaMemcpyAsync(
                d_in,
                compressed_inputs[i].data(),
                compressed_inputs[i].size(),
                cudaMemcpyHostToDevice,
                stream
            )
        );

        h_input_ptrs[i] = d_in;
        h_output_ptrs[i] = d_out;
        h_input_sizes[i] = compressed_inputs[i].size();
    }

    NGLR_CHECK_CUDA(cudaMemcpyAsync(
        d_input_ptrs,
        h_input_ptrs.data(),
        N * sizeof(void*),
        cudaMemcpyHostToDevice,
        stream
    ));

    NGLR_CHECK_CUDA(cudaMemcpyAsync(
        d_output_ptrs,
        h_output_ptrs.data(),
        N * sizeof(void*),
        cudaMemcpyHostToDevice,
        stream
    ));

    NGLR_CHECK_CUDA(cudaMemcpyAsync(
        d_input_sizes,
        h_input_sizes.data(),
        N * sizeof(size_t),
        cudaMemcpyHostToDevice,
        stream
    ));

    NGLR_CHECK_CUDA(cudaMemcpyAsync(
        d_output_sizes,
        h_output_sizes.data(),
        N * sizeof(size_t),
        cudaMemcpyHostToDevice,
        stream
    ));

    NGLR_CHECK_NVCOMP(
        nvcompBatchedZstdDecompressAsync(
            reinterpret_cast<const void* const*>(d_input_ptrs),
            d_input_sizes,
            d_output_sizes,
            d_output_sizes,
            N,
            d_temp,
            temp_bytes,
            reinterpret_cast<void* const*>(d_output_ptrs),
            decomp_opts,
            d_statuses,
            stream
        )
    );

    std::vector<nvcompStatus_t> h_statuses(N);

    NGLR_CHECK_CUDA(cudaMemcpyAsync(
        h_statuses.data(),
        d_statuses,
        N * sizeof(nvcompStatus_t),
        cudaMemcpyDeviceToHost,
        stream
    ));

    NGLR_CHECK_CUDA(cudaStreamSynchronize(stream));

    for (size_t i = 0; i < N; ++i) {
        if (h_statuses[i] != nvcompSuccess) {
            throw std::runtime_error(
                "NGLR nvCOMP Zstd decompress failed on bitplane " +
                std::to_string(i)
            );
        }

        outputs[i].resize(expected_size);

        NGLR_CHECK_CUDA(
            cudaMemcpy(
                outputs[i].data(),
                d_decomp_pool + i * expected_size,
                expected_size,
                cudaMemcpyDeviceToHost
            )
        );
    }

    cudaFree(d_comp_pool);
    cudaFree(d_decomp_pool);
    if (d_temp) {
        cudaFree(d_temp);
    }
    cudaFree(d_input_ptrs);
    cudaFree(d_output_ptrs);
    cudaFree(d_input_sizes);
    cudaFree(d_output_sizes);
    cudaFree(d_statuses);
    cudaStreamDestroy(stream);

    return outputs;
#else
    (void)compressed_inputs;
    (void)expected_size;
    throw std::runtime_error(
        "NGLR requires nvCOMP. Reconfigure CAESAR with ENABLE_NVCOMP."
    );
#endif
}

// The math is first done in int64 so it is hard to overflow. Before sending values to
// the bitplane codec, we check that they really fit in int32.
void ensure_int64_fits_int32(
    const torch::Tensor& values,
    const std::string& what
) {
    const int64_t min_value =
        values.min().item<int64_t>();
    const int64_t max_value =
        values.max().item<int64_t>();

    if (min_value < std::numeric_limits<int32_t>::min() ||
        max_value > std::numeric_limits<int32_t>::max()) {
        throw std::overflow_error(
            what + " does not fit int32: min=" +
            std::to_string(min_value) +
            " max=" +
            std::to_string(max_value)
        );
    }
}

std::vector<EncodedDeltaBlock> encode_delta_blocks_batch(
    const std::vector<torch::Tensor>& deltas,
    int zstd_level,
    bool use_cpu_zstd
) {
    std::vector<EncodedDeltaBlock> blocks;
    blocks.reserve(deltas.size());

    std::vector<std::vector<uint8_t>> all_packed_bitplanes;
    std::vector<std::pair<size_t, size_t>> stream_ranges;
    stream_ranges.reserve(deltas.size());

    for (const auto& delta : deltas) {
        torch::Tensor d =
            delta.to(torch::kCPU).to(torch::kInt32).contiguous();

        EncodedDeltaBlock block;
        block.T = d.size(0);
        block.H = d.size(1);
        block.W = d.size(2);

        std::vector<uint32_t> zz =
            zigzag_encode_int32_vector(d);

        block.bit_count =
            compute_bit_count(zz);

        const size_t begin =
            all_packed_bitplanes.size();

        if (block.bit_count > 0) {
            for (int b = 0; b < block.bit_count; ++b) {
                all_packed_bitplanes.push_back(
                    pack_bitplane_little(zz, b)
                );
            }
        }

        stream_ranges.push_back(
            {begin, all_packed_bitplanes.size()}
        );

        blocks.push_back(std::move(block));
    }

    if (!all_packed_bitplanes.empty()) {
        std::vector<std::vector<uint8_t>> compressed =
            compress_bitplanes_batch(all_packed_bitplanes, zstd_level, use_cpu_zstd);

        for (size_t i = 0; i < blocks.size(); ++i) {
            const auto [begin, end] = stream_ranges[i];
            blocks[i].streams.reserve(end - begin);

            for (size_t j = begin; j < end; ++j) {
                blocks[i].streams.push_back(
                    std::move(compressed[j])
                );
            }
        }
    }

    return blocks;
}

torch::Tensor decode_delta_block(
    const EncodedDeltaBlock& block,
    bool use_cpu_zstd
) {
    if (block.bit_count == 0) {
        return torch::zeros(
            {block.T, block.H, block.W},
            torch::TensorOptions()
                .dtype(torch::kInt32)
                .device(torch::kCPU)
        );
    }

    const int64_t num_values = block.T * block.H * block.W;
    const size_t packed_size =
        static_cast<size_t>((num_values + 7) / 8);

    std::vector<std::vector<uint8_t>> packed_streams =
        decompress_bitplanes_batch(
            block.streams,
            packed_size,
            use_cpu_zstd
        );

    std::vector<uint32_t> zz_back =
        unpack_bitplanes_little(
            packed_streams,
            block.bit_count,
            num_values
        );

    return zigzag_decode_uint32_vector_to_tensor(
        zz_back,
        block.T,
        block.H,
        block.W
    );
}

// Move an encoded block into the public format used by the rest of CAESAR. The move is
// intentional so large compressed byte buffers are not copied.
NGLRBlockStream to_public_block(EncodedDeltaBlock&& encoded) {
    NGLRBlockStream block;

    block.bit_count = encoded.bit_count;
    block.streams = std::move(encoded.streams);

    return block;
}

EncodedDeltaBlock to_encoded_block(
    const NGLRBlockStream& block_stream,
    const BlockSlice& slice
) {
    EncodedDeltaBlock encoded;

    encoded.bit_count = block_stream.bit_count;
    encoded.T = slice.t1 - slice.t0;
    encoded.H = slice.h1 - slice.h0;
    encoded.W = slice.w1 - slice.w0;
    encoded.streams = block_stream.streams;

    return encoded;
}

struct BlockShape {
    int64_t T = 0;
    int64_t H = 0;
    int64_t W = 0;
};

BlockShape block_shape_from_slice(const BlockSlice& sl) {
    return {sl.t1 - sl.t0, sl.h1 - sl.h0, sl.w1 - sl.w0};
}

bool same_block_shape(const BlockShape& a, const BlockShape& b) {
    return a.T == b.T && a.H == b.H && a.W == b.W;
}

size_t same_shape_batch_end(
    const std::vector<BlockSlice>& slices,
    size_t start,
    size_t max_batch
) {
    const BlockShape first = block_shape_from_slice(slices[start]);
    size_t end = start + 1;

    while (end < slices.size() && end - start < max_batch) {
        if (!same_block_shape(first, block_shape_from_slice(slices[end]))) {
            break;
        }
        ++end;
    }

    return end;
}

torch::Tensor gather_qhat_context_value(
    const torch::Tensor& qhat,
    const torch::Tensor& t_idx,
    const torch::Tensor& h_idx,
    const torch::Tensor& w_idx,
    const torch::Tensor& mask,
    int64_t dt,
    int64_t dh,
    int64_t dw
) {
    const int64_t nb = qhat.size(0);
    const int64_t n = t_idx.numel();

    torch::Tensor out =
        torch::zeros({nb, n}, qhat.options().dtype(torch::kInt32));

    if (!mask.any().item<bool>()) {
        return out;
    }

    torch::Tensor mt = t_idx.index({mask}) + dt;
    torch::Tensor mh = h_idx.index({mask}) + dh;
    torch::Tensor mw = w_idx.index({mask}) + dw;

    torch::Tensor vals =
        qhat.index({
            torch::indexing::Slice(),
            mt,
            mh,
            mw
        });

    out.index_put_(
        {torch::indexing::Slice(), mask},
        vals
    );

    return out;
}

std::pair<torch::Tensor, torch::Tensor> torch_lorenzo_context(
    const torch::Tensor& qhat,
    const torch::Tensor& t_idx,
    const torch::Tensor& h_idx,
    const torch::Tensor& w_idx,
    double q_scale
) {
    torch::Tensor v1 =
        gather_qhat_context_value(qhat, t_idx, h_idx, w_idx, t_idx > 0, -1, 0, 0);

    torch::Tensor v2 =
        gather_qhat_context_value(qhat, t_idx, h_idx, w_idx, h_idx > 0, 0, -1, 0);

    torch::Tensor v3 =
        gather_qhat_context_value(qhat, t_idx, h_idx, w_idx, w_idx > 0, 0, 0, -1);

    torch::Tensor v4 =
        gather_qhat_context_value(qhat, t_idx, h_idx, w_idx, (t_idx > 0) & (h_idx > 0), -1, -1, 0);

    torch::Tensor v5 =
        gather_qhat_context_value(qhat, t_idx, h_idx, w_idx, (t_idx > 0) & (w_idx > 0), -1, 0, -1);

    torch::Tensor v6 =
        gather_qhat_context_value(qhat, t_idx, h_idx, w_idx, (h_idx > 0) & (w_idx > 0), 0, -1, -1);

    torch::Tensor v7 =
        gather_qhat_context_value(qhat, t_idx, h_idx, w_idx, (t_idx > 0) & (h_idx > 0) & (w_idx > 0), -1, -1, -1);

    torch::Tensor pred =
        v1 + v2 + v3 - v4 - v5 - v6 + v7;

    const double denom = std::max(q_scale, 1.0);

    torch::Tensor ctx =
        torch::stack({
            v1.to(torch::kFloat32) / denom,
            v2.to(torch::kFloat32) / denom,
            v3.to(torch::kFloat32) / denom,
            v4.to(torch::kFloat32) / denom,
            v5.to(torch::kFloat32) / denom,
            v6.to(torch::kFloat32) / denom,
            v7.to(torch::kFloat32) / denom,
            pred.to(torch::kFloat32) / denom
        }, 2);

    return {ctx, pred};
}

// Batched causal NGLR encode. Blocks with the same shape are processed together on CUDA.
std::vector<torch::Tensor> strict_encode_delta_blocks(
    const std::vector<torch::Tensor>& q_blocks,
    const std::vector<torch::Tensor>& r_blocks,
    CausalNeuralLorenzoNet model,
    const NGLRMetaData& meta,
    torch::Device device
) {
    torch::NoGradGuard no_grad;

    const int64_t nb =
        static_cast<int64_t>(q_blocks.size());

    std::vector<torch::Tensor> out;
    if (nb == 0) {
        return out;
    }

    torch::Tensor q_ref =
        torch::stack(q_blocks, 0)
            .to(device)
            .to(torch::kInt32)
            .contiguous();

    torch::Tensor r =
        torch::stack(r_blocks, 0)
            .to(device)
            .to(torch::kFloat32)
            .contiguous();

    const int64_t T = q_ref.size(1);
    const int64_t H = q_ref.size(2);
    const int64_t W = q_ref.size(3);

    torch::Tensor qhat =
        torch::zeros({nb, T, H, W}, q_ref.options().dtype(torch::kInt32));

    torch::Tensor delta =
        torch::zeros({nb, T, H, W}, q_ref.options().dtype(torch::kInt32));

    model->to(device);
    model->eval();

    torch::Tensor rf =
        model->encode_recons(r.unsqueeze(1).contiguous())
            .to(device)
            .to(torch::kFloat32)
            .contiguous();

    const int64_t ch = rf.size(1);

    auto diagonals =
        diagonal_indices(T, H, W);

    for (const auto& diag : diagonals) {
        const auto& ts = std::get<0>(diag);
        const auto& hs = std::get<1>(diag);
        const auto& ws = std::get<2>(diag);

        const int64_t n =
            static_cast<int64_t>(ts.size());

        torch::Tensor t_idx =
            torch::from_blob(
                const_cast<int64_t*>(ts.data()),
                {n},
                torch::kInt64
            ).clone().to(device);

        torch::Tensor h_idx =
            torch::from_blob(
                const_cast<int64_t*>(hs.data()),
                {n},
                torch::kInt64
            ).clone().to(device);

        torch::Tensor w_idx =
            torch::from_blob(
                const_cast<int64_t*>(ws.data()),
                {n},
                torch::kInt64
            ).clone().to(device);

        auto ctx_pred =
            torch_lorenzo_context(
                qhat,
                t_idx,
                h_idx,
                w_idx,
                meta.q_scale
            );

        torch::Tensor qctx =
            ctx_pred.first
                .reshape({nb * n, 8, 1, 1, 1})
                .contiguous();

        torch::Tensor pred =
            ctx_pred.second;

        torch::Tensor rf_sel =
            rf.index({
                torch::indexing::Slice(),
                torch::indexing::Slice(),
                t_idx,
                h_idx,
                w_idx
            })
            .permute({0, 2, 1})
            .contiguous()
            .view({nb * n, ch, 1, 1, 1});

        torch::Tensor bias_norm =
            model->forward_from_feature(rf_sel, qctx)
                .reshape({nb, n})
                .to(device)
                .to(torch::kFloat32)
                .contiguous();

        torch::Tensor ref =
            reference_from_pred_bias(pred, bias_norm, meta.d_scale, device);

        torch::Tensor cur =
            q_ref.index({
                torch::indexing::Slice(),
                t_idx,
                h_idx,
                w_idx
            });

        torch::Tensor d64 =
            cur.to(torch::kInt64) - ref.to(torch::kInt64);

        ensure_int64_fits_int32(d64, "NGLR delta");

        torch::Tensor d =
            d64.to(torch::kInt32);

        delta.index_put_(
            {torch::indexing::Slice(), t_idx, h_idx, w_idx},
            d
        );

        qhat.index_put_(
            {torch::indexing::Slice(), t_idx, h_idx, w_idx},
            cur
        );
    }

#if defined(DEBUG_MODE)
    if (!torch::equal(qhat, q_ref)) {
        int64_t maxdiff =
            (qhat.to(torch::kCPU).to(torch::kInt64) -
             q_ref.to(torch::kCPU).to(torch::kInt64))
                .abs()
                .max()
                .item<int64_t>();

        throw std::runtime_error(
            "NGLR batched GPU encode mismatch, maxdiff=" +
            std::to_string(maxdiff)
        );
    }
#endif

    out.reserve(static_cast<size_t>(nb));

    for (int64_t i = 0; i < nb; ++i) {
        out.push_back(
            delta.index({i})
                .to(torch::kCPU)
                .to(torch::kInt32)
                .contiguous()
        );
    }

    return out;
}

// Batched causal NGLR decode mirrors the encode order and reconstructs qhat on CUDA.
std::vector<torch::Tensor> strict_decode_delta_blocks(
    const std::vector<torch::Tensor>& delta_blocks,
    const std::vector<torch::Tensor>& r_blocks,
    CausalNeuralLorenzoNet model,
    const NGLRMetaData& meta,
    torch::Device device
) {
    torch::NoGradGuard no_grad;

    const int64_t nb =
        static_cast<int64_t>(delta_blocks.size());

    std::vector<torch::Tensor> out;
    if (nb == 0) {
        return out;
    }

    torch::Tensor deltas =
        torch::stack(delta_blocks, 0)
            .to(device)
            .to(torch::kInt32)
            .contiguous();

    torch::Tensor r =
        torch::stack(r_blocks, 0)
            .to(device)
            .to(torch::kFloat32)
            .contiguous();

    const int64_t T = deltas.size(1);
    const int64_t H = deltas.size(2);
    const int64_t W = deltas.size(3);

    torch::Tensor qhat =
        torch::zeros({nb, T, H, W}, deltas.options().dtype(torch::kInt32));

    model->to(device);
    model->eval();

    torch::Tensor rf =
        model->encode_recons(r.unsqueeze(1).contiguous())
            .to(device)
            .to(torch::kFloat32)
            .contiguous();

    const int64_t ch = rf.size(1);

    auto diagonals =
        diagonal_indices(T, H, W);

    for (const auto& diag : diagonals) {
        const auto& ts = std::get<0>(diag);
        const auto& hs = std::get<1>(diag);
        const auto& ws = std::get<2>(diag);

        const int64_t n =
            static_cast<int64_t>(ts.size());

        torch::Tensor t_idx =
            torch::from_blob(
                const_cast<int64_t*>(ts.data()),
                {n},
                torch::kInt64
            ).clone().to(device);

        torch::Tensor h_idx =
            torch::from_blob(
                const_cast<int64_t*>(hs.data()),
                {n},
                torch::kInt64
            ).clone().to(device);

        torch::Tensor w_idx =
            torch::from_blob(
                const_cast<int64_t*>(ws.data()),
                {n},
                torch::kInt64
            ).clone().to(device);

        auto ctx_pred =
            torch_lorenzo_context(
                qhat,
                t_idx,
                h_idx,
                w_idx,
                meta.q_scale
            );

        torch::Tensor qctx =
            ctx_pred.first
                .reshape({nb * n, 8, 1, 1, 1})
                .contiguous();

        torch::Tensor pred =
            ctx_pred.second;

        torch::Tensor rf_sel =
            rf.index({
                torch::indexing::Slice(),
                torch::indexing::Slice(),
                t_idx,
                h_idx,
                w_idx
            })
            .permute({0, 2, 1})
            .contiguous()
            .view({nb * n, ch, 1, 1, 1});

        torch::Tensor bias_norm =
            model->forward_from_feature(rf_sel, qctx)
                .reshape({nb, n})
                .to(device)
                .to(torch::kFloat32)
                .contiguous();

        torch::Tensor ref =
            reference_from_pred_bias(pred, bias_norm, meta.d_scale, device);

        torch::Tensor d =
            deltas.index({
                torch::indexing::Slice(),
                t_idx,
                h_idx,
                w_idx
            });

        torch::Tensor q64 =
            ref.to(torch::kInt64) + d.to(torch::kInt64);

        ensure_int64_fits_int32(q64, "NGLR decoded q");

        qhat.index_put_(
            {torch::indexing::Slice(), t_idx, h_idx, w_idx},
            q64.to(torch::kInt32)
        );
    }

    out.reserve(static_cast<size_t>(nb));

    for (int64_t i = 0; i < nb; ++i) {
        out.push_back(
            qhat.index({i})
                .to(torch::kCPU)
                .to(torch::kInt32)
                .contiguous()
        );
    }

    return out;
}

}

void save_metadata(
    std::ostream& out,
    const NGLRMetaData& meta,
    const NGLRCompressedData& compressed
) {
    out.write(reinterpret_cast<const char*>(&meta.mean), sizeof(meta.mean));
    out.write(reinterpret_cast<const char*>(&meta.scale), sizeof(meta.scale));
    out.write(reinterpret_cast<const char*>(&meta.step), sizeof(meta.step));
    out.write(reinterpret_cast<const char*>(&meta.q_scale), sizeof(meta.q_scale));
    out.write(reinterpret_cast<const char*>(&meta.d_scale), sizeof(meta.d_scale));

    out.write(reinterpret_cast<const char*>(&meta.block_t), sizeof(meta.block_t));
    out.write(reinterpret_cast<const char*>(&meta.block_h), sizeof(meta.block_h));
    out.write(reinterpret_cast<const char*>(&meta.block_w), sizeof(meta.block_w));
    out.write(reinterpret_cast<const char*>(&meta.hidden), sizeof(meta.hidden));
    out.write(reinterpret_cast<const char*>(&meta.q_hidden), sizeof(meta.q_hidden));
    out.write(reinterpret_cast<const char*>(&meta.model_blocks), sizeof(meta.model_blocks));

    size_t size = meta.shape.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(
        reinterpret_cast<const char*>(meta.shape.data()),
        static_cast<std::streamsize>(size * sizeof(int64_t))
    );

    const uint32_t model_magic = kNGLRModelTensorMagic;
    const uint32_t model_version = kNGLRModelTensorVersion;
    out.write(reinterpret_cast<const char*>(&model_magic), sizeof(model_magic));
    out.write(reinterpret_cast<const char*>(&model_version), sizeof(model_version));

    size = meta.model_tensors.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));

    for (const auto& tensor : meta.model_tensors) {
        size = tensor.name.size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        out.write(
            tensor.name.data(),
            static_cast<std::streamsize>(tensor.name.size())
        );

        size = tensor.shape.size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        out.write(
            reinterpret_cast<const char*>(tensor.shape.data()),
            static_cast<std::streamsize>(size * sizeof(int64_t))
        );

        size = tensor.values.size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        out.write(
            reinterpret_cast<const char*>(tensor.values.data()),
            static_cast<std::streamsize>(size * sizeof(float))
        );
    }

    size = compressed.blocks.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));

    for (const auto& block : compressed.blocks) {
        uint32_t bit_count = static_cast<uint32_t>(block.bit_count);
        out.write(reinterpret_cast<const char*>(&bit_count), sizeof(bit_count));

        for (const auto& stream : block.streams) {
            uint64_t stream_size = static_cast<uint64_t>(stream.size());
            out.write(reinterpret_cast<const char*>(&stream_size), sizeof(stream_size));
            out.write(
                reinterpret_cast<const char*>(stream.data()),
                static_cast<std::streamsize>(stream_size)
            );
        }
    }
}

void load_metadata(
    std::istream& in,
    NGLRMetaData& meta,
    NGLRCompressedData& compressed
) {
    size_t size = 0;
    const std::streampos meta_start = in.tellg();

    auto read_clean_meta = [&]() {
        in.read(reinterpret_cast<char*>(&meta.mean), sizeof(meta.mean));
        in.read(reinterpret_cast<char*>(&meta.scale), sizeof(meta.scale));
        in.read(reinterpret_cast<char*>(&meta.step), sizeof(meta.step));
        in.read(reinterpret_cast<char*>(&meta.q_scale), sizeof(meta.q_scale));
        in.read(reinterpret_cast<char*>(&meta.d_scale), sizeof(meta.d_scale));

        in.read(reinterpret_cast<char*>(&meta.block_t), sizeof(meta.block_t));
        in.read(reinterpret_cast<char*>(&meta.block_h), sizeof(meta.block_h));
        in.read(reinterpret_cast<char*>(&meta.block_w), sizeof(meta.block_w));
        in.read(reinterpret_cast<char*>(&meta.hidden), sizeof(meta.hidden));
        in.read(reinterpret_cast<char*>(&meta.q_hidden), sizeof(meta.q_hidden));
        in.read(reinterpret_cast<char*>(&meta.model_blocks), sizeof(meta.model_blocks));

        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (size > 8) {
            in.setstate(std::ios::failbit);
            return;
        }

        meta.shape.resize(size);
        in.read(
            reinterpret_cast<char*>(meta.shape.data()),
            static_cast<std::streamsize>(size * sizeof(int64_t))
        );
    };

    read_clean_meta();

    const bool clean_meta_ok =
        in.good() &&
        meta.scale > 0.0 &&
        meta.block_t > 0 &&
        meta.block_h > 0 &&
        meta.block_w > 0 &&
        meta.hidden > 0 &&
        meta.q_hidden > 0 &&
        meta.model_blocks > 0 &&
        !meta.shape.empty() &&
        meta.shape.size() <= 8;

    if (!clean_meta_ok) {
        in.clear();
        in.seekg(meta_start);

        bool old_correction_occur = false;
        double old_target = 0.0;
        double old_base_nrmse = 0.0;
        double old_quant_nrmse = 0.0;
        double old_best_loss = 0.0;
        int old_best_epoch = 0;
        int old_train_epochs = 0;
        int old_zstd_level = 0;
        int64_t old_original_bytes = 0;
        int64_t old_latent_bit = 0;
        int64_t old_correction_bytes = 0;

        in.read(reinterpret_cast<char*>(&old_correction_occur), sizeof(old_correction_occur));
        in.read(reinterpret_cast<char*>(&old_target), sizeof(old_target));
        in.read(reinterpret_cast<char*>(&meta.mean), sizeof(meta.mean));
        in.read(reinterpret_cast<char*>(&meta.scale), sizeof(meta.scale));
        in.read(reinterpret_cast<char*>(&meta.step), sizeof(meta.step));
        in.read(reinterpret_cast<char*>(&meta.q_scale), sizeof(meta.q_scale));
        in.read(reinterpret_cast<char*>(&meta.d_scale), sizeof(meta.d_scale));

        in.read(reinterpret_cast<char*>(&meta.block_t), sizeof(meta.block_t));
        in.read(reinterpret_cast<char*>(&meta.block_h), sizeof(meta.block_h));
        in.read(reinterpret_cast<char*>(&meta.block_w), sizeof(meta.block_w));
        in.read(reinterpret_cast<char*>(&meta.hidden), sizeof(meta.hidden));
        in.read(reinterpret_cast<char*>(&meta.q_hidden), sizeof(meta.q_hidden));
        in.read(reinterpret_cast<char*>(&meta.model_blocks), sizeof(meta.model_blocks));
        in.read(reinterpret_cast<char*>(&old_train_epochs), sizeof(old_train_epochs));
        in.read(reinterpret_cast<char*>(&old_zstd_level), sizeof(old_zstd_level));

        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        meta.shape.resize(size);
        in.read(
            reinterpret_cast<char*>(meta.shape.data()),
            static_cast<std::streamsize>(size * sizeof(int64_t))
        );

        in.read(reinterpret_cast<char*>(&old_original_bytes), sizeof(old_original_bytes));
        in.read(reinterpret_cast<char*>(&old_latent_bit), sizeof(old_latent_bit));
        in.read(reinterpret_cast<char*>(&old_correction_bytes), sizeof(old_correction_bytes));
        in.read(reinterpret_cast<char*>(&old_base_nrmse), sizeof(old_base_nrmse));
        in.read(reinterpret_cast<char*>(&old_quant_nrmse), sizeof(old_quant_nrmse));
        in.read(reinterpret_cast<char*>(&old_best_loss), sizeof(old_best_loss));
        in.read(reinterpret_cast<char*>(&old_best_epoch), sizeof(old_best_epoch));

        (void)old_correction_occur;
        (void)old_target;
        (void)old_base_nrmse;
        (void)old_quant_nrmse;
        (void)old_best_loss;
        (void)old_best_epoch;
        (void)old_train_epochs;
        (void)old_zstd_level;
        (void)old_original_bytes;
        (void)old_latent_bit;
        (void)old_correction_bytes;
    }

    meta.model_tensors.clear();

    if (clean_meta_ok) {
        uint32_t model_magic = 0;
        const std::streampos before_model = in.tellg();
        in.read(reinterpret_cast<char*>(&model_magic), sizeof(model_magic));

        if (in.good() && model_magic == kNGLRModelTensorMagic) {
            uint32_t model_version = 0;
            in.read(reinterpret_cast<char*>(&model_version), sizeof(model_version));
            if (model_version != kNGLRModelTensorVersion) {
                throw std::runtime_error("Unsupported NGLR model tensor metadata version");
            }

            in.read(reinterpret_cast<char*>(&size), sizeof(size));
            if (size > 10000) {
                throw std::runtime_error("Invalid NGLR model tensor count in metadata");
            }

            meta.model_tensors.resize(size);

            for (auto& tensor : meta.model_tensors) {
                in.read(reinterpret_cast<char*>(&size), sizeof(size));
                if (size > 4096) {
                    throw std::runtime_error("Invalid NGLR model tensor name length");
                }

                tensor.name.resize(size);
                in.read(
                    tensor.name.data(),
                    static_cast<std::streamsize>(tensor.name.size())
                );

                in.read(reinterpret_cast<char*>(&size), sizeof(size));
                if (size > 8) {
                    throw std::runtime_error("Invalid NGLR model tensor rank");
                }

                tensor.shape.resize(size);
                in.read(
                    reinterpret_cast<char*>(tensor.shape.data()),
                    static_cast<std::streamsize>(size * sizeof(int64_t))
                );

                in.read(reinterpret_cast<char*>(&size), sizeof(size));
                tensor.values.resize(size);
                in.read(
                    reinterpret_cast<char*>(tensor.values.data()),
                    static_cast<std::streamsize>(size * sizeof(float))
                );
            }
        } else {
            in.clear();
            in.seekg(before_model);
        }
    }

    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    compressed.blocks.resize(size);

    for (auto& block : compressed.blocks) {
        uint32_t bit_count = 0;
        in.read(reinterpret_cast<char*>(&bit_count), sizeof(bit_count));
        block.bit_count = static_cast<int>(bit_count);

        block.streams.resize(block.bit_count);

        for (auto& stream : block.streams) {
            uint64_t stream_size = 0;
            in.read(reinterpret_cast<char*>(&stream_size), sizeof(stream_size));
            stream.resize(static_cast<size_t>(stream_size));
            in.read(
                reinterpret_cast<char*>(stream.data()),
                static_cast<std::streamsize>(stream_size)
            );
        }
    }
}

torch::Tensor lorenzo_pred(const torch::Tensor& q_in) {
    torch::Tensor q =
        q_in.to(torch::kCPU).to(torch::kInt64).contiguous();

    if (q.dim() != 3) {
        throw std::runtime_error("lorenzo_pred expects [T,H,W]");
    }

    auto sizes = q.sizes();

    int64_t T = sizes[0];
    int64_t H = sizes[1];
    int64_t W = sizes[2];

    torch::Tensor pred = torch::zeros_like(q);

    auto q_acc = q.accessor<int64_t, 3>();
    auto p_acc = pred.accessor<int64_t, 3>();

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                int64_t v1 = (t > 0) ? q_acc[t - 1][h][w] : 0;
                int64_t v2 = (h > 0) ? q_acc[t][h - 1][w] : 0;
                int64_t v3 = (w > 0) ? q_acc[t][h][w - 1] : 0;

                int64_t v4 = (t > 0 && h > 0) ? q_acc[t - 1][h - 1][w] : 0;
                int64_t v5 = (t > 0 && w > 0) ? q_acc[t - 1][h][w - 1] : 0;
                int64_t v6 = (h > 0 && w > 0) ? q_acc[t][h - 1][w - 1] : 0;
                int64_t v7 =
                    (t > 0 && h > 0 && w > 0)
                        ? q_acc[t - 1][h - 1][w - 1]
                        : 0;

                p_acc[t][h][w] =
                    v1 + v2 + v3 - v4 - v5 - v6 + v7;
            }
        }
    }

    return pred;
}

torch::Tensor lorenzo_delta(const torch::Tensor& q) {
    return q.to(torch::kInt64) - lorenzo_pred(q);
}

torch::Tensor zigzag_encode(const torch::Tensor& delta_in) {
    torch::Tensor d = delta_in.to(torch::kInt64).contiguous();

    torch::Tensor non_negative = d.ge(0);
    torch::Tensor pos = d * 2;
    torch::Tensor neg = -2 * d - 1;

    return torch::where(non_negative, pos, neg).to(torch::kInt64);
}

torch::Tensor zigzag_decode(const torch::Tensor& zz_in) {
    torch::Tensor z = zz_in.to(torch::kInt64).contiguous();

    torch::Tensor even = z.bitwise_and(1).eq(0);
    torch::Tensor pos = z / 2;
    torch::Tensor neg = -((z + 1) / 2);

    return torch::where(even, pos, neg).to(torch::kInt32);
}

// Quantize residuals, run causal NGLR correction, and store compressed bitplanes.
NGLRResult encode_correction(
    const torch::Tensor& original,
    const torch::Tensor& reconstruction,
    CausalNeuralLorenzoNet model,
    NGLRMetaData meta,
    torch::Device device
) {
    torch::Device nglr_device = resolve_nglr_device(device);
    const bool use_zstd_cpu = !nglr_device.is_cuda();

    torch::Tensor x =
        original.to(torch::kCPU).to(torch::kFloat32).contiguous();

    torch::Tensor r =
        reconstruction.to(torch::kCPU).to(torch::kFloat32).contiguous();

    NGLRResult result;
    result.meta = std::move(meta);

    result.meta.zstd_level =
        get_env_int_or_default("CAESAR_NGLR_ZSTD_LEVEL", result.meta.zstd_level);

    model->to(nglr_device);
    model->eval();

    std::vector<BlockSlice> slices =
        iter_block_slices(
            result.meta.shape,
            result.meta.block_t,
            result.meta.block_h,
            result.meta.block_w
        );

    const size_t batch_blocks =
        static_cast<size_t>(get_env_int_or_default("CAESAR_NGLR_BATCH_BLOCKS", 32));

    result.compressed.blocks.clear();
    result.compressed.blocks.reserve(slices.size());

    size_t i = 0;
    while (i < slices.size()) {
        const size_t next_i =
            same_shape_batch_end(
                slices,
                i,
                batch_blocks
            );

        std::vector<torch::Tensor> q_blocks;
        std::vector<torch::Tensor> r_blocks;

        q_blocks.reserve(next_i - i);
        r_blocks.reserve(next_i - i);

        for (size_t j = i; j < next_i; ++j) {
            const BlockSlice& sl = slices[j];

            torch::Tensor x_block =
                slice_5d_to_3d(x, sl);

            torch::Tensor r_block_raw =
                slice_5d_to_3d(r, sl);

            torch::Tensor x_block_norm =
                (x_block - result.meta.mean) / result.meta.scale;

            torch::Tensor r_block_norm =
                (r_block_raw - result.meta.mean) / result.meta.scale;

            torch::Tensor q_block =
                torch::round(
                    (x_block_norm - r_block_norm) / result.meta.step
                ).to(torch::kInt32).contiguous();

            q_blocks.push_back(q_block);
            r_blocks.push_back(r_block_norm.contiguous());
        }
        std::vector<torch::Tensor> deltas =
            strict_encode_delta_blocks(
                q_blocks,
                r_blocks,
                model,
                result.meta,
                nglr_device
            );

        std::vector<EncodedDeltaBlock> encoded_blocks =
            encode_delta_blocks_batch(deltas, result.meta.zstd_level, use_zstd_cpu);

        for (auto& encoded : encoded_blocks) {
            result.compressed.blocks.push_back(
                to_public_block(std::move(encoded))
            );
        }

        i = next_i;

    }

    if (!result.compressed.blocks.empty()) {
        result.meta.model_tensors = export_model_tensors(model);
    }

    return result;
}

// Decode compressed bitplanes, replay causal NGLR reconstruction, and apply correction.
torch::Tensor decompress(
    const torch::Tensor& reconstruction,
    const NGLRMetaData& meta,
    const NGLRCompressedData& compressed,
    torch::Device device
) {
    torch::Device nglr_device = resolve_nglr_device(device);
    const bool use_zstd_cpu = !nglr_device.is_cuda();

    torch::Tensor r =
        reconstruction.to(torch::kCPU).to(torch::kFloat32).contiguous();

    torch::Tensor r_norm =
        (r - meta.mean) / meta.scale;

    torch::Tensor corrected =
        r.clone();

    if (compressed.blocks.empty()) {
        return corrected.to(device);
    }

    CausalNeuralLorenzoNet model =
        load_nglr_model(meta, nglr_device);

    std::vector<BlockSlice> slices =
        iter_block_slices(
            meta.shape,
            meta.block_t,
            meta.block_h,
            meta.block_w
        );

    if (slices.size() != compressed.blocks.size()) {
        throw std::runtime_error(
            "NGLR decompress block count mismatch: expected " +
            std::to_string(slices.size()) +
            ", got " +
            std::to_string(compressed.blocks.size())
        );
    }

    const size_t batch_blocks =
        static_cast<size_t>(get_env_int_or_default("CAESAR_NGLR_BATCH_BLOCKS", 32));

    size_t i = 0;

    while (i < slices.size()) {
        const size_t next_i =
            same_shape_batch_end(
                slices,
                i,
                batch_blocks
            );

        std::vector<torch::Tensor> delta_blocks;
        std::vector<torch::Tensor> r_blocks;

        delta_blocks.reserve(next_i - i);
        r_blocks.reserve(next_i - i);

        for (size_t j = i; j < next_i; ++j) {
            EncodedDeltaBlock encoded =
                to_encoded_block(compressed.blocks[j], slices[j]);

            delta_blocks.push_back(
                decode_delta_block(encoded, use_zstd_cpu)
            );

            r_blocks.push_back(
                slice_5d_to_3d(r_norm, slices[j])
                    .to(torch::kCPU)
                    .to(torch::kFloat32)
                    .contiguous()
            );
        }
        std::vector<torch::Tensor> q_blocks =
            strict_decode_delta_blocks(
                delta_blocks,
                r_blocks,
                model,
                meta,
                nglr_device
            );

        for (size_t j = i; j < next_i; ++j) {
            const size_t local = j - i;
            const BlockSlice& sl = slices[j];

            torch::Tensor corrected_block =
                (r_blocks[local] +
                 q_blocks[local].to(torch::kCPU).to(torch::kFloat32) * meta.step)
                    * meta.scale + meta.mean;

            corrected.index_put_(
                {
                    sl.b,
                    sl.c,
                    torch::indexing::Slice(sl.t0, sl.t1),
                    torch::indexing::Slice(sl.h0, sl.h1),
                    torch::indexing::Slice(sl.w0, sl.w1)
                },
                corrected_block
            );
        }

        i = next_i;

    }

    return corrected.to(device);
}

}
