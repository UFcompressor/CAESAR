#include "lbrc.h"
/*
 * The LBRC algorithm implemented here is based on:
 * Zhu, L., Ranka, S., and Rangarajan, A.
 * "Residual Modeling for High-Fidelity Learned Compression of Scientific Data."
 * arXiv:2606.05389, 2026.
 * DOI: 10.48550/arXiv.2606.05389
 */
namespace caesar::lbrc {
namespace {
struct Shape5 {
  int64_t B, C, T, H, W;
};

static Shape5 shape_of(const torch::Tensor &t) {
  TORCH_CHECK(t.dim() == 5, "LBRC expects a 5-D tensor (B,C,T,H,W)");
  return {t.size(0), t.size(1), t.size(2), t.size(3), t.size(4)};
}

static inline int64_t idx5(const Shape5 &s, int64_t b, int64_t c, int64_t t,
                           int64_t h, int64_t w) {
  return (((b * s.C + c) * s.T + t) * s.H + h) * s.W + w;
}

struct SliceDesc {
  int64_t b, c, t0, t1, h0, h1, w0, w1;
};

static std::vector<SliceDesc>
make_slices(const Shape5 &S, const std::array<int64_t, 3> &block_size) {
  const int64_t bt = block_size[0];
  const int64_t bh = block_size[1];
  const int64_t bw = block_size[2];
  TORCH_CHECK(bt > 0 && bh > 0 && bw > 0,
              "LBRC: block_size entries must be positive");

  std::vector<SliceDesc> slices;
  for (int64_t b = 0; b < S.B; ++b)
    for (int64_t c = 0; c < S.C; ++c)
      for (int64_t t = 0; t < S.T; t += bt)
        for (int64_t h = 0; h < S.H; h += bh)
          for (int64_t w = 0; w < S.W; w += bw)
            slices.push_back({b, c, t, std::min(t + bt, S.T), h,
                              std::min(h + bh, S.H), w, std::min(w + bw, S.W)});
  return slices;
}

template <class F> static void parallel_for(int64_t n, int workers, F f) {
  if (workers <= 1 || n <= 1) {
    for (int64_t i = 0; i < n; ++i)
      f(i);
    return;
  }
  std::atomic<int64_t> next{0};
  const int64_t W = std::min<int64_t>(workers, n);
  std::vector<std::thread> th;
  th.reserve(W);
  for (int64_t k = 0; k < W; ++k) {
    th.emplace_back([&] {
      while (true) {
        int64_t i = next.fetch_add(1);
        if (i >= n)
          break;
        f(i);
      }
    });
  }
  for (auto &t : th)
    t.join();
}

// todo make zstd muti-threaded like in the GEA maybe ???
static std::vector<uint8_t> zstd_compress(const std::vector<uint8_t> &in,
                                          int level) {
  std::vector<uint8_t> out(ZSTD_compressBound(in.size()));
  size_t n = ZSTD_compress(out.data(), out.size(), in.data(), in.size(), level);
  if (ZSTD_isError(n))
    throw std::runtime_error(std::string("zstd compress: ") +
                             ZSTD_getErrorName(n));
  out.resize(n);
  return out;
}

static std::vector<uint8_t> zstd_decompress(const std::vector<uint8_t> &in,
                                            size_t nout) {
  std::vector<uint8_t> out(nout);
  size_t n = ZSTD_decompress(out.data(), nout, in.data(), in.size());
  if (ZSTD_isError(n) || n != nout)
    throw std::runtime_error("zstd decompress failed");
  return out;
}

static inline uint32_t zzenc(int32_t v) {
  return v >= 0 ? static_cast<uint32_t>(v) * 2u
                : static_cast<uint32_t>(-2LL * v - 1LL);
}

static inline int32_t zzdec(uint32_t u) {
  return (u & 1u) ? -static_cast<int32_t>((u + 1u) >> 1)
                  : static_cast<int32_t>(u >> 1);
}

static void pack_bits(const std::vector<uint32_t> &v, uint32_t bit,
                      std::vector<uint8_t> &out) {
  out.assign((v.size() + 7) / 8, 0);
  for (size_t i = 0; i < v.size(); ++i)
    if ((v[i] >> bit) & 1u)
      out[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
}

static void unpack_bits(const std::vector<uint8_t> &p, uint32_t bit,
                        std::vector<uint32_t> &out) {
  for (size_t i = 0; i < out.size(); ++i)
    if ((p[i / 8] >> (i % 8)) & 1u)
      out[i] |= 1u << bit;
}

static std::vector<int32_t> lorenzo_3d(const std::vector<int32_t> &q, int64_t T,
                                       int64_t H, int64_t W) {
  auto idx = [&](int64_t t, int64_t h, int64_t w) -> int64_t {
    return (t * H + h) * W + w;
  };
  std::vector<int32_t> d(q.size());
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < H; ++h)
      for (int64_t w = 0; w < W; ++w) {
        int64_t v = q[idx(t, h, w)];
        if (t > 0)
          v -= q[idx(t - 1, h, w)];
        if (h > 0)
          v -= q[idx(t, h - 1, w)];
        if (w > 0)
          v -= q[idx(t, h, w - 1)];
        if (t > 0 && h > 0)
          v += q[idx(t - 1, h - 1, w)];
        if (t > 0 && w > 0)
          v += q[idx(t - 1, h, w - 1)];
        if (h > 0 && w > 0)
          v += q[idx(t, h - 1, w - 1)];
        if (t > 0 && h > 0 && w > 0)
          v -= q[idx(t - 1, h - 1, w - 1)];
        d[idx(t, h, w)] = static_cast<int32_t>(v);
      }
  return d;
}

// Inverse: three sequential prefix-sum passes W, H, T
static std::vector<int32_t> inv_lorenzo_3d(std::vector<int32_t> q, int64_t T,
                                           int64_t H, int64_t W) {
  auto idx = [&](int64_t t, int64_t h, int64_t w) -> int64_t {
    return (t * H + h) * W + w;
  };
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < H; ++h)
      for (int64_t w = 1; w < W; ++w)
        q[idx(t, h, w)] += q[idx(t, h, w - 1)];
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 1; h < H; ++h)
      for (int64_t w = 0; w < W; ++w)
        q[idx(t, h, w)] += q[idx(t, h - 1, w)];
  for (int64_t t = 1; t < T; ++t)
    for (int64_t h = 0; h < H; ++h)
      for (int64_t w = 0; w < W; ++w)
        q[idx(t, h, w)] += q[idx(t - 1, h, w)];
  return q;
}

static double block_sse(const float *x, const float *x0n, const float *r,
                        int64_t N, double step, float mean, float scale) {
  const float st = static_cast<float>(step);
  double sse = 0.0;
  for (int64_t i = 0; i < N; ++i) {
    float q = std::nearbyint(r[i] / st);
    float y = (x0n[i] + q * st) * scale + mean;
    float e = (x[i] - y) / scale;
    sse += static_cast<double>(e) * static_cast<double>(e);
  }
  return sse;
}

struct QuantResult {
  double step;
  std::vector<int32_t> q;
  double sse;
};

static QuantResult quantize_block(const float *x, const float *x0n,
                                  const float *r, int64_t N,
                                  double target_nrmse, int qiter, float mean,
                                  float scale) {
  const double target_sse =
      target_nrmse * target_nrmse * static_cast<double>(N);

  double sse0 = 0.0;
  for (int64_t i = 0; i < N; ++i) {
    float y = x0n[i] * scale + mean;
    float e = (x[i] - y) / scale;
    sse0 += static_cast<double>(e) * static_cast<double>(e);
  }

  if (sse0 <= target_sse)
    return {1.0, std::vector<int32_t>(N, 0), sse0};

  double low = 0.0;
  double high = std::max(target_nrmse * std::sqrt(12.0), 1e-12);
  while (block_sse(x, x0n, r, N, high, mean, scale) <= target_sse) {
    low = high;
    high *= 2.0;
  }
  for (int i = 0; i < qiter; ++i) {
    double mid = 0.5 * (low + high);
    if (block_sse(x, x0n, r, N, mid, mean, scale) <= target_sse)
      low = mid;
    else
      high = mid;
  }

  double step = std::max(low, 1e-12);
  double sse = block_sse(x, x0n, r, N, step, mean, scale);

  std::vector<int32_t> q(N);
  const float st = static_cast<float>(step);
  for (int64_t i = 0; i < N; ++i)
    q[i] = static_cast<int32_t>(std::nearbyint(r[i] / st));

  return {step, std::move(q), sse};
}

static LBRCBlock encode_block(const float *x_ptr, const float *x0n_ptr,
                              const float *r_ptr, const Shape5 &S, int64_t b,
                              int64_t c, int64_t t0, int64_t t1, int64_t h0,
                              int64_t h1, int64_t w0, int64_t w1,
                              double target_nrmse, int zstd_level, int qiter,
                              float mean, float scale) {
  const int64_t T = t1 - t0;
  const int64_t H = h1 - h0;
  const int64_t W = w1 - w0;
  const int64_t N = T * H * W;

  std::vector<float> xb(N), x0b(N), rb(N);
  int64_t p = 0;
  for (int64_t t = t0; t < t1; ++t)
    for (int64_t h = h0; h < h1; ++h)
      for (int64_t w = w0; w < w1; ++w, ++p) {
        int64_t id = idx5(S, b, c, t, h, w);
        xb[p] = x_ptr[id];
        x0b[p] = x0n_ptr[id];
        rb[p] = r_ptr[id];
      }

  auto qr = quantize_block(xb.data(), x0b.data(), rb.data(), N, target_nrmse,
                           qiter, mean, scale);

  auto d = lorenzo_3d(qr.q, T, H, W);
  std::vector<uint32_t> zz(d.size());
  uint32_t mx = 0;
  for (size_t i = 0; i < d.size(); ++i) {
    zz[i] = zzenc(d[i]);
    mx = std::max(mx, zz[i]);
  }
  uint32_t bits = 1;
  while ((mx >> bits) != 0)
    ++bits;

  LBRCBlock blk;
  blk.step = qr.step;
  blk.bit_count = bits;
  blk.streams.resize(bits);

  std::vector<uint8_t> packed;
  for (uint32_t bit = 0; bit < bits; ++bit) {
    pack_bits(zz, bit, packed);
    blk.streams[bit] = zstd_compress(packed, zstd_level);
  }

  return blk;
}

static void compress_cpu(const torch::Tensor &original,
                         const torch::Tensor &recons, double target_nrmse,
                         LBRCMetaData &meta, std::vector<LBRCBlock> &blocks,
                         int workers) {
  TORCH_CHECK(original.dtype() == torch::kFloat32 &&
                  recons.dtype() == torch::kFloat32,
              "LBRC compress: tensors must be float32");
  TORCH_CHECK(original.sizes() == recons.sizes(),
              "LBRC compress: shape mismatch");
  TORCH_CHECK(original.dim() == 5,
              "LBRC compress: expected 5-D tensor (B,C,T,H,W)");

  const Shape5 S = shape_of(original);

  torch::Tensor orig_c = original.contiguous();
  torch::Tensor rec_c = recons.contiguous();
  const float *x_ptr = orig_c.data_ptr<float>();
  const float *x0_ptr = rec_c.data_ptr<float>();
  const int64_t N_total = orig_c.numel();

  float x_mean = orig_c.mean().item<float>();
  float x_max = orig_c.max().item<float>();
  float x_min = orig_c.min().item<float>();
  float scale = x_max - x_min;
  TORCH_CHECK(scale > 0, "LBRC compress: zero data range");

  std::vector<float> x0n(N_total), r(N_total);
  for (int64_t i = 0; i < N_total; ++i) {
    float xn = (x_ptr[i] - x_mean) / scale;
    x0n[i] = (x0_ptr[i] - x_mean) / scale;
    r[i] = xn - x0n[i];
  }

  const auto slices = make_slices(S, meta.block_size);
  if (workers <= 0)
    workers = get_allocated_cores();

  int zstd_level = 21;
  int quantity_iter = 16;
  blocks.resize(slices.size());

  parallel_for(static_cast<int64_t>(slices.size()), workers, [&](int64_t i) {
    const auto &sl = slices[i];
    blocks[i] = encode_block(x_ptr, x0n.data(), r.data(), S, sl.b, sl.c, sl.t0,
                             sl.t1, sl.h0, sl.h1, sl.w0, sl.w1, target_nrmse,
                             zstd_level, quantity_iter, x_mean, scale);
  });

  meta.x_mean = x_mean;
  meta.scale = scale;
  meta.lbrc_correction_occur = true;
}

static torch::Tensor decompress_cpu(const torch::Tensor &recons,
                                    const LBRCMetaData &meta,
                                    const std::vector<LBRCBlock> &blocks,
                                    int workers) {
  TORCH_CHECK(recons.dtype() == torch::kFloat32,
              "LBRC decompress: recons must be float32");
  TORCH_CHECK(recons.dim() == 5,
              "LBRC decompress: expected 5-D tensor (B,C,T,H,W)");

  const Shape5 S = shape_of(recons);
  torch::Tensor rec_c = recons.contiguous();
  const float *x0_ptr = rec_c.data_ptr<float>();
  const int64_t N_total = rec_c.numel();

  const float x_mean = meta.x_mean;
  const float scale = meta.scale;

  std::vector<float> x0n(N_total);
  for (int64_t i = 0; i < N_total; ++i)
    x0n[i] = (x0_ptr[i] - x_mean) / scale;

  torch::Tensor out = torch::empty_like(rec_c);
  float *out_ptr = out.data_ptr<float>();

  const auto slices = make_slices(S, meta.block_size);
  TORCH_CHECK(slices.size() == blocks.size(),
              "LBRC decompress: derived block grid (", slices.size(),
              ") does not match stored block count (", blocks.size(), ")");

  if (workers <= 0)
    workers = get_allocated_cores();

  parallel_for(static_cast<int64_t>(blocks.size()), workers, [&](int64_t bi) {
    const LBRCBlock &blk = blocks[bi];
    const SliceDesc &sl = slices[bi];
    const int64_t T = sl.t1 - sl.t0;
    const int64_t H = sl.h1 - sl.h0;
    const int64_t W = sl.w1 - sl.w0;
    const int64_t N = T * H * W;

    std::vector<uint32_t> zz(N, 0u);
    for (uint32_t bit = 0; bit < blk.bit_count; ++bit) {
      const size_t packed_bytes = static_cast<size_t>((N + 7) / 8);
      auto packed = zstd_decompress(blk.streams[bit], packed_bytes);
      unpack_bits(packed, bit, zz);
    }

    std::vector<int32_t> d(N);
    for (int64_t i = 0; i < N; ++i)
      d[i] = zzdec(zz[i]);
    auto q = inv_lorenzo_3d(std::move(d), T, H, W);

    const float st = static_cast<float>(blk.step);
    int64_t p = 0;
    for (int64_t t = sl.t0; t < sl.t1; ++t)
      for (int64_t h = sl.h0; h < sl.h1; ++h)
        for (int64_t w = sl.w0; w < sl.w1; ++w, ++p) {
          int64_t id = idx5(S, sl.b, sl.c, t, h, w);
          out_ptr[id] =
              (x0n[id] + static_cast<float>(q[p]) * st) * scale + x_mean;
        }
  });

  return out;
}

} // anonymous namespace

// Everything from normalize through bit-packing stays on device as batched
// tensor ops. Blocks are extracted with `unfold` after padding T,H,W up to a
// multiple of block_size; a bool mask rides alongside so padded voxels never
// affect SSE, bit_count, or the reconstructed output.

namespace gpu {

// x: [B,C,T,H,W] -> [Nblocks, bt,bh,bw]. Requires block_size to evenly
// divide (T,H,W) -- no padding, no mask, no boundary-block logic anywhere
// downstream. If your data doesn't divide evenly, pick a block_size that
// does (e.g. a divisor of the grid dims) rather than adding padding back.
static torch::Tensor to_blocks(const torch::Tensor &x,
                               const std::array<int64_t, 3> &block_size) {
  TORCH_CHECK(
      x.size(2) % block_size[0] == 0 && x.size(3) % block_size[1] == 0 &&
          x.size(4) % block_size[2] == 0,
      "LBRC (gpu): block_size must evenly divide (T,H,W); got T=", x.size(2),
      " H=", x.size(3), " W=", x.size(4), " block_size=(", block_size[0], ",",
      block_size[1], ",", block_size[2], ")");
  auto u = x.unfold(2, block_size[0], block_size[0])
               .unfold(3, block_size[1], block_size[1])
               .unfold(4, block_size[2], block_size[2]);
  return u.contiguous().view({-1, block_size[0], block_size[1], block_size[2]});
}

// Inverse of to_blocks.
static torch::Tensor from_blocks(const torch::Tensor &blocks_t, const Shape5 &S,
                                 const std::array<int64_t, 3> &block_size) {
  int64_t bt = block_size[0], bh = block_size[1], bw = block_size[2];
  int64_t Tb = S.T / bt, Hb = S.H / bh, Wb = S.W / bw;
  auto v = blocks_t.view({S.B, S.C, Tb, Hb, Wb, bt, bh, bw});
  return v.permute({0, 1, 2, 5, 3, 6, 4, 7})
      .contiguous()
      .view({S.B, S.C, S.T, S.H, S.W});
}

// q: [Nblocks, bt,bh,bw] int32 -> mixed 3rd-difference (batched Lorenzo).
static torch::Tensor lorenzo_3d(const torch::Tensor &q) {
  auto shift1 = [](const torch::Tensor &t, int64_t dim) {
    auto z = torch::zeros_like(t);
    z.narrow(dim, 1, t.size(dim) - 1).copy_(t.narrow(dim, 0, t.size(dim) - 1));
    return z;
  };
  auto t_ = shift1(q, 1), h_ = shift1(q, 2), w_ = shift1(q, 3);
  auto th = shift1(t_, 2), tw = shift1(t_, 3), hw = shift1(h_, 3);
  auto thw = shift1(th, 3);
  return q - t_ - h_ - w_ + th + tw + hw - thw;
}

static torch::Tensor inv_lorenzo_3d(torch::Tensor q) {
  q = q.cumsum(3);
  q = q.cumsum(2);
  q = q.cumsum(1);
  return q;
}

static torch::Tensor zigzag_encode(const torch::Tensor &d /*int32*/) {
  auto d64 = d.to(torch::kInt64);
  return torch::where(d64 >= 0, d64 * 2, -2 * d64 - 1);
}

static torch::Tensor zigzag_decode(const torch::Tensor &zz /*int64, >=0*/) {
  auto odd = (zz % 2) == 1;
  return torch::where(odd, -(zz + 1) / 2, zz / 2).to(torch::kInt32);
}

static torch::Tensor pack_bits(const torch::Tensor &bits01) {
  auto sizes = bits01.sizes().vec();
  int64_t N = sizes.back();
  TORCH_CHECK(N % 8 == 0, "pack_bits: last dim must be a multiple of 8");
  auto grouped = bits01.reshape({-1, N / 8, 8}).to(torch::kInt32);
  auto w = torch::tensor({1, 2, 4, 8, 16, 32, 64, 128}, grouped.options());
  auto packed = (grouped * w).sum(-1).to(torch::kUInt8);
  sizes.back() = N / 8;
  return packed.reshape(sizes);
}

static torch::Tensor unpack_bits(const torch::Tensor &packed, int64_t N) {
  auto p32 = packed.to(torch::kInt32).unsqueeze(-1);
  auto w = torch::tensor({1, 2, 4, 8, 16, 32, 64, 128}, p32.options());
  auto bits = ((p32 & w) > 0).to(torch::kUInt8);
  auto sizes = packed.sizes().vec();
  sizes.back() = N;
  return bits.reshape(sizes);
}

// Batched binary-search quantization across all Nblocks at once.
static std::pair<torch::Tensor, torch::Tensor>
quantize_batched(const torch::Tensor &x, const torch::Tensor &x0n,
                 const torch::Tensor &r, double target_nrmse, int qiter,
                 float mean, float scale) {
  int64_t Nb = x.size(0);
  int64_t N = x.size(1) * x.size(2) * x.size(3);
  double target_sse = target_nrmse * target_nrmse * static_cast<double>(N);

  // One lambda covers both sse0 (step so large q rounds to 0 everywhere,
  // i.e. the uncorrected base-recon error) and every bisection probe.
  auto sse_at = [&](const torch::Tensor &step4) {
    auto q = torch::round(r / step4);
    auto y = (x0n + q * step4) * scale + mean;
    auto e = (x - y) / scale;
    return (e * e).sum({1, 2, 3});
  };

  auto huge = torch::full({Nb, 1, 1, 1}, 1e30, x.options());
  auto sse0 = sse_at(huge);
  auto zero_correction = sse0 <= target_sse; // [Nblocks] bool

  auto opts = x.options();
  auto low = torch::zeros({Nb}, opts);
  const double init_high =
      target_nrmse * 3.4641016151377544; // target*sqrt(12), literal constant
  auto high = torch::full({Nb}, init_high > 1e-12 ? init_high : 1e-12, opts);

  const int max_growth =
      40; // fixed iters, no host sync -- same style as qiter below
  for (int g = 0; g < max_growth; ++g) {
    auto passing = sse_at(high.view({Nb, 1, 1, 1})) <= target_sse;
    low = torch::where(passing, high, low);
    high = torch::where(passing, high * 2.0, high);
  }

  for (int i = 0; i < qiter; ++i) {
    auto mid = 0.5 * (low + high);
    auto ok = sse_at(mid.view({Nb, 1, 1, 1})) <= target_sse;
    low = torch::where(ok, mid, low);
    high = torch::where(ok, high, mid);
  }

  auto step = torch::clamp_min(low, 1e-12);
  step = torch::where(zero_correction, torch::ones_like(step), step);

  auto step4 = step.view({Nb, 1, 1, 1});
  auto q = torch::round(r / step4).to(torch::kInt32);
  q = torch::where(zero_correction.view({Nb, 1, 1, 1}), torch::zeros_like(q),
                   q);

  return {step, q};
}

static std::vector<std::vector<uint8_t>> compress_planes(
    const torch::Tensor &planes /*[Nblocks, packed_bytes] uint8, CUDA*/,
    int zstd_level, int workers) {
#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
  (void)zstd_level;
  (void)workers;
  // Row of a contiguous [Nb, bytes] tensor is already contiguous -- this is
  // pointer slicing, not a copy, and nothing here touches the host.
  int64_t Nb = planes.size(0);
  std::vector<torch::Tensor> rows(Nb);
  for (int64_t i = 0; i < Nb; ++i)
    rows[i] = planes[i];

  auto results = nvcomp_batch_compress(rows); // one batched device call

  std::vector<std::vector<uint8_t>> out(Nb);
  for (int64_t i = 0; i < Nb; ++i) {
    auto &t = results[i].compressed; // CPU kUInt8 tensor, already D2H'd
    const uint8_t *p = t.data_ptr<uint8_t>();
    out[i].assign(p, p + t.numel());
  }
  return out;
#else
  auto planes_cpu = planes.to(torch::kCPU).contiguous();
  int64_t Nb = planes_cpu.size(0), bytes = planes_cpu.size(1);
  const uint8_t *base = planes_cpu.data_ptr<uint8_t>();
  std::vector<std::vector<uint8_t>> out(Nb);
  parallel_for(Nb, workers, [&](int64_t i) {
    std::vector<uint8_t> raw(base + i * bytes, base + (i + 1) * bytes);
    out[i] = zstd_compress(raw, zstd_level);
  });
  return out;
#endif
}

static torch::Tensor
decompress_planes(const std::vector<std::vector<uint8_t>> &compressed,
                  int64_t packed_bytes, const torch::TensorOptions &gpu_opts,
                  int workers) {
#if defined(USE_CUDA) && defined(ENABLE_NVCOMP)
  (void)workers;
  int64_t Nb = static_cast<int64_t>(compressed.size());
  std::vector<const uint8_t *> comp_ptrs(Nb);
  std::vector<size_t> comp_sizes(Nb);
  std::vector<size_t> decomp_sizes(Nb, static_cast<size_t>(packed_bytes));
  for (int64_t i = 0; i < Nb; ++i) {
    comp_ptrs[i] = compressed[i].data();
    comp_sizes[i] = compressed[i].size();
    if (compressed[i].empty())
      decomp_sizes[i] = 0; // no plane at this bit
  }

  auto results = nvcomp_batch_decompress(comp_ptrs, comp_sizes, decomp_sizes);

  torch::Tensor out_cpu = torch::zeros({Nb, packed_bytes}, torch::kUInt8);
  uint8_t *base = out_cpu.data_ptr<uint8_t>();
  for (int64_t i = 0; i < Nb; ++i) {
    if (results[i].empty())
      continue; // stays 0, same as zstd path
    std::memcpy(base + i * packed_bytes, results[i].data(), packed_bytes);
  }
  return out_cpu.to(gpu_opts.device());
#else
  int64_t Nb = static_cast<int64_t>(compressed.size());
  torch::Tensor out_cpu = torch::zeros({Nb, packed_bytes}, torch::kUInt8);
  uint8_t *base = out_cpu.data_ptr<uint8_t>();
  parallel_for(Nb, workers, [&](int64_t i) {
    if (compressed[i].empty())
      return; // block has no plane at this bit -> stays 0
    auto bytes =
        zstd_decompress(compressed[i], static_cast<size_t>(packed_bytes));
    std::memcpy(base + i * packed_bytes, bytes.data(), packed_bytes);
  });
  return out_cpu.to(gpu_opts.device());
#endif
}

static void compress_gpu(const torch::Tensor &original,
                         const torch::Tensor &recons, double target_nrmse,
                         LBRCMetaData &meta, std::vector<LBRCBlock> &blocks,
                         int workers) {
  TORCH_CHECK(original.dtype() == torch::kFloat32 &&
                  recons.dtype() == torch::kFloat32,
              "LBRC compress (gpu): tensors must be float32");
  TORCH_CHECK(original.sizes() == recons.sizes(),
              "LBRC compress (gpu): shape mismatch");
  TORCH_CHECK(original.dim() == 5,
              "LBRC compress (gpu): expected 5-D tensor (B,C,T,H,W)");

  const Shape5 S = shape_of(original);
  auto orig_c = original.contiguous();
  auto rec_c = recons.contiguous();

  auto stats = torch::stack({orig_c.mean(), orig_c.amax(), orig_c.amin()})
                   .to(torch::kCPU);
  const float x_mean = stats[0].item<float>();
  const float scale = stats[1].item<float>() - stats[2].item<float>();
  TORCH_CHECK(scale > 0, "LBRC compress (gpu): zero data range");

  auto x0n = (rec_c - x_mean) / scale;
  auto r = (orig_c - x_mean) / scale - x0n;

  auto x_blk = to_blocks(orig_c, meta.block_size);
  auto x0n_blk = to_blocks(x0n, meta.block_size);
  auto r_blk = to_blocks(r, meta.block_size);

  int qiter = 16;
  auto [step, q] = quantize_batched(x_blk, x0n_blk, r_blk, target_nrmse, qiter,
                                    x_mean, scale);

  auto d = lorenzo_3d(q);
  auto zz = zigzag_encode(d);                   // int64, >=0
  auto mx = zz.amax({1, 2, 3}).to(torch::kCPU); // host sync, metadata only

  int64_t Nb = q.size(0);
  int64_t bt = meta.block_size[0], bh = meta.block_size[1],
          bw = meta.block_size[2];
  int64_t N = bt * bh * bw;
  int64_t N8 = (N + 7) / 8 * 8; // pad flattened voxel count to a byte boundary

  auto mx_acc = mx.accessor<int64_t, 1>();
  std::vector<uint32_t> bit_count(Nb);
  uint32_t max_bits = 1;
  for (int64_t i = 0; i < Nb; ++i) {
    uint32_t bits = 1;
    int64_t v = mx_acc[i];
    while ((v >> bits) != 0)
      ++bits;
    bit_count[i] = bits;
    max_bits = std::max(max_bits, bits);
  }

  auto zz_flat = zz.view({Nb, N});
  if (N8 != N)
    zz_flat = torch::nn::functional::pad(
        zz_flat, torch::nn::functional::PadFuncOptions({0, N8 - N}));

  blocks.resize(Nb);
  for (auto &blk : blocks)
    blk.streams.clear();

  int zstd_level = 21;
  if (workers <= 0)
    workers = get_allocated_cores();

  for (uint32_t bit = 0; bit < max_bits; ++bit) {
    // no Tensor>>scalar operator in LibTorch -- bit b of a nonnegative
    // int is (x / 2^b) % 2, same result via arithmetic.
    int64_t shift = static_cast<int64_t>(1) << bit;
    auto bits01 = torch::remainder(torch::div(zz_flat, shift,
                                              /*rounding_mode=*/"trunc"),
                                   2)
                      .to(torch::kUInt8);
    auto packed = pack_bits(bits01);
    auto planes = compress_planes(packed, zstd_level, workers);
    for (int64_t i = 0; i < Nb; ++i)
      if (bit < bit_count[i])
        blocks[i].streams.push_back(std::move(planes[i]));
  }

  auto step_cpu = step.to(torch::kFloat64).to(torch::kCPU);
  auto step_acc = step_cpu.accessor<double, 1>();
  for (int64_t i = 0; i < Nb; ++i) {
    blocks[i].bit_count = bit_count[i];
    blocks[i].step = step_acc[i];
  }

  meta.x_mean = x_mean;
  meta.scale = scale;
  meta.lbrc_correction_occur = true;
}

static torch::Tensor decompress_gpu(const torch::Tensor &recons,
                                    const LBRCMetaData &meta,
                                    const std::vector<LBRCBlock> &blocks,
                                    int workers) {
  TORCH_CHECK(recons.dtype() == torch::kFloat32,
              "LBRC decompress (gpu): recons must be float32");
  TORCH_CHECK(recons.dim() == 5,
              "LBRC decompress (gpu): expected 5-D tensor (B,C,T,H,W)");

  const Shape5 S = shape_of(recons);
  auto rec_c = recons.contiguous();
  auto x0n = (rec_c - meta.x_mean) / meta.scale;
  auto x0n_blk = to_blocks(x0n, meta.block_size);

  int64_t Nb = static_cast<int64_t>(blocks.size());
  int64_t bt = meta.block_size[0], bh = meta.block_size[1],
          bw = meta.block_size[2];
  int64_t N = bt * bh * bw;
  int64_t N8 = (N + 7) / 8 * 8;
  int64_t packed_bytes = N8 / 8;

  uint32_t max_bits = 0;
  for (auto &b : blocks)
    max_bits = std::max(max_bits, b.bit_count);

  if (workers <= 0)
    workers = get_allocated_cores();
  auto zz_flat = torch::zeros({Nb, N8}, x0n_blk.options().dtype(torch::kInt64));

  for (uint32_t bit = 0; bit < max_bits; ++bit) {
    std::vector<std::vector<uint8_t>> plane_bytes(Nb);
    for (int64_t i = 0; i < Nb; ++i)
      if (bit < blocks[i].bit_count)
        plane_bytes[i] = blocks[i].streams[bit];

    auto packed =
        decompress_planes(plane_bytes, packed_bytes,
                          x0n_blk.options().dtype(torch::kUInt8), workers);
    auto bits01 = unpack_bits(packed, N8).to(torch::kInt64);
    // bits are disjoint per plane, so += is equivalent to bitwise-OR
    // and avoids relying on Tensor |= support.
    zz_flat += bits01 * (static_cast<int64_t>(1) << bit);
  }

  auto zz = zz_flat.narrow(1, 0, N).view({Nb, bt, bh, bw});
  auto d = zigzag_decode(zz);
  auto q = inv_lorenzo_3d(d);

  std::vector<double> steps(Nb);
  for (int64_t i = 0; i < Nb; ++i)
    steps[i] = blocks[i].step;
  auto step_t = torch::from_blob(steps.data(), {Nb}, torch::kFloat64)
                    .to(q.options().dtype(torch::kFloat32).device(q.device()))
                    .view({Nb, 1, 1, 1});

  auto out_blk =
      (x0n_blk + q.to(torch::kFloat32) * step_t) * meta.scale + meta.x_mean;
  return from_blocks(out_blk, S, meta.block_size);
}

} // namespace gpu

void compress(const torch::Tensor &original, const torch::Tensor &recons,
              double target_nrmse, LBRCMetaData &meta,
              std::vector<LBRCBlock> &blocks, int workers) {
  // only suport nvida/amd
  if (torch::cuda::is_available()) {
    gpu::compress_gpu(original, recons, target_nrmse, meta, blocks, workers);
  } else {
    compress_cpu(original, recons, target_nrmse, meta, blocks, workers);
  }
}

torch::Tensor decompress(const torch::Tensor &recons, const LBRCMetaData &meta,
                         const std::vector<LBRCBlock> &blocks, int workers) {
  if (recons.is_cuda()) {
    return gpu::decompress_gpu(recons, meta, blocks, workers);
  }
  return decompress_cpu(recons, meta, blocks, workers);
}

} // namespace caesar::lbrc
