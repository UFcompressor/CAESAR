#include "nglr.h"
namespace nglr {
namespace {

constexpr std::array<char, 8> kMagic{{'N', 'G', 'L', 'R', 'M', 'E', 'M', '1'}};
constexpr uint32_t kFormatVersion = 1;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("nglr: " + message);
}

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

c10::Device select_model_device() {
  return torch::cuda::is_available() ? c10::Device(c10::kCUDA)
                                     : c10::Device(c10::kCPU);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

class ByteWriter {
public:
  template <class T> void put(const T &value) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "binary field required");
    const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
    bytes_.insert(bytes_.end(), bytes, bytes + sizeof(T));
  }
  void put_bytes(const void *data, size_t count) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    bytes_.insert(bytes_.end(), bytes, bytes + count);
  }
  std::vector<uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<uint8_t> bytes_;
};

class ByteReader {
public:
  explicit ByteReader(const std::vector<uint8_t> &bytes) : bytes_(bytes) {}
  template <class T> T get() {
    static_assert(std::is_trivially_copyable<T>::value,
                  "binary field required");
    need(sizeof(T));
    T value;
    std::memcpy(&value, bytes_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return value;
  }
  std::vector<uint8_t> get_bytes(size_t count) {
    need(count);
    std::vector<uint8_t> out(bytes_.begin() + static_cast<std::ptrdiff_t>(pos_),
                             bytes_.begin() +
                                 static_cast<std::ptrdiff_t>(pos_ + count));
    pos_ += count;
    return out;
  }
  bool at_end() const { return pos_ == bytes_.size(); }

private:
  void need(size_t count) const {
    if (count > bytes_.size() - pos_)
      fail("truncated in-memory correction payload");
  }
  const std::vector<uint8_t> &bytes_;
  size_t pos_ = 0;
};

std::vector<uint8_t> zstd_compress(const std::vector<uint8_t> &source,
                                   int level) {
  const size_t bound = ZSTD_compressBound(source.size());
  std::vector<uint8_t> out(bound);
  const size_t written = ZSTD_compress(out.data(), out.size(), source.data(),
                                       source.size(), level);
  if (ZSTD_isError(written))
    fail(std::string("zstd compression failed: ") + ZSTD_getErrorName(written));
  out.resize(written);
  return out;
}

std::vector<uint8_t> zstd_decompress(const std::vector<uint8_t> &source,
                                     size_t output_size) {
  std::vector<uint8_t> out(output_size);
  const size_t written =
      ZSTD_decompress(out.data(), out.size(), source.data(), source.size());
  if (ZSTD_isError(written) || written != output_size)
    fail("zstd decompression failed or produced an unexpected size");
  return out;
}

struct BlockSlice {
  int64_t b, c, t0, t1, h0, h1, w0, w1;
};

std::vector<BlockSlice> block_slices(const torch::Tensor &x,
                                     const NGLRMeta &meta) {
  require(x.dim() == 5, "expected a 5-D [B,C,T,H,W] tensor");
  require(meta.block_t > 0 && meta.block_h > 0 && meta.block_w > 0,
          "block dimensions must be positive");
  std::vector<BlockSlice> out;
  for (int64_t b = 0; b < x.size(0); ++b)
    for (int64_t c = 0; c < x.size(1); ++c)
      for (int64_t t = 0; t < x.size(2); t += meta.block_t)
        for (int64_t h = 0; h < x.size(3); h += meta.block_h)
          for (int64_t w = 0; w < x.size(4); w += meta.block_w)
            out.push_back({b, c, t, std::min(t + meta.block_t, x.size(2)), h,
                           std::min(h + meta.block_h, x.size(3)), w,
                           std::min(w + meta.block_w, x.size(4))});
  return out;
}

torch::Tensor slice_block(const torch::Tensor &x, const BlockSlice &s) {
  using torch::indexing::Slice;
  return x
      .index({Slice(s.b, s.b + 1), Slice(s.c, s.c + 1), Slice(s.t0, s.t1),
              Slice(s.h0, s.h1), Slice(s.w0, s.w1)})
      .contiguous();
}

void put_block(torch::Tensor &x, const BlockSlice &s,
               const torch::Tensor &block) {
  using torch::indexing::Slice;
  x.index_put_({Slice(s.b, s.b + 1), Slice(s.c, s.c + 1), Slice(s.t0, s.t1),
                Slice(s.h0, s.h1), Slice(s.w0, s.w1)},
               block);
}

using Diagonal = std::array<torch::Tensor, 3>;
std::vector<Diagonal> diagonal_indices(int64_t t_size, int64_t h_size,
                                       int64_t w_size) {
  std::vector<Diagonal> result;
  const auto options = torch::TensorOptions().dtype(torch::kInt64);
  for (int64_t sum = 0; sum <= t_size + h_size + w_size - 3; ++sum) {
    std::vector<int64_t> ts, hs, ws;
    for (int64_t t = std::max<int64_t>(0, sum - (h_size - 1) - (w_size - 1));
         t <= std::min(t_size - 1, sum); ++t) {
      for (int64_t h = std::max<int64_t>(0, sum - t - (w_size - 1));
           h <= std::min(h_size - 1, sum - t); ++h) {
        ts.push_back(t);
        hs.push_back(h);
        ws.push_back(sum - t - h);
      }
    }
    if (!ts.empty())
      result.push_back({torch::tensor(ts, options), torch::tensor(hs, options),
                        torch::tensor(ws, options)});
  }
  return result;
}

torch::Tensor gather_with_zero_boundary(const torch::Tensor &qhat,
                                        const torch::Tensor &ts,
                                        const torch::Tensor &hs,
                                        const torch::Tensor &ws) {
  using torch::indexing::Slice;
  const auto valid = (ts >= 0) & (hs >= 0) & (ws >= 0);
  auto values =
      qhat.index({Slice(), ts.clamp_min(0), hs.clamp_min(0), ws.clamp_min(0)});
  return values * valid.to(values.dtype()).unsqueeze(0);
}

std::pair<torch::Tensor, torch::Tensor>
lorenzo_context(const torch::Tensor &qhat, const torch::Tensor &ts,
                const torch::Tensor &hs, const torch::Tensor &ws,
                double scale) {
  const auto v1 = gather_with_zero_boundary(qhat, ts - 1, hs, ws);
  const auto v2 = gather_with_zero_boundary(qhat, ts, hs - 1, ws);
  const auto v3 = gather_with_zero_boundary(qhat, ts, hs, ws - 1);
  const auto v4 = gather_with_zero_boundary(qhat, ts - 1, hs - 1, ws);
  const auto v5 = gather_with_zero_boundary(qhat, ts - 1, hs, ws - 1);
  const auto v6 = gather_with_zero_boundary(qhat, ts, hs - 1, ws - 1);
  const auto v7 = gather_with_zero_boundary(qhat, ts - 1, hs - 1, ws - 1);
  const auto prediction = v1 + v2 + v3 - v4 - v5 - v6 + v7;
  auto context = torch::stack({v1, v2, v3, v4, v5, v6, v7, prediction}, 2)
                     .to(torch::kFloat32);
  return {context / std::max(scale, 1.0), prediction};
}

torch::Tensor gather_recons_features(const torch::Tensor &features,
                                     const torch::Tensor &ts,
                                     const torch::Tensor &hs,
                                     const torch::Tensor &ws) {
  using torch::indexing::Slice;
  const int64_t channels = features.size(1);
  return features.index({Slice(), Slice(), ts, hs, ws})
      .permute({0, 2, 1})
      .contiguous()
      .view({-1, channels, 1, 1, 1});
}

torch::Tensor strict_delta_encode(const NGLRModel &model,
                                  const torch::Tensor &q_block,
                                  const torch::Tensor &recons_block,
                                  const NGLRMeta &meta) {
  using torch::indexing::Slice;
  const auto device = model.device();
  const int64_t nb = q_block.size(0), t = q_block.size(2), h = q_block.size(3),
                w = q_block.size(4);
  const auto target = q_block.index({Slice(), 0}).to(device, torch::kInt32);
  auto qhat =
      torch::zeros({nb, t, h, w},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto delta = torch::zeros_like(qhat);
  const auto features =
      model.encode_recons(recons_block.to(device, torch::kFloat32));
  for (const auto &diagonal : diagonal_indices(t, h, w)) {
    const auto ts = diagonal[0].to(device), hs = diagonal[1].to(device),
               ws = diagonal[2].to(device);
    const auto context_and_prediction =
        lorenzo_context(qhat, ts, hs, ws, meta.q_context_scale);
    const auto context = context_and_prediction.first.reshape({-1, 8, 1, 1, 1});
    const auto reference =
        torch::round(
            context_and_prediction.second.to(torch::kFloat64) +
            model.forward_from_recons_feature(
                     gather_recons_features(features, ts, hs, ws), context)
                    .reshape({nb, -1})
                    .to(torch::kFloat64) *
                meta.delta_scale)
            .to(torch::kInt32);
    const auto d = target.index({Slice(), ts, hs, ws}) - reference;
    delta.index_put_({Slice(), ts, hs, ws}, d);
    qhat.index_put_({Slice(), ts, hs, ws}, reference + d);
  }
  require(torch::equal(qhat, target),
          "strict encoder reconstruction check failed");
  return delta.to(torch::kCPU).contiguous();
}

torch::Tensor strict_delta_decode(const NGLRModel &model,
                                  const torch::Tensor &delta_block,
                                  const torch::Tensor &recons_block,
                                  const NGLRMeta &meta) {
  using torch::indexing::Slice;
  const auto device = model.device();
  const int64_t nb = delta_block.size(0), t = delta_block.size(1),
                h = delta_block.size(2), w = delta_block.size(3);
  const auto delta = delta_block.to(device, torch::kInt32);
  auto qhat =
      torch::zeros({nb, t, h, w},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));
  const auto features =
      model.encode_recons(recons_block.to(device, torch::kFloat32));
  for (const auto &diagonal : diagonal_indices(t, h, w)) {
    const auto ts = diagonal[0].to(device), hs = diagonal[1].to(device),
               ws = diagonal[2].to(device);
    const auto context_and_prediction =
        lorenzo_context(qhat, ts, hs, ws, meta.q_context_scale);
    const auto context = context_and_prediction.first.reshape({-1, 8, 1, 1, 1});
    const auto reference =
        torch::round(
            context_and_prediction.second.to(torch::kFloat64) +
            model.forward_from_recons_feature(
                     gather_recons_features(features, ts, hs, ws), context)
                    .reshape({nb, -1})
                    .to(torch::kFloat64) *
                meta.delta_scale)
            .to(torch::kInt32);
    qhat.index_put_({Slice(), ts, hs, ws},
                    reference + delta.index({Slice(), ts, hs, ws}));
  }
  return qhat.to(torch::kCPU).contiguous();
}

struct EncodedBlock {
  uint32_t bits;
  uint64_t values;
  std::vector<std::vector<uint8_t>> streams;
};

EncodedBlock encode_bitplanes(const torch::Tensor &delta, int level,
                              NGLREncodeStats &stats) {
  const auto flat = delta.contiguous().view({-1});
  const auto *values = flat.data_ptr<int32_t>();
  const uint64_t count = static_cast<uint64_t>(flat.numel());
  uint32_t max_value = 0;
  for (uint64_t i = 0; i < count; ++i) {
    const uint32_t zigzag = (static_cast<uint32_t>(values[i]) << 1) ^
                            static_cast<uint32_t>(values[i] >> 31);
    max_value = std::max(max_value, zigzag);
    stats.delta_abs_mean += std::abs(static_cast<double>(values[i]));
    stats.delta_zero_count += values[i] == 0;
  }
  uint32_t bit_count = 1;
  while (bit_count < 32 && (max_value >> bit_count) != 0)
    ++bit_count;
  const uint64_t bytes_per_plane = (count + 7) / 8;
  EncodedBlock result{bit_count, count, {}};
  result.streams.reserve(bit_count);
  for (uint32_t bit = 0; bit < bit_count; ++bit) {
    std::vector<uint8_t> packed(bytes_per_plane, 0);
    for (uint64_t i = 0; i < count; ++i) {
      const uint32_t zigzag = (static_cast<uint32_t>(values[i]) << 1) ^
                              static_cast<uint32_t>(values[i] >> 31);
      packed[i / 8] |= static_cast<uint8_t>(((zigzag >> bit) & 1U) << (i % 8));
    }
    stats.uncompressed_bitplane_bytes += packed.size();
    result.streams.push_back(zstd_compress(packed, level));
  }
  return result;
}

torch::Tensor decode_bitplanes(ByteReader &reader, int64_t t, int64_t h,
                               int64_t w) {
  const uint32_t bits = reader.get<uint32_t>();
  const uint64_t count = reader.get<uint64_t>();
  require(bits >= 1 && bits <= 32, "invalid bit count in correction payload");
  require(count == static_cast<uint64_t>(t * h * w),
          "block value count does not match its dimensions");
  std::vector<uint32_t> zigzag(count, 0);
  const size_t raw_size = static_cast<size_t>((count + 7) / 8);
  for (uint32_t bit = 0; bit < bits; ++bit) {
    const uint64_t compressed_size = reader.get<uint64_t>();
    require(compressed_size <= std::numeric_limits<size_t>::max(),
            "compressed stream is too large");
    const auto packed = zstd_decompress(
        reader.get_bytes(static_cast<size_t>(compressed_size)), raw_size);
    // Decode into the same logical element order used by encode.  This is the
    // important bitplane fix: each decoded bit is ORed into zigzag[i], not a
    // discarded temporary tensor/view.
    for (uint64_t i = 0; i < count; ++i)
      zigzag[i] |= static_cast<uint32_t>((packed[i / 8] >> (i % 8)) & 1U)
                   << bit;
  }
  auto output =
      torch::empty({t, h, w}, torch::TensorOptions().dtype(torch::kInt32));
  auto *decoded = output.data_ptr<int32_t>();
  for (uint64_t i = 0; i < count; ++i)
    decoded[i] =
        static_cast<int32_t>((zigzag[i] >> 1) ^ (0U - (zigzag[i] & 1U)));
  return output;
}

void write_header(ByteWriter &writer, const torch::Tensor &x,
                  const NGLRMeta &meta, uint64_t blocks, int zstd_level) {
  writer.put_bytes(kMagic.data(), kMagic.size());
  writer.put(kFormatVersion);
  writer.put(meta.x_mean);
  writer.put(meta.scale);
  writer.put(meta.step);
  writer.put(meta.q_context_scale);
  writer.put(meta.delta_scale);
  writer.put(meta.block_t);
  writer.put(meta.block_h);
  writer.put(meta.block_w);
  for (int i = 0; i < 5; ++i)
    writer.put(x.size(i));
  writer.put(blocks);
  writer.put(static_cast<int32_t>(zstd_level));
}

void check_header(ByteReader &reader, const torch::Tensor &recons,
                  const NGLRMeta &meta) {
  const auto magic = reader.get_bytes(kMagic.size());
  require(std::equal(magic.begin(), magic.end(), kMagic.begin()),
          "bad correction payload magic");
  require(reader.get<uint32_t>() == kFormatVersion,
          "unsupported correction payload version");
  const double x_mean = reader.get<double>(), scale = reader.get<double>(),
               step = reader.get<double>();
  const double q_context_scale = reader.get<double>(),
               delta_scale = reader.get<double>();
  const int64_t bt = reader.get<int64_t>(), bh = reader.get<int64_t>(),
                bw = reader.get<int64_t>();
  require(x_mean == meta.x_mean && scale == meta.scale && step == meta.step &&
              q_context_scale == meta.q_context_scale &&
              delta_scale == meta.delta_scale && bt == meta.block_t &&
              bh == meta.block_h && bw == meta.block_w,
          "payload metadata differs from the loaded model metadata");
  for (int i = 0; i < 5; ++i)
    require(reader.get<int64_t>() == recons.size(i),
            "recons shape differs from correction payload");
}

} // namespace

NGLRMeta load_meta_from_model_dir(const std::string &model_dir) {
  std::ifstream input(model_dir + "/nglr_meta.txt");
  if (!input)
    fail("could not open " + model_dir + "/nglr_meta.txt");
  std::unordered_map<std::string, std::string> values;
  for (std::string line; std::getline(input, line);) {
    const auto equal = line.find('=');
    if (equal != std::string::npos)
      values[trim(line.substr(0, equal))] = trim(line.substr(equal + 1));
  }
  const auto number = [&values](const char *key) {
    const auto it = values.find(key);
    if (it == values.end())
      fail(std::string("metadata missing key ") + key);
    try {
      return std::stod(it->second);
    } catch (...) {
      fail(std::string("metadata key is not numeric: ") + key);
    }
  };
  NGLRMeta meta;
  meta.x_mean = number("x_mean");
  meta.scale = number("scale");
  meta.step = number("step");
  meta.q_context_scale = number("q_context_scale");
  meta.delta_scale = number("delta_scale");
  meta.block_t = static_cast<int64_t>(number("block_t"));
  meta.block_h = static_cast<int64_t>(number("block_h"));
  meta.block_w = static_cast<int64_t>(number("block_w"));
  require(std::isfinite(meta.scale) && meta.scale != 0.0 &&
              std::isfinite(meta.step) && meta.step > 0.0,
          "scale must be nonzero and step must be positive");
  return meta;
}

NGLRModel::NGLRModel(const std::string &path,
                     std::optional<c10::Device> requested)
    : device_(requested.value_or(select_model_device())) {
  try {
    module_ = torch::jit::load(path, device_);
    module_.eval();
  } catch (const c10::Error &e) {
    fail("failed to load scripted model '" + path + "': " + e.what());
  }
}

torch::Tensor NGLRModel::encode_recons(const torch::Tensor &recons) const {
  torch::NoGradGuard guard;
  return module_.get_method("encode_recons")({recons}).toTensor();
}

torch::Tensor
NGLRModel::forward_from_recons_feature(const torch::Tensor &feature,
                                       const torch::Tensor &context) const {
  torch::NoGradGuard guard;
  return module_.get_method("forward_from_recons_feature")({feature, context})
      .toTensor();
}

NGLRBundle NGLRBundle::load(const std::string &model_dir,
                            std::optional<c10::Device> device) {
  return {NGLRModel(model_dir + "/nglr_model.pt", device),
          load_meta_from_model_dir(model_dir)};
}

NGLREncodeStats nglr_encode(const torch::Tensor &original,
                            const torch::Tensor &recons, const NGLRModel &model,
                            const NGLRMeta &meta,
                            std::vector<uint8_t> &correction_out,
                            int zstd_level) {
  require(original.defined() && recons.defined() &&
              original.sizes() == recons.sizes(),
          "original and recons must be defined tensors with identical shape");
  require(original.dim() == 5, "original and recons must be 5-D [B,C,T,H,W]");
  const auto original_norm =
      ((original.to(torch::kCPU, torch::kFloat32) - meta.x_mean) / meta.scale)
          .contiguous();
  const auto recons_norm =
      ((recons.to(torch::kCPU, torch::kFloat32) - meta.x_mean) / meta.scale)
          .contiguous();
  const auto q = torch::round((original_norm - recons_norm) / meta.step)
                     .to(torch::kInt32)
                     .contiguous();
  const auto slices = block_slices(q, meta);
  ByteWriter writer;
  write_header(writer, q, meta, slices.size(), zstd_level);
  NGLREncodeStats stats;
  for (const auto &slice : slices) {
    const auto delta = strict_delta_encode(
        model, slice_block(q, slice), slice_block(recons_norm, slice), meta);
    const auto encoded = encode_bitplanes(delta, zstd_level, stats);
    writer.put(slice.t1 - slice.t0);
    writer.put(slice.h1 - slice.h0);
    writer.put(slice.w1 - slice.w0);
    writer.put(encoded.bits);
    writer.put(encoded.values);
    for (const auto &stream : encoded.streams) {
      writer.put(static_cast<uint64_t>(stream.size()));
      writer.put_bytes(stream.data(), stream.size());
    }
    ++stats.block_count;
  }
  correction_out = writer.take();
  stats.correction_bytes = correction_out.size();
  const uint64_t all_values = static_cast<uint64_t>(q.numel());
  if (all_values != 0)
    stats.delta_abs_mean /= static_cast<double>(all_values);
  return stats;
}

torch::Tensor nglr_decode(const torch::Tensor &recons, const NGLRModel &model,
                          const NGLRMeta &meta,
                          const std::vector<uint8_t> &correction) {
  require(recons.defined() && recons.dim() == 5,
          "recons must be a defined 5-D [B,C,T,H,W] tensor");
  ByteReader reader(correction);
  check_header(reader, recons, meta);
  const uint64_t block_count = reader.get<uint64_t>();
  (void)reader.get<int32_t>(); // producer's zstd level is informational.
  const auto recons_norm =
      ((recons.to(torch::kCPU, torch::kFloat32) - meta.x_mean) / meta.scale)
          .contiguous();
  auto qhat =
      torch::zeros(recons.sizes(), torch::TensorOptions().dtype(torch::kInt32));
  const auto slices = block_slices(qhat, meta);
  require(block_count == slices.size(),
          "payload block count does not match tensor shape/metadata");
  for (const auto &slice : slices) {
    const int64_t t = reader.get<int64_t>(), h = reader.get<int64_t>(),
                  w = reader.get<int64_t>();
    require(t == slice.t1 - slice.t0 && h == slice.h1 - slice.h0 &&
                w == slice.w1 - slice.w0,
            "payload block dimensions do not match expected block layout");
    const auto delta = decode_bitplanes(reader, t, h, w).unsqueeze(0);
    put_block(
        qhat, slice,
        strict_delta_decode(model, delta, slice_block(recons_norm, slice), meta)
            .unsqueeze(1));
  }
  require(reader.at_end(), "trailing bytes in correction payload");
  return ((recons_norm + qhat.to(torch::kFloat32) * meta.step) * meta.scale +
          meta.x_mean)
      .to(recons.device(), recons.scalar_type());
}

} // namespace nglr
