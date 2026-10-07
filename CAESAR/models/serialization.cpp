#include "serialization.h"
#include "caesar_compress.h"

#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace caesar {
namespace {
// A single field list drives both reading and writing, preventing drift.
#define FIELDS(Type, ...)                                                      \
  template <                                                                   \
      class A, class T,                                                        \
      std::enable_if_t<std::is_same_v<std::remove_const_t<T>, Type>, int> = 0> \
  void fields(A &a, T &v) {                                                    \
    a(__VA_ARGS__);                                                            \
  }
FIELDS(PaddingInfo, v.original_shape, v.original_length, v.padded_shape, v.H,
       v.W, v.was_padded)
FIELDS(CompressionMetaData, v.offsets, v.scales, v.indexes, v.block_info,
       v.data_input_shape, v.filtered_blocks, v.global_scale, v.global_offset,
       v.pad_T, v.all_filtered)
FIELDS(GAEMetaData, v.GAE_correction_occur, v.padding_recon_info, v.pcaBasis,
       v.uniqueVals, v.quanBin, v.nVec, v.prefixLength, v.dataBytes,
       v.coeffIntBytes)
FIELDS(LBRCMetaData, v.lbrc_correction_occur, v.x_mean, v.scale, v.block_size)
FIELDS(LBRCBlock, v.step, v.bit_count, v.streams)
FIELDS(nglr::NGLRMeta, v.x_mean, v.scale, v.step, v.q_context_scale,
       v.delta_scale, v.block_t, v.block_h, v.block_w)
FIELDS(nglr::NGLRWeight, v.name, v.shape, v.values)
FIELDS(nglr::NGLRMetaData, v.schema_version, v.correction_occurred,
       v.constant_input, v.quantization, v.hidden, v.q_hidden, v.model_blocks,
       v.shape, v.weights)
FIELDS(CompressionResult, v.model_id, v.correction_method, v.n_frame,
       v.original_shape, v.shape_info, v.encoded_latents,
       v.encoded_hyper_latents, v.compressionMetaData, v.gaeMetaData,
       v.gae_comp_data, v.lbrcMetaData, v.lbrc_blocks, v.nglrMetaData,
       v.nglr_comp_data)
#undef FIELDS

struct Writer {
  std::vector<uint8_t> data;
  bool measuring = false;
  size_t measured = 0;
  void measure(size_t n) {
    if (n > std::numeric_limits<size_t>::max() - measured)
      throw std::length_error("CAESAR serialized size overflow");
    measured += n;
  }
  template <class... T> void operator()(const T &...v) { (put(v), ...); }
  template <class T> void put(const T &v) {
    if (measuring) {
      if constexpr (std::is_arithmetic_v<T> || std::is_enum_v<T>)
        measure(std::is_same_v<T, bool> ? 1 : sizeof(T));
      else
        fields(*this, v);
      return;
    }
    if constexpr (std::is_same_v<T, bool>) {
      data.push_back(v ? 1 : 0);
    } else if constexpr (std::is_enum_v<T>) {
      put(static_cast<std::underlying_type_t<T>>(v));
    } else if constexpr (std::is_integral_v<T>) {
      using U = std::make_unsigned_t<T>;
      U bits = static_cast<U>(v);
      for (size_t i = 0; i < sizeof(T); ++i) {
        data.push_back(static_cast<uint8_t>(bits));
        bits >>= 8;
      }
    } else if constexpr (std::is_floating_point_v<T>) {
      static_assert(std::numeric_limits<T>::is_iec559);
      using U = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
      U bits;
      std::memcpy(&bits, &v, sizeof(T));
      put(bits);
    } else {
      fields(*this, v);
    }
  }
  void put(const std::string &v) {
    put(uint64_t(v.size()));
    if (measuring) {
      measure(v.size());
      return;
    }
    data.insert(data.end(), v.begin(), v.end());
  }
  template <class T> void put(const std::vector<T> &v) {
    put(uint64_t(v.size()));
    if constexpr (std::is_arithmetic_v<T>) {
      if (measuring) {
        if (v.size() > std::numeric_limits<size_t>::max() / sizeof(T))
          throw std::length_error("CAESAR serialized vector size overflow");
        measure(v.size() * sizeof(T));
        return;
      }
    }
    for (const auto &item : v)
      put(item);
  }
  template <class T, size_t N> void put(const std::array<T, N> &v) {
    for (const auto &item : v)
      put(item);
  }
  template <class X, class Y> void put(const std::pair<X, Y> &v) {
    (*this)(v.first, v.second);
  }
  template <class... T> void put(const std::tuple<T...> &v) {
    std::apply([this](const auto &...items) { (*this)(items...); }, v);
  }
};

struct Reader {
  const uint8_t *data;
  size_t size, pos = 0;
  void require(size_t n) const {
    if (n > size - pos)
      throw std::runtime_error("Truncated CAESAR serialized buffer");
  }
  template <class... T> void operator()(T &...v) { (get(v), ...); }
  template <class T> void get(T &v) {
    if constexpr (std::is_same_v<T, bool>) {
      uint8_t b;
      get(b);
      if (b > 1)
        throw std::runtime_error("Invalid serialized boolean");
      v = b != 0;
    } else if constexpr (std::is_enum_v<T>) {
      std::underlying_type_t<T> b;
      get(b);
      v = correction_method_from_byte(b);
    } else if constexpr (std::is_integral_v<T>) {
      require(sizeof(T));
      using U = std::make_unsigned_t<T>;
      U bits = 0;
      for (size_t i = 0; i < sizeof(T); ++i)
        bits |= U(data[pos++]) << (8 * i);
      std::memcpy(&v, &bits, sizeof(T));
    } else if constexpr (std::is_floating_point_v<T>) {
      using U = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
      U bits;
      get(bits);
      std::memcpy(&v, &bits, sizeof(T));
    } else {
      fields(*this, v);
    }
  }
  size_t count() {
    uint64_t n;
    get(n);
    if (n > size - pos)
      throw std::runtime_error("Invalid serialized collection length");
    return static_cast<size_t>(n);
  }
  void get(std::string &v) {
    size_t n = count();
    require(n);
    v.assign(reinterpret_cast<const char *>(data + pos), n);
    pos += n;
  }
  template <class T> void get(std::vector<T> &v) {
    size_t n = count();
    if constexpr (std::is_arithmetic_v<T>) {
      if (n > (size - pos) / sizeof(T))
        throw std::runtime_error("Invalid serialized vector length");
      // The complete scalar payload fits the input. Allocate once rather than
      // growing geometrically, which temporarily holds two large allocations.
      v.reserve(n);
    }
    // Composite elements are decoded before growing their destination because
    // their variable-length payloads have not yet been validated.
    for (size_t i = 0; i < n; ++i) {
      T item{};
      get(item);
      v.push_back(std::move(item));
    }
  }
  template <class T, size_t N> void get(std::array<T, N> &v) {
    for (auto &item : v)
      get(item);
  }
  template <class X, class Y> void get(std::pair<X, Y> &v) {
    (*this)(v.first, v.second);
  }
  template <class... T> void get(std::tuple<T...> &v) {
    std::apply([this](auto &...items) { (*this)(items...); }, v);
  }
};
constexpr uint32_t magic = 0x52534143; // "CASR"
constexpr uint32_t version = 1;
static_assert(sizeof(int) == 4 && sizeof(size_t) == 8,
              "Serializer v1 currently requires 32-bit int and 64-bit size_t");
} // namespace

std::vector<uint8_t> serialize(const CompressionResult &result) {
  correction_method_from_byte(static_cast<uint8_t>(result.correction_method));
  // Measure without copying payloads, then allocate the output once.
  Writer counter;
  counter.measuring = true;
  counter(magic, version, result);
  Writer writer;
  writer.data.reserve(counter.measured);
  writer(magic, version, result);
  return std::move(writer.data);
}
CompressionResult deserialize(const void *data, size_t size) {
  if (!data && size)
    throw std::invalid_argument("Null CAESAR serialized buffer");
  Reader reader{static_cast<const uint8_t *>(data), size};
  uint32_t m, v;
  reader(m, v);
  if (m != magic)
    throw std::runtime_error("Invalid CAESAR serialized buffer magic");
  if (v != version)
    throw std::runtime_error("Unsupported CAESAR serialization version");
  CompressionResult result{};
  reader(result);
  if (reader.pos != size)
    throw std::runtime_error("Trailing bytes in CAESAR serialized buffer");
  return result;
}
CompressionResult deserialize(const std::vector<uint8_t> &data) {
  return deserialize(data.data(), data.size());
}
void save(const CompressionResult &result, const std::string &path) {
  const auto bytes = serialize(result);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file ||
      !file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size()))
    throw std::runtime_error("Cannot write CAESAR file: " + path);
  file.close();
  if (!file)
    throw std::runtime_error("Cannot close CAESAR file: " + path);
}
CompressionResult load(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("Cannot open CAESAR file: " + path);
  const auto length = file.tellg();
  if (length < 0)
    throw std::runtime_error("Cannot size CAESAR file: " + path);
  std::vector<uint8_t> bytes(static_cast<size_t>(length));
  file.seekg(0);
  if (!file.read(reinterpret_cast<char *>(bytes.data()), bytes.size()))
    throw std::runtime_error("Cannot read CAESAR file: " + path);
  return deserialize(bytes);
}
} // namespace caesar
