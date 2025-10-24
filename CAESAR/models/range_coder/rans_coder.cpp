#include "rans_coder.hpp"
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace {

/* probability range, same as CompressAI */
constexpr int precision = PRECISION;
constexpr uint16_t bypass_precision = BYPASS_PRECISION;
constexpr uint16_t max_bypass_val = MAX_BYPASS_VAL;

void assert_cdfs(const std::vector<std::vector<int32_t>> &cdfs,
                 const std::vector<int32_t> &cdfs_sizes) {
  for (int i = 0; i < static_cast<int>(cdfs.size()); ++i) {
    assert(cdfs[i][0] == 0);
    assert(cdfs[i][cdfs_sizes[i] - 1] == (1 << precision));
    for (int j = 0; j < cdfs_sizes[i] - 1; ++j) {
      assert(cdfs[i][j + 1] > cdfs[i][j]);
    }
  }
}

/* Support only 16 bits word max */
inline void Rans64EncPutBits(Rans64State *r, uint32_t **pptr,
                             uint32_t val, uint32_t nbits) {
  assert(nbits <= 16);
  assert(val < (1u << nbits));

  uint64_t x = *r;
  uint32_t freq = 1 << (16 - nbits);
  uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
  if (x >= x_max) {
    *pptr -= 1;
    **pptr = (uint32_t)x;
    x >>= 32;
  }
  *r = (x << nbits) | val;
}

inline uint32_t Rans64DecGetBits(Rans64State *r, uint32_t **pptr,
                                 uint32_t n_bits) {
  uint64_t x = *r;
  uint32_t val = x & ((1u << n_bits) - 1);

  x = x >> n_bits;
  if (x < RANS64_L) {
    x = (x << 32) | **pptr;
    *pptr += 1;
  }

  *r = x;
  return val;
}

} // namespace


// ====================== Encoder ======================

void BufferedRansEncoder::encode_with_indexes(
    const std::vector<int32_t> &symbols, const std::vector<int32_t> &indexes,
    const std::vector<std::vector<int32_t>> &cdfs,
    const std::vector<int32_t> &cdfs_sizes,
    const std::vector<int32_t> &offsets) {
  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  for (size_t i = 0; i < symbols.size(); ++i) {
    const int32_t cdf_idx = indexes[i];
    const auto &cdf = cdfs[cdf_idx];
    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;

    int32_t value = symbols[i] - offsets[cdf_idx];
    uint32_t raw_val = 0;

    if (value < 0) {
      raw_val = -2 * value - 1;
      value = max_value;
    } else if (value >= max_value) {
      raw_val = 2 * (value - max_value);
      value = max_value;
    }

    _syms.push_back({(uint16_t)cdf[value],
                     (uint16_t)(cdf[value + 1] - cdf[value]),
                     false});

    if (value == max_value) {
      int32_t n_bypass = 0;
      while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
        ++n_bypass;
      }
      int32_t val = n_bypass;
      while (val >= max_bypass_val) {
        _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
        val -= max_bypass_val;
      }
      _syms.push_back({(uint16_t)val, (uint16_t)(val + 1), true});
      for (int32_t j = 0; j < n_bypass; ++j) {
        int32_t v = (raw_val >> (j * bypass_precision)) & max_bypass_val;
        _syms.push_back({(uint16_t)v, (uint16_t)(v + 1), true});
      }
    }
  }
}

std::string BufferedRansEncoder::flush() {
  Rans64State rans;
  Rans64EncInit(&rans);

  std::vector<uint32_t> output(_syms.size(), 0xCC);
  uint32_t *ptr = output.data() + output.size();

  while (!_syms.empty()) {
    const RansSymbol sym = _syms.back();
    if (!sym.bypass) {
      Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
    } else {
      Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
    }
    _syms.pop_back();
  }

  Rans64EncFlush(&rans, &ptr);

  const int nbytes =
      std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t);
  return std::string(reinterpret_cast<char *>(ptr), nbytes);
}

std::string RansEncoder::encode_with_indexes(
    const std::vector<int32_t> &symbols,
    const std::vector<int32_t> &indexes,
    const std::vector<std::vector<int32_t>> &cdfs,
    const std::vector<int32_t> &cdfs_sizes,
    const std::vector<int32_t> &offsets) {
  BufferedRansEncoder enc;
  enc.encode_with_indexes(symbols, indexes, cdfs, cdfs_sizes, offsets);
  return enc.flush();
}


// ====================== Decoder ======================

std::vector<int32_t> RansDecoder::decode_with_indexes(
    const std::string &encoded, const std::vector<int32_t> &indexes,
    const std::vector<std::vector<int32_t>> &cdfs,
    const std::vector<int32_t> &cdfs_sizes,
    const std::vector<int32_t> &offsets) {
  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  std::vector<int32_t> output(indexes.size());

  Rans64State rans;
  uint32_t *ptr = (uint32_t *)encoded.data();
  Rans64DecInit(&rans, &ptr);

  for (size_t i = 0; i < indexes.size(); ++i) {
    const auto &cdf = cdfs[indexes[i]];
    const int32_t max_value = cdfs_sizes[indexes[i]] - 2;
    const int32_t offset = offsets[indexes[i]];

    uint32_t cum_freq = Rans64DecGet(&rans, precision);
    auto it = std::find_if(cdf.begin(), cdf.begin() + cdfs_sizes[indexes[i]],
                           [cum_freq](uint32_t v) { return v > cum_freq; });
    uint32_t s = std::distance(cdf.begin(), it) - 1;

    Rans64DecAdvance(&rans, &ptr, cdf[s], cdf[s + 1] - cdf[s], precision);
    int32_t value = (int32_t)s;

    if (value == max_value) {
      int32_t val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
      int32_t n_bypass = val;
      while (val == max_bypass_val) {
        val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        n_bypass += val;
      }
      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        raw_val |= val << (j * bypass_precision);
      }
      value = (raw_val & 1) ? -(raw_val >> 1) - 1 : (raw_val >> 1) + max_value;
    }

    output[i] = value + offset;
  }

  return output;
}

void RansDecoder::set_stream(const std::string &encoded) {
  _stream = encoded;
  _ptr = (uint32_t *)_stream.data();
  Rans64DecInit(&_rans, &_ptr);
}

std::vector<int32_t> RansDecoder::decode_stream(
    const std::vector<int32_t> &indexes,
    const std::vector<std::vector<int32_t>> &cdfs,
    const std::vector<int32_t> &cdfs_sizes,
    const std::vector<int32_t> &offsets) {
  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  std::vector<int32_t> output(indexes.size());

  for (size_t i = 0; i < indexes.size(); ++i) {
    const auto &cdf = cdfs[indexes[i]];
    const int32_t max_value = cdfs_sizes[indexes[i]] - 2;
    const int32_t offset = offsets[indexes[i]];

    uint32_t cum_freq = Rans64DecGet(&_rans, precision);
    auto it = std::find_if(cdf.begin(), cdf.begin() + cdfs_sizes[indexes[i]],
                           [cum_freq](uint32_t v) { return v > cum_freq; });
    uint32_t s = std::distance(cdf.begin(), it) - 1;

    Rans64DecAdvance(&_rans, &_ptr, cdf[s], cdf[s + 1] - cdf[s], precision);
    int32_t value = (int32_t)s;

    if (value == max_value) {
      int32_t val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
      int32_t n_bypass = val;
      while (val == max_bypass_val) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        n_bypass += val;
      }
      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        raw_val |= val << (j * bypass_precision);
      }
      value = (raw_val & 1) ? -(raw_val >> 1) - 1 : (raw_val >> 1) + max_value;
    }

    output[i] = value + offset;
  }

  return output;
}
