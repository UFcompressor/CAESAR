#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "rans64.h"

// Constants consistent with CompressAI
constexpr int PRECISION = 16;
constexpr uint16_t BYPASS_PRECISION = 4;
constexpr uint16_t MAX_BYPASS_VAL = (1 << BYPASS_PRECISION) - 1;

struct RansSymbol {
  uint16_t start;
  uint16_t range;
  bool bypass; // bypass flag to write raw bits to the stream
};

/* NOTE: Warning, we buffer everything for now...
 * In case of large files we should split the bitstream into chunks...
 * Or for a memory-bounded encoder
 **/
class BufferedRansEncoder {
public:
  BufferedRansEncoder() = default;

  BufferedRansEncoder(const BufferedRansEncoder &) = delete;
  BufferedRansEncoder(BufferedRansEncoder &&) = delete;
  BufferedRansEncoder &operator=(const BufferedRansEncoder &) = delete;
  BufferedRansEncoder &operator=(BufferedRansEncoder &&) = delete;

  void encode_with_indexes(const std::vector<int32_t> &symbols,
                           const std::vector<int32_t> &indexes,
                           const std::vector<std::vector<int32_t>> &cdfs,
                           const std::vector<int32_t> &cdfs_sizes,
                           const std::vector<int32_t> &offsets);

  std::string flush();  // Previously returned py::bytes, now returns std::string directly

private:
  std::vector<RansSymbol> _syms;
};

class RansEncoder {
public:
  RansEncoder() = default;

  RansEncoder(const RansEncoder &) = delete;
  RansEncoder(RansEncoder &&) = delete;
  RansEncoder &operator=(const RansEncoder &) = delete;
  RansEncoder &operator=(RansEncoder &&) = delete;

  std::string encode_with_indexes(const std::vector<int32_t> &symbols,
                                  const std::vector<int32_t> &indexes,
                                  const std::vector<std::vector<int32_t>> &cdfs,
                                  const std::vector<int32_t> &cdfs_sizes,
                                  const std::vector<int32_t> &offsets);
};

class RansDecoder {
public:
  RansDecoder() = default;

  RansDecoder(const RansDecoder &) = delete;
  RansDecoder(RansDecoder &&) = delete;
  RansDecoder &operator=(const RansDecoder &) = delete;
  RansDecoder &operator=(RansDecoder &&) = delete;

  std::vector<int32_t>
  decode_with_indexes(const std::string &encoded,
                      const std::vector<int32_t> &indexes,
                      const std::vector<std::vector<int32_t>> &cdfs,
                      const std::vector<int32_t> &cdfs_sizes,
                      const std::vector<int32_t> &offsets);

  void set_stream(const std::string &stream);

  std::vector<int32_t>
  decode_stream(const std::vector<int32_t> &indexes,
                const std::vector<std::vector<int32_t>> &cdfs,
                const std::vector<int32_t> &cdfs_sizes,
                const std::vector<int32_t> &offsets);

private:
  Rans64State _rans;
  std::string _stream;
  uint32_t *_ptr = nullptr;
};
