#include <caesar/models/model_metadata.h>
#include <caesar/models/range_coder/rans_coder.hpp>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
  if (argc != 2)
    throw std::runtime_error("Expected installed model directory");
  const auto metadata = read_model_metadata(std::filesystem::path(argv[1]) /
                                            "model_metadata.txt");
  metadata.require_dims(5);
  const std::vector<int32_t> symbols{0, 1, 1, 0};
  const std::vector<int32_t> indexes{0, 0, 0, 0};
  const std::vector<std::vector<int32_t>> cdfs{{0, 32768, 65536}};
  const std::vector<int32_t> lengths{3}, offsets{0};
  RansEncoder encoder;
  RansDecoder decoder;
  const auto encoded =
      encoder.encode_with_indexes(symbols, indexes, cdfs, lengths, offsets);
  if (decoder.decode_with_indexes(encoded, indexes, cdfs, lengths, offsets) !=
      symbols)
    throw std::runtime_error("Installed library round trip failed");
  for (const auto *name :
       {"caesar_compressor.pt2", "caesar_decompressor.pt2",
        "caesar_hyper_decompressor.pt2", "vbr_quantized_cdf.bin",
        "vbr_cdf_length.bin", "vbr_offset.bin", "gs_quantized_cdf.bin",
        "gs_cdf_length.bin", "gs_offset.bin"}) {
    const auto path = std::filesystem::path(argv[1]) / name;
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path) == 0)
      throw std::runtime_error("Missing installed model artifact: " +
                               path.string());
  }
  std::cout
      << "Installed package, model artifacts, and library round trip passed\n";
}
