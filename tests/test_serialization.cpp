#include "models/caesar_compress.h"
#include "models/serialization.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>

void check(bool ok) {
  if (!ok)
    throw std::runtime_error("Serialization test failed");
}
template <class F> void rejects(F f) {
  bool failed = false;
  try {
    f();
  } catch (const std::exception &) {
    failed = true;
  }
  check(failed);
}
int main() {
  for (auto method :
       {caesar::CorrectionMethod::GAE, caesar::CorrectionMethod::LBRC,
        caesar::CorrectionMethod::NGLR}) {
    CompressionResult r{};
    r.model_id = "ufl:2@sha256:example";
    r.correction_method = method;
    r.n_frame = 8;
    r.original_shape = {2, 3, 8, 256, 256};
    r.shape_info = {{8, 256, 256}, 524288, {1, 1, 8, 256, 256}, 256, 256, true};
    r.encoded_latents = {std::string("a\0b", 3), ""};
    r.encoded_hyper_latents = {"hyper"};
    auto &m = r.compressionMetaData;
    m.offsets = {-2.5f, 0.5f};
    m.scales = {1, 2};
    m.indexes = {{-1, 2}, {}};
    m.block_info = {2, 3, {0, 1}};
    m.data_input_shape = {1, 1, 8, 256, 256};
    m.filtered_blocks = {{-1, 2.5f}};
    m.global_scale = 4;
    m.global_offset = -3;
    m.pad_T = 2;
    m.all_filtered = true;
    r.gaeMetaData = {true, {1, 2}, {{0.5f, -1}}, {2}, 0.25, 1, 2, 3, 4};
    r.gae_comp_data = {0, 255, 17};
    r.lbrcMetaData = {true, -3, 4, {8, 256, 256}};
    r.lbrc_blocks = {{0.125, 3, {{0, 255}, {}, {7}}}};
    r.nglrMetaData.correction_occurred = true;
    r.nglrMetaData.constant_input = true;
    r.nglrMetaData.quantization = {-2, 3, 0.1, 4, 5, 6, 7, 8};
    r.nglrMetaData.shape = {1, 1, 8, 256, 256};
    r.nglrMetaData.weights = {{"weight", {2}, {0.5f, -0.25f}}};
    r.nglr_comp_data = {11, 0, 255};
    const auto bytes = caesar::serialize(r);
    const auto restored = caesar::deserialize(bytes);
    check(restored.model_id == r.model_id && restored.shape_info.was_padded);
    check(restored.compressionMetaData.indexes == m.indexes);
    check(restored.encoded_latents == r.encoded_latents);
    check(restored.lbrc_blocks[0].streams == r.lbrc_blocks[0].streams);
    check(restored.nglrMetaData.weights[0].values ==
          r.nglrMetaData.weights[0].values);
    check(caesar::serialize(restored) == bytes);
    for (size_t n = 0; n < bytes.size(); ++n)
      rejects([&] { caesar::deserialize(bytes.data(), n); });
    auto bad = bytes;
    bad[0] ^= 1;
    rejects([&] { caesar::deserialize(bad); });
    bad = bytes;
    bad[4] = 99;
    rejects([&] { caesar::deserialize(bad); });
    bad = bytes;
    bad[16 + r.model_id.size()] = 255;
    rejects([&] { caesar::deserialize(bad); });
    bad = bytes;
    bad.push_back(0);
    rejects([&] { caesar::deserialize(bad); });
    bad = bytes;
    for (size_t i = 8; i < 16; ++i)
      bad[i] = 255;
    rejects([&] { caesar::deserialize(bad); });
    auto path = (std::filesystem::temp_directory_path() /
                 "caesar-serialization-test.bin")
                    .string();
    caesar::save(r, path);
    check(caesar::serialize(caesar::load(path)) == bytes);
    std::filesystem::remove(path);
  }
  // Non-power-of-two payload exposes geometric growth and retained excess
  // capacity. Exercise large correction data without loading any model.
  {
    CompressionResult large{};
    constexpr size_t payload_size = 8 * 1024 * 1024 + 1;
    large.gae_comp_data.assign(payload_size, 0x5a);
    large.gaeMetaData.pcaBasis = {std::vector<float>(1024 * 1024 + 1, 0.25f)};
    auto bytes = caesar::serialize(large);
    check(bytes.capacity() == bytes.size());
    auto restored = caesar::deserialize(bytes);
    check(restored.gae_comp_data == large.gae_comp_data);
    check(restored.gae_comp_data.capacity() == payload_size);
    check(restored.gaeMetaData.pcaBasis == large.gaeMetaData.pcaBasis);
    check(restored.gaeMetaData.pcaBasis[0].capacity() ==
          restored.gaeMetaData.pcaBasis[0].size());
  }
  CompressionResult empty{};
  check(caesar::serialize(caesar::deserialize(caesar::serialize(empty))) ==
        caesar::serialize(empty));
  std::cout << "Serialization round-trip and malformed-buffer tests passed\n";
}
