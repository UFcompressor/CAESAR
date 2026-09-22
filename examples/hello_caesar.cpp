/**
 * hello_caesar.cpp
 *
 * Minimal in-memory exmple of compressing and decompressing a 3D field
 * with CAESAR.
 *
 * CAESAR expects 5D tensors in the form:
 * [variable, channel, time, height, width].
 */

#include <iostream>

#include "../CAESAR/data_utils.h"
#include "../CAESAR/dataset/dataset.h"
#include "../CAESAR/models/array_utils.h"
#include "../CAESAR/models/caesar_compress.h"
#include "../CAESAR/models/caesar_decompress.h"

int main() {
  try {
    const int64_t dim_x = 256;
    const int64_t dim_y = 256;
    const int64_t n_time = 256;
    const std::vector<int64_t> shape = {1, 1, n_time, dim_x, dim_y};

    // Generate a synthetic time-varying 3D field.
    torch::Tensor raw = torch::empty(shape, torch::kFloat32);
    {
      auto xs = torch::linspace(0, 2 * M_PI, dim_x);
      auto ys = torch::linspace(0, 2 * M_PI, dim_y);
      auto ts = torch::linspace(0, 2 * M_PI, n_time);

      auto grid = torch::sin(xs.view({1, -1, 1})) *
                  torch::cos(ys.view({1, 1, -1})) *
                  torch::sin(ts.view({-1, 1, 1}));

      raw.copy_(grid.reshape(shape));
    }

    // Convert [1, 1, T, H, W] to [T, H, W] before CAESAR preprocessing.
    raw = raw.squeeze();

    float raw_min = raw.min().item<float>();
    float raw_max = raw.max().item<float>();

    std::cout << "Generated data: shape " << raw.sizes() << ", min=" << raw_min
              << ", max=" << raw_max << "\n";

    // Convert the input to CAESAR's 5D representation and pad if necessary.
    //  TODO REMOVE PADDING NEEDS TO HAPPEN interally for CAESAR included in meta data
    PaddingInfo padding_info;
    torch::Tensor padded_5d;
    std::tie(padded_5d, padding_info) = to_5d(raw);
    padded_5d = padded_5d.contiguous();
    raw = torch::Tensor();

    torch::Device device = select_model_device();

    DatasetConfig config;
    config.memory_data = padded_5d;

    // Number of time frames processed per temporal window.
    config.n_frame = 8;

    const float rel_eb = 1e-4f;
// you compile it for a ceratin device you have have to do it for a different device so no point in device
    Compressor compressor;
    CompressionResult compressed =
        compressor.compress(config,rel_eb);

    // Calculate the size of the encoded latent streams.
    uint64_t compressed_bytes = 0;
    for (const auto &stream : compressed.encoded_latents)
      compressed_bytes += stream.size();

    for (const auto &stream : compressed.encoded_hyper_latents)
      compressed_bytes += stream.size();

    std::cout << "Compressed to " << compressed_bytes << " bytes\n";

    // Decompress directly from the in-memory compression result.
    Decompressor decompressor(device);
    torch::Tensor recon =
        decompressor.decompress(batch_size, config.n_frame, compressed);

    if (!recon.defined() || recon.numel() == 0) {
      std::cerr << "Decompression failed: reconstructed tensor is "
                   "empty/undefined.\n";
      return 1;
    }

    // Remove padding and restore the original tensor dimensions.
    torch::Tensor restored = restore_from_5d(recon, padding_info);

    std::cout << "Reconstructed shape: " << restored.sizes() << "\n";
    std::cout << "Done\n";

#ifndef _WIN32
    ModelCache::instance().clear();
#endif

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
