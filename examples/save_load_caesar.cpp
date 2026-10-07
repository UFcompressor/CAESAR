#include "models/caesar_compress.h"
#include "models/caesar_decompress.h"
#include "models/serialization.h"
#include <iostream>
#include <torch/torch.h>

// Compile against an installed CAESAR library, then run with a matching model
// installation. The same save/load calls also work with LBRC and NGLR results.
int main(int argc, char **argv) {
  try {
    const std::string path = argc > 1 ? argv[1] : "field.cae";
    CompressionConfig config;
    config.memory_data = torch::rand({8, 256, 256});
    config.n_frame = 8;
    Compressor compressor;
    auto compressed = compressor.compress(config, 0.001f);
    caesar::save(compressed, path);

    auto reloaded = caesar::load(path);
    Decompressor decompressor;
    auto restored = decompressor.decompress(reloaded);
    auto direct = decompressor.decompress(compressed);
    if (!torch::equal(restored, direct))
      throw std::runtime_error("Disk round trip changed the reconstruction");
    std::cout << "Saved and restored " << restored.numel() << " values through "
              << path << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
