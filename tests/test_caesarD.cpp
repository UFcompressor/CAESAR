#include "../CAESAR/cli_result_header.h"
#include "../CAESAR/data_utils.h"
#include "../CAESAR/dataset/dataset.h"
#include "../CAESAR/models/caesar_compress.h"
#include "../CAESAR/models/caesar_decompress.h"

std::vector<std::string> load_encoded_streams(const std::string &filename) {
  std::vector<std::string> out;
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file to read: " << filename << std::endl;
    return out;
  }
  uint64_t len;
  while (file.read(reinterpret_cast<char *>(&len), sizeof(len))) {
    std::string s;
    if (len) {
      s.resize(len);
      if (!file.read(&s[0], static_cast<std::streamsize>(len))) {
        std::cerr << "Error: truncated read while reading " << filename
                  << std::endl;
        break;
      }
    }
    out.push_back(std::move(s));
  }
  file.close();
  return out;
}

torch::Tensor loadRawBinary(const std::string &bin_path,
                            const std::vector<int64_t> &shape) {
  std::ifstream file(bin_path, std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Cannot open binary file: " + bin_path);

  size_t num_elems = 1;
  for (auto d : shape) {
    if (d <= 0)
      throw std::runtime_error("Invalid shape dimension");
    num_elems *= static_cast<size_t>(d);
  }

  std::vector<float> buf(num_elems);
  file.read(reinterpret_cast<char *>(buf.data()),
            static_cast<std::streamsize>(num_elems * sizeof(float)));
  if (!file)
    throw std::runtime_error("Failed to read expected floats from " + bin_path);
  file.close();

  torch::Tensor t =
      torch::from_blob(buf.data(), torch::IntArrayRef(shape), torch::kFloat32)
          .clone();
  std::cout << "Loaded " << bin_path << " with shape " << t.sizes() << "\n";
  std::cout << "  Min: " << t.min().item<float>()
            << ", Max: " << t.max().item<float>() << "\n";
  return t;
}

CompressionResult
load_compression_result_metadata(const std::string &filename) {
  CompressionResult result;
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Cannot open metadata file: " + filename);

  caesar::cli::read_result_header(file, result);

  // Helper lambda to read vector
  auto read_vector = [&file](auto &vec) {
    uint64_t size;
    file.read(reinterpret_cast<char *>(&size), sizeof(size));
    vec.resize(size);
    if (size > 0) {
      file.read(reinterpret_cast<char *>(vec.data()),
                static_cast<std::streamsize>(size * sizeof(vec[0])));
    }
  };

  // Helper lambda to read 2d vector
  auto read_2d_vector = [&file](auto &vec_2d) {
    uint64_t outer_size;
    file.read(reinterpret_cast<char *>(&outer_size), sizeof(outer_size));
    vec_2d.resize(outer_size);
    for (uint64_t i = 0; i < outer_size; ++i) {
      uint64_t inner_size;
      file.read(reinterpret_cast<char *>(&inner_size), sizeof(inner_size));
      vec_2d[i].resize(inner_size);
      if (inner_size > 0) {
        file.read(
            reinterpret_cast<char *>(vec_2d[i].data()),
            static_cast<std::streamsize>(inner_size * sizeof(vec_2d[i][0])));
      }
    }
  };

  auto &meta = result.compressionMetaData;
  auto &gae_meta = result.gaeMetaData;
  auto &lbrc_meta = result.lbrcMetaData;

  // Load compressionMetaData
  read_vector(meta.offsets);
  read_vector(meta.scales);
  read_2d_vector(meta.indexes);

  int32_t nH, nW;
  file.read(reinterpret_cast<char *>(&nH), sizeof(nH));
  file.read(reinterpret_cast<char *>(&nW), sizeof(nW));
  std::vector<int32_t> padding_vec;
  read_vector(padding_vec);
  meta.block_info = std::make_tuple(nH, nW, padding_vec);

  read_vector(meta.data_input_shape);
  read_vector(meta.filtered_blocks);

  file.read(reinterpret_cast<char *>(&meta.global_scale),
            sizeof(meta.global_scale));
  file.read(reinterpret_cast<char *>(&meta.global_offset),
            sizeof(meta.global_offset));
  file.read(reinterpret_cast<char *>(&meta.pad_T), sizeof(meta.pad_T));

  // Load gaeMetaData
  file.read(reinterpret_cast<char *>(&gae_meta.GAE_correction_occur),
            sizeof(gae_meta.GAE_correction_occur));
  read_vector(gae_meta.padding_recon_info);
  read_2d_vector(gae_meta.pcaBasis);
  read_vector(gae_meta.uniqueVals);
  file.read(reinterpret_cast<char *>(&gae_meta.quanBin),
            sizeof(gae_meta.quanBin));
  file.read(reinterpret_cast<char *>(&gae_meta.nVec), sizeof(gae_meta.nVec));
  file.read(reinterpret_cast<char *>(&gae_meta.prefixLength),
            sizeof(gae_meta.prefixLength));
  file.read(reinterpret_cast<char *>(&gae_meta.dataBytes),
            sizeof(gae_meta.dataBytes));
  file.read(reinterpret_cast<char *>(&gae_meta.coeffIntBytes),
            sizeof(gae_meta.coeffIntBytes));

  // Load gae_comp_data
  read_vector(result.gae_comp_data);

  // // Load latent_indexes
  // read_2d_vector(result.latent_indexes);

  // Load correction_method
  file.read(reinterpret_cast<char *>(&result.correction_method),
            sizeof(result.correction_method));

  // Load lbrcMetaData
  file.read(reinterpret_cast<char *>(&lbrc_meta.lbrc_correction_occur),
            sizeof(lbrc_meta.lbrc_correction_occur));
  file.read(reinterpret_cast<char *>(&lbrc_meta.x_mean),
            sizeof(lbrc_meta.x_mean));
  file.read(reinterpret_cast<char *>(&lbrc_meta.scale),
            sizeof(lbrc_meta.scale));
  file.read(reinterpret_cast<char *>(&lbrc_meta.block_size),
            sizeof(lbrc_meta.block_size));

  // Load lbrc_blocks
  uint64_t num_blocks;
  file.read(reinterpret_cast<char *>(&num_blocks), sizeof(num_blocks));
  result.lbrc_blocks.resize(num_blocks);
  for (uint64_t b = 0; b < num_blocks; ++b) {
    auto &blk = result.lbrc_blocks[b];
    file.read(reinterpret_cast<char *>(&blk.step), sizeof(blk.step));
    file.read(reinterpret_cast<char *>(&blk.bit_count), sizeof(blk.bit_count));

    uint64_t num_streams;
    file.read(reinterpret_cast<char *>(&num_streams), sizeof(num_streams));
    blk.streams.resize(num_streams);
    for (uint64_t s = 0; s < num_streams; ++s) {
      uint64_t stream_len;
      file.read(reinterpret_cast<char *>(&stream_len), sizeof(stream_len));
      if (stream_len > 0) {
        blk.streams[s].resize(stream_len);
        file.read(reinterpret_cast<char *>(&blk.streams[s][0]),
                  static_cast<std::streamsize>(stream_len));
      }
    }
  }

  if (result.correction_method == caesar::CorrectionMethod::NGLR) {
    auto &m = result.nglrMetaData;
    const auto begin = file.tellg();
    file.seekg(0, std::ios::end);
    const auto end = file.tellg();
    file.seekg(begin);
    auto scalar = [&](auto &value) {
      if (!file.read(reinterpret_cast<char *>(&value), sizeof(value)))
        throw std::runtime_error("Truncated NGLR metadata");
    };
    auto vector = [&](auto &values) {
      uint64_t n = 0;
      scalar(n);
      using T = typename std::decay_t<decltype(values)>::value_type;
      if (n > static_cast<uint64_t>(end - file.tellg()) / sizeof(T))
        throw std::runtime_error("Truncated NGLR vector");
      values.resize(n);
      if (n &&
          !file.read(reinterpret_cast<char *>(values.data()), n * sizeof(T)))
        throw std::runtime_error("Truncated NGLR vector");
    };
    scalar(m.schema_version);
    scalar(m.correction_occurred);
    scalar(m.constant_input);
    scalar(m.quantization.x_mean);
    scalar(m.quantization.scale);
    scalar(m.quantization.step);
    scalar(m.quantization.q_context_scale);
    scalar(m.quantization.delta_scale);
    scalar(m.quantization.block_t);
    scalar(m.quantization.block_h);
    scalar(m.quantization.block_w);
    scalar(m.hidden);
    scalar(m.q_hidden);
    scalar(m.model_blocks);
    vector(m.shape);
    uint64_t count = 0;
    scalar(count);
    if (count > 1024)
      throw std::runtime_error("Invalid NGLR parameter count");
    m.weights.resize(count);
    for (auto &weight : m.weights) {
      vector(weight.name);
      vector(weight.shape);
      vector(weight.values);
    }
    vector(result.nglr_comp_data);
    nglr::validate_metadata(m);
  }
  file.close();
  return result;
}

float load_error_bound(const std::string &filename) {
  float rel_eb = 0.0f;
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Cannot open error bound file: " + filename);
  file.read(reinterpret_cast<char *>(&rel_eb), sizeof(rel_eb));
  file.close();
  return rel_eb;
}

int main() {
  std::cout.setf(std::ios::unitbuf);
  try {
    std::set_terminate([]() {
      std::cerr << "FATAL: std::terminate() was called - "
                   "likely an uncaught exception on a non-main thread.\n";
      std::abort();
    });

    const std::string out_dir = "./output/";
    const std::string raw_path = "TCf48.bin.f32";
    const std::vector<int64_t> shape = {1, 1, 20, 256, 256};

    std::cout << "\n===== DECOMPRESSION =====\n";

    // Load error bound
    float rel_eb;
    try {
      rel_eb = load_error_bound(out_dir + "error_bound.bin");
      std::cout << "Error bound: " << rel_eb << "\n";
    } catch (const std::exception &e) {
      std::cerr << "Warning: " << e.what() << " - Using default 0.0001\n";
      rel_eb = 0.0001f;
    }

    // Load original data for metrics
    std::cout << "Loading original data...\n";
    torch::Tensor raw_for_metrics = loadRawBinary(raw_path, shape).squeeze();
    float raw_min = raw_for_metrics.min().item<float>();
    float raw_max = raw_for_metrics.max().item<float>();
    std::cout << "Original min: " << raw_min << ", max: " << raw_max << "\n";

    // Load encoded streams
    std::cout << "Loading encoded streams...\n";
    std::vector<std::string> loaded_latents =
        load_encoded_streams(out_dir + "encoded_latents.bin");
    std::vector<std::string> loaded_hyper =
        load_encoded_streams(out_dir + "encoded_hyper_latents.bin");

    if (loaded_latents.empty() || loaded_hyper.empty()) {
      std::cerr << "Error: Failed to load encoded streams\n";
      return 1;
    }
    std::cout << loaded_latents.size() << " latent streams and "
              << loaded_hyper.size() << " hyper streams\n";

    // Load metadata
    std::cout << "Loading metadata...\n";
    CompressionResult comp;
    try {
      comp = load_compression_result_metadata(out_dir +
                                              "compression_metadata.bin");
    } catch (const std::exception &e) {
      std::cerr << "Error: " << e.what() << "\n";
      return 1;
    }

    // Assign loaded streams to result
    comp.encoded_latents = loaded_latents;
    comp.encoded_hyper_latents = loaded_hyper;

    // Decompress
    torch::Device decompression_device = select_model_device();

    auto start_timeD = std::chrono::high_resolution_clock::now();
    Decompressor decompressor;
    torch::Tensor recon = decompressor.decompress(comp);
    auto end_timeD = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> secondsD =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_timeD -
                                                                  start_timeD);
    std::cout << "\nTime taken for decompression: " << secondsD.count()
              << " s\n";

    if (!recon.defined() || recon.numel() == 0) {
      std::cerr << "FAIL: Reconstructed tensor is empty\n";
      return 1;
    }

    std::cout << "Reconstructed shape: " << recon.sizes() << "\n";

    // Shape restoration is part of decompression.
    torch::Tensor restored = recon;
    recon = torch::Tensor();

    // Calculate metrics
    std::cout << "\nCalculating metrics...\n";
    torch::Tensor recon_cpu = restored.to(torch::kCPU);
    restored = torch::Tensor();

    torch::Tensor diff = recon_cpu - raw_for_metrics;
    raw_for_metrics = torch::Tensor();

    float mse = diff.pow(2).mean().item<float>();
    diff = torch::Tensor();
    float rmse = std::sqrt(mse);
    float nrmse =
        rmse / (static_cast<float>(raw_max) - static_cast<float>(raw_min));
    float tolerance = std::numeric_limits<float>::epsilon();

    // Print results
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "=== QUALITY METRICS ===\n";
    std::cout << std::string(50, '=') << "\n";
    std::cout << "NRMSE:              " << nrmse << "\n";
    std::cout << "Error bound:        " << rel_eb << "\n";
    std::cout << "NRMSE <= Error bound: "
              << (nrmse <= rel_eb + tolerance ? "YES" : "NO") << "\n";

    bool passed = nrmse <= rel_eb + tolerance;
    std::cout << "\n";
    if (passed) {
      std::cout << "┌─────────────────────────────────────────┐\n";
      std::cout << "│        ✓ TEST PASSED ✓                  │\n";
      std::cout << "│  Reconstruction within error bounds     │\n";
      std::cout << "└─────────────────────────────────────────┘\n";
    } else {
      std::cout << "┌─────────────────────────────────────────┐\n";
      std::cout << "│        ✗ TEST FAILED ✗                  │\n";
      std::cout << "│  NRMSE exceeds error bound              │\n";
      std::cout << "│  Excess: " << (nrmse - rel_eb) << std::string(20, ' ')
                << "│\n";
      std::cout << "└─────────────────────────────────────────┘\n";
    }
    std::cout << std::string(50, '=') << "\n\n";

#ifndef _WIN32
    ModelCache &cache = ModelCache::instance();
    cache.clear();
#endif

    return passed ? 0 : 1;

  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
