#include "../CAESAR/data_utils.h"
#include "../CAESAR/dataset/dataset.h"
#include "../CAESAR/models/array_utils.h"
#include "../CAESAR/models/caesar_compress.h"
#include "../CAESAR/models/caesar_decompress.h"

bool save_encoded_streams(const std::vector<std::string> &streams,
                          const std::string &filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file to write: " << filename << std::endl;
    return false;
  }
  for (const auto &s : streams) {
    uint64_t len = static_cast<uint64_t>(s.size());
    file.write(reinterpret_cast<const char *>(&len), sizeof(len));
    if (len)
      file.write(s.data(), static_cast<std::streamsize>(len));
  }
  file.close();
  return true;
}

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

template <typename T> size_t get_vector_data_size(const std::vector<T> &vec) {
  if (vec.empty()) {
    return 0;
  }
  return vec.size() * sizeof(T);
}

template <typename T>
size_t get_2d_vector_data_size(const std::vector<std::vector<T>> &vec_2d) {
  size_t total_bytes = 0;
  for (const auto &inner_vec : vec_2d) {
    total_bytes += inner_vec.size() * sizeof(T);
  }
  return total_bytes;
}

size_t calculate_metadata_size(const CompressionResult &result) {
  size_t total_bytes = 0;

  total_bytes += get_vector_data_size(result.gae_comp_data);

  const auto &meta = result.compressionMetaData;

  total_bytes += get_vector_data_size(meta.offsets);
  total_bytes += get_vector_data_size(meta.scales);
  total_bytes += get_2d_vector_data_size(meta.indexes);
  total_bytes += sizeof(std::get<0>(meta.block_info));
  total_bytes += sizeof(std::get<1>(meta.block_info));
  total_bytes += get_vector_data_size(std::get<2>(meta.block_info));
  total_bytes += get_vector_data_size(meta.data_input_shape);
  total_bytes += get_vector_data_size(meta.filtered_blocks);
  total_bytes += sizeof(meta.global_scale);
  total_bytes += sizeof(meta.global_offset);
  total_bytes += sizeof(meta.pad_T);

  const auto &gae_meta = result.gaeMetaData;

  total_bytes += sizeof(gae_meta.GAE_correction_occur);
  total_bytes += get_vector_data_size(gae_meta.padding_recon_info);
  total_bytes += get_2d_vector_data_size(gae_meta.pcaBasis);
  total_bytes += get_vector_data_size(gae_meta.uniqueVals);
  total_bytes += sizeof(gae_meta.quanBin);
  total_bytes += sizeof(gae_meta.nVec);
  total_bytes += sizeof(gae_meta.prefixLength);
  total_bytes += sizeof(gae_meta.dataBytes);
  total_bytes += sizeof(gae_meta.coeffIntBytes);

  const auto &lbrc_meta = result.lbrcMetaData;
  total_bytes += sizeof(lbrc_meta.lbrc_correction_occur);
  total_bytes += sizeof(lbrc_meta.x_mean);
  total_bytes += sizeof(lbrc_meta.scale);
  total_bytes += sizeof(lbrc_meta.block_size);

  for (const auto &blk : result.lbrc_blocks) {
    total_bytes += sizeof(blk.step);
    total_bytes += sizeof(blk.bit_count);
    for (const auto &s : blk.streams)
      total_bytes += s.size();
  }

  return total_bytes;
}

bool save_padding_info(const PaddingInfo &padding_info,
                       const std::string &filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file to write: " << filename << std::endl;
    return false;
  }

  // Save original_shape
  uint64_t shape_size = padding_info.original_shape.size();
  file.write(reinterpret_cast<const char *>(&shape_size), sizeof(shape_size));
  for (int64_t dim : padding_info.original_shape) {
    file.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
  }

  // Save padded_shape
  uint64_t padded_size = padding_info.padded_shape.size();
  file.write(reinterpret_cast<const char *>(&padded_size), sizeof(padded_size));
  for (int64_t dim : padding_info.padded_shape) {
    file.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
  }

  // Save padding_values
  file.write(reinterpret_cast<const char *>(&padding_info.original_length),
             sizeof(padding_info.original_length));

  // Save H, W, was_padded
  file.write(reinterpret_cast<const char *>(&padding_info.H),
             sizeof(padding_info.H));
  file.write(reinterpret_cast<const char *>(&padding_info.W),
             sizeof(padding_info.W));
  file.write(reinterpret_cast<const char *>(&padding_info.was_padded),
             sizeof(padding_info.was_padded));
  file.close();
  return true;
}

bool save_compression_result_metadata(const CompressionResult &result,
                                      const std::string &filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file to write: " << filename << std::endl;
    return false;
  }

  const auto &meta = result.compressionMetaData;
  const auto &gae_meta = result.gaeMetaData;
  const auto &lbrc_meta = result.lbrcMetaData;

  // Helper lambda to write vector
  auto write_vector = [&file](const auto &vec) {
    uint64_t size = vec.size();
    file.write(reinterpret_cast<const char *>(&size), sizeof(size));
    if (!vec.empty()) {
      file.write(reinterpret_cast<const char *>(vec.data()),
                 static_cast<std::streamsize>(vec.size() * sizeof(vec[0])));
    }
  };

  // Helper lambda to write 2d vector
  auto write_2d_vector = [&file](const auto &vec_2d) {
    uint64_t outer_size = vec_2d.size();
    file.write(reinterpret_cast<const char *>(&outer_size), sizeof(outer_size));
    for (const auto &inner : vec_2d) {
      uint64_t inner_size = inner.size();
      file.write(reinterpret_cast<const char *>(&inner_size),
                 sizeof(inner_size));
      if (!inner.empty()) {
        file.write(
            reinterpret_cast<const char *>(inner.data()),
            static_cast<std::streamsize>(inner.size() * sizeof(inner[0])));
      }
    }
  };

  // Save compressionMetaData
  write_vector(meta.offsets);
  write_vector(meta.scales);
  write_2d_vector(meta.indexes);

  int32_t nH = std::get<0>(meta.block_info);
  int32_t nW = std::get<1>(meta.block_info);
  file.write(reinterpret_cast<const char *>(&nH), sizeof(nH));
  file.write(reinterpret_cast<const char *>(&nW), sizeof(nW));
  write_vector(std::get<2>(meta.block_info));

  write_vector(meta.data_input_shape);
  write_vector(meta.filtered_blocks);

  file.write(reinterpret_cast<const char *>(&meta.global_scale),
             sizeof(meta.global_scale));
  file.write(reinterpret_cast<const char *>(&meta.global_offset),
             sizeof(meta.global_offset));
  file.write(reinterpret_cast<const char *>(&meta.pad_T), sizeof(meta.pad_T));

  // Save gaeMetaData
  file.write(reinterpret_cast<const char *>(&gae_meta.GAE_correction_occur),
             sizeof(gae_meta.GAE_correction_occur));
  write_vector(gae_meta.padding_recon_info);
  write_2d_vector(gae_meta.pcaBasis);
  write_vector(gae_meta.uniqueVals);
  file.write(reinterpret_cast<const char *>(&gae_meta.quanBin),
             sizeof(gae_meta.quanBin));
  file.write(reinterpret_cast<const char *>(&gae_meta.nVec),
             sizeof(gae_meta.nVec));
  file.write(reinterpret_cast<const char *>(&gae_meta.prefixLength),
             sizeof(gae_meta.prefixLength));
  file.write(reinterpret_cast<const char *>(&gae_meta.dataBytes),
             sizeof(gae_meta.dataBytes));
  file.write(reinterpret_cast<const char *>(&gae_meta.coeffIntBytes),
             sizeof(gae_meta.coeffIntBytes));

  // Save gae_comp_data
  write_vector(result.gae_comp_data);

  // // Save latent_indexes
  // write_2d_vector(result.latent_indexes);

  // Save use_lbrc
  file.write(reinterpret_cast<const char *>(&result.use_lbrc),
             sizeof(result.use_lbrc));

  // Save lbrcMetaData
  file.write(reinterpret_cast<const char *>(&lbrc_meta.lbrc_correction_occur),
             sizeof(lbrc_meta.lbrc_correction_occur));
  file.write(reinterpret_cast<const char *>(&lbrc_meta.x_mean),
             sizeof(lbrc_meta.x_mean));
  file.write(reinterpret_cast<const char *>(&lbrc_meta.scale),
             sizeof(lbrc_meta.scale));
  file.write(reinterpret_cast<const char *>(&lbrc_meta.block_size),
             sizeof(lbrc_meta.block_size));

  // Save lbrc_blocks
  uint64_t num_blocks = result.lbrc_blocks.size();
  file.write(reinterpret_cast<const char *>(&num_blocks), sizeof(num_blocks));
  for (const auto &blk : result.lbrc_blocks) {
    file.write(reinterpret_cast<const char *>(&blk.step), sizeof(blk.step));
    file.write(reinterpret_cast<const char *>(&blk.bit_count),
               sizeof(blk.bit_count));

    uint64_t num_streams = blk.streams.size();
    file.write(reinterpret_cast<const char *>(&num_streams),
               sizeof(num_streams));
    for (const auto &s : blk.streams) {
      uint64_t stream_len = s.size();
      file.write(reinterpret_cast<const char *>(&stream_len),
                 sizeof(stream_len));
      if (stream_len) {
        file.write(reinterpret_cast<const char *>(s.data()),
                   static_cast<std::streamsize>(stream_len));
      }
    }
  }

  file.close();
  return true;
}

bool save_error_bound(float rel_eb, const std::string &filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file to write: " << filename << std::endl;
    return false;
  }
  file.write(reinterpret_cast<const char *>(&rel_eb), sizeof(rel_eb));
  file.close();
  return true;
}

int main(int argc, char *argv[]) {
  std::cout.setf(std::ios::unitbuf);
  try {
    std::set_terminate([]() {
      std::cerr << "FATAL: std::terminate() was called - "
                   "likely an uncaught exception on a non-main thread.\n";
      std::abort();
    });

    // Parse error bound from CLI
    float rel_eb = 0.0001f;
    if (argc > 1) {
      try {
        rel_eb = std::stof(argv[1]);
      } catch (...) {
        std::cerr
            << "Error: Invalid error bound argument. Using default 0.0001\n";
        rel_eb = 0.0001f;
      }
    }

    const std::vector<int64_t> shape = {1, 1, 20, 256, 256};
    const std::string raw_path = "TCf48.bin.f32";
    const std::string out_dir = "./output/";

    std::filesystem::create_directories(out_dir);

    const int batch_size = 128;
    const int n_frame = 8;
    torch::Tensor raw = loadRawBinary(raw_path, shape);

    raw = raw.squeeze();
    std::cout << "After squeeze, shape: " << raw.sizes() << "\n";

    torch::Tensor raw_5d;
    PaddingInfo padding_info;
    bool force_padding = false;

    if (shape.size() >= 5 && shape[3] >= 128 && shape[4] >= 128) {
      std::tie(raw_5d, padding_info) =
          to_5d_and_pad(raw, shape[3], shape[4], force_padding);
    } else if (shape.size() == 4 || shape.size() == 3) {
      std::tie(raw_5d, padding_info) =
          to_5d_and_pad(raw, 128, 128, force_padding);
    } else {
      std::tie(raw_5d, padding_info) =
          to_5d_and_pad(raw, 256, 256, force_padding);
    }

    raw = torch::Tensor();

    torch::Device compression_device = torch::Device(torch::kCPU);

    std::cout << "\n===== COMPRESSION =====\n";
    Compressor compressor(compression_device);

    DatasetConfig config;
    config.memory_data = raw_5d;
    config.variable_idx = 0;
    config.n_frame = n_frame;
    config.dataset_name = "TCf48 Dataset";
    config.section_range = std::nullopt;
    config.frame_range = std::nullopt;
    config.train_size = 256;
    config.inst_norm = true;
    config.norm_type = "mean_range";
    config.train_mode = false;
    config.n_overlap = 0;
    config.test_size = {256, 256};
    config.augment_type = {};

    raw_5d = torch::Tensor();

    std::cout << "Error bound for compression: " << rel_eb << "\n";
    auto start_timeC = std::chrono::high_resolution_clock::now();
    CompressionResult comp = compressor.compress(config, batch_size, rel_eb);
    auto end_timeC = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> secondsC =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_timeC -
                                                                  start_timeC);
    std::cout << "\nTime taken for compression: " << secondsC.count() << " s\n";

    // Save encoded streams
    std::string latents_file = out_dir + "encoded_latents.bin";
    std::string hyper_file = out_dir + "encoded_hyper_latents.bin";
    if (!save_encoded_streams(comp.encoded_latents, latents_file)) {
      std::cerr << "Failed to save encoded_latents\n";
      return 1;
    }
    if (!save_encoded_streams(comp.encoded_hyper_latents, hyper_file)) {
      std::cerr << "Failed to save encoded_hyper_latents\n";
      return 1;
    }
    std::cout << "Encoded streams written to " << out_dir << "\n";

    // Save metadata
    std::string metadata_file = out_dir + "compression_metadata.bin";
    if (!save_compression_result_metadata(comp, metadata_file)) {
      std::cerr << "Failed to save metadata\n";
      return 1;
    }
    std::cout << "Metadata written to " << metadata_file << "\n";

    // Save padding info
    std::string padding_file = out_dir + "padding_info.bin";
    if (!save_padding_info(padding_info, padding_file)) {
      std::cerr << "Failed to save padding info\n";
      return 1;
    }
    std::cout << "Padding info written to " << padding_file << "\n";

    // Save error bound
    std::string error_bound_file = out_dir + "error_bound.bin";
    if (!save_error_bound(rel_eb, error_bound_file)) {
      std::cerr << "Failed to save error bound\n";
      return 1;
    }
    std::cout << "Error bound written to " << error_bound_file << "\n";

    // Compression stats
    uint64_t compressed_bytes = 0;
    for (const auto &s : comp.encoded_latents)
      compressed_bytes += s.size();
    for (const auto &s : comp.encoded_hyper_latents)
      compressed_bytes += s.size();

    uint64_t num_elements = 1;
    for (auto d : shape)
      num_elements *= static_cast<uint64_t>(d);
    uint64_t uncompressed_bytes = num_elements * sizeof(float);

    size_t comp_all_meta_size = calculate_metadata_size(comp);

    double CR_without_meta = (compressed_bytes > 0)
                                 ? static_cast<double>(uncompressed_bytes) /
                                       static_cast<double>(compressed_bytes)
                                 : 0.0;
    double CR_with_meta = (compressed_bytes + comp_all_meta_size > 0)
                              ? static_cast<double>(uncompressed_bytes) /
                                    (static_cast<double>(compressed_bytes) +
                                     static_cast<double>(comp_all_meta_size))
                              : 0.0;

    std::cout << "\n=== Compression Stats ===\n";
    std::cout << "  Uncompressed: " << uncompressed_bytes << " bytes\n";
    std::cout << "  Compressed:   " << compressed_bytes << " bytes\n";
    std::cout << "  Metadata:     " << comp_all_meta_size << " bytes\n";
    std::cout << "  CR (no metadata):   " << CR_without_meta << "\n";
    std::cout << "  CR (with metadata): " << CR_with_meta << "\n";

#ifndef _WIN32
    ModelCache &cache = ModelCache::instance();
    cache.clear();
#endif

    std::cout
        << "\nCompressionPhase complete. Run test_caesar_D to decompress.\n";
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}