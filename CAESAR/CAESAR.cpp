#include "cli_result_header.h"
#include "data_utils.h"
#include "dataset/dataset.h"
#include "models/caesar_compress.h"
#include "models/caesar_decompress.h"

void save_complete_metadata(const std::string &filename,
                            const CompressionResult &comp) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open metadata file for writing: " +
                             filename);
  }

  caesar::cli::write_result_header(file, comp);

  size_t size;

  // Save CompressionMetaData
  const auto &meta = comp.compressionMetaData;

  size = meta.offsets.size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size));
  file.write(reinterpret_cast<const char *>(meta.offsets.data()),
             size * sizeof(float));

  size = meta.scales.size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size));
  file.write(reinterpret_cast<const char *>(meta.scales.data()),
             size * sizeof(float));

  size = meta.indexes.size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size));
  for (const auto &idx_vec : meta.indexes) {
    size_t inner_size = idx_vec.size();
    file.write(reinterpret_cast<const char *>(&inner_size), sizeof(inner_size));
    file.write(reinterpret_cast<const char *>(idx_vec.data()),
               inner_size * sizeof(int32_t));
  }

  auto nH = std::get<0>(meta.block_info);
  auto nW = std::get<1>(meta.block_info);
  file.write(reinterpret_cast<const char *>(&nH), sizeof(nH));
  file.write(reinterpret_cast<const char *>(&nW), sizeof(nW));
  size = std::get<2>(meta.block_info).size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size));
  file.write(
      reinterpret_cast<const char *>(std::get<2>(meta.block_info).data()),
      size * sizeof(int32_t));

  size = meta.data_input_shape.size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size));
  file.write(reinterpret_cast<const char *>(meta.data_input_shape.data()),
             size * sizeof(int32_t));

  size = meta.filtered_blocks.size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size));
  for (const auto &fb : meta.filtered_blocks) {
    file.write(reinterpret_cast<const char *>(&fb.first), sizeof(fb.first));
    file.write(reinterpret_cast<const char *>(&fb.second), sizeof(fb.second));
  }

  file.write(reinterpret_cast<const char *>(&meta.global_scale),
             sizeof(meta.global_scale));
  file.write(reinterpret_cast<const char *>(&meta.global_offset),
             sizeof(meta.global_offset));
  file.write(reinterpret_cast<const char *>(&meta.pad_T), sizeof(meta.pad_T));

  // Save GAEMetaData
  const auto &gae_meta = comp.gaeMetaData;

  file.write(reinterpret_cast<const char *>(&gae_meta.GAE_correction_occur),
             sizeof(gae_meta.GAE_correction_occur));

  size = gae_meta.padding_recon_info.size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size));
  file.write(reinterpret_cast<const char *>(gae_meta.padding_recon_info.data()),
             size * sizeof(int32_t));

  size = gae_meta.pcaBasis.size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size));
  for (const auto &basis_vec : gae_meta.pcaBasis) {
    size_t inner_size = basis_vec.size();
    file.write(reinterpret_cast<const char *>(&inner_size), sizeof(inner_size));
    file.write(reinterpret_cast<const char *>(basis_vec.data()),
               inner_size * sizeof(float));
  }

  size = gae_meta.uniqueVals.size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size));
  file.write(reinterpret_cast<const char *>(gae_meta.uniqueVals.data()),
             size * sizeof(float));

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
  size = comp.gae_comp_data.size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size));
  file.write(reinterpret_cast<const char *>(comp.gae_comp_data.data()), size);

  // Save correction_method
  file.write(reinterpret_cast<const char *>(&comp.correction_method),
             sizeof(comp.correction_method));

  // Save LBRCMetaData
  const auto &lbrc_meta = comp.lbrcMetaData;
  file.write(reinterpret_cast<const char *>(&lbrc_meta.lbrc_correction_occur),
             sizeof(lbrc_meta.lbrc_correction_occur));
  file.write(reinterpret_cast<const char *>(&lbrc_meta.x_mean),
             sizeof(lbrc_meta.x_mean));
  file.write(reinterpret_cast<const char *>(&lbrc_meta.scale),
             sizeof(lbrc_meta.scale));
  file.write(reinterpret_cast<const char *>(&lbrc_meta.block_size),
             sizeof(lbrc_meta.block_size));

  // Save lbrc_blocks
  size = comp.lbrc_blocks.size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size));
  for (const auto &blk : comp.lbrc_blocks) {
    file.write(reinterpret_cast<const char *>(&blk.step), sizeof(blk.step));
    file.write(reinterpret_cast<const char *>(&blk.bit_count),
               sizeof(blk.bit_count));

    size_t num_streams = blk.streams.size();
    file.write(reinterpret_cast<const char *>(&num_streams),
               sizeof(num_streams));
    for (const auto &s : blk.streams) {
      size_t stream_len = s.size();
      file.write(reinterpret_cast<const char *>(&stream_len),
                 sizeof(stream_len));
      if (stream_len) {
        file.write(reinterpret_cast<const char *>(s.data()),
                   static_cast<std::streamsize>(stream_len));
      }
    }
  }

  if (comp.correction_method == caesar::CorrectionMethod::NGLR) {
    const auto &m = comp.nglrMetaData;
    auto scalar = [&](const auto &value) {
      file.write(reinterpret_cast<const char *>(&value), sizeof(value));
    };
    auto vector = [&](const auto &values) {
      const uint64_t n = values.size();
      scalar(n);
      using T = typename std::decay_t<decltype(values)>::value_type;
      if (n)
        file.write(reinterpret_cast<const char *>(values.data()),
                   n * sizeof(T));
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
    scalar(static_cast<uint64_t>(m.weights.size()));
    for (const auto &weight : m.weights) {
      vector(weight.name);
      vector(weight.shape);
      vector(weight.values);
    }
    vector(comp.nglr_comp_data);
    if (!file)
      throw std::runtime_error("Failed writing NGLR metadata");
  }
  file.close();
}

CompressionResult load_complete_metadata(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open metadata file for reading: " +
                             filename);
  }

  CompressionResult comp;
  caesar::cli::read_result_header(file, comp);

  size_t size;

  // Load CompressionMetaData
  CompressionMetaData meta;

  file.read(reinterpret_cast<char *>(&size), sizeof(size));
  meta.offsets.resize(size);
  file.read(reinterpret_cast<char *>(meta.offsets.data()),
            size * sizeof(float));

  file.read(reinterpret_cast<char *>(&size), sizeof(size));
  meta.scales.resize(size);
  file.read(reinterpret_cast<char *>(meta.scales.data()), size * sizeof(float));

  file.read(reinterpret_cast<char *>(&size), sizeof(size));
  meta.indexes.resize(size);
  for (auto &idx_vec : meta.indexes) {
    size_t inner_size;
    file.read(reinterpret_cast<char *>(&inner_size), sizeof(inner_size));
    idx_vec.resize(inner_size);
    file.read(reinterpret_cast<char *>(idx_vec.data()),
              inner_size * sizeof(int32_t));
  }

  int32_t nH, nW;
  file.read(reinterpret_cast<char *>(&nH), sizeof(nH));
  file.read(reinterpret_cast<char *>(&nW), sizeof(nW));
  file.read(reinterpret_cast<char *>(&size), sizeof(size));
  std::vector<int32_t> padding(size);
  file.read(reinterpret_cast<char *>(padding.data()), size * sizeof(int32_t));
  meta.block_info = std::make_tuple(nH, nW, padding);

  file.read(reinterpret_cast<char *>(&size), sizeof(size));
  meta.data_input_shape.resize(size);
  file.read(reinterpret_cast<char *>(meta.data_input_shape.data()),
            size * sizeof(int32_t));

  file.read(reinterpret_cast<char *>(&size), sizeof(size));
  meta.filtered_blocks.resize(size);
  for (auto &fb : meta.filtered_blocks) {
    file.read(reinterpret_cast<char *>(&fb.first), sizeof(fb.first));
    file.read(reinterpret_cast<char *>(&fb.second), sizeof(fb.second));
  }

  file.read(reinterpret_cast<char *>(&meta.global_scale),
            sizeof(meta.global_scale));
  file.read(reinterpret_cast<char *>(&meta.global_offset),
            sizeof(meta.global_offset));
  file.read(reinterpret_cast<char *>(&meta.pad_T), sizeof(meta.pad_T));

  comp.compressionMetaData = meta;

  // Load GAEMetaData
  GAEMetaData gae_meta;

  file.read(reinterpret_cast<char *>(&gae_meta.GAE_correction_occur),
            sizeof(gae_meta.GAE_correction_occur));

  file.read(reinterpret_cast<char *>(&size), sizeof(size));
  gae_meta.padding_recon_info.resize(size);
  file.read(reinterpret_cast<char *>(gae_meta.padding_recon_info.data()),
            size * sizeof(int32_t));

  file.read(reinterpret_cast<char *>(&size), sizeof(size));
  gae_meta.pcaBasis.resize(size);
  for (auto &basis_vec : gae_meta.pcaBasis) {
    size_t inner_size;
    file.read(reinterpret_cast<char *>(&inner_size), sizeof(inner_size));
    basis_vec.resize(inner_size);
    file.read(reinterpret_cast<char *>(basis_vec.data()),
              inner_size * sizeof(float));
  }

  file.read(reinterpret_cast<char *>(&size), sizeof(size));
  gae_meta.uniqueVals.resize(size);
  file.read(reinterpret_cast<char *>(gae_meta.uniqueVals.data()),
            size * sizeof(float));

  file.read(reinterpret_cast<char *>(&gae_meta.quanBin),
            sizeof(gae_meta.quanBin));
  file.read(reinterpret_cast<char *>(&gae_meta.nVec), sizeof(gae_meta.nVec));
  file.read(reinterpret_cast<char *>(&gae_meta.prefixLength),
            sizeof(gae_meta.prefixLength));
  file.read(reinterpret_cast<char *>(&gae_meta.dataBytes),
            sizeof(gae_meta.dataBytes));
  file.read(reinterpret_cast<char *>(&gae_meta.coeffIntBytes),
            sizeof(gae_meta.coeffIntBytes));

  comp.gaeMetaData = gae_meta;

  // Load gae_comp_data
  file.read(reinterpret_cast<char *>(&size), sizeof(size));
  comp.gae_comp_data.resize(size);
  file.read(reinterpret_cast<char *>(comp.gae_comp_data.data()), size);

  // Load correction_method
  file.read(reinterpret_cast<char *>(&comp.correction_method),
            sizeof(comp.correction_method));

  // Load LBRCMetaData
  LBRCMetaData lbrc_meta;
  file.read(reinterpret_cast<char *>(&lbrc_meta.lbrc_correction_occur),
            sizeof(lbrc_meta.lbrc_correction_occur));
  file.read(reinterpret_cast<char *>(&lbrc_meta.x_mean),
            sizeof(lbrc_meta.x_mean));
  file.read(reinterpret_cast<char *>(&lbrc_meta.scale),
            sizeof(lbrc_meta.scale));
  file.read(reinterpret_cast<char *>(&lbrc_meta.block_size),
            sizeof(lbrc_meta.block_size));
  comp.lbrcMetaData = lbrc_meta;

  // Load lbrc_blocks
  file.read(reinterpret_cast<char *>(&size), sizeof(size));
  comp.lbrc_blocks.resize(size);
  for (auto &blk : comp.lbrc_blocks) {
    file.read(reinterpret_cast<char *>(&blk.step), sizeof(blk.step));
    file.read(reinterpret_cast<char *>(&blk.bit_count), sizeof(blk.bit_count));

    size_t num_streams;
    file.read(reinterpret_cast<char *>(&num_streams), sizeof(num_streams));
    blk.streams.resize(num_streams);
    for (auto &s : blk.streams) {
      size_t stream_len;
      file.read(reinterpret_cast<char *>(&stream_len), sizeof(stream_len));
      if (stream_len) {
        s.resize(stream_len);
        file.read(reinterpret_cast<char *>(s.data()),
                  static_cast<std::streamsize>(stream_len));
      }
    }
  }

  if (comp.correction_method == caesar::CorrectionMethod::NGLR) {
    auto &m = comp.nglrMetaData;
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
    vector(comp.nglr_comp_data);
    nglr::validate_metadata(m);
  }
  file.close();

  return comp;
}

torch::Tensor load_raw_binary(const std::string &bin_path,
                              const std::vector<int64_t> &shape,
                              bool verbose = false) {
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

  if (verbose) {
    std::cout << "Loaded " << bin_path << " with shape " << t.sizes() << "\n";
    std::cout << "  Min: " << t.min().item<float>()
              << ", Max: " << t.max().item<float>() << "\n";
  }
  return t;
}

void save_tensor_to_bin(const torch::Tensor &tensor,
                        const std::string &filename, bool verbose = false) {
  torch::Tensor cpu = tensor.to(torch::kCPU).contiguous();
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Error opening " + filename + " for write");
  }
  file.write(reinterpret_cast<const char *>(cpu.data_ptr<float>()),
             static_cast<std::streamsize>(cpu.numel() * sizeof(float)));
  file.close();
  if (verbose) {
    std::cout << "Saved tensor to " << filename << "\n";
  }
}

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
    throw std::runtime_error("Cannot open stream file: " + filename);
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

void print_usage(const char *program_name) {
  std::cout << "CAESAR Compression Tool\n\n";
  std::cout << "Usage:\n";
  std::cout << "  " << program_name << " compress <input> [options]\n";
  std::cout << "  " << program_name << " decompress <input> [options]\n\n";
  std::cout << "Common Options:\n";
  std::cout << "  -o, --output <file>      Output file path\n";
  std::cout << "  -s, --shape <shape>      Data shape (e.g., 1,24,256,256)\n";
  std::cout
      << "  -f, --n-frame <n>        Required for compression (must be 8)\n";
  std::cout << "  -t, --timing             Show timing information\n";
  std::cout << "  -v, --verbose            Verbose output\n";
  std::cout << "  -q, --quiet              Suppress output\n";
  std::cout << "  -h, --help               Show this help message\n\n";
  std::cout << "Compression Options:\n";
  std::cout << "  -e, --error-bound <val>  Error bound (default: 0.001)\n";
  std::cout << "  --correction <method>    gae (default), lbrc, or nglr\n";
  std::cout << "  --metadata               Show detailed metadata\n";
  std::cout << "  --metrics-csv <file>     Save metrics to CSV\n\n";
  std::cout << "Decompression Options:\n";
  std::cout << "  --verify                  Verify reconstruction\n";
  std::cout << "  --original <file>         Original file for verification\n";
}

std::vector<int64_t> parse_shape(const std::string &shape_str) {
  std::vector<int64_t> shape;
  std::stringstream ss(shape_str);
  std::string item;
  while (std::getline(ss, item, ',')) {
    shape.push_back(std::stoll(item));
  }
  return shape;
}

double calculate_psnr(const torch::Tensor &original,
                      const torch::Tensor &reconstructed) {
  torch::Tensor orig_cpu = original.to(torch::kCPU);
  torch::Tensor recon_cpu = reconstructed.to(torch::kCPU);

  double max_val = orig_cpu.max().item<double>();
  double min_val = orig_cpu.min().item<double>();
  double range = max_val - min_val;

  torch::Tensor diff = recon_cpu - orig_cpu;
  double mse = diff.pow(2).mean().item<double>();

  if (mse == 0.0)
    return std::numeric_limits<double>::infinity();

  double psnr = 20.0 * std::log10(range) - 10.0 * std::log10(mse);
  return psnr;
}

void save_metrics_to_csv(
    const std::string &filename, const std::string &input_file,
    const std::vector<int64_t> &shape, double compression_time,
    double decompression_time, uint64_t uncompressed_bytes,
    uint64_t compressed_bytes, size_t metadata_bytes, double cr_with_meta,
    double cr_without_meta, double nrmse, double psnr, float error_bound,
    int batch_size, int n_frame, const std::string &model_type,
    const std::string &compress_device, const std::string &decompress_device) {
  bool file_exists = std::filesystem::exists(filename);
  std::ofstream file(filename, std::ios::app);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open CSV file: " << filename << std::endl;
    return;
  }

  if (!file_exists) {
    file << "timestamp,input_file,shape,error_bound,batch_size,n_frame,model,"
         << "compress_device,decompress_device,"
         << "uncompressed_bytes,compressed_bytes,metadata_bytes,"
         << "cr_with_meta,cr_without_meta,"
         << "compression_time_s,decompression_time_s,total_time_s,"
         << "nrmse,psnr\n";
  }

  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  std::stringstream timestamp;
  timestamp << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");

  std::stringstream shape_str;
  shape_str << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    shape_str << shape[i];
    if (i < shape.size() - 1)
      shape_str << "x";
  }
  shape_str << "]";

  file << timestamp.str() << "," << input_file << "," << shape_str.str() << ","
       << error_bound << "," << batch_size << "," << n_frame << ","
       << model_type << "," << compress_device << "," << decompress_device
       << "," << uncompressed_bytes << "," << compressed_bytes << ","
       << metadata_bytes << "," << std::fixed << std::setprecision(4)
       << cr_with_meta << "," << cr_without_meta << "," << std::setprecision(6)
       << compression_time << "," << decompression_time << ","
       << (compression_time + decompression_time) << "," << std::setprecision(8)
       << nrmse << "," << std::setprecision(4) << psnr << "\n";

  file.close();
}

int compress_file(const std::string &input_file, const std::string &output_file,
                  const std::vector<int64_t> &shape, float error_bound,
                  int n_frame, const std::string &model_type, bool show_timing,
                  bool show_metadata, bool verbose, bool quiet,
                  const std::string &metrics_csv,
                  caesar::CorrectionMethod correction_method) {
  const auto compress_device = select_model_device();
  if (!quiet) {
    std::cout << "=== CAESAR COMPRESSION ===\n";
    std::cout << "Input file: " << input_file << "\n";
    std::cout << "Output file: " << output_file << "\n";
    std::cout << "Model: " << model_type << "\n";
    std::cout << "Compression device: " << compress_device << "\n";
    std::cout << "Error bound: " << error_bound << "\n";
    std::cout << "N-frame: " << n_frame << "\n\n";
  }

  torch::Tensor raw = load_raw_binary(input_file, shape, verbose);

  if (verbose) {
    std::cout << "Input shape: " << raw.sizes() << "\n";
  }

  Compressor compressor;

  CompressionConfig config;
  config.memory_data = raw;

  config.n_frame = n_frame;

  auto start_time_c = std::chrono::high_resolution_clock::now();
  config.correction_method = correction_method;
  CompressionResult comp = compressor.compress(config, error_bound);
  auto end_time_c = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> compression_time = end_time_c - start_time_c;

  if (show_timing || verbose) {
    std::cout << "\n  Compression time: " << compression_time.count() << " s\n";
  }

  std::string base_output =
      output_file.empty() ? input_file + ".cae" : output_file;
  std::string latents_file = base_output + ".latents";
  std::string hyper_file = base_output + ".hyper";
  std::string metadata_file = base_output + ".meta";

  if (!save_encoded_streams(comp.encoded_latents, latents_file)) {
    std::cerr << "Failed to save encoded_latents\n";
    return 1;
  }
  if (!save_encoded_streams(comp.encoded_hyper_latents, hyper_file)) {
    std::cerr << "Failed to save encoded_hyper_latents\n";
    return 1;
  }

  try {
    save_complete_metadata(metadata_file, comp);
  } catch (const std::exception &e) {
    std::cerr << "Failed to save metadata: " << e.what() << "\n";
    return 1;
  }

  if (!quiet) {
    std::cout << "\n Compression complete!\n";
    std::cout << "Output files:\n";
    std::cout << "  - " << latents_file << "\n";
    std::cout << "  - " << hyper_file << "\n";
    std::cout << "  - " << metadata_file << "\n";
  }

  return 0;
}

int decompress_file(const std::string &input_base,
                    const std::string &output_file, bool show_timing,
                    bool verbose, bool quiet, bool verify,
                    const std::string &original_file,
                    const std::string &metrics_csv) {
  const auto decompress_device = select_model_device();
  if (!quiet) {
    std::cout << "=== CAESAR DECOMPRESSION ===\n";
    std::cout << "Input base: " << input_base << "\n";
    std::cout << "Output file: " << output_file << "\n";
    std::cout << "Decompression device: " << decompress_device << "\n\n";
  }

  std::string latents_file = input_base + ".latents";
  std::string hyper_file = input_base + ".hyper";
  std::string metadata_file = input_base + ".meta";

  std::vector<std::string> loaded_latents = load_encoded_streams(latents_file);
  std::vector<std::string> loaded_hyper = load_encoded_streams(hyper_file);

  if (verbose) {
    std::cout << "Loaded " << loaded_latents.size() << " latent streams and "
              << loaded_hyper.size() << " hyper streams\n";
  }

  CompressionResult comp;

  try {
    comp = load_complete_metadata(metadata_file);
  } catch (const std::exception &e) {
    std::cerr << "Failed to load metadata: " << e.what() << "\n";
    return 1;
  }

  comp.encoded_latents = loaded_latents;
  comp.encoded_hyper_latents = loaded_hyper;

  std::cout << "Metadata loaded successfully\n";

  auto start_time_d = std::chrono::high_resolution_clock::now();
  Decompressor decompressor;
  torch::Tensor recon = decompressor.decompress(comp);
  auto end_time_d = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> decompression_time = end_time_d - start_time_d;

  if (show_timing || verbose) {
    std::cout << "\n  Decompression time: " << decompression_time.count()
              << " s\n";
  }

  if (!recon.defined() || recon.numel() == 0) {
    std::cerr << "Error: Decompression failed - empty tensor\n";
    return 1;
  }

  if (verbose) {
    std::cout << "Reconstructed tensor shape: " << recon.sizes() << "\n";
  }

  torch::Tensor restored = recon;

  if (verbose) {
    std::cout << "Restored tensor shape: " << restored.sizes() << "\n";
  }

  save_tensor_to_bin(restored, output_file, verbose);

  if (verify && !original_file.empty()) {
    if (!quiet)
      std::cout << "\n Verifying reconstruction...\n";

    torch::Tensor original =
        load_raw_binary(original_file, comp.original_shape, false);
    if (original.dim() == 5)
      original = original.narrow(0, 0, 1);

    torch::Tensor orig_cpu = original.to(torch::kCPU);
    torch::Tensor recon_cpu = restored.to(torch::kCPU);

    torch::Tensor diff = recon_cpu - orig_cpu;
    double mse = diff.pow(2).mean().item<double>();
    double rmse = std::sqrt(mse);
    double range =
        orig_cpu.max().item<double>() - orig_cpu.min().item<double>();
    double nrmse = range == 0.0 ? rmse : rmse / range;
    double psnr = calculate_psnr(original, restored);

    if (!quiet) {
      std::cout << "\n Quality Metrics:\n";
      std::cout << "  NRMSE: " << std::scientific << std::setprecision(6)
                << nrmse << "\n";
      std::cout << "  PSNR:  " << std::fixed << std::setprecision(2) << psnr
                << " dB\n";
    }

    if (!metrics_csv.empty()) {
      uint64_t num_elements = 1;
      for (auto d : comp.shape_info.original_shape)
        num_elements *= static_cast<uint64_t>(d);
      uint64_t uncompressed_bytes = num_elements * sizeof(float);

      save_metrics_to_csv(metrics_csv, original_file,
                          comp.shape_info.original_shape, 0.0,
                          decompression_time.count(), uncompressed_bytes, 0, 0,
                          0.0, 0.0, nrmse, psnr, 0.0, 128, comp.n_frame, "V",
                          "N/A", decompress_device.is_cuda() ? "cuda" : "cpu");
    }
  }

  if (!quiet) {
    std::cout << "\n Decompression complete!\n";
    std::cout << "Output: " << output_file << "\n";
  }

  return 0;
}

int main(int argc, char *argv[]) {
  try {
    if (argc < 2) {
      print_usage(argv[0]);
      return 1;
    }

    std::string command = argv[1];
    if (command == "-h" || command == "--help") {
      print_usage(argv[0]);
      return 0;
    }

    if (command != "compress" && command != "decompress") {
      std::cerr << "Error: Unknown command '" << command << "'\n";
      print_usage(argv[0]);
      return 1;
    }

    if (argc < 3) {
      std::cerr << "Error: Missing input file\n";
      print_usage(argv[0]);
      return 1;
    }

    std::string input_file = argv[2];
    std::string output_file;
    std::vector<int64_t> shape;
    float error_bound = 0.001f;
    auto correction_method = caesar::CorrectionMethod::GAE;
    int n_frame = 0;
    std::string model_type = get_model_name();
    bool show_timing = false;
    bool show_metadata = false;
    bool verbose = false;
    bool quiet = false;
    bool verify = false;
    std::string metrics_csv;
    std::string original_file;

    for (int i = 3; i < argc; ++i) {
      std::string arg = argv[i];

      if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
        output_file = argv[++i];
      } else if ((arg == "-s" || arg == "--shape") && i + 1 < argc) {
        shape = parse_shape(argv[++i]);
      } else if ((arg == "-e" || arg == "--error-bound") && i + 1 < argc) {
        error_bound = std::stof(argv[++i]);
      } else if ((arg == "-f" || arg == "--n-frame") && i + 1 < argc) {
        n_frame = std::stoi(argv[++i]);
      } else if (arg == "--correction" && i + 1 < argc) {
        correction_method = caesar::correction_method_from_string(argv[++i]);
      } else if (arg == "-t" || arg == "--timing") {
        show_timing = true;
      } else if (arg == "--metadata") {
        show_metadata = true;
      } else if (arg == "-v" || arg == "--verbose") {
        verbose = true;
      } else if (arg == "-q" || arg == "--quiet") {
        quiet = true;
      } else if (arg == "--verify") {
        verify = true;
      } else if (arg == "--metrics-csv" && i + 1 < argc) {
        metrics_csv = argv[++i];
      } else if (arg == "--original" && i + 1 < argc) {
        original_file = argv[++i];
      } else {
        throw std::invalid_argument("Unknown option or missing value: " + arg);
      }
    }

    if (command == "compress") {
      if (shape.empty()) {
        std::cerr
            << "Error: Shape is required for compression (-s or --shape)\n";
        return 1;
      }

      if (n_frame != 8)
        throw std::invalid_argument("Compression requires --n-frame 8");

      if (output_file.empty()) {
        output_file = input_file + ".cae";
      }

      return compress_file(input_file, output_file, shape, error_bound, n_frame,
                           model_type, show_timing, show_metadata, verbose,
                           quiet, metrics_csv, correction_method);

    } else if (command == "decompress") {
      if (n_frame != 0 || !shape.empty())
        throw std::invalid_argument("Decompression reads shape and n_frame "
                                    "from metadata; omit -s and -f");

      if (output_file.empty()) {
        std::string base = input_file;
        if (base.size() >= 4 && base.substr(base.size() - 4) == ".cae") {
          base = base.substr(0, base.size() - 4);
        }
        output_file = base + ".decompressed.bin";
      }

      if (verify && original_file.empty()) {
        std::cerr << "Warning: --verify requires --original <file>\n";
        verify = false;
      }

      return decompress_file(input_file, output_file, show_timing, verbose,
                             quiet, verify, original_file, metrics_csv);
    }

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
