#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "data_utils.h"
#include "dataset/dataset.h"
#include "models/array_utils.h"
#include "models/caesar_compress.h"
#include "models/caesar_decompress.h"

void save_complete_metadata(const std::string& filename,
                            const PaddingInfo& padding_info,
                            const CompressionResult& comp) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open metadata file for writing: " +
                             filename);
  }

  // Save PaddingInfo
  size_t size = padding_info.original_shape.size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));
  file.write(reinterpret_cast<const char*>(padding_info.original_shape.data()),
             size * sizeof(int64_t));

  file.write(reinterpret_cast<const char*>(&padding_info.original_length),
             sizeof(padding_info.original_length));

  size = padding_info.padded_shape.size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));
  file.write(reinterpret_cast<const char*>(padding_info.padded_shape.data()),
             size * sizeof(int64_t));

  file.write(reinterpret_cast<const char*>(&padding_info.H),
             sizeof(padding_info.H));
  file.write(reinterpret_cast<const char*>(&padding_info.W),
             sizeof(padding_info.W));
  file.write(reinterpret_cast<const char*>(&padding_info.was_padded),
             sizeof(padding_info.was_padded));

  // Save num_samples and num_batches
  file.write(reinterpret_cast<const char*>(&comp.num_samples),
             sizeof(comp.num_samples));
  file.write(reinterpret_cast<const char*>(&comp.num_batches),
             sizeof(comp.num_batches));

  // Save CompressionMetaData
  const auto& meta = comp.compressionMetaData;

  size = meta.offsets.size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));
  file.write(reinterpret_cast<const char*>(meta.offsets.data()),
             size * sizeof(float));

  size = meta.scales.size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));
  file.write(reinterpret_cast<const char*>(meta.scales.data()),
             size * sizeof(float));

  size = meta.indexes.size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));
  for (const auto& idx_vec : meta.indexes) {
    size_t inner_size = idx_vec.size();
    file.write(reinterpret_cast<const char*>(&inner_size), sizeof(inner_size));
    file.write(reinterpret_cast<const char*>(idx_vec.data()),
               inner_size * sizeof(int32_t));
  }

  auto nH = std::get<0>(meta.block_info);
  auto nW = std::get<1>(meta.block_info);
  file.write(reinterpret_cast<const char*>(&nH), sizeof(nH));
  file.write(reinterpret_cast<const char*>(&nW), sizeof(nW));
  size = std::get<2>(meta.block_info).size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));
  file.write(reinterpret_cast<const char*>(std::get<2>(meta.block_info).data()),
             size * sizeof(int32_t));

  size = meta.data_input_shape.size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));
  file.write(reinterpret_cast<const char*>(meta.data_input_shape.data()),
             size * sizeof(int32_t));

  size = meta.filtered_blocks.size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));
  for (const auto& fb : meta.filtered_blocks) {
    file.write(reinterpret_cast<const char*>(&fb.first), sizeof(fb.first));
    file.write(reinterpret_cast<const char*>(&fb.second), sizeof(fb.second));
  }

  file.write(reinterpret_cast<const char*>(&meta.global_scale),
             sizeof(meta.global_scale));
  file.write(reinterpret_cast<const char*>(&meta.global_offset),
             sizeof(meta.global_offset));
  file.write(reinterpret_cast<const char*>(&meta.pad_T), sizeof(meta.pad_T));

  // Save GAEMetaData
  const auto& gae_meta = comp.gaeMetaData;

  file.write(reinterpret_cast<const char*>(&gae_meta.GAE_correction_occur),
             sizeof(gae_meta.GAE_correction_occur));

  size = gae_meta.padding_recon_info.size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));
  file.write(reinterpret_cast<const char*>(gae_meta.padding_recon_info.data()),
             size * sizeof(int32_t));

  size = gae_meta.pcaBasis.size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));
  for (const auto& basis_vec : gae_meta.pcaBasis) {
    size_t inner_size = basis_vec.size();
    file.write(reinterpret_cast<const char*>(&inner_size), sizeof(inner_size));
    file.write(reinterpret_cast<const char*>(basis_vec.data()),
               inner_size * sizeof(float));
  }

  size = gae_meta.uniqueVals.size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));
  file.write(reinterpret_cast<const char*>(gae_meta.uniqueVals.data()),
             size * sizeof(float));

  file.write(reinterpret_cast<const char*>(&gae_meta.quanBin),
             sizeof(gae_meta.quanBin));
  file.write(reinterpret_cast<const char*>(&gae_meta.nVec),
             sizeof(gae_meta.nVec));
  file.write(reinterpret_cast<const char*>(&gae_meta.prefixLength),
             sizeof(gae_meta.prefixLength));
  file.write(reinterpret_cast<const char*>(&gae_meta.dataBytes),
             sizeof(gae_meta.dataBytes));
  file.write(reinterpret_cast<const char*>(&gae_meta.coeffIntBytes),
             sizeof(gae_meta.coeffIntBytes));

  // Save gae_comp_data
  size = comp.gae_comp_data.size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));
  file.write(reinterpret_cast<const char*>(comp.gae_comp_data.data()), size);

  file.close();
}

CompressionResult load_complete_metadata(const std::string& filename,
                                         PaddingInfo& padding_info) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open metadata file for reading: " +
                             filename);
  }

  CompressionResult comp;

  // Load PaddingInfo
  size_t size;
  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  padding_info.original_shape.resize(size);
  file.read(reinterpret_cast<char*>(padding_info.original_shape.data()),
            size * sizeof(int64_t));

  file.read(reinterpret_cast<char*>(&padding_info.original_length),
            sizeof(padding_info.original_length));

  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  padding_info.padded_shape.resize(size);
  file.read(reinterpret_cast<char*>(padding_info.padded_shape.data()),
            size * sizeof(int64_t));

  file.read(reinterpret_cast<char*>(&padding_info.H), sizeof(padding_info.H));
  file.read(reinterpret_cast<char*>(&padding_info.W), sizeof(padding_info.W));
  file.read(reinterpret_cast<char*>(&padding_info.was_padded),
            sizeof(padding_info.was_padded));

  // Load num_samples and num_batches
  file.read(reinterpret_cast<char*>(&comp.num_samples),
            sizeof(comp.num_samples));
  file.read(reinterpret_cast<char*>(&comp.num_batches),
            sizeof(comp.num_batches));

  // Load CompressionMetaData
  CompressionMetaData meta;

  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  meta.offsets.resize(size);
  file.read(reinterpret_cast<char*>(meta.offsets.data()), size * sizeof(float));

  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  meta.scales.resize(size);
  file.read(reinterpret_cast<char*>(meta.scales.data()), size * sizeof(float));

  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  meta.indexes.resize(size);
  for (auto& idx_vec : meta.indexes) {
    size_t inner_size;
    file.read(reinterpret_cast<char*>(&inner_size), sizeof(inner_size));
    idx_vec.resize(inner_size);
    file.read(reinterpret_cast<char*>(idx_vec.data()),
              inner_size * sizeof(int32_t));
  }

  int32_t nH, nW;
  file.read(reinterpret_cast<char*>(&nH), sizeof(nH));
  file.read(reinterpret_cast<char*>(&nW), sizeof(nW));
  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  std::vector<int32_t> padding(size);
  file.read(reinterpret_cast<char*>(padding.data()), size * sizeof(int32_t));
  meta.block_info = std::make_tuple(nH, nW, padding);

  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  meta.data_input_shape.resize(size);
  file.read(reinterpret_cast<char*>(meta.data_input_shape.data()),
            size * sizeof(int32_t));

  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  meta.filtered_blocks.resize(size);
  for (auto& fb : meta.filtered_blocks) {
    file.read(reinterpret_cast<char*>(&fb.first), sizeof(fb.first));
    file.read(reinterpret_cast<char*>(&fb.second), sizeof(fb.second));
  }

  file.read(reinterpret_cast<char*>(&meta.global_scale),
            sizeof(meta.global_scale));
  file.read(reinterpret_cast<char*>(&meta.global_offset),
            sizeof(meta.global_offset));
  file.read(reinterpret_cast<char*>(&meta.pad_T), sizeof(meta.pad_T));

  comp.compressionMetaData = meta;

  // Load GAEMetaData
  GAEMetaData gae_meta;

  file.read(reinterpret_cast<char*>(&gae_meta.GAE_correction_occur),
            sizeof(gae_meta.GAE_correction_occur));

  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  gae_meta.padding_recon_info.resize(size);
  file.read(reinterpret_cast<char*>(gae_meta.padding_recon_info.data()),
            size * sizeof(int32_t));

  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  gae_meta.pcaBasis.resize(size);
  for (auto& basis_vec : gae_meta.pcaBasis) {
    size_t inner_size;
    file.read(reinterpret_cast<char*>(&inner_size), sizeof(inner_size));
    basis_vec.resize(inner_size);
    file.read(reinterpret_cast<char*>(basis_vec.data()),
              inner_size * sizeof(float));
  }

  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  gae_meta.uniqueVals.resize(size);
  file.read(reinterpret_cast<char*>(gae_meta.uniqueVals.data()),
            size * sizeof(float));

  file.read(reinterpret_cast<char*>(&gae_meta.quanBin),
            sizeof(gae_meta.quanBin));
  file.read(reinterpret_cast<char*>(&gae_meta.nVec), sizeof(gae_meta.nVec));
  file.read(reinterpret_cast<char*>(&gae_meta.prefixLength),
            sizeof(gae_meta.prefixLength));
  file.read(reinterpret_cast<char*>(&gae_meta.dataBytes),
            sizeof(gae_meta.dataBytes));
  file.read(reinterpret_cast<char*>(&gae_meta.coeffIntBytes),
            sizeof(gae_meta.coeffIntBytes));

  comp.gaeMetaData = gae_meta;

  // Load gae_comp_data
  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  comp.gae_comp_data.resize(size);
  file.read(reinterpret_cast<char*>(comp.gae_comp_data.data()), size);

  file.close();

  return comp;
}

torch::Tensor load_raw_binary(const std::string& bin_path,
                              const std::vector<int64_t>& shape,
                              bool verbose = false) {
  std::ifstream file(bin_path, std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Cannot open binary file: " + bin_path);

  size_t num_elems = 1;
  for (auto d : shape) {
    if (d <= 0) throw std::runtime_error("Invalid shape dimension");
    num_elems *= static_cast<size_t>(d);
  }

  std::vector<float> buf(num_elems);
  file.read(reinterpret_cast<char*>(buf.data()),
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

void save_tensor_to_bin(const torch::Tensor& tensor,
                        const std::string& filename, bool verbose = false) {
  torch::Tensor cpu = tensor.to(torch::kCPU).contiguous();
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Error opening " + filename + " for write");
  }
  file.write(reinterpret_cast<const char*>(cpu.data_ptr<float>()),
             static_cast<std::streamsize>(cpu.numel() * sizeof(float)));
  file.close();
  if (verbose) {
    std::cout << "Saved tensor to " << filename << "\n";
  }
}

bool save_encoded_streams(const std::vector<std::string>& streams,
                          const std::string& filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file to write: " << filename << std::endl;
    return false;
  }
  for (const auto& s : streams) {
    uint64_t len = static_cast<uint64_t>(s.size());
    file.write(reinterpret_cast<const char*>(&len), sizeof(len));
    if (len) file.write(s.data(), static_cast<std::streamsize>(len));
  }
  file.close();
  return true;
}

std::vector<std::string> load_encoded_streams(const std::string& filename) {
  std::vector<std::string> out;
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file to read: " << filename << std::endl;
    return out;
  }
  uint64_t len;
  while (file.read(reinterpret_cast<char*>(&len), sizeof(len))) {
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

void print_usage(const char* program_name) {
  std::cout << "CAESAR Compression Tool\n\n";
  std::cout << "Usage:\n";
  std::cout << "  " << program_name << " compress <input> [options]\n";
  std::cout << "  " << program_name << " decompress <input> [options]\n\n";
  std::cout << "Common Options:\n";
  std::cout << "  -o, --output <file>      Output file path\n";
  std::cout << "  -s, --shape <shape>      Data shape (e.g., 1,24,256,256)\n";
  std::cout << "  -b, --batch-size <n>     Batch size (default: 128)\n";
  std::cout << "  -f, --n-frame <n>        Number of frames (default: 8)\n";
  std::cout << "  -t, --timing             Show timing information\n";
  std::cout << "  -v, --verbose            Verbose output\n";
  std::cout << "  -q, --quiet              Suppress output\n";
  std::cout << "  -h, --help               Show this help message\n\n";
  std::cout << "Compression Options:\n";
  std::cout << "  -e, --error-bound <val>  Error bound (default: 0.001)\n";
  std::cout << "  -m, --model <V|D>        Model type (default: V)\n";
  std::cout << "  --compress-device <dev>  Device (cpu/cuda)\n";
  std::cout << "  --metadata               Show detailed metadata\n";
  std::cout << "  --force-padding          Force padding\n";
  std::cout << "  --metrics-csv <file>     Save metrics to CSV\n\n";
  std::cout << "Decompression Options:\n";
  std::cout << "  --decompress-device <dev> Device (cpu/cuda)\n";
  std::cout << "  --verify                  Verify reconstruction\n";
  std::cout << "  --original <file>         Original file for verification\n";
}

std::vector<int64_t> parse_shape(const std::string& shape_str) {
  std::vector<int64_t> shape;
  std::stringstream ss(shape_str);
  std::string item;
  while (std::getline(ss, item, ',')) {
    shape.push_back(std::stoll(item));
  }
  return shape;
}

torch::Device auto_select_device() {
  if (torch::cuda::is_available()) {
    return torch::Device(torch::kCUDA, 0);
  }
  return torch::Device(torch::kCPU);
}

torch::Device parse_device(const std::string& device_str) {
  if (device_str == "cpu") {
    return torch::Device(torch::kCPU);
  } else if (device_str.substr(0, 4) == "cuda") {
    if (!torch::cuda::is_available()) {
      std::cerr << "Warning: CUDA not available, using CPU\n";
      return torch::Device(torch::kCPU);
    }
    if (device_str.size() > 5 && device_str[4] == ':') {
      int device_id = std::stoi(device_str.substr(5));
      return torch::Device(torch::kCUDA, device_id);
    }
    return torch::Device(torch::kCUDA, 0);
  }
  throw std::runtime_error("Invalid device string: " + device_str);
}

double calculate_psnr(const torch::Tensor& original,
                      const torch::Tensor& reconstructed) {
  torch::Tensor orig_cpu = original.to(torch::kCPU);
  torch::Tensor recon_cpu = reconstructed.to(torch::kCPU);

  double max_val = orig_cpu.max().item<double>();
  double min_val = orig_cpu.min().item<double>();
  double range = max_val - min_val;

  torch::Tensor diff = recon_cpu - orig_cpu;
  double mse = diff.pow(2).mean().item<double>();

  if (mse == 0.0) return std::numeric_limits<double>::infinity();

  double psnr = 20.0 * std::log10(range) - 10.0 * std::log10(mse);
  return psnr;
}

void save_metrics_to_csv(
    const std::string& filename, const std::string& input_file,
    const std::vector<int64_t>& shape, double compression_time,
    double decompression_time, uint64_t uncompressed_bytes,
    uint64_t compressed_bytes, size_t metadata_bytes, double cr_with_meta,
    double cr_without_meta, double nrmse, double psnr, float error_bound,
    int batch_size, int n_frame, const std::string& model_type,
    const std::string& compress_device, const std::string& decompress_device) {
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
    if (i < shape.size() - 1) shape_str << "x";
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

int compress_file(const std::string& input_file, const std::string& output_file,
                  const std::vector<int64_t>& shape, float error_bound,
                  int batch_size, int n_frame, const std::string& model_type,
                  torch::Device compress_device, bool show_timing,
                  bool show_metadata, bool verbose, bool quiet,
                  bool force_padding, const std::string& metrics_csv) {
  if (!quiet) {
    std::cout << "=== CAESAR COMPRESSION ===\n";
    std::cout << "Input file: " << input_file << "\n";
    std::cout << "Output file: " << output_file << "\n";
    std::cout << "Model: CAESAR-" << model_type << "\n";
    std::cout << "Compression device: " << compress_device << "\n";
    std::cout << "Error bound: " << error_bound << "\n";
    std::cout << "Batch size: " << batch_size << "\n";
    std::cout << "N-frame: " << n_frame << "\n\n";
  }

  torch::Tensor raw = load_raw_binary(input_file, shape, verbose);
  raw = raw.squeeze();

  if (verbose) {
    std::cout << "After squeeze, shape: " << raw.sizes() << "\n";
  }

  torch::Tensor raw_5d;
  PaddingInfo padding_info;

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

  Compressor compressor(compress_device);

  DatasetConfig config;
  config.memory_data = raw_5d;
  config.device = compress_device;
  config.variable_idx = 0;
  config.n_frame = n_frame;
  config.dataset_name = "CAESAR Compression Dataset";
  config.section_range = std::nullopt;
  config.frame_range = std::nullopt;
  config.train_size = 256;
  config.inst_norm = true;
  config.norm_type = "mean_range";
  config.train_mode = false;
  config.n_overlap = 0;
  config.test_size = {256, 256};
  config.augment_type = {};

  auto start_time_c = std::chrono::high_resolution_clock::now();
  CompressionResult comp = compressor.compress(config, batch_size, error_bound);
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
    save_complete_metadata(metadata_file, padding_info, comp);
  } catch (const std::exception& e) {
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

int decompress_file(const std::string& input_base,
                    const std::string& output_file,
                    const std::vector<int64_t>& original_shape, int batch_size,
                    int n_frame, torch::Device decompress_device,
                    bool show_timing, bool verbose, bool quiet, bool verify,
                    const std::string& original_file,
                    const std::string& metrics_csv) {
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

  if (loaded_latents.empty() || loaded_hyper.empty()) {
    std::cerr << "Error: Failed to load compressed streams\n";
    return 1;
  }

  if (verbose) {
    std::cout << "Loaded " << loaded_latents.size() << " latent streams and "
              << loaded_hyper.size() << " hyper streams\n";
  }

  PaddingInfo padding_info;
  CompressionResult comp;

  try {
    comp = load_complete_metadata(metadata_file, padding_info);
  } catch (const std::exception& e) {
    std::cerr << "Failed to load metadata: " << e.what() << "\n";
    return 1;
  }

  comp.encoded_latents = loaded_latents;
  comp.encoded_hyper_latents = loaded_hyper;

  if (verbose) {
    std::cout << "Loaded metadata:\n";
    std::cout << "  Number of samples: " << comp.num_samples << "\n";
    std::cout << "  Number of batches: " << comp.num_batches << "\n";
  }

  std::cout << "Metadata loaded successfully\n";

  auto start_time_d = std::chrono::high_resolution_clock::now();
  Decompressor decompressor(decompress_device);
  torch::Tensor recon = decompressor.decompress(batch_size, n_frame, comp);
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

  torch::Tensor restored = restore_from_5d(recon, padding_info);

  if (verbose) {
    std::cout << "Restored tensor shape: " << restored.sizes() << "\n";
  }

  save_tensor_to_bin(restored, output_file, verbose);

  if (verify && !original_file.empty()) {
    if (!quiet) std::cout << "\n Verifying reconstruction...\n";

    torch::Tensor original =
        load_raw_binary(original_file, original_shape, false);
    original = original.squeeze();

    torch::Tensor orig_cpu = original.to(torch::kCPU);
    torch::Tensor recon_cpu = restored.to(torch::kCPU);

    torch::Tensor diff = recon_cpu - orig_cpu;
    double mse = diff.pow(2).mean().item<double>();
    double rmse = std::sqrt(mse);
    double nrmse =
        rmse / (orig_cpu.max().item<double>() - orig_cpu.min().item<double>());
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
      for (auto d : original_shape) num_elements *= static_cast<uint64_t>(d);
      uint64_t uncompressed_bytes = num_elements * sizeof(float);

      save_metrics_to_csv(metrics_csv, original_file, original_shape, 0.0,
                          decompression_time.count(), uncompressed_bytes, 0, 0,
                          0.0, 0.0, nrmse, psnr, 0.0, batch_size, n_frame, "V",
                          "N/A", decompress_device.is_cuda() ? "cuda" : "cpu");
    }
  }

  if (!quiet) {
    std::cout << "\n Decompression complete!\n";
    std::cout << "Output: " << output_file << "\n";
  }

  return 0;
}

int main(int argc, char* argv[]) {
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
    int batch_size = 128;
    int n_frame = 8;
    std::string model_type = "V";
    std::string compress_device_str;
    std::string decompress_device_str;
    bool show_timing = false;
    bool show_metadata = false;
    bool verbose = false;
    bool quiet = false;
    bool verify = false;
    bool force_padding = false;
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
      } else if ((arg == "-b" || arg == "--batch-size") && i + 1 < argc) {
        batch_size = std::stoi(argv[++i]);
      } else if ((arg == "-f" || arg == "--n-frame") && i + 1 < argc) {
        n_frame = std::stoi(argv[++i]);
      } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
        model_type = argv[++i];
        std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                       ::toupper);
      } else if (arg == "--compress-device" && i + 1 < argc) {
        compress_device_str = argv[++i];
      } else if (arg == "--decompress-device" && i + 1 < argc) {
        decompress_device_str = argv[++i];
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
      } else if (arg == "--force-padding") {
        force_padding = true;
      } else if (arg == "--metrics-csv" && i + 1 < argc) {
        metrics_csv = argv[++i];
      } else if (arg == "--original" && i + 1 < argc) {
        original_file = argv[++i];
      }
    }

    if (command == "compress") {
      if (shape.empty()) {
        std::cerr
            << "Error: Shape is required for compression (-s or --shape)\n";
        return 1;
      }

      torch::Device compress_device = compress_device_str.empty()
                                          ? auto_select_device()
                                          : parse_device(compress_device_str);

      if (output_file.empty()) {
        output_file = input_file + ".cae";
      }

      return compress_file(input_file, output_file, shape, error_bound,
                           batch_size, n_frame, model_type, compress_device,
                           show_timing, show_metadata, verbose, quiet,
                           force_padding, metrics_csv);

    } else if (command == "decompress") {
      torch::Device decompress_device =
          decompress_device_str.empty() ? auto_select_device()
                                        : parse_device(decompress_device_str);

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

      return decompress_file(input_file, output_file, shape, batch_size,
                             n_frame, decompress_device, show_timing, verbose,
                             quiet, verify, original_file, metrics_csv);
    }

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
