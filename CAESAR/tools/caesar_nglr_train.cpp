#include <torch/torch.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "data_utils.h"
#include "models/caesar_compress.h"
#include "models/caesar_decompress.h"
#include "nglr_train.h"

CompressionResult load_complete_metadata(
    const std::string& filename,
    PaddingInfo& padding_info
);

void save_complete_metadata(
    const std::string& filename,
    const PaddingInfo& padding_info,
    const CompressionResult& comp
);

std::vector<std::string> load_encoded_streams(const std::string& filename);

torch::Tensor load_raw_binary(
    const std::string& bin_path,
    const std::vector<int64_t>& shape,
    bool verbose = false
);

std::vector<int64_t> parse_shape(const std::string& shape_str);
torch::Device parse_device(const std::string& device_str);
torch::Device auto_select_device();

namespace {

void print_usage(const char* program_name) {
  std::cout
      << "CAESAR NGLR Training Tool\n\n"
      << "Usage:\n"
      << "  " << program_name << " <base.cae> --original <file> --shape <shape> [options]\n\n"
      << "Options:\n"
      << "  -e, --target <val>       Target NRMSE/error bound (default: 0.001)\n"
      << "  -b, --batch-size <n>     CAESAR decompression batch size (default: 128)\n"
      << "  -f, --n-frame <n>        Number of frames (default: 8)\n"
      << "  --device <dev>           Training/encoding device (cpu/cuda/mps)\n"
      << "  -v, --verbose            Verbose output\n"
      << "  -h, --help               Show this help message\n\n"
      << "The tool expects <base.cae>.latents, <base.cae>.hyper, and <base.cae>.meta\n"
      << "from a base CAESAR compression run, preferably --correction none.\n";
}

std::string sizes_to_string(const c10::IntArrayRef& sizes) {
  std::ostringstream out;
  out << sizes;
  return out.str();
}

torch::Tensor load_original_5d(
    const std::string& original_file,
    const std::vector<int64_t>& shape,
    bool verbose
) {
  torch::Tensor raw = load_raw_binary(original_file, shape, verbose);
  raw = raw.squeeze();

  PaddingInfo ignored;
  torch::Tensor raw_5d;
  if (shape.size() >= 5 && shape[3] >= 128 && shape[4] >= 128) {
    std::tie(raw_5d, ignored) = to_5d_and_pad(raw, shape[3], shape[4], false);
  } else if (shape.size() == 4 || shape.size() == 3) {
    std::tie(raw_5d, ignored) = to_5d_and_pad(raw, 128, 128, false);
  } else {
    std::tie(raw_5d, ignored) = to_5d_and_pad(raw, 256, 256, false);
  }

  return raw_5d.to(torch::kCPU).to(torch::kFloat32).contiguous();
}

CompressionResult load_base_result(const std::string& base, PaddingInfo& padding_info) {
  CompressionResult comp = load_complete_metadata(base + ".meta", padding_info);

  comp.encoded_latents = load_encoded_streams(base + ".latents");
  comp.encoded_hyper_latents = load_encoded_streams(base + ".hyper");

  if (comp.encoded_latents.empty() || comp.encoded_hyper_latents.empty()) {
    throw std::runtime_error("Failed to load base CAESAR latent/hyper streams");
  }

  return comp;
}

void disable_existing_corrections(CompressionResult& comp) {
  comp.use_lbrc = false;
  comp.lbrc_blocks.clear();

  comp.gaeMetaData.GAE_correction_occur = false;
  comp.gae_comp_data.clear();

  comp.use_nglr = false;
  comp.nglrMetaData = caesar::nglr::NGLRMetaData();
  comp.nglrCompressedData.blocks.clear();
}

} // namespace

int main(int argc, char* argv[]) {
  try {
    if (argc < 2) {
      print_usage(argv[0]);
      return 1;
    }

    std::string base = argv[1];
    if (base == "-h" || base == "--help") {
      print_usage(argv[0]);
      return 0;
    }

    std::string original_file;
    std::vector<int64_t> shape;
    double target = 1e-3;
    int batch_size = 128;
    int n_frame = 8;
    std::string device_arg;
    bool verbose = false;

    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];

      if (arg == "--original" && i + 1 < argc) {
        original_file = argv[++i];
      } else if ((arg == "-s" || arg == "--shape") && i + 1 < argc) {
        shape = parse_shape(argv[++i]);
      } else if ((arg == "-e" || arg == "--target") && i + 1 < argc) {
        target = std::stod(argv[++i]);
      } else if ((arg == "-b" || arg == "--batch-size") && i + 1 < argc) {
        batch_size = std::stoi(argv[++i]);
      } else if ((arg == "-f" || arg == "--n-frame") && i + 1 < argc) {
        n_frame = std::stoi(argv[++i]);
      } else if (arg == "--device" && i + 1 < argc) {
        device_arg = argv[++i];
      } else if (arg == "-v" || arg == "--verbose") {
        verbose = true;
      } else if (arg == "-h" || arg == "--help") {
        print_usage(argv[0]);
        return 0;
      } else {
        throw std::runtime_error("Unknown argument: " + arg);
      }
    }

    if (original_file.empty()) {
      throw std::runtime_error("--original is required");
    }
    if (shape.empty()) {
      throw std::runtime_error("--shape is required");
    }

    torch::Device device =
        device_arg.empty() ? auto_select_device() : parse_device(device_arg);

    PaddingInfo padding_info;
    CompressionResult comp = load_base_result(base, padding_info);

    disable_existing_corrections(comp);

    auto start = std::chrono::high_resolution_clock::now();

    Decompressor decompressor(device);
    torch::Tensor reconstruction =
        decompressor.decompress(
            static_cast<unsigned int>(batch_size),
            static_cast<unsigned int>(n_frame),
            comp
        ).to(torch::kCPU).to(torch::kFloat32).contiguous();

    torch::Tensor original = load_original_5d(original_file, shape, verbose);

    if (original.sizes() != reconstruction.sizes()) {
      throw std::runtime_error(
          "Original/base reconstruction shape mismatch: original=" +
          sizes_to_string(original.sizes()) +
          " reconstruction=" +
          sizes_to_string(reconstruction.sizes())
      );
    }

    caesar::nglr::NGLRTrainConfig train_config =
        caesar::nglr::default_train_config(target);

    caesar::nglr::NGLRTrainResult trained =
        caesar::nglr::train_nglr_model(
            original,
            reconstruction,
            train_config,
            device
        );

    if (!trained.correction_required) {
      comp.use_nglr = false;
      save_complete_metadata(base + ".meta", padding_info, comp);

      std::cout << "NGLR skipped: base reconstruction already satisfies target "
                << std::scientific << std::setprecision(6)
                << target << " (base NRMSE " << trained.base_nrmse << ")\n";
      return 0;
    }

    caesar::nglr::NGLRResult encoded =
        caesar::nglr::encode_correction(
            original,
            reconstruction,
            trained.model,
            trained.meta,
            device
        );

    if (encoded.compressed.blocks.empty()) {
      comp.use_nglr = false;
    } else {
      comp.use_nglr = true;
      comp.nglrMetaData = std::move(encoded.meta);
      comp.nglrCompressedData = std::move(encoded.compressed);
    }

    save_complete_metadata(base + ".meta", padding_info, comp);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "NGLR correction attached to " << base << ".meta\n";
    std::cout << "Target NRMSE: " << std::scientific << std::setprecision(6)
              << target << "\n";
    std::cout << "Base NRMSE:   " << trained.base_nrmse << "\n";
    std::cout << "Quant NRMSE:  " << trained.quant_nrmse << "\n";
    std::cout << "Time:         " << std::fixed << std::setprecision(3)
              << elapsed.count() << " s\n";

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
