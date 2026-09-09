#include "caesar_decompress.h"
#include <ATen/Parallel.h>
#include <algorithm>
#include <c10/core/thread_pool.h>
#include <future>

namespace {
// Join every task before returning or propagating an error, so output buffers
// remain alive until all workers have finished writing their disjoint rows.
template <class F>
void decode_parallel(c10::ThreadPool &pool, int64_t count, const F &decode) {
  const int64_t chunk = (count + pool.size() - 1) / pool.size();
  std::vector<std::future<void>> results;
  try {
    for (int64_t begin = 0; begin < count; begin += chunk) {
      auto task = std::make_shared<std::packaged_task<void()>>(
          [&, begin] { decode(begin, std::min(begin + chunk, count)); });
      results.push_back(task->get_future());
      pool.run([task] { (*task)(); });
    }
  } catch (...) {
    pool.waitWorkComplete();
    throw;
  }
  pool.waitWorkComplete();
  for (auto &result : results)
    result.get();
}
} // namespace

torch::Tensor deblockHW(const torch::Tensor &data, int64_t nH, int64_t nW,
                        const std::vector<int64_t> &padding);

std::tuple<torch::Tensor, std::vector<int>>
padding(const torch::Tensor &data, std::pair<int, int> block_size = {8, 8});
torch::Tensor recons_data(const torch::Tensor &recons_data,
                          const std::vector<int32_t> &shape, int64_t pad_T) {
  int64_t stop_t = shape[2] - pad_T;
  return recons_data.index({torch::indexing::Slice(), torch::indexing::Slice(),
                            torch::indexing::Slice(0, stop_t),
                            torch::indexing::Slice(),
                            torch::indexing::Slice()});
}
torch::Tensor unpadding(const torch::Tensor &padded_data,
                        const std::vector<int> &padding);

template <typename T>
std::vector<T> tensor_to_vector(const torch::Tensor &tensor) {
  torch::Tensor cpu_tensor = tensor.cpu().contiguous();
  const T *tensor_data_ptr = cpu_tensor.data_ptr<T>();
  int64_t num_elements = cpu_tensor.numel();
  return std::vector<T>(tensor_data_ptr, tensor_data_ptr + num_elements);
}

torch::Tensor build_indexes_tensor(const std::vector<int32_t> &size) {
  int64_t dims = size.size();
  TORCH_CHECK(dims >= 2,
              "Input size must have at least 2 dimensions (N, C, ...)");
  int64_t C = size[1];
  std::vector<int64_t> view_dims = {1, C};
  view_dims.insert(view_dims.end(), dims - 2, 1);
  torch::Tensor indexes = torch::arange(C).view(view_dims);
  std::vector<int64_t> size_int64(size.begin(), size.end());
  return indexes.expand(size_int64).to(torch::kInt32);
}

Decompressor::Decompressor(torch::Device device) : device_(device) {
  at::globalContext().setDeterministicAlgorithms(true, false);
  load_models();
  load_probability_tables();
}

void Decompressor::load_models() {
  hyper_decompressor_model_ =
      ModelCache::instance().get_hyper_decompressor_model();
  decompressor_model_ = ModelCache::instance().get_decompressor_model();
}

void Decompressor::load_probability_tables() {
  vbr_quantized_cdf_ = ModelCache::instance().get_vbr_quantized_cdf();
  vbr_cdf_length_ = ModelCache::instance().get_vbr_cdf_length();
  vbr_offset_ = ModelCache::instance().get_vbr_offset();
  gs_quantized_cdf_ = ModelCache::instance().get_gs_quantized_cdf();
  gs_cdf_length_ = ModelCache::instance().get_gs_cdf_length();
  gs_offset_ = ModelCache::instance().get_gs_offset();
}

torch::Tensor Decompressor::reshape_batch_2d_3d(const torch::Tensor &batch_data,
                                                int64_t batch_size,
                                                int64_t n_frame) {
  auto sizes = batch_data.sizes();
  TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4D.");
  int64_t BT = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
  int64_t T = BT / batch_size;
  torch::Tensor reshaped_data = batch_data.view({batch_size, T, C, H, W});
  torch::Tensor permuted_data = reshaped_data.permute({0, 2, 1, 3, 4});
  return permuted_data;
}

torch::Tensor Decompressor::decompress(const unsigned int batch_size,
                                       const unsigned int n_frame,
                                       const CompressionResult &comp_result) {
  c10::InferenceMode guard;

  DecompressionResult result;
  result.num_samples = 0;
  result.num_batches = 0;

  auto &meta = comp_result.compressionMetaData;

  torch::TensorOptions opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(device_);
  torch::Tensor offsets_tensor = torch::tensor(meta.offsets, opts);
  torch::Tensor scales_tensor = torch::tensor(meta.scales, opts);

  std::vector<int64_t> input_shape(meta.data_input_shape.begin(),
                                   meta.data_input_shape.end());
  torch::Tensor recon_tensor = torch::zeros(input_shape, opts);
  input_shape.clear();
  input_shape.shrink_to_fit();

  // this is for saving latents for portablity
  //   std::vector<torch::Tensor> row_tensors;
  //   row_tensors.reserve(comp_result.latent_indexes.size());
  //   for (const auto& row : comp_result.latent_indexes) {
  //     auto t = torch::from_blob((void*)row.data(), {(long)row.size()},
  //     torch::kUInt8).clone(); row_tensors.push_back(t.to(device_));
  //   }

  //  torch::Tensor latent_indexes_recon = torch::stack(row_tensors,
  //  0).to(torch::kInt32);
  //   row_tensors.clear();
  //   row_tensors.shrink_to_fit();

  // All hyper streams use the same channel indexes. Build them once.
  const std::vector<int32_t> hyper_index_vec = tensor_to_vector<int32_t>(
      build_indexes_tensor({1, 64, 4, 4}).contiguous().reshape(-1));
  const auto cpu_float_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  // Reuse a bounded pool across batches. This also works in builds without
  // OpenMP, where the header-only at::parallel_for would execute serially.
  const int workers = std::max(1, std::min(8, at::get_num_threads()));
  c10::ThreadPool decode_pool(workers);

  for (size_t lat_start = 0; lat_start < comp_result.encoded_latents.size();
       lat_start += (size_t)batch_size * 2) {
    size_t lat_end = std::min(lat_start + (size_t)batch_size * 2,
                              comp_result.encoded_latents.size());
    size_t cur_latents = lat_end - lat_start;
    TORCH_CHECK(cur_latents % 2 == 0, "cur_latents must be even.");
    size_t cur_samples = cur_latents / 2;
    size_t sample_start = lat_start / 2;

    // Workers touch only CPU data and disjoint output rows. Decode straight
    // into float staging storage, avoiding a temporary tensor per stream.
    torch::Tensor decoded_hyper_latents =
        torch::empty({(long)cur_latents, 64, 4, 4}, cpu_float_opts);
    float *hyper_data = decoded_hyper_latents.data_ptr<float>();
    decode_parallel(decode_pool, (int64_t)cur_latents,
                    [&](int64_t begin, int64_t end) {
                      RansDecoder decoder;
                      for (int64_t i = begin; i < end; ++i) {
                        auto decoded = decoder.decode_with_indexes(
                            comp_result.encoded_hyper_latents[lat_start + i],
                            hyper_index_vec, vbr_quantized_cdf_,
                            vbr_cdf_length_, vbr_offset_);
                        std::copy(decoded.begin(), decoded.end(),
                                  hyper_data + i * 64 * 4 * 4);
                      }
                    });

    std::vector<torch::Tensor> hyper_outputs =
        hyper_decompressor_model_->run({decoded_hyper_latents.to(device_)});
    torch::Tensor mean = hyper_outputs[0].to(torch::kFloat32);
    // The CPU entropy decoder needs every index in this batch. Transfer once,
    // rather than synchronizing a device-to-host copy for each latent sample.
    torch::Tensor latent_indexes_recon =
        hyper_outputs[1].to(torch::kInt32).to(torch::kCPU).contiguous();

    constexpr int64_t latent_elements = 64 * 16 * 16;
    TORCH_CHECK(latent_indexes_recon.numel() ==
                    (int64_t)cur_latents * latent_elements,
                "Unexpected latent index shape");
    torch::Tensor decoded_latents_before_offset =
        torch::empty({(long)cur_latents, 64, 16, 16}, cpu_float_opts);
    const int32_t *index_data = latent_indexes_recon.data_ptr<int32_t>();
    float *latent_data = decoded_latents_before_offset.data_ptr<float>();
    decode_parallel(
        decode_pool, (int64_t)cur_latents, [&](int64_t begin, int64_t end) {
          RansDecoder decoder;
          std::vector<int32_t> latent_index(latent_elements);
          for (int64_t i = begin; i < end; ++i) {
            std::copy_n(index_data + i * latent_elements, latent_elements,
                        latent_index.begin());
            auto decoded = decoder.decode_with_indexes(
                comp_result.encoded_latents[lat_start + i], latent_index,
                gs_quantized_cdf_, gs_cdf_length_, gs_offset_);
            std::copy(decoded.begin(), decoded.end(),
                      latent_data + i * latent_elements);
          }
        });

    torch::Tensor q_latent_with_offset =
        decoded_latents_before_offset.to(device_) + mean;

    auto decoded_latents_sizes = q_latent_with_offset.sizes();
    std::vector<int64_t> new_shape = {-1, 2};
    new_shape.insert(new_shape.end(), decoded_latents_sizes.begin() + 1,
                     decoded_latents_sizes.end());
    torch::Tensor reshaped_latents = q_latent_with_offset.reshape(new_shape);

    std::vector<torch::Tensor> decompressor_outputs =
        decompressor_model_->run({reshaped_latents.to(torch::kFloat32)});
    torch::Tensor raw_output = decompressor_outputs[0].to(torch::kFloat32);

    torch::Tensor norm_output =
        reshape_batch_2d_3d(raw_output, (long)cur_samples, n_frame);

    torch::Tensor batched_offsets =
        offsets_tensor.narrow(0, (long)sample_start, (long)cur_samples)
            .view({-1, 1, 1, 1, 1});
    torch::Tensor batched_scales =
        scales_tensor.narrow(0, (long)sample_start, (long)cur_samples)
            .view({-1, 1, 1, 1, 1});
    torch::Tensor denorm_output =
        norm_output * batched_scales + batched_offsets;

    // Use host placement metadata directly; the reconstructed data stays on
    // device.
    for (int64_t i = 0; i < (int64_t)cur_samples; ++i) {
      const auto &index_row = meta.indexes.at(sample_start + i);
      int64_t idx0 = index_row.at(0);
      int64_t idx1 = index_row.at(1);
      int64_t start_t = index_row.at(2);
      int64_t end_t = index_row.at(3);

      torch::Tensor source_slice_3d = denorm_output.select(0, i).squeeze(0);
      torch::Tensor dest_slice =
          recon_tensor.select(0, idx0).select(0, idx1).slice(0, start_t, end_t);
      dest_slice.copy_(source_slice_3d);
    }

    result.num_samples += cur_samples;
    result.num_batches++;
  }

  offsets_tensor = torch::Tensor();
  scales_tensor = torch::Tensor();
  if (!meta.filtered_blocks.empty()) {
    const int64_t S = static_cast<int64_t>(meta.data_input_shape[1]);
    const int64_t T = static_cast<int64_t>(meta.data_input_shape[2]);
    const int64_t nf = 8;
    const int64_t samples = T / nf;
    const int64_t SS = S * samples;

    for (const auto &pair : meta.filtered_blocks) {
      const int64_t label = pair.first;
      const float value = pair.second;
      const int64_t v = label / SS;
      const int64_t remain = label % SS;
      const int64_t s = remain / samples;
      const int64_t blk = remain % samples;
      recon_tensor.select(0, v)
          .select(0, s)
          .slice(0, blk * nf, (blk + 1) * nf)
          .fill_(value);
    }
  }

  auto [b1_i32, b2_i32, pad_i32] = meta.block_info;

  int64_t block_info_1 = b1_i32;
  int64_t block_info_2 = b2_i32;
  std::vector<int64_t> block_info_3(pad_i32.begin(), pad_i32.end());

  torch::Tensor recon_tensor_deblock =
      deblockHW(recon_tensor, block_info_1, block_info_2, block_info_3);
  recon_tensor = torch::Tensor();

  //  ---- LBRC path --------------------------------
  //  ---------------------------------------------------------
  if (comp_result.use_lbrc) {
    torch::Tensor recon_ =
        recon_tensor_deblock.to(device_).to(torch::kFloat32).contiguous();
    recon_tensor_deblock = torch::Tensor();

    torch::Tensor corrected = caesar::lbrc::decompress(
        recon_, comp_result.lbrcMetaData, comp_result.lbrc_blocks,
        get_allocated_cores());

    torch::Tensor final_recon =
        recons_data(corrected, comp_result.compressionMetaData.data_input_shape,
                    comp_result.compressionMetaData.pad_T);

    return final_recon;
  }

  // GAE path  ----------------------------------------
  if (comp_result.gaeMetaData.GAE_correction_occur) {
    std::tuple<torch::Tensor, std::vector<int>> padding_recon =
        padding(recon_tensor_deblock);
    recon_tensor_deblock = torch::Tensor();
    torch::Tensor padded_recon_tensor = std::get<0>(padding_recon);
    std::vector<int> padding_recon_info = std::get<1>(padding_recon);
    padding_recon = {};

    float global_scale = meta.global_scale;
    float global_offset = meta.global_offset;
    torch::Tensor padded_recon_tensor_norm =
        (padded_recon_tensor - global_offset) / global_scale;
    padded_recon_tensor = torch::Tensor();
    double quan_factor = 2.0;

    std::string codec_alg = "Zstd";
    std::pair<int, int> patch_size = {8, 8};
    double rel_eb = 1e-3;

    std::string pca_device_str = device_.is_cuda()  ? "cuda"
                                 : device_.is_mps() ? "mps"
                                                    : "cpu";

    PCACompressor pca_compressor(rel_eb, quan_factor, pca_device_str, codec_alg,
                                 patch_size);

    MetaData gae_record_metaData;
    CompressedData gae_record_compressedData;

    int64_t pca_rows = comp_result.gaeMetaData.pcaBasis.size();
    int64_t pca_cols = comp_result.gaeMetaData.pcaBasis[0].size();

    std::vector<float> pca_vec;
    pca_vec.reserve(pca_rows * pca_cols);

    for (const auto &row_vec : comp_result.gaeMetaData.pcaBasis) {
      pca_vec.insert(pca_vec.end(), row_vec.begin(), row_vec.end());
    }
    torch::Tensor pca_vec_1d = torch::tensor(pca_vec);
    torch::Tensor pcaBasis = pca_vec_1d.reshape({pca_rows, pca_cols});

    gae_record_metaData.pcaBasis = pcaBasis.to(device_);
    gae_record_metaData.uniqueVals =
        torch::tensor(comp_result.gaeMetaData.uniqueVals).to(device_);
    gae_record_metaData.quanBin = comp_result.gaeMetaData.quanBin;
    gae_record_metaData.nVec = comp_result.gaeMetaData.nVec;
    gae_record_metaData.prefixLength = comp_result.gaeMetaData.prefixLength;
    gae_record_metaData.dataBytes = comp_result.gaeMetaData.dataBytes;

    gae_record_compressedData.data = comp_result.gae_comp_data;
    gae_record_compressedData.dataBytes = comp_result.gaeMetaData.dataBytes;
    gae_record_compressedData.coeffIntBytes =
        comp_result.gaeMetaData.coeffIntBytes;

    torch::Tensor recons_gae =
        pca_compressor.decompress(padded_recon_tensor_norm, gae_record_metaData,
                                  gae_record_compressedData);
    padded_recon_tensor_norm = torch::Tensor();
    padded_recon_tensor = torch::Tensor();

    torch::Tensor recons_gae_unpadded =
        unpadding(recons_gae, comp_result.gaeMetaData.padding_recon_info);
    padding_recon_info.clear();
    padding_recon_info.shrink_to_fit();
    torch::Tensor final_recon_norm =
        recons_data(recons_gae_unpadded, meta.data_input_shape, meta.pad_T);
    torch::Tensor final_recon =
        final_recon_norm * meta.global_scale + meta.global_offset;

    return final_recon;
  } else {
    torch::Tensor final_recon =
        recons_data(recon_tensor_deblock, meta.data_input_shape, meta.pad_T);

    return final_recon;
  }
}
