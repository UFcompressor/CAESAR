#include "caesar_compress.h"


template<typename T>
std::vector<std::vector<T>> tensor_to_2d_vector(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.dim() == 2 , "Input tensor must be 2-dimensional.");

    torch::Tensor cpu_tensor = tensor.is_cpu() ? tensor.contiguous() : tensor.cpu().contiguous();
    const int64_t rows = cpu_tensor.size(0);
    const int64_t cols = cpu_tensor.size(1);
    const T* data_ptr = cpu_tensor.data_ptr<T>();

    std::vector<std::vector<T>> vec_2d;
    vec_2d.reserve(rows);

    for (int64_t r = 0; r < rows; ++r) {
        const T* row_start_ptr = data_ptr + (r * cols);
        std::vector<T> inner_vec(row_start_ptr , row_start_ptr + cols);
        vec_2d.push_back(inner_vec);
    }
    return vec_2d;
}

torch::Tensor Compressor::recons_data(const torch::Tensor& recons_data , std::vector<int32_t> shape , int64_t pad_T) const {
    int64_t stop_t = shape[2] - pad_T;
    return recons_data.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(0, stop_t),
        torch::indexing::Slice(),
        torch::indexing::Slice()
        });
}

torch::Tensor Compressor::reshape_batch_2d_3d(const torch::Tensor& batch_data , int64_t batch_size) {
    auto sizes = batch_data.sizes();
    TORCH_CHECK(sizes.size() == 4 , "Input tensor must be 4-dimensional.");

    int64_t BT = sizes[0];
    int64_t C  = sizes[1];
    int64_t H  = sizes[2];
    int64_t W  = sizes[3];

    int64_t T = BT / batch_size;
    torch::Tensor reshaped_data = batch_data.view({ batch_size, T, C, H, W });
    torch::Tensor permuted_data = reshaped_data.permute({ 0, 2, 1, 3, 4 });

    return permuted_data;
}

torch::Tensor Compressor::deblockHW(const torch::Tensor& data ,
    int64_t nH ,
    int64_t nW ,
    const std::vector<int64_t>& padding) {
    if (padding.size() != 4)
        throw std::invalid_argument("padding must have 4 values: top, down, left, right");

    auto sizes = data.sizes();
    if (sizes.size() != 5)
        throw std::invalid_argument("Expected 5D input tensor (V, S_blk, T, h_block, w_block)");

    int64_t V      = sizes[0];
    int64_t sBlk   = sizes[1];
    int64_t T      = sizes[2];
    int64_t hBlock = sizes[3];
    int64_t wBlock = sizes[4];
    int64_t top    = padding[0];
    int64_t down   = padding[1];
    int64_t left   = padding[2];
    int64_t right  = padding[3];

    if (sBlk % (nH * nW) != 0)
        throw std::invalid_argument("sBlk must be divisible by nH * nW");
    int64_t sOrig = sBlk / (nH * nW);

    auto reshaped = data.reshape({ V, sOrig, nH, nW, T, hBlock, wBlock });
    auto merged   = reshaped.permute({ 0, 1, 4, 2, 5, 3, 6 })
                            .reshape({ V, sOrig, T, nH * hBlock, nW * wBlock });

    int64_t H = nH * hBlock - top - down;
    int64_t W = nW * wBlock - left - right;

    return merged.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(top, top + H),
        torch::indexing::Slice(left, left + W)
    });
}

std::tuple<torch::Tensor , std::vector<int>> padding(
    const torch::Tensor& data ,
    std::pair<int , int> block_size = { 8, 8 })
{
    int h_block = block_size.first;
    int w_block = block_size.second;

    auto sizes = data.sizes();
    int ndim = sizes.size();
    int H = sizes[ndim - 2];
    int W = sizes[ndim - 1];

    int H_target = std::ceil((float)H / h_block) * h_block;
    int W_target = std::ceil((float)W / w_block) * w_block;
    int dh = H_target - H;
    int dw = W_target - W;

    int top   = dh / 2;
    int down  = dh - top;
    int left  = dw / 2;
    int right = dw - left;
    
    if (dh == 0 && dw == 0) {
        return { data, { 0, 0, 0, 0 } };
    }

    auto leading_dims = data.sizes().vec();
    int leading_size = 1;
    for (size_t i = 0; i < leading_dims.size() - 2; ++i)
        leading_size *= leading_dims[i];

    auto data_padded = torch::nn::functional::pad(
        data.view({ leading_size, H, W }) ,
        torch::nn::functional::PadFuncOptions({ left, right, top, down })
            .mode(torch::kConstant).value(0));

    auto new_shape = leading_dims;
    new_shape[new_shape.size() - 2] = data_padded.size(-2);
    new_shape[new_shape.size() - 1] = data_padded.size(-1);

    return { data_padded.view(new_shape), { top, down, left, right } };
}

torch::Tensor unpadding(const torch::Tensor& padded_data , const std::vector<int>& padding)
{
    int top  = padding[0];
    int down = padding[1];
    int left = padding[2];
    int right = padding[3];

    auto sizes = padded_data.sizes();
    int ndim = sizes.size();
    int H = sizes[ndim - 2];
    int W = sizes[ndim - 1];

    return padded_data.index({
        torch::indexing::Ellipsis,
        torch::indexing::Slice(top, H - down),
        torch::indexing::Slice(left, W - right)
    });
}

template<typename T>
std::vector<T> tensor_to_vector(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.is_cpu() ? tensor.contiguous() : tensor.cpu().contiguous();
    const T* ptr = cpu_tensor.data_ptr<T>();
    return std::vector<T>(ptr , ptr + cpu_tensor.numel());
}

Compressor::Compressor(torch::Device device) : device_(device) {
    load_models();
    load_probability_tables();
}

void Compressor::load_models() {
    compressor_model_          = ModelCache::instance().get_compressor_model();
    hyper_decompressor_model_  = ModelCache::instance().get_hyper_decompressor_model();
    decompressor_model_        = ModelCache::instance().get_decompressor_model();
}

void Compressor::load_probability_tables() {
    vbr_quantized_cdf_ = ModelCache::instance().get_vbr_quantized_cdf();
    vbr_cdf_length_    = ModelCache::instance().get_vbr_cdf_length();
    vbr_offset_        = ModelCache::instance().get_vbr_offset();
    gs_quantized_cdf_  = ModelCache::instance().get_gs_quantized_cdf();
    gs_cdf_length_     = ModelCache::instance().get_gs_cdf_length();
    gs_offset_         = ModelCache::instance().get_gs_offset();
}

CompressionResult Compressor::compress(const DatasetConfig& config,
                                        int batch_size, float rel_eb) {
  c10::InferenceMode guard;

  ScientificDataset dataset(config);

  CompressionResult result;

  int64_t pad_T = dataset.get_pad_T();
  result.compressionMetaData.pad_T = pad_T;

  {
    const auto& data_input_shape = dataset.get_data_input().sizes();
    std::vector<int32_t> data_input_shape_i32;
    data_input_shape_i32.reserve(data_input_shape.size());
    for (int64_t dim : data_input_shape)
      data_input_shape_i32.push_back(static_cast<int32_t>(dim));
    result.compressionMetaData.data_input_shape = data_input_shape_i32;
  }

  {
    const auto& filtered_blocks = dataset.get_filtered_blocks();
    result.compressionMetaData.filtered_blocks.reserve(filtered_blocks.size());
    for (const auto& pair : filtered_blocks)
      result.compressionMetaData.filtered_blocks.emplace_back(
          static_cast<int32_t>(pair.first), pair.second);
  }

  {
    auto block_info = dataset.get_block_info();
    int32_t nH_i32 = static_cast<int32_t>(std::get<0>(block_info));
    int32_t nW_i32 = static_cast<int32_t>(std::get<1>(block_info));
    const std::vector<int64_t>& padding_i64 = std::get<2>(block_info);
    std::vector<int32_t> padding_i32;
    padding_i32.reserve(padding_i64.size());
    for (int64_t v : padding_i64)
      padding_i32.push_back(static_cast<int32_t>(v));
    result.compressionMetaData.block_info =
        std::make_tuple(nH_i32, nW_i32, padding_i32);
  }

  std::vector<torch::Tensor> batch_inputs;
  batch_inputs.reserve(batch_size);
  std::vector<float> batch_offsets_vec;
  batch_offsets_vec.reserve(batch_size);
  std::vector<float> batch_scales_vec;
  batch_scales_vec.reserve(batch_size);
  std::vector<torch::Tensor> batch_indexes;
  batch_indexes.reserve(batch_size);

  result.compressionMetaData.offsets.reserve(dataset.size());
  result.compressionMetaData.scales.reserve(dataset.size());
  result.compressionMetaData.indexes.reserve(dataset.size());

  const std::vector<int32_t>& shape_i32 =
      result.compressionMetaData.data_input_shape;
  std::vector<int64_t> input_shape(shape_i32.begin(), shape_i32.end());
  torch::Tensor recon_tensor =
      torch::zeros(input_shape, torch::TensorOptions().device(device_));

  std::vector<torch::Tensor> all_q_latent;
  std::vector<torch::Tensor> all_latent_indexes;
  std::vector<torch::Tensor> all_q_hyper_latent;
  std::vector<torch::Tensor> all_hyper_indexes;
  int64_t total_latent_codes = 0;


  for (size_t i = 0; i < dataset.size(); i++) {
    auto sample = dataset.get_item(i);

    torch::Tensor input_tensor  = sample["input"].to(device_);
    torch::Tensor offset_tensor = sample["offset"];
    torch::Tensor scale_tensor  = sample["scale"];
    torch::Tensor index_tensor  = sample["index"];

    batch_inputs.push_back(input_tensor);
    batch_offsets_vec.push_back(offset_tensor.item<float>());
    batch_scales_vec.push_back(scale_tensor.item<float>());
    batch_indexes.push_back(index_tensor.view({1, index_tensor.sizes()[0]}));
    result.compressionMetaData.offsets.push_back(offset_tensor.item<float>());
    result.compressionMetaData.scales.push_back(scale_tensor.item<float>());

    {
      std::vector<int32_t> index_vec;
      index_vec.reserve(index_tensor.numel());
      const int64_t* index_data_ptr = index_tensor.data_ptr<int64_t>();
      for (int j = 0; j < index_tensor.numel(); ++j)
        index_vec.push_back(static_cast<int32_t>(index_data_ptr[j]));
      result.compressionMetaData.indexes.push_back(std::move(index_vec));
    }

    if (batch_inputs.size() == static_cast<size_t>(batch_size) ||
        i == dataset.size() - 1) {

      int64_t num_input_samples = static_cast<int64_t>(batch_inputs.size());

      torch::Tensor batched_input = torch::cat(batch_inputs, 0);
      torch::Tensor batched_offsets =
          torch::tensor(batch_offsets_vec,
                        torch::TensorOptions().device(device_))
              .view({-1, 1, 1, 1, 1});
      torch::Tensor batched_scales =
          torch::tensor(batch_scales_vec,
                        torch::TensorOptions().device(device_))
              .view({-1, 1, 1, 1, 1});
      torch::Tensor batched_indexes = torch::cat(batch_indexes, 0).to(device_);
      
      std::vector<torch::Tensor> outputs = compressor_model_->run({batched_input.to(torch::kFloat32)}); 
      torch::Tensor q_latent       = outputs[0];
      torch::Tensor latent_indexes = outputs[1];
      torch::Tensor q_hyper_latent = outputs[2];
      torch::Tensor hyper_indexes  = outputs[3];
      outputs.clear();


      std::vector<torch::Tensor> hyper_outputs =
          hyper_decompressor_model_->run({q_hyper_latent.to(torch::kFloat32)});
      torch::Tensor mean = hyper_outputs[0].to(torch::kFloat32);
      hyper_outputs.clear();

      all_q_latent.push_back(q_latent);
      all_latent_indexes.push_back(latent_indexes);
      all_q_hyper_latent.push_back(q_hyper_latent);
      all_hyper_indexes.push_back(hyper_indexes);
      total_latent_codes += q_latent.sizes()[0];


      torch::Tensor q_latent_with_offset =
          q_latent.to(torch::kFloat32) + mean;

      auto decoded_sizes = q_latent_with_offset.sizes();
      std::vector<int64_t> new_shape = {-1, 2};
      new_shape.insert(new_shape.end(),
                       decoded_sizes.begin() + 1,
                       decoded_sizes.end());

      std::vector<torch::Tensor> decompressor_outputs =
          decompressor_model_->run(
              {q_latent_with_offset.reshape(new_shape).to(torch::kFloat32)});
      torch::Tensor raw_output = decompressor_outputs[0].to(torch::kFloat32);
      decompressor_outputs.clear();
      torch::Tensor norm_output   = reshape_batch_2d_3d(raw_output, num_input_samples);
      torch::Tensor denorm_output = norm_output * batched_scales + batched_offsets;

      torch::Tensor indexes_cpu  = batched_indexes.cpu();
      auto index_accessor = indexes_cpu.accessor<int64_t, 2>();
      for (int64_t ii = 0; ii < num_input_samples; ++ii) {
        recon_tensor.select(0, index_accessor[ii][0])
                    .select(0, index_accessor[ii][1])
                    .slice(0, index_accessor[ii][2], index_accessor[ii][3])
                    .copy_(denorm_output.select(0, ii).squeeze(0));
      }

      batch_inputs.clear();
      batch_offsets_vec.clear();
      batch_scales_vec.clear();
      batch_indexes.clear();
    }
  }


  torch::Tensor cat_q_latent       = torch::cat(all_q_latent, 0);
  torch::Tensor cat_latent_indexes = torch::cat(all_latent_indexes, 0);
  torch::Tensor cat_q_hyper        = torch::cat(all_q_hyper_latent, 0);
  torch::Tensor cat_hyper_indexes  = torch::cat(all_hyper_indexes, 0);
  all_q_latent.clear();
  all_latent_indexes.clear();
  all_q_hyper_latent.clear();
  all_hyper_indexes.clear();

  torch::Tensor cpu_q_latent       = cat_q_latent.to(torch::kCPU, true);
  torch::Tensor cpu_latent_indexes = cat_latent_indexes.to(torch::kCPU,  true);
  torch::Tensor cpu_q_hyper        = cat_q_hyper.to(torch::kCPU,  true);
  torch::Tensor cpu_hyper_indexes  = cat_hyper_indexes.to(torch::kCPU, true);

#ifdef USE_CUDA
  torch::cuda::synchronize();
#endif

  cat_q_latent       = torch::Tensor();
  cat_latent_indexes = torch::Tensor();
  cat_q_hyper        = torch::Tensor();
  cat_hyper_indexes  = torch::Tensor();

  result.encoded_latents.resize(total_latent_codes);
  result.encoded_hyper_latents.resize(total_latent_codes);

  const int workers = get_allocated_cores();
  std::vector<std::thread> threads;
  threads.reserve(workers);
  const int64_t chunk = (total_latent_codes + workers - 1) / workers;

  for (int w = 0; w < workers; ++w) {
    int64_t start = w * chunk;
    int64_t end   = std::min(start + chunk, total_latent_codes);
    if (start >= end) break;
    threads.emplace_back([&, start, end]() {
      RansEncoder enc;
      for (int64_t j = start; j < end; ++j) {
        auto latent_syms = tensor_to_vector<int32_t>(cpu_q_latent.select(0, j).reshape(-1));
        auto latent_idxs = tensor_to_vector<int32_t>(cpu_latent_indexes.select(0, j).reshape(-1));
        auto hyper_syms  = tensor_to_vector<int32_t>(cpu_q_hyper.select(0, j).reshape(-1));
        auto hyper_idxs  = tensor_to_vector<int32_t>(cpu_hyper_indexes.select(0, j).reshape(-1));
        result.encoded_latents[j] = enc.encode_with_indexes(
            latent_syms, latent_idxs, gs_quantized_cdf_, gs_cdf_length_, gs_offset_);
        result.encoded_hyper_latents[j] = enc.encode_with_indexes(
            hyper_syms, hyper_idxs, vbr_quantized_cdf_, vbr_cdf_length_, vbr_offset_);
      }
    });
  }
  for (auto& t : threads) t.join();
//   int64_t total = cpu_latent_indexes.size(0);
// result.latent_indexes.resize(total);
// for (int64_t j = 0; j < total; ++j) {
//     result.latent_indexes[j] = tensor_to_vector<int32_t>(
//         cpu_latent_indexes.select(0, j).reshape(-1));
// }


  if (!result.compressionMetaData.filtered_blocks.empty()) {
    const int64_t V =
        static_cast<int64_t>(result.compressionMetaData.data_input_shape[0]);
    const int64_t S =
        static_cast<int64_t>(result.compressionMetaData.data_input_shape[1]);
    const int64_t T =
        static_cast<int64_t>(result.compressionMetaData.data_input_shape[2]);
    const int64_t nf      = 8;
    const int64_t samples = T / nf;

    if (samples == 0 || S == 0) {
      std::cerr << "Error: 'samples' or 'S' is zero, cannot calculate indexes.\n";
    } else {
      const int64_t SS = S * samples;
      for (const auto& pair : result.compressionMetaData.filtered_blocks) {
        const int32_t label = pair.first;
        const float   value = pair.second;
        const int64_t v      = label / SS;
        const int64_t remain = label % SS;
        const int64_t s      = remain / samples;
        const int64_t blk    = remain % samples;
        recon_tensor.select(0, v)
                    .select(0, s)
                    .slice(0, blk * nf, (blk + 1) * nf)
                    .fill_(value);
      }
    }
  }

  int64_t block_info_1 =
      static_cast<int64_t>(std::get<0>(result.compressionMetaData.block_info));
  int64_t block_info_2 =
      static_cast<int64_t>(std::get<1>(result.compressionMetaData.block_info));
  std::vector<int64_t> block_info_3;
  for (int32_t v : std::get<2>(result.compressionMetaData.block_info))
    block_info_3.push_back(static_cast<int64_t>(v));

  torch::Tensor recon_deblk =
      deblockHW(recon_tensor, block_info_1, block_info_2, block_info_3);

  recon_tensor = torch::Tensor();
#ifdef USE_CUDA
  c10::cuda::CUDACachingAllocator::emptyCache();
#endif

    // ---- LBRC path hard coded for now !!!!!!!!!!!!!!!!!!!!  ---------------------------------------------------------
    result.use_lbrc = true;  // hard-coded later add logic for the rel eb to make this make more else and change the decine
    // hard code for the cpu

    if (result.use_lbrc) {
        torch::Tensor original_cpu =
            dataset.original_data().to(torch::kCPU).to(torch::kFloat32).contiguous();
        torch::Tensor recon_cpu =
            recon_deblk.to(torch::kCPU).to(torch::kFloat32).contiguous();
        recon_deblk = torch::Tensor();
        dataset.clear();

        result.lbrcMetaData.quant_iter = 16;

        caesar::lbrc::compress(
            original_cpu, recon_cpu,
            static_cast<double>(rel_eb),
            result.lbrcMetaData,
            result.lbrc_blocks,
            get_allocated_cores());

        return result;
    }
    // ---- GAE path  ----------------------------------------
  torch::Tensor original_full = dataset.original_data();

  // crop reflected tail frames (pad_T) before computing global stats
  int64_t T_full = original_full.size(2);
  torch::Tensor original_for_stats = (pad_T > 0)
      ? original_full.index({torch::indexing::Slice(), torch::indexing::Slice(),
                              torch::indexing::Slice(0, T_full - pad_T)})
      : original_full;

  torch::Tensor stats =
      torch::stack({original_for_stats.max(), original_for_stats.min(),
                    original_for_stats.mean()});
  float global_scale  = stats[0].item<float>() - stats[1].item<float>();
  float global_offset = stats[2].item<float>();

  auto [padded_recon_tensor, padding_recon_info] = padding(recon_deblk);
  recon_deblk = torch::Tensor();

  auto [padded_original_tensor, dummy_pad] = padding(original_full);
  dataset.clear();

  result.gaeMetaData.padding_recon_info = padding_recon_info;

  result.compressionMetaData.global_scale  = global_scale;
  result.compressionMetaData.global_offset = global_offset;

  padded_original_tensor.sub_(global_offset).div_(global_scale);
  torch::Tensor& padded_original_tensor_norm = padded_original_tensor;

  padded_recon_tensor.sub_(global_offset).div_(global_scale);
  torch::Tensor& padded_recon_tensor_norm = padded_recon_tensor;


  double quan_factor      = 2.0;
  std::string codec_alg   = "Zstd";
  std::pair<int, int> patch_size = {8, 8};

  PCACompressor pca_compressor(rel_eb, quan_factor,
                               device_.is_cuda() ? "cuda" : "cpu",
                               codec_alg, patch_size);

  auto gae_compression_result = pca_compressor.compress(
      padded_original_tensor_norm, padded_recon_tensor_norm);
  padded_original_tensor_norm = torch::Tensor();

    result.gaeMetaData.GAE_correction_occur =
      gae_compression_result.metaData.GAE_correction_occur;

  result.gaeMetaData.quanBin       = gae_compression_result.metaData.quanBin;
  result.gaeMetaData.nVec          = gae_compression_result.metaData.nVec;
  result.gaeMetaData.prefixLength  = gae_compression_result.metaData.prefixLength;
  result.gaeMetaData.dataBytes     = gae_compression_result.metaData.dataBytes;
  result.gaeMetaData.coeffIntBytes = gae_compression_result.compressedData->coeffIntBytes;
  result.gae_comp_data             = gae_compression_result.compressedData->data;

  if (result.gaeMetaData.GAE_correction_occur) {
        result.gaeMetaData.pcaBasis =
        tensor_to_2d_vector<float>(gae_compression_result.metaData.pcaBasis);
    result.gaeMetaData.uniqueVals =
        tensor_to_vector<float>(gae_compression_result.metaData.uniqueVals);

    MetaData gae_record_metaData;
    gae_record_metaData.pcaBasis     = gae_compression_result.metaData.pcaBasis.to(device_);
    gae_record_metaData.uniqueVals   = gae_compression_result.metaData.uniqueVals.to(device_);
    gae_record_metaData.quanBin      = result.gaeMetaData.quanBin;
    gae_record_metaData.nVec         = result.gaeMetaData.nVec;
    gae_record_metaData.prefixLength = result.gaeMetaData.prefixLength;
    gae_record_metaData.dataBytes    = result.gaeMetaData.dataBytes;

    CompressedData gae_record_compressedData;
    gae_record_compressedData.data          = result.gae_comp_data;
    gae_record_compressedData.dataBytes     = result.gaeMetaData.dataBytes;
    gae_record_compressedData.coeffIntBytes = result.gaeMetaData.coeffIntBytes;

  } 

  return result;
}
