#include "networkComponents.h"


std::vector<uint32_t> pmf_to_quantized_cdf(const std::vector<float> &pmf, int precision) {
    for (float p : pmf) {
        if (p < 0 || !std::isfinite(p)) {
            throw std::domain_error(
                std::string("Invalid `pmf`, non-finite or negative element found: ") +
                std::to_string(p)
            );
        }
    }

    std::vector<uint32_t> cdf(pmf.size() + 1);
    cdf[0] = 0;

    std::transform(pmf.begin(), pmf.end(), cdf.begin() + 1,
                   [=](float p) { return std::round(p * (1 << precision)); });

    const uint32_t total = std::accumulate(cdf.begin(), cdf.end(), 0u);
    if (total == 0) {
        throw std::domain_error("Invalid `pmf`: at least one element must have non-zero probability.");
    }

    std::transform(cdf.begin(), cdf.end(), cdf.begin(),
                   [precision, total](uint32_t p) {
                       return ((static_cast<uint64_t>(1 << precision) * p) / total);
                   });

    std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
    cdf.back() = 1 << precision;

    for (int i = 0; i < static_cast<int>(cdf.size()) - 1; ++i) {
        if (cdf[i] == cdf[i + 1]) {
            uint32_t best_freq = ~0u;
            int best_steal = -1;
            for (int j = 0; j < static_cast<int>(cdf.size()) - 1; ++j) {
                uint32_t freq = cdf[j + 1] - cdf[j];
                if (freq > 1 && freq < best_freq) {
                    best_freq = freq;
                    best_steal = j;
                }
            }
            assert(best_steal != -1);

            if (best_steal < i) {
                for (int j = best_steal + 1; j <= i; ++j) cdf[j]--;
            } else {
                assert(best_steal > i);
                for (int j = i + 1; j <= best_steal; ++j) cdf[j]++;
            }
        }
    }

    assert(cdf[0] == 0);
    assert(cdf.back() == (1 << precision));
    for (int i = 0; i < static_cast<int>(cdf.size()) - 1; ++i) {
        assert(cdf[i + 1] > cdf[i]);
    }

    return cdf;
}

torch::Tensor pmfToQuantizedCDFTensor(const torch::Tensor& pmf, int precision){
    
    std::vector<float> pmfVec(pmf.data_ptr<float>(), pmf.data_ptr<float>() + pmf.numel());
    std::vector<uint32_t> cdfVec = pmf_to_quantized_cdf(pmfVec, precision);

    torch::Tensor cdfTensor = torch::from_blob(cdfVec.data(), {(int)cdfVec.size()}, torch::kInt32).clone();
    return cdfTensor;
}










