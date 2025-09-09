#include "runGaeCuda.h"


PCA::PCA(int numComponents, const std::string& device)
    : numComponents_(numComponents), device_(torch::Device(device)) {}

PCA& PCA::fit(const torch::Tensor& x) {
    auto xDevice = x.to(device_);
    mean_ = torch::mean(xDevice, 0);
    auto xCentered = xDevice - mean_;

    auto C = torch::matmul(xCentered.t(), xCentered) / (xCentered.size(0) - 1);

    auto eigen = torch::linalg_eigh(C);
    auto evals = std::get<0>(eigen);
    auto evecs = std::get<1>(eigen);

    auto idx = std::get<1>(torch::sort(evals, /*dim=*/0, /*descending=*/true));
    auto Vt = torch::index_select(evecs, 1, idx).t();

    if (numComponents_ > 0) {
        Vt = Vt.slice(0, 0, numComponents_);
    }

    components_ = Vt;

    return *this;
}

// this may work correly
torch::Tensor block2Vector(const torch::Tensor& blockData, std::pair<int, int> patchSize) {
    int patchH = patchSize.first;
    int patchW = patchSize.second;

    auto sizes = blockData.sizes();
    int dims = sizes.size();

    int T = sizes[dims - 3];
    int H = sizes[dims - 2];
    int W = sizes[dims - 1];

    int nH = H / patchH;
    int nW = W / patchW;


    std::vector<int64_t> newShape;
    for (int i = 0; i < dims - 3; ++i)
        newShape.push_back(sizes[i]);
    newShape.push_back(T);
    newShape.push_back(nH);
    newShape.push_back(patchH);
    newShape.push_back(nW);
    newShape.push_back(patchW);

    auto reshaped = blockData.reshape(newShape);


    std::vector<int64_t> permuteOrder;
    for (int i = 0; i < dims - 3; ++i) permuteOrder.push_back(i);
    permuteOrder.push_back(dims - 3);
    permuteOrder.push_back(dims - 1); 
    permuteOrder.push_back(dims - 2);
    permuteOrder.push_back(dims);   
    permuteOrder.push_back(dims + 1); 

    auto permuted = reshaped.permute(permuteOrder);

    int64_t finalDim = patchH * patchW;

    std::vector<int64_t> finalShape;
    for (int i = 0; i < dims - 3; ++i) finalShape.push_back(sizes[i]);
    finalShape.push_back(T * nH * nW);
    finalShape.push_back(finalDim);

    return permuted.reshape(finalShape);
}
// idk if this works properly
torch::Tensor vector2Block(const torch::Tensor& vectors,
                           const std::vector<int64_t>& originalShape,
                           std::pair<int, int> patchSize) {
    int patchH = patchSize.first;
    int patchW = patchSize.second;

    int dims = originalShape.size();
    int T = originalShape[dims - 3];
    int H = originalShape[dims - 2];
    int W = originalShape[dims - 1];

    int nH = H / patchH;
    int nW = W / patchW;


    std::vector<int64_t> reshapedShape;
    for (int i = 0; i < dims - 3; ++i) reshapedShape.push_back(originalShape[i]);
    reshapedShape.push_back(nH);
    reshapedShape.push_back(nW);
    reshapedShape.push_back(T); 
    reshapedShape.push_back(patchH);
    reshapedShape.push_back(patchW);

    auto reshaped = vectors.reshape(reshapedShape);

std::vector<int64_t> permuteOrder;
int batchDims = dims - 3;
for (int i = 0; i < batchDims; ++i) permuteOrder.push_back(i);


permuteOrder.push_back(batchDims + 2); 
permuteOrder.push_back(batchDims + 0);
permuteOrder.push_back(batchDims + 3);
permuteOrder.push_back(batchDims + 1);
permuteOrder.push_back(batchDims + 4);


    auto permuted = reshaped.permute(permuteOrder).contiguous();

    return permuted.reshape(originalShape);
}

