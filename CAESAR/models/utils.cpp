#include "utils.h"



 std::vector<int>numToGroups(int num, int divisor){
    if(divisor <= 0){
        throw std::invalid_argument("Divisor must be positive +");
    }
    if(num < 0){
        throw std::invalid_argument("Num must be non-negative");
    }

    int groups = num / divisor;
    int rem = num % divisor;

    std::vector<int> arr(groups, divisor);
    if(rem > 0){
        arr.push_back(rem);
    }

    return arr;
}
torch::Tensor extract(const torch::Tensor& a, const torch::Tensor& t, const std::vector<int64_t>& x_shape) {

    int64_t b = t.size(0);
    
    torch::Tensor out = a.gather(-1, t);
    
    std::vector<int64_t> reshape_dims;
    reshape_dims.push_back(b);

    for (size_t i = 1; i < x_shape.size(); ++i) {
        reshape_dims.push_back(1);
    }
    
    return out.reshape(reshape_dims);
}

