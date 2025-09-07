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


torch::Tensor extractTensor(const torch::Tensor& a, 
        const torch::Tensor& t, const torch::Tensor& placeHolder){
   
    TORCH_CHECK(t.dtype() == torch::kLong, "Index tenosr t must be of type Long");

    auto device = a.device();
    auto arrangeIdx = torch::arange(t.size(0), torch::TensorOptions().dtype(torch::kLong).device(device));

    return a.index({t, arrangeIdx});
}

// do not trust this  till you have noise --------------------------
torch::Tensor noiseLike(const std::vector<int64_t>& shape, 
        torch::Device device, bool repeat){
    TORCH_CHECK(!shape.empty(), "Shape can't be empty");

    if (repeat){
        std::vector<int64_t> smallShape(shape.begin() + 1, shape.end());
        auto noise = torch::randn(smallShape, torch::TensorOptions().device(device));

        noise = noise.unsqueeze(0);

        std::vector<int64_t> repeats(shape.size(),1);
        repeats[0] = shape[0];
        return noise.repeat(repeats);
    } 
    else {
       // they used a lambda func and it returns this essintally
       //  different from the noise method --------
         return torch::randn(shape, torch::TensorOptions().device(device));
    }
}

// not scale is a float not double ask later IDK
torch::Tensor noise(const torch::Tensor& input, float scale){
    auto randTensor = torch::rand_like(input)-0.5;
    return input + scale * randTensor;
}


torch::Tensor cosineBetaSchedule(int64_t timeSteps, double s) {
    int64_t steps = timeSteps + 1;
    auto x = torch::linspace(0, steps, steps, torch::TensorOptions().dtype(torch::kDouble));

    auto alphasCumprod = torch::cos(((x / steps)/ + s) / (1.0 + s) * M_PI * 0.5);
    alphasCumprod = alphasCumprod.pow(2);


     alphasCumprod = alphasCumprod / alphasCumprod[0];
   
    auto alphasCumprodNext = alphasCumprod.slice(0,1);
    auto alphasCumprodPrev = alphasCumprod.slice(0,0, -1);

    auto  betas = 1.0 - (alphasCumprodNext / alphasCumprodPrev);

    betas = betas.clamp(0.0,0.999);
    
    return betas;
}

torch::Tensor linearBetaSchedule(int64_t timeSteps){
    TORCH_CHECK(timeSteps > 0, "Time Steps must be postive");

    double scale = 1000.0 / static_cast<double>(timeSteps);
    double betaStart = scale * 0.0001;
    double betaEnd = scale * 0.02;


    return torch::linspace(betaStart, betaEnd, timeSteps,
                           torch::TensorOptions().dtype(torch::kDouble));

}



torch::Tensor roundWOffset(const torch::Tensor& input,
        const torch::Tensor& loc){
    TORCH_CHECK(input.sizes() == loc.sizes(),
            "roundWOffset: input and loc must have the same shapes");
    auto diff = STERound::apply(input - loc);
    return diff + loc; 


}

torch::Tensor STERound::forward(torch::autograd::AutogradContext* ctx,
                                const torch::Tensor& x) {
    return torch::round(x);
}


torch::autograd::tensor_list STERound::backward(torch::autograd::AutogradContext* ctx,
                                                torch::autograd::tensor_list grad_outputs) {
    return {grad_outputs[0]};
}


torch::Tensor quantize(const torch::Tensor& x, const std::string& mode, const torch::Tensor& offset){

    if(mode == "noise"){
        return noise(x,1.0);
    }
    else if (mode == "round"){
        return  STERound::apply(x);
    }
    else if (mode == "dequantize"){
        if(!offset.defined()){
            throw std::invalid_argument("METHOD: 'quantize' mode: 'dequantize' requires a valid offset tenosr ");
        }
        return roundWOffset(x, offset);
      }
    else{
        throw std::invalid_argument("METHOD: 'quantize' mode: " + mode + " not implemented");
    }
}

torch::Tensor LowerBound::forward(torch::autograd::AutogradContext* ctx,
                                  const torch::Tensor& inputs,
                                  const torch::Tensor& bound) {

    auto b = torch::full_like(inputs, bound.item<double>());


    ctx->save_for_backward({inputs, b});

    return torch::max(inputs, b);
}


torch::autograd::tensor_list LowerBound::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs) {

    auto saved = ctx->get_saved_variables();
    auto inputs = saved[0];
    auto b = saved[1];

    auto gradOutput = grad_outputs[0];

    auto passThrough = (inputs >= b) | (gradOutput < 0);

    return {passThrough.to(gradOutput.dtype()) * gradOutput, torch::Tensor()};
}














