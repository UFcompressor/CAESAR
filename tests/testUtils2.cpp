#include "../CAESAR/models/utils.h"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <list>
#include <deque>
#include <array>
#include <memory>
#include <chrono>


int main() {
    torch::Tensor a = torch::randn({5, 3}); // (rows, cols)
    torch::Tensor t = torch::tensor({1, 3, 0}, torch::dtype(torch::kLong)); // indices

    auto result = extractTensor(a, t);

    std::cout << "Input tensor a:\n" << a << "\n\n";
    std::cout << "Index tensor t:\n" << t << "\n\n";
    std::cout << "Result of extractTensor:\n" << result << "\n\n";



    std::cout << "Done testing extractTensor " << std::endl;
    std::cout << "Testing noiseLike placeholder..." << std::endl;
    std::vector<int64_t> shape = {3, 4, 5};
    torch::Device device(torch::kCPU);
    try {
  
        auto noise = noiseLike(shape, device, false);
        std::cout << "noiseLike output:\n" << noise << "\n";
    } catch (const std::exception& e) {
        std::cout << "Caught exception from noiseLike: " << e.what() << std::endl;
    }

    try {
 
        auto repeat_noise = noiseLike(shape, device, true);
        std::cout << "repeat noiseLike output:\n" << repeat_noise << "\n";
    } catch (const std::exception& e) {
        std::cout << "Caught exception from repeat noiseLike: " << e.what() << std::endl;
    }

    std::cout << "Done testing noiseLike " << std::endl;


    std::cout<<"Testing Noise "<<std::endl;

 torch::Tensor input1 = torch::ones({3, 3});
    float scale1 = 0.2;

    auto output1 = noise(input1, scale1);

    std::cout << "Input (all ones):\n" << input1 << "\n\n";
    std::cout << "Output after noise (scale=0.2):\n" << output1 << "\n\n";

    // -------------------------
    // Test 2: Larger random tensor
    // -------------------------
    torch::Tensor input2 = torch::randn({2, 4});
    float scale2 = 0.5;

    auto output2 = noise(input2, scale2);

    std::cout << "Input (random):\n" << input2 << "\n\n";
    std::cout << "Output after noise (scale=0.5):\n" << output2 << "\n\n";

    std::cout << "Done testing noise " << std::endl;


    int timesteps1 = 5;
    int timesteps2 = 10;

    auto betas1 = cosineBetaSchedule(timesteps1);
    auto betas2 = cosineBetaSchedule(timesteps2);
    std::cout << "Test with timesteps = " << timesteps1 << ":\n";
    std::cout << betas1 << "\n";
    std::cout << "Test with timesteps = " << timesteps2 << ":\n";
    std::cout << betas2 << "\n";
    std::cout << "Done testing cosineBetaSchedule \n";
    std::cout << "Starting test: linearBetaSchedule\n";
  auto betas3 = linearBetaSchedule(timesteps1);
    auto betas4 = linearBetaSchedule(timesteps2);
    std::cout << "Test with timesteps = " << timesteps1 << ":\n";
    std::cout << betas3 << "\n";

    std::cout << "Test with timesteps = " << timesteps2 << ":\n";
    std::cout << betas4 << "\n";

    std::cout << "Done testing linearBetaSchedule \n";

    std::cout << "Starting test: roundWOffset\n";

    torch::Tensor input = torch::tensor({0.2, 1.7, 2.5, -0.6}, torch::kDouble);
    torch::Tensor loc   = torch::tensor({0.5, 1.2, 2.3, -1.0}, torch::kDouble);

     result = roundWOffset(input, loc);

    std::cout << "Input:\n" << input << "\n";
    std::cout << "Loc:\n"   << loc   << "\n";
    std::cout << "Result:\n" << result << "\n";

    std::cout << "Done testing roundWOffset \n";


std::cout << "Starting test roundWOffset" << std::endl;

input = torch::tensor({0.2, 1.7, -2.3});
loc   = torch::tensor({0.5, 1.5, -2.0});

result = roundWOffset(input, loc);

std::cout << "Input:\n" << input << std::endl;
std::cout << "Loc:\n" << loc << std::endl;
std::cout << "Result:\n" << result << std::endl;

std::cout << "done testing roundwoffset" << std::endl;




std::cout<<"Starting testing quatize "<<std::endl;
auto x = torch::randn({4,4});

// noise mode
auto q1 = quantize(x, "noise");

// round mode
auto q2 = quantize(x, "round");

// dequantize mode (need offset)
auto offset = torch::randn({4,4});
auto q3 = quantize(x, "dequantize", offset);



std::cout << "done testing quantize" << std::endl;
auto z = torch::tensor({0.5, 1.2, -0.3}, torch::requires_grad());
auto bound = torch::tensor(0.0);
auto y = LowerBound::apply(z, bound);
std::cout << y << std::endl;

auto grad = torch::ones_like(y);
y.backward(grad); 

std::cout << "done testing LowerBound" << std::endl;


std::cout<<"Starting to test UpperBound"<<std::endl;

auto X = torch::tensor({0.5, 1.2, -0.3}, torch::requires_grad());
auto Bound = torch::tensor(1.0);

auto Y = UpperBound::apply(X, Bound);
std::cout << Y << std::endl;

auto Grad = torch::ones_like(Y);
Y.backward(Grad);

std::cout << "done testing UpperBound" << std::endl;



std::cout<<"Testing NormalDistrubtion" <<std::endl;

    auto Loc = torch::tensor({0.0, 1.0, -1.0});
    auto scale = torch::tensor({1.0, 0.5, 2.0});

    NormalDistribution dist(Loc, scale);

    std::cout << "Mean: " << dist.mean() << std::endl;

    auto sample = dist.sample();
    std::cout << "Sample: " << sample << std::endl;

    auto T = torch::tensor({0.5, 1.2, -0.3});
    auto likelihood = dist.likelihood(T);
    std::cout << "Likelihood: " << likelihood << std::endl;

    auto scaled_likelihood = dist.scaledLikelihood(T, 2.0);
    std::cout << "Scaled Likelihood: " << scaled_likelihood << std::endl;

    std::cout<<"Done testing NormalDistrubtion" <<std::endl;






    std::cout<< "DONE TESTING\n";
    return 0;
}

