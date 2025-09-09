#include "../CAESAR/models/runGaeCuda.h" 
#include <iostream>

int main() {
    std::cout<<"Starting to test runGAECUDA"<<std::endl;

    std::cout<<"Testing PCA now"<<std::endl;
    torch::Tensor X = torch::rand({10, 5}, torch::kFloat32);

    // Instantiate PCA, keep 3 components
    PCA pca(3, "cpu"); // you can use "cuda" if GPU is available

    // Fit PCA
    pca.fit(X);

    // Print mean
    std::cout << "Mean:\n" << pca.mean() << std::endl;

    // Print components
    std::cout << "Components:\n" << pca.components() << std::endl;
    std::cout<<"Done testing PCA"<<std::endl;


    std::cout<<"Testing block2Vector"<<std::endl;
    X = torch::rand({2, 4, 16, 16});

    auto result = block2Vector(X, {8,8});

    std::cout << "Input shape: " << X.sizes() << std::endl;
    std::cout << "Output shape: " << result.sizes() << std::endl;
    
    std::cout<<"Done testing block2Vector"<<std::endl;

    std::cout << "Testing block2Vector <-> vector2Block round-trip..." << std::endl;

    // Example block: shape [2, 3, 4, 8, 8] (can adjust for testing)
    auto original = torch::rand({2, 3, 4, 8, 8});

    std::pair<int,int> patchSize = {4, 4};

    // Convert block to vector
    auto vectors = block2Vector(original, patchSize);

    // Convert back to block
    std::vector<int64_t> originalShape(original.sizes().begin(), original.sizes().end());
    auto reconstructed = vector2Block(vectors, originalShape, patchSize);

    // Check if original and reconstructed are equal
    if (torch::allclose(original, reconstructed)) {
        std::cout << "PASS: Round-trip successful!" << std::endl;
    } else {
        std::cerr << "FAIL: Round-trip failed!" << std::endl;
        std::cerr << "Original: " << original << std::endl;
        std::cerr << "Reconstructed: " << reconstructed << std::endl;
    }
    std::cout<<"done testing both block2Vector and vector2Block"<<std::endl;


    std::cout<<"Done testing runGAECUDA"<<std::endl;
    return 0;
}
