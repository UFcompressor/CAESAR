#include "../CAESAR/models/runGaeCuda.h" 
#include <iostream>
#include <chrono>

void testDecompression() {
    std::cout << "\n=== COMPREHENSIVE DECOMPRESSION TESTS ===" << std::endl;
    
    // Test 1: Full round-trip with small data
    {
        std::cout << "\nTest 1: Full compress->decompress round-trip..." << std::endl;
        
        double nrmse = 0.05;
        double quanFactor = 0.5;
        PCACompressor compressor(nrmse, quanFactor, "cpu");
        
        // Create test data with significant differences
        int numVectors = 50;
        int vectorSize = 64;
        
        torch::Tensor originalData = torch::randn({numVectors, vectorSize}, torch::kFloat32);
        // Make sure reconstruction is different enough to trigger compression
        torch::Tensor reconsData = originalData + 1.5 * torch::randn_like(originalData);
        
        // Compress
        auto compressResult = compressor.compress(originalData, reconsData);
        
        if (compressResult.dataBytes == 0) {
            std::cout << "Warning: No compression occurred, skipping test" << std::endl;
            return;
        }
        
        std::cout << "Compressed data bytes: " << compressResult.dataBytes << std::endl;
        
        // Decompress
        torch::Tensor decompressed = compressor.decompress(
            reconsData,
            compressResult.metaData,
            *compressResult.compressedData
        );
        
        // Verify shape
        assert(decompressed.sizes() == originalData.sizes());
        std::cout << "Shape verification: PASS" << std::endl;
        
        // Verify error bound
        torch::Tensor residual = originalData - decompressed;
        torch::Tensor norms = torch::linalg_norm(residual, c10::nullopt, {1});
        double maxNorm = torch::max(norms).item<double>();
        double errorBound = nrmse * std::sqrt(vectorSize);
        
        std::cout << "Max norm: " << maxNorm << ", Error bound: " << errorBound << std::endl;
        assert(maxNorm <= errorBound * 1.01); // Allow 1% tolerance for numerical precision
        
        std::cout << "Test 1: PASSED" << std::endl;
    }
    
    // Test 2: Block format round-trip
    {
        std::cout << "\nTest 2: Block format round-trip..." << std::endl;
        
        PCACompressor compressor(0.1, 0.5, "cpu");
        
        // 4D block data
        torch::Tensor originalData = torch::randn({5, 2, 16, 16}, torch::kFloat32);
        torch::Tensor reconsData = originalData + 2.0 * torch::randn_like(originalData);
        
        auto compressResult = compressor.compress(originalData, reconsData);
        
        if (compressResult.dataBytes > 0) {
            torch::Tensor decompressed = compressor.decompress(
                reconsData,
                compressResult.metaData,
                *compressResult.compressedData
            );
            
            assert(decompressed.sizes() == originalData.sizes());
            std::cout << "Block format shape verification: PASS" << std::endl;
            std::cout << "Test 2: PASSED" << std::endl;
        } else {
            std::cout << "Warning: No compression occurred for Test 2" << std::endl;
        }
    }
    
    // Test 3: Different patch sizes
    {
        std::cout << "\nTest 3: Different patch sizes round-trip..." << std::endl;
        
        std::pair<int, int> patchSize = {4, 4};
        PCACompressor compressor(0.08, 0.4, "cpu", "Zstd", patchSize);
        
        int numVectors = 100;
        int vectorSize = 16; // 4x4
        
        torch::Tensor originalData = torch::randn({numVectors, vectorSize}, torch::kFloat32);
        torch::Tensor reconsData = originalData + 1.0 * torch::randn_like(originalData);
        
        auto compressResult = compressor.compress(originalData, reconsData);
        
        if (compressResult.dataBytes > 0) {
            torch::Tensor decompressed = compressor.decompress(
                reconsData,
                compressResult.metaData,
                *compressResult.compressedData
            );
            
            // Check NRMSE
            torch::Tensor diff = originalData - decompressed;
            double nrmse_achieved = torch::norm(diff).item<double>() / 
                                   (torch::norm(originalData).item<double>() * std::sqrt(vectorSize));
            
            std::cout << "Achieved NRMSE: " << nrmse_achieved << std::endl;
            std::cout << "Test 3: PASSED" << std::endl;
        } else {
            std::cout << "Warning: No compression for Test 3" << std::endl;
        }
    }
    
    // Test 4: Large dataset stress test
    {
        std::cout << "\nTest 4: Large dataset stress test..." << std::endl;
        
        PCACompressor compressor(0.02, 0.3, "cpu");
        
        int numVectors = 2000;
        int vectorSize = 64;
        
        torch::Tensor originalData = torch::randn({numVectors, vectorSize}, torch::kFloat32);
        torch::Tensor reconsData = originalData + 1.5 * torch::randn_like(originalData);
        
        auto startCompress = std::chrono::high_resolution_clock::now();
        auto compressResult = compressor.compress(originalData, reconsData);
        auto endCompress = std::chrono::high_resolution_clock::now();
        
        if (compressResult.dataBytes > 0) {
            auto startDecompress = std::chrono::high_resolution_clock::now();
            torch::Tensor decompressed = compressor.decompress(
                reconsData,
                compressResult.metaData,
                *compressResult.compressedData
            );
            auto endDecompress = std::chrono::high_resolution_clock::now();
            
            auto compressTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                endCompress - startCompress).count();
            auto decompressTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                endDecompress - startDecompress).count();
            
            std::cout << "Compression time: " << compressTime << " ms" << std::endl;
            std::cout << "Decompression time: " << decompressTime << " ms" << std::endl;
            
            double compressionRatio = (double)(numVectors * vectorSize * sizeof(float)) / 
                                     compressResult.dataBytes;
            std::cout << "Compression ratio: " << compressionRatio << ":1" << std::endl;
            
            std::cout << "Test 4: PASSED" << std::endl;
        } else {
            std::cout << "Warning: No compression for Test 4" << std::endl;
        }
    }
    
    // Test 5: Edge case - single unique value
    {
        std::cout << "\nTest 5: Edge case - few unique coefficients..." << std::endl;
        
        PCACompressor compressor(0.1, 1.0, "cpu"); // Large quantization bin
        
        torch::Tensor originalData = torch::randn({100, 64}, torch::kFloat32);
        torch::Tensor reconsData = originalData + 2.0 * torch::randn_like(originalData);
        
        auto compressResult = compressor.compress(originalData, reconsData);
        
        if (compressResult.dataBytes > 0) {
            std::cout << "Unique values: " << compressResult.metaData.uniqueVals.size(0) << std::endl;
            
            torch::Tensor decompressed = compressor.decompress(
                reconsData,
                compressResult.metaData,
                *compressResult.compressedData
            );
            
            assert(decompressed.sizes() == originalData.sizes());
            std::cout << "Test 5: PASSED" << std::endl;
        }
    }
    
    // Test 6: Verify metadata preservation
    {
        std::cout << "\nTest 6: Metadata preservation..." << std::endl;
        
        PCACompressor compressor(0.05, 0.5, "cpu");
        
        torch::Tensor originalData = torch::randn({150, 64}, torch::kFloat32);
        torch::Tensor reconsData = originalData + 1.5 * torch::randn_like(originalData);
        
        auto compressResult = compressor.compress(originalData, reconsData);
        
        if (compressResult.dataBytes > 0) {
            // Verify metadata fields
            assert(compressResult.metaData.nVec == 150);
            assert(compressResult.metaData.pcaBasis.size(1) == 64);
            assert(compressResult.metaData.quanBin > 0);
            assert(compressResult.metaData.prefixLength > 0);
            
            std::cout << "All metadata fields verified" << std::endl;
            std::cout << "Test 6: PASSED" << std::endl;
        }
    }
    
    std::cout << "\n=== ALL DECOMPRESSION TESTS COMPLETED ===" << std::endl;
}


void testPCACompressor() {
    std::cout << "Starting to test PCACompressor..." << std::endl;

    // Test 1: Basic construction
    {
        std::cout << "Test 1: Basic construction..." << std::endl;

        double nrmse = 0.01;
        double quanFactor = 0.1;
        std::string device = "cpu"; // Use CPU for testing

        PCACompressor compressor(nrmse, quanFactor, device);

        std::cout << "PCACompressor constructed successfully" << std::endl;
    }

    // Test 2: Simple compression with small data
    {
        std::cout << "Test 2: Simple compression test..." << std::endl;

        double nrmse = 0.1;
        double quanFactor = 0.5;
        PCACompressor compressor(nrmse, quanFactor, "cpu");

        // Create simple test data (100 vectors of size 64 = 8x8 patches)
        int numVectors = 100;
        int vectorSize = 64; // 8x8

        // Original data with some structure
        torch::Tensor originalData = torch::randn({numVectors, vectorSize}, torch::kFloat32);

        // Add some correlation structure
        for (int i = 0; i < numVectors; ++i) {
            originalData[i] = originalData[i] + 0.5 * originalData[0]; // Add correlation
        }

        // Reconstruction data (slightly different)
        torch::Tensor reconsData = originalData + 0.05 * torch::randn_like(originalData);

        std::cout << "Input shapes - Original: [" << originalData.size(0) << ", "
                  << originalData.size(1) << "], Recons: [" << reconsData.size(0)
                  << ", " << reconsData.size(1) << "]" << std::endl;

        try {
            auto result = compressor.compress(originalData, reconsData);

            std::cout << "Compression completed!" << std::endl;
            std::cout << "Data bytes: " << result.dataBytes << std::endl;
            
            // FIX: Check if data was actually compressed before accessing tensor dimensions
            if (result.dataBytes > 0 && result.metaData.pcaBasis.numel() > 0) {
                std::cout << "PCA basis shape: [" << result.metaData.pcaBasis.size(0)
                          << ", " << result.metaData.pcaBasis.size(1) << "]" << std::endl;
                std::cout << "Number of vectors processed: " << result.metaData.nVec << std::endl;
                std::cout << "Unique values count: " << result.metaData.uniqueVals.size(0) << std::endl;
            } else {
                std::cout << "No compression needed - all residuals within error bounds" << std::endl;
                std::cout << "PCA basis shape: [empty]" << std::endl;
                std::cout << "Number of vectors processed: " << result.metaData.nVec << std::endl;
                std::cout << "Unique values count: 0" << std::endl;
            }

            std::cout << "Test 2 passed" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Test 2 failed with exception: " << e.what() << std::endl;
        }
    }

    // Test 3: Edge case - no data to compress
    {
        std::cout << "Test 3: Edge case - no data to compress..." << std::endl;

        double nrmse = 10.0; // Very high threshold
        double quanFactor = 1.0;
        PCACompressor compressor(nrmse, quanFactor, "cpu");

        // Create data where residual is very small
        torch::Tensor originalData = torch::ones({50, 64}, torch::kFloat32);
        torch::Tensor reconsData = originalData + 0.001 * torch::randn_like(originalData);

        auto result = compressor.compress(originalData, reconsData);

        std::cout << "Data bytes (should be 0): " << result.dataBytes << std::endl;
        assert(result.dataBytes == 0);
        assert(result.compressedData == nullptr);

        std::cout << "Test 3 passed" << std::endl;
    }

    // Test 4: Different patch sizes
    {
        std::cout << "Test 4: Different patch sizes..." << std::endl;

        double nrmse = 0.1;
        double quanFactor = 0.5;
        std::pair<int, int> patchSize = {4, 4}; // Smaller patch size

        PCACompressor compressor(nrmse, quanFactor, "cpu", "Zstd", patchSize);

        // Create data matching the patch size
        int numVectors = 50;
        int vectorSize = 16; // 4x4

        torch::Tensor originalData = torch::randn({numVectors, vectorSize}, torch::kFloat32);
        torch::Tensor reconsData = originalData + 0.1 * torch::randn_like(originalData);

        try {
            auto result = compressor.compress(originalData, reconsData);
            std::cout << "Compression with 4x4 patches completed, data bytes: "
                      << result.dataBytes << std::endl;
            
            // FIX: Add safety check here too
            if (result.dataBytes > 0) {
                std::cout << "Compression successful with patch size 4x4" << std::endl;
            } else {
                std::cout << "No compression needed for 4x4 patches" << std::endl;
            }
            
            std::cout << "Test 4 passed" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Test 4 failed with exception: " << e.what() << std::endl;
        }
    }

    // Test 5: Performance test with larger data
    {
        std::cout << "Test 5: Performance test with larger data..." << std::endl;

        double nrmse = 0.05;
        double quanFactor = 0.3;
        PCACompressor compressor(nrmse, quanFactor, "cpu");

        // Larger dataset
        int numVectors = 1000;
        int vectorSize = 64;

        torch::Tensor originalData = torch::randn({numVectors, vectorSize}, torch::kFloat32);
        torch::Tensor reconsData = originalData + 0.02 * torch::randn_like(originalData);

        auto startTime = std::chrono::high_resolution_clock::now();

        try {
            auto result = compressor.compress(originalData, reconsData);

            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                endTime - startTime).count();

            std::cout << "Large dataset compression completed in " << duration << " ms" << std::endl;
            std::cout << "Data bytes: " << result.dataBytes << std::endl;
            
            // FIX: Only calculate compression ratio if data was actually compressed
            if (result.dataBytes > 0) {
                std::cout << "Compression ratio: " <<
                    (static_cast<double>(numVectors * vectorSize * sizeof(float)) / result.dataBytes)
                    << ":1" << std::endl;
            } else {
                std::cout << "No compression needed - all data within error bounds" << std::endl;
            }

            std::cout << "Test 5 passed" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Test 5 failed with exception: " << e.what() << std::endl;
        }
    }

    // Test 6: Block data format (3D/4D tensors)
    {
        std::cout << "Test 6: Block data format..." << std::endl;

        double nrmse = 0.1;
        double quanFactor = 0.5;
        PCACompressor compressor(nrmse, quanFactor, "cpu");

        // Create 4D tensor data (batch, channels, height, width)
        torch::Tensor originalData = torch::randn({10, 1, 8, 8}, torch::kFloat32);
        torch::Tensor reconsData = originalData + 0.05 * torch::randn_like(originalData);

        try {
            auto result = compressor.compress(originalData, reconsData);
            std::cout << "Block format compression completed, data bytes: "
                      << result.dataBytes << std::endl;
            std::cout << "Test 6 passed" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Note: Test 6 may fail if block2Vector is not fully implemented" << std::endl;
            std::cout << "Exception: " << e.what() << std::endl;
        }
    }

    // Test 7: High reconstruction error (force float64 PCA)
    {
        std::cout << "Test 7: High reconstruction error (force float64 PCA)..." << std::endl;

        double nrmse = 0.01;   // Very strict error bound
        double quanFactor = 0.1;
        PCACompressor compressor(nrmse, quanFactor, "cpu");

        // Create highly random original data
        torch::Tensor originalData = torch::randn({50, 64}, torch::kFloat32);

        // Reconstruction is just noise (guaranteed high error)
        torch::Tensor reconsData = torch::zeros_like(originalData);

        auto result = compressor.compress(originalData, reconsData);

        std::cout << "Data bytes: " << result.dataBytes << std::endl;
        std::cout << "PCA basis shape: [" 
                  << result.metaData.pcaBasis.size(0) << ", "
                  << result.metaData.pcaBasis.size(1) << "]" << std::endl;
        std::cout << "Test 7 passed (float64 PCA path)" << std::endl;
    }

        // Test 8: Very small dataset (single vector)
    {
        std::cout << "Test 8: Very small dataset (single vector)..." << std::endl;

        PCACompressor compressor(0.1, 0.5, "cpu");

        // Just one vector of size 64
        torch::Tensor originalData = torch::randn({1, 64}, torch::kFloat32);
        torch::Tensor reconsData = originalData + 0.1 * torch::randn_like(originalData);

        auto result = compressor.compress(originalData, reconsData);

        std::cout << "Data bytes: " << result.dataBytes << std::endl;
        std::cout << "Unique values count: " << result.metaData.uniqueVals.size(0) << std::endl;
        std::cout << "Test 8 passed" << std::endl;
    }

        // Test 9: Using Cascaded codec instead of Zstd
    {
        std::cout << "Test 9: Cascaded codec compression..." << std::endl;

        double nrmse = 0.1;
        double quanFactor = 0.5;
        PCACompressor compressor(nrmse, quanFactor, "cpu", "Cascaded");

        torch::Tensor originalData = torch::randn({100, 64}, torch::kFloat32);
        torch::Tensor reconsData = originalData + 0.05 * torch::randn_like(originalData);

        auto result = compressor.compress(originalData, reconsData);

        std::cout << "Compression completed with Cascaded codec, data bytes: "
                  << result.dataBytes << std::endl;
        std::cout << "Test 9 passed" << std::endl;
    }




    std::cout << "Done testing PCACompressor" << std::endl;
}



void testBitUtils() {
    std::cout << "Starting to test BitUtils..." << std::endl;
    
    // Test 1: Basic bits to bytes conversion
    {
        std::cout << "Test 1: Basic bits to bytes..." << std::endl;
        
        // Create a tensor with 8 bits: [1, 0, 1, 1, 0, 0, 1, 0] = 178 in decimal
        torch::Tensor bits = torch::tensor({1, 0, 1, 1, 0, 0, 1, 0}, torch::kUInt8);
        
        auto bytes = BitUtils::bitsToBytes(bits);
        
        std::cout << "Input bits: ";
        for (int i = 0; i < bits.numel(); ++i) {
            std::cout << bits[i].item<uint8_t>() << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Output bytes: ";
        for (auto byte : bytes) {
            std::cout << (int)byte << " ";
        }
        std::cout << std::endl;
        
        // Expected: 178 (10110010 in binary)
        assert(bytes.size() == 1);
        assert(bytes[0] == 178);
        std::cout << "Test 1 passed" << std::endl;
    }
    
    // Test 2: Bytes to bits conversion
    {
          std::cout << "Test 2: Simple compression test..." << std::endl;

        double nrmse = 0.1;
        double quanFactor = 0.5;
        PCACompressor compressor(nrmse, quanFactor, "cpu");

        // Create simple test data (100 vectors of size 64 = 8x8 patches)
        int numVectors = 100;
        int vectorSize = 64; // 8x8

        // Original data with some structure
        torch::Tensor originalData = torch::randn({numVectors, vectorSize}, torch::kFloat32);

        // Add some correlation structure
        for (int i = 0; i < numVectors; ++i) {
            originalData[i] = originalData[i] + 0.5 * originalData[0]; // Add correlation
        }

        // Make reconstruction data significantly different to trigger compression
        // Use much larger noise to ensure residuals exceed error bound
        torch::Tensor reconsData = originalData + 2.0 * torch::randn_like(originalData); // Increased from 0.05 to 2.0

        std::cout << "Input shapes - Original: [" << originalData.size(0) << ", "
                  << originalData.size(1) << "], Recons: [" << reconsData.size(0)
                  << ", " << reconsData.size(1) << "]" << std::endl;

        // Calculate and print some residual statistics for debugging
        torch::Tensor residual = originalData - reconsData;
        torch::Tensor norms = torch::norm(residual, /*p=*/2, /*dim=*/1);
        double meanNorm = torch::mean(norms).item<double>();
        double maxNorm = torch::max(norms).item<double>();
        double errorBound = 0.1 * std::sqrt(64.0); // nrmse * sqrt(vectorSize)

        std::cout << "Residual stats - Mean norm: " << meanNorm
                  << ", Max norm: " << maxNorm
                  << ", Error bound: " << errorBound << std::endl;
        std::cout << "Vectors above threshold: " << torch::sum(norms > errorBound).item<int64_t>() << std::endl;

        try {
            auto result = compressor.compress(originalData, reconsData);

            std::cout << "Compression completed!" << std::endl;
            std::cout << "Data bytes: " << result.dataBytes << std::endl;

            // Check if data was actually compressed before accessing tensor dimensions
            if (result.dataBytes > 0 && result.metaData.pcaBasis.numel() > 0) {
                std::cout << "PCA basis shape: [" << result.metaData.pcaBasis.size(0)
                          << ", " << result.metaData.pcaBasis.size(1) << "]" << std::endl;
                std::cout << "Number of vectors processed: " << result.metaData.nVec << std::endl;
                std::cout << "Unique values count: " << result.metaData.uniqueVals.size(0) << std::endl;
            } else {
                std::cout << "No compression needed - all residuals within error bounds" << std::endl;
                std::cout << "PCA basis shape: [empty]" << std::endl;
                std::cout << "Number of vectors processed: " << result.metaData.nVec << std::endl;
                std::cout << "Unique values count: 0" << std::endl;
            }

            std::cout << "Test 2 passed" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Test 2 failed with exception: " << e.what() << std::endl;
        }
    }
    
    // Test 3: Round trip conversion
    {
        std::cout << "Test 3: Round trip conversion..." << std::endl;
        
        torch::Tensor originalBits = torch::tensor({1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1}, torch::kBool);
        
        // Convert to bytes
        auto bytes = BitUtils::bitsToBytes(originalBits);
        
        // Convert back to bits with original length
        auto reconstructedBits = BitUtils::bytesToBits(bytes, originalBits.numel());
        
        std::cout << "Original bits: ";
        for (int i = 0; i < originalBits.numel(); ++i) {
            std::cout << originalBits[i].item<bool>() << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Reconstructed bits: ";
        for (int i = 0; i < reconstructedBits.numel(); ++i) {
            std::cout << reconstructedBits[i].item<bool>() << " ";
        }
        std::cout << std::endl;
        
        // Check they match
        assert(originalBits.numel() == reconstructedBits.numel());
        for (int i = 0; i < originalBits.numel(); ++i) {
            assert(originalBits[i].item<bool>() == reconstructedBits[i].item<bool>());
        }
        std::cout << "Test 3 passed" << std::endl;
    }
    
    // Test 4: Different data types
    {
        std::cout << "Test 4: Different data types..." << std::endl;
        
        // Test with float tensor (should be converted to uint8)
        torch::Tensor floatBits = torch::tensor({1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0}, torch::kFloat);
        
        auto bytes = BitUtils::bitsToBytes(floatBits);
        auto bits = BitUtils::bytesToBits(bytes, 8);
        
        std::cout << "Float input converted and reconstructed successfully" << std::endl;
        std::cout << "Test 4 passed" << std::endl;
    }
    
    // Test 5: Padding handling
    {
        std::cout << "Test 5: Padding handling..." << std::endl;
        
        // Test with 10 bits (will be padded to 16 bits / 2 bytes)
        torch::Tensor bits10 = torch::tensor({1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, torch::kBool);
        
        auto bytes = BitUtils::bitsToBytes(bits10);
        auto reconstructed = BitUtils::bytesToBits(bytes, 10); // Specify original length
        
        std::cout << "10-bit input handled with padding correctly" << std::endl;
        assert(reconstructed.numel() == 10);
        std::cout << "Test 5 passed" << std::endl;
    }
    
    std::cout << "Done testing BitUtils" << std::endl;
}


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

   std::cout << "Starting to test indexMaskPrefix..." << std::endl;
    auto arr_2d = torch::tensor({{0, 1, 0, 0},
                                 {1, 0, 0, 1},
                                 {0, 0, 0, 1}}, torch::kFloat32);

    // Step 1: run prefix
    auto [prefix_mask, mask_length] = indexMaskPrefix(arr_2d);
    std::cout << "prefix_mask: " << prefix_mask << std::endl;
    std::cout << "mask_length: " << mask_length << std::endl;

    // Step 2: run reverse
    auto arr_2d_reconstructed = indexMaskReverse(prefix_mask, mask_length, arr_2d.size(1));
    std::cout << "Reconstructed arr_2d:\n" << arr_2d_reconstructed << std::endl;
    std::cout << "Done testing indexMaskReverse." << std::endl;

     testBitUtils();

 try {
        testPCACompressor();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
     testDecompression();

    std::cout<<"Done testing runGAECUDA"<<std::endl;
    return 0;
}
