#include "../CAESAR/data_utils.h"
#include <cassert>
#include <iostream>

// Verifies to_5d is a pure reshape: same numel, same data, in original
// order, just right-aligned into 5 dims with leading 1s. Then verifies
// restore_from_5d reverses it exactly.
void check_roundtrip(torch::Tensor data,
                     const std::vector<int64_t> &expected_5d) {
  torch::Tensor original = data.clone(); // to_5d nulls its input arg
  std::vector<int64_t> original_shape(original.sizes().begin(),
                                      original.sizes().end());

  auto [reshaped, info] = to_5d(data);

  std::cout << "Original shape: " << original_shape << std::endl;
  std::cout << "5D shape: " << reshaped.sizes() << std::endl;

  assert(reshaped.dim() == 5);
  assert(reshaped.sizes().vec() == expected_5d);
  assert(reshaped.numel() == original.numel());
  assert(info.original_shape == original_shape);
  assert(info.original_length == original.numel());

  // Data itself must be untouched, not just the shape -- reshape must not
  // reorder or fabricate elements.
  torch::Tensor flat_before = original.flatten();
  torch::Tensor flat_after = reshaped.flatten();
  assert(torch::allclose(flat_before, flat_after));

  torch::Tensor restored = restore_from_5d(reshaped, info);
  std::cout << "Restored shape: " << restored.sizes() << std::endl;

  assert(restored.sizes().vec() == original_shape);
  assert(torch::allclose(restored, original));
}

void test_2d() {
  std::cout << "\n=== Test 2D ===" << std::endl;
  check_roundtrip(torch::randn({50, 50}), {1, 1, 1, 50, 50});
}

void test_3d() {
  std::cout << "\n=== Test 3D ===" << std::endl;
  check_roundtrip(torch::randn({5, 100, 100}), {1, 1, 5, 100, 100});
}

void test_4d() {
  std::cout << "\n=== Test 4D ===" << std::endl;
  check_roundtrip(torch::randn({2, 5, 100, 100}), {1, 2, 5, 100, 100});
}

void test_5d() {
  std::cout << "\n=== Test 5D (already 5D, dims pass through unchanged) ==="
            << std::endl;
  check_roundtrip(torch::randn({2, 3, 10, 256, 256}), {2, 3, 10, 256, 256});
}

// Real GX shapes from bpls, not just synthetic sizes -- this is the case
// that actually broke in production.
void test_gx_density_shape() {
  std::cout << "\n=== Test GX Density shape {2,42,83,96,2} ===" << std::endl;
  check_roundtrip(torch::randn({2, 42, 83, 96, 2}), {2, 42, 83, 96, 2});
}

void test_gx_phi_shape() {
  std::cout << "\n=== Test GX Phi shape {42,83,96,2} ===" << std::endl;
  check_roundtrip(torch::randn({42, 83, 96, 2}), {1, 42, 83, 96, 2});
}

// to_5d must reject shapes it can't represent, rather than silently
// misbehaving.
void test_rejects_1d() {
  std::cout << "\n=== Test 1D input is rejected ===" << std::endl;
  torch::Tensor data = torch::randn({1000});
  bool threw = false;
  try {
    to_5d(data);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);
}

void test_rejects_6d() {
  std::cout << "\n=== Test 6D input is rejected ===" << std::endl;
  torch::Tensor data = torch::randn({2, 2, 2, 2, 2, 2});
  bool threw = false;
  try {
    to_5d(data);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  assert(threw);
}

// restore_from_5d must catch upstream data loss (e.g. a decompressor that
// only returns part of the data) instead of letting LibTorch's reshape
// throw an opaque InferSize error.
void test_restore_detects_element_mismatch() {
  std::cout << "\n=== Test restore_from_5d detects size mismatch ==="
            << std::endl;
  torch::Tensor data = torch::randn({2, 42, 83, 96, 2});
  auto [reshaped, info] = to_5d(data);

  // Simulate a decompressor that dropped half the data.
  torch::Tensor truncated =
      reshaped.flatten().slice(0, 0, reshaped.numel() / 2).clone();

  bool threw = false;
  try {
    restore_from_5d(truncated, info);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  assert(threw);
}

int main() {
  try {
    std::cout << "Starting data_utils tests..." << std::endl;

    test_2d();
    test_3d();
    test_4d();
    test_5d();
    test_gx_density_shape();
    test_gx_phi_shape();
    test_rejects_1d();
    test_rejects_6d();
    test_restore_detects_element_mismatch();

    std::cout << "\n✓ All tests passed!" << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "\n✗ Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
}
