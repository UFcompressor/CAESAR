#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace caesar {

// Explicit values are part of the compressed-buffer metadata contract.
enum class CorrectionMethod : uint8_t { GAE = 0, LBRC = 1, NGLR = 2 };

inline CorrectionMethod correction_method_from_byte(uint8_t value) {
  switch (value) {
  case 0:
    return CorrectionMethod::GAE;
  case 1:
    return CorrectionMethod::LBRC;
  case 2:
    return CorrectionMethod::NGLR;
  default:
    throw std::invalid_argument("Unknown CAESAR correction method: " +
                                std::to_string(value));
  }
}

inline CorrectionMethod correction_method_from_string(const std::string &name) {
  if (name == "gae")
    return CorrectionMethod::GAE;
  if (name == "lbrc")
    return CorrectionMethod::LBRC;
  if (name == "nglr")
    return CorrectionMethod::NGLR;
  throw std::invalid_argument("Unknown CAESAR correction method: " + name);
}

} // namespace caesar
