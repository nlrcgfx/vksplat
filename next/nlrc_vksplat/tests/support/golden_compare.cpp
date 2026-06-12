#include "golden_compare.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace nlrc::vksplat::tests {

bool compare_float_buffers(Span<const float> expected,
                           Span<const float> actual,
                           double epsilon,
                           std::string *first_mismatch) {
  if (expected.size() != actual.size()) {
    if (first_mismatch != nullptr) {
      std::ostringstream stream;
      stream << "Size mismatch: expected " << expected.size() << " values got " << actual.size();
      *first_mismatch = stream.str();
    }
    return false;
  }

  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double diff = std::abs(static_cast<double>(expected[i]) - static_cast<double>(actual[i]));
    if (diff > epsilon) {
      if (first_mismatch != nullptr) {
        std::ostringstream stream;
        stream << "Mismatch at index " << i << ": expected " << expected[i] << " got " << actual[i];
        *first_mismatch = stream.str();
      }
      return false;
    }
  }
  return true;
}

void assert_no_nan_inf(Span<const float> values) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i])) {
      throw std::runtime_error("Non-finite value in buffer");
    }
  }
}

} // namespace nlrc::vksplat::tests
