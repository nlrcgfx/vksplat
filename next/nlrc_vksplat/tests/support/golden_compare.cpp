#include "golden_compare.hpp"

#include <sstream>
#include <stdexcept>

namespace nlrc::vksplat::tests {

bool compare_float_buffers(const float *expected,
                           const float *actual,
                           std::size_t count,
                           double epsilon,
                           std::string *first_mismatch) {
  for (std::size_t i = 0; i < count; ++i) {
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

void assert_no_nan_inf(const float *values, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) {
      throw std::runtime_error("Non-finite value in buffer");
    }
  }
}

} // namespace nlrc::vksplat::tests
