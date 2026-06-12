#pragma once

#include <cmath>
#include <cstddef>
#include <string>

namespace nlrc::vksplat::tests {

inline constexpr double kDefaultFloatEpsilon = 1e-5;

[[nodiscard]] bool compare_float_buffers(const float *expected,
                                         const float *actual,
                                         std::size_t count,
                                         double epsilon = kDefaultFloatEpsilon,
                                         std::string *first_mismatch = nullptr);

void assert_no_nan_inf(const float *values, std::size_t count);

} // namespace nlrc::vksplat::tests
