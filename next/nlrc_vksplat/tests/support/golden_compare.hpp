#pragma once

#include <string>

#include "span.hpp"

namespace nlrc::vksplat::tests {

inline constexpr double kDefaultFloatEpsilon = 1e-5;

[[nodiscard]] bool compare_float_buffers(Span<const float> expected,
                                         Span<const float> actual,
                                         double epsilon = kDefaultFloatEpsilon,
                                         std::string *first_mismatch = nullptr);

void assert_no_nan_inf(Span<const float> values);

} // namespace nlrc::vksplat::tests
