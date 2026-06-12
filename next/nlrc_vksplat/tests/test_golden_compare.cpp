#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "golden_compare.hpp"
#include "span.hpp"

TEST_CASE("Golden float compare passes within epsilon", "[host]") {
  const std::vector<float> expected{1.0F, 2.0F, 3.0F};
  const std::vector<float> actual{1.0F, 2.000001F, 2.999999F};
  REQUIRE(nlrc::vksplat::tests::compare_float_buffers(nlrc::vksplat::make_span(expected),
                                                      nlrc::vksplat::make_span(actual)));
}

TEST_CASE("Golden float compare fails outside epsilon", "[host]") {
  const std::vector<float> expected{1.0F, 2.0F};
  const std::vector<float> actual{1.0F, 2.1F};
  std::string message;
  REQUIRE_FALSE(nlrc::vksplat::tests::compare_float_buffers(nlrc::vksplat::make_span(expected),
                                                            nlrc::vksplat::make_span(actual), 1e-5, &message));
  REQUIRE_FALSE(message.empty());
}

TEST_CASE("Golden float compare reports size mismatch", "[host]") {
  const std::vector<float> expected{1.0F, 2.0F};
  const std::vector<float> actual{1.0F};
  std::string message;
  REQUIRE_FALSE(nlrc::vksplat::tests::compare_float_buffers(nlrc::vksplat::make_span(expected),
                                                            nlrc::vksplat::make_span(actual), 1e-5, &message));
  REQUIRE_FALSE(message.empty());
}

TEST_CASE("NaN detection rejects non-finite values", "[host]") {
  const std::vector<float> values{1.0F, std::numeric_limits<float>::quiet_NaN()};
  REQUIRE_THROWS_AS(nlrc::vksplat::tests::assert_no_nan_inf(nlrc::vksplat::make_span(values)), std::runtime_error);
}
