#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "golden_compare.hpp"
#include "span.hpp"

TEST_CASE("Golden float compare passes within epsilon", "[host]") {
  const std::vector<float> actual{1.0F, 2.000001F, 2.999999F};
  auto actual_span = nlrc::vksplat::make_span(actual);

  const std::vector<float> expected{1.0F, 2.0F, 3.0F};
  auto expected_span = nlrc::vksplat::make_span(expected);

  REQUIRE(nlrc::vksplat::tests::compare_float_buffers(expected_span, actual_span));
}

TEST_CASE("Golden float compare fails outside epsilon", "[host]") {
  const std::vector<float> actual{1.0F, 2.1F};
  auto actual_span = nlrc::vksplat::make_span(actual);

  const std::vector<float> expected{1.0F, 2.0F};
  auto expected_span = nlrc::vksplat::make_span(expected);

  std::string message;
  const double epsilon = 1e-5;

  REQUIRE_FALSE(nlrc::vksplat::tests::compare_float_buffers(expected_span, actual_span, epsilon, &message));
  REQUIRE_FALSE(message.empty());
}

TEST_CASE("Golden float compare reports size mismatch", "[host]") {
  const std::vector<float> actual{1.0F};
  auto actual_span = nlrc::vksplat::make_span(actual);

  const std::vector<float> expected{1.0F, 2.0F};
  auto expected_span = nlrc::vksplat::make_span(expected);

  std::string message;
  const double epsilon = 1e-5;

  REQUIRE_FALSE(nlrc::vksplat::tests::compare_float_buffers(expected_span, actual_span, epsilon, &message));
  REQUIRE_FALSE(message.empty());
}

TEST_CASE("NaN detection rejects non-finite values", "[host]") {
  const std::vector<float> values{1.0F, std::numeric_limits<float>::quiet_NaN()};
  auto values_span = nlrc::vksplat::make_span(values);

  REQUIRE_THROWS_AS(nlrc::vksplat::tests::assert_no_nan_inf(values_span), std::runtime_error);
}

TEST_CASE("Infinity detection rejects non-finite values", "[host]") {
  const std::vector<float> values{1.0F, std::numeric_limits<float>::infinity()};
  auto values_span = nlrc::vksplat::make_span(values);

  REQUIRE_THROWS_AS(nlrc::vksplat::tests::assert_no_nan_inf(values_span), std::runtime_error);
}
