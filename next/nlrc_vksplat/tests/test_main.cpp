// Catch2 main is provided by Catch2::Catch2WithMain.
// This translation unit anchors the test executable target.

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Catch2 harness is linked", "[host]") {
  REQUIRE(true);
}
