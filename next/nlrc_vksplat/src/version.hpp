#pragma once

#include <cstddef>

namespace nlrc::vksplat {

[[nodiscard]] const char *version();

// Non-zero when embedded smoke shader bytecode is linked (shader build pipeline).
[[nodiscard]] std::size_t embedded_shader_smoke_words();

} // namespace nlrc::vksplat
