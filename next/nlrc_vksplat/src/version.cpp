#include "version.hpp"

#include "smoke_spirv.hpp"

namespace nlrc::vksplat {

[[nodiscard]] const char *version() {
  return "0.0.0-scaffold";
}

[[nodiscard]] std::size_t embedded_shader_smoke_words() {
  return shaders::kSmokeSpirvWordCount;
}

} // namespace nlrc::vksplat
