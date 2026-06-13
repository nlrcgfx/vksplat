#include <cstddef>
#include <cstdlib>
#include <iostream>

#include "gpu/shader_descriptors.hpp"

#include <nlohmann/json.hpp>

int main() {
  const auto shaders = nlrc::vksplat::gpu::shader_interfaces();

  nlohmann::ordered_json output;
  output["schema"] = 2;
  output["shaders"] = nlohmann::ordered_json::object();

  for (std::size_t shader_index = 0; shader_index < shaders.size(); ++shader_index) {
    const auto &shader = shaders[shader_index];
    auto bindings = nlohmann::ordered_json::array();
    for (std::size_t binding_index = 0; binding_index < shader.bindings.size(); ++binding_index) {
      bindings.push_back(shader.bindings[binding_index].name);
    }
    output["shaders"][shader.logical_name] = std::move(bindings);
  }

  std::cout << output.dump(2) << '\n';

  return EXIT_SUCCESS;
}
