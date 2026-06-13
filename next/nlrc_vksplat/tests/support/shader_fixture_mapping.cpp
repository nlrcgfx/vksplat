#include "shader_fixture_mapping.hpp"

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fixture_loader.hpp"

#include <nlohmann/json.hpp>

namespace nlrc::vksplat::tests {
namespace {

struct Route final {
  std::string subgraph;
  std::string stage_name;
  std::string stage_name_prefix;
  std::string side;
  std::string contract;
  bool untracked{false};
};

struct BindingContracts final {
  std::map<std::string, std::vector<std::string>> shaders;
  std::map<std::string, std::vector<std::string>> fixture_contracts;
  std::vector<Route> routes;
  std::vector<Route> untracked_routes;
};

[[nodiscard]] bool has_prefix(const std::string &value, std::string_view prefix) {
  return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

[[nodiscard]] std::string side_name(FixtureManifestSide side) {
  switch (side) {
    case FixtureManifestSide::Fixture:
      return "fixture";
    case FixtureManifestSide::Golden:
      return "golden";
  }
  throw std::invalid_argument("Unknown fixture manifest side");
}

[[nodiscard]] std::map<std::string, std::vector<std::string>> parse_contract_group(const nlohmann::json &root,
                                                                                   const char *group_name) {
  const auto &group = root.at(group_name);
  if (!group.is_object()) {
    throw std::runtime_error(std::string(group_name) + " must be a JSON object");
  }

  std::map<std::string, std::vector<std::string>> contracts;
  for (auto iter = group.begin(); iter != group.end(); ++iter) {
    if (iter.key().empty() || !iter.value().is_array()) {
      throw std::runtime_error(std::string(group_name) + " contains an invalid contract");
    }

    std::vector<std::string> bindings;
    bindings.reserve(iter.value().size());
    for (const auto &binding : iter.value()) {
      if (!binding.is_string()) {
        throw std::runtime_error(std::string(group_name) + "/" + iter.key() + " binding must be a string");
      }
      bindings.push_back(binding.get<std::string>());
    }
    contracts.emplace(iter.key(), std::move(bindings));
  }
  return contracts;
}

[[nodiscard]] std::string optional_route_string(const nlohmann::json &route, const char *key) {
  if (!route.contains(key)) {
    return {};
  }
  if (!route.at(key).is_string()) {
    throw std::runtime_error(std::string("route field must be a string: ") + key);
  }
  return route.at(key).get<std::string>();
}

[[nodiscard]] Route parse_route(const nlohmann::json &value, bool require_contract) {
  if (!value.is_object()) {
    throw std::runtime_error("route entries must be JSON objects");
  }

  Route route;
  route.subgraph = optional_route_string(value, "subgraph");
  route.stage_name = optional_route_string(value, "stage_name");
  route.stage_name_prefix = optional_route_string(value, "stage_name_prefix");
  route.side = value.contains("side") ? optional_route_string(value, "side") : "both";
  route.contract = optional_route_string(value, "contract");
  route.untracked = !require_contract;

  if (route.subgraph.empty()) {
    throw std::runtime_error("route is missing subgraph");
  }
  if ((route.stage_name.empty()) == (route.stage_name_prefix.empty())) {
    throw std::runtime_error("route must set exactly one stage matcher");
  }
  if (route.side != "fixture" && route.side != "golden" && route.side != "both") {
    throw std::runtime_error("route has invalid side: " + route.side);
  }
  if (require_contract && route.contract.empty()) {
    throw std::runtime_error("tracked route is missing contract");
  }
  if (!require_contract && !route.contract.empty()) {
    throw std::runtime_error("untracked route must not set contract");
  }
  return route;
}

[[nodiscard]] std::pair<std::string, std::string> split_contract_ref(const std::string &contract_ref) {
  const auto separator = contract_ref.find('/');
  if (separator == std::string::npos || separator == 0 || separator + 1 >= contract_ref.size()) {
    throw std::runtime_error("invalid binding contract reference: " + contract_ref);
  }
  return {contract_ref.substr(0, separator), contract_ref.substr(separator + 1)};
}

void validate_contract_ref(const BindingContracts &contracts, const std::string &contract_ref) {
  const auto [group, name] = split_contract_ref(contract_ref);
  if (group == "shaders") {
    if (contracts.shaders.find(name) == contracts.shaders.end()) {
      throw std::runtime_error("route references missing shader contract: " + contract_ref);
    }
    static_cast<void>(gpu::shader_interface(name));
    return;
  }
  if (group == "fixture_contracts") {
    if (contracts.fixture_contracts.find(name) == contracts.fixture_contracts.end()) {
      throw std::runtime_error("route references missing fixture contract: " + contract_ref);
    }
    return;
  }
  throw std::runtime_error("route references invalid binding contract group: " + contract_ref);
}

[[nodiscard]] BindingContracts load_binding_contracts() {
  const auto contracts_path = test_data_root() / "shader_binding_contracts.json";
  std::ifstream input(contracts_path);
  if (!input.is_open()) {
    throw std::runtime_error("could not open shader binding contracts: " + contracts_path.string());
  }

  const auto root = nlohmann::json::parse(input);
  if (root.at("schema").get<int>() != 2) {
    throw std::runtime_error("unsupported shader binding contract schema: " + contracts_path.string());
  }

  BindingContracts contracts;
  contracts.shaders = parse_contract_group(root, "shaders");
  contracts.fixture_contracts = parse_contract_group(root, "fixture_contracts");

  const auto &routes = root.at("routes");
  const auto &untracked_routes = root.at("untracked_routes");
  if (!routes.is_array() || !untracked_routes.is_array()) {
    throw std::runtime_error("shader binding routes must be JSON arrays");
  }

  contracts.routes.reserve(routes.size());
  for (const auto &route : routes) {
    contracts.routes.push_back(parse_route(route, true));
  }

  contracts.untracked_routes.reserve(untracked_routes.size());
  for (const auto &route : untracked_routes) {
    contracts.untracked_routes.push_back(parse_route(route, false));
  }

  for (const auto &route : contracts.routes) {
    validate_contract_ref(contracts, route.contract);
  }
  return contracts;
}

[[nodiscard]] const BindingContracts &binding_contracts() {
  static const BindingContracts contracts = load_binding_contracts();
  return contracts;
}

[[nodiscard]] bool route_matches(const Route &route, const FixtureManifest &manifest, FixtureManifestSide side) {
  const auto requested_side = side_name(side);
  if (route.side != "both" && route.side != requested_side) {
    return false;
  }
  if (route.subgraph != manifest.subgraph) {
    return false;
  }
  if (!route.stage_name.empty()) {
    return route.stage_name == manifest.stage_name;
  }
  return has_prefix(manifest.stage_name, route.stage_name_prefix);
}

[[nodiscard]] const Route *route_for_manifest(const FixtureManifest &manifest, FixtureManifestSide side) {
  const auto &contracts = binding_contracts();
  for (const auto &route : contracts.routes) {
    if (route_matches(route, manifest, side)) {
      return &route;
    }
  }
  for (const auto &route : contracts.untracked_routes) {
    if (route_matches(route, manifest, side)) {
      return &route;
    }
  }
  throw std::runtime_error("no shader binding contract route for stage " + manifest.stage_name + " on " +
                           side_name(side) + " side");
}

[[nodiscard]] std::vector<std::string> binding_names_for_contract(const std::string &contract_ref) {
  const auto [group, name] = split_contract_ref(contract_ref);
  if (group == "shaders") {
    return gpu::binding_names(gpu::shader_interface(name));
  }
  if (group == "fixture_contracts") {
    const auto &contracts = binding_contracts().fixture_contracts;
    const auto contract_it = contracts.find(name);
    if (contract_it == contracts.end()) {
      throw std::runtime_error("unknown fixture contract: " + contract_ref);
    }
    return contract_it->second;
  }
  throw std::runtime_error("invalid binding contract group: " + contract_ref);
}

} // namespace

FixtureBindingResolution fixture_binding_resolution(const FixtureManifest &manifest, FixtureManifestSide side) {
  const auto *route = route_for_manifest(manifest, side);
  if (route->untracked) {
    return {"", {}, true};
  }
  return {route->contract, binding_names_for_contract(route->contract), false};
}

const gpu::ShaderInterface *shader_interface_for_fixture(const FixtureManifest &manifest, FixtureManifestSide side) {
  const auto *route = route_for_manifest(manifest, side);
  if (route->untracked) {
    return nullptr;
  }

  const auto [group, name] = split_contract_ref(route->contract);
  if (group != "shaders") {
    return nullptr;
  }
  return &gpu::shader_interface(name);
}

bool fixture_is_intentionally_untracked(const FixtureManifest &manifest) {
  const auto &contracts = binding_contracts();

  // NOLINTNEXTLINE(readability-use-anyofallof)
  for (const auto &route : contracts.untracked_routes) {
    if (route.subgraph != manifest.subgraph) {
      continue;
    }
    if (!route.stage_name.empty() && route.stage_name == manifest.stage_name) {
      return true;
    }
    if (!route.stage_name_prefix.empty() && has_prefix(manifest.stage_name, route.stage_name_prefix)) {
      return true;
    }
  }
  return false;
}

} // namespace nlrc::vksplat::tests
