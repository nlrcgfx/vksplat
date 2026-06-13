
set(NLRC_VKSPLAT_GPU_TESTS "AUTO" CACHE STRING "GPU test policy: AUTO, REQUIRE, or OFF")
set_property(CACHE NLRC_VKSPLAT_GPU_TESTS PROPERTY STRINGS AUTO REQUIRE OFF)

set(NLRC_VKSPLAT_TEST_DATA_DIR "${PROJECT_SOURCE_DIR}/../../test_data" CACHE PATH
    "Root directory for fixture and golden test data")

set(NLRC_VKSPLAT_FIXTURE_GENERATOR "${NLRC_VKSPLAT_TEST_DATA_DIR}/generate_fixtures.py")
set(NLRC_VKSPLAT_BINDING_CONTRACTS "${NLRC_VKSPLAT_TEST_DATA_DIR}/shader_binding_contracts.json")
set(NLRC_VKSPLAT_BINDING_CONTRACT_EXPORTER
    "${PROJECT_SOURCE_DIR}/scripts/export_shader_binding_contracts.py")

set(NLRC_VKSPLAT_DEP_ARCHIVE_DIR "${PROJECT_SOURCE_DIR}/contrib" CACHE PATH
    "Directory containing optional local dependency archives")

set(_NLRC_VKSPLAT_GPU_POLICY_VALUE 0)
if(NLRC_VKSPLAT_GPU_TESTS STREQUAL "REQUIRE")
  set(_NLRC_VKSPLAT_GPU_POLICY_VALUE 1)
elseif(NLRC_VKSPLAT_GPU_TESTS STREQUAL "OFF")
  set(_NLRC_VKSPLAT_GPU_POLICY_VALUE 2)
endif()

set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)

set(NLRC_VKSPLAT_CATCH2_VERSION 3.7.1)
set(NLRC_VKSPLAT_CATCH2_ARCHIVE "${NLRC_VKSPLAT_DEP_ARCHIVE_DIR}/Catch2-v${NLRC_VKSPLAT_CATCH2_VERSION}.tar.gz")
set(NLRC_VKSPLAT_CATCH2_REMOTE_URL "https://github.com/catchorg/Catch2/archive/refs/tags/v${NLRC_VKSPLAT_CATCH2_VERSION}.tar.gz")
set(NLRC_VKSPLAT_CATCH2_SHA256 "SHA256=c991b247a1a0d7bb9c39aa35faf0fe9e19764213f28ffba3109388e62ee0269c")

nlrc_vksplat_fetch_dependency(
  NAME Catch2
  URL "${NLRC_VKSPLAT_CATCH2_REMOTE_URL}"
  LOCAL_ARCHIVE "${NLRC_VKSPLAT_CATCH2_ARCHIVE}"
  URL_HASH "${NLRC_VKSPLAT_CATCH2_SHA256}"
)

set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)

set(NLRC_VKSPLAT_NLOHMANN_JSON_VERSION 3.11.3)
set(NLRC_VKSPLAT_NLOHMANN_JSON_ARCHIVE "${NLRC_VKSPLAT_DEP_ARCHIVE_DIR}/json-v${NLRC_VKSPLAT_NLOHMANN_JSON_VERSION}.tar.gz")

set(NLRC_VKSPLAT_NLOHMANN_JSON_REMOTE_URL
    "https://github.com/nlohmann/json/archive/refs/tags/v${NLRC_VKSPLAT_NLOHMANN_JSON_VERSION}.tar.gz")

set(NLRC_VKSPLAT_NLOHMANN_JSON_SHA256
    "SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406")

nlrc_vksplat_fetch_dependency(
  NAME nlohmann_json
  URL "${NLRC_VKSPLAT_NLOHMANN_JSON_REMOTE_URL}"
  LOCAL_ARCHIVE "${NLRC_VKSPLAT_NLOHMANN_JSON_ARCHIVE}"
  URL_HASH "${NLRC_VKSPLAT_NLOHMANN_JSON_SHA256}"
)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_library(nlrc_vksplat_test_support STATIC
  support/fixture_manifest.cpp
  support/fixture_loader.cpp
  support/fixtures.cpp
  support/golden_compare.cpp
  support/gpu_available.cpp
  support/manifest_profile.cpp
  support/shader_fixture_mapping.cpp
)

target_include_directories(nlrc_vksplat_test_support PUBLIC support)

target_link_libraries(nlrc_vksplat_test_support
  PUBLIC
    nlrc_vksplat_core
  PRIVATE
    Catch2::Catch2
    nlohmann_json::nlohmann_json
)

target_compile_definitions(nlrc_vksplat_test_support PUBLIC
  NLRC_VKSPLAT_TEST_DATA_DIR="${NLRC_VKSPLAT_TEST_DATA_DIR}"
  NLRC_VKSPLAT_GPU_TEST_POLICY=${_NLRC_VKSPLAT_GPU_POLICY_VALUE}
)

add_executable(nlrc_vksplat_shader_binding_registry_export
  export_shader_binding_registry.cpp
)

target_link_libraries(nlrc_vksplat_shader_binding_registry_export PRIVATE
  nlrc_vksplat_core
  nlohmann_json::nlohmann_json
)

add_executable(nlrc_vksplat_tests)

target_sources(nlrc_vksplat_tests
  PRIVATE
    test_main.cpp
    test_compute_tile_ranges.cpp
    test_fixture_catalog.cpp
    test_fixture_loader.cpp
    test_forward.cpp
    test_generate_keys.cpp
    test_golden_compare.cpp
    test_gpu_smoke.cpp
    test_manifest_profile.cpp
    test_projection_forward.cpp
    test_radix_sort.cpp
    test_rasterize_forward.cpp
    test_shader_descriptors.cpp
    test_shader_config_profile.cpp
    test_utility_shaders.cpp
)

target_link_libraries(nlrc_vksplat_tests PRIVATE
  nlrc_vksplat_core
  nlrc_vksplat_test_support
  Catch2::Catch2WithMain
  nlohmann_json::nlohmann_json
)

target_compile_definitions(nlrc_vksplat_tests PRIVATE
  NLRC_VKSPLAT_SHADER_CONFIG_JSON="${NLRC_VKSPLAT_SHADER_GENERATED_DIR}/shader_config.json"
  NLRC_VKSPLAT_SHADER_GENERATED_DIR="${NLRC_VKSPLAT_SHADER_GENERATED_DIR}"
)

if(TARGET nlrc_vksplat_shaders)
  nlrc_vksplat_link_shaders(nlrc_vksplat_tests)
endif()

include(Catch)
catch_discover_tests(nlrc_vksplat_tests DISCOVERY_MODE PRE_TEST)

add_test(
  NAME nlrc_vksplat_fixture_generation_check
  COMMAND ${Python_EXECUTABLE} ${NLRC_VKSPLAT_FIXTURE_GENERATOR} --check
)
set_tests_properties(nlrc_vksplat_fixture_generation_check PROPERTIES
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_test(
  NAME nlrc_vksplat_shader_binding_contracts_check
  COMMAND
    ${Python_EXECUTABLE}
    ${NLRC_VKSPLAT_BINDING_CONTRACT_EXPORTER}
    --check
    --binding-contracts ${NLRC_VKSPLAT_BINDING_CONTRACTS}
    --registry-exporter $<TARGET_FILE:nlrc_vksplat_shader_binding_registry_export>
)
set_tests_properties(nlrc_vksplat_shader_binding_contracts_check PROPERTIES
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
