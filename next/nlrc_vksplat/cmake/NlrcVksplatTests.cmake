include_guard(GLOBAL)

include(FetchContent)
include("${CMAKE_CURRENT_LIST_DIR}/NlrcVksplatFetchDependency.cmake")

if(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
  set(_NLRC_VKSPLAT_DEFAULT_BUILD_TESTS OFF)
else()
  set(_NLRC_VKSPLAT_DEFAULT_BUILD_TESTS ON)
endif()

option(NLRC_VKSPLAT_BUILD_TESTS "Build Catch2 tests" ${_NLRC_VKSPLAT_DEFAULT_BUILD_TESTS})
set(NLRC_VKSPLAT_GPU_TESTS "AUTO" CACHE STRING "GPU test policy: AUTO, REQUIRE, or OFF")
set_property(CACHE NLRC_VKSPLAT_GPU_TESTS PROPERTY STRINGS AUTO REQUIRE OFF)

set(NLRC_VKSPLAT_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../test_data" CACHE PATH
    "Root directory for fixture and golden test data")
set(NLRC_VKSPLAT_DEP_ARCHIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/contrib" CACHE PATH
    "Directory containing optional local dependency archives")

set(_NLRC_VKSPLAT_GPU_POLICY_VALUE 0)
if(NLRC_VKSPLAT_GPU_TESTS STREQUAL "REQUIRE")
  set(_NLRC_VKSPLAT_GPU_POLICY_VALUE 1)
elseif(NLRC_VKSPLAT_GPU_TESTS STREQUAL "OFF")
  set(_NLRC_VKSPLAT_GPU_POLICY_VALUE 2)
endif()

add_library(nlrc_vksplat_gpu STATIC
  src/gpu/headless_context.cpp
  src/gpu/storage_buffer.cpp
  src/gpu/compute_pipeline.cpp
)

target_include_directories(nlrc_vksplat_gpu PUBLIC src)
target_link_libraries(nlrc_vksplat_gpu PUBLIC Vulkan::Vulkan)

target_link_libraries(nlrc_vksplat_gpu PRIVATE
  nlrc_vksplat_warnings
  nlrc_vksplat_compile_options
)

nlrc_vksplat_apply_shader_profile(nlrc_vksplat_gpu)

function(nlrc_vksplat_setup_tests)
  if(NOT NLRC_VKSPLAT_BUILD_TESTS)
    return()
  endif()

  set(NLRC_VKSPLAT_CATCH2_VERSION 3.7.1)
  set(NLRC_VKSPLAT_CATCH2_ARCHIVE
      "${NLRC_VKSPLAT_DEP_ARCHIVE_DIR}/Catch2-v${NLRC_VKSPLAT_CATCH2_VERSION}.tar.gz"
  )
  set(NLRC_VKSPLAT_CATCH2_REMOTE_URL
      "https://github.com/catchorg/Catch2/archive/refs/tags/v${NLRC_VKSPLAT_CATCH2_VERSION}.tar.gz"
  )
  set(NLRC_VKSPLAT_CATCH2_SHA256 "SHA256=c991b247a1a0d7bb9c39aa35faf0fe9e19764213f28ffba3109388e62ee0269c")

  nlrc_vksplat_fetch_dependency(
    NAME Catch2
    URL "${NLRC_VKSPLAT_CATCH2_REMOTE_URL}"
    LOCAL_ARCHIVE "${NLRC_VKSPLAT_CATCH2_ARCHIVE}"
    URL_HASH "${NLRC_VKSPLAT_CATCH2_SHA256}"
  )

  set(JSON_BuildTests OFF CACHE INTERNAL "")
  set(NLRC_VKSPLAT_NLOHMANN_JSON_VERSION 3.11.3)
  set(NLRC_VKSPLAT_NLOHMANN_JSON_ARCHIVE
      "${NLRC_VKSPLAT_DEP_ARCHIVE_DIR}/json-v${NLRC_VKSPLAT_NLOHMANN_JSON_VERSION}.tar.gz"
  )
  set(NLRC_VKSPLAT_NLOHMANN_JSON_REMOTE_URL
      "https://github.com/nlohmann/json/archive/refs/tags/v${NLRC_VKSPLAT_NLOHMANN_JSON_VERSION}.tar.gz"
  )
  set(NLRC_VKSPLAT_NLOHMANN_JSON_SHA256
      "SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406"
  )

  nlrc_vksplat_fetch_dependency(
    NAME nlohmann_json
    URL "${NLRC_VKSPLAT_NLOHMANN_JSON_REMOTE_URL}"
    LOCAL_ARCHIVE "${NLRC_VKSPLAT_NLOHMANN_JSON_ARCHIVE}"
    URL_HASH "${NLRC_VKSPLAT_NLOHMANN_JSON_SHA256}"
  )

  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

  add_executable(nlrc_vksplat_tests
    tests/test_main.cpp
    tests/test_fixture_loader.cpp
    tests/test_golden_compare.cpp
    tests/test_gpu_smoke.cpp
    tests/test_manifest_profile.cpp
    tests/test_shader_config_profile.cpp
    tests/support/fixture_manifest.cpp
    tests/support/fixture_loader.cpp
    tests/support/golden_compare.cpp
    tests/support/gpu_available.cpp
    tests/support/manifest_profile.cpp
  )

  target_include_directories(nlrc_vksplat_tests PRIVATE tests/support)
  target_link_libraries(nlrc_vksplat_tests PRIVATE
    nlrc_vksplat_gpu
    Catch2::Catch2WithMain
    nlohmann_json::nlohmann_json
  )

  target_compile_definitions(nlrc_vksplat_tests PRIVATE
    NLRC_VKSPLAT_TEST_DATA_DIR="${NLRC_VKSPLAT_TEST_DATA_DIR}"
    NLRC_VKSPLAT_GPU_TEST_POLICY=${_NLRC_VKSPLAT_GPU_POLICY_VALUE}
    NLRC_VKSPLAT_SHADER_CONFIG_JSON="${NLRC_VKSPLAT_SHADER_GENERATED_DIR}/shader_config.json"
    NLRC_VKSPLAT_SHADER_GENERATED_DIR="${NLRC_VKSPLAT_SHADER_GENERATED_DIR}"
  )

  if(TARGET nlrc_vksplat_shaders)
    nlrc_vksplat_link_shaders(nlrc_vksplat_tests)
  endif()

  include(Catch)
  catch_discover_tests(nlrc_vksplat_tests DISCOVERY_MODE PRE_TEST)

  add_test(NAME nlrc_vksplat_tests COMMAND nlrc_vksplat_tests)
  set_tests_properties(nlrc_vksplat_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()
