include_guard(GLOBAL)

include(FetchContent)

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

if(TARGET nlrc_vksplat_shaders)
  add_dependencies(nlrc_vksplat_gpu nlrc_vksplat_shaders)
  target_include_directories(nlrc_vksplat_gpu PRIVATE ${NLRC_VKSPLAT_SHADER_GENERATED_DIR})
  target_compile_definitions(nlrc_vksplat_gpu PUBLIC
    VKSPLAT_USE_EMULATED_INT64=${NLRC_VKSPLAT_USE_EMULATED_INT64_VALUE}
    VKSPLAT_USE_EMULATED_F32_ATOMIC=${NLRC_VKSPLAT_USE_EMULATED_F32_ATOMIC_VALUE}
  )
endif()

function(nlrc_vksplat_setup_tests)
  if(NOT NLRC_VKSPLAT_BUILD_TESTS)
    return()
  endif()

  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(Catch2)

  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)

  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

  add_executable(nlrc_vksplat_tests
    tests/test_main.cpp
    tests/test_fixture_loader.cpp
    tests/test_golden_compare.cpp
    tests/test_gpu_smoke.cpp
    tests/test_manifest_profile.cpp
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
  )

  if(TARGET nlrc_vksplat_shaders)
    add_dependencies(nlrc_vksplat_tests nlrc_vksplat_shaders)
    target_include_directories(nlrc_vksplat_tests PRIVATE ${NLRC_VKSPLAT_SHADER_GENERATED_DIR})
  endif()

  include(Catch)
  catch_discover_tests(nlrc_vksplat_tests DISCOVERY_MODE PRE_TEST)

  add_test(NAME nlrc_vksplat_tests COMMAND nlrc_vksplat_tests)
  set_tests_properties(nlrc_vksplat_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()
