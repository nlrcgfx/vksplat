cmake_minimum_required(VERSION 3.28)
project(nlrc_vksplat VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NLRC_VKSPLAT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

if(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
  # Suppress -gcc-toolchain warnings from OHOS SDK
  add_compile_options(-Qunused-arguments)
endif()

include(cmake/NlrcVksplatWarnings.cmake)
include(cmake/NlrcVksplatCompileOptions.cmake)

# Fetch dependencies: Catch2, nlohmann_json
include(cmake/NlrcVksplatFetchDependency.cmake)

# Shader build and embedded SPIR-V headers
include(cmake/NlrcVksplatShaders.cmake)

find_package(Vulkan REQUIRED)

add_library(nlrc_vksplat_core STATIC)

target_sources(nlrc_vksplat_core
  PRIVATE
    src/gpu/compute_pipeline.cpp
    src/gpu/headless_context.cpp
    src/gpu/shader_execution.cpp
    src/gpu/storage_buffer.cpp
    src/vulkan_check.hpp
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS src
    FILES
      src/gpu/constants.hpp
      src/gpu/compute_pipeline.hpp
      src/gpu/headless_context.hpp
      src/gpu/shader_descriptors.hpp
      src/gpu/shader_execution.hpp
      src/gpu/storage_buffer.hpp
      src/nlrc_vksplat_config.hpp
      src/span.hpp
)

target_link_libraries(nlrc_vksplat_core PUBLIC Vulkan::Vulkan)

target_link_libraries(nlrc_vksplat_core PRIVATE
  nlrc_vksplat_warnings
  nlrc_vksplat_compile_options
)

# Apply shader profile to the core library
nlrc_vksplat_apply_shader_profile(nlrc_vksplat_core)

if(TARGET nlrc_vksplat_shaders)
  add_dependencies(nlrc_vksplat_core nlrc_vksplat_shaders)
  target_include_directories(nlrc_vksplat_core PRIVATE ${NLRC_VKSPLAT_SHADER_GENERATED_DIR})
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
  set(_NLRC_VKSPLAT_DEFAULT_BUILD_TESTING OFF)
else()
  set(_NLRC_VKSPLAT_DEFAULT_BUILD_TESTING ON)
endif()

option(BUILD_TESTING "Build tests" ${_NLRC_VKSPLAT_DEFAULT_BUILD_TESTING})
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Sync compile commands for clangd and other tools
include(cmake/NlrcVksplatCompileCommands.cmake)
nlrc_vksplat_add_compile_commands_sync_target()
