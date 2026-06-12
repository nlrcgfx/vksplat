cmake_minimum_required(VERSION 3.28)
project(nlrc_vksplat VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
  add_compile_options(-Qunused-arguments)
endif()

option(NLRC_VKSPLAT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

include(cmake/NlrcVksplatWarnings.cmake)
include(cmake/NlrcVksplatCompileOptions.cmake)
include(cmake/NlrcVksplatShaders.cmake)
include(cmake/NlrcVksplatLint.cmake)

find_package(Vulkan REQUIRED)

add_library(nlrc_vksplat_core STATIC)

target_sources(nlrc_vksplat_core PRIVATE src/version.cpp)
target_include_directories(nlrc_vksplat_core PUBLIC src)

target_link_libraries(nlrc_vksplat_core PUBLIC Vulkan::Vulkan)

target_link_libraries(nlrc_vksplat_core PRIVATE
  nlrc_vksplat_warnings
  nlrc_vksplat_compile_options
)

nlrc_vksplat_link_shaders(nlrc_vksplat_core)
nlrc_vksplat_add_lint_targets(nlrc_vksplat_core)

include(cmake/NlrcVksplatTests.cmake)
if(NLRC_VKSPLAT_BUILD_TESTS)
  include(CTest)
  enable_testing()
endif()
nlrc_vksplat_setup_tests()

# Sync compile commands for clangd and other tools
include(cmake/NlrcVksplatCompileCommands.cmake)
nlrc_vksplat_add_compile_commands_sync_target()
