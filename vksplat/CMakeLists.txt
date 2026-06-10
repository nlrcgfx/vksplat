cmake_minimum_required(VERSION 3.28)
project(VkSplat VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(VKSPLAT_BUILD_PYTHON "Build pybind11 Python module" ON)

if(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
    add_compile_options(-Qunused-arguments)
endif()

include(cmake/VkSplatWarnings.cmake)
include(cmake/VkSplatCompileOptions.cmake)

include(cmake/FetchDependency.cmake)

include(cmake/FetchGLM.cmake)
include(cmake/FetchVulkanHeaders.cmake)

find_package(Vulkan REQUIRED)

set(VKSPLAT_CORE_SOURCES
    src/buffer.cpp
    src/colmap_reader.cpp
    src/knn.cpp
    src/scheduler.cpp
    src/gs_pipeline.cpp
    src/gs_renderer.cpp
    src/gs_trainer.cpp
    src/perf_timer.cpp
)

set(VKSPLAT_CORE_HEADERS
    src/config.h
    src/buffer.h
    src/colmap_reader.h
    src/knn.h
    src/scheduler.h
    src/gs_pipeline.h
    src/gs_renderer.h
    src/gs_trainer.h
    src/perf_timer.h
)

add_library(vksplat_core STATIC ${VKSPLAT_CORE_SOURCES} ${VKSPLAT_CORE_HEADERS})

target_link_libraries(vksplat_core PRIVATE
    vksplat_warnings
    vksplat_compile_options
)

target_include_directories(vksplat_core PUBLIC src)

target_link_libraries(vksplat_core PUBLIC
    Vulkan::Headers
    Vulkan::Vulkan
    glm::glm
)

set_target_properties(vksplat_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

if(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
    target_compile_definitions(vksplat_core PUBLIC VK_USE_PLATFORM_OHOS_KHR)
elseif(WIN32)
    target_compile_definitions(vksplat_core PUBLIC VK_USE_PLATFORM_WIN32_KHR)
elseif(UNIX AND NOT APPLE)
    target_compile_definitions(vksplat_core PUBLIC VK_USE_PLATFORM_XLIB_KHR)
elseif(APPLE)
    target_compile_definitions(vksplat_core PUBLIC VK_USE_PLATFORM_MACOS_MVK)
endif()

if(VKSPLAT_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)

    # Find pybind11
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        # If pybind11 is not found, download it
        message(STATUS "Downloading pybind11")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11
            GIT_TAG        v2.11.1
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(vksplat MODULE src/python_bindings.cpp)

    target_link_libraries(vksplat PRIVATE
        vksplat_core
        vksplat_warnings
        vksplat_compile_options
    )

    set_target_properties(vksplat PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )

    install(TARGETS vksplat DESTINATION .)
endif()

include(cmake/VkSplatCompileCommands.cmake)
vksplat_add_compile_commands_sync_target()
