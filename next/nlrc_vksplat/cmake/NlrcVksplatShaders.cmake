include_guard(GLOBAL)

option(NLRC_VKSPLAT_EMULATE_INT64 "Compile C++ and shaders for emulated int64 tile bounds" OFF)
option(NLRC_VKSPLAT_EMULATE_F32_ATOMIC "Compile C++ and shaders for emulated f32 atomics" OFF)

set(NLRC_VKSPLAT_USE_EMULATED_INT64_VALUE 0)
if(NLRC_VKSPLAT_EMULATE_INT64)
  set(NLRC_VKSPLAT_USE_EMULATED_INT64_VALUE 1)
endif()

set(NLRC_VKSPLAT_USE_EMULATED_F32_ATOMIC_VALUE 0)
if(NLRC_VKSPLAT_EMULATE_F32_ATOMIC)
  set(NLRC_VKSPLAT_USE_EMULATED_F32_ATOMIC_VALUE 1)
endif()

find_package(Python COMPONENTS Interpreter REQUIRED)
find_program(NLRC_VKSPLAT_SLANGC slangc REQUIRED)
find_program(NLRC_VKSPLAT_GLSLC glslc REQUIRED)

set(NLRC_VKSPLAT_SHADER_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")

set(NLRC_VKSPLAT_SHADER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/compile_shaders.py")
set(NLRC_VKSPLAT_CONFIG_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/nlrc_vksplat_config.hpp")
set(NLRC_VKSPLAT_SHADER_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_shader_config.py")
set(NLRC_VKSPLAT_SMOKE_SLANG "${CMAKE_CURRENT_SOURCE_DIR}/slang/smoke.slang")

set(NLRC_VKSPLAT_SHADER_OUTPUTS
  "${NLRC_VKSPLAT_SHADER_GENERATED_DIR}/config_generated.slang"
  "${NLRC_VKSPLAT_SHADER_GENERATED_DIR}/config_generated.glsl"
  "${NLRC_VKSPLAT_SHADER_GENERATED_DIR}/shader_config.json"
  "${NLRC_VKSPLAT_SHADER_GENERATED_DIR}/shader_manifest.json"
  "${NLRC_VKSPLAT_SHADER_GENERATED_DIR}/smoke_spirv.hpp"
)

add_custom_command(
  OUTPUT ${NLRC_VKSPLAT_SHADER_OUTPUTS}
  COMMAND
    ${Python_EXECUTABLE}
    ${NLRC_VKSPLAT_SHADER_SCRIPT}
    --project-dir ${CMAKE_CURRENT_SOURCE_DIR}
    --generated-dir ${NLRC_VKSPLAT_SHADER_GENERATED_DIR}
    --emulate-int64 ${NLRC_VKSPLAT_USE_EMULATED_INT64_VALUE}
    --emulate-f32-atomic ${NLRC_VKSPLAT_USE_EMULATED_F32_ATOMIC_VALUE}
    --slangc ${NLRC_VKSPLAT_SLANGC}
    --glslc ${NLRC_VKSPLAT_GLSLC}
  DEPENDS
    ${NLRC_VKSPLAT_SHADER_SCRIPT}
    ${NLRC_VKSPLAT_SHADER_GENERATOR}
    ${NLRC_VKSPLAT_CONFIG_HEADER}
    ${NLRC_VKSPLAT_SMOKE_SLANG}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating shader config and embedded SPIR-V headers"
  VERBATIM
)

add_custom_target(nlrc_vksplat_shaders DEPENDS ${NLRC_VKSPLAT_SHADER_OUTPUTS})

function(nlrc_vksplat_link_shaders target)
  if(NOT TARGET nlrc_vksplat_shaders)
    message(FATAL_ERROR "nlrc_vksplat_shaders target is not defined")
  endif()

  add_dependencies(${target} nlrc_vksplat_shaders)
  target_include_directories(${target} PRIVATE ${NLRC_VKSPLAT_SHADER_GENERATED_DIR})
  target_compile_definitions(${target} PUBLIC
    VKSPLAT_USE_EMULATED_INT64=${NLRC_VKSPLAT_USE_EMULATED_INT64_VALUE}
    VKSPLAT_USE_EMULATED_F32_ATOMIC=${NLRC_VKSPLAT_USE_EMULATED_F32_ATOMIC_VALUE}
  )
endfunction()
