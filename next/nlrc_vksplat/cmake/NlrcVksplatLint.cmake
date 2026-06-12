include_guard(GLOBAL)

find_program(NLRC_VKSPLAT_CLANG_FORMAT clang-format REQUIRED)
find_program(NLRC_VKSPLAT_CLANG_TIDY clang-tidy REQUIRED)

set(NLRC_VKSPLAT_LINT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/lint_cpp.py")

function(nlrc_vksplat_add_lint_targets core_target)
  if(NOT TARGET nlrc_vksplat_lint)
    add_custom_target(nlrc_vksplat_lint
      COMMAND
        ${Python_EXECUTABLE}
        ${NLRC_VKSPLAT_LINT_SCRIPT}
        --project-dir ${CMAKE_CURRENT_SOURCE_DIR}
        --build-dir ${CMAKE_BINARY_DIR}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Linting rewrite C++ sources (clang-format + clang-tidy)"
      VERBATIM
    )
  endif()

  if(TARGET nlrc_vksplat_shaders)
    add_dependencies(nlrc_vksplat_lint nlrc_vksplat_shaders)
  endif()

  if(TARGET ${core_target})
    add_dependencies(${core_target} nlrc_vksplat_lint)
  else()
    message(FATAL_ERROR "nlrc_vksplat_add_lint_targets: target ${core_target} does not exist")
  endif()

  if(TARGET nlrc_vksplat_gpu)
    add_dependencies(nlrc_vksplat_gpu nlrc_vksplat_lint)
  endif()

  if(TARGET nlrc_vksplat_tests)
    add_dependencies(nlrc_vksplat_tests nlrc_vksplat_lint)
  endif()
endfunction()
