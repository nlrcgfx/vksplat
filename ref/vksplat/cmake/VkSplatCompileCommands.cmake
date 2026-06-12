include_guard(GLOBAL)

function(vksplat_add_compile_commands_sync_target)
  if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
    return()
  endif()

  if(TARGET vksplat_sync_compile_commands)
    return()
  endif()

  set(_vksplat_compile_commands_source "${CMAKE_BINARY_DIR}/compile_commands.json")
  set(_vksplat_compile_commands_destination "${PROJECT_SOURCE_DIR}/build/compile_commands.json")

  set(_vksplat_sync_compile_commands_script
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/VkSplatSyncCompileCommands.cmake")

  add_custom_target(vksplat_sync_compile_commands ALL
    COMMAND
      "${CMAKE_COMMAND}"
      "-DVKSPLAT_COMPILE_COMMANDS_SOURCE=${_vksplat_compile_commands_source}"
      "-DVKSPLAT_COMPILE_COMMANDS_DESTINATION=${_vksplat_compile_commands_destination}"
      -P "${_vksplat_sync_compile_commands_script}"
    COMMENT "Syncing active compile_commands.json for clangd and other tools"
    VERBATIM
  )
endfunction()