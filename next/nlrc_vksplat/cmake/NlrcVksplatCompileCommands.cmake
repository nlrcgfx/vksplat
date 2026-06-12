include_guard(GLOBAL)

function(nlrc_vksplat_add_compile_commands_sync_target)
  if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
    return()
  endif()

  if(TARGET nlrc_vksplat_sync_compile_commands)
    return()
  endif()

  set(_nlrc_vksplat_compile_commands_source "${CMAKE_BINARY_DIR}/compile_commands.json")
  set(_nlrc_vksplat_compile_commands_destination "${PROJECT_SOURCE_DIR}/build/compile_commands.json")

  set(_nlrc_vksplat_sync_compile_commands_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/NlrcVksplatSyncCompileCommands.cmake")

  add_custom_target(nlrc_vksplat_sync_compile_commands ALL
    COMMAND
      "${CMAKE_COMMAND}"
      "-DNLRC_VKSPLAT_COMPILE_COMMANDS_SOURCE=${_nlrc_vksplat_compile_commands_source}"
      "-DNLRC_VKSPLAT_COMPILE_COMMANDS_DESTINATION=${_nlrc_vksplat_compile_commands_destination}"
      -P "${_nlrc_vksplat_sync_compile_commands_script}"
    COMMENT "Syncing active compile_commands.json for clangd and other tools"
    VERBATIM
  )
endfunction()
