if(NOT DEFINED VKSPLAT_COMPILE_COMMANDS_SOURCE OR
   VKSPLAT_COMPILE_COMMANDS_SOURCE STREQUAL "")
  message(FATAL_ERROR "VKSPLAT_COMPILE_COMMANDS_SOURCE is required")
endif()

if(NOT DEFINED VKSPLAT_COMPILE_COMMANDS_DESTINATION OR
   VKSPLAT_COMPILE_COMMANDS_DESTINATION STREQUAL "")
  message(FATAL_ERROR "VKSPLAT_COMPILE_COMMANDS_DESTINATION is required")
endif()

if(NOT EXISTS "${VKSPLAT_COMPILE_COMMANDS_SOURCE}")
  message(STATUS "compile_commands.json not found; skipping sync")
  return()
endif()

get_filename_component(_vksplat_compile_commands_dir
                       "${VKSPLAT_COMPILE_COMMANDS_DESTINATION}" DIRECTORY)

file(MAKE_DIRECTORY "${_vksplat_compile_commands_dir}")

file(COPY_FILE
  "${VKSPLAT_COMPILE_COMMANDS_SOURCE}"
  "${VKSPLAT_COMPILE_COMMANDS_DESTINATION}"
  ONLY_IF_DIFFERENT
  INPUT_MAY_BE_RECENT
  RESULT _vksplat_copy_result
)

if(NOT _vksplat_copy_result STREQUAL "0")
  message(FATAL_ERROR "Failed to sync compile_commands.json: ${_vksplat_copy_result}")
endif()

message(STATUS "Synced compile_commands.json to ${VKSPLAT_COMPILE_COMMANDS_DESTINATION}")
