if(NOT DEFINED NLRC_VKSPLAT_COMPILE_COMMANDS_SOURCE OR
   NLRC_VKSPLAT_COMPILE_COMMANDS_SOURCE STREQUAL "")
  message(FATAL_ERROR "NLRC_VKSPLAT_COMPILE_COMMANDS_SOURCE is required")
endif()

if(NOT DEFINED NLRC_VKSPLAT_COMPILE_COMMANDS_DESTINATION OR
   NLRC_VKSPLAT_COMPILE_COMMANDS_DESTINATION STREQUAL "")
  message(FATAL_ERROR "NLRC_VKSPLAT_COMPILE_COMMANDS_DESTINATION is required")
endif()

if(NOT EXISTS "${NLRC_VKSPLAT_COMPILE_COMMANDS_SOURCE}")
  message(STATUS "compile_commands.json not found; skipping sync")
  return()
endif()

get_filename_component(_nlrc_vksplat_compile_commands_dir
                       "${NLRC_VKSPLAT_COMPILE_COMMANDS_DESTINATION}" DIRECTORY)

file(MAKE_DIRECTORY "${_nlrc_vksplat_compile_commands_dir}")

file(COPY_FILE
  "${NLRC_VKSPLAT_COMPILE_COMMANDS_SOURCE}"
  "${NLRC_VKSPLAT_COMPILE_COMMANDS_DESTINATION}"
  ONLY_IF_DIFFERENT
  INPUT_MAY_BE_RECENT
  RESULT _nlrc_vksplat_copy_result
)

if(NOT _nlrc_vksplat_copy_result STREQUAL "0")
  message(FATAL_ERROR "Failed to sync compile_commands.json: ${_nlrc_vksplat_copy_result}")
endif()

message(STATUS "Synced compile_commands.json to ${NLRC_VKSPLAT_COMPILE_COMMANDS_DESTINATION}")
