include_guard(GLOBAL)

include(FetchContent)

function(nlrc_vksplat_fetch_dependency)
  set(options)
  set(one_value_args NAME URL URL_HASH LOCAL_ARCHIVE SOURCE_SUBDIR)
  set(multi_value_args)
  cmake_parse_arguments(NLRC_VKSPLAT_DEP "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT NLRC_VKSPLAT_DEP_NAME)
    message(FATAL_ERROR "nlrc_vksplat_fetch_dependency requires NAME")
  endif()

  if(NOT NLRC_VKSPLAT_DEP_URL_HASH)
    message(FATAL_ERROR "nlrc_vksplat_fetch_dependency(${NLRC_VKSPLAT_DEP_NAME}) requires URL_HASH")
  endif()

  if(NLRC_VKSPLAT_DEP_LOCAL_ARCHIVE AND EXISTS "${NLRC_VKSPLAT_DEP_LOCAL_ARCHIVE}")
    set(_nlrc_vksplat_source_url "${NLRC_VKSPLAT_DEP_LOCAL_ARCHIVE}")
    message(STATUS "Using local archive for ${NLRC_VKSPLAT_DEP_NAME}: ${NLRC_VKSPLAT_DEP_LOCAL_ARCHIVE}")
  elseif(NLRC_VKSPLAT_DEP_URL)
    set(_nlrc_vksplat_source_url "${NLRC_VKSPLAT_DEP_URL}")
    message(STATUS "Fetching ${NLRC_VKSPLAT_DEP_NAME} from ${NLRC_VKSPLAT_DEP_URL}")
  else()
    message(FATAL_ERROR
      "nlrc_vksplat_fetch_dependency(${NLRC_VKSPLAT_DEP_NAME}) requires URL or existing LOCAL_ARCHIVE"
    )
  endif()

  set(_nlrc_vksplat_declare_args
    URL "${_nlrc_vksplat_source_url}"
    URL_HASH "${NLRC_VKSPLAT_DEP_URL_HASH}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )

  if(NLRC_VKSPLAT_DEP_SOURCE_SUBDIR)
    list(APPEND _nlrc_vksplat_declare_args SOURCE_SUBDIR "${NLRC_VKSPLAT_DEP_SOURCE_SUBDIR}")
  endif()

  FetchContent_Declare(
    "${NLRC_VKSPLAT_DEP_NAME}"
    ${_nlrc_vksplat_declare_args}
  )

  FetchContent_MakeAvailable("${NLRC_VKSPLAT_DEP_NAME}")

  FetchContent_GetProperties(
    "${NLRC_VKSPLAT_DEP_NAME}"
    SOURCE_DIR _nlrc_vksplat_source_dir
    BINARY_DIR _nlrc_vksplat_binary_dir
  )

  string(TOLOWER "${NLRC_VKSPLAT_DEP_NAME}" _nlrc_vksplat_dep_name_lower)
  set("${_nlrc_vksplat_dep_name_lower}_SOURCE_DIR" "${_nlrc_vksplat_source_dir}" PARENT_SCOPE)
  set("${_nlrc_vksplat_dep_name_lower}_BINARY_DIR" "${_nlrc_vksplat_binary_dir}" PARENT_SCOPE)
  set("${NLRC_VKSPLAT_DEP_NAME}_SOURCE_DIR" "${_nlrc_vksplat_source_dir}" PARENT_SCOPE)
  set("${NLRC_VKSPLAT_DEP_NAME}_BINARY_DIR" "${_nlrc_vksplat_binary_dir}" PARENT_SCOPE)
endfunction()
