include_guard(GLOBAL)

include(FetchContent)

function(vksplat_fetch_dependency)
  set(options)
  set(one_value_args NAME URL URL_HASH LOCAL_ARCHIVE SOURCE_SUBDIR)
  set(multi_value_args)
  cmake_parse_arguments(VKSPLAT_DEP "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT VKSPLAT_DEP_NAME)
    message(FATAL_ERROR "vksplat_fetch_dependency requires NAME")
  endif()

  if(NOT VKSPLAT_DEP_URL_HASH)
    message(FATAL_ERROR "vksplat_fetch_dependency(${VKSPLAT_DEP_NAME}) requires URL_HASH")
  endif()

  if(VKSPLAT_DEP_LOCAL_ARCHIVE AND EXISTS "${VKSPLAT_DEP_LOCAL_ARCHIVE}")
    set(_vksplat_source_url "${VKSPLAT_DEP_LOCAL_ARCHIVE}")
    message(STATUS "Using local archive for ${VKSPLAT_DEP_NAME}: ${VKSPLAT_DEP_LOCAL_ARCHIVE}")
  elseif(VKSPLAT_DEP_URL)
    set(_vksplat_source_url "${VKSPLAT_DEP_URL}")
    message(STATUS "Fetching ${VKSPLAT_DEP_NAME} from ${VKSPLAT_DEP_URL}")
  else()
    message(FATAL_ERROR "vksplat_fetch_dependency(${VKSPLAT_DEP_NAME}) requires URL or existing LOCAL_ARCHIVE")
  endif()

  set(_vksplat_declare_args
    URL "${_vksplat_source_url}"
    URL_HASH "${VKSPLAT_DEP_URL_HASH}"
  )
  if(VKSPLAT_DEP_SOURCE_SUBDIR)
    list(APPEND _vksplat_declare_args SOURCE_SUBDIR "${VKSPLAT_DEP_SOURCE_SUBDIR}")
  endif()

  FetchContent_Declare(
    "${VKSPLAT_DEP_NAME}"
    ${_vksplat_declare_args}
  )

  FetchContent_MakeAvailable("${VKSPLAT_DEP_NAME}")
endfunction()
