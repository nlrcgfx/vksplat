include_guard(GLOBAL)

add_library(nlrc_vksplat_warnings INTERFACE)

target_compile_options(nlrc_vksplat_warnings INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/W4>>"
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>,$<BOOL:${NLRC_VKSPLAT_WARNINGS_AS_ERRORS}>>:/WX>"
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wnon-virtual-dtor;-Wold-style-cast;-Woverloaded-virtual;-Wformat=2>>"
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<BOOL:${NLRC_VKSPLAT_WARNINGS_AS_ERRORS}>>:-Werror>"
)
