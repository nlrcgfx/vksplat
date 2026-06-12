include_guard(GLOBAL)

add_library(vksplat_warnings INTERFACE)

target_compile_options(vksplat_warnings INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/W4>>"
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall;-Wextra>>"
)
