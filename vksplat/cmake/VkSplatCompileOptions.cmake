include_guard(GLOBAL)

add_library(vksplat_compile_options INTERFACE)

target_compile_options(vksplat_compile_options INTERFACE
    # All configs — MSVC standards / large TUs
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/permissive-;/Zc:__cplusplus;/bigobj>>"

    # Debug — MSVC
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:MSVC>>:/Od;/Zi;/RTC1>>"

    # Debug — GCC/Clang
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<AND:$<CONFIG:Debug>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Og;-fno-omit-frame-pointer>>"

    # Release — MSVC
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/O2>>"

    # Release — GCC/Clang
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O3>>"
)

# Sanitizer placeholders (not enabled by default):
# MSVC Debug:   /fsanitize=address
# GCC/Clang Debug: -fsanitize=address -fno-omit-frame-pointer -fno-common
