include_guard(GLOBAL)

add_library(nlrc_vksplat_compile_options INTERFACE)

target_compile_options(nlrc_vksplat_compile_options INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/permissive-;/Zc:__cplusplus;/bigobj>>"

    "$<$<COMPILE_LANGUAGE:CXX>:$<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:MSVC>>:/Od;/Zi;/RTC1>>"

    "$<$<COMPILE_LANGUAGE:CXX>:$<$<AND:$<CONFIG:Debug>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Og;-fno-omit-frame-pointer>>"

    "$<$<COMPILE_LANGUAGE:CXX>:$<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/O2>>"

    "$<$<COMPILE_LANGUAGE:CXX>:$<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O3>>"
)
