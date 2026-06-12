include(${PROJECT_SOURCE_DIR}/cmake/FetchCLI11.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchNlohmannJson.cmake)

add_executable(vksplat_train
    main.cpp
)

target_link_libraries(vksplat_train PRIVATE
    vksplat_core
    vksplat_warnings
    vksplat_compile_options
    CLI11::CLI11
    nlohmann_json::nlohmann_json
)

target_include_directories(vksplat_train PRIVATE
    ${PROJECT_SOURCE_DIR}/contrib
)

target_compile_definitions(vksplat_train PRIVATE
    VKSPLAT_DEFAULT_SHADER_DIR="${PROJECT_SOURCE_DIR}/shader/"
)

set_target_properties(vksplat_train PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

install(TARGETS vksplat_train DESTINATION bin)
