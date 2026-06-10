include_guard(GLOBAL)

set(VKSPLAT_GLM_VERSION 1.0.3)
set(VKSPLAT_GLM_ARCHIVE "${CMAKE_CURRENT_LIST_DIR}/../contrib/glm-${VKSPLAT_GLM_VERSION}.zip")
set(VKSPLAT_GLM_REMOTE_URL
    "https://github.com/g-truc/glm/releases/download/${VKSPLAT_GLM_VERSION}/glm-${VKSPLAT_GLM_VERSION}.zip"
)
set(VKSPLAT_GLM_SHA256 "SHA256=1c0a0fced9b0d87c7b7bc94e40be490cff6d4c83c25db8488d8f33754e7fdeb2")

vksplat_fetch_dependency(
    NAME glm
    URL "${VKSPLAT_GLM_REMOTE_URL}"
    LOCAL_ARCHIVE "${VKSPLAT_GLM_ARCHIVE}"
    URL_HASH "${VKSPLAT_GLM_SHA256}"
)
