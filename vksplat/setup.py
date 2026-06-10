import hashlib
import os
import sys
import platform
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup, Extension

__version__ = "0.1.0"

# Keep in sync with cmake/FetchGLM.cmake
VKSPLAT_GLM_VERSION = "1.0.3"
VKSPLAT_GLM_URL = (
    f"https://github.com/g-truc/glm/releases/download/"
    f"{VKSPLAT_GLM_VERSION}/glm-{VKSPLAT_GLM_VERSION}.zip"
)
VKSPLAT_GLM_SHA256 = "1c0a0fced9b0d87c7b7bc94e40be490cff6d4c83c25db8488d8f33754e7fdeb2"

VKSPLAT_ROOT = Path(__file__).resolve().parent
VKSPLAT_CONTRIB = VKSPLAT_ROOT / "contrib"

SYSTEM_GLM_PATHS = [
    "/usr/include/glm",
    "/usr/local/include/glm",
    "/opt/homebrew/include/glm",  # macOS Homebrew
]

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "vksplat",
        [
            "src/buffer.cpp",
            "src/colmap_reader.cpp", 
            "src/knn.cpp",
            "src/scheduler.cpp",
            "src/gs_pipeline.cpp",
            "src/gs_renderer.cpp",
            "src/gs_trainer.cpp",
            "src/perf_timer.cpp",
            "src/python_bindings.cpp",
        ],
        include_dirs=[
            "src/",
            pybind11.get_include(),
        ],
        language='c++',
        cxx_std=17,
    ),
]

def find_vulkan():
    """Find Vulkan SDK installation"""
    # Check environment variable first
    vulkan_sdk = os.environ.get('VULKAN_SDK')
    if vulkan_sdk and os.path.exists(vulkan_sdk):
        return vulkan_sdk
    
    # Platform-specific default locations
    if platform.system() == "Windows":
        # Windows default installation paths
        possible_paths = [
            "C:/VulkanSDK/*/",
            os.path.expanduser("~/VulkanSDK/*/")
        ]
    elif platform.system() == "Darwin":  # macOS
        possible_paths = [
            "/usr/local/vulkan/macOS/",
            "~/VulkanSDK/macOS/"
        ]
    else:  # Linux
        possible_paths = [
            "/usr/include/vulkan/",
            "/usr/local/include/vulkan/",
            "~/VulkanSDK/*/x86_64/"
        ]
    
    import glob
    for path_pattern in possible_paths:
        expanded = os.path.expanduser(path_pattern)
        matches = glob.glob(expanded)
        if matches:
            return matches[0]
    
    return None

def ensure_glm_contrib():
    """Download and extract GLM into contrib/ if not already present."""
    archive = VKSPLAT_CONTRIB / f"glm-{VKSPLAT_GLM_VERSION}.zip"
    include_dir = VKSPLAT_CONTRIB / "glm"
    marker = include_dir / "glm" / "glm.hpp"

    if marker.exists():
        return str(include_dir)

    VKSPLAT_CONTRIB.mkdir(parents=True, exist_ok=True)

    if not archive.exists():
        print(f"Downloading GLM {VKSPLAT_GLM_VERSION} to {archive}")
        urllib.request.urlretrieve(VKSPLAT_GLM_URL, archive)

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != VKSPLAT_GLM_SHA256:
        raise RuntimeError(
            f"GLM archive hash mismatch: expected {VKSPLAT_GLM_SHA256}, got {digest}"
        )

    if not marker.exists():
        print(f"Extracting GLM to {VKSPLAT_CONTRIB}")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(VKSPLAT_CONTRIB)

    return str(include_dir)


def resolve_glm_include_dir():
    """Resolve GLM include dir: contrib first (matches CMake), then system paths."""
    archive = VKSPLAT_CONTRIB / f"glm-{VKSPLAT_GLM_VERSION}.zip"
    marker = VKSPLAT_CONTRIB / "glm" / "glm" / "glm.hpp"

    if marker.exists() or archive.exists():
        return ensure_glm_contrib()

    if not archive.exists():
        try:
            return ensure_glm_contrib()
        except urllib.error.URLError:
            pass

    for glm_path in SYSTEM_GLM_PATHS:
        glm_header = Path(glm_path) / "glm.hpp"
        if glm_header.exists():
            return str(glm_header.parent.parent)

    raise RuntimeError(
        "GLM not found: place glm-{0}.zip in {1}, install a system GLM, or allow network download".format(
            VKSPLAT_GLM_VERSION, VKSPLAT_CONTRIB
        )
    )

def configure_build():
    """Configure build settings based on platform and available libraries"""

    # Find Vulkan
    vulkan_path = find_vulkan()
    if not vulkan_path:
        raise RuntimeError("Vulkan SDK not found. Please install Vulkan SDK and set VULKAN_SDK environment variable.")
    print(f"Found Vulkan SDK at: {vulkan_path}")
    
    # Configure extension
    ext = ext_modules[0]
    
    # Add Vulkan include and library paths
    if platform.system() == "Windows":
        ext.include_dirs.extend([
            os.path.join(vulkan_path, "Include"),
        ])
        ext.library_dirs = [os.path.join(vulkan_path, "Lib")]
        ext.libraries = ["vulkan-1"]
        ext.define_macros = [("VK_USE_PLATFORM_WIN32_KHR", None)]
    elif platform.system() == "Darwin":  # macOS
        ext.include_dirs.extend([
            os.path.join(vulkan_path, "include"),
        ])
        ext.library_dirs = [os.path.join(vulkan_path, "lib")]
        ext.libraries = ["vulkan"]
        ext.define_macros = [("VK_USE_PLATFORM_MACOS_MVK", None)]
    else:  # Linux
        ext.include_dirs.extend([
            os.path.join(vulkan_path, "include") if "/include/vulkan" not in vulkan_path else vulkan_path.replace("/vulkan", ""),
            # "/usr/include",
            # "/usr/local/include"
        ])
        ext.library_dirs = [
            os.path.join(vulkan_path, "lib") if vulkan_path else "/usr/lib",
            # "/usr/local/lib"
        ]
        ext.libraries = ["vulkan"]
        ext.define_macros = [("VK_USE_PLATFORM_XLIB_KHR", None)]
    
    # Add GLM (header-only library)
    glm_include = resolve_glm_include_dir()
    ext.include_dirs.append(glm_include)
    print(f"Using GLM from: {glm_include}")
    
    # Compiler flags
    if platform.system() != "Windows":
        ext.extra_compile_args = [
            "-Wall",
            "-Wextra",
            "-O3",  # Release optimization
            "-DVERSION_INFO=\"{}\"".format(__version__)
        ]
    else:
        ext.extra_compile_args = [
            "/O2",  # Release optimization  
            "/std:c++17",
            "/DVERSION_INFO=\"{}\"".format(__version__)
        ]
    cflags = os.getenv("CFLAGS")
    if cflags is not None:
        ext.extra_compile_args += cflags.strip().split()

    # Set visibility and optimization
    if platform.system() != "Windows":
        ext.extra_compile_args.extend([
            "-fvisibility=hidden"
        ])


class CustomBuildExt(build_ext):
    """Custom build extension to handle dependencies"""
    
    def build_extensions(self):
        configure_build()
        super().build_extensions()


setup(
    name="vksplat",
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "opencv-python",
        "tqdm",
        "torchmetrics[image]>=1.0.1"  # TODO: possibly get a Vulkan version
    ],
    setup_requires=[
        "pybind11>=2.11.1",
        "setuptools",
    ],
)
