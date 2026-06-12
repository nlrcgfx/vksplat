#!/usr/bin/env python3
"""Compile rewrite shaders to embedded SPIR-V C++ headers."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_shader_config import generate_shader_config  # noqa: E402


@dataclass
class CompileConfig:
    project_dir: Path
    build_dir: Path
    slang_src_path: Path
    shader_dst_path: Path
    generated_dir: Path
    config_header_path: Path
    slangc_compile_args: list[str] = field(
        default_factory=lambda: ["-stage", "compute", "-O", "-fp-mode", "fast", "-line-directive-mode", "none"]
    )
    glslc_compile_args: list[str] = field(default_factory=lambda: ["-O", "--target-spv=spv1.5", "--target-env=vulkan1.2"])
    slangc_path: Path | None = None
    glslc_path: Path | None = None
    emulate_int64: int | None = None
    emulate_f32_atomic: int | None = None


@dataclass
class ShaderJob:
    name: str
    defines: dict[str, int] = field(default_factory=dict)


@dataclass
class ShaderSource:
    source: str
    language: str
    jobs: list[ShaderJob] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)


def spirv_symbols(job_name: str) -> tuple[str, str]:
    """Map a snake_case job name to clang-tidy-compliant embed symbol names."""
    parts = [part for part in job_name.split("_") if part]
    if not parts:
        raise ValueError("shader job name must not be empty")
    camel = "".join(part[:1].upper() + part[1:] for part in parts)
    return f"k{camel}Spirv", f"k{camel}SpirvWordCount"


def spirv_to_header(spirv_bytes: bytes, job_name: str) -> str:
    array_symbol, word_count_symbol = spirv_symbols(job_name)
    if len(spirv_bytes) % 4 != 0:
        raise ValueError(f"SPIR-V size for {job_name} is not a multiple of 4 bytes")
    words = [
        int.from_bytes(spirv_bytes[i : i + 4], byteorder="little")
        for i in range(0, len(spirv_bytes), 4)
    ]
    body = ",\n    ".join(f"0x{word:08x}u" for word in words)
    return "\n".join(
        [
            "#pragma once",
            "",
            "#include <cstdint>",
            "",
            "namespace nlrc::vksplat::shaders {",
            "",
            f"inline constexpr uint32_t {array_symbol}[] = {{",
            f"    {body}",
            "};",
            "",
            f"inline constexpr uint32_t {word_count_symbol} = {len(words)};",
            "",
            "}  // namespace nlrc::vksplat::shaders",
            "",
        ]
    )


def _find_clang_format() -> Path:
    clang_format = shutil.which("clang-format")
    if clang_format is None:
        raise RuntimeError("clang-format not found. Install LLVM clang-format or add it to PATH")
    return Path(clang_format)


def _find_clang_tidy() -> Path:
    clang_tidy = shutil.which("clang-tidy")
    if clang_tidy is None:
        raise RuntimeError("clang-tidy not found. Install LLVM clang-tidy or add it to PATH")
    return Path(clang_tidy)


def clang_format_file(path: Path, style_file: Path) -> None:
    clang_format = _find_clang_format()
    subprocess.run(
        [str(clang_format), f"-style=file:{style_file}", "-i", str(path)],
        check=True,
    )
    subprocess.run(
        [str(clang_format), f"-style=file:{style_file}", "--Werror", "--dry-run", str(path)],
        check=True,
    )


def verify_header_tidy(header_path: Path, project_dir: Path, build_dir: Path, generated_dir: Path) -> None:
    clang_tidy = _find_clang_tidy()
    config_file = project_dir / ".clang-tidy"
    if not config_file.exists():
        raise RuntimeError(f"clang-tidy config not found: {config_file}")

    lint_dir = generated_dir / ".lint"
    lint_dir.mkdir(parents=True, exist_ok=True)
    lint_cpp = lint_dir / f"{header_path.stem}_lint.cpp"
    lint_cpp.write_text(f'#include "{header_path.name}"\n', encoding="utf-8")

    compile_commands_dir = build_dir
    compile_commands = compile_commands_dir / "compile_commands.json"
    cmd = [
        str(clang_tidy),
        str(lint_cpp),
        f"--config-file={config_file}",
        "--header-filter=.*/generated/.*",
        "-warnings-as-errors=*",
    ]
    if compile_commands.exists():
        cmd.append(f"-p={compile_commands_dir}")
    else:
        print(
            "[WARN] build/compile_commands.json not found; "
            "run cmake configure for full tidy flags",
            file=sys.stderr,
        )

    cmd.extend(
        [
            "--",
            "-std=c++17",
            f"-I{generated_dir}",
        ]
    )
    result = subprocess.run(cmd, cwd=generated_dir, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr or result.stdout or f"clang-tidy failed for {header_path.name}"
        raise RuntimeError(message)


def verify_generated_cpp_header(header_path: Path, project_dir: Path, build_dir: Path, generated_dir: Path) -> None:
    style_file = project_dir / ".clang-format"
    if not style_file.exists():
        raise RuntimeError(f"clang-format config not found: {style_file}")
    clang_format_file(header_path, style_file)
    verify_header_tidy(header_path, project_dir, build_dir, generated_dir)


class ShaderCompiler:
    def __init__(self, config: CompileConfig):
        self.config = config
        self.config.generated_dir.mkdir(parents=True, exist_ok=True)
        self.shader_config = self._generate_shader_config()
        self.slangc = self._find_slangc()
        self.glslc = self._find_glslc()
        self.manifest_modules: list[dict[str, Any]] = []

    def _generate_shader_config(self) -> dict[str, Any]:
        return generate_shader_config(
            header=self.config.config_header_path,
            slang_out=self.config.generated_dir / "config_generated.slang",
            glsl_out=self.config.generated_dir / "config_generated.glsl",
            json_out=self.config.generated_dir / "shader_config.json",
            emulate_int64=self.config.emulate_int64,
            emulate_f32_atomic=self.config.emulate_f32_atomic,
        )

    def _find_slangc(self) -> Path:
        if self.config.slangc_path and self.config.slangc_path.exists():
            return self.config.slangc_path
        slangc = shutil.which("slangc")
        if slangc:
            return Path(slangc)
        raise RuntimeError("slangc not found. Install the Vulkan SDK or pass --slangc")

    def _find_glslc(self) -> Path:
        if self.config.glslc_path and self.config.glslc_path.exists():
            return self.config.glslc_path
        glslc = shutil.which("glslc")
        if glslc:
            return Path(glslc)
        raise RuntimeError("glslc not found. Install the Vulkan SDK or pass --glslc")

    def _create_shader_jobs(self) -> list[ShaderSource]:
        # Add kernel jobs here as stages are ported (Phase 1+).
        return [
            ShaderSource(source="smoke.slang", language="slang", jobs=[ShaderJob("smoke", {})]),
        ]

    def _compile_slang_job(self, source_file: str, job: ShaderJob) -> Path:
        source_path = self.config.slang_src_path / source_file
        spirv_path = self.config.generated_dir / f"{job.name}.spv"
        header_path = self.config.generated_dir / f"{job.name}_spirv.hpp"

        cmd = [str(self.slangc), str(source_path)]
        for define, value in job.defines.items():
            cmd.append(f"-D{define}={value}")
        cmd.extend(["-target", "spirv"])
        cmd.extend(self.config.slangc_compile_args)
        cmd.extend(["-I", str(self.config.generated_dir)])
        cmd.extend(["-I", str(self.config.slang_src_path)])
        cmd.extend(["-o", str(spirv_path)])

        subprocess.run(cmd, capture_output=True, text=True, check=True)
        header_path.write_text(
            spirv_to_header(spirv_path.read_bytes(), job.name),
            encoding="utf-8",
        )
        spirv_path.unlink(missing_ok=True)
        verify_generated_cpp_header(
            header_path,
            self.config.project_dir,
            self.config.build_dir,
            self.config.generated_dir,
        )

        array_symbol, word_count_symbol = spirv_symbols(job.name)
        self.manifest_modules.append(
            {
                "name": job.name,
                "source": source_file,
                "header": header_path.name,
                "array_symbol": array_symbol,
                "word_count_symbol": word_count_symbol,
                "defines": job.defines,
            }
        )
        return header_path

    def compile_all(self) -> None:
        for source in self._create_shader_jobs():
            if source.language != "slang":
                raise ValueError(f"Unsupported shader language for {source.source}: {source.language}")
            for job in source.jobs:
                print(f">>> Compiling {source.source} ({job.name})")
                header_path = self._compile_slang_job(source.source, job)
                print(f"[OK] Wrote {header_path.name}")

        manifest = {
            "schema": 1,
            "use_emulated_int64": self.shader_config["use_emulated_int64"],
            "use_emulated_f32_atomic": self.shader_config["use_emulated_f32_atomic"],
            "modules": self.manifest_modules,
        }
        manifest_path = self.config.generated_dir / "shader_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] Wrote {manifest_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=SCRIPT_DIR.parents[0],
        help="nlrc_vksplat project root",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=None,
        help="Output directory for generated headers and config (required from CMake)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="CMake binary directory containing compile_commands.json",
    )
    parser.add_argument("--slangc", type=Path, default=None)
    parser.add_argument("--glslc", type=Path, default=None)
    parser.add_argument("--emulate-int64", type=int, choices=[0, 1], required=True)
    parser.add_argument("--emulate-f32-atomic", type=int, choices=[0, 1], required=True)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    generated_dir = (
        args.generated_dir.resolve()
        if args.generated_dir is not None
        else (project_dir / "build" / "generated")
    )
    build_dir = args.build_dir.resolve() if args.build_dir is not None else generated_dir.parent

    config = CompileConfig(
        project_dir=project_dir,
        build_dir=build_dir,
        slang_src_path=project_dir / "slang",
        shader_dst_path=project_dir / "shader",
        generated_dir=generated_dir,
        config_header_path=project_dir / "src" / "nlrc_vksplat_config.hpp",
        slangc_path=args.slangc,
        glslc_path=args.glslc,
        emulate_int64=args.emulate_int64,
        emulate_f32_atomic=args.emulate_f32_atomic,
    )

    try:
        compiler = ShaderCompiler(config)
        compiler.compile_all()
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or exc.stdout or str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("[OK] Shader build complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
