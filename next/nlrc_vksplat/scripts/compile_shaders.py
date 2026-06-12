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
    slang_src_path: Path
    shader_dst_path: Path
    generated_dir: Path
    config_header_path: Path
    stamp_path: Path | None = None
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


def spirv_symbol(job_name: str) -> str:
    """Map a snake_case job name to a clang-tidy-compliant embed symbol name."""
    parts = [part for part in job_name.split("_") if part]
    if not parts:
        raise ValueError("shader job name must not be empty")
    camel = "".join(part[:1].upper() + part[1:] for part in parts)
    return f"k{camel}Spirv"


def spirv_to_header(spirv_bytes: bytes, job_name: str) -> tuple[str, int]:
    array_symbol = spirv_symbol(job_name)
    if len(spirv_bytes) % 4 != 0:
        raise ValueError(f"SPIR-V size for {job_name} is not a multiple of 4 bytes")
    words = [
        int.from_bytes(spirv_bytes[i : i + 4], byteorder="little")
        for i in range(0, len(spirv_bytes), 4)
    ]
    body = ",\n    ".join(f"0x{word:08x}u" for word in words)
    header = "\n".join(
        [
            "#pragma once",
            "",
            "#include <array>",
            "#include <cstdint>",
            "",
            "namespace nlrc::vksplat::shaders {",
            "",
            f"inline constexpr std::array<uint32_t, {len(words)}> {array_symbol} = {{",
            f"    {body}",
            "};",
            "",
            "}  // namespace nlrc::vksplat::shaders",
            "",
        ]
    )
    return header, len(words)


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
        return [
            ShaderSource(source="smoke.slang", language="slang", jobs=[ShaderJob("smoke", {})]),
            ShaderSource(
                source="cumsum.slang",
                language="slang",
                jobs=[
                    ShaderJob("cumsum_single_pass", {"CUMSUM_PHASE": 0}),
                    ShaderJob("cumsum_block_scan", {"CUMSUM_PHASE": 1}),
                    ShaderJob("cumsum_scan_block_sums", {"CUMSUM_PHASE": 2}),
                    ShaderJob("cumsum_add_block_offsets", {"CUMSUM_PHASE": 3}),
                ],
                deps=["config.slang"],
            ),
            ShaderSource(source="sum.slang", language="slang", jobs=[ShaderJob("sum", {})], deps=["config.slang"]),
            ShaderSource(source="where.slang", language="slang", jobs=[ShaderJob("where", {})], deps=["config.slang"]),
            ShaderSource(
                source="vertex_shader.slang",
                language="slang",
                jobs=[ShaderJob("projection_forward", {"EXPORT_MODE": 0})],
                deps=["config.slang", "utils.slang", "spherical_harmonics.slang"],
            ),
            ShaderSource(
                source="tile_shader.slang",
                language="slang",
                jobs=[ShaderJob("generate_keys", {"ENTRY": 1})],
                deps=["config.slang", "utils.slang"],
            ),
            ShaderSource(
                source="radix_sort/upsweep.comp",
                language="glsl",
                jobs=[ShaderJob("radix_sort_upsweep", {})],
                deps=["radix_sort/config.glsl", "config_generated.glsl"],
            ),
            ShaderSource(
                source="radix_sort/spine.comp",
                language="glsl",
                jobs=[ShaderJob("radix_sort_spine", {})],
                deps=["radix_sort/config.glsl", "config_generated.glsl"],
            ),
            ShaderSource(
                source="radix_sort/downsweep.comp",
                language="glsl",
                jobs=[ShaderJob("radix_sort_downsweep", {})],
                deps=["radix_sort/config.glsl", "config_generated.glsl"],
            ),
        ]

    def _compile_slang_job(self, source: ShaderSource, job: ShaderJob) -> Path:
        source_path = self.config.slang_src_path / source.source
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
        header_text, word_count = spirv_to_header(spirv_path.read_bytes(), job.name)
        header_path.write_text(header_text, encoding="utf-8")
        spirv_path.unlink(missing_ok=True)

        self.manifest_modules.append(
            {
                "name": job.name,
                "source": source.source,
                "language": source.language,
                "header": header_path.name,
                "array_symbol": spirv_symbol(job.name),
                "word_count": word_count,
                "defines": job.defines,
                "deps": source.deps,
            }
        )
        return header_path

    def _compile_glsl_job(self, source: ShaderSource, job: ShaderJob) -> Path:
        source_path = self.config.shader_dst_path / source.source
        spirv_path = self.config.generated_dir / f"{job.name}.spv"
        header_path = self.config.generated_dir / f"{job.name}_spirv.hpp"

        cmd = [str(self.glslc), str(source_path)]
        for define, value in job.defines.items():
            cmd.append(f"-D{define}={value}")
        cmd.extend(self.config.glslc_compile_args)
        cmd.extend(["-I", str(self.config.generated_dir)])
        cmd.extend(["-I", str(self.config.shader_dst_path)])
        cmd.extend(["-o", str(spirv_path)])

        subprocess.run(cmd, capture_output=True, text=True, check=True)
        header_text, word_count = spirv_to_header(spirv_path.read_bytes(), job.name)
        header_path.write_text(header_text, encoding="utf-8")
        spirv_path.unlink(missing_ok=True)

        self.manifest_modules.append(
            {
                "name": job.name,
                "source": source.source,
                "language": source.language,
                "header": header_path.name,
                "array_symbol": spirv_symbol(job.name),
                "word_count": word_count,
                "defines": job.defines,
                "deps": source.deps,
            }
        )
        return header_path

    def compile_all(self) -> None:
        for source in self._create_shader_jobs():
            for job in source.jobs:
                print(f">>> Compiling {source.source} ({job.name})")
                if source.language == "slang":
                    header_path = self._compile_slang_job(source, job)
                elif source.language == "glsl":
                    header_path = self._compile_glsl_job(source, job)
                else:
                    raise ValueError(f"Unsupported shader language for {source.source}: {source.language}")
                print(f"[OK] Wrote {header_path.name}")

        manifest = {
            "schema": 2,
            "use_emulated_int64": self.shader_config["use_emulated_int64"],
            "use_emulated_f32_atomic": self.shader_config["use_emulated_f32_atomic"],
            "modules": self.manifest_modules,
        }
        manifest_path = self.config.generated_dir / "shader_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] Wrote {manifest_path.name}")

        if self.config.stamp_path is not None:
            self.config.stamp_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.stamp_path.write_text("ok\n", encoding="utf-8")
            print(f"[OK] Wrote {self.config.stamp_path.name}")


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
    parser.add_argument("--slangc", type=Path, default=None)
    parser.add_argument("--glslc", type=Path, default=None)
    parser.add_argument("--stamp", type=Path, default=None, help="Stable stamp output for CMake dependency tracking")
    parser.add_argument("--emulate-int64", type=int, choices=[0, 1], required=True)
    parser.add_argument("--emulate-f32-atomic", type=int, choices=[0, 1], required=True)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    generated_dir = (
        args.generated_dir.resolve()
        if args.generated_dir is not None
        else (project_dir / "build" / "generated")
    )
    config = CompileConfig(
        project_dir=project_dir,
        slang_src_path=project_dir / "slang",
        shader_dst_path=project_dir / "shader",
        generated_dir=generated_dir,
        config_header_path=project_dir / "src" / "nlrc_vksplat_config.hpp",
        stamp_path=args.stamp.resolve() if args.stamp is not None else None,
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
