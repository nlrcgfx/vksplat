#!/usr/bin/env python3
"""
Shader compiler for VkSplat (reference copy).

Run from the ref/ directory:

    python compile_shaders.py
"""

import os
import sys
import hashlib
import subprocess
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Literal
import argparse
import shutil
import json

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "vksplat" / "scripts"))
from generate_shader_config import generate_shader_config


@dataclass
class CompileConfig:
    """Configuration for shader compilation"""
    slang_src_path: Path = Path("./vksplat/slang")
    shader_dst_path: Path = Path("./vksplat/shader")
    generated_dst_path: Path = Path("./vksplat/shader/generated")
    config_header_path: Path = Path("./vksplat/src/vksplat_config.h")
    config_generator_path: Path = Path("./vksplat/scripts/generate_shader_config.py")
    # slangc_compile_args: str = "-stage compute -O3 -fp-mode fast -line-directive-mode none"
    slangc_compile_args: str = "-stage compute -O -fp-mode fast -line-directive-mode none"
    slangc_glsl_compile_args: str = "-stage compute -fp-mode fast -line-directive-mode standard"
    glslc_compile_args: str = "-O --target-spv=spv1.5 --target-env=vulkan1.2"
    slangc_path: Optional[Path] = None
    glslc_path: Optional[Path] = None
    max_workers: Optional[int] = None
    emit_glsl: bool = False
    emulate_int64: Optional[int] = None
    emulate_f32_atomic: Optional[int] = None


@dataclass
class ShaderJob:
    """Represents a single shader compilation job"""
    name: str
    defines: Dict[str, int]


class ShaderCompiler:
    def __init__(self, config: CompileConfig):
        self.config = config

        self.config.shader_dst_path.mkdir(exist_ok=True)
        self.config.generated_dst_path.mkdir(exist_ok=True)
        self.shader_config = self._generate_shader_config()

        self.slangc = self._find_slangc()
        self._glslc: Optional[Path] = None

    def _generate_shader_config(self) -> Dict:
        return generate_shader_config(
            header=self.config.config_header_path,
            slang_out=self.config.slang_src_path / "config_generated.slang",
            glsl_out=self.config.shader_dst_path / "radix_sort" / "config_generated.glsl",
            json_out=self.config.generated_dst_path / "shader_config.json",
            emulate_int64=self.config.emulate_int64,
            emulate_f32_atomic=self.config.emulate_f32_atomic,
        )

    @property
    def glslc(self) -> Path:
        if self._glslc is None:
            self._glslc = self._find_glslc()
        return self._glslc

    def _find_slangc(self) -> Path:
        """Find slangc executable"""
        if self.config.slangc_path and self.config.slangc_path.exists():
            return self.config.slangc_path
        
        slangc = shutil.which("slangc")
        if slangc:
            return Path(slangc)
        
        common_paths = [
            Path.home() / ".local/lib/python3.10/site-packages/slangtorch/bin/slangc",
        ]
        for path in common_paths:
            if path.exists():
                return path
        
        raise RuntimeError("slangc not found. Please install Slang or specify path with --slangc")

    def _find_glslc(self) -> Path:
        """Find glslc executable"""
        if self.config.glslc_path and self.config.glslc_path.exists():
            return self.config.glslc_path
        
        glslc = shutil.which("glslc")
        if glslc:
            return Path(glslc)
        
        common_paths = [
        ]
        for path in common_paths:
            if path.exists():
                return path
        
        raise RuntimeError("glslc not found. Please install glslc or specify path with --glslc")

    def _compute_checksum(self, files: List[Path], for_glsl: bool = False) -> str:
        """Compute MD5 checksum of multiple files"""
        files_expanded = []
        for path in files:
            if '.' not in path:
                path = os.path.join(self.config.shader_dst_path, path)
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        full_path = Path(root) / file
                        files_expanded.append(full_path)
            else:
                files_expanded.append(self.config.slang_src_path / path)

        hasher = hashlib.md5()
        for file_path in files_expanded:
            assert file_path.exists(), f"{file_path} does not exist"
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
        with open(__file__, 'rb') as f:
            hasher.update(f.read())
        for extra_file in [
            self.config.config_header_path,
            self.config.config_generator_path,
            self.config.generated_dst_path / "shader_config.json",
        ]:
            if extra_file.exists():
                with open(extra_file, 'rb') as f:
                    hasher.update(f.read())
        if for_glsl:
            hasher.update(bytearray(self.config.slangc_glsl_compile_args, 'utf-8'))
        else:
            hasher.update(bytearray(self.config.slangc_compile_args, 'utf-8'))
            hasher.update(bytearray(self.config.glslc_compile_args, 'utf-8'))
        return hasher.hexdigest()

    def _should_compile(self, checksum_key: str, checksum: str) -> bool:
        """Check if compilation is needed based on checksum"""
        checksum_path = self.config.shader_dst_path / "checksum.json"
        try:
            with open(checksum_path, 'r') as fp:
                checksums = json.load(fp)
        except:
            checksums = {}

        if checksum_key in checksums and checksums[checksum_key] == checksum:
            return False
        
        checksums[checksum_key] = checksum
        with open(checksum_path, 'w') as fp:
            json.dump(checksums, fp, indent=4)
        return True
    
    def _compile_shader_slang(self, source_file: Path, job: ShaderJob,
                              target: Literal["glsl", "spirv", "host-callable"],
                              ext: Literal["comp", "spv"]) -> Tuple[bool, str]:
        source_file = str(self.config.slang_src_path / source_file)
        cmd = [str(self.slangc), source_file]
        job_name = job.name + '.' + ext
        output_file = self.config.generated_dst_path / job_name  # type: Path
        if output_file.exists():
            output_file.unlink()
        
        # Add defines
        for define, value in job.defines.items():
            cmd.extend([f"-D{define}={value}"])
        
        # Add target and other args
        cmd.extend(["-target", target])
        compile_args = (
            self.config.slangc_glsl_compile_args if target == "glsl"
            else self.config.slangc_compile_args
        )
        cmd.extend(compile_args.split())
        cmd.extend(["-o", str(output_file)])
        
        output = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output

    def _compile_shader_glsl(self, source_file: Path, job: ShaderJob, target: Literal["glsl", "spirv"], ext: Literal["comp", "spv"]) -> Tuple[bool, str]:
        cmd = [str(self.glslc), str(self.config.shader_dst_path / source_file / job.name)]
        job_name = Path(job.name).with_suffix('.'+ext)
        output_file = self.config.shader_dst_path / source_file / job_name
        if output_file.exists():
            output_file.unlink()
        
        # Add defines
        for define, value in job.defines.items():
            cmd.extend([f"-D{define}={value}"])
        
        # Add target and other args
        if target is not None:
            cmd.extend(["-target", target])
        cmd.extend(self.config.glslc_compile_args.split())
        cmd.extend(["-o", str(output_file)])
        
        return subprocess.run(cmd, capture_output=True, text=True, check=True)

    def _compile_shader(self, source_file: Path, job: ShaderJob, target: Literal["glsl", "spirv"], ext: Literal["comp", "spv"]) -> Tuple[bool, str]:
        """Compile a single shader"""
        try:
            if '.' in source_file:
                job_name = job.name + '.' + ext
                output = self._compile_shader_slang(source_file, job, target, ext)
            else:
                job_name = os.path.join(source_file, job.name + '.' + ext)
                output = self._compile_shader_glsl(source_file, job, target, ext)
            if output.stdout != "" or output.stderr != "":
                return False, f"O Compiled {job_name} with warning: {output.stdout} {output.stderr}"
            return True, f"[OK] Compiled {job_name}"
        except subprocess.CalledProcessError as e:
            return False, f"[FAILED] Failed to compile {job_name}: {e.stderr}"
        except Exception as e:
            return False, f"[FAILED] Error compiling {job_name}: {str(e)}"

    def _create_shader_jobs(self) -> List[Tuple[str, List[ShaderJob]]]:
        """Create all shader compilation jobs grouped by category"""
        jobs = []

        forward, backward = 0, 1  # export modes
        config_deps = ["config.slang", "config_generated.slang"]

        # Radix sort
        jobs.append(("radix_sort", [
            ShaderJob("upsweep.comp", {}),
            ShaderJob("spine.comp", {}),
            ShaderJob("downsweep.comp", {}),
        ], []))

        # Morton sorting Functions
        jobs.append(("morton_sort.slang", [
            ShaderJob(
                f"morton_sort_{phase_name}", {"MORTON_SORT_PHASE": phase_id, "EXPORT_MODE": forward}
            ) for phase_name, phase_id in [
                ("compute_stats", 0),
                ("generate_keys", 1),
                ("apply_indices", 2),
                ("apply_indices_sh", 3),
                ("update_buffer", 4),
                ("update_buffer_sh", 5),
            ]
        ], ["strategy_utils.slang"] + config_deps))

        # MCMC Functions
        jobs.append(("mcmc.slang", [
            ShaderJob(
                f"mcmc_{phase_name}", {"MCMC_PHASE": phase_id, "EXPORT_MODE": forward}
            ) for phase_name, phase_id in [
                ("inject_noise", 0), ("compute_probs", 1),
                ("compute_relocation_index_map", 10), ("compute_relocation", 11), ("update_relocation", 12),
                ("compute_add_index_map", 20), ("compute_add", 21), ("update_add", 22),
            ]
        ], ["strategy_utils.slang"] + config_deps))

        # Default Densification Functions
        jobs.append(("default.slang", [
            ShaderJob(
                f"default_{phase_name}", {"DEFAULT_PHASE": phase_id, "EXPORT_MODE": forward}
            ) for phase_name, phase_id in [
                ("update_state", 0), ("compute_grow_mask", 1), ("duplicate", 2),
                ("split", 3), ("compute_prune_mask", 4),
                ("prune", 5), ("prune_mean", 51), ("prune_sh", 52),
                ("reset_opa", 6)
            ]
        ], ["strategy_utils.slang"] + config_deps))

        # Where (torch.where equivalent)
        jobs.append(("where.slang", [
            ShaderJob("where", {}),
        ], config_deps))

        # Summation
        jobs.append(("sum.slang", [
            ShaderJob("sum", {}),
        ], config_deps))

        # Fused projection backward and optimizer
        jobs.append(("fused_projection_backward_optimizer.slang", [
            ShaderJob("fused_projection_backward_optimizer", {"EXPORT_MODE": backward}),
        ], config_deps + ["utils.slang", "spherical_harmonics.slang"]))

        # SSIM
        jobs.append(("ssim.slang", [
            ShaderJob("ssim_forward", {"EXPORT_MODE": forward}),
            ShaderJob("ssim_backward", {"EXPORT_MODE": backward}),
        ], config_deps))

        # Prefix Sum
        jobs.append(("cumsum.slang", [
            ShaderJob(
                f"cumsum_{phase_name}",
                {"CUMSUM_PHASE": phase_id}
            ) for phase_name, phase_id in [
                ("block_scan", 1), ("scan_block_sums", 2),
                ("add_block_offsets", 3), ("single_pass", 0)
            ]
        ], config_deps))

        # Alpha Blending
        tensor_bwd_configs = [
            (
                cfg["use_subgroup_operations"],
                cfg["splat_batch_size"],
                cfg["group_reduce_before_atomic"],
            )
            for cfg in self.shader_config["tensor_backward_configs"]
        ]
        jobs.append(("alphablend_shader.slang", [
            ShaderJob(f"rasterize_forward", {"EXPORT_MODE": forward}),
        ] + [
            ShaderJob(f"rasterize_backward_{i}", {
                "EXPORT_MODE": backward, "BACKWARD_MODE": min(i, 2),
                "USE_SUBGROUP_OPERATIONS": tensor_bwd_configs[max(i-2,0)][0],
                "SPLAT_BATCH_SIZE": tensor_bwd_configs[max(i-2,0)][1],
                "GROUP_REDUCE_BEFORE_ATOMIC": tensor_bwd_configs[max(i-2,0)][2],
            })
            for i in range(2+len(tensor_bwd_configs))
        ], config_deps + ["utils.slang",
            "alphablend_shader_bwd_per_pixel.slang",
            "alphablend_shader_bwd_per_splat.slang",
            "alphablend_shader_bwd_tensor.slang"
        ]))

        # Tile Shaders
        jobs.append(("tile_shader.slang", [
            ShaderJob(f"generate_keys", {"EXPORT_MODE": forward, "ENTRY": 1}),
            ShaderJob("compute_tile_ranges", {"EXPORT_MODE": forward, "ENTRY": 2}),
        ], config_deps + ["utils.slang"]))

        # Vertex Shader
        jobs.append(("vertex_shader.slang", [
            ShaderJob(f"projection_forward", {"EXPORT_MODE": forward}),
        ], config_deps + ["utils.slang", "spherical_harmonics.slang"]))

        return jobs

    def compile_all(self, force: bool = False) -> bool:
        """Compile all shaders"""
        job_groups = self._create_shader_jobs()
        all_successful = True

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:

            futures = {}

            for source_file, shader_jobs, deps in job_groups:
                if not shader_jobs:
                    continue

                checksum_key = source_file
                spirv_checksum = self._compute_checksum([source_file] + deps)
                spirv_needed = force or self._should_compile(checksum_key, spirv_checksum)

                glsl_needed = False
                if self.config.emit_glsl and '.' in source_file:
                    glsl_checksum = self._compute_checksum([source_file] + deps, for_glsl=True)
                    glsl_needed = force or self._should_compile(
                        f"{checksum_key}#glsl", glsl_checksum
                    )

                if not spirv_needed and not glsl_needed:
                    continue

                print(f">>> Adding \"{source_file}\" to compilation queue")

                for job in shader_jobs:
                    if '.' in source_file:
                        if spirv_needed:
                            futures[executor.submit(
                                self._compile_shader, source_file, job, "spirv", "spv"
                            )] = None
                        if glsl_needed:
                            futures[executor.submit(
                                self._compile_shader, source_file, job, "glsl", "comp"
                            )] = None
                    else:
                        futures[executor.submit(
                            self._compile_shader, source_file, job, None, "spv"
                        )] = None

            for future in concurrent.futures.as_completed(futures):
                success, message = future.result()
                print(message)
                if not success:
                    all_successful = False

        return all_successful


def main():
    parser = argparse.ArgumentParser(description="Parallel Slang shader compiler")
    parser.add_argument("--slangc", type=Path, help="Path to slangc executable")
    parser.add_argument("--glslc", type=Path, help="Path to glslc executable")
    parser.add_argument("--src", type=Path, default="./vksplat/slang",
                       help="Source directory for Slang files")
    parser.add_argument("--dst", type=Path, default="./vksplat/shader",
                       help="Destination directory for compiled shaders")
    parser.add_argument("-j", "--jobs", type=int, help="Number of parallel jobs")
    parser.add_argument("--force", action="store_true", help="Force recompilation of all shaders")
    parser.add_argument(
        "--emit-glsl", action="store_true",
        help="Emit GLSL intermediate (.comp) for every Slang shader alongside SPIR-V"
    )
    parser.add_argument(
        "--emulate-int64", type=int, choices=[0, 1], default=None,
        help="Generate shaders with USE_EMULATED_INT64 set to 0 or 1"
    )
    parser.add_argument(
        "--emulate-f32-atomic", type=int, choices=[0, 1], default=None,
        help="Generate shaders with USE_EMULATED_F32_ATOMIC set to 0 or 1"
    )

    args = parser.parse_args()

    config = CompileConfig(
        slang_src_path=args.src,
        shader_dst_path=args.dst,
        generated_dst_path=(args.dst / "generated"),
        slangc_path=args.slangc,
        glslc_path=args.glslc,
        max_workers=args.jobs,
        emit_glsl=args.emit_glsl,
        emulate_int64=args.emulate_int64,
        emulate_f32_atomic=args.emulate_f32_atomic,
    )
    compiler = ShaderCompiler(config)
    success = compiler.compile_all(force=args.force)
    
    if success:
        print("[OK] All shaders compiled successfully!")
        sys.exit(0)
    else:
        print("[FAILED] Some shaders failed to compile!")
        sys.exit(1)


if __name__ == "__main__":
    main()
