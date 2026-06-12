#!/usr/bin/env python3
"""Run clang-format and clang-tidy on rewrite C++ sources."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SOURCE_DIRS = ("src", "apps", "tests")


def discover_sources(project_dir: Path) -> tuple[list[Path], list[Path]]:
    headers: list[Path] = []
    sources: list[Path] = []
    for directory in SOURCE_DIRS:
        root = project_dir / directory
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".hpp":
                headers.append(path)
            elif path.suffix == ".cpp":
                sources.append(path)
    return headers, sources


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} not found. Install LLVM tools or add them to PATH")
    return path


def run_clang_format(paths: list[Path], style_file: Path, fix: bool) -> None:
    if not paths:
        return
    clang_format = require_tool("clang-format")
    style_arg = f"file:{style_file}"
    if fix:
        subprocess.run(
            [clang_format, f"-style={style_arg}", "-i", *[str(path) for path in paths]],
            check=True,
        )
    subprocess.run(
        [
            clang_format,
            f"-style={style_arg}",
            "--Werror",
            "--dry-run",
            *[str(path) for path in paths],
        ],
        check=True,
    )


def run_clang_tidy(sources: list[Path], project_dir: Path) -> None:
    if not sources:
        return

    compile_commands = project_dir / "build" / "compile_commands.json"
    if not compile_commands.exists():
        raise RuntimeError(
            "build/compile_commands.json not found. "
            "Run `cmake --preset <preset>` before nlrc_vksplat_lint"
        )

    clang_tidy = require_tool("clang-tidy")
    default_config = project_dir / ".clang-tidy"
    tests_config = project_dir / "tests" / ".clang-tidy"
    for source in sources:
        if tests_config.exists():
            try:
                source.relative_to(project_dir / "tests")
                config_file = tests_config
            except ValueError:
                config_file = default_config
        else:
            config_file = default_config
        cmd = [
            clang_tidy,
            str(source),
            f"-p={project_dir / 'build'}",
            f"--config-file={config_file}",
            "-warnings-as-errors=*",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            message = result.stderr or result.stdout or f"clang-tidy failed for {source}"
            raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=PROJECT_DIR,
        help="nlrc_vksplat project root",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply clang-format fixes before checking",
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    style_file = project_dir / ".clang-format"
    if not style_file.exists():
        print(f"clang-format config not found: {style_file}", file=sys.stderr)
        return 1

    headers, sources = discover_sources(project_dir)
    all_paths = headers + sources
    if not all_paths:
        print("[OK] No C++ sources to lint")
        return 0

    try:
        run_clang_format(all_paths, style_file, args.fix)
        run_clang_tidy(sources, project_dir)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"[OK] Lint passed ({len(all_paths)} file(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
