#!/usr/bin/env python3
"""Pre-commit lint hooks for the rewrite C++ tree."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "next" / "nlrc_vksplat"
SOURCE_DIRS = ("src", "apps", "tests")
DEFAULT_BUILD_DIR = PROJECT_DIR / "build" / "windows-debug"
COMMAND_CHUNK_SIZE = 50


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


def run_command(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode == 0:
        return
    if result.stdout:
        print(result.stdout, file=sys.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def chunked(values: list[Path]) -> list[list[Path]]:
    return [values[index : index + COMMAND_CHUNK_SIZE] for index in range(0, len(values), COMMAND_CHUNK_SIZE)]


def run_clang_format(project_dir: Path, fix: bool) -> None:
    headers, sources = discover_sources(project_dir)
    all_paths = headers + sources
    if not all_paths:
        print("[OK] No C++ files to format")
        return

    style_file = project_dir / ".clang-format"
    if not style_file.exists():
        raise RuntimeError(f"clang-format config not found: {style_file}")

    clang_format = require_tool("clang-format")
    style_arg = f"file:{style_file}"
    if fix:
        for paths in chunked(all_paths):
            run_command([clang_format, f"-style={style_arg}", "-i", *[str(path) for path in paths]])
    for paths in chunked(all_paths):
        run_command([clang_format, f"-style={style_arg}", "--Werror", "--dry-run", *[str(path) for path in paths]])
    print(f"[OK] clang-format passed ({len(all_paths)} file(s))")


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def clang_tidy_config(project_dir: Path, source: Path) -> Path:
    tests_config = project_dir / "tests" / ".clang-tidy"
    if tests_config.exists() and is_under(source, project_dir / "tests"):
        return tests_config
    return project_dir / ".clang-tidy"


def run_clang_tidy(project_dir: Path, build_dir: Path) -> None:
    _, sources = discover_sources(project_dir)
    if not sources:
        print("[OK] No C++ sources to tidy")
        return

    compile_commands = build_dir / "compile_commands.json"
    if not compile_commands.exists():
        raise RuntimeError(
            f"{compile_commands} not found. Run `cmake --preset windows-debug` first, "
            "or set NLRC_VKSPLAT_TIDY_BUILD_DIR to another configured build directory."
        )

    clang_tidy = require_tool("clang-tidy")
    for source in sources:
        run_command(
            [
                clang_tidy,
                str(source),
                f"-p={build_dir}",
                f"--config-file={clang_tidy_config(project_dir, source)}",
                "-warnings-as-errors=*",
            ],
            cwd=project_dir,
        )
    print(f"[OK] clang-tidy passed ({len(sources)} file(s))")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    format_parser = subparsers.add_parser("format", help="Check clang-format over the rewrite C++ tree")
    format_parser.add_argument("--fix", action="store_true", help="Apply clang-format before checking")

    subparsers.add_parser("tidy", help="Run clang-tidy over the rewrite C++ source tree")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    build_dir = Path(os.environ.get("NLRC_VKSPLAT_TIDY_BUILD_DIR", DEFAULT_BUILD_DIR)).resolve()
    try:
        if args.command == "format":
            run_clang_format(PROJECT_DIR, args.fix)
        elif args.command == "tidy":
            run_clang_tidy(PROJECT_DIR, build_dir)
        else:
            parser.error(f"Unsupported command: {args.command}")
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
