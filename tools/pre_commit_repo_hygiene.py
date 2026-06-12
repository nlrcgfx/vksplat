#!/usr/bin/env python3
"""Small repository hygiene checks for pre-commit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

BINARY_SAMPLE_SIZE = 8192
MERGE_MARKERS = (b"<<<<<<< ", b"=======\n", b">>>>>>> ")
BINARY_SUFFIXES = {
    ".bin",
    ".exe",
    ".gif",
    ".gz",
    ".jpg",
    ".jpeg",
    ".lib",
    ".pdf",
    ".png",
    ".pyd",
    ".ply",
    ".so",
    ".spv",
    ".zip",
}


def existing_paths(paths: list[str]) -> list[Path]:
    return [Path(path) for path in paths if Path(path).is_file()]


def is_binary(path: Path) -> bool:
    if path.suffix.lower() in BINARY_SUFFIXES:
        return True
    sample = path.read_bytes()[:BINARY_SAMPLE_SIZE]
    return b"\0" in sample


def text_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not is_binary(path)]


def report_failures(success_message: str, failure_title: str, failures: list[str]) -> int:
    if not failures:
        print(f"[OK] {success_message}")
        return 0
    print(failure_title, file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    return 1


def check_trailing_whitespace(paths: list[Path]) -> int:
    failures: list[str] = []
    for path in text_paths(paths):
        for line_number, line in enumerate(path.read_bytes().splitlines(keepends=True), start=1):
            content = line.rstrip(b"\r\n")
            if content.endswith((b" ", b"\t")):
                failures.append(f"{path}:{line_number}")
    return report_failures("No trailing whitespace", "Trailing whitespace found:", failures)


def check_final_newline(paths: list[Path]) -> int:
    failures: list[str] = []
    for path in text_paths(paths):
        data = path.read_bytes()
        if data and not data.endswith((b"\n", b"\r\n")):
            failures.append(str(path))
    return report_failures("All text files end with a newline", "Files missing a final newline:", failures)


def check_merge_conflict_markers(paths: list[Path]) -> int:
    failures: list[str] = []
    for path in text_paths(paths):
        for line_number, line in enumerate(path.read_bytes().splitlines(keepends=True), start=1):
            if any(line.startswith(marker) for marker in MERGE_MARKERS):
                failures.append(f"{path}:{line_number}")
    return report_failures("No merge conflict markers", "Merge conflict markers found:", failures)


def check_large_files(paths: list[Path], max_kib: int) -> int:
    max_bytes = max_kib * 1024
    failures = [f"{path} ({path.stat().st_size} bytes)" for path in paths if path.stat().st_size > max_bytes]
    return report_failures(f"No files larger than {max_kib} KiB", f"Files larger than {max_kib} KiB found:", failures)


def check_json(paths: list[Path]) -> int:
    failures: list[str] = []
    for path in paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            failures.append(f"{path}: {exc}")
    return report_failures("JSON syntax passed", "Invalid JSON files found:", failures)


def check_toml(paths: list[Path]) -> int:
    if tomllib is None:
        print("TOML syntax checks require Python 3.11 or newer", file=sys.stderr)
        return 1

    failures: list[str] = []
    for path in paths:
        try:
            tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            failures.append(f"{path}: {exc}")
    return report_failures("TOML syntax passed", "Invalid TOML files found:", failures)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("trailing-whitespace", "final-newline", "merge-conflict", "large-files", "json", "toml"),
    )
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--max-kib", type=int, default=1024)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    paths = existing_paths(args.paths)

    if args.command == "trailing-whitespace":
        return check_trailing_whitespace(paths)
    if args.command == "final-newline":
        return check_final_newline(paths)
    if args.command == "merge-conflict":
        return check_merge_conflict_markers(paths)
    if args.command == "large-files":
        return check_large_files(paths, args.max_kib)
    if args.command == "json":
        return check_json(paths)
    if args.command == "toml":
        return check_toml(paths)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
