#!/usr/bin/env python3
"""Export or check shader binding contracts against the C++ descriptor registry."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_BINDING_CONTRACTS = PROJECT_ROOT / "test_data" / "shader_binding_contracts.json"

VALID_ROUTE_SIDES = {"fixture", "golden", "both"}
VALID_CONTRACT_GROUPS = {"shaders", "fixture_contracts"}


def json_string_array_map(value: Any, path: Path, group: str) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {group} must be an object")

    result: dict[str, list[str]] = {}
    for name, bindings in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: {group} contains an invalid contract name")
        if not isinstance(bindings, list) or not all(isinstance(binding, str) for binding in bindings):
            raise ValueError(f"{path}: {group}/{name} must be an array of strings")
        result[name] = list(bindings)
    return result


def validate_contract_ref(document: dict[str, Any], contract_ref: str, path: Path) -> None:
    try:
        group, name = contract_ref.split("/", 1)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid contract reference {contract_ref!r}") from exc

    if group not in VALID_CONTRACT_GROUPS:
        raise ValueError(f"{path}: invalid contract group in {contract_ref!r}")
    if not name or name not in document[group]:
        raise ValueError(f"{path}: missing contract reference {contract_ref!r}")


def validate_route(value: Any, path: Path, *, require_contract: bool) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: route entries must be objects")

    subgraph = value.get("subgraph")
    stage_name = value.get("stage_name")
    stage_name_prefix = value.get("stage_name_prefix")
    side = value.get("side", "both")
    contract = value.get("contract")

    if not isinstance(subgraph, str) or not subgraph:
        raise ValueError(f"{path}: route is missing non-empty subgraph")
    has_exact = isinstance(stage_name, str) and bool(stage_name)
    has_prefix = isinstance(stage_name_prefix, str) and bool(stage_name_prefix)
    if has_exact == has_prefix:
        raise ValueError(f"{path}: route for {subgraph!r} must set exactly one stage matcher")
    if side not in VALID_ROUTE_SIDES:
        raise ValueError(f"{path}: route for {subgraph!r} has invalid side {side!r}")

    if require_contract:
        if not isinstance(contract, str) or not contract:
            raise ValueError(f"{path}: tracked route for {subgraph!r} is missing contract")
    elif contract is not None:
        raise ValueError(f"{path}: untracked route for {subgraph!r} must not set contract")


def validate_document(document: dict[str, Any], path: Path) -> None:
    if document.get("schema") != 2:
        raise ValueError(f"{path}: expected schema version 2")

    document["shaders"] = json_string_array_map(document.get("shaders"), path, "shaders")
    document["fixture_contracts"] = json_string_array_map(
        document.get("fixture_contracts"), path, "fixture_contracts"
    )

    routes = document.get("routes")
    untracked_routes = document.get("untracked_routes")
    if not isinstance(routes, list):
        raise ValueError(f"{path}: routes must be an array")
    if not isinstance(untracked_routes, list):
        raise ValueError(f"{path}: untracked_routes must be an array")

    for route in routes:
        validate_route(route, path, require_contract=True)
        validate_contract_ref(document, route["contract"], path)
    for route in untracked_routes:
        validate_route(route, path, require_contract=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_registry_shaders(registry_exporter: Path) -> dict[str, list[str]]:
    completed = subprocess.run(
        [str(registry_exporter)],
        check=True,
        capture_output=True,
        text=True,
    )
    snapshot = json.loads(completed.stdout)
    if not isinstance(snapshot, dict):
        raise ValueError(f"{registry_exporter}: exporter did not emit a JSON object")
    if snapshot.get("schema") != 2:
        raise ValueError(f"{registry_exporter}: exporter emitted unsupported schema")
    return json_string_array_map(snapshot.get("shaders"), registry_exporter, "shaders")


def merged_document(existing: dict[str, Any], registry_shaders: dict[str, list[str]], path: Path) -> dict[str, Any]:
    document = {
        "schema": 2,
        "shaders": registry_shaders,
        "fixture_contracts": existing.get("fixture_contracts"),
        "routes": existing.get("routes"),
        "untracked_routes": existing.get("untracked_routes"),
    }
    validate_document(document, path)
    return document


def normalized_json(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Fail if the checked-in JSON is not current")
    mode.add_argument("--write", action="store_true", help="Update the checked-in JSON")
    parser.add_argument(
        "--binding-contracts",
        type=Path,
        default=DEFAULT_BINDING_CONTRACTS,
        help="Path to test_data/shader_binding_contracts.json",
    )
    parser.add_argument(
        "--registry-exporter",
        type=Path,
        required=True,
        help="Compiled executable that emits registry shader bindings as JSON",
    )
    args = parser.parse_args()

    contracts_path = args.binding_contracts.resolve()
    registry_exporter = args.registry_exporter.resolve()

    try:
        existing = load_json(contracts_path)
        validate_document(existing, contracts_path)
        registry_shaders = load_registry_shaders(registry_exporter)
        expected = merged_document(existing, registry_shaders, contracts_path)
        expected_text = normalized_json(expected)
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"Failed to export shader binding contracts: {exc}", file=sys.stderr)
        return 1

    if args.write:
        contracts_path.write_text(expected_text, encoding="utf-8")
        print(f"[OK] Wrote {contracts_path}")
        return 0

    actual_text = contracts_path.read_text(encoding="utf-8")
    if actual_text == expected_text:
        print("[OK] Shader binding contracts match the C++ registry")
        return 0

    print("Shader binding contract drift detected.", file=sys.stderr)
    print(
        "Regenerate with: python next/nlrc_vksplat/scripts/export_shader_binding_contracts.py "
        "--write --registry-exporter <path-to-exporter>",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
