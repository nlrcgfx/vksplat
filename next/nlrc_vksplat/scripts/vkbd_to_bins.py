#!/usr/bin/env python3
"""Convert ref .vkbd dumps to raw .bin fixtures (read-only ref tool wrapper).

Future usage:

    python scripts/vkbd_to_bins.py \\
        --vkbd ../../../outputs/.../xyz_ws.vkbd \\
        --out ../../../test_data/fixtures/D_cumsum/xyz_ws.bin \\
        --dtype float32

This gap #5 stub documents the workflow; implementation will shell out to
ref/vksplat/scripts/vkbd_tool.py without linking ref CMake targets.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vkbd", type=str, help="Source .vkbd path")
    parser.add_argument("--out", type=str, help="Destination .bin path")
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    if not args.vkbd or not args.out:
        print("vkbd_to_bins.py: specify --vkbd and --out (stub not yet implemented)", file=sys.stderr)
        return 2

    print("vkbd_to_bins.py: not implemented; use ref vkbd_tool.py manually for now", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
