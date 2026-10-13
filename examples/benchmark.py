"""Single benchmark entry point; each suite retains its own timing boundaries.

Use `benchmark.py SUITE --help` for suite options. Omitting SUITE runs RCA and
preserves existing `benchmark.py --no-matrix ...` commands.
"""

from __future__ import annotations

import argparse
import importlib
import sys

SUITES = {
    "rca": "rca_benchmark",
    "two-stage": "two_stage_benchmark",
    "matrix-backends": "matrix_backends",
    "ffdas-paper": "ffdas_paper",
    "matrix-chunks": "matrix_chunks",
}


def main(argv=None) -> None:
    """Dispatch without importing optional GPU/backend dependencies until selected."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", choices=SUITES, default="rca")
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["--help"], ["-h"]):
        parser.print_help()
        return
    suite = "rca"
    if args and not args[0].startswith("-"):
        suite = args.pop(0)
        if suite not in SUITES:
            parser.error(f"Unknown suite {suite!r}; choose from {', '.join(SUITES)}")
    importlib.import_module(f"_support.{SUITES[suite]}").main(args)


if __name__ == "__main__":
    main()
