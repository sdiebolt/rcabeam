"""Single napari entry point: RCA methods, dual-stage OPW or interpolation.

Use `visualize.py COMPARISON --help` for options. Omitting COMPARISON runs RCA.
--headless reconstructs without opening a window, for compute-only checks.
"""

from __future__ import annotations

import argparse
import importlib
import sys

COMPARISONS = {
    "rca": "rca_visualization",
    "two-stage": "two_stage_visualization",
    "interpolation": "interpolation",
}


def main(argv=None):
    """Open one selected comparison; never run benchmark repetitions or PSF sweeps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", nargs="?", choices=COMPARISONS, default="rca")
    parser.add_argument("--headless", action="store_true", help="Reconstruct without opening napari.")
    args = list(sys.argv[1:] if argv is None else argv)
    show = "--headless" not in args
    args = [arg for arg in args if arg != "--headless"]
    if args in (["--help"], ["-h"]):
        parser.print_help()
        return
    comparison = "rca"
    if args and not args[0].startswith("-"):
        comparison = args.pop(0)
        if comparison not in COMPARISONS:
            parser.error(f"Unknown comparison {comparison!r}; choose from {', '.join(COMPARISONS)}")
    return importlib.import_module(f"_support.{COMPARISONS[comparison]}").main(args, show=show)


if __name__ == "__main__":
    main()
