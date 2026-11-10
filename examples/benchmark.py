"""Benchmark reconstruction, clutter filters and conventional/two-stage pipelines.

No arguments preserve the existing all-method reconstruction benchmark.
Filtering requires explicit rank bounds; extended precision checks are opt-in.
"""

import argparse
import importlib
import json
import math
from pathlib import Path


def main(argv=None):
    """Validate options before importing GPU suites; write one aggregate report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("reconstruction", "clutter", "pipeline", "all"), default="reconstruction")
    for name in ("grid", "elements", "angles", "frames", "repeat"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--rank-start", type=int, help="First retained zero-based rank; required for filtering")
    parser.add_argument("--rank-stop", type=int, help="Exclusive final rank; defaults to original frame count")
    parser.add_argument("--density", type=float, nargs="+", help="Pipeline samples/wavelength; default: 8 16 32")
    parser.add_argument("--frame-chunk", type=int, help="Two-stage pipeline chunk; default: 32")
    parser.add_argument(
        "--extended-clutter",
        action="store_true",
        help="Also run the adversarial 256x32 precision sweep for both source dtypes",
    )
    parser.add_argument("--output", type=Path, help="Save all selected suites to one strict JSON report")
    args = parser.parse_args(argv)
    common = {
        name: getattr(args, name)
        for name in ("grid", "elements", "angles", "frames", "repeat")
        if getattr(args, name) is not None
    }
    if any(value < (2 if name == "frames" else 1) for name, value in common.items()):
        parser.error("Positive dimensions/repeats and at least two frames required")
    if args.suite == "reconstruction":
        if args.rank_start is not None or args.rank_stop is not None:
            parser.error("Rank bounds require --suite clutter, pipeline or all")
    else:
        if args.rank_start is None:
            parser.error("Filtering requires --rank-start (no automatic cutoff)")
        frames = common.get("frames", 200)
        stop = frames if args.rank_stop is None else args.rank_stop
        if not 0 <= args.rank_start <= stop <= frames:
            parser.error("Require 0 <= rank-start <= rank-stop <= frames")
    if args.extended_clutter and args.suite not in ("clutter", "all"):
        parser.error("--extended-clutter requires --suite clutter or all")
    if (args.density is not None or args.frame_chunk is not None) and args.suite not in ("pipeline", "all"):
        parser.error("--density/--frame-chunk require --suite pipeline or all")
    if args.frame_chunk is not None and args.frame_chunk < 1:
        parser.error("--frame-chunk must be positive")
    if args.density is not None and (
        any(not math.isfinite(d) or d < 2 for d in args.density) or len(set(args.density)) != len(args.density)
    ):
        parser.error("Densities must be unique, finite and >=2")

    shared = []
    for name, value in common.items():
        shared.extend((f"--{name}", str(value)))
    filtering = list(shared)
    for name in ("rank_start", "rank_stop"):
        if (value := getattr(args, name)) is not None:
            filtering.extend((f"--{name.replace('_', '-')}", str(value)))
    report = {
        "suites": {},
        "targets": {"frames": 200, "prf_hz": 500, "reconstruction_only_ms": 400, "complete_pipeline_ms": 400},
    }
    for suite in ("reconstruction", "clutter", "pipeline"):
        if args.suite not in (suite, "all"):
            continue
        if suite == "reconstruction":
            result = importlib.import_module("_support.rca_benchmark").main(**common)
        elif suite == "clutter":
            result = importlib.import_module("_support.clutter_benchmark").main(filtering)
        else:
            options = list(filtering)
            if args.density is not None:
                options.extend(("--density", *(str(d) for d in args.density)))
            if args.frame_chunk is not None:
                options.extend(("--frame-chunk", str(args.frame_chunk)))
            result = importlib.import_module("_support.two_stage_clutter").main(options)
        report["suites"][suite] = result
    if args.extended_clutter:
        accuracy = importlib.import_module("_support.clutter_accuracy")
        report["suites"]["clutter_accuracy"] = {
            dtype: accuracy.main(["--input-dtype", dtype]) for dtype in ("complex64", "complex128")
        }
    print(
        "Targets for 200 frames / 500 Hz: 400 ms reconstruction-only; 400 ms complete pipeline including filtering/transfers."
    )
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Details saved to {args.output}")
    return report


if __name__ == "__main__":
    main()
