"""Run every RCA method, including two-stage OPW, in one Rich table.

Include the FPM reference when mach is installed.

No configuration arguments: uv run examples/benchmark.py
"""

import argparse


def main(argv=None) -> None:
    """Parse help before importing GPU dependencies."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    from _support.rca_benchmark import main as run

    run()


if __name__ == "__main__":
    main()
