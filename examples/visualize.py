"""Show every RCA method, including two-stage OPW, in one napari viewer.

Run: uv run examples/visualize.py
The FPM reference is included when mach is installed. --headless computes only.
"""

import argparse


def main(argv=None):
    """Parse help before importing optional GPU/viewer dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Reconstruct without opening napari.")
    args = parser.parse_args(argv)
    from _support.rca_visualization import main as run

    return run(show=not args.headless)


if __name__ == "__main__":
    main()
