"""Compare conventional, two-stage and baseband beamformers in napari."""

import argparse


def main(argv=None):
    """Parse help before importing optional GPU/viewer dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        choices=("targets", "mixture", "filtered"),
        help="Five point targets (default) or 200 frames of tissue, blood and noise, before or after clutter filtering.",
    )
    parser.add_argument(
        "--view",
        choices=("all", "opw", "st-sw", "dmas"),
        help="Show all methods (default) or focus on one family. The St-SW view includes similarity weights.",
    )
    files = parser.add_mutually_exclusive_group()
    files.add_argument("--save", metavar="FILE", help="Save the reconstructed volumes for later viewing.")
    files.add_argument("--load", metavar="FILE", help="Open a saved comparison without CUDA.")
    parser.add_argument("--headless", action="store_true", help="Run without opening napari.")
    args = parser.parse_args(argv)
    if args.load and (args.scene or args.view):
        parser.error("--load uses the scene and view in the saved file")

    if args.load:
        from _support.rca_visualization import open_saved

        return open_saved(args.load, show=not args.headless)
    from _support.rca_visualization import main as reconstruct

    return reconstruct(show=not args.headless, scene=args.scene or "targets", view=args.view or "all", save=args.save)


if __name__ == "__main__":
    main()
