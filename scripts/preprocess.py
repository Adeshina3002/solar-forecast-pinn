"""
Build the preprocessed dataset.

Thin wrapper over solar_forecast.preprocess.build_and_cache(). All real logic
lives in the package; this script exists so the pipeline can be invoked from
the command line without thinking about Python paths.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --force      # rebuild even if cache exists
"""

from __future__ import annotations

import argparse
import logging
import sys

from solar_forecast.preprocess import build_and_cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild the parquet cache even if it exists")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show INFO-level log messages from the pipeline")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    out_path = build_and_cache(force=args.force)
    print(f"\n✓ Dataset ready at {out_path}")
    print("  Load it from a notebook with:")
    print("    from solar_forecast import load_dataset")
    print("    df = load_dataset()")
    return 0


if __name__ == "__main__":
    sys.exit(main())
