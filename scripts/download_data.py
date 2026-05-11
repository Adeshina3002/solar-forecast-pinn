"""
Download OPSD time series and weather data packages.

Pinned to specific versioned releases so the dataset is reproducible — if OPSD
publishes a new version, this script keeps fetching the version the model was
trained on. To upgrade, bump the constants below and re-run.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --force  # re-download even if files exist
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen, urlretrieve

# Pin to specific OPSD releases. These are the latest stable as of project start.
# Bump these only when you intentionally want fresher data and are prepared to
# re-validate the data quality report.
TIME_SERIES_VERSION = "2020-10-06"
WEATHER_VERSION = "2020-09-16"

BASE_URL = "https://data.open-power-system-data.org"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DATASETS = {
    "time_series": {
        "version": TIME_SERIES_VERSION,
        "filename": "time_series_60min_singleindex.csv",
        "url": f"{BASE_URL}/time_series/{TIME_SERIES_VERSION}/time_series_60min_singleindex.csv",
        "datapackage": f"{BASE_URL}/time_series/{TIME_SERIES_VERSION}/datapackage.json",
    },
    "weather": {
        "version": WEATHER_VERSION,
        "filename": "weather_data.csv",
        "url": f"{BASE_URL}/weather_data/{WEATHER_VERSION}/weather_data.csv",
        "datapackage": f"{BASE_URL}/weather_data/{WEATHER_VERSION}/datapackage.json",
    },
}


# OPSD is inconsistent about which hash they publish: time_series uses SHA-256,
# weather uses MD5. We detect by hex length — there's no overlap.
_HASH_ALGOS = {32: "md5", 40: "sha1", 64: "sha256"}


def hash_of(path: Path, algo: str, chunk_size: int = 1 << 20) -> str:
    """Compute `algo` digest of a file in 1 MiB chunks (memory-safe for big files)."""
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def expected_hash(datapackage_url: str, filename: str) -> tuple[str, str] | None:
    """
    Pull the expected hash for `filename` from OPSD's datapackage.json.
    Returns (algorithm, hex_digest) or None if not found / not fetchable.
    """
    try:
        with urlopen(datapackage_url, timeout=30) as response:
            metadata = json.loads(response.read())
    except Exception as e:
        print(f"  ⚠ Could not fetch datapackage.json ({e}); skipping hash check")
        return None

    for resource in metadata.get("resources", []):
        if resource.get("path", "").endswith(filename):
            digest = resource.get("hash", "")
            algo = _HASH_ALGOS.get(len(digest))
            if algo is None:
                print(f"  ⚠ Unrecognised hash length ({len(digest)} chars); skipping check")
                return None
            return algo, digest
    return None


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    """Pretty progress bar for urlretrieve — keeps users sane on 200 MB downloads."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, 100.0 * downloaded / total_size)
        mb_done = downloaded / (1 << 20)
        mb_total = total_size / (1 << 20)
        bar = "█" * int(pct / 2.5) + "░" * (40 - int(pct / 2.5))
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}% ({mb_done:6.1f}/{mb_total:6.1f} MB)")
        sys.stdout.flush()
        if downloaded >= total_size:
            sys.stdout.write("\n")


def download_one(name: str, spec: dict, *, force: bool = False) -> None:
    """Fetch a single dataset, verify integrity, skip if already present."""
    target = DATA_DIR / spec["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        print(f"✓ {name}: already at {target} (use --force to re-download)")
        return

    print(f"↓ {name} (version {spec['version']})")
    print(f"  from {spec['url']}")
    urlretrieve(spec["url"], target, reporthook=_progress)

    # Best-effort hash verification — failing the lookup shouldn't abort the run,
    # but a mismatch should, since it means we have corrupted or wrong-version data.
    result = expected_hash(spec["datapackage"], spec["filename"])
    if result:
        algo, expected = result
        actual = hash_of(target, algo)
        if actual.lower() == expected.lower():
            print(f"  ✓ {algo.upper()} verified")
        else:
            target.unlink()  # poison-pill: don't leave bad data on disk
            raise RuntimeError(
                f"Hash mismatch for {spec['filename']}: "
                f"expected {algo} {expected}, got {actual}. File deleted."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-download files even if they already exist")
    args = parser.parse_args()

    print(f"Downloading OPSD data to {DATA_DIR}\n")
    for name, spec in DATASETS.items():
        download_one(name, spec, force=args.force)
        print()

    print("Done. Next step: python scripts/data_quality_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())