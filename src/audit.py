"""CLI tool to audit dataset authenticity, verify Kaggle source, and check autocorrelation."""

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import audit_dataset


def main():
    parser = argparse.ArgumentParser(description="Audit AQI dataset authenticity and autocorrelation.")
    parser.add_argument("--data-path", type=str, default=None, help="Optional direct path to CSV file")
    parser.add_argument("--city", type=str, default="Delhi", help="City name (default: Delhi)")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        default=False,
        help="Allow fallback to synthetic dataset (default: False)",
    )
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        default=False,
        help="Disable automatic download of real Kaggle dataset if missing",
    )
    args = parser.parse_args()

    audit_info = audit_dataset(
        data_path=args.data_path,
        city=args.city,
        allow_synthetic=args.allow_synthetic,
        auto_download=not args.no_auto_download,
        verbose=True,
    )
    return audit_info


if __name__ == "__main__":
    main()
