"""Fetch independent ENTSO-E imbalance outcomes with explicit coverage reporting."""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from outcomes import OperationalOutcomeFetcher, save_outcomes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--start", default="20230101")
parser.add_argument("--end", default="20240101")
parser.add_argument("--zone", default="DE_LU")
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "data" / "external" / "operational_outcomes.parquet",
)

if __name__ == "__main__":
    args = parser.parse_args()
    frame, coverage = OperationalOutcomeFetcher().fetch(args.start, args.end, args.zone)
    save_outcomes(frame, coverage, args.output)
    print(coverage)
    if coverage["coverage_fraction"] == 0:
        raise SystemExit("No operational outcome coverage was returned")
