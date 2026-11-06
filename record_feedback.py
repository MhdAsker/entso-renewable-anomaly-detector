"""Record operator feedback for a persisted alert event."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from alerts import record_feedback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("event_id")
parser.add_argument("label", choices=["true_positive", "false_positive", "uncertain"])
parser.add_argument("--notes", default="")
parser.add_argument("--alerts", type=Path, default=ROOT / "runtime" / "alerts.csv")

if __name__ == "__main__":
    args = parser.parse_args()
    record_feedback(args.alerts, args.event_id, args.label, args.notes)
    print(f"Recorded {args.label} for {args.event_id}")
