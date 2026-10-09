"""Command-line interface for CSV-based withinpatient reports."""
from __future__ import annotations

import argparse

import pandas as pd

from .report import evaluate, to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise pooled and within-patient forecast performance.")
    parser.add_argument("file", help="Input CSV file")
    parser.add_argument("--patient", default="patient", help="Patient identifier column")
    parser.add_argument("--y", default="y", help="Binary outcome column")
    parser.add_argument("--forecast", default="forecast", help="Probability forecast column")
    parser.add_argument("--time", help="Optional time column")
    parser.add_argument("--threshold", type=float, help="Optional probability threshold")
    parser.add_argument("--runin", type=float, help="Optional run-in period in time units")
    parser.add_argument("--prior-rate", type=float, help="Optional external prior event rate for the expanding reference")
    parser.add_argument("--prior-weight", type=float, default=1.0, help="Pseudo-count weight for --prior-rate (default: 1)")
    parser.add_argument("--permutations", type=int, default=2000, help="Within-patient forecast shuffles (default: 2000)")
    parser.add_argument("--bootstrap", type=int, default=0, help="Whole-patient bootstrap resamples; 0 disables intervals")
    args = parser.parse_args()
    frame = pd.read_csv(args.file)
    print(to_markdown(evaluate(frame, patient=args.patient, y=args.y, forecast=args.forecast,
                               time=args.time, threshold=args.threshold, runin_days=args.runin,
                               prior_rate=args.prior_rate, prior_weight=args.prior_weight,
                               n_perm=args.permutations, n_boot=args.bootstrap)))


if __name__ == "__main__":
    main()
