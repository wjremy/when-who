"""Command-line interface for the tailored within-patient report."""
from __future__ import annotations

import argparse

import pandas as pd

from .evaluation import evaluate, to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score patient forecasts against within-patient permutation chance."
    )
    parser.add_argument("file", help="Input CSV file")
    parser.add_argument("--patient", default="patient", help="Patient identifier column")
    parser.add_argument("--time", default="time", help="Time column")
    parser.add_argument("--outcome", default="y", help="Binary outcome column")
    parser.add_argument("--forecast", default="forecast", help="Forecast probability column")
    parser.add_argument("--runin", type=float, help="Time cutoff; earlier rows supply history only")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: cohort scored event rate)")
    parser.add_argument("--rebuild", choices=("running_rate", "window_count"),
                        help="Rebuild an outcome-history forecast in each shuffle")
    parser.add_argument("--window", type=int, help="Window width for --rebuild window_count")
    parser.add_argument("--window-count", action="store_true",
                        help="Return counts instead of rates for window_count; use ranking metrics")
    parser.add_argument("--yardstick", help="Reference column, patient_rate, or running_rate")
    parser.add_argument("--permutations", type=int, default=200,
                        help="Within-patient outcome shuffles (default: 200)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    args = parser.parse_args()
    frame = pd.read_csv(args.file)
    rebuild = args.rebuild
    if rebuild == "window_count":
        if args.window is None:
            parser.error("--rebuild window_count requires --window")
        rebuild = {"name": "window_count", "w": args.window, "as_rate": not args.window_count}
    elif args.window is not None or args.window_count:
        parser.error("--window and --window-count require --rebuild window_count")
    print(to_markdown(evaluate(
        frame, patient=args.patient, time=args.time, outcome=args.outcome,
        forecast=args.forecast, runin=args.runin, threshold=args.threshold,
        rebuild=rebuild, n_perm=args.permutations, seed=args.seed,
        yardstick=args.yardstick,
    )))


if __name__ == "__main__":
    main()
