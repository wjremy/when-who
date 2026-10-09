"""Regenerate Tables S2, S7 and S14 from the synthetic designs in the supplement.

Table S7 reproduces the realisation of code/calibration.py (seed 11) with a
converged Newton fit; Table S14 follows code/history_bias.py (scoring from
zero-based day 30, ties counted one half).

Only numpy and the Python standard library are required. Run from the repository
root with the project runtime, for example:
  python code/supp_tables.py --table S2
"""
from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "code" / "supp_tables_output.md"
MEANS = (0.05, 0.10, 0.20, 0.30, 0.50)
SDS = (0.025, 0.05, 0.10, 0.15, 0.20)
PRINTED_S2 = {
    (0.05, .025): .64, (0.05, .05): .77,
    (0.10, .025): .58, (0.10, .05): .65, (0.10, .10): .79, (0.10, .15): .88,
    (0.20, .025): .54, (0.20, .05): .59, (0.20, .10): .67, (0.20, .15): .76, (0.20, .20): .83,
    (0.30, .025): .53, (0.30, .05): .57, (0.30, .10): .63, (0.30, .15): .70, (0.30, .20): .77,
    (0.50, .025): .53, (0.50, .05): .56, (0.50, .10): .61, (0.50, .15): .67, (0.50, .20): .73,
}
PRINTED_S7 = np.array([[.98, .99, 4.05], [.99, .99, 1.01], [.64, .98, .34]])
PRINTED_S14 = {
    # Printed since 9 Oct 2026 = the seed-20261006 regeneration below (ledger NC).
    60: (.2773, .3571, .4188), 100: (.3555, .4164, .4639),
    180: (.4034, .4572, .4829), 365: (.4360, .4800, .4924),
    730: (.4562, .4905, .4965),
}


def _beta_a0(mu: float, sd: float, grid_n: int = 400_000) -> float:
    """Exact A0 = 1/2 + E|X-X'| / (4 mu (1-mu)) for beta-distributed patient rates.

    The integral is taken on a midpoint grid in u = x**alpha, which removes the
    singularity of the density at zero when alpha < 1 (a uniform grid in x is
    off by 0.004 for mean 0.10 and SD 0.15); for alpha >= 1 the grid is uniform in x.
    """
    var = sd * sd
    concentration = mu * (1 - mu) / var - 1
    alpha, beta = mu * concentration, (1 - mu) * concentration
    if alpha <= 0 or beta <= 0:
        return math.nan
    u = (np.arange(grid_n, dtype=np.float64) + 0.5) / grid_n
    if alpha < 1:
        x = u ** (1 / alpha)
        mass = (1 - x) ** (beta - 1)
    else:
        x = u
        mass = np.exp((alpha - 1) * np.log(x) + (beta - 1) * np.log1p(-x))
    mass /= mass.sum()
    lower_mass = np.cumsum(mass) - mass
    lower_moment = np.cumsum(mass * x) - mass * x
    e_abs = 2 * np.sum(mass * (x * lower_mass - lower_moment))
    return 0.5 + e_abs / (4 * mu * (1 - mu))


def table_s2() -> tuple[str, list[tuple[str, float, float, float]]]:
    rows, comparisons = [], []
    for mu in MEANS:
        cells = []
        for sd in SDS:
            if sd >= 2 * mu:
                cells.append("")
                continue
            var = sd * sd
            if var >= mu * (1 - mu):
                cells.append("")
                continue
            exact = _beta_a0(mu, sd)
            rounded = round(exact + 1e-12, 2)
            cells.append(f"{exact:.4f} ({rounded:.2f})")
            printed = PRINTED_S2.get((mu, sd))
            if printed is not None:
                comparisons.append((f"mu={mu:.2f}, sd={sd:.3f}", rounded, printed, abs(rounded - printed)))
        rows.append("| " + f"{mu:.2f}" + " | " + " | ".join(cells) + " |")
    body = "\n".join([
        "## Table S2 — exact beta rate benchmark",
        "Midpoint quadrature (400,000 points; in u = p**alpha when alpha < 1) for E|p-p'|; cells show four decimals and the two-decimal value in parentheses.",
        "| Mean event rate | SD 0.025 | SD 0.05 | SD 0.10 | SD 0.15 | SD 0.20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |", *rows,
    ])
    body += "\n\nPrinted-value comparison (two-decimal rounded values):\n\n| Cell | Regenerated | Printed | Difference |\n| --- | ---: | ---: | ---: |\n"
    body += "\n".join(f"| {label} | {new:.2f} | {printed:.2f} | {diff:.2f} |" for label,new,printed,diff in comparisons)
    return body, comparisons


def _expit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def _irls(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Unpenalised Newton/IRLS for binary or fractional outcomes."""
    design = np.column_stack((np.ones(len(y)), X))
    weights = np.ones(len(y)) if weights is None else weights
    beta = np.zeros(design.shape[1], dtype=float)
    ym = np.average(y, weights=weights)
    beta[0] = math.log(ym / (1 - ym)) if 0 < ym < 1 else 0.0
    for _ in range(80):
        p = _expit(design @ beta)
        w = weights * np.maximum(p * (1 - p), 1e-10)
        hessian = design.T @ (w[:, None] * design)
        step = np.linalg.solve(hessian + np.eye(len(beta)) * 1e-10,
                               design.T @ (weights * (y - p)))
        beta += step
        if np.max(np.abs(step)) < 2e-8:
            return beta
    raise RuntimeError("IRLS did not converge")


def _slopes(y: np.ndarray, forecast_logit: np.ndarray, n_days: int) -> np.ndarray:
    """Pooled slope, and between- and within-patient slopes from the joint model.

    forecast_logit is a flattened (patients x days) array; the between-patient
    term is the realised mean forecast logit of each patient diary.
    """
    mean = np.repeat(forecast_logit.reshape(-1, n_days).mean(axis=1), n_days)
    pooled = _irls(forecast_logit[:, None], y)[1]
    joint = _irls(np.column_stack((mean, forecast_logit - mean)), y)
    return np.array([pooled, joint[1], joint[2]])


def _s3_printed_example() -> tuple[np.ndarray, float]:
    """The single realisation printed in Table S7 (code/calibration.py, seed 11).

    calibration.py draws the tau = 0.5 setting first and the tau = 1.5 setting
    second from one generator; the table reports the second. Forecast
    probabilities are clipped to [1e-6, 1 - 1e-6] before the logit, as there.
    """
    rng = np.random.default_rng(11)
    K, T, omega, intercept = 400, 300, 0.5, math.log(.15 / .85)
    bound = math.log((1 - 1e-6) / 1e-6)
    for tau in (0.5, 1.5):
        patient = intercept + rng.normal(0, tau, K)
        daily = rng.normal(0, 1, (K, T))
        outcome = (rng.random((K, T)) < _expit(patient[:, None] + omega * daily)).astype(float).ravel()
    rows = [_slopes(outcome, np.clip(patient[:, None] + c * omega * daily, -bound, bound).ravel(), T)
            for c in (.25, 1., 3.)]
    return np.vstack(rows), float(outcome.mean())


def _s3_expected_check() -> tuple[np.ndarray, float]:
    """Outcomes replaced by their expectations: 4,000 patients of 180 days, marginal rate 0.15.

    Same design and seed as the original verification of Supplement S7: the common
    intercept is found by bisection so that the mean risk is 0.15.
    """
    rng = np.random.default_rng(20261005)
    K, T = 4000, 180
    patient0 = rng.normal(0, 1.5, K)
    z = rng.normal(size=(K, T))
    lo, hi = -12., 4.
    for _ in range(70):
        mid = (lo + hi) / 2
        if _expit(mid + patient0[:, None] + .5 * z).mean() > .15:
            hi = mid
        else:
            lo = mid
    patient = patient0 + (lo + hi) / 2
    risk = _expit(patient[:, None] + .5 * z).ravel()
    rows = [_slopes(risk, (patient[:, None] + c * .5 * z).ravel(), T) for c in (.25, 1., 3.)]
    return np.vstack(rows), float(risk.mean())


def table_s7(repetitions: int = 200, seed: int = 73105) -> tuple[str, list[tuple[str, float, float, float]]]:
    """Printed realisation (seed 11), its sampling variation, and the expected-outcome check."""
    labels = ("0.25", "1", "3")
    names = ("pooled", "between", "within")
    printed_example, event_proportion = _s3_printed_example()

    # Sampling variation of the same design over fresh realisations.
    rng = np.random.default_rng(seed)
    K, T, tau, omega = 400, 300, 1.5, 0.5
    intercept = math.log(.15 / .85)
    values = np.empty((repetitions, 3, 3), dtype=float)
    for r in range(repetitions):
        patient = intercept + rng.normal(0, tau, K)
        daily = rng.normal(0, 1, (K, T))
        outcome = (rng.random((K, T)) < _expit(patient[:, None] + omega * daily)).astype(float).ravel()
        for j, scale in enumerate((.25, 1., 3.)):
            values[r, j] = _slopes(outcome, (patient[:, None] + scale * omega * daily).ravel(), T)
    means, sds = values.mean(axis=0), values.std(axis=0, ddof=1)

    expected, mean_risk = _s3_expected_check()
    target = np.array([[1., 1., 4.], [1., 1., 1.], [.62, .99, .33]])

    lines = [
        "## Table S7 — calibration simulation",
        f"Printed realisation: 400 patients × 300 days, tau=1.5, omega=0.5, fixed intercept=logit(0.15), generator seed 11 as in code/calibration.py; realised event proportion {event_proportion:.3f}.",
        "| Daily logit deviation scale | Pooled slope | Between slope | Within slope |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for j, label in enumerate(labels):
        lines.append(f"| {label} | " + " | ".join(f"{v:.4f}" for v in printed_example[j]) + " |")
    lines.extend([
        "",
        f"### Sampling variation ({repetitions} fresh realisations of the same design, seed={seed}; mean (SD))",
        "| Daily logit deviation scale | Pooled slope | Between slope | Within slope |",
        "| ---: | ---: | ---: | ---: |",
    ])
    for j, label in enumerate(labels):
        lines.append(f"| {label} | " + " | ".join(f"{means[j,k]:.3f} ({sds[j,k]:.3f})" for k in range(3)) + " |")
    lines.extend([
        "",
        "### Expected-outcome check",
        f"4,000 patients × 180 days, tau=1.5, omega=0.5, intercept set so that the mean risk is {mean_risk:.4f}; outcomes replaced by conditional risks (seed 20261005, as in the original verification).",
        "| Daily logit deviation scale | Pooled slope | Between slope | Within slope |",
        "| ---: | ---: | ---: | ---: |",
    ])
    for j, label in enumerate(labels):
        lines.append(f"| {label} | " + " | ".join(f"{v:.4f}" for v in expected[j]) + " |")
    lines.extend(["", "### Printed-value comparison", "Both checks are deterministic given their seeds; a value matches if it rounds to the printed two decimals.",
                  "| Quantity | Regenerated | Printed | Difference | Tolerance result |", "| --- | ---: | ---: | ---: | --- |"])
    comps = []
    for kind, regenerated, printed in (("seed 11", printed_example, PRINTED_S7), ("expected", expected, target)):
        for j, label in enumerate(labels):
            for k, name in enumerate(names):
                value, ref = float(regenerated[j, k]), float(printed[j, k])
                match = round(value + 1e-12, 2) == round(ref + 1e-12, 2)
                comps.append((f"{kind}, scale={label}, {name}", value, ref, 0.0 if match else abs(value - ref)))
                lines.append(f"| {kind}, scale={label}, {name} | {value:.4f} | {ref:.2f} | {abs(value - ref):.4f} | "
                             f"{'matches at printed precision' if match else 'does not match at printed precision'} |")
    return "\n".join(lines), comps


def _diary_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Tie-aware AUC in one diary using rank sums."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return math.nan
    order = np.argsort(score, kind="mergesort")
    ss, yy = score[order], y[order]
    starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
    ends = np.r_[starts[1:], len(ss)]
    ranksum = 0.0
    for lo, hi in zip(starts, ends):
        # Average of one-based ranks lo+1 through hi.
        ranksum += yy[lo:hi].sum() * (lo + 1 + hi) / 2
    return float((ranksum - n1 * (n1 + 1) / 2) / (n1 * n0))


def _batch_auc(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tie-aware AUC for a batch of diaries, retaining exact score ties."""
    n_diary, n_days = y.shape
    order = np.argsort(score, axis=1, kind="mergesort")
    sorted_score = np.take_along_axis(score, order, axis=1)
    sorted_y = np.take_along_axis(y, order, axis=1)
    index = np.broadcast_to(np.arange(n_days), (n_diary, n_days))
    # a tie group starts where the score differs from the previous one and ends where it differs from the next
    first = np.ones_like(sorted_score, dtype=bool)
    first[:, 1:] = sorted_score[:, 1:] != sorted_score[:, :-1]
    last = np.ones_like(sorted_score, dtype=bool)
    last[:, :-1] = sorted_score[:, :-1] != sorted_score[:, 1:]
    starts = np.maximum.accumulate(np.where(first, index, 0), axis=1)
    ends = np.minimum.accumulate(np.where(last, index, n_days - 1)[:, ::-1], axis=1)[:, ::-1]
    avg_rank = (starts + ends) * .5 + 1  # one-based midrank of the tie group
    ranksum = np.sum(sorted_y * avg_rank, axis=1)
    positives = y.sum(axis=1)
    pairs = positives * (n_days - positives)
    eligible = pairs > 0
    aucs = np.full(n_diary, np.nan, dtype=float)
    aucs[eligible] = (ranksum[eligible] - positives[eligible] * (positives[eligible] + 1) * .5) / pairs[eligible]
    return aucs, eligible


def _history_block(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-diary AUCs of the three history forecasts for one block of diaries."""
    T = y.shape[1]
    cum = np.cumsum(y, axis=1, dtype=np.int32)
    prefix = np.pad(cum, ((0, 0), (1, 0)))  # prefix[:, t] = events before day t
    day = np.arange(T)
    expanding = prefix[:, :T] / np.maximum(1, day)[None, :]
    roll30 = prefix[:, day] - prefix[:, np.maximum(0, day - 30)]
    roll7 = prefix[:, day] - prefix[:, np.maximum(0, day - 7)]
    # scoring starts at zero-based day 30, as in code/history_bias.py
    return [_batch_auc(y[:, 30:], scores[:, 30:]) for scores in (expanding, roll30, roll7)]


def table_s14(diaries: int = 100_000, seed: int = 20261006, block: int = 5_000) -> tuple[str, list[tuple[str, float, float, float]]]:
    """Monte Carlo within-diary AUCs for prior-history count/rate forecasts."""
    rng = np.random.default_rng(seed)
    lines = [
        "## Table S14 — independent Bernoulli history forecasts",
        f"{diaries:,} diaries per length; Bernoulli(0.15), scoring starts at day 30, seed={seed}; SD of diary AUCs / sqrt(eligible diaries) is the Monte Carlo SE.",
        "| Diary length | Expanding rate | Previous 30-day count | Previous 7-day count |",
        "| ---: | ---: | ---: | ---: |",
    ]
    comparisons = []
    for T in (60, 100, 180, 365, 730):
        collected = [[], [], []]
        for start in range(0, diaries, block):
            y = (rng.random((min(block, diaries - start), T)) < .15).astype(np.uint8)
            for j, (aucs, eligible) in enumerate(_history_block(y)):
                collected[j].append(aucs[eligible])
        per_method = []
        for vals in map(np.concatenate, collected):
            per_method.append((float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(len(vals))), int(len(vals))))
        lines.append(f"| {T} | " + " | ".join(f"{m:.4f} ({se:.5f}; n={n})" for m,se,n in per_method) + " |")
        for j, (mean, se, n) in enumerate(per_method):
            label = f"T={T}, " + ("expanding" if j == 0 else f"previous {30 if j == 1 else 7} days")
            comparisons.append((label, mean, PRINTED_S14[T][j], abs(mean - PRINTED_S14[T][j]), se))
    lines.extend(["", "### Printed-value comparison",
                  "The printed values are themselves Monte Carlo means of 100,000 diaries, so the difference of two independent runs has standard error sqrt(2) x SE; the criterion is 3 x sqrt(2) x SE.",
                  "| Quantity | Regenerated | Printed | Difference | Tolerance result |", "| --- | ---: | ---: | ---: | --- |"])
    for label, mean, printed, diff, se in comparisons:
        lines.append(f"| {label} | {mean:.4f} (SE {se:.5f}) | {printed:.4f} | {diff:.4f} | {'within tolerance' if diff <= 3*math.sqrt(2)*se else 'outside tolerance'} |")
    return "\n".join(lines), comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", choices=("S2", "S7", "S14", "all"), default="all")
    args = parser.parse_args()
    started = time.perf_counter()
    funcs = {"S2": table_s2, "S7": table_s7, "S14": table_s14}
    selected = tuple(funcs) if args.table == "all" else (args.table,)
    sections, reports = [], []
    for name in selected:
        t0 = time.perf_counter()
        section, comp = funcs[name]()
        elapsed = time.perf_counter() - t0
        sections.append(section)
        reports.append((name, elapsed, comp))
    # Preserve previously generated sections when a single table is run, and
    # always write them in supplement order.
    old = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
    blocks: dict[str, str] = {}
    for block in re.split(r"(?m)(?=^## Table S[234] )", old):
        block = block.strip()
        for name in ("S2", "S7", "S14"):
            if block.startswith(f"## Table {name} ") and name not in selected:
                blocks[name] = block
    for name, section in zip(selected, sections):
        blocks[name] = section
    ordered = [blocks[name] for name in ("S2", "S7", "S14") if name in blocks]
    OUTPUT.write_text("\n\n".join(ordered) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    for name, elapsed, comp in reports:
        print(f"{name}: {elapsed:.2f}s")
        if name == "S2":
            bad = [x for x in comp if x[1] != x[2]]
            diff = max((x[3] for x in comp), default=0)
            print(f"  rounded cells differing: {len(bad)}/{len(comp)}; maximum absolute difference={diff:.2f}")
        elif name == "S7":
            bad = [x for x in comp if x[3] > 0]
            print(f"  values not matching at printed precision: {len(bad)}/{len(comp)}")
        else:
            bad = [x for x in comp if x[3] > 3 * math.sqrt(2) * x[4]]
            print(f"  outside tolerance: {len(bad)}/{len(comp)}")
    print(f"Total elapsed: {time.perf_counter()-started:.2f}s")


if __name__ == "__main__":
    main()
