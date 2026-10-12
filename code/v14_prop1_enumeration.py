"""Exact check of the extension of Proposition 1 (v14, B4).

Setting of Proposition 1: a record of T scored days with n events
(0 < n < T), every arrangement of the events equally likely, and a forecast
for each day that is a fixed rule of the other days' outcomes and never
decreases when another day becomes an event day.

For every T from 2 to MAX_T, every n and every rule below, the script
enumerates all C(T, n) arrangements and evaluates, in exact rational
arithmetic (fractions.Fraction), with q = n / T, M the number of alarms
(forecast >= cutoff) and every cutoff that changes the alarms:

  auc      E[AUC] <= 1/2                                 (Proposition 1)
  youden   E[sensitivity + specificity - 1] <= 0         (Proposition 1)
  i        E[true alarms] <= q E[M]
  ii       E[true non-alarms] <= (1 - q) E[T - M]
  iii      E[net benefit at tau] <= (E[M] / T) (q - tau) / (1 - tau)   (iii_sharp)
           and <= max(0, (q - tau) / (1 - tau))                        (iii)
           for tau equal to the cutoff, tau = q and tau = 1/20, ..., 19/20
  iv       E[Brier] >= q (1 - q) + mean_t E[(f_t - q)^2]
  v        E[AP] <= R(T, n), the expected AP of a uniformly random strict
           ranking; AP with tied forecasts grouped (v_grouped) and with ties
           broken at random, in expectation (v_random)
  vi       E[PPV | M > 0] <= q   and   E[NPV | M < T] <= 1 - q

Each rule is first verified exhaustively, on all 2^T outcome vectors, to
ignore the day's own outcome and to be nondecreasing in every other day.

Output: results/v14/B4_prop1_enumeration.csv and .md.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_T = 12
HALF = Fr(1, 2)


# ---------------------------------------------------------------- rules
# A rule maps (y, t, T) to the forecast for day t; y is a tuple of 0/1.

def rate_so_far_smoothed(y, t, T):
    return Fr(1 + sum(y[:t]), 2 + t)


def rate_so_far_raw(y, t, T):
    return Fr(sum(y[:t]), t) if t else HALF


def rate_other_days(y, t, T):
    return Fr(sum(y) - y[t], T - 1)


def rate_later_days(y, t, T):
    return Fr(1 + sum(y[t + 1:]), 1 + T - t)


def previous_day(y, t, T):
    return Fr(1, 10) + Fr(8, 10) * (y[t - 1] if t else 0)


def window(w):
    def rule(y, t, T):
        past = y[max(0, t - w):t]
        return Fr(1 + sum(past), 2 + len(past))
    rule.__name__ = f"window_{w}_smoothed"
    return rule


def two_of_last_three(y, t, T):
    return Fr(int(sum(y[max(0, t - 3):t]) >= 2))


def any_earlier_event(y, t, T):
    return Fr(int(any(y[:t])))


def exponential_weights(y, t, T):
    return sum((Fr(1, 2 ** (t - s)) * y[s] for s in range(t)), Fr(0))


def or_and(y, t, T):
    others = [(t + j) % T for j in range(1, T)]
    if len(others) >= 3:
        a, b, c = others[:3]
        hit = y[a] or (y[b] and y[c])
    else:
        hit = any(y[j] for j in others)
    return Fr(15, 100) + Fr(70, 100) * int(bool(hit))


def random_weighted_threshold(seed):
    cache = {}

    def rule(y, t, T):
        if (T, t) not in cache:
            rng = np.random.default_rng([seed, T, t])
            weights = [int(v) for v in rng.integers(1, 10, size=T)]
            cut = int(rng.integers(1, max(2, sum(weights) - weights[t])))
            cache[(T, t)] = (weights, cut)
        weights, cut = cache[(T, t)]
        total = sum(weights[j] * y[j] for j in range(T) if j != t)
        return Fr(1, 10) + Fr(8, 10) * int(total >= cut)
    rule.__name__ = f"random_weighted_threshold_{seed}"
    return rule


def random_weighted_sum(seed):
    """Graded monotone rule with few ties: a random positive weighted sum."""
    cache = {}

    def rule(y, t, T):
        if (T, t) not in cache:
            rng = np.random.default_rng([seed, T, t, 7])
            cache[(T, t)] = [int(v) for v in rng.integers(1, 50, size=T)]
        weights = cache[(T, t)]
        total = sum(weights[j] for j in range(T) if j != t)
        return Fr(sum(weights[j] * y[j] for j in range(T) if j != t), total)
    rule.__name__ = f"random_weighted_sum_{seed}"
    return rule


def random_dnf(seed):
    cache = {}

    def rule(y, t, T):
        if (T, t) not in cache:
            rng = np.random.default_rng([seed, T, t, 11])
            others = [j for j in range(T) if j != t]
            clauses = []
            for _ in range(3):
                k = int(rng.integers(1, min(4, len(others)) + 1))
                clauses.append([int(v) for v in rng.choice(others, size=k, replace=False)])
            cache[(T, t)] = clauses
        hit = any(all(y[j] for j in clause) for clause in cache[(T, t)])
        return Fr(1, 10) + Fr(8, 10) * int(hit)
    rule.__name__ = f"random_dnf_{seed}"
    return rule


def constant(y, t, T):
    return Fr(3, 10)


def fixed_ranking(y, t, T):
    return Fr(t + 1, T + 1)


RULES = [
    rate_so_far_smoothed, rate_so_far_raw, rate_other_days, rate_later_days,
    previous_day, window(2), window(4), two_of_last_three, any_earlier_event,
    exponential_weights, or_and,
    random_weighted_threshold(1), random_weighted_threshold(2), random_weighted_threshold(3),
    random_weighted_sum(1), random_weighted_sum(2), random_weighted_sum(3),
    random_dnf(1), random_dnf(2),
    constant, fixed_ranking,
]
OUTCOME_FREE = {"constant", "fixed_ranking"}


# ---------------------------------------------------------------- helpers

def random_ranking_ap(T: int, n: int) -> Fr:
    return sum((Fr(1) + Fr((r - 1) * (n - 1), T - 1)) / r for r in range(1, T + 1)) / T


def ap_both(y_row, rank_row, n):
    """AP of one record with ties grouped, and its expectation with ties broken at random."""
    order = sorted(range(len(y_row)), key=lambda i: -rank_row[i])
    grouped, randomised = Fr(0), Fr(0)
    tp, k, i, T = 0, 0, 0, len(order)
    while i < T:
        j = i
        while j < T and rank_row[order[j]] == rank_row[order[i]]:
            j += 1
        g = j - i
        h = sum(y_row[order[m]] for m in range(i, j))
        if h:
            grouped += Fr(tp + h, k + g) * h
            for pos in range(1, g + 1):
                inside = Fr((pos - 1) * (h - 1), g - 1) if g > 1 else Fr(0)
                randomised += Fr(h, g) * (tp + 1 + inside) / (k + pos)
        tp += h
        k += g
        i = j
    return grouped / n, randomised / n


class Tally:
    def __init__(self):
        self.cases = defaultdict(int)
        self.equal = defaultdict(int)
        self.worst = {}
        self.closest = {}
        self.violations = defaultdict(list)

    def add(self, check, gap, context):
        """gap = (side that must be smaller) - (side that must be larger); a violation is gap > 0."""
        self.cases[check] += 1
        if gap == 0:
            self.equal[check] += 1
        if check not in self.worst or gap > self.worst[check][0]:
            self.worst[check] = (gap, context)
        if gap < 0 and (check not in self.closest or gap > self.closest[check][0]):
            self.closest[check] = (gap, context)
        if gap > 0:
            self.violations[check].append((gap, context))


# ---------------------------------------------------------------- enumeration

def verify_rule(ranks: np.ndarray, T: int, name: str) -> None:
    masks = np.arange(1 << T)
    for t in range(T):
        off = masks[(masks >> t) & 1 == 0]
        if not np.array_equal(ranks[off, t], ranks[off | (1 << t), t]):
            raise AssertionError(f"{name}, T={T}: forecast for day {t} uses its own outcome")
        for j in range(T):
            if j == t:
                continue
            off = masks[(masks >> j) & 1 == 0]
            if np.any(ranks[off, t] > ranks[off | (1 << j), t]):
                raise AssertionError(f"{name}, T={T}: day {t} decreases when day {j} becomes an event")


def run(max_t: int = MAX_T):
    overall, per_rule = Tally(), defaultdict(Tally)
    arrangements = 0
    grid = [Fr(k, 20) for k in range(1, 20)]

    for T in range(2, max_t + 1):
        masks = np.arange(1 << T)
        Y_all = ((masks[:, None] >> np.arange(T)[None, :]) & 1).astype(np.int64)
        counts_all = Y_all.sum(axis=1)
        for rule in RULES:
            name = rule.__name__
            values = [[rule(tuple(int(v) for v in y), t, T) for t in range(T)] for y in Y_all]
            unique = sorted({v for row in values for v in row})
            rank_of = {v: i for i, v in enumerate(unique)}
            ranks = np.array([[rank_of[v] for v in row] for row in values], dtype=np.int64)
            verify_rule(ranks, T, name)

            for n in range(1, T):
                rows = np.flatnonzero(counts_all == n)
                Y, R = Y_all[rows], ranks[rows]
                m = len(rows)
                q = Fr(n, T)
                ctx = (name, T, n)
                tallies = (overall, per_rule[name])
                if rule is RULES[0]:
                    arrangements += m

                def add(check, gap, extra=None):
                    for tally in tallies:
                        tally.add(check, gap, ctx + ((extra,) if extra is not None else ()))

                # AUC: twice the concordant pairs plus the tied pairs, over 2 n (T - n).
                num = 0
                for y_row, r_row in zip(Y, R):
                    pos, neg = r_row[y_row == 1], r_row[y_row == 0]
                    diff = pos[:, None] - neg[None, :]
                    num += 2 * int((diff > 0).sum()) + int((diff == 0).sum())
                add("auc", Fr(num, 2 * n * (T - n) * m) - HALF)

                # Brier score, exactly, from counts of (outcome, forecast value).
                brier, spread = Fr(0), Fr(0)
                for r in np.unique(R):
                    v = unique[int(r)]
                    c1 = int(((R == r) & (Y == 1)).sum())
                    c0 = int(((R == r) & (Y == 0)).sum())
                    brier += c1 * (1 - v) ** 2 + c0 * v ** 2
                    spread += (c0 + c1) * (v - q) ** 2
                brier, spread = brier / (m * T), spread / (m * T)
                add("iv", q * (1 - q) + spread - brier)

                # Average precision, both conventions for ties.
                ap_g, ap_r = Fr(0), Fr(0)
                for y_row, r_row in zip(Y.tolist(), R.tolist()):
                    g, r = ap_both(y_row, r_row, n)
                    ap_g += g
                    ap_r += r
                target = random_ranking_ap(T, n)
                add("v_grouped", ap_g / m - target)
                add("v_random", ap_r / m - target)

                # Alarm rules: forecast >= cutoff, for every cutoff that changes the alarms.
                present = np.unique(R)
                for r in list(present) + [int(present.max()) + 1]:
                    A = R >= r
                    M = A.sum(axis=1)
                    TP = (A & (Y == 1)).sum(axis=1)
                    TN = (~A & (Y == 0)).sum(axis=1)
                    sM, sTP, sTN = int(M.sum()), int(TP.sum()), int(TN.sum())
                    cut = unique[int(r)] if int(r) < len(unique) else None
                    add("i", Fr(sTP, m) - q * Fr(sM, m), cut)
                    add("ii", Fr(sTN, m) - (1 - q) * Fr(m * T - sM, m), cut)
                    add("youden", Fr(sTP, m * n) + Fr(sTN, m * (T - n)) - 1, cut)

                    has = M > 0
                    if has.any():
                        pairs, cnt = np.unique(np.stack([TP[has], M[has]]), axis=1, return_counts=True)
                        ppv = sum(Fr(int(a), int(b)) * int(c) for (a, b), c in zip(pairs.T, cnt)) / int(has.sum())
                        add("vi_ppv", ppv - q, cut)
                    has = M < T
                    if has.any():
                        pairs, cnt = np.unique(np.stack([TN[has], T - M[has]]), axis=1, return_counts=True)
                        npv = sum(Fr(int(a), int(b)) * int(c) for (a, b), c in zip(pairs.T, cnt)) / int(has.sum())
                        add("vi_npv", npv - (1 - q), cut)

                    taus = set(grid) | {q}
                    if cut is not None and 0 < cut < 1:
                        taus.add(cut)
                    for tau in taus:
                        w = tau / (1 - tau)
                        nb = (Fr(sTP, m) - w * Fr(sM - sTP, m)) / T
                        sharp = Fr(sM, m * T) * (q - tau) / (1 - tau)
                        add("iii_sharp", nb - sharp, (cut, tau))
                        add("iii", nb - max(Fr(0), (q - tau) / (1 - tau)), (cut, tau))
        print(f"T={T} done", flush=True)
    return overall, per_rule, arrangements


CHECKS = [
    ("auc", "E[AUC] <= 1/2 (Proposition 1)"),
    ("youden", "E[sensitivity + specificity - 1] <= 0 (Proposition 1)"),
    ("i", "(i) E[true alarms] <= (n/T) E[alarms]"),
    ("ii", "(ii) E[true non-alarms] <= (1 - n/T) E[non-alarms]"),
    ("iii_sharp", "(iii) E[net benefit] <= (E[alarms]/T)(q - tau)/(1 - tau)"),
    ("iii", "(iii) E[net benefit] <= that of always or never alarming"),
    ("iv", "(iv) E[Brier] >= (n/T)(1 - n/T) + mean E[(f_t - n/T)^2]"),
    ("v_grouped", "(v) E[AP] <= random-ranking value, ties grouped"),
    ("v_random", "(v) E[AP] <= random-ranking value, ties broken at random"),
    ("vi_ppv", "(vi) E[PPV | at least one alarm] <= n/T"),
    ("vi_npv", "(vi) E[NPV | at least one day without alarm] <= 1 - n/T"),
]


def main() -> None:
    max_t = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_T
    overall, per_rule, arrangements = run(max_t)
    out_dir = ROOT / "results" / "v14"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["rule,check,cases,violations,equalities,largest_gap_exact,largest_gap,context_of_largest_gap,"
             "closest_strict_gap,context_of_closest_strict_gap"]
    for name, tally in [("ALL", overall)] + [(r.__name__, per_rule[r.__name__]) for r in RULES]:
        for key, _ in CHECKS:
            gap, context = tally.worst[key]
            near, near_context = tally.closest.get(key, (None, None))
            lines.append(f'{name},{key},{tally.cases[key]},{len(tally.violations[key])},'
                         f'{tally.equal[key]},{gap},{float(gap):.6g},"{context}",'
                         f'{"" if near is None else format(float(near), ".6g")},"{near_context}"')
    (out_dir / "B4_prop1_enumeration.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    monotone = [r.__name__ for r in RULES if r.__name__ not in OUTCOME_FREE]
    md = [
        "# B4. Exact check of the extension of Proposition 1",
        "",
        "Script: `code/v14_prop1_enumeration.py`. Numbers per rule: `results/v14/B4_prop1_enumeration.csv`.",
        "",
        f"Exact enumeration in rational arithmetic: every record length T from 2 to {max_t}, every event count "
        f"0 < n < T, all C(T, n) arrangements ({arrangements:,} arrangements per rule), {len(RULES)} forecast rules, "
        "and for the alarm-based measures every cutoff that changes the alarms. Each rule was first verified on all "
        "2^T outcome vectors to ignore the day's own outcome and never to decrease when another day becomes an event day.",
        "",
        "Rules (" + str(len(monotone)) + " that respond to outcomes, plus a constant and a fixed ranking as equality cases): "
        + ", ".join(f"`{name}`" for name in monotone) + "; `constant`, `fixed_ranking`.",
        "",
        "A case is one combination of T, n, rule and, where it applies, cutoff and tau. Gap = the side that must be "
        "smaller minus the side that must be larger; a violation is a gap above zero.",
        "",
        "| Statement | Cases | Violations | Equalities | Strict cases | Gap closest to zero among strict cases |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in CHECKS:
        near, _ = overall.closest[key]
        strict = overall.cases[key] - overall.equal[key] - len(overall.violations[key])
        md.append(f"| {label} | {overall.cases[key]:,} | {len(overall.violations[key])} | "
                  f"{overall.equal[key]:,} | {strict:,} | {float(near):.3g} |")
    md.append("")
    any_violation = False
    for key, label in CHECKS:
        for gap, context in sorted(overall.violations[key], reverse=True)[:5]:
            any_violation = True
            md.append(f"- COUNTEREXAMPLE {label}: gap {gap} ({float(gap):.6g}) at (rule, T, n, cutoff[, tau]) = {context}")
    if not any_violation:
        md.append("No counterexample to any statement.")
    md += [
        "",
        "Statements (i), (ii) and the sensitivity-plus-specificity statement of Proposition 1 are one inequality: "
        "E[true non-alarms] - (1 - q) E[non-alarms] = E[true alarms] - q E[alarms], and E[sensitivity + specificity - 1] "
        "is that quantity divided by n (1 - q). Their cases, equalities and signs therefore coincide.",
        "",
        "Equalities occur where the alarms or the ranking do not depend on the outcomes (all or no days alarmed, "
        "the constant forecast, the fixed ranking, forecasts that are constant for the given T and n) and, for the bound by "
        "always or never alarming, where tau = q. With ties grouped, a constant forecast has average precision n/T, "
        "below the random-ranking value; with ties broken at random it has exactly that value.",
    ]
    (out_dir / "B4_prop1_enumeration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
