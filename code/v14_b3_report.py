"""Write results/v14/B3_simulations.md from the three B3 result files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v14"


def span(series) -> str:
    return f"{series.min():.2f} to {series.max():.2f}"


def main() -> None:
    a = pd.read_csv(OUT / "B3a_who_when.csv")
    b = pd.read_csv(OUT / "B3b_refit.csv")
    c = pd.read_csv(OUT / "B3c_shared_course.csv")
    L = [
        "# B3. Reruns of three simulations of the evidence audit (12 Oct 2026)", "",
        "Clean scripts, written and reviewed by Claude; the agent versions in `work/2026-10-12_evidence/sim/` "
        "were used only for the designs. All seeds derive from 20261012. Numbers are means over replications; "
        "Monte Carlo standard errors are in the CSV files. This file is written by `code/v14_b3_report.py`.", "",
        "## B3a. Who and when in the pooled, per-patient and per-day AUC", "",
        "Script `code/v14_who_when_auc.py`, numbers `results/v14/B3a_who_when.csv` "
        f"(48 cells, {int(a.replications.iloc[0])} cohorts each; largest standard error "
        f"{a[[k for k in a.columns if k.endswith('_se')]].max().max():.3f}).", "",
        "Design. K patients by T days. Patient levels are Beta with mean 0.20 and SD 0.05 or 0.15. Outcomes are "
        "independent with logit risk = logit(level) + beta x + d: x is a daily signal (standard normal, independent "
        "over patients and days, no patient level), d a day effect shared by all patients (normal, SD 0 or 1). "
        "Forecasts: who only (the patient's level), when only (x), shared day only (d), both (the true risk). "
        "beta = 0.5 (weak) or 1.5 (strong); with beta = 0 the when-only forecast has no skill. Per-patient and "
        "per-day AUC: mean over patients or days with both outcomes.", "",
        "K = 100, T = 365 (all 48 cells, with K = 10 and T = 30, are in the CSV):", "",
        "| Level SD | Daily signal | Shared day SD | Forecast | Pooled | Per patient | Per day |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    names = {"who": "who only", "when": "when only", "day": "shared day only", "both": "both (true risk)"}
    signal = {0.0: "none", 0.5: "weak", 1.5: "strong"}
    for _, r in a[(a.K == 100) & (a["T"] == 365)].iterrows():
        for f in names:
            if (f == "day" and r.day_sd == 0) or (f == "when" and r.beta == 0):
                continue
            L.append(f"| {r.level_sd:.2f} | {signal[r.beta]} | {r.day_sd:.0f} | {names[f]} | "
                     f"{r[f + '_pooled']:.3f} | {r[f + '_patient']:.3f} | {r[f + '_day']:.3f} |")

    x = a[a.beta > 0].copy()
    x["share"] = (x.when_pooled - 0.5) / (x.when_patient - 0.5)
    nd, wd = x[x.day_sd == 0], x[x.day_sd == 1]
    who0, who1 = a[a.day_sd == 0], a[a.day_sd == 1]
    pick = lambda K: a[(a.K == K) & (a["T"] == 365) & (a.level_sd == 0.15) & (a.beta == 1.5)
                       & (a.day_sd == 0)].iloc[0]
    day_only = a[(a.day_sd == 1) & (a.beta == 0)]
    L += [
        "", "Readings over all 48 cells.", "",
        "- A forecast that knows only when keeps most of its per-patient AUC in the pooled AUC. Share of the "
        "per-patient excess over 0.5 that the pooled AUC retains, (pooled - 0.5) / (per patient - 0.5): "
        f"{span(nd[nd.level_sd == 0.05].share)} with level SD 0.05 and {span(nd[nd.level_sd == 0.15].share)} with "
        f"level SD 0.15 (no shared day effect, all sizes, both strengths); {span(wd.share)} with a shared day "
        "effect. It does not shrink with the number of patients: strong signal, level SD 0.15, T = 365, "
        f"K = 10 against K = 100: pooled {pick(10).when_pooled:.3f} against {pick(100).when_pooled:.3f} "
        f"(per patient {pick(10).when_patient:.3f} and {pick(100).when_patient:.3f}).",
        "- A forecast that knows only who scores 0.500 per patient in every cell. Pooled, without a shared day "
        f"effect, it scores {span(who0[who0.level_sd == 0.05].who_pooled)} (level SD 0.05) and "
        f"{span(who0[who0.level_sd == 0.15].who_pooled)} (level SD 0.15). Its pooled AUC is then within "
        f"{(who0.who_pooled - who0.who_day).abs().max():.3f} of its per-day AUC (K = 100: within "
        f"{(who0[who0.K == 100].who_pooled - who0[who0.K == 100].who_day).abs().max():.3f}); with a shared day "
        f"effect the pooled value is below the per-day value by up to {(who1.who_day - who1.who_pooled).max():.3f}.",
        "- A forecast that knows only a day effect shared by all patients scores "
        f"{span(day_only.day_pooled)} pooled and {span(day_only.day_patient)} per patient when there is no other "
        "signal, and 0.500 per day in every cell (all patients are tied within a day).",
        "- So 'the pooled AUC is close to the per-day AUC' holds for a who-only forecast without shared time "
        "variation, not in general. And the pooled AUC does not measure 'almost nothing of when': it takes up "
        "who, when and a shared course alike and cannot tell them apart.", "",
        "Difference from the agent's version: the day effect has SD 1 on the logit scale (agent: 2.5, with which "
        "the shared-day forecast reached 0.87 pooled and 0.90 per patient), and level SD 0.15 is the value of the "
        "task (the agent's 'high' spread was smaller: who-only pooled 0.71). The qualitative results are the same; "
        "quote the numbers of this file.", "",
        "## B3b. A model fitted on the scored record: frozen against refitted coefficients", "",
        "Script `code/v14_refit_shuffle.py`, numbers `results/v14/B3b_refit.csv`.", "",
        "Design. Records of independent days with risk 0.25 (no forecast has skill), "
        f"{int(b.records.max()):,} records per length, {int(b.shuffles.iloc[0])} shuffles per record. Pipeline: five "
        "rotating folds within the patient (day index modulo 5); for each fold a logistic regression is fitted on "
        "the other four folds, with a noise covariate and the patient's event rate among the other training days "
        "as inputs (ridge penalty 1 on the two slopes), and predicts the held-out days. Frozen: the coefficients "
        "fitted on the observed record are kept in every shuffle and only the outcome-derived input is rebuilt. "
        "Refitted: the whole pipeline is run again on every shuffled record. One-sided p = (1 + number of shuffled "
        "AUCs at or above the observed) / (shuffles + 1).", "",
        "| Record length | Shuffle | Records | Observed per-patient AUC | Mean shuffled AUC | Excess (SE) | Rejections at 5% |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in b.iterrows():
        L.append(f"| {r['T']} | {r.shuffle_scope} | {r.records:,} | {r.auc_observed_mean:.3f} | "
                 f"{r.auc_shuffled_mean:.3f} | {r.excess_mean:+.3f} ({r.excess_se:.3f}) | {100 * r.rejection_rate:.1f}% |")
    fz, rf = b[b.shuffle_scope == "frozen"], b[b.shuffle_scope == "refit"]
    L += [
        "", "Readings. The no-skill pipeline itself scores "
        f"{fz.auc_observed_mean.iloc[0]:.2f} (60 days) and {fz.auc_observed_mean.iloc[1]:.2f} (120 days) per "
        "patient: the finite-record bias reaches a cross-validated model through the training rate. With frozen "
        f"coefficients the shuffled baseline sits at 0.50, the excess is {fz.excess_mean.iloc[0]:+.2f} and "
        f"{fz.excess_mean.iloc[1]:+.2f}, and the test rejects in {100 * fz.rejection_rate.iloc[0]:.1f}% and "
        f"{100 * fz.rejection_rate.iloc[1]:.1f}% of records: a forecast with real skill would be judged against a "
        "baseline that is too high. With the pipeline refitted in every shuffle the baseline follows the bias "
        f"({rf.auc_shuffled_mean.iloc[0]:.2f}, {rf.auc_shuffled_mean.iloc[1]:.2f}), the excess is zero within "
        f"simulation error ({rf.excess_mean.iloc[0]:+.3f}, {rf.excess_mean.iloc[1]:+.3f}) and the test has its "
        f"nominal size ({100 * rf.rejection_rate.iloc[0]:.1f}%, {100 * rf.rejection_rate.iloc[1]:.1f}%). Rule: keep "
        "what was learned without the shuffled outcomes; refit whatever was learned from them. (Agent version, "
        "60 days, 1,000 records, 39 shuffles: -0.11 and -0.01.)", "",
        "## B3c. A time course shared by all patients, and the block shuffle", "",
        "Script `code/v14_shared_course.py`, numbers `results/v14/B3c_shared_course.csv`.", "",
        f"Design. Records of 100 independent days, {int(c.records.max()):,} records per course, "
        f"{int(c.shuffles.iloc[0])} shuffles per record. Courses: (1) risk falls linearly from 50% to 10%, the same "
        "for every patient; (2) constant risk 30% (control for the size of the test); (3) the falling course plus "
        "a real daily signal, logit risk = logit(course) + x. Forecasts: minus the study day; the rate so far "
        "(rebuilt in every shuffle; 0.30 on day 1); a constant; the daily signal x. Shuffles: all days of the "
        "record (full), or within consecutive blocks of 20 or 10 days, which keeps the number of events in every "
        "block. Excess and rejections (one-sided, 5%) are per record.", "",
        "| Course | Forecast | Per-patient AUC | Full shuffle: excess, rejections | Blocks of 20: excess, rejections | "
        "Blocks of 10: excess, rejections |",
        "|---|---|---:|---:|---:|---:|",
    ]
    courses = {"falling_0.50_to_0.10": "falling 50% to 10%", "constant_0.30": "constant 30%",
               "falling_with_daily_signal": "falling, with daily signal"}
    forecasts = {"study_day": "study day", "rate_so_far": "rate so far", "constant": "constant",
                 "daily_signal": "daily signal x"}
    for course, course_label in courses.items():
        for f, f_label in forecasts.items():
            d = c[(c.course == course) & (c.forecast == f)].set_index("shuffle")
            cell = lambda s: f"{d.loc[s, 'excess_mean']:+.3f}, {100 * d.loc[s, 'rejection_rate']:.1f}%"
            L.append(f"| {course_label} | {f_label} | {d.loc['full', 'auc_mean']:.3f} | {cell('full')} | "
                     f"{cell('block_20')} | {cell('block_10')} |")
    g = lambda course, f, s, col: c[(c.course == course) & (c.forecast == f) & (c.shuffle == s)][col].iloc[0]
    fall, sig, flat = "falling_0.50_to_0.10", "falling_with_daily_signal", "constant_0.30"
    size = c[(c.course == flat) & (c.forecast != "constant")].rejection_rate
    L += [
        "", f"Standard errors of the excess are at most {c.excess_se.max():.4f}.", "", "Readings.", "",
        "- With a shared falling course, the study day alone has an excess of "
        f"{g(fall, 'study_day', 'full', 'excess_mean'):.2f} over the full shuffle and the rate so far "
        f"{g(fall, 'rate_so_far', 'full', 'excess_mean'):.2f} "
        f"({100 * g(fall, 'study_day', 'full', 'rejection_rate'):.0f}% and "
        f"{100 * g(fall, 'rate_so_far', 'full', 'rejection_rate'):.0f}% of single records reject). Both forecasts "
        "know only the shared course; under the definition of Section 2 that is knowing when. A constant forecast "
        "has excess 0 and never rejects. (Agent version: 0.16, 0.13, 0.)",
        "- The block shuffle removes this: the excess falls to "
        f"{g(fall, 'study_day', 'block_20', 'excess_mean'):.3f} and "
        f"{g(fall, 'rate_so_far', 'block_20', 'excess_mean'):.3f} with blocks of 20 days and to "
        f"{g(fall, 'study_day', 'block_10', 'excess_mean'):.3f} and "
        f"{g(fall, 'rate_so_far', 'block_10', 'excess_mean'):.3f} with blocks of 10 days. Rejections stay somewhat "
        f"above 5% ({100 * g(fall, 'study_day', 'block_20', 'rejection_rate'):.0f}% and "
        f"{100 * g(fall, 'rate_so_far', 'block_20', 'rejection_rate'):.0f}% with blocks of 20; "
        f"{100 * g(fall, 'study_day', 'block_10', 'rejection_rate'):.0f}% and "
        f"{100 * g(fall, 'rate_so_far', 'block_10', 'rejection_rate'):.0f}% with blocks of 10), because the risk "
        "still falls within a block. Shorter blocks remove more of the course.",
        "- The block shuffle keeps skill that is specific to the patient's days: the daily signal has an excess of "
        f"{g(sig, 'daily_signal', 'full', 'excess_mean'):.2f} over the full shuffle and "
        f"{g(sig, 'daily_signal', 'block_20', 'excess_mean'):.2f} and "
        f"{g(sig, 'daily_signal', 'block_10', 'excess_mean'):.2f} over blocks of 20 and 10 days, with "
        f"{100 * c[(c.course == sig) & (c.forecast == 'daily_signal')].rejection_rate.min():.0f}% rejections or more "
        "in all three.",
        f"- With a constant risk every test has its size ({100 * size.min():.1f}% to {100 * size.max():.1f}%), and "
        f"the rate so far scores {g(flat, 'rate_so_far', 'full', 'auc_mean'):.3f} per patient with excess 0: the "
        "shuffled baseline absorbs the finite-record bias under the block shuffle as well.",
    ]
    (OUT / "B3_simulations.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
