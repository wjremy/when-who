# B3. Reruns of three simulations of the evidence audit (12 Oct 2026)

Clean scripts, written and reviewed by Claude; the agent versions in `work/2026-10-12_evidence/sim/` were used only for the designs. All seeds derive from 20261012. Numbers are means over replications; Monte Carlo standard errors are in the CSV files. This file is written by `code/v14_b3_report.py`.

## B3a. Who and when in the pooled, per-patient and per-day AUC

Script `code/v14_who_when_auc.py`, numbers `results/v14/B3a_who_when.csv` (48 cells, 400 cohorts each; largest standard error 0.003).

Design. K patients by T days. Patient levels are Beta with mean 0.20 and SD 0.05 or 0.15. Outcomes are independent with logit risk = logit(level) + beta x + d: x is a daily signal (standard normal, independent over patients and days, no patient level), d a day effect shared by all patients (normal, SD 0 or 1). Forecasts: who only (the patient's level), when only (x), shared day only (d), both (the true risk). beta = 0.5 (weak) or 1.5 (strong); with beta = 0 the when-only forecast has no skill. Per-patient and per-day AUC: mean over patients or days with both outcomes.

K = 100, T = 365 (all 48 cells, with K = 10 and T = 30, are in the CSV):

| Level SD | Daily signal | Shared day SD | Forecast | Pooled | Per patient | Per day |
|---|---|---|---|---:|---:|---:|
| 0.05 | none | 0 | who only | 0.587 | 0.500 | 0.588 |
| 0.05 | none | 0 | both (true risk) | 0.587 | 0.500 | 0.588 |
| 0.05 | none | 1 | who only | 0.575 | 0.500 | 0.588 |
| 0.05 | none | 1 | shared day only | 0.740 | 0.744 | 0.500 |
| 0.05 | none | 1 | both (true risk) | 0.752 | 0.744 | 0.588 |
| 0.05 | weak | 0 | who only | 0.584 | 0.500 | 0.585 |
| 0.05 | weak | 0 | when only | 0.633 | 0.636 | 0.633 |
| 0.05 | weak | 0 | both (true risk) | 0.658 | 0.636 | 0.658 |
| 0.05 | weak | 1 | who only | 0.573 | 0.500 | 0.585 |
| 0.05 | weak | 1 | when only | 0.616 | 0.617 | 0.634 |
| 0.05 | weak | 1 | shared day only | 0.731 | 0.735 | 0.500 |
| 0.05 | weak | 1 | both (true risk) | 0.770 | 0.763 | 0.658 |
| 0.05 | strong | 0 | who only | 0.565 | 0.500 | 0.565 |
| 0.05 | strong | 0 | when only | 0.814 | 0.817 | 0.814 |
| 0.05 | strong | 0 | both (true risk) | 0.821 | 0.817 | 0.821 |
| 0.05 | strong | 1 | who only | 0.559 | 0.500 | 0.566 |
| 0.05 | strong | 1 | when only | 0.785 | 0.788 | 0.816 |
| 0.05 | strong | 1 | shared day only | 0.688 | 0.690 | 0.500 |
| 0.05 | strong | 1 | both (true risk) | 0.852 | 0.849 | 0.823 |
| 0.15 | none | 0 | who only | 0.754 | 0.500 | 0.756 |
| 0.15 | none | 0 | both (true risk) | 0.754 | 0.500 | 0.756 |
| 0.15 | none | 1 | who only | 0.728 | 0.500 | 0.757 |
| 0.15 | none | 1 | shared day only | 0.715 | 0.746 | 0.500 |
| 0.15 | none | 1 | both (true risk) | 0.816 | 0.746 | 0.757 |
| 0.15 | weak | 0 | who only | 0.747 | 0.500 | 0.749 |
| 0.15 | weak | 0 | when only | 0.618 | 0.636 | 0.617 |
| 0.15 | weak | 0 | both (true risk) | 0.774 | 0.636 | 0.776 |
| 0.15 | weak | 1 | who only | 0.725 | 0.500 | 0.752 |
| 0.15 | weak | 1 | when only | 0.604 | 0.620 | 0.618 |
| 0.15 | weak | 1 | shared day only | 0.708 | 0.739 | 0.500 |
| 0.15 | weak | 1 | both (true risk) | 0.827 | 0.767 | 0.779 |
| 0.15 | strong | 0 | who only | 0.706 | 0.500 | 0.708 |
| 0.15 | strong | 0 | when only | 0.786 | 0.821 | 0.786 |
| 0.15 | strong | 0 | both (true risk) | 0.857 | 0.821 | 0.857 |
| 0.15 | strong | 1 | who only | 0.691 | 0.500 | 0.710 |
| 0.15 | strong | 1 | when only | 0.763 | 0.793 | 0.787 |
| 0.15 | strong | 1 | shared day only | 0.674 | 0.695 | 0.500 |
| 0.15 | strong | 1 | both (true risk) | 0.877 | 0.853 | 0.859 |

Readings over all 48 cells.

- A forecast that knows only when keeps most of its per-patient AUC in the pooled AUC. Share of the per-patient excess over 0.5 that the pooled AUC retains, (pooled - 0.5) / (per patient - 0.5): 0.98 to 0.99 with level SD 0.05 and 0.86 to 0.90 with level SD 0.15 (no shared day effect, all sizes, both strengths); 0.87 to 1.00 with a shared day effect. It does not shrink with the number of patients: strong signal, level SD 0.15, T = 365, K = 10 against K = 100: pooled 0.789 against 0.786 (per patient 0.821 and 0.821).
- A forecast that knows only who scores 0.500 per patient in every cell. Pooled, without a shared day effect, it scores 0.56 to 0.59 (level SD 0.05) and 0.69 to 0.75 (level SD 0.15). Its pooled AUC is then within 0.022 of its per-day AUC (K = 100: within 0.002); with a shared day effect the pooled value is below the per-day value by up to 0.046.
- A forecast that knows only a day effect shared by all patients scores 0.71 to 0.74 pooled and 0.74 to 0.75 per patient when there is no other signal, and 0.500 per day in every cell (all patients are tied within a day).
- So 'the pooled AUC is close to the per-day AUC' holds for a who-only forecast without shared time variation, not in general. And the pooled AUC does not measure 'almost nothing of when': it takes up who, when and a shared course alike and cannot tell them apart.

Difference from the agent's version: the day effect has SD 1 on the logit scale (agent: 2.5, with which the shared-day forecast reached 0.87 pooled and 0.90 per patient), and level SD 0.15 is the value of the task (the agent's 'high' spread was smaller: who-only pooled 0.71). The qualitative results are the same; quote the numbers of this file.

## B3b. A model fitted on the scored record: frozen against refitted coefficients

Script `code/v14_refit_shuffle.py`, numbers `results/v14/B3b_refit.csv`.

Design. Records of independent days with risk 0.25 (no forecast has skill), 2,000 records per length, 199 shuffles per record. Pipeline: five rotating folds within the patient (day index modulo 5); for each fold a logistic regression is fitted on the other four folds, with a noise covariate and the patient's event rate among the other training days as inputs (ridge penalty 1 on the two slopes), and predicts the held-out days. Frozen: the coefficients fitted on the observed record are kept in every shuffle and only the outcome-derived input is rebuilt. Refitted: the whole pipeline is run again on every shuffled record. One-sided p = (1 + number of shuffled AUCs at or above the observed) / (shuffles + 1).

| Record length | Shuffle | Records | Observed per-patient AUC | Mean shuffled AUC | Excess (SE) | Rejections at 5% |
|---:|---|---:|---:|---:|---:|---:|
| 60 | frozen | 2,000 | 0.400 | 0.503 | -0.103 (0.003) | 2.2% |
| 60 | refit | 2,000 | 0.400 | 0.398 | +0.002 (0.003) | 5.1% |
| 120 | frozen | 2,000 | 0.429 | 0.502 | -0.073 (0.002) | 2.2% |
| 120 | refit | 2,000 | 0.429 | 0.429 | -0.000 (0.002) | 4.8% |

Readings. The no-skill pipeline itself scores 0.40 (60 days) and 0.43 (120 days) per patient: the finite-record bias reaches a cross-validated model through the training rate. With frozen coefficients the shuffled baseline sits at 0.50, the excess is -0.10 and -0.07, and the test rejects in 2.2% and 2.2% of records: a forecast with real skill would be judged against a baseline that is too high. With the pipeline refitted in every shuffle the baseline follows the bias (0.40, 0.43), the excess is zero within simulation error (+0.002, -0.000) and the test has its nominal size (5.1%, 4.8%). Rule: keep what was learned without the shuffled outcomes; refit whatever was learned from them. (Agent version, 60 days, 1,000 records, 39 shuffles: -0.11 and -0.01.)

## B3c. A time course shared by all patients, and the block shuffle

Script `code/v14_shared_course.py`, numbers `results/v14/B3c_shared_course.csv`.

Design. Records of 100 independent days, 4,000 records per course, 199 shuffles per record. Courses: (1) risk falls linearly from 50% to 10%, the same for every patient; (2) constant risk 30% (control for the size of the test); (3) the falling course plus a real daily signal, logit risk = logit(course) + x. Forecasts: minus the study day; the rate so far (rebuilt in every shuffle; 0.30 on day 1); a constant; the daily signal x. Shuffles: all days of the record (full), or within consecutive blocks of 20 or 10 days, which keeps the number of events in every block. Excess and rejections (one-sided, 5%) are per record.

| Course | Forecast | Per-patient AUC | Full shuffle: excess, rejections | Blocks of 20: excess, rejections | Blocks of 10: excess, rejections |
|---|---|---:|---:|---:|---:|
| falling 50% to 10% | study day | 0.661 | +0.161, 81.8% | +0.007, 12.7% | +0.001, 6.8% |
| falling 50% to 10% | rate so far | 0.543 | +0.125, 61.6% | +0.009, 10.3% | +0.002, 6.3% |
| falling 50% to 10% | constant | 0.500 | +0.000, 0.0% | +0.000, 0.0% | +0.000, 0.0% |
| falling 50% to 10% | daily signal x | 0.500 | +0.000, 4.9% | +0.000, 4.5% | -0.000, 4.5% |
| constant 30% | study day | 0.497 | -0.003, 3.9% | -0.000, 4.7% | -0.000, 4.5% |
| constant 30% | rate so far | 0.415 | -0.002, 4.0% | -0.001, 4.3% | -0.001, 5.0% |
| constant 30% | constant | 0.500 | +0.000, 0.0% | +0.000, 0.0% | +0.000, 0.0% |
| constant 30% | daily signal x | 0.501 | +0.001, 4.6% | +0.001, 4.9% | +0.001, 5.3% |
| falling, with daily signal | study day | 0.641 | +0.141, 73.2% | +0.006, 11.8% | +0.002, 7.4% |
| falling, with daily signal | rate so far | 0.522 | +0.102, 52.0% | +0.007, 9.3% | +0.002, 6.4% |
| falling, with daily signal | constant | 0.500 | +0.000, 0.0% | +0.000, 0.0% | +0.000, 0.0% |
| falling, with daily signal | daily signal x | 0.729 | +0.229, 98.5% | +0.220, 98.8% | +0.209, 98.3% |

Standard errors of the excess are at most 0.0013.

Readings.

- With a shared falling course, the study day alone has an excess of 0.16 over the full shuffle and the rate so far 0.12 (82% and 62% of single records reject). Both forecasts know only the shared course; under the definition of Section 2 that is knowing when. A constant forecast has excess 0 and never rejects. (Agent version: 0.16, 0.13, 0.)
- The block shuffle removes this: the excess falls to 0.007 and 0.009 with blocks of 20 days and to 0.001 and 0.002 with blocks of 10 days. Rejections stay somewhat above 5% (13% and 10% with blocks of 20; 7% and 6% with blocks of 10), because the risk still falls within a block. Shorter blocks remove more of the course.
- The block shuffle keeps skill that is specific to the patient's days: the daily signal has an excess of 0.23 over the full shuffle and 0.22 and 0.21 over blocks of 20 and 10 days, with 98% rejections or more in all three.
- With a constant risk every test has its size (3.9% to 5.3%), and the rate so far scores 0.415 per patient with excess 0: the shuffled baseline absorbs the finite-record bias under the block shuffle as well.
