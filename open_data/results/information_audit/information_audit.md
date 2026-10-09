# Information audit of pooled versus within-patient forecast AUC

Entries are point estimates with 95% percentile intervals from 2,000 whole-patient resamples (seed 20261009).
A0 is recomputed as each test patient's observed event rate across that patient's scored period. A0,time is a time-only forecast: the TRAINING patients' event rate at each time point (overall training rate where fewer than five training patients), the same for every test patient at that time; its pooled AUC covers all scored test observations; its per-time-point AUC averages only points with both outcomes and at least five patients.
The who part is each patient's mean GB probability. The when part is the GB probability minus that patient's mean plus the row-weighted cohort mean, all on the probability scale. Per-patient AUCs use patients with both outcomes.
For College Experience, A0,time is shown for both follow-up week and calendar date.

| Cohort | A0 | A0,time | GB pooled | who part pooled | when part pooled | when part per-patient | share reproduced |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| REPLACE-BG | 0.752 (0.709 to 0.785) | Night since enrollment: pooled 0.500 (0.475 to 0.526); per-time-point 0.500 (0.500 to 0.500) | 0.829 (0.800 to 0.855) | 0.736 (0.692 to 0.771) | 0.702 (0.650 to 0.758) | 0.759 (0.727 to 0.789) | 0.719 (0.613 to 0.793) |
| CTN-0094 | 0.932 (0.915 to 0.946) | Study week since randomization: pooled 0.527 (0.505 to 0.548); per-time-point 0.500 (0.500 to 0.500) | 0.929 (0.914 to 0.942) | 0.922 (0.904 to 0.938) | 0.669 (0.641 to 0.695) | 0.638 (0.593 to 0.681) | 0.984 (0.963 to 1.002) |
| College Experience | 0.855 (0.787 to 0.900) | Week since first complete report: pooled 0.547 (0.491 to 0.612); per-time-point 0.500 (0.500 to 0.500)<br>Calendar date: pooled 0.530 (0.493 to 0.568); per-time-point 0.500 (0.500 to 0.500) | 0.935 (0.906 to 0.955) | 0.853 (0.784 to 0.899) | 0.804 (0.760 to 0.849) | 0.800 (0.756 to 0.840) | 0.812 (0.670 to 0.896) |
