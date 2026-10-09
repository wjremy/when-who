## Table S2 — exact beta rate benchmark
Midpoint quadrature (400,000 points; in u = p**alpha when alpha < 1) for E|p-p'|; cells show four decimals and the two-decimal value in parentheses.
| Mean event rate | SD 0.025 | SD 0.05 | SD 0.10 | SD 0.15 | SD 0.20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.6446 (0.64) | 0.7669 (0.77) |  |  |  |
| 0.10 | 0.5779 (0.58) | 0.6533 (0.65) | 0.7860 (0.79) | 0.8840 (0.88) |  |
| 0.20 | 0.5441 (0.54) | 0.5879 (0.59) | 0.6743 (0.67) | 0.7569 (0.76) | 0.8324 (0.83) |
| 0.30 | 0.5336 (0.53) | 0.5672 (0.57) | 0.6345 (0.63) | 0.7016 (0.70) | 0.7679 (0.77) |
| 0.50 | 0.5282 (0.53) | 0.5565 (0.56) | 0.6134 (0.61) | 0.6710 (0.67) | 0.7294 (0.73) |

Printed-value comparison (two-decimal rounded values):

| Cell | Regenerated | Printed | Difference |
| --- | ---: | ---: | ---: |
| mu=0.05, sd=0.025 | 0.64 | 0.64 | 0.00 |
| mu=0.05, sd=0.050 | 0.77 | 0.77 | 0.00 |
| mu=0.10, sd=0.025 | 0.58 | 0.58 | 0.00 |
| mu=0.10, sd=0.050 | 0.65 | 0.65 | 0.00 |
| mu=0.10, sd=0.100 | 0.79 | 0.79 | 0.00 |
| mu=0.10, sd=0.150 | 0.88 | 0.88 | 0.00 |
| mu=0.20, sd=0.025 | 0.54 | 0.54 | 0.00 |
| mu=0.20, sd=0.050 | 0.59 | 0.59 | 0.00 |
| mu=0.20, sd=0.100 | 0.67 | 0.67 | 0.00 |
| mu=0.20, sd=0.150 | 0.76 | 0.76 | 0.00 |
| mu=0.20, sd=0.200 | 0.83 | 0.83 | 0.00 |
| mu=0.30, sd=0.025 | 0.53 | 0.53 | 0.00 |
| mu=0.30, sd=0.050 | 0.57 | 0.57 | 0.00 |
| mu=0.30, sd=0.100 | 0.63 | 0.63 | 0.00 |
| mu=0.30, sd=0.150 | 0.70 | 0.70 | 0.00 |
| mu=0.30, sd=0.200 | 0.77 | 0.77 | 0.00 |
| mu=0.50, sd=0.025 | 0.53 | 0.53 | 0.00 |
| mu=0.50, sd=0.050 | 0.56 | 0.56 | 0.00 |
| mu=0.50, sd=0.100 | 0.61 | 0.61 | 0.00 |
| mu=0.50, sd=0.150 | 0.67 | 0.67 | 0.00 |
| mu=0.50, sd=0.200 | 0.73 | 0.73 | 0.00 |

## Table S3 — calibration simulation
Printed realisation: 400 patients × 300 days, tau=1.5, omega=0.5, fixed intercept=logit(0.15), generator seed 11 as in code/calibration.py; realised event proportion 0.237.
| Daily logit deviation scale | Pooled slope | Between slope | Within slope |
| ---: | ---: | ---: | ---: |
| 0.25 | 0.9807 | 0.9876 | 4.0505 |
| 1 | 0.9887 | 0.9865 | 1.0127 |
| 3 | 0.6351 | 0.9814 | 0.3374 |

### Sampling variation (200 fresh realisations of the same design, seed=73105; mean (SD))
| Daily logit deviation scale | Pooled slope | Between slope | Within slope |
| ---: | ---: | ---: | ---: |
| 0.25 | 0.998 (0.008) | 1.000 (0.008) | 3.998 (0.065) |
| 1 | 1.000 (0.008) | 1.000 (0.008) | 1.000 (0.016) |
| 3 | 0.614 (0.013) | 0.998 (0.008) | 0.333 (0.005) |

### Expected-outcome check
4,000 patients × 180 days, tau=1.5, omega=0.5, intercept set so that the mean risk is 0.1500; outcomes replaced by conditional risks (seed 20261005, as in the original verification).
| Daily logit deviation scale | Pooled slope | Between slope | Within slope |
| ---: | ---: | ---: | ---: |
| 0.25 | 1.0005 | 1.0001 | 3.9996 |
| 1 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 0.6155 | 0.9948 | 0.3330 |

### Printed-value comparison
Both checks are deterministic given their seeds; a value matches if it rounds to the printed two decimals.
| Quantity | Regenerated | Printed | Difference | Tolerance result |
| --- | ---: | ---: | ---: | --- |
| seed 11, scale=0.25, pooled | 0.9807 | 0.98 | 0.0007 | matches at printed precision |
| seed 11, scale=0.25, between | 0.9876 | 0.99 | 0.0024 | matches at printed precision |
| seed 11, scale=0.25, within | 4.0505 | 4.05 | 0.0005 | matches at printed precision |
| seed 11, scale=1, pooled | 0.9887 | 0.99 | 0.0013 | matches at printed precision |
| seed 11, scale=1, between | 0.9865 | 0.99 | 0.0035 | matches at printed precision |
| seed 11, scale=1, within | 1.0127 | 1.01 | 0.0027 | matches at printed precision |
| seed 11, scale=3, pooled | 0.6351 | 0.64 | 0.0049 | matches at printed precision |
| seed 11, scale=3, between | 0.9814 | 0.98 | 0.0014 | matches at printed precision |
| seed 11, scale=3, within | 0.3374 | 0.34 | 0.0026 | matches at printed precision |
| expected, scale=0.25, pooled | 1.0005 | 1.00 | 0.0005 | matches at printed precision |
| expected, scale=0.25, between | 1.0001 | 1.00 | 0.0001 | matches at printed precision |
| expected, scale=0.25, within | 3.9996 | 4.00 | 0.0004 | matches at printed precision |
| expected, scale=1, pooled | 1.0000 | 1.00 | 0.0000 | matches at printed precision |
| expected, scale=1, between | 1.0000 | 1.00 | 0.0000 | matches at printed precision |
| expected, scale=1, within | 1.0000 | 1.00 | 0.0000 | matches at printed precision |
| expected, scale=3, pooled | 0.6155 | 0.62 | 0.0045 | matches at printed precision |
| expected, scale=3, between | 0.9948 | 0.99 | 0.0048 | matches at printed precision |
| expected, scale=3, within | 0.3330 | 0.33 | 0.0030 | matches at printed precision |

## Table S4 — independent Bernoulli history forecasts
100,000 diaries per length; Bernoulli(0.15), scoring starts at day 30, seed=20261006; SD of diary AUCs / sqrt(eligible diaries) is the Monte Carlo SE.
| Diary length | Expanding rate | Previous 30-day count | Previous 7-day count |
| ---: | ---: | ---: | ---: |
| 60 | 0.2773 (0.00042; n=99248) | 0.3571 (0.00043; n=99248) | 0.4188 (0.00041; n=99248) |
| 100 | 0.3555 (0.00027; n=99997) | 0.4164 (0.00026; n=99997) | 0.4639 (0.00028; n=99997) |
| 180 | 0.4034 (0.00018; n=100000) | 0.4572 (0.00019; n=100000) | 0.4829 (0.00020; n=100000) |
| 365 | 0.4360 (0.00012; n=100000) | 0.4800 (0.00013; n=100000) | 0.4924 (0.00013; n=100000) |
| 730 | 0.4562 (0.00009; n=100000) | 0.4905 (0.00009; n=100000) | 0.4965 (0.00009; n=100000) |

### Printed-value comparison
The printed values are themselves Monte Carlo means of 100,000 diaries, so the difference of two independent runs has standard error sqrt(2) x SE; the criterion is 3 x sqrt(2) x SE.
| Quantity | Regenerated | Printed | Difference | Tolerance result |
| --- | ---: | ---: | ---: | --- |
| T=60, expanding | 0.2773 (SE 0.00042) | 0.2769 | 0.0004 | within tolerance |
| T=60, previous 30 days | 0.3571 (SE 0.00043) | 0.3569 | 0.0002 | within tolerance |
| T=60, previous 7 days | 0.4188 (SE 0.00041) | 0.4191 | 0.0003 | within tolerance |
| T=100, expanding | 0.3555 (SE 0.00027) | 0.3560 | 0.0005 | within tolerance |
| T=100, previous 30 days | 0.4164 (SE 0.00026) | 0.4165 | 0.0001 | within tolerance |
| T=100, previous 7 days | 0.4639 (SE 0.00028) | 0.4639 | 0.0000 | within tolerance |
| T=180, expanding | 0.4034 (SE 0.00018) | 0.4033 | 0.0001 | within tolerance |
| T=180, previous 30 days | 0.4572 (SE 0.00019) | 0.4574 | 0.0002 | within tolerance |
| T=180, previous 7 days | 0.4829 (SE 0.00020) | 0.4833 | 0.0004 | within tolerance |
| T=365, expanding | 0.4360 (SE 0.00012) | 0.4360 | 0.0000 | within tolerance |
| T=365, previous 30 days | 0.4800 (SE 0.00013) | 0.4801 | 0.0001 | within tolerance |
| T=365, previous 7 days | 0.4924 (SE 0.00013) | 0.4923 | 0.0001 | within tolerance |
| T=730, expanding | 0.4562 (SE 0.00009) | 0.4563 | 0.0001 | within tolerance |
| T=730, previous 30 days | 0.4905 (SE 0.00009) | 0.4906 | 0.0001 | within tolerance |
| T=730, previous 7 days | 0.4965 (SE 0.00009) | 0.4964 | 0.0001 | within tolerance |
