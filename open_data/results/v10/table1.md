# Table 1. One forecast, two patients

| Metric | patient A | patient B | per-patient mean | pooled |
| --- | ---: | ---: | ---: | ---: |
| AUC | 0.50 | 0.50 | 0.50 | 0.82 |
| AP | 0.667 | 0.067 | 0.367 | 0.639 |
| Sensitivity | 100% | 0% | 50% | 91% |
| Specificity | 0% | 100% | 50% | 74% |
| PPV | 0.67 (excess +0.00) | n/a | 0.67 (excess +0.00) | 0.67 (excess +0.30) |
| NPV | n/a | 0.93 (excess +0.00) | 0.93 (excess +0.00) | 0.93 (excess +0.30) |
| Net benefit | 0.000 | 0.000 | 0.000 | 0.235 |
| Brier skill | 0% | 0% | 0% | 39% |
| Calibration slope | n/a | n/a | n/a | 1.00 |

*n/a:* PPV is undefined for patient B because no days are alarmed; NPV is undefined for patient A because no days are unalarmed. Per-patient calibration slopes and their mean are undefined because each patient's forecast is constant, so there is no within-patient forecast variation.

*Record detail:* the manuscript reports 30 days with 20 events for A and 30 days with 2 events for B, and forecasts equal to each patient's own rate (2/3 and 2/30; displayed as 0.67 and 0.07). It does not report event-day labels or their order. The reported grouped counts are sufficient because forecasts are constant within patient; no day order has been imputed.

*Definitions:* AP is non-interpolated average precision at distinct forecast levels; its expected chance references for random orderings without ties are 0.701 for A and 0.163 for B. PPV/NPV cells show the value and its excess over the patient's own event/non-event rate (pooled: over the population rate). Per-patient net benefit is the model's net benefit minus the best policy supported by that patient's own event rate; pooled net benefit is standard net benefit at t = 0.37. Per-patient Brier skill uses each own rate as reference; pooled Brier skill uses the population rate. The population event rate is 0.37 when rounded as in the manuscript. The high forecast group (shown as 0.67) is the alarm group; threshold t = 0.37 selects the same days for net benefit. Youden's index is 0 in each patient, as printed in Table 1.
