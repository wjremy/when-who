# REPLACE-BG daily hypoglycaemia forecasting analysis

Primary event: at least 15 minutes below 54 mg/dL in the 00:00–06:00 night, defined by three consecutive 5-minute CGM readings below threshold. Nights and prior predictor days each require at least 70% of nominal CGM slots. The first 14 eligible nights per patient form the forecasting run-in; patients require at least 60 eligible post-run-in nights.

## Cohort

| Quantity | Value |
|---|---:|
| Patients | 224 |
| Scored nights | 46702 |
| Primary events | 2749 |
| Primary event rate | 0.05886257547856623 |
| Mean patient event rate | 0.06023293939812164 |
| SD patient event rate | 0.0519549540961884 |
| Event-rate ICC | 0.04566711280538863 |
| Median eligible-night gap (days) | 1.0 |
| Within-patient pair share | 0.004318098409908193 |
| Night below 70 rate | 0.1575735514538992 |
| Prior-day basal dose estimate available | 0.8963641814055072 |
| Next-24h below 54 rate (covered days) | 0.2144229283048864 |

## Ranking and benchmark

| Forecast | Pooled AUC (95% CI) | Within-patient AUC (95% CI) | Eligible patients |
|---|---:|---:|---:|
| training_cohort_rate | 0.5 ([0.5, 0.5]) | 0.5 ([0.5, 0.5]) | 42 |
| runin_rate | 0.5971073660374072 ([0.5243364856739329, 0.6617163480707989]) | 0.5 ([0.5, 0.5]) | 42 |
| runin_rate_shrunk_10 | 0.5971073660374072 ([0.5243364856739329, 0.6617163480707989]) | 0.5 ([0.5, 0.5]) | 42 |
| expanding_rate | 0.7066441408005194 ([0.6547688566372091, 0.7423560648792241]) | 0.3870664140861947 ([0.36042025541436795, 0.41414089363144496]) | 42 |
| recent_28_night_rate | 0.6837289215890039 ([0.6359357798147626, 0.720714322646195]) | 0.4692132343751356 ([0.44664639756050023, 0.49126579345358323]) | 42 |
| persistence | 0.5365961199294532 ([0.5173024924524281, 0.552816467205721]) | 0.5062029783623406 ([0.49429808497671557, 0.5176671067757816]) | 42 |
| retrospective_rate_reference | 0.7518087003683711 ([0.7090254087904425, 0.7822097787204548]) | 0.5 ([0.5, 0.5]) | 42 |
| logistic_regression | 0.8181307489949465 ([0.7912165724213727, 0.8428457005856554]) | 0.750388338185026 ([0.7151081270422536, 0.7842801499861883]) | 42 |
| gradient_boosting | 0.8285656882364701 ([0.7996464071555783, 0.8537493613845042]) | 0.7586922688546608 ([0.7261732451659114, 0.7910232221882627]) | 42 |
| within_patient_centred_gradient_boosting | 0.784827799848376 ([0.7449592102281358, 0.8233516801128486]) | 0.7598249085880727 ([0.7199821221586435, 0.7960639725929354]) | 42 |
| benchmark | retrospective 0.7518087003683711; run-in 0.5971073660374072; expanding 0.7066441408005194 | — | — |

## Scores and skill

| Forecast | Brier pooled | Brier patient mean | Log-loss pooled | Log-loss patient mean | Skill vs run-in | Skill vs expanding | Skill vs retrospective |
|---|---:|---:|---:|---:|---:|---:|---:|
| training_cohort_rate | 0.056279493122475106 | 0.0568531764771633 | 0.22677946859300774 | 0.22861794837647814 | 0.07029598784226199 | -0.03053015050575314 | -0.05284717635281133 |
| runin_rate | 0.0605348502173899 | 0.0610549833290982 | 0.41904811626322586 | 0.4367561440329498 | 0.0 | -0.108449718436741 | -0.13245416023242917 |
| runin_rate_shrunk_10 | 0.05694037290988709 | 0.057481683072349 | 0.2292686715838797 | 0.23171035443263588 | 0.05937864378278779 | -0.04263147745455442 | -0.06521056805165171 |
| expanding_rate | 0.05461217519434519 | 0.05527157155693085 | 0.23038814523330184 | 0.23494917672634238 | 0.09783909602114282 | 0.0 | -0.021655868909905784 |
| recent_28_night_rate | 0.05562606053174598 | 0.05629186676114326 | 0.3015467285479277 | 0.3053925887536095 | 0.08109030860761546 | -0.0185651886926812 | -0.040623102895203544 |
| persistence | 0.10435791917273399 | 0.10596917210890562 | 1.4417588297791155 | 1.4640191101190687 | -0.7239312362708217 | -0.9108910934486951 | -0.9522731004695268 |
| retrospective_rate_reference | 0.05345457003307357 | 0.054094946257658344 | 0.20242624610862484 | 0.20508386604216894 | 0.11696205010650829 | 0.02119683307160214 | 0.0 |
| logistic_regression | 0.0495461979552326 | 0.0502340577370823 | 0.1846943164449477 | 0.18746169367291346 | 0.18152605024536062 | 0.09276278084666256 | 0.07311577055849039 |
| gradient_boosting | 0.047963325082300105 | 0.0485280285093282 | 0.17992120051262345 | 0.18218163481611707 | 0.20767417594895377 | 0.12174666342045903 | 0.10272732429380504 |
| within_patient_centred_gradient_boosting | 0.05066735305856736 | 0.051284887161684216 | 0.19255758464971048 | 0.19482677164812737 | 0.16300522960553876 | 0.07223338242323474 | 0.05214179017400555 |

## Calibration and history bias

| Analysis | Forecast | Pooled / observed | Between / reference | Within / difference |
|---|---|---:|---:|---:|
| Calibration slopes: logistic_regression | Day weighted | 1.1525336417817802 | 1.43983562045042 | 1.0281811203274895 |
| Calibration slopes: gradient_boosting | Day weighted | 1.0592481201856472 | 1.5242569490825126 | 0.9515557101478844 |
| History bias | recent_28_night_rate | 0.46921323437513557 | 0.4575107165502806 | 0.011702517824854949 |
| History bias | expanding_rate | 0.38706641408619463 | 0.3733306848508288 | 0.013735729235365802 |
| History bias | persistence | 0.5062029783623405 | 0.49787760471147585 | 0.00832537365086476 |

Treatment assignment and a randomization day are not available in the inspected HRandomization table, so these results do not estimate treatment effects or separate protocol phases. Bolus and carbohydrate variables indicate pump-record availability; missing entries are not evidence of zero exposure. Basal insulin is estimated from recorded rate multiplied by actual duration, split at midnight; it may be incomplete when pump records are absent.
