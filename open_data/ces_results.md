# Nepal openESM analysis results

## Cohort

| Quantity | Value |
| --- | --- |
| included_people | 186 |
| scored_forecasts | 30369 |
| event_rate | 0.3823635944548717 |
| mean_personal_event_rate | 0.4123060899365669 |
| sd_personal_event_rate | 0.3142021565169628 |
| forecast_person_event_rate_mean | 0.4123060899365669 |
| forecast_person_event_rate_sd | 0.3142021565169628 |
| icc_day_weighted | 0.40039525468357207 |
| median_gap_days_consecutive_complete_reports | 5.0 |
| within_person_pair_share | 0.006463285657366274 |
| complete_reports | 35348 |
| people_with_complete_reports | 218 |
| people_ge60_reports | 186 |
| median_reports_per_person | 169.5 |

## Ranking with benchmark

| Forecast / reference | Pooled AUC | Within AUC | Eligible people | Pooled minus within |
| --- | --- | --- | --- | --- |
| training_cohort_rate | 0.5 | 0.5 | 37 | 0.0 |
| runin_rate | 0.6950602319974009 | 0.5 | 37 | 0.19506023199740086 |
| runin_rate_shrunk_10 | 0.6950602319974009 | 0.5 | 37 | 0.19506023199740086 |
| expanding_rate | 0.8424691655548098 | 0.5782657484604353 | 37 | 0.26420341709437445 |
| last12_rate | 0.9051800610380594 | 0.7040164428982489 | 37 | 0.20116361813981043 |
| persistence | 0.9055509835192292 | 0.7764710853864096 | 37 | 0.12907989813281961 |
| retrospective_rate | 0.8549228221485199 | 0.5 | 37 | 0.3549228221485199 |
| logistic_regression | 0.9339318875619905 | 0.8055983136687709 | 37 | 0.12833357389321964 |
| gradient_boosting | 0.9352484897162168 | 0.800308448089229 | 37 | 0.1349400416269878 |
| within_centred_gradient_boosting | 0.8927501636862304 | 0.7965719147777331 | 37 | 0.09617824890849724 |
| retrospective exact AUC | 0.8549228221485199 | None | None | None |
| run-in exact AUC | 0.6950602319974009 | None | None | None |
| expanding exact AUC | 0.8424691655548098 | None | None | None |
| normal approx observed SD | 0.8121844236106119 | None | None | None |
| normal approx corrected SD | 0.8110050434693278 | None | None | None |

## Scores and skill

| Forecast | Brier pooled | Brier person mean | Log loss pooled | Log loss person mean | Skill vs references (summed-loss skill; person-mean Brier difference) |
| --- | --- | --- | --- | --- | --- |
| training_cohort_rate | 0.2397895702162751 | 0.24915782630133174 | 0.6725864871726641 | 0.6916116328879859 |  |
| runin_rate | 0.23703938060053453 | 0.23606489183351845 | 1.2081065352195435 | 1.2119010439272542 |  |
| runin_rate_shrunk_10 | 0.22069769155445632 | 0.2220529675094993 | 0.6319682475874144 | 0.6347908953168901 |  |
| expanding_rate | 0.16092370789169094 | 0.1615525676162122 | 0.49913580718547246 | 0.5065818086325882 |  |
| last12_rate | 0.1202521004733707 | 0.12598051715446448 | 0.7595574377867731 | 0.7964360106616932 |  |
| persistence_total_div_12 | 0.19392365023007788 | 0.21142503031144813 | 0.7643783051425165 | 0.8250386162987523 |  |
| retrospective_rate | 0.14951586258368937 | 0.14754300619833613 | 0.4522145302729926 | 0.44644115162381154 |  |
| logistic_regression | 0.0980889715520698 | 0.10261994199611772 | 0.32241333537422223 | 0.33514338771407437 | training_cohort_rate: skill 0.591, person-mean Brier diff -0.147; runin_rate: skill 0.586, person-mean Brier diff -0.133; runin_rate_shrunk_10: skill 0.556, person-mean Brier diff -0.119; expanding_rate: skill 0.390, person-mean Brier diff -0.059; last12_rate: skill 0.184, person-mean Brier diff -0.023; persistence_total_div_12: skill 0.494, person-mean Brier diff -0.109; retrospective_rate: skill 0.344, person-mean Brier diff -0.045 |
| gradient_boosting | 0.09723059399018717 | 0.10152682049329076 | 0.3180356640609568 | 0.3308319853252092 | training_cohort_rate: skill 0.595, person-mean Brier diff -0.148; runin_rate: skill 0.590, person-mean Brier diff -0.135; runin_rate_shrunk_10: skill 0.559, person-mean Brier diff -0.121; expanding_rate: skill 0.396, person-mean Brier diff -0.060; last12_rate: skill 0.191, person-mean Brier diff -0.024; persistence_total_div_12: skill 0.499, person-mean Brier diff -0.110; retrospective_rate: skill 0.350, person-mean Brier diff -0.046 |
| within_centred_gradient_boosting | 0.12885951157605266 | 0.1437970652259363 | 0.41066844985948003 | 0.4477603349501903 | training_cohort_rate: skill 0.463, person-mean Brier diff -0.105; runin_rate: skill 0.456, person-mean Brier diff -0.092; runin_rate_shrunk_10: skill 0.416, person-mean Brier diff -0.078; expanding_rate: skill 0.199, person-mean Brier diff -0.018; last12_rate: skill -0.072, person-mean Brier diff 0.018; persistence_total_div_12: skill 0.336, person-mean Brier diff -0.068; retrospective_rate: skill 0.138, person-mean Brier diff -0.004 |

## Calibration and history bias

| Analysis | Quantity | Estimate | 95% interval |
| --- | --- | --- | --- |
| logistic_regression day_weighted | pooled slope | 1.062389931051896 | [0.9646943905942884, 1.1653770454322894] |
| logistic_regression day_weighted | between / within slope | 1.1157787802104058 / 0.9962545917121238 | [1.0196456392090627, 1.2313907551949752] / [0.8562234753821923, 1.122910762048556] |
| logistic_regression equal_person_weighted | pooled slope | 1.035705270510758 | [0.9384109611826416, 1.1328170742844124] |
| logistic_regression equal_person_weighted | between / within slope | 1.0873831207228275 / 0.9569198700750642 | [0.9934003120229414, 1.1950846079151798] / [0.8447859622793633, 1.0668511309470727] |
| logistic_regression | between / within logit variance share | 0.6103117969092122 / 0.3896882030907879 |  |
| gradient_boosting day_weighted | pooled slope | 1.071626962691515 | [0.9935518529619363, 1.1485539631006032] |
| gradient_boosting day_weighted | between / within slope | 1.138076589881441 / 1.0033948438660771 | [1.0585877692405772, 1.2301818478204525] / [0.8922101061421099, 1.101956441858808] |
| gradient_boosting equal_person_weighted | pooled slope | 1.0541121522087329 | [0.9732976719210914, 1.1323364509114386] |
| gradient_boosting equal_person_weighted | between / within slope | 1.1221224552994105 / 0.9704272915790606 | [1.045351926252059, 1.2132587339600356] / [0.8722960718067818, 1.0666138196119817] |
| gradient_boosting | between / within logit variance share | 0.6301621134115836 / 0.36983788658841643 |  |
| test last12_rate | observed AUC | 0.7040164428982489 |  |
| test last12_rate | permutation reference AUC | 0.472966328291067 |  |
| test last12_rate | observed - permutation AUC | 0.23105011460718217 | [0.18961308585616743, 0.2760909859367635] |
| test expanding_rate | observed AUC | 0.5782657484604353 |  |
| test expanding_rate | permutation reference AUC | 0.4099148240546684 |  |
| test expanding_rate | observed - permutation AUC | 0.16835092440576688 | [0.1076101811281971, 0.22479293500075945] |
| test persistence | observed AUC | 0.7764710853864096 |  |
| test persistence | permutation reference AUC | 0.49602364704604596 |  |
| test persistence | observed - permutation AUC | 0.2804474383403637 | [0.2400432761198285, 0.3166536000399206] |
| all_included last12_rate | observed AUC | 0.6813562313904771 |  |
| all_included last12_rate | permutation reference AUC | 0.47194443721274126 |  |
| all_included last12_rate | observed - permutation AUC | 0.20941179417773575 | [0.19002216800897703, 0.2291308115145908] |
| all_included expanding_rate | observed AUC | 0.5573475893904026 |  |
| all_included expanding_rate | permutation reference AUC | 0.40620735686893217 |  |
| all_included expanding_rate | observed - permutation AUC | 0.15114023252147046 | [0.12618888525615554, 0.17362065249417033] |
| all_included persistence | observed AUC | 0.7648877345016454 |  |
| all_included persistence | permutation reference AUC | 0.49597285555895915 |  |
| all_included persistence | observed - permutation AUC | 0.26891487894268634 | [0.24999818322580217, 0.28787335031457456] |
