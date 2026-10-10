# REPLACE-BG daily hypoglycaemia forecasting analysis

Primary event: at least 15 minutes below 54 mg/dL in the 00:00–06:00 night, defined by three consecutive 5-minute CGM readings below threshold. Nights and prior predictor days each require at least 70% of nominal CGM slots. The first 14 eligible nights per patient form the forecasting run-in; patients require at least 60 eligible post-run-in nights.

## Cohort

| Quantity | Value |
|---|---:|
| Patients | 224 |
| Scored nights | 46696 |
| Primary events | 2762 |
| Primary event rate | 0.059148535206441664 |
| Mean patient event rate | 0.06056433440895289 |
| SD patient event rate | 0.05208114658115657 |
| Event-rate ICC | 0.04575794948666458 |
| Median eligible-night gap (days) | 1.0 |
| Within-patient pair share | 0.004316040580520574 |
| Night below 70 rate | 0.1583433270515676 |
| Prior-day basal dose estimate available | 0.8963722802809663 |
| Next-24h below 54 rate (covered days) | 0.21454754228679945 |

## Ranking and benchmark

| Forecast | Pooled AUC (95% CI) | Within-patient AUC (95% CI) | Eligible patients |
|---|---:|---:|---:|
| training_cohort_rate | 0.5 ([0.5, 0.5]) | 0.5 ([0.5, 0.5]) | 42 |
| runin_rate | 0.6085700385464616 ([0.5381844805750038, 0.6672948530062349]) | 0.5 ([0.5, 0.5]) | 42 |
| runin_rate_shrunk_10 | 0.6085700385464616 ([0.5381844805750038, 0.6672948530062349]) | 0.5 ([0.5, 0.5]) | 42 |
| expanding_rate | 0.7092734080626513 ([0.6566556953977901, 0.7449734054392649]) | 0.387534378704246 ([0.36047592478968093, 0.4146100529861686]) | 42 |
| recent_28_night_rate | 0.686379331060333 ([0.6369177888464422, 0.7236947406907396]) | 0.4701997955066838 ([0.4482378078684495, 0.49237752975626314]) | 42 |
| persistence | 0.5360795471332424 ([0.5170231720609136, 0.5518700694468107]) | 0.5057970923834996 ([0.49412760382719195, 0.5173432312915822]) | 42 |
| retrospective_rate_reference | 0.7527413881059447 ([0.709817586364785, 0.7833438734872074]) | 0.5 ([0.5, 0.5]) | 42 |
| logistic_regression | 0.8194930460765139 ([0.7925404179253585, 0.8450441362101921]) | 0.7513418508829076 ([0.71543888432066, 0.7852193648274174]) | 42 |
| gradient_boosting | 0.8300423085332057 ([0.8015975495787256, 0.8558599458966517]) | 0.7593517215709531 ([0.7264235307924337, 0.7913820855108029]) | 42 |
| within_patient_centred_gradient_boosting | 0.7847478525661858 ([0.7427421530445844, 0.8226551933954173]) | 0.7590742109189454 ([0.7199126396231991, 0.7948218846726945]) | 42 |
| benchmark | retrospective 0.7527413881059447; run-in 0.6085700385464616; expanding 0.7092734080626513 | — | — |

## Scores and skill

| Forecast | Brier pooled | Brier patient mean | Log-loss pooled | Log-loss patient mean | Skill vs run-in | Skill vs expanding | Skill vs retrospective |
|---|---:|---:|---:|---:|---:|---:|---:|
| training_cohort_rate | 0.056557926983519455 | 0.057107492218055354 | 0.22764767784752016 | 0.22940622402105962 | 0.06528909987658138 | -0.031892667867185365 | -0.053789671191693644 |
| runin_rate | 0.06050847056138062 | 0.06094939292100435 | 0.3914683308140762 | 0.40335492327173356 | 0.0 | -0.10396986675873254 | -0.12739636507133056 |
| runin_rate_shrunk_10 | 0.05705519615960935 | 0.05754263865516397 | 0.22796281193749954 | 0.22991302438276032 | 0.057070925272656114 | -0.04096528498968066 | -0.06305481136768054 |
| expanding_rate | 0.05480989326188236 | 0.05544575862226886 | 0.2297113878812096 | 0.2339711098771684 | 0.0941781745039737 | 0.0 | -0.021220233466497262 |
| recent_28_night_rate | 0.05582220669807985 | 0.0564632623447076 | 0.3008522394456202 | 0.30439641028591363 | 0.07744806338390176 | -0.018469538544084507 | -0.04008169993050559 |
| persistence | 0.10499103091695684 | 0.10654907703436418 | 1.4505055911326763 | 1.4720307921594438 | -0.7351460042351015 | -0.915548903102372 | -0.9561972980426969 |
| retrospective_rate_reference | 0.053670982483212316 | 0.054292638910434565 | 0.2029756093233935 | 0.2055847565054128 | 0.11300051075051154 | 0.02077929203817852 | 0.0 |
| logistic_regression | 0.0495633126556363 | 0.05023563443844968 | 0.18467089456766253 | 0.18739064724203217 | 0.1808863751504245 | 0.09572324071455185 | 0.07653427676418711 |
| gradient_boosting | 0.04787285110584452 | 0.048439608084404895 | 0.17960306733917852 | 0.18180136210359793 | 0.20882397684665244 | 0.12656551113669512 | 0.1080310273653251 |
| within_patient_centred_gradient_boosting | 0.05097890814575136 | 0.05159037862744635 | 0.19357461388729147 | 0.19581222751519156 | 0.15749137810320857 | 0.06989586894151578 | 0.05015884213230137 |

## Calibration and history bias

| Analysis | Forecast | Pooled / observed | Between / reference | Within / difference |
|---|---|---:|---:|---:|
| Calibration slopes: logistic_regression | Day weighted | 1.154582886128714 | 1.4373831333682132 | 1.031724292334079 |
| Calibration slopes: gradient_boosting | Day weighted | 1.0631441622304512 | 1.5335483961506173 | 0.954578903179195 |
| History bias | recent_28_night_rate | 0.47019979550668367 | 0.4574482601038805 | 0.012751535402803129 |
| History bias | expanding_rate | 0.387534378704246 | 0.37321437802003155 | 0.01432000068421442 |
| History bias | persistence | 0.5057970923834995 | 0.49787423538748854 | 0.007922856996010955 |

Treatment assignment and a randomization day are not available in the inspected HRandomization table, so these results do not estimate treatment effects or separate protocol phases. Bolus and carbohydrate variables indicate pump-record availability; missing entries are not evidence of zero exposure. Basal insulin is estimated from recorded rate multiplied by actual duration, split at midnight; it may be incomplete when pump records are absent.
