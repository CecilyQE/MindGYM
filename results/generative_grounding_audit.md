# Generative grounding audit

Every `make_generative_env` factory entry is annotated with a grounding profile.
Kinds: `transcript_calibrated` (JSONL-derived defaults), `paper_documented` (design from publication), `mixed`, `partial`.

| experiment_id | kind | sources | caveats |
|---------------|------|---------|---------|
| `badham2017deficits/exp1.csv` | paper_documented | paper:Badham2017, psych101:stimulus_inventory | — |
| `bahrami2020four/exp.csv` | mixed | paper:Bahrami2020, psych101:reward_pools | — |
| `cox2017information/exp1.csv` | paper_documented | paper:Cox2017, psych101:word_pool | — |
| `enkavi2019digitspan/exp1.csv` | transcript_calibrated | paper:Enkavi2019, psych101:end_keys_and_lengths | — |
| `enkavi2019gonogo/exp1.csv` | transcript_calibrated | paper:Enkavi2019, psych101:go_keys_and_trial_counts | — |
| `flesch2018comparing/exp1.csv` | transcript_calibrated | paper:Flesch2018, psych101:response_key_pairs | — |
| `frey2017cct/exp1.csv` | paper_documented | paper:Frey2017, psych101:task_rules | — |
| `frey2017risk/exp1.csv` | paper_documented | paper:Frey2017, psych101:instruction | — |
| `gershman2018deconstructing/exp1.csv` | paper_documented | paper:Gershman2018, psych101:instruction | — |
| `gershman2018deconstructing/exp2.csv` | paper_documented | paper:Gershman2018, psych101:instruction | — |
| `gershman2020reward/exp1.csv` | paper_documented | paper:Gershman2020, psych101:instruction | — |
| `hilbig2014generalized/exp1.csv` | paper_documented | paper:Hilbig2014, psych101:expert_weights | — |
| `kool2016when/exp1.csv` | paper_documented | paper:Kool2016, psych101:instruction | — |
| `kool2016when/exp2.csv` | paper_documented | paper:Kool2016, psych101:instruction | — |
| `kool2017cost/exp1.csv` | transcript_calibrated | paper:Kool2017, psych101:session_topologies | — |
| `kool2017cost/exp2.csv` | transcript_calibrated | paper:Kool2017, psych101:session_topologies | — |
| `lefebvre2017behavioural/exp1.csv` | mixed | paper:Lefebvre2017, psych101:reward_pools | — |
| `lefebvre2017behavioural/exp2.csv` | mixed | paper:Lefebvre2017, psych101:reward_pools | — |
| `peterson2021using/exp1.csv` | mixed | psych101:recorded_payoff_pairs, paper:Peterson2021 | Displayed outcome probabilities are fixed at 0.5/0.5 placeholders until Corr joint sampling is implemented. |
| `plonsky2018when/exp1.csv` | transcript_calibrated | paper:Plonsky2018, psych101:trial_counts | — |
| `schulz2020finding/exp1.csv` | paper_documented | paper:Schulz2020, psych101:instruction | — |
| `schulz2020finding/exp2.csv` | paper_documented | paper:Schulz2020, psych101:instruction | — |
| `schulz2020finding/exp3.csv` | paper_documented | paper:Schulz2020, psych101:instruction | — |
| `schulz2020finding/exp4.csv` | paper_documented | paper:Schulz2020, psych101:instruction | — |
| `schulz2020finding/exp5.csv` | paper_documented | paper:Schulz2020, psych101:instruction | — |
| `speekenbrink2008learning/exp1.csv` | paper_documented | paper:Speekenbrink2008, psych101:card_rules | — |
| `steingroever2015data/exp1.csv` | paper_documented | paper:Steingroever2015, psych101:IGT_payoffs | — |
| `steingroever2015data/exp3.csv` | paper_documented | paper:Steingroever2015, psych101:IGT_payoffs | — |
| `tomov2021multitask/exp1.csv` | mixed | paper:Tomov2021, psych101:door_labels | Generative graph is schematic, not transcript-isomorphic. |
| `tomov2021multitask/exp3.csv` | mixed | paper:Tomov2021, psych101:door_labels | Generative graph is schematic, not transcript-isomorphic. |
| `wilson2014humans/exp1.csv` | paper_documented | paper:Wilson2014, psych101:instruction | — |
| `wu2018generalisation/exp1.csv` | paper_documented | paper:Wu2018, psych101:instruction | — |
| `wulff2018description/exp1.csv` | paper_documented | paper:Wulff2018, psych101:lottery_lines | — |
| `wulff2018sampling/exp1.csv` | paper_documented | paper:Wulff2018, psych101:KDX_phases | — |

**Total:** 34 generative environments.
**Partial:** 0 — review before claiming transcript fidelity.
**Mixed:** 6 — paper structure + heuristic or placeholder simulation.
