# Unsupported Generative Source Audit

Date: 2026-05-30

Scope: the current 18 `tier_C` experiment ids in `data/generated/generative_setting_tiers.yaml`, plus upgrade/demotion notes from the 2026-05-30 generative eligibility pass.
Goal: identify what is missing for a paper-setting `make_generative_env`, and whether external sources can plausibly fill the gap.

## Source Policy

An experiment can be upgraded from unsupported only if the full generative setting can be specified from paper, supplement, OSF/GitHub/task code, or an equivalent accurate source. Psych-101 transcripts may be used as validation targets and to recover action labels or observed trial layouts, but they must not be the only source for hidden reward rules or full counterfactual feedback.

For stochastic reward tasks, exact transcript matching is not required for a fresh seed. Validation should combine:

- human-path conformance with transcript-observed random draws bound as latent outcomes;
- distributional validation over many seeds against paper/source probabilities.

## External Source Leads

| Source | Use |
|---|---|
| Psych-101 on Hugging Face | Local transcript source; 60,092 rows and 76 experiment ids. Useful for validation and label/action extraction, not sufficient by itself for hidden generative rules. |
| Psych-201 / Psych-201-discrete | Potential row-level structured successor to Psych-101; useful to check whether currently text-only tasks have structured columns. |
| THINGS / THINGS OSF | Candidate source for `hebart2023things/exp1.csv` stimulus concepts/images and similarity/odd-one-out metadata. |
| Original paper supplements / OSF / GitHub | Required when the transcript does not expose hidden mappings, distributions, or non-press scoring rules. |
| choices13k GitHub | Example of an acceptable external source: it exposes `Corr` and `Amb` fields for Peterson-style gambles, so paper-external but accurate task schema can make a task generative-eligible. |

## Upgrade Triage

Legend:

- **Upgrade-now**: current transcript + paper-level rule in prompt is probably enough for a first generative implementation.
- **Source-needed**: likely implementable, but needs original paper/supplement/code or structured data before registering fresh generative.
- **Interface-needed**: setting may be recoverable, but current `press key -> feedback` interface is too narrow.
- **Keep-unsupported**: do not register generative until both source and interface issues are resolved.

## Confirmed Moves Into Generative Groups

These ids have enough information from the transcript instruction and/or an existing isomorphic engine to be moved out of the unsupported list for design purposes. They still need code registration and tests before `make_generative_env` should expose them.

| experiment_id | New group | Why confirmed |
|---|---|---|
| `enkavi2019adaptivenback/exp1.csv` | deterministic rule-feedback | Instructions specify N, match/non-match keys, adaptive thresholds, and block lengths; RT can be omitted in v0. |
| `gershman2020reward/exp1.csv` | deterministic rule-feedback with sampled mapping | Paper-style fresh episodes can sample game-local stimulus-response mappings; transcript-exact audit remains partial for unrevealed labels. |
| `ruggeri2022globalizability/exp1.csv` | deterministic/no-reward choice task | Option texts and action keys are explicit; no hidden reward/correctness is required. |
| `wu2023chunking/exp1.csv`, `exp2.csv` | deterministic rule-feedback | Each trial states the instructed key and correctness rule; RT/chunk timing can be optional v1 metadata. |
| `feng2021dynamics/exp1.csv` | stochastic reward / latent-state | Transcript is isomorphic to Wilson two-arm instructed/free slot-machine task. |
| `ludwig2023human/exp0.csv`–`exp2.csv` | stochastic/latent-state market task | Instruction specifies two-step market navigation and dot-product reward; fresh market layouts can be sampled per block. |
| `peterson2021using/exp1.csv` | stochastic risky-choice task | choices13k exposes problem distributions plus `Corr` and `Amb`, so fresh joint outcome sampling no longer needs placeholder probabilities. |
| `sadeghiyeh2020temporal/exp1.csv` | stochastic reward / latent-state | Transcript is isomorphic to Wilson two-arm instructed/free slot-machine task. |
| `somerville2017charting/exp1.csv` | stochastic reward / latent-state | Transcript is isomorphic to Wilson two-arm instructed/free slot-machine task. |
| `waltz2020differential/exp1.csv` | stochastic reward / latent-state | Transcript is isomorphic to Wilson two-arm instructed/free slot-machine task. |
| `xiong2023neural/exp1.csv` | stochastic reward / latent-state | Instructions state hazard-rate reward changes and uniform reset range; distributional checks are well-defined. |
| `zorowitz2023data/exp1.csv` | stochastic/latent-state two-step task | Instruction specifies ship transition uncertainty, planet-specific alien choices, and slowly drifting treasure probabilities. |

| experiment_id | Current missing data / blocker | Candidate fill source | Recommendation |
|---|---|---|---|
| `collsiöö2023MCPL/exp2.csv` | Test-phase stimulus-label combinations are not recovered from each session's feedback phase. | MCPL paper/supplement; possibly infer rule family from exp1/exp3 if documented. | Source-needed. Upgrade only after rule/label mapping is source-backed. |
| `enkavi2019adaptivenback/exp1.csv` | Adaptive n-back state machine and no-press/press timing are not implemented. | Transcript instructions already specify N update thresholds and block lengths. | Upgrade-now for correctness-only text env; ignore RT initially. |
| `feng2021dynamics/exp1.csv` | Marked unreviewed, but transcript is Wilson-like instructed/free two-arm bandit. | Reuse Wilson two-arm slot engine after confirming paper parameters. | Upgrade-now / Source-light. |
| `gershman2020reward/exp1.csv` | Some session-local stimulus-response labels never appear with positive feedback, blocking exact transcript reconstruction. | Paper rule permits fresh random mapping; transcript exact audit can skip unrecovered labels. | Upgrade-now for fresh generative; exact transcript audit remains partial. |
| `hebart2023things/exp1.csv` | Needs THINGS object concept/image set and odd-one-out stimulus triplets or similarity labels. | THINGS initiative / OSF / THINGS-data. | Source-needed. High value but separate stimulus project. |
| `jansen2021dunningkruger/exp1.csv` | Grammar quiz plus self-report estimates; `You say` numeric/free responses, not press-only. | Original quiz items/answer key if available; Psych-101 text may include item wording. | Interface-needed; keep unsupported for current press Sim. |
| `kumar2023disentangling/exp1.csv` | 7x7 coordinate action API and hidden binary pattern generator are missing. | Original task code/paper/supplement needed for pattern family. | Source-needed + interface-needed. |
| `levering2020revisiting/exp1.csv` | Category-learning rule families and test-block labels are not registered. | Paper/supplement; feedback phase may recover some rules but should not be sole source. | Source-needed. |
| `levering2020revisiting/exp2.csv` | Same as exp1. | Same as exp1. | Source-needed. |
| `popov2023intent/exp1.csv` | Multi-cycle memory task with size judgments, timed arithmetic, and free recall scoring. | Word lists and scoring rules from paper/supplement/task code. | Interface-needed; keep unsupported for current press Sim. |
| `popov2023intent/exp2.csv` | Same family, word-pair judgments and recall. | Same as exp1. | Interface-needed; keep unsupported. |
| `popov2023intent/exp3.csv` | Same family, word-pair judgments and recall. | Same as exp1. | Interface-needed; keep unsupported. |
| `ruggeri2022globalizability/exp1.csv` | Fixed intertemporal/risk preference choice list; no objective reward/correctness. | Transcript has option text; paper/supplement can validate item list. | Upgrade-now as no-reward choice env; not RL reward env. |
| `sadeghiyeh2020temporal/exp1.csv` | Marked unreviewed, but transcript is Wilson-like instructed/free two-arm bandit. | Reuse Wilson two-arm slot engine after confirming paper parameters. | Upgrade-now / Source-light. |
| `somerville2017charting/exp1.csv` | Marked unreviewed, but transcript is Wilson-like instructed/free two-arm bandit. | Reuse Wilson two-arm slot engine after confirming paper parameters. | Upgrade-now / Source-light. |
| `tomov2020discovery/exp2.csv`, `exp4.csv`, `exp5.csv`, `exp7.csv` | Current fresh graph generator is schematic rather than the paper/source graph setting. | Original task code/supplement or structured graph parameters. | Demoted; keep recorded-path validation only until exact graph generator is implemented. |
| `tomov2021multitask/exp1.csv`, `exp3.csv` | Current castle generator is schematic and not source-isomorphic. | Original room graph, resource distributions, market process, and transition parameters. | Demoted; unsupported until full process is recovered. |
| `waltz2020differential/exp1.csv` | Marked unreviewed, but transcript is Wilson-like instructed/free two-arm bandit. | Reuse Wilson two-arm slot engine after confirming paper parameters. | Upgrade-now / Source-light. |
| `wise2019acomputational/exp1.csv` | Continuous probability rating bar, visual outcomes, shock scheduling, and non-press action interface. | Original task code/paper parameters for shock volatility and visual outcome process. | Source-needed + interface-needed. |
| `wu2023chunking/exp1.csv` | Sequence/chunking with RT; current env lacks RT/chunk generator. | Transcript instruction gives instructed-key feedback; paper needed for sequence generator. | Upgrade-now for deterministic correctness without RT; distribution/RT later. |
| `wu2023chunking/exp2.csv` | Same as exp1 with errors and RT. | Same as exp1. | Upgrade-now for deterministic correctness without RT; distribution/RT later. |
| `xiong2023neural/exp1.csv` | Hazard-rate restless bandit engine not implemented. | Transcript instruction exposes hazard rate and uniform reset range; paper can confirm parameters. | Upgrade-now / Source-light for distributional bandit. |
| `zhu2020bayesian/exp1.csv` | Typed probability estimates and normative query model; no press-choice interface. | Original query set / Bayesian model from paper/supplement. | Interface-needed; keep unsupported until estimate action API exists. |
| `zhu2020bayesian/exp2.csv` | Same as exp1. | Same as exp1. | Interface-needed; keep unsupported until estimate action API exists. |

## Recommended Upgrade Order

1. **Wilson-like bandits**: `feng2021dynamics`, `sadeghiyeh2020temporal`, `somerville2017charting`, `waltz2020differential`.
   - Reuse `TwoArmSlotGenerativeEnv`.
   - Validate instructed/free phase counts and reward distribution.

2. **Deterministic/near-deterministic correctness tasks**: `enkavi2019adaptivenback`, `wu2023chunking` exp1/2, `gershman2020reward`.
   - Add task-specific parsers and rule engines.
   - Ignore RT in v0; store RT as optional audit field only.

3. **Hazard / two-step bandits**: `xiong2023neural`, `zorowitz2023data`, `ludwig2023human`.
   - Need source lookup before final registration.

4. **External stimulus / non-press tasks**: `hebart2023things`, `zhu2020bayesian`, `wise2019acomputational`, `popov2023intent`, `jansen2021dunningkruger`, `kumar2023disentangling`.
   - Do not register as current press-action generative until either the interface or external assets are added.

## Notes

- `Psych-101` can validate text feedback and action formatting, but it is not by itself a full source for hidden task generators.
- `Psych-201` should be checked before hand-coding difficult tasks; it may already expose structured variables missing from Psych-101 text.
- `THINGS` appears source-fillable because the THINGS initiative and OSF repository expose object concepts/images/metadata.
- Peterson is upgraded after connecting `choices13k` problem distributions, `Corr`, and `Amb` as the joint outcome/display model.
