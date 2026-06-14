# Generative Distribution Validation

Fresh `make_generative_env` episodes sampled over multiple seeds. This validates generic rollout and distribution-health invariants, not transcript exact matching.

| status | count |
|--------|-------|
| pass | 53 |
| warning | 5 |
| fail | 0 |

| experiment_id | status | grounding | episodes | unique signatures | steps mean | reward mean | top issue |
|---------------|--------|-----------|----------|-------------------|------------|-------------|-----------|
| `badham2017deficits/exp1.csv` | pass | paper_documented | 5/5 | 5 | 12.0 | 12.00 | — |
| `bahrami2020four/exp.csv` | pass | mixed | 5/5 | 5 | 8.0 | 420.44 | — |
| `collsiöö2023MCPL/exp1.csv` | pass | paper_documented | 5/5 | 5 | 10.0 | 1.40 | — |
| `collsiöö2023MCPL/exp3.csv` | pass | paper_documented | 5/5 | 5 | 10.0 | 1.40 | — |
| `cox2017information/exp1.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | 8.00 | — |
| `enkavi2019adaptivenback/exp1.csv` | pass | paper_documented | 5/5 | 5 | 12.0 | 12.00 | — |
| `enkavi2019digitspan/exp1.csv` | pass | transcript_calibrated | 5/5 | 5 | 11.8 | 0.60 | — |
| `enkavi2019gonogo/exp1.csv` | pass | transcript_calibrated | 5/5 | 5 | 12.0 | 10.80 | — |
| `enkavi2019recentprobes/exp1.csv` | pass | transcript_calibrated | 5/5 | 5 | 8.0 | 0.00 | — |
| `flesch2018comparing/exp1.csv` | pass | transcript_calibrated | 5/5 | 5 | 10.0 | -130.00 | — |
| `frey2017cct/exp1.csv` | pass | paper_documented | 5/5 | 5 | 15.6 | 1158.60 | — |
| `frey2017risk/exp1.csv` | pass | paper_documented | 5/5 | 5 | 17.2 | 0.00 | — |
| `garcia2023experiential/exp1.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | 0.00 | — |
| `garcia2023experiential/exp2.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | 0.00 | — |
| `garcia2023experiential/exp3.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | 0.00 | — |
| `garcia2023experiential/exp4.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | 0.00 | — |
| `gershman2018deconstructing/exp1.csv` | pass | paper_documented | 5/5 | 5 | 10.0 | -9.80 | — |
| `gershman2018deconstructing/exp2.csv` | pass | paper_documented | 5/5 | 5 | 10.0 | 19.20 | — |
| `gershman2020reward/exp1.csv` | pass | paper_documented | 5/5 | 5 | 10.0 | 5.00 | — |
| `hilbig2014generalized/exp1.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | 4.20 | — |
| `kool2016when/exp1.csv` | pass | paper_documented | 5/5 | 5 | 4.0 | 11.40 | — |
| `kool2016when/exp2.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | 7.40 | — |
| `kool2017cost/exp1.csv` | pass | transcript_calibrated | 5/5 | 5 | 4.0 | 35.60 | — |
| `kool2017cost/exp2.csv` | pass | transcript_calibrated | 5/5 | 5 | 8.0 | 10.59 | — |
| `krueger2022identifying/exp1.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | -110.60 | — |
| `lefebvre2017behavioural/exp1.csv` | pass | mixed | 5/5 | 5 | 8.0 | 2.10 | — |
| `lefebvre2017behavioural/exp2.csv` | pass | mixed | 5/5 | 5 | 8.0 | 2.10 | — |
| `ludwig2023human/exp0.csv` | pass | paper_documented | 5/5 | 5 | 12.0 | -2.00 | — |
| `ludwig2023human/exp1.csv` | pass | paper_documented | 5/5 | 5 | 12.0 | -2.00 | — |
| `ludwig2023human/exp2.csv` | pass | paper_documented | 5/5 | 5 | 12.0 | -2.00 | — |
| `peterson2021using/exp1.csv` | pass | paper_documented | 5/5 | 5 | 6.0 | 76.00 | — |
| `plonsky2018when/exp1.csv` | pass | transcript_calibrated | 5/5 | 5 | 12.0 | 240.76 | — |
| `schulz2020finding/exp1.csv` | pass | paper_documented | 5/5 | 5 | 10.0 | 228.73 | — |
| `schulz2020finding/exp2.csv` | pass | paper_documented | 5/5 | 5 | 10.0 | 228.73 | — |
| `schulz2020finding/exp3.csv` | pass | paper_documented | 5/5 | 5 | 10.0 | 228.73 | — |
| `schulz2020finding/exp4.csv` | pass | paper_documented | 5/5 | 5 | 10.0 | 228.73 | — |
| `schulz2020finding/exp5.csv` | pass | paper_documented | 5/5 | 5 | 10.0 | 228.73 | — |
| `speekenbrink2008learning/exp1.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | 3.40 | — |
| `steingroever2015data/exp1.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | -770.04 | — |
| `steingroever2015data/exp2.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | -770.04 | — |
| `steingroever2015data/exp3.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | -770.04 | — |
| `wilson2014humans/exp1.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | 527.60 | — |
| `wilson2014humans/exp2.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | 527.60 | — |
| `wilson2014humans/exp3.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | 527.60 | — |
| `wilson2014humans/exp4.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | 527.60 | — |
| `wilson2014humans/exp5.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | 527.60 | — |
| `wu2018generalisation/exp1.csv` | pass | paper_documented | 5/5 | 5 | 4.0 | 199.40 | — |
| `wu2023chunking/exp1.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | 8.00 | — |
| `wu2023chunking/exp2.csv` | pass | paper_documented | 5/5 | 5 | 8.0 | 8.00 | — |
| `wulff2018description/exp1.csv` | pass | paper_documented | 5/5 | 5 | 3.0 | 4.26 | — |
| `wulff2018sampling/exp1.csv` | pass | paper_documented | 5/5 | 5 | 5.0 | 0.00 | — |
| `xiong2023neural/exp1.csv` | pass | paper_documented | 5/5 | 5 | 10.0 | 401.20 | — |
| `zorowitz2023data/exp1.csv` | pass | paper_documented | 5/5 | 5 | 16.0 | 3.20 | — |
| `feng2021dynamics/exp1.csv` | warning | paper_documented | 5/5 | 5 | 300.0 | 18327.80 | step budget exhausted before episode end |
| `ruggeri2022globalizability/exp1.csv` | warning | paper_documented | 5/5 | 1 | 10.0 | 0.00 | — |
| `sadeghiyeh2020temporal/exp1.csv` | warning | paper_documented | 5/5 | 5 | 300.0 | 18327.80 | step budget exhausted before episode end |
| `somerville2017charting/exp1.csv` | warning | paper_documented | 5/5 | 5 | 300.0 | 18327.80 | step budget exhausted before episode end |
| `waltz2020differential/exp1.csv` | warning | paper_documented | 5/5 | 5 | 300.0 | 18327.80 | step budget exhausted before episode end |
