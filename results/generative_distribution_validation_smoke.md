# Generative Distribution Validation

Fresh `make_generative_env` episodes sampled over multiple seeds. This validates generic rollout and distribution-health invariants, not transcript exact matching.

| status | count |
|--------|-------|
| pass | 5 |
| warning | 0 |
| fail | 0 |

| experiment_id | status | grounding | episodes | unique signatures | steps mean | reward mean | top issue |
|---------------|--------|-----------|----------|-------------------|------------|-------------|-----------|
| `badham2017deficits/exp1.csv` | pass | paper_documented | 2/2 | 2 | 12.0 | 12.00 | — |
| `lefebvre2017behavioural/exp1.csv` | pass | mixed | 2/2 | 2 | 8.0 | 2.75 | — |
| `peterson2021using/exp1.csv` | pass | paper_documented | 2/2 | 2 | 6.0 | 64.00 | — |
| `wilson2014humans/exp1.csv` | pass | paper_documented | 2/2 | 2 | 8.0 | 578.00 | — |
| `wulff2018sampling/exp1.csv` | pass | paper_documented | 2/2 | 2 | 5.0 | 0.00 | — |
