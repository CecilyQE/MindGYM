# MindGYM — PsycEnvir Generative Export

从 [`PsycEnvir`](../PsycEnvir) 导出的 **generative environment** 代码、文档、数据与验收结果快照（2026-05-30）。

## 目录结构

```
MindGYM/
  README.md                 # 本文件
  src/psycenvir/
    generative/             # 44 个 generative env 实现
    core/                   # generative_registry, registry, base, replay, gymnasium
    sim/                    # transcript-bound exact 后端（from_transcript 依赖）
    audit/                  # transcript_human_path 审计
    psych101/               # transcript 解析
    session_texts/          # 指令注册表（grounding / 指令对齐）
    experiment_specs/       # Tier A 实验 YAML
    models.py, errors.py, specs.py
  scripts/
    validate_generative_distribution.py
    run_transcript_human_path_audit.py
    audit_generative_grounding.py
    evaluate_generative.py
  tests/                    # generative + transcript audit + 各 family smoke tests
  docs/
    GENERATIVE_SUMMARY.md   # generative 工作总结（主文档）
    SIMENV_DESIGN.md        # 设计文档（含 §2.8 / §8.9 generative）
  data/
    generated/              # generative_setting_tiers.yaml, calibration, manifest
    raw/prompts_training.jsonl   # not in git (~819MB); see below
    external/choices13k/    # Peterson generative 依赖
  results/                  # distribution validation, grounding audit, transcript audit
```

## 快速使用

```bash
cd MindGYM
export PYTHONPATH=src

# Fresh generative episode
python -c "from psycenvir import make_generative_env; e=make_generative_env('badham2017deficits/exp1.csv', seed=0); print(e.reset()[1])"

# 跑 generative 测试
python -m pytest tests/test_generative_setting_tier_a.py tests/test_generative_envs.py -q

# 多 seed 分布验收
python scripts/validate_generative_distribution.py

# Transcript-bound 全量审计（慢；需 jsonl）
python scripts/run_transcript_human_path_audit.py
```

### 数据：`prompts_training.jsonl`

未纳入 git（单文件 ~819MB，超 GitHub 限制）。本地放至 `data/raw/prompts_training.jsonl`，可从 [PsycEnvir](../PsycEnvir) 复制，或按 Psych-101 / HuggingFace 导出获取。

## 未包含（有意排除）

- **Policy eval** track（`policy_eval.py`, `run_policy_eval.py`, `POLICY_EVAL_METRICS.md`）
- **Recorded-only** sim 扩展与 benchmark/agent 全流程
- **Session texts** 构建脚本与 `session_texts.yaml`（仅保留 instruction registry 源码）
- 非 generative 的 results / 其它 docs

## 来源与同步

本目录为 **只读快照**。若在 PsycEnvir 主仓继续开发 generative，请重新导出或直接从 `PsycEnvir` 工作。

关键验收状态（详见 `docs/GENERATIVE_SUMMARY.md`）：

| 验收 | 结果 |
|------|------|
| Transcript-bound audit | 64 experiments, 44,452 sessions, 0 fail |
| Fresh distribution (Tier A, 5 seeds) | 53 pass / 5 warning / 0 fail |
| Tier A 注册 | 58 experiments |
