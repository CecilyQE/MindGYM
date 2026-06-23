# MindGYM Task Type Integration Plan

主要目标是搭建一个和人类被试得到同类 feedback 的 LLM harness： 在同一 task、同一 condition、同一 latent dynamics 下跑 baseline LLM 和加 skill 的 agent， 看行为是否更接近 human transcripts。Psych-201 catalog 用作入口，但 raw `other / uncategorized` 先按 reference 描述清洗归类。

- **76 / 76** Psych-101 experiment_id 在 Psych-201 catalog 中 exact match

- **8 + 38** 8 个 Yifei family + 38 个 Yifei task/env；额外未实现任务按具体内容命名

- **LLM + skill** 比较 baseline agent、skill agent 和 human 行为距离

## 核心结论

到目前为止，关于 MindGYM 两个版本和 family 化改造的判断。

### Cecily 适合作为底层 env

Cecily 的目标更接近真实人类实验模拟：paper/source grounded、transcript audited，并且对 LLM 未在 transcript 中出现的合法动作也应能按 latent dynamics 反馈。

Tags: 机制保真, exact env

### Yifei 适合作为测评层

Yifei 的强项是统一动作接口、runner、logs、seed/batch、metric aggregation 和 leaderboard。它更像 paper-inspired benchmark，不是严格复现实验设计。

Tags: LLM runner, 工程统一

### 两层体系最稳

清洗后的 task type 负责 catalog / coverage；mechanism subtype 负责 env engine 复用；experiment spec 负责 paper、condition、group 和 variant 差异。

Tags: primary_task_type, mechanism template, per-experiment spec

## 研究目标

这个 study 的核心不是替代所有人类实验，而是先验证：如果 LLM/agent 得到和人类被试同构的 observation、action space、feedback 和 history，它的行为能否复现人类的 task-level 与 condition-level 行为模式；再测试加 skill 后是否更接近 human transcripts。

### Harness 要保证什么

同一 paper / condition 下，agent 每一步看到的信息、可选动作、reward/feedback、phase 转换和历史长度，要和人类被试 transcript 对齐。没有这个底座，skill 是否“更像人”无法解释。

Tags: same feedback, same latent dynamics, same metrics

### Skill 实验怎么问

比较 baseline LLM、LLM + skill、human data 三者在 trial choice、learning curve、condition effect、switch/stay、exploration、confidence calibration 等指标上的距离，而不是只看总分。

Tags: baseline agent, skill agent, human distance

### Control group 够不够

不一定只能看 control group。原则是 agent 必须和被比较的人类 subset 处在同一个 condition。control group 最干净；experimental groups 也能用，但必须有 condition metadata 或能从 transcript/prompt/feedback schedule 可靠识别。

Tags: condition-matched, not pooled blindly

### Transcript 能不能推断 group

能，但要分级：metadata 明确记录 group 最可靠；prompt 文案、block schedule、reward mapping、feedback type 可唯一识别时可 inferred；如果只能从行为反推，就只能标 weak-inferred，不能做 exact condition claim。

Tags: recorded, inferred, unknown

| 组件 | 现在缺什么 | 融合后负责什么 |
| --- | --- | --- |
| **Yifei / Psych-201** | 缺 paper-faithful latent dynamics、condition/group fidelity、transcript-grounded feedback generator；很多任务是 paper-inspired parser/benchmark，能跑但不一定和人类被试经历一致。 | 负责大 catalog、batch runner、LLM action interface、logging、baseline aggregation、leaderboard 和 psych201 额外 research 的入口。 |
| **Cecily / Psych-101** | 缺大规模 LLM harness、统一 action adapter、skill-agent 比较层、condition/group metadata 标准化；覆盖面也小于 Psych-201。 | 负责 exact / generative env、human-feedback-equivalent dynamics、transcript audit、fidelity tier、paper-level VariantSpec。 |
| **融合层** | 缺一个明确 contract：哪些 experiment 是 exact human feedback，哪些只是 paradigm_sim；哪些 transcript 有 group，哪些 group 是 inferred。 | 建立 `VariantSpec` + `ConditionSpec` + `LLMEnvAdapter` + `SkillAgentRunner`，让同一个 env 能跑 human replay、baseline LLM、skill agent。 |

## 覆盖关系

本地 catalog 检查得到的关系：Psych-101 可以作为 Psych-201 的严格子集底座。

| 对象 | 当前理解 | 融合含义 |
| --- | --- | --- |
| **Cecily / Psych-101** | 约 76 个 experiment_id；已有 generative env、tiers、experiment specs 和 transcript-aware 逻辑。 | 优先作为 exact / generative env 底座，尤其 Tier A。 |
| **Yifei / Psych-201** | 约 264 个 experiment variants；包含 Psych-101 的 76/76 个 exact experiment_id。 | 用来补更大 catalog、runner、baseline parsing、metric aggregation。 |
| **重合部分** | 不是少量 overlap，而是 Psych-101 被 Psych-201 catalog 完整包含。 | 先让重合的 76 个 experiment 在 Cecily exact env 上跑通 LLM benchmark。 |
| **Psych-201 额外部分** | 超出 Psych-101 的 variants 不能直接承诺 exact。 | 同机制则新增 VariantSpec；规则不同则标 paradigm_sim 或 Tier C。 |

## Task Type

顶层 **Task type** 指心理学任务范式：被试/agent 看见什么、能做什么动作、收到什么 feedback，以及用什么指标和 human transcripts 对齐。 这样分是因为本 study 要比较 baseline LLM、skill agent 和 matched humans 在同一 observation-action-feedback 结构下的行为距离。

Yifei 页面有两层：8 个 **domain family** 和 38 个已实现的 **task/env**。 本表先按 8 个 family 分组；每个已实现任务直接用 Yifei 的 task/env 名作为 **Task type**，这样可以直接对应代码入口、runner 和 logs。 Yifei 尚未覆盖的研究任务用具体任务内容命名，并标为 not included in Yifei's task/env。

每个 task type 下面再拆 **subtype / engine**。 **Task type 共用底座** 是该类任务共享的最小 runner、action parser、feedback loop 和通用 metric。 **继承 / 复用** 表示可从相近 engine 起步。 **本 subtype 改** 表示必须按具体 paper/condition 重建的部分，例如 reward schedule、stimulus materials、latent transition、scoring rule、group/condition metadata。

颜色是 **subtype 层级** 的覆盖状态，不是 task type 整体状态： 绿色 表示 Cecily/Psych-101 已有可作为 generative/exact env 起点的底座； 灰色 表示这个 research 在 Psych-101 里缺材料/condition/latent dynamics，或在下方审计中被判定不能只靠 paper + Psych-201 transcript 搭完整 exact env； 蓝色 表示它属于 Psych-201 扩展集、不在 Psych-101，Cecily 未覆盖，需要先确认材料和机制能否复原。

**other / uncategorized** 不保留为 task type，因为它不是心理学任务范式。 能判断范式的条目已放到对应 task type；仍需核实的条目放在最接近的 subtype 下并标蓝，表示需要确认材料和机制能否复原。

### Risk & Value-based Decision (10 groups)

#### `risk_mpl`

**Task type 共用底座：**described price-list lottery -> switching choice -> risk-aversion summary。*与 risky_choice 共用 lottery parser，但 MPL 重点是 staircase/switching metric。*

##### multiple price list / lottery battery

- **继承 / 复用：**described lottery parser、risk preference summary metrics。
- **本 subtype 改：**price list staircase, switching metric, gain/loss framing, payoff conversion。
- Coverage (not in 101 / confirm): frey2017mpl, frey2017lotteries, spektor2024lossaversion

#### `risky_choice`

**Task type 共用底座：**described gamble / lottery display -> option choice -> EV/CPT-style feature extraction。*这是静态 described-risk choice 的主 engine。*

##### static described gamble choice

- **继承 / 复用：**two-option gamble renderer、EV/CPT features、choice parser。
- **本 subtype 改：**stimulus set, probability/payoff format, feedback policy, theory-specific metrics。
- Coverage (Cecily done): krueger2022identifying, wulff2018description, xiong2023neural, peterson2021using
- Coverage (not in 101 / confirm): Thoma_et_al_2025_risky_choice, hussain2024risk, pirrone_unpublished_lottery, russek2024heuristics

##### value-based two-option choice

- **继承 / 复用：**two-option choice parser and value-feature extraction。
- **本 subtype 改：**food/value stimulus set, reaction-time use, no stochastic outcome unless paper specifies one。
- Coverage (not in 101 / confirm): pirrone_unpublished_food

#### `skewness_choice`

**Task type 共用底座：**described risky stimuli with skewness manipulation -> option choice -> skewness preference metric。

##### skewness / optimized risky stimuli

- **继承 / 复用：**static gamble choice and value-feature extraction。
- **本 subtype 改：**skewness manipulation, optimized stimulus generation, theory-discriminating contrasts。
- Coverage (not in 101 / confirm): olschewski2024skewness, olschewski2025optimal

#### `moral_machine`

**Task type 共用底座：**moral dilemma vignette -> forced choice -> moral-dimension aggregation。*Yifei 放在 Risk family；机制上仍是 moral/social judgment。*

##### moral machine / sacrificial dilemma

- **继承 / 复用：**vignette/choice runner and condition parser。
- **本 subtype 改：**scenario materials, moral dimensions, response labels, scoring interpretation。
- Coverage (not in 101 / confirm): awad2018moral, ciranka_vandenbos_2024

#### `bart`

**Task type 共用底座：**pump/cash action -> explosion or accumulated reward -> risk-taking metrics。

##### classic BART

- **继承 / 复用：**pump/cash state machine、explosion/cashout metrics。
- **本 subtype 改：**threshold distribution, balloon count, money conversion, instructions。
- Coverage (Cecily done): frey2017risk

##### clinical BART variant

- **继承 / 复用：**BART pump/cash engine。
- **本 subtype 改：**clinical metadata, instruction framing, variant-specific thresholds or outcomes。
- Coverage (not in 101 / confirm): pike2023catastrophizing

#### `sampling_choice`

**Task type 共用底座：**latent lottery -> sample/description exposure -> final risky choice。

##### decisions from experience

- **继承 / 复用：**lottery latent state, sample history, final choice parser。
- **本 subtype 改：**sample-then-feedback rule, description/experience condition, stopping/free-sampling rule。
- Coverage (Cecily done): garcia2023experiential, plonsky2018when, wulff2018sampling
- Coverage (not in 101 / confirm): frey2017dfe

#### `columbia_card`

**Task type 共用底座：**sequential card flip / stop -> gain-loss feedback -> risk-taking sensitivity metrics。

##### CCT card-flipping risk task

- **继承 / 复用：**sequential risk state machine and early-stop parser。
- **本 subtype 改：**loss-card probability, gain/loss amount, fresh deck generation, money conversion。
- Coverage (Cecily done): frey2017cct

#### `iowa`

**Task type 共用底座：**four-deck draw -> gain/loss feedback -> advantageous deck preference metrics。

##### classic IGT deck schedules

- **继承 / 复用：**four-deck draw loop, gain/loss bookkeeping, deck-preference metrics。
- **本 subtype 改：**deck payoff schedule, instructions, block metrics, variant-specific deck labels。
- Coverage (Cecily done): steingroever2015data

#### `intertemporal`

**Task type 共用底座：**sooner/later choice -> discounting / deferral metric。

##### delay discounting / intertemporal choice

- **继承 / 复用：**binary choice parser and discount-rate/AUC scoring。
- **本 subtype 改：**amount, delay, gain/loss framing, country/context metadata。
- Coverage (101 material/condition gap): ruggeri2022globalizability

#### not included in Yifei's task/env; **context effects / multi-attribute choice**

**Task type 共用底座：**multi-option attribute table -> choice -> context-effect metric。

##### attraction / compromise / similarity choice sets

- **继承 / 复用：**multi-option choice parser、attribute table renderer、choice-share metrics。
- **本 subtype 改：**decoy construction, attribute normalization, context condition, dominance relation。
- Coverage (not in 101 / confirm): spektor2019contexteffects

##### consumer multi-attribute choice / search or deferral

- **继承 / 复用：**attribute-table choice parser and option/search/deferral action vocabulary。
- **本 subtype 改：**hotel/price-history materials, search option, buy-now vs defer target, price-change history。
- Coverage (not in 101 / confirm): evangelidis2023upscaling, gunadi2021deferral

### Reinforcement Learning (7 groups)

#### `k_armed`

**Task type 共用底座：**arm choice -> stochastic reward/feedback -> trial-level learning metrics。

##### stationary / ordinary k-arm bandit

- **继承 / 复用：**arm/action/reward loop、Q-learning/regret/switch metrics。
- **本 subtype 改：**arm count、reward distribution、trial schedule、instruction framing。
- Coverage (Cecily done): bahrami2020four, feng2021dynamics, gershman2018deconstructing, sadeghiyeh2020temporal, somerville2017charting, waltz2020differential
- Coverage (not in 101 / confirm): anvari2024armed_bandit, anllo2024weird, hartley2024twoarmedbandit, fan2022trait, rutledge2023happiness

##### non-stationary / instructed horizon bandit

- **继承 / 复用：**bandit loop plus drift / horizon / instructed-exposure controls。
- **本 subtype 改：**reward-probability drift, instructed trials, horizon 1/6, safety or exploration constraints。
- Coverage (101 material/condition gap): speekenbrink2008learning
- Coverage (not in 101 / confirm): dubois2022value

#### `counterfactual_bandit`

**Task type 共用底座：**bandit choice loop + chosen/unchosen feedback -> counterfactual learning metrics。

##### counterfactual feedback bandit

- **继承 / 复用：**bandit choice loop、chosen reward update。
- **本 subtype 改：**unchosen-arm feedback, counterfactual logging, learning metric。
- Coverage (not in 101 / confirm): chambon2020feedback

#### `reward_learning`

**Task type 共用底座：**reward observation under different scale/range conditions -> range adaptation metrics。

##### range / magnitude adaptation bandit

- **继承 / 复用：**bandit loop、reward observation、choice parser。
- **本 subtype 改：**reward scale, range normalization, magnitude condition。
- Coverage (not in 101 / confirm): bavard2018magnitude, bavard2021range, bavard2023functional

#### `probability_learning`

**Task type 共用底座：**stimulus -> response -> probabilistic feedback -> matching/maximizing metric。

##### probability learning / zero-outcome variants

- **继承 / 复用：**stimulus-response-feedback loop、learning-rate and bias metrics。
- **本 subtype 改：**rewarded response, probability schedule, omission/zero outcome semantics。
- Coverage (not in 101 / confirm): Thoma_et_al_2025_probability_learning, jagadish2023zero

#### `reversal_learning`

**Task type 共用底座：**stimulus-response learning with changing contingencies -> reversal adaptation metric。

##### reversal / flexibility learning

- **继承 / 复用：**instrumental learning skeleton and block/phase parser。
- **本 subtype 改：**reversal schedule, criterion, feedback semantics, perseveration metric。
- Coverage (not in 101 / confirm): reversal_learning

#### `optimism_learning`

**Task type 共用底座：**probabilistic feedback learning -> asymmetric update / optimism bias metric。

##### optimism / asymmetric RL

- **继承 / 复用：**stimulus-response-feedback loop、learning-rate and bias metrics。
- **本 subtype 改：**positive/negative update asymmetry, belief report, feedback schedule。
- Coverage (Cecily done): lefebvre2017behavioural
- Coverage (not in 101 / confirm): optimism_learning

#### `reward_pairs`

**Task type 共用底座：**action-outcome learning -> devaluation/manipulation -> goal-directed vs habit metric。

##### outcome devaluation / goal-directed choice

- **继承 / 复用：**action-outcome mapping and choice parser。
- **本 subtype 改：**devaluation phase, habit/goal metric, clinical grouping, stimulus set。
- Coverage (not in 101 / confirm): gillan2016characterizing, holton2024goalcommitment, reward_pairs

### Explore-Exploit & Model-Based RL (8 groups)

#### `two_step`

**Task type 共用底座：**stage-1 action -> latent transition -> stage-2 choice/reward -> model-based/model-free readout。

##### canonical two-step

- **继承 / 复用：**stage-1 transition、stage-2 reward drift、stay/MB index。
- **本 subtype 改：**stimulus theme、trial count、transition/reward schedule。
- Coverage (Cecily done): kool2016when, zorowitz2023data
- Coverage (not in 101 / confirm): decker2016twostep, potter2017twostep, shahar2019twosteptask

##### cost / instruction / clinical two-step

- **继承 / 复用：**two-step transition skeleton、model-based/model-free readout。
- **本 subtype 改：**cost calculation、instruction/debrief condition、clinical group metadata、block schedule。
- Coverage (Cecily done): kool2017cost
- Coverage (not in 101 / confirm): castro_rodrigues2022twostep, sandbrink2024metacontrol, nussenbaum2020twostep, phaneuf-hadd_2025_cogeff

#### `spatial_search`

**Task type 共用底座：**grid or spatial latent function -> reveal/search action -> exploration/generalization metric。

##### grid search / spatial generalization

- **继承 / 复用：**grid-state representation、query action parser、spatial generalization metrics。
- **本 subtype 改：**kernel/function prior, payoff landscape, query budget, developmental or empowerment condition。
- Coverage (Cecily done): wu2018generalisation
- Coverage (not in 101 / confirm): Meder (2021), Schulz (2019), braendle2023empowerment, giron2023developmentalExploration

#### `observe_or_bet`

**Task type 共用底座：**observe information or place bet -> hidden-state belief update -> value-of-information metric。

##### observe action vs bet action

- **继承 / 复用：**information-value action parser and belief/update metrics。
- **本 subtype 改：**hidden urn/generative process, observe cost, bet payoff, color/state schedule。
- Coverage (not in 101 / confirm): anvari2024observe_bet

#### `horizon`

**Task type 共用底座：**forced-choice prelude -> horizon-limited free choice -> directed/random exploration metrics。

##### forced-choice prelude + horizon 1/6

- **继承 / 复用：**two-arm payoff sampler、forced schedule、directed/random exploration metrics。
- **本 subtype 改：**horizon length, information asymmetry, payoff setup, block schedule。
- Coverage (Cecily done): wilson2014humans

#### not included in Yifei's task/env; **planning / navigation**

**Task type 共用底座：**graph/map state -> navigation action -> goal/reward/planning metric。

##### castle / graph navigation planning

- **继承 / 复用：**graph transition engine, goal/reward state, planning metrics。
- **本 subtype 改：**map, goal visibility, reward schedule, block structure, discovery condition。
- Coverage (Cecily done): tomov2020discovery

#### not included in Yifei's task/env; **multi-task RL / transfer**

**Task type 共用底座：**task blocks -> repeated action/reward learning -> transfer metric。

##### multi-task graph transfer

- **继承 / 复用：**task-switch runner, reward/action parser, transfer metrics。
- **本 subtype 改：**latent task graph, transfer relation, task order, block schedule。
- Coverage (Cecily done): tomov2021multitask

#### not included in Yifei's task/env; **information sampling**

**Task type 共用底座：**buy/reveal information or answer -> cost/value bookkeeping -> sampling efficiency metric。

##### buy information vs answer

- **继承 / 复用：**buy-info/answer action parser、cost-sensitive sampling metrics。
- **本 subtype 改：**information cost/value, reveal rule, answer scoring, memory subtask linkage。
- Coverage (Cecily done): cox2017information

#### not included in Yifei's task/env; **optional stopping / secretary search**

**Task type 共用底座：**sequential reveal/search -> stop or recall -> payoff and optimal-threshold metric。

##### optional stopping with recall

- **继承 / 复用：**sequential search parser and stopping-position metrics。
- **本 subtype 改：**recall availability, opening cost, sequence length, reward distribution。
- Coverage (not in 101 / confirm): optional_stopping_with_recall

### Reasoning (4 groups)

#### `crt`

**Task type 共用底座：**short problem statement -> answer -> accuracy / intuitive-lure metric。

##### Cognitive Reflection Test

- **继承 / 复用：**multiple-choice/free-answer parser、correctness scoring、lure-choice metric。
- **本 subtype 改：**item bank, numeric answer normalization, intuitive-lure labels, external human baseline。
- Coverage (not in 101 / confirm): crt

#### `matrix_reasoning`

**Task type 共用底座：**visual/symbolic pattern matrix -> option choice -> rule-consistency scoring。

##### Raven-style matrix reasoning

- **继承 / 复用：**matrix/item renderer、option parser、accuracy metric。
- **本 subtype 改：**pattern generator, rule dimensions, distractor construction, visual/text rendering fidelity。
- Coverage (not in 101 / confirm): matrix_reasoning

#### `causal_inference`

**Task type 共用底座：**causal structure -> intervention/prediction response -> outcome/scoring rule。

##### causal prediction / intervention

- **继承 / 复用：**causal graph / intervention API, prediction parser。
- **本 subtype 改：**graph structure, outcome generator, intervention costs, scoring rule。
- Coverage (not in 101 / confirm): cohen2020causal, witte2024interventionStudy, bramley2017

#### not included in Yifei's task/env; **algorithmic advice / augmented decision**

**Task type 共用底座：**base decision problem + optional advice/augmentation -> reliance and accuracy metric。

##### algorithmic augmentation with arithmetic decisions

- **继承 / 复用：**binary decision parser and advice-condition logging。
- **本 subtype 改：**subtraction/divisibility item generator, advice quality, time feedback, reliance scoring。
- Coverage (not in 101 / confirm): xu2023augmenting

### Memory & Cognitive Control (8 groups)

#### `drm`

**Task type 共用底座：**semantic word-list study -> recall/recognition -> false-memory metric。

##### DRM false memory

- **继承 / 复用：**study-list -> recognition/free-recall wrapper。
- **本 subtype 改：**semantic lure lists, false-alarm scoring, external human baseline。
- Coverage (not in 101 / confirm): drm

#### `source_monitoring`

**Task type 共用底座：**study episode with source labels -> source recognition -> source-memory accuracy。

##### source monitoring

- **继承 / 复用：**study/test memory wrapper and source-choice parser。
- **本 subtype 改：**speaker/source assignment, source labels, lure construction, external human baseline。
- Coverage (not in 101 / confirm): source_monitoring

#### `digit_span`

**Task type 共用底座：**ordered sequence presentation -> immediate reproduction -> span length / accuracy metric。

##### digit span

- **继承 / 复用：**sequence presentation, ordered response parser, span/accuracy metric。
- **本 subtype 改：**sequence length, forward/backward rule, adaptive schedule, scoring tolerance。
- Coverage (Cecily done): enkavi2019digitspan

#### `nback`

**Task type 共用底座：**sequential stimuli -> n-back target/non-target response -> accuracy/adaptation metrics。

##### adaptive n-back

- **继承 / 复用：**n-back sequence engine、target/non-target parser、accuracy metrics。
- **本 subtype 改：**adaptive schedule, n-level rule, target mapping, timing。
- Coverage (Cecily done): enkavi2019adaptivenback

#### `stroop`

**Task type 共用底座：**stimulus -> automatic response conflict -> interference-control metric。

##### Stroop / interference control

- **继承 / 复用：**stimulus-response classification runner、congruency-condition metrics。
- **本 subtype 改：**color-word materials, congruent/incongruent mapping, timing/RT policy, interference metric。
- Coverage (not in 101 / confirm): busch2024_stroop

#### not included in Yifei's task/env; **recent-probe / intention / associative memory**

**Task type 共用底座：**study phase -> delay/probe/test phase -> recall/recognition/source scoring。

##### recent-probe working memory

- **继承 / 复用：**study/test phase wrapper、probe presentation、accuracy/RT metrics。
- **本 subtype 改：**probe lag, foil construction, response window, adaptive or fixed schedule。
- Coverage (Cecily done): enkavi2019recentprobes

##### intention / source-memory gap

- **继承 / 复用：**memory phase wrapper and response parsing。
- **本 subtype 改：**missing source/intention materials and latent study/test mapping before exact env。
- Coverage (101 material/condition gap): popov2023intent

##### associative / episodic / hindsight transfer memory

- **继承 / 复用：**study-test scheduler and recall/recognition scoring shell。
- **本 subtype 改：**image/word materials, association graph, lures, transfer labels, free-recall scoring。
- Coverage (not in 101 / confirm): gross2023hindsightTransferLearning, guenther2024associations, haridi2024memory_1, haridi2024memory_2, haridi2024memory_3, rausch_unpublished_replication

#### not included in Yifei's task/env; **memory / chunking**

**Task type 共用底座：**sequence presentation -> ordered response -> chunk-level scoring。

##### sequence chunking / hierarchical chunks

- **继承 / 复用：**sequence presentation, response capture, chunk-level scoring。
- **本 subtype 改：**chunk hierarchy, sequence grammar, response window, scoring decomposition。
- Coverage (Cecily done): wu2023chunking

#### not included in Yifei's task/env; **go / no-go response inhibition**

**Task type 共用底座：**stimulus -> respond/withhold -> commission/omission metrics。

##### go / no-go

- **继承 / 复用：**go/no-go state machine、commission/omission metrics。
- **本 subtype 改：**stimulus-go mapping, timing, feedback, adaptive or fixed schedule。
- Coverage (Cecily done): enkavi2019gonogo
- Coverage (not in 101 / confirm): moutoussis2018pavlovian

### Social & Economic Games (6 groups)

#### `ultimatum`

**Task type 共用底座：**economic game role/action -> payoff and fairness metric。

##### ultimatum responder / proposer variant

- **继承 / 复用：**role/action parser and payoff bookkeeping。
- **本 subtype 改：**offer distribution, responder/proposer role, acceptance threshold metric。
- Coverage (not in 101 / confirm): heffner2022economicgames

#### `repeated_games`

**Task type 共用底座：**repeated 2x2 payoff matrix -> action choice -> cooperation/strategy metric。

##### repeated matrix games

- **继承 / 复用：**game-action runner and payoff aggregation。
- **本 subtype 改：**payoff matrix, partner model, repetition, history window, strategy metric。
- Coverage (not in 101 / confirm): akata2023repeatedgames, heffner2022economicgames

#### `contribution_game`

**Task type 共用底座：**public-goods contribution -> group payoff feedback -> cooperation metric。

##### public goods game

- **继承 / 复用：**contribution parser and group-payoff aggregation。
- **本 subtype 改：**group size, multiplier, feedback visibility, round structure。
- Coverage (not in 101 / confirm): alsobay2025publicGoodsGame, heffner2022economicgames

#### `dictator_game`

**Task type 共用底座：**allocation choice -> recipient/self payoff -> generosity metric。

##### dictator allocation

- **继承 / 复用：**allocation parser and payoff bookkeeping。
- **本 subtype 改：**stake size, recipient framing, anonymity/context manipulation。
- Coverage (Cecily done): hilbig2014generalized

#### `social_hierarchy`

**Task type 共用底座：**pairwise social comparison -> hidden rank/grid update -> transitive-inference metric。

##### social rank / hierarchy learning

- **继承 / 复用：**pairwise comparison parser、training/test phase wrapper、transitive inference metric。
- **本 subtype 改：**competence/popularity grid, person materials, training pairs, test-distance scoring。
- Coverage (not in 101 / confirm): park2021socialhierarchy

#### not included in Yifei's task/env; **strategic / security / matrix games**

**Task type 共用底座：**strategic choice or monitoring decision -> payoff/risk feedback -> strategic metric。

##### security / investment / one-shot matrix variants

- **继承 / 复用：**payoff-matrix or strategic-choice runner。
- **本 subtype 改：**attack/defer monitoring risk, investment race payoff, one-shot matrix-game payoff structure。
- Coverage (not in 101 / confirm): aggarwal2023, hunter2021increased, zhu2024games

### Language & Pragmatics (4 groups)

#### `rsa_pragmatics`

**Task type 共用底座：**text context/question -> choice/rating/free-text response -> pragmatic target scoring。

##### probabilistic pragmatics / RSA-style judgments

- **继承 / 复用：**text prompt runner、choice/rating parser、condition-level aggregation。
- **本 subtype 改：**context construction, literal/pragmatic target, speaker/listener role, answer key。
- Coverage (not in 101 / confirm): hu2023lm-pragmatics, tesslerfranke_2018_not_unreasonable, vantiel2020-probabilistic_pragmatics, vantiel2020probabilistic_pragmatics, vantiel2021probabilisticpragmatics, vantiel2022meaninguse

#### not included in Yifei's task/env; **comprehension / substitution judgments**

**Task type 共用底座：**text material -> forced-choice or judgment response -> semantic target scoring。

##### comprehension / substitution

- **继承 / 复用：**text material loader、forced-choice parser。
- **本 subtype 改：**sentence/item bank, substitution candidates, semantic correctness target。
- Coverage (not in 101 / confirm): guenther2024comprehension, guenther2024substitutions

#### not included in Yifei's task/env; **reference-game reasoning / production**

**Task type 共用底座：**speaker/listener context -> message or referent choice -> pragmatic/reference success metric。

##### reference-game reasoning / production

- **继承 / 复用：**reference-game vignette runner and RSA-style listener/speaker scoring。
- **本 subtype 改：**speaker vs listener role, display composition, target object, message-option set。
- Coverage (not in 101 / confirm): frankedegen2016reasoning-exp1, frankedegen2016reasoning-exp2, franke2024bayesian

#### not included in Yifei's task/env; **lexical / relational / grammaticality / association judgments**

**Task type 共用底座：**lexical/text item -> judgment/free response -> label or semantic scoring。

##### lexical and linguistic judgments

- **继承 / 复用：**text material loader and lexical/free-response parser。
- **本 subtype 改：**lexical-decision target, sensibility/relational labels, grammaticality key, free-association scoring。
- Coverage (not in 101 / confirm): guenther2020LDT, guenther2020TS, guenther2022Relational, guenther2023Grammaticality, guenther2024associations_individual, sun2025rat

### Judgment & Metacognition (6 groups)

#### `metacognition`

**Task type 共用底座：**primary judgment/choice + confidence/rating -> calibration/discrimination metrics。

##### knowledge calibration / Dunning-Kruger

- **继承 / 复用：**question-answer-confidence parser、calibration metrics。
- **本 subtype 改：**question bank, ground truth, confidence scale, feedback policy。
- Coverage (101 material/condition gap): jansen2021dunningkruger

##### choice + confidence / latent belief tasks

- **继承 / 复用：**choice runner plus confidence/belief report parser。
- **本 subtype 改：**belief latent state, confidence scale, sampling paradigm, calibration metric。
- Coverage (not in 101 / confirm): anvari2024sampling_paradigm, baar2021latent, hellmann_unpublished_brightness, schiekiera2025metascience

#### `multi_cue_judgment`

**Task type 共用底座：**cue vector -> criterion/category estimate -> cue-use/function-learning metric。

##### MCPL / cue-to-criterion learning

- **继承 / 复用：**cue table parser、criterion response、learning/test split。
- **本 subtype 改：**cue validity, criterion function, feedback schedule, missing test labels。
- Coverage (101 material/condition gap): collsiöö2023MCPL

##### function estimation / shuffle / yes-no cue judgment

- **继承 / 复用：**cue-to-response shell and feedback/test metrics。
- **本 subtype 改：**function family, shuffled feedback, omission/yes-no bias scoring。
- Coverage (not in 101 / confirm): barnby2022knowing, breslav2022shuffle, cheung2025omissionyesnobias

#### `probability_estimation`

**Task type 共用底座：**probabilistic statement/cue -> numeric probability estimate -> calibration/error metric。

##### Bayesian / probabilistic estimation

- **继承 / 复用：**continuous/numeric estimate parser and calibration metrics。
- **本 subtype 改：**stimulus generation, posterior target, scoring rule。
- Coverage (101 material/condition gap): zhu2020bayesian

##### likelihood / probability / XOR judgments

- **继承 / 复用：**probability judgment parser。
- **本 subtype 改：**premise format, target event, likelihood/rating scale, XOR structure。
- Coverage (not in 101 / confirm): Ying2023NIPE, bhatia2024likelihoodratings, tsvilodub-2023xorsome

#### `things_odd_one_out`

**Task type 共用底座：**triplet display -> odd-one-out / similarity choice -> similarity-reference scoring。

##### THINGS triplet odd-one-out

- **继承 / 复用：**triplet choice parser and similarity-response metric。
- **本 subtype 改：**image/material mapping and ground-truth similarity structure。
- Coverage (101 material/condition gap): hebart2023things

##### visual semantic / spatial triplets

- **继承 / 复用：**triplet/odd-one-out runner。
- **本 subtype 改：**visual materials, semantic embedding target, answer key or human-reference scoring。
- Coverage (not in 101 / confirm): guenther2023ViSpa

#### not included in Yifei's task/env; **perceptual / detection**

**Task type 共用底座：**stimulus -> detection/classification response -> accuracy or discrimination metric。

##### Navon / perceptual detection

- **继承 / 复用：**classification runner、accuracy/RT parsing。
- **本 subtype 改：**visual stimuli, congruency/detail feedback, correct label generation。
- Coverage (not in 101 / confirm): busch2024_navon

##### perceptual numerosity / dot decision

- **继承 / 复用：**perceptual forced-choice parser and accuracy/RT metrics。
- **本 subtype 改：**dot-array generator, numerosity or motion ratio, speed-accuracy tradeoff scoring。
- Coverage (not in 101 / confirm): pirrone_2018_dots

##### phishing / real-world detection

- **继承 / 复用：**classification runner and binary/graded decision parser。
- **本 subtype 改：**email/web materials, ground-truth labels, feedback wording, risk metric。
- Coverage (not in 101 / confirm): singh2019phishing

#### not included in Yifei's task/env; **category learning**

**Task type 共用底座：**stimulus -> category response -> feedback/test accuracy。

##### rule-based category learning

- **继承 / 复用：**stimulus-feedback loop、category-response parser、accuracy metrics。
- **本 subtype 改：**category rule, stimulus generator, block/test schedule。
- Coverage (Cecily done): badham2017deficits
- Coverage (101 material/condition gap): flesch2018comparing

##### revisiting / category variant gap

- **继承 / 复用：**category learning shell。
- **本 subtype 改：**exact materials/rules, feedback timing, transfer/test construction。
- Coverage (101 material/condition gap): levering2020revisiting


## Cecily 未做类别的 env 可搭建性审计

当前 Task Type 表中没有任何绿色 Cecily/Psych-101 底座的类别共 33 个。下面每类选 1-3 篇代表 paper，判断是否能只依靠 paper 描述和 Psych-201 transcript 的 observation/action/feedback 格式搭出完整 text-env。

判定标准：必须能复原 trial generator 或 replay item bank、legal action、feedback/latent schedule、phase transition 和主要 metric。`可完整搭` 表示可作为下一批 Cecily env；`不能完整搭` 表示只能做 transcript replay、text proxy，或需要额外材料，网页中标灰。

### risk_mpl
- frey2017mpl: 可完整搭。MPL 行、A/B lottery、无即时反馈、switching metric 都在 transcript/paper 中可复原。
- frey2017lotteries: 可完整搭。两选 described lottery，概率/金额/动作齐全；随机兑现可按 paper 作为末端 bonus rule。
- spektor2024lossaversion: 可完整搭。50/50 mixed-gamble 对、session/context 与动作键在 transcript 中显式出现。
- Cecily next step: 新增 lottery/MPL VariantSpec；复用 risky-choice parser，补 switching/loss-aversion metric。

### skewness_choice
- olschewski2024skewness: 不能完整搭。自动呈现的 dividend experience stream 和 stock-market latent distribution 不能只靠截断 transcript 稳定生成。
- olschewski2025optimal: 可完整搭。described lottery / sampling block 的 A/B 选择、概率、金额和 sample rule 可从 transcript/paper 实现。
- Cecily next step: 先做 olschewski2025optimal；olschewski2024 只做 transcript replay 或等原始 outcome stream。

### moral_machine
- awad2018moral: 可完整搭。两种 outcome vignette、人物属性、law status、forced choice 全在 trial text 中。
- ciranka_vandenbos_2024: 可完整搭。jar choice、safe/risky rule、bonus、belief/report action 都在 transcript 中，feedback rule 可复原。
- Cecily next step: 可先做 vignette/binary-choice shell，再接 marble-jar social-info variant。

### intertemporal
- ruggeri2022globalizability: 可完整搭。SS/LL 金额、delay、gain/loss framing 与按键都在 transcript 中；无动态 feedback。
- Cecily next step: 新增 delay-discounting VariantSpec 和 AUC/discount-rate metric。

### context effects / multi-attribute choice
- spektor2019contexteffects: 可完整搭。2/3-option choice set、option labels、反馈值和 block 结构在 transcript 中。
- evangelidis2023upscaling: 可完整搭。hotel attributes + search action 是单轮 choice，可直接生成/replay。
- gunadi2021deferral: 可完整搭。price history、buy-now/defer action 明确，属于静态 vignette choice。
- Cecily next step: 做 attribute-table renderer；condition 记录 attraction/compromise/search/deferral。

### counterfactual_bandit
- chambon2020feedback: 可完整搭。free/forced choice、chosen feedback、complete/partial feedback 文本足够搭 bandit loop。
- Cecily next step: 复用 k-arm bandit，补 forced trial 和 unchosen-feedback logging。

### reward_learning
- bavard2018magnitude: 可完整搭。stimulus pair、reward/punishment magnitude、learning/transfer phase 在 transcript 中。
- bavard2021range: 可完整搭。contextual magnitude/range 条件、training/test feedback 可从 transcript/paper 参数化。
- bavard2023functional: 可完整搭。complete feedback for all stimuli 已出现在 transcript，适合做 range/frequency adaptation env。
- Cecily next step: 建立 probabilistic-instrumental-learning shell，单独配置 partial/complete feedback。

### probability_learning
- Thoma_et_al_2025_probability_learning: 可完整搭。two-house prediction、single/both/none outcome 与 trial feedback 齐全。
- jagadish2023zero: 可完整搭。casino/slot-machine options、5-trial blocks、coins feedback 和 zero-outcome semantics 可复原。
- Cecily next step: 补 binary prediction + multi-option zero-outcome 两个 subtype。

### reversal_learning
- suthaharan2021paranoia: 可完整搭。3-arm reward probabilities、block/reversal-like belief updating、feedback 可由 transcript replay/parameterize。
- wise2019acomputational: 不能完整搭。aversive reversal 的 shock-probability belief reports、eye-tracking/attention components不能只靠 transcript 完整复原。
- full_REV_data.csv: 不能完整搭。当前是文件名级条目，缺 paper-level condition/source mapping，不能承诺 exact env。
- Cecily next step: 先做 suthaharan；把 generic reversal_learning tag 拆成具体 paper 后再接。

### reward_pairs
- holton2024goalcommitment: 可完整搭。seafood/net state、changing availability、goal switching cost 和 reward 可由 transcript 搭。
- gillan2016characterizing: 可完整搭。实际更接近 two-step；可复用 two_step skeleton，不应作为 reward_pairs 主代表。
- reward_pairs: 不能完整搭。当前 HTML 只有 generic env id，没有映射到具体 Psych-201 paper/transcript。
- Cecily next step: 把 gillan 迁回 two-step；reward_pairs 需先补具体 source。

### observe_or_bet
- anvari2024observe_bet: 可完整搭。observe/guess action、latent color probability、block change和 feedback 文本完整。
- Cecily next step: 直接新增 observe/bet HMM-like VariantSpec。

### optional stopping / secretary search
- optional_stopping_with_recall: 可完整搭。box values、opening cost、stop/continue/recall payoff rule 在 transcript/paper 中可搭。
- Cecily next step: 实现 sequential search state machine，记录 opening cost 和 recalled best value。

### crt
- crt: 不能完整搭。当前 HTML/Yifei env 是 generic id；Psych-201 catalog 中没有明确 CRT paper/transcript 对应。
- Cecily next step: 先补 paper_key 和 item bank，否则不做 exact claim。

### matrix_reasoning
- matrix_reasoning: 不能完整搭。当前没有 Psych-201 paper/transcript 映射；矩阵图片/选项生成规则不在 HTML 中。
- Cecily next step: 需要 source materials 或生成规则后再搭。

### causal_inference
- cohen2020causal: 可完整搭。gold/rock feedback、territory/intervener rule、yes/no intervention questions在 transcript 中。
- witte2024interventionStudy: 不能完整搭。novel intervention game 的 tutorial/round mechanics 很长，当前 transcript 不足以复原完整 latent game generator。
- bramley2017: 不能完整搭。当前无本地 Psych-201 transcript entry，只能作为待查 paper。
- Cecily next step: 先做 cohen；witte/bramley 标 gray。

### algorithmic advice / augmented decision
- xu2023augmenting: 可完整搭。N1/N2/N3 arithmetic item、time-feedback condition、binary correct key 都在 transcript。
- Cecily next step: 可搭 arithmetic/advice shell，condition 记录 augmentation/time feedback。

### drm
- drm: 不能完整搭。当前只有 Yifei env id，没有对应 Psych-201 paper/transcript；缺 word-list materials 和 lure structure。
- Cecily next step: 需要具体 DRM paper/list materials。

### source_monitoring
- source_monitoring: 不能完整搭。当前只有 env id，没有具体 source-memory paper/transcript mapping；缺 study/test item-source map。
- Cecily next step: 先补 material provenance。

### stroop
- busch2024_stroop: 可完整搭。color-word stimulus、congruency、correct key、timing/RT policy 可从 transcript/paper 搭 text-env。
- Cecily next step: 复用 classification runner，RT 只作 metadata。

### ultimatum
- heffner2022economicgames: 可完整搭。ultimatum/proposer-responder payoff and action format 可从 economic-games variants 生成。
- Cecily next step: 从 heffner battery 中拆 ultimatum VariantSpec。

### repeated_games
- akata2023repeatedgames: 可完整搭。2x2 payoff matrix、same-partner repeated rounds、opponent action/payoff feedback齐全。
- heffner2022economicgames: 可完整搭。economic-games battery 里的 repeated games 可按 payoff matrix replay/生成。
- Cecily next step: 复用 game-action runner，固定或脚本化 opponent schedule。

### contribution_game
- heffner2022economicgames: 可完整搭。4-player public-goods contribution/payoff formula 清楚，可用 fixed other-player schedule。
- alsobay2025publicGoodsGame: 不能完整搭。20-player online game含奖励/惩罚/其他玩家动态；没有完整对手 policy 就不能承诺 counterfactual exact feedback。
- Cecily next step: 先做 heffner PGG；alsobay 仅 replay 或需完整 group logs。

### social_hierarchy
- park2021socialhierarchy: 不能完整搭。当前本地 catalog 没有 experiment transcript；缺 person materials、training pairs 和 test grid。
- Cecily next step: 等 transcript/materials 后再搭。

### strategic / security / matrix games
- aggarwal2023: 可完整搭。IAG target reward/penalty/mProb、attack/defer、monitoring feedback 可搭。
- hunter2021increased: 可完整搭。patent-race endowment/prize/opponent investment/payoff 在 transcript 中。
- zhu2024games: 可完整搭。one-shot 2x2 payoff matrix 和 row choice 完整；无需动态 feedback。
- Cecily next step: 补 security/investment/matrix payoff engines。

### rsa_pragmatics
- hu2023lm-pragmatics: 可完整搭。story、question、multiple-choice options完整，适合 text harness。
- tesslerfranke_2018_not_unreasonable: 可完整搭。gradable predicate/scalar vignette可由 transcript item bank replay。
- vantiel2021probabilisticpragmatics: 可完整搭。scalar/pragmatic inference rating task可按 item text + rating scale搭。
- Cecily next step: 做 pragmatic text-item runner，metric 对 human response distribution。

### comprehension / substitution judgments
- guenther2024comprehension: 可完整搭。sentence、question、one-word answer action完整。
- guenther2024substitutions: 可完整搭。sentence/substitution candidates and judgment response可由 transcript搭。
- Cecily next step: 补 text material loader 和 yes/no/free-response normalizer。

### reference-game reasoning / production
- frankedegen2016reasoning-exp1: 可完整搭。display 被 transcript 文本化为 creature/accessory descriptions，可搭 text reference game。
- frankedegen2016reasoning-exp2: 可完整搭。production/display variants可按 transcript 生成。
- franke2024bayesian: 可完整搭。Bayesian reference-game items可按 vignette/options replay。
- Cecily next step: 若要视觉 exact 需图片；LLM text-env 可先做。

### lexical / relational / grammaticality / association judgments
- guenther2020LDT: 可完整搭。word/nonword item、K/M action、lexical decision labels完整。
- guenther2022Relational: 可完整搭。compound relation selection item/options可搭。
- guenther2023Grammaticality: 可完整搭。sentence grammaticality item and binary/choice response可搭。
- Cecily next step: 优先做 closed-form lexical/grammaticality；free association 单独做 distribution metric。

### metacognition
- jansen2021dunningkruger: 可完整搭。pre/post confidence、20-item quiz、percentile estimate和 answers 可搭。
- baar2021latent: 可完整搭。social prediction game、confidence rating、feedback可由 transcript搭。
- hellmann_unpublished_brightness: 不能完整搭。brightness discrimination依赖视觉刺激和 joystick confidence；text transcript不能保真 human observation。
- Cecily next step: 先做 quiz/social-prediction；视觉 psychophysics 标 gray。

### multi_cue_judgment
- collsiöö2023MCPL: 可完整搭。cue values、criterion estimate、feedback/no-feedback phase齐全。
- barnby2022knowing: 可完整搭。trust/dictator payoff choices和反馈可搭，虽然应迁到 social/economic subtype。
- breslav2022shuffle: 不能完整搭。当前 transcript显示 deck sequence/revealed cards，缺 shuffle/generator rule，不能生成完整 counterfactual env。
- Cecily next step: 先做 MCPL；barnby 迁类；breslav 需 source rule。

### probability_estimation
- zhu2020bayesian: 可完整搭。probability query text、numeric response scale完整。
- Ying2023NIPE: 可完整搭。inverse-planning scenario、candidate goal、1-7 rating完整。
- tsvilodub-2023xorsome: 可完整搭。story、statement、0-100 slider可搭；attention checks按 transcript处理。
- Cecily next step: 补 numeric/rating parser；objective scoring按 paper model或 human distribution。

### things_odd_one_out
- hebart2023things: 不能完整搭。triplet object names足够 replay text task，但原始 THINGS image/material index和完整 similarity reference不在 paper+transcript内。
- guenther2023ViSpa: 可完整搭。word-pair visual-similarity judgments以文本呈现，可按 item set和rating/choice搭。
- Cecily next step: THINGS exact 需 material index；ViSpa text-env 可先做。

### perceptual / detection
- busch2024_navon: 不能完整搭。Navon 依赖大/小字母视觉图形、mask和位置；文字描述会改变 observation。
- pirrone_2018_dots: 不能完整搭。dot-array视觉/数量感知和RT是核心，人类看到的刺激不能只用文本等价。
- singh2019phishing: 可完整搭。email sender/subject/body、ham/phishing label、feedback phase完整。
- Cecily next step: 先做 phishing；视觉 psychophysics 只做非保真 text proxy。

## Cecily / Yifei 同步怎么改

目标是两边同时推进：Cecily 保住 exact feedback/env，Yifei 补齐可批量运行、可对比 skill-agent 的 benchmark 层。两边通过同一套 VariantSpec / ConditionSpec / metric contract 对齐。

### Cecily 侧要加什么

### 先补 VariantSpec / ConditionSpec

每个 experiment 明确 paper、condition、group、phase、reward/feedback schedule、action set、metric。group 记录为 `recorded`、`inferred` 或 `unknown`。

### 加 human transcript replay

用同一个 env 重放 human transcript，检查 observation、legal action、feedback、phase transition 是否和记录一致；这是 LLM harness 的验收测试。

### 加 LLMEnvAdapter

把真实实验动作包装成统一 LLM action schema，但不改变 latent dynamics。adapter 负责 prompt、legal actions、parse、invalid action policy、history window。

### 加 SkillAgentRunner

同一 VariantSpec 下运行 baseline LLM 和 LLM + skill，输出 trial-level log 和 condition-level human-distance metrics。

### 保留 fidelity / claim 标记

每个结果标 `human_feedback_exact`、`condition_matched`、`paradigm_sim` 或 `transcript_bound`，避免把可跑 benchmark 误写成可替代人类被试。

### Yifei 侧要加什么

### 补 psych201 condition catalog

在现有 `psych201_catalog.json` 外增加 condition/group 字段：control/experimental、provenance、feedback type、phase schedule、human subset filter。

### 加 Cecily exact-env bridge

runner 不能只调 Yifei 自己的简化 parser；要能按 experiment_id 调 Cecily exact env，或通过 adapter RPC/package 调用同一套 step/feedback。

### 保留 benchmark fidelity

每个 task 标 `human_feedback_exact`、`condition_matched`、`paradigm_sim`。Yifei 原有 simplified env 可以保留，但只能作为 paradigm baseline。

### 加 skill-agent experiment runner

同一 VariantSpec 下批量跑 baseline LLM、skill agent、ablation skill；输出 trial logs、condition metrics、human-distance summary。

### 加 crosswalk / cleaning layer

维护 raw Psych-201 paper_key -> cleaned task type -> Cecily experiment_id / engine 的映射，特别处理原始 other / uncategorized。

### 两边共享的 contract

| 模块 | Cecily 负责 | Yifei 负责 |
| --- | --- | --- |
| **VariantSpec** | 定义 exact env 参数、latent dynamics、feedback/reward generator、paper-level metric。 | 引用同一 spec 做 catalog、batch selection、leaderboard grouping、Psych-201 crosswalk。 |
| **ConditionSpec** | 验证 condition 是否能生成 human-equivalent feedback；标 group provenance。 | 按 condition 过滤 human subset，不再只做 pooled baseline。 |
| **Human replay** | 重放 transcript，检查 step feedback、phase、reward 和 legal action。 | 导出/规范化 transcript 和 group metadata，接收 replay pass/fail 状态。 |
| **LLMEnvAdapter** | 把 exact env 包成 observation/legal_actions/step，不改变 dynamics。 | 统一模型调用、prompt template、invalid action policy、parallel runner。 |
| **SkillAgentRunner** | 提供可复现 env seeds、trial-level feedback、human-comparable metrics。 | 运行 baseline、skill、ablation，生成 human-distance report。 |
| **Fidelity labels** | 标 `human_feedback_exact` / `transcript_bound` / blocked reason。 | 在结果和 leaderboard 中显示 fidelity，不把 paradigm_sim 混入 exact claim。 |

## 实施优先级

Cecily 和 Yifei 要并行推进：Cecily 先证明 same-feedback exactness，Yifei 同时把 runner/catalog 改成能消费这些 exact specs。

**Phase 1**
### 双仓对齐最小 contract

Cecily 定义 VariantSpec / ConditionSpec 和 replay pass/fail；Yifei 同步加 crosswalk、condition fields、fidelity labels，确保同一个 experiment_id 在两边含义一致。

**Phase 2**
### 选 5-10 个 Tier A exact env 做端到端试跑

Cecily 用 human transcript replay 验证每一步 feedback；Yifei 调 Cecily adapter 跑 baseline LLM 和最小 skill agent，先用 bandit、two-step、BART/CCT 这类 metric 清楚的任务。

**Phase 3**
### 补 group / condition provenance audit

Cecily 负责判断 feedback/condition 是否 exact；Yifei 负责在 catalog 和 human subset filter 里落地 `recorded`、`inferred`、`weak_inferred`、`unknown`。

**Phase 4**
### 再接 Psych-201 额外 variants

Yifei 按清洗后 task type 提供候选和 human data；Cecily 判断能否复原 latent dynamics。同机制且 feedback generator 可复原的才升级为 exact；其余保留 `paradigm_sim`。

**Phase 5**
### 评估“是否可能减少人类被试”

只在 condition-matched、human-feedback-exact、skill agent 明显缩小 human distance 的任务上讨论。结论应限定到具体 task family 和 metric，不能泛化到所有心理学实验。

## 保真原则

这些原则决定什么时候能做 human-matched claim，什么时候只能说是 benchmark 或范式模拟。

### Same-feedback 才能比较

agent 必须看到和 human subset 同构的 observation、legal actions、feedback、reward、history 和 phase transition。只共享题目文本或 headline metric 不够。

Tags: observation, feedback, history

### Condition 不能乱 pooled

control group 最干净，但 experimental group 也能用；关键是 group/condition 必须 recorded 或从 prompt、schedule、feedback type 强推断。unknown 不能做 condition claim。

Tags: recorded, inferred, unknown

### Skill claim 要看距离

skill 有效不是看得分更高，而是看它是否把 agent 的 choice distribution、learning curve、switch/stay、exploration 或 confidence calibration 拉近 matched humans。

Tags: human distance, condition effect

### 能跑不等于能替代人类

Yifei 风格 parser/runner 可以做大规模 benchmark；但只有 exact feedback、condition provenance、human-distance improvement 都成立时，才能讨论某类 research 是否减少人类被试需求。

Tags: benchmark, paradigm_sim, human_feedback_exact
