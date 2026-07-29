---
title: Vitality Scoring Long-Term Evolution
status: idea
owner: patchouli
scope: vitality-and-lifecycle-evolution-exploration
related_current:
  - docs/patchouli/lifecycle.md
  - docs/patchouli/memory-library.md
  - docs/patchouli/retrieval.md
last_reviewed: 2026-07-29
---

# 生命力分数计算：长期演进方向

本文创建于 2026-07-26，源于“生命力分数 100→67”缺陷修复后的进一步思考。它保存候选方向，不承诺实施顺序；当前公式、强化事件、gardening 与 archive/revive 语义以[记忆生命周期](../patchouli/lifecycle.md)为准，存储状态以 [MemoryLibrary](../patchouli/memory-library.md)为准。

## 0. 复核结论与设计矛盾

当前三段式公式、四种强化事件、定期 gardening 和显式 archive/revive 已经落地，因此本文对它们的描述仍可作为问题背景。其余五个方向均未实现：`MetaData` 没有 per-memory `decay_rate`、review/salience 字段，Lifecycle 没有 Promoter 或 INTERFERENCE 事件，也没有持久化复习历史。

其中“短→中→长逐层晋升”的原始设想与当前存储语义存在关键冲突。ShortTermMemoryStore 保存的是对话 topic buffers，并不是 MemoryAtom 的初级层；Generation 从结算材料生成 MemoryAtom 后直接写入 MidTerm；LongTerm 当前是退出普通检索热集合的 archive store，而不是高价值记忆的成熟层。未来若研究 consolidation，必须先定义“语义巩固”究竟意味着生成摘要、合并版本、冻结策略还是新的存储状态，不能直接复用现有 `archive/revive` 名称制造相反语义。

这个 Idea 升级为 Plan 前至少需要：

1. 收集真实命中、引用、反馈、归档与复活数据，建立当前公式的校准基线；
2. 每次只引入一个可解释的新状态，证明收益超过参数和迁移复杂度；
3. 明确事件历史、用户隔离、schema migration、回滚和旧 atom 默认值；
4. 若涉及 retrieval，先证明新信号不会把“需要复习”误当成“与当前问题相关”；
5. 若涉及 consolidation，先形成与当前 MemoryLibrary 状态机一致的独立语义，再建立可验收 Plan。

## 背景

本次 bug 的直接修复采用了**三段式语义**公式：

```
V(t) = V_0 · D(t) + A(access) + B(events)
- V_0   = 100                  # 初始强度，固定高值，与 confidence 解耦
- D(t)  = exp(-λ_eff · t)      # 艾宾浩斯式时间衰减
         λ_eff = λ · (2 - I)   # 高价值记忆 (I 接近 1) 衰减更慢
- A     = k · log(1 + n)       # 访问加成，对数曲线自然饱和
- B     = event_vitality_boost # 事件累积加成，单独存储
```

这套公式已能满足"记忆最初高值 → 随时间衰退 → 命中/反馈事件增强"的基本语义，止住了 confidence 压低起点造成的假性跳变。但作为**模拟人类记忆衰退与强化的系统**，仍有多个维度的能力缺口，下文按优先级整理。

## 优先级总览

下表是待验证的优先级假设，不是实施排期；真实顺序必须由使用数据、schema 成本和产品语义共同决定。

| 方向 | 假设 ROI | 与现有架构契合度 | 新概念引入 | 验证顺序建议 |
|---|---|---|---|---|
| 1. 自阻尼 λ (per-memory) | 高 | 高 | 极少 | 1st |
| 2. 巩固化循环（语义待定义） | 潜在高 | 待澄清（当前分层不是成熟度阶梯） | 高 | 证据后再排 |
| 3. FSRS / SM-2 复习时机 | 中高 | 中 | 算法引进 | 3rd |
| 4. 来源 salience 进公式 | 中 | 高 | 少 | 4th |
| 5. 干扰建模 | 中 | 中 | 少 | 5th |

---

## 方向 1: 自阻尼 λ (per-memory decay rate) — *优先评估候选*

### 核心想法

将衰减率 `λ` 从全局配置改为**每条记忆的私有字段**。每次成功的 HIT/CITATION 让该记忆的 `λ *= 0.85`（举例），即**"被反复回忆成功的记忆衰减得更慢"**。

这是艾宾浩斯曲线最核心的生物学直觉：**记住一次，下次遗忘得更慢**。当前全局 λ 让"被反复验证的核心记忆"和"昙花一现的记忆"走同一条衰减曲线，丢失了分化信号。

### 落点

- `MetaData` 新增 `decay_rate: float` 字段（初始化为 `config.decay_lambda * (2 - I)`）
- `VitalityCalculator._calculate_decay` 读取 `memory.meta.decay_rate`，不再依赖全局 `λ_eff` 计算
- `DynamicReinforcementEngine.reinforce` 在 HIT/CITATION/FEEDBACK_POSITIVE 后递减 `decay_rate`：
  `decay_rate *= 0.85`，下限 `0.1 * config.decay_lambda`
- FEEDBACK_NEGATIVE 反向：`decay_rate *= 1.2`，上限 `2 * config.decay_lambda`

### 期望效果

- 单次 HIT 立刻有体感：下次重算时分数明显高于"未被命中的同年龄记忆"
- 自然累积出"长尾核心记忆"与"昙花一现记忆"的分化
- 无需冷路径同步——`decay_rate` 存在记忆本身，重算即生效

### 验证手段

加一个调试脚本：可视化一条记忆在「30 天内被 HIT 5 次」与「30 天内未被命中」的活力曲线对比，直观看到自阻尼带来的累积差异。

---

## 方向 2: 巩固化循环 (Consolidation Loop) — *需要先重定义存储语义*

### 核心想法

当前代码有 ShortTerm topic buffer、MidTerm MemoryAtom 检索热集合和 LongTerm archive store，并通过 `archive/revive` 进行冷热搬运。它们是不同对象和访问状态，不是 MemoryAtom 从幼年到成熟的三级阶梯；把高生命力记忆直接“晋升”到现有 LongTerm 反而会使它退出普通检索。

仍值得保留的核心问题是：能否把**生物学巩固（consolidation）**显式建模为一次可解释的语义演化，例如把稳定、反复验证的一组记忆合并为更高层摘要，冻结特定版本，或形成新的可检索稳定态。具体落点必须经过数据验证后再定，不能预设为短→中→长搬运。

### 落点

- 先定义 consolidation 的产物：新 MemoryAtom、现有 atom 的新版本、Artifact，或独立的存储状态；
- 再决定 Lifecycle 是只提供“持续高位”信号，还是拥有触发权；内容合并仍应由 Generation/专用 consolidator 完成；
- 若需要 `high_watermark_sustain_days`，必须有可持久化的持续时间证据，不能只读取一次即时分数；
- 调度继续复用 System 全局 scheduler，但不能让 scheduler 拥有合并或存储迁移规则。

### 期望效果

- 让高价值记忆从“分数高”演进为可解释、可追溯的稳定知识；
- 让 consolidation 结果通过 provenance 指回组成它的证据和旧版本；
- 在不破坏现有 archive/revive 冷热语义的前提下，为后续检索提供更少、更稳的候选。

### 风险

- 自动合并可能放大错误记忆，必须保留来源、版本和人工纠正入口；
- “稳定”与“正确”不能由 vitality 单独推出；
- 如果需要新的成熟度状态，应先扩展正式状态机和迁移语义，而不是改写现有 LongTerm archive 的含义。

---

## 方向 3: FSRS / SM-2 复习时机 (Active Review Scheduling)

### 核心想法

从「被动打分」升级到「主动召回调度」：不只算当前 vitality，同时算出**最优复习时间点** `next_review_at`。检索时用 `now - next_review_at` 作为优先级——「该复习了」的记忆排前；「还没到」的不打扰。

成熟的间隔重复算法：
- **SM-2**: SuperMemo 经典算法，公式简单（`interval *= ease_factor`），Anki 早期默认
- **FSRS** (Free Spaced Repetition Scheduler): Anki 当代默认，基于心理学三参数（稳定性、可提取性、可恢复性）建模，论文与开源实现均有

### 落点

- `MetaData` 新增 `next_review_at: datetime`、`stability: float`、`retrievability: float`（FSRS 三参数可选）
- 新增 `engines/lifecycle/review_scheduler.py`：根据 HIT/CITATION 事件更新 `next_review_at`
- `RetrievalModeConfig.time_weight` 目前只有配置字段，fusion 主路径尚未消费；若未来接入 `next_review_at`，需要先定义它与相关性、recency 和 vitality 的组合语义

### 期望效果

- 让检索主动暴露"濒临遗忘但仍值得保住"的记忆
- 主动复习调度比"被动衰减打分"更接近人脑的"主动遗忘与召回"

### 风险/前置

- FSRS 真正发挥价值需要采集**复习历史**（每次 HIT 的时间戳），需扩 `MemoryEventLog` 持久化
- 算法引入要写一遍 schema migration 与单测，工程量比方向 1、2 大

---

## 方向 4: 来源 salience 进公式 (Source-aware Salience)

### 核心想法

现在公式是扁平的 `V_0 · D(t) + A + B`，**不区分用户主动 WRITE 与 Agent 被动观察**。但代码里已有现成的语义切分点：

- **Mode B (WRITE)**: 用户/Agent 明确要求保存 → extractor 强制 `confidence=1.0`，对用户重要度高
- **Mode A (被动观察)**: LLM 判断有价值才入库，重要度中等
- **Agent REFLECTION**: Agent 自我反思生成，重要度中等偏上

引入 `salience` 维度让公式按"来源重要度"分级，更贴近用户主观记忆强度。

### 落点

- `MetaData` 新增 `salience: Literal["passive", "reflection", "write", "pinned"]`
  - `pinned`: 用户手动置顶/钉住，永不衰减（`D(t) := 1`）
- `VitalityCalculator.calculate` 按 salience 调制：
  - `V_0_eff = V_0 * salience_multiplier` (passive=0.9, reflection=1.0, write=1.0, pinned=1.0)
  - `λ_eff = λ * (2 - I) * salience_decay_factor` (passive=1.0, write=0.7 衰减更慢, pinned=0.0)
- `MemoryGenerationEngine._draft_to_memory` 写入路径根据 Mode 设置 salience
- 用户接口加 `pin_memory(memory_id)` 与 `unpin_memory` API

### 期望效果

- 用户主动强调的记忆得到保护，不被均匀的衰减淹没
- 产品端具备"置顶"能力，扩展表达力

### 风险

- 多一个维度会让调参复杂度非线性上升，建议先并行记录 salience 但暂时只让 pinned 起作用，观察数据

---

## 方向 5: 干扰建模 (Retroactive / Proactive Interference)

### 核心想法

认知科学里"前摄/后摄干扰"：学新知识会让旧的相关记忆退化。当前查重仅做 CREATE/UPDATE/TOUCH 决策（见 `engines/generation/engine.py:326-355`），**完全不影响旧记忆 vitality**。

增加一步：当 dedup 决策为 CREATE 但与某条旧记忆高相似度（不构成 update 但有重叠）时，给旧记忆施加小幅干扰：
- `event_vitality_boost -= 2` (或类似小幅度)
- `decay_rate *= 1.05` (如果方向 1 已落地)

### 落点

- `MemoryGenerationEngine._dedup_and_resolve` 把 candidates 传出来做一次"软干扰"阈值判断
- 新增 `DeduplicatorConfig.interference_threshold` (例如 0.6) 与 `interference_penalty`
- 在 `engines/lifecycle/reinforcement.py` 新增 `EventType.INTERFERENCE` 事件类型

### 期望效果

- 自动避免陈旧相似记忆沉淀堆积，让长尾保持干净
- 与「方向 1 自阻尼 λ」配合后形成"新记忆挤压旧记忆 / 旧记忆靠复习抵御挤压"的动态平衡

### 风险

- 误判会无脑打压有效记忆，阈值需要谨慎调参
- 不在主线需求路径上，价值密度低于前 4 个方向

---

## 推荐演进路径

1. **方向 1 先做离线校准**（自阻尼 λ）：用真实事件历史比较固定 λ 与 per-memory λ，不以“公式更像人脑”替代收益证据
2. **方向 4 先验证产品语义**（salience）：尤其是用户明确保存/置顶是否能提供比隐式命中更可靠的信号；不要一次引入全部倍率
3. **方向 3 先补数据再谈算法**（FSRS）：只有存在持久化复习历史和清晰产品动作后，主动复习时机才可评估
4. **方向 2 独立重做语义设计**（巩固化循环）：不得把现有 ShortTerm/MidTerm/LongTerm 直接解释为成熟度迁移
5. **方向 5 最后验证**（干扰）：只有去重误差和冲突样本表明它有净收益时，才考虑让相似新记忆影响旧记忆

---

## 现状记 (修复前公式与艾宾浩斯意图的偏差)

本次修复前的旧公式 `V = (C × I) × D(t) × 100 + A` 存在以下与"模拟人类记忆衰退/强化"意图的系统性偏差，是 100→67 bug 的根因，也提示了上述演进方向：

1. **confidence 既是"初值系数"又是"质量分"，语义混淆** — 低置信度记忆不是"衰减得更慢/更快"，而是从一开始就低，违背艾宾浩斯 V_0 应为高值的直觉。本次修复已通过把 V_0 固定解耦解决。

2. **D(t) 只跟 `updated_at` 走，且所有事件重置 updated_at** — 衰减钟一直被归零，时间衰减实际只在"从未被访问"的记忆上起作用。本次修复已通过「HIT 不重置 updated_at，仅 CITATION 重置」解决。

3. **access_boost 封顶太低 (10 次封顶)** — 强化效果被压平，"反复强化累积"失去梯度。本次修复已通过改对数曲线解决。

4. **clamp 在 100 截断** — 公式原始输出 `[0, 120]`，靠 clamp 到 100。提示 base+boost 的设计没有预留 hit_boost/feedback_boost 的叠加空间。本次修复保留 clamp，但建议未来公式扩展时重新审视上限。

5. **confidence 缺乏上行通道** — `reinforce()` 只在负反馈时改 confidence（×0.5），正向事件都不提升 confidence。"被反复验证有效应提升记忆强度"的语义在当前公式实现不了。**该偏差本次修复未处理**，留待方向 1 / 4 落地时一并解决。

---

## 附: 公式与各方向叠加后的目标形态

```
V(t) = V_0(salience) · D(t; λ_eff) + A(access) + B(events)

V_0(salience)   = base_vitality * salience_multiplier          # 方向 4
λ_eff           = memory.decay_rate                            # 方向 1 (per-memory 自阻尼)
A               = access_boost_coef * log(1 + access_count)    # 本次已落地
B               = memory.event_vitality_boost                 # 本次已落地
next_review_at  = FSRS_next_review(stability, retrievability)  # 方向 3
语义巩固          = candidate consolidator + provenance          # 方向 2，具体状态待定义
干扰             = EventType.INTERFERENCE on dedup_create       # 方向 5
```
