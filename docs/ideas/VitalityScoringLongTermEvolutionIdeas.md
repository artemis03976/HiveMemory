# 生命力分数计算 - 长期演进方向 Ideas

> 状态: 设计设想阶段，未排期
> 创建: 2026-07-26
> 关联: 本次「生命力分数 100→67 bug」修复后整理的未来演进方向

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

| 方向 | ROI | 与现有架构契合度 | 新概念引入 | 推荐顺序 |
|---|---|---|---|---|
| 1. 自阻尼 λ (per-memory) | 高 | 高 | 极少 | 1st |
| 2. 巩固化循环 (短中长流动) | 高 | 极高 (已有物理分层) | 中等 | 2nd |
| 3. FSRS / SM-2 复习时机 | 中高 | 中 | 算法引进 | 3rd |
| 4. 来源 salience 进公式 | 中 | 高 | 少 | 4th |
| 5. 干扰建模 | 中 | 中 | 少 | 5th |

---

## 方向 1: 自阻尼 λ (per-memory decay rate) — *最高 ROI*

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

## 方向 2: 巩固化循环 (Consolidation Loop) — *与架构契合度最高*

### 核心想法

现有代码已有 `ShortTerm / MidTerm / LongTerm` 三层存储 + `archive/resurrect` 的物理分层（见 `patchouli/memory_library/library.py`、`adapters/long_term.py`），但 vitality 当前只在归档时被动利用一次（跌破 `low_watermark` 触发归档）。

把**生物学巩固（consolidation）**显式建模为正向流动：周期性任务把"vitality 持续高位 + 命中次数 ≥ N"的记忆从短→中→长逐层晋升；长期层则触发永久冻结。让 vitality 不仅是"打分"，而是**驱动记忆在三层之间流动的喷淋水位**。

### 落点

- 扩展 `engines/lifecycle/engine.py`，引入对称的 **`Promoter`**（与 `GarbageCollector` 对偶）：
  - `promote(memory_id)`：把 vitality 长期保持高位的记忆晋升一层
  - 配置 `high_watermark_sustain_days`：要求"持续高位 N 天"才晋升，避免短期波动误晋升
- 与现有的 `GarbageCollector`（低水位归档）形成双臂调度器
- 可在 `patchouli/services/lifecycle.py` 中绑定周期调度

### 期望效果

- 这是普通 RAG 系统做不到、而本项目架构已具备土壤的能力
- 让长尾核心记忆"沉淀"到长期层（成本更低、检索更稳）
- 让消失中的记忆先在 mid 层"徘徊"，避免误归档丢失

### 风险

- 三层之间的位移动作需要保证一致性（建议涉及写时同步、读时降级回退）
- 晋升阈值需要观测后调参，避免长期层积压太多低活跃记忆

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
- `engines/retrieval/fusion.py` 的 `RetrievalModeConfig.time_weight` 已有吸收时间信号的字段，可改为读 `now - next_review_at`

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

1. **先做方向 1**（自阻尼 λ）：风险最低，立刻有体感，是其他方向的基础设施
2. **再做方向 2**（巩固固化循环）：把 vitality 变成"驱动记忆在三层间流动"的力，最大化现有架构价值
3. **方向 3**（FSRS）：在 1、2 落地、有复习历史数据后引入，提升到主动召回调度
4. **方向 4**（salience）：在产品语义上分层，更适合产品演进期落地
5. **方向 5**（干扰）：作为长尾清理机制，最后锦上添花

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
层间流动          = Promoter / GarbageCollector 周期调度         # 方向 2
干扰             = EventType.INTERFERENCE on dedup_create       # 方向 5
```