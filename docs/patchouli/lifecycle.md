---
title: Patchouli Memory Lifecycle
status: current
owner: patchouli
scope: vitality-reinforcement-gardening-and-archive
code_paths:
  - src/hivememory/engines/lifecycle/
  - src/hivememory/patchouli/services/lifecycle.py
  - src/hivememory/patchouli/memory_library/library.py
related_contracts:
  - docs/system/runtime-and-bus.md
  - docs/contracts/routes-and-events.md
last_reviewed: 2026-07-28
---

# 记忆生命周期

长期记忆不能只增不减。若每次检索命中都永久提高优先级、每次对话都新增副本而从不衰减，书库最终会被“曾经有用”但已经过时的内容占满。Lifecycle 为记忆建立一组可解释的时间、使用与反馈信号，并把低生命力记忆移出中期检索热集合，同时保留显式复活路径。

生命周期不是自动判真器。高生命力只表示一条记忆近期、常用或得到正向信号，不代表内容一定正确；负反馈降低 confidence 与 vitality，也不会替代人工删除或版本修订。

## 1. 分层责任

```text
LifecycleFamiliar
  -> MemoryLifecycleEngine
       -> VitalityCalculator
       -> DynamicReinforcementEngine
       -> PeriodicGarbageCollector
  -> MemoryLibrary.archive / revive
```

- Familiar 暴露业务入口与 scheduler callback；
- LifecycleEngine 协调评分、事件和 GC；
- Calculator 是纯分数计算；
- ReinforcementEngine 读取、修改并持久化单条记忆；
- GarbageCollector 筛选候选，跨层搬运只调用 MemoryLibrary。

全局调度器只决定 gardening 何时运行，不拥有生命力公式或 archive 规则。

## 2. 生命力模型

当前公式为：

```text
V(t) = clamp(V0 * D(t) + A(access) + B(events), 0, 100)

D(t) = exp(-lambda_eff * days_since_update)
lambda_eff = decay_lambda * (2 - intrinsic_value)
A(access) = access_boost_coef * log(1 + access_count)
B(events) = event_vitality_boost
```

- `V0` 默认 100，使新记忆从高生命力开始；
- intrinsic value 按 memory type 调节衰减速度，code/fact 默认比 work-in-progress 衰减慢；
- access boost 使用对数曲线，访问越多仍增长，但边际增益下降；
- event boost 单独累积，避免每次重算时丢失强化历史；
- confidence 当前不参与起始分数或衰减公式，只在负反馈等其他环节使用。

这种模型保留了“遗忘是默认趋势，使用与复习可以抵抗遗忘”的设计理念。它不是统计学习得出的最优公式；权重仍是可配置的工程启发式，需要以实际命中质量评估。

## 3. 强化事件

当前四种事件为：

| 事件 | 默认 event boost | access count | updated_at | confidence |
|:---|---:|:---:|:---:|:---:|
| `HIT` | +5 | +1 | 不重置 | 不变 |
| `CITATION` | +20 | +1 | 重置 | 不变 |
| `FEEDBACK_POSITIVE` | +50 | +1 | 不重置 | 不变 |
| `FEEDBACK_NEGATIVE` | -50 | +1 | 不重置 | ×0.5 |

HIT 表示一次被动检索命中，不应不断把更新时间刷新到“现在”，否则经常被召回的旧事实永不衰减。CITATION 表示 Agent 或用户显式使用，当前把它视作主动复习并重置时间衰减。所有事件随后统一重算 vitality 并 upsert MemoryAtom。

事件 history 默认只在 ReinforcementEngine 内存中保留，最大 10000 条，用于当前进程的调试与统计；它不是持久化审计日志。MemoryAtom 中的 event boost、access count、confidence 和 updated time 才随 atom 持久化。

## 4. Gardening 与垃圾回收

`PatchouliSystem` 向全局 scheduler 注册 `memory_gardening`。一次 gardening：

1. scroll 最多 10000 条中期记忆；
2. 重新计算每条 vitality 并持久化；
3. 选择 `vitality_score <= low_watermark` 的候选；
4. 按分数升序截取 `batch_size`；
5. 对尚未归档的候选调用 `MemoryLibrary.archive()`；
6. 返回 archived count、duration 与 error 摘要供维护观测。

默认 low watermark 为 20，batch size 为 10。`force` 参数当前会沿调用链传递，但 collector 没有额外调度限制可绕过，因此不会改变实际筛选逻辑。

## 5. Archive 与 Revive

Archive 顺序为长期 persist 后中期 delete，并在 atom 中追加 ARCHIVED event；Revive 顺序为长期 load、中期 upsert、长期 remove，并追加 REVIVED event。状态转移由 MemoryLibrary 统一执行，Archiver 与 GC 不直接同时持有两个后端。

归档后的记忆退出普通 Qdrant retrieval。Revive 是显式 local capability，不是 retrieval miss 的自动行为。这样可以避免一次宽泛搜索意外把大量冷记忆重新搬回热集合，也使恢复成本保持可控。

## 6. 当前调用点

- Patchouli finalize 对 prepare 阶段实际注入的 memories 记录 HIT；
- MTP/公开接口可记录 CITATION；
- Memory management API 可记录正/负反馈；
- Retrieval 返回时可临时刷新结果 atoms 的 vitality，但默认不持久化；
- Scheduler 定期执行 gardening；
- 内部 local route 可执行 revive。

Lifecycle 不从 Gateway 的 `worth_saving` 直接调整分数。入口价值预判只影响短期材料是否进入结算，正式 MemoryAtom 的长期演化使用自己的证据。

## 7. 维护与可观测性

Gardening 作为全局 maintenance task 运行，调度器提供 interval、非重入、启停和 RuntimeEvent；LifecycleFamiliar 捕获业务异常并返回 `success/error/duration_ms/archived_count`。GC 还维护当前进程累计 scanned/archived/skipped/runs stats。

业务状态与观测状态保持分开：事件发布或统计失败不能改变已经完成的 archive，日志也不能作为恢复 MemoryAtom 的来源。

## 8. 当前限制

- 公式与权重是启发式配置，尚无基于真实任务的系统校准；
- `confidence` 尚未进入衰减调制，配置注释仍保留未来方向；
- `high_watermark` 当前没有进入 Engine/GC 主路径；
- reinforcement event history 与 GC stats 只在进程内；
- gardening 每次最多 scroll 10000 条，没有分页游标或分片 job；
- archive/revive 不是跨存储事务，失败可能形成重复副本或中间态；
- 冷存储没有自动检索与复活策略；
- `force` 当前不改变 collector 行为；
- 生命力高不等于事实正确，生命周期不能替代 provenance、版本和用户修订。

未来的 split/merge、版本回档或自动复活都应建立在当前所有权上：Lifecycle 提供信号，MemoryLibrary 执行状态转移，Artifact 保存历史；任何一个组件都不应独自取得三者的全部职责。
