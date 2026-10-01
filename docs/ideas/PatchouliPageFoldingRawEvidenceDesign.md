---
title: Patchouli Page Folding Raw Evidence
status: idea
horizon: candidate
owner: patchouli
scope: page-folding-raw-evidence-exploration
related_current:
  - docs/patchouli/perception.md
  - docs/patchouli/artifacts.md
  - docs/patchouli/generation.md
  - docs/system/passive-ingress.md
related_ideas:
  - docs/ideas/long-running-agent-intra-turn-context-folding.md
  - docs/ideas/external-session-and-topic-projection.md
last_reviewed: 2026-10-01
---

# Patchouli Page Folding Raw Evidence 设计备忘

本文是一项未排期的开放设计，不是 Patchouli 当前能力说明。当前 Page Folding、Artifact 与 Generation 的真实边界分别以[感知与短期话题](../patchouli/perception.md)、[Artifacts 与来源追踪](../patchouli/artifacts.md)和[记忆生成](../patchouli/generation.md)为准。一个未结束 turn 内多次 compact 的工作集管理另见[长时间运行 Agent 的 Turn 内上下文折叠](./long-running-agent-intra-turn-context-folding.md)；两项 Idea 可以共享 evidence refs，但不能互相充当已经落地的能力。

## 0.0 定位调整（owner，2026-10-01）

ConversationSession 落地后，前台对话与后台的记忆 Topic 管理是两项各自独立的工作。前台的上下文压缩归 CPU（用户以 `/compact` 触发，见[会话 `/compact` 指令 Todo](../todo/conversation-compact-command.md)）；后台的 Topic 管理继续使用 page folding。本 Idea 针对的是后台问题：Topic 累计消息过长时，如何保证输入 Patchouli 记忆生成的资料不会导致上下文爆炸。

影响（分析）：

- 下文把 `TOKEN_OVERFLOW` 描述为“面向 Agent 热路径的设计决策”，并设计了 Agent Context Flow（3.1）。这反映的是当前代码：Alice 每轮的历史来自 Topic 的 `state_summary` 与最近 blocks（`prompts/assembler.py`）。ConversationSession 落地后，Agent 的上下文改由会话提供，page folding 不再服务 Agent 的上下文，相关段落需要按新的定位重新审视；
- 第 0 节所述“Agent 的工作上下文与系统保存的原始证据是两种不同资产”仍然成立，只是 Agent 的工作上下文从此归前台、由 CPU 管理。

## 0. 复核结论与成立条件

这个 Idea 所保护的核心理念仍然成立：Agent 的工作上下文与系统保存的原始证据是两种不同资产。旧页退出 active buffer，不等于它们在证据层也应当消失；反过来，保存证据也不意味着每轮都应把全部历史重新注入 Agent。

当前代码已经具备三块可复用基础：`TOKEN_OVERFLOW` 会生成 `state_summary` 并保留配置数量的最近 blocks，Generation 接受结算后的 `TopicMaterializeTask`，Artifact 层也能保存 append-oriented 的 InteractionArtifact。但三者尚未形成本文设想的旁路：overflow 当前不 Settle，被裁剪的旧前缀只进入有损摘要；没有 `FoldResult`、folded evidence artifact 类型或 raw turn store；`TopicMaterializeTask` 也不携带 folded artifact refs。现有 Artifact 基础因此不能被解释为“折叠原文已经保全”。

本方向只有在以下条件同时满足后才应升级为 Plan：

1. 通过可复现案例证明 overflow 原文丢失确实影响记忆质量、审计或调试，而不是只存在理论风险；
2. 明确证据保留期限、用户隔离、敏感内容处理、容量上限和删除语义；
3. 决定写入所有权与失败语义，并证明 artifact 旁路不会阻塞 Agent 热路径；
4. 定义默认 settlement 与 high-fidelity refinement 的质量/延迟基线、去重策略和验收测试；
5. 形成独立 Plan，明确修改 Perception、Artifacts、Generation 和配置时必须同步更新的当前文档。

## 1. 背景

当前 Patchouli 感知层的 `TOKEN_OVERFLOW` 行为是一个面向 Agent 热路径的设计决策：

```text
active topic blocks
-> token overflow
-> compact into state_summary
-> discard folded raw blocks from active buffer
```

这套机制的核心目标不是完整保全短期对话原文，而是为 Agent 腾出可用上下文空间。换言之，Page Folding 的一等语义是 **Agent context compression**，不是 archive。

因此，compact 后旧 blocks 从 active buffer 中移除是合理的：对 Agent 来说，这些 blocks 已经不可见，后续上下文只通过 `state_summary + recent_blocks` 接力。

但这也带来一个明确代价：summary 是有损压缩。被折叠的原始 blocks 如果没有被保存到其他位置，就无法用于后续高保真记忆生成、审计、debug 或历史重放。

本文记录一个未来可实现的优化方向：在不破坏当前热路径简洁性的前提下，为 Page Folding 增加 raw evidence side-channel。

## 2. 当前设计取舍

### 2.1 当前策略

当前策略可以概括为：

```text
TOKEN_OVERFLOW = compact + discard raw pages from active buffer
```

优点：

- Agent 热路径简单，active buffer 大小可控。
- `state_summary` 是唯一接力状态，topic context 组装逻辑稳定。
- 不需要让 generation extractor 直接承接无限增长的历史 blocks。
- 不引入额外的 artifact / cold store 读写路径。

缺点：

- summary 会丢失细节。
- 被折叠 blocks 未必已经被 settlement 生成过长期记忆。
- 后续 idle / LRU / manual settlement 只能看到 summary 和 compact 后的新 blocks。
- debug、审计和高保真重放缺少原始证据。

### 2.2 不建议简单改为始终保留所有 blocks

一个直接方案是：Page Folding 只更新 summary，但 active buffer 始终保留所有 blocks。

这个方案不推荐作为默认策略，原因是：

- active buffer 控制流会重新变复杂。
- Agent topic context 仍然不能无界注入这些 blocks，保留原文并不等于 Agent 可见。
- settlement 时 extractor 可能因历史过长超过上下文窗口。
- LRU / idle / manual settlement 的语义会混入“压缩页是否仍在 active buffer 中”的细节。
- perception 层会被迫同时管理 Agent 可见上下文和原文保全，两种职责再次耦合。

更好的方向是拆流。

## 3. 候选设计

未来优化应拆成三条语义独立的流：

```text
1. Agent Context Flow
   active buffer -> state_summary + recent_blocks

2. Raw Evidence Flow
   folded blocks -> append-only artifact / raw turn store

3. Generation Flow
   state_summary + recent_blocks by default
   optional raw evidence replay for high-fidelity generation
```

### 3.1 Agent Context Flow

这条流保持当前策略：

- `TOKEN_OVERFLOW` 触发 compact。
- compact 后旧 blocks 从 active buffer 移除。
- active topic context 只暴露 `state_summary + recent_blocks`。
- Agent 不感知 folded raw blocks。

这保证热路径仍然轻量，Page Folding 继续承担上下文压缩职责。

### 3.2 Raw Evidence Flow

在 folded blocks 被移出 active buffer 前，将它们写入 append-only 原始证据仓库。

候选落点：

- `ArtifactStore`
- 独立 `RawTurnStore`
- `InteractionArtifact`
- 未来的 cold raw transcript store

关键约束：

- raw evidence 不参与 active topic context。
- raw evidence 不影响 LRU / idle / manual topic 控制流。
- perception 层不负责决定何时消费 raw evidence。
- 写入应是 append-only，便于审计和重放。

### 3.3 Generation Flow

默认 settlement 仍只使用：

```text
state_summary + recent_blocks
```

但 `TopicMaterializeTask` 可以携带 raw evidence 引用：

```text
topic_id
state_summary
recent_blocks
folded_artifact_refs
reason
```

generation coordinator 根据策略决定是否读取 raw evidence。

默认策略应保持低延迟：

```text
SETTLEMENT default:
  build context from state_summary + recent_blocks
  run normal generation
```

高保真策略可以异步执行：

```text
HIGH_FIDELITY_SETTLEMENT:
  load folded evidence refs
  chunk raw blocks
  run extraction per chunk
  dedup / merge generated atoms
```

## 4. 候选演进顺序

### Phase 1: 提取 fold 原语

为 `ShortTermMemoryStore` 增加明确的 fold API：

```python
def fold_blocks(
    self,
    topic_id: str,
    summary: str,
    retain_count: int,
) -> FoldResult:
    ...
```

`FoldResult` 建议包含：

```python
class FoldResult:
    topic_id: str
    folded_blocks: tuple[LogicalBlock, ...]
    retained_blocks: tuple[LogicalBlock, ...]
    state_summary: str
```

职责边界：

- Store 负责裁剪 active blocks 和更新 token 计数。
- Store 不负责写 artifact。
- TriggerManager / PerceptionFamiliar 决定是否把 `folded_blocks` 交给 side-channel。

### Phase 2: 引入 raw evidence artifact

新增 artifact 类型，例如：

```text
ArtifactType.FOLDED_TOPIC_BLOCKS
```

建议字段：

```text
artifact_id
topic_id
user_id
created_at
fold_sequence
source_reason = TOKEN_OVERFLOW
state_summary_before
state_summary_after
blocks
block_ids
turn_event_ids
```

注意：artifact 是原始证据，不是长期记忆原子，不参与正常 retrieval ranking。

### Phase 3: 在 TopicMaterializeTask 中携带引用

扩展 `TopicMaterializeTask`：

```python
folded_artifact_refs: list[ArtifactRef] = []
```

这只是 metadata，不意味着 generation 必须读取 artifact。

### Phase 4: 高保真后台 refinement

增加一个可选后台任务：

```text
memory_task.submit_high_fidelity_settlement
```

输入：

```text
topic_id
state_summary
recent_blocks
folded_artifact_refs
```

处理：

```text
load artifacts
-> chunk raw blocks
-> extract candidates per chunk
-> dedup / merge with existing atoms
-> optionally update artifacts / source links
```

该任务不应阻塞 Agent finalize / shutdown drain 的热路径。

## 5. 关键边界

### 5.1 Perception 层

Perception 仍然只负责短期 MMU：

- ingest turn payload
- maintain active topic buffer
- fold pages
- settle topic into `TopicMaterializeTask`

Perception 不应：

- 直接运行 generation。
- 决定 raw evidence 如何生成 memory。
- 为了 artifact replay 改变 active context 行为。

### 5.2 Artifact / Raw Store

Artifact / Raw Store 负责不可变证据保存：

- folded blocks
- turn events
- action traces
- interaction snapshots

它不负责：

- Agent context 组装。
- generation 策略。
- memory lifecycle。

### 5.3 Generation Coordinator

Generation Coordinator 决定如何消费输入：

- 默认只消费 `state_summary + recent_blocks`。
- 可选读取 `folded_artifact_refs`。
- 可将高保真处理拆到后台任务。

这能避免把 generation 的策略复杂度泄漏回 perception。

## 6. 配置建议

未来可增加配置：

```yaml
patchouli:
  perception:
    page_folding:
      retain_recent_blocks: 3
      persist_folded_evidence: true
      folded_evidence_store: artifact

  generation:
    settlement:
      use_folded_evidence: false
      high_fidelity_refinement_enabled: false
      folded_evidence_chunk_blocks: 8
```

默认值应偏向当前行为：

```text
persist_folded_evidence = false 或 true 但不消费
use_folded_evidence = false
high_fidelity_refinement_enabled = false
```

即使先打开 evidence 保存，也不应立即改变 generation 结果路径。

## 7. 风险与注意事项

### 7.1 不要让 raw evidence 回流成 active context

raw evidence 的存在只是为了保真和后处理，不应让 Agent 每轮重新看到全部 folded history。

否则 Page Folding 失去意义。

### 7.2 不要让 extractor 一次消费全部历史

如果 raw evidence 很长，应分片提取：

```text
artifact chunks -> candidate memories -> dedup / merge
```

不要把所有 folded blocks 拼回一个超长 generation context。

### 7.3 不要在 perception 中加入“哪些 blocks 已生成记忆”的复杂状态

主动写入、被动归档、高保真 refinement 可能都消费同一段原始证据。

去重应由 generation deduplicator / memory merge 机制兜底，而不是让 perception 追踪消费状态。

### 7.4 Artifact 写入失败策略

raw evidence 是增强能力，不应阻塞 Agent 热路径。

建议：

- artifact 写入失败记录 warning / runtime event。
- active buffer folding 仍继续。
- 后续可通过 observability 发现证据缺失。

## 8. 理想候选形态

理想状态下，Page Folding 的语义是：

```text
For Agent:
  old blocks are compressed into state_summary and removed from active context.

For System:
  old blocks may be preserved as immutable raw evidence outside active buffer.

For Generation:
  default settlement stays lightweight.
  high-fidelity extraction can replay raw evidence asynchronously.
```

这保留了当前设计中最重要的热路径性质，同时为未来更高质量的记忆生成、审计、回放和调试提供基础设施。

## 9. 后台折叠的已知缺口与待设计范围

2026-10-01 并入：原占位计划“话题折叠、Actor 上下文与原始证据统一改造”（`docs/plans/topic-folding-context-and-raw-evidence.md`）退回 Idea，原 Todo“Page Folding 跨入口上下文与证据后续技术债”（`docs/todo/page-folding-cross-ingress-follow-ups.md`）拆分。两份文档中属于后台 Topic 管理的内容并入本节，属于前台上下文的内容并入[长时间运行 Agent 的 Turn 内上下文折叠](./long-running-agent-intra-turn-context-folding.md)第 14 节，原文件随后删除，删除前最后版本见 commit `74b5056`。

### 9.1 已知缺口（代码核对，2026-07-30 记录，2026-10-01 复核配置）

- **软水位线**：overflow 后只按 block 数保留后缀。单个超大 turn 或最近 blocks 本身可能继续超过 `fold_token_threshold`，因此它只是软水位线。按 0.0 节的定位，后果是 Topic 交给记忆生成的资料可能超出预算。
- **不支持 summary-only Topic**：公开配置拒绝 `fold_retain_recent_blocks=0`（`config/patchouli.py`，`ge=1`）。底层 Store 把零值定义为“清空全部 blocks”，但 `TopicData.is_empty`、shutdown settlement、Generation identity 与 InteractionArtifact 尚不支持只剩摘要的 Topic。
- **缺少入口来源**：主动与被动入口共享 `InteractionPayload` 和短期 Topic，payload 与 block 没有稳定的 ingress origin 与 connector provenance，因此不能安全地按入口类型切换整个 Topic 的折叠策略。若直接根据 `PASSIVE_MEMORY` 跳过折叠，Patchouli 内部 buffer 与 settlement generation 可能无界增长。
- **compact-only**：`TOKEN_OVERFLOW` 只压缩，被折叠的旧前缀不会自动形成 settlement 或 raw evidence artifact（见第 0、1 节），长时间运行的被动对话尤其可能只剩有损摘要，高保真的原始 turn 可能在正式记忆或 artifact 生成前被裁剪。

“何时清空 blocks”本质上是数据耐久性边界，而不只是一个摘要算法参数；调整这些语义时，必须同时检查 Generation、Artifacts、Passive Ingress 与 shutdown drain。

### 9.2 待设计范围（原占位计划的后台部分）

| 工作域 | 需要覆盖的问题 |
|:---|:---|
| 折叠算法 | RelayController 的定位、触发策略与摘要算法的分离、token 与 block 数的联合预算、摘要自身的增长、超大 block 与保留后缀 |
| 原始证据 | 保存时机、存储归属、引用与覆盖范围、折叠前后的材料交接、容量、保留期限、删除与写入失败 |
| Generation 与生命周期 | 普通 settlement、高保真分块处理、去重与来源关联、幂等、背压、关闭及持久化恢复的承诺边界 |

### 9.3 升级为 Plan 前需要回答的问题

在第 0 节的成立条件之外，还需要回答（候选，未决定）：

1. count limit 与 token budget 的联合保留算法如何处理单个超大 block，需要哪些可观测事件；
2. 是否支持 retain zero：若支持，需要完整定义 summary-only Topic 的非空判断、身份归属、shutdown/idle settlement、Generation 与 artifact 行为；若不支持，继续在 schema 与当前文档中明确拒绝；
3. 在需要按入口区分折叠策略之前，如何定义 interaction origin/provenance 以及混合来源 Topic 的冲突规则，而不是根据 connector 名称或单个事件临时推断；
4. folded prefix 采用 checkpoint settlement、append-only raw evidence 还是其他耐久化路径，以及写入失败、容量、隐私、删除与 shutdown 语义。

每项跨系统能力在实施前都应形成独立 Plan，并同步更新 Perception、Generation、Artifacts、Passive Ingress 与相关契约文档。
