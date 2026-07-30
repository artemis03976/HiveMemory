---
title: Long-Running Agent Intra-Turn Context Folding
status: idea
owner: patchouli-alice-system
scope: long-running-agent-intra-turn-context-management
related_current:
  - docs/patchouli/perception.md
  - docs/patchouli/generation.md
  - docs/patchouli/artifacts.md
  - docs/alice/agent-runtime.md
  - docs/system/passive-ingress.md
related_ideas:
  - docs/ideas/PatchouliPageFoldingRawEvidenceDesign.md
related_todos:
  - docs/todo/page-folding-cross-ingress-follow-ups.md
last_reviewed: 2026-07-30
---

# 长时间运行 Agent 的 Turn 内上下文折叠

本文记录一项面向未来生产级长时间运行 Agent 的开放设计，不是当前能力说明，也不代表近期排期。当前短期话题、Agent runtime、Passive Ingress、Generation 和 Artifact 的真实行为分别以 front matter 中列出的当前文档为准。

## 0. 核心判断与升级门槛

长时间运行 Agent 会让一次 user 请求对应的 turn 持续数十分钟甚至更久，并在同一个 turn 内经历多次模型 context overflow 与 compact。此时，“一个 turn”仍是可靠的业务和因果边界，却不再是合适的容量管理边界。

本 Idea 的核心判断是：

> Turn 是因果单位，Segment 是容量单位，Checkpoint 是接力单位，LogicalBlock 是默认记忆生成单位，Artifact 是高保真证据单位。

因此，不应简单扩大现有 LogicalBlock，或把每次运行中 compact 都伪装成一个完整 turn。更合理的方向是在现有 topic Page Folding 之前增加 Turn 内工作集管理，形成两级折叠：

```text
Level 1: Intra-Turn Context Folding
Agent event stream
  -> open-turn semantic segments
  -> context checkpoints
  -> bounded open-turn working set

Level 2: Inter-Turn Topic Page Folding
sealed turn
  -> bounded LogicalBlock
  -> state_summary + recent blocks
```

本方向只有在以下条件同时满足后才应升级为 Plan：

1. 收集真实长时间运行样本，证明单 turn 膨胀已经影响 Agent 连续运行、Patchouli 短期工作集或 Generation，而不是只存在理论风险；
2. 明确 Alice、Patchouli、System connector 与外部 harness 对 compact 决策、checkpoint 生成和上下文返回的所有权；
3. 定义可迁移的数据模型、幂等与恢复协议，不破坏现有完整 turn、settlement 和 artifact 语义；
4. 给出 token、延迟、记忆质量、原始证据容量和失败恢复基线；
5. 形成独立 Plan，并列出需要同步更新的当前设计、契约、配置、可观测性和兼容策略。

## 1. 背景与当前约束

当前主路径以完整 turn 为基本摄入单位：Agent run 结束后构造 `InteractionPayload`，Perception 再把其中的 user query、assistant final text、turn events、actions 和 traces 封装为一个 `TurnRecord` 与一个 `LogicalBlock`。Topic Page Folding 发生在 blocks 已经进入 SemanticBuffer 之后。

这套模型适用于“一个 turn 在一次模型上下文内自然结束”的常规对话，但面对长时间运行 Agent 会出现四类张力：

- Agent runtime 已经在一个 turn 内 compact 多次，HiveMemory 却只在最终完成时看到一个巨大 turn；
- shell、网页、文件、数据库结果和子 Agent transcript 可能让单个事件本身超过短期工作集预算；
- 若把每个中间片段直接当作 LogicalBlock，Generation 与 settlement 会误把 partial turn 当作完整记忆生成单位；
- 外部 harness 的 continuation summary 面向任务继续执行，不一定适合用户记忆、事实提取和来源审计。

还存在一个不可绕过的能力边界：如果 Passive connector 只在 turn 完成后提交最终结果，HiveMemory 不可能反向管理该 Agent 运行期间已经发生的 context overflow。它最多只能在收到数据后保护自身的短期存储与 Generation 输入。

## 2. 概念分层

### 2.1 Turn：因果与产品边界

Turn 仍表示一次 user 请求到最终 assistant outcome 的完整因果过程。它负责承载：

- 原始 user query；
- 最终 assistant 结果与终态；
- 完整操作链的归属；
- worth-saving 与记忆生成边界；
- settlement、InteractionArtifact 和用户可见历史的默认分组。

Turn 不应因为运行中发生一次 context compact 就提前结束。

### 2.2 Context Epoch：模型容量周期

Context Epoch 表示两次运行时 compact 之间的上下文阶段。一次 turn 可以包含零个、一个或多个 epoch。Epoch 是描述 Agent 实际看到什么、为何需要 compact 的运行视图，不是新的用户交互。

### 2.3 TurnSegment：不可随意拆分的容量单元

TurnSegment 是 Turn 内可被工作集管理的最小语义片段。候选字段如下，名称仅用于探索：

```text
TurnSegment
├── segment_id
├── turn_id
├── ordinal
├── events[]
├── semantic_kind
├── inline_token_count
├── created_at
└── raw_evidence_refs[]
```

Segment 应按语义边界形成，而不是机械地每 N tokens 切分。以下内容原则上不能被拆开：

- tool call 与对应 tool result；
- 一个 MTP action 与它的多个 results；
- Alice child frame 的创建、关键产出和终态；
- 一组必须共同解释的 thought/action/observation；
- 外部 Agent 声明为原子的结构化操作。

### 2.4 ContextCheckpoint：运行接力单位

ContextCheckpoint 表示一次 compact 已经覆盖的稳定前缀及其继续执行视图。候选字段如下：

```text
ContextCheckpoint
├── turn_id
├── checkpoint_id
├── previous_checkpoint_id
├── covers_through_sequence
├── continuation_summary
├── memory_working_summary
├── retained_segment_ids[]
├── compact_reason
├── model_context_limit
├── producer
└── evidence_refs[]
```

`covers_through_sequence` 是关键语义：它明确 summary 覆盖到哪个事件，哪些 segments 仍作为未覆盖后缀保留，从而避免 summary 与 recent segments 重复承载同一事实。

### 2.5 LogicalBlock：sealed turn 的默认记忆生成单位

LogicalBlock 继续优先表示已经结束的完整 turn。未来它可以携带 bounded turn view、checkpoint refs 和 raw evidence refs，但不应因为工作集切页而自动复制成多个完整 turns。

### 2.6 Artifact：高保真证据单位

Artifact 或未来 raw evidence store 保存不适合长期驻留在 prompt working set 中的大对象和原始片段。保存证据不等于每轮都把证据重新注入 Agent；它与 active context 必须继续保持职责分离。

## 3. 候选两级架构

### 3.1 OpenTurnWorkspace

在现有 SemanticBuffer 之前增加一个管理未结束 turn 的候选工作区：

```text
OpenTurnWorkspace
├── turn_id
├── accumulated_summary
├── recent_segments[]
├── checkpoint_chain[]
├── next_sequence
├── inline_token_count
├── lifecycle_state
└── raw_evidence_refs[]
```

职责边界建议为：

- OpenTurnWorkspace 管理一个仍在运行的 turn、segment 和 checkpoint；
- SemanticBuffer 管理多个已经 sealed 的 turn blocks；
- turn finalize 后才从 bounded open-turn view 形成 TurnRecord/LogicalBlock；
- 两层可以复用“总结旧前缀、保留最近后缀”的纯算法，但不能复用同一生命周期状态机。

### 3.2 候选数据流

```text
Agent/Alice/connector event
  -> append raw event or evidence ref
  -> validate turn_id + sequence
  -> assemble semantic segment
  -> append OpenTurnWorkspace
  -> check inline token budget
      -> below limit: continue
      -> overflow: checkpoint stable prefix
                   retain bounded recent segments
                   continue same turn
  -> final outcome
  -> seal OpenTurnWorkspace
  -> materialize bounded LogicalBlock
  -> append SemanticBuffer
  -> existing topic Page Folding / settlement
```

这个顺序使原始事实先于派生摘要落定，避免 compact 成功但对应原文从未进入任何可靠载体。

## 4. Turn 内 Folding 算法

Turn 内 folding 不应只使用 segment 数量，而应优先满足多个 token budget：

```text
working set budget
  = total model budget
  - system / tool schema budget
  - retrieval memory budget
  - rolling summary budget
  - next-step generation headroom
```

候选算法：

1. 从最新 segment 向前选择要保留的后缀；
2. 同时满足 segment count、inline token 和 generation headroom；
3. 不拆分原子语义 segment；
4. 把未保留的稳定前缀交给 checkpoint summarizer；
5. 新 summary 只覆盖旧前缀，不覆盖 retained segment IDs；
6. summary 超过自身预算时，对 checkpoint summary 进行有界再压缩，而不是无限拼接；
7. 原子更新 summary、coverage、retained segments 和 token count；
8. compact 失败时保留最后一个有效 checkpoint，并进入可观测的降级或背压状态。

这意味着 count limit 只能作为次级上限，不能再被描述为严格 context 安全保证。

## 5. 单个超大事件与大对象外置

超长 turn 最常见的膨胀来源是 tool result、文件、网页、数据库结果和子 Agent transcript。若单个 segment 已超过预算，仅靠总结多个 blocks 无法解决问题。

候选路径：

```text
large event payload
  -> append-only artifact / blob store
  -> checksum + size + MIME + provenance
  -> inline semantic digest + bounded excerpt + artifact ref
  -> count only inline representation in hot working set
```

高保真 Generation 或审计按需沿 artifact ref 分块读取。主 Agent context、短期话题和普通 settlement 不重新内联全部正文。

需要特别定义：

- 外置写入失败时是阻塞、降级还是截断；
- 二进制、敏感数据和凭证的过滤；
- retention、容量、租户隔离和用户删除语义；
- artifact ref 的完整性验证；
- chunking、去重与重放顺序。

这些问题与 [Page Folding Raw Evidence](./PatchouliPageFoldingRawEvidenceDesign.md) 高度相关，进入 Plan 时应决定共享基础设施还是保持独立数据面。

## 6. Agent summary 与 Memory summary

运行时 continuation summary 和 Patchouli memory summary 面向不同目标：

| 视图 | 优先保留 | 主要消费者 |
|:---|:---|:---|
| continuation summary | 当前计划、未完成步骤、工具状态、下一步动作 | Agent runtime |
| memory working summary | 用户偏好、确认事实、关键决策、来源和可复用经验 | Patchouli / Generation |

因此不建议只保存一个无来源的 `summary` 字段。早期实现可以把外部或 Alice 产生的 continuation summary 作为 memory summarizer 的输入或故障降级，但不能默认把它当作正式记忆事实。

每份 summary 至少应记录 producer、model、prompt/policy version、coverage 和来源 refs，使未来可以判断它是 Agent 自述、HiveMemory 派生视图还是高保真 refinement。

## 7. Checkpoint 契约与不变量

生产级 checkpoint 至少需要以下不变量：

- `turn_id + checkpoint_id` 幂等；
- event/segment sequence 在单 turn 内单调，重复提交不会重复计入；
- checkpoint coverage 只能前进，不能覆盖尚未接收或尚未持久化的事件；
- summary coverage 与 retained segment IDs 不重叠；
- tool/action 原子边界不被 folding 拆开；
- sealed、cancelled 或 failed turn 对迟到事件有明确拒绝或补偿语义；
- checkpoint 是派生视图，不能静默删除唯一原始证据；
- summary 生成失败不能让已接收事件从 open-turn 状态消失；
- shutdown 能区分完整 sealed turn、可恢复 open turn 和不可恢复残片；
- 多 worker 或进程重启后的所有权、lease 和恢复顺序明确。

如果 OpenTurnWorkspace 仍是纯进程内状态，它只能作为早期验证，不应被宣称为生产级长时间运行保障。

## 8. Active 与 Passive 的能力差异

### 8.1 Active Alice

Alice 能知道模型实际何时 compact，因此是首选验证路径：

```text
Alice runtime compact callback
  -> publish ContextCheckpoint
  -> advance OpenTurnWorkspace coverage
  -> continue same Agent turn
```

Patchouli 消费 compact 事实，但不反向取得 Alice 的执行控制权。谁决定何时 compact、谁保存短期记忆、谁触发 Generation 必须继续分层。

### 8.2 Passive、具备 compact 能力的外部 Agent

connector 可以选择转发外部 checkpoint，包括外部 summary、覆盖范围、retained IDs 和 context metadata。外部 summary 必须保留 provenance，不能冒充 HiveMemory 自己生成的摘要。

需要验证不同 Agent framework 是否能提供稳定的 message/event IDs 和 coverage；如果只能提供一段无范围的摘要，它最多是 hint，不能驱动安全裁剪。

### 8.3 Passive、没有 compact 能力的轻量 bot

HiveMemory 可以根据持续收到的事件流自行 segment/checkpoint，保护内部 memory generation。若还要帮助 bot 管理自己的 prompt，则需要单独的 opt-in managed-context 契约，例如：

```text
ManagedContextSnapshot
├── checkpoint_id
├── summary
├── recent_messages[]
├── covered_through_event_id
└── replace_before_event_id
```

调用方只有在确认应用成功后才能推进自身 cursor；响应必须支持幂等重试和旧版本检测。不能根据 `source=discord` 等名称自动推断上下文所有权。

### 8.4 Turn 完成后才提交的外部系统

HiveMemory 只能 post-hoc 分块、外置和 compact，保护自身的存储与 Generation。它不能解决外部 Agent 运行中已经发生的 context overflow。若产品需要这项能力，connector 必须升级为事件流或 checkpoint 协议。

## 9. Turn Seal 与下游 Generation

turn 结束时，OpenTurnWorkspace 应通过显式 seal 形成稳定的 bounded turn view。候选 LogicalBlock 内容包括：

```text
sealed turn metadata
user query
final assistant outcome
bounded turn summary
important recent segments
checkpoint refs
raw evidence refs
worth-saving / identity / provenance
```

普通 settlement 使用 bounded summary + recent segments；高保真 refinement 才按策略读取 checkpoint/raw refs。Generation 必须按 parent turn 分组，避免一个 turn 的多个 epochs 重复生成同一记忆。

取消和失败 turn 需要单独裁定：

- 是否形成可结算的 partial turn；
- final outcome 缺失时如何标记；
- 已完成工具事实是否仍可进入记忆；
- 用户主动取消与系统崩溃是否采用不同 provenance；
- 恢复后继续原 turn 还是 seal 旧 turn 并创建 successor。

## 10. 不推荐的简化方案

### 10.1 直接把每个 segment 当作 LogicalBlock

会让 partial turn 提前进入 settlement，导致重复记忆、缺少最终 assistant outcome、worth-saving 漂移和 InteractionArtifact 碎片化。

### 10.2 只在 turn 完成后切大 block

可以保护 Patchouli 的后处理，却无法帮助 Agent runtime 在 turn 内继续运行，也无法准确还原每次外部 compact 的 coverage。

### 10.3 只保存 Agent compact summary

summary 面向任务继续执行且有损，不能替代 raw evidence、来源审计或 memory-oriented refinement。

### 10.4 根据 active/passive 或 connector 名称切换 folding

入口模式不能表达外部 context ownership，同一 topic 还可能混合多种来源。应使用显式 capability、provenance 和 coverage 契约。

### 10.5 无限拼接 previous summary

summary 自身最终也会溢出，必须拥有独立预算、版本和再压缩策略。

## 11. 候选演进顺序

### Phase 1：大对象外置与观测

- 统计真实 turn/event/segment 大小和 compact 次数；
- 为大 tool result 建立 bounded inline view + artifact ref；
- 补齐 inline/raw token、artifact size、截断和失败指标；
- 不改变现有 LogicalBlock 与 settlement。

### Phase 2：TurnSegment 与序号协议

- 定义 `turn_id + sequence + segment boundary`；
- 保持完整 turn 仍是公开提交语义；
- 验证 tool/action 原子分段和幂等重放；
- 明确 ingress provenance。

### Phase 3：OpenTurnWorkspace

- 支持未结束 turn 的有界 recent segments；
- 建立 checkpoint coverage 与 rolling summary；
- 先采用进程内原型验证算法和质量，不宣称生产耐久性；
- 保持现有 SemanticBuffer 与 LogicalBlock 下游兼容。

### Phase 4：Alice Runtime Checkpoint

- 接入真实 compact callback；
- 验证一个 turn 多次 compact、失败、取消、恢复与 child frame；
- 建立 continuation summary 与 memory summary 的对照基线。

### Phase 5：Passive Capability 与 Managed Context

- 允许有能力的 connector 提交外部 checkpoint；
- 为轻量 bot 评估 opt-in managed-context API；
- 定义 cursor、覆盖、替换、确认和版本冲突协议。

### Phase 6：持久化恢复与高保真 Generation

- durable open-turn journal/checkpoint chain；
- folded evidence 与分块 replay；
- 多进程 lease、shutdown drain 和灾难恢复；
- 高保真 extraction、dedup、成本与延迟验收。

阶段编号只是候选依赖顺序，不是版本承诺。任何阶段进入实施前都应形成独立 Plan。

## 12. 生产级指标与验收方向

### 容量与性能

- open-turn inline tokens、segment count 和 summary tokens 始终有界；
- compact 延迟、Agent pause 时间和 checkpoint 写入延迟可观测；
- 单个超大事件不会直接进入模型 hot context；
- summary 再压缩不会形成无界递归或持续抖动。

### 正确性与恢复

- coverage 无重叠、无缺口、单调推进；
- 重复、乱序和迟到事件不会重复生成 segment 或记忆；
- compact/持久化/进程崩溃的每个边界都有可恢复测试；
- shutdown 能报告 sealed、recoverable open 和 orphaned 状态。

### 记忆质量

- bounded view 相比完整 raw transcript 的事实召回、决策保留和误提取基线；
- continuation summary 与 memory summary 的用途差异有实验支持；
- 多 epoch turn 不会产生明显重复 MemoryAtom；
- high-fidelity replay 的质量收益足以覆盖额外成本。

### 安全与治理

- raw evidence 与 summary 都符合租户隔离、敏感数据和删除语义；
- 外部 checkpoint 不被无条件信任；
- artifact/ref 完整性可验证；
- 日志和 RuntimeEvent 不泄露大 payload 正文。

## 13. 开放问题

1. OpenTurnWorkspace 应归属于 Alice runtime、Patchouli，还是由 System 提供独立 journal，才能既不反转控制权又支持耐久恢复？
2. Segment boundary 应由 Agent runtime 声明、由 HiveMemory 归并，还是采用混合策略？
3. continuation summary 与 memory summary 是否需要同时同步生成，还是允许 memory summary 异步滞后？
4. checkpoint raw evidence 的保留期限、容量和删除语义是否与 InteractionArtifact 一致？
5. cancelled/failed turn 的 partial facts 何时值得进入正式记忆？
6. 一个外部 checkpoint 缺少稳定 coverage 时，HiveMemory 应拒绝、仅作 hint，还是自行重建 coverage？
7. 多 Agent/child frame 应共享 parent turn workspace，还是形成带父子关系的子 workspace？
8. managed-context API 是否属于 Passive Ingress 扩展，还是应成为独立的 conversation-context service？
9. token budget 应按具体模型动态解析，还是保持与模型无关的保守统一预算？
10. 哪些真实使用指标足以证明该复杂度优于“外部 Agent 自行 compact，HiveMemory 只做 post-hoc 记忆生成”？

这些问题在获得生产样本与明确所有权之前应保持开放，不应仅凭模型类名或配置草案提前固化。
