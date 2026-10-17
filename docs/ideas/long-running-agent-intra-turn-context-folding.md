---
title: Long-Running Agent Intra-Turn Context Folding
status: idea
horizon: candidate
owner: patchouli-alice-system
scope: long-running-agent-intra-turn-context-management
related_current:
  - docs/patchouli/perception.md
  - docs/patchouli/generation.md
  - docs/patchouli/artifacts.md
  - docs/alice/agent-runtime.md
  - docs/system/passive-ingress.md
related_ideas:
  - docs/ideas/short-term-memory-and-context-compaction.md
  - docs/ideas/PatchouliPageFoldingRawEvidenceDesign.md
  - docs/ideas/pending-intent-migration.md
related_todos:
  - docs/todo/conversation-compact-command.md
last_reviewed: 2026-10-17
---

# 长时间运行 Agent 的 Turn 内上下文折叠

本文记录一项面向未来生产级长时间运行 Agent 的开放设计，不是当前能力说明，也不代表近期排期。当前短期话题、Agent runtime、Passive Ingress、Generation 和 Artifact 的真实行为分别以 front matter 中列出的当前文档为准。

通用的执行工作记忆、Topic 记忆形成工作集、Alice compact 与 page folding 关系集中在[短期记忆与上下文压缩](./short-term-memory-and-context-compaction.md)；Session → Topic 完整资料 → Artifact 的原始数据保存链集中在[全链路原始证据 Idea](./PatchouliPageFoldingRawEvidenceDesign.md)。本文只讨论未结束 turn 的事件流接入、语义分段、独立覆盖进度、超大事件和封口恢复，不重复定义通用压缩算法或证据存储。

> 定位调整（owner，2026-10-01）：ConversationSession 落地后，前台对话与后台的记忆 Topic 管理是两项各自独立的工作。前台的上下文压缩归 CPU，用户以 `/compact` 触发（[会话 `/compact` 指令 Todo](../todo/conversation-compact-command.md)）；后台的 Topic 管理继续使用 page folding。本文据此分别讨论 CPU 的 turn 内执行视图与 Patchouli 的未结束交互资料视图，两边各自推进，不能串成前一级 compact 完成后再交给后一级 folding 的流水线。

> 方向（owner，2026-10-13）：对话以流式方式同步到 Patchouli 侧，仍保留一轮 Interaction 的概念，但 Topic 侧接收更原子的输入结构，以解决一轮对话或执行过长导致的上下文爆炸。Topic 的生成材料相应变为之前全部的 Interaction 记录，加上当前轮已流式接收的消息。写入意图派发时“只取过去若干轮历史与 `state_summary`、当前一轮不参与”的局限随之消失（现状见 [Patchouli 生成](../patchouli/generation.md)第 3.2 节，材料方向见[写入意图后续方向 Idea](./pending-intent-migration.md)第 8 节）。影响（分析）：第 1 节“当前主路径以完整 turn 为基本摄入单位”是这一方向要改变的现状；第 3.1 节接收未结束 turn 的 OpenTurnWorkspace 是一种候选承载；第 10.1 节“partial turn 提前进入 settlement”的顾虑仍然成立，流式接收的消息作为生成材料不等于提前结算。接收流式输入需要 Patchouli 侧先有容器保存关联信息，并供生成取出已接收的内容；容器是话题本身，还是第 3.1 节那样单独接收未结束 Interaction 的容器，决定话题是否需要预建，在此之前话题保留预建（[任务进程 Idea](./task-process-table-and-registration-entry.md#21-交互被接纳后即关闭进程q-1) 2.1，owner，2026-10-13）。

## 0. 核心判断与升级门槛

长时间运行 Agent 会让一次 user 请求对应的 turn 持续数十分钟甚至更久，并在同一个 turn 内经历多次模型 context overflow 与 compact。此时，“一个 turn”仍是可靠的业务和因果边界，却不再是合适的容量管理边界。

本 Idea 的核心判断是：

> Turn 是因果单位，Segment 是候选容量单位；执行检查点由 CPU 管理，Topic 摄入与折叠进度由 Patchouli 管理；原始资料另有独立的保存链。

不应简单扩大现有 LogicalBlock，或把每次运行中 compact 都伪装成一个完整 turn。两边共享交互事实来源，而不是消费对方已经压缩后的状态：

```text
accepted interaction events
  ├─ CPU execution view -> execution checkpoint + recent execution context
  └─ Patchouli received materials -> open-turn material view + Topic page folding

raw records / source references -> Session / Topic complete materials -> Artifact
```

CPU 可能已处理但 Patchouli 尚未收到某条事件。CPU compact 的回调和覆盖范围不能充当 Patchouli 的摄入确认；Patchouli 已接收资料也不意味着相应 Memory 已生成。上图不要求先对话封口再同步材料，流式接收正是本专项要探索的方向。

本方向只有在以下条件同时满足后才应升级为 Plan：

1. 收集真实长时间运行样本，证明单 turn 膨胀已经影响 Agent 连续运行、Patchouli 短期工作集或 Generation，而不是只存在理论风险；
2. 明确 Alice、Patchouli、System connector 与外部 harness 对 compact 决策、checkpoint 生成和上下文返回的所有权；
3. 定义可迁移的数据模型、幂等与恢复协议，不破坏现有完整 turn、settlement 和 artifact 语义；
4. 给出 token、延迟、记忆质量、原始证据容量和失败恢复基线；
5. 形成独立 Plan，并列出需要同步更新的当前设计、契约、配置、可观测性和兼容策略。

## 1. 背景与当前约束

当前主路径以完整 turn 为基本摄入单位：任务进程在 CPU 执行结束后封口 `InteractionPayload`，Perception 再把其中的 user query、assistant final text、turn events、actions 和 traces 封装为一个 `TurnRecord` 与一个 `LogicalBlock`。Topic Page Folding 发生在 blocks 已经进入 Topic 工作集之后，详见[感知与短期话题](../patchouli/perception.md)。

这套模型适用于“一个 turn 在一次模型上下文内自然结束”的常规对话，但面对长时间运行 Agent 会出现四类张力：

- 外部 harness 可能已在一个 turn 内 compact 多次，HiveMemory 却只在最终完成时看到一个巨大 turn；
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

### 2.4 执行检查点与资料覆盖进度

ContextCheckpoint 只表示 CPU 一次 compact 覆盖的稳定前缀及其继续执行视图。候选字段如下；`turn_id` 是概念名称，落地应关联既有 `interaction_id`，不另造一套来源身份：

```text
ContextCheckpoint
├── turn_id
├── checkpoint_id
├── previous_checkpoint_id
├── covers_through_sequence
├── continuation_summary
├── retained_segment_ids[]
├── compact_reason
├── model_context_limit
├── producer
└── evidence_refs[]
```

`covers_through_sequence` 只解释执行摘要覆盖到哪个来源事件；它与 retained segments 不重复覆盖同一前缀。Patchouli 则单独记录实际已接收事件的进度，以及其记忆工作视图已折叠的范围。三者可以关联同一来源序号，但不是同一个游标，也不由同一检查点对象拥有。

接入协议还需区分已接收、已纳入资料视图、已生成摘要、已完成记忆生成等事实。它们不因一次 CPU compact 成功而一同成立；checkpoint 也不携带一个由 CPU 负责推进的 `memory_working_summary`。

### 2.5 LogicalBlock：sealed turn 的默认记忆生成单位

LogicalBlock 继续优先表示已经结束的完整 turn。未来它与原子流式材料如何共同组织，仍需设计；不应因为执行上下文切页或记忆材料切段就自动复制成多个完整 turns。封口后的资料必须能关联完整来源，不能只保留 CPU 的 bounded view。

### 2.6 Artifact：高保真证据单位

原始资料从完整交互记录进入 Topic 的资料集合，并在记忆生成阶段由 Artifact 固定关联来源；具体载体、引用交接与保留规则见[全链路原始证据 Idea](./PatchouliPageFoldingRawEvidenceDesign.md)。保存完整资料不等于每轮都把它们重新注入 Agent 或 Generation。本文不把先实现会话持久化作为流式材料探索的前置条件；进程重启恢复的承诺须另行满足。

## 3. 独立的执行视图与流式资料视图

### 3.1 OpenTurnWorkspace

OpenTurnWorkspace 是 Patchouli 接收未结束交互资料的一种候选容器，可独立存在，也可整合进 Topic 的资料组织。它不是 Alice 执行工作区，不能持有 CPU 的运行状态或依赖 CPU compact 才推进：

```text
OpenTurnWorkspace
├── turn_id
├── accumulated_summary
├── recent_segments[]
├── received_through_sequence
├── folded_through_sequence
├── pending_event_ranges[]
├── inline_token_count
├── lifecycle_state
└── raw_evidence_refs[]
```

以上字段只是讨论不同进度所需的语义，不是已确定模型。职责边界建议为：

- CPU 自行管理执行上下文与 compact 检查点；
- Patchouli 的候选工作区管理实际收到的资料、来源、分段与记忆工作视图；
- 当前轮已接收的资料可以进入生成材料选择，不必等待整个 turn 封口，也不因此提前触发 settlement；
- turn 封口后与最终 Interaction 对齐，合并为稳定资料，不因重放封口 payload 重复摄入；
- 工作视图可以有界，完整资料集合的保留及向 Artifact 的交接由原始证据设计负责。

### 3.2 候选数据流

```text
actual interaction events from Alice / external connector
  -> validate source + interaction_id + sequence / event identity
  -> accept into Patchouli material container and retain source records / refs
  -> advance Patchouli received progress
  -> assemble material segments and maintain bounded memory view
  -> expose received current-turn materials to Generation selection

final outcome
  -> reconcile final Interaction with received events
  -> seal material container without duplicate facts
  -> continue Topic lifecycle / settlement under their own rules

CPU compact callback -> update CPU execution checkpoint independently
```

事件接纳与资料保留的具体原子边界仍需定义。来源记录或稳定引用确认之前，不能因派生视图更新而丢弃唯一原文；这里的“确认”不预设数据库持久化。若需要可靠流式交接，必须使用可返回确定结果的提交协议，不能用 best-effort RuntimeEvent 代替。

## 4. Turn 内的容量管理专项约束

完整请求预算、摘要边界、保留后缀、校验与失败后保留有效检查点等通用机制见[短期记忆与上下文压缩](./short-term-memory-and-context-compaction.md)。长 turn 还需要解决以下专项问题：

- 只有已经闭合的稳定事件前缀适合折叠；未返回的工具调用和未结束的子 frame 不能因 token 水位任意拆散；
- CPU 与 Patchouli 分别计算各自消费者的完整请求预算，不能共用一个“当前上下文 token 数”；
- 对尚未闭合、单独就超过预算的操作，需要有界表示、分块读取或显式失败策略；
- 摘要与保留片段必须绑定相同来源范围，原子更新该视图自己的 coverage，不能顺带推进另一边；
- 压缩失败是否允许继续 turn 按该视图的预算与失败策略决定，不能丢掉已收到的流式资料；显式用户停止仍沿原生取消传播，不被当作普通压缩失败吞掉。

Segment count 只能作为次级容量限制。任何严格的 context 上限都需要把单个大事件、摘要以及请求中其他部分一起计入。

## 5. 单个超大事件与有界表示

超长 turn 最常见的膨胀来源是 tool result、文件、网页、数据库结果和子 Agent transcript。若单个 segment 已超过预算，仅靠总结多个 blocks 无法解决问题。

候选路径：

```text
large event payload retained in the raw-material chain
  -> source reference + size / MIME / provenance
  -> inline semantic digest + bounded excerpt + source ref
  -> count only inline representation in hot working set
```

Generation 或审计可以按需沿来源引用分块读取；记忆生成时再按原始证据设计固定 Artifact 关联。是否要对大对象采用独立 blob 载体属于保存设计的选择，不能因本节直接增加第二套必需存储服务。主 Agent context 和记忆工作视图不因保留完整资料而重新内联全部正文。

需要特别定义：

- 完整资料或来源引用接纳失败时如何返回错误或施加背压；若选择截断，如何显式表达缺口；
- 二进制、敏感数据和凭证的过滤；
- retention、容量、租户隔离和用户删除语义；
- 来源引用的完整性和生命周期验证；
- chunking、去重与重放顺序。

资料保存、引用交接与删除规则由[全链路原始证据 Idea](./PatchouliPageFoldingRawEvidenceDesign.md)维护；本专项只验证大事件经过流式协议后仍可形成有界工作表示，并能定位完整来源。

## 6. Agent summary 与 Memory summary

运行时 continuation summary 和 Patchouli memory summary 面向不同目标：

| 视图 | 优先保留 | 主要消费者 |
|:---|:---|:---|
| continuation summary | 当前计划、未完成步骤、工具状态、下一步动作 | Agent runtime |
| memory working summary | 用户偏好、确认事实、关键决策、来源和可复用经验 | Patchouli / Generation |

两份视图由各自消费者决定何时生成、何时接受和何时失效，不要求同步生成，也不共用一个拥有双摘要字段的检查点。具体策略见[短期记忆与上下文压缩](./short-term-memory-and-context-compaction.md)。

Alice 的 continuation summary 不进入 Topic、InteractionPayload 或 Patchouli 的记忆生成材料。外部 harness 另行交付的 summary 若在未来被评估为有价值的辅助输入，需要显式声明来源、覆盖范围和有损性质；它不能替代未收到的事件或原始证据，也不能作为 Patchouli 摘要失败时自动接受的事实来源。

每份 summary 至少应记录 producer、model、prompt/policy version、coverage 和来源 refs，使未来可以判断它是 Agent 自述、HiveMemory 派生视图还是高保真 refinement。

## 7. Checkpoint 契约与不变量

流式交接与各自检查点至少需要以下不变量：

- `turn_id + checkpoint_id` 幂等；
- event/segment sequence 在单 turn 内单调，重复提交不会重复计入；
- Patchouli 的已接收进度只由实际接收的交互推进；有缺口时不得把“最大已见序号”当成完整接收前缀；
- 每份视图的折叠 coverage 只能覆盖其自己已接纳、且保留规则已得到满足的来源；CPU 覆盖进度不证明 Patchouli 已收到同一范围；
- summary coverage 与 retained segment IDs 不重叠；
- tool/action 原子边界不被 folding 拆开；
- sealed、cancelled 或 failed turn 对迟到事件有明确拒绝或补偿语义；
- checkpoint 是派生视图，不能静默删除唯一原始证据；
- summary 生成失败不能让已接收事件从 open-turn 状态消失；
- shutdown 能区分完整 sealed turn、可恢复 open turn 和不可恢复残片；
- 多 worker 或进程重启后的所有权、lease 和恢复顺序明确。

若 OpenTurnWorkspace 与原始资料仍仅在进程内保留，可以验证流式接纳和容量行为；它不能承诺跨进程重启恢复。是否增加会话或 open-turn 持久化可以另行规划，不能把未承诺的耐久性写成本专项的既有能力。

## 8. Active 与 Passive 的能力差异

### 8.1 Active Alice

Alice 未来实现 turn 内 compact 后，可以准确报告执行视图的检查点；流式材料接入仍走独立路径：

```text
Alice runtime compact callback
  -> accept CPU ContextCheckpoint
  -> continue same Agent turn

actual Alice interaction events
  -> Patchouli acceptance
  -> advance Patchouli received-material progress
```

不要求 Patchouli 订阅 compact 回调。若需要观测 compact，可记录旁路运行事实；它不能驱动材料接纳、确认或裁剪。谁决定何时 compact、谁维护记忆工作视图、谁触发 Generation 必须继续分层。

### 8.2 Passive、具备 compact 能力的外部 Agent

connector 可以另行转发外部 checkpoint 的执行元数据，包括覆盖范围、retained IDs 和 context metadata；外部 summary 的辅助用途须按第 6 节显式约定，不能默认进入 Topic。摄入真实交互与报告外部执行检查点必须是可区分的契约。

需要验证不同 Agent framework 是否能提供稳定的 message/event IDs 和 coverage；即使能提供准确的 CPU coverage，也不能据此确认 Patchouli 收到了原文。如果只有一段无范围的摘要，它最多是有来源的辅助线索，不能驱动安全裁剪。

### 8.3 Passive、没有 compact 能力的轻量 bot

Patchouli 可以根据实际收到的事件流组织 segment 与记忆工作视图，保护内部 memory generation。若还要帮助 bot 管理自己的 prompt，则需要单独的 opt-in managed-context 契约，由显式受委托的上下文管理方负责执行视图，例如：

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

turn 结束时，Patchouli 的候选工作区通过显式 seal，与最终 Interaction 核对来源范围、结局和已接收事实。封口不能只用 CPU 摘要替换此前收到的原始资料。候选记忆工作表示包括：

```text
sealed turn metadata
user query
final assistant outcome
bounded turn summary
important recent segments
memory-view coverage
raw evidence refs
worth-saving / identity / provenance
```

在封口前，生成材料可以选择当前轮已流式接收的资料，与过去的 Interaction 记录共同使用；这不等于提前执行 Topic settlement。封口后的有界工作视图与完整资料继续分离，Generation 按需求读取来源，并由 Artifact 固定相应关联。材料选择、生成和实际持久化是各自独立的进度。

Generation 需要保留 parent interaction 的分组与来源范围，避免把一个 turn 的多个执行 epochs 或重放事件误当成独立经历；具体去重规则仍需与 WRITE/UPDATE 和 settlement 的现有职责协调。

当前有输出的 completed/cancelled/failed 均会封口交互并进入后续提交，已接纳的 WRITE/UPDATE 不因 CPU 或进程失败撤销，见[记忆生成](../patchouli/generation.md)第 7 节；流式接入应保留这一边界。新增的专项问题是尚未完整封口的事件流如何结束和恢复：

- 进程崩溃等导致 final outcome 缺失时如何标记，与正常取消封口如何区分；
- 不完整事件流中的已完成工具事实是否、何时可以进入记忆；
- 用户主动取消与系统崩溃是否采用不同 provenance；
- 恢复后继续原 turn 还是 seal 旧 turn 并创建 successor。

## 10. 不推荐的简化方案

### 10.1 直接把每个 segment 当作 LogicalBlock

会让 partial turn 提前进入 settlement，导致重复记忆、缺少最终 assistant outcome、worth-saving 漂移和 InteractionArtifact 碎片化。

### 10.2 只在 turn 完成后切大 block

可以保护 Patchouli 的后处理，却不能提供当前轮已接收材料的持续使用，也无法帮助 Agent runtime 在 turn 内继续运行。前者需要流式接入，后者需要执行方自己的 compact；两项能力分别验证。

### 10.3 只保存 Agent compact summary

summary 面向任务继续执行且有损，不能替代 raw evidence、来源审计或 memory-oriented refinement。

### 10.4 根据 active/passive 或 connector 名称切换 folding

入口模式不能表达外部 context ownership，同一 topic 还可能混合多种来源。应使用显式 capability、provenance 和 coverage 契约。

### 10.5 无限拼接 previous summary

summary 自身最终也会溢出，必须拥有独立预算、版本和再压缩策略，通用讨论见[短期记忆与上下文压缩](./short-term-memory-and-context-compaction.md)。同样不能因两边都生成摘要，就合并它们的状态或接受条件。

## 11. 候选验证工作

以下工作尚未排期，分组表达验证范围，不代表统一实施顺序。CPU turn 内 compact、Patchouli 流式资料接入和原始证据保存可以按各自边界形成独立切片；具体版本仍需另行决定。

### 11.1 大事件与容量观测

- 统计真实 turn/event/segment 大小和 compact 次数；
- 为大 tool result 验证 bounded inline view + 完整来源引用；
- 补齐 inline/raw token、artifact size、截断和失败指标；
- 分别测量执行请求和记忆生成请求的容量，避免一个预算替代全部消费者。

### 11.2 TurnSegment 与流式接纳协议

- 基于既有 `interaction_id` 定义来源 sequence、事件标识和 segment boundary；
- 保持完整 turn 的因果与封口语义，允许此前逐步接纳原子事件；
- 验证 tool/action 原子分段和幂等重放；
- 明确 ingress provenance、乱序缺口、接纳确认与背压。

### 11.3 Patchouli 未结束交互资料容器

- 选择直接由 Topic 接收，还是使用 OpenTurnWorkspace 等独立容器；
- 独立记录已接收进度与记忆视图折叠进度；
- 支持生成选取当前轮已接收的材料，验证其与封口、settlement 的关系；
- 验证最终 Interaction 对账不会重复摄入，并与完整资料保存链交接。

### 11.4 Alice turn 内执行检查点

- 在根 frame 跨历史轮次 compact 之外，单独评估当前 turn 迭代增长的处理；v0.7.0 的 Alice 压缩只对单轮增长设预算守卫，超出即以结构化错误结束（[Agent Runtime](../alice/agent-runtime.md#31-会话上下文视图)），轮内压缩仍属本专项；
- 验证一个 turn 多次 compact、失败、取消、恢复与 child frame；
- 明确执行检查点只影响 CPU 自己的视图；
- 建立 continuation summary 与记忆视图摘要的质量对照，不强制同步生成。

### 11.5 Passive capability 与 managed context

- 验证 connector 的真实事件流与外部执行 checkpoint 各自能提供什么；
- 为轻量 bot 评估 opt-in managed-context API；
- 定义 cursor、覆盖、替换、确认和版本冲突协议。

### 11.6 另行评估的耐久恢复

- 若产品要求跨重启恢复，再评估 durable open-turn journal 与各方自己的检查点；
- 引用完整原始资料链验证分块 replay；
- 多进程 lease、shutdown drain 和灾难恢复；
- 评估耐久能力增加的质量、成本和延迟，不把原始会话持久化混入近期 compact 的默认范围。

## 12. 生产级指标与验收方向

### 容量与性能

- open-turn inline tokens、segment count 和 summary tokens 始终有界；
- compact 延迟、Agent pause 时间和 checkpoint 写入延迟可观测；
- 单个超大事件不会直接进入模型 hot context；
- summary 再压缩不会形成无界递归或持续抖动。

### 正确性与恢复

- 各方自己的 coverage 无重叠、无缺口、单调推进，CPU compact 不推进 Patchouli 摄入进度；
- 重复、乱序和迟到事件不会重复生成 segment 或记忆；
- compact 与流式接纳失败不破坏最后有效状态；若承诺持久化恢复，再覆盖崩溃与恢复边界；
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

1. Patchouli 接收当前轮材料的容器直接放在 Topic 中，还是独立设置 OpenTurnWorkspace；这对 Topic 预建和最终封口有什么影响？
2. Segment boundary 应由 Agent runtime 声明、由 HiveMemory 归并，还是采用混合策略？
3. 流式原子事件接纳、完整 Interaction 封口与写入意图读取材料之间，如何冻结一致的已接收快照？
4. 完整原始资料链中的引用期限和删除规则，如何覆盖未结束 turn、延迟事件和后来生成的 Artifact？
5. 尚未完整封口的事件流如何表达 partial facts，并与已有 cancelled/failed 封口记录区分？
6. 外部事件标识或 sequence 不稳定时，如何检出重放与缺口；外部 checkpoint 可以提供哪些辅助诊断而不替代接收事实？
7. 多 Agent/child frame 的材料如何关联 parent interaction；CPU 各 frame 的执行检查点如何独立定位这些来源？
8. managed-context API 是否属于 Passive Ingress 扩展，还是应成为独立的 conversation-context service？
9. 具体消费者的 token budget 在模型切换时如何重新验证；超大且尚未闭合的 segment 采用什么有界表示？
10. 哪些真实使用指标足以证明该复杂度优于“外部 Agent 自行 compact，HiveMemory 只做 post-hoc 记忆生成”？

这些问题在获得生产样本与明确所有权之前应保持开放，不应仅凭模型类名或配置草案提前固化。

## 14. 跨入口上下文的已知缺口与专项范围

2026-10-01 并入：原占位计划“话题折叠、Actor 上下文与原始证据统一改造”（`docs/plans/topic-folding-context-and-raw-evidence.md`）退回 Idea，原 Todo“Page Folding 跨入口上下文与证据后续技术债”（`docs/todo/page-folding-cross-ingress-follow-ups.md`）拆分。两份文档中属于前台上下文的内容并入本节，属于后台 Topic 管理的内容并入原始证据 Idea；原文件随后删除，删除前最后版本见 commit `74b5056`。

2026-10-16 整理：一般性的 CPU compact、Topic page folding 边界与优化由[短期记忆与上下文压缩](./short-term-memory-and-context-compaction.md)集中维护，Session → Topic 资料 → Artifact 的保存规划由[全链路原始证据 Idea](./PatchouliPageFoldingRawEvidenceDesign.md)承接。本节保留跨入口 managed-context、未结束 turn 和原子事件流的专项问题，供已有引用继续定位；不再把所有前台 compact 归入长 turn 项目。

### 14.1 已知缺口（代码核对，2026-07-30 记录）

- **没有托管上下文能力**：Passive Ingress 默认由外部 harness 管理 prompt history，公共响应只返回 retrieval memory，不返回 `state_summary` 与 recent blocks。系统没有 `external | hivememory` 这类显式的上下文所有权能力，也没有带版本或覆盖游标的压缩后上下文输出。因此 Discord 等轻量 bot 不能把 HiveMemory 内部的折叠当作外部 prompt 压缩服务使用（对应第 8.3 节）。
- 按开头的定位调整，前台上下文的压缩归 CPU。若要支持由 HiveMemory 托管的 bot 上下文，需要形成独立 Plan，定义上下文所有权、版本/覆盖游标、返回模型，以及调用方替换历史的幂等契约（候选，未决定；与第 13 节问题 8 相关）。

### 14.2 待设计范围（原占位计划的前台部分）

| 工作域 | 需要覆盖的问题 |
|:---|:---|
| 长 turn 与运行中折叠 | segment/checkpoint 候选模型、事件顺序、覆盖关系、工具调用的原子边界、seal/cancel/failure 及恢复范围 |
| Alice 与外部 harness | 自主管理上下文、外部摘要/checkpoint 的来源、混合来源 Topic，以及显式委托上下文管理的适用范围 |
| 执行与记忆的边界 | 独立的接收、摘要覆盖与生成进度（第 2.4、6、7 节）；CPU compact 不改写 Topic 状态，不构成流式摄入确认；通用边界见短期记忆 Idea |

`InteractionPayload` 表示一次完成或明确封口的交互；本文的长 turn 分片与 checkpoint 是运行中的预算与覆盖机制，两者生命周期不同。证据与折叠覆盖关系应引用 `interaction_id`（controller 模式下取 `process_id` 的值）与 `TurnEvent` 范围建立，不另建一套来源 ID。
