---
title: 外部会话消息的接收与 Topic 投影
status: idea
horizon: current
serves_version: v0.7.0
owner: project
scope: conversation-session-interaction-event-topic-projection
related_docs:
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/workspace-network-task-process-architecture.md
  - docs/ideas/pending-intent-migration.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/architecture/decisions/0006-memory-library-custody-criteria-and-independence-contract.md
  - docs/ideas/PatchouliPageFoldingRawEvidenceDesign.md
  - docs/ideas/long-running-agent-intra-turn-context-folding.md
  - docs/ideas/PatchouliPageFoldingRawEvidenceDesign.md
  - docs/ideas/long-running-agent-intra-turn-context-folding.md
last_reviewed: 2026-10-01
---

# 外部会话消息的接收与 Topic 投影

## 0. 文档性质

本文由原 v0.7.0 A3 计划（Conversation Session 与 Topic 投影边界）于 2026-09-27 退回 Idea：删除了阶段划分、交付切片、验收门禁、跨计划依赖与文档更新清单，设计内容保留。计划的最后版本见 commit `dda9d9d` 中的 `docs/plans/v0.7.0-a3-conversation-session-and-topic-projection.md`。

- 要解决的问题：Topic 体系不能接收外部 Actor 的会话消息（owner 表述，2026-09-27）。
- **版本归属**（owner，2026-09-27）：本方向在 v0.7.0 内完成，Alice 作为第一个使用者（[总 Idea](./workspace-network-task-process-architecture.md#61-已决定事项) 6.1）。外部 Actor 分为两种接入模式（[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#11-两种接入模式owner2026-09-27) 1.1）：controller 模式在 v0.7.1，plugin 模式在其后的 v0.7.x。
- plugin 模式的会话归外部 harness 所有；Alice 与 controller 模式的会话由 HiveMemory 发起。本文最初面向外部 harness 自有的会话。
- **Topic 与会话解耦**（owner，2026-09-28）：Topic 将与 conversation session 解耦，不再承担上下文，但仍是记忆生成的历史材料来源；conversation session 不是记忆的材料来源；任务进程不再预先创建临时话题，Gateway 的 Topic 路由决定跨阶段传递到结算，提交后依此按需创建 Topic（[任务进程 Idea](./task-process-table-and-registration-entry.md#12-任务进程的结构owner2026-09-28) 1.2、Q-9）。本文第 3 节 Topic 的 working set 职责、第 4.3 节的 prepare 生命周期，以及第 8 节 Topic 生命周期事项中的预创建与条件清理，需要按此决定重新审视。
- **对话上下文由 Session 提供**（owner，2026-09-28）：实际使用的对话上下文由 ConversationSession 提供，原样积累，不再由外界干涉；Topic 作为内部记忆生成的资料，Gateway 话题路由与 Topic 只为记忆生成服务。这是本文一开始就定下的前提（见下段与第 3 节）。第 3 节表中 ConversationSession“不负责模型工作集”一格需要按此修订。
- **会话模型与 Topic 池**（owner，2026-09-28）：见 0.1。
- 原计划中的“决定”“冻结”在本文中均为候选设计，不是已采纳的方案；原计划留待 A3-0 冻结的事项汇总为第 8 节的开放问题。例外：第 2.2 节的 `InteractionPayload` 是 owner 提出的共用提交模型，不是待选方案（owner，2026-09-28）；字段细节仍按第 8 节待定。
- 本文的 `ConversationSession` 是[任务进程 Idea](./task-process-table-and-registration-entry.md#q-9-对话连续性的承载) Q-9 选项 B（保留独立的会话记录）的一种形态。2026-09-28 Q-9 已选 B：实际使用的对话上下文由 ConversationSession 提供；模型字段等细节仍是候选设计；主动进程的交互记录去向见同文 Q-14，Import Bus（现有 Passive Ingress 链路）交互的 Topic 落位见[总 Idea](./workspace-network-task-process-architecture.md) Q-11，该链路不在 v0.7.0 范围。
- 文中“A1”指已归档的 A1 访问边界（当前事实见 [Workspace 架构](../architecture/workspace.md)第 4 节）；原文提到的 A2、A5、A6 计划已作废删除，相关表述改为中性描述。

外部 harness 通常拥有确定性的 session/conversation，而 Patchouli 需要根据内容把交互分配到 Topic，以便组织记忆生成资料。两者都是有意义的分桶，但回答不同问题：Session 负责对外会话连续性、顺序和展示；Topic 负责记忆侧的相关性划分和短期工作集。本文讨论这个数据模型和交接时机，不涉及完整折叠算法；后台 Topic 折叠与原始证据见 [Page Folding Raw Evidence Idea](./PatchouliPageFoldingRawEvidenceDesign.md)，前台上下文与长 turn 见[长时间运行 Agent 的 Turn 内上下文折叠 Idea](./long-running-agent-intra-turn-context-folding.md)。

### 0.1 会话模型与 Topic 池（owner，2026-09-28）

**背景**（owner 表述）：Topic 机制最初同时承担短期记忆与 Agent 上下文控制，希望实现自动的上下文场景切换：前端对应自动的 Topic 卡片切换与话题池展示，以及半自动的上下文管理，包括为适应长上下文而压缩对话。这与主流的 session 视角冲突：

1. session 由用户主动创建，Topic 完全由后台自动控制；
2. 一般的 harness 都自带运行时的上下文压缩，与 page folding 冲突；
3. Topic 机制也不符合厂商推广 KV cache 的方向：输入一直在变，命中不了缓存，反而可能没有 token 经济效益。代码核对（2026-09-28，`prompts/assembler.py`）：Alice 每轮的 system prompt 依次包含 MTP 说明、persona、本轮检索到的记忆、附件与 Topic 的 `state_summary`，历史是 Topic 最近 5 个 block 的滑动窗口，每轮能稳定命中缓存的只有 MTP 说明与 persona。

要使用外部 Actor，无论如何都必须解决 session 与 Topic 体系的不一致。Topic 机制背后的理念（对话语义的一致性与连贯性）仍然是目标，在记忆生成一侧保持；前台的对话体验回归常规 Agent 软件的 session 概念。

**决定**：

| 事项 | 决定 |
|:---|:---|
| 会话操作 | 保留三个操作：新建（清空上下文）、恢复（resume，回到原会话继续追加）、压缩（compact，在会话内由 CPU 当场生成摘要替换较早的上下文）。不设“新会话接续旧会话”的交接摘要：把旧会话的完整历史带入新会话，等同于恢复旧会话；而 `topic_title`、`topic_summary` 只供前端展示，`state_summary` 不一定已经生成，都不能作为交接摘要的来源 |
| 压缩的归属 | 压缩由 CPU 负责，作用于 CPU 自己基于 Session 派生的上下文视图；Session 记录保持原样。外部 harness 自带压缩与恢复；Alice 需要实现自己的压缩。Topic 的 page folding 只整理记忆材料，两者作用于不同对象。补充（owner，2026-10-01）：ConversationSession 落地后，前台对话与后台的记忆 Topic 管理是两项各自独立的工作；compact 归属前台，用户的 `/compact` 指令对外部 Actor 触发其 harness 自带的压缩，对 Alice 由 Alice 自己实现（[会话 `/compact` 指令 Todo](../todo/conversation-compact-command.md)）；后台的 Topic 管理继续使用 page folding，面向 Topic 累计消息过长时保证输入记忆生成的资料不会导致上下文爆炸（[Page Folding Raw Evidence Idea](./PatchouliPageFoldingRawEvidenceDesign.md)） |
| Topic 与 Session | Topic 不绑定 Session，workspace 共享一个 Topic 池；Gateway 的路由候选为整个 workspace 的 Topic。同一主题在不同会话中的讨论路由到同一个 Topic，记忆材料的语义保持连贯。Session 与 Topic 之间只有路由关联：会话包含若干进程，每个进程的交互记录路由到某个 Topic |
| 跨会话连续性 | 用户有意分开的会话，靠中期与长期记忆接续；想回到原来的话题时恢复原会话 |
| Gateway 的会话提示 | 由 Gateway 识别“本轮消息属于另一个会话的话题”并提示用户恢复该会话，放到后续版本实现；现在 Gateway 的话题路由只服务于后台的 Topic 路由 |
| 前端 | 左侧改为 session 列表（新建、恢复、重命名、加载历史）；取消“当前 Topic”概念；`topic_info` 改为异步的记忆标注（进程结束后得知本轮归入了哪个 Topic）；Topic 池移到记忆面板，作为短期库的整理视图，保留 settle 与 delete 管理 |

**影响**（分析）：

- 本文第 3 节“新 Session 的自动路由候选默认限制在已关联 Topic”与第 4 节“用户新建 Session 只创建会话容器；显式新 Topic 才约束内层路由”，按本决定废止或需要修订；第 7 节“Alice 可以继续选 Topic 工作集、短期窗口和检索结果组织 prompt”不再成立；
- 一个 Topic 中会交错出现来自多个会话的 block；提交按 Topic 串行处理，不影响记忆生成，但 Topic 不能再作为某个会话的对话视图展示；
- controller 模式下，恢复会话对外部 harness 意味着恢复它的原会话，取决于 harness 是否支持（例如 ACP 的 `session/load` 是可选能力）；不支持时只能新建；
- Alice 一侧的压缩与候选 Idea [长时间运行 Agent 的 Turn 内上下文折叠](./long-running-agent-intra-turn-context-folding.md)相关；占位计划“话题折叠、Actor 上下文与原始证据统一改造”把 Topic 折叠与 Actor 上下文合在一起，其前提已不成立（2026-10-01 退回 Idea 后删除，前台部分并入[Turn 内上下文折叠 Idea](./long-running-agent-intra-turn-context-folding.md)，后台部分并入 [Page Folding Raw Evidence Idea](./PatchouliPageFoldingRawEvidenceDesign.md)）；
- 版本安排（owner，2026-09-28）：前端改造在 v0.7.0 完成，新建与恢复两个会话操作随前端改造一起完成；Alice 的压缩算法滞后，大约在 v0.7.1 完成；Gateway 的会话提示放到后续版本。v0.7.0 期间 Alice 没有会话内压缩，长会话的上下文长度由用户新建会话来控制（分析）。

## 1. 目标与非目标

目标是新增 `ConversationSession` 承接会话资源，在现有 `InteractionPayload`、`TurnEvent` 和提交结果上补齐共同语义，让 System、Alice、Passive Ingress 和外部 adapter 提交同一种封口交互；Patchouli 内部继续产生 `LogicalBlock`、`TopicData` 和交互到 Topic 的路由关联。Session 历史可以跨多个 Topic，不把全部历史自动送入模型。非目标包括：把 Session 变成 Topic 的另一份 canonical store、要求外部 Actor 构造 Alice 的 `TurnRecord`、在此决定摘要算法、实现全历史导入或替换外部 harness 的上下文压缩。

Topic prepare/handle、assignment、条件清理和资料引用交接的生命周期集中在第 4 节定义。

代码依据：`core/models/identity.py` 保留兼容 `session_id`；`core/protocol/models.py` 的 `InteractionPayload` 混合事件、物化任务和价值信号；`core/models/interaction.py` 已有可选 `action_id` 的 TurnEvent、ActionReducer 和 TurnRecord；`core/models/topic.py` 的 LogicalBlock 内嵌 TurnRecord。`patchouli/application/interaction_submission_service.py` 已有 `InteractionSubmitResult`，当前仅表达接纳结果，使用 `topic:{requested_topic_id}` 排序；`patchouli/control/interaction_apply_journal.py` 的 `InteractionApplyRecord` 已按 interaction_id 保存 topic_id、应用阶段和摘要，但只提供有界内部记录。`system/services/passive/turn_buffer.py` 已把部分外部事件转换为 TurnEvent；其启动条件、容量截断和封口规则仍不足以承接一般外部会话。以上路径已于 2026-09-27 复核存在；不能据此声称新的 Session 模型、无损输入处理或公共应用结果查询已实现。

## 2. 数据模型

候选设计的复用方向如下；具体字段、类型、必需性、长度/容量、缺省和兼容样例仍待定（第 8 节）。下列内容不是已发布 schema。

| 原拟模型 | 候选设计 | 保留的职责 |
|:---|:---|:---|
| ConversationSession | 新增会话资源 | 会话身份、历史顺序、生命周期与保留范围 |
| ConversationSegment | 不新建；演进现有 InteractionPayload | 一次封口交互的共同内容输入 |
| ConversationPart | 不新建；扩展现有 TurnEvent | 有序消息、工具事件及其来源 |
| SessionRef | 不作为必需包装；可信访问范围内使用 session_id | 跨范围寻址确有需要时可定义小型复合值类型 |
| InteractionContext | 不新增通用容器 | 会话/来源关联归交互输入，授权归 A1 access |
| TopicAssignment | 不要求独立实体；扩展既有应用记录与公开结果投影 | interaction_id → topic_id 的稳定关联及真实路由信息 |

减少的是重复内容、重复身份和重复状态所有者，不是限制 Python 类的数量。路由指令、类型化 handle 或只读查询结果可按实际契约定义小型值类型；不能因此再复制整份交互正文或创建同义生命周期。

### 2.1 ConversationSession

```text
ConversationSession
  session_id
  source
  external_conversation_id
  workspace_ref
  state: open | paused | closed
  ordered_interaction_refs
  optional route_cursor_projection
  created_at / updated_at
```

Session 是由用户或调用软件控制的会话总容器，维护稳定来源、顺序、生命周期和历史引用。它作为 Workspace 归属的会话资源，由 workspace 共享设施中的会话服务持有本地记录（2026-09-28，总 Idea D-9；原候选为 System 会话服务），adapter 负责输入与生命周期触发；Patchouli 不持有第二份完整历史。外部 harness 仍拥有自己原生历史，本地只记录它交付的部分，不能宣称获得未提供的完整会话。Session 的管理/读取同样经过 A1 与资源 policy。

`ordered_interaction_refs` 指向本地保留的封口 `InteractionPayload`，表达逻辑顺序，不要求把全部历史装进一个无限增长对象。Session 记录与封口内容共同构成本地接收历史；内容的存放位置、分页、保留范围和容量待定（第 8 节），不能用内部 queue 或 apply journal 代替历史存储。进程内首版只承诺实际保存窗口，容量不足必须显式拒绝/报告，不能静默丢事件。会话资源不能当派生 cache 随便淘汰；删除/关闭 Session 不默认级联删除已形成的 Topic 或 Memory，provenance 引用过期时明确不可回查。route cursor 仅为可重建的路由投影，不决定既有交互的来源或内容。

Session 不在事务上拥有 Topic，也不把 `session_id` 继续塞进 `ActorIdentity` 作为身份事实。交互输入直接携带本地 `session_id`，结合 A1 的可信访问范围定位并校验 Session；现有 `ActorIdentity.session_id` 仅做兼容读取，最终迁移由契约和代码一起完成，不要求增加 SessionRef/InteractionContext 两层包装。不同来源相同 external_conversation_id 不合并；受信 source namespace 与 Workspace、用户的映射构成 Session 定位坐标，消息声称的 speaker 不授予 Session 访问权。

2026-09-23 归属确认（原边界宪章 §6.3，判据见 [ADR-0006](../architecture/decisions/0006-memory-library-custody-criteria-and-independence-contract.md)）：本节的 Session/Topic 切分即为宪章确认的归属——Session 是 workspace 归属的会话连续性真相源，Topic 是库对已提交素材的管护；短期记忆库的管辖权语义是"库已接收素材的接入暂存（intake buffer）"，不是 actor 可见记忆分层（命名是否随之调整待定）。由此得到四点确认：① 提交可靠性是过线契约，active finalize 的“Interaction applied 硬成功边界”与被动链的 InteractionSubmissionQueue 是该契约的现状形态，只指认、不重建（2026-09-28：任务进程在交互被提交队列接纳后即结束，不再等待 applied，见[任务进程 Idea](./task-process-table-and-registration-entry.md#q-1-进程何时关闭) Q-1）；② 结算触发（idle/LRU/shutdown）是库管理自身接入积压的库内事务，不改为由会话生命周期驱动，关闭 Session 不级联 Topic/Memory；③ Session 持封口 payload、Topic 持路由 blocks 的双份内容是接受的成本，不新增第三份；④ 短期库的管辖权语义如上。

对外能力上，会话相关操作可归为三类：session 管理/读取、`interaction.submit`（InteractionPayload 提交）与 topic 资料读取；领域行为只在 Patchouli 领域实现维护，能力层不复制。原计划把这三类方法接入 `workspace/capability`，排在 A2 能力层骨架之后；workspace 包的现有实现需要重新调查，见[总 Idea](./workspace-network-task-process-architecture.md)第 6.1 节。

### 2.2 InteractionPayload：共同封口交互

`InteractionPayload` 是 owner 提出的共用提交模型（2026-09-28 确认）。controller 模式下，任务进程在结算阶段提交它，`interaction_id` 字段始终取 `process_id` 的值；只有 completed 的进程才提交（[任务进程 Idea](./task-process-table-and-registration-entry.md#q-14-主动进程的交互记录去向) Q-14、Q-16）。

```text
InteractionPayload（演进现有模型）
  interaction_id
  session_id
  sequence（Session 内交互顺序）
  started_at / completed_at
  completion_state: completed | partial | cancelled | failed
  seal_reason
  turn_events: ordered TurnEvent
  topic_routing_directive
  source_metadata
  optional execution_reference（来源关联，不携带第二份正文）
  used_attachments / optional analysis hints（见第 6 节）
```

`InteractionPayload` 表达已经完成或明确封口的一次交互，是跨 Actor 的共同提交输入。它可以包含普通消息、助手回复、工具调用和工具结果，不等同于 Alice run，也不要求特定 prompt history。旧草案的 Segment 仅保留为历史概念名，不再引入独立 `segment_id` 或 Segment→Payload 内容转换层。

`interaction_id` 同时标识封口交互及其逻辑提交；同一次提交重试必须复用它。既有 envelope、公开结果与 apply journal 沿用同一值，不能各自产生新 ID；兼容参数与 payload 同时携带 ID 时须校验一致。`work_id` 仍是队列工作身份，外部 turn/event ID 和 execution reference 仍是来源关联，不与 interaction_id 互换。Session 顺序由可信 Session key 与交互 sequence 保证，不能继续简单使用 `topic:{topic_id}` 作为会话排序键；TurnEvent.sequence 则只表达交互内部顺序。

开放段是 accumulator 的可变工作记录，公开提交的是封口后的不可变快照；保留类名不意味着保留当前模型的可变提交语义。`completion_state` 描述交互实际结束结果，`seal_reason` 描述为何封口，两者都不表示 Patchouli 已接纳。取消/失败也可有可保存内容，不应因没有成功 final reply 而丢弃全部记录。一份交互不硬性要求恰好一条 user 和一条 assistant；可包含多条消息或不完整工具事件。可信 scope 仍由 access/内部 envelope 承载，正文中的来源信息不能成为第二份授权身份。controller 模式下只有 completed 的进程提交交互记录（任务进程 Idea Q-14），其余取值的用途见第 8 节第 6 条。

路由指令表达本次交互的处理意图；prepare handle（2026-09-28 起在 controller 模式下失去前提，见第 4 节开头）作为可选提交参数交接，不成为另一份正文或授权容器。其与 interaction_id、指令和幂等摘要的绑定方式待定（第 8 节）；无论采用哪种绑定，都要说明未接纳时如何显式重新准备；接纳结果未知或已经接纳时，须先定位原结果，不能以换 handle 为由改变既有路由。

### 2.3 TurnEvent：复用并扩展事件表达

```text
TurnEvent（在现有字段上补齐表达能力）
  sequence
  kind: user_message | assistant_message | system_message
         | thought | tool_call | tool_result | external_summary | unknown
  role / content / occurred_at
  optional action_id
  tool_name / tool_args / source call association
  status
  source_metadata
```

现有 TurnEvent 已能表示消息、thought、工具调用与结果，并允许缺少 action_id；候选设计补齐 system_message、external_summary、unknown、事件时间、外部 event/call 关联和 provenance 的表达，不另建等价 Part 模型。来源字段采用独立字段还是受约束 metadata 待定；外部 event ID 可选，不能为了统一强造新的全局 part_id。来源未知须可显式表达。

未知或缺少关联的工具事件必须保留在共同 TurnEvent 序列中，不因无法构造完整 action 而丢弃或伪造关系。source、occurred_at、received_at、外部 event id 和 connector provenance 是来源元数据，不是 Actor 认证凭证。

工具相关字段只适用于相应 kind，schema 应按类型约束，避免所有字段随意可空。外部 call ID 用于关联原始工具调用，不要求等于 Alice action_id；映射时保留来源，缺少关联时不能仅凭相邻位置补全。delta/chunk 在开放段合并为消息或显式分片后才投影，不把每个 token 伪装成独立动作。附件沿用 ref、revision 与实际使用快照，TurnEvent 按需补齐多模态引用表达，不把图片/文件强制塞进文本。`thought` 仅保留调用方实际提供的信息，不要求外部 harness 暴露内部推理。

## 3. Session 与 Topic 的边界

| 对象 | 负责 | 不负责 |
|:---|:---|:---|
| ConversationSession | 对外连续性、历史顺序、open/paused/closed、展示索引 | 记忆生成的相关性、Topic 摘要和模型工作集 |
| InteractionPayload | 一次封口交互、TurnEvent 顺序、来源、提交幂等 | 直接决定最终 Memory 或强制绑定一个用户 Topic |
| Topic | Patchouli 内的记忆材料分桶、blocks、summary、working set | 外部 session 生命周期和全量对话展示 |
| LogicalBlock | 交互转换后的 Patchouli 内容投影 | 保存整个 Session 或代替原始来源 |
| TopicData | Topic 内容事实、blocks、summary、bindings 的不可变快照 | 保存执行器 prompt history |

一个 Session 可以跨多个 Topic；首版一次交互只应用一个 primary Topic，避免拆分一次工具动作并重复记忆生成。跨 Topic 拆分交互属于后续算法决定，不在本文范围。Topic 的资源归属仍是 Workspace；Session→Topic 只保存路由关联，不新增 Topic 的 Session owner。

（2026-09-28 废止，见 0.1：Topic 不绑定 Session，路由候选为整个 workspace 的 Topic。以下为原候选设计。）新 Session 的自动路由候选默认限制在已关联 Topic；需要复用另一 Session 的 Topic 时显式选择并授权 attach。这个默认方向需要用现有 Alice/Passive Ingress 样例确认：不能把它误做新的资源硬隔离，也不能直接重写旧 Workspace 范围 Topic 的归属。迁移期保留 legacy 路由规则并明确版本；跨 Session 关联只改变可选路由范围，不合并 Session 历史。

`LogicalBlock` 可以保存 `session_id`、`interaction_id` 和 source provenance 引用，但不复制整个 Session。Session 保留提交时的内容，Topic 的 TurnRecord/LogicalBlock 是记忆侧投影；Topic 折叠、摘要或淘汰不回写 Session 历史。既有 blocks/summary/bindings 和 TopicData 快照保持 Patchouli 领域语义；新增关联优先使用兼容字段或现有应用记录，不能为此重写整个 Topic 存储模型。

## 4. Topic 路由指令与交接

（2026-09-28：任务进程不再预先创建 Topic；Gateway 的路由决定由进程携带到结算，提交后按需创建 Topic（0.1；任务进程 Idea 1.2）。本节中 prepare handle 与预创建 Topic 的条件清理，在 controller 模式下失去前提，待本方向的实施批次重审；路由指令、路由关联与应用结果的内容不受影响。）

本节集中定义 Topic 交接生命周期，包含没有 Session 的 prepare/主动意图场景，不意味着 Topic 属于 Session。这部分的公开方法、总线路由映射、operation 绑定和领域实现应在同一处定义；生成侧只引用本节，不另行决定 handle/assignment 的有效性、清理或来源绑定。

首版指令采用窄判别模型：`auto`、`continue(topic_id)`、`explicit(topic_id)` 和 `new`。交互路由处理形成稳定的 interaction_id → topic_id 关联，优先由现有应用记录持有，再通过授权的公开结果投影观察：

```text
InteractionSubmitResult / 应用结果查询投影（扩展既有结果语义）
  interaction_id / work_id
  admission / application status
  topic_id（路由确定后可用）
  optional route_reason / router_revision / confidence / manual_override
```

上述为候选的结果语义与内部指令，不意味着当前已提供同名外部接口。现有 `InteractionApplyRecord` 可承接路由关联，但其有界内部 journal 不能直接充当公共历史或查询 API；还需补齐结果可见范围、保留期、过期行为和 A1 授权。`InteractionSubmitResult` 是初次调用的不可变返回值，不会在后台自动变为 applied；后续查询读取新的结果投影，不暴露内部 queue/work record。不另设 TopicAssignment 实体或与 application status 竞争的状态机。

`auto` 在允许的候选中判断；`continue` 将上次 Topic 作为提示，允许明确的改路由；`explicit` 在通过授权后固定目标；`new` 强制新建并排除自动并回相似 Topic。若 `continue/explicit` 在真实调用中没有不同语义，应合并枚举，不能留下两个同义分支。confidence/router_revision 只有路由器实际提供时才返回，不伪造精度和版本。

（2026-09-28：会话操作与 Topic 路由的关系按 0.1 修订。）用户新建 Session 只创建会话容器；显式新 Topic 才约束内层路由。清空 Actor prompt context 也不等同删除 Session 历史或 Topic，adapter 若还要强制新 Topic 须另发 `new`。这让手动选择和自动路由遵循显式优先级，避免“标题相似”覆盖用户选择。UI 按钮、会话分支与批量重路由另行规划。

若在 prepare 阶段已经得出路由，应返回稳定的 prepare handle，submit 时携带该引用，避免前后重新路由造成不一致；它与交互路由关联的区别见第 4.2 节。未调用 prepare 的交互可按携带的路由指令在提交后处理；handle 失效不能被偷偷改写成“未提供 handle”。adapter 不把外部 session_id 伪装成 topic_id。

交互流程是：

```text
外部事件 -> Session 开放段中的 TurnEvent 序列 -> seal/flush
  -> 同一份 InteractionPayload 封口快照
  -> A1 access + interaction.submit 公共路由
  -> Patchouli 接纳交互（返回初始收据）、路由 Topic
  -> InteractionPayload -> TurnRecord / LogicalBlock / TopicData
  -> 通过结果查询观察实际 Topic 关联与应用状态
```

Session 服务保证输入去重和开放段的本地顺序；Patchouli application 必须独立验证交互提交幂等与顺序，不能完全相信 adapter。重试复用唯一的 interaction_id，服务端绑定 Workspace、Session、来源及规范化载荷，同键不同内容报 conflict。同一 Session 的同一 sequence 不得以另一个 ID 重复提交不同快照。Session 序列化与 Topic mutation 串行是两种约束：同 Session 跨 Topic 仍按交互顺序路由，不同 Session 显式 attach 同一 Topic 后仍由 Patchouli 的 Topic 占用/lease 保证安全，不为此引入第二套通用队列拓扑。

二者不使用跨领域大事务，而通过稳定 ID、路由关联、幂等和应用记录交接。accepted、routed、applied 是不同事实，即使共用一个结果结构也必须可区分。接纳时可能还没有最终 Topic，初始收据只报告 accepted；路由后可观察 topic_id，是否已应用仍须单独表达。已接纳且应用成功的重试必须定位同一结果，不能重新创建 Topic/block。结果已过保留窗口必须明确返回不可查询/已过期，不能据此认定从未提交再执行一次；幂等窗口与结果保留的协调待定。prepare、接纳、应用和清理之间的约束统一见第 4.3 节；具体待决项见第 8 节。

### 4.1 分开维护的状态与迟到事件

| 对象 | 状态由谁推进 | 与其他状态的关系 |
|:---|:---|:---|
| Session | workspace 会话服务（总 Idea D-9）；open/paused/closed | 不等于 Actor 正在运行或 Topic 已 settle |
| 开放段/封口快照 | adapter + Session accumulator；open→sealed，附 completion outcome | sealed 不等于已接纳或成功回复 |
| interaction 提交 | Patchouli application/queue；accepted→applied/failed | applied 不保证生成 Memory |
| Topic 路由关联 | Patchouli 路由/应用结果 | 对同一已应用交互稳定，不成为访问凭证；不新增独立状态机 |
| Memory/Pending 物化 | Patchouli 领域任务；属于后续消费 | 独立于 Session 关闭和普通交互结果 |

同一结束信号重复到达返回已有封口/提交关联。未接纳时保留封口快照，响应丢失后用同一 ID 查询或受控重试。迟到消息/工具结果不能改写已封口的 InteractionPayload：首版显式拒绝/报告迟到，或由调用方以新的交互带来源引用补交；具体支持方式待定，不隐式 reopen 或重放整个交互。sequence 缺口、乱序和时间戳冲突需定义可接受窗口/拒绝策略，不承诺任意自动重排。

### 4.2 Topic 引用词汇与责任

下表统一既有计划中 handle、assignment reference 和 Topic 快照的含义，不要求为每个概念新增一个服务或 Python 类。

当前 `TopicManagementService.prepare_topic()` 返回 topic_id 字符串，旧 `cleanup_prepared_agent_run()` 还依赖 PreparedAgentRun、新建标记与交互接纳状态。本节 handle 是目标交接语义；旧字段如何承接它待定（第 8 节），不宣称现有实现已有 handle 注册、预留或过期机制。

| 对象 | 表达什么 | 不代表什么 |
|:---|:---|:---|
| Topic identity/ref | Workspace 中的具体 Topic 身份 | 访问许可、资料版本或某次交互已应用 |
| prepare handle（原 route handle） | 一次准备操作的稳定路由交接引用，绑定原 scope、路由决定及必要关联 | 交互的应用结果、内容快照或永久有效的访问凭证；是否预创建/预留 Topic 仍待冻结 |
| Topic 路由关联（assignment） | 交互与 Topic 的路由关联及可用路由来源信息，保存在应用记录/结果中；对已应用交互稳定 | prepare handle 的别名；存在关联不单独证明 blocks 已应用 |
| interaction receipt | 交互接纳、应用或失败及其关联结果 | Topic 的整个生命周期，也不等于 Memory 已生成 |
| TopicData/资料快照 | 某一读取时点实际可用的 blocks、summary、bindings/provenance | 永远新鲜的内容、全部 Session 历史或完整原文档案；router_revision 也不是内容版本 |
| 主动意图的 Topic 资料绑定 | 该意图实际选择的 Topic 来源及资料快照关联 | Session 的可变 route cursor；不默认依赖 prepare handle 持续存活 |

Patchouli 拥有 handle 的验证、路由关联、资料读取及条件清理语义；Topic 领域实现只有一处，API 目录和主动生成只是消费者，不产生新的运行时所有者。System/Actor adapter 持有引用并请求操作，不能凭“由我 prepare”绕过后续授权，不能从 handle 字符串自行推断 Topic 身份和清理权限。handle 也不默认等同于 Asset lease 或 Topic 排他锁。无论编码为 opaque string 还是类型化值，背后都必须保留验证、有效期、资源创建事实与接纳责任转移所需的状态；不能以减少模型为由省去这些生命周期事实。

### 4.3 prepare → submit/apply → cleanup 生命周期

以下是行为约束和责任转移，不是提前增加一套固定状态枚举。prepare 是需要预先选定 Topic 的调用方可使用的步骤；普通交互和主动意图不因本表被强制增加一次 prepare 往返。

| 阶段/触发 | 必须保持的行为 | 后续责任 |
|:---|:---|:---|
| prepare 成功 | 通过当前访问检查；以稳定 handle 交接路由决定；记录本次是否创建了资源的真实结果 | 是否预创建/预留、重复 prepare 的关联键与有效期待定（第 8 节） |
| 尚未提交或接纳结果未知 | 调用方保留原 handle、封口内容和提交 identity；响应丢失先查询/受控重试 | 结果未知不能推断为未接纳并清理；不悄悄换 Topic |
| interaction.submit 校验与接纳 | 验证 handle 的 scope、关联、有效性及与请求指令的一致性；无 handle 按公开路由指令处理 | 接纳后由 Patchouli 接管已承诺的领域工作和必要 Topic 保护；adapter 断连不撤销 |
| interaction apply | 使用已确认的路由关联应用 blocks；重试复用同一交互结果 | 路由结果与 applied 分别观察；应用成功固定关联，不重复创建 Topic/block |
| prepare 放弃、调用方失败或 handle 到期 | 可请求释放准备关联；若曾预创建 Topic，是否回收由领域条件检查决定 | handle 失效/释放不等于删除 Topic，不撤销已接纳工作 |
| 领域处理失败后的补偿 | Patchouli 根据实际应用、接纳和占用事实判断是否可回收 | adapter 不以通用 evict 代替条件清理；已经应用的交互与后置物化结果仍各自成立 |

条件清理只面向本次准备可回收的资源：复用的既有 Topic 不因当前调用方失败被删除；预创建 Topic 也须在同一领域并发边界检查仍为空、没有已接纳工作或其他有效占用需要它，满足条件才清理。不能由 adapter 先读取“为空”再无条件 evict。具体占用/保留实现和“空”的判定待定（第 8 节），不要求新增通用 lease 系统。重复释放/清理必须有可解释的幂等结果；不可清理与失败须可区分。

Topic 条件清理与管理员主动 settle/evict 是不同操作语义，不能借管理权限替代 prepare 补偿。附件 lease 的持有与释放仍按既有 W1 契约处理；Topic 无法清理不意味着附件 lease 可以泄漏，释放附件也不意味着 Topic 可以删除。

### 4.4 向主动意图交接资料

Topic 身份、prepare handle、路由关联和资料快照分别交接。主动意图不要求先有 Session、交互或 prepare；若调用方明确要求纳入某次交互，先通过授权结果查询确认该交互 applied，并取得其实际 topic_id，再提交意图。只收到 prepare handle、accepted 或路由结果不能证明该交互内容已进入 Topic。两条提交不合并事务，也不建立等待未来交互的全局屏障。

| 输入情形 | 共同处理要求 |
|:---|:---|
| 明确指定已有 Topic | 校验存在、Workspace 归属和当前读取权限，再交付真实资料；失败不当作空资料 |
| Topic 合法但 blocks 为空 | 返回成功的空资料；不把内容为空视为意图无效或读取失败 |
| 只提供 prepare handle | 按确定后的 handle 契约验证；若允许用于主动意图，领域侧转换为稳定资料绑定；是否接受该输入待定（第 8 节） |
| 来源为已应用交互 | 按实际应用结果中的 Topic 关联选择，不使用当前 Session cursor 重选 |
| 未提供 Topic/交互 | 保留意图独立生成；采用无辅助 Topic，还是由领域准备空 Topic，待定（第 8 节） |

资料读取与绑定须固定本次任务的来源，后续 Session 路由或 handle 状态变化不能让同一任务重试悄悄切换 Topic。若处理需要继续访问活 Topic，应明确保护何时取得和释放；若已取得独立资料快照，则按快照保留约定处理，不靠临时 handle 充当无限期资料保留。选择哪种绑定、取样时点、Topic 内容变化后的重试行为以及可用版本证据待定（第 8 节），不假定现有 Topic 已有可持久化 revision 或通用 pin 能力。

本节涉及资料身份、全部可用内容的读取/绑定和保留/释放；生成侧的预算编译与 Pending 状态不在本文范围，见[写入意图体系迁移](./pending-intent-migration.md)。资料为空、原文已经折叠而不可恢复、读取失败和无权读取必须区分。

### 4.5 实现复用方向（候选）

复用 `TopicManagementService`、`InteractionSubmissionService` 与现有全局路由地址，对尚未公开的方法补齐明确地址和 operation；不另包一层 provider 或 Workspace service。演进后的 `InteractionPayload` 是公开提交输入，`InteractionSubmission` 仍是内部 queue envelope；二者沿用同一个 interaction_id。apply/路由关联观察应与提交入口同时可用。

## 5. 结构化事件的构建时机

外部 adapter 在接收阶段把实际收到的事件归一化为 TurnEvent，放入开放段；共同归一化规则应复用，来源特有协议解释留在 adapter。交互封口可以由完成回调、assistant 结束、工具结果完整、显式 flush、shutdown 或取消信号触发；缺少结束信号不能推断成功完成，partial/cancelled 必须保留其状态。

以阶段而非定时器确定构建时机：接收时建立最小事件，封口时冻结内容，Patchouli 在应用前建立领域投影。已有足够结构可以直接复用，只有依赖完整交互才延后处理：

1. 接收时保存有序 `TurnEvent` 及来源；未知 kind、缺失说话者或工具关联按 schema 显式表达，不等完整 Action 才记录。
2. 封口时冻结 `InteractionPayload`、interaction_id、Session sequence 与结束状态；不再另造 Segment 或 execution_projection 正文。
3. Patchouli 应用前，在可靠调用关联下由 ActionReducer 聚合 `AgentAction`。调用已知而结果缺失可形成显式 pending/incomplete action；孤立结果保留为事件，不能虚构成功结果或凭相邻位置强配。
4. 形成 `TurnRecord` 和 `LogicalBlock` 时保留 interaction/event provenance；trace 等摘要由同一投影过程派生，不反向覆盖原事件。不完整事件未形成 action 不等于可以从输入历史中删除。

外部客户端只须按外部接入协议（见[外部 Actor 的接入登记与运行时访问](./external-actor-registration-and-runtime-access.md)）提交它实际拥有的消息/工具事件；内部 adapter 负责转换为 TurnEvent，不要求客户端实例化 Python 类型、构造 `AgentAction` 或 `TurnRecord`。Alice adapter 直接保留既有 TurnEvent 的结构与 action 关联，提交相同 InteractionPayload，Patchouli 使用同一领域投影规则。`TurnRecord.identity` 不再承担 session 身份；资源授权只使用 A1 的 access context。

复用事件类不等于当前 reducer 已具备正确语义：`ActionReducer._resolve_action_id` 会把没有 action_id 的 thought/tool_result 关联到最近动作，这对缺失或乱序的外部关联不可靠。需要限制这一推断，确定显式关联、可证明的来源映射及无法关联时的保留规则；Alice 已有可靠 ID 应直接保留。TraceReducer 的输出不能代替原事件，也不能成为第二份提交正文。

目标上 Session 保存封口 InteractionPayload，TurnRecord 是 Topic 内的结构化内容投影；现有代码将 TurnRecord 称为“单轮内容真相”，迁移时明确它在 Topic 领域内的含义，不让两份正文相互回写。同步评估当前依赖 user 文本的完整性判定，允许无 user 的合法事件记录；保留记录不意味着必须生成记忆。未知 kind/多模态 ref 无法完整进入旧 Topic 投影时保留在共同记录及 provenance，并在结果中表达投影缺口，不能冒充无损应用。

## 6. InteractionPayload、队列 envelope 与主动意图

当前 `InteractionPayload` 混合 user/assistant 文本、turn_events、mtp_traces、worth_saving、附件使用和 `materialize_tasks`。问题在职责与字段语义，不需要通过新增同义正文模型解决。迁移目标是：

- 演进 `InteractionPayload` 为正式共同输入，补齐 Session、稳定 identity、顺序、封口与来源语义；以 turn_events 为内容来源。
- 原 user/final 文本和 trace 逐项退出独立正文地位，兼容解码或只读派生视图必须可追溯到同一事件序列。
- `InteractionSubmission` 继续是进入 Patchouli submission lane 的内部 envelope，携带可信 scope、同一 interaction_id 和 ordering；queue/work 身份不外化为会话身份。
- `materialize_tasks` 不进入新版本 interaction 输入；主动 WRITE/UPDATE 保持独立提交责任，其公共模型见[写入意图体系迁移](./pending-intent-migration.md)。

| 现有字段 | 目标承接 | 兼容要求 |
|:---|:---|:---|
| user_message / assistant_final_text | 旧纯文本输入转换为消息 TurnEvent；新输入的文本视图由事件派生 | 已有事件时核对内容一致性，不重复追加；不硬性要求非空 user；不凭两段文本伪造工具调用 |
| turn_events | 唯一有序事件正文；扩展来源与不完整事件表达 | Alice 原 action 关联保留；外部缺失允许表达；不再增加 parts/execution_projection |
| mtp_traces | Patchouli 内部派生 trace；旧输入只作兼容资料 | 有事件时派生并校验，不维持平行真相；旧 trace 独有内容保留来源/报告缺口，不能静默丢弃或反推完整动作 |
| rewritten_query / worth_saving | 标注来源的分析提示 | 不覆盖原话、不代替 Patchouli 最终领域判断；价值体系重设另行计划 |
| used_attachments | 已实际编译使用的 ref/revision 快照与 provenance | retry 原样重用；应用成功后才建立 binding，不回查 UI 或上传列表 |
| model_used | 来源展示信息/可选执行元数据 | 外部未提供时未知，不要求绑定本地模型注册表 |
| materialize_tasks | 调用侧独立保留的主动任务，不进入新版本 InteractionPayload | 新输入拒绝混入任务；混合旧载荷不静默丢任务，消费者切换时分别提交一次 |

现有 `InteractionSubmission` codec 已有 schema_version 和严格字段校验，引入新载荷时必须显式升级版本并验证规范化摘要与往返一致性。旧版本按受信兼容解码路径转换，不能用一组可选字段同时猜测新旧语义。同一次封口保留唯一 interaction_id；旧 envelope 已有 ID 时原样沿用，缺失时在首次封口/受信转换边界生成并保存，重试不重新生成。随机 turn_id/block_id 不作为提交幂等依据。

一次交互可以没有主动意图；一个主动意图可以没有交互或 Session。交互结果向主动意图交接 Topic 资料的条件统一见第 4.4 节。尚未切换的混合旧入口需要受信兼容窗口，不能将任务丢弃后冒充转换成功；独立主动任务的提交不与交互入口合并。

## 7. RelayController 与外部上下文

RelayController 的新边界是 Topic working set 的折叠器：管理 Topic 侧的 token/block 预算、折叠原文的引用和供 Patchouli 生成使用的摘要。它不负责 ConversationSession 生命周期，也不应替外部 harness 压缩 prompt history。外部 harness 提供的 summary/checkpoint 作为扩展后的 `TurnEvent(kind=external_summary)` 或带 provenance 的辅助材料进入系统，不自动覆盖 Topic canonical `state_summary`。

更细的设计分别见两份 Idea：page folding、raw evidence、容量与后台恢复见 [Page Folding Raw Evidence](./PatchouliPageFoldingRawEvidenceDesign.md)，长 turn checkpoint 与前台上下文见[长时间运行 Agent 的 Turn 内上下文折叠](./long-running-agent-intra-turn-context-folding.md)；本文只涉及其必须依赖的 Session/interaction/Topic 关联和事件来源边界。

（2026-09-28：本段中 Alice 以 Topic 工作集组织 prompt 的表述已不成立，对话上下文由 Session 提供、压缩由 CPU 负责，见 0.1。）Session 完整历史的展示与 prompt 预算是两件事。Alice 可以继续选 Topic 工作集、短期窗口和检索结果组织 prompt；外部 harness 自主管理 prompt 与厂商缓存。不把所有 Session 内容自动注入模型，也不以重构为由删除现有记忆侧短期工作集。“接收记录就替外部 Actor 改写执行上下文”的耦合需要解除；折叠质量算法本身继续保留现有实现，直到专项替换。

## 8. 开放问题

原计划中留待 A3-0 冻结的事项如下，均未决定。

模型、字段与语义：

1. TurnEvent 新增 kind、来源/时间、外部调用关联和多模态引用的字段；ActionReducer 的可靠关联规则；Payload 文本/trace 的派生与兼容限制。第 2 节的候选设计复用现有模型，不另建 Segment/Part 双模型。
2. System Session 及封口 Payload 的物理位置（2026-09-28：ConversationSession 位于 workspace 的共享设施，见[总 Idea](./workspace-network-task-process-architecture.md#d-9-chat-编排与-chat-run-注册表的最终归属) D-9）、最小创建/读取/封口/关闭能力与 A1 operation、历史分页、保留期/容量/溢出行为；暂停/删除是否暴露及其边界。不要求创建新的大子系统或耐久历史平台。
3. 旧 Passive Ingress `source + external_conversation_id + actor` key 的兼容映射，补齐可信 Workspace 分区及 scope/Actor 漂移规则；移除 identity.session_id 对 equality/hash/cache key 的影响。speaker、来源与调用主体分别处理。
4. 无 session 的旧 Alice/单次交互如何映射为明确的临时 Session，不能静默合并全用户历史；主动意图仍可无 Session。旧 Topic 只保存可证来源，不能编造完整 Session。
5. interaction_id 在封口、Session 历史、公开参数、queue envelope、apply record/result 间的一致映射；work_id、外部 ID 和 turn/block ID 各自含义；新旧 codec、版本与规范化摘要的兼容样例（2026-09-28：controller 模式下 `interaction_id` 始终取 `process_id` 的值，见任务进程 Idea Q-16）。
6. 封口信号、completion outcome、partial/cancel/failed、迟到/重复/sequence 缺口策略；Session 与 Topic 两种序列化约束；开放段容量和无 user 事件的接收规则。
7. 路由指令优先级、默认候选范围和显式 attach（2026-09-28：默认候选范围已决定为整个 workspace，见 0.1）；Topic prepare/handle、条件清理、资料绑定、结果观察/保留与 API 决定见下方 Topic 生命周期事项；跨 Session 复用不改变 Workspace ownership。

Topic 生命周期事项：

| 事项 | 需要回答的问题 |
|:---|:---|
| prepare 副作用与引用 | 是否预创建/预留 Topic（2026-09-28：任务进程不再预先创建，见第 0 节）；handle 的 scope/调用关联、引用编码与提交参数；同 interaction_id/路由指令/幂等摘要的绑定及冲突行为 |
| handle 存续与交接 | 有效期、一次/多次使用、重复 prepare、接纳时保护转移、过期后的查询/冲突/显式重新准备 |
| 条件清理 | 本次创建的证据、空/占用判定、与并发 submit/资料读取的协调、幂等结果、关闭时收尾 |
| 资料来源与保留 | 省略 Topic 的唯一行为；是否接受 prepare handle；快照时点/版本证据、重试绑定及必要保护释放 |
| 应用结果与路由关联 | 扩展既有 apply record/result 的字段；accepted/routed/applied 的观察方式、授权、保留/过期与幂等窗口 |
| API/授权/兼容 | prepare、资料读取、assignment/apply 查询、条件清理、interaction.submit 的参数/结果、operation 与旧 prepare_topic 返回 topic_id 的迁移 |

## 9. 设计需满足的行为约束

以下约束来自原计划的验收要求，是候选设计必须满足的行为，不是测试计划：

- 同 Session 跨多个 Topic 仍按序提交；显式 new/continue/attach 不合并历史；Topic 折叠后保留窗口内的 Session 原事件不变。
- 两个相邻工具调用后的无关联结果不会被强配到最后一个动作；孤立结果、无 user 消息、未知事件、多模态引用和部分失败交互可保留并报告投影缺口。
- 重复 seal/提交沿用同一 interaction_id；响应丢失后重试不新建 Topic/block；同 ID 不同内容、同 Session sequence 冲突及迟到事件按契约处理。
- 同名外部 conversation 在不同 source/Workspace 中隔离；Session 读取和结果查询均校验权限，猜中 interaction_id/work_id 不授予访问权。
- 初次 accepted 与后续 routed/applied 查询可区分；结果过期不当作未提交；Session、结果及幂等记录的保留边界可解释。
- 新旧 codec 的字段、摘要和往返转换稳定；旧纯文本/结构化事件不重复，混合主动任务不被转换过程静默丢弃。
