---
title: 外部会话消息的接收与 Topic 投影
status: idea
horizon: long-term
owner: project
scope: conversation-session-interaction-event-topic-projection
related_docs:
  - docs/ideas/short-term-memory-and-context-compaction.md
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/fifth-generation-architecture.md
  - docs/archive/ideas/pending-intent-migration.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/architecture/decisions/0006-memory-library-custody-criteria-and-independence-contract.md
  - docs/ideas/PatchouliPageFoldingRawEvidenceDesign.md
  - docs/ideas/long-running-agent-intra-turn-context-folding.md
  - docs/todo/conversation-compact-command.md
  - docs/archive/ideas/unified-interaction-memory.md
  - docs/archive/plans/import-bus-authorization-in-capability-layer.md
last_reviewed: 2026-11-24
---

# 外部会话消息的接收与 Topic 投影

## 0. 文档性质

本文是会话、交互输入与 Topic 之间关联与交接的设计，要解决的问题是：Topic 体系要能接收外部 Actor 的会话消息。正文是已定的设计，候选设计标“候选”；已经实现的部分以当前事实文档为准。

- **已经实现**：ConversationSession 承接 Alice 的会话资源（[会话资源](../workspace/sessions.md)）；主动链路与 Import Bus 共用封口函数，提交同一种 `InteractionPayload`，经 root API 的“提交交互”（[任务进程](../workspace/task-process.md)第 8 节、[Import Bus](../system/import-bus.md)）；话题按需建立（第 10 节）。
- **仍是候选**：外部输入的事件表达、顺序与封口语义，面向外部的结果查询，以及 controller 模式下的会话交接。
- **两种接入模式**：plugin 模式的会话归外部 harness 所有；Alice 与 controller 模式的会话由 HiveMemory 发起（[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#11-两种接入模式owner2026-09-27) 1.1）。controller 模式是候选方向。
- **共用的提交模型**：`InteractionPayload` 是 owner 提出的共用提交模型，不是待选方案；外部输入要补的字段细节待定（第 8 节）。
- 文中“A1”指访问边界：两阶段认证与两阶段授权（[身份与授权](../workspace/identity-and-authorization.md#2-访问边界两阶段认证与两阶段授权)第 2 节）。

外部 harness 通常拥有确定性的 session/conversation，而 Patchouli 需要根据内容把交互分配到 Topic，以便组织记忆生成资料。两者都是有意义的分桶，但回答不同问题：Session 负责对外会话连续性、顺序和展示；Topic 负责记忆侧的相关性划分和短期工作集。压缩机制见[短期记忆](./short-term-memory-and-context-compaction.md)，完整资料保存见 [Raw Evidence](./PatchouliPageFoldingRawEvidenceDesign.md)，长 turn 的输入与封口见[专项 Idea](./long-running-agent-intra-turn-context-folding.md)。

**Topic 与会话解耦**：Topic 不再承担上下文的控制，它组织记忆生成的材料，并以话题的进展向前台提供“进行到哪一步”：另一个会话被路由回已有话题、或会话压缩之后推送，只读出、不控制（[短期记忆](./short-term-memory-and-context-compaction.md)第 7 节）。实际使用的对话上下文由 ConversationSession 提供，原样积累，不受外界干涉；CPU 管理派生视图的预算与压缩。Patchouli 通过接收到的交互组织材料，不在生成时反向读取 Session 或 CPU 上下文。

### 0.1 会话模型与 Topic 池

**背景**：Topic 机制最初同时承担短期记忆与 Agent 上下文控制，希望实现自动的上下文场景切换：前端对应自动的 Topic 卡片切换与话题池展示，以及半自动的上下文管理，包括为适应长上下文而压缩对话。这与主流的 session 视角冲突：

1. session 由用户主动创建，Topic 完全由后台自动控制；
2. 一般的 harness 都自带运行时的上下文压缩，与 page folding 冲突；
3. Topic 机制也不符合厂商推广 KV cache 的方向：输入一直在变，命中不了缓存，反而可能没有 token 经济效益。

要使用外部 Actor，必须解决 session 与 Topic 体系的不一致。Topic 机制背后的理念（对话语义的一致性与连贯性）在记忆生成一侧保持；前台的对话体验回归常规 Agent 软件的 session 概念。

**设计**：

| 事项 | 设计 |
|:---|:---|
| 会话操作 | 新建与恢复控制对外会话连续性；会话内 compact 的语义与“不设跨会话交接摘要”由[短期记忆](./short-term-memory-and-context-compaction.md#81-已完成的范围)维护 |
| Topic 与 Session | Topic 不绑定 Session，workspace 共享一个 Topic 池；Gateway 的路由候选为整个 workspace 的 Topic。同一主题在不同会话中的讨论路由到同一个 Topic，记忆材料的语义保持连贯。Session 与 Topic 之间只有路由关联：会话包含若干进程，每个进程的交互记录路由到某个 Topic |
| 跨会话连续性 | 用户有意分开的会话，靠中期与长期记忆接续；想回到原来的话题时恢复原会话 |
| Gateway 的会话提示 | 由 Gateway 识别“本轮消息属于另一个会话的话题”，向用户提议恢复已有的会话或新开会话；放到后续版本，现在 Gateway 的话题路由只服务于后台的 Topic 路由 |
| 前端 | 左侧是 session 列表（新建、恢复、重命名、加载历史）；没有“当前 Topic”；`topic_info` 是异步的记忆标注（进程结束后得知本轮归入了哪个 Topic）；Topic 池在记忆面板，作为短期库的整理视图，保留 settle 与 delete 管理 |

**影响**：

- 一个 Topic 中会交错出现来自多个会话的 block；提交按 Topic 串行处理，不影响记忆生成，但 Topic 不能再作为某个会话的对话视图展示；
- controller 模式下，恢复会话对外部 harness 意味着恢复它的原会话，取决于 harness 是否支持（例如 ACP 的 `session/load` 是可选能力）；不支持时只能新建。

### 0.2 会话上下文的后续方向

会话资源、Alice 与 Gateway 的会话上下文、有输出的取消与失败提交、前端会话模型已经实现，当前模型与失败语义以[会话资源](../workspace/sessions.md)为准。仍是候选的方向：

- **会话压缩**：由[短期记忆](./short-term-memory-and-context-compaction.md)维护，本文只保留外部会话与 CPU 能力之间的关联。
- **执行环境与外部 harness**：外部会话的恢复和压缩依赖 harness 的能力；会话绑定执行环境的选择见[执行环境 Idea](./execution-environment.md) T-8a，同一 harness 会话切换 Agent 是尽力应用另一份 Profile（第五代架构 P-10a）。面向外部输入的模型演进留给 controller 模式。
- **会话持久化**：会话只在进程内保存，跨后端重启恢复要另行立项。
- **Gateway 会话提示**：最近 N 轮与每轮 Topic 关联可以为它提供输入，但提示判断还没有实现；会话 ID 不需要进入 Patchouli。不能用上一轮 Topic 强制偏置路由，否则不利于识别属于另一个会话的内容；提议本身不改变用户的上下文。
  - **路由上下文的矛盾**（开放）：会话中指代不清的输入（如“把它修好”）只有结合会话上下文才能路由到正确的话题；会话提示却要求路由不受会话惯性影响，才能识别“本轮属于另一个会话的话题”。话题路由不带会话、只看消息与话题菜单时，指代衔接不上；先做依赖上一轮会话内容的意图理解，或在路由中带入会话，会话提示就无法生效。这一矛盾自话题路由与查询理解拆成两步时就存在。现状偏向指代消解：两项分析都接收最近 N 轮；带各话题最后一轮的菜单（`gateway/topic_context.py` 的 `render_topic_snapshots`）没有消费者，路由器使用只含标题与摘要的菜单。
  - **实测约束**：会话轮次以用户/助手对话消息交给分析模型时，真实模型会接着对话回答用户，JSON 模式下返回空白，两项分析随之降级；现在改为与话题菜单同形的编号列表，嵌入 system 提示词（[Gateway 分析](../gateway/analysis.md)第 4.1 节）。以后提供会话上下文时同样不能使用对话消息的形式。
  - **候选方向**（需以真实模型实验验证）：把指代展开与路由分开。先用会话上下文只展开指代，能独立理解的输入原样保留；再以展开后的消息做不带会话的路由，会话惯性不进入路由判断；查询理解仍在路由之后进行。

## 1. 目标与非目标

目标是在现有 `InteractionPayload`、`TurnEvent` 和提交结果上补齐会话、顺序与来源语义，让增量接收的外部 adapter 也能提交同一种封口交互；Patchouli 内部继续产生 `LogicalBlock`、`TopicData` 和交互到 Topic 的路由关联。Session 历史可以跨多个 Topic；Alice 取得全量历史，外部 harness 的上下文交接按各自能力设计。非目标包括：把 Session 变成 Topic 的另一份 canonical store、要求外部 Actor 构造 Alice 的 `TurnRecord`、在此决定摘要算法、实现全历史导入或替换外部 harness 的上下文压缩。

Topic 的路由、提交、应用与资料交接集中在第 4 节定义。

代码依据：actor 身份不含 `session_id`，HTTP chat 请求体以该字段绑定 workspace 会话资源；`core/protocol/models.py` 的 `InteractionPayload` 只有 `outcome`、`turn_events`、`model_used` 与 `used_attachments`，交互内容只以 `turn_events` 表达，构造时校验事件集以序号 0 的用户消息开头；`core/models/interaction.py` 的 `TurnEvent` 有 `user_message`、`assistant_message`、`thought`、`subagent_message`、`tool_call` 五种 kind，一次工具调用与它的结局是同一个事件，`TurnRecord` 只保存结局、标识、actor 与事件集；`core/models/topic.py` 的 LogicalBlock 内嵌 TurnRecord。`patchouli/application/interaction_submission_service.py` 接纳后等待应用，返回的 `InteractionSubmitResult`（定义在 `patchouli/contracts/interaction_submission.py`）带进度、成功时实际所在的话题与失败时的异常类名，提交队列以 `topic:{topic_id}` 排序，同一交互的幂等只由提交队列负责。外部 harness 的整轮交互由连接器读出、经 Import Bus（`system/services/import_bus/`）导入，按（user、Agent、source、external_conversation_id）串行并推导 `interaction_id`。Patchouli 的进度查询与等待路由只在 root API 内部调用；外部无损输入（开放段与增量接收）和面向外部的应用结果查询没有实现。

## 2. 数据模型（候选）

复用方向如下；具体字段、类型、必需性、长度与容量、缺省和兼容样例待定（第 8 节）。下列内容不是已发布 schema。

| 原拟模型 | 候选设计 | 保留的职责 |
|:---|:---|:---|
| ConversationSession | 扩展已实现的会话资源以承接外部输入 | 会话身份、历史顺序、生命周期与保留范围 |
| ConversationSegment | 不新建；演进现有 InteractionPayload | 一次封口交互的共同内容输入 |
| ConversationPart | 不新建；扩展现有 TurnEvent | 有序消息、工具事件及其来源 |
| SessionRef | 不作为必需包装；可信访问范围内使用 session_id | 跨范围寻址确有需要时可定义小型复合值类型 |
| InteractionContext | 不新增通用容器 | 会话/来源关联归交互输入，授权归 A1 access |
| TopicAssignment | 不要求独立实体；扩展既有应用记录与公开结果投影 | interaction_id → topic_id 的稳定关联及真实路由信息 |

减少的是重复内容、重复身份和重复状态所有者，不是限制 Python 类的数量。路由指令或只读查询结果可按实际契约定义小型值类型；不能因此再复制整份交互正文或创建同义生命周期。

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

Session 是由用户或调用软件控制的会话总容器，维护稳定来源、顺序、生命周期和历史引用。它是 Workspace 归属的会话资源，由 workspace 共享设施中的会话服务持有本地记录，adapter 负责输入与生命周期触发；Patchouli 不持有第二份完整历史。外部 harness 仍拥有自己原生的历史，本地只记录它交付的部分，不能宣称获得了未提供的完整会话。Session 的管理与读取同样经过 A1 与资源 policy。

`ordered_interaction_refs` 指向本地保留的封口 `InteractionPayload`，表达逻辑顺序，不要求把全部历史装进一个无限增长对象。Session 记录与封口内容共同构成本地接收历史；内容的存放位置、分页、保留范围和容量待定（第 8 节），不能用内部 queue 代替历史存储。进程内首版只承诺实际保存窗口，容量不足必须显式拒绝或报告，不能静默丢事件。会话资源不能当派生 cache 随便淘汰；删除或关闭 Session 不默认级联删除已形成的 Topic 或 Memory，provenance 引用过期时明确不可回查。route cursor 只是可重建的路由投影，不决定既有交互的来源或内容。

完整资料的跨环节保留按[全链路 Raw Evidence](./PatchouliPageFoldingRawEvidenceDesign.md)收敛：Topic 要在自己的生命周期内持有已交接的资料或稳定引用，不能只保留依赖 Session 不被删除的临时指针。这里的保留窗口与引用过期不是允许 folding 删除唯一原文的例外，会话持久化另行计划。

Session 不在事务上拥有 Topic，`session_id` 也不进入 `ActorIdentity` 作为身份事实：actor 身份不含 `session_id`，旧交互 Artifact 只读兼容额外键并保留原始 hash 校验。候选交互输入直接携带本地 `session_id`，结合 A1 的可信访问范围定位并校验 Session，不增加 SessionRef/InteractionContext 两层包装。不同来源相同 external_conversation_id 不合并；受信 source namespace 与 Workspace、用户的映射构成 Session 定位坐标，消息声称的 speaker 不授予 Session 访问权。

**Session 与 Topic 的归属**（判据见 [ADR-0006](../architecture/decisions/0006-memory-library-custody-criteria-and-independence-contract.md)）：Session 是 workspace 归属的会话连续性真相源，Topic 是库对已提交素材的管护；短期记忆库的管辖权语义是“库已接收素材的接入暂存（intake buffer）”，不是 actor 可见的记忆分层。由此：

1. 提交可靠性是过线契约：主动链路与 Import Bus 共用的 root API“提交交互”提交并等待应用（第 10 节）。
2. 结算触发（idle、LRU、shutdown）是库管理自身接入积压的库内事务，不改为由会话生命周期驱动，关闭 Session 不级联 Topic 或 Memory。
3. Session 持封口 payload、Topic 持路由 blocks 的双份内容是接受的成本，不新增第三份。

对外能力上，会话相关操作可归为三类：session 管理与读取、交互提交、topic 资料读取；领域行为只在 Patchouli 领域实现维护，能力层不复制。现状：会话管理在 root API；交互提交是 root API 的“提交交互”，由系统执行、参与的 Agent 显式传递，任务进程与 Import Bus 都经它提交（[身份与授权](../workspace/identity-and-authorization.md#4-操作授权)）。外部交互提交面的设计没有实施。

### 2.2 InteractionPayload：共同封口交互

任务进程在结算阶段提交 `InteractionPayload`，`interaction_id` 取 `process_id` 的值；有输出的 completed/cancelled/failed 都带结局提交（[任务进程](../workspace/task-process.md)）。下列外部输入字段草案不代表当前 schema。

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
  used_attachments / model_used（见第 6 节）
```

`InteractionPayload` 表达已经完成或明确封口的一次交互，是跨 Actor 的共同提交输入。它可以包含普通消息、助手回复、工具调用和工具结果，不等同于 Alice run，也不要求特定 prompt history；不引入独立 `segment_id` 或 Segment→Payload 的内容转换层。

`interaction_id` 同时标识封口交互及其逻辑提交；同一次提交重试必须复用它。既有 envelope、公开结果与提交队列沿用同一值，不能各自产生新 ID；兼容参数与 payload 同时携带 ID 时须校验一致。`work_id` 仍是队列工作身份，外部 turn/event ID 和 execution reference 仍是来源关联，不与 interaction_id 互换。Session 顺序由可信 Session key 与交互 sequence 保证，不能继续简单使用 `topic:{topic_id}` 作为会话排序键；TurnEvent.sequence 只表达交互内部顺序。

开放段是 accumulator 的可变工作记录，公开提交的是封口后的不可变快照。`completion_state` 描述交互实际结束结果，`seal_reason` 描述为何封口，两者都不表示 Patchouli 已接纳。取消与失败也可有可保存内容，不因没有成功的最终回复而丢弃全部记录。一份交互不硬性要求恰好一条 user 和一条 assistant；可包含多条消息或不完整工具事件。可信 scope 仍由 access 与内部 envelope 承载，正文中的来源信息不能成为第二份授权身份。Alice 使用 `outcome=completed/cancelled/failed`，有输出即封口提交；外部输入的 `completion_state` 与 `seal_reason` 是候选，不是另一套现行结局字段。

路由指令表达本次交互的处理意图，不成为另一份正文或授权容器。话题不预建，root API 的“为一轮确定话题”直接返回话题标识，没有 prepare 结果模型或 handle（第 10 节）；接纳结果未知或已经接纳时，须先定位原结果，不能改变既有路由。

### 2.3 TurnEvent：复用并扩展事件表达

```text
TurnEvent 现有字段
  kind: user_message | assistant_message | thought | subagent_message | tool_call
  sequence / content
  tool_call：action_id、verb 或 tool_name（互斥）、targets、tool_args
  tool_call 的结局：status / result、produced_aliases

外部输入的候选扩展
  kind: system_message | external_summary | unknown
  occurred_at / received_at
  source_metadata（来源、外部 event id、外部 call 关联、connector provenance）
  多模态引用
```

工具调用相关字段已经实现：调用必带 `action_id`，`verb` 与 `tool_name` 互斥，结局记在同一个调用事件上，没有结局表示结局没有记录（[数据模型](../architecture/data-model.md#49-交互记录turnevent-与-interactionpayload)）。上面只有第二部分是候选设计。

候选设计补齐 system_message、external_summary、unknown、事件时间、外部 event/call 关联和 provenance 的表达，不另建等价 Part 模型。来源字段采用独立字段还是受约束 metadata 待定；外部 event ID 可选，不能为了统一强造新的全局 part_id。来源未知须可显式表达。

未知或缺少关联的工具事件必须保留在共同 TurnEvent 序列中，不因无法配对调用与结局而丢弃或伪造关系。source、occurred_at、received_at、外部 event id 和 connector provenance 是来源元数据，不是 Actor 认证凭证。

工具相关字段只适用于相应 kind，schema 应按类型约束，避免所有字段随意可空。外部 call ID 用于关联原始工具调用，不要求等于 Alice action_id；映射时保留来源，缺少关联时不能仅凭相邻位置补全。delta/chunk 在开放段合并为消息或显式分片后才投影，不把每个 token 伪装成独立动作。附件沿用 ref、revision 与实际使用快照，TurnEvent 按需补齐多模态引用表达，不把图片或文件强制塞进文本。`thought` 只保留调用方实际提供的信息，不要求外部 harness 暴露内部推理。

## 3. Session 与 Topic 的边界

| 对象 | 负责 | 不负责 |
|:---|:---|:---|
| ConversationSession | 对外连续性、历史顺序、open/paused/closed、展示索引 | 记忆生成的相关性、Topic 摘要及 CPU 派生工作视图的预算算法 |
| InteractionPayload | 一次封口交互、TurnEvent 顺序、来源、提交幂等 | 直接决定最终 Memory 或强制绑定一个用户 Topic |
| Topic | Patchouli 内的记忆材料分桶、blocks、summary、working set | 外部 session 生命周期和全量对话展示 |
| LogicalBlock | 交互转换后的 Patchouli 内容投影 | 保存整个 Session 或代替原始来源 |
| TopicData | Topic 内容事实、blocks、summary、bindings 的不可变快照 | 保存执行器 prompt history |

一个 Session 可以跨多个 Topic；首版一次交互只应用一个 primary Topic，避免拆分一次工具动作并重复记忆生成。跨 Topic 拆分交互属于后续算法决定，不在本文范围。Topic 的资源归属是 Workspace；Session→Topic 只保存路由关联，不新增 Topic 的 Session owner。路由候选是整个 workspace 的 Topic（0.1），跨 Session 的路由不合并 Session 历史。

`LogicalBlock` 可以保存 `session_id`、`interaction_id` 和 source provenance 引用，但不复制整个 Session。Session 保留提交时的内容，Topic 的 TurnRecord 与 LogicalBlock 是记忆侧投影；Topic 折叠、摘要或淘汰不回写 Session 历史。既有 blocks、summary、bindings 和 TopicData 快照保持 Patchouli 领域语义；新增关联优先使用兼容字段或现有应用记录，不为此重写整个 Topic 存储模型。

## 4. Topic 路由指令与交接

话题不预建：第一次写入或交互应用时建立，任何路径都不补偿；建立后没有接收交互的话题停下超过沉淀阈值时直接退役，或在池满时退役（第 10 节）。本节定义路由、提交、应用与资料交接，包含没有 Session 的主动意图场景，不意味着 Topic 属于 Session。公开方法、总线路由映射、operation 绑定和领域实现应在同一处定义；生成侧只引用本节，不另行决定路由关联的有效性或来源绑定。

首版指令（候选）采用窄判别模型：`auto`、`continue(topic_id)`、`explicit(topic_id)` 和 `new`。交互路由处理形成稳定的 interaction_id → topic_id 关联，优先由现有应用记录持有，再通过授权的公开结果投影观察：

```text
InteractionSubmitResult / 应用结果查询投影（扩展既有结果语义）
  interaction_id / work_id
  admission / application status
  topic_id（路由确定后可用）
  optional route_reason / router_revision / confidence / manual_override
```

上述为候选的结果语义与内部指令，不意味着已经提供同名外部接口。提交队列的有界内部工作记录不能直接充当公共历史或查询 API；还需补齐结果可见范围、保留期、过期行为和 A1 授权。`InteractionSubmitResult` 是初次调用的不可变返回值，不会在后台自动变为 applied；后续查询读取新的结果投影，不暴露内部 queue 或 work record。不另设 TopicAssignment 实体或与 application status 竞争的状态机。

`auto` 在允许的候选中判断；`continue` 将上次 Topic 作为提示，允许明确的改路由；`explicit` 在通过授权后固定目标；`new` 强制新建并排除自动并回相似 Topic。若 `continue` 与 `explicit` 在真实调用中没有不同语义，应合并枚举，不能留下两个同义分支。confidence 与 router_revision 只有路由器实际提供时才返回，不伪造精度和版本。

用户新建 Session 只创建会话容器；显式新 Topic 才约束内层路由。清空 Actor 的 prompt context 也不等同删除 Session 历史或 Topic，adapter 若还要强制新 Topic 须另发 `new`。这让手动选择和自动路由遵循显式优先级，避免“标题相似”覆盖用户选择。UI 按钮、会话分支与批量重路由另行规划。adapter 不把外部 session_id 伪装成 topic_id。

交互流程是：

```text
外部事件 -> Session 开放段中的 TurnEvent 序列 -> seal/flush
  -> 同一份 InteractionPayload 封口快照
  -> A1 access + 交互提交
  -> Patchouli 接纳交互（返回初始收据）、路由 Topic
  -> InteractionPayload -> TurnRecord / LogicalBlock / TopicData
  -> 通过结果查询观察实际 Topic 关联与应用状态
```

Session 服务保证输入去重和开放段的本地顺序；Patchouli application 必须独立验证交互提交幂等与顺序，不能完全相信 adapter。重试复用唯一的 interaction_id，服务端绑定 Workspace、Session、来源及规范化载荷，同键不同内容报 conflict。同一 Session 的同一 sequence 不得以另一个 ID 重复提交不同快照。Session 序列化与 Topic mutation 串行是两种约束：同 Session 跨 Topic 仍按交互顺序路由，不同 Session 路由到同一 Topic 后由 Patchouli 的 Topic 占用与 lease 保证安全，不为此引入第二套通用队列拓扑。

二者不使用跨领域大事务，而通过稳定 ID、路由关联、幂等和应用记录交接。accepted、routed、applied 是不同事实，即使共用一个结果结构也必须可区分。接纳时可能还没有最终 Topic，初始收据只报告 accepted；路由后可观察 topic_id，是否已应用仍须单独表达。已接纳且应用成功的重试必须定位同一结果，不能重新创建 Topic 或 block。结果已过保留窗口必须明确返回不可查询或已过期，不能据此认定从未提交再执行一次；幂等窗口与结果保留的协调待定（第 8 节）。

### 4.1 分开维护的状态与迟到事件

| 对象 | 状态由谁推进 | 与其他状态的关系 |
|:---|:---|:---|
| Session | workspace 会话服务；open/paused/closed | 不等于 Actor 正在运行或 Topic 已 settle |
| 开放段/封口快照 | adapter + Session accumulator；open→sealed，附 completion outcome | sealed 不等于已接纳或成功回复 |
| interaction 提交 | Patchouli application/queue；accepted→applied/failed | applied 不保证生成 Memory |
| Topic 路由关联 | Patchouli 路由/应用结果 | 对同一已应用交互稳定，不成为访问凭证；不新增独立状态机 |
| Memory/写入意图的物化 | Patchouli 领域任务；属于后续消费 | 独立于 Session 关闭和普通交互结果 |

同一结束信号重复到达返回已有封口或提交关联。未接纳时保留封口快照，响应丢失后用同一 ID 查询或受控重试。迟到消息或工具结果不能改写已封口的 InteractionPayload：首版显式拒绝或报告迟到，或由调用方以新的交互带来源引用补交；具体支持方式待定，不隐式 reopen 或重放整个交互。sequence 缺口、乱序和时间戳冲突需定义可接受窗口或拒绝策略，不承诺任意自动重排。

### 4.2 Topic 引用词汇与责任

`TopicManagementService.prepare_topic()` 返回 topic_id 字符串，经 root API 的“为一轮确定话题”使用；没有 prepare 结果模型、handle 注册、预留或过期机制（第 10 节）。下表统一引用的含义，不要求为每个概念新增一个服务或 Python 类。

| 对象 | 表达什么 | 不代表什么 |
|:---|:---|:---|
| Topic identity/ref | Workspace 中的具体 Topic 身份 | 访问许可、资料版本或某次交互已应用 |
| Topic 路由关联（assignment） | 交互与 Topic 的路由关联及可用路由来源信息，保存在应用记录或结果中；对已应用交互稳定 | 存在关联不单独证明 blocks 已应用 |
| interaction receipt | 交互接纳、应用或失败及其关联结果 | Topic 的整个生命周期，也不等于 Memory 已生成 |
| TopicData/资料快照 | 某一读取时点实际可用的 blocks、summary、bindings/provenance | 永远新鲜的内容、全部 Session 历史或完整原文档案；router_revision 也不是内容版本 |
| 主动意图的 Topic 资料绑定 | 该意图实际选择的 Topic 来源及资料快照关联 | Session 的可变 route cursor |

Patchouli 拥有路由关联与资料读取的语义；Topic 领域实现只有一处，API 目录和主动生成只是消费者，不产生新的运行时所有者。System 与 Actor adapter 持有引用并请求操作，不能凭持有引用绕过后续授权，也不能从引用字符串自行推断 Topic 身份。

### 4.3 提交与应用的生命周期

以下是行为约束和责任转移，不是一套固定状态枚举。

| 阶段/触发 | 必须保持的行为 | 后续责任 |
|:---|:---|:---|
| 尚未提交或接纳结果未知 | 调用方保留封口内容和提交 identity；响应丢失先查询或受控重试 | 结果未知不能推断为未接纳；不悄悄换 Topic |
| 交互提交的校验与接纳 | 校验 scope 与路由指令；需要路由时在提交方法内查重、路由 | 接纳后由 Patchouli 接管已承诺的领域工作；adapter 断连不撤销 |
| interaction apply | 使用已确认的路由关联应用 blocks；重试复用同一交互结果 | 路由结果与 applied 分别观察；应用成功固定关联，不重复创建 Topic 或 block |
| 调用方失败或放弃 | 不补偿、不清理：已经应用的交互与写入的结果各自成立 | 建立后没有接收交互的话题停下超过沉淀阈值时直接退役，或在池满时退役 |

管理员主动 settle 或 evict 是另一种操作语义，不代替提交路径上的任何补偿。附件 lease 的持有与释放按既有契约处理，与 Topic 的回收互不依赖。

### 4.4 向主动意图交接资料

Topic 身份、路由关联和资料快照分别交接。主动意图不要求先有 Session 或交互；若调用方明确要求纳入某次交互，先通过授权结果查询确认该交互 applied，并取得其实际 topic_id，再提交意图。只收到 accepted 或路由结果不能证明该交互内容已进入 Topic。两条提交不合并事务，也不建立等待未来交互的全局屏障。

| 输入情形 | 共同处理要求 |
|:---|:---|
| 明确指定已有 Topic | 校验存在、Workspace 归属和当前读取权限，再交付真实资料；失败不当作空资料 |
| Topic 合法但 blocks 为空 | 返回成功的空资料；不把内容为空视为意图无效或读取失败 |
| 来源为已应用交互 | 按实际应用结果中的 Topic 关联选择，不使用当前 Session cursor 重选 |
| 未提供 Topic/交互 | 意图独立生成；写入需要话题，由“为一轮确定话题”在派发前确定，需要新话题时在第一次写入时建立，派发时为空（第 10 节） |

资料读取与绑定须固定本次任务的来源，后续 Session 路由变化不能让同一任务重试悄悄切换 Topic。若处理需要继续访问活 Topic，应明确保护何时取得和释放；若已取得独立资料快照，则按快照保留约定处理。选择哪种绑定、取样时点、Topic 内容变化后的重试行为以及可用版本证据待定（第 8 节），不假定 Topic 已有可持久化 revision 或通用 pin 能力。

本节涉及资料身份、全部可用内容的读取与绑定、保留与释放；主动写入生成时的材料见 v0.8 [写入链路 Idea](./memory-flow-write-path.md)第 2.2 节，写入意图见[读取视图与写入意图](../workspace/read-view-and-write-intents.md)。资料为空、原文已经折叠而不可恢复、读取失败和无权读取必须区分。

### 4.5 实现复用方向（候选）

复用 `TopicManagementService`、`InteractionSubmissionService` 与现有全局路由地址，对尚未公开的方法补齐明确地址和 operation；不另包一层 provider 或 Workspace service。演进后的 `InteractionPayload` 是公开提交输入，`InteractionSubmission` 仍是内部 queue envelope；二者沿用同一个 interaction_id。apply 与路由关联观察应与提交入口同时可用。

## 5. 结构化事件的构建时机

交互内容只以 `turn_events` 表达，一次工具调用与它的结局是同一个事件，没有可靠关联的调用不经归约器配对（[数据模型](../architecture/data-model.md#49-交互记录turnevent-与-interactionpayload)）。

外部 adapter 在接收阶段把实际收到的事件归一化为 TurnEvent，放入开放段（候选）；共同归一化规则复用，来源特有的协议解释留在 adapter。交互封口可以由完成回调、assistant 结束、工具结果完整、显式 flush、shutdown 或取消信号触发；缺少结束信号不能推断成功完成，partial/cancelled 必须保留其状态。

以阶段而非定时器确定构建时机：接收时建立最小事件，封口时冻结内容，Patchouli 在应用前建立领域投影。已有足够结构的直接复用，只有依赖完整交互的才延后处理：

1. 接收时保存有序 `TurnEvent` 及来源；未知 kind、缺失说话者或工具关联按 schema 显式表达，不等完整 Action 才记录。
2. 封口时冻结 `InteractionPayload`、interaction_id、Session sequence 与结束状态；不另造 Segment 或 execution_projection 正文。
3. 调用与结局的配对在接收时完成：调用事件写出后，结局到达时按调用标识补到同一个事件上；结局缺失的调用照常保留，表示结局没有记录，不虚构成功结果，也不凭相邻位置把无法关联的结果配给某个调用。
4. 形成 `TurnRecord` 和 `LogicalBlock` 时保留 interaction/event provenance；生成材料等派生视图由事件渲染，不反向覆盖原事件。结局缺失的调用不等于可以从输入历史中删除。

外部客户端只须按外部接入协议（见[外部 Actor 的接入登记与运行时访问](./external-actor-registration-and-runtime-access.md)）提交它实际拥有的消息与工具事件；内部 adapter 负责转换为 TurnEvent，不要求客户端实例化 Python 类型或构造 `TurnRecord`。Alice 直接保留既有 TurnEvent 的结构与 action 关联，提交相同的 InteractionPayload，Patchouli 使用同一领域投影规则。`TurnRecord.identity` 不承担 session 身份；资源授权只使用 A1 的 access context。

现有来源各自给出可靠关联：Alice 的 MTP 调用带自己的 action_id，Claude Code 与 dsh 的连接器按工具调用标识把结果补到调用上，找不到调用的结果跳过（[Import Bus](../system/import-bus.md)第 8.3、9.3 节）。外部输入沿用这一规则：只接受显式关联或可证明的来源映射，无法关联时按上一条保留，不引入按相邻位置推断的归约。

目标上 Session 保存封口 InteractionPayload，TurnRecord 是 Topic 内的结构化内容投影；TurnRecord 现在定义为单轮内容记录、事件集是唯一的内容事实，迁移时要明确它在 Topic 领域内的含义，不让两份正文相互回写。事件集的不变量要求以序号 0 的用户消息开头，外部输入若要接收无 user 的合法事件记录，需要先修订这一不变量；保留记录不意味着必须生成记忆。未知 kind 或多模态 ref 无法完整进入 Topic 投影时，保留在共同记录及 provenance 中，并在结果中表达投影缺口，不能冒充无损应用。

## 6. InteractionPayload、队列 envelope 与主动意图

`InteractionPayload` 只有 `outcome`、`turn_events`、`model_used` 与 `used_attachments`：用户消息与回复正文以只读属性从事件读取，不保存、不序列化。迁移目标（候选）：

- 演进 `InteractionPayload` 为外部输入的正式共同输入，补齐 Session、顺序、封口与来源语义；`turn_events` 已是唯一的内容来源。
- `InteractionSubmission` 继续是进入 Patchouli submission lane 的内部 envelope，携带可信 scope、同一 interaction_id 和 ordering；queue/work 身份不外化为会话身份。
- 主动 WRITE/UPDATE 保持独立提交责任，不进入 interaction 输入，其公共模型 `MemoryIntent` 见[公开路由与事件](../contracts/routes-and-events.md)。

| 现有字段 | 目标承接 | 兼容要求 |
|:---|:---|:---|
| outcome | 三种结局已实现；外部输入的 `completion_state` 与 `seal_reason` 是候选 | 不另起一套与 `outcome` 并行的现行结局字段 |
| turn_events | 唯一有序事件正文（已是现状）；扩展来源与不完整事件表达 | Alice 与连接器的调用关联保留；外部缺失允许表达；不再增加 parts/execution_projection |
| used_attachments | 已实际编译使用的 ref/revision 快照与 provenance | retry 原样重用；应用成功后才建立 binding，不回查 UI 或上传列表 |
| model_used | 来源展示信息/可选执行元数据 | 外部未提供时未知，不要求绑定本地模型注册表 |

`InteractionSubmission` codec 有 schema_version 和严格字段校验，引入新载荷时必须显式升级版本并验证规范化摘要与往返一致性。旧版本按受信兼容解码路径转换，不能用一组可选字段同时猜测新旧语义。同一次封口保留唯一 interaction_id；旧 envelope 已有 ID 时原样沿用，缺失时在首次封口或受信转换边界生成并保存，重试不重新生成。随机 turn_id 与 block_id 不作为提交幂等依据。

一次交互可以没有主动意图；一个主动意图可以没有交互或 Session。交互结果向主动意图交接 Topic 资料的条件见第 4.4 节。独立主动任务的提交不与交互入口合并。

## 7. 外部上下文与输入资料的交接边界

外部 harness 自主管理 prompt history；adapter 接收它实际提供的交互事件，不因接收记录就接管执行上下文。外部 summary/checkpoint 需携带可识别的来源，不能替代原始交互或直接覆盖 Topic 摘要；详细压缩关系由[短期记忆](./short-term-memory-and-context-compaction.md#82-两边的关系与不变量)维护。

本文只维护 Session、interaction 与 Topic 的关联及共同输入的来源表达。完整资料保存和 Artifact 来源链见 [Raw Evidence](./PatchouliPageFoldingRawEvidenceDesign.md)，未结束 turn 的摄入见[长 Turn 专项](./long-running-agent-intra-turn-context-folding.md)。

## 8. 开放问题

本文的剩余事项属于外部输入、路由与资料交接，是候选方向；旧 Passive Ingress 会话键的映射与 Workspace 分区已由 Import Bus 实现（[Import Bus](../system/import-bus.md)第 4.3 节），Alice 的会话压缩由[短期记忆](./short-term-memory-and-context-compaction.md)维护。

### 8.1 设计问题的归属

外部输入立项时，根据第 2 至 6 节收敛事件表达、幂等、顺序、路由结果与资料交接。完整资料、保留和来源 Artifact 的跨环节约束由 Raw Evidence 维护，压缩预算、Relay 与 KV cache 议题由短期记忆 Idea 维护。

### 8.2 外部会话的后续事项

各留一行，需要时再展开；括号内为相关章节与归属方向。

- TurnEvent 新增 kind、来源与时间、外部调用关联和多模态引用；外部来源的调用关联规则（第 2.3、5 节；controller 模式）。
- `InteractionPayload` 增加会话与顺序字段，`InteractionSubmission` codec 的新旧兼容样例；interaction_id 与 work_id、外部 ID、turn/block ID 的映射（第 2.2、6 节；controller 模式）。
- 封口信号、迟到、重复与 sequence 缺口；开放段容量与无 user 事件的接收（第 4.1、5 节；controller 模式）。
- Topic 资料与结果：资料来源与保留、面向外部的 accepted/routed/applied 结果查询与保留（第 4 节）。
- 路由指令的优先级与显式 attach（第 4 节；默认候选范围为整个 workspace）。
- Gateway 的会话提示（恢复已有会话或新开会话）的判断方式、提议的呈现与确认（0.1、0.2；后续版本，路由上下文的矛盾见 0.2）；会话 ID 无需为此进入 Patchouli。

## 9. 设计需满足的行为约束

以下是候选设计必须满足的行为，不是测试计划：

- 同 Session 跨多个 Topic 仍按序提交；显式 new/continue/attach 不合并历史；Topic 折叠不改写仍在保留范围内的 Session 原事件；完整资料集合与删除语义按 Raw Evidence 收敛。
- 两个相邻工具调用后的无关联结果不会被强配到最后一个动作；孤立结果、无 user 消息、未知事件、多模态引用和部分失败交互可保留并报告投影缺口。
- 重复 seal 与提交沿用同一 interaction_id；响应丢失后重试不新建 Topic 或 block；同 ID 不同内容、同 Session sequence 冲突及迟到事件按契约处理。
- 同名外部 conversation 在不同 source 或 Workspace 中隔离；Session 读取和结果查询均校验权限，猜中 interaction_id 或 work_id 不授予访问权。
- 初次 accepted 与后续 routed/applied 查询可区分；结果过期不当作未提交；Session、结果及幂等记录的保留边界可解释。
- 新旧 codec 的字段、摘要和往返转换稳定。

## 10. 交互提交收束与话题按需创建

已经实现，当前事实见[任务进程](../workspace/task-process.md)与 [Import Bus](../system/import-bus.md)。

### 10.1 背景

- **预建话题把提交绑在准备上**：结算要补偿准备阶段新建的空话题，并为准备阶段检索到的记忆记录命中；Patchouli 为此要持有结算续行、对实际所在话题的核对与清理路由。
- **交互提交曾有两个互不相通的入口**：Alice 一侧由 Patchouli 内部入队并等待，plugin 一侧的查重、等待与失败归类都在 Import Bus 里。
- **写入在派发时需要话题**：WRITE/UPDATE 在执行期间派发给 Patchouli 生成，生成按话题取材料（折叠摘要与最近几个块；话题不存在时按空材料），并按话题登记主动写入生成的记忆；workspace 一侧的意图登记挂在这一轮上（`process_id`）。
- **清理会误删**：Agent 先写入、CPU 随后没有输出时，按“是否接收过交互”判空的清理会把写入过的话题当空话题删除。

### 10.2 设计

1. **话题按需建立**：不预建话题；第一次写入时建立，否则推迟到接受交互时建立。
2. **不做补偿**：发生写入之后话题就不再为空，任何路径都不补偿或清理话题；建立后没有接收交互的话题停下超过沉淀阈值时直接退役，或在池满时退役。
3. **检索命中在每次检索时记录**，不等到交互提交。
4. **准备与提交拆成单独的能力层调用**，不与 Patchouli 强耦合。
5. **提交逻辑只在一处**：交互提交队列收束提交接口、完整封装提交逻辑，入口不各自管理。
6. **Alice 一侧与 Import Bus 共用 root API 的方法**，这是统一交互提交的核心。
7. **写入先授权、再建立话题**，MCP 同样：被拒的写入不触发路由、建立话题或 LRU 结算。
8. **交互提交与写入时确定话题收束为 root API 的两个方法**，入口各授权一次：
   - “提交交互”接收封口后的交互与话题来源（已确定的话题、Gateway 的新话题决定，或一轮待路由的导入）。需要路由时先在方法内查重，未接纳或应用失败的才做 PASSIVE_MEMORY 路由，然后交给 Patchouli 的“提交并等待应用”，返回结构化结局；
   - “为一轮确定话题”：按 Gateway 的决定，或在方法内查重、路由之后，确认或建立话题；
   - 查进度、路由与等待不是单独的 root 方法；接纳失败、应用失败与超时的归类只在 Patchouli 一侧做一次；
   - 写入的顺序是：授权 → UPDATE 的基础冷读 → 确定话题 → 同步段登记 → 派发；请求级提交的派发话题参数是一个确定话题的回调。
9. **主线程在 PREPARE 开始时派生**：Profile 解析之后派生主线程，预检索按 Gateway 的检索计划经操作入口以主线程提交，与 Alice 的 SEARCH 走同一条路；请求级提交只给不建进程的入口。主线程从 PREPARE 存在到 Actor 阶段结束，停止被接受后预检索随线程结束被拒绝。

### 10.3 与短期记忆形态的关系

写入需要话题，所以由第一次写入建立话题，写入派发即生成的时点不变，话题何时、由谁建立收在一处。不采用的做法：写入挂在这一轮的交互标识上，交互应用到哪个话题，写入就归到哪个话题，写入从不建立话题；它等于先采用按会话时间线切段的短期记忆形态，与已归档的[统一的交互记忆 Idea](../archive/ideas/unified-interaction-memory.md)第 3.3 节的切点设想同向。

短期记忆的单位与形态（话题按内容跨会话归并，还是按会话时间线切段）尚未确定，要在 v1.0 发布前确定；本设计不替它决定。形态改变时，需要重做的是话题的建立时点，以及 Import Bus 在写入时确定话题的路径。
