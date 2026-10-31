---
title: Patchouli Perception
status: current
owner: patchouli
scope: structured-ingestion-and-short-term-topics
code_paths:
  - src/hivememory/engines/perception/
  - src/hivememory/infrastructure/llm/base.py
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/patchouli/services/topic_working_set.py
  - src/hivememory/patchouli/memory_library/stores.py
  - src/hivememory/patchouli/control/interaction_submission.py
  - src/hivememory/patchouli/application/interaction_submission_service.py
  - src/hivememory/core/protocol/sealing.py
  - src/hivememory/core/models/interaction.py
  - src/hivememory/core/models/topic.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/system/import-bus.md
related_docs:
  - docs/workspace/README.md
last_reviewed: 2026-10-31
---

# 感知与短期话题

Perception 负责把一轮已经发生的交互转化为 Patchouli 可以持续维护的短期事实。它不猜测用户下一步要做什么，也不直接决定某段文字应成为哪条正式记忆；它保存结构化 turn、维护相互隔离的话题 buffer，并在容量、空闲、驱逐、手动操作或停机边界上形成可交给 Generation 的材料。

“感知”在这里更接近记忆管理单元，而不是第二个 Gateway。Gateway 先解释入口，Alice 先完成 run，Perception 随后观察结果。这个先后关系防止记忆系统为了整理历史而重新取得入口决策或 Agent 执行权。

## 1. 模型分层

### 1.1 内容事实：TurnRecord

一轮交互的内容真相由 `core.models.TurnRecord` 表达，只有四项：本轮结局 `outcome`、`turn_id`、产出本轮的 `ActorIdentity`（不含 session），以及以序号 0 的用户消息开头的有序 `TurnEvent` 集。事件集是唯一的内容事实：用户消息取序号 0 的事件正文，回复正文是全部助手正文按序号以空行拼接，二者都在读取时从事件得到，不在记录中另存一份。

`TurnEvent` 有用户消息、助手正文、思考与工具调用四种；一次工具调用与它的结局是同一个事件，字段与一轮的不变量见[数据模型](../architecture/data-model.md)。事件集之外不保存由它归并的动作或轨迹：同一事实存两份时，两份会各自漂移，读取方也会从不同来源得到不一致的解读。Perception 不会从一段拼接文本逆向解析工具事实。

`TurnRecord.outcome` 保存 completed/cancelled/failed（默认 completed），从 `InteractionPayload` 原样转入。中断和失败只要有输出，也沿同一提交与幂等应用链形成 block；结局不是成功知识的标志，生成转写会注明不完整。会话上下文属于 workspace，Topic 仅作为记忆材料，不反向填充 Alice 的历史。

### 1.2 感知容器：LogicalBlock

`LogicalBlock` 在 TurnRecord 外只增加感知层元数据：block id、created time 与估算 token（`total_tokens`，第 6 节）。用户消息与回复正文是从事件读取的只读属性。它是不可变模型，表明同一轮内容事实进入短期话题后的感知视图；Gateway 的入口判断（意图、改写后的查询）不进入块。

可以简写为：

```text
LogicalBlock = TurnRecord + perception metadata
```

### 1.3 话题：原始数据与工作视图

Perception 以 `TopicData` 快照维护话题。快照只存两份原始数据：`interactions` 按接收顺序保存话题接收到的全部交互（`LogicalBlock`），只追加，话题结束前不移除；`folds` 按顺序保存每次折叠的 `TopicFold`，即这一次被折叠的块标识、折叠后的状态摘要与折叠时间。快照另含展示 title/summary、资产绑定与最近模型名。

工作视图由两份原始数据派生，不单独存储：`blocks` 是最后一次折叠之后的交互记录，`state_summary` 是最后一次折叠的摘要（没有折叠时为空串），`total_tokens` 是 `blocks` 的 token 之和。读取话题内容的消费方（折叠选择、Gateway 查询分析、主动生成与话题快照）读的都是工作视图。工作视图不另存一份，是为了不让同一事实有两份表达：分开存储的副本需要在每次写入与折叠时维持“工作视图是交互记录的后缀”，派生则不会失去一致。构造时校验全部折叠记录依次拼接，必须等于交互记录开头同样数量的块标识；`TopicData` 拒绝未声明的字段，以 `blocks=` 等工作视图名构造时显式失败。

话题只经 core 话题模型的两个写函数写入：`merge_interaction_into_topic` 只向交互记录追加一轮（以及首次资产绑定），`fold_topic` 只追加一条折叠记录。两者都以完整构造（`TopicData.model_validate`）产生新快照：Pydantic 的 `model_copy(update=...)` 不运行校验器，用它写回会绕过构造校验。

快照只承载内容事实：执行占用不建模为记录字段（原 `BufferState` 状态机与 `last_accessed_at` 访问时间已删除），跨 await 的占用权由 `TopicWorkingSet` 的 lease 表达，访问顺序由 WorkingSet 的 LRU 索引维护。短期 adapter 直接存储 frozen `TopicData`，Store/Port 与 Perception 之间只交接不可变快照；生命周期写回由 PerceptionFamiliar 编排。

## 2. 结构化摄入

`InteractionPayload` 是主动链路与 [Import Bus](../system/import-bus.md) 共享的协议。交互内容只以 `turn_events` 表达，其余字段是本轮结局 `outcome`、执行者使用的模型 `model_used` 与实际使用的附件引用 `used_attachments`（Import Bus 的提交为空），它们不是对事件的解析；用户消息与回复正文是从事件读取的只读属性，不保存、不序列化。载荷构造时校验一轮事件集的不变量，违反时显式失败，不修补。两条链路都由 `core.protocol.sealing` 中同一个封口函数 `seal_interaction` 组装它，交给 Patchouli 的载荷因此同形；封口只组装、不解析事件，序号 0 的用户消息事件由提交方写出（Alice 执行循环、Import Bus 服务端），执行方只交出回复正文、没有写助手正文事件时，封口在末尾补一条助手正文事件。它不重复保存本轮 Workspace 身份；两条链路在资源 owner 边界拆分 `IdentityScope`，进入提交队列时由外层 `InteractionSubmission.belong_to` 与 `from_actor` 独立携带归属和本轮发起者。

`InteractionSubmission` 另外携带可选的 `new_topic_title` 与 `new_topic_summary`：只在 `requested_topic_id` 为 `NEW_TOPIC` 时允许非空，已有话题的标题与摘要由话题自身持有，提交不能借此改写。`InteractionSubmissionCodec` 当前为 v6，只注册当前布局：缺失字段、额外字段、旧 `identity_scope` 布局及嵌套对象不能规范往返的载荷均被拒绝。队列使用 `InMemoryWorkStore`，提交快照与幂等收据只驻留于当前进程，不提供旧版本队列载荷迁移或重启恢复。

队列按 `interaction_id` 幂等：等待中或已成功的交互以相同载荷重交时复用原收据，载荷不同明确冲突。为此保留的旁路索引只存编码载荷的摘要与归属，冲突比较不需要留住整份载荷。应用失败（工作项已到终态但没有成功）的交互允许以同一标识重新提交，载荷可以不同，例如原落点已被驱逐、提交方重新路由后再交；新提交以递增的尝试号作为工作项标识入队。这条例外只放开已经失败的交互：等待中的交互重交会被判为重复或冲突，已成功的交互不会被第二份载荷重复应用。同一轮也不会被写入两次：把 block 写入话题是应用中最后一个可能失败的步骤，应用要么失败且没有写入，要么写入并成功，因此重新提交的只会是没有写入的交互，感知一侧不另设应用记录。在写入之后加入可能失败的步骤会破坏这一点。

公开路由 `patchouli.public.interaction.status` 把队列中的工作状态归为四种进度：`pending`、`succeeded`（附真实落点，`NEW_TOPIC` 已解析为实际话题）、`failed` 与 `unknown`（旁路索引仍在、工作记录已超出终态保留窗口，接纳过但结局不再可知），未被接纳时为空。`patchouli.public.interaction.wait` 在交互仍为 `pending` 时等待到终态或超时，超时时仍报告 `pending`，队列停止时返回停止那一刻的进度。两者都先校验归属，同名交互属于另一个 Workspace 时以 `WorkspaceMismatchError` 拒绝，不把其他 Workspace 的接纳当作重复。查询只覆盖队列的进程内保留窗口，窗口外的交互视为未接纳。公开提交路由在队列容量已满或已停止时以 `InteractionAdmissionUnavailableError` 拒绝，表示这次提交没有被接管，调用方可用同一标识重试。

当前布局、幂等拒绝与失败后重交的回归入口为 [InteractionSubmission 测试](../../tests/unit/patchouli/control/test_interaction_submission.py)，进度查询与等待见[提交服务测试](../../tests/unit/patchouli/application/test_submission_services.py)；Import Bus 经过真实队列与感知存储的身份交接、查重与新话题标题见 [Import Bus 集成测试](../../tests/integration/system/test_import_bus_submission.py)。

交互应用成功后，Perception 才会把用户明确使用的 `(asset_id, asset_ref)` 记录为 Topic 的 `TopicAssetBinding`。上传、候选列表或 UI 选择本身不会进入 Topic 事实。由此，感知层只接收已完成身份与资源前置校验的交接输入，不拥有 `WorkspaceAssetStore` 的生命周期。

```text
InteractionPayload（构造时已校验一轮的不变量）
  -> TurnRecord(outcome, identity, turn_events)
  -> 渲染整轮材料并估算 token（engines.perception.material）
  -> LogicalBlock
  -> merge_interaction_into_topic：追加到交互记录
  -> ShortTermMemoryStore.put(新快照)   应用到此完成
  -> needs_compact：提交队列排入折叠项 -> compact_topic（Page Folding，第 5 节）
```

主动流程由任务进程封口 payload、Patchouli finalize 原样提交；Import Bus 以同一封口函数封口导入的整轮记录，经公开提交路由提交。两者先进入同一 `InteractionSubmissionQueue`，再由 `PerceptionFamiliar.apply_interaction()` 编排 `MemoryPerceptionEngine.build_block()`（纯算法）、占用检查与 Store 写入，不存在导入专用的扁平文本主链，也不存在中间感知层。应用在写入后即完成，不执行折叠；话题需要折叠时，由提交队列在这次应用之后另行排入折叠项。

结构化摄入的边界来自外部 harness 的真实事件形态：Import Bus 接收的一轮整轮记录中，用户消息由服务端写为序号 0 的事件，其后是助手正文、思考与工具调用三类事件。两种来源写出的事件同形，来源之间真实存在的差异只体现为取值：调用 HiveMemory 能力（Alice 的 MTP 指令、外部 harness 经 HiveMemory MCP 工具的检索、读取、写入与更新）带动词与作用对象，外部工具带工具名，下游不判断来源。思考、工具调用与工具返回必须以 `TurnEvent` 保留，不能为了生成一段看似完整的 transcript 而提前丢弃。

对事件集的解读只在感知层进行一次：`engines/perception/material.py` 把一轮事件渲染为感知层唯一的材料文本，记忆生成视图（[记忆生成](./generation.md#21-材料与渲染)）、LLM 接力摘要与逻辑块的 token 统计都使用它，按同一个工具结果上限得到同样的材料。入口层与封口不另行构造第二套摘要，否则历史重放、主动生成与导入归档会拥有互相漂移的事实来源。

目标话题属于提交前的 Gateway 落位决定，而不是应用时重新计算的感知结果。任务进程在 Gateway 阶段取得落位，并在 prepare 时建立新话题；Import Bus 在提交前以 `PASSIVE_MEMORY` 为需要提交的每一轮调用 Gateway，把落位作为 `requested_topic_id` 提交。目标为 `NEW_TOPIC` 时，Perception 以提交携带的 `new_topic_title`/`new_topic_summary` 建立话题，缺省时使用默认标题“新建话题”与空摘要；后续轮次的落位依赖候选话题的标题与摘要，因此 Import Bus 新建话题时带上 Gateway 给出的标题与摘要。外部 harness 经 MCP 写入时，Import Bus 在写入时经 prepare 预建这一轮的话题（以同样的标题与摘要），这一轮回流时以该话题的真实 id 提交（[Import Bus](../system/import-bus.md#47-写入时落位)）。应用失败的交互没有写入任何话题，重新提交时按提交方这一次交来的落位应用。Import Bus 拥有外部会话的串行、查重与确认，Gateway 只负责入口判断，两者都不持有话题内容。

形成 settlement payload 时，话题接收到的全部交互都进入结算，不在入口筛选。入口只看到用户消息与已有话题上下文，看不到这一轮的回答、工具调用与结果，“请提交”“继续”这类短消息背后常是整轮实现；按入口判断筛选会把这些轮次整轮丢出生成，交互 Artifact 也不再保存它们。一轮是否有长期价值，由生成阶段在完整材料上判断；Gateway 的决定只提供话题落位，不携带价值信号。

## 3. 话题准备与 LRU

`prepare_topic()` 在 Agent run 之前确保目标话题存在：

- `NEW_TOPIC` 创建 UUID、title 和 summary；
- 已存在话题继续使用并刷新访问顺序；
- 指定话题不存在时拒绝请求，不把跨 Workspace 的全局 ID 误当作新话题。

当需要创建新话题且活跃池达到 `max_resident_topics` 时，`TopicWorkingSet.needs_eviction()` 判定容量、`select_lru_candidate()` 从驻留索引（OrderedDict，O(k)）选出最久未访问且未被占用（lease）的话题，按 `LRU_EVICTION` 结算并提交后台 generation task，然后从 Store 与驻留集合移除。已有话题命中不会为了“置顶”而误驱逐另一话题；候选被占用时改选其他话题，全部被占用时抛出 `TopicBusyError` 留待重试。容量与候选以 WorkingSet 驻留索引为准，Store 只表达内容事实。

Prepare 与 ingest 都会再次确保话题存在，以覆盖 prepare 后异常清理或其他边界变化。若 prepare 已创建新话题但 Agent run 失败，Patchouli cleanup 只在它仍为空时删除，避免误删已经摄入内容的话题。

## 4. 触发原因与具名用例

settle / compact / evict 是三个概念独立的具名用例，由 `PerceptionFamiliar` 编排，不再通过决策矩阵把触发原因解释成动作组合。`TriggerReason` 只作为 `TopicMaterializeTask` 的 provenance 标签（记录“这次结算为什么发生”），不驱动分支：

| 原因（provenance） | 触达的用例 | 当前结果 |
|:---|:---|:---|
| `TOKEN_OVERFLOW` | compact | 折叠工作视图的旧前缀、追加折叠记录，话题继续存活 |
| `IDLE_TIMEOUT` | settle | 维护扫描中提交候选材料并移出活跃池 |
| `LRU_EVICTION` | settle | 为新话题腾出位置 |
| `SHUTDOWN` | settle | 停机前结算并驱逐（含真正空 Topic） |
| `MANUAL_SETTLE` | settle | 手动结算为记忆资产并结束 Topic 生命周期 |
| `MANUAL_COMPACT` | compact | 历史触发原因；当前无公开手动 compact 入口，compact 仅在 token 溢出时自动触发 |
| `MANUAL_DELETE` | evict | 历史触发原因；手动删除通过 `evict_topic` 用例表达，不写记忆 |

三个用例的职责边界：

- **settle**（`_settle_topic` 统一时序，所有来源共用）：获取 lease → 冻结全部交互记录与折叠记录，构建 `TopicMaterializeTask` → 锁外 generation admission → 删除话题并移除驻留。admission 失败时话题内容原样保留可重试，无需 abort——记录从未被改动。
- **compact**（`compact_topic`，是否需要由 `needs_compact` 判断）：取得 lease 后，把工作视图中保留后缀之前的旧 blocks 交给 RelayController 生成新的状态摘要，再经 `fold_topic` 追加一条折叠记录写回。被折叠的块仍留在交互记录中，工作视图向后推进，token 统计随之只计入保留的块。话题已被结算或驱逐时直接结束，正被占用时以 `TopicBusyError` 留待重试；摘要返回后才一次写回，中途失败或取消都不改动话题。
- **evict**（`evict_topic` / `discard_if_empty`）：从 Store 与驻留集合移除话题，不结算、不写记忆；话题被占用时报 `removed=False` 而不是阻塞。

历史上曾由 `TriggerManager`/`TRIGGER_PLANS` 决策矩阵与记录字段状态机（`BufferState`）承担这些流程，现已删除：占用权建模为 WorkingSet 的 lease，而不是记录状态，补偿校验随之消失。

`TopicMaterializeTask` 是 Perception 交给 Generation 的冻结快照：携带话题的全部交互记录与折叠记录，工作视图用与 `TopicData` 相同的 core 派生函数得到；此外只携带 `belong_to` 与已确认的 `asset_bindings`，不携带触发结算的 actor 或 `IdentityScope`。生成材料是工作视图（`state_summary` 与未折叠的 blocks），来源 InteractionArtifact 覆盖全部交互记录并带折叠记录（[记忆生成](./generation.md)、[Artifacts](./artifacts.md)）。Coordinator 为 manual、idle、LRU、shutdown 统一构造所属 Workspace 的 `system` 发起者，实际参与 Agent 从 turn identity 聚合为贡献者；结算查重只能读取本 Workspace 的 PUBLIC 记忆。进入队列后，后续处理使用冻结的交接事实，不能从当前进程状态重新推导 Workspace、读取权限或资产关系。

手动用例互不混杂：`manual_settle_topic` 只结算并在结算材料被可靠接纳后删除话题；token 溢出 compact 只压缩工作集，不结算、不驱逐（当前没有公开的手动 compact 入口，`MANUAL_COMPACT` 仅作为 provenance 保留）；`evict_topic` 只驱逐、不生成记忆。manual settle 的提交顺序固定为冻结 settlement payload → generation admission 成功 → 删除；admission 失败抛出受控错误且 Topic 的交互记录与折叠记录保持完整可重试。无任务（真正空 Topic）时 settle 仍按契约结束生命周期并返回成功，以 `generation_submitted` 表达是否建立后台任务。

### 内容判空语义

Topic 是否为空只看是否接收过交互：`has_content` 为交互记录非空，`is_empty` 为其否定。折叠不移除交互记录，有折叠摘要的话题一定有交互记录；话题列表（排除空话题时）、prepare 失败后的空话题清理与折叠判断都使用这一判断，接收过交互的话题不会被当作空话题删除。真正空 Topic 没有可 settle/compact 的内容，但 settle 仍会正常结束其生命周期。compact 不支持“总结后清空全部 blocks”的 `retain_count=0` 语义：所有 compact 配置与内部入口都要求 `retain_recent_blocks >= 1`。

这层返回值边界很重要：底层感知算法可以决定“应交出哪些材料”，但后台任务、事件和取消属于上层控制面。

## 5. Page Folding

当话题工作视图的 token 超过 `fold_token_threshold`（默认 32768）且有可折叠的旧前缀时，话题需要折叠。`fold_retain_recent_blocks`（默认 2，必须大于 0）定义工作视图中保留的最近 blocks 数；旧前缀与 previous summary 由 RelayController 合成为新的状态摘要，写回只追加一条折叠记录，工作视图从被折叠的块之后开始，保留后缀继续参与下一轮上下文。Relay 不总结保留后缀，避免 `state_summary + recent_blocks` 重复承载同一轮事实。由于 Relay 摘要生成的整个窗口内 `compact_topic` 持有该话题的 lease，其他调用无法同时取得占用权，写回不再需要旧的 expected_state 前缀补偿校验。

折叠是话题自身的维护，不属于交互应用。交互写入话题即完成提交；交互提交队列在这次应用之后检查 `needs_compact`，需要时排入一个折叠项。折叠项与该话题的提交共用排序键 `topic:{topic_id}`（`topic_ordering_key`），按键串行的 lane 因此在折叠执行期间挡住同一话题的后续提交，它们在队列中等待，等待不计入各自的处理时限；其他话题的提交照常执行。由此：

- 交互的确认（Import Bus 的应用等待、主动链路的 finalize）不包含折叠耗时；同一话题的下一轮要等折叠结束，Import Bus 等待超过 `import_bus.apply_wait_seconds` 时返回可重试的 503 `apply_pending`；
- 同一话题同时至多一个未结束的折叠项；队列已满或排入时出错，就放弃这一次，交互结果不受影响，下一次写入该话题时重新判断；
- 折叠失败只结束这个折叠项，话题保持原样；话题正被结算等用例占用时重试；
- LLM relay 的每次摘要调用以 `summary_timeout_seconds` 为时限（第 6 节），超时回退为规则摘要，使一次折叠的耗时有上限，并让话题回到阈值以下。

交互应用的时限（30 秒）由提交队列的 handler 只对应用本身施加；lane 不设统一时限，折叠不被它截断。

Page Folding 的设计目标是把 context window 视为工作集，而不是无限日志：旧页可以折叠为接力摘要，新的交互继续发生在同一话题。摘要是工作视图，不是原始证据：折叠不丢弃原文，被折叠的块留在交互记录中，话题结算时随全部记录冻结进来源 InteractionArtifact。

Page Folding 是 Patchouli 的内部 topic working-set compaction，主动链路与 Import Bus 共享同一语义。`state_summary` 与近期 blocks 服务于 Gateway 辅助分析、主动 WRITE/UPDATE 的生成背景与后续 settlement；Alice 的对话历史从 Session 重放。Import Bus 的响应只报告这一轮的导入结果，不返回检索记忆或折叠上下文。外部 harness 自行 compact 对话不会缩减 Patchouli 已摄入的 blocks，因此不能作为跳过内部 folding 的依据。

overflow 不执行 Settle：折叠只推进工作视图。生成材料仍是工作视图，被折叠的轮次只以摘要进入生成，原文在结算时进入来源 InteractionArtifact。当工作视图中的 blocks 数量不大于保留数时没有可折叠前缀，本轮 compact 会延后，因此阈值是工作集软水位线，而不是严格的模型 context 上限。

## 6. MemoryPerceptionEngine 与 RelayController

`MemoryPerceptionEngine` 是无状态的短期记忆摄入与 compact 算法引擎，职责包括：

- `build_block`: 把 `InteractionPayload` 转化为不可变 `LogicalBlock` 并估算 token。块的 `total_tokens` 是对整轮渲染材料的估算：用户消息、助手正文、思考、能力调用的一行摘要与外部工具步骤，只有外部工具结果按 `patchouli.generation.transcript.max_result_chars` 截断，折叠阈值因此直接反映生成实际看到的材料；能力调用读回的记忆不渲染，也不计入；
- `should_compact`: 判断话题工作视图的 token 是否超过折叠阈值；
- `select_blocks_to_fold`: 选择待折叠的旧 blocks 前缀（保留最近 N 个）；
- `generate_fold_summary`: 异步生成 Page Folding 摘要（委托给持有的 `RelayController`）。

Engine 持有 `RelayController`，封装了完整的 compact 算法能力。`PerceptionFamiliar` 通过 Engine 的 `generate_fold_summary()` 获取摘要，不再直接持有 Relay 组件。

RelayController 有三种实现：

- **SimpleRelayController**: 以确定性规则形成简要接力摘要：用户请求数、用到的工具（能力调用取动词的小写，外部工具取工具名，无法解析的 MTP 指令不计）与 token 总数；
- **LLMRelayController**: 经组合根注入的共享 LLM 服务，以 `shared.llm.librarian` 转换的请求参数异步生成更丰富摘要。近期事件是各块与记忆生成视图相同的材料，块之间空一行：被折叠的块此后只以摘要参与工作视图与生成，近期事件因此保留思考与外部工具过程，不只有问答；每次调用以 `patchouli.perception.engine.relay.engine.summary_timeout_seconds`（默认 60 秒）为时限。只在没有可用 LLM（Librarian 未配置模型）、提供商失败（`ModelProviderError`，含超窗）或调用超时时回退为规则摘要，回退的规则摘要接在 previous summary 之后，已折叠的历史不丢失；程序错误不被规则摘要掩盖，沿 compact 路径传播；
- **NoOpRelayController**: 用于关闭折叠能力。

Relay 控制器由 Runtime 按配置创建并注入 Engine 构造。关闭 perception（`engine.enable=false`）时 Runtime 注入 `engine=None`，Familiar 的摄入路径变为无副作用 no-op（原 NullPerceptionLayer 语义），维护用例对空存储自然空转。

Engine 不持有 Store / Queue，也不导入 `hivememory.patchouli.*`，可被其他 runtime 复用、可纯单元测试。

## 7. 维护与关闭

全局 scheduler 定期调用 `scan_idle_buffers_once()`；候选来自 WorkingSet 的 idle 查询（访问时间超时且未被 lease 占用），逐个执行统一 settle 时序，admission 失败记录警告并留待下一轮。关闭时 Patchouli 先关闭折叠——排队中的折叠项直接结束，进行中的折叠被取消，不改动话题——再排空提交队列（一并等待折叠项收尾，释放 lease），之后 `flush_all_for_shutdown()` 对全部驻留话题执行同一时序，结算带走话题的全部交互记录与折叠记录；真正空 Topic 没有可提交材料，但仍正常结束生命周期并计入 `generation_skipped_topic_ids`。单个话题的占用冲突或 admission 异常被隔离到报告的 `failed_topic_ids`，不阻止其余话题清理。

手动 settle 返回 `TopicSettleResult`，通过可选 `generation_task_id` 与派生的 `generation_submitted` 表达是否建立后台任务；无任务不等于生命周期失败。手动 evict（删除话题）返回 `TopicEvictionResult`，明确“不触发结算、不写记忆”，适合用户主动丢弃短期话题；manual compact 只压缩工作集，不结算、不驱逐。三种手动操作互不混用。Patchouli 业务结果不再放在 `services/perception.py`，server 层只在 HTTP 边界投影为响应模型。shutdown 批处理使用 `runtime.models.TopicShutdownFlushReport` 记录已结算 Topic、未建立 generation task 的正常 skip 子集，以及结算前驻留 block 数量；该运行时报告不进入 HTTP 链路，异常也不会被归入正常 skip。

## 8. 当前限制

- 短期话题是进程内状态，异常退出可能丢失未结算的交互记录；
- 话题在结算前保存全部交互记录，折叠不释放内存；进程内没有容量上限或背压，单个话题的内存占用与结算载荷随轮次增长；
- token 统计是对整轮材料（只截断工具结果）的估算，不是模型级精确 tokenizer 预算；
- 块的 token 统计计入完整的思考、工具参数与截断后的工具结果，工具密集的轮次会更早触发折叠；单个轮次没有总量上限，LLM relay 的近期事件同样包含这些材料，输入越大越可能触及 `summary_timeout_seconds` 或超出模型窗口而回退为规则摘要，规则摘要只列工具名；
- `fold_retain_recent_blocks` 只限制 block 数量，不保证保留后缀的 token 总量低于阈值；单个超大 block 也可能超过软水位线；
- 所有 compact 配置与 Engine 的折叠选择都拒绝小于 1 的 retain 值，折叠后工作视图至少保留一个最新 block；
- 被折叠的轮次只以有损的 `state_summary` 进入生成材料，原文只保存在来源 InteractionArtifact 中，生成不回读；
- Relay 的摘要调用在 lease 持有期间进行：LLM relay 等待模型时不阻塞事件循环，但该话题一直被占用，同一话题的后续提交在队列中等待，手动结算会遇到忙碌；
- 折叠项占用提交队列的一个执行槽（共 4 个），一次最多占用摘要时限（默认 60 秒）；多个话题同时折叠时，其他话题的提交要等执行槽；
- 折叠项排在已入队的同一话题提交之后：同一话题积压多轮时，这些轮次都写入之后才折叠一次，这次的摘要输入相应更大；Import Bus 每个外部会话同时只有一轮在途，积压通常很小；
- WorkingSet 的容量检查与候选选择运行在单事件循环内（方法同步、不内部 await），跨线程使用需调用方先行串行化；两个并发 NEW_TOPIC 的容量检查存在与旧实现等价的短暂竞态窗口；
- flush 触发器的历史 e2e 用例随感知层删除而移除，重建需要真实 LLM/Qdrant 栈；当前回归由 Familiar 集成测试（tests/integration/patchouli/test_perception.py）覆盖；
- 旧文档中的 `assistant_message` fallback、扁平 `context_messages` 和 Perception 私有 `InteractionPayload` 均已退出主路径。

调整这些语义时必须同时检查 Generation、Artifacts、Import Bus 与 shutdown drain，因为“何时丢弃交互记录”本质上是数据耐久性边界，而不只是一个摘要算法参数。

token-aware 保留、summary-only Topic 与 Relay 优化等议题见[短期记忆与上下文压缩 Idea](../ideas/short-term-memory-and-context-compaction.md)第 6 节；完整资料与来源保存见 [Raw Evidence Idea](../ideas/PatchouliPageFoldingRawEvidenceDesign.md)第 9 节；长 turn 与跨入口专项见[Turn 内上下文折叠 Idea](../ideas/long-running-agent-intra-turn-context-folding.md)第 14 节。
