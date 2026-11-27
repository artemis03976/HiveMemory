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
  - src/hivememory/memory_library/stores.py
  - src/hivememory/memory_library/topic_log.py
  - src/hivememory/memory_library/adapters/short_term.py
  - src/hivememory/patchouli/control/interaction_submission.py
  - src/hivememory/patchouli/application/interaction_submission_service.py
  - src/hivememory/patchouli/application/topic_management_service.py
  - src/hivememory/core/protocol/sealing.py
  - src/hivememory/core/models/interaction.py
  - src/hivememory/core/models/topic.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/system/import-bus.md
related_docs:
  - docs/workspace/README.md
  - docs/workspace/read-view-and-write-intents.md
last_reviewed: 2026-11-26
---

# 感知与短期话题

Perception 负责把一轮已经发生的交互转化为 Patchouli 可以持续维护的短期事实。它不猜测用户下一步要做什么，也不直接决定某段文字应成为哪条正式记忆；它保存结构化 turn、维护相互隔离的话题 buffer，并在话题停下够久、话题池满或手动操作时形成可交给 Generation 的材料。

“感知”在这里更接近记忆管理单元，而不是第二个 Gateway。Gateway 先解释入口，Alice 先完成 run，Perception 随后观察结果。这个先后关系防止记忆系统为了整理历史而重新取得入口决策或 Agent 执行权。

## 1. 模型分层

### 1.1 内容事实：TurnRecord

一轮交互的内容真相由 `core.models.TurnRecord` 表达，只有四项：本轮结局 `outcome`、`turn_id`、产出本轮的 `ActorIdentity`（不含 session），以及以序号 0 的用户消息开头的有序 `TurnEvent` 集。事件集是唯一的内容事实：用户消息取序号 0 的事件正文，回复正文是全部助手正文按序号以空行拼接，二者都在读取时从事件得到，不在记录中另存一份。

`TurnEvent` 有用户消息、助手正文、思考、子 Agent 交回的消息与工具调用五种；一次工具调用与它的结局是同一个事件，模型产生的事件记下来自本轮第几次模型响应（`response_index`），字段与一轮的不变量见[数据模型](../architecture/data-model.md#49-交互记录turnevent-与-interactionpayload)。事件集之外不保存由它归并的动作或轨迹：同一事实存两份时，两份会各自漂移，读取方也会从不同来源得到不一致的解读。Perception 不会从一段拼接文本逆向解析工具事实。

`TurnRecord.outcome` 保存 completed/cancelled/failed（默认 completed），从 `InteractionPayload` 原样转入。中断和失败只要有输出，也沿同一提交与幂等应用链形成 block；结局不是成功知识的标志，生成转写会注明不完整。会话上下文属于 workspace，Topic 仅作为记忆材料，不反向填充 Alice 的历史。

### 1.2 感知容器：LogicalBlock

`LogicalBlock` 在 TurnRecord 外只增加感知层元数据：block id、created time 与估算 token（`total_tokens`，第 6 节）。用户消息与回复正文是从事件读取的只读属性。它是不可变模型，表明同一轮内容事实进入短期话题后的感知视图；Gateway 的入口判断（意图、改写后的查询）不进入块。

可以简写为：

```text
LogicalBlock = TurnRecord + perception metadata
```

### 1.3 话题：原始数据与工作视图

话题的读取值是 `TopicData`，由话题的日志依次套用每条记录组装而成（第 7 节），只存两份原始数据：`interactions` 按接收顺序保存话题接收到的全部交互（`LogicalBlock`），只追加，话题结束前不移除；`folds` 按顺序保存每次折叠的 `TopicFold`，即这一次被折叠的块标识、折叠后的摘要与折叠时间。读取值另含展示 title/summary、资产绑定、最近模型名，`settled_until`：已经完成的沉淀覆盖到第几个交互（第 4 节），以及 `topic_generated_memory_ids`：本话题生成的记忆，即本话题主动写入结算到的记忆与此前几次沉淀新建的记忆（[记忆生成](./generation.md#42-本话题生成的记忆)第 4.2 节）。`has_unsettled_interactions` 由它派生，表示最近一次沉淀之后是否接收过新交互，没有接收过交互的话题为假。`progress_record` 是话题的进展记录（第 7 节）：按写下的顺序排列的 `TopicProgressEntry`，每一条是正文、指回原文的区间（覆盖的首轮与末轮的轮次标识、末轮最后一个事件的序号，与记忆引用是同一套坐标，区间从首轮序号 0 的用户消息开始）、覆盖到第几个交互（`covered_until`，交互记录的前缀长度）与写下的时间。`recorded_until` 由它派生，是最后一条的覆盖位置，还没有写过一条时为 0；其后的交互是还没有整理的原始交互记录，二者合起来就是话题的进展，回答话题进行到哪一步。构造时校验各条依次覆盖相邻的区间：每一条从上一条覆盖到的位置开始、往后至少一轮、不超过交互记录，指回的区间与覆盖的位置一致。话题快照（`TopicSnapshot`）带出进展记录、还没有整理的轮数（`unrecorded_turn_count`），以及其中最近至多 3 轮的节选（`unrecorded_turns`：轮次号，用户消息与回复各截到 200 个字符，超出时加省略号），话题池界面据此展示话题的进展；节选取自交互记录，已经折叠、还没有整理的轮次也在内。Gateway 读快照但不渲染这几项。

工作视图由两份原始数据派生，不单独存储：`blocks` 是最后一次折叠之后的交互记录，`fold_summary` 是最后一次折叠的摘要（没有折叠时为空串），`total_tokens` 是 `blocks` 的 token 之和。读取话题内容的消费方（折叠选择、Gateway 查询分析、主动生成与话题快照）读的都是工作视图；快照中的进展记录与还没有整理的轮次例外，取自交互记录。工作视图不另存一份，是为了不让同一事实有两份表达：分开存储的副本需要在每次写入与折叠时维持“工作视图是交互记录的后缀”，派生则不会失去一致。构造时校验全部折叠记录依次拼接，必须等于交互记录开头同样数量的块标识；`TopicData` 拒绝未声明的字段，以 `blocks=` 等工作视图名构造时显式失败。

**话题的位置**。话题上有六个“处理到第几轮”的位置，都是交互记录的前缀长度。日志里的四个是 `TopicData` 的属性：`interaction_count`（接收了几轮）、`folded_until`（折叠到第几轮，由折叠记录推出，工作视图从这里开始）、`recorded_until`（进展记录写到第几轮）与 `settled_until`（已经完成的沉淀覆盖到第几轮）。进程内的两个在驻留记录 `ResidentTopic` 上（第 7 节）：`settling_until`（被生成队列接纳、还没到终态的沉淀覆盖到第几轮）与 `progress_failed_until`（最近一次写进展记录失败时要覆盖到第几轮）。它们彼此独立推进，谁也不等谁，也都不切分材料：沉淀总是冻结全部交互与折叠记录，进展记录只决定下一条从哪一轮写起。每个触发看哪几个位置、推进哪一个、失败与关闭时怎样，按触发逐条写在 [`patchouli/services/perception.py`](../../src/hivememory/patchouli/services/perception.py) 的模块说明（触发表与 lease 规则）中。

写入话题的方法归短期存储（`ShortTermMemoryStore`），话题上发生的事各有一个写入方法，写入就是在日志末尾追加一条记录：`append_interaction` 只向交互记录追加一轮（以及首次资产绑定），`append_fold` 只追加一条折叠记录，`append_settled` 只推进沉淀进度、不改内容与最近写入时间，`append_generated_memories` 只改本话题生成的记忆（合并改指只在被吸收的记忆在话题当时的记录中时生效），`append_progress_entry` 只在进展记录末尾追加一条、不改内容与最近写入时间，只记到池中的话题，另有建立话题的 `create` 与退役的 `retire`。每种记录有一个套用函数（`memory_library.topic_log`），以完整构造（`TopicData.model_validate`）产生新的读取值：Pydantic 的 `model_copy(update=...)` 不运行校验器，用它写回会绕过构造校验。套用函数是纯的，时间取记录里的发生时间，运行时的写入与启动时的重放用同一组函数，重放得到的读取值与写入时内存中的相同。core 话题模型只保留 `TopicData`、`TopicFold`、`TopicProgressEntry` 与读取侧的派生（`topic_folded_until`、`topic_working_blocks`、`topic_fold_summary`）。

读取值只承载内容事实：执行占用不建模为记录字段，跨 await 的占用权由 `TopicWorkingSet` 的 lease 表达（本话题生成的记忆、沉淀完成与进展记录的写入不取 lease，第 4、7 节）。进程内每个驻留话题有一条 `ResidentTopic`（访问时间、在途沉淀与写进展记录失败的位置），按访问顺序排列，就是 LRU 的顺序；lease 表与它分开，取得 lease 不要求话题驻留。占用、访问时间、两个进程内的位置与 LRU 都只在进程内，不写进日志，日志只记已经发生的事。短期存储在内存中放池中话题的 frozen `TopicData`，读取原样返回、不碰磁盘，Store 与 Perception 之间只交接不可变的读取值；生命周期写回由 PerceptionFamiliar 编排。

## 2. 结构化摄入

`InteractionPayload` 是主动链路与 [Import Bus](../system/import-bus.md) 共享的协议。交互内容只以 `turn_events` 表达，其余字段是本轮结局 `outcome`、执行者使用的模型 `model_used` 与实际使用的附件引用 `used_attachments`（Import Bus 的提交为空），它们不是对事件的解析；用户消息与回复正文是从事件读取的只读属性，不保存、不序列化。载荷构造时校验一轮事件集的不变量，违反时显式失败，不修补。两条链路都由 `core.protocol.sealing` 中同一个封口函数 `seal_interaction` 组装它，交给 Patchouli 的载荷因此同形；封口只组装、不解析事件，序号 0 的用户消息事件由提交方写出（Alice 执行循环、Import Bus 服务端），执行方只交出回复正文、没有写助手正文事件时，封口在末尾补一条助手正文事件，取本轮最后一个响应序号（本轮没有响应序号时不带），一轮的响应序号因此仍全有或全无。它不重复保存本轮 Workspace 身份；两条链路在资源 owner 边界拆分 `IdentityScope`，进入提交队列时由外层 `InteractionSubmission.belong_to` 与 `from_actor` 独立携带归属和本轮发起者。

`InteractionSubmission` 另外携带可选的 `new_topic_title` 与 `new_topic_summary`：只在 `requested_topic_id` 为 `NEW_TOPIC` 时允许非空，已有话题的标题与摘要由话题自身持有，提交不能借此改写。`InteractionSubmissionCodec` 当前为 v8（事件可带响应序号），只注册当前布局：缺失字段、额外字段、旧 `identity_scope` 布局及嵌套对象不能规范往返的载荷均被拒绝。队列使用 `InMemoryWorkStore`，提交快照与幂等收据只驻留于当前进程，不提供旧版本队列载荷迁移或重启恢复。

队列按 `interaction_id` 幂等：等待中或已成功的交互以相同载荷重交时复用原收据，载荷不同明确冲突。为此保留的旁路索引只存编码载荷的摘要与归属，冲突比较不需要留住整份载荷。应用失败（工作项已到终态但没有成功）的交互允许以同一标识重新提交，载荷可以不同，例如原先指定的话题已退役或被删除、提交方重新路由后再交；新提交以递增的尝试号作为工作项标识入队。这条例外只放开已经失败的交互：等待中的交互重交会被判为重复或冲突，已成功的交互不会被第二份载荷重复应用。同一轮也不会被写入两次：把 block 写入话题是应用中最后一个可能失败的步骤，应用要么失败且没有写入，要么写入并成功，因此重新提交的只会是没有写入的交互，感知一侧不另设应用记录。在写入之后加入可能失败的步骤会破坏这一点。

公开提交路由 `patchouli.public.interaction.submit` 接纳交互后等待它应用到终态或时限，返回结局：成功时带实际所在的话题（`NEW_TOPIC` 已解析），应用失败时带异常类名，到时限仍未应用完成时为 `pending`，时限为空时等到终态；调用方取消等待不撤回已接纳的交互，它照常应用。`patchouli.public.interaction.status` 把队列中的工作状态归为四种进度：`pending`、`succeeded`（附实际所在的话题，`NEW_TOPIC` 已解析为实际话题）、`failed` 与 `unknown`（旁路索引仍在、工作记录已超出终态保留窗口，接纳过但结局不再可知），未被接纳时为空。`patchouli.public.interaction.wait` 在交互仍为 `pending` 时等待到终态或超时，超时时仍报告 `pending`，队列停止时返回停止那一刻的进度。两者都先校验归属，同名交互属于另一个 Workspace 时以 `WorkspaceMismatchError` 拒绝，不把其他 Workspace 的接纳当作重复。查询只覆盖队列的进程内保留窗口，窗口外的交互视为未接纳。公开提交路由在队列容量已满或已停止时以 `InteractionAdmissionUnavailableError` 拒绝，表示这次提交没有被接管，调用方可用同一标识重试。三条路由的生产调用方都是 root API 的提交交互与为一轮确定话题，查重与续接上一轮所在的话题在那里完成（[子系统公共契约](../contracts/subsystem-contracts.md#36-交互提交进度与等待)）。

当前布局、幂等拒绝与失败后重交的回归入口为 [InteractionSubmission 测试](../../tests/unit/patchouli/control/test_interaction_submission.py)，提交并等待应用的结局、进度查询与等待见[提交服务测试](../../tests/unit/patchouli/application/test_submission_services.py)；Import Bus 经过真实队列与感知存储的身份交接、查重与新话题标题见 [Import Bus 集成测试](../../tests/integration/system/test_import_bus_submission.py)。

交互应用成功后，Perception 才会把用户明确使用的 `(asset_id, asset_ref)` 记录为 Topic 的 `TopicAssetBinding`。上传、候选列表或 UI 选择本身不会进入 Topic 事实。由此，感知层只接收已完成身份与资源前置校验的交接输入，不拥有 `WorkspaceAssetStore` 的生命周期。

```text
InteractionPayload（构造时已校验一轮的不变量）
  -> TurnRecord(outcome, identity, turn_events)
  -> 渲染整轮材料并估算 token（engines.perception.material）
  -> LogicalBlock
  -> ShortTermMemoryStore.append_interaction：日志追加一条 interaction 记录，
     读取值的交互记录随之追加   应用到此完成
  -> needs_fold：提交队列排入折叠项 -> fold_topic（折叠，第 5 节）
```

任务进程与 Import Bus 以同一封口函数封口，经 root API 的提交交互调用同一条公开提交路由，载荷原样提交。两者先进入同一 `InteractionSubmissionQueue`，再由 `PerceptionFamiliar.receive_interaction()` 编排 `MemoryPerceptionEngine.build_block()`（纯算法）、占用检查与 Store 写入，不存在导入专用的扁平文本主链，也不存在中间感知层。应用在写入后即完成，不执行折叠；话题需要折叠时，由提交队列在这次应用之后另行排入折叠项。

结构化摄入的边界来自外部 harness 的真实事件形态：Import Bus 接收的一轮整轮记录中，用户消息由服务端写为序号 0 的事件，其后是助手正文、思考与工具调用三类事件。两种来源写出的事件同形，来源之间真实存在的差异只体现为取值：调用 HiveMemory 能力（Alice 的 MTP 指令、外部 harness 经 HiveMemory MCP 工具的检索、读取、写入与更新）带动词与作用对象，外部工具带工具名，下游不判断来源。思考、工具调用与工具返回必须以 `TurnEvent` 保留，不能为了生成一段看似完整的 transcript 而提前丢弃。

对事件集的解读只在感知层进行一次：`engines/perception/material.py` 把一轮事件渲染为感知层唯一的材料文本，记忆生成视图（[记忆生成](./generation.md#21-材料与渲染)）、用模型写的折叠摘要与逻辑块的 token 统计都使用它，按同一个工具结果上限得到同样的材料。入口层与封口不另行构造第二套摘要，否则历史重放、主动生成与导入归档会拥有互相漂移的事实来源。

一轮的段也由感知层从事件推出（`engines/perception/segments.py`，纯函数，输入只有一轮满足不变量的事件）。生成的记忆要能指回一轮中的具体位置：一轮常有几十上百次工具调用，只引用到轮次时同一轮产出的多条记忆全都指向同一处；直接引用事件，模型又要在几十个条目里挑。段介于二者之间，是一个决定连同其后没有正文的若干响应：

- 第一段从用户消息开始，包含第一次响应；之后每次带正文或思考正文的响应开启新的一段，其余响应并入当前段；
- 一段已有 8 次响应（`SEGMENT_MAX_RESPONSES`）时，下一次响应开启新的一段，只在响应边界上切。这个上限只截断长尾：在 1,216 轮真实 Claude Code 记录上，每段响应数 P90 为 5，不设上限时最多 18；
- 子 Agent 交回的消息没有响应序号，并入其后的段，位于末尾时并入最后一段；
- 没有响应序号的轮次（Codex，以及加入响应序号之前的记录）整轮一段。

各段的事件序号区间首尾相接，覆盖整轮。段不保存，读取时计算：切段以原始交互记录为准，生成材料只是它的一个视图；规则以后调整时，已经保存的引用（轮次标识加事件序号区间）不受影响。段只用于生成材料的标签与记忆引用，渲染函数本身不带段标签，token 统计与折叠摘要不受影响。

目标话题属于提交前的 Gateway 路由决定，而不是应用时重新计算的感知结果。任务进程在 Gateway 阶段取得话题路由：指向已有话题时在 PREPARE 确认它存在并刷新驻留，提交到该话题；指向新话题时，本轮第一次写入时经话题准备以 Gateway 的标题与摘要建立话题，没有写入则以 `NEW_TOPIC` 携带标题与摘要提交，在应用时建立。Import Bus 在提交前以 `PASSIVE_MEMORY` 为需要提交的每一轮调用 Gateway（在 root API 的提交方法内），把路由得到的目标话题作为 `requested_topic_id` 提交。目标为 `NEW_TOPIC` 时，Perception 以提交携带的 `new_topic_title`/`new_topic_summary` 建立话题，缺省时使用默认标题“新建话题”与空摘要；后续轮次的路由依赖候选话题的标题与摘要，因此 Import Bus 新建话题时带上 Gateway 给出的标题与摘要。外部 harness 经 MCP 写入或读当前话题时，Import Bus 在那时经话题准备建立这一轮的话题（以同样的标题与摘要），这一轮回流时以该话题的真实 id 提交（[Import Bus](../system/import-bus.md#47-写入与读当前话题时确定话题)）。应用失败的交互没有写入任何话题，重新提交时按提交方这一次交来的目标话题应用。Import Bus 拥有外部会话的串行、查重与确认，Gateway 只负责入口判断，两者都不持有话题内容。

冻结沉淀材料时，话题接收到的全部交互都进入沉淀，不在入口筛选。入口只看到用户消息与已有话题上下文，看不到这一轮的回答、工具调用与结果，“请提交”“继续”这类短消息背后常是整轮实现；按入口判断筛选会把这些轮次整轮丢出生成，交互 Artifact 也不再保存它们。一轮是否有长期价值，由生成阶段在完整材料上判断；Gateway 的决定只提供话题路由，不携带价值信号。

## 3. 话题准备与 LRU

`prepare_topic()`（公开路由 `patchouli.public.topic.prepare`）为一轮确认或建立目标话题，不检索：

- `NEW_TOPIC` 创建 UUID、title 和 summary；
- 已存在话题继续使用并刷新访问顺序；
- 指定话题不存在时拒绝请求，不把跨 Workspace 的全局 ID 误当作新话题。

它服务于写入：写入需要一个已存在的话题作为派发话题，话题的折叠摘要与最近 blocks 是生成材料。任务进程在 PREPARE 确认 Gateway 指向的已有话题并冻结为派发话题，在本轮第一次写入时建立 Gateway 指向的新话题；Import Bus 为外部 harness 写入时的一轮确定话题。交互本身不需要提前建立话题：Gateway 指向新话题而本轮没有写入时，以 `NEW_TOPIC` 提交，由 `receive_interaction()` 在应用时建立。两条路径共用同一段 `_ensure_topic`：已有话题确认它还在池中；`NEW_TOPIC` 时池满先腾位置（`_retire_lru_for_new_topic`），再建立话题并登记驻留。新话题因此只在第一次写入或交互应用时进入话题池。

当需要创建新话题且池中的话题数达到 `patchouli.perception.max_resident_topics`（每个 Workspace 计，默认 5）时，`TopicWorkingSet.needs_eviction()` 判定容量、`select_lru_candidate()` 从按访问顺序排列的驻留记录（OrderedDict，O(k)）选出最久未访问且未被占用（lease）的话题让它退役：有未沉淀的交互时先按 `LRU_EVICTION` 沉淀并提交后台 generation task，然后让它离开 Store 与驻留集合；没有时直接退役。池中的话题超出容量时（驻留容量在两次运行之间调小，第 7 节），同一次建立话题逐个退役到容量以内。已有话题命中不会为了“置顶”而让另一话题退役；候选被占用时改选其他话题，全部被占用时抛出 `TopicBusyError` 留待重试。容量与候选以 WorkingSet 的驻留记录为准，Store 只表达内容事实。池满时的退役随建立话题发生：没有建立话题的轮次不触发退役；腾位置失败（候选都被占用、沉淀材料未被生成队列接纳，接纳失败的错误原样抛出）发生在第一次写入时，错误交回这次写入；发生在交互应用时，交互按提交队列的规则重试（只重试占用冲突）或应用失败。

话题准备与应用都会确保话题存在：已有话题在两者之间可能已经退役或被删除，应用时话题已不存在则以 `ResourceNotFoundError` 失败，交互没有写入任何话题。建立之后没有接收交互的话题（例如确定话题之后进程收到停止、写入后 CPU 没有输出、外部 harness 的这一轮没有回流）不另行补偿：它不是 Gateway 的候选话题，停下超过沉淀阈值时直接退役，或在池满时退役，不触发生成。

## 4. 触发原因与具名用例

沉淀（含沉淀并退役）、折叠与删除是概念独立的具名用例，由 `PerceptionFamiliar` 编排，不再通过决策矩阵把触发原因解释成动作组合。`TriggerReason` 只作为 `TopicSettleMaterial` 的 provenance 标签（记录“这次沉淀为什么发生”），不驱动分支；折叠与删除不建立沉淀任务，没有触发原因。枚举的字符串值写进任务、事件与话题日志的退役记录，是保存格式，不随名字改变：

| 原因（provenance） | 触达的用例 | 当前结果 |
|:---|:---|:---|
| `IDLE_TIMEOUT`（值 `idle_timeout`） | 沉淀 | 沉淀扫描中，话题停下超过 `settle_after_seconds` 且有未沉淀交互时沉淀，话题留在话题池；从没接收过交互的空话题直接退役 |
| `LRU_EVICTION` | 沉淀并退役 | 池满时为新话题腾出位置，退役前沉淀未沉淀的交互 |
| `MANUAL_SETTLE` | 沉淀并退役 | 手动沉淀未沉淀的交互，然后退役 |

三个用例的职责边界：

- **沉淀**（`_settle_topic`，三个触发共用）：沉淀把话题新的交互交给生成，话题留在话题池；退役让话题离开话题池，退役前先沉淀还没有沉淀的交互。用户会断断续续地回到同一件事，话题停下一段时间就退役会把连续的工作切成几个话题，所以停下够久时只沉淀，退役只在池满与手动沉淀并退役时发生（停下够久时，从没接收过交互的空话题退役）；进程关闭不让话题退役，也不为话题发起沉淀（第 7 节）。是否退役由调用的用例决定，不读触发原因。时序为：获取 lease → 话题在 `settled_until` 与 `settling_until` 中较远的那个之后还有交互时，冻结全部交互记录与折叠记录（不只是新增的部分），构建带 `retires_topic` 的 `TopicSettleMaterial` → 持有 lease 等待生成队列接纳 → 接纳之后在驻留记录上记下 `settling_until`，话题留在存储与驻留中、不刷新访问时间；退役让话题离开存储与驻留。等待接纳的整个期间持有 lease，存储自己的锁只在每一次读写时持有，两者是两把锁。没有新交互时不生成：只沉淀时什么也不做，退役时直接让话题退役。接纳失败时话题与各个位置原样保留可重试，无需 abort——记录从未被改动。
- **沉淀进度在生成完成时推进**：话题随时落盘（第 7 节），进度若在接纳时就写进日志，进程崩溃时话题以为已经沉淀，生成却随进程丢失，这段交互再也不会沉淀。所以接纳之后只在驻留记录上记下 `settling_until`，只在进程内；判断有没有新交互以 `settled_until` 与 `settling_until` 中较远的为准，在途期间沉淀扫描不会再沉淀同一段。SETTLE 任务到达终态时，控制器经本地路由 `topic.record_settled(topic_id, *, belong_to, settled_until)` 交给 `record_settled`，位置取任务冻结的全部交互数；感知写一条 `settled` 记录（话题在池中时同时更新读取值），清除不超过它的 `settling_until`。成功、失败与用户取消都算完成：失败的沉淀不自动重试，取消的不重做；关闭时取消的不算，`settled_until` 停在上一次完成的位置，重启后由沉淀扫描重新沉淀。哪些终态算完成由生成控制器判断。协调器在材料里没有可生成的轮次时不建任务，接纳后立即推进 `settled_until`；工作视图总保留最近几轮，正常不会走到这一支。同一话题的沉淀按话题顺序键依次执行；用户取消排队中的沉淀时完成的顺序可能颠倒，沉淀进度只往前，较早的沉淀后完成不把它拉回去。生成已经写下记忆、日志还没记下完成时崩溃，重新沉淀会复述这些记忆，交给查重。
- **退役与日志**：退役在日志中记一条（带退役原因），话题随即离开存储与驻留，日志保留到它最后一次沉淀完成后再删除：`record_settled` 遇到已退役的话题照常记下，这一次之后再没有未沉淀的交互即是最后一次，随即删除日志。在途的沉淀已经覆盖全部交互时退役不再生成，日志同样等这次在途的沉淀完成。沉淀都已完成的退役（包括从没接收过交互的空话题）直接删除日志。日志删除之前，退役话题照样记下生成的记忆、经本地路由 `topic.get_generated_memories` 交出本话题生成的记忆：话题退役时可能还有排在前面的同话题沉淀，最后一次沉淀要认得它们写下的记忆（[记忆生成](./generation.md#42-本话题生成的记忆)第 4.2 节）；退役的话题不回到话题池，`topic.get` 与话题列表都看不到它。
- **折叠**（`fold_topic`，是否需要由 `needs_fold` 判断）：取得 lease 后，把工作视图中保留后缀之前的旧 blocks 交给折叠摘要写作者生成新的折叠摘要，再经 `append_fold` 追加一条折叠记录写回，`folded_until` 随之推进。被折叠的块仍留在交互记录中，工作视图向后推进，token 统计随之只计入保留的块。话题已退役或删除时直接结束，正被占用时以 `TopicBusyError` 留待重试；摘要返回后才一次写回，中途失败或取消都不改动话题。
- **删除**（`delete_topic`，话题池的手动删除）：从 Store 与驻留记录移除话题并立即删除它的日志，不沉淀、不写记忆；话题被占用或不存在时报 `removed=False` 而不是阻塞。已经被生成队列接纳的沉淀不撤回：接纳即交接，沉淀进度之前的交互照常生成记忆，排队中的沉淀在话题删除之后照常执行，完成时话题已经不在，`topic.record_settled` 与 `topic.record_generated_memories` 都不记。

生成执行之后，控制器经本地路由 `topic.record_generated_memories` 把这次生成的记忆与合并改指交给 `record_generated_memories`，感知经 `append_generated_memories` 记到话题的日志上；话题已经退役、日志还在时照样记，话题被删除或日志已经删除时不记。记下生成的记忆与沉淀完成不取 lease：本话题生成的记忆只往一个与交互、折叠和沉淀位置都不相交的集合里加，`settled_until` 只往前推进，不需要跨 await 占用话题。短期存储的每次写入都在锁内依次读当前的读取值、用套用函数算出新值（构造校验失败直接抛出，什么都不写）、经 Port 追加这一行、替换内存中的值：await 之前读到的读取值不会被写回，折叠等待摘要、沉淀等待接纳期间记下的生成的记忆因此不被覆盖；lease 仍保证这期间交互记录与折叠记录不变，折叠前缀与冻结的交互数照样成立。写入方法不检查话题处在什么状态，话题不存在时什么都不写、返回空值；持有 lease 时找不到话题说明占用被绕过，以 `ResourceNotFoundError` 显式失败。

这些用例的结局各有一个运行时事件（[事件契约](../contracts/routes-and-events.md#4-runtimeevent-观测契约)第 4 节）：新建话题发 `topic.created`（`created_by` 为 `interaction` 或 `prepare`），每次沉淀或退役发 `topic.settled`（带触发原因、是否提交了生成，以及是否随之退役），删除话题发 `topic.deleted`。`topic.settled` 在沉淀被生成队列接纳时发出，不等沉淀到达终态，与日志中沉淀到达终态时写下的 `settled` 记录不是一件事，`settled_until` 只由后者推进。接纳失败、话题保留时不发 `topic.settled`；折叠不发话题事件，以 Work Queue 的 `work.*` 出现。事件只是旁路观测，发布失败不影响话题的存储与沉淀。

历史上曾由 `TriggerManager`/`TRIGGER_PLANS` 决策矩阵与记录字段状态机（`BufferState`）承担这些流程，现已删除：占用权建模为 WorkingSet 的 lease，而不是记录状态，补偿校验随之消失。

`TopicSettleMaterial` 是沉淀时 Perception 交给 Generation 的冻结材料：携带话题的全部交互记录与折叠记录，工作视图用与 `TopicData` 相同的 core 派生函数得到；此外只携带 `belong_to`、已确认的 `asset_bindings`、`retires_topic` 与冻结时本话题生成的记忆。同一话题会被沉淀多次，每次都冻结全部交互，此前沉淀出的记忆记在本话题生成的记忆中（[记忆生成](./generation.md#42-本话题生成的记忆)第 4.2 节），不携带触发沉淀的 actor 或 `IdentityScope`。生成材料是工作视图（`fold_summary` 与未折叠的 blocks），来源 InteractionArtifact 覆盖全部交互记录并带折叠记录（[记忆生成](./generation.md)、[Artifacts](./artifacts.md)）。Coordinator 为手动、沉淀扫描与池满三个来源统一构造所属 Workspace 的 `system` 发起者，实际参与 Agent 从 turn identity 聚合为贡献者；沉淀的查重（一次性生成）与馆长的读取都只能读到本 Workspace 的 PUBLIC 记忆。沉淀交给 Generation 之后，按 `patchouli.generation.settle_mode` 选择的方式生成（一次性生成或馆长，见[记忆生成](./generation.md#31-mode-a被动观察--archive)第 3.1 节）；感知侧的沉淀时序、冻结的交接事实与接纳不随方式变化。进入队列后，后续处理使用冻结的交接事实，不能从当前进程状态重新推导 Workspace、读取权限或资产关系。

手动用例互不混杂：`settle_and_retire_topic` 沉淀未沉淀的交互、在材料被可靠接纳后让话题退役（没有未沉淀的交互时直接退役）；折叠只推进工作视图，不沉淀、不退役（没有公开的手动折叠入口）；`delete_topic` 只删除、不生成记忆。手动沉淀并退役的顺序固定为冻结沉淀材料 → 生成队列接纳成功 → 退役；话题不存在时以 `ResourceNotFoundError` 拒绝，队列已满或已停止时抛出 `TopicSettleAdmissionError`，话题的交互记录与折叠记录保持完整可重试。没有新交互（包括真正空的话题）时仍按契约退役并返回成功，以 `generation_submitted` 表达是否建立后台任务。

### 内容判空语义

Topic 是否为空只看是否接收过交互：`has_content` 为交互记录非空，`is_empty` 为其否定。折叠不移除交互记录，有折叠摘要的话题一定有交互记录；话题列表（排除空话题时）与折叠判断都使用这一判断，接收过交互的话题不会被当作空话题。没有接收交互的话题可能已经承载过写入的派发，Perception 不据此立即删除它，它在停下超过沉淀阈值时直接退役，或在池满与手动沉淀并退役时退役，都不生成。真正空的话题没有可沉淀或折叠的内容，但沉淀并退役仍会正常结束其生命周期。折叠不支持“总结后清空全部 blocks”的 `retain_count=0` 语义：折叠配置与内部入口都要求保留的块数不小于 1。

这层返回值边界很重要：底层感知算法可以决定“应交出哪些材料”，但后台任务、事件和取消属于上层控制面。

## 5. 折叠

当话题工作视图的 token 超过 `fold_token_threshold`（默认 32768）且有可折叠的旧前缀时，话题需要折叠。`fold_retain_recent_blocks`（默认 2，必须大于 0）定义工作视图中保留的最近 blocks 数；旧前缀与上一次的折叠摘要由折叠摘要写作者合成为新的折叠摘要，写回只追加一条折叠记录，工作视图从被折叠的块之后开始，保留后缀继续参与下一轮上下文。写作者不总结保留后缀，避免 `fold_summary + recent_blocks` 重复承载同一轮事实。由于生成摘要的整个窗口内 `fold_topic` 持有该话题的 lease，其他调用无法同时取得占用权，写回不需要前缀补偿校验。

折叠是话题自身的维护，不属于交互应用。交互写入话题即完成提交；交互提交队列在这次应用之后检查 `needs_fold`，需要时排入一个折叠项（工作标识以 `topic_fold:` 开头）。折叠项是话题维护项（`TopicMaintenance`）的一种，另一种是写一条进展记录（第 7 节），两种的调度相同。折叠项与该话题的提交共用排序键 `topic:{topic_id}`（`topic_ordering_key`），按键串行的 lane 因此在折叠执行期间挡住同一话题的后续提交，它们在队列中等待，等待不计入各自的处理时限；其他话题的提交照常执行。由此：

- 交互的确认（Import Bus 与任务进程的提交并等待应用）不包含折叠耗时；同一话题的下一轮要等折叠结束，Import Bus 等待超过 `import_bus.apply_wait_seconds` 时返回可重试的 503 `apply_pending`；
- 同一话题同时至多一个未结束的维护项（折叠或写一条进展记录），已有时不再排入；队列已满或排入时出错，就放弃这一次，交互结果不受影响，下一次写入该话题时重新判断；
- 折叠失败只结束这个折叠项，话题保持原样，日志原因码 `topic_fold_failed`；话题正被沉淀等用例占用时重试；
- 用模型写折叠摘要时，每次调用以 `summary_timeout_seconds` 为时限（第 6 节），超时回退为规则摘要，使一次折叠的耗时有上限，并让话题回到阈值以下。

交互应用的时限（30 秒）由提交队列的 handler 只对应用本身施加；lane 不设统一时限，折叠不被它截断。

折叠的设计目标是把 context window 视为工作集，而不是无限日志：旧页可以折叠为折叠摘要，新的交互继续发生在同一话题。摘要是工作视图，不是原始证据：折叠不丢弃原文，被折叠的块留在交互记录中，沉淀时随全部记录冻结进来源 InteractionArtifact。

折叠是 Patchouli 内部对话题工作集的维护，主动链路与 Import Bus 共享同一语义。`fold_summary` 与近期 blocks 服务于 Gateway 辅助分析、主动 WRITE/UPDATE 的生成背景与之后的沉淀；Alice 的对话历史从 Session 重放。Import Bus 的响应只报告这一轮的导入结果，不返回检索记忆或折叠上下文。外部 harness 自行压缩对话不会缩减 Patchouli 已摄入的 blocks，因此不能作为跳过内部折叠的依据。

超过阈值不触发沉淀：折叠只推进工作视图。生成材料仍是工作视图，被折叠的轮次只以摘要进入生成，原文在沉淀时进入来源 InteractionArtifact。当工作视图中的 blocks 数量不大于保留数时没有可折叠前缀，折叠延后，因此阈值是工作集软水位线，而不是严格的模型 context 上限。

## 6. MemoryPerceptionEngine 与两个写作者

`MemoryPerceptionEngine` 是无状态的短期记忆摄入与折叠算法引擎，职责只有三项：

- `build_block`: 把 `InteractionPayload` 转化为不可变 `LogicalBlock` 并估算 token。块的 `total_tokens` 是对整轮渲染材料的估算：用户消息、助手正文、思考、能力调用的一行摘要与外部工具步骤，只有外部工具结果按 `patchouli.generation.transcript.max_result_chars` 截断，折叠阈值因此直接反映生成实际看到的材料；能力调用读回的记忆不渲染，也不计入；
- `should_fold`: 判断话题工作视图的 token 是否超过折叠阈值；
- `select_blocks_to_fold`: 选择待折叠的旧 blocks 前缀（保留最近 N 个）。

折叠摘要与进展记录由两个写作者写，`PerceptionFamiliar` 直接持有它们，Engine 不持有、也不转发。两者由 Runtime 按 `patchouli.perception.engine.summary_writer` 创建（`type` 为 `llm` 或 `simple`）：

折叠摘要的写作者（`engines/perception/fold_summary.py`，基类 `FoldSummaryWriter`，工厂 `create_fold_summary_writer`）有两种：

- **RuleFoldSummaryWriter**（`type: simple`）：以确定性规则形成简要摘要：用户请求数、用到的工具（能力调用取动词的小写，外部工具取工具名，无法解析的 MTP 指令不计）与 token 总数，接在上一次的折叠摘要之后；
- **LLMFoldSummaryWriter**（`type: llm`）：经组合根注入的共享 LLM 服务，以 `patchouli.llm` 经模型注册表解析得到的请求参数异步生成更丰富摘要。近期事件是各块与记忆生成视图相同的材料，块之间空一行：被折叠的块此后只以摘要参与工作视图与生成，近期事件因此保留思考与外部工具过程，不只有问答；每次调用以 `patchouli.perception.engine.summary_writer.summary_timeout_seconds`（默认 60 秒）为时限。只在没有可用 LLM（Librarian 未配置模型）、提供商失败（`ModelProviderError`，含超窗）或调用超时时回退为规则摘要，回退的规则摘要接在上一次的折叠摘要之后，已折叠的历史不丢失；程序错误不被规则摘要掩盖，沿折叠路径传播。提示词由 `get_fold_summary_prompt_text` 按语言取得。

`ProgressRecordWriter`（`engines/perception/progress_record.py`）写进展记录的一条，与用模型写折叠摘要用同一个模型与时限：Runtime 只在 `summary_writer.type` 为 `llm`、且 `patchouli.llm` 配置了模型时创建它，否则交给 `PerceptionFamiliar` 的是 `None`，话题不写进展记录。它每次调用以 `summary_timeout_seconds` 为时限，提供商失败重试一次。输入只有两部分：旧进展记录逐条列出，只作上下文；还没整理的交互按感知层唯一的材料渲染，每一轮标上它在话题中的轮次号（`[Turn N]`），结局不是 completed 的轮次在轮首注明，与记忆生成视图相同。提示词按语言取自 `i18n/prompts.py`，只有一份：话题还没有写过一条时，旧进展记录一节写“暂无”（英文 “None”），提示词写的是“以旧进展记录（如有）作为参考”。要求 JSON 输出，结果是 `{"entry": "..."}`；不是这个形状、正文为空、超时或提供商失败都抛出，不回退为规则摘要：进展记录只往后追加，一条写坏了不能再改，宁可这一次不写。

输入同时有旧进展记录与新交互：看得到旧进展记录，写的人知道前面在谈什么、做什么，这一段推进了多少，又不与前面重复；只看新交互时没有上下文，留下的常是“请修改后提交”这类离开前文就读不懂的短句。旧进展记录只作上下文，不改写，新的一条只写这一段新交互带来的进展，也不做结构化的条目。不每次由全部材料重新生成整份进展记录：那只能产出一句始终翻新的总结，前后两次没有约束，容易不一致、变成笼统概括；进展记录比折叠生成得更频繁，全量的成本也高。话题可能是一项工作，也可能是一段讨论或闲聊，提示词不假设话题是工作。

引擎总是建立，感知没有关闭开关。Engine 不持有 Store / Queue 与写作者，也不导入 `hivememory.patchouli.*`，可被其他 runtime 复用、可纯单元测试。

## 7. 维护、存储与启动时读回

**话题停下多久**。沉淀与写进展记录两个扫描看的是同一个信号：话题停下多久，即驻留记录的访问时间（接收交互、为写入预先确定话题与启动读回都会刷新它），候选都来自 `TopicWorkingSet.list_inactive`（超过阈值且未被 lease 占用）。两者只是阈值不同：`patchouli.perception.progress_record_after_seconds`（默认 1800 秒）比 `settle_after_seconds`（默认 3600 秒）短，话题停下时先写一条进展记录，再沉淀。

**沉淀扫描**。全局 scheduler 定期调用 `scan_for_settle_once()`（任务 `perception_settle`，默认每 30 秒）：停下超过 `settle_after_seconds` 的话题中，有新交互的逐个沉淀，从没接收过交互的空话题直接退役，接纳失败（任何异常）记录警告并留待下一轮。沉淀之后话题留在池中、访问时间不变，没有新交互就不会再被沉淀，话题停下的同一段时间里因此不会反复沉淀；话题接收新交互后重新计时。

**写进展记录**。读话题的进展的，是停下之后回来的人或另一个会话；交互连续进行时，前台自己的上下文就够用。所以话题停下够久时写一条：全局 scheduler 定期调用 `scan_for_progress_record_once()`（任务 `perception_progress_record`，默认每 30 秒），停下超过 `progress_record_after_seconds`、有还没整理的交互、上一次没有在同一位置失败的话题，由交互提交队列排入一个写一条的维护项（第 5 节），与折叠共用调度：与该话题的提交共用排序键，同一话题同时至多一个未结束的维护项，已有、维护已关闭或队列已满时这一轮不排、下一轮再看。写进展记录的阈值比沉淀的短，沉淀时通常已经写过一条，沉淀这一触发因此不另外再写。写下一条之后访问时间不变，没有新交互就不会再写。

写一条（`write_progress_entry`）读话题最新的读取值，取旧进展记录与第 `recorded_until` 个之后的交互，交给写作者生成一段总结，完成后经 `append_progress_entry` 追加，`recorded_until` 推进到生成开始时的 `interaction_count`。它不取 lease：折叠持有 lease 是因为它折叠的是此刻工作视图的前缀，交互与沉淀都会改变它；写一条的前提只有“从当前覆盖到的位置开始”，这个位置只有写进展记录会改，同一话题同时至多一个维护项，交互只往后追加，前提不会被别的写入破坏，存储的位置校验兜住违反这一点的错误。生成期间沉淀与 LRU 照常取得 lease；话题因此退役或被删除时，`append_progress_entry` 找不到它，这一条不记。生成完成才追加，进程中途退出时什么都没有写下，重启后话题停下够久时再写，不需要在途状态。生成失败（提供商错误、超时、输出不合格）不改动话题，驻留记录上记下 `progress_failed_until`（这次要覆盖到的位置），同一位置不再重试，话题有了新交互再判断，避免提供商故障时每一轮扫描都重试；失败以 Work Queue 的 `work.*` 出现，日志原因码 `progress_record_failed`。写下一条记一行 info 日志（话题、覆盖的轮数与覆盖到的位置、正文字数，不记内容）。进展记录不进入记忆生成的材料，沉淀冻结的仍是交互记录与折叠记录。

**按句柄读出**。进展记录的读者是前台：人在话题池里看，agent 经 actor API 的 READ 按话题句柄读（`topic:<话题 id>` 读索引，`topic:<话题 id>#<n>` 读一条的全文与它覆盖的原文，[读取视图与写入意图](../workspace/read-view-and-write-intents.md#41-话题句柄)第 4.1 节）。Patchouli 一侧只是公开路由 `patchouli.public.topic.get_data` 交出 `TopicData`，按 scope 的 Workspace 复查归属，不为前台另写读取方法；渲染属于 MemoryCompiler。读取只交出池中的话题：退役之后，日志要等最后一次沉淀完成才删除，其间读不到，读取按已过期交付。退役到日志删除之间只隔一次生成，日常的 LRU 退役也很少，所以不为退役话题开放读取，前台的读取与沉淀之后的话题不纠缠。读话题不刷新访问时间，不改变话题停下多久。

**话题的日志**。每个话题是一条只追加的日志，存成一个 JSON Lines 文件 `<topic_dir>/<topic_id>.jsonl`（`patchouli.perception.topic_dir`，默认 `.hivememory/topics`，相对路径按工作目录解析），一行一条记录：`created`（格式版本、Workspace 归属、标题与展示摘要）、`interaction`、`fold`、`settled`（沉淀覆盖到的交互数；字段名 `interaction_count` 是日志格式的一部分，套用后成为 `settled_until`）、`generated_memories`（本话题生成的记忆与合并改指）、`progress_entry`（进展记录的一条）与 `retired`（退役原因）；每条带发生时间，内容沿用现有模型的 JSON 形式。话题 ID 全局唯一，Workspace 归属写在第一条记录里，目录不按 Workspace 分层：数据根的路径长度有限制。话题上发生的事发生时即落盘，不改写已有的记录：话题按顺序保存全部交互，折叠不丢原文，沉淀进度、本话题生成的记忆与进展记录都只往后加，它本来就是一条只增长的日志；把它当作整块快照保存，每写一点都要重建整个话题，也只能在正常关闭时整池写出、崩溃即全部丢失。唯一的 Port 实现 `JsonlShortTermStorage` 同步写一行并交给操作系统（不经 Python 的缓冲），进程崩溃不丢；不 fsync，断电可能丢失最后几行。追加失败时文件截回追加之前的长度，日志里不留半行，存储先落盘再改内存，内存中的读取值不变、错误照常传播。健康检查报告话题目录是否可写。

**关闭**。关闭时 Patchouli 先关闭话题维护——排队中的维护项直接结束，进行中的折叠与写一条进展记录被取消，不改动话题——再排空提交队列（一并等待维护项收尾，折叠释放 lease），交互在关闭之前都已写进日志。之后只处理生成工作：生成队列停止取新工作，未执行的写入、更新与合并保存到磁盘，未执行的沉淀不保存、以关闭为原因取消，已开始写入的沉淀等它写完并记下进度（[记忆生成](./generation.md#62-提交阶段关闭时保存与启动时继续)）。话题不在关闭时沉淀或退役，也不为停下的话题写进展记录（退出不等待模型调用），空话题留到重启后由沉淀扫描退役。

**启动时读回**。下次启动时，Patchouli 在挂载公开路由与启动交互提交之前读回全部日志并重放（`ShortTermMemoryStore.load`），Gateway 与新交互看到的是上次运行留下的话题池，不论上次是正常关闭还是进程崩溃：

- 没有退役的话题回到存储，`restore_topics` 按 `last_update` 从旧到新登记驻留，最久没有写入的是第一个退役候选；驻留的访问时间不写进日志，这只是上次运行访问顺序的近似，话题停下多久从读回时重新计时，`settling_until` 与 `progress_failed_until` 从 0 开始。读回的 `settled_until` 停在上一次完成的沉淀，上次运行中在途或关闭时取消的沉淀，在之后的沉淀扫描中重新沉淀。进展记录随日志重放，话题停下够久时照常再写；升级前已有的话题没有进展记录，停下够久之后从第 1 个交互开始写第一条。
- 已经退役、最后一次沉淀没有完成的话题不回到话题池，在生成队列启动之后由 `resettle_retired_topics` 重新发起这次沉淀（材料由日志组装，`retires_topic=True`，带原来的退役原因），完成时删除日志；没有被生成队列接纳的保留日志，下次启动再发起。已经退役、沉淀都已完成的（删除日志之前进程退出）直接删除日志。
- 同一 Workspace 读回的话题超过 `max_resident_topics`（配置在两次运行之间调小）时全部读回，下一次需要新话题时逐个退役到池有空位，不丢未沉淀的交互。
- 最后一行不完整（没有换行，写入中途进程退出）时丢弃并从文件中截掉这半行，其余照常读回（`topic_log_partial_line`）；中间某一行解析或校验失败、格式版本不认识、或重放时构造校验失败时，整个文件改名为同目录的 rejected 文件（`<topic_id>.rejected-<UTC 时间>.jsonl`），这个话题不读回（`topic_log_unreadable`）。

读回是一次 `patchouli.restore_topic_pool` 操作，发布开始与完成事件，报告回到话题池的话题数（`restored`）、待重新发起最后一次沉淀的退役话题数（`retired`）与进入 rejected 的日志数（`rejected`）；有日志进入 rejected 时状态为 `completed_with_rejections`、级别 error。重新发起的结果另记一行日志。

手动沉淀并退役（公开路由 `patchouli.public.topic.settle_and_retire`）返回 `TopicSettleResult`，通过可选 `generation_task_id` 与派生的 `generation_submitted` 表达是否建立后台任务；无任务不等于生命周期失败。手动删除话题（公开路由 `patchouli.public.topic.delete`）返回 `TopicDeleteResult`，明确“不沉淀、不写记忆”，适合用户主动丢弃短期话题。两种手动操作互不混用。Patchouli 业务结果不再放在 `services/perception.py`，server 层只在 HTTP 边界投影为响应模型。读回报告 `runtime.models.RestoredTopicPoolReport` 只供运行时观测，不进入 HTTP 链路。

## 8. 当前限制

- 话题的日志不 fsync（第 7 节）：断电可能丢失最后几行，最后一行不完整时丢弃；生成工作、写入意图登记与任务历史仍只在正常关闭时写盘，崩溃时随进程丢失，读回的话题因此可能等不到崩溃前在途的生成，由重新沉淀补上；读回时话题停下多久从头计时，驻留顺序只近似上次运行的访问顺序；话题的附件绑定随话题保存，但 WorkspaceAsset 在关闭时清理，重启后提升附件按尽力而为跳过；
- 沉淀进度只在生成完成时记下：生成已经写下记忆、日志还没记下完成时崩溃，或关闭时取消了还没开始写入的沉淀，重启后重新沉淀，复述已写下记忆的草稿交给查重；话题池的“有未沉淀的交互”按日志中的进度判断，沉淀生成期间仍显示为有未沉淀的交互；
- 写日志是同步写，占着事件循环：一条 `interaction` 记录带着整轮材料，长轮次可能有几 MB；
- 话题在退役前保存全部交互记录，折叠与沉淀都不释放内存，日志只增长、退役后整个删除；进程内没有容量上限以外的背压，话题的寿命只受容量与手动沉淀并退役约束、不受重启限制，单个话题的内存占用、日志大小与沉淀材料随轮次增长，启动时重放全部日志，每次沉淀都保存一份覆盖全部交互的来源 InteractionArtifact；
- 降级到不认识话题目录的旧版本时，日志既不读取也不删除；再升级会读回降级之前的日志，话题与沉淀进度停在降级那一刻，降级期间沉淀过的交互会再沉淀一次；
- token 统计是对整轮材料（只截断工具结果）的估算，不是模型级精确 tokenizer 预算；
- 块的 token 统计计入完整的思考、工具参数与截断后的工具结果，工具密集的轮次会更早触发折叠；单个轮次没有总量上限，用模型写折叠摘要时的近期事件同样包含这些材料，输入越大越可能触及 `summary_timeout_seconds` 或超出模型窗口而回退为规则摘要，规则摘要只列工具名；
- `fold_retain_recent_blocks` 只限制 block 数量，不保证保留后缀的 token 总量低于阈值；单个超大 block 也可能超过软水位线；
- 折叠配置与 Engine 的折叠选择都拒绝小于 1 的 retain 值，折叠后工作视图至少保留一个最新 block；
- 被折叠的轮次只以有损的 `fold_summary` 进入生成材料，原文只保存在来源 InteractionArtifact 中，生成不回读；
- 折叠摘要的生成在 lease 持有期间进行：等待模型时不阻塞事件循环，但该话题一直被占用，同一话题的后续提交在队列中等待，手动沉淀并退役会遇到忙碌；
- 维护项（折叠与写一条进展记录）占用提交队列的一个执行槽（共 4 个），一次最多占用摘要时限（默认 60 秒）；多个话题同时维护时，其他话题的提交要等执行槽；写一条期间同一话题新到的交互排在它之后等待，与折叠相同，虽然写一条本身不需要排在交互之间；
- 进展记录只在话题停下够久时写：会话切到别的话题、一轮之内都不写；话题停下多久看驻留的访问时间，为写入确定话题时也会刷新，跨重启后从读回时重新计时。旧条目全部保留、全部进入写下一条的输入，没有合并或截取，长话题的输入随条数增长，话题每停下够久一次就是一次模型调用；写一条失败后，同一位置只在本进程内不再重试，重启后照常再写。`summary_writer` 不是 `llm`、或没有配置 `patchouli.llm` 时不写进展记录。进展记录经话题列表的 HTTP 响应交给话题池，经话题句柄交给 agent（第 7 节）；退役之后的话题读不到；
- 折叠项排在已入队的同一话题提交之后：同一话题积压多轮时，这些轮次都写入之后才折叠一次，这次的摘要输入相应更大；Import Bus 每个外部会话同时只有一轮在途，积压通常很小；
- WorkingSet 的容量检查与候选选择运行在单事件循环内（方法同步、不内部 await），跨线程使用需调用方先行串行化；两个并发 NEW_TOPIC 的容量检查存在与旧实现等价的短暂竞态窗口；
- 话题停下后沉淀没有端到端用例（旧的用例随旧感知层删除而移除，重建需要真实 LLM/Qdrant 栈）；当前回归由 Familiar 集成测试（tests/integration/patchouli/test_perception.py）覆盖；
- 旧文档中的 `assistant_message` fallback、扁平 `context_messages` 和 Perception 私有 `InteractionPayload` 均已退出主路径。

调整这些语义时必须同时检查 Generation、Artifacts、Import Bus、shutdown drain 与启动时读回，因为“何时丢弃交互记录”本质上是数据耐久性边界，而不只是一个摘要算法参数。

token-aware 保留、summary-only Topic 与折叠摘要优化等议题见[短期记忆 Idea](../ideas/short-term-memory-and-context-compaction.md)第 6 节；完整资料与来源保存见 [Raw Evidence Idea](../ideas/PatchouliPageFoldingRawEvidenceDesign.md)第 9 节；长 turn 与跨入口专项见[Turn 内上下文折叠 Idea](../ideas/long-running-agent-intra-turn-context-folding.md)第 14 节。
