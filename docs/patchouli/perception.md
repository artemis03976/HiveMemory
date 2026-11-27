---
title: Patchouli Perception
status: current
owner: patchouli
scope: structured-ingestion
code_paths:
  - src/hivememory/engines/perception/
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/patchouli/control/interaction_submission.py
  - src/hivememory/patchouli/application/interaction_submission_service.py
  - src/hivememory/core/protocol/sealing.py
  - src/hivememory/core/models/interaction.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/system/import-bus.md
related_docs:
  - docs/architecture/short-term-memory.md
  - docs/workspace/README.md
last_reviewed: 2026-11-27
---

# 感知：交互的接收

Perception 负责把一轮已经发生的交互转化为 Patchouli 可以持续维护的短期事实：保存结构化 turn，并把它追加进话题。它不猜测用户下一步要做什么，也不直接决定某段文字应成为哪条正式记忆。话题怎样记下进展、沉淀、退役、折叠与读回前台，见[短期记忆](../architecture/short-term-memory.md)。

“感知”在这里更接近记忆管理单元，而不是第二个 Gateway。Gateway 先解释入口，Alice 先完成 run，Perception 随后观察结果。这个先后关系防止记忆系统为了整理历史而重新取得入口决策或 Agent 执行权。

## 1. 模型分层

### 1.1 内容事实：TurnRecord

一轮交互的内容真相由 `core.models.TurnRecord` 表达，只有四项：本轮结局 `outcome`、`turn_id`、产出本轮的 `ActorIdentity`（不含 session），以及以序号 0 的用户消息开头的有序 `TurnEvent` 集。事件集是唯一的内容事实：用户消息取序号 0 的事件正文，回复正文是全部助手正文按序号以空行拼接，二者都在读取时从事件得到，不在记录中另存一份。

`TurnEvent` 有用户消息、助手正文、思考、子 Agent 交回的消息与工具调用五种；一次工具调用与它的结局是同一个事件，模型产生的事件记下来自本轮第几次模型响应（`response_index`），字段与一轮的不变量见[数据模型](../architecture/data-model.md#49-交互记录turnevent-与-interactionpayload)。事件集之外不保存由它归并的动作或轨迹：同一事实存两份时，两份会各自漂移，读取方也会从不同来源得到不一致的解读。Perception 不会从一段拼接文本逆向解析工具事实。

`TurnRecord.outcome` 保存 completed/cancelled/failed（默认 completed），从 `InteractionPayload` 原样转入。中断和失败只要有输出，也沿同一提交与幂等应用链形成 block；结局不是成功知识的标志，生成转写会注明不完整。会话上下文属于 workspace，Topic 仅作为记忆材料，不反向填充 Alice 的历史。

### 1.2 感知容器：LogicalBlock

`LogicalBlock` 在 TurnRecord 外只增加感知层元数据：block id、created time 与估算 token（`total_tokens`，第 3 节）。用户消息与回复正文是从事件读取的只读属性。它是不可变模型，表明同一轮内容事实进入短期话题后的感知视图；Gateway 的入口判断（意图、改写后的查询）不进入块。

可以简写为：

```text
LogicalBlock = TurnRecord + perception metadata
```

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
  -> needs_fold：提交队列排入折叠项 -> fold_topic（折叠，见短期记忆第 7 节）
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

目标话题属于提交前的路由决定，而不是应用时重新计算的感知结果：任务进程取自 Gateway 的话题路由，Import Bus 取自为这一轮确定的话题或提交前的 `PASSIVE_MEMORY` 路由，以 `requested_topic_id` 提交（[短期记忆](../architecture/short-term-memory.md#5-一轮交互属于哪个话题)第 5 节）。目标为 `NEW_TOPIC` 时，Perception 以提交携带的 `new_topic_title`/`new_topic_summary` 建立话题，缺省时使用默认标题“新建话题”与空摘要；后续轮次的路由依赖候选话题的标题与摘要，因此 Import Bus 新建话题时带上 Gateway 给出的标题与摘要。应用失败的交互没有写入任何话题，重新提交时按提交方这一次交来的目标话题应用。Import Bus 拥有外部会话的串行、查重与确认，Gateway 只负责入口判断，两者都不持有话题内容。

## 3. MemoryPerceptionEngine

`MemoryPerceptionEngine` 是无状态的摄入与折叠算法引擎，职责只有三项：

- `build_block`: 把 `InteractionPayload` 转化为不可变 `LogicalBlock` 并估算 token。块的 `total_tokens` 是对整轮渲染材料的估算：用户消息、助手正文、思考、能力调用的一行摘要与外部工具步骤，只有外部工具结果按 `patchouli.generation.transcript.max_result_chars` 截断，折叠阈值因此直接反映生成实际看到的材料；能力调用读回的记忆不渲染，也不计入；
- `should_fold`: 判断话题工作视图的 token 是否超过折叠阈值；
- `select_blocks_to_fold`: 选择待折叠的旧 blocks 前缀（保留最近 N 个）。

什么时候折叠、折叠摘要与进展记录的两个写作者，见[短期记忆](../architecture/short-term-memory.md#44-折叠)第 5、7 节。

引擎总是建立，感知没有关闭开关。Engine 不持有 Store / Queue 与写作者，也不导入 `hivememory.patchouli.*`，可被其他 runtime 复用、可纯单元测试。

## 4. 当前限制

- token 统计是对整轮材料（只截断工具结果）的估算，不是模型级精确 tokenizer 预算；
- 块的 token 统计计入完整的思考、工具参数与截断后的工具结果，工具密集的轮次会更早触发折叠；单个轮次没有总量上限，用模型写折叠摘要时的近期事件同样包含这些材料，输入越大越可能触及 `summary_timeout_seconds` 或超出模型窗口而回退为规则摘要，规则摘要只列工具名；
- 旧文档中的 `assistant_message` fallback、扁平 `context_messages` 和 Perception 私有 `InteractionPayload` 均已退出主路径。
