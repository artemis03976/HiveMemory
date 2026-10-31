---
title: Patchouli Memory Generation
status: current
owner: patchouli
scope: memory-materialization-and-background-tasks
code_paths:
  - src/hivememory/engines/generation/
  - src/hivememory/infrastructure/llm/base.py
  - src/hivememory/prompts/transcript/generation.py
  - src/hivememory/engines/perception/material.py
  - src/hivememory/i18n/prompts.py
  - src/hivememory/patchouli/services/memory_generation.py
  - src/hivememory/patchouli/control/memory_generation/
related_contracts:
  - docs/contracts/mtp.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/subsystem-contracts.md
related_docs:
  - docs/workspace/README.md
last_reviewed: 2026-10-31
---

# 记忆生成

记忆生成是把短期材料或显式保存意图转化为正式 `MemoryAtom` 的冷路径。它必须同时避免两种对称错误：把每段对话都机械保存，会让书库迅速被噪音淹没；把保存完全交给不透明的 LLM 判断，又可能让用户明确要求保留的内容悄悄消失。

当前设计因此保留三种生成模式，并把“构造任务”“执行生成”“写 artifacts 和持久化”“发布任务/settlement 终态”拆成不同层。生成不再是 Perception 的一个回调，也不再由一个 LibrarianCore 同时管理计算与后台任务。

每个 `MemoryGenerationTaskSpec` 都独立携带创建任务时冻结的 `belong_to: WorkspaceIdentity`、`from_actor: ActorIdentity` 和 `topic_id`；任务快照沿用相同字段，不保存 `IdentityScope` 或访问 context。Generation 先限定资源归属，再按发起者读取策略查重，不从进程当前上下文重新推导 Workspace 或权限。WorkspaceAsset 不属于 Generation，生成链只接收 settlement 交接中已经冻结的资产绑定事实；Store 自身的归属与访问接口拆分见[后续待办](../todo/workspace-asset-ownership-identity-split.md)。

提取阶段选择 LLM 而不是固定规则，是因为“是否值得长期保存”以及如何把自然语言重组为 title、summary、content 与 tags，本质上依赖跨表达方式的语义判断；只靠关键词或消息计数会把调用方措辞固化成记忆结构。代价是延迟、成本与非确定性，因此它被放在后台冷路径，并为显式 WRITE 保留确定性的 fallback；LLM 可以提出记忆草稿，但不能独自决定任务终态、持久化成功或用户承诺。

## 1. 控制面与数据面

```text
MemoryGenerationCoordinator
  -> MemoryGenerationTaskSpec
  -> MemoryGenerationTaskController
       -> stable task_id
       -> private queue work envelope
       -> MemoryGenerationQueue / TaskHandle
       -> local route GENERATION_EXECUTE_SPEC
  -> MemoryGenerationFamiliar
       -> MemoryGenerationEngine.process
       -> ArtifactEngine
       -> MidTermMemoryStore.upsert
       -> MemoryGenerationResult
  -> PendingAtom settlement + RuntimeEvent
```

- Coordinator 把 settlement 或 PendingAtom materialize task 变成统一 spec；
- TaskController 创建不可变 work 并持有 typed handle，负责领域快照、取消、等待与事件时机；
- MemoryTaskEventEmitter 负责把领域快照投影为 `memory.task.*` RuntimeEvent；
- Familiar 是生成数据面，执行 compute -> artifact -> persist；
- GenerationEngine 负责提取、合并、去重和构造 outcome，不直接持久化或发布事件。

这条边界使 GenerationEngine 的返回值可以在落库前被补充来源信息，也让取消/失败只由 TaskController 写入一次终态。

## 2. 统一输入视图

Generation 不再消费拼接的 `context_messages`。`GenerationTranscriptBuilder` 从 `LogicalBlock[]` 构造：

```text
GenerationContext
  ├─ state_summary
  └─ GenerationTurn[]
       ├─ outcome（completed/cancelled/failed）
       ├─ user_message
       ├─ material_entries（用户消息之后的材料条目，只有工具结果按上限截断）
       └─ identity
```

它保留对记忆提取有意义的语义与过程材料，丢弃能力调用读回的内容（READ、SEARCH 读回的已有记忆会被重复提取）与工具结果超出上限的中段。Alice 历史从会话 payload 重放，记忆生成从 Topic 的 TurnRecord 取材，两者各自生成适合任务的视图；Generation 不能反向依赖 Alice 的消息渲染格式。

### 2.1 材料与渲染

每轮的材料由感知层唯一的渲染函数（`engines/perception/material.py`）从事件集得到，Generation 只组装话题状态与各轮次。同一个函数也用于 LLM 接力摘要的近期事件与逻辑块的 token 统计（见[感知与短期话题](./perception.md)），三处按同一个工具结果上限得到同样的材料。

用户消息之后的事件按序号就是材料的顺序，助手正文与过程条目按出现位置穿插：

| 事件 | 渲染 |
|:---|:---|
| 用户消息 | `[User]: …`，在轮首 |
| 助手正文 | `[Assistant]: …` |
| 思考 | `- THOUGHT: …`；空白的思考（例如只有签名的思考块）不渲染。思考只按先后排列，不归属某个工具调用；Alice 当前不产生思考事件 |
| 能力调用（带动词） | 一行摘要，不渲染返回原文：`- SEARCH: "查询"`、`- READ: a, b`、`- RUN: 目标 (状态)`，没有结局时状态为 `no result`；WRITE、UPDATE、CALL 不渲染 |
| 外部工具调用（带工具名） | `- TOOL: 工具名 参数`，参数是完整的 JSON；有结局时其后一行 `RESULT: …`，出错时为 `RESULT (error): …` |
| 无法解析的 MTP 指令 | 不渲染 |

能力调用的规则不区分来源：外部 harness 经 HiveMemory 的 MCP 工具检索与读取时，连接器把调用写成同样的 `SEARCH`、`READ` 与作用对象（[MCP 记忆访问](../contracts/mcp.md)），同一次读取无论来自 Alice 的 MTP 指令还是外部 harness，渲染相同，读取显示所读的 alias。外部工具的结局是对外部系统的观察，是记忆提取的重要事实来源。调用没有结局表示结局没有记录：执行者拿到结果之前被取消，或来源没有记下结果（Claude Code 的会话记录会漏记并行调用的结果），因此渲染不断言调用没有完成。

一轮交互的内容不受任何总量上限约束：思考、工具参数与一轮的全部条目原样渲染，不降级、不省略。只有单条外部工具结果按 `patchouli.generation.transcript.max_result_chars`（默认 800）截断：工具输出可能是整份文件或日志，其余内容是对话与推理本身，任何一个上限数字都不应截掉一轮的过程。截断保留首尾、省略中段并以 `…[N chars omitted]…` 标出省略的字符数，长输出末尾的报错因此得以保留。

截断在构建 `GenerationContext` 时一次完成，渲染为提示词文本时不再截断，多行条目的后续行缩进到条目之下。

Mode B/C 的背景同样经这个构建器生成。三种模式的提取提示词（中英文各一份）都说明材料格式：每轮以 `[User]` 开头，其后按发生顺序排列 `[Assistant]` 正文与以 `- ` 开头的过程条目；THOUGHT 是 Agent 的推理，不能作为事实依据（Mode A 中主要由它得出的内容降低置信度）；TOOL/RESULT 是外部工具的调用与返回，可以作为依据，`(error)` 表示调用出错；SEARCH/READ/RUN 是记忆系统内部操作的一行摘要，Mode A 另外提示读回的已有记忆不要重复提取；工具结果中省略标记处的内容不要推测。提示词同时要求不把密钥、令牌、密码、证书或连接串中的凭据写入记忆，需要记录相关配置时只保留名称与用途。

渲染与截断规则的回归入口为 [材料测试](../../tests/unit/engines/perception/test_material.py)，转写格式见 [生成视图构建器测试](../../tests/unit/engines/generation/test_generation_transcript_builder.py)。

`GenerationTranscriptBuilder` 将 `TurnRecord.outcome` 传给 `GenerationTurn`；cancelled/failed 在转写的轮首附 `[Outcome: ...]` 和未正常完成的说明，避免把半截回复视为完整结论。

## 3. 三种生成模式

### 3.1 Mode A：被动观察 / ARCHIVE

当 idle、LRU、shutdown 或 manual settle 形成 `TopicMaterializeTask` 时，Coordinator 用其中的工作视图（未折叠的 blocks 与 `state_summary`）构造 GenerationContext，被折叠的轮次只以摘要进入生成；同一任务的 `InteractionArtifactInput` 携带话题的全部交互记录与折叠记录，来源 InteractionArtifact 因此覆盖已折叠的轮次（第 5 节）。没有 turns 时跳过任务。

GenerationEngine 渲染 transcript，调用 extractor 判断长期价值并生成 `ExtractedMemoryDraft`；没有草稿或 `has_value=false` 时不产生记忆。有效草稿随后进入 dense top-1 查重。

Extractor 经组合根注入的共享 LLM 服务异步调用模型，请求参数由 `shared.llm.librarian` 转换而来，等待模型期间不阻塞事件循环；Librarian 未配置模型时 extractor 以 No-Op 运行。它区分两种结果：模型输出为空或无法解析为草稿是“没有产出”；提供商失败只重试一次，耗尽后抛出 `ModelProviderError`。Mode A 不捕获该错误，生成任务以 `memory_generation_failed` 失败收尾，不再把提供商故障记录成“对话没有价值”。超窗、程序错误与取消不重试，原样传播；分类见[错误模型](../contracts/error-model.md#46-模型调用错误)。

Mode A 的“被动”指没有显式 WRITE/UPDATE focus，而不是同步发生在用户响应内，也与 Import Bus 无关：导入的轮次与主动链路的轮次一样进入话题，之后经 settle 参与 Mode A。它仍作为 Patchouli 后台 memory task 执行。

四种 settle 入口（manual / idle / LRU / shutdown）共享同一发起者裁定：Coordinator 按 `belong_to` 构造 `user_id=owner_user_id`、`agent_id=SYSTEM_AGENT_ID`、`team_id=None` 的 `from_actor`，不继承最后访问者或驱逐触发者。`system` 不属于 Team，也不能成为 PRIVATE policy 的目标，普通读取策略因此把 SETTLE 查重限定为本 Workspace 的 PUBLIC 记忆，避免后台结算触碰某个参与 Agent 才可见的 PRIVATE/TEAM 记忆。新建记忆使用 PUBLIC policy、`source_agent_id=SYSTEM_AGENT_ID` 与 `SYSTEM` creation intent；实际参与内容的 Agent 由 `contributing_agent_ids` 从 block identity 聚合（去重、保持首次出现顺序、不含 `system`）。

Mode B 新建记忆以提交 WRITE 的 actor Agent 为来源，贡献者先记录发起 Agent 本身（无上下文的主动写入仍由发起 Agent 产出），再合并上下文轮次贡献者。WRITE/UPDATE 保留物化意图的 `from_actor`，按该 actor 的普通 PUBLIC/TEAM/PRIVATE 可见性读取，不能沿用 SETTLE 的 system 身份。版本更新（dedup 合并、Mode C）保留已有 `meta.provenance` 的来源主体字段，并把本次生成的贡献者并入已有集合，合并结果随 MemoryAtom 与对应 MemoryVersionArtifact 持久化。来源与贡献者字段只记录 provenance，不参与读取授权。

四种触发的发起者、SETTLE/WRITE 查重可见性与完整空闲结算持久化由[结算身份集成测试](../../tests/integration/patchouli/test_settlement_identity.py)验证。

### 3.2 Mode B：主动 WRITE

Alice 的 MTP WRITE 经线程 context 绑定的操作请求进入 workspace 能力层，在共享 `WriteIntentRegistry` 登记 MATERIALIZING 意图，并立即调用 Patchouli 的 `memory_intent.submit`。服务沿用登记的 `intent_id` 与 `pending_alias` 构造内部 `PendingAtomMaterializeTask`，生成队列接纳后才返回 ACK。意图登记与终态由 workspace 持有，正式记忆生成与任务状态由 Patchouli 持有。

Coordinator 使用话题的 `state_summary` 与过去 N 个 block 作为背景，以 `WriteFocus` 为保存核心。N 由 `patchouli.generation.active_recent_blocks` 配置（默认 5，允许 0），与感知折叠保留轮数无关。派发发生在交互封口前，当前轮不进入背景；新话题为空时只依赖 focus。话题不存在时按空材料生成，话题读取抛错时因尚无任务入队，逐项结算 FAILED 并拒绝。

Extractor 遇到提供商失败（重试耗尽）或输出无法解析时，Mode B 会直接从 WriteFocus 构建 fallback draft，保证明确保存意图不会因为 Librarian LLM 暂时失败而无声丢失；程序错误不进入 fallback，任务按失败收尾。Fallback 仍要经过去重，因而最终可能 CREATE、UPDATE、TOUCH 或 DISCARD；ACK 从未承诺“一定新建一条独立记忆”。

### 3.3 Mode C：主动 UPDATE

UPDATE spec 构建时，Coordinator 解析 `UpdateFocus.base_uuid` 并通过 local route 读取现有 MemoryAtom。UUID 非法或目标不存在只使该 active spec 失败，并发布对应 pending failure，不阻断同批其他任务。

Engine 用 extractor merge 生成新正文与 changelog；merge 遇到提供商失败或输出无法解析时：

- focus 携带 content：把内容作为带日期的更新段追加到旧正文；
- 只有 instruction：保留旧正文，记录无内容变更的 fallback changelog。

内容合并结果与现有正文一致时不创建新版本——outcome 降级为 TOUCH，只经受限 patch 推进访问统计。真实内容变化时，Familiar 在提交边界（一次取 `now`）推进 `meta.version` 与 `updated_at`/`decay_anchor_at`、重置置信度，前后完整快照交给 Artifact 层生成 MemoryVersionArtifact 后写回中期库。

## 4. 去重与演化

Mode A/B 使用 `title + summary` 对中期库执行 dense top-1 搜索，Deduplicator 根据相似度与内容关系返回：

| 决策 | 当前行为 |
|:---|:---|
| `CREATE` | 从 draft 构造新 MemoryAtom 与 canonical alias |
| `UPDATE` | 用 draft 内容覆盖现有 head；提交边界递增 version 并写版本 Artifact；合并结果与现有内容一致时降级为 `TOUCH`（§3.2：无变化不增版本） |
| `TOUCH` | 经受限 `patch_payload` 推进 access count 与 last_accessed_at；不改内容时间、不升版本、不重算向量 |
| `DISCARD` | 不写 MemoryAtom，返回低质量重复消息 |

`TOUCH` 与 `UPDATE` 都作用于查重命中的已有记忆，Deduplicator 必须随决策返回该记忆；违反这一约定时 GenerationEngine 显式失败，不以缺失的记忆继续构造结果。

去重 UPDATE 当前使用轻量覆盖，不调用强合并 prompt；这是刻意保守的当前实现，但也意味着 draft 质量直接决定新 head。历史由 MemoryVersionArtifact 保存，不能通过向正文不断追加旧版本来模拟版本控制。

Alias 由 `AliasGenerator`（`engines/generation/alias.py`）按 memory type 前缀和 extractor 给出的 suffix 构造候选，例如 `code_* / fact_* / url_* / ref_* / user_* / wip_* / agent_*`；suffix 清洗后为空时改由 title 派生，仍为空则该记忆没有 alias。候选先经中期库确认在同一 Workspace 内空闲，被占用时依次追加 `_2`、`_3`…… 消歧后缀重试，全部被占用时抛 `MemoryAliasConflictError`（`reason=alias_candidates_exhausted`）。生成侧只负责让正常路径避开冲突；同一 Workspace 内 alias 唯一的不变量由中期库写入前校验兜底，见 [MemoryLibrary](./memory-library.md)第 1.2 节。Pending alias 只在 Alice 运行期定位候选意图；settlement 后的 canonical alias 才属于正式 MemoryAtom。

## 5. Artifact 与持久化顺序

MemoryGenerationFamiliar 执行：

1. 尝试捕获 InteractionArtifact：话题结算时覆盖全部交互记录并带折叠记录，主动生成时只有最近的 blocks、没有折叠记录；
2. 在提交边界取一次 `now`，调用 GenerationEngine（传入同一时点用于内容日期）得到 outcomes；
3. 对每个非 DISCARD 的 outcome 依次完成：提交字段分配（UPDATE 递增 version、推进内容时间与衰减基准、重置置信度）→ 构建 creation/version artifacts（强制，`now` 贯通 `changed_at` 与事件时间）→ 挂载 refs/events；
4. 执行 MidTerm `upsert(atom, recompute_vectors=<embedding 输入是否变化>)`；TOUCH 走受限 `patch_payload`；
5. 把 outcome 投影为 `MemoryGenerationResult` 与可选 PendingAtomSettlement，并通知变更。

InteractionArtifact 在生成之前捕获，失败只记录 warning、生成照常进行。因此提取失败（包括超窗）或结果为 TOUCH、DISCARD 时，原文仍保存在 InteractionArtifact 中，只是没有记忆引用它。

`InteractionArtifactInput.asset_bindings` 随 settlement task 一起冻结并穿过队列边界；它是 Topic 已确认使用过的资产关系，而不是 Artifact 本身。CREATE/UPDATE 持久化后，Familiar 通过注入的 `WorkspaceAssetMaterializationReaderPort`，按归属与 `asset_ref` 租借 READY representation、创建独立 DocumentArtifact 并释放租借；失败按 best-effort 跳过，不回滚已提交的 Memory。该端口不携带 actor 或 scope，Store 的既有访问接口适配由组合根承担；详情见[Artifacts 与来源追踪](./artifacts.md)。

**版本记录是提交成功的前置条件**：版本存储关闭（NoOp builder）或构建失败时任务直接失败，不发布无历史的新 canonical；canonical upsert 失败同样使任务失败，已写入的版本记录保留为孤立 Artifact。`recompute_vectors` 由提交前后 embedding 输入（index 的 title/memory_type/tags/summary）对比决定，仅 `payload.agent_config` 等非 embedding 变化不重算向量。详见[Artifacts 与来源追踪](./artifacts.md)。

### 5.1 重试边界

Memory Generation 的业务 handler 固定为单次 attempt。整条数据面包含 artifact 写入、LLM 生成和
`MemoryAtom` upsert 等副作用；在这些步骤部分成功后无法确认外部状态时，自动重放可能制造重复记忆或重复
artifact。因此 `TimeoutError`、`ConnectionError`、模型异常以及普通业务异常都会直接把任务投影为
`FAILED`，不会由 Memory Generation lane 自动重试。

这不等于系统完全没有重试：通用 Work Queue Runtime 仍保留 retry 状态机，Interaction Submission 也只对
明确标记的瞬态提交错误做有限重试；Generation extractor 内部的 LLM 边界重试属于另一层策略，只对提供商
失败重试一次，超窗、程序错误与取消不重试。当前
`FAILED` 只表示本次任务未完成，不保证没有任何外部副作用。Active WRITE/UPDATE 已使用 PendingAtom
`intent_id` 派生稳定 task/work identity：进程内重复 dispatch 会复用已有任务，且相同 identity 携带不同
spec 会被拒绝。跨重启结果记录、数据面内部副作用幂等和 reconciliation 仍留待后续持久化/I2 阶段处理。

手工 memory create/update 同样通过 Familiar，两者都只接收调用方提交的字段材料：创建由 Familiar 构造原子，记忆 ID、来源记录（取本次发起者）、默认访问策略、生命周期与完整时间戳都在提交边界确定，字段非法以 `InvalidMemoryFieldError` 拒绝，类型相关约束与 alias 唯一性在写版本记录之前检查；记忆管理入口不接受 `AGENT_PROFILE`，Profile 只经 Profile 管理入口创建，类型由入口固定。

类型相关约束有两条：`AGENT_PROFILE` 必须带 alias（按 alias 寻址与 CALL）；`payload.agent_config` 只允许 `AGENT_PROFILE` 设置。`agent_config` 的唯一读取方是 Profile 解析，写在其他类型上只会生成无人使用的内容版本，调用方也得不到字段无效的反馈。记忆与 Profile 共用编辑路由，HTTP 的 `PATCH /api/v1/memories/{id}` 同时是前端保存 Agent 的入口，入口无法按类型区分，因此由 Familiar 在提交边界按原子类型检查，违反时以 `InvalidMemoryFieldError` 拒绝且不写版本记录。约束放在提交边界而不是 Memory 模型：模型约束同样作用于读取解码，收紧会让既有不合规记录无法读取。

编辑捕获 before snapshot，仅当传入值与当前值不同（值相等判定）才递增 version 并生成 MANUAL_EDIT version artifact，无变化调用原样返回。

## 6. 后台任务与终态

`MemoryGenerationTaskSpec` 是控制面与 Familiar 共享的业务输入。Controller 只在队列边界内把
`task_id + spec` 包装为私有不可变工作信封；该队列适配细节不再作为一份公开领域模型。公开 API 使用
`MemoryGenerationTask` 只读快照表达下列领域状态：

```text
PENDING -> RUNNING -> COMPLETED | FAILED | CANCELLED
```

私有工作信封 codec 的 `schema_version` 当前为字符串 `"1.1"`，`spec` 独立编码归属与发起者，只注册当前版本。解码校验完整字段集与嵌套对象的规范往返，拒绝旧 scope 布局、未知版本、额外字段与身份篡改。底层仍是 `InMemoryWorkStore`；版本化载荷用于进程内冻结和一致性校验，不代表已有持久化队列或旧任务迁移能力。

`WorkRecord` 是执行状态的唯一真相源，Queue 的 `TaskHandle` 提供 snapshot/wait/cancel；
TaskController 只在查询时映射非终态快照，并在 work 终止后执行一次 settlement 与终态事件。
提交返回的 `MemoryGenerationTask` 不会原地更新，调用方需要通过 list/get/wait 获取新快照。

Familiar 内部以 Generation Engine 返回的 `GenerationOutcome` 完成 compute、artifact 与 persist；只有这些
步骤成功退出后才投影为跨域 `MemoryGenerationResult`。该结果仅保留 canonical identity 和可选的
`PendingAtomSettlement`，不会把 `MemoryAtom`、`DuplicateDecision`、before snapshot 或 changelog 泄漏到
Controller。由此，Engine 计算语义与 Patchouli 已持久化事实保持明确边界。

控制面提供 list/get/cancel/wait/wait_many/wait_all。`wait_task()` 直接返回
`MemoryGenerationTask | None`，`wait_many()` 按请求顺序返回快照或 `None` 的列表，不再复制一层
WaitResult/WaitSummary。等待使用 `asyncio.shield`，超时返回当前 `PENDING/RUNNING` 快照，不会接管或
自动取消后台任务；仅 shutdown drain 会把超时窗口后仍未终结的快照汇总为观测计数，并显式 cancel 那批任务。

任务状态通过 RuntimeEvent 发布。TaskController 只决定生命周期事件的发生时机，`MemoryTaskEventEmitter` 集中选择 `memory.task.*` 类型并组装稳定 payload；通用队列仍独立发布 `work.*` 基础设施事件，两者不互相替代。主动任务完成后，settlement 会通过 PatchouliBridge 转发为全局 PendingAtom event，由 workspace registry 更新登记终态；发布失败不会回滚已持久化记忆，Controller 会 best-effort 发布 pending failure。PendingAtom settlement、failed、cancelled 仍属于功能事件并继续留在 TaskController；当前没有事件未送达补偿或跨重启恢复，不能保证 registry 在通知丢失后自行补齐终态。

PendingAtom settlement 只更新意图状态，不回填 canonical 缓存。中期 Store 的独立变更事件负责原子、alias 与 Profile 派生项失效，所有正式记忆重新读取都经过当前授权；两种事件的作用见[MemoryLibrary](./memory-library.md#12-中期当前可检索书库)。

## 7. 实时派发与交互 finalize

主动写入与交互提交分别执行：

```text
WRITE/UPDATE -> workspace 登记 -> memory_intent.submit
  -> Coordinator 构建材料与 spec -> Controller 接纳生成任务 -> ACK
  -> 后台生成、持久化与结算事件

CPU 有输出（completed/cancelled/failed）-> 封口 InteractionPayload -> 同步写入会话 -> finalize
  -> submit as-is -> wait until interaction applied
  -> best-effort record prepared retrieval HITs -> None
```

每次提交只携带一个意图：Coordinator 的 `submit_active()` 为它构建 spec，交给 Controller 的 `admit_generation()`，返回单个 `MemoryGenerationAdmission`。accepted 携带新建或幂等复用的任务快照；rejected 由 Controller 结算对应 PendingAtom failed，不伪造 FAILED 任务；unknown 不结算，可继续由实际接纳的任务给出终态。Coordinator 在构建 spec 时遇到确定性失败（如 UPDATE 基础不存在）同样结算 failed 并返回 rejected，遇到无法判断的失败或接纳请求本身失败则返回 unknown。稳定的 `active:{intent_id}` 是任务标识，同一 intent 携带不同 spec 重放时结果为 unknown，既有任务保留结算权，不能抢先发布 failed。

Interaction applied 仍是 Chat 的硬成功边界，finalize 不派发主动任务，也不返回任务列表。预检索 HIT 失败只记录 warning，不反向改写 Chat 终态。Finalization continuation 由 Patchouli 进程级持有，HTTP/SSE 调用方取消不会中止已经接管的提交与 post-apply 义务。CPU 失败、取消、进程关闭与失败 CALL 子帧都不撤回此前的显式写入。

主动生成不等待当前交互 applied，不读取当前轮，也不会为了 WRITE/UPDATE 调用话题 settlement、summary 或 clear。有输出轮次的当前交互留在话题中，之后可在 idle/LRU/shutdown 时参与 Mode A；主动保存与被动归档不能通过清空 buffer 相互替代。未完成进程的预建空话题仍可 cleanup，已派发任务保留该 topic_id 作观测，正式原子不保存 topic_id。

这条解耦边界是对旧链路实际问题的回应。旧实现曾把 MTP WRITE/UPDATE 当作话题结算理由，执行 `archive + compact + clear`：连续两次 WRITE 时，第二次只能看到已经被压扁的 summary；每次主动写还会额外触发一次 summary，使原始细节在明确保存意图下反复丢失。当前实现直接使用 topic 最近 context，并以 `WriteFocus`/`UpdateFocus` 驱动 Mode B/C；主动生成与被动归档保持两条正交流，主动写入后的 blocks 仍可在之后进入 Mode A。

这并不保证主动写出的正文与被动归档结果完全不重复。重复是当前可接受的成本，由 deduplicator 的 `CREATE`、`UPDATE`、`TOUCH`、`DISCARD` 决定最终落库动作；为了消除重复，不能重新让 Perception 感知 focus 或 pending 状态，否则会把控制意图污染为对话事实。当前每个主动 task 也会独立捕获 `InteractionArtifact`，因此 provenance 可能重复，这是已知的可追踪性成本，而不是重新合并任务输入的理由。

## 8. 当前限制

- 一次话题结算至多产出一条记忆：Mode A 的 extractor 对整段话题材料只返回一份草稿（`BaseMemoryExtractor.extract` 返回 `ExtractedMemoryDraft | None`），同一话题中讨论过的多项决定、事实或偏好只能压进一条记忆，次要内容往往只留在正文里。检索的稠密与稀疏向量以及重排输入都只由标题、类型、标签与摘要编译（MemoryCompiler 的 `DENSE_EMBEDDING`、`SPARSE_EMBEDDING` 目标），只写在正文里的内容因此检索不到。从一段话题材料中提取多条记忆，需要提取输出、查重与演化都改为逐条进行。真实样本见已归档的 [Import Bus 真实使用观察](../archive/todo/import-bus-real-use-observation.md)；
- 结果未知或结算事件丢失时，句柄可能长期保持 MATERIALIZING；没有对账恢复。写入一旦开始派发就不因 CPU、进程或 CALL 子帧失败取消；
- UPDATE 意图不携带登记时的基础版本，spec 构建时按 UUID 读取当时的最新版本，并发 UPDATE 不检测基础版本冲突；后续方向见[写入意图后续方向 Idea](../ideas/pending-intent-migration.md)第 6 节；
- memory work、typed result 与终态快照只存在于当前进程，重启后不可恢复；持久化与恢复边界见[耐久性与故障恢复治理](../governance/reliability/durability-and-recovery.md)；
- 终态快照与 Queue 使用同一 `terminal_retention` 上限；
- Controller 的 admission 持有同一把锁串行执行，并发提交的意图按取得锁的顺序入队；
- extractor/merge 的 LLM 调用没有独立超时，提供商长时间无响应时任务一直等待，只受取消与 shutdown drain 约束；
- 后台抽取与 chat 共用提供商配额，限流表现为提供商失败：Mode A 按任务失败收尾，Mode B/C 使用 fallback；当前没有生成侧的并发控制；
- Mode A/B 去重仍只取 dense top-1；检索调用链独立携带归属与发起者，Memory store/adapter 先强制 Workspace ownership hard filter，再重验 actor policy；其他去重策略限制仍待收敛；
- Dedup UPDATE 是直接覆盖 draft content，不是强语义 merge；
- 生成材料只截断工具结果，单个轮次没有总量上限，生成也不分片读取：工作视图保留的 blocks 很大时（例如单轮上百次工具调用），结算的提取可能超出模型窗口，任务以 `memory_generation_failed` 失败（超窗不重试），原文仍在来源 InteractionArtifact 中但没有记忆引用；
- Active tasks 复用相同 blocks 输入，但会各自写 InteractionArtifact；
- memory 版本 Artifact 失败会阻断该次内容提交（无历史不成功）；provenance 写入与 MemoryAtom 发布仍不是原子事务，canonical 失败会留下孤立 Artifact。

未来若引入 durable queue，首先要保持私有工作信封 codec、`WorkRecord` 单一状态源和 settlement 语义不变，避免基础设施升级重新把控制面塞回 GenerationEngine。
