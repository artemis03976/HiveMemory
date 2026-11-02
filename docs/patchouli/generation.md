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
last_reviewed: 2026-11-02
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
- GenerationEngine 负责提取、Mode C 修订、查重、两条记忆的合并计算和构造 outcome，不直接持久化或发布事件。

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
| 子 Agent 消息 | `[Subagent]: …`，全文渲染。子 Agent 的过程不在材料中，它结束时交回的输出是它唯一留下的内容，因此不按工具结果截断；当前只由 dsh 插件的导入写出 |
| 能力调用（带动词） | 一行摘要，不渲染返回原文：`- SEARCH: "查询"`、`- READ: a, b`、`- RUN: 目标 (状态)`，没有结局时状态为 `no result`；WRITE、UPDATE、CALL 不渲染 |
| 外部工具调用（带工具名） | `- TOOL: 工具名 参数`，参数是完整的 JSON；有结局时其后一行 `RESULT: …`，出错时为 `RESULT (error): …` |
| 无法解析的 MTP 指令 | 不渲染 |

能力调用的规则不区分来源：外部 harness 经 HiveMemory 的 MCP 工具检索、读取、写入与更新时，连接器把调用写成同样的 `SEARCH`、`READ`、`WRITE`、`UPDATE` 与作用对象（[MCP 记忆访问](../contracts/mcp.md)），同一次调用无论来自 Alice 的 MTP 指令还是外部 harness，渲染相同：读取显示所读的 alias，显式写入与更新不渲染，Mode A 不会把它们的内容再提取一遍。外部工具的结局是对外部系统的观察，是记忆提取的重要事实来源。调用没有结局表示结局没有记录：执行者拿到结果之前被取消，或来源没有记下结果（Claude Code 的会话记录会漏记并行调用的结果），因此渲染不断言调用没有完成。

一轮交互的内容不受任何总量上限约束：思考、子 Agent 消息、工具参数与一轮的全部条目原样渲染，不降级、不省略。只有单条外部工具结果按 `patchouli.generation.transcript.max_result_chars`（默认 800）截断：工具输出可能是整份文件或日志，其余内容是对话与推理本身，任何一个上限数字都不应截掉一轮的过程。截断保留首尾、省略中段并以 `…[N chars omitted]…` 标出省略的字符数，长输出末尾的报错因此得以保留。

截断在构建 `GenerationContext` 时一次完成，渲染为提示词文本时不再截断，多行条目的后续行缩进到条目之下。

Mode B/C 的背景同样经这个构建器生成。三种模式的提取提示词（中英文各一份）都说明材料格式：每轮以 `[User]` 开头，其后按发生顺序排列 `[Assistant]` 正文、`[Subagent]` 消息与以 `- ` 开头的过程条目；`[Subagent]` 是被委托的子 Agent 结束时交回的通知，带它的结局与最后的输出，按 Agent 的结论对待，与助手正文同等；THOUGHT 是 Agent 的推理，不能作为事实依据（Mode A 中主要由它得出的内容降低置信度）；TOOL/RESULT 是外部工具的调用与返回，可以作为依据，`(error)` 表示调用出错；SEARCH/READ/RUN 是记忆系统内部操作的一行摘要，Mode A 另外提示读回的已有记忆不要重复提取；工具结果中省略标记处的内容不要推测。提示词同时要求不把密钥、令牌、密码、证书或连接串中的凭据写入记忆，需要记录相关配置时只保留名称与用途。

渲染与截断规则的回归入口为 [材料测试](../../tests/unit/engines/perception/test_material.py)，转写格式见 [生成视图构建器测试](../../tests/unit/engines/generation/test_generation_transcript_builder.py)。

`GenerationTranscriptBuilder` 将 `TurnRecord.outcome` 传给 `GenerationTurn`；cancelled/failed 在转写的轮首附 `[Outcome: ...]` 和未正常完成的说明，避免把半截回复视为完整结论。

## 3. 三种生成模式

### 3.1 Mode A：被动观察 / ARCHIVE

当 idle、LRU、shutdown 或 manual settle 形成 `TopicMaterializeTask` 时，Coordinator 用其中的工作视图（未折叠的 blocks 与 `state_summary`）构造 GenerationContext，被折叠的轮次只以摘要进入生成；同一任务的 `InteractionArtifactInput` 携带话题的全部交互记录与折叠记录，来源 InteractionArtifact 因此覆盖已折叠的轮次（第 5 节）。没有 turns 时跳过任务。

GenerationEngine 渲染 transcript，调用 extractor 的话题提取（`extract`）。一段话题常包含多项彼此独立的决定、事实与经验，而检索向量只由标题、类型、标签与摘要编译，压进同一条记忆的次要内容只留在正文里，检索不到；因此模型一次输出一组记忆草稿（`ExtractionResult`），每件之后会被单独查找的事写成一条：一项决定及其理由、一个事实或配置、一个已解决的问题及其解法、一条偏好或经验、一项工作的当前状态。提示词同时要求同一件事的背景、过程与结论不拆开，顺带提及、没有结论的内容不单独成条，也不为凑数拆分；整段没有值得保存的内容时输出空列表。

`has_value=false` 的草稿被丢弃，没有草稿时不产生记忆。其余草稿按输出顺序逐条查重，各自产出一条结果（第 4 节）。

Extractor 经组合根注入的共享 LLM 服务异步调用模型，请求参数由 `patchouli.llm` 经模型注册表解析得到，等待模型期间不阻塞事件循环；Librarian 未配置模型时 extractor 以 No-Op 运行。一段话题的全部草稿共用一次调用的输出上限，Librarian 的 `max_tokens` 默认为 32768。

Extractor 区分以下结果：

- **没有产出**：模型输出为空，或无法解析为草稿列表（包括只给出一条草稿对象、没有 `memories` 列表）。整段输出按一个对象解析，任一条草稿不符合结构时整段都没有产出。
- **输出被截断**：输出达到 `max_tokens` 上限时抛出 `ModelProviderError`，不重试，因为同一请求会再次截断。截断的列表无法解析，若当作没有产出，整段话题不留记忆而任务成功结束。
- **提供商失败**：只重试一次，耗尽后抛出 `ModelProviderError`。

Mode A 不捕获 `ModelProviderError`，生成任务以 `memory_generation_failed` 失败收尾，不再把提供商故障或截断记录成“对话没有价值”。超窗、程序错误与取消不重试，原样传播；分类见[错误模型](../contracts/error-model.md#46-模型调用错误)。

Mode A 的“被动”指没有显式 WRITE/UPDATE focus，而不是同步发生在用户响应内，也与 Import Bus 无关：导入的轮次与主动链路的轮次一样进入话题，之后经 settle 参与 Mode A。它仍作为 Patchouli 后台 memory task 执行。

四种 settle 入口（manual / idle / LRU / shutdown）共享同一发起者裁定：Coordinator 按 `belong_to` 构造 `user_id=owner_user_id`、`agent_id=SYSTEM_AGENT_ID`、`team_id=None` 的 `from_actor`，不继承最后访问者或驱逐触发者。`system` 不属于 Team，也不能成为 PRIVATE policy 的目标，普通读取策略因此把 SETTLE 查重限定为本 Workspace 的 PUBLIC 记忆，避免后台结算触碰某个参与 Agent 才可见的 PRIVATE/TEAM 记忆。新建记忆使用 PUBLIC policy、`source_agent_id=SYSTEM_AGENT_ID` 与 `SYSTEM` creation intent；实际参与内容的 Agent 由 `contributing_agent_ids` 从 block identity 聚合（去重、保持首次出现顺序、不含 `system`）。

Mode B 新建记忆以提交 WRITE 的 actor Agent 为来源，贡献者先记录发起 Agent 本身（无上下文的主动写入仍由发起 Agent 产出），再合并上下文轮次贡献者。WRITE/UPDATE 保留物化意图的 `from_actor`，按该 actor 的普通 PUBLIC/TEAM/PRIVATE 可见性读取，不能沿用 SETTLE 的 system 身份。Mode C 修订保留已有 `meta.provenance` 的来源主体字段，并把本次生成的贡献者并入已有集合；两条记忆合并时沿用保留记忆的来源主体，贡献者取两者的并集（第 4.1 节）。结果随 MemoryAtom 与对应 MemoryVersionArtifact 持久化。来源与贡献者字段只记录 provenance，不参与读取授权。

四种触发的发起者、SETTLE/WRITE 查重可见性与完整空闲结算持久化由[结算身份集成测试](../../tests/integration/patchouli/test_settlement_identity.py)验证。

### 3.2 Mode B：主动 WRITE

WRITE 经操作请求进入 workspace 能力层，在共享 `WriteIntentRegistry` 登记 MATERIALIZING 意图，并立即调用 Patchouli 的 `memory_intent.submit`。两种来源走同一条链路：Alice 的 MTP WRITE 以线程 context 提交，派发话题是 CPU 分配冻结的本轮话题；外部 harness（plugin 模式）的 `memory_write` 以请求级 context 提交，派发话题是 Import Bus 在写入时为这一轮落位得到的话题，这一轮回流时落在同一话题（[Import Bus](../system/import-bus.md#47-写入时落位)）。服务沿用登记的 `intent_id` 与 `pending_alias` 构造内部 `PendingAtomMaterializeTask`，生成队列接纳后才返回 ACK。意图登记与终态由 workspace 持有，正式记忆生成与任务状态由 Patchouli 持有。

Coordinator 使用话题的 `state_summary` 与过去 N 个 block 作为背景，以 `WriteFocus` 为保存核心。N 由 `patchouli.generation.active_recent_blocks` 配置（默认 5，允许 0），与感知折叠保留轮数无关。派发发生在交互封口前，当前轮不进入背景；新话题为空时只依赖 focus。plugin 模式下同样如此：落位预建的新话题在派发时为空，落到已有话题时背景是该话题此前回流的几轮。话题不存在时按空材料生成，话题读取抛错时因尚无任务入队，逐项结算 FAILED 并拒绝。

Extractor 遇到提供商失败（重试耗尽）或输出无法解析时，Mode B 会直接从 WriteFocus 构建 fallback draft，保证明确保存意图不会因为 Librarian LLM 暂时失败而无声丢失；程序错误不进入 fallback，任务按失败收尾。Fallback 仍要经过查重，因而最终可能新建、只触碰已有记忆，或新建后被合并进已有记忆；ACK 从未承诺“一定保留为一条独立记忆”。

### 3.3 Mode C：主动 UPDATE

UPDATE 与 WRITE 的两种来源和派发话题规则相同：Alice 的 MTP UPDATE 与外部 harness 的 `memory_update` 都经 actor API 解析可读的正式基础原子后登记并派发。

UPDATE spec 构建时，Coordinator 解析 `UpdateFocus.base_uuid` 并通过 local route 读取现有 MemoryAtom。UUID 非法或目标不存在只使该 active spec 失败，并发布对应 pending failure，不阻断同批其他任务。

Engine 用 extractor 的修订（`revise`，结果为 `RevisionResult`）生成新正文与 changelog；修订遇到提供商失败或输出无法解析时：

- focus 携带 content：把内容作为带日期的更新段追加到旧正文；
- 只有 instruction：保留旧正文，记录无内容变更的 fallback changelog。

修订结果与现有正文一致时不创建新版本——outcome 为 TOUCH，只经受限 patch 推进访问统计。真实内容变化时，Familiar 在提交边界（一次取 `now`）推进 `meta.version` 与 `updated_at`/`decay_anchor_at`、重置置信度，前后完整快照交给 Artifact 层生成 MemoryVersionArtifact 后写回中期库。

## 4. 查重与合并

Mode A 的每条草稿与 Mode B 的草稿都按 MemoryCompiler 的 `DENSE_EMBEDDING` 目标编译为查询文本，对中期库执行 dense top-1 搜索。已有记忆的稠密向量由同一目标的编译文本编码，查询与入库的文本形态一致，内容相同的草稿才能达到 IDENTICAL 所需的相似度；早先查询只拼接 `title summary`，与带字段标签的入库文本形态不同，内容完全相同的草稿也只有约 0.90，触碰分支实际走不到。真实 embedding 与 Qdrant 下内容相同的草稿只触碰已有记忆，由[查重召回测试](../../tests/integration/patchouli/test_generation_dedup_recall.py)验证。Deduplicator 按相似度与内容一致性给出判断（`DuplicateDecision`），GenerationEngine 再把判断翻译为提交动作（`GenerationAction`）：

| 判断 | 条件（默认阈值） | 提交动作 |
|:---|:---|:---|
| `DISTINCT` | 没有候选，或相似度不高于下限 0.75 | `CREATE`：从 draft 构造新 MemoryAtom 与 canonical alias |
| `SIMILAR` | 相似度高于上限 0.95 但内容不一致，或落在 (0.75, 0.95] | `CREATE` 新记忆 N；满足合并条件时携带命中的记忆 M 作为合并目标，不修改 M |
| `IDENTICAL` | 相似度高于上限，标题相同且正文词集合的 Jaccard 相似度高于 0.9 | `TOUCH`：经受限 `patch_payload` 推进 access count 与 last_accessed_at；不改内容时间、不升版本、不重算向量 |

阈值由 `patchouli.generation.deduplicator` 配置。`SIMILAR` 与 `IDENTICAL` 都作用于命中的已有记忆，Deduplicator 必须随判断返回该记忆；违反这一约定时 GenerationEngine 显式失败，不以缺失的记忆继续构造结果。

查重命中后不改写已有记忆。早先的实现用草稿整体覆盖命中的记忆：同一项目的记忆增多后，同一工作的前后阶段被空闲结算拆到不同话题，后一阶段的草稿连续覆盖前一阶段的记忆，最初的设计决定只剩在版本记录里。现在命中时先把草稿保存为独立的新记忆，再由生成队列中的合并任务把两条记忆融合为一条。草稿先落库，因此关闭、队列拒绝或合并失败最多留下两条相似的记忆，不会丢失内容。合并不复用 Mode C：Mode C 的输入是一条记忆加修改指令与新材料，合并的输入是两条地位对等的正式记忆。

合并条件是 M 与 N 的访问策略相同，且两者都不是 `AGENT_PROFILE`。策略不同的两条记忆合并后会改变一方的可见范围；Profile 承载结构化配置，不能交给模型融合。生成的记忆都是 PUBLIC，策略不同只会出现在经管理入口改过策略的记忆上。不满足条件时 N 是一条普通的新记忆。

### 4.1 合并任务

Familiar 完成 N 的创建提交后，在 `MemoryGenerationResult` 中携带合并提名（`MergeFocus`：保留记忆 M 与被吸收记忆 N 的 ID；N 来自主动 WRITE 时另带这次写入的 `pending_alias` 与 `intent_id`），WRITE 的首次结算照常为 `CREATED`。Controller 在原任务成功并发布结算之后，以原任务的归属、发起者与话题提交一个 `MERGE` 来源的任务，提名放在 `GenerationRequest.merge_focus`。合并任务不持有写入意图的终态，任务标识一次性生成，失败或取消都不发布句柄事件。队列已停止或容量不足时只记录 warning，两条记忆都保留；shutdown drain 不等待合并任务，尚未执行的合并在关闭时丢失，结果同样是两条都保留。

Familiar 的合并路径：

1. 以发起者的可见性读取 M 与 N，检查合并条件，并拒绝同一条记忆吸收自身；
2. GenerationEngine 调用 extractor 的合并（`merge` 提示词）：模型只看到两条记忆的标题、摘要、标签、类型与正文，并按内容更新时间得知哪一条较新；它输出完整的标题、摘要、标签、类型、正文与一句 changelog（`MergeResult`，越界的标题与摘要按索引上限截断）。alias、来源主体与访问状态沿用 M，贡献者取 M 与 N 的并集，类型无法识别时沿用 M 的类型；
3. 写回前按复合键重新读取 M 与 N，确认两者都还存在、对发起者可读、内容版本未变，并再次检查合并条件；生命周期与访问策略以重读得到的 M 为准；
4. 按内容修订的规则提交 M 的新版本：推进 version、内容时间与衰减基准，重置置信度；版本记录来源为 `MERGE`，来源引用包含 N 的全部引用（交互来源、创建记录与 v1 版本记录），N 的交互来源引用同时并入 M；向量按第 5 节的规则只在编译文本变化时重算；
5. 删除 N，结果携带 M 的身份与被吸收的 N；N 带句柄时附 `MERGED` 结算，由 Controller 发布改指事件（第 6 节）。

先写 M 再删 N：删除失败时 N 作为重复留下，M 已包含它的内容；N 的完整内容始终保存在它自己的 v1 版本记录中。模型融合两条记忆的结果不会与 M 完全相同，因此合并总是产生 M 的新版本，不设“结果不变时跳过写回”的分支。

以下情况以 `MemoryMergeAbortedError`（`workspace.memory.merge_aborted`）让任务失败，两条记忆都不变，不回退为追加或覆盖：任一记忆缺失或不可见（`memory_missing`）、合并条件不满足（`not_mergeable`）、写回前版本已变（`version_changed`）、模型输出为空、无法解析或把类型改为 Profile（`model_output_invalid`）。提供商失败重试一次后原样传播 `ModelProviderError`。版本检查与写回之间没有原子保证，与 alias 唯一性的“检查→写入”相同。

Alias 由 `AliasGenerator`（`engines/generation/alias.py`）按 memory type 前缀和 extractor 给出的 suffix 构造候选，例如 `code_* / fact_* / url_* / ref_* / user_* / wip_* / agent_*`；suffix 清洗后为空时改由 title 派生，仍为空则该记忆没有 alias。候选先经中期库确认在同一 Workspace 内空闲，被占用时依次追加 `_2`、`_3`…… 消歧后缀重试，全部被占用时抛 `MemoryAliasConflictError`（`reason=alias_candidates_exhausted`）。同一次结算的多条草稿全部算完之后才由 Familiar 依次落库，中期库查不到本批已分配的 alias，因此引擎把它们作为已占用传给后续草稿，后缀相同的草稿得到消歧后缀。生成侧只负责让正常路径避开冲突；同一 Workspace 内 alias 唯一的不变量由中期库写入前校验兜底，见 [MemoryLibrary](./memory-library.md)第 1.2 节。Pending alias 只在 Alice 运行期定位候选意图；settlement 后的 canonical alias 才属于正式 MemoryAtom。

## 5. Artifact 与持久化顺序

MemoryGenerationFamiliar 执行：

1. 尝试捕获 InteractionArtifact：话题结算时覆盖全部交互记录并带折叠记录，主动生成时只有最近的 blocks、没有折叠记录；
2. 在提交边界取一次 `now`，调用 GenerationEngine（传入同一时点用于内容日期）得到 outcomes；
3. 对每个 outcome 依次完成：提交字段分配（UPDATE 递增 version、推进内容时间与衰减基准、重置置信度）→ 构建 creation/version artifacts（强制，`now` 贯通 `changed_at` 与事件时间）→ 挂载 refs/events；
4. 执行 MidTerm `upsert(atom, recompute_vectors=<embedding 输入是否变化>)`；TOUCH 走受限 `patch_payload`；
5. 把 outcome 投影为 `MemoryGenerationResult`，附可选的 PendingAtomSettlement 与合并提名（第 4.1 节），并通知变更。

InteractionArtifact 在生成之前捕获，失败只记录 warning、生成照常进行。因此提取失败（包括超窗与截断）、没有产出或结果为 TOUCH 时，原文仍保存在 InteractionArtifact 中，只是没有记忆引用它。一次结算产出的多条记忆共享这一份 InteractionArtifact，各自挂载对它的来源引用。

`InteractionArtifactInput.asset_bindings` 随 settlement task 一起冻结并穿过队列边界；它是 Topic 已确认使用过的资产关系，而不是 Artifact 本身。CREATE/UPDATE 持久化后，Familiar 通过注入的 `WorkspaceAssetMaterializationReaderPort`，按归属与 `asset_ref` 租借 READY representation、创建独立 DocumentArtifact 并释放租借；失败按 best-effort 跳过，不回滚已提交的 Memory。该端口不携带 actor 或 scope，Store 的既有访问接口适配由组合根承担；详情见[Artifacts 与来源追踪](./artifacts.md)。

**版本记录是提交成功的前置条件**：版本存储关闭（NoOp builder）或构建失败时任务直接失败，不发布无历史的新 canonical；canonical upsert 失败同样使任务失败，已写入的版本记录保留为孤立 Artifact。`recompute_vectors` 由提交前后编译出的向量输入文本对比决定：比较的是 MemoryCompiler 的 `DENSE_EMBEDDING` 与 `SPARSE_EMBEDDING` 两份文本，也就是写入时实际编码的内容，而不是一份字段清单。新建总是计算向量；正文、`payload.agent_config` 等不进入向量文本的变化不重算，标签顺序变化会重算。修订、合并写回与手动编辑都按同一规则判断。详见[Artifacts 与来源追踪](./artifacts.md)。

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

任务来源（`MemoryGenerationSource`）有四种：`WRITE`、`UPDATE`、`SETTLE` 与 `MERGE`，任务快照与 `memory.task.*` 事件都携带来源。来源同时决定版本记录的来源类型：`MERGE` 任务写出的版本记录来源为 `MERGE`，其余来源的内容修订为 `UPDATE`。

Familiar 内部以 Generation Engine 返回的 `GenerationOutcome`（原子与提交动作 `GenerationAction`）完成 compute、artifact 与 persist；只有这些
步骤成功退出后才投影为跨域 `MemoryGenerationResult`。该结果仅保留 canonical identity、可选的
`PendingAtomSettlement`、合并提名与被吸收记忆的 ID，不会把 `MemoryAtom`、查重判断、提交动作、before snapshot 或 changelog 泄漏到
Controller。由此，Engine 计算语义与 Patchouli 已持久化事实保持明确边界。

控制面提供 list/get/cancel/wait/wait_many/wait_all。`wait_task()` 直接返回
`MemoryGenerationTask | None`，`wait_many()` 按请求顺序返回快照或 `None` 的列表，不再复制一层
WaitResult/WaitSummary。等待使用 `asyncio.shield`，超时返回当前 `PENDING/RUNNING` 快照，不会接管或
自动取消后台任务；仅 shutdown drain 会把超时窗口后仍未终结的快照汇总为观测计数，并显式 cancel 那批任务。

任务状态通过 RuntimeEvent 发布。TaskController 只决定生命周期事件的发生时机，`MemoryTaskEventEmitter` 集中选择 `memory.task.*` 类型并组装稳定 payload；通用队列仍独立发布 `work.*` 基础设施事件，两者不互相替代。主动任务完成后，settlement 会通过 PatchouliBridge 转发为全局 PendingAtom event，由 workspace registry 更新登记终态；发布失败不会回滚已持久化记忆，Controller 会 best-effort 发布 pending failure。PendingAtom settlement、failed、cancelled 仍属于功能事件并继续留在 TaskController；当前没有事件未送达补偿或跨重启恢复，不能保证 registry 在通知丢失后自行补齐终态。

合并任务成功且结果带句柄时，Controller 经结算发布者发布改指事件 `patchouli.events.pending_atom.merged`，由 PatchouliBridge 转发为同名全局事件，载荷是 `MERGED` 结算与被吸收记忆的 ID。句柄此前已经以 `CREATED` 结算过一次，改指因此不占用“每个 alias 一次终态”的保留，发布失败只记录 warning，不改写任何任务终态。workspace 登记只对仍以 `CREATED` 指向被吸收记忆的句柄替换结算（[读取视图与写入意图](../workspace/read-view-and-write-intents.md)）。

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

这并不保证主动写出的正文与被动归档结果完全不重复。重复是当前可接受的成本，由查重决定新建、只触碰已有记忆，或新建后合并进已有记忆；为了消除重复，不能重新让 Perception 感知 focus 或 pending 状态，否则会把控制意图污染为对话事实。当前每个主动 task 也会独立捕获 `InteractionArtifact`，因此 provenance 可能重复，这是已知的可追踪性成本，而不是重新合并任务输入的理由。

## 8. 当前限制

- 检索的稠密与稀疏向量以及重排输入都只由标题、类型、标签与摘要编译（MemoryCompiler 的 `DENSE_EMBEDDING`、`SPARSE_EMBEDDING` 目标），只写在一条记忆正文里的细节检索不到。话题结算把彼此独立的事拆成多条记忆，各自有标题与摘要，但拆分粒度由模型按提示词判断，没有校验。真实样本见已归档的 [Import Bus 真实使用观察](../archive/todo/import-bus-real-use-observation.md)；
- 同一次结算的多条草稿之间不互相查重，只与已经落库的记忆比较；两条描述同一件事的草稿会各自新建；
- 一次结算的多条结果依次提交，中途失败（例如存储或版本记录故障）时已提交的记忆保留而任务失败；控制面只在任务成功时提交合并提名，这一批的合并因此不会发生，留下重复；
- 一段话题的全部草稿共用一次调用的输出上限：草稿多、正文长的话题被截断时任务失败，原文仍在来源 InteractionArtifact 中，但没有记忆引用它；
- 结果未知或结算事件丢失时，句柄可能长期保持 MATERIALIZING；没有对账恢复。写入一旦开始派发就不因 CPU、进程或 CALL 子帧失败取消；
- UPDATE 意图不携带登记时的基础版本，spec 构建时按 UUID 读取当时的最新版本，并发 UPDATE 不检测基础版本冲突；后续方向见[写入意图后续方向 Idea](../ideas/pending-intent-migration.md)第 6 节；
- memory work、typed result 与终态快照只存在于当前进程，重启后不可恢复；持久化与恢复边界见[耐久性与故障恢复治理](../governance/reliability/durability-and-recovery.md)；
- 终态快照与 Queue 使用同一 `terminal_retention` 上限；
- Controller 的 admission 持有同一把锁串行执行，并发提交的意图按取得锁的顺序入队；
- 提取、修订与合并的 LLM 调用没有独立超时，提供商长时间无响应时任务一直等待，只受取消与 shutdown drain 约束；
- 后台抽取与 chat 共用提供商配额，限流表现为提供商失败：Mode A 与合并任务按任务失败收尾（合并失败时两条记忆都保留），Mode B/C 使用 fallback；当前没有生成侧的并发控制；
- Mode A/B 去重仍只取 dense top-1；检索调用链独立携带归属与发起者，Memory store/adapter 先强制 Workspace ownership hard filter，再重验 actor policy；其他去重策略限制仍待收敛；
- 查重只按 dense top-1 余弦与固定阈值判断，分不开同一工作与同项目的其他记忆：误判的两条无关记忆也会被合并，只能经版本记录与人工编辑纠正；判定信号的证据与后续见[查重决策信号 Todo](../todo/generation-dedup-decision-signal.md)；
- 同一工作的各阶段不断并入同一条记忆，正文持续变长，摘要越来越难覆盖全部要点；正文过长时合并可能超出模型窗口，任务失败，两条记忆都保留；
- 新建到合并完成之间，检索与前端同时看到 M 与 N；
- 改指事件与结算事件属于同一可靠性等级，没有补偿：丢失时句柄仍指向已删除的 N，读取得到不带原子的重定向；
- 两个话题同时命中同一条 M 时（生成队列并发为 2），后写回的合并因版本已变失败，留下重复；
- 生成材料只截断工具结果，单个轮次没有总量上限，生成也不分片读取：工作视图保留的 blocks 很大时（例如单轮上百次工具调用），结算的提取可能超出模型窗口，任务以 `memory_generation_failed` 失败（超窗不重试），原文仍在来源 InteractionArtifact 中但没有记忆引用；
- Active tasks 复用相同 blocks 输入，但会各自写 InteractionArtifact；
- memory 版本 Artifact 失败会阻断该次内容提交（无历史不成功）；provenance 写入与 MemoryAtom 发布仍不是原子事务，canonical 失败会留下孤立 Artifact。

未来若引入 durable queue，首先要保持私有工作信封 codec、`WorkRecord` 单一状态源和 settlement 语义不变，避免基础设施升级重新把控制面塞回 GenerationEngine。
