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
last_reviewed: 2026-11-21
---

# 记忆生成

记忆生成是把短期材料或显式保存意图转化为正式 `MemoryAtom` 的冷路径。它必须同时避免两种对称错误：把每段对话都机械保存，会让书库迅速被噪音淹没；把保存完全交给不透明的 LLM 判断，又可能让用户明确要求保留的内容悄悄消失。

当前设计因此保留三种生成模式（话题结算又分一次性生成与馆长两种方式，由用户选择，第 3.1 节），并把“构造任务”“执行生成”“写 artifacts 和持久化”“发布任务/settlement 终态”拆成不同层。生成不再是 Perception 的一个回调，也不再由一个 LibrarianCore 同时管理计算与后台任务。

每个 `MemoryGenerationTaskSpec` 都独立携带创建任务时冻结的 `belong_to: WorkspaceIdentity`、`from_actor: ActorIdentity` 和 `topic_id`；任务快照沿用相同字段，不保存 `IdentityScope` 或工作 actor。Generation 先限定资源归属，再按发起者读取策略查重，不从进程当前上下文重新推导 Workspace 或权限。WorkspaceAsset 不属于 Generation，生成链只接收 settlement 交接中已经冻结的资产绑定事实；Store 的归属接口见 [WorkspaceAsset](../workspace/assets.md#3-身份与归属)。

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
       -> MemoryGenerationEngine.process（话题结算：一次性提取加查重，或馆长循环）
       -> ArtifactEngine
       -> MidTermMemoryStore.upsert
       -> MemoryGenerationResult
  -> PendingAtom settlement + RuntimeEvent
```

- Coordinator 把 settlement 或 PendingAtom materialize task 变成统一 spec；
- TaskController 创建不可变 work 并持有 typed handle，负责领域快照、取消、等待与事件时机；
- MemoryTaskEventEmitter 负责把领域快照投影为 `memory.task.*` RuntimeEvent；
- Familiar 是生成数据面，执行 compute -> artifact -> persist；
- GenerationEngine 负责提取、话题结算的馆长循环（`engines/generation/librarian.py`）、Mode C 修订、查重、两条记忆的合并计算和构造 outcome，不直接持久化或发布事件。

这条边界使 GenerationEngine 的返回值可以在落库前被补充来源信息，也让取消/失败只由 TaskController 写入一次终态。

## 2. 统一输入视图

Generation 不再消费拼接的 `context_messages`。`GenerationTranscriptBuilder` 从 `LogicalBlock[]` 构造：

```text
GenerationContext
  ├─ state_summary
  └─ GenerationTurn[]
       ├─ turn_id
       ├─ outcome（completed/cancelled/failed）
       ├─ user_message
       ├─ material_entries（用户消息之后的材料条目，只有工具结果按上限截断）
       ├─ segments（GenerationSegment[]：事件序号区间与渲染出的条目数，第 2.2 节）
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

Mode B/C 的背景同样经这个构建器生成。三种模式的提取提示词与馆长的提示词（中英文各一份）都说明材料格式：每轮以 `[User]` 开头，其后按发生顺序排列 `[Assistant]` 正文、`[Subagent]` 消息与以 `- ` 开头的过程条目；`[Subagent]` 是被委托的子 Agent 结束时交回的通知，带它的结局与最后的输出，按 Agent 的结论对待，与助手正文同等；THOUGHT 是 Agent 的推理，不能作为事实依据（Mode A 中主要由它得出的内容降低置信度）；TOOL/RESULT 是外部工具的调用与返回，可以作为依据，`(error)` 表示调用出错；SEARCH/READ/RUN 是记忆系统内部操作的一行摘要，Mode A 与馆长另外提示读回的已有记忆不要重复提取（馆长的提示词还说明主动写入不在材料中，要以工具查看）；工具结果中省略标记处的内容不要推测。提示词同时要求不把密钥、令牌、密码、证书或连接串中的凭据写入记忆，需要记录相关配置时只保留名称与用途。

渲染与截断规则的回归入口为 [材料测试](../../tests/unit/engines/perception/test_material.py)，转写格式见 [生成视图构建器测试](../../tests/unit/engines/generation/test_generation_transcript_builder.py)。

`GenerationTranscriptBuilder` 将 `TurnRecord.outcome` 传给 `GenerationTurn`；cancelled/failed 在转写的轮首附 `[Outcome: ...]` 和未正常完成的说明，避免把半截回复视为完整结论。

### 2.2 段标签与记忆引用

生成的记忆要能指回具体的交互：被动结算下的复核要指出依据在哪几段，要看到写入与更新发生在哪里。这只能由模型在生成时给出，事后无法重建。构建器按感知层的切段规则把每轮切段（[感知与短期话题](./perception.md#2-结构化摄入)），`GenerationTurn` 带上 `turn_id` 与各段；逐段渲染再拼接与整轮渲染得到同样的条目，各段记下事件序号区间与渲染出的条目数。转写在每段之前印出段标签 `[Segment 位置.段号]`，位置是这一轮在材料中的序数（与 `[Turn N]` 相同），第一段从用户消息开始：

```text
[Turn 2]
[Segment 2.1]
[User]: …
- TOOL: …
[Segment 2.2]
[Assistant]: …
```

段标签只出现在送给模型的生成材料中：渲染函数本身不带标签，token 估算、接力摘要与 Alice 的历史转写不受影响。升级前保存的生成任务（第 6 节）没有段，转写不印标签。

模型输出中的 `source_segments` 是支撑这一条内容的段标签：被动提取与主动写入的格式说明由 `ExtractedMemoryDraft` 的 JSON Schema 自动带上这个字段（Schema 的说明文字随提示词语言，见[全局 i18n](../system/i18n.md)），修订（Mode C 与馆长的 `revise_memory`）的编辑集格式在中英文提示词中手写，带这个字段（`MemoryEditSet.source_segments`），馆长 `write_memory` 的参数同样在提示词中手写、带这个字段；合并没有交互材料，不引用段；四类提示词都说明段标签的写法，并要求只引用直接支撑这条内容的段，话题状态摘要中的内容没有段、不引用。引用的容错按“只丢弃这一处引用”处理：

- 格式不对（不是形如 `2.3` 的字符串，或整个字段不是列表）的引用在解析时丢弃，不让草稿或整份被动输出作废；
- 材料中不存在的标签丢弃，并记一条 INFO 日志（`count`）；
- 退回的草稿（Mode B 提供商失败或输出无法解析）与追加的修订（Mode C）没有模型引用的段。

引擎把合格的标签按引用顺序去重，换算为 `CitedSegment`（轮次标识加首尾都包含的事件序号区间），经 `GenerationOutcome.cited_segments` 交给提交；新建（含查重为相似时的新建与馆长的新建）与修订带引用，查重命中而只触碰已有记忆（TOUCH）时不带。Familiar 把它写进这一版的 MemoryVersionArtifact（第 5 节），合并与手动编辑的版本记录为空。被引用的轮次就在同一份版本记录 `source_artifacts` 指向的交互快照中；被动结算捕获话题的全部交互，主动写入与修订捕获作为背景的最近几轮。引用不进入记忆原子、`ArtifactRef` 与 Qdrant 载荷，管理接口与 MCP 的读取都不返回它；生成侧现在也没有读取方，它供之后对已有记忆的复核读取。

在 1,216 轮真实 Claude Code 记录拼成的材料上回放被动提取（deepseek-chat，30 份材料各 3 次），引用全部格式合格、标签全部存在，同一份材料各次运行引用的段集合两两 Jaccard 均值 0.90；抽查中多数引用指向事实或决定所在的段，少数偏向这一轮开头的段。引用没有标准答案，准确性只能靠合格率、一致性与抽查判断。回放工具是 `scripts/memory_replay.py` 的 `turns-claude-code`、`segments`、`extract-turns` 与 `citations` 子命令。

## 3. 三种生成模式

被动提取与主动写入的提示词（中英文）与馆长的提示词给出同一组类型：

| 类型 | 写什么 |
|:---|:---|
| `FACT` | 明确的事实、业务规则与参数定义。陈述某样东西怎样运作的内容，即使引用了文件路径或网址，也是 FACT；只存在于对话中的代码（写给用户、没有提交的脚本，在控制台执行过的查询）也归 FACT，代码原样保留 |
| `REFLECTION` | 经验总结、错误反思与教训 |
| `USER_PROFILE` | 用户的偏好与习惯 |
| `RESOURCE` | 资源记忆：对一个外部资源（网页、文件、代码、文档）的记忆。正文说明它是什么、为什么值得去看、在哪里，把读者引向源头；不抄录源头的内容，真相留在源头 |

模型给出的类型名不在其中时，提取、馆长与写入落为 FACT；修订与合并不改类型（第 3.4 节）。`WORK_IN_PROGRESS`、`CODE_SNIPPET` 与 `URL_RESOURCE` 只作为历史取值留在 `MemoryType` 中（`HISTORICAL_MEMORY_TYPES`）：版本快照记着写入当时的类型，仍要按 schema 2.1 严格解码；模型照旧写出这些名字时同样按未知类型处理。“做到哪了”留在交互记录上，不写成记忆：一条进度只在产生时决定一次类型，修订改不了它，也没有关闭的状态，最终成了名义上一直在进行的工作。

### 3.1 Mode A：被动观察 / ARCHIVE

当 idle、LRU、shutdown 或 manual settle 形成 `TopicMaterializeTask` 时，Coordinator 用其中的工作视图（未折叠的 blocks 与 `state_summary`）构造 GenerationContext，被折叠的轮次只以摘要进入生成；同一任务的 `InteractionArtifactInput` 携带话题的全部交互记录与折叠记录，来源 InteractionArtifact 因此覆盖已折叠的轮次（第 5 节）。没有 turns 时跳过任务。

话题结算有两种生成方式，由 `patchouli.generation.settle_mode` 选择，是用户选项：一次性生成（`one_shot`，默认）与馆长（`librarian`）。一次性生成只调用一次模型，但写之前看不到本话题已经写入的记忆，也看不到库，只能先尽量多地提取，再由查重、触碰、合并提名与命中本话题写入即丢弃这一串规则补救（第 4 节）；馆长先查看本话题主动写入过的记忆，再检索并阅读库中相关的记忆，只写两者都没有覆盖的内容，代价是一次结算要调用模型数次。两种方式并存，因为馆长明显更慢、开销更大（owner，2026-11-19）；一次性生成按现状冻结，它的重复问题不在这里修补。两者共用 `extractor.enabled` 开关（关闭时都不运行，结算不产生记忆）、结算的发起者与来源裁定（本节末尾）、材料与段引用（第 2 节）、别名分配（第 4.1 节末段）与提交管线（第 5 节）。组合根只在选馆长、提取开启且配置了模型时为引擎装配馆长，引擎有馆长就以馆长结算，否则走一次性生成；两项配置保存后重启生效，关闭时保存的结算任务在下次启动后按当时配置的方式执行。

#### 一次性生成

GenerationEngine 渲染 transcript，调用 extractor 的话题提取（`extract`）。一段话题常包含多项彼此独立的决定、事实与经验，而检索向量只由标题、类型、标签与摘要编译，压进同一条记忆的次要内容只留在正文里，检索不到；因此模型一次输出一组记忆草稿（`ExtractionResult`），每件之后会被单独查找的事写成一条：一项决定及其理由、一个事实或配置、一个已解决的问题及其解法、一条偏好或经验、一个值得以后去看的外部资源；一项工作做到哪了（进度、下一步、待办）不写成记忆，其中得出的结论、决定与尚未解决的问题本身照常成条。提示词同时要求同一件事的背景、过程与结论不拆开，顺带提及、没有结论的内容不单独成条，也不为凑数拆分；整段没有值得保存的内容时输出空列表。

`has_value=false` 的草稿被丢弃，没有草稿时不产生记忆。其余草稿按输出顺序逐条查重，各自产出一条结果（第 4 节）；查重命中本话题主动写入结算到的记忆时，草稿被丢弃（第 4.2 节）。

Extractor 经组合根注入的共享 LLM 服务异步调用模型，请求参数由 `patchouli.llm` 经模型注册表解析得到，等待模型期间不阻塞事件循环；Librarian 未配置模型时 extractor 以 No-Op 运行。一段话题的全部草稿共用一次调用的输出上限，Librarian 的 `max_tokens` 默认为 32768。

Extractor 区分以下结果：

- **没有产出**：模型输出为空，或无法解析为草稿列表（包括只给出一条草稿对象、没有 `memories` 列表）。整段输出按一个对象解析，任一条草稿不符合结构时整段都没有产出。
- **输出被截断**：输出达到 `max_tokens` 上限时抛出 `ModelProviderError`，不重试，因为同一请求会再次截断。截断的列表无法解析，若当作没有产出，整段话题不留记忆而任务成功结束。
- **提供商失败**：只重试一次，耗尽后抛出 `ModelProviderError`。

Mode A 不捕获 `ModelProviderError`，生成任务以 `model_provider_failed`（超窗为 `model_context_window_exceeded`）失败收尾（第 6.1 节），不再把提供商故障或截断记录成“对话没有价值”。超窗、程序错误与取消不重试，原样传播；分类见[错误模型](../contracts/error-model.md#46-模型调用错误)。

#### 馆长

馆长（`engines/generation/librarian.py`）是 Patchouli 内部的工具循环：工具与循环由 Patchouli 自己定义，不复用 Alice 的执行循环与 MTP，也不经过 Workspace 的 actor API，只操作 Patchouli 自己拥有的数据（owner，2026-11-19）。

- **动作协议**：模型服务只有文本接口，没有原生工具调用。每一步，馆长把对话历史交给 `complete`（要求 JSON 输出），模型输出一个 JSON 动作 `{"tool": "<名称>", "args": {…}}`；Patchouli 执行工具，把结果连同剩余步数（`steps_left`）作为下一条消息交回。材料（话题状态与带段标签的各轮）在第一条消息中给出，不需要工具读取。
- **工具**：

  | 工具 | 作用 |
  |:---|:---|
  | `list_topic_writes` | 列出本话题主动写入结算到的记忆（第 4.2 节的记录）：别名、标题、摘要 |
  | `search_memories` | 以一句查询检索中期库，`mode` 为 `dense`（缺省）或 `sparse`，单向量、不融合、不重排；返回别名、标题、摘要、类型与分数，最多 `patchouli.generation.librarian.search_top_k`（默认 5）条 |
  | `read_memory` | 按别名读一条记忆的标题、摘要、标签、类型与正文 |
  | `write_memory` | 暂存一条新记忆：标题、摘要、标签、类型、正文、置信度、别名后缀与引用的段 |
  | `revise_memory` | 暂存对一条本次读过的记忆的修订：alias 加编辑集（第 3.4 节） |
  | `finish` | 结束 |

- **读取**：三个读取工具以结算的发起者（Workspace 的 system actor）直接访问中期库，只读得到本 Workspace 的 PUBLIC 记忆；不经过检索服务（`RetrievalFamiliar`），因此不记录命中、不刷新生命力。本话题的写入中 PRIVATE、TEAM 或已删除的不列出：私有内容因此不会被写进公开记忆，代价是馆长可能与它们重叠。
- **写入**：`write_memory` 只暂存草稿，字段与被动提取的草稿相同，一律视为有价值：是否值得写由馆长在写之前判断。循环结束后暂存的草稿一次交给引擎，按 CREATE 提交：分配别名、引用的段写进版本记录、入库；不查重、不触碰、不提名合并。最后仍有的重叠接受，不加复核、去重或兜底（owner，2026-11-19）。
- **修订**：`revise_memory` 只改本次运行经 `read_memory` 读过的记忆：材料中有不可信内容，而馆长能改全库，读过才能给出依据；`read_memory` 记下读到的那一版。一条记忆一次结算只修订一次，几处修改放进同一个编辑集。编辑集在循环中以读到的版本为底应用，不合格时作为工具错误交回、计一步、不暂存，通过时暂存并交回 `{"revised": alias}`。循环结束后引擎把修订转成 `UPDATE` 结果：修改前的完整快照、并入结算贡献者的贡献者集合与引用的段；Familiar 提交前按基础变化规则重读、重放或丢弃（第 3.4 节）。版本记录的 `update_source` 为 `UPDATE`，来源主体不变；修订的记忆可以是本话题的写入，也可以是库中已有的记忆，包括一次性生成写下、合并过的。
- **结束与预算**：模型调用 `finish`，或步数达到 `patchouli.generation.librarian.max_steps`（默认 12，每输出一个动作算一步），或生成 lane 的时限到期（任务以 `memory_generation_timed_out` 失败，暂存的不提交）。步数用完时已暂存的照常提交，并记一条 warning。每次结算记一行日志：结束方式、步数、暂存条数、模型报告的 token 合计、各工具的调用次数、修订条数与被拒的修订次数。在开发实例上实测（deepseek-chat，2026-11-19），一次结算 5–8 步，模型报告的 token 合计约 1.5 万到 3.8 万。
- **失败**：动作无法解析、工具不存在或参数不合格时，错误作为工具结果交回模型并计一步，循环继续。模型服务失败不重试（提取器重试一次），与超窗一样原样传播，任务以 `model_provider_failed` 或 `model_context_window_exceeded` 失败（第 6.1 节），暂存的草稿不提交，来源 InteractionArtifact 照常保存。
- **提示词**：馆长的系统提示词（中英文）规定工作顺序：先看本话题写过什么，再查库里相关的，最后对每件事作三种判断：已有记忆足以支持下次使用就不动；材料改变了某条读过的记忆的结论、适用条件或当前状态，就修订那一条；只有已有记忆承载不了的内容才新建。同时给出动作格式与工具说明、编辑集的写法（逐字抄出原文片段，只改变化的部分）、步数与检索条数的上限、记忆类型、材料格式、助手正文中的事实怎样对待（助手转述的用户事实按用户的事实记，正文不写出处说明；助手自己的建议用户采纳了才记；与用户的说法矛盾时以用户为准）、凭据约束、段引用与拆分粒度。被动提示词与 `ExtractionResult` 只服务一次性生成。

循环、预算、六个工具与修订的校验由[馆长测试](../../tests/unit/engines/generation/test_librarian.py)覆盖，两种方式的分派与馆长产出的提交动作由[引擎测试](../../tests/unit/engines/generation/test_engine.py)覆盖，“先写入、后由馆长结算：列出写入、检索库、新建入库”由[生成流水线集成测试](../../tests/integration/patchouli/test_memory_generation_pipeline.py)验证。

#### 发起者与来源

Mode A 的“被动”指没有显式 WRITE/UPDATE focus，而不是同步发生在用户响应内，也与 Import Bus 无关：导入的轮次与主动链路的轮次一样进入话题，之后经 settle 参与 Mode A。它仍作为 Patchouli 后台 memory task 执行。

四种 settle 入口（manual / idle / LRU / shutdown）共享同一发起者裁定：Coordinator 按 `belong_to` 构造 `user_id=owner_user_id`、`agent_id=SYSTEM_AGENT_ID`、`team_id=None` 的 `from_actor`，不继承最后访问者或驱逐触发者。`system` 不属于 Team，也不能成为 PRIVATE policy 的目标，普通读取策略因此把 SETTLE 的查重与馆长的读取限定为本 Workspace 的 PUBLIC 记忆，避免后台结算触碰某个参与 Agent 才可见的 PRIVATE/TEAM 记忆。新建记忆使用 PUBLIC policy、`source_agent_id=SYSTEM_AGENT_ID` 与 `SYSTEM` creation intent；实际参与内容的 Agent 由 `contributing_agent_ids` 从 block identity 聚合（去重、保持首次出现顺序、不含 `system`）。

Mode B 新建记忆以提交 WRITE 的 actor Agent 为来源，贡献者先记录发起 Agent 本身（无上下文的主动写入仍由发起 Agent 产出），再合并上下文轮次贡献者。WRITE/UPDATE 保留物化意图的 `from_actor`，按该 actor 的普通 PUBLIC/TEAM/PRIVATE 可见性读取，不能沿用 SETTLE 的 system 身份。Mode C 修订保留已有 `meta.provenance` 的来源主体字段，并把本次生成的贡献者并入已有集合；两条记忆合并时沿用保留记忆的来源主体，贡献者取两者的并集（第 4.1 节）。结果随 MemoryAtom 与对应 MemoryVersionArtifact 持久化。来源与贡献者字段只记录 provenance，不参与读取授权。

四种触发的发起者、SETTLE/WRITE 查重可见性与完整空闲结算持久化由[结算身份集成测试](../../tests/integration/patchouli/test_settlement_identity.py)验证。

### 3.2 Mode B：主动 WRITE

WRITE 经操作请求进入 workspace 能力层，在共享 `WriteIntentRegistry` 登记 MATERIALIZING 意图，并立即调用 Patchouli 的 `memory_intent.submit`。两种来源走同一条链路：Alice 的 MTP WRITE 经执行线程提交、以线程的工作 actor 授权，派发话题是进程这一轮的话题：Gateway 指向已有话题时是 PREPARE 确认的话题，指向新话题时由本轮第一次写入建立（[执行线程](../workspace/execution-threads.md#8-派发话题)）；外部 harness（plugin 模式）的 `memory_write` 经请求级提交、以每次调用认证得到的工作 actor 授权，派发话题是 Import Bus 在写入时为这一轮确定的话题，这一轮回流时落在同一话题（[Import Bus](../system/import-bus.md#47-写入时确定话题)）。服务沿用登记的 `intent_id` 与 `pending_alias` 构造内部 `PendingAtomMaterializeTask`，生成队列接纳后才返回 ACK。意图登记与终态由 workspace 持有，正式记忆生成与任务状态由 Patchouli 持有。

Coordinator 使用话题的 `state_summary` 与过去 N 个 block 作为背景，以 `WriteFocus` 为保存核心。N 由 `patchouli.generation.active_recent_blocks` 配置（默认 5，允许 0），与感知折叠保留轮数无关。派发发生在交互封口前，当前轮不进入背景；新话题为空时只依赖 focus：任务进程第一次写入建立的新话题与 plugin 模式下写入时建立的新话题在派发时都为空，落到已有话题时背景是该话题此前的几轮。话题不存在时按空材料生成，话题读取抛错时因尚无任务入队，逐项结算 FAILED 并拒绝。

提取器把 agent 拟定的标题（`WriteFocus.title`，可以为空，空时提示词写"未提供"）连同正文与保存理由交给模型。提示词要求沿用这个标题、不改写措辞，只在它与草稿内容不符，或缺少让人检索到它的关键名词（项目、组件、工具名）时修改，修改时保留原标题的判断与表述。检索只比较标题、类型、标签与摘要，agent 斟酌过的标题是这条记忆能否被找到的主要依据；改写第一当事人拟定的标题属于越权到措辞。系统不比较模型输出与 agent 的标题，是否修改由模型按规则判断；摘要、标签与正文照常由生成整理，合并仍会重新综合两条记忆的标题（第 4.1 节）。

Extractor 遇到提供商失败（重试耗尽）或输出无法解析时，Mode B 会直接从 WriteFocus 构建 fallback draft，保证明确保存意图不会因为 Librarian LLM 暂时失败而无声丢失；程序错误不进入 fallback，任务按失败收尾。Fallback 仍要经过查重，因而最终可能新建、只触碰已有记忆，或新建后被合并进已有记忆；ACK 从未承诺“一定保留为一条独立记忆”。相似命中本话题其他主动写入结算到的记忆时照常新建、不提名合并（第 4.2 节）。

### 3.3 Mode C：主动 UPDATE

UPDATE 与 WRITE 的两种来源和派发话题规则相同：Alice 的 MTP UPDATE 与外部 harness 的 `memory_update` 都经 actor API 解析可读的正式基础原子后登记并派发。

UPDATE spec 构建时，Coordinator 解析 `UpdateFocus.base_uuid` 并通过 local route 读取现有 MemoryAtom。UUID 非法或目标不存在只使该 active spec 失败，并发布对应 pending failure，不阻断同批其他任务。

执行时 Familiar 按复合键重读基础记忆，以此刻的版本为底计算：任务可能排队很久，agent 的指令作用在此刻的内容上，排队期间别处的修改模型也看得到。基础已删除或对发起者不可读时不修订，任务以 `ResourceNotFoundError` 失败（原因码按领域错误的规则为 `workspace.resource.not_found`）：发起者不能修订它读不到的记忆，写回也会让已删除的记忆重新出现。workspace 登记 UPDATE 时记下的基础内容版本（`UpdateFocus.base_version`）只作记录，不参与写回判断。

Engine 用 extractor 的修订（`revise`）请模型写出对这条记忆的编辑集（第 3.4 节），输入是标题、摘要、标签、alias、当前正文、指令、新材料与参考上下文。编辑集以执行时的版本为底应用；不能应用或输出无法解析时，把上一次的输出与错误附上再请一次。仍不合格、提供商失败或没有模型时回退：

- focus 携带 content：把内容作为带日期的更新段追加到正文末尾，只改正文、不改索引，没有模型引用的段；
- 只有 instruction：不改（TOUCH）。

UPDATE 的 ACK 表示意图已被接纳，内容不因模型没能写成编辑集而丢失。编辑集为空表示不需要改，同样降为 TOUCH，只经受限 patch 推进访问统计。真实内容变化时，Familiar 在提交边界（一次取 `now`）推进 `meta.version` 与 `updated_at`/`decay_anchor_at`、重置置信度，前后完整快照交给 Artifact 层生成 MemoryVersionArtifact 后写回中期库；标题、摘要或标签变了时编译文本随之变化，向量重算，检索与查重看到修订后的索引。

提交之前按第 3.4 节的基础变化规则对上基础的当前版本：模型调用期间基础的内容被修改过（另一个 UPDATE、馆长的修订或管理页编辑）时，把编辑集重放到当前版本上，期间别处的修改只要没碰到编辑引用的原文就得以保留；重放不了时任务以 `MemoryUpdateBaseChangedError` 失败（原因码 `workspace.memory.update_base_changed`，第 6.1 节），读取句柄时提示重新读取后再更新；基础在期间被删除或变得不可读时以 `ResourceNotFoundError` 失败。访问统计与读取策略的变动不推进内容版本，修订沿用此刻的值，不把用户在管理页修改的策略改回去。

### 3.4 编辑集：修订已有记忆的共同形式

馆长的修订（第 3.1 节）、主动 UPDATE（第 3.3 节）与查重提名的合并（第 4.1 节）都以编辑集（`MemoryEditSet`，`engines/generation/edits.py`）改一条已有记忆。整篇重写在多次修订中会让细节漂移：此前 Mode C 输出整篇新正文，合并输出完整的标题、摘要、标签、类型与正文，第一次 HaluMem 评测中一次性生成的 628 次合并有 98% 改了标题，同一条记忆合并 15 次之后标题成了逐次累加的关键词清单（[写入链路 Idea](../ideas/memory-flow-write-path.md#21-对照已有记忆判断与修订的形态)第 2.1 节）。输出格式只有“给出新的全文”时，模型没有“有限度地改”的选项；片段替换从结构上给出这个选项。

| 操作 | 字段 | 作用 |
|:---|:---|:---|
| `replace` | 正文、标题、摘要 | 把一处原文换成新文；原文必须在当前文本中逐字出现且只出现一次 |
| `append` | 正文 | 在正文末尾追加一段 |
| `rewrite` | 正文 | 整篇改写正文；只在大部分内容都变了时使用 |

标签以 `add_tags` 与 `remove_tags` 逐个增删，按小写、去首尾空白比较；类型、访问策略、alias 与来源主体不在编辑集中，沿用原记忆。编辑按顺序应用，后一处在前一处的结果上查找原文。结果的标题与摘要越界时拒绝而不是截断，标题不能为空，至少保留一个标签；什么也没有改的编辑集不合格，空编辑集表示不需要改。不合格时的说明指出第几处编辑与原因，原样交回模型。

检索与查重的向量只由标题、类型、标签与摘要编译（第 8 节），只改正文的修订在检索中等于没有修订，因此修订可以改索引。索引的修改幅度由上面的结构限制：标题与摘要只能片段替换，标签只能逐个增删，类型不变；不再另外比较修改前后的向量。修改前后的稠密余弦分不开状态真的变了的修订与同一主题的整条重写（整条重写的合并中位 0.930、最低 0.800，正确的修订最低 0.876），而在编辑集的格式下，合并改了标题的从 16 次中 16 次降到 3 次。一次 `replace` 仍可以换掉整句摘要，状态真的变了的修订需要这样做。

校验失败时，馆长在工具循环中收到错误，在剩余的步数内改正；Mode C 与合并没有工具循环，把上一次的输出与错误附上再请模型一次，仍不合格时 Mode C 回退、合并中止。

**基础变化规则**：编辑集以计算时读到的版本为底；提交前重读基础记忆，三条路径共用同一条规则，失败的方式随路径对调用方的承诺不同：

| | 基础的版本没变 | 版本变了、编辑集重放成功 | 重放不了 | 基础已删除或不可读 |
|:---|:---|:---|:---|:---|
| 馆长修订 | 提交 | 以当前版本为底提交 | 丢弃这一条修订，记 warning | 丢弃这一条修订，记 warning |
| Mode C | 提交 | 以当前版本为底提交 | `MemoryUpdateBaseChangedError` | `ResourceNotFoundError` |
| 合并的保留记忆 | 写回 | 以当前版本为底写回 | 以 `version_changed` 中止 | 先沿吸收关系改指，找不到以 `memory_missing` 中止 |

馆长的一次结算有多条结果，一条失败不应连累其余；Mode C 对应 agent 的一次请求，失败要让 agent 知道；合并中止时两条都保留，不丢内容。重放的结果以当前版本为底，期间别处的修改只要没碰到编辑引用的原文就得以保留，访问统计与读取策略沿用此刻的值，内容贡献者取两边的并集。编辑集的解析、应用与重放由[模型测试](../../tests/unit/engines/generation/test_models.py)覆盖，三条路径的基础变化规则由 [Familiar 测试](../../tests/unit/patchouli/services/test_memory_generation.py)覆盖。

## 4. 查重与合并

查重与合并服务主动写入（Mode B）与一次性生成；馆长写下的草稿不查重，也不提名合并（第 3.1 节）。一次性生成的每条草稿与 Mode B 的草稿都按 MemoryCompiler 的 `DENSE_EMBEDDING` 目标编译为查询文本，对中期库执行 dense top-1 搜索。已有记忆的稠密向量由同一目标的编译文本编码，查询与入库的文本形态一致，内容相同的草稿才能达到 IDENTICAL 所需的相似度；早先查询只拼接 `title summary`，与带字段标签的入库文本形态不同，内容完全相同的草稿也只有约 0.90，触碰分支实际走不到。真实 embedding 与 Qdrant 下内容相同的草稿只触碰已有记忆，由[查重召回测试](../../tests/integration/patchouli/test_generation_dedup_recall.py)验证。Deduplicator 按相似度与内容一致性给出判断（`DuplicateDecision`），GenerationEngine 再把判断翻译为提交动作（`GenerationAction`）：

| 判断 | 条件（默认阈值） | 提交动作 |
|:---|:---|:---|
| `DISTINCT` | 没有候选，或相似度不高于下限 0.75 | `CREATE`：从 draft 构造新 MemoryAtom 与 canonical alias |
| `SIMILAR` | 相似度高于上限 0.95 但内容不一致，或落在 (0.75, 0.95] | `CREATE` 新记忆 N；满足合并条件时携带命中的记忆 M 作为合并目标，不修改 M |
| `IDENTICAL` | 相似度高于上限，标题相同且正文词集合的 Jaccard 相似度高于 0.9 | `TOUCH`：经受限 `patch_payload` 推进 access count 与 last_accessed_at；不改内容时间、不升版本、不重算向量 |

阈值由 `patchouli.generation.deduplicator` 配置。`SIMILAR` 与 `IDENTICAL` 都作用于命中的已有记忆，Deduplicator 必须随判断返回该记忆；违反这一约定时 GenerationEngine 显式失败，不以缺失的记忆继续构造结果。

查重命中后不改写已有记忆。早先的实现用草稿整体覆盖命中的记忆：同一项目的记忆增多后，同一工作的前后阶段被空闲结算拆到不同话题，后一阶段的草稿连续覆盖前一阶段的记忆，最初的设计决定只剩在版本记录里。现在命中时先把草稿保存为独立的新记忆，再由生成队列中的合并任务把它独有的内容补进命中的记忆。草稿先落库，因此关闭、队列拒绝或合并失败最多留下两条相似的记忆，不会丢失内容。合并与 Mode C 共用编辑集（第 3.4 节），输入不同：Mode C 是一条记忆加修改指令与新材料，合并是保留记忆 M 与被吸收记忆 N 两条正式记忆。

合并条件是 M 与 N 的访问策略相同，且对 WRITE 而言 M 不是本话题主动写入结算到的记忆（第 4.2 节）。策略不同的两条记忆合并后会改变一方的可见范围。生成的记忆都是 PUBLIC，策略不同只会出现在经管理入口改过策略的记忆上。不满足条件时 N 是一条普通的新记忆。

### 4.1 合并任务

Familiar 完成 N 的创建提交后，在 `MemoryGenerationResult` 中携带合并提名（`MergeFocus`：保留记忆 M 与被吸收记忆 N 的 ID；N 来自主动 WRITE 时另带这次写入的 `pending_alias` 与 `intent_id`），WRITE 的首次结算照常为 `CREATED`。Controller 在原任务成功并发布结算之后，以原任务的归属、发起者与话题提交一个 `MERGE` 来源的任务，提名放在 `GenerationRequest.merge_focus`。合并任务不持有写入意图的终态，任务标识一次性生成，失败或取消都不发布句柄事件。队列已停止或容量不足时只记录 warning，两条记忆都保留；关闭时尚未执行的合并与其他生成任务一起保存，下次启动后执行（第 6.2 节）。

Familiar 的合并路径：

1. 以 Workspace 为键经串行门（`KeyedSerialGate`）依次执行，从读取两条记忆到写回、删除都在门内：同一 Workspace 的两个合并不会同时写同一条保留记忆，也不会在模型调用期间删掉对方的目标；新建、修订与其他 Workspace 的任务不受这道门约束；
2. 保留记忆已被此前的合并吸收时，沿进程内记下的吸收关系改指仍然存在的那条。几次提名串成一串时（另一个话题的草稿命中了刚落库、还在等着被并走的新记忆），无论哪个先执行，内容都依次进入同一条记忆。吸收关系只在进程内，重启后为空；
3. 以发起者的可见性读取 M 与 N，检查合并条件，并拒绝同一条记忆吸收自身；
4. GenerationEngine 调用 extractor 的合并（`merge` 提示词）：模型看到两条记忆的标题、摘要、标签、类型与正文，并按内容更新时间得知哪一条较新；它以 M 为底写出编辑集（第 3.4 节）：补入 N 中 M 没有的事实，用 `replace` 改掉 M 中被 N 推翻的说法（以较新的一条为准），标签只增加；N 的内容 M 都已有时输出空编辑集。类型、alias、来源主体与访问状态沿用 M，贡献者取 M 与 N 的并集。编辑集不合格时带着错误再请一次；
5. 写回前按复合键重读两条：M 的版本变了时把编辑集重放到当前的 M 上（第 3.4 节）；N 的版本变了时不重放，要吸收的内容已经不同；再次检查合并条件；
6. 按内容修订的规则提交 M 的新版本：推进 version、内容时间与衰减基准，重置置信度；版本记录来源为 `MERGE`，来源引用包含 N 的全部引用（交互来源、创建记录与 v1 版本记录），N 的交互来源引用同时并入 M；向量按第 5 节的规则只在编译文本变化时重算；
7. 删除 N，在进程内记下“N 已被 M 吸收”，结果携带 M 的身份与被吸收的 N；N 带句柄时附 `MERGED` 结算，由 Controller 发布改指事件（第 6 节）。

先写 M 再删 N：删除失败时 N 作为重复留下，M 已包含它的内容；N 的完整内容始终保存在它自己的 v1 版本记录中。合并总是产生 M 的新版本：空编辑集时 M 的内容不变，版本记下这次吸收（贡献者并集、N 的来源引用并入）。保留记忆是底，原有的细节不会被较新的那条覆盖。

以下情况以 `MemoryMergeAbortedError`（`workspace.memory.merge_aborted`）让任务失败，两条记忆都不变，不回退为追加或覆盖：任一记忆缺失或不可见、改指之后也找不到（`memory_missing`）；合并条件不满足（`not_mergeable`）；写回前 M 的编辑集重放不了，或 N 的版本变了（`version_changed`）；模型输出为空、无法解析，或编辑集带着错误重试一次后仍不能应用、要删去 M 的标签（`model_output_invalid`）。提供商失败重试一次后原样传播 `ModelProviderError`。版本检查与写回之间没有原子保证，与 alias 唯一性的“检查→写入”相同；串行门只约束合并之间，模型调用期间的修订、删除或策略变更由写回前的重读处理。

Alias 由 `AliasGenerator`（`engines/generation/alias.py`）按 memory type 前缀和 extractor 给出的 suffix 构造候选，例如 `fact_* / ref_* / user_* / res_* / agent_*`；只作历史取值的类型不再产生新记忆，没有前缀，已有记忆的 alias 是对外标识，不随类型改变；suffix 清洗后为空时改由 title 派生，仍为空（例如纯中文或纯符号）时回退为 `mem_` 加记忆 ID 的 32 位十六进制（`fallback_alias`），不带连字符，MTP 的多目标列表可以直接引用；记忆 ID 唯一，回退别名因此不查询占用，引擎也因此先构造原子、再以原子的 ID 分配 alias。由前缀与 suffix 或 title 构成的候选先经中期库确认在同一 Workspace 内空闲，被占用时依次追加 `_2`、`_3`…… 消歧后缀重试，全部被占用时抛 `MemoryAliasConflictError`（`reason=alias_candidates_exhausted`）。同一次结算的多条草稿全部算完之后才由 Familiar 依次落库，中期库查不到本批已分配的 alias，因此引擎把它们作为已占用传给后续草稿，后缀相同的草稿得到消歧后缀。生成侧只负责让正常路径避开冲突；同一 Workspace 内 alias 唯一的不变量由中期库写入前校验兜底，见 [MemoryLibrary](./memory-library.md)第 1.2 节。Pending alias 只在 Alice 运行期定位候选意图；settlement 后的 canonical alias 才属于正式 MemoryAtom。

### 4.2 本话题的主动写入

查重与合并本身不区分来源。只按相似度判断时，被动提取会从同一份材料里重新推导出 agent 已经写入的内容，判为相似后并入；合并按新旧定主次，较新的被动草稿成为骨架，把 agent 有意精简掉的风险清单、行号与未决问题带回记忆。同一轮里 agent 有意分开写的几条记忆，也会因相似度越过阈值而互相合并。证据与考虑过的做法见[已归档的收敛 Idea](../archive/ideas/generation-source-convergence.md)第 2、3 节。因此同一话题内，主动写入结算到的记忆优先：

- **记录**：控制器在进程内按话题记录本话题主动写入结算到的正式记忆（两种结算方式共用这份记录），即 WRITE、UPDATE 的结算目标，以及吸收了写入新建记忆的合并所保留的那条。带 settlement 的结果只来自主动写入链路，数据面返回后即记下。
- **传递**：WRITE 与 SETTLE 执行时，控制器把该话题当前的记录放进生成请求（`GenerationRequest.topic_write_memory_ids`）；SETTLE 执行成功后清除这个话题的记录，话题结算之后不再接收写入；关闭时保存的 SETTLE 带上这份记录（第 6.2 节）。读取发生在执行时而不是接纳时：同一话题先后派发的写入，前一条完成后，后一条执行时才能看到它。
- **一次性生成（Mode A）**：草稿查重为 IDENTICAL 或 SIMILAR、且命中记录中的记忆时丢弃，不新建、不触碰、不提名合并，只记一行日志。一段话题的草稿全部被丢弃时任务照常成功，没有新记忆，来源 InteractionArtifact 照常保存。被动提取照常运行，写入没有覆盖的事实照常进入记忆。
- **馆长**：以 `list_topic_writes` 列出记录中 system actor 读得到的记忆，在写之前查看，不按相似度丢弃草稿（第 3.1 节）。
- **Mode B**：SIMILAR 命中记录中的记忆时照常新建、不提名合并；IDENTICAL 照常只触碰，同一轮写了两次相同的内容时触碰仍然正确。
- **不变的部分**：跨话题的查重、合并提名与合并语义不变；Mode C 不查重，不受影响；结算事件、任务快照与任务 API 不变。

这一规则按话题判定，不知道一条被动草稿复述的是哪一次写入；两条写入路径在哪里汇合见 [v0.8 写入链路 Idea](../ideas/memory-flow-write-path.md)第 5 节。引擎的丢弃与不提名、控制器的记录与传递分别由引擎与控制器的单元测试覆盖，"先写入、后结算时复述写入的草稿不产生合并、库中只有写入的那条"由[生成流水线集成测试](../../tests/integration/patchouli/test_memory_generation_pipeline.py)验证。

## 5. Artifact 与持久化顺序

MemoryGenerationFamiliar 执行：

1. 尝试捕获 InteractionArtifact：话题结算时覆盖全部交互记录并带折叠记录，主动生成时只有最近的 blocks、没有折叠记录；
2. 在提交边界取一次 `now`，调用 GenerationEngine（传入同一时点用于内容日期）得到 outcomes；
3. 对每个 outcome 依次完成：提交字段分配（UPDATE 递增 version、推进内容时间与衰减基准、重置置信度）→ 构建 creation/version artifacts（强制，`now` 贯通 `changed_at` 与事件时间）→ 挂载 refs/events；
4. 执行 MidTerm `upsert(atom, recompute_vectors=<embedding 输入是否变化>)`；TOUCH 走受限 `patch_payload`；
5. 把 outcome 投影为 `MemoryGenerationResult`，附可选的 PendingAtomSettlement 与合并提名（第 4.1 节），并通知变更。

InteractionArtifact 在生成之前捕获，失败只记录 warning、生成照常进行。因此提取失败（包括超窗与截断）、没有产出或结果为 TOUCH 时，原文仍保存在 InteractionArtifact 中，只是没有记忆引用它。一次结算产出的多条记忆共享这一份 InteractionArtifact，各自挂载对它的来源引用。

`InteractionArtifactInput.asset_bindings` 随 settlement task 一起冻结并穿过队列边界；它是 Topic 已确认使用过的资产关系，而不是 Artifact 本身。CREATE/UPDATE 持久化后，Familiar 通过注入的资产读取端口 `WorkspaceAssetReaderPort`，按归属与 `asset_ref` 租借 READY representation、创建独立 DocumentArtifact 并释放租借；失败按 best-effort 跳过，不回滚已提交的 Memory。该端口只接收归属，不携带 actor 或 scope，组合根直接注入资产 Store；详情见[Artifacts 与来源追踪](./artifacts.md)。

**版本记录是提交成功的前置条件**：版本存储关闭（NoOp builder）或构建失败时任务直接失败，不发布无历史的新 canonical；canonical upsert 失败同样使任务失败，已写入的版本记录保留为孤立 Artifact。`recompute_vectors` 由提交前后编译出的向量输入文本对比决定：比较的是 MemoryCompiler 的 `DENSE_EMBEDDING` 与 `SPARSE_EMBEDDING` 两份文本，也就是写入时实际编码的内容，而不是一份字段清单。新建总是计算向量；正文、`payload.agent_config` 等不进入向量文本的变化不重算，标签顺序变化会重算。修订、合并写回与手动编辑都按同一规则判断。详见[Artifacts 与来源追踪](./artifacts.md)。

### 5.1 重试边界

Memory Generation 的业务 handler 固定为单次 attempt。整条数据面包含 artifact 写入、LLM 生成和
`MemoryAtom` upsert 等副作用；在这些步骤部分成功后无法确认外部状态时，自动重放可能制造重复记忆或重复
artifact。因此 `TimeoutError`、`ConnectionError`、模型异常以及普通业务异常都会直接把任务投影为
`FAILED`，不会由 Memory Generation lane 自动重试。

这不等于系统完全没有重试：通用 Work Queue Runtime 仍保留 retry 状态机，Interaction Submission 也只对
明确标记的瞬态提交错误做有限重试；Generation extractor 内部的 LLM 边界重试属于另一层策略，只对提供商
失败重试一次，超窗、程序错误与取消不重试；馆长不重试提供商失败。当前
`FAILED` 只表示本次任务未完成，不保证没有任何外部副作用。Active WRITE/UPDATE 已使用 PendingAtom
`intent_id` 派生稳定 task/work identity：进程内重复 dispatch 会复用已有任务，且相同 identity 携带不同
spec 会被拒绝。跨重启结果记录、数据面内部副作用幂等和 reconciliation 仍留待后续持久化/I2 阶段处理。

手工 memory create/update 同样通过 Familiar，两者都只接收调用方提交的字段材料：创建由 Familiar 构造原子，记忆 ID、来源记录（取本次发起者）、默认访问策略、生命周期与完整时间戳都在提交边界确定，字段非法以 `InvalidMemoryFieldError` 拒绝，alias 唯一性在写版本记录之前检查；创建时没有给出 alias 的记忆同样以 `mem_<记忆 ID>` 命名并落库。HTTP 的记忆管理入口不接受只作历史取值的类型（包括 `AGENT_PROFILE`）。

Agent Profile 不是记忆，不经过 Familiar：它是 AgentProfileStore 中的文件，由 Patchouli 的 Profile 管理服务直接读写，没有版本记录（[MemoryLibrary](./memory-library.md#15-agent-profile人偶图纸)第 1.5 节）。

编辑捕获 before snapshot，仅当传入值与当前值不同（值相等判定）才递增 version 并生成 MANUAL_EDIT version artifact，无变化调用原样返回。

## 6. 后台任务与终态

`MemoryGenerationTaskSpec` 是控制面与 Familiar 共享的业务输入。Controller 只在队列边界内把
`task_id + spec` 包装为私有不可变工作信封；该队列适配细节不再作为一份公开领域模型。公开 API 使用
`MemoryGenerationTask` 只读快照表达下列领域状态：

```text
PENDING -> RUNNING -> COMPLETED | FAILED | CANCELLED
```

私有工作信封 codec 的 `schema_version` 当前为字符串 `"1.3"`，`spec` 独立编码归属与发起者；1.3 起生成材料的轮次带 `turn_id` 与段，交互记录的事件带响应序号。另注册两个读取器，只用于读回升级前保存的工作：`"1.2"` 的材料没有轮次标识与段、事件没有响应序号，读回时补成空串、空列表与空值，工作照常执行，材料不印段标签，生成的版本记录也就不带引用；`"1.1"` 的修订另外没有 `base_version`，读回时取任务构建时读到的基础记忆（`existing_memory`）的版本补上（只作记录，执行时以重读的版本为底），然后按 1.2 的读回规则继续补齐。补齐后都按当前版本的规则严格解码。解码校验完整字段集与嵌套对象的规范往返，拒绝旧 scope 布局、未知版本、额外字段与身份篡改。底层仍是 `InMemoryWorkStore`，队列本身不持久化；版本化载荷用于进程内冻结和一致性校验，也是关闭时保存工作的格式（第 6.2 节）。保存的载荷在下次启动时由另一个进程解码并重新编码比对，因此载荷必须与进程无关：`GenerationRequest.topic_write_memory_ids` 按排序输出。改动信封格式时，同一改动必须让新版本读得回上一版本保存的工作，否则这些工作在升级后进入 rejected 文件。

`WorkRecord` 是执行状态的唯一真相源，Queue 的 `TaskHandle` 提供 snapshot/wait/cancel；
TaskController 只在查询时映射非终态快照，并在 work 终止后执行一次 settlement 与终态事件。
提交返回的 `MemoryGenerationTask` 不会原地更新，调用方需要通过 list/get/wait 获取新快照。

任务来源（`MemoryGenerationSource`）有四种：`WRITE`、`UPDATE`、`SETTLE` 与 `MERGE`，任务快照与 `memory.task.*` 事件都携带来源。来源同时决定版本记录的来源类型：`MERGE` 任务写出的版本记录来源为 `MERGE`，其余来源的内容修订为 `UPDATE`。

Familiar 内部以 Generation Engine 返回的 `GenerationOutcome`（原子与提交动作 `GenerationAction`）完成 compute、artifact 与 persist；只有这些
步骤成功退出后才投影为跨域 `MemoryGenerationResult`。该结果仅保留 canonical identity、可选的
`PendingAtomSettlement`、合并提名与被吸收记忆的 ID，不会把 `MemoryAtom`、查重判断、提交动作、before snapshot 或 changelog 泄漏到
Controller。由此，Engine 计算语义与 Patchouli 已持久化事实保持明确边界。

已结束的任务在正常关闭时写进任务历史 `patchouli.shutdown.task_history_file`（默认 `.hivememory/memory_tasks.json`），下次启动时、读回保存的工作之前读回，对话页的记忆生成任务列表因此不随重启清空。历史只供 list 与 get：两者在本进程的任务之外合并历史，同一任务 ID 以本进程的为准；历史不能取消或等待，也不参与接纳的幂等判重。读回的历史与本进程已结束的任务共用 `terminal_retention`（100 条）上限，新任务结束时先淘汰最早的历史；写盘与读回都只保留结束于 `task_history_retention_days`（默认 7 天）之内的。关闭时保存的工作（取消原因 `shutdown_persisted`）不写进历史：它在下次启动后继续执行，写入与更新以同一任务 ID 重新出现，历史中留下“已取消”会与它冲突。读回后文件即删除，崩溃时这段时间的历史丢失；读不回的文件改名为同目录的 `memory_tasks.rejected-<UTC 时间>.json` 保留，写盘失败记录 `task_history_write_failed`。

控制面提供 list/get/cancel/wait/wait_many/wait_all。`wait_task()` 直接返回
`MemoryGenerationTask | None`，`wait_many()` 按请求顺序返回快照或 `None` 的列表，不再复制一层
WaitResult/WaitSummary。等待使用 `asyncio.shield`，超时返回当前 `PENDING/RUNNING` 快照，不会接管或
自动取消后台任务。

任务状态通过 RuntimeEvent 发布。TaskController 只决定生命周期事件的发生时机，`MemoryTaskEventEmitter` 集中选择 `memory.task.*` 类型并组装稳定 payload；通用队列仍独立发布 `work.*` 基础设施事件，两者不互相替代。主动任务完成后，settlement 会通过 PatchouliBridge 转发为全局 PendingAtom event，由 workspace registry 更新登记终态；发布失败不会回滚已持久化记忆，Controller 会 best-effort 发布 pending failure。PendingAtom settlement、failed、cancelled 仍属于功能事件并继续留在 TaskController；failed 带上任务失败的原因码（第 6.1 节）。跨重启由 workspace 登记的写盘与启停顺序保证结算事件有落点（[读取视图与写入意图](../workspace/read-view-and-write-intents.md#31-写盘读回与保留期)）；没有事件未送达补偿，不能保证 registry 在通知丢失后自行补齐终态。

合并任务成功且结果带句柄时，Controller 经结算发布者发布改指事件 `patchouli.events.pending_atom.merged`，由 PatchouliBridge 转发为同名全局事件，载荷是 `MERGED` 结算与被吸收记忆的 ID。句柄此前已经以 `CREATED` 结算过一次，改指因此不占用“每个 alias 一次终态”的保留，发布失败只记录 warning，不改写任何任务终态。workspace 登记只对仍以 `CREATED` 指向被吸收记忆的句柄替换结算（[读取视图与写入意图](../workspace/read-view-and-write-intents.md)）。

PendingAtom settlement 只更新意图状态，不回填 canonical 缓存。中期 Store 的独立变更事件负责原子与 alias 派生项失效，所有正式记忆重新读取都经过当前授权；两种事件的作用见[MemoryLibrary](./memory-library.md#12-中期当前可检索书库)。

### 6.1 失败原因码

生成 lane 固定单次尝试，失败分类（`MemoryGenerationHandler.classify_failure`）只决定原因码，不驱动重试或业务判断。按异常自上而下判断：

| 异常 | 原因码 |
|:---|:---|
| `MemoryMergeAbortedError` | `merge_aborted.<details 中的 reason>`，即 `memory_missing`、`not_mergeable`、`version_changed` 或 `model_output_invalid`（第 4.1 节） |
| `ModelContextWindowExceededError` | `model_context_window_exceeded` |
| `ModelProviderError` | `model_provider_failed` |
| `TimeoutError` | `memory_generation_timed_out`：生成 lane 的时限（`patchouli.generation.queue_timeout_seconds`，默认 300 秒）到期，或数据面内部抛出的超时 |
| 其他领域错误（`DomainError`） | 该错误的 `code` |
| 其他异常 | `memory_generation_failed` |

超窗是提供商失败的子类，先于它判断。原因码写入工作记录，由 `work.failed` 投影；Controller 发布 `memory.task.failed` 时 `reason` 取工作记录中的原因码，载荷的 `error_class` 为异常类名，取消终态的 `reason` 仍是取消原因。payload 解码失败与分类器自身出错不经过这张表，原因分别为 `invalid_work_payload` 与 `failure_classifier_error`。

同一次失败只在分类处记一行 error 日志：`work_id`（`memory_generation:<task_id>`）、原因码与异常类名；领域错误另记 `details`，未预期的异常带堆栈，提供商失败与超时不带堆栈（提取器重试前已以 warning 记录第一次失败的原因）。日志不带话题与来源，它们在同一任务的 `memory.task.failed` 中，按 `task_id` 在运行时事件记录文件里对应（[可观测性](../components/observability.md#51-事件记录文件)第 5.1 节）。任务快照与任务 API 不变，公开 `error` 仍是固定说明。写入与更新任务失败时，写入意图的失败事件带上同一个原因码，workspace 登记记在句柄上，读取失败的句柄时交付；接纳前的确定性拒绝没有原因码。分类与日志由生成队列的单元测试覆盖，原因码到 `work.failed` 与 `memory.task.failed` 的传递由控制器的单元测试覆盖。

### 6.2 提交阶段、关闭时保存与启动时继续

Familiar 把一次任务分成两段：

- **提交之前**：为结算保存来源 InteractionArtifact，调用模型提取、修订或合并（结算选馆长时是馆长的整个循环，含它对中期库的读取），查重；合并任务另有写回之前的重读与校验，修订写回之前重读基础记忆（第 3.3 节）。这一段除了交互 Artifact 不写任何东西。
- **提交**：写版本记录、upsert 中期存储、提升附件；合并任务是写回保留记忆、删除被吸收的记忆。只有本地写入，一般在秒级。

开始提交之前，Familiar 同步调用执行路由 `generation.execute_spec` 的 `on_commit_started`，控制器把它记在任务句柄上。提交这一段以 `join_shielded` 执行，期间收到的取消等写完再传出，因此在提交之前被中止的任务可以从头重跑，不会留下写了一半的记忆。用户取消或 lane 时限恰好落在提交阶段时，任务写完后以取消或超时结束：写入意图报告取消或失败，记忆已经落库；写入本身出错时，错误随取消一起丢弃。

后端关闭时不等待模型调用，Patchouli 依次：

1. 生成队列停止取新工作，入队照常接受；
2. 结算全部驻留话题，SETTLE 任务只入队、不执行（[感知与短期话题](./perception.md#7-维护与关闭)）；
3. 在跑的任务中，还没开始提交的立即取消，按未执行保存；已开始提交的写完再结束，结算与合并提名照常完成，提名的合并入队后一并保存。取消与开始提交之间没有原子性，因此以是否开始提交、而不是是否被取消决定要不要保存，不会重复生成；
4. 队列中全部未执行的工作按入队顺序写进 `patchouli.shutdown.pending_work_file`（默认 `.hivememory/pending_generation.json`，相对路径按工作目录解析），先写临时文件再原子替换，没有工作时不写。每项是工作信封的 `kind`、`schema_version` 与规范载荷；SETTLE 带上本话题主动写入的记录（第 4.2 节）；
5. 被保存的任务在本进程内以取消结束，原因 `shutdown_persisted`；其中的 WRITE、UPDATE 不发布写入意图的取消事件，它们在下次启动后还会结算。

文件没写出（例如磁盘错误）时记录 `saved_work_write_failed`，这些工作随进程结束丢失。

下次启动时，Patchouli 在生成队列启动之后读回保存文件：

- 先删除文件，再逐项经控制器重新接纳，与新任务走同一条路；只入队，执行在后台进行，不阻塞启动。接纳过程中崩溃会丢失这些工作，但不会在下一次启动重复生成。
- 整个文件无法解析或格式版本不认识时，文件改名为同目录的 `pending_generation.rejected-<UTC 时间>.json`；单项的 `kind` 或 `schema_version` 不认识、载荷不规范或接纳被拒绝时，这些项写进同样命名的文件，其余照常接纳。两种情况都记录错误日志并发布事件，启动继续。
- 读回的工作以新任务出现，主动写入的任务 ID 沿用 `active:{intent_id}`。写入意图登记同样在正常关闭时写盘，并在 Patchouli 启动之前读回，恢复的写入结算时事件落到原句柄上（[读取视图与写入意图](../workspace/read-view-and-write-intents.md#31-写盘读回与保留期)）。

观测：`patchouli.shutdown_drain` 的完成事件报告按来源分的保存数、等待提交结束的任务数与文件是否写出，没写出时状态为 `completed_without_saved_work`、级别 error。读回作为一次 `patchouli.restore_saved_generation` 操作发布开始与完成事件，报告读回、重新接纳与写进 rejected 文件的数量；有内容进入 rejected 文件时状态为 `completed_with_rejections`、级别 error。

停止取新工作由工作队列的单元测试覆盖，提交阶段不被取消打断由 Familiar 的单元测试覆盖，保存、读回与 rejected 文件由控制器的单元测试覆盖；“关闭时结算产生的任务不调用提取、新进程读回后落库”由[生成流水线集成测试](../../tests/integration/patchouli/test_memory_generation_pipeline.py)验证。

## 7. 实时派发与交互提交

主动写入与交互提交分别执行：

```text
WRITE/UPDATE -> 授权 -> 取得派发话题 -> workspace 登记 -> memory_intent.submit
  -> Coordinator 构建材料与 spec -> Controller 接纳生成任务 -> ACK
  -> 后台生成、持久化与结算事件

CPU 有输出（completed/cancelled/failed）-> 封口 InteractionPayload
  -> root API 提交交互 -> interaction.submit：原样提交，等待交互应用
  -> 写入会话（话题取实际所在的话题）
```

每次提交只携带一个意图：Coordinator 的 `submit_active()` 为它构建 spec，交给 Controller 的 `admit_generation()`，返回单个 `MemoryGenerationAdmission`。accepted 携带新建或幂等复用的任务快照；rejected 由 Controller 结算对应 PendingAtom failed，不伪造 FAILED 任务；unknown 不结算，可继续由实际接纳的任务给出终态。Coordinator 在构建 spec 时遇到确定性失败（如 UPDATE 基础不存在）同样结算 failed 并返回 rejected，遇到无法判断的失败或接纳请求本身失败则返回 unknown。稳定的 `active:{intent_id}` 是任务标识，同一 intent 携带不同 spec 重放时结果为 unknown，既有任务保留结算权，不能抢先发布 failed。

Interaction applied 仍是 Chat 的硬成功边界，交互提交不派发主动任务，也不返回任务列表。提交任务由任务进程持有并以 shield 等待，HTTP/SSE 调用方离开不会中止已经开始的提交；已接纳的交互也不随调用方取消撤回。CPU 失败、取消、进程关闭与失败 CALL 子帧都不撤回此前的显式写入。

主动生成不等待当前交互 applied，不读取当前轮，也不会为了 WRITE/UPDATE 调用话题 settlement、summary 或 clear。有输出轮次的当前交互留在话题中，之后可在 idle/LRU/shutdown 时参与 Mode A；主动保存与被动归档不能通过清空 buffer 相互替代。写入建立的话题若最终没有接收交互（例如写入后 CPU 没有输出），不被补偿删除，由空闲结算或 LRU 按没有材料的话题结束；已派发任务保留该 topic_id 作观测，正式原子不保存 topic_id。

这条解耦边界是对旧链路实际问题的回应。旧实现曾把 MTP WRITE/UPDATE 当作话题结算理由，执行 `archive + compact + clear`：连续两次 WRITE 时，第二次只能看到已经被压扁的 summary；每次主动写还会额外触发一次 summary，使原始细节在明确保存意图下反复丢失。当前实现直接使用 topic 最近 context，并以 `WriteFocus`/`UpdateFocus` 驱动 Mode B/C；主动生成与被动归档保持两条正交流，主动写入后的 blocks 仍可在之后进入 Mode A。

这并不保证主动写出的正文与被动归档结果完全不重复。一次性生成中，同一话题内复述主动写入的被动草稿由查重丢弃（第 4.2 节）；馆长在写之前查看本话题的写入与库中相关记忆，最后仍有的重叠接受（第 3.1 节）；跨话题或相似度不高于下限的重复是当前可接受的成本，由查重决定新建、只触碰已有记忆，或新建后合并进已有记忆；为了消除重复，不能重新让 Perception 感知 focus 或 pending 状态，否则会把控制意图污染为对话事实。当前每个主动 task 也会独立捕获 `InteractionArtifact`，因此 provenance 可能重复，这是已知的可追踪性成本，而不是重新合并任务输入的理由。

## 8. 当前限制

- 检索的稠密与稀疏向量以及重排输入都只由标题、类型、标签与摘要编译（MemoryCompiler 的 `DENSE_EMBEDDING`、`SPARSE_EMBEDDING` 目标），只写在一条记忆正文里的细节检索不到。话题结算把彼此独立的事拆成多条记忆，各自有标题与摘要，但拆分粒度由模型按提示词判断，没有校验。真实样本见已归档的 [Import Bus 真实使用观察](../archive/todo/import-bus-real-use-observation.md)；
- 同一次结算的多条草稿之间不互相查重，只与已经落库的记忆比较；两条描述同一件事的草稿会各自新建；
- 一次结算的多条结果依次提交，中途失败（例如存储或版本记录故障）时已提交的记忆保留而任务失败；控制面只在任务成功时提交合并提名，这一批的合并因此不会发生，留下重复；
- 一段话题的全部草稿共用一次调用的输出上限：草稿多、正文长的话题被截断时任务失败，原文仍在来源 InteractionArtifact 中，但没有记忆引用它；
- 派发重试后结局仍然不明、或结算事件丢失时，句柄保持 MATERIALIZING，直到写入句柄的保留期满被删除；没有对账恢复。写入一旦开始派发就不因 CPU、进程或 CALL 子帧失败取消；
- 正常关闭时未执行的工作保存到磁盘、下次启动后继续（第 6.2 节），崩溃或强制结束时随进程丢失；typed result 只存在于当前进程，已结束任务的快照在正常关闭时写进任务历史（第 6 节），崩溃时丢失；持久化与恢复边界见[耐久性与故障恢复治理](../governance/reliability/durability-and-recovery.md)；
- 终态快照（含读回的历史）与 Queue 使用同一 `terminal_retention` 上限；
- Controller 的 admission 持有同一把锁串行执行，并发提交的意图按取得锁的顺序入队；
- 提取、修订、合并与馆长的 LLM 调用没有独立超时，只受生成 lane 的时限（默认 300 秒，到期以 `memory_generation_timed_out` 失败）与取消约束；
- 后台抽取与 chat 共用提供商配额，限流表现为提供商失败：一次性生成、馆长与合并任务按任务失败收尾（合并失败时两条记忆都保留），Mode B/C 使用 fallback；当前没有生成侧的并发控制；
- Mode A/B 去重仍只取 dense top-1；检索调用链独立携带归属与发起者，Memory store/adapter 先强制 Workspace ownership hard filter，再重验 actor policy；其他去重策略限制仍待收敛；
- 查重只按 dense top-1 余弦与固定阈值判断，分不开同一工作与同项目的其他记忆：误判的两条无关记忆也会被合并，只能经版本记录与人工编辑纠正；判定信号的证据与后续见[查重决策信号 Todo](../todo/generation-dedup-decision-signal.md)；
- 同一工作的各阶段不断并入同一条记忆，正文持续变长，摘要越来越难覆盖全部要点；正文过长时合并可能超出模型窗口，任务失败，两条记忆都保留；
- 新建到合并完成之间，检索与前端同时看到 M 与 N；
- 改指事件与结算事件属于同一可靠性等级，没有补偿：丢失时句柄仍指向已删除的 N，读取得到不带原子的重定向；
- 合并的吸收关系只在进程内，只增不减：关闭时保存的合并若以已被吸收的记忆为目标，下次启动后改指不到，按 `memory_missing` 中止，两条都保留；
- 同一 Workspace 的合并依次执行，在门外等待的合并仍占着生成 lane 的一个执行位（并发为 2），同一时间的结算可能排在合并之后；
- 一次 `replace` 可以换掉整个标题或摘要；合并讲的不是同一件事时没有机械的拒绝，靠查重提名与合并提示词把关，两条是否融合交给维护；
- 本话题主动写入的记录只在进程内（正常关闭时随保存的结算任务写盘），进程崩溃时丢失，之后的结算退回不区分来源的查重；话题未经结算即被删除，或结算开始之后才完成的同话题写入，留下的记录保留到进程结束；
- 生成 lane 不按话题排序：结算开始时仍在执行的同话题写入不在记录中，它与被动草稿之间照旧按查重合并；
- 复述写入但相似度不高于下限的被动草稿照常新建；话题路由错误时（[跨会话话题路由 Todo](../todo/cross-conversation-topic-placement.md)），另一段对话里与这些写入相似的草稿也会被丢弃；被动草稿里比主动写入更细的内容随草稿一起丢弃；
- 生成材料只截断工具结果，单个轮次没有总量上限，生成也不分片读取：工作视图保留的 blocks 很大时（例如单轮上百次工具调用），结算的提取可能超出模型窗口，任务以 `model_context_window_exceeded` 失败（超窗不重试），原文仍在来源 InteractionArtifact 中但没有记忆引用；
- Active tasks 复用相同 blocks 输入，但会各自写 InteractionArtifact；关闭时在提交之前被中止的结算任务重跑时再保存一份来源 InteractionArtifact，前一份没有记忆引用；
- 馆长不提名合并，只修订读过的记忆：一次性生成留下的相近记忆不会被它合并，两种方式产出的记忆在库中混合；模型没有查到或判断为不重叠的重复照常留下；
- 同一 Workspace 的两次馆长结算可以同时运行（生成 lane 并发 2），彼此看不到对方暂存的记忆；同一次结算暂存的草稿之间也不查重；
- 馆长只读中期库中的记忆本身：不读其他话题的原始轨迹（经段引用回到原文需要读取交互快照的组件，现在没有）、冷藏库与版本记录；本话题由 UPDATE 修订过的私有记忆不会被列出；
- 馆长的动作是文本中的 JSON，模型不按协议输出时只能交回错误、消耗步数；它不重试提供商失败，一次结算的多次调用中任一次失败，整次结算失败、暂存的草稿不提交；
- memory 版本 Artifact 失败会阻断该次内容提交（无历史不成功）；provenance 写入与 MemoryAtom 发布仍不是原子事务，canonical 失败会留下孤立 Artifact。

未来若引入 durable queue，首先要保持私有工作信封 codec、`WorkRecord` 单一状态源和 settlement 语义不变，避免基础设施升级重新把控制面塞回 GenerationEngine。
