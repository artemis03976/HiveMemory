---
title: 工具解析链路重写
status: archived
archived_at: 2026-10-30
implemented_by: v0.7.1 工具解析链路重写（archive/plans/v0.7.1-tool-call-parsing-chain.md）
superseded_by: docs/architecture/data-model.md
owner: project
scope: turn-event-as-single-interaction-record-for-mtp-and-mcp-tool-calls
code_paths:
  - src/hivememory/core/models/interaction.py
  - src/hivememory/core/protocol/sealing.py
  - src/hivememory/core/protocol/models.py
  - src/hivememory/agent_runtime/loop.py
  - src/hivememory/agent_runtime/runtime.py
  - src/hivememory/system/services/import_bus/models.py
  - integrations/claude-code/connector/claude_code/reader.py
  - src/hivememory/mcp_server/server.py
  - src/hivememory/prompts/transcript/
  - src/hivememory/alice/context/references.py
  - src/hivememory/server/models/session.py
  - frontend/src/stores/chat/sessionHistory.ts
related_docs:
  - docs/governance/baselines/tool-call-data-shapes-2026-10-30.md
  - docs/archive/plans/v0.7.1-tool-call-parsing-chain.md
  - docs/patchouli/perception.md
  - docs/system/import-bus.md
  - docs/contracts/mtp.md
  - docs/contracts/mcp.md
  - docs/ideas/long-running-agent-intra-turn-context-folding.md
last_reviewed: 2026-10-30
---

# 工具解析链路重写

> 状态：Archived（2026-10-30，随[工具解析链路重写计划](../plans/v0.7.1-tool-call-parsing-chain.md)实施后归档；当前事实见[数据模型](../../architecture/data-model.md)、[感知与短期话题](../../patchouli/perception.md)与 [Patchouli 生成](../../patchouli/generation.md)）。以下为设计讨论原文；实施中另决定把一次工具调用与它的结局合并为一个事件，与本文第 3–6 节的调用、结果分开的形态不同，以计划第 5.1 节为准。

**文档状态**：Idea。2026-10-30 已形成[计划](../plans/v0.7.1-tool-call-parsing-chain.md)：本文的候选设计在计划中采纳或调整，此后以计划为准，本文不再同步。

## 0. 文档性质

- **owner 决定（2026-10-30）**：
  - 现有的 `ActionReducer` 与 `TraceReducer` 都不符合预期；`InteractionPayload` 同时保留 `mtp_traces` 与 `turn_events`，后续解析的来源不一致；后续各处对工具路径的渲染行为也不可见。整条工具解析链路需要重写。
  - `TurnEvent` 必须承载唯一的 Agent 交互原子事件。重写分两部分：第一部分是 MTP 与 MCP 的工具调用结果如何统一进入 `TurnEvent`；第二部分是感知层如何解析这一份 `TurnEvent` 集。先完成第一部分。
  - `tool_name` 与 `verb` 不得重复表达同一个调用对象：HiveMemory 的 MCP 工具与动词一一对应（第 3.2 节）。
  - HiveMemory MCP 工具的参数在连接器归一化（原待决问题 TC-1，第 6 节）。
  - MTP 调用事件的 `content` 保留指令原文（第 3.2 节）。
  - `InteractionPayload` 只能携带 `turn_events`，由感知层解析：交互内容只以事件表达，派生的与和事件重复的表达不进入载荷；`outcome`、`model_used`、`used_attachments` 保留（第 7 节）。
- **本文范围**：第一部分见第 3–6 节，第二部分见第 7–9 节；两部分共用第 2 节的原则，迁移与暂不处理合并在第 10、11 节。
- **现状依据**：[工具调用数据形态调查基线](../../governance/baselines/tool-call-data-shapes-2026-10-30.md)，下文称“基线”。字段级现状以基线为准，本文不复制。
- 标注“候选设计”的内容不是 owner 决定，给出的是合理默认值，可在计划评审时调整。两部分目前都没有待决问题。

### 0.1 与其他 Idea 的关系

- **[外部会话 Idea](../../ideas/external-session-and-topic-projection.md#23-turnevent复用并扩展事件表达) 2.3**：那里的候选扩展有 system_message、external_summary、unknown 等事件种类，事件时间与来源元数据，以及可选的 `action_id`。本文只定义工具调用相关的字段，不新增事件种类。本文要求工具事件必带 `action_id`，因为两个现有来源都有调用标识；没有调用标识的 harness 出现时，由它的连接器生成。那一节的候选扩展待本文落地后，依据新的事实修订。
- **[执行环境 Idea](../../ideas/execution-environment.md#t-10-cpu-输出与会话历史的词汇归属) T-10**：其中“会话历史”一项（MTP 原文、`tool_kind` 取 MTP 动词、`render_as`）由本文的设计处理：动词作为 HiveMemory 能力动词对两种来源统一，`render_as` 删除。交互事件与失败原因两项不在本文范围。

## 1. 问题

基线第 8 节列出的问题中，下面这些属于第一部分，即工具调用进入 `TurnEvent` 这一侧：

- **字段含义重叠**：`tool_name` 在 MTP 一侧是指令目标，在外部一侧是工具名；MTP 的 `target` 与 `tool_name` 重复；读取方取值的优先级互相矛盾。
- **两条链路不同形**：两条链路写出的事件在 `role`、`content`、`target`、`tool_args`、`status`、`render_as` 上都不同。下游如果要正确处理，就得知道事件来自哪一侧。
- **同一事实写在多处**：
  - MTP 调用事件的 `content` 含指令前正文，这段正文已另存为一条 `assistant_message`；
  - MTP 结果事件重复调用事件的 `tool_kind`、`tool_name`、`target`；
  - 调用事件的 `status` 在执行后被改写为结果的状态。
- **派生表达随载荷传递**：`mtp_traces` 由事件派生，却和事件一起随 `InteractionPayload` 传递。
- **结构化事实写进正文再解析回来**：WRITE、UPDATE 的 ACK 中的 pending alias 生产时就已知道，却只写进回填正文；Alice 的引用线索再用正则从正文里取回。
- **没有消费方的字段**：`assistant_message` 上的 `status`、按 `render_as` 补系统前缀（回填正文本身已带标题）。

下面这些属于第二部分，即感知层解析这一侧（现状见基线第 7 节）：

- **解析分散在多处**：动作在封口与感知各归约一次，轨迹沿用载荷；生成视图按 `action_id` 把动作与轨迹拼回去，LLM 接力摘要只读轨迹，规则摘要读动作的 `tool_name`，token 统计两者混用。
- **各处看到的材料不同**：生成视图含思考与外部工具；LLM 接力摘要只有 MTP 轨迹与问答，被折叠的块在结算前失去思考与工具过程的影响（[真实使用观察 Todo](../todo/import-bus-real-use-observation.md) 已记录）。
- **正文与动作的先后丢失**：生成视图把全部动作排在前面、最终回复排在最后；而 Alice 的回复由各次迭代的正文拼成，与动作本来是穿插的。
- **载荷与逻辑块携带重复或无用的字段**：`user_message`、`assistant_final_text` 与事件重复；`worth_saving` 不参与任何决定；逻辑块上的 `gateway_intent` 从未被赋值，`anchor_text`、`is_complete` 没有消费方。
- **渲染不可见**：渲染规则分散在生成视图、接力摘要与 token 统计各自的代码中，渲染出的材料只在新建记忆时随创建 Artifact 保存。

## 2. 设计原则

候选设计，依据第 0 节的 owner 决定。

1. **唯一记录**：一轮交互的过程只有一份 `TurnEvent` 集。动作、轨迹、生成材料等一切表达都在读取时从它派生，不随载荷传递，也不持久化成第二份事实。
2. **同形**：两种来源写出的事件，字段含义完全相同。下游不需要、也不能通过判断来源来解释字段。来源之间真实存在的差异，只体现为字段的取值（例如结构化调用没有调用原文），不体现为字段含义不同。
3. **一个事实一个字段**：
   - 调用了什么只由一个字段表达：HiveMemory 能力用 `verb`，外部工具用 `tool_name`，二者互斥；
   - 调用一方的事实（调用了什么、对什么、带什么参数）只写在调用事件上；
   - 结局一方的事实（返回了什么、状态如何、产出了什么）只写在结果事件上；
   - 两者以 `action_id` 配对，结果事件不重复调用事件的字段。
4. **记录执行者实际收发的内容**：`content` 是执行者写出或看到的原文：Alice 看到的回填文本，或 Claude Code 收到的工具返回。重放与审计据此还原，不在读取时重新格式化。
5. **生产时已知的结构化事实直接结构化记录**，不写进正文后再解析。

## 3. 候选设计：统一后的 TurnEvent

### 3.1 各种事件的字段

| kind | content | action_id | tool_name | verb | targets | tool_args | status | produced_aliases |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| user_message | 用户消息 | — | — | — | — | — | — | — |
| assistant_message | 助手正文的一段 | — | — | — | — | — | — | — |
| thought | 思考正文 | — | — | — | — | — | — | — |
| tool_call | 调用原文：MTP 为指令片段，结构化调用为空串 | 必填 | 外部工具的名字；调用 HiveMemory 能力时为空 | HiveMemory 能力动词；调用外部工具时为空 | 动词的作用对象，可为空 | 动词或工具的参数，可为空 | — | — |
| tool_result | 执行者收到的返回原文 | 必填，指向本轮已出现的调用 | — | — | — | — | 必填 | 本次调用产出的 alias，可为空 |

另有 `kind` 与 `sequence` 两个公共字段。

### 3.2 字段语义

- **`sequence`**：轮内唯一、严格递增；用户消息固定为 0。
- **`content`**：见原则 4。
  - MTP 调用事件只存指令片段（例如 `⟪ READ | [fact_a, fact_b] | ⟫`），指令前正文只存在于它前面那条 `assistant_message`；
  - 结构化工具调用（MCP、普通工具）没有调用原文，为空串。
  - MTP 指令片段与 `verb`、`targets`、`tool_args` 描述的是同一次调用，但不算同一事实写两遍：片段是执行者写出的原话，供重放还原；结构化字段是 HiveMemory 对它的解析，下游只按结构化字段解读。解析失败时只有原话。owner 决定（2026-10-30）保留原话，不改为重放时由结构化字段重新生成指令文本（那样重放出的文本与模型当时写的不完全一致）。
- **`action_id`**：调用事件必填且轮内唯一，结果事件必填且指向本轮已出现的调用。来源：
  - MTP 为 `action_{iteration}_{sequence}`；
  - Claude Code 为 `tool_use.id`。
  - 思考与正文不带 `action_id`：它们与动作的先后关系由 `sequence` 表达，是否归属某个动作由第二部分决定。
- **`verb`**（取代 `tool_kind`）与 **`tool_name`**：二者互斥，共同回答“调用了什么”。
  - `verb` 是 HiveMemory 能力动词，取 SEARCH、READ、RUN、WRITE、UPDATE、CALL。MTP 调用取指令动词；外部 harness 调用 HiveMemory 的 MCP 工具时取对应的动词（目前 `memory_search` 为 SEARCH，`memory_read` 为 READ），harness 给工具起的调用名不记录：工具与动词一一对应，调用名只多出 harness 的命名规则，HiveMemory 用不到。
  - `tool_name` 是外部工具的名字，例如 `Bash`、`Read`，不再用来存放目标。
  - `verb` 非空即表示“这是对 HiveMemory 能力的操作”，回声去重等规则据此判断。
  - 不合并为一个字段：动词是封闭词表，外部工具名是开放词表，两者会重名（Claude Code 有 `Read`、`Write` 工具）。合并后只能靠命名约定区分，读取时又要解析字符串。
  - MTP 指令无法解析时二者都为空，原文仍在 `content` 中，对应结果为 error。
- **`targets`**（取代 `target`）：动词的作用对象，元组。
  - READ 为要读取的 alias 列表；
  - RUN 为工具 alias 或代码记忆 alias；
  - UPDATE 为被更新的 alias；
  - CALL 为被调用 Agent 的 alias；
  - SEARCH 与 WRITE 的通配目标记为空元组；
  - 不再以逗号拼接成一个字符串。
- **`tool_args`**：只放参数，不放作用对象。
  - MTP 为指令参数（值为字符串），例如 SEARCH 的 `query`、`filter`，WRITE 的 `content`、`title`、`reason`；
  - MCP 调用的参数里属于作用对象的部分移入 `targets`，例如 `memory_read` 的 `aliases`；其余参数保留，例如 `memory_search` 的 `query`；
  - 两种来源的 SEARCH 因此都从 `tool_args["query"]` 取查询；
  - 没有参数时为空。
- **`status`**：只在结果事件上，取 `success`、`ack`、`error`、`cancelled` 之一。
  - MTP 取回填的响应状态。`warning` 从不作为状态产生（警告挂在 success 上），不纳入；`suspend` 只是 CALL 的中间状态，不写入事件。
  - MCP 结果为 `error`（`is_error`）或 `success`。
  - 调用没有结果事件，表示这次调用没有完成（被停止或交付方离开）。
- **`produced_aliases`**：本次调用产出的 alias。
  - WRITE、UPDATE 的 ACK 为 pending alias；
  - CALL 为子帧回填的 `artifact_aliases`；
  - 其他为空。
  - 生产使用方是 Alice 引用线索：不再用正则从正文取 ACK 中的 alias。

### 3.3 删除的字段

| 字段 | 去向 |
|:---|:---|
| `role` | 由 `kind` 决定：user_message 为用户；assistant_message、thought、tool_call 为助手；tool_result 是工具返回。Alice 重放时把工具返回放进 user 消息，这是 Alice 自己的渲染规则，不是事件数据 |
| `target` | 由 `targets` 取代 |
| `tool_kind` | 由 `verb` 取代 |
| `render_as` | 回填正文本身已带标题；Alice 减重原先靠它识别 CALL 回填，改为看调用事件的 `verb` |
| 调用事件上的 `status` | 结局只在结果事件上 |
| assistant_message 上的 `status` | 没有消费方；本轮结局由 `InteractionPayload.outcome` 表达 |

### 3.4 一轮事件集的不变量

- 恰有一条 `user_message`，序号为 0；序号轮内唯一、严格递增。
- 每个调用的 `action_id` 轮内唯一；每个结果都指向本轮更早出现的调用；每个调用至多一个结果。
- 字段只出现在 3.1 表中允许的事件种类上。`verb` 与 `tool_name` 不同时出现；`verb` 为空时 `targets` 为空；`produced_aliases` 只出现在结果上。

### 3.5 校验位置与载荷边界

- 单个事件的字段规则由 `TurnEvent` 自身的校验执行；一轮的不变量由 core 中的一个函数校验，`InteractionPayload` 构造时调用它，因此任何载荷都满足不变量，违反时显式失败，不修补。共用封口 `seal_interaction` 只负责组装。
- Import Bus 在请求校验阶段提前执行同样的规则，违反时返回 422，不等到封口。
- 载荷的边界见第 7 节。

## 4. 两个来源如何写出

### 4.1 Alice（MTP）

Alice 执行循环写出事件的规则（现状见基线第 4.3 节）：

| 时机 | 写出的事件 |
|:---|:---|
| 生成中断于 MTP 指令，且指令前有正文 | `assistant_message`，content 为 `prefix_text` |
| 指令解析成功 | `tool_call`：content 为指令片段；`verb` 为指令动词；`targets` 取 `MTPTarget.aliases`（通配为空）；`tool_args` 为指令参数 |
| 指令解析失败 | `tool_call`：content 为指令片段，`verb` 与 `tool_name` 为空；随后是 error 结果 |
| 指令执行完成（CALL 除外） | `tool_result`：content 为回填文本；`status` 为响应状态；`produced_aliases` 取 `pending_alias` |
| CALL 挂起 | 只写 `tool_call`，不写结果；回填时不再改写调用事件，结局只写在结果上 |
| CALL 回填 | `tool_result`：content 为格式化的 CALL 响应；`status` 为子帧结局；`produced_aliases` 取 `artifact_aliases` |
| 取消或失败时的半截正文 | `assistant_message`，不带状态 |

- 执行器对含 `⟪` 的文本不会返回空结果。现有的“执行器没有结果时写失败结果”分支在生产中走不到，删除；遇到时显式失败。
- 运行时的 working history 不变：助手消息仍是指令前正文加指令片段的全文。Alice 重放会话历史时，把紧邻的 `assistant_message` 与随后的 `tool_call` 合并成一条助手消息，与运行时一致。这顺带消除了基线第 8 节第 7 项的重复。

### 4.2 外部 harness（Claude Code 与 HiveMemory MCP）

| Claude Code 会话记录中的块 | 写出的事件 |
|:---|:---|
| `text` | `assistant_message` |
| `thinking`（非空白） | `thought` |
| `tool_use` | `tool_call`：content 为空串；`action_id` 为 `tool_use.id`。外部工具填 `tool_name`（调用名）与 `tool_args`；HiveMemory 的 MCP 工具只填 `verb` 与 `targets`（连接器归一化，见第 6 节），不记录调用名，参数中属于作用对象的部分移出 `tool_args` |
| `tool_result` | `tool_result`：content 为返回文本；`status` 为 `error`（`is_error`）或 `success`；不带调用一方的字段 |

- 导入契约随之改为接受 `verb`、`targets` 与必填的结果 `status`，不再接受 `tool_kind`。`role`、`render_as` 与第 3.3 节删除的字段一律拒绝。
- 连接器跳过子代理内部过程、只记签名的空白思考等取舍不变。

### 4.3 同一次读取在两侧的形态

读取两条记忆：

| 字段 | Alice（MTP） | Claude Code（MCP） |
|:---|:---|:---|
| tool_call.content | `⟪ READ \| [fact_a, fact_b] \| ⟫` | 空串 |
| tool_call.action_id | `action_2_3` | `toolu_…` |
| tool_call.tool_name | 空 | 空 |
| tool_call.verb | READ | READ |
| tool_call.targets | `(fact_a, fact_b)` | `(fact_a, fact_b)` |
| tool_call.tool_args | 空 | 空 |
| tool_result.content | 带标题的 `<mtp_response>` 回填 | 编译后的记忆全文 |
| tool_result.status | success 或 error | success 或 error |

两侧只在调用原文与返回原文上不同，这些都是真实存在的差异。下游只看 `verb`、`targets`、`tool_args`、`status` 就能得到相同的解释。

已知差异：读取部分 alias 未命中时，MTP 在 success 结果上附带警告，MCP 也返回 success 并在正文末尾列出未命中的 alias。两侧的 `status` 都是 success，未命中的信息只在正文里。

## 5. 随之调整的读取方（感知层以外）

| 读取方 | 调整 |
|:---|:---|
| Alice 会话历史重放（`prompts/transcript/history.py`） | 角色由 `kind` 决定；合并指令前正文与调用（第 4.1 节）；不再按 `render_as` 补前缀 |
| Alice 工具结果减重（`prompts/transcript/reduction.py`） | 动词与目标取自配对的调用事件；CALL 回填按 `verb` 识别 |
| Alice 引用线索（`alice/context/references.py`） | READ 的每个 alias 各算一条引用，修复 `memory:a,b` 被当成一条的问题；ACK 的 alias 改读 `produced_aliases` |
| 会话 HTTP（`server/models/session.py`） | 响应字段改为新字段集 |
| 前端会话历史（`frontend/src/stores/chat/sessionHistory.ts`） | 改读 `verb`、`targets`；MTP 卡片的命令文本只剩指令片段 |
| 交互 Artifact 快照、感知、记忆生成视图、接力摘要 | 第二部分，见第 8、9 节 |

Alice 的流式交付事件（`mtp_start`、`mtp_result`）是交给前端的传输事件，不是 `TurnEvent`，不在本次范围。

## 6. HiveMemory MCP 工具参数的归一化

**owner 决定（2026-10-30）**：在连接器归一化（原待决问题 TC-1）。

问题背景：`memory_read` 的 `aliases` 要移入 `targets`。工具的参数结构由 HiveMemory 的 `mcp_server` 定义，与 harness 无关；harness 给工具起的调用名因 harness 而异，识别“这是 HiveMemory 的 MCP 工具”只能在连接器里做。当时的另一个选项是由 Import Bus 归一化：参数映射只有服务端一份，所有 harness 共用，但 MCP 工具的参数结构要定义在 core，供 `mcp_server` 与 Import Bus 共用（Import Bus 位于 system 层，不能导入入口层的 `mcp_server`）。目前只有一个 harness，没有为第二个实现新增这一概念。

候选设计：

- **连接器**：识别表由“调用名 → 动词”扩展为“调用名 → 动词与作用对象参数”：`memory_read` 为 READ，作用对象是 `aliases`；`memory_search` 为 SEARCH，没有作用对象。作用对象参数从 `tool_args` 移入 `targets`，其余参数保留在 `tool_args`；调用名不上传。
- **Import Bus**：不认识 MCP 工具的参数结构，只校验通用规则：导入的 `verb` 只能是 SEARCH 或 READ；READ 的 `targets` 非空；`verb` 为空时 `targets` 为空；`verb` 与 `tool_name` 不同时出现。违反时返回 422。
- **同步**：MCP 工具的参数改名时，`mcp_server` 与连接器的识别表在同一次改动中修改；连接器的测试覆盖识别表到事件字段的映射。
- 接入第二个 harness 时，再评估是否把参数映射收回服务端。

## 7. 第二部分：载荷边界

**owner 决定（2026-10-30）**：`InteractionPayload` 只能携带 `turn_events`，由感知层解析。含义是交互内容只以事件表达，派生的与和事件重复的表达不进入载荷；载荷仍是封口交互的数据类型，`outcome`、`model_used`、`used_attachments` 保留。

### 7.1 各字段的处理

候选设计：

| 现有字段 | 处理 | 理由 |
|:---|:---|:---|
| `turn_events` | 保留，是唯一的交互记录 | — |
| `mtp_traces` | 删除 | 由事件派生 |
| `user_message` | 删除，由序号 0 的事件得到（第 7.2 节） | 与事件重复 |
| `assistant_final_text` | 删除，由 `assistant_message` 事件得到（第 7.2 节） | 与事件重复 |
| `rewritten_query` | 删除 | 是 Gateway 对用户消息的改写，不是交互事件；感知层只有接力摘要使用它，改为用用户消息与本轮过程（第 8.4 节） |
| `worth_saving` | 删除 | 不参与任何决定，只供观测；Import Bus 的观测事件直接取 Gateway 的决定，不经载荷。它与 Gateway 价值预判的整体去留仍随[生成材料 Todo](../todo/generation-material-filtering-and-truncation.md)与记忆价值策略决定 |
| `outcome`、`model_used`、`used_attachments` | 保留（owner，2026-10-30） | 本轮的结局、执行者使用的模型与实际使用的附件，不是对事件的解析，也不与事件重复 |

### 7.2 从事件得到的两段文字

用户消息与回复正文有多个读取方，其中有些在感知层以外：Gateway 的会话上文与话题上文、话题快照的最后一轮、会话 HTTP 与会话标题。

候选设计：

- 由 core 中两个函数从事件集得到，载荷与逻辑块都以只读属性提供，不保存、不序列化。
- **用户消息**：序号 0 的 `user_message` 的正文。
- **回复正文**：全部 `assistant_message` 按序号以空行拼接。两种来源统一按空行拼接：Alice 自己的 `final_text` 直接相连，连接器以空行拼接。CPU 交给流式 `done` 的最终回复仍由 CPU 组装，不受影响。
- 它们只是读取事件正文，不解释工具调用；对工具过程的解析只在感知层（第 8 节）。

## 8. 第二部分：感知层的解析与渲染

### 8.1 解析

候选设计：

- **输入**：载荷中满足第 3.4 节不变量的事件集。感知层不再修补或容错。
- **输出**：用户消息之后按序号排列的**步骤**，每一步是以下之一：
  - 一段助手正文（`assistant_message`）；
  - 一段思考（`thought`）；
  - 一个工具步骤，即调用事件与它的结果事件（可能没有结果）。
- **工具步骤的位置**：放在调用事件的位置，按 `action_id` 取结果。多个调用先后发出、结果随后到达时（Claude Code 的并行工具调用），各步骤仍按调用顺序排列。
- **思考不归属动作**：只按先后排列，不再像现有归约器那样把思考吸附到最近的动作。
- **新增与删除的概念**：新增一个类型表示工具步骤（调用与结果）。删除 `ActionReducer`、`TraceReducer`、`AgentAction`、`TraceItem`，以及 `TurnRecord` 上的 `actions`、`semantic_traces`、`user_query`、`rewritten_query`、`assistant_final_text`。
- **位置**：解析与第 8.2 节的渲染放在感知层的同一个模块（`engines/perception`）。生成视图的组装（`prompts/transcript/generation.py`）调用它；现有的 `prompts/transcript/actions.py` 并入该模块，`engines/perception` 不再反向依赖 `prompts`。

### 8.2 唯一的渲染

**材料**：一轮在感知层的文本表示。生成视图、LLM 接力摘要与 token 统计都使用同一个渲染函数，输入只有事件集与截断配置（`patchouli.generation.transcript`），渲染规则只有这一处。

候选的渲染规则（沿用现有生成视图的规则，改为按步骤顺序输出）：

| 步骤 | 渲染 |
|:---|:---|
| 用户消息 | `[User]: …` |
| 助手正文 | `[Assistant]: …`，按出现位置穿插在动作之间 |
| 思考 | `- THOUGHT: …`，按 `max_thought_chars` 截断 |
| 能力步骤（有 `verb`） | 一行摘要，不渲染结果（取回的是已有记忆，渲染会被重复提取）：`SEARCH: "查询"`、`READ: a, b`、`RUN: 目标 (状态)`；WRITE、UPDATE、CALL 不渲染 |
| 外部工具步骤（有 `tool_name`） | `- TOOL: 工具名 参数摘要`，其后逐行 `RESULT: …` 或 `RESULT (error): …`，按 `max_args_chars`、`max_result_chars` 截断 |
| 无法解析的 MTP 指令（两者皆空） | 不渲染 |

- **单轮上限**：`max_turn_action_chars` 只约束思考、能力步骤与外部工具步骤，用户消息与助手正文不计入。超出时的降级与省略顺序沿用现有规则。
- **结局标注**：本轮结局不是 completed 时，在轮首标注 `[Outcome: …]`。

示例：

```text
[User]: Import Bus 遇到 4xx 时游标怎么处理？
- THOUGHT: 先查记忆里有没有当时的决定。
- SEARCH: "Import Bus 4xx 游标"
- READ: fact_import_cursor
[Assistant]: 找到一条相关记忆，再核对一下代码。
- TOOL: Bash {"command": "rg -n advance connector/core/sync.py"}
  RESULT: 238:            self._advance(path, turn, next_offset)
[Assistant]: 连接器遇到 4xx 会记录拒绝并推进游标……
```

### 8.3 记忆生成视图

候选设计：

- `GenerationTurn` 改为结局、用户消息、按顺序的材料条目（含助手正文）与来源身份。
- 转写文本按第 8.2 节输出，不再有单独的 `[Actions]` 段与末尾的 `[Assistant]` 行。
- 记忆创建 Artifact 中的 `generation_view` 随之换成新形态（以 dict 保存，只写不读）。

### 8.4 接力摘要

候选设计：

- **LLM 接力**：近期事件改用第 8.2 节的材料。被折叠的块在结算前因此仍保留思考与外部工具过程。代价是接力摘要的输入变大，单块最多约为单轮上限加上问答正文。接力提示词中“近期事件”的说明（目前写作“对话轮次和系统动作（MTP 语义轨迹）”）随之修改。
- **规则接力**（LLM 不可用或超时时的回退）：请求计数取用户消息，工具列表取工具步骤（`verb` 的小写或 `tool_name`），不再读取 `rewritten_query`。

### 8.5 token 统计

逻辑块的 `total_tokens` 改为对第 8.2 节渲染出的整轮材料估算，折叠阈值因此直接反映生成实际看到的材料，不再分别统计问答、轨迹与动作条目。

### 8.6 逻辑块、话题与交互 Artifact

候选设计：

- `TurnRecord` 只保留 `outcome`、`turn_id`、`identity`、`turn_events`；用户消息、回复正文与步骤都在读取时从事件得到。
- `LogicalBlock` 删除 `worth_saving`、`gateway_intent`，以及没有消费方的 `anchor_text`、`is_complete`。
- 话题快照的最后一轮、Gateway 的话题上文改读第 7.2 节的两段文字。
- 交互 Artifact 快照只保存事件，以及块标识、时间、来源身份与结局，不再保存问答副本、动作与轨迹。

### 8.7 渲染的可见性

候选设计：

- 渲染只有一个函数，输入是事件与截断配置，同样的输入总得到同样的材料。任何读取方看到的材料，都可以由交互 Artifact 中保存的事件按当时的配置复现。
- 生成视图仍随新建记忆保存，作为“模型当时看到了什么”的记录。
- 为排查提供的查看入口（例如按话题或交互 Artifact 输出材料）不在最小范围内，见第 11 节。

## 9. 第二部分：载荷变化波及的感知层以外读取方

| 读取方 | 调整 |
|:---|:---|
| 任务进程的 Gateway 会话上文（`workspace/process/runner.py`） | 用户消息与回复正文改读第 7.2 节的属性 |
| 会话标题（`workspace/sessions/service.py`） | 同上 |
| 会话 HTTP（`server/models/session.py`） | 响应中的用户消息与回复正文改由属性得到，响应字段不变 |
| Alice 会话历史重放 | 删除“没有事件时用用户消息与回复拼出历史”的分支：每份载荷都至少有序号 0 的用户消息 |
| Import Bus | 导入契约删除 `assistant_final_text`，由事件得到；是否有输出改为看用户消息之后有没有事件 |
| 共用封口 | 不再接收 `user_message` 与 Gateway 决定。序号 0 的用户消息事件由提交方写出：Alice 执行循环已经写出，Import Bus 由服务端写出。Alice 只在用户消息非空时才写这条事件，因此 chat 入口拒绝空消息（422），使载荷不变量在生产中始终成立。CPU 只返回正文、没有 `assistant_message` 事件时，仍由封口补一条事件 |

## 10. 迁移与兼容

- **会话**：只在进程内保存，重启即清空，没有旧数据需要迁移。
- **交互 Artifact 与记忆创建 Artifact**：快照与生成视图以 dict 保存且只写不读。旧记录保留旧字段；读取它们的需求出现时再处理。
- **连接器与服务端同时升级**：导入契约变更后，旧版连接器提交的轮次会被以 422 拒绝。连接器遇到 4xx 会记录拒绝并推进游标，所以升级窗口内的轮次会被跳过，但拒绝记录可查。候选：不保留兼容分支，仓库内的连接器与服务端在同一次改动中升级，并在交付说明中提示重新安装插件。
- **短期话题**：话题在进程内保存，升级后的进程从空话题开始，不存在旧形态的逻辑块。
- **测试**：源码中构造 `TurnEvent` 的有 6 个文件、13 处，测试中有 27 个文件、72 处，另有 31 个测试文件读取相关字段。契约有意变更，旧契约的测试在同一改动中改写或删除。

## 11. 暂不处理

- Alice 流式交付事件的字段与 `TurnEvent` 不同名（`target`、`args`），只影响前端实时视图。
- `MTPExecutionResult.command` 的类型为 `Any`。
- MTP READ 截断正式记忆正文的问题，见 [Todo](../../todo/mtp-read-content-truncation.md)。
- 读取部分未命中在两侧都是 success，未命中的 alias 只在正文中。
- 渲染材料的查看入口（第 8.7 节）。
- 生成材料每轮上限的校准，见[生成材料 Todo](../todo/generation-material-filtering-and-truncation.md)。
