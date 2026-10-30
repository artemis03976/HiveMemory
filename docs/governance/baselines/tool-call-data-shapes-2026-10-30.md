---
title: 工具调用数据形态调查（2026-10-30）
status: baseline
owner: project
scope: alice-mtp-and-external-mcp-tool-call-data-shapes-and-reduction
snapshot_at: 2026-10-30
source_commit: 7c0ebbe8cb1c40c64cdedae3ff481383f431b923
code_paths:
  - src/hivememory/core/models/interaction.py
  - src/hivememory/core/protocol/sealing.py
  - src/hivememory/core/protocol/models.py
  - src/hivememory/core/mtp/models.py
  - src/hivememory/core/mtp/formatter.py
  - src/hivememory/agent_runtime/loop.py
  - src/hivememory/agent_runtime/runtime.py
  - src/hivememory/agent_runtime/model/generator.py
  - src/hivememory/agent_runtime/mtp/
  - src/hivememory/alice/interface/output.py
  - src/hivememory/alice/run/result.py
  - src/hivememory/alice/context/references.py
  - src/hivememory/mcp_server/server.py
  - src/hivememory/system/services/import_bus/models.py
  - src/hivememory/server/models/import_bus.py
  - src/hivememory/server/models/session.py
  - src/hivememory/engines/perception/
  - src/hivememory/engines/artifacts/
  - src/hivememory/prompts/transcript/
  - integrations/claude-code/connector/
  - frontend/src/stores/chat/sessionHistory.ts
related_docs:
  - docs/archive/plans/v0.7.1-tool-call-parsing-chain.md
  - docs/contracts/mtp.md
  - docs/contracts/mcp.md
  - docs/system/import-bus.md
  - docs/patchouli/perception.md
  - docs/patchouli/generation.md
last_reviewed: 2026-10-30
---

# 工具调用数据形态调查

本文冻结 `source_commit`（v0.7.1 开发期间的 `master`）上两条工具调用链路的数据形态：Alice 的 MTP 指令，与外部 harness（Claude Code）经 HiveMemory MCP 工具的调用。最新行为仍以代码、测试与当前设计为准。

**起因**：处理 Todo《导入的 MCP 读取在轨迹中显示 alias》（2026-10-30 并入[工具解析链路重写计划](../../archive/plans/v0.7.1-tool-call-parsing-chain.md)）时发现，问题不在渲染一处。owner 判断（2026-10-30）：`ActionReducer` 与 `TraceReducer` 的行为都不符合预期；`InteractionPayload` 同时保留 `mtp_traces` 与 `turn_events`，后面的解析来源因此不一致；后续各处对这些工具路径的渲染行为也不可见。整条工具解析链路需要重写。本文为重写提供现状证据，不提出设计。

## 1. 范围与方法

- **范围**：从工具调用产生（LLM 输出、MCP 工具返回、Claude Code 会话记录）到全部消费方（感知、记忆生成视图、接力摘要、交互 Artifact、Alice 会话历史、会话 HTTP 与前端会话历史）的每一层数据结构与字段。
- **方法**：只读调查，逐层核对代码。两处结论用同一套代码实际复现：封口与轨迹归约（第 6.1 节），Alice 会话历史重放（第 6.2 节）。
- **不在范围**：Gateway 的会话上文只使用用户消息与最终回复，不读取工具事件；MTP 的授权与执行细节见 [MTP 契约](../../contracts/mtp.md)。

## 2. 结论

- 同一轮的工具过程有三份表达：结构化事件 `turn_events`、由它归约的动作、由动作归约的轨迹。动作在封口与感知各归约一次，轨迹只在封口时归约一次、随载荷传递；没有机制保证三者一致，生成视图以 `action_id` 把它们重新拼合（第 5 节）。
- 两条链路写出的事件在 `role`、`content`、`tool_name`、`target`、`tool_args`、`status`、`render_as` 上都不同形（第 4 节）。其中 `tool_name` 在 MTP 一侧是指令目标，在外部一侧是工具名；归约器取 `tool_name` 优先，而 Alice 会话历史的读取方取 `target` 优先。
- 渲染结果基本不可见：渲染后的生成材料只在新建记忆时随创建 Artifact 保存，接力摘要的输入与 token 统计不留痕迹（第 6 节第 8 项）。

## 3. 两条链路的汇合关系

```text
Alice MTP                                     外部 harness（Claude Code + HiveMemory MCP）
LLM 文本 → GenerationResult                    MCP 工具返回 CallToolResult
  → MTPCommand → MTPResponse                   → Claude Code 会话记录 JSONL（tool_use / tool_result 块）
  → MTPExecutionResult                         → 连接器导入事件（dict）→ POST /api/v1/import/turns
  → TurnEvent（Alice 执行循环写出）             → ImportedTurnEvent → TurnEvent（to_turn_events）
  → CPUExecutionResult.turn_events
  （另有流式交付：mtp_start / mtp_result）
                  ╲                               ╱
                   seal_interaction（共用封口）
                   → InteractionPayload（turn_events 与 mtp_traces 并存）
                   ├→ 感知：LogicalBlock.turn（重新归约动作，轨迹沿用载荷）
                   │    → 记忆生成视图 / token 统计 / 接力摘要 / 交互 Artifact 快照
                   └→（只有 Alice）ConversationSession
                        → 历史重放、工具结果减重、引用线索、会话 HTTP、前端会话历史
```

## 4. Alice MTP 一侧

### 4.1 生成与解析

- **`GenerationResult`**（`agent_runtime/model/generator.py`）：`text`、`finish_reason`、`was_mtp_interrupted`、`prefix_text`、`mtp_fragment`、`model_used`。生成以 stop sequence `⟫` 停止且文本含 `⟪` 时视为 MTP 中断：补齐右定界符，以最后一个 `⟪` 切分，`prefix_text` 是指令之前的正文，`mtp_fragment` 是指令片段，`text` 是二者之和。
- **`MTPCommand`**（`core/mtp/models.py`）：`verb`（SEARCH、READ、RUN、WRITE、UPDATE、CALL）、`target: MTPTarget(is_wildcard, aliases: list[str])`、`args: dict[str, str]`（值均为字符串）、`raw_text`。

### 4.2 执行结果与回填

- **`MTPResponse`**（`core/mtp/models.py`）：`status`（success、error、ack、warning、suspend、cancelled）、`content`、`execution_time_ms`、`pending_alias`（不序列化）、`call_request`（不序列化）、`error: MTPErrorInfo(code, message_key, severity, params, cause)`、`warnings: list[MTPWarningInfo(message_key, params)]`。
- **`MTPExecutionResult`**（`agent_runtime/mtp/context.py`）：`command`（类型为 `Any`）、`response_status: str`、`response_content`、`formatted_response`、`success`、`execution_time_ms`、`pending_alias`、`call_request`。
- **回填文本 `formatted_response`**（`core/mtp/formatter.py`）：本地化标题一行，其后为 `<mtp_response status=".." time="..">` 容器，依次放正文、错误与警告。READ 的正文按 `MTP_READ` 编译，正式记忆超过 500 字截断；SEARCH 的正文按 `RETRIEVAL_CONTEXT` 信封编译。
- **CALL**：执行返回 `suspend` 与 `MTPCallRequest(target_alias, task, context_refs)`；子帧结束后组装 `MTPCallResponse(status, agent_alias, reply, artifact_aliases, error)`，以单独的标题格式化回填。

### 4.3 执行循环写出的结构化事件

来源：`agent_runtime/loop.py` 的帧循环、`agent_runtime/runtime.py` 的 CALL 回填、`alice/orchestration/frames.py` 的首条用户消息。

| 事件 | role | content | action_id | tool_kind | tool_name | target | tool_args | status | render_as |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 首条用户消息 | user | 用户消息 | — | — | — | — | — | — | plain |
| 指令前正文（非空时） | assistant | `prefix_text` | — | — | — | — | — | — | plain |
| tool_call | assistant | `text` 全文（指令前正文加指令片段） | `action_{iteration}_{sequence}` | 动词；解析失败为 `UNKNOWN` | 目标提示 | 与 tool_name 相同 | `args`（值为字符串） | 执行后改为 `response_status`；执行器没有结果时为 failed；CALL 挂起时不设状态，回填时改为子帧结局 | plain |
| tool_result | user | `formatted_response` | 同调用 | 动词 | 目标提示 | 目标提示 | — | `response_status` | `system_tool_result` |
| tool_result（执行器没有结果） | user | `mtp_fragment` | 同调用 | 动词 | 目标提示 | 无 | — | failed | plain |
| CALL 的 tool_result | user | 格式化的 CALL 响应 | 同调用 | CALL | `target_alias` | 无 | — | 子帧结局 | `system_call_response` |
| 取消或失败时的半截正文 | assistant | 已流出的文字 | — | — | — | — | — | cancelled 或 failed | plain |
| 自然收敛的回复 | assistant | 正文 | — | — | — | — | — | — | plain |

- **目标提示的编码**（`AgentLoopExecutor._extract_command_info`）：通配为 `*`，别名列表以逗号拼接为一个字符串（如 `fact_a,fact_b`），没有目标时为空串。
- **CALL 子帧**：子帧自己的事件不进入根帧结果；调用方只保留 CALL 的调用事件与回填结果，子帧的回复在回填文本中。
- **最终回复**：`CPUExecutionResult.final_text` 拼接帧进度中的全部 `text_segments`（各次迭代的指令前正文与最后的回复）。

### 4.4 流式交付的工具事件

来源：`alice/interface/output.py`。这些事件只交给前端，不进入封口或 Patchouli。

- `mtp_start`：`verb, target, args, raw_text, iteration, action_id`；
- `mtp_result`：在 `mtp_start` 的字段之外再加 `status`；
- 二者都带 `agent_run_id`、`stream_sequence`，以及 `scope, depth, agent_id, frame_id` 元数据；
- CALL 另发 `sub_agent_start`（`agent_id, task, iteration, action_id, scope, depth, frame_id`）与 `sub_agent_end`（另带 `status, final_text`，以及可选的 `terminal_status`、`error_code`）。

### 4.5 交给任务进程的结果

`CPUExecutionResult`（`workspace/contracts/cpu.py`）：`status, failure_reason, compaction_checkpoint, final_text, turn_events, model_used`。任务进程以 `turn_events` 封口；流式 `done` 不下发 `turn_events` 与检查点。

## 5. 外部 MCP 一侧（Claude Code）

### 5.1 HiveMemory MCP 工具的返回

来源：`mcp_server/server.py`。

- **`memory_search(query: str)`**：返回 `CallToolResult(content=[TextContent(text)])`，正文是 `RETRIEVAL_CONTEXT` 编译文本（不含面向 Alice 的头尾引导）；没有结果时为“没有检索到相关记忆。”。
- **`memory_read(aliases: list[str])`**：按请求顺序逐条以 `MTP_READ` 编译并交付全文；部分 alias 找不到时末尾追加“未找到以下 alias：…”；全部找不到时返回错误结果。
- **错误结果**：`is_error=True`，正文是错误投影的 JSON（错误码、安全说明与白名单内的 reason）。
- 返回没有 XML 容器，也没有类似 MTP 的状态字段。

### 5.2 Claude Code 会话记录

连接器（`integrations/claude-code/connector/claude_code/reader.py`）读取的部分：

- **条目**：`type`（user 或 assistant）、`uuid`、`sessionId`、`turnOrigin`、`timestamp`、`apiBlockIndex`；`isSidechain`、`isMeta`、`isCompactSummary` 为真的条目整条跳过，子代理的内部过程因此不导入。
- **`message`**：`id`、`model`、`stop_reason`、`content`（字符串或块列表）。
- **块**：`text{text}`、`thinking{thinking}`、`image`、`tool_use{id, name, input}`、`tool_result{tool_use_id, content: str | [{type: text, text} | {type: image}], is_error}`。

### 5.3 连接器写出的导入事件

每个事件是 dict，`sequence` 从 1 递增，值为 `None` 的字段省略。

| 来源块 | kind | content | action_id | tool_name | tool_kind | tool_args | status |
|:---|:---|:---|:---|:---|:---|:---|:---|
| text | assistant_message | 正文 | — | — | — | — | — |
| thinking | thought | 思考正文；空白时丢弃（较新的模型多只记录签名） | — | — | — | — | — |
| tool_use | tool_call | 空串 | `tool_use.id` | 完整调用名，如 `mcp__plugin_hivememory-connector_hivememory__memory_read` | 只有本插件两个工具标为 `SEARCH`、`READ` | `input`；不是 dict 时为 `{"input": 原值}` | — |
| tool_result | tool_result | 文本，图片记为 `[image]` | `tool_use_id` | 无 | 无 | 无 | `is_error` 时为 `error`，否则省略 |

- `assistant_final_text` 以空行拼接该轮全部助手正文。
- 调用名按 Claude Code 的规则由插件名与 server 名拼出；规则变化时 `tool_kind` 标记静默失效。
- **请求体**：`source, external_conversation_id, external_turn_id, agent_id, outcome, user_message, events, assistant_final_text, model_used, recent_turns[{user_message, assistant_text, external_turn_id}]`。

### 5.4 服务端接收与转换

来源：`server/models/import_bus.py`、`system/services/import_bus/models.py`。

- `ImportTurnRequest` 是 `ImportedTurn` 加 `agent_id`（只作认证声明）。
- `ImportedTurnEvent` 为 `extra=forbid`：不接受 `role`、`target`、`render_as`；`tool_kind` 只能是 `SEARCH` 或 `READ`；`tool_args: dict[str, Any]`；序号严格递增。
- `to_turn_events()`：用户消息写在序号 0；role 按种类填写，`tool_result` 为 `system`；`target` 一律为空，`render_as` 一律为 `plain`；`tool_args` 递归冻结。

## 6. 两条链路的工具事件对照

| 字段 | MTP tool_call | MCP tool_call | MTP tool_result | MCP tool_result |
|:---|:---|:---|:---|:---|
| role | assistant | assistant | user | system |
| content | 生成全文（含指令前正文） | 空串 | 带标题的 XML 回填 | 工具返回文本或错误 JSON |
| tool_name | 目标（记忆 alias、工具 alias 或 Agent alias） | 工具调用名 | 目标；CALL 为被调用 Agent | 无 |
| target | 与 tool_name 相同 | 无 | 同 tool_name；CALL 与执行失败时没有 | 无 |
| tool_kind | 六个动词或 `UNKNOWN` | `SEARCH`、`READ` 或空 | 动词 | 无（归约时从调用补齐） |
| tool_args | `dict[str, str]`；READ 一般为空 | 任意 JSON；READ 为 `{"aliases": [...]}`，SEARCH 为 `{"query": ...}` | 无 | 无 |
| status | success、error、ack、warning、failed；CALL 为子帧结局 | 无 | 同左 | `error` 或无 |
| render_as | plain | plain | `system_tool_result` 或 `system_call_response` | plain |

## 7. 汇合之后的共同下游

### 7.1 封口与归约

- **封口**（`core/protocol/sealing.py` 的 `seal_interaction`）：输出 `InteractionPayload`（`core/protocol/models.py`），字段为 `outcome, user_message, rewritten_query, assistant_final_text, turn_events, mtp_traces, worth_saving, model_used, used_attachments`。事件中没有助手正文时，在末尾补一条 `assistant_message`。
- **`ActionReducer`**（`core/models/interaction.py`）：按 `action_id` 把 thought、tool_call、tool_result 归成 `AgentAction(action_id, thought, tool_name, tool_kind, tool_args, results: tuple[TurnEvent, ...], status)`。
  - 调用事件的 `tool_name` 取 `event.tool_name or event.target`；结果事件只在动作缺少名字或种类时补齐；
  - 没有 `action_id` 的调用以 `tool_call_{sequence}` 为标识，思考与结果吸附到最近的动作；
  - `status` 取最后一个带状态的事件。
- **`TraceReducer`**：按 `tool_kind` 产生 `TraceItem(action, action_id, target, query, tool, status)`。
  - READ、CALL、WRITE、UPDATE 的 `target` 取动作的 `tool_name`，后三者另带 `status`（缺省为 unknown）；
  - RUN 的 `tool` 取 `tool_name`，`status` 缺省为 unknown；
  - SEARCH 的 `query` 取 `tool_args["query"]`；
  - READ 与 SEARCH 不带状态；没有 `tool_kind` 的动作不产生轨迹。

### 7.2 感知

`MemoryPerceptionEngine.build_block` 产生 `LogicalBlock(block_id, turn, created_at, total_tokens, gateway_intent, worth_saving)`，其中 `TurnRecord` 有 `outcome, turn_id, identity, user_query, rewritten_query, assistant_final_text, turn_events, actions, semantic_traces`。**动作由事件重新归约，`semantic_traces` 直接取载荷中的 `mtp_traces`。**

### 7.3 各消费方读取的表达

| 消费方 | turn_events | 动作 | 轨迹 |
|:---|:---|:---|:---|
| 记忆生成视图（`prompts/transcript/actions.py`） | 思考与动作的先后 | 外部工具的名字、参数与结果 | 按 `action_id` 对上动作后渲染 MTP 一行摘要；对不上的补在末尾 |
| 感知 token 统计 | 经渲染间接计入 | 经渲染间接计入 | 计入 `query` 与 `target` |
| LLM 接力摘要的近期事件（`relay_controller.py`） | — | — | 只读轨迹 |
| 规则接力摘要的“使用了工具” | — | 读 `tool_name`（MTP READ 的 alias 也被列为工具） | 读 `tool` 与 `action` |
| 交互 Artifact 快照（`engines/artifacts/interaction.py`） | 以 dict 持久化 | 以 dict 持久化 | 以 dict 持久化；三份都只写不读 |
| 记忆创建 Artifact | — | — | 只保存渲染后的生成视图，且只在新建记忆时保存 |

**只有 Alice 的支线**：`ConversationSession` 的每一轮 `ConversationTurn(interaction_id, agent_id, payload, topic_id, created_at)` 保存同一份 `InteractionPayload`。

- 历史重放（`prompts/transcript/history.py`）：每个事件按 `role` 渲染为一条消息，按 `render_as` 补系统前缀；
- 工具结果减重（`prompts/transcript/reduction.py`）与引用线索（`alice/context/references.py`）：目标取 `target` 优先；
- 会话 HTTP（`server/models/session.py`）：只返回 `kind, sequence, content, action_id, tool_kind, target, tool_args, status`，不含 `tool_name`、`role`、`render_as`；
- 前端会话历史（`frontend/src/stores/chat/sessionHistory.ts`）：以 `tool_kind`、`target`、`tool_args`、`content`、`status` 渲染，按 `action_id` 配对调用与结果。

## 8. 发现的问题

1. **三份表达、来源不一致**：`turn_events`、两次归约的动作、只归约一次的轨迹并存于载荷与逻辑块，各消费方读取不同的表达。
2. **字段含义重叠**：`tool_name` 在 MTP 一侧是目标、在外部一侧是工具名；MTP 的 `target` 与 `tool_name` 重复；`ActionReducer` 取 `tool_name` 优先，Alice 会话历史的读取方取 `target` 优先。
3. **结果的 role 不同**：MTP 工具结果为 user，导入的工具结果为 system。
4. **状态词汇不同**：MCP 调用成功时状态为空，MTP 为 success；READ 与 SEARCH 的轨迹不带状态。
5. **`tool_kind` 承担两种语义**：MTP 一侧是动词，导入一侧是“记忆操作”标记；回声去重依赖这个标记，而标记取决于连接器按 Claude Code 命名规则拼出的调用名。
6. **多个 alias 的表示不同**：MTP 为逗号拼接的字符串，MCP 为参数中的列表；Alice 的引用线索把 `a,b` 当作一个引用 `memory:a,b`。
7. **指令前正文重复**：MTP 调用事件的 `content` 含指令前正文，这段正文已另存为一条 `assistant_message`。运行时的 working history 中它只出现一次（助手消息为 `text` 全文），重放会话历史时出现两次；前端会话历史中 MTP 卡片的命令文本同样带着它。
8. **渲染结果不可见**：渲染后的生成材料只在新建记忆时随记忆创建 Artifact 保存；查重判为更新、触碰或丢弃时不保存；接力摘要的输入与 token 统计都不留痕迹。

### 8.1 复现：封口与轨迹归约

以第 5.3 节的形态构造一轮导入（一次检索、一次读取两条记忆、一次 Bash），另以第 4.3 节的形态构造一次 MTP READ，经 `seal_interaction` 与生成视图得到：

| 动作 | 动作的 `tool_name` | 轨迹 | 生成视图 |
|:---|:---|:---|:---|
| MTP READ（两条） | `fact_a,fact_b` | `target=fact_a,fact_b` | `READ: fact_a,fact_b` |
| 导入的 MCP 读取 | `mcp__plugin_hivememory-connector_hivememory__memory_read` | `target` 同左 | `READ: mcp__plugin_hivememory-connector_hivememory__memory_read` |
| 导入的 MCP 检索 | `mcp__plugin_hivememory-connector_hivememory__memory_search` | `query=游标 4xx` | `SEARCH: "游标 4xx"` |
| 导入的 Bash | `Bash` | 无 | `TOOL: Bash {"command": "ls"}` 与结果 |

### 8.2 复现：Alice 会话历史重放

以第 4.3 节的形态构造一轮“指令前正文、READ 调用、READ 结果、回复”，经 `HistoryTranscriptBuilder.build_messages` 重放：

```text
user      | 查一下
assistant | 我先读一下记忆。
assistant | 我先读一下记忆。⟪ READ | fact_a ⟫
user      | [System MTP Execution Result] / <mtp_response status="success">正文</mtp_response>
assistant | 结论是……
```

## 9. 后续处置

| 项 | 处置 |
|:---|:---|
| 工具解析链路重写 | owner 决定重写（2026-10-30）；设计见[工具解析链路重写 Idea](../../archive/ideas/tool-call-parsing-chain.md)，实施见[计划](../../archive/plans/v0.7.1-tool-call-parsing-chain.md) |
| Todo《导入的 MCP 读取在轨迹中显示 alias》 | 是第 8 节第 2 项的一个表现，2026-10-30 并入[工具解析链路重写计划](../../archive/plans/v0.7.1-tool-call-parsing-chain.md) |
| 第 8 节其余各项 | 只记录，供重写设计参考 |
