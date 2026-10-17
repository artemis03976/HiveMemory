---
title: Alice Agent Runtime
status: current
owner: alice
scope: single-agent-frame-execution
code_paths:
  - src/hivememory/agent_runtime/
  - src/hivememory/alice/runtime/streaming.py
  - src/hivememory/alice/orchestration/context_view.py
  - src/hivememory/alice/orchestration/relay_summary.py
  - src/hivememory/prompts/
  - src/hivememory/core/protocol/models.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/mtp.md
  - docs/contracts/error-model.md
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-10-17
---

# Agent Runtime

Agent Runtime 负责把一个 Agent 的一帧运行到自然收敛、取消或控制流陷入。它是 Alice 所使用的执行层，却刻意不知道“主 Agent”“子 Agent 团队”或“下一步应该调度谁”：给它一个包含观测标签、图纸、消息、操作提交函数和执行坐标的 `ExecutionFrame`，它负责生成、解释本帧 MTP、回填结果并持续推进；一旦遇到 CALL，就把控制权交还 Alice。

这个边界来自一个很朴素的判断：执行一条指令与决定派生哪个进程不是同一种责任。若单 Agent loop 自己递归创建子 Agent，它就同时成为 CPU 和调度器，Alice 无法再拥有真实的编排语义；若每次工具调用都让编排层介入，执行循环又会被切碎。当前设计以 frame 作为两层之间的稳定货币，让执行层可以独立重入，让编排层只在 CALL trap 上接管。

## 1. 层级与依赖方向

```text
AgentRunService
  -> RunExecutor
       -> AgentRuntime facade
       -> AgentLoopExecutor
       -> WorkerAgentService
       -> MTPExecutor port -> KoakumaRuntime
       -> frame submit_operation -> WorkspaceOperationEntry -> workspace actor API
```

`src/hivememory/agent_runtime/` 是共享执行层，不是子系统：

- 不实现 `SubsystemProtocol`，不注册 `GlobalSystemBus` route；
- 不拥有 start/stop/health 生命周期；
- 不直接 import Alice 的 RunExecutor、CallCoordinator、CallContextProvider；
- 只消费注入的 MTP port、配置、模型注册表和运行时状态；
- WRITE/UPDATE、引用读取与 SEARCH 经 frame 的线程 context 绑定提交函数进入 workspace；CALL Profile 读取由 Alice 编排层提交请求，引用记录由 workspace 读取能力自动完成。

`AgentRuntime` 门面与 frame 级契约保留在 `agent_runtime/` 根部，`execution/` 收拢 loop 与 WorkerAgent，`mtp/` 保存协议执行。AliceRuntime 只装配执行机制；写入意图登记、完整引用解析、原子与 Profile 缓存由 workspace 持有，执行层只消费 CPU 驱动交付的提交函数。

当前对外只有一个 frame 执行入口：`AgentRuntime.run_frame(frame, *, generation_options, output_sink)`。非流式与流式调用分别注入 `NullFrameOutputSink` 和支持 token 的 frame output sink，但共享同一条 loop 与 `FrameExecutionResult` 语义；旧的 `run_frame_stream()`、`run_frame_emitting()` 与 callback adapter 已删除。Agent Runtime 不接收 Chat Run 取消句柄，也不轮询取消状态；外层 task cancellation 直接沿 await 传播。Agent Runtime 不接收额外的 generation mode，而是只读取 `output_sink.streams_tokens`：为 `false` 时调用完整生成，为 `true` 时调用 token stream。

## 2. ExecutionFrame：可恢复的运行 PCB

`ExecutionFrame` 是一次单 Agent 执行的进程控制块：

```text
ExecutionFrame
  ├─ RuntimeScope(labels, run_id, frame_id, action_id)
  ├─ AgentProfile
  ├─ working_history[]
  ├─ resolved_model（生成参数、展示名、input_token_limit 的解析快照）
  ├─ topic_id | None
  ├─ submit_operation（本帧所属线程的提交函数）
  ├─ harvested_aliases[]（本帧收到的 ACK alias）
  └─ ExecutionProgress
       ├─ text_segments[]
       ├─ turn_events[]
       ├─ iteration
       ├─ sequence
       └─ model_used
```

把执行进度放在 frame 而不是 `execute_frame()` 的局部变量中，是 CALL 能安全重入的前提。frame 运行到 CALL 时会返回 `SUSPENDED`；Alice 处理完被调用 frame 后把同一个 caller frame 再交给执行层。此前产生的正文、事件序号和迭代预算都留在 `ExecutionProgress` 中，因此不会因为 Python 函数返回而丢失，也不会在恢复时从第 0 次迭代重新开始。

`RuntimeScope` 的冻结 `ExecutionLabels(agent_id, workspace_id)` 仅用于展示和观测，不携带访问 context、操作身份或资源归属。frame 与 `MTPExecutionContext` 也不暴露 `identity` / `identity_scope` 派生属性。`RuntimeScope` 不表达主/子拓扑、`parent_frame_id` 或 `depth`。Alice 通过无状态 `FrameFactory` 创建普通 frame，并在 `RunSession` 的 frame registry 与 `CallRecord` 中保存调用关系；Agent Runtime 只根据传入 scope 执行该 frame。

## 3. 输入上下文与 Prompt 组装

Agent Runtime 不接收原始 Gateway 输入，而消费 `AgentRunContext` 已准备好的事实：`ExecutionLabels`、话题、当前用户消息、全量会话 `session_history`、检索 atoms、已编译 memory context、已编译附件文本、Agent Profile 和 storage availability。`AgentRunContext` 由 `AgentRunService` 从任务进程交来的 `CPUInputManifest` 转换而来：Profile 由进程解析，记忆与附件由进程编译，Alice 只决定这些文本在提示词中的位置。

Alice 在创建主帧前按“三明治”顺序组装消息：

```text
Top     MTP protocol + permitted verbs/tools + storage-offline notice
Middle  Agent Profile persona
        earlier conversation summary + reference hints (when available)
Bottom  compiled memory context + compiled attachment context
History complete turns after summary coverage, reduced or raw by checkpoint
User    current user message
```

这种层次不是为了依赖某种 prompt 技巧，而是让机器约束、角色偏好和工作状态拥有不同来源。MTP 教学从结构化权限生成，persona 不得暗中扩张权限，memory context 也只是本轮工作视图；三者不能拼成一段后再由 Runtime 反向解析。

历史消息由结构化 `TurnEvent` 重放，`current_agent_id` 使用输入清单的注册标签。若历史轮次来自另一个 Agent，assistant 消息会增加 `[From: agent_id]` 前缀，避免当前 Agent 把同事的旧输出误认成自己曾经作出的承诺。工具结果则按 `render_as` 添加本地化系统前缀。会话 payload 缺少事件时回退到用户/助手文本；cancelled/failed 轮次的本地化未完成说明附在该轮最后一条消息末尾，不在历史中间插入 system 消息（部分模型适配会把 system 消息合并到开头，或要求角色严格交替）。Topic 状态摘要与 block 不进入 Alice 的上下文。Alice 接收全量会话原文，再按以下规则构建预算内的派生视图，不改写会话原文。

### 3.1 会话上下文视图

`SessionContextView` 属于 Alice 编排层，每次 run 只在创建根 frame 前构建一次。会话服务持有检查点、Alice 解释其格式：当前接受 `producer=alice`、`format_version=1`，未知格式、无法匹配原文的游标或摘要与覆盖不一致时回退到全量重放。关闭压缩时忽略检查点。检查点结构见[子系统契约](../contracts/subsystem-contracts.md#421-会话压缩检查点)，接受与生命周期见 [Workspace](../architecture/workspace.md#55-会话资源与历史所有权)。

先估算协议、人设、检索记忆、附件与本轮用户消息；固定输入已超过输入上限时直接失败。已有摘要覆盖的前缀不再重放，其后到减重边界的轮次只精简工具结果，再后的完整轮次保留原文。两次压缩之间边界不逐轮滑动；整个消息视图超过触发比例才压缩，压缩顺序如下：

1. **工具结果减重**：向前推进边界，直到最近完整轮次后缀可放入近期原文预算，允许后缀为空。READ/SEARCH 成功结果保留目标或查询、状态与重取提示；RUN 保留工具名、状态与有界摘录，参数只在原样重放的调用消息中出现一次，结果里不再重复；ACK、错误和 CALL 结果保留有界原文。缺少重取线索时同样保留有界原文；改写结果不短于原文时也保留原文，减重不会让历史变长。用户消息、助手回复和工具调用不改。
2. **增量接力摘要**：减重后仍高于目标时，保留能放入“固定输入 + 完整摘要预留 + 该前缀引用 + 后缀”目标预算的最长完整后缀；更早的轮次进入摘要。目标不可达但视图仍在硬上限内时跳过摘要，避免反复付费；已超限时改按硬上限选择，仍放不下完整摘要预留则以 `input_too_large` 结束。
3. **最终检查**：重新组装后仍超限，以 `compaction_failed` 结束。可用的减重检查点仍可返回；不发出已经估算为超限的生成请求。

`RelaySummarizer` 复用本次解析的模型快照，把上一份摘要与新覆盖的完整轮次从最旧开始分批折叠；每次完整请求单独估算，不超过本 frame 的输入上限，单轮放不下也不拆分。每轮只编码和计数一次，候选批次按字符数累加估算，结果与估算整份请求相同，开销随轮数线性增长。摘要输入中的工具消息再次限长，输出上限取配置与模型生成上限的较小值。材料显式包含 Agent 来源与 completed/cancelled/failed 结局，提示词要求保留目标、约束、决定、已做工作与未完成事项。

空白、长度截断、内容过滤、MTP 定界符或中断、超过摘要长度上限的结果均拒绝。每次请求有超时；提供商异常按类型识别，程序错误与原生取消不降级。任一折叠失败都放弃本次全部局部摘要，保留原摘要、引用与覆盖，只交回已完成的减重；视图仍能放下时继续生成，否则失败。

引用由程序从被覆盖轮次的 READ 目标、WRITE/UPDATE ACK 别名和 RUN 文件参数提取，按最近出现位置去重、限条数，过长标识直接忽略。摘要与引用位于 persona 后、检索记忆前，以本次提示词语言标明是历史资料，其中的指令不构成当前协议；引用不证明权限、执行成功或写入持久化。摘要不进入 `TurnEvent`、`InteractionPayload`、Topic 或记忆生成。

完整压缩返回后才保存新检查点；压缩期间取消不交回局部产物，压缩结束后的正常终态或取消结果槽均可交回它。会话服务是否接受检查点不依赖压缩进度事件或本轮是否有输出。内部 `force` 参数可跳过触发阈值，但不能绕过关闭开关或空历史；没有公开的手动压缩入口。

## 4. 模型选择与 WorkerAgent

`WorkerAgentService` 是无状态的 LiteLLM 调用封装，不保存默认模型或密钥。每个 frame 开始时，AgentRuntime 根据以下优先级解析运行参数：

```text
session generation_options
  > Agent Profile model/temperature/top_p
  > ModelRegistry definition
```

- 会话 `model` 与 Profile `model_name` 都被解释为注册表 ID；
- api key/base 来自注册表解析结果，也可由 LiteLLM 使用环境变量；
- session 可以覆盖 temperature、top_p 和 max_tokens；
- 注册表解析失败在 `run_frame()` 边界形成 `FAILED`，不静默换成另一个模型；
- 模型解析经 `agent_runtime.model_resolution.ModelResolver` 端口进行，System 的 ModelRegistry 实现该端口并由组合根注入；未注入时，调用方必须在 generation options 中直接提供可执行 model，否则 WorkerAgent 抛出 `ValueError`。

模型注册表启用时，frame 记录的是展示名，供执行结果的 `model_used` 与话题 UI 使用。未启用注册表的兼容路径可以正常生成，但当前不会把 WorkerAgent 返回的底层 model 名重新写入 frame，因此 `model_used` 可能为空。

`AgentRuntime.resolve_model()` 返回 `ResolvedFrameModel`：固定实际生成参数、展示名与输入上限。根 frame 在上下文组装前解析，CALL 子 frame 按自身模型独立解析；同一 frame 从 CALL 恢复时复用快照，避免注册表变动导致预算与请求漂移。输入上限为总 `context_window` 减去解析后的 `max_tokens` 和安全余量，缺少窗口登记时采用保守默认；非正上限不会自动扩大。预算字段不透传给 LiteLLM。估算沿用 `TokenEstimator`，另计每条消息 4 token 开销；配置与默认值见 [System 配置](../system/configuration.md#12-alice-会话预算与压缩)。

## 5. 单帧执行循环

默认最大循环次数为 10。每轮执行：

```text
0. estimate complete working_history; exceed input_token_limit -> FAILED, no generation
1. WorkerAgent generate with stop=["⟫"]
2. natural output?
     -> append assistant text/event -> complete
3. MTP fragment?
     -> build action-scoped MTPExecutionContext
     -> Koakuma execute
     -> append tool_call event
4. response == suspend?
     -> return FrameExecutionResult(SUSPENDED)
5. ordinary result
     -> append assistant command + formatted tool result to history
     -> append tool_result event
     -> continue
```

WorkerAgent 只识别 MTP 左定界符，并在 stop sequence 截断了右定界符时补齐文本；它不知道 SEARCH、WRITE 或 CALL 的业务含义。Koakuma 经窄 `MTPExecutor` port 负责真正解析与执行。这个分离使模型适配、协议解释和多 Agent 编排可以独立变化。

自然语言正文和工具事实同时投影为 `TurnEvent`。每个 MTP action 使用 `action_{iteration}_{sequence}` 关联 `tool_call` 与 `tool_result`，下游用 core 的归约器从事件得到 action/trace（任务进程封口时归约 MTP trace，Perception 归约 action），而不需要解析已经格式化的响应字符串。

CALL 是唯一会返回 `SUSPENDED` 的 MTP 路径。执行层不创建子 frame、不解析 Profile、不组装 IPC；这些动作属于 Alice 编排层。自然停止返回 `COMPLETED`，取消返回 `CANCELLED`，无法形成有效结果返回 `FAILED`，达到 `max_loop_iterations` 且尚未自然收敛返回 `BUDGET_EXHAUSTED`。只有 `COMPLETED` 表示本帧成功完成。

轮内守卫检查包含所有新工具结果的完整 `working_history`，根帧与 CALL 子帧分别按自身预算执行。子帧被拦截沿现有 CALL error 回到父帧，父帧可以继续；根帧被拦截后已有的输出仍按 failed 封口。守卫不压缩未结束轮次。提供商仍报告超窗时，Worker 按 LiteLLM 异常类型映射，不按错误文本猜测，也不重试；稳定失败码见[错误模型](../contracts/error-model.md#22-agent-run)。

## 6. 交互输出与流式执行

Agent Runtime 不再直接构造 SSE dict，也不依赖名为 EventBus/Sink 的全局观测设施。`agent_runtime/output.py` 定义窄 `FrameOutputSink`，单帧循环只会发送 `TokenDelta`、`MTPStarted` 和 `MTPFinished` 三类强类型输出。Alice 的 `AgentRunOutput` 再把 frame 输出与 CALL 边界合并，`alice/runtime/streaming.py` 才负责投影为兼容的交互事件：

- `token`：尚未检测到 MTP 左定界符的自然语言增量；
- `mtp_start`：Runtime 已识别并准备执行一条指令；
- `mtp_result`：指令的 success/error/ack/suspend 等状态；
- 被调用 frame 事件仍使用相同类型，通过 `scope/frame_id/action_id/agent_id` 命名空间区分；为兼容既有 SSE 客户端，Alice 仍可提供 `depth` 展示字段，但它不再参与 Runtime 控制流；
- frame 展示的 `agent_id` 优先取 Profile 的来源 alias（`AgentProfile.agent_id`），空值回退 `RuntimeScope.labels.agent_id`；
- `context_compaction`：根 frame 创建前的压缩开始与结束统计，不属于帧输出或轮次事实；
- `done` 由 AgentRunService 在 RunExecutor 完成主 run 收尾后组装，而不是由 WorkerAgent 直接发出。

每次流式 run 由 `AgentRunStreamAdapter` 创建容量为 256 的有界 FIFO queue、runner task 与 run-local `stream_sequence`。`QueueAgentRunOutput` 使用 `await put()` 保留背压，不丢弃 token 或控制事件；消费者提前关闭时，适配器取消并 join 自己创建的 runner，不影响其他 run。RunExecutor 本身没有 `run_stream()`、queue 或 sequence，它只调用统一的 `run(..., run_output=...)`。

检测到 MTP 后，本轮剩余协议文本不再作为普通 token 推给用户，而是等待 Runtime 执行并发出结构化输出。CALL 时，`RunExecutor` 取得 `SUSPENDED` 结果，调用 `CallCoordinator.begin_call()`；Coordinator 先通过 `CallContextProvider` 获得目标 Profile 和共享上下文，再准备 callee。随后 Executor 递归等待同一个 `_execute_frame(callee)`，返回后调用 `complete_call()`，最后通过 `AgentRuntime.apply_call_response()` 重入同一个 caller frame。Runtime 不感知入口/被调用角色，挂起关系由 Executor 的协程调用栈表达。

交互输出流与 RuntimeEvent 是两条严格独立的数据面。前者面向当前调用方，参与背压和断流取消，不能丢弃，并保留 `token/mtp_start/mtp_result/sub_agent_start/sub_agent_end/context_compaction/done` 兼容事件名；后者是全局 best-effort 观测旁路，可以缓冲、回放和丢弃慢订阅者数据，记录 `agent.run.*` 生命周期与 `agent.context.*` 预算统计。FrameOutput 不会自动桥接到 RuntimeEventBus，RuntimeEvent 也不驱动 frame、CALL 或 run 状态。

非流式 LiteLLM await、流式 async iterator 的每次 pull，以及 MTP await 都直接响应外层 task cancellation。Agent loop 在 `finally` 中关闭 Worker generator，Worker 再关闭 LiteLLM/provider response；各层 close 失败只记录日志，不能替换正在传播的 `CancelledError`。RunExecutor 的 CALL/run 收尾遵循同一优先级：先做 best-effort 本地清理，再原样重抛取消。同步 syscall 一旦开始执行，事件循环仍必须等待函数返回，这属于 Python task cancellation 无法解决的同步边界。

LLM 流已交付的文字同步积累到 frame 的 `streaming_text`；正常结果到达后清空以避免重复，原生取消或模型失败时封为带状态的未完成 `assistant_message` 并加入正文。非流式没有 token 缓冲，只保留已完成迭代。

## 7. 执行结果的组装边界

Agent Runtime 返回的是 frame 级 `FrameExecutionResult`；面向任务进程的 CPU 中立执行结果（`CPUExecutionResult`，`workspace.contracts`）必须由 AgentRunService 组装：

- `final_text` 来自主 frame 累积正文；
- `turn_events` 是当前用户消息、assistant 输出和工具事件的有序事实（`TurnEvent`）；
- `status` 由取消状态与运行终态确定；
- `model_used` 来自主 frame 模型解析；
- `failure_reason` 仅 failed 时投影可识别的预算/压缩失败码；
- `compaction_checkpoint` 是本次完成的历史视图检查点，不属于本轮输出，也不进入 done。

迭代统计（`mtp_iterations/total_iterations`）来自主 frame PCB，但属于 Alice 的观测：它们只进入 `agent.run.*` 终态事件的载荷，不进入执行结果。

执行层不维护写入意图或物化任务副本。Agent loop 只在 ACK 时记录 alias；成功子帧的 `finalize_frame()` 返回 `FrameProducts.artifact_aliases`，服务于当前 CALL 的回填。写入由 workspace 在每次操作内派发，正式物化由 Patchouli 独立完成，AgentRuntime 不再提供 `finalize_run()`，RunExecutor 只记录根帧终态。

## 8. 关键不变量与矛盾检查

- Agent Runtime 必须与 Agent 数量无关；若 loop 开始解析子 Agent Profile、决定拓扑或组 IPC，编排责任已经回流；
- `ExecutionFrame` 是重入状态的唯一载体，不能在另一个 service 中并行维护 iteration、sequence 或 text accumulator；
- WorkerAgent 只负责模型生成与 MTP 定界符检测，不执行权限或记忆语义；
- 输入清单与由它转换的 AgentRunContext 是本轮只读快照，Alice 不在执行中取得话题或长期记忆的可变所有权；
- 模型解析和 generation/provider 故障必须稳定形成 `FAILED`，不能为了可用性静默运行错误模型；
- 取消不能被包装成普通 success，流式路径也必须以 `done` 或异常明确结束；
- 引擎事件必须保持 sequence 单调、tool_call/tool_result action_id 对齐，不能只保留用户可见正文。

## 9. 配置、观测与测试

执行循环配置位于 `AliceConfig.runtime.max_loop_iterations`，历史压缩与输入预算使用 `AliceConfig.compaction`；模型、密钥与采样默认值由 ModelRegistry 和 shared config 管理，单次请求可以覆盖。Koakuma 与 prompt 配置见 [MTP Runtime](./mtp-runtime.md)。

`AgentRunEventEmitter` 为主 run 产生 `agent.run.started/completed/cancelled/failed` 观测事件，包含 process、agent run、topic、agent/workspace 注册标签、status、迭代统计；`RuntimeEventPublisher` 统一补充 scope/context、payload 安全转换和 best-effort 异常隔离。AgentRunService 只在明确的业务分支调用这些语义方法。frame 内部过程主要通过交互输出事件和结构化 TurnEvent 暴露；Alice 另投影根/子帧守卫拦截的数值与 frame 坐标，不带消息正文。压缩和守卫事件字段见[事件契约](../contracts/routes-and-events.md#43-上下文预算观测)。

主要验证入口：

- `tests/unit/agent_runtime/execution/test_loop_turn_events.py`；
- `tests/unit/agent_runtime/execution/test_loop_stream.py`；
- `tests/unit/agent_runtime/execution/test_worker.py`；
- `tests/unit/agent_runtime/test_runtime.py`；
- `tests/unit/alice/orchestration/test_context_view.py`、`test_relay_summary.py`、`test_summary_references.py`；
- `tests/e2e/system/test_chat_session.py`（HTTP 长会话、检查点复用与守卫封口）；
- `tests/unit/alice/application/test_agent_run_service.py`；
- `tests/unit/alice/orchestration/test_run_executor.py`；
- `tests/unit/alice/runtime/test_streaming.py`、`test_runtime_events.py`；
- `tests/e2e/pipeline/test_agent_loop_e2e.py`、`test_chat_run_e2e.py`。

## 10. 当前限制

- frame、执行进度和消息历史只在内存中，进程重启、worker 崩溃或请求迁移后不能恢复；
- `BUDGET_EXHAUSTED` 能区分循环预算耗尽，但当前没有动态扩容、自动任务分解或执行进度恢复策略；
- 主 run 的失败和预算耗尽都由 Alice 组装为执行结果的 `failed`，它是可观察且稳定的常规终态，不应被改写为 `cancelled`；
- 未使用 ModelRegistry 时 `model_used` 可能为空，即使 WorkerAgent 实际已经使用了调用方提供的模型；
- 资源操作的提交函数不提供访问 context、Workspace 选择或执行者重新绑定能力；CALL 子帧使用以被调用方身份派生的子线程提交函数；
- 流式取消只能在 LiteLLM chunk 或 MTP checkpoint 处生效，不能保证立即中断同步 syscall；
- 输入预算使用字符估算，不能保证与提供商精确计数一致；没有每 run 的累计费用配额、并发限流或持久化执行检查点；
- 摘要是有损历史视图，原文仍在会话内；固定样本与 HTTP 验证使用提供商替身，真实 LLM 的事实保留、延迟与成本尚未评估；
- `health()` 只返回 loop/worker 固定 `ok`，不验证模型端点、正在运行的 frame 或迭代耗尽率。

这些限制不改变当前层级判断：扩展可靠性应围绕 frame、取消和可恢复执行状态建设，不能通过把编排重新塞回 loop 来绕过。
