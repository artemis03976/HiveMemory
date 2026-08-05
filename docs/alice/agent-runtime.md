---
title: Alice Agent Runtime
status: current
owner: alice
scope: single-agent-frame-execution
code_paths:
  - src/hivememory/agent_runtime/
  - src/hivememory/alice/runtime/streaming.py
  - src/hivememory/prompts/
  - src/hivememory/core/protocol/models.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/mtp.md
  - docs/contracts/error-model.md
last_reviewed: 2026-08-04
---

# Agent Runtime

Agent Runtime 负责把一个 Agent 的一帧运行到自然收敛、取消或控制流陷入。它是 Alice 所使用的执行层，却刻意不知道“主 Agent”“子 Agent 团队”或“下一步应该调度谁”：给它一个包含身份、图纸、消息和执行坐标的 `ExecutionFrame`，它负责生成、解释本帧 MTP、回填结果并持续推进；一旦遇到 CALL，就把控制权交还 Alice。

这个边界来自一个很朴素的判断：执行一条指令与决定派生哪个进程不是同一种责任。若单 Agent loop 自己递归创建子 Agent，它就同时成为 CPU 和调度器，Alice 无法再拥有真实的编排语义；若每次工具调用都让编排层介入，执行循环又会被切碎。当前设计以 frame 作为两层之间的稳定货币，让执行层可以独立重入，让编排层只在 CALL trap 上接管。

## 1. 层级与依赖方向

```text
AgentRunService
  -> RunExecutor
       -> AgentRuntime facade
       -> AgentLoopExecutor
       -> WorkerAgentService
       -> MTPExecutor port -> KoakumaRuntime
       -> PendingAtomRuntime management facade
```

`src/hivememory/agent_runtime/` 是共享执行层，不是子系统：

- 不实现 `SubsystemProtocol`，不注册 `GlobalSystemBus` route；
- 不拥有 start/stop/health 生命周期；
- 不直接 import Alice 的 RunExecutor、CallCoordinator、CallContextProvider 或 ProfileResolver；
- 只消费注入的 MTP port、配置、模型注册表和运行时状态；
- 持久化记忆、Profile 读取与 citation 均通过 Alice 装配的 local bus 间接访问 Patchouli。

`AgentRuntime` 门面与 frame 级稳定契约保留在 `agent_runtime/` 根部；`execution/` 收拢 loop 与 WorkerAgent，`aliases/` 收拢热缓存和三级解析，`mtp/`、`pending_atom/` 分别保存协议执行与写缓冲能力。AliceRuntime 在进程启动时构造这组资源，AgentRunService 把同一个门面交给每次 run 的 RunExecutor；执行层不再位于 Alice 编排目录中。

当前对外只有一个 frame 执行入口：`AgentRuntime.run_frame(frame, *, generation_options, output_sink)`。非流式与流式调用分别注入 `NullFrameOutputSink` 和支持 token 的 frame output sink，但共享同一条 loop 与 `FrameExecutionResult` 语义；旧的 `run_frame_stream()`、`run_frame_emitting()` 与 callback adapter 已删除。Agent Runtime 不接收 Chat Run 取消句柄，也不轮询取消状态；外层 task cancellation 直接沿 await 传播。Agent Runtime 不接收额外的 generation mode，而是只读取 `output_sink.streams_tokens`：为 `false` 时调用完整生成，为 `true` 时调用 token stream。

## 2. ExecutionFrame：可恢复的运行 PCB

`ExecutionFrame` 是一次单 Agent 执行的进程控制块：

```text
ExecutionFrame
  ├─ RuntimeScope(run_id, frame_id, action_id)
  ├─ AgentProfile
  ├─ Identity
  ├─ working_history[]
  ├─ topic_id | None
  ├─ harvested_aliases[]
  └─ ExecutionProgress
       ├─ text_segments[]
       ├─ turn_events[]
       ├─ iteration
       ├─ sequence
       └─ model_used
```

把执行进度放在 frame 而不是 `execute_frame()` 的局部变量中，是 CALL 能安全重入的前提。frame 运行到 CALL 时会返回 `SUSPENDED`；Alice 处理完被调用 frame 后把同一个 caller frame 再交给执行层。此前产生的正文、事件序号和迭代预算都留在 `ExecutionProgress` 中，因此不会因为 Python 函数返回而丢失，也不会在恢复时从第 0 次迭代重新开始。

`RuntimeScope` 不再表达主/子拓扑、`parent_frame_id` 或 `depth`。Alice 通过无状态 `FrameFactory` 创建普通 frame，并在 `RunSession` 的 frame registry 与 `CallRecord` 中保存调用关系；Agent Runtime 只根据传入 scope 执行该 frame。

## 3. 输入上下文与 Prompt 组装

Agent Runtime 不接收原始 Gateway 输入，而消费 `AgentRunContext` 已准备好的事实：Identity、话题、当前用户消息、最近话题 blocks、检索 atoms、已编译 memory context、Agent Profile 和 storage availability。

Alice 在创建主帧前按“三明治”顺序组装消息：

```text
Top     MTP protocol + permitted verbs/tools + storage-offline notice
Middle  Agent Profile persona
Bottom  compiled memory context + topic state summary
History latest five LogicalBlocks rendered as messages
User    current user message
```

这种层次不是为了依赖某种 prompt 技巧，而是让机器约束、角色偏好和工作状态拥有不同来源。MTP 教学从结构化权限生成，persona 不得暗中扩张权限，memory context 也只是本轮工作视图；三者不能拼成一段后再由 Runtime 反向解析。

历史消息由结构化 `TurnEvent` 重放。若历史 block 来自另一个非默认 Agent，assistant 消息会增加 `[From: agent_id]` 前缀，避免当前 Agent 把同事的旧输出误认成自己曾经作出的承诺。工具结果则按 `render_as` 添加本地化系统前缀。旧 block 只有在缺少事件时才回退到 `assistant_final_text` 等兼容字段。

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
- 未注入 ModelRegistry 时，调用方必须在 generation options 中直接提供可执行 model，否则 WorkerAgent 抛出 `ValueError`。

模型注册表启用时，frame 记录的是展示名，供 `AgentRunResult.model_used` 与话题 UI 使用。未启用注册表的兼容路径可以正常生成，但当前不会把 WorkerAgent 返回的底层 model 名重新写入 frame，因此 `model_used` 可能为空。

## 5. 单帧执行循环

默认最大循环次数为 10。每轮执行：

```text
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

自然语言正文和工具事实同时投影为 `TurnEvent`。每个 MTP action 使用 `action_{iteration}_{sequence}` 关联 `tool_call` 与 `tool_result`，Perception 后续可以从事件归约 action/trace，而不需要解析已经格式化的响应字符串。

CALL 是唯一会返回 `SUSPENDED` 的 MTP 路径。执行层不创建子 frame、不解析 Profile、不组装 IPC；这些动作属于 Alice 编排层。自然停止返回 `COMPLETED`，取消返回 `CANCELLED`，无法形成有效结果返回 `FAILED`，达到 `max_loop_iterations` 且尚未自然收敛返回 `BUDGET_EXHAUSTED`。只有 `COMPLETED` 表示本帧成功完成。

## 6. 交互输出与流式执行

Agent Runtime 不再直接构造 SSE dict，也不依赖名为 EventBus/Sink 的全局观测设施。`agent_runtime/output.py` 定义窄 `FrameOutputSink`，单帧循环只会发送 `TokenDelta`、`MTPStarted` 和 `MTPFinished` 三类强类型输出。Alice 的 `AgentRunOutput` 再把 frame 输出与 CALL 边界合并，`alice/runtime/streaming.py` 才负责投影为兼容的交互事件：

- `token`：尚未检测到 MTP 左定界符的自然语言增量；
- `mtp_start`：Runtime 已识别并准备执行一条指令；
- `mtp_result`：指令的 success/error/ack/suspend 等状态；
- 被调用 frame 事件仍使用相同类型，通过 `scope/frame_id/action_id/agent_id` 命名空间区分；为兼容既有 SSE 客户端，Alice 仍可提供 `depth` 展示字段，但它不再参与 Runtime 控制流；
- `done` 由 AgentRunService 在 RunExecutor 完成主 run 收尾后组装，而不是由 WorkerAgent 直接发出。

每次流式 run 由 `AgentRunStreamAdapter` 创建容量为 256 的有界 FIFO queue、runner task 与 run-local `stream_sequence`。`QueueAgentRunOutput` 使用 `await put()` 保留背压，不丢弃 token 或控制事件；消费者提前关闭时，适配器取消并 join 自己创建的 runner，不影响其他 run。RunExecutor 本身没有 `run_stream()`、queue 或 sequence，它只调用统一的 `run(..., run_output=...)`。

检测到 MTP 后，本轮剩余协议文本不再作为普通 token 推给用户，而是等待 Runtime 执行并发出结构化输出。CALL 时，`RunExecutor` 取得 `SUSPENDED` 结果，调用 `CallCoordinator.begin_call()`；Coordinator 先通过 `CallContextProvider` 获得目标 Profile 和共享上下文，再准备 callee。随后 Executor 递归等待同一个 `_execute_frame(callee)`，返回后调用 `complete_call()`，最后通过 `AgentRuntime.apply_call_response()` 重入同一个 caller frame。Runtime 不感知入口/被调用角色，挂起关系由 Executor 的协程调用栈表达。

交互输出流与 RuntimeEvent 是两条严格独立的数据面。前者面向当前调用方，参与背压和断流取消，不能丢弃，并保留 `token/mtp_start/mtp_result/sub_agent_start/sub_agent_end/done` 兼容事件名；后者是全局 best-effort 观测旁路，可以缓冲、回放和丢弃慢订阅者数据，只记录 `agent.run.*` 等生命周期摘要。FrameOutput 不会自动桥接到 RuntimeEventBus，RuntimeEvent 也不驱动 frame、CALL 或 run 状态。

非流式 LiteLLM await、流式 async iterator 的每次 pull，以及 MTP await 都直接响应外层 task cancellation。Worker 在流式 iterator 的 `finally` 中关闭底层 iterator；同步 syscall 一旦开始执行，事件循环仍必须等待函数返回，这属于 Python task cancellation 无法解决的同步边界。

## 7. AgentRunResult 的组装边界

Agent Runtime 返回的是 frame 级 `FrameExecutionResult`；面向跨子系统的 `AgentRunResult` 必须由 AgentRunService 组装：

- `final_text` 来自主 frame 累积正文；
- `turn_events` 是当前用户消息、assistant 输出和工具事件的有序事实；
- `mtp_iterations/total_iterations` 来自主 frame PCB；
- `materialize_tasks` 由 PendingAtomRuntime 按共享 run_id 认领，包含父子帧写意图；
- `status` 由取消状态与运行终态确定；
- `model_used` 来自主 frame 模型解析。

执行层不应为了组装最终响应重新维护 write focus、pending alias 或子 Agent 结果副本。PendingAtomRuntime 已拥有写缓冲真相，AgentRunService 只在 run 边界投影稳定公共结果。

当前收尾由两个显式产品模型分开：`finalize_frame()` 产生只供当前 CALL 使用的 `FrameProducts.artifact_aliases`，`finalize_run()` 产生交给 Patchouli 的 `RuntimeProducts.materialize_tasks`。前者可由 `CallCoordinator` 调用，后者只由 `RunExecutor` 在入口 frame 终态调用一次；编排层不再遍历 PendingAtom store 的内部集合。

## 8. 关键不变量与矛盾检查

- Agent Runtime 必须与 Agent 数量无关；若 loop 开始解析子 Agent Profile、决定拓扑或组 IPC，编排责任已经回流；
- `ExecutionFrame` 是重入状态的唯一载体，不能在另一个 service 中并行维护 iteration、sequence 或 text accumulator；
- WorkerAgent 只负责模型生成与 MTP 定界符检测，不执行权限或记忆语义；
- AgentRunContext 是本轮只读快照，Alice 不在执行中取得话题或长期记忆的可变所有权；
- 模型解析和 generation/provider 故障必须稳定形成 `FAILED`，不能为了可用性静默运行错误模型；
- 取消不能被包装成普通 success，流式路径也必须以 `done` 或异常明确结束；
- 引擎事件必须保持 sequence 单调、tool_call/tool_result action_id 对齐，不能只保留用户可见正文。

## 9. 配置、观测与测试

当前执行层配置位于 `AliceConfig.runtime.max_loop_iterations`；模型、密钥与采样默认值由 ModelRegistry 和 shared config 管理，单次请求可以覆盖。Koakuma 与 prompt 配置见 [MTP Runtime](./mtp-runtime.md)。

`AgentRunEventEmitter` 为主 run 产生 `agent.run.started/completed/cancelled/failed` 观测事件，包含 generation、agent run、topic、agent、status、迭代与 materialize task 数量；`RuntimeEventPublisher` 统一补充 scope/context、payload 安全转换和 best-effort 异常隔离。AgentRunService 只在明确的业务分支调用这些语义方法。frame 内部过程则通过交互输出事件和结构化 TurnEvent 暴露，不进入 RuntimeEventBus。

主要验证入口：

- `tests/unit/agent_runtime/execution/test_loop_turn_events.py`；
- `tests/unit/agent_runtime/execution/test_loop_stream.py`；
- `tests/unit/agent_runtime/execution/test_worker.py`；
- `tests/unit/agent_runtime/test_runtime.py`；
- `tests/unit/alice/application/test_agent_run_service.py`；
- `tests/unit/alice/orchestration/test_run_executor.py`；
- `tests/unit/alice/runtime/test_streaming.py`、`test_runtime_events.py`；
- `tests/e2e/pipeline/test_kernel_loop_e2e.py`、`test_active_mode_e2e.py`。

## 10. 当前限制

- frame、执行进度和消息历史只在内存中，进程重启、worker 崩溃或请求迁移后不能恢复；
- `BUDGET_EXHAUSTED` 能区分循环预算耗尽，但当前没有动态扩容、自动任务分解或 checkpoint 恢复策略；
- 主 run 的失败和预算耗尽都由 Alice 组装为 `AgentRunStatus.FAILED`，它是可观察且稳定的常规终态，不应被改写为 `cancelled`；
- `AgentRunResult.turn_events` 在公共模型中仍声明为 `list[Any]`，类型边界没有完全收紧到 `TurnEvent`；
- 未使用 ModelRegistry 时 `model_used` 可能为空，即使 WorkerAgent 实际已经使用了调用方提供的模型；
- AliceRuntime 仍直接持有 PendingAtom settlement/cache 刷新逻辑；这部分属于进程级运行时投影，但尚未进一步提取为窄事件处理器；
- 流式取消只能在 LiteLLM chunk 或 MTP checkpoint 处生效，不能保证立即中断同步 syscall；
- 执行层没有每 run 的资源配额、token budget、并发限流、持久化 checkpoint 或回放能力；
- `health()` 只返回 loop/worker 固定 `ok`，不验证模型端点、正在运行的 frame 或迭代耗尽率。

这些限制不改变当前层级判断：扩展可靠性应围绕 frame、取消和可恢复执行状态建设，不能通过把编排重新塞回 loop 来绕过。
