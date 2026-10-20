---
title: Alice Multi-Agent Orchestration
status: current
owner: alice
scope: call-frame-scheduling-and-sub-agent-return
code_paths:
  - src/hivememory/alice/run/service.py
  - src/hivememory/alice/orchestration/executor.py
  - src/hivememory/alice/orchestration/session.py
  - src/hivememory/alice/orchestration/call/coordinator.py
  - src/hivememory/alice/orchestration/call/context.py
  - src/hivememory/alice/orchestration/call/record.py
  - src/hivememory/alice/orchestration/frames.py
  - src/hivememory/alice/context/view.py
  - src/hivememory/alice/orchestration/output.py
  - src/hivememory/alice/interface/output.py
  - src/hivememory/alice/interface/cpu.py
  - src/hivememory/alice/events.py
  - src/hivememory/prompts/compilation.py
  - src/hivememory/prompts/mtp.py
related_contracts:
  - docs/contracts/mtp.md
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/error-model.md
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-10-19
---

# 多 Agent 编排

Alice 的多 Agent 能力当前不是一个会自主拆解任务图的“超级大脑”，而是一套有限、可解释的 CALL 控制流。主 Agent 在自己的生成过程中决定是否委派；Alice 在协议 trap 上接管执行，建立一个隔离的子 frame，将受控上下文交给目标 Profile，并把结果压缩成结构化 CALL response 后恢复主 Agent。

这套设计追求的不是让 Agent 彼此无限对话，而是让专项能力像可调用进程一样被发现和使用。主 Agent 保留用户任务与最终回答责任，子 Agent 只处理被委派的局部任务；子帧的试错过程不进入主话题，真正需要跨帧传递的知识必须通过 task、context refs、自然语言结果或 PendingAtom alias 明确表达。

## 1. 编排层组件

```text
AliceSystem
  ├─ AliceCPU -> SinkRunOutput（推送到任务进程的事件 sink）
  │    └─ AgentRunService -> RootContextBuilder
  │         ├─ RunSession（frame registry / CallRecord）
  │         ├─ RunExecutor -> CallCoordinator -> CallContextProvider
  │         │                -> FrameFactory / CALL response
  │         └─ CPUExecutionResult + AgentRunEventEmitter
  └─ build_agent_runtime -> AgentRuntime -> loop / MTPGenerator / Koakuma
                                            -> frame submit_operation -> workspace
```

- AgentRunService 是内部运行用例：`run_agent` 面向运行层的 `RunOutput`，创建根 frame 与每次 run 的 Executor，返回 CPU 中立执行结果并发布终态观测；模型预算、视图与压缩交给独立的 `RootContextBuilder`，结果投影放在 `run/result.py`；它是 Alice 作为 CPU 的最外层，被取消时把已产生的输出与已完成检查点组装为 cancelled 结果返回，不再传播取消，没有可交回的内容时才原样传播；
- AliceCPU 实现 workspace 定义的 CPU 端口，由组合根注入任务进程。它把进程交来的主线程绑定为线程句柄、检查宿主启停，每次执行都以 `SinkRunOutput` 直接等待运行服务，交互事件推送给进程的 sink，类型化结果作为返回值交回，不区分交付形态；
- AliceSystem 是子系统装配根，通过内核工厂构造 AgentRuntime。Profile 读取与缓存归 workspace，内核不参与多帧调度；
- RunSession 只拥有一次 run 的 frame registry 与 CALL record，不保存取消信号、活动 frame、frame 调度状态或传输层 stream sequence；
- RunExecutor 是唯一调用 `AgentRuntime.run_frame()` 的 Alice 编排组件。它以协程递归执行 CALL 派生 frame，并在根帧终态只记录一次 run 结果；意图收尾属于任务进程；
- AgentRunOutput 是调度与当前请求交互输出之间的窄端口，只承载 frame 输出与 CALL 边界；上下文层的 `ContextProgressOutput` 只承载压缩进度。运行层的 `RunOutput` 合并两者并补充 run 绑定与取消观测信息，生产实现只有 `interface/output.py` 的 `SinkRunOutput`，不缓存事件；
- SinkRunOutput 只负责把输出投影为交互事件并推送，AgentRunEventEmitter 只负责全局观测投影，两者不参与 frame 求值；
- CallContextProvider 经 caller 句柄的 `derive_child` 取得子线程的句柄（被调用方）并读取 `context_refs`，返回由被调用方句柄与共享上下文组成、不含 frame 或 CALL ledger 状态的 `CallContext`；
- CallCoordinator 把 CALL 拆为 `begin_call()` 与 `complete_call()`：消费 `CallContext` 组装 callee、投影 outcome，并通过 `AgentRuntime.apply_call_response()` exactly-once 恢复 caller；它在 CALL 收尾时经子线程句柄结束子线程（第 8 节），不解析 Profile/记忆，不运行 frame，也不收尾整个 run；
- FrameFactory 无状态地创建普通 frame，不表达主/子拓扑；
- `ResolveReferencesRequest` 让 context refs 与 READ 使用 workspace 的同一套引用解析、授权和引用记录；
- AgentRuntime 只运行给定 frame，不接触多 Agent 拓扑。

`RunExecutor` 是每次 run 独立创建的递归解释器，不维护 active-frame 状态机，也不把 caller/callee 写成两套绝对角色。`_execute_frame(frame)` 运行任意已登记 frame；遇到 CALL 时，它 `await` 递归执行新 frame，返回后继续同一个 caller。协程调用栈自然表达挂起与重入，`RunSession` 因而不再充当程序计数器。如果 AgentRuntime 开始创建 callee，或 CallCoordinator 再次调用 `run_frame()`，说明这组责任重新混合。

## 2. 主帧与运行作用域

每次 `run_agent()` 创建一个新的主 frame：

- `run_id=agent_run_<uuid>`，也是 `RunSession.agent_run_id`；任务进程的必填 `process_id` 随输入清单传入，作为外层关联值；
- 唯一且无拓扑含义的 `frame_id`；
- `topic_id` 指向 Patchouli 已准备的话题；
- `submit_operation` 是主线程的句柄，也是资源请求的唯一交付入口，只交出主线程描述，不暴露访问 context 或目标选择；
- `ExecutionLabels` 取自主线程描述，即注册时绑定的 agent/workspace 观测标签，不参与资源授权；
- `agent_profile` 是主线程描述中的图纸（进程在 prepare 之前解析），模型解析也用它；MTP 动词由主线程描述的操作快照翻译。`agent.run.*` 与历史中的当前 Agent 取输入清单的注册标签，与主线程标签相同；
- `working_history` 来自 `SessionContextView` 组装的预算内视图，预算与生成参数来自同一个 `ResolvedFrameModel`。

根 frame 创建前只构建一次会话视图：先解析模型，再检查固定输入、复用已有检查点并按需减重/摘要。算法见 [Agent Runtime](./agent-runtime.md#31-会话上下文视图)。Alice 不拥有跨 run 的检查点存储；完整压缩结束才保存本次产物，由 CPU 结果交回会话服务。CALL 子 frame 不接收会话历史或检查点，只按自身模型执行轮内守卫；caller 重入复用原解析快照。

FrameFactory 会把当前 user message 插入 `TurnEvent` 序列首位，使最终事件流拥有完整的一轮事实。随后 `RunExecutor._execute_frame()` 调用 `AgentRuntime.run_frame(frame)`：`SUSPENDED` 进入 CALL 事务并递归求值派生 frame；`COMPLETED/CANCELLED/FAILED/BUDGET_EXHAUSTED` 返回上一层。最外层入口 frame 的终态由 Alice 映射为 CPU 结果，任务进程据此封口交互或跳过 finalize，已提交写入不受影响。

frame 是实际可恢复状态。恢复 caller 时必须继续使用原 frame，不能从消息重新构造一个“看似等价”的新 frame，否则迭代预算、事件序号、已经产生的正文和 ACK alias 清单都会分叉。父子、caller action 等调用关系只记录在 Alice 的 `CallRecord` 与事件元数据中，不进入 `RuntimeScope`。

RunSession 只登记 frame 与 CALL record，并校验 frame 属于当前 run、调用 action 唯一、callee 与 record 绑定一致。当前正在执行哪个 frame、caller 在哪里等待，都由 Python 协程调用栈表达；Session 不保存 `PENDING/RUNNABLE/RUNNING/WAITING/TERMINATED` 之类的调度状态。重复入口/callee、跨 run 绑定、重复 CALL action 和重复 apply 仍作为编排不变量抛出。

## 3. CALL trap 与重入

CALL 的稳定语法和参数见 [MTP 契约](../contracts/mtp.md)。在编排层，它是一种控制流陷入：

```text
current frame emits CALL
  -> Koakuma validates FrameExecutionPolicy and returns SUSPEND
  -> AgentLoopExecutor records tool_call and returns FrameExecutionResult
  -> RunExecutor calls CallCoordinator.begin_call()
  -> register CallRecord before the first await
  -> CallContextProvider derives the child thread via the caller handle's derive_child
     (agent.call, target profile, callee admission) and resolves context_refs with the caller's handle
  -> CallCoordinator creates a normal callee frame from CallContext; the frame holds the child handle
  -> return DispatchCallee(callee)
  -> RunExecutor awaits _execute_frame(callee)
  -> CallCoordinator.complete_call() ends the child thread and finalizes the callee frame
  -> call_response maps the finalized result to one MTPCallResponse
  -> AgentRuntime.apply_call_response() commits the response once
  -> emit sub_agent_end
  -> return ResumeCaller; task cancellation unwinds the recursive call stack
```

`suspend` 不能被当作一条正文为空的成功响应。如果执行循环直接继续，Agent 会在被调用任务尚未执行时向下生成，CALL 的任务身份、结果和取消边界都会丢失。RunExecutor 必须先等待被调用任务完成，再以同一 action_id 通过 `AgentRuntime.apply_call_response()` 回填 tool result。

`SUSPENDED` 结果必须同时携带 `MTPCallRequest` 与 action_id。RunExecutor 在进入 CallCoordinator 之前经 `FrameExecutionResult.require_call_suspension()` 校验，缺失任一项即显式失败、不进入 CALL 事务；CallCoordinator 的 begin/complete/cancel 与响应提交也只经该方法取得挂起信息，不各自假定字段存在。

### 3.1 Profile 解析

未提供 alias 时使用内置 `OMNI_DOLL_PROFILE`；`default` 与 `omni_doll` 是对同一内置 Profile 的显式选择，不是加载失败后的降级。Omni-Doll 对当前 verb/tool 使用显式白名单，因此后续新增能力不会自动穿透 fallback 边界。

所有 alias，包括未指定、`default` 与 `omni_doll`，都由 CallContextProvider 经调用方句柄的 `derive_child` 提交 `DeriveCallThreadRequest`：workspace 的操作入口以调用方的线程授权 `agent.call` 并解析目标图纸，再由线程表以被调用方（Profile 的 `agent_id`，内置图纸为 `omni_doll`）做 Workspace 准入、派生并登记子线程；CPU 驱动把子线程绑定为它自己的句柄，CallContextProvider 只拿到该句柄，它交出的子线程描述包含目标 Profile、被调用方的操作快照与被调用方的观测标签。`context_refs` 另以调用方的 READ 请求逐项解析，需要调用方的 `memory.read`，仅缺失引用或明确的冷读依赖故障被跳过，授权、线程结束与程序错误传播到 CALL 出口，已派生的子线程随之结束。读取视图统一承担 Profile 解析、Workspace/alias 坐标定位、当前 actor 命中重验与 canonical 源变更失效。Patchouli 作为 Profile atom 所有者执行 Workspace ownership 和资源 policy 校验，再解析 persona、模型与语言。Alice 不保存另一份 Profile cache；管理入口修改源原子后，下一次 CALL 会通过失效视图读取新配置。子 frame 的 `FrameExecutionPolicy` 按每次 CALL 由被调用方的操作快照翻译后去掉 CALL 得到，不从 Profile 派生，也不进入 cache。

子帧消息由 `CallCoordinator` 直接组装：MTP 教学 → 被调用方人格 → 共享上下文，随后是原样任务用户消息；空片段省略，全部为空时不创建系统消息。语言取被调用方 Profile，缺省时回退进程默认值；人格复用 `prompts.compilation.render_persona`。教学读取 Alice 的 Koakuma 配置，直接使用子帧策略已排除 CALL 的动词集合，布局层不再过滤一次。子帧不继承根帧的固定输入或会话摘要；它是 Alice 的内部 harness 行为，根帧的进程侧编译不承担这项职责。

显式失败通过 `MTPCallResponse.error` 回填，不再启动子 frame：

| 场景 | 稳定 code | message key |
|:---|:---|:---|
| alias 不存在或源原子不可见 | `mtp.alias.not_found` | `mtp.call.profile_not_found` |
| 缺少 `agent.call` 操作授权 | `mtp.permission.denied` | `mtp.permission.verb_denied` |
| alias 类型不符 / Profile 配置无效 | `mtp.memory.type_mismatch` / `mtp.argument.invalid` | `mtp.call.profile_type_mismatch` / `mtp.call.profile_invalid` |
| Profile route 或读取失败 | 对应 `mtp.system.*` | `mtp.call.profile_load_failed` 或底层稳定 key |
| Profile 引用的模型不可用 | `mtp.system.service_unavailable` | `mtp.call_response.model_unavailable` |

Profile 生产者和读取视图只抛中立错误。`preparation_error_response()` 负责上下文/装配失败的唯一呈现，最终子帧结果通过具名函数构造 `MTPErrorInfo`，不再为已确定的预算耗尽或模型不可用创建只供序列化的异常对象。参数逐项筛选，分类与恢复动作见[错误模型](../contracts/error-model.md)。

CALL 的 `tool_call` 与 `tool_result` 使用同一个最终 success/error/cancelled 状态；内部 cause 只进入日志，不回填给 Agent。

Agent Profile 作为记忆存在，使服务发现可以复用预检索与 SEARCH：相关图纸可以在 memory context 中以 Agent Profile 菜单出现，主 Agent 随后用 alias 发起 CALL。Alice 当前不维护硬编码 team，也不会根据模糊需求动态创建 Profile。

### 3.2 共享上下文

`context_refs` 不是直接复制父 frame 的全部 history。CallContextProvider 逐个使用 caller frame 的 `submit_operation(ResolveReferencesRequest(...))` 解析：

- pending：共享同一 Workspace 中仍可回读的写入意图；
- redirect：共享已经结算后的 canonical atom；
- atom：共享正式记忆；
- discarded/failed/expired/not-found：记录 warning 并跳过。

有效 sources 交给 MemoryCompiler 的 `SHARED_CONTEXT_INJECTION` target，形成子 Agent system prompt 中的受控视图。这样父子进程共享的是明确引用，而不是一段无法追踪来源的任意拼接历史。

逐项 best effort 只覆盖超时、存储离线/读取失败和资源暂不可用；单个冷读失败不会丢弃其余已命中引用，全部不可用时可以只带 task 运行。READ 授权拒绝、线程结束、解析结果违约与程序错误结束 CALL 准备并回填错误，`CancelledError` 原样传播；两种情形下子帧都不会运行，已派生的子线程先同步结束。交付正式 atom 或可读 redirect 时，workspace 在返回前自动记录引用，READ 后再次作为 context ref 共享会形成另一次读取记录。当前 CALL response 不把这些跳过项作为结构化 warning 返回给主 Agent，只能从日志观察。

## 4. 瞬态子帧

子 frame 当前具有：

- 与 caller 相同的 `run_id`；
- 由 `FrameFactory` 生成的唯一 `frame_id`，不携带 `parent_frame_id` 或 `depth`；
- `topic_id=None`，不直接挂载 Patchouli 话题；
- 目标 Agent Profile（取自子线程描述）；
- 由 persona、裁剪后的 MTP 教学、shared context 和 task 组成的全新消息历史；
- 子线程描述中的 `ExecutionLabels`：被调用方与注册时的 workspace，不沿用调用方，由 workspace 的线程表生成；
- 子线程的句柄：以被调用方身份授权，写入意图记在被调用方名下并记录子线程标识。

子帧不读取主话题完整 history，也不会把内部 token、SEARCH/RUN 重试或工具结果写回主 frame 的 working history。只有子帧以 `COMPLETED` 自然结束后，CallCoordinator 才取 `text_segments` 形成 reply，并把运行期间收到 ACK 的 aliases 放入 CALL artifacts。取消、失败或预算耗尽的子帧不会收割 reply/artifact，已提交写入仍由 Patchouli 继续生成。`SUSPENDED` 是 RunExecutor 必须继续递归消费的非终态 trap；若它进入 `complete_call()`，属于编排不变量违约，不构造 CALL response。

这种黑盒隔离避免主 Agent 与 Perception 被子任务细节淹没，但它并不等于子任务没有证据。子帧流事件仍可被 UI 观察，CALL 在主 frame 中有结构化 tool_call/tool_result，PendingAtom 又保存写入意图；只是这些事实目前没有被组合成持久化子任务 artifact。

## 5. 当前单层 CALL 策略

当前产品策略允许入口 Agent 串行调用多个子 Agent，被调用 Agent 不能再次 CALL：

```text
root frame
  ├─ CALL -> callee A
  ├─ CALL -> callee B
  └─ continue root frame
```

软限制由 CallCoordinator 组装子帧提示词时实现：MTP 教学按被调用方的动词集合渲染，CALL 已在构造策略时移除，因此不会出现在教学中。硬限制由 `FrameExecutionPolicy` 实现：CallCoordinator 消费 CallContext 后，以被调用方的操作快照（`mtp_verbs_for_operations` 翻译，RUN 始终可用）构造子帧策略并显式移除 CALL；被调用方 Profile 只提供 persona、模型与语言。动词集合只是提示与执行前检查，子帧的资源操作仍由 actor API 按被调用方的白名单授权（[Workspace 架构](../architecture/workspace.md)第 5.4 节）。Koakuma 以 frame policy 校验。即使模型仍输出 CALL，也返回 `PermissionDeniedError`，执行循环把错误回填给该 Agent，让它改用自然语言或其他获准能力继续。

这是能力策略，不是 RunExecutor 的结构限制：执行器对所有 frame 使用同一个递归入口，不包含 `RootTurn/CalleeTurn` 分支。未来允许嵌套 CALL 时可以沿用这条递归路径；并行 sibling CALL 可在一次 suspension 能表达多个请求后，于当前递归层使用结构化并发。持久化 DAG、跨进程恢复和 review loop 仍属于后置方向。

## 6. 结果收割与回填

子帧成功结束后，CallCoordinator 通过 `AgentRuntime.finalize_frame()` 建立 artifact alias 列表：

`FrameProducts` 只投影本 frame 收到 ACK 后去重记录的 alias，不加入 UPDATE 的原基础 alias；caller 收到成功 CALL response 后把子帧产物加入自己的 alias 清单。

自然语言 reply 与 alias 列表组成 success `MTPCallResponse`。CallCoordinator 不直接操作 history 容器，而把 success/error/cancelled 终态响应交给 `AgentRuntime.apply_call_response()` 一次性加入 caller working history，并形成与原 CALL action_id 对应的 `tool_result`。caller 随后可以 READ pending alias、把它作为另一个 CALL 的 context ref，或直接根据子 Agent reply 继续任务。

CALL 故意没有配套的 MTP `RETURN` 动词。返回描述的是子 frame 生命周期的自然完成，不是一项新的记忆或工具动作；若再要求模型生成 `RETURN`，就会在已有执行终态之外增加一条语法、权限和 formatter 都可能失败的路径。当前由子帧自然结束触发返回，以自然语言 reply 表达结论，以 PendingAtom alias 收割表达可继续寻址的副作用，两者共同组成 CALL response。隐式返回只消除了重复协议动作，并不把任何退出都视作成功：`alice/orchestration/call/response.py` 仅将 `COMPLETED` 映射为 success，将 `CANCELLED` 映射为 cancelled，将 `FAILED`、`BUDGET_EXHAUSTED` 映射为带稳定 error code 的 error；`SUSPENDED` 不属于可映射终态。

caller 与 callee 各自使用自己线程的句柄：callee 登记的意图发起者是被调用方，进程关联与 caller 相同，操作内立即派发到相同的进程话题。子线程随 CALL 返回结束（第 8 节），之后以子线程句柄提交的请求以线程已结束拒绝；callee 未成功结束不回填 IPC alias，也不撤回已经提交的写入。IPC alias 服务于 caller 当前认知，正式生成由能力层提交后独立运行；CPU 结果不携带物化任务，Alice 不管理 run 级登记收尾。

## 7. 流式事件

流式 CALL 除普通 `token/mtp_start/mtp_result` 外增加：

- `sub_agent_start`：目标 alias、task、父迭代、depth 与 scope；
- 子帧自身的 token/MTP 事件：`scope=sub`，并带 depth/frame_id；
- `sub_agent_end`：最终 success/error/cancelled、子帧 `terminal_status`、目标 alias 与 frame id；success 携带 reply，error 携带稳定 `error_code`。

这些事件服务当前请求的实时 UI 与调试，不是业务结果来源。每次 run 独占的 `SinkRunOutput` 为事件补全 `agent_run_id/frame_id/action_id/stream_sequence`，逐项 `await` 任务进程的事件 sink；背压由进程的输出通道施加，通道写满时 sink 等待。`depth` 仅保留为兼容展示字段，不再是执行坐标。`sub_agent_start` 在 callee frame 创建后才发布，因此 `frame_id` 不为空。运行服务返回的 `CPUExecutionResult` 中的 `turn_events` 才是结构化的一轮事实，由任务进程封口后交给 Patchouli。

frame 的展示 `agent_id` 优先取 `AgentProfile.agent_id`（源 alias，内置图纸为 `omni_doll`），空值回退该 frame 所属线程的标签：主 frame 为注册标签，子 frame 为被调用方；`sub_agent_start`、`sub_agent_end` 的 `agent_id` 仍取 CALL 请求的目标 alias。`agent.run.*` 的 agent/workspace 字段始终取输入清单的注册标签，不从 Profile 或子帧重新推导身份；逐帧的预算守卫拦截 `agent.context.budget_exceeded` 报告拦截 frame 所属线程的标签，子帧被拦截时报告被调用方。

交互输出不会自动转发到 RuntimeEventBus。后者通过 `AgentRunEventEmitter` 记录主 `agent.run.*` 生命周期、压缩统计与逐帧预算守卫拦截，采用 best-effort、可回放且允许慢订阅者丢失的语义；前者具有背压，交付方离开时经停止请求取消本次 run。即使二者包含相同的 `agent_run_id/process_id` 关联字段，也不能把 RuntimeEvent 当作 token/CALL 流的备份或业务控制输入。

## 8. 失败、取消与降级

- 模型解析、generation/provider 等可归一化故障在最窄边界形成 `FrameExecutionResult.FAILED`；root 对外映射为失败 run，callee 对外映射为 CALL error；
- frame 注册、action/target、重复 apply、callee 关联与重复 finalize 等编排不变量继续抛出，不被 Executor 外层宽泛吞掉；
- CallContextProvider 的 Profile/共享上下文解析错误，以及后续模型调用或执行异常，会在 CALL 边界形成 error `MTPCallResponse`，主 Agent 得到错误后可以调整方案；
- 子帧预算耗尽映射为 `mtp.call_response.budget_exhausted`，子帧取消保持 cancelled 终态；生产 policy 会在 Agent loop 内拒绝被调用 frame 的 CALL，因此不会形成第二层 suspension；若未来开放该权限，RunExecutor 会直接递归执行，而不需要增加 root/callee 状态分支；
- 无法解析的单个 context ref 只跳过，不使 CALL 失败；
- 任务进程不向主/子 AgentRuntime 传递取消 token。用户 stop 与交付方离开都以原生取消到达等待 CPU 的子 task，可能落在模型调用上，也可能落在等待事件 sink 的推送上；RunExecutor 捕获 `CancelledError` 做本地 unwind，活跃 CALL 只清理 callee frame 和 record，不伪造 caller response，收尾整个 run 一次，并保留原生异常传播；`CancelledError` 沿递归协程栈展开，各层清理尚未 apply 的 CALL；
- RunExecutor、Agent loop 与 Worker 的 unwind/close 都是 best effort：清理异常记录日志，但不能替换正在传播的 `CancelledError`；
- 运行服务是取消唯一被转换的地方：RunExecutor 同步落定 CANCELLED 后，服务把根帧文字/事件及已完成的压缩检查点投影为 cancelled 的 `CPUExecutionResult` 并返回，内部各层仍原样传播取消；执行器尚未开始且没有完成的检查点时没有可交回的内容，服务原样传播取消；
- 压缩期间取消不交回本次局部检查点；压缩结束后即使根 frame 尚无输出，取消后返回的结果也可携带检查点。done 不携带检查点，检查点由任务进程从返回结果中接受；
- CALL 子帧取消时不伪造 caller response；根帧只保留未完成的 CALL，子帧半截正文不混入根帧结果或会话历史。
- 子 Agent 没有独立的公开取消句柄、重试策略或超时配置，生命周期依附于父 run；
- 子线程随 CALL 返回结束：`complete_call` 在子帧以任何终态（完成、失败、取消、预算耗尽）收尾时、`cancel_call` 在取消沿递归栈清理 CALL 时，以及派生成功之后、子帧开始运行之前的任何失败（共享上下文读取失败或被取消、子帧创建失败、会话绑定失败），都调用子线程句柄的 `end()`。`end()` 是同步调用，可以在取消收尾中执行，不引入新的挂起点；结束幂等，与 Actor 阶段的兜底结束不冲突。子帧的在途操作先随取消中断，已登记的写入在派发时不再复查线程；Alice 不结束主线程；
- 取消路径不提交 actor 操作，收尾只用已经产生的内容组装结果，满足 CPU 端口对取消后收尾的要求。

将子 Agent 失败包装成 CALL error 是局部容错，不代表子任务成功；主 Agent 是否还能完成用户请求由后续生成决定。相反，主 frame 基础设施失败没有可用的上层 Agent 继续纠正，因此必须结束本次 run。

## 9. 关键不变量与矛盾检查

- CALL trap 必须由 Alice 恢复，不能由 Koakuma formatter 或 Agent loop 吞成普通响应；
- 子帧只接收 task 与显式 shared context，不能默认复制主 frame 全部工作历史；
- 主 Agent 最终负责用户回复，子 Agent 不直接写入主话题或向任务进程交出第二个 CPU 终态；
- Profile 的 persona 不能提升结构化权限，调用方 task 也不能替被调用者改写白名单；
- Pending alias 的 IPC 收割与能力层的意图实时派发是两条不同用途的数据流；
- `context_refs` 必须经过操作请求的 workspace 引用解析与 MemoryCompiler，不能通过裸 UUID 或字符串拼接绕过 alias/状态语义；
- CALL 权限与预算必须随 frame policy 传播，不能只依赖 prompt 告诫模型；取消沿拥有 Alice run 的 task 递归展开，不通过 session 建立第二套控制面；
- Profile 与完整引用解析缓存只由 workspace 持有，Alice 不另建派生状态真相；观测标签不能成为资源授权依据。

## 10. 当前限制

- 能否 CALL 某个 Agent 只取决于调用方的 `agent.call`、目标 Profile 对调用方可见与被调用方的准入，没有“谁能调用谁”的细粒度策略；被调用方读到的 PRIVATE 记忆可能经 CALL 返回值交给调用方；
- frame registry 与 CallRecord 由每次 run 新建的 `RunSession` 持有；执行位置由 RunExecutor 的协程调用栈表达；stream sequence 由每次 run 独占的 `SinkRunOutput` 持有，当前没有共享 frame stack、活动 frame 状态机或共享输出队列；
- context ref 跳过只写日志，CALL response 没有 partial warning 列表；
- 子任务没有持久化 task id、独立 timeout/retry、并发额度、结果 artifact 或恢复机制；
- 当前产品 policy 只允许串行单层 CALL；RunExecutor 已能递归求值 frame，但尚未开放嵌套 CALL，也没有 parallel fan-out、动态 DAG、review loop 或 Alice 自主规划。

这些限制说明后续高级编排首先需要收紧身份、失败和并发状态，而不是先增加更多拓扑语法。只要同一个 frame 或 Profile 仍可能被错误归属，扩大 CALL 深度只会放大不可观测的矛盾。
