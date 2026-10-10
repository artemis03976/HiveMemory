---
title: Alice Multi-Agent Orchestration
status: current
owner: alice
scope: call-frame-scheduling-and-sub-agent-return
code_paths:
  - src/hivememory/alice/application/agent_run_service.py
  - src/hivememory/alice/orchestration/run_executor.py
  - src/hivememory/alice/orchestration/run_session.py
  - src/hivememory/alice/orchestration/sub_agent/call_coordinator.py
  - src/hivememory/alice/orchestration/sub_agent/call_context_provider.py
  - src/hivememory/alice/orchestration/sub_agent/call_record.py
  - src/hivememory/alice/orchestration/frame_factory.py
  - src/hivememory/alice/orchestration/run_output.py
  - src/hivememory/alice/runtime/streaming.py
  - src/hivememory/alice/runtime/runtime_events.py
  - src/hivememory/prompts/assembler.py
related_contracts:
  - docs/contracts/mtp.md
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/error-model.md
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-10-09
---

# 多 Agent 编排

Alice 的多 Agent 能力当前不是一个会自主拆解任务图的“超级大脑”，而是一套有限、可解释的 CALL 控制流。主 Agent 在自己的生成过程中决定是否委派；Alice 在协议 trap 上接管执行，建立一个隔离的子 frame，将受控上下文交给目标 Profile，并把结果压缩成结构化 CALL response 后恢复主 Agent。

这套设计追求的不是让 Agent 彼此无限对话，而是让专项能力像可调用进程一样被发现和使用。主 Agent 保留用户任务与最终回答责任，子 Agent 只处理被委派的局部任务；子帧的试错过程不进入主话题，真正需要跨帧传递的知识必须通过 task、context refs、自然语言结果或 PendingAtom alias 明确表达。

## 1. 编排层组件

```text
AliceSystem
  ├─ AliceCPU
  │    └─ workspace CPUPort 的实现：经 alice.public.run_agent 调用，done -> CPUExecutionResult
  ├─ AgentRunService
  │    ├─ unified run API (stream 参数) / root frame bootstrap / CPUExecutionResult / stream done
  │    ├─ RunSession
  │    │    └─ frame registry / CallRecord ledger
  │    ├─ RunExecutor
  │    │    └─ recursive frame evaluation + root outcome recording
  │    ├─ CallCoordinator
  │    │    ├─ begin/complete CALL, frame preparation, response apply
  │    │    └─ CallContextProvider -> target profile + context_refs -> CallContext
  │    ├─ FrameFactory
  │    │    └─ create ordinary ExecutionFrame from FrameSpec
  │    ├─ AgentRunOutput / AgentRunStreamAdapter
  │    │    └─ frame output binding + bounded streaming queue
  │    └─ AgentRunEventEmitter
  │         └─ global best-effort agent.run.* observability
  └─ AliceRuntime
       ├─ KoakumaRuntime -> frame submit_operation -> workspace 操作入口
       └─ AgentRuntime facade -> execute one frame to a terminal/trap outcome
```

- AgentRunService 是 Alice 的公开 run 用例入口：`run_agent` 以 `stream` 参数控制是否流式，内部只有一套执行骨架，负责创建入口 frame、为每次 run 构造 Executor、组装 CPU 中立的执行结果，并在流式终态后发出唯一 `done`。两种模式的 `agent.run.*` 终态事件语义一致：完成、失败与终态前被关闭或取消分别只发布一次；queue、runner task、stream sequence 与 RuntimeEvent envelope 实现均不放在 application 层；
- AliceCPU 实现 workspace 定义的 CPU 端口，由组合根注入任务进程；它经全局总线调用 Alice 的统一执行路由，绑定任务进程的执行凭据形成提交函数，原样转交交互事件并把 `done` 转换为执行结果，端口输出流关闭时一并关闭 Alice 的事件流，因此 Alice 的运行时仍在公开路由之后；
- AliceSystem 是子系统装配根；AliceRuntime 只持有进程级执行机制，Profile 读取与缓存归 workspace，不参与单次 run 的控制链；
- RunSession 只拥有一次 run 的 frame registry 与 CALL record，不保存取消信号、活动 frame、frame 调度状态或传输层 stream sequence；
- RunExecutor 是唯一调用 `AgentRuntime.run_frame()` 的 Alice 编排组件。它以协程递归执行 CALL 派生 frame，并在根帧终态只记录一次 run 结果；意图收尾属于任务进程；
- AgentRunOutput 是调度与当前请求交互输出之间的窄端口；非流式使用 null 实现，流式使用 Alice runtime 的 queue-backed 实现；
- AgentRunStreamAdapter 只负责流式传输适配，AgentRunEventEmitter 只负责全局观测投影，两者不参与 frame 求值；
- CallContextProvider 通过 caller 的 `submit_operation` 提交目标 Profile 与 `context_refs` 读取请求，返回不含 frame 或 CALL ledger 状态的 `CallContext`；
- CallCoordinator 把 CALL 拆为 `begin_call()` 与 `complete_call()`：消费 `CallContext` 组装 callee、投影 outcome，并通过 `AgentRuntime.apply_call_response()` exactly-once 恢复 caller；它不解析 Profile/记忆，不运行 frame，也不收尾整个 run；
- FrameFactory 无状态地创建普通 frame，不表达主/子拓扑；
- `ResolveReferencesRequest` 让 context refs 与 READ 使用 workspace 的同一套引用解析、授权和引用记录；
- AgentRuntime 只运行给定 frame，不接触多 Agent 拓扑。

`RunExecutor` 是每次 run 独立创建的递归解释器，不维护 active-frame 状态机，也不把 caller/callee 写成两套绝对角色。`_execute_frame(frame)` 运行任意已登记 frame；遇到 CALL 时，它 `await` 递归执行新 frame，返回后继续同一个 caller。协程调用栈自然表达挂起与重入，`RunSession` 因而不再充当程序计数器。如果 AgentRuntime 开始创建 callee，或 CallCoordinator 再次调用 `run_frame()`，说明这组责任重新混合。

## 2. 主帧与运行作用域

每次 `run_agent()` 创建一个新的主 frame：

- `run_id=agent_run_<uuid>`，也是 `RunSession.agent_run_id`；任务进程的 `process_id` 随输入清单传入，作为外层关联值；
- 唯一且无拓扑含义的 `frame_id`；
- `topic_id` 指向 Patchouli 已准备的话题；
- `ExecutionLabels` 来自输入清单（经 `AgentRunContext` 转换），只保存注册时绑定的 agent/workspace 观测标签，不参与资源授权；
- 凭据绑定的 `submit_operation` 是资源请求的唯一交付入口，不暴露访问 context 或目标选择；
- `agent_profile` 是本次主 Agent 图纸；
- `working_history` 已由 PromptAssembler 组装。

FrameFactory 会把当前 user message 插入 `TurnEvent` 序列首位，使最终事件流拥有完整的一轮事实。随后 `RunExecutor._execute_frame()` 调用 `AgentRuntime.run_frame(frame)`：`SUSPENDED` 进入 CALL 事务并递归求值派生 frame；`COMPLETED/CANCELLED/FAILED/BUDGET_EXHAUSTED` 返回上一层。最外层入口 frame 的终态由 Alice 映射为 CPU 结果，任务进程据此认领或取消意图。

frame 是实际可恢复状态。恢复 caller 时必须继续使用原 frame，不能从消息重新构造一个“看似等价”的新 frame，否则迭代预算、事件序号、已经产生的正文和 ACK alias 清单都会分叉。父子、caller action 等调用关系只记录在 Alice 的 `CallRecord` 与事件元数据中，不进入 `RuntimeScope`。

RunSession 只登记 frame 与 CALL record，并校验 frame 属于当前 run、调用 action 唯一、callee 与 record 绑定一致。当前正在执行哪个 frame、caller 在哪里等待，都由 Python 协程调用栈表达；Session 不保存 `PENDING/RUNNABLE/RUNNING/WAITING/TERMINATED` 之类的调度状态。重复入口/callee、跨 run 绑定、重复 CALL action 和重复 apply 仍作为编排不变量抛出。

## 3. CALL trap 与重入

CALL 的稳定语法和参数见 [MTP 契约](../contracts/mtp.md)。在编排层，它是一种控制流陷入：

```text
current frame emits CALL
  -> Koakuma validates profile + FrameExecutionPolicy and returns SUSPEND
  -> AgentLoopExecutor records tool_call and returns FrameExecutionResult
  -> RunExecutor calls CallCoordinator.begin_call()
  -> register CallRecord before the first await
  -> CallContextProvider resolves target profile/context_refs
  -> CallCoordinator creates a normal callee frame from CallContext
  -> return DispatchCallee(callee)
  -> RunExecutor awaits _execute_frame(callee)
  -> CallCoordinator.complete_call() finalizes the callee frame
  -> call_response maps the finalized result to one MTPCallResponse
  -> AgentRuntime.apply_call_response() commits the response once
  -> emit sub_agent_end
  -> return ResumeCaller; task cancellation unwinds the recursive call stack
```

`suspend` 不能被当作一条正文为空的成功响应。如果执行循环直接继续，Agent 会在被调用任务尚未执行时向下生成，CALL 的任务身份、结果和取消边界都会丢失。RunExecutor 必须先等待被调用任务完成，再以同一 action_id 通过 `AgentRuntime.apply_call_response()` 回填 tool result。

### 3.1 Profile 解析

未提供 alias 时使用内置 `OMNI_DOLL_PROFILE`；`default` 与 `omni_doll` 是对同一内置 Profile 的显式选择，不是加载失败后的降级。Omni-Doll 对当前 verb/tool 使用显式白名单，因此后续新增能力不会自动穿透 fallback 边界。

所有 alias，包括未指定、`default` 与 `omni_doll`，都由 CallContextProvider 提交 `GetAgentProfileRequest`，经 workspace 能力层的 `profile.read` 授权。`WorkspaceOperationEntry` 只使用主线程执行凭据绑定的身份与目标；读取视图统一承担 Profile 解析、Workspace/alias 坐标定位、当前 actor 命中重验与 canonical 源变更失效。Patchouli 作为 Profile atom 所有者执行 Workspace ownership 和资源 policy 校验，再解析 persona、模型与权限。Alice 不保存另一份 Profile cache；管理入口修改源原子后，下一次 CALL 会通过失效视图读取新配置。`FrameExecutionPolicy` 仍按每次 CALL 从 Profile 派生，不进入 cache。

显式失败通过 `MTPCallResponse.error` 回填，不再启动子 frame：

| 场景 | 稳定 code | message key |
|:---|:---|:---|
| alias 不存在或源原子不可见 | `mtp.alias.not_found` | `mtp.call.profile_not_found` |
| 缺少 `profile.read` 操作授权 | `mtp.permission.denied` | `mtp.permission.verb_denied` |
| alias 类型不符 / Profile 配置无效 | `mtp.memory.type_mismatch` / `mtp.argument.invalid` | `mtp.call.profile_type_mismatch` / `mtp.call.profile_invalid` |
| Profile route 或读取失败 | 对应 `mtp.system.*` | `mtp.call.profile_load_failed` 或底层稳定 key |
| Profile 引用的模型不可用 | `mtp.system.service_unavailable` | `mtp.call_response.model_unavailable` |

CALL 的 `tool_call` 与 `tool_result` 使用同一个最终 success/error/cancelled 状态；内部 cause 只进入日志，不回填给 Agent。

Agent Profile 作为记忆存在，使服务发现可以复用预检索与 SEARCH：相关图纸可以在 memory context 中以 Agent Profile 菜单出现，主 Agent 随后用 alias 发起 CALL。Alice 当前不维护硬编码 team，也不会根据模糊需求动态创建 Profile。

### 3.2 共享上下文

`context_refs` 不是直接复制父 frame 的全部 history。CallContextProvider 逐个使用 caller frame 的 `submit_operation(ResolveReferencesRequest(...))` 解析：

- pending：共享同一 Workspace 中仍可回读的写入意图；
- redirect：共享已经结算后的 canonical atom；
- atom：共享正式记忆；
- discarded/failed/expired/not-found：记录 warning 并跳过。

有效 sources 交给 MemoryCompiler 的 `SHARED_CONTEXT_INJECTION` target，形成子 Agent system prompt 中的受控视图。这样父子进程共享的是明确引用，而不是一段无法追踪来源的任意拼接历史。

解析失败采用逐项 best effort：单个 ref 失败不阻断其他 ref 或整个 CALL；全部失败时子 Agent 仍只带 task 运行；`CancelledError` 原样传播。交付正式 atom 或可读 redirect 时，workspace 在返回前自动记录引用，READ 后再次作为 context ref 共享会形成另一次读取记录。当前 CALL response 不把这些跳过项作为结构化 warning 返回给主 Agent，只能从日志观察。

## 4. 瞬态子帧

子 frame 当前具有：

- 与 caller 相同的 `run_id`；
- 由 `FrameFactory` 生成的唯一 `frame_id`，不携带 `parent_frame_id` 或 `depth`；
- `topic_id=None`，不直接挂载 Patchouli 话题；
- 目标 Agent Profile；
- 由 persona、裁剪后的 MTP 教学、shared context 和 task 组成的全新消息历史；
- 继承父 frame 的同一组 `ExecutionLabels` 与凭据绑定的操作提交函数。

子帧不读取主话题完整 history，也不会把内部 token、SEARCH/RUN 重试或工具结果写回主 frame 的 working history。只有子帧以 `COMPLETED` 自然结束后，CallCoordinator 才取 `text_segments` 形成 reply，并把运行期间收到 ACK 的 aliases 放入 CALL artifacts。取消、失败或预算耗尽的子帧不会收割 reply/artifact，CallCoordinator 会经提交函数撤回该子帧收到 ACK 且仍为 PENDING 的意图；取消协程时由根进程关闭同步取消全部未认领意图。`SUSPENDED` 是 RunExecutor 必须继续递归消费的非终态 trap；若它进入 `complete_call()`，属于编排不变量违约，不构造 CALL response。

这种黑盒隔离避免主 Agent 与 Perception 被子任务细节淹没，但它并不等于子任务没有证据。子帧流事件仍可被 UI 观察，CALL 在主 frame 中有结构化 tool_call/tool_result，PendingAtom 又保存写入意图；只是这些事实目前没有被组合成持久化子任务 artifact。

## 5. 当前单层 CALL 策略

当前产品策略允许入口 Agent 串行调用多个子 Agent，被调用 Agent 不能再次 CALL：

```text
root frame
  ├─ CALL -> callee A
  ├─ CALL -> callee B
  └─ continue root frame
```

软限制由 PromptAssembler 实现：被调用 frame 的 prompt 始终移除 CALL 动词教学。硬限制由 `FrameExecutionPolicy` 实现：CallCoordinator 消费 CallContext 后，在 Profile 权限基础上显式移除 CALL；Koakuma 同时校验 Profile 与 frame policy。即使模型仍输出 CALL，也返回 `PermissionDeniedError`，执行循环把错误回填给该 Agent，让它改用自然语言或其他获准能力继续。

这是能力策略，不是 RunExecutor 的结构限制：执行器对所有 frame 使用同一个递归入口，不包含 `RootTurn/CalleeTurn` 分支。未来允许嵌套 CALL 时可以沿用这条递归路径；并行 sibling CALL 可在一次 suspension 能表达多个请求后，于当前递归层使用结构化并发。持久化 DAG、跨进程恢复和 review loop 仍属于后置方向。

## 6. 结果收割与回填

子帧成功结束后，CallCoordinator 通过 `AgentRuntime.finalize_frame()` 建立 artifact alias 列表：

`FrameProducts` 只投影本 frame 收到 ACK 后去重记录的 alias，不加入 UPDATE 的原基础 alias；caller 收到成功 CALL response 后把子帧产物加入自己的 alias 清单。

自然语言 reply 与 alias 列表组成 success `MTPCallResponse`。CallCoordinator 不直接操作 history 容器，而把 success/error/cancelled 终态响应交给 `AgentRuntime.apply_call_response()` 一次性加入 caller working history，并形成与原 CALL action_id 对应的 `tool_result`。caller 随后可以 READ pending alias、把它作为另一个 CALL 的 context ref，或直接根据子 Agent reply 继续任务。

CALL 故意没有配套的 MTP `RETURN` 动词。返回描述的是子 frame 生命周期的自然完成，不是一项新的记忆或工具动作；若再要求模型生成 `RETURN`，就会在已有执行终态之外增加一条语法、权限和 formatter 都可能失败的路径。当前由子帧自然结束触发返回，以自然语言 reply 表达结论，以 PendingAtom alias 收割表达可继续寻址的副作用，两者共同组成 CALL response。隐式返回只消除了重复协议动作，并不把任何退出都视作成功：`call_response.py` 仅将 `COMPLETED` 映射为 success，将 `CANCELLED` 映射为 cancelled，将 `FAILED`、`BUDGET_EXHAUSTED` 映射为带稳定 error code 的 error；`SUSPENDED` 不属于可映射终态。

caller 与 callee 使用同一个凭据绑定的操作提交函数，登记中的意图因而关联同一个 process ID。callee 未成功结束时，CallCoordinator 提交 `CancelIntentsRequest` 撤回 callee 已收到 ACK 的意图（只影响仍为 PENDING 的记录），避免它们在根帧 completed 后被认领物化。IPC alias 服务于 caller 当前认知，最终物化任务由任务进程在 completed 后直接从 workspace 登记认领；CPU 结果不携带物化任务，Alice 不再提供 run 级登记收尾。

## 7. 流式事件

流式 CALL 除普通 `token/mtp_start/mtp_result` 外增加：

- `sub_agent_start`：目标 alias、task、父迭代、depth 与 scope；
- 子帧自身的 token/MTP 事件：`scope=sub`，并带 depth/frame_id；
- `sub_agent_end`：最终 success/error/cancelled、子帧 `terminal_status`、目标 alias 与 frame id；success 携带 reply，error 携带稳定 `error_code`。

这些事件服务当前请求的实时 UI 与调试，不是业务结果来源。`AgentRunStreamAdapter` 为每次流式 run 创建容量为 256 的有界 FIFO queue，所有事件通过 `await put()` 施加背压；`QueueAgentRunOutput` 为事件补全 `agent_run_id/frame_id/action_id/stream_sequence`。`depth` 仅保留为兼容展示字段，不再是执行坐标。`sub_agent_start` 在 callee frame 创建后才发布，因此 `frame_id` 不为空。最终 `done` 携带的执行结果中的 `turn_events` 才是结构化的一轮事实，由任务进程封口后交给 Patchouli。

frame 的展示 `agent_id` 优先取 `AgentProfile.agent_id`（源 alias），空值回退注册标签；`agent.run.*` 的 agent/workspace 字段始终取注册标签，不从 Profile 或子帧重新推导身份。

交互输出不会自动转发到 RuntimeEventBus。后者只通过 `AgentRunEventEmitter` 记录主 `agent.run.*` 生命周期，采用 best-effort、可回放且允许慢订阅者丢失的语义；前者具有背压与断流取消语义。即使二者包含相同的 `agent_run_id/process_id` 关联字段，也不能把 RuntimeEvent 当作 token/CALL 流的备份或业务控制输入。

## 8. 失败、取消与降级

- 模型解析、generation/provider 等可归一化故障在最窄边界形成 `FrameExecutionResult.FAILED`；root 对外映射为失败 run，callee 对外映射为 CALL error；
- frame 注册、action/target、重复 apply、callee 关联与重复 finalize 等编排不变量继续抛出，不被 Executor 外层宽泛吞掉；
- CallContextProvider 的 Profile/共享上下文解析错误，以及后续模型调用或执行异常，会在 CALL 边界形成 error `MTPCallResponse`，主 Agent 得到错误后可以调整方案；
- 子帧预算耗尽映射为 `mtp.call_response.budget_exhausted`，子帧取消保持 cancelled 终态；生产 policy 会在 Agent loop 内拒绝被调用 frame 的 CALL，因此不会形成第二层 suspension；若未来开放该权限，RunExecutor 会直接递归执行，而不需要增加 root/callee 状态分支；
- 无法解析的单个 context ref 只跳过，不使 CALL 失败；
- Chat application 不向主/子 AgentRuntime 传递取消 token。用户 stop 取消当前 Alice task；RunExecutor 捕获 `CancelledError` 做本地 unwind，活跃 CALL 只清理 callee frame 和 record，不伪造 caller response，最外层收尾整个 run 一次，并保留原生异常传播；
- 流生成器提前关闭时，`AgentRunStream` 取消并 join 自己创建的 runner；`CancelledError` 沿递归协程栈展开，各层清理尚未 apply 的 CALL，且不再向关闭的 consumer 阻塞发送事件；
- RunExecutor、Agent loop 与 Worker 的 unwind/close 都是 best effort：清理异常记录日志，但不能替换正在传播的 `CancelledError`；
- 子 Agent 没有独立的公开取消句柄、重试策略或超时配置，生命周期依附于父 run。

将子 Agent 失败包装成 CALL error 是局部容错，不代表子任务成功；主 Agent 是否还能完成用户请求由后续生成决定。相反，主 frame 基础设施失败没有可用的上层 Agent 继续纠正，因此必须结束本次 run。

## 9. 关键不变量与矛盾检查

- CALL trap 必须由 Alice 恢复，不能由 Koakuma formatter 或 Agent loop 吞成普通响应；
- 子帧只接收 task 与显式 shared context，不能默认复制主 frame 全部工作历史；
- 主 Agent 最终负责用户回复，子 Agent 不直接写入主话题或向客户端产生第二个 done；
- Profile 的 persona 不能提升结构化权限，调用方 task 也不能替被调用者改写白名单；
- Pending alias 的 IPC 收割与任务进程的意图认领是两条不同用途的数据流；
- `context_refs` 必须经过操作请求的 workspace 引用解析与 MemoryCompiler，不能通过裸 UUID 或字符串拼接绕过 alias/状态语义；
- CALL 权限与预算必须随 frame policy 传播，不能只依赖 prompt 告诫模型；取消沿拥有 Alice run 的 task 递归展开，不通过 session 建立第二套控制面；
- Profile 与完整引用解析缓存只由 workspace 持有，Alice 不另建派生状态真相；观测标签不能成为资源授权依据。

## 10. 当前限制

- `AgentProfile` 保留来源 alias，但子 frame 共享父帧的凭据绑定提交函数，写意图的发起者仍是主线程 Actor；当前没有子线程独立授权；
- frame registry 与 CallRecord 由每次 run 新建的 `RunSession` 持有；执行位置由 RunExecutor 的协程调用栈表达；stream sequence 由每次流式 run 独占的 `QueueAgentRunOutput` 持有，当前没有共享 frame stack、活动 frame 状态机或共享输出队列；
- context ref 跳过只写日志，CALL response 没有 partial warning 列表；
- 子任务没有持久化 task id、独立 timeout/retry、并发额度、结果 artifact 或恢复机制；
- 当前产品 policy 只允许串行单层 CALL；RunExecutor 已能递归求值 frame，但尚未开放嵌套 CALL，也没有 parallel fan-out、动态 DAG、review loop 或 Alice 自主规划。

这些限制说明后续高级编排首先需要收紧身份、失败和并发状态，而不是先增加更多拓扑语法。只要同一个 frame 或 Profile 仍可能被错误归属，扩大 CALL 深度只会放大不可观测的矛盾。
