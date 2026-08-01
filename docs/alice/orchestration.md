---
title: Alice Multi-Agent Orchestration
status: current
owner: alice
scope: call-frame-scheduling-and-sub-agent-return
code_paths:
  - src/hivememory/alice/runtime/orchestrator.py
  - src/hivememory/alice/runtime/agent/frame_scheduler.py
  - src/hivememory/alice/runtime/agent/profile_resolver.py
  - src/hivememory/prompts/assembler.py
related_contracts:
  - docs/contracts/mtp.md
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/error-model.md
last_reviewed: 2026-08-01
---

# 多 Agent 编排

Alice 的多 Agent 能力当前不是一个会自主拆解任务图的“超级大脑”，而是一套有限、可解释的 CALL 控制流。主 Agent 在自己的生成过程中决定是否委派；Alice 在协议 trap 上接管执行，建立一个隔离的子 frame，将受控上下文交给目标 Profile，并把结果压缩成结构化 CALL response 后恢复主 Agent。

这套设计追求的不是让 Agent 彼此无限对话，而是让专项能力像可调用进程一样被发现和使用。主 Agent 保留用户任务与最终回答责任，子 Agent 只处理被委派的局部任务；子帧的试错过程不进入主话题，真正需要跨帧传递的知识必须通过 task、context refs、自然语言结果或 PendingAtom alias 明确表达。

## 1. 编排层组件

```text
AgentOrchestrator
  ├─ FrameScheduler
  │    ├─ create main frame
  │    ├─ suspend / resume stack
  │    └─ fork transient sub frame
  ├─ AgentProfileResolver
  │    └─ alias -> Patchouli AgentProfile -> LRU cache
  ├─ RuntimeAliasResolver
  │    └─ context_refs -> pending / redirect / atom
  └─ AgentRuntime facade
       └─ run one frame to completed / suspended / cancelled / failed / budget_exhausted
```

- Orchestrator 拥有 CALL 重入序列和最终 `AgentRunResult` 组装；
- FrameScheduler 拥有帧坐标、挂起栈与子帧消息构造；
- ProfileResolver 负责把可读 agent alias 解析为运行图纸；
- RuntimeAliasResolver 让 context refs 复用与 READ 相同的运行时寻址；
- AgentRuntime 只运行给定 frame，不接触多 Agent 拓扑。

如果 AgentRuntime 开始创建子帧，或 FrameScheduler 开始执行 MTP，说明这组责任重新混合。

## 2. 主帧与运行作用域

每次 `run_agent()` 创建一个新的主 frame：

- `run_id=run_<uuid>`，贯穿主帧和所有子帧；
- `frame_id=frame_main_<counter>`；
- `depth=0`；
- `topic_id` 指向 Patchouli 已准备的话题；
- `identity` 来自 `AgentRunContext`；
- `agent_profile` 是本次主 Agent 图纸；
- `working_history` 已由 PromptAssembler 组装。

Orchestrator 会把当前 user message 插入 `TurnEvent` 序列首位，使最终事件流拥有完整的一轮事实。随后它重复调用 `AgentRuntime.run_frame(main_frame)`：`SUSPENDED` 进入 CALL 重入，`COMPLETED/CANCELLED/FAILED/BUDGET_EXHAUSTED` 映射为主 run 的最终终态。

frame 是实际可恢复状态，FrameScheduler 的栈只表示当前编排挂起关系。恢复主 Agent 时必须继续使用原 frame，不能从消息重新构造一个“看似等价”的新 frame，否则迭代预算、事件序号、已经产生的正文和 PendingAtom action scope 都会分叉。

## 3. CALL trap 与重入

CALL 的稳定语法和参数见 [MTP 契约](../contracts/mtp.md)。在编排层，它是一种控制流陷入：

```text
main Agent emits CALL
  -> Koakuma validates root depth and returns SUSPEND
  -> AgentLoopExecutor records tool_call and returns FrameExecutionResult
  -> Orchestrator appends normalized CALL assistant message
  -> suspend main frame
  -> resolve target profile
  -> resolve and compile context_refs
  -> fork + run sub frame
  -> harvest result and aliases
  -> format MTPCallResponse
  -> append response + tool_result to main frame
  -> re-enter main frame
```

`suspend` 不能被当作一条正文为空的成功响应。如果执行循环直接继续，主 Agent 会在子任务尚未执行时向下生成，CALL 的任务身份、结果和取消边界都会丢失。Orchestrator 必须先补齐子任务，再以同一 action_id 回填 tool result。

### 3.1 Profile 解析

未提供 alias 时使用内置 `OMNI_DOLL_PROFILE`；`default` 与 `omni_doll` 是对同一内置 Profile 的显式选择，不是加载失败后的降级。Omni-Doll 对当前 verb/tool 使用显式白名单，因此后续新增能力不会自动穿透 fallback 边界。

其他 alias 必须随父 frame 的 `Identity` 解析。Alice 先查 Identity + alias 维度的 32 项 LRU cache，再通过 local bus 请求 Patchouli 的 `GET_AGENT_PROFILE` 能力；并发 cache miss 会串行复查，避免一个身份的授权结果污染另一个请求。Patchouli 作为 Profile atom 所有者执行 user 与 PUBLIC / WORKSPACE / PRIVATE 可见性校验，再解析 persona、模型和权限。

显式失败通过 `MTPCallResponse.error` 回填，不再启动子 frame：

| 场景 | 稳定 code | message key |
|:---|:---|:---|
| alias 不存在 | `mtp.alias.not_found` | `mtp.call.profile_not_found` |
| 当前 Identity 无权访问 | `mtp.permission.denied` | `mtp.call.profile_permission_denied` |
| alias 类型不符 / Profile 配置无效 | `mtp.memory.type_mismatch` / `mtp.argument.invalid` | `mtp.call.profile_type_mismatch` / `mtp.call.profile_invalid` |
| Profile route 或读取失败 | 对应 `mtp.system.*` | `mtp.call.profile_load_failed` 或底层稳定 key |
| Profile 引用的模型不可用 | `mtp.system.service_unavailable` | `mtp.call_response.model_unavailable` |

CALL 的 `tool_call` 与 `tool_result` 使用同一个最终 success/error/cancelled 状态；内部 cause 只进入日志，不回填给 Agent。

Agent Profile 作为记忆存在，使服务发现可以复用预检索与 SEARCH：相关图纸可以在 memory context 中以 Agent Profile 菜单出现，主 Agent 随后用 alias 发起 CALL。Alice 当前不维护硬编码 team，也不会根据模糊需求动态创建 Profile。

### 3.2 共享上下文

`context_refs` 不是直接复制父 frame 的全部 history。Orchestrator 逐个使用 RuntimeAliasResolver 解析：

- pending：共享尚未物化的本轮写意图；
- redirect：共享已经结算后的 canonical atom；
- atom：共享正式记忆；
- discarded/failed/expired/not-found：记录 warning 并跳过。

有效 sources 交给 MemoryCompiler 的 `SHARED_CONTEXT_INJECTION` target，形成子 Agent system prompt 中的受控视图。这样父子进程共享的是明确引用，而不是一段无法追踪来源的任意拼接历史。

解析失败采用逐项 best effort：单个 ref 失败不阻断其他 ref 或整个 CALL；全部失败时子 Agent 仍只带 task 运行。当前 CALL response 不把这些跳过项作为结构化 warning 返回给主 Agent，只能从日志观察。

## 4. 瞬态子帧

子 frame 当前具有：

- 与父帧相同的 `run_id`；
- 新的 `frame_sub_<counter>` 与 `parent_frame_id`；
- `depth=1`；
- `topic_id=None`，不直接挂载 Patchouli 话题；
- 目标 Agent Profile；
- 由 persona、裁剪后的 MTP 教学、shared context 和 task 组成的全新消息历史；
- 继承父 frame 的 `Identity`。

子帧不读取主话题完整 history，也不会把内部 token、SEARCH/RUN 重试或工具结果写回主 frame 的 working history。只有子帧以 `COMPLETED` 自然结束后，Orchestrator 才取 `text_segments` 形成 reply，并把运行期间产生的 pending aliases 放入 CALL artifacts。取消、失败、预算耗尽或意外挂起的子帧不会收割 reply/artifact，其 frame 内尚未结算的 PendingAtom 会被取消。

这种黑盒隔离避免主 Agent 与 Perception 被子任务细节淹没，但它并不等于子任务没有证据。子帧流事件仍可被 UI 观察，CALL 在主 frame 中有结构化 tool_call/tool_result，PendingAtom 又保存写入意图；只是这些事实目前没有被组合成持久化子任务 artifact。

## 5. 单层星型拓扑

当前拓扑是根 Agent 可以串行调用多个子 Agent，子 Agent 不能再次 CALL：

```text
main(depth=0)
  ├─ CALL -> sub A(depth=1)
  ├─ CALL -> sub B(depth=1)
  └─ continue main
```

软限制由 PromptAssembler 实现：为 depth >= 1 的子帧移除 CALL 动词教学。硬限制由 Koakuma 实现：即使模型仍输出 CALL，也返回 `PermissionDeniedError(call_depth_exceeded)`，执行循环把错误回填给该子 Agent，让它改用自然语言或其他获准能力继续。

限制深度不是把复杂拓扑永久排除，而是确保当前 frame、取消、预算和错误语义尚未持久化时，不会出现无法收束的递归调用。并行 specialist、DAG 和 review loop 仍属于后置方向。

## 6. 结果收割与回填

子帧成功结束后，Orchestrator 从两个来源建立 artifact alias 列表：

1. 按 `frame_id` 查询共享 PendingAtomRuntime，取得子帧 WRITE/UPDATE 产生的 pending alias；
2. 对 UPDATE tool event 进行兼容 fallback，补入尚未登记为 pending 的 target alias。

自然语言 reply 与 alias 列表组成 `MTPCallResponse`。成功响应加入主 frame working history，并形成与原 CALL action_id 对应的 `tool_result`；主 Agent 随后可以 READ pending alias、把它作为另一个 CALL 的 context ref，或直接根据子 Agent reply 继续任务。

CALL 故意没有配套的 MTP `RETURN` 动词。返回描述的是子 frame 生命周期的自然完成，不是一项新的记忆或工具动作；若再要求模型生成 `RETURN`，就会在已有执行终态之外增加一条语法、权限和 formatter 都可能失败的路径。当前由子帧自然结束触发返回，以自然语言 reply 表达结论，以 PendingAtom alias 收割表达可继续寻址的副作用，两者共同组成 CALL response。隐式返回只消除了重复协议动作，并不把任何退出都视作成功：Orchestrator 检查 `FrameExecutionResult`，仅将 `COMPLETED` 映射为 success，将 `CANCELLED` 映射为 cancelled，将 `FAILED`、`BUDGET_EXHAUSTED` 和意外 `SUSPENDED` 映射为带稳定 error code 的 error。

父子帧共享 run_id，因此最终物化任务不依赖这份 IPC harvest：run 结束时 PendingAtomRuntime 会按 run_id 收集父子帧全部 PENDING 原子。IPC aliases 服务于主 Agent 当前认知，materialize task 服务于 Alice -> Patchouli 的数据交接，两者不能混为一份真相。

## 7. 流式事件

流式 CALL 除普通 `token/mtp_start/mtp_result` 外增加：

- `sub_agent_start`：目标 alias、task、父迭代、depth 与 scope；
- 子帧自身的 token/MTP 事件：`scope=sub`，并带 depth/frame_id；
- `sub_agent_end`：最终 success/error/cancelled、子帧 `terminal_status`、目标 alias 与 frame id；success 携带 reply，error 携带稳定 `error_code`。

这些事件服务实时 UI 与调试，不是业务结果来源。最终 `done.AgentRunResult.turn_events` 才是交给 Patchouli 的结构化一轮事实；RuntimeEvent 则只记录主 `agent.run.*` 生命周期。

## 8. 失败、取消与降级

- 主 frame 的模型或执行异常直接向 AliceRuntime 抛出，整个 agent run 失败；
- 子 Agent Profile 解析、共享上下文、模型调用或执行异常会被捕获并形成 error `MTPCallResponse`，主 Agent 得到错误后可以调整方案；
- 子帧预算耗尽和意外再次挂起分别使用 `mtp.call_response.budget_exhausted` 和 `mtp.call_response.unexpected_suspend`；子帧取消则保持 cancelled 终态；
- 无法解析的单个 context ref 只跳过，不使 CALL 失败；
- run 取消 token 会传给主/子 AgentRuntime；最终主结果为 cancelled，并取消本 run 尚未结算的 PendingAtom，不交出 materialize tasks；
- 流生成器提前关闭时，AliceRuntime 设置调用方 cancel token 并发出 cancelled 观测事件；
- 子 Agent 没有独立的公开取消句柄、重试策略或超时配置，生命周期依附于父 run。

将子 Agent 失败包装成 CALL error 是局部容错，不代表子任务成功；主 Agent 是否还能完成用户请求由后续生成决定。相反，主 frame 基础设施失败没有可用的上层 Agent 继续纠正，因此必须结束本次 run。

## 9. 关键不变量与矛盾检查

- CALL trap 必须由 Alice 恢复，不能由 Koakuma formatter 或 Agent loop 吞成普通响应；
- 子帧只接收 task 与显式 shared context，不能默认复制主 frame 全部工作历史；
- 主 Agent 最终负责用户回复，子 Agent 不直接写入主话题或向客户端产生第二个 done；
- Profile 的 persona 不能提升结构化权限，调用方 task 也不能替被调用者改写白名单；
- Pending alias 的 IPC 收割与 run 级 materialize task 收集是两条不同用途的数据流；
- `context_refs` 必须经过 RuntimeAliasResolver 与 MemoryCompiler，不能通过裸 UUID 或字符串拼接绕过 alias/状态语义；
- 调用深度、预算与取消必须随 RuntimeScope/frame 传播，不能只依赖 prompt 告诫模型；
- Alice 可以持有 Profile 运行时 cache，却不能把它当成 Patchouli 中 Profile 记忆的第二份权威事实。

## 10. 当前限制

- AgentProfile cache 已按 Identity + alias 隔离、上限固定为 32，但没有 TTL、版本检查或管理事件失效；Profile 更新要等 LRU 淘汰或进程重启才可靠生效；
- `AgentProfile` 模型不保存来源 atom alias，子 frame 又继承父 Identity。执行层子事件可能把 `agent_id` 标为父 Agent，子帧创建的 PendingAtom 也无法仅凭 Identity 证明真实 CALL 目标；
- FrameScheduler 的 `_frame_stack` 是 AliceRuntime 级共享列表，不是 task-local。并发 run 的 CALL 可以交错 push/pop，当前代码又不核对 resume 返回的 frame；
- context ref 跳过只写日志，CALL response 没有 partial warning 列表；
- 子任务没有持久化 task id、独立 timeout/retry、并发额度、结果 artifact 或恢复机制；
- 当前只能串行单层 CALL，没有 parallel fan-out、动态 DAG、review loop 或 Alice 自主规划。

这些限制说明后续高级编排首先需要收紧身份、失败和并发状态，而不是先增加更多拓扑语法。只要同一个 frame 或 Profile 仍可能被错误归属，扩大 CALL 深度只会放大不可观测的矛盾。
