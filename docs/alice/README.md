---
title: Alice
status: current
owner: alice
scope: agent-execution-and-multi-agent-orchestration
code_paths:
  - src/hivememory/alice/
  - src/hivememory/agent_runtime/
  - src/hivememory/prompts/
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/mtp.md
  - docs/architecture/boundaries.md
last_reviewed: 2026-08-03
---

# Alice

Alice 是 HiveMemory 的 Agent 执行与多智能体编排子系统。若说 Gateway 决定一次输入应如何进入系统，Patchouli 决定哪些内容能够成为长期知识，那么 Alice 负责把已经准备好的本轮上下文变成行动：装载人偶图纸，驱动模型生成，在协议边界上调用记忆与工具，必要时派生一个瞬态子 Agent，最后交回结构化的运行结果。

“爱丽丝·玛格特罗依德”与“人偶使”的隐喻强调的不是一个无所不知的超级 Agent，而是控制面与执行单元的分离。人偶图纸可以决定人格、模型和权限，单 Agent Runtime 可以把一帧运行到收敛，Koakuma 可以解释 MTP 指令；Alice 自己维护的是谁在运行、何时挂起、何时派生、如何回填，以及本轮临时状态在什么边界上结束。

## 1. 设计定位

### 1.1 Alice 拥有什么

Alice 当前拥有：

- `AgentRunContext -> AgentRunResult` 的执行边界；
- 主 Agent 与子 Agent 的 `ExecutionFrame`、帧进度、CALL 挂起与恢复；
- 单 Agent generate -> MTP -> 回填循环的装配与调用；
- Agent Profile 的运行时解析、人格注入、模型选择和权限应用；
- Koakuma MTP runtime、运行时 syscall registry 与格式化错误回填；
- PendingAtom 的进程内写缓冲、临时 alias、物化任务投影和 settlement 运行时视图；
- 非流式与流式 Agent run，以及 `agent.run.*` RuntimeEvent；
- Alice 私有 local bus，并经它代理 Patchouli 的公开记忆能力。

这里的“拥有”主要指运行时语义，而不是所有代码都必须位于 `alice/`。`agent_runtime/` 是 Alice 消费的单 Agent 执行层，`AgentProfile`、`PendingAtom` 与 `AgentRunResult` 等跨边界模型位于 `core`，prompt 组装位于 `prompts`。Alice 决定这些部件如何组成一次 run，但不能因此取得长期记忆、入口决策或顶层 chat 用例的所有权。

### 1.2 Alice 不拥有什么

Alice 不负责：

- 分析原始入口、识别系统命令、选择话题或形成 `GatewayDecision`；
- 创建、检索、归档或修订长期记忆的权威事实；
- 决定一次已完成 run 是否进入 Perception，或直接调用 Patchouli 内部 Runtime/Store；
- 持有 HTTP、SSE 连接、chat 总超时、跨子系统取消或 prepare/finalize 补偿；
- 自动生成 Agent Profile、维护持久化任务图或执行通用 plan-and-execute；
- 把 `WRITE` / `UPDATE` ACK 当作正式记忆落库成功；
- 为不受信任代码提供强安全沙箱。

System 应用层拥有完整 chat 顺序和取消控制；Gateway 拥有入口解释；Patchouli 拥有 Profile 与 MemoryAtom 的持久化事实。Alice 只通过公开 route、公共模型和 settlement event 与它们交接。完整边界见[系统边界](../architecture/boundaries.md)与[子系统公共契约](../contracts/subsystem-contracts.md)。

## 2. Alice 与 AgentRuntime 为什么是两层

早期实现曾把 Agent loop、MTP 执行、PendingAtom、CALL 派生和帧调度压在同一命名空间中。它能跑通功能，却无法回答一个关键问题：当模型产生 CALL 时，究竟是“单个 Agent 的执行器”应自行递归，还是“多 Agent 调度器”应接回控制权。只要这个问题含糊，执行循环就会逐渐知道子 Agent、拓扑、结果收割和 IPC，Alice 反而只剩一个空壳名字。

当前设计用两层解决这个矛盾：

```text
AliceSystem
  ├─ AgentRunService                  public run / run_stream use case
  │    -> RunExecutor + RunSession    recursive execution + run-local ledger
  │         -> CallCoordinator + CallContextProvider + FrameFactory
  │         -> AgentRuntime facade
  ├─ AliceRuntime                     process-local execution resources
  │    -> Koakuma + PendingAtomRuntime + alias/profile caches
  └─ AliceBridge                      public routes + Patchouli proxies/events
```

- `agent_runtime/` 是“CPU”：只关心把一个 `ExecutionFrame` 运行到自然收敛、取消或 CALL trap，不决定下一步该调度谁；
- Alice 编排层是“进程调度器”：创建主帧，消费 CALL trap，通过 CallContextProvider 取得子 Agent Profile/共享上下文，派生子帧，收割结果并恢复主帧；
- Patchouli 是长期存储与记忆域：Alice 可以提出物化请求，却不能自行确认长期事实已经成立。

`agent_runtime/` 是顶层共享层，不是第四个子系统。它不实现 `SubsystemProtocol`，不注册全局公开路由，也不拥有独立启停；依赖的 bus、配置、模型注册表和 PendingAtomRuntime 均由 Alice 注入。反向出现 `agent_runtime -> alice` 的领域依赖，或让执行循环重新决定子 Agent 拓扑，都意味着这道边界开始失效。

## 3. 人偶图纸：人格与权限分离

Agent Profile 是 Patchouli 中 `MemoryType.AGENT_PROFILE` 记忆的运行时投影。它延续“万物皆记忆”的设计：专业 Agent 不通过新增 Python 子类注册，而以可存储、可检索、可演化的图纸存在；Alice 在需要运行时把图纸解析为 `AgentProfile`。

一张图纸分为两部分：

- “灵魂”：`persona`，来自 MemoryAtom 正文，描述角色、工作方式和领域偏好；
- “骨架”：模型名、temperature、top_p、语言、`allowed_mtp_verbs` 与 `allowed_sys_tools`，来自结构化 `agent_config`。

权限与 persona 分离不是排版选择。若把工具权限只写入自然语言人设，模型幻觉或 prompt injection 就可能绕过它；当前实现既按 Profile 裁剪 MTP 教学与工具菜单，也在 Koakuma 执行前再次执行 verb/tool 白名单检查。提示词层降低误用概率，Runtime 层才是当前真正的执行闸门。

未指定主 Agent 时使用 `OMNI_DOLL_PROFILE`；显式选择 `default` / `omni_doll` 也会直接选择同一个内置 Profile，但不属于错误 fallback。Omni-Doll 无特定 persona、模型名为 `default`，verb/tool 使用当前内置能力的显式白名单，而不是 `None=未来所有能力也自动允许`。因此新增 MTP verb 或 syscall 时必须同步审查并更新白名单，不能悄悄扩大 fallback 权限。

自定义 Profile 必须携带调用 `Identity` 交由 Patchouli 解析。Patchouli 先按 user 边界读取 atom，再检查 PUBLIC / WORKSPACE / PRIVATE 可见性、MemoryType 与 `agent_config`；Alice 只缓存已经授权的结果，cache key 包含 user、agent、team 与 alias。显式 alias 不存在、越权、配置无效、读取失败或模型不可用都保持为结构化失败，不会改以 Omni-Doll 身份继续执行。

## 4. 当前主流程

### 4.1 主 Agent run

```text
Patchouli AgentRunContext
  -> warm pre-retrieval MemoryAtoms into alias cache
  -> assemble MTP + persona + memory + topic messages
  -> create root ExecutionFrame(topic_id=...)
  -> RunExecutor -> AgentRuntime.run_frame()
       -> LLM generate
       -> natural stop | MTP execute | CALL suspend | cancel
  -> AgentRunService assembles AgentRunResult
  -> System decides whether Patchouli finalize may run
```

Alice 接收的是 Patchouli 已经准备好的本轮快照，不在 run 中重新分析 Gateway，也不重新构造长期记忆上下文。主帧只保存运行所需的消息与 `ExecutionProgress`；最后的 `AgentRunResult` 包含自然语言正文、结构化 `TurnEvent[]`、迭代统计、实际模型展示名和 `PendingAtomMaterializeTask[]`。

`FrameExecutionResult.FAILED` 与 `BUDGET_EXHAUSTED` 会由 Alice 稳定映射为 `AgentRunStatus.FAILED`，基础设施异常才向上抛出并由 System 结束 chat 用例；取消返回 `cancelled`，不交出物化任务。只有完成的 run 才会在 System 管理的主动流程中进入 Patchouli finalize。

### 4.2 CALL 与瞬态子帧

```text
root frame emits CALL
  -> Koakuma returns SUSPEND + MTPCallRequest
  -> RunExecutor 让 CallCoordinator 建立 CallRecord 并准备 callee
  -> CallContextProvider resolve target Agent Profile
  -> CallContextProvider resolve context_refs + compile shared context
  -> FrameFactory 创建普通 callee frame
  -> RunExecutor 递归等待 callee frame
  -> finalize_frame 投影 pending aliases + natural-language reply
  -> MTPCallResponse -> AgentRuntime.apply_call_response()
  -> RunExecutor 恢复同一个 caller frame 并继续生成
```

子 Agent 的完整试错轨迹留在自己的 frame 中，主话题只接收 CALL、结构化返回与最终主 Agent 输出。这种黑盒隔离不是为了隐藏事实，而是为了避免一次专项执行的迭代细节污染主上下文；需要共享的记忆由 `context_refs` 显式解析并通过 MemoryCompiler 注入。

当前只允许根 frame 发起 CALL。callee frame 的 `FrameExecutionPolicy` 会移除 CALL，Koakuma 同时以 policy 硬拒绝递归调用，形成单层串行星型拓扑。

## 5. MTP 与临时写状态

Agent 使用 MTP 在生成过程中发现、读取和使用记忆，也可以提出长期写入或委派意图。完整语法和动词契约只在 [MTP 契约](../contracts/mtp.md)维护；Alice 文档只说明运行时怎样兑现它。

`WRITE` / `UPDATE` 不同步修改 Qdrant。Koakuma 立即创建 PendingAtom 并返回 `draft_*` 或 `rev_*` 临时句柄，使当前 run 可以继续 READ；run 完成时 Alice 将仍为 PENDING 的原子投影为 materialize task，Patchouli 后台生成完成后再通过 settled/failed/cancelled event 回填运行时状态。

这条写缓冲边界使 Agent 可以在同一 run 内读到自己的写意图，又不会让半完成、失败或取消的执行直接污染长期记忆。详细状态与回收语义见 [PendingAtom](./pending-atom.md)。

## 6. 启停、公开能力与观测

`AliceSystem.start()` 通过 AliceBridge 向 `GlobalSystemBus` 注册 `alice.public.run_agent` 与 `alice.public.run_agent_stream`，并在 AliceBus 上挂载访问 Patchouli 公开能力的代理；停止时按相反顺序卸载。Alice 不再为 run workflow 维护一套无人消费的 local route。Alice 没有独立后台 worker 或 shutdown drain，运行中的 chat 取消和连接关闭由 System 应用层持有的控制状态处理。

AliceRuntime 还订阅 PatchouliBridge 发布的 PendingAtom settled/failed/cancelled 业务事件。事件只更新 Alice 的运行时投影；正式记忆是否落库仍以 Patchouli 为准。

每次主 run 产生 `agent.run.started` 和 completed/cancelled/failed RuntimeEvent。流式路径必须出现 `done` 才被视为正常终态；流生成器在终态前关闭时，Alice 会请求取消并发出 cancelled 观测事件。RuntimeEvent 是 best-effort 旁路，不参与业务成功判定。

## 7. 当前设计文档

- [Agent Runtime](./agent-runtime.md)：单 Agent 执行层、ExecutionFrame、prompt、模型解析、循环与流式输出；
- [多 Agent 编排](./orchestration.md)：RunExecutor、CallCoordinator、CallContextProvider、CALL trap、Profile 解析、共享上下文、结果回流与当前单层策略；
- [PendingAtom](./pending-atom.md)：运行时写缓冲、状态机、物化任务、settlement、redirect 与回收；
- [MTP Runtime](./mtp-runtime.md)：Koakuma、权限、verb 分发、syscall、错误、取消和真实安全边界。

跨子系统 route、event、MTP 语法和错误类型不在本目录复制，分别以[公开路由与事件](../contracts/routes-and-events.md)、[MTP 契约](../contracts/mtp.md)和[错误模型](../contracts/error-model.md)为准。

## 8. 代码与测试入口

| 责任 | 当前入口 |
|:---|:---|
| 子系统装配与生命周期 | `src/hivememory/alice/system.py` |
| Agent run 应用用例 | `src/hivememory/alice/application/agent_run_service.py` |
| Alice 进程级资源与 local bus | `src/hivememory/alice/runtime/core.py`、`bus.py` |
| 多 Agent 编排 | `src/hivememory/alice/orchestration/run_executor.py`、`src/hivememory/alice/orchestration/sub_agent/`、`src/hivememory/alice/orchestration/run_session.py` |
| 单 Agent 执行层 | `src/hivememory/agent_runtime/`、`agent_runtime/runtime.py` |
| Prompt 与历史视图 | `src/hivememory/prompts/`、`engines/perception/context_converter.py` |
| 公共运行模型 | `src/hivememory/core/protocol/models.py`、`core/models/{agent,pending}.py` |
| Alice 应用与编排测试 | `tests/unit/alice/application/`、`tests/unit/alice/orchestration/` |
| 执行、PendingAtom 与 MTP 测试 | `tests/unit/agent_runtime/` |
| 主动与 CALL 流程 | `tests/e2e/pipeline/test_agent_loop_e2e.py`、`test_chat_run_e2e.py`、`test_sub_agent_call_e2e.py` |

## 9. 当前限制与设计张力

- AgentProfile cache 是进程内、按 Identity + alias 隔离的 32 项 LRU，但仍没有 TTL、更新事件或显式失效入口；Profile 修改可能在进程内长期不可见；
- `ExecutionFrame.identity` 在子帧中继承父帧，`AgentProfile` 又不携带解析 alias；因此部分子帧流事件和 PendingAtom provenance 会记录父 Agent，而不是实际 CALL 目标；
- KoakumaAtomCache 与 PendingAtomRuntime 都由 AliceRuntime 进程级共享。L0/L1 alias 命中当前不会再次校验调用 Identity，尚未满足跨用户并发运行所需的隔离；
- 每次 run 的 frame registry 与 CallRecord 由独立 `RunSession` 持有，stream sequence 由流式输出端口持有；`RunExecutor` 用协程递归表达 CALL 的挂起与重入，不维护单活动 frame 状态机；Chat application 在更上层拥有可取消阶段 task。
- 子 Agent 异常会被包装为 CALL error 交给主 Agent 继续处理；取消、预算耗尽和意外挂起分别保持 cancelled 或稳定 error，不会被视作成功返回；
- Agent frame、PendingAtom、alias cache 与 Profile cache 均不持久化，进程重启后不能恢复；统一恢复边界见[耐久性与故障恢复治理](../governance/reliability/durability-and-recovery.md)；
- Alice 当前只有单层 CALL，不具备持久化 DAG、并行 specialist、review loop、配额或 backpressure；
- Koakuma 的若干配置字段和同步 syscall 仍有实现缺口，RUN 也不是不受信任代码的安全边界，详见 [MTP Runtime](./mtp-runtime.md)；
- `health()` 目前主要报告 AgentRuntime 与 Koakuma 的固定 `ok`，不探测模型、syscall、缓存隔离或正在运行的 frame。

这些缺口限定了 Alice 当前能够声称的是“单进程、有限 CALL、具备运行时记忆工具的 Agent 执行系统”，而不是一个可恢复、可横向扩展或强隔离的通用多 Agent 平台。未来高级编排只有在形成 Plan 并落地后，才能改写本目录中的当前能力描述。
