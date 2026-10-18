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
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-10-17
---

# Alice

Alice 是 HiveMemory 的 Agent 执行与多智能体编排子系统。若说 Gateway 决定一次输入应如何进入系统，Patchouli 决定哪些内容能够成为长期知识，那么 Alice 负责把已经准备好的本轮上下文变成行动：装载人偶图纸，驱动模型生成，在协议边界上调用记忆与工具，必要时派生一个瞬态子 Agent，最后交回结构化的运行结果。

“爱丽丝·玛格特罗依德”与“人偶使”的隐喻强调的不是一个无所不知的超级 Agent，而是控制面与执行单元的分离。人偶图纸可以决定人格与模型，访问登记决定操作权限，单 Agent Runtime 可以把一帧运行到收敛，Koakuma 可以解释 MTP 指令；Alice 自己维护的是谁在运行、何时挂起、何时派生、如何回填，以及本轮临时状态在什么边界上结束。

## 1. 设计定位

### 1.1 Alice 拥有什么

Alice 当前拥有：

- `CPUInputManifest -> CPUExecutionResult` 的执行边界：CPU 端口的 `execute` 以 `stream` 参数控制是否流式，接收任务进程组装的输入清单（`workspace.contracts`），在内部转换为提示词组装使用的 `AgentRunContext`；Alice 以 `AliceCPU` 实现 workspace 定义的 CPU 端口，由组合根注入任务进程；
- 主 Agent 与子 Agent 的 `ExecutionFrame`、帧进度、CALL 挂起与恢复；
- 单 Agent generate -> MTP -> 回填循环的装配与调用；
- 人格注入、模型选择和权限应用；主 Agent 的 Profile 由输入清单提供（任务进程在 CPU 分配时解析），CALL 子 Agent 的 Profile 随子线程派生由 workspace 能力层经操作请求读取；
- Koakuma MTP runtime、运行时 syscall registry 与格式化错误回填；
- 经线程 context 绑定的 `submit_operation` 提交资源操作请求、消费中立结果与记录本帧收到的 ACK alias；
- 共用一套执行骨架的非流式与流式 Agent run，以及 `agent.run.*` RuntimeEvent；

这里的“拥有”主要指运行时语义，而不是所有代码都必须位于 `alice/`。chat 任务类型的编排与进程控制也不在 Alice：任务进程表与 `TaskProcessService` 位于 `workspace.process`，经公开路由调用 Gateway 与 Patchouli、经 CPU 端口调用 Alice，不参与 Agent loop。`agent_runtime/` 是 Alice 消费的单 Agent 执行层，`AgentProfile` 与 `PendingAtom` 等跨边界模型位于 `core`，CPU 端口、输入清单与执行结果位于 `workspace.contracts`，prompt 组装位于 `prompts`。Alice 决定这些部件如何组成一次 run，但不能因此取得长期记忆、入口决策或顶层 chat 用例的所有权。

### 1.2 Alice 不拥有什么

Alice 不负责：

- 分析原始入口、识别系统命令、选择话题或形成 `GatewayDecision`；
- 创建、检索、归档或修订长期记忆的权威事实；
- 决定一次已完成 run 是否进入 Perception，或直接调用 Patchouli 内部 Runtime/Store；
- 在 Agent 执行运行时中持有 HTTP、SSE 连接、chat 总超时、跨子系统取消或 prepare/finalize 补偿（这些属于 `workspace.process` 的任务进程编排）；
- 自动生成 Agent Profile、维护持久化任务图或执行通用 plan-and-execute；
- 把 `WRITE` / `UPDATE` ACK 当作正式记忆落库成功；
- 为不受信任代码提供强安全沙箱。

chat 任务进程编排（`workspace.process`）拥有完整 chat 顺序和取消控制；Gateway 拥有入口解释；Patchouli 拥有 Profile 与 MemoryAtom 的持久化事实。Alice 通过 CPU 端口、公共模型与 `workspace.contracts.OperationSubmitter` 交接；资源请求由 `WorkspaceOperationEntry` 校验 CPU 附上的线程 context 后分派，写入意图登记、完整引用解析和全部原子/Profile 读取缓存属于 workspace。完整边界见[系统边界](../architecture/boundaries.md)与[子系统公共契约](../contracts/subsystem-contracts.md)。

## 2. Alice 与 AgentRuntime 为什么是两层

早期实现曾把 Agent loop、MTP 执行、PendingAtom、CALL 派生和帧调度压在同一命名空间中。它能跑通功能，却无法回答一个关键问题：当模型产生 CALL 时，究竟是“单个 Agent 的执行器”应自行递归，还是“多 Agent 调度器”应接回控制权。只要这个问题含糊，执行循环就会逐渐知道子 Agent、拓扑、结果收割和 IPC，Alice 反而只剩一个空壳名字。

当前设计用两层解决这个矛盾：

```text
AliceSystem
  ├─ AliceCPU                         CPU 端口、线程提交函数绑定与启停闸门
  │    ├─ AgentRunStreamAdapter       队列、runner task 与类型化终态交付
  │    └─ AgentRunService             同一运行用例，返回 CPUExecutionResult
  │         ├─ RootContextBuilder    根帧输入、会话视图与压缩
  │         └─ RunExecutor + RunSession
  │              -> CallCoordinator + CallContextProvider + FrameFactory
  │              -> AgentRuntime facade
  └─ build_agent_runtime             内核工厂：Koakuma、循环与 LLMClient
```

- `agent_runtime/` 是“CPU”：只关心把一个 `ExecutionFrame` 运行到自然收敛、取消或 CALL trap，不决定下一步该调度谁；
- Alice 编排层是“进程调度器”：创建主帧，消费 CALL trap，通过 CallContextProvider 取得子 Agent Profile/共享上下文，派生子帧，收割结果并恢复主帧；
- Patchouli 是长期存储与记忆域：Alice 可以提出物化请求，却不能自行确认长期事实已经成立。

Alice 内部依赖方向为 `interface → run → {context, orchestration} → events`；`context` 与 `orchestration` 互不导入。运行服务不依赖接口层，输出差异通过窄端口注入，避免传输与调度成环。Alice 与组合根只从 `hivememory.agent_runtime` 包根导入公开符号。

`agent_runtime/` 是顶层共享层，不是第四个子系统。它不实现 `SubsystemProtocol`，不注册全局公开路由，也不拥有独立启停；依赖的配置与模型注册表由 Alice 注入，线程 context 绑定的操作提交函数随 frame 传入。反向出现 `agent_runtime -> alice` 的领域依赖，或让执行循环重新决定子 Agent 拓扑，都意味着这道边界开始失效。

## 3. 人偶图纸：人格与权限分离

Agent Profile 是 Patchouli 中 `MemoryType.AGENT_PROFILE` 记忆的运行时投影。它延续“万物皆记忆”的设计：专业 Agent 不通过新增 Python 子类注册，而以可存储、可检索、可演化的图纸存在；Alice 在需要运行时把图纸解析为 `AgentProfile`。

一张图纸分为两部分：

- “灵魂”：`persona`，来自 MemoryAtom 正文，描述角色、工作方式和领域偏好；
- “骨架”：模型名、temperature、top_p 与语言，来自结构化 `agent_config`。

`AgentProfile.agent_id` 是 Profile 的身份，解析时取自源原子的 `index.alias`，覆盖 `agent_config` 中的同名键；builtin Profile 的 `agent_id` 为 `None`。由于 alias 在同一 Workspace 的中期库内唯一（见 [MemoryLibrary](../patchouli/memory-library.md)第 1.2 节），`agent_id` 在同一 Workspace 内同样唯一。可见性不进入 Profile 模型，仍由源原子的 `MemoryAccessPolicy` 表达。

图纸不承载权限。Agent 的操作权限只由 Workspace 访问登记的 actor 操作白名单表达：它与 Profile 原子是否存在、由谁编辑无关，也同样适用于没有 Profile 原子的外部 Actor。Alice 把输入清单携带的操作快照翻译为 frame 的 MTP 动词集合，用它裁剪 MTP 教学并在 Koakuma 执行前检查动词；资源操作仍由 actor API 逐次授权（[MTP Runtime](./mtp-runtime.md#3-动词集合的三层检查)第 3 节）。若把权限只写入自然语言人设，模型幻觉或 prompt injection 就可能绕过它，因此提示词层只降低误用概率，执行层与 workspace 授权才是闸门。`agent_config` 中残留的旧权限键（`allowed_mtp_verbs`、`allowed_sys_tools`）在解析时被忽略。

未指定主 Agent 时使用 `OMNI_DOLL_PROFILE`；显式选择 `default` / `omni_doll` 也会直接选择同一个内置 Profile，但不属于错误 fallback。Omni-Doll 无特定 persona、模型名为 `default`，verb/tool 使用当前内置能力的显式白名单，而不是 `None=未来所有能力也自动允许`。因此新增 MTP verb 或 syscall 时必须同步审查并更新白名单，不能悄悄扩大 fallback 权限。

主 Agent 的 Profile 由任务进程在 CPU 分配时经阶段授权读取，不要求任何 actor 操作；CALL 目标（包括内置 Profile）通过派生请求 `DeriveCallThreadRequest` 读取，经 workspace actor API 的 `agent.call` 授权，并由任务进程以被调用方准入后派生子线程。操作入口把请求与 CPU 附上的线程 context 交给 actor API，actor API 以线程的驻留 Workspace 为目标授权后，才把一次操作的 `IdentityScope` 交给读取视图和 Patchouli 公开边界；两条路径都由读取视图按当前 actor 检查源原子可见性。Patchouli 先检查 Workspace 归属，再检查资源可见性、MemoryType 与 `agent_config`。Profile 缓存由 workspace 按 Workspace/alias 坐标定位，条目保存源原子的 policy 与 UUID，命中时对当前 actor 重验；canonical 源变更失效与在途回填代次检查统一维护，详见 [Workspace 架构](../architecture/workspace.md)。显式 alias 不存在、越权、配置无效、读取失败或模型不可用都保持为结构化失败，不会改以 Omni-Doll 身份继续执行。

## 4. 当前主流程

### 4.1 主 Agent run

```text
任务进程的 CPUInputManifest
  -> 转换为 AgentRunContext（Alice 内部）
  -> attach thread-context-bound submit_operation to root frame
  -> resolve model budget; build MTP + persona + summary + memory/attachments + session view
  -> create root ExecutionFrame(topic_id=...)
  -> RunExecutor -> AgentRuntime.run_frame()
       -> LLM generate
       -> natural stop | MTP execute | CALL suspend | cancel
  -> AgentRunService assembles CPUExecutionResult
  -> 任务进程据执行结果决定是否封口交互记录并进入 Patchouli finalize
```

Alice 接收的是 workspace 任务进程组装的本轮输入清单，不在 run 中重新分析 Gateway，也不重新构造长期记忆上下文。主帧只保存运行所需的消息与 `ExecutionProgress`；最后的执行结果包含终态、自然语言正文、结构化 `TurnEvent[]`、实际模型展示名、可选失败原因和压缩检查点；迭代统计只出现在 `agent.run.*` 观测事件中。

`FrameExecutionResult.FAILED` 与 `BUDGET_EXHAUSTED` 会由 Alice 稳定映射为执行结果的 `failed`，基础设施异常才向上抛出并由任务进程按失败结束；取消返回 `cancelled`，不交出物化任务。有输出的 completed/cancelled/failed 都由任务进程封口、追加会话并进入 Patchouli finalize；Alice 只提供结果，CPU 取消时通过结果槽交回部分输出。主帧历史来自输入清单的全量会话轮次，Alice 按模型预算先减重旧工具结果，再按需生成增量摘要；原文与检查点由会话资源保存。根帧与 CALL 子帧每次生成前都有输入预算守卫；算法与取消规则见 [Agent Runtime](./agent-runtime.md#31-会话上下文视图)。Topic 内容不作为对话上下文。

### 4.2 CALL 与瞬态子帧

```text
root frame emits CALL
  -> Koakuma returns SUSPEND + MTPCallRequest
  -> RunExecutor 让 CallCoordinator 建立 CallRecord 并准备 callee
  -> CallContextProvider derive_call_thread：agent.call + 目标 Profile + 进程以被调用方准入派生子线程
  -> CallContextProvider resolve context_refs + compile shared context
  -> FrameFactory 创建普通 callee frame
  -> RunExecutor 递归等待 callee frame
  -> finalize_frame 投影 pending aliases + natural-language reply
  -> MTPCallResponse -> AgentRuntime.apply_call_response()
  -> RunExecutor 恢复同一个 caller frame 并继续生成
```

子 Agent 的完整试错轨迹留在自己的 frame 中，主话题只接收 CALL、结构化返回与最终主 Agent 输出。这种黑盒隔离不是为了隐藏事实，而是为了避免一次专项执行的迭代细节污染主上下文；需要共享的记忆由 `context_refs` 显式解析并通过 MemoryCompiler 注入。

当前只允许根 frame 发起 CALL。callee frame 是以被调用方身份运行的子线程：它的 `FrameExecutionPolicy` 由被调用方的操作快照翻译并移除 CALL，资源操作按被调用方的白名单与可见性授权，写入意图记在被调用方名下；Koakuma 以 policy 硬拒绝递归调用，形成单层串行星型拓扑。被调用方未在本 Workspace 准入时，CALL 以“agent 不可用”回填，不启动子 frame。

## 5. MTP 与临时写状态

Agent 使用 MTP 在生成过程中发现、读取和使用记忆，也可以提出长期写入或委派意图。完整语法和动词契约只在 [MTP 契约](../contracts/mtp.md)维护；Alice 文档只说明运行时怎样兑现它。

`WRITE` / `UPDATE` 不同步修改 Qdrant。Koakuma 提交操作请求，在 workspace 登记意图并立即交给 Patchouli 意图路由；生成队列接纳后返回 ACK 与 `draft_*` 或 `rev_*` 句柄，供当前及后续进程 READ。结算事件更新 workspace 登记。

这条写缓冲边界使 Agent 可以在同一 run 内读到自己的写意图，正式生成由 Patchouli 异步处理，随后执行失败或取消不撤回已接纳写入。Alice 的适配与 frame 产物见 [PendingAtom](./pending-atom.md)，登记状态和读取缓存的权威说明见 [Workspace 架构](../architecture/workspace.md)。

## 6. 启停、公开能力与观测

`AliceSystem.start()` 打开 CPU 端口的执行闸门，`stop()` 关闭它。CPU 端口始终可取得；未启动或已停止时，`execute()` 仍返回句柄，首次拉取输出才以 `AliceUnavailableError` 失败，由任务进程按失败收尾；异常上的诊断标识见[错误模型](../contracts/error-model.md)。Alice 不注册执行总线路由，也不需要全局总线参数。

`AliceCPU` 直接调用 `AgentRunService`，非流式只交回结果，流式依次交回交互事件与最后一个 `CPUExecutionResult`。检查点直接随结果传递，取消时由结果槽回传；交互流关闭必须先取消并等待 runner 收尾。Alice 没有独立后台 worker 或 shutdown drain，运行中的 chat 取消和连接关闭由 workspace 任务进程处理，System 在停止子系统前关闭任务进程。

Alice 与 agent_runtime 不持有 local bus，也不请求 Patchouli 公开路由；检索、引用与 CALL Profile 读取均提交 workspace 操作请求。PendingAtom 结算由 workspace 登记消费，派生读取缓存由 `WorkspaceRuntime.close()` 清理。

`CPUInputManifest`、内部 `AgentRunContext` 与 `RuntimeScope` 只保留冻结的 `ExecutionLabels(agent_id, workspace_id)` 观测标签，不携带授权身份；清单的操作快照在 `AgentRunContext` 中被翻译为 `permitted_mtp_verbs`，只供提示词与 frame 执行策略使用。标签取自注册入口绑定的同一组值，子帧沿用；提示词历史归并与运行事件使用注册标签，流式 frame 展示优先使用 `AgentProfile.agent_id`（源 alias），空值回退标签。标签不参与资源授权。

每次主 run 产生 `agent.run.started` 和 completed/cancelled/failed RuntimeEvent，终态事件载荷中的迭代统计从主 frame 进度取得。CPU 流以类型化执行结果标识正常终态；任务进程负责面向客户端的 `done` 投影。流式与非流式共用一套骨架，在终态前被关闭或取消时，两种模式都发出 cancelled 观测事件，失败时发出同一条 failed 事件。RuntimeEvent 是 best-effort 旁路，不参与业务成功判定。

## 7. 当前设计文档

- [Agent Runtime](./agent-runtime.md)：单 Agent 执行层、会话视图与压缩、ExecutionFrame、模型预算、循环与流式输出；
- [多 Agent 编排](./orchestration.md)：RunExecutor、CallCoordinator、CallContextProvider、CALL trap、Profile 解析、共享上下文、结果回流与当前单层策略；
- [PendingAtom](./pending-atom.md)：操作请求的写入意图适配、回读与 ACK alias 收割；
- [MTP Runtime](./mtp-runtime.md)：Koakuma、权限、verb 分发、syscall、错误、取消和真实安全边界。

跨子系统 route、event、MTP 语法和错误类型不在本目录复制，分别以[公开路由与事件](../contracts/routes-and-events.md)、[MTP 契约](../contracts/mtp.md)和[错误模型](../contracts/error-model.md)为准。

## 8. 代码与测试入口

| 责任 | 当前入口 |
|:---|:---|
| 子系统装配与生命周期 | `src/hivememory/alice/system.py` |
| Agent run 应用用例 | `src/hivememory/alice/run/service.py` |
| 内核工厂与门面 | `src/hivememory/agent_runtime/runtime.py` |
| CPU 端口与交互输出 | `src/hivememory/alice/interface/` |
| 根帧上下文与压缩 | `src/hivememory/alice/context/` |
| 多 Agent 编排 | `src/hivememory/alice/orchestration/executor.py`、`src/hivememory/alice/orchestration/call/`、`src/hivememory/alice/orchestration/session.py` |
| 单 Agent 执行层 | `src/hivememory/agent_runtime/`、`agent_runtime/runtime.py` |
| Prompt 与历史视图 | `src/hivememory/prompts/`、`prompts/transcript/history.py` |
| 公共运行模型 | `src/hivememory/workspace/contracts/process.py`（输入清单）、`src/hivememory/core/protocol/models.py`、`core/models/agent.py`、`agent_runtime/frame.py` |
| Alice 应用与编排测试 | `tests/unit/alice/run/`、`tests/unit/alice/orchestration/` |
| 执行与 MTP 测试 | `tests/unit/agent_runtime/` |
| 主动与 CALL 流程 | `tests/e2e/pipeline/test_agent_loop_e2e.py`、`test_chat_run_e2e.py`、`test_sub_agent_call_e2e.py`；CALL 子线程按被调用方授权与写入归属见 `tests/e2e/system/test_chat_mtp_operations.py`、`tests/integration/alice/orchestration/call/test_profile_requests.py` |

## 9. 当前限制与设计张力

- 能否 CALL 某个 Agent 只取决于调用方的 `agent.call`、目标 Profile 对调用方可见与被调用方的准入，没有“谁能调用谁”的细粒度策略；被调用方读到的 PRIVATE 记忆可能经 CALL 返回值交给调用方；
- 每次 run 的 frame registry 与 CallRecord 由独立 `RunSession` 持有，stream sequence 由流式输出端口持有；`RunExecutor` 用协程递归表达 CALL 的挂起与重入，不维护单活动 frame 状态机；Chat application 在更上层拥有可取消阶段 task。
- 子 Agent 异常会被包装为 CALL error 交给主 Agent 继续处理；取消、预算耗尽和意外挂起分别保持 cancelled 或稳定 error，不会被视作成功返回；
- Alice 的 Agent frame 不持久化，进程重启后不能恢复；workspace 的派生 cache 同样不跨重启保存；统一恢复边界见[耐久性与故障恢复治理](../governance/reliability/durability-and-recovery.md)；
- Alice 当前只有单层 CALL，不具备持久化 DAG、并行 specialist、review loop、配额或 backpressure；
- Koakuma 的若干配置字段和同步 syscall 仍有实现缺口，RUN 也不是不受信任代码的安全边界，详见 [MTP Runtime](./mtp-runtime.md)；
- `health()` 目前主要报告 AgentRuntime 与 Koakuma 的固定 `ok`，不探测模型、syscall 或正在运行的 frame。

这些缺口限定了 Alice 当前能够声称的是“单进程、有限 CALL、具备运行时记忆工具的 Agent 执行系统”，而不是一个可恢复、可横向扩展或强隔离的通用多 Agent 平台。未来高级编排只有在形成 Plan 并落地后，才能改写本目录中的当前能力描述。
