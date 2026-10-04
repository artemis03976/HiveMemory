---
title: System Application Services
status: current
owner: system
scope: application-use-cases-and-cross-subsystem-orchestration
code_paths:
  - src/hivememory/system/application/
  - src/hivememory/workspace/process/
  - src/hivememory/workspace/capability/
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
related_docs:
  - docs/architecture/workspace.md
  - docs/architecture/boundaries.md
  - docs/system/attachments.md
last_reviewed: 2026-10-04
---

# 应用服务

应用服务是 transport 与子系统之间的用例层。它们由 System 组合根装配、经 `HiveMemorySystem` 门面交给入口，但按归属分布在三处：

- 任务进程表与 chat 任务进程编排：`workspace.process`（`ProcessTable`/`ProcessRecord` 进程表、`TaskProcessService` 唯一注册入口与 `TaskProcess` 四阶段骨架）；
- 资源能力层：`workspace.capability`（Memory、MemoryTask、Agent Profile、Topic、WorkspaceAsset 服务）；
- 系统级服务：`system.application`（被动摄入与就绪检查）。

本文统一描述它们共同遵守的用例层规则。它们回答“这次请求应该按什么顺序跨边界运行”，而不回答“记忆如何检索”“Agent 如何生成”或“Gateway 如何分析”。

这一层存在，是因为 HTTP、SSE、CLI 和未来外部 adapter 都需要共享同一条主动 chat、取消、被动摄入和管理 API 链路。若每个 router 自己拼装 Gateway、Patchouli 和 Alice，就会再次出现多套 prepare/finalize、错误处理和取消语义。

## 1. 共同边界

应用服务的共同规则是：

1. 通过 `GlobalSystemBus` 请求公开 route；
2. 在边界处接受访问 context 与目标 Workspace、请求模型或面向 API 的结果，不接受调用方另行组装的 `IdentityScope`；
3. 不持有另一个子系统的 Runtime、Service、Controller 或 local bus；
4. 不把内部 execution state、fallback 原因和观测事件直接返回给外部客户端；
5. 对取消、失败和 cleanup 保持与 Contracts 一致的终态。

身份与访问的交接遵循两阶段认证与两阶段授权（[Workspace 架构](../architecture/workspace.md)第 4 节）：server 在 `server/deps.py` 用 `resolve_request_identity_claims` 把用户导向身份选择（`user_id + workspace_id`，Agent action 附加 `agent_id`）解析为身份声明，交给统一认证网关取得访问 context——chat 由任务进程的注册入口认证，管理操作与取消取得绑定本次请求的请求级 context。应用服务不解析身份，也不执行默认解析；非 Agent action 的声明使用保留 `system` actor，只标记"没有具体 Agent 作为操作来源主体"。后台 task、retry 和 finalize 不重新读取进程当前 Workspace；它们使用自身 DTO 中保存的 scope，在最终访问 Workspace-owned 资源时由领域所有者校验。应用服务不会因此拥有 Workspace 资源，也不会为共享 runtime 创建按 Workspace 分区的状态。

应用服务可以保存一次用例的短期控制状态，例如任务进程编排的进程表，但不能保存 Patchouli 的长期记忆状态或 Gateway 的请求级 workflow state。

workspace 能力层（Memory、Agent、Topic、Task 能力服务与附件上传服务）是授权点：方法只接收访问 context、目标 Workspace 与业务参数，先经操作授权者授权，再用返回的 `IdentityScope` 构造领域对象并调用 Patchouli 公开路由，不向下传 context。能力层的读取分两族：管理读取（Memory 与 Profile 的管理 get/list，绑定 `management.memory`）经 Patchouli 公开路由直接读取；actor 可见读取（Memory 点读、alias 读取、语义检索、Profile 读取）经 workspace 读取视图读取并在交付前逐次授权，目前没有生产调用方，HTTP 路由使用管理方法。

### 1.1 Transport / Router 边界

FastAPI router 是 transport adapter，而不是另一层业务编排者。它的职责应收敛为：解析和校验 request、取得窄化的应用服务依赖、调用一个用例、把结果转换成 HTTP/SSE response，并在 transport 边界映射状态码或公开错误。

Router 不得直接访问 `HiveMemorySystem.patchouli`、Alice/Gateway runtime、Store、scheduler 或内部 bus，也不得重新拼装 prepare/run/finalize。`server/deps.py` 应暴露面向用例的 service dependency，而不是把整个 System 当作万能 facade 注入每个入口。

这一边界同样约束应用服务自身：System 是 composition root，不是 God Facade；应用服务可以跨公开 route 编排一个用例，却不能逐步吸收子系统算法、存储访问和所有运行时状态，成为新的万能 runtime。Transport、用例编排和领域所有权保持分离，才允许 HTTP、SSE、CLI 与未来 adapter 共享同一业务语义。

## 2. 服务分工

| 服务（位置） | 当前职责 | 主要依赖 |
|:---|:---|:---|
| `TaskProcessService`（`workspace.process`） | 任务请求的唯一注册入口：注册时完成两阶段认证并创建进程，返回不透明的进程句柄；运行已注册的进程（主动非流式/流式 chat、command short-circuit、四阶段编排与阶段授权）；CPU 分配（Profile 解析、附件租借、附件与记忆编译、组装 `CPUInputManifest`）；进程表登记每个任务进程，提供取消、状态查询与关闭 | 认证网关、操作授权者；Gateway、Patchouli public routes；CPU 端口（组合根注入，当前为 Alice 的实现）；WorkspaceAsset reader 端口；MemoryCompiler、AttachmentCompiler；RuntimeEventPublisher（经 `TaskProcessEventEmitter` 投影 `chat.run.*`） |
| `PassiveIngressService`（`system.application`） | 外部事件摄入、idle maintenance 注册、显式 flush、shutdown drain | Passive Ingressor、Gateway/Patchouli public routes、scheduler |
| `MemoryApplicationService`（`workspace.capability`） | Memory 管理 CRUD、feedback 和查询参数转换（`management.memory`）；actor 可见读取经读取视图（`resource.read` / `resource.search`） | 操作授权者；Patchouli memory routes；workspace 读取视图 |
| `MemoryTaskApplicationService`（`workspace.capability`） | 查询/取消 Patchouli 拥有的 memory generation task（观察 `task.observe`、取消 `management.task`） | 操作授权者；Patchouli task routes |
| `AgentApplicationService`（`workspace.capability`） | 构造 Agent Profile atom 并调用 Patchouli profile routes（管理创建与列表绑定 `management.memory`）；Profile 读取经读取视图（`profile.read`） | 操作授权者；Patchouli profile routes；workspace 读取视图 |
| `TopicApplicationService`（`workspace.capability`） | 活跃话题列表、手动 settle、evict（均绑定 `management.topic`） | 操作授权者；Patchouli topic routes |
| `SystemReadinessService`（`system.application`） | 模型 warmup、ready 和简短 readiness 状态 | Patchouli readiness routes |
| `WorkspaceAssetApplicationService`（`workspace.capability`） | 编排 Chat 附件的接收、原子注册和请求内解析，保留首次创建/重放回执语义；在接收与注册副作用之前授权（`management.asset`），注册只使用授权返回的 scope | workspace 的 WorkspaceAssetStore（命令端口）、附件接收函数、AttachmentParseService、上传串行门、操作授权者；链路事实见[Chat 附件链路](./attachments.md) |

这些服务的“拥有”只指顶层用例入口，不改变表中后端子系统的状态所有权。例如 `MemoryTaskApplicationService` 可以取消任务，但任务生命周期仍由 Patchouli 负责。

附件上传同样遵守这一边界：应用层只有 `upload_asset()` 用例入口，文件名规则、受限读取和 SHA-256 计算由附件接收函数实现，解析接纳、来源校验和失败/取消收尾由 `AttachmentParseService` 实现。应用层决定串行门覆盖整个请求，并保留 Store 返回的 `created` 标记；它不实现解析算法，也不维护另一份资产状态。完整职责与错误语义统一维护在[Chat 附件链路](./attachments.md)。上传方法只接收访问 context 与目标 Workspace，调用方不能另传 scope，注册与解析交接都使用授权返回的 scope。

## 3. 主动 chat：唯一编排者

`TaskProcessService` 是任务请求的唯一注册入口（当前只有主动 chat 经它进入），注册与运行分开：`register_process()` 完成认证并创建进程、返回进程句柄（第 4 节），`run_process(handle, stream=…)` 运行已注册的进程，`stream`（默认 `True`）只决定交付形态；注册参数 `message` 是交给 Gateway 分析的指令文本。每次注册创建一个 `TaskProcess`（`workspace/process/task_process.py`），它既是本进程的状态容器（进程记录、工作集、事件投影），也承载唯一的编排骨架 `run()` 与唯一的关闭流程 `close()`。骨架只产出类型化的阶段产出（`workspace/process/outputs.py`），流式交付把它们投影为 SSE 事件，非流式交付只取终态产出；两种形态共用同一条阶段顺序、同一组取消响应点与同一个关闭流程。它们在执行上只有两处差异：CPU 以流式还是非流式产出，finalize 之后的话题池读取只服务于流式 `done` 事件。编排途中的异常在两种形态下都先记录失败终态、完成关闭，再由流式交付翻译为 `error` 事件（Workspace 领域错误携带安全文案与错误码，其余统一为系统错误），或由非流式交付原样上抛。

CPU 分配（Profile 解析、附件租借与编译、记忆编译与清单组装）由 `CPUAllocator`（`workspace/process/allocation.py`）完成；`chat.run.*` 观测事件由领域 emitter `TaskProcessEventEmitter`（`workspace/process/events.py`）投影。

Actor 执行经 CPU 端口完成：`TaskProcessService` 由组合根注入一个 `CPUPort`（`workspace.contracts`，当前唯一的实现是 Alice 的 `AliceCPU`），进程只调用端口的 `execute(manifest, *, generation_options, stream)`，不出现任何具体 CPU 的路由名或结果类型。端口返回的异步生成器先产出交互事件（流式时），最后产出唯一的终态结果 `CPUExecutionResult`；Actor 阶段只有一个拉取循环，流式与非流式共用，每次拉取都可被停止请求中断。进程拿到终态结果后立即关闭这条输出流，让 CPU 在 finalize 之前释放自己的资源；关闭流程中的再次关闭只作兜底。端口与结果的契约见[子系统公共契约](../contracts/subsystem-contracts.md#4-cpu-端口与-alice-实现)第 4 节。端口定义在 workspace、由 CPU 实现，是为了让 CPU 可以替换而不改动进程：Alice 之外的 CPU（测试中的 `ScriptedCPU`）能跑完整个任务进程。

### 3.1 非流式链路

```text
TaskProcessService.register_process()：认证、创建进程、登记进程表，返回进程句柄
TaskProcessService.run_process(handle, stream=False) -> TaskProcess.run()
  -> Gateway public process (ACTIVE_CHAT)             阶段授权 resource.read
  -> command: return command outcome
  -> decision: 解析 Agent Profile（Patchouli get_agent_profile）  profile.read
  -> Patchouli prepare_agent_run（Topic 与检索；interaction_id 取 process_id 值）  resource.search
  -> CPU 分配：附件租借与编译（asset.acquire）、记忆编译、组装 CPUInputManifest
  -> 进入 Actor 前检查 interaction.submit
  -> Actor 执行：CPU 端口 execute（CPUInputManifest，非流式只产出终态结果）
  -> completed: 封口交互记录（InteractionPayload）-> Patchouli finalize_agent_run
  -> cancelled/failed: Patchouli cleanup_prepared_agent_run（不做阶段授权）
  -> 关闭：TaskProcess.close() 释放附件租借；注册入口撤销 context、从进程表注销
```

每次阶段调用前，进程以注册时通过认证的 Workspace 为目标调用 `authorize_operation`，把返回的 `IdentityScope` 传给对应路由；某一阶段缺少 operation 时在该阶段失败，不产生后续副作用。`interaction.submit` 提前到进入 Actor 执行前检查，避免 CPU 执行完才在结算被拒。CPU 输入清单中的 `IdentityScope` 由操作授权者的过渡方法 `cpu_execution_identity` 组装（[Workspace 架构](../architecture/workspace.md)第 4.4 节）。

CPU 分配由进程完成：Patchouli prepare 只返回话题准备结果与未编译的检索原子（`PreparedAgentRun`）；进程用共享引擎 `MemoryCompiler` 把检索结果编译为 `RETRIEVAL_CONTEXT` 文本，用 `AttachmentCompiler` 编译附件并得出实际使用的附件，再把两者与已解析的 Profile 一起组装为输入清单经 CPU 端口交给 CPU。编译放在进程而不是执行者一侧，是为了让不同执行者共用同一份编译结果，而不必各自调用引擎。

Agent Profile 属于 CPU 分配，但目前在 prepare 之前解析：prepare 可能按路由决定新建 Topic，话题池已满时还会先按 LRU 结算一个已有话题，Profile 缺失的请求应在这些副作用发生之前失败。Profile 暂时经 Patchouli 公开路由解析，不经能力层。

交互记录也由进程封口：Actor 正常完成、进入 finalize 之后，进程以入口消息、Gateway 决定、Actor 的执行结果与实际使用的附件组装 `InteractionPayload`（`workspace/process/sealing.py`），MTP 轨迹由 core 的归约器从轮次事件得到；finalize 原样提交。这与被动链路由提交方（turn buffer）封口一致，Patchouli 不需要读懂执行者的运行结果。字段来源见[子系统公共契约](../contracts/subsystem-contracts.md#32-finalizeagentrun) 3.2。

本进程的 prepare 结果、附件租借与附件编译得出的实际使用引用由进程工作集（`ProcessWorkingSet`，`workspace/process/working_set.py`）持有；输入清单在 CPU 分配后直接交给 CPU 端口，不留在工作集中。进程无论以何种结局结束都经注册入口关闭：`TaskProcess.close()` 先同步释放全部租借，再关闭 CPU 输出流（若尚未关闭）、请求 cleanup；注册入口随后撤销进程绑定的访问 context 并从进程表注销，这两步放在内层 `finally`，因此即使这些 `await` 被取消，租借、context 与进程登记也不会泄漏。prepare 返回的结果先写入工作集再做身份校验，校验失败时仍会交回 cleanup，以补偿 prepare 可能预建的 Topic。

Gateway 返回 command outcome 时，结果只携带命令解析结果，服务立即完成本次 run，不进入 topic、retrieval、Actor 执行或主动记忆生成。命令终态由进程按解析状态产生（`workspace/process/command_terminal.py`）：解析成功时命令暂不可用（`not_implemented`、`command.unavailable`），解析失败时拒绝（`rejected`、`command.parse.<状态>`），均不带客户端动作；进程仍以 completed 结束。这是控制消息与普通对话之间的语义隔离，不是一个性能优化开关。

普通决策进入 prepare 后，Profile 解析、prepare 与 CPU 分配期间收到的停止请求会被进程记录；分配完成、进入 Actor 执行前统一检查一次，命中即跳过 Actor 执行和 finalize，返回 cancelled 结果，并在关闭流程中释放租借、请求 cleanup。只有在 CPU 执行结果的 `status == completed` 且进程未取消时才允许进入 finalize；finalize 成功后才将 prepared 标记为已接管，不再 cleanup。

### 3.2 流式链路

`run_process(handle)`（默认 `stream=True`）运行同一个骨架，CPU 以流式产出，阶段产出按以下顺序投影为 transport 事件：

```text
process_id
  -> Gateway decision / command
  -> command_result + done
  -> 或 Profile 解析、prepare、CPU 分配
       topic_info
       memory_refs
       CPU 交互事件（原样转交）
       run_status(finalizing)
       done(completed + memory_task_ids + pool_topics)
```

流式执行必须收到 CPU 的终态结果才能结束 Actor 阶段。若输出流在没有终态结果时结束，服务按协议错误处理；`done` 只携带交付方使用的执行结果字段（终态、最终回复、模型名）与进程字段：轮次事件与物化任务只用于封口交互记录，不下发；也不含执行者专属的统计。客户端提前关闭时，SSE adapter 以句柄形式的 `cancel_process(handle, reason="client_disconnected")` 取消当前进程，先取消并 join 自己创建的 stream-pull task，再经注册入口关闭进程；生成器从未开始迭代时（例如关停信号在注册期间到达），SSE 响应在自身收尾时兜底关闭进程。进程的关闭流程随后以 `stream_closed` 收口尚未发布终态的进程，释放附件租借、关闭 CPU 输出流，并对尚未 finalize 的 prepared run 执行 cleanup。`topic_info` 与 `memory_refs` 由进程从 prepare 结果与输入清单推导，只在 CPU 分配成功后发出。

流式 `done`、`command_result` 和 `error` 是 transport 可消费的事件，不是新的跨子系统业务契约；它们的来源和调用顺序仍由本服务和 Contracts 共同约束。

## 4. 注册入口、进程表与取消

**注册**（在流式响应开始之前完成）：

1. 拒绝保留的 `system` 作为任务请求的 actor：任务进程必须由具体 Agent 执行；
2. 经认证网关完成两阶段认证，运行绑定为本进程的 `process_id`（由 server 入口在进入编排服务前生成并冻结）；认证失败直接抛出，HTTP 入口据此返回 403，不创建、不登记进程；
3. 创建进程记录（写入访问 context）与 `TaskProcess`，以通过认证的声明绑定 `chat.run.*` 的观测标签（`workspace_id`、`agent_id`），登记到进程表；认证通过后、登记完成前失败时撤销已签发的 context；
4. 签发并返回不透明的进程句柄。

**进程表**（`ProcessTable`，`workspace/process/table.py`）是唯一的进程注册表，也是 workspace 的进程内共享设施：以 `process_id` 为键登记 `TaskProcess`，注销只移除同一个进程对象；进程记录作为进程的控制面经进程取得，不单独登记。进程记录（`ProcessRecord`）有：

- `process_id`：进程唯一标识；
- `access`：注册时签发、绑定本进程的访问 context，只交给操作授权者使用，不保存 actor、驻留 Workspace 等身份字段；
- `events`：注册时绑定观测标签的事件发布器；
- `phase`：`created/gateway/prepare/actor/finalize/terminal`（`actor` 即 Actor 执行阶段）；
- `outcome`：`running/stop_requested/cancelled/completed/failed`；
- 首次接受的 `stop_reason`；
- 仅在 Gateway 或 Actor 执行阶段存在的 `active_task` 身份引用。

进程表不保存 `Event`、Token 或 waiter。

**进程句柄**（`ProcessHandle`）是入口 adapter 与已注册进程之间唯一的引用：只暴露 `process_id`，不暴露进程记录、其中的 context 或 `TaskProcess`。句柄只由注册入口签发，按对象身份判定有效：句柄私下记着它对应的进程，注册入口解析时要求进程表中登记的正是这一个。按 `process_id` 重新构造的对象、进程关闭后的旧句柄都不是有效句柄——运行时以 `process_handle_unknown` 拒绝，取消时返回 `not_found` 且不发布事件，关闭是空操作。持有有效句柄即为该进程生命周期的所有者。

**取消**：`cancel_process` 是唯一的取消方法，取消的依据作为参数：

| 调用形式 | 依据 | 授权 | 找不到或无权时 |
|:---|:---|:---|:---|
| `cancel_process(handle, reason=…)` | 进程的所有者，例如 chat 路由在客户端断开时 | 不经进程控制授权；不接受另传的 access | 返回 `not_found`，不发布事件 |
| `cancel_process(process_id, access=…, reason=…)` | 以请求级 context 发起的控制请求（`POST /chat/stop`） | 经进程控制授权：请求方与进程记录驻留在同一 owner 与 Workspace | 返回 `not_found`，发布带请求方观测标签的事件 |

两种形式解析出进程后共用同一份实现：同步调用 `request_stop()`，首次 stop 固定 reason，重复 stop 返回同一判定且不重复取消 task，事件使用进程注册时绑定的观测标签。取消不新增 operation，不是 Agent action；跨 user/workspace 的取消得到 `not_found`，不泄露进程是否存在。状态查询 `process_status(process_id, access=…)` 使用同一进程控制授权，不可控与不存在统一为 `None`。

取消响应点只有两处：只有 Gateway 与 Actor 执行两个阶段会取消当前 `active_task`，因为这两个阶段在前台调用 LLM；进入 Actor 执行前统一检查一次停止请求（CPU 分配之后的检查与 `_run_interruptible` 的入口检查），Profile 解析、prepare 与 CPU 分配期间的停止请求都在这些工作照常完成后于此生效；阶段交接窗口只记录 stop，下一阶段不会启动；Finalize 与 Terminal 拒绝 stop。外部取消的入口是 `POST /chat/stop`（必须携带明确的 `process_id`）与 chat 路由内的客户端断开处理。

用户 stop 在编排骨架内被翻译为私有 `_ProcessCancelled` 分支，下游只传播原生 `asyncio.CancelledError`。`_run_interruptible()` 同时区分“stop 取消 child task”和“进程 owner 被 ASGI/shutdown 取消”，后者必须原样向上传播。资源所有者在 unwind 中关闭自己创建的 stream、runner 与 provider response；收尾异常只记录日志，不能替换正在传播的 `CancelledError`。

**关闭**：`close_process(handle)` 是注册入口的关闭路径，幂等；交付路径结束时注册入口直接关闭进程。进程关闭后从进程表注销、绑定的 context 被撤销，此后取消与状态查询都按 `not_found` 处理。

当前进程表是进程内短期控制状态，不是可恢复的长期工作记录。进程重启后不能据此恢复进程；System 停止流程不排空进程表。用户可见长期任务与后台 Agent workflow 当前未立项，需在真实负载和执行能力成立后独立设计。

## 5. 管理类应用服务

### 5.1 Memory 与 Profile

`MemoryApplicationService` 用授权返回的 `IdentityScope` 将 API 字段构造成 `MemoryAtom`，再通过 Patchouli public route 执行 create/list/get/update/delete/feedback。列表查询会显式排除 `AGENT_PROFILE`，避免普通记忆管理 API 与 Profile 资产混为一类；不存在的 get/update 被翻译为 `MemoryNotFoundError`。管理读写按 owner-management 语义执行：在 Workspace ownership hard boundary 之内访问该 Workspace 的全部 Memory，不执行 Agent 级 `MemoryAccessPolicy` 可见性过滤（见[Workspace 架构](../architecture/workspace.md)与[MemoryLibrary](../patchouli/memory-library.md)）。

`AgentApplicationService` 用同样的 atom 结构创建 `AGENT_PROFILE`，Profile 的实际持久化和可见性仍由 Patchouli 负责。这里的 `agent_config` 是资产内容，不是 System 直接解释的运行时权限。

### 5.2 Task、Topic 与 Readiness

`MemoryTaskApplicationService` 授权后转发 task list/get/cancel（观察绑定 `task.observe`、取消绑定 `management.task`），任务归属由 Patchouli 按 `IdentityScope` 投影判断；它不从 Patchouli task 对象推导第二套状态机。

`TopicApplicationService` 提供活跃话题列表、手动 settle 和 evict 入口，三者都绑定 `management.topic`（管理用例的 actor 为保留 `system`，它的白名单不含 `resource.read`）；授权后以返回的 `IdentityScope` 通过 Patchouli topic route 交给 Topic 所有者，不根据 ID 或缓存自行判断 Workspace 可见性和生命周期。`SystemReadinessService` 提供模型 warmup、ready 查询和 `ready/warming_up` 摘要，不参与 Workspace 资源授权。

## 6. 错误、观测与 cleanup

- 业务拒绝或资源不存在应使用服务定义的稳定结果/异常；
- Gateway 与 Actor 执行（CPU）的 task cancellation 不应被包装成普通 success 或 failed；Gateway timeout 仍按自己的 deadline/fallback 契约处理；
- `chat.run.*` 事件由 `TaskProcessEventEmitter` 从进程记录投影阶段与终态，经 `RuntimeEventPublisher` best-effort 发布：发布失败不改变返回值，失败事件只携带 Workspace 领域错误码或固定摘要，不写入异常正文；
- cleanup 是 prepare 失败后的有限补偿，不是跨子系统 rollback；
- 应用服务捕获的错误应保留原始因果，不能通过“空列表/空话题”掩盖 route 缺失或契约违约。

## 7. 应用层矛盾检查

新增应用服务或新入口时，检查：

1. 是否复制了 `TaskProcessService` 的 prepare/run/finalize 顺序？
2. 是否绕过 `GlobalSystemBus` 直接持有子系统 Runtime？
3. 是否把 command outcome 继续送进普通 chat？
4. 是否在取消后仍调用 finalize，或 finalize 失败后忘记 cleanup？
5. 是否把 memory task、topic 或 run registry 的临时状态写成另一个权威来源？
6. 是否把内部 RuntimeEvent、fallback 原因或 trace 当成公开 API 字段？
7. Router 是否只处理 transport，还是直接访问了 System/子系统内部对象或复制了用例顺序？
8. 新应用服务是否开始拥有本应属于子系统的算法、实体或长期状态？

## 8. 验证入口

- `tests/unit/workspace/process/test_gateway_chat_flow.py`
- `tests/unit/workspace/process/test_chat_run_control_contract.py`
- `tests/unit/workspace/process/test_cancel_hardening.py`
- `tests/unit/workspace/process/test_cpu_allocation.py`（CPU 分配、租借释放与关闭顺序）
- `tests/unit/workspace/process/test_command_terminal.py`（命令解析结果转换为命令终态）
- `tests/unit/workspace/process/test_cpu_port.py`（经 CPU 端口执行：测试 CPU 跑通任务进程、自报结局、停止、缺终态与输出流关闭）
- `tests/unit/workspace/process/test_seal_interaction.py`（交互记录封口）
- `tests/unit/workspace/process/test_process_events.py`（`chat.run.*` 投影与 best-effort 边界）
- `tests/unit/server/routers/test_chat.py`（注册后运行、客户端断开取消、流从未开始时的关闭）
- `tests/unit/system/application/test_api_services.py`
- `tests/unit/system/application/test_identity_entry_guards.py`（服务签名/身份入口守卫）
- `tests/unit/workspace/capability/`（Memory、MemoryTask、Agent Profile、Topic 能力服务）
- `tests/unit/system/application/test_readiness_service.py`
- `tests/integration/workspace/capability/test_assets.py`、`tests/integration/system/application/test_workspace_asset_parsing.py`（真实附件服务、解析服务与 Store 的上传协作）
- `tests/integration/workspace/test_published_registration_chain.py`（随仓库发布的登记文件驱动的管理路由、chat 与取消链路）
