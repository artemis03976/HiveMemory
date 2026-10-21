---
title: System Composition and Lifecycle
status: current
owner: system
scope: composition-root-and-lifecycle
code_paths:
  - src/hivememory/system/assembler.py
  - src/hivememory/system/system.py
  - src/hivememory/infrastructure/llm/
  - src/hivememory/core/contracts/subsystem.py
  - src/hivememory/config/app.py
  - src/hivememory/components/
  - src/hivememory/workspace/
  - src/hivememory/system/services/import_bus/
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/architecture/boundaries.md
related_docs:
  - docs/workspace/README.md
  - docs/system/import-bus.md
last_reviewed: 2026-10-21
---

# System 组合根与生命周期

`HiveMemorySystem` 是进程内顶层宿主与入口使用的门面。它把 Gateway、Patchouli、Alice、workspace 设施、应用服务和共享运行时放到同一条可推理的生命周期里，但不替任何子系统执行领域工作。

这个边界是项目从旧对象图中抽离出来的关键结果。若 HTTP router 或某个子系统自己装配依赖，就会出现多套总线、不同的取消状态和不一致的启停顺序；若 System 进一步拥有记忆算法、Gateway workflow 或 Agent loop，又会重新成为一个无法替换的全能总管。因此 System 只拥有组合关系、进程级状态与系统级能力，并位于依赖图顶点：它可以导入任何下层包完成装配，而除入口外没有包导入它。组件需要的配置由 System 从根配置（`config.app`）中取出对应配置段注入，组件不持有根配置。

## 1. 组合结构

`SystemAssembler.assemble()` 按七步生成中间产物，最后交给 `HiveMemorySystem`：

```text
HiveMemorySystem.build(config)
  -> SystemAssembler
       -> runtime bundle
             GlobalSystemBus / GlobalMaintenanceScheduler（components）
             RuntimeEventBus / NullRuntimeEventSink（components）
             InMemoryWorkspaceAssetStore（workspace.assets；进程级唯一）
             WorkspaceReadView（写入意图登记与读取视图；子系统构建前订阅全局事件）
       -> registries bundle
            ProviderRegistry
            ModelRegistry
       -> llm bundle（在注册表预解析组件 LLM 配置之后）
            LiteLLMService（infrastructure.llm；进程级唯一，Gateway / Patchouli / Alice 共享）
            Gateway 与 Librarian 的请求参数（component_llm_params）
       -> access-control bundle（两类登记分别从 configs/system_principals.yaml、configs/workspace_actors.yaml 装载）
            SystemActorAccessRegistry / WorkspaceActorAccessRegistry
            WorkspaceAuthenticator（workspace.authentication；准入、签发、主线程与子线程 context 的派生与撤销）
            WorkspaceOperationAuthorizer（workspace.authorization；操作授权，只依赖访问注册表）
            ActorAuthenticationGateway（workspace.authentication，注入 SystemPrincipalAuthenticator 与 WorkspaceAuthenticator；同时实现线程 context 生命周期端口）
            ProcessTable（workspace.process；在构建 actor API 之前创建，与注册入口共享）
            ProcessThreads（workspace.process；进程线程管理，注入进程表、认证网关（生命周期端口）与操作授权者）
       -> capabilities bundle
            actor API：ActorMemoryService / ActorAgentService（workspace.capability.actor；注入操作授权者与统一读取视图，ActorAgentService 另注入 ProcessThreads 作为子线程派生端口）
            WorkspaceOperationEntry（workspace.capability；注入 actor API、操作授权者与写入意图登记；只接受线程 context，按请求类型直接分派到 actor API）
       -> subsystem bundle
            GatewaySystem / PatchouliSystem / AliceSystem（各自只接收自己的配置段与共享 LLM 服务；Alice 注入操作入口）
       -> service bundle
            CPUAllocator（workspace.process；注入操作授权者、Profile 读取端口（读取视图的 ProfileResolver）、AssetStore 只读 reader 与 memory_compiler / attachment_compiler 配置段）
            TaskProcessRunner（workspace.process，四阶段骨架；注入全局总线、CPU 端口（AliceSystem.cpu_port）、CPUAllocator、ProcessThreads、写入意图登记、操作授权者与 Gateway 请求超时）
            TaskProcessService（workspace.process，注册入口；注入 TaskProcessRunner、共享的进程表、认证网关、操作授权者与 root RuntimeEventPublisher）
            ImportBusService（system.application；注入操作授权者与编排器 ImportBusIngestor，后者只取得全局总线、Gateway 请求时限与最近轮数、等待应用的时限与观测 sink）
            SystemReadinessService（system.application）
            root API：RootMemoryService / RootAgentProfileService / RootTopicService / RootMemoryTaskService / RootWorkspaceAssetService（workspace.capability.root；注入操作授权者，不接触读取视图）
```

七个 Bundle 是装配器的私有交接对象，不是公共协议。它们的作用是让依赖顺序显式可读：运行时先存在，注册表再解析模型配置，LLM 服务随后创建并取得组件请求参数，访问控制装载两类登记并组装认证网关与操作授权者，随后先构建 actor API 与操作入口，Alice 构建时即可注入入口；root API 在服务阶段构建，只交给门面；子系统共享全局基础设施，服务最后只拿到公共总线、认证网关或操作授权者、必要配置与 AssetStore 读取端口。认证网关注入任务进程的注册入口并经 `HiveMemorySystem.access_gateway` 暴露给 server，另以只含派生与撤销线程 context 的 `ThreadContextLifecycle` 端口注入进程线程管理 `ProcessThreads`；进程表在访问控制阶段创建，`ProcessThreads` 因此能在 actor API 之前构建，注入执行器（主线程）与 actor API（子线程派生），无需延迟绑定；操作授权者注入 root API（root 授权）、actor API（actor 操作授权）、操作入口（线程 context 校验与进程关联）、任务进程的执行器与 CPU 分配（阶段授权）、注册入口（进程控制授权），以及 Import Bus 服务（导入授权）；任务进程的编排依赖只交给执行器，注册入口只拿到执行器与生命周期依赖（[任务进程](../workspace/task-process.md)）；Gateway、Patchouli 与 Alice 不注入任何认证或授权对象（访问模型见[身份与授权](../workspace/identity-and-authorization.md#2-访问边界两阶段认证与两阶段授权)第 2 节）。

### 1.1 Runtime bundle

机制实现均位于 `components` 包，System 只创建实例并管理启停：

- `GlobalSystemBus`：跨子系统 public route 的 RPC/Pub/Sub 交接面；
- `GlobalMaintenanceScheduler`：在当前主 `asyncio` loop 调度维护任务；
- `RuntimeEventBus`：启用时保存有界观测事件和订阅队列；
- `NullRuntimeEventSink`：观测关闭时的无副作用替代实现；
- `InMemoryWorkspaceAssetStore`：workspace 的进程内 WorkspaceAsset working set，保存当前资产、representation、opaque ref 和 lease；不按 Workspace 复制实例，由组合根创建并在关闭时最后清理；
- `WorkspaceReadView`：共享写入意图登记与读取视图（完整原子缓存、Profile 解析缓存、失效代次，以及记忆与 Profile 两个读取器），L2 冷读经 `GlobalSystemBus` 调用 Patchouli backing 路由；Alice 全部资源读取（含 SEARCH、CALL Profile）经 actor API 使用该视图，CPU 分配的 Profile 读取在阶段授权后直接使用该视图的 `ProfileResolver`。组合根创建后立即订阅 canonical 变更与三种写入意图结算事件，早于子系统构建及请求接收。

写入意图登记、完整引用解析、原子与 Profile 读取缓存由 workspace 唯一持有；Alice 不持有自己的资源缓存或 resolver。CPUAllocator 只依赖 Profile 读取协议，runner 与 actor API 只经 `ProcessThreads` 派生与撤销线程 context，不导入认证一侧，能力层与任务进程仍互不导入；CPU 驱动绑定线程 context 形成操作提交函数，入口与 runner 使用同一份意图登记。WorkspaceAsset 命令端口由上传应用服务直接持有，附件上传不经过全局总线。

观测设施和业务总线在装配阶段就分开，是为了让 RuntimeEvent 的失败不会阻塞一次正常业务调用。

组合根另创建共享 `ConversationSessionService`，把相同实例注入 `RootSessionService`、任务注册入口与执行器；它只接收 `WorkspaceSessionConfig`。root 会话服务由门面供 HTTP 管理路由取得。

### 1.2 Registry bundle

`ProviderRegistry` 保存 provider 凭证与环境配置的合并结果；`ModelRegistry` 根据模型引用解析 Gateway/Librarian 的 LLM 配置。System 在子系统构建前完成这一步，解析结果随后转换为请求参数（第 1.3 节），使 Gateway 和 Patchouli 使用已经解析过的配置，而不是在请求路径里各自重新读取凭证。

注册表是 System 的全局资源，但它不拥有模型调用，也不决定某次请求使用哪条业务流程。具体运行时仍由 Gateway、Patchouli 或 Alice 按各自契约消费。

### 1.3 LLM bundle

System 只创建一个 `LiteLLMService`，它是 `infrastructure.llm.base.BaseLLMService` 的唯一实现，注入 Gateway、Patchouli 与 Alice。服务无状态：不读取配置、不持有默认模型或凭证，每次调用的模型、凭证与采样参数都由请求参数 `LLMRequestParams` 携带。服务不绑定配置是为了让共享实例不可能串用参数：按配置绑定实例再以单例共享时，后构造的组件会静默拿到先构造组件的模型与凭证；逐次携带参数从结构上排除这种情况。

`component_llm_params` 把注册表解析后的 `shared.llm.gateway` 与 `shared.llm.librarian` 各自转换为请求参数，只含模型、凭证、temperature 与 max_tokens；`top_p` 与停止序列不属于组件调参，由提供商取默认值。组件未配置模型时参数为 `None`：Gateway 按无 LLM 降级，Patchouli 的记忆抽取器以 No-Op 运行，Page Folding 回退为规则摘要。Alice 的请求参数逐帧由模型解析得到，见 [Agent Runtime](../alice/agent-runtime.md#4-模型选择与-mtp-生成器)。

服务接口只有一个入口 `complete(messages, params, *, stream=False)`：

- 完整调用返回 `LLMCompletion`（首个候选的文本、提供商报告的 finish_reason 与可缺省用量）；提供商没有返回文本时以 `ModelProviderError` 结束，空字符串仍是文本；
- 流式调用返回可关闭的异步生成器，首次拉取时才发出请求；中间项是非空文本增量，正常耗尽时最后一项携带 finish_reason。流在耗尽、被关闭或拉取时被取消都会关闭提供商响应，关闭失败只记录日志，不替换正在传播的异常或取消；
- 请求参数逐项传给 SDK，为 `None` 或为空的项不发送，凭证缺省时由 SDK 从环境变量读取；JSON 输出要求只在 SDK 明确报告不支持该参数时退回普通补全；
- SDK 异常只在实现中转换为 `core.errors` 的中立错误，完整调用、流式启动与流式拉取采用同一转换，程序错误与取消原样传播，见[错误模型](../contracts/error-model.md#46-模型调用错误)。

提供商 SDK 只在 `infrastructure/llm/` 中导入，接口模块 `base.py` 不依赖 SDK；除组合根外，引擎与子系统从 LLM 包只导入接口模块，不导入实现模块或包根。这三条由分层架构测试守护，使更换 SDK 或加入其他实现时，使用方不需要随之修改。

### 1.4 Subsystem bundle

三个子系统以平级宿主装配：

| 子系统 | System 负责的部分 | 子系统自己负责的部分 |
|:---|:---|:---|
| Gateway | 注入 `gateway` 配置段、共享 LLM 服务与 Gateway 请求参数、全局总线和观测 sink | GatewayRuntime、命令、上下文、workflow 与公共 process route |
| Patchouli | 注入 `patchouli` / `shared` / `scheduler` 配置段、共享 LLM 服务与 Librarian 请求参数、全局总线、维护调度器、观测 sink、AssetStore 只读 reader（供 Artifact promotion） | 记忆、话题、检索、感知、生成任务与 prepare/finalize |
| Alice | 注入 `alice` / `memory_compiler` 配置段、共享 LLM 服务、全局总线、模型解析端口（`ModelRegistry` 实现 `ModelResolver`）、操作入口（`WorkspaceOperationEntry` 实现 `OperationEntry`）和观测 sink | Agent run、frame、MTP 生成与执行、工具与 ACK alias 集合 |

System 不通过这些宿主的具体 Runtime 互相串联；跨边界链路由服务（chat 编排、Import Bus、能力层）通过 `GlobalSystemBus` 发起。

## 2. 启动顺序

当前 `HiveMemorySystem.start()` 的有效顺序是：

```text
Gateway.start
  -> Patchouli.start
  -> Alice.start
  -> GlobalMaintenanceScheduler.start
  -> SYSTEM_READY
```

顺序背后的理由是：先挂载三个子系统的 route，使应用服务有可用的交接面；再启动调度器。Import Bus 没有维护任务，也没有启动步骤，组合根构建完成后即可接收导入。`start()` 要求在运行中的 `asyncio` loop 内执行，因为维护调度器不创建隐藏 event loop。

重复 `start()` 不再重复挂载 route 或重启 scheduler，而是发布一次带 `already_started=true` 的生命周期观测并返回。启动失败会发布 `system.start_failed`，保留已完成步骤和第一个未完成步骤后继续抛出原异常。

当前实现没有在 `start()` 失败后自动逆序回滚已启动的子系统；调用方不能把失败事件误读为已经完成清理。这个限制需要在后续生命周期增强中单独处理，不通过隐式补偿掩盖。

## 3. 停止顺序

当前 `stop()` 的有效顺序是：

```text
ActorAuthenticationGateway.close
  -> GlobalMaintenanceScheduler.stop
  -> ImportBusService.shutdown（停止接收，等待在途导入结束，含等待应用的时间；不排空提交队列）
  -> TaskProcessService.shutdown（关闭未启动的进程，以 system_shutdown 停止已启动的进程并等待进程 task 结束，超时后硬取消；进程在自己的 task 中收尾并释放会话认领）
  -> ConversationSessionService.close_and_clear
  -> Alice.stop
  -> Patchouli.stop
  -> Gateway.stop
  -> WorkspaceReadView.close
  -> WorkspaceAssetStore.close_and_clear
  -> 撤销全部已签发的访问 context（ActorAuthenticationGateway.revoke_all_contexts）
  -> SYSTEM_STOPPED
```

认证网关最先关闭：此后拒绝新的认证，已签发的访问 context 照常可用，在途请求与任务进程仍能完成各自的授权；全部已签发 context 在最后一步撤销。操作授权者无状态，不需要关闭。先停调度器是为了阻止新的维护 tick；随后 Import Bus 等待在途导入结束：认证网关已经关闭，新的导入以 503 拒绝，门外等待的导入以 `shutting_down` 拒绝，正在处理的导入继续到应用确认，因为 Patchouli 的提交队列要到 Patchouli 停止时才排空（[Import Bus](./import-bus.md#61-关闭)）。全部任务进程关闭后清空会话，再停止 Alice；Alice 停止时卸载自己的 run route；此后 Patchouli 才会按自己的顺序 drain interaction submission、active finalize、Topic settlement/generation 和 memory-generation queue，避免消费者仍需反查 asset ref 时 Store 已经消失。Gateway 撤销后，System 关闭 workspace 运行时（停止新读、取消结算与 canonical 变更订阅、清理派生缓存，不触碰 canonical 数据），最后调用 `WorkspaceAssetStore.close_and_clear()`；Store 不调用 Patchouli controller 的 `wait_all`，也不查询 Topic 或 binding。

重复 `stop()` 会保持幂等：scheduler 已停止时不重复等待，未启动的系统仍会等待 Import Bus 的在途导入并发布 `already_stopped=true`。`system.stopped` 与 `system.stop_failed` 的 `import_bus_shutdown` 字段报告 Import Bus 开始关闭时的在途请求数（`waited_requests`）。任一步骤失败都会发布 `system.stop_failed`，记录已完成步骤、失败步骤、scheduler 状态和 Import Bus 的等待结果后抛出异常。

对于从未成功启动的 System，`stop()` 在关闭认证网关、等待 Import Bus 在途导入、关闭任务进程并清空会话后，即可关闭读取视图、清空 AssetStore、撤销全部访问 context 并返回，不会伪造 Alice、Patchouli 或 Gateway 已完成停止；正常已启动实例才执行上面列出的完整逆序。

## 4. 健康状态与公共入口

`HiveMemorySystem.health()` 汇总：

- 当前 System 是否处于 started 状态；
- Gateway、Patchouli、Alice 各自的 `health()`；
- Patchouli 模型是否 ready。

健康状态是观测和管理入口，不替代业务契约。模型尚未 ready 不等于所有 route 都不存在；反过来，健康返回 `ok` 也不保证一次具体检索或生成调用一定成功。

System 作为门面对外暴露服务属性和 registry/sink 查询，例如 `process_service`（workspace 任务进程编排）、`import_bus_service`（Import Bus）、`root_memory_service` 等 root API 服务（actor API 只由操作入口持有，不经门面暴露）、`access_gateway`、`runtime_events`、`model_registry`。这些属性方便 HTTP 或其他 adapter 注入依赖，但 adapter 仍应调用服务，不应从属性继续下钻到子系统 Runtime。

## 5. 生命周期不变量与矛盾检查

- 所有 public route 的挂载和撤销必须由对应子系统宿主完成；
- 服务（chat 编排、Import Bus、能力层）不直接持有 Gateway/Patchouli/Alice 实例；
- 除入口外没有包导入 `system`；组件只接收自己的配置段，不持有根配置；
- 维护任务必须在 scheduler 注册，不能由业务组件偷偷创建第二个 interval loop；
- `SYSTEM_READY` 只在所有启动步骤完成后发布，RuntimeEvent 失败不能改变这个判断；
- `SYSTEM_STOPPED` 的观测摘要不等于 submission 已跨进程持久化：`import_bus_shutdown` 只报告 Import Bus 等待的在途请求，交互提交队列是进程内队列，由 Patchouli 停止时排空；
- `WorkspaceAssetStore.close_and_clear()` 必须晚于 Patchouli drain；失败时不得发布伪装成正常完成的 `SYSTEM_STOPPED`；
- 全进程只存在一个 WorkspaceAssetStore 与一个 WorkspaceReadView；写入意图登记、完整引用解析与原子/Profile 读取缓存由 workspace 持有，结算订阅保留到 Patchouli drain 完成后才撤销；
- registry 解析失败、子系统启停失败和业务请求失败不能被统一降级成健康 `ok`。

评审新的组合代码时，优先检查是否出现第二个 GlobalSystemBus、应用层直连子系统 Runtime、启动失败后仍接受请求，或把 RuntimeEvent 当作控制信号的情况。

## 6. 验证入口

- `src/hivememory/system/assembler.py`
- `src/hivememory/system/system.py`
- `tests/unit/system/test_hivememory_system.py`
- `tests/unit/system/test_lifecycle.py`
- `tests/unit/system/test_assembler.py`（组件 LLM 请求参数各自独立）
- `tests/unit/architecture/test_package_layers.py`（含 LLM SDK 与实现模块的导入边界）
- `tests/unit/workspace/assets/test_store.py`
- `tests/integration/system/test_workspace_asset_runtime.py`（对象图与停止顺序）
- `tests/unit/system/contracts/test_contracts.py`
- `tests/integration/system/test_workspace_intent_subscription.py`（构建即订阅与停止即取消订阅）
