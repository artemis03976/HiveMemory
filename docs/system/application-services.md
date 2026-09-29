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
last_reviewed: 2026-09-28
---

# 应用服务

应用服务是 transport 与子系统之间的用例层。它们由 System 组合根装配、经 `HiveMemorySystem` 门面交给入口，但按归属分布在三处：

- 任务进程表与 chat 任务进程编排：`workspace.process`（`ProcessTable`/`ProcessRecord` 进程表与 `TaskProcessService` 四阶段骨架）；
- 资源能力层：`workspace.capability`（Memory、MemoryTask、Agent Profile、Topic、WorkspaceAsset 服务）；
- 系统级服务：`system.application`（被动摄入与就绪检查）。

本文统一描述它们共同遵守的用例层规则。它们回答“这次请求应该按什么顺序跨边界运行”，而不回答“记忆如何检索”“Agent 如何生成”或“Gateway 如何分析”。

这一层存在，是因为 HTTP、SSE、CLI 和未来外部 adapter 都需要共享同一条主动 chat、取消、被动摄入和管理 API 链路。若每个 router 自己拼装 Gateway、Patchouli 和 Alice，就会再次出现多套 prepare/finalize、错误处理和取消语义。

## 1. 共同边界

应用服务的共同规则是：

1. 通过 `GlobalSystemBus` 请求公开 route；
2. 在边界处接受 server 冻结的 `IdentityScope`、请求模型或面向 API 的结果；
3. 不持有另一个子系统的 Runtime、Service、Controller 或 local bus；
4. 不把内部 execution state、fallback 原因和观测事件直接返回给外部客户端；
5. 对取消、失败和 cleanup 保持与 Contracts 一致的终态。

应用服务公共方法只接受 `identity_scope: IdentityScope` 唯一入口，不接受裸 `user_id`（由签名守卫测试约束）。用户导向身份选择（`user_id + workspace_id`，Agent action 附加 `agent_id`）由 `server/deps.py resolve_request_identity_scope` 在 server 边界一次性校验并冻结为不可变 `IdentityScope`，随后沿 route 和领域 payload 传递；应用服务不再解析身份，也不得再次执行默认解析。非 Agent action 的 scope 由 server 注入保留 `system` actor，只标记"没有具体 Agent 作为操作来源主体"。后台 task、retry 和 finalize 不重新读取进程当前 Workspace；它们使用自身 DTO 中保存的 scope，在最终访问 Workspace-owned 资源时由领域所有者校验。应用服务不会因此拥有 Workspace 资源，也不会为共享 runtime 创建按 Workspace 分区的状态。

应用服务可以保存一次用例的短期控制状态，例如任务进程编排的进程表，但不能保存 Patchouli 的长期记忆状态或 Gateway 的请求级 workflow state。

Memory、Agent、Topic、Task 能力服务和附件上传服务还接收统一认证网关签发的 `WorkspaceAccessContext`（`access` 参数）。管理方法（Memory CRUD/feedback、Profile 管理、Topic 管理、Task 观察/取消）将 access **原样透传**给 Patchouli 公共路由：本层不解释、不裁剪 access，也不以 DTO scope 覆盖可信坐标，行为授权在 Patchouli application 落实；能力层的 actor 可见读取方法（Memory 点读、alias 读取、语义检索、Profile 读取）则在本层按 operation 授权后经 workspace 读取视图读取，Patchouli backing 只校验 context 有效性（见[Workspace 架构](../architecture/workspace.md)第 4 节）。目前没有生产入口调用这些 actor 可见读取方法，HTTP 路由使用管理方法。`access` 缺省时依赖下游冻结的迁移期兼容分支（裸 `IdentityScope` 受信适配），兼容窗口由 A6 完成生产消费者切换后关闭。

### 1.1 Transport / Router 边界

FastAPI router 是 transport adapter，而不是另一层业务编排者。它的职责应收敛为：解析和校验 request、取得窄化的应用服务依赖、调用一个用例、把结果转换成 HTTP/SSE response，并在 transport 边界映射状态码或公开错误。

Router 不得直接访问 `HiveMemorySystem.patchouli`、Alice/Gateway runtime、Store、scheduler 或内部 bus，也不得重新拼装 prepare/run/finalize。`server/deps.py` 应暴露面向用例的 service dependency，而不是把整个 System 当作万能 facade 注入每个入口。

这一边界同样约束应用服务自身：System 是 composition root，不是 God Facade；应用服务可以跨公开 route 编排一个用例，却不能逐步吸收子系统算法、存储访问和所有运行时状态，成为新的万能 runtime。Transport、用例编排和领域所有权保持分离，才允许 HTTP、SSE、CLI 与未来 adapter 共享同一业务语义。

## 2. 服务分工

| 服务（位置） | 当前职责 | 主要依赖 |
|:---|:---|:---|
| `TaskProcessService`（`workspace.process`） | 主动非流式/流式 chat、command short-circuit、取消和 prepare/run/finalize 四阶段编排；进程表（`ProcessTable`）登记每个任务进程并以 `process_id` 暴露 stop 控制面 | Gateway、Patchouli、Alice public routes；RuntimeEventSink |
| `PassiveIngressService`（`system.application`） | 外部事件摄入、idle maintenance 注册、显式 flush、shutdown drain | Passive Ingressor、Gateway/Patchouli public routes、scheduler |
| `MemoryApplicationService`（`workspace.capability`） | Memory CRUD、feedback 和查询参数转换，透传访问上下文；actor 可见读取经读取视图 | Patchouli memory routes；workspace 读取视图 |
| `MemoryTaskApplicationService`（`workspace.capability`） | 查询/取消 Patchouli 拥有的 memory generation task；透传观察/取消访问上下文 | Patchouli task routes |
| `AgentApplicationService`（`workspace.capability`） | 构造 Agent Profile atom 并调用 Patchouli profile routes；Profile 读取经读取视图 | Patchouli profile routes；workspace 读取视图 |
| `TopicApplicationService`（`workspace.capability`） | 活跃话题查询、手动 settle、evict；透传读取/Topic 管理访问上下文 | Patchouli topic routes |
| `SystemReadinessService`（`system.application`） | 模型 warmup、ready 和简短 readiness 状态 | Patchouli readiness routes |
| `WorkspaceAssetApplicationService`（`workspace.capability`） | 编排 Chat 附件的接收、原子注册和请求内解析，保留首次创建/重放回执语义；携带 access 时先经共享行为检查（`management.asset`） | workspace 的 WorkspaceAssetStore（命令端口）、附件接收函数、AttachmentParseService、上传串行门、Workspace 行为检查；链路事实见[Chat 附件链路](./attachments.md) |

这些服务的“拥有”只指顶层用例入口，不改变表中后端子系统的状态所有权。例如 `MemoryTaskApplicationService` 可以取消任务，但任务生命周期仍由 Patchouli 负责。

附件上传同样遵守这一边界：应用层只有 `upload_asset()` 用例入口，文件名规则、受限读取和 SHA-256 计算由附件接收函数实现，解析接纳、来源校验和失败/取消收尾由 `AttachmentParseService` 实现。应用层决定串行门覆盖整个请求，并保留 Store 返回的 `created` 标记；它不实现解析算法，也不维护另一份资产状态。完整职责与错误语义统一维护在[Chat 附件链路](./attachments.md)。已知缺陷：带 access 上传时行为权限与传入 scope 缺少一致性校验（详见[Todo：WorkspaceAsset 上传的认证上下文与 scope 不一致](../todo/workspace-asset-upload-access-scope-mismatch.md)），修复前带 access 的上传路径不视为已通过身份一致性验收。

## 3. 主动 chat：唯一编排者

### 3.1 非流式链路

```text
TaskProcessService.chat_scoped
  -> register ProcessRecord（进程表）
  -> Gateway public process (ACTIVE_CHAT)
  -> command: return command outcome
  -> decision: Patchouli prepare_agent_run（interaction_id 取 process_id 值）
  -> Alice run_agent（process_id）
  -> completed: Patchouli finalize_agent_run
  -> cancelled/failed: Patchouli cleanup_prepared_agent_run
  -> close 进程表
```

Gateway 返回 command outcome 时，服务立即完成本次 run，不进入 topic、retrieval、Alice 或主动记忆生成。这是控制消息与普通对话之间的语义隔离，不是一个性能优化开关。

普通决策进入 prepare 后，prepare 期间收到的停止请求会被进程表记录；prepare 正常返回后在进入 Alice 前统一检查一次，命中即跳过 Alice 和 finalize，返回 cancelled 结果并在 finally 中请求 cleanup。Alice 只有在 `AgentRunResult.status == completed` 且 run 未取消时才允许进入 finalize；finalize 成功后才将 prepared 标记为已接管，不再 cleanup。

### 3.2 流式链路

`chat_stream_scoped()` 保持同一条阶段顺序，但把阶段事实以事件交给 transport：

```text
process_id
  -> Gateway decision / command
  -> command_result + done
  -> 或 prepare
       topic_info
       memory_refs
       Alice stream events
       run_status(finalizing)
       done(completed + memory_task_ids + pool_topics)
```

流式生成必须收到 Alice 的最终 `done` 才能构造 `AgentRunResult`。若流在没有终态事件时结束，服务按协议错误处理；客户端提前关闭时，SSE adapter 以 `process_id` 请求停止当前进程，先取消并 join 自己创建的 stream-pull task，再关闭 Chat generator。`chat_stream_scoped()` 随后关闭 Alice 子流，并对尚未 finalize 的 prepared run 执行 cleanup。

流式 `done`、`command_result` 和 `error` 是 transport 可消费的事件，不是新的跨子系统业务契约；它们的来源和调用顺序仍由本服务和 Contracts 共同约束。

## 4. 进程表与取消

`ProcessTable`（`workspace/process/table.py`）是任务进程编排拥有的进程内控制表，也是 workspace 的进程内共享设施。每条进程记录（`ProcessRecord`）有：

- `process_id`：进程唯一标识，由 server 入口在进入编排服务前生成并冻结（Q-16）；
- `phase`：`created/gateway/prepare/alice/finalize/terminal`；
- `outcome`：`running/stop_requested/cancelled/completed/failed`；
- 首次接受的 `stop_reason`；
- 仅在 Gateway 或 Alice 阶段存在的 `active_task` 身份引用。

进程表不保存 `Event`、Token 或 waiter。`cancel()` 按 `process_id` 查找进程后同步调用 `request_stop()`：不存在返回 `not_found`；首次 stop 固定 reason；重复 stop 返回同一判定且不重复取消 task。取消是 owner/workspace 校验（同一用户、同一 Workspace），不是 Agent action。进程表只向同 owner/workspace 的控制请求暴露进程记录，跨 user/workspace 的取消得到 `not_found`。

取消响应点按 Q-15 收口：只有 Gateway 与 Actor 执行两个阶段会取消当前 `active_task`；进入 Alice 前统一检查一次停止请求（`_run_interruptible` 的入口检查与 prepare 之后的检查），prepare 期间的停止请求照常完成 prepare 后在此生效；阶段交接窗口只记录 stop，下一阶段不会启动；Finalize 与 Terminal 拒绝 stop。外部取消必须携带明确的 `process_id`，唯一入口是 `POST /chat/stop` 与路由内的客户端断开处理。

用户 stop 在编排服务内被翻译为私有 `_ProcessCancelled` 分支，下游只传播原生 `asyncio.CancelledError`。`_run_interruptible()` 同时区分“stop 取消 child task”和“进程 owner 被 ASGI/shutdown 取消”，后者必须原样向上传播。资源所有者在 unwind 中关闭自己创建的 stream、runner 与 provider response；收尾异常只记录日志，不能替换正在传播的 `CancelledError`。

当前进程表是进程内短期控制状态，不是可恢复的长期工作记录。进程重启后不能据此恢复进程；用户可见长期任务与后台 Agent workflow 当前未立项，需在真实负载和执行能力成立后独立设计。

## 5. 管理类应用服务

### 5.1 Memory 与 Profile

`MemoryApplicationService` 在显式 `IdentityScope` 中将 API 字段构造成 `MemoryAtom`，再通过 Patchouli public route 执行 create/list/get/update/delete/feedback。列表查询会显式排除 `AGENT_PROFILE`，避免普通记忆管理 API 与 Profile 资产混为一类；不存在的 get/update 被翻译为 `MemoryNotFoundError`。管理读写按 owner-management 语义执行：在 Workspace ownership hard boundary 之内访问该 Workspace 的全部 Memory，不执行 Agent 级 `MemoryAccessPolicy` 可见性过滤（见[Workspace 架构](../architecture/workspace.md)与[MemoryLibrary](../patchouli/memory-library.md)）。

`AgentApplicationService` 用同样的 atom 结构创建 `AGENT_PROFILE`，Profile 的实际持久化和可见性仍由 Patchouli 负责。这里的 `agent_config` 是资产内容，不是 System 直接解释的运行时权限。

### 5.2 Task、Topic 与 Readiness

`MemoryTaskApplicationService` 只转发 task list/get/cancel 并透传访问上下文（观察绑定 `task.observe`、取消绑定 `management.task`）；它不从 Patchouli task 对象推导第二套状态机。

`TopicApplicationService` 提供活跃话题、手动 settle 和 evict 入口；它只透传 server 冻结的 `IdentityScope`（管理用例的 actor 为保留 `system`），再通过 Patchouli topic route 交给 Topic 所有者，不根据 ID 或缓存自行判断 Workspace 可见性和生命周期。`SystemReadinessService` 提供模型 warmup、ready 查询和 `ready/warming_up` 摘要，不参与 Workspace 资源授权。

## 6. 错误、观测与 cleanup

- 业务拒绝或资源不存在应使用服务定义的稳定结果/异常；
- Gateway/Alice 的 task cancellation 不应被包装成普通 success 或 failed；Gateway timeout 仍按自己的 deadline/fallback 契约处理；
- `RuntimeEventSink` 只记录 chat run、状态和失败，不改变返回值；
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
- `tests/unit/server/routers/test_chat.py`
- `tests/unit/system/application/test_api_services.py`
- `tests/unit/system/application/test_identity_entry_guards.py`（服务签名/身份入口守卫）
- `tests/unit/workspace/capability/`（Memory、MemoryTask、Agent Profile、Topic 能力服务）
- `tests/unit/system/application/test_readiness_service.py`
- `tests/integration/workspace/capability/test_assets.py`、`tests/integration/system/application/test_workspace_asset_parsing.py`（真实附件服务、解析服务与 Store 的上传协作）
