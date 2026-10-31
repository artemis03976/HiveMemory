---
title: System Application Services
status: current
owner: system
scope: application-use-case-layer-rules-transport-boundary-and-system-services
code_paths:
  - src/hivememory/system/application/
  - src/hivememory/server/deps.py
  - src/hivememory/server/routers/
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
related_docs:
  - docs/workspace/README.md
  - docs/workspace/task-process.md
  - docs/workspace/capability.md
  - docs/architecture/boundaries.md
  - docs/system/import-bus.md
last_reviewed: 2026-10-31
---

# 应用服务

应用服务是 transport 与子系统之间的用例层。它们由 System 组合根装配、经 `HiveMemorySystem` 门面交给入口，但按归属分布在三处：

- 任务进程表与 chat 任务进程编排：`workspace.process`，见[任务进程](../workspace/task-process.md)；
- 资源能力层：`workspace.capability` 的 root API（只供 `system`）与 actor API（只经操作入口），门面只向入口暴露 root API，见[能力层](../workspace/capability.md)；
- 系统级服务：`system.application`（Import Bus 与就绪检查），见本文第 3 节。

本文统一描述它们共同遵守的用例层规则。它们回答“这次请求应该按什么顺序跨边界运行”，而不回答“记忆如何检索”“Agent 如何生成”或“Gateway 如何分析”。

这一层存在，是因为 HTTP、SSE、CLI 和未来外部 adapter 都需要共享同一条主动 chat、取消、交互导入和管理 API 链路。若每个 router 自己拼装 Gateway、Patchouli 和 Alice，就会再次出现多套 prepare/finalize、错误处理和取消语义。

## 1. 共同边界

应用服务的共同规则是：

1. 跨资源 owner 时通过 `GlobalSystemBus` 请求公开 route；执行者的资源操作通过绑定线程 context 的提交函数进入 workspace 能力入口；
2. 在边界处接受访问 context 与目标 Workspace、请求模型或面向 API 的结果，不接受调用方另行组装的 `IdentityScope`；
3. 不持有另一个子系统的 Runtime、Service、Controller 或 local bus；
4. 不把内部 execution state、fallback 原因和观测事件直接返回给外部客户端；
5. 对取消、失败和 cleanup 保持与 Contracts 一致的终态。

身份与访问的交接遵循两阶段认证与两阶段授权（[身份与授权](../workspace/identity-and-authorization.md#2-访问边界两阶段认证与两阶段授权)）：server 在 `server/deps.py` 用 `resolve_request_identity_claims` 把用户导向身份选择（`user_id + workspace_id`，Agent action 附加 `agent_id`）解析为身份声明，交给统一认证网关取得访问 context：chat 由任务进程的注册入口认证，管理操作、取消与交互导入取得绑定本次请求的请求级 context。应用服务不解析身份，也不执行默认解析；Agent action（Chat 与交互导入）必须声明具体 Agent，非 Agent action 的声明使用保留 `system` actor，标记“没有具体 Agent 作为操作来源主体”，它也是 root API 唯一接受的 actor。`IdentityScope` 只用于一次操作到公开资源边界的调用；Patchouli 在该边界拆出 `belong_to` 与 `from_actor`，后台 task 和 retry 使用这些独立字段，由领域所有者校验归属与可见性。应用服务不会因此拥有 Workspace 资源，也不会为共享 runtime 创建按 Workspace 分区的状态。

应用服务可以保存一次用例的短期控制状态，例如任务进程编排的进程表，但不能保存 Patchouli 的长期记忆状态或 Gateway 的请求级 workflow state。workspace 能力层的两个 API 面都是授权点，先授权、再用返回的 `IdentityScope` 调用资源 owner，不向下传 context（[能力层](../workspace/capability.md#5-授权点的共同约定)）。

### 1.1 Transport / Router 边界

FastAPI router 是 transport adapter，而不是另一层业务编排者。它的职责应收敛为：解析和校验 request、取得窄化的应用服务依赖、调用一个用例、把结果转换成 HTTP/SSE response，并在 transport 边界映射状态码或公开错误。

Router 不得直接访问 `HiveMemorySystem.patchouli`、Alice/Gateway runtime、Store、scheduler 或内部 bus，也不得重新拼装 prepare/run/finalize。`server/deps.py` 应暴露面向用例的 service dependency，而不是把整个 System 当作万能 facade 注入每个入口。HTTP 管理路由只调用 root API；chat 路由只经任务进程的注册入口注册、运行与取消（[任务进程](../workspace/task-process.md)）。

这一边界同样约束应用服务自身：System 是 composition root，不是 God Facade；应用服务可以跨公开 route 编排一个用例，却不能逐步吸收子系统算法、存储访问和所有运行时状态，成为新的万能 runtime。Transport、用例编排和领域所有权保持分离，才允许 HTTP、SSE、CLI 与未来 adapter 共享同一业务语义。

## 2. 服务分工

| 服务（位置） | 当前职责 | 详见 |
|:---|:---|:---|
| `TaskProcessService`（`workspace.process`） | 任务请求的唯一注册入口：注册时完成两阶段认证并创建进程，以流式交付运行已注册的进程（四阶段编排），提供取消与关闭 | [任务进程](../workspace/task-process.md) |
| `RootMemoryService`、`RootMemoryTaskService`、`RootAgentProfileService`、`RootSessionService`、`RootTopicService`、`RootWorkspaceAssetService`（`workspace.capability.root`） | 以 root 授权提供 Memory、记忆任务、Profile、会话、话题与附件上传的管理用例 | [能力层](../workspace/capability.md#3-root-api) |
| `ImportBusService`（`system.application`） | 外部 harness 一轮完整交互的导入：导入授权后委托编排器；关闭时等待在途导入 | 第 3 节；[Import Bus](./import-bus.md) |
| `SystemReadinessService`（`system.application`） | 模型 warmup、ready 和简短 readiness 状态 | 第 3 节 |

这些服务的“拥有”只指顶层用例入口，不改变后端子系统的状态所有权。actor API 的 `ActorMemoryService` 与 `ActorAgentService` 不经门面暴露，只由操作入口持有。

## 3. 系统级服务

`ImportBusService` 是 `POST /api/v1/import/turns` 背后的用例入口：以 workspace 的导入授权（`authorize_import`，只校验准入与目标、不检查 actor 操作白名单、拒绝保留的 `system`）把请求级 context 兑现为本次请求的 `IdentityScope` 与 principal 标识，再交给 `system.services.import_bus` 的编排器 `ImportBusIngestor`。编排器按外部会话串行，经 Patchouli 公开路由查询进度、提交与等待应用，经 Gateway 的 `PASSIVE_MEMORY` 入口只取得话题落位，以与任务进程共用的封口函数组装交互记录，应用完成后才确认。它不检索记忆、不返回记忆上下文，不运行 Alice、MTP、命令或回复生成，不经任务进程的注册入口，不持有 Patchouli 的提交队列，也不向调度器注册维护任务。链路、失败分类与 Claude Code 连接器见 [Import Bus](./import-bus.md)，导入授权在访问模型中的位置见[身份与授权](../workspace/identity-and-authorization.md)。

`SystemReadinessService` 提供模型 warmup、ready 查询和 `ready/warming_up` 摘要，经 Patchouli readiness routes 取得状态，不参与 Workspace 资源授权。ready 表示每个启用的推理模型自进程启动后至少成功加载过一次，且当前不处于加载失败状态：启动预热完成之前为 `warming_up`；空闲卸载不改变 ready，ready 也不表示模型此刻在内存中；重新加载失败时回到 `warming_up`，直到下一次加载成功。驻留与卸载见 [Patchouli](../patchouli/README.md#41-模型驻留与空闲卸载)。

## 4. 错误、观测与 cleanup

- 业务拒绝或资源不存在应使用服务定义的稳定结果/异常；
- task cancellation 不应被包装成普通 success 或 failed；
- 观测事件经 `RuntimeEventPublisher` best-effort 发布：发布失败不改变返回值，失败事件只携带领域错误码或固定摘要，不写入异常正文；chat 任务进程的 `chat.run.*` 投影见[任务进程](../workspace/task-process.md#14-错误与观测)；
- cleanup 是 prepare 失败后的有限补偿，不是跨子系统 rollback；
- 应用服务捕获的错误应保留原始因果，不能通过“空列表/空话题”掩盖 route 缺失或契约违约。

## 5. 应用层矛盾检查

新增应用服务或新入口时，检查：

1. 是否复制了 `TaskProcessService` 的 prepare/run/finalize 顺序？
2. 是否绕过 `GlobalSystemBus` 直接持有子系统 Runtime？
3. 是否把 command outcome 继续送进普通 chat？
4. 是否在取消后仍调用 finalize，或 finalize 失败后忘记 cleanup？
5. 是否把 memory task、topic 或 run registry 的临时状态写成另一个权威来源？
6. 是否把内部 RuntimeEvent、fallback 原因或 trace 当成公开 API 字段？
7. Router 是否只处理 transport，还是直接访问了 System/子系统内部对象或复制了用例顺序？
8. 新应用服务是否开始拥有本应属于子系统的算法、实体或长期状态？

## 6. 验证入口

- `tests/unit/system/application/test_api_services.py`（门面与 server 依赖交出的服务，含 Import Bus 与请求体上限）
- `tests/unit/system/application/test_readiness_service.py`
- `tests/unit/architecture/test_access_boundaries.py`（能力层、任务进程入口与 Import Bus 服务只经访问 context 接收身份；导入授权只由 Import Bus 服务调用）
- `tests/integration/system/application/test_workspace_asset_parsing.py`（真实附件服务、解析服务与 Store 的上传协作）
- `tests/integration/workspace/test_published_registration_chain.py`（随仓库发布的登记文件驱动的管理路由、chat 与取消链路）

任务进程与能力层的测试入口见[任务进程](../workspace/task-process.md#16-代码与测试入口)与[能力层](../workspace/capability.md#10-代码与测试入口)，Import Bus 的测试入口见 [Import Bus](./import-bus.md#11-代码与测试入口)。
