---
title: 认证与访问登记
status: current
owner: workspace
scope: two-phase-authentication-access-registration-and-sealed-access-context
code_paths:
  - src/hivememory/workspace/authentication.py
  - src/hivememory/workspace/registry.py
  - src/hivememory/core/access.py
  - src/hivememory/config/access.py
  - src/hivememory/system/access/
  - src/hivememory/server/deps.py
  - configs/system_principals.yaml
  - configs/workspace_actors.yaml
related_contracts:
  - docs/contracts/error-model.md
related_docs:
  - docs/workspace/identity-and-authorization.md
  - docs/workspace/execution-threads.md
  - docs/system/configuration.md
  - docs/governance/security/identity-and-execution-safety.md
last_reviewed: 2026-10-19
---

# 认证与访问登记

本文说明 actor 如何进入 Workspace：两类访问登记、统一认证网关编排的两阶段认证，以及认证签发的访问 context。四个访问阶段的总览与第 3 阶段的操作授权见[身份与授权](./identity-and-authorization.md#2-访问边界两阶段认证与两阶段授权)。

## 1. 两阶段认证

| 阶段 | 检查 | 负责者 |
|:---|:---|:---|
| 1 Principal 认证 | 调用来源已登记并启用、可经该 adapter 接入、可代表声明中的用户 | System：`core.access.PrincipalAuthenticator` 端口，由 `system/access/` 的 `SystemPrincipalAuthenticator` 实现 |
| 2 Workspace 认证 | actor 用户等于要进入的 Workspace 的 owner，准入记录存在且启用 | workspace：`WorkspaceAuthenticator` |

两个阶段由唯一对外的认证入口 `ActorAuthenticationGateway.authenticate(adapter, principal, actor, workspace, binding)` 依次完成，通过后签发绑定本次运行的访问 context。运行持有者是 server（请求级 context）与任务进程的注册入口（进程 context）；它们只面对认证网关，不直接持有 `WorkspaceAuthenticator`。第 1 阶段属于 System，是因为“哪个调用来源可以接入”是部署的配置，与任何 Workspace 无关；第 2 阶段属于 workspace，是因为“谁能进入哪个 Workspace”是 Workspace 的配置。workspace 以端口声明对第 1 阶段的需要，由组合根注入实现，因此不导入 System。

HTTP 入口以自身登记的 principal 与 `http` adapter 调用认证网关；请求头中的用户身份不做证明，这是本地单用户部署的信任假设（第 8 节）。

## 2. 两类访问登记

两类登记各用一个 YAML 文件，对应各自的配置所有者，由 `config/access.py` 的 `load_access_registration()` 在启动时装载：

| 文件 | 内容 | 配置所有者 | 装载为 |
|:---|:---|:---|:---|
| `configs/system_principals.yaml` | `principals`：稳定来源标识 `principal_id`、连接方式 `kind`、是否启用、可用 adapter、可选的 `allowed_user_ids` 收紧 | System | `system/access/registry.py` 的接入注册表 |
| `configs/workspace_actors.yaml` | `workspace_actors`：`(owner_user_id, workspace_id, user_id, agent_id)` 坐标、是否启用、`allowed_operations` | Workspace | `workspace/registry.py` 的 Workspace Actor 访问注册表 |

- principal 登记不授予任何 Workspace operation；一条 Workspace 访问记录同时承载准入状态与 actor 操作白名单，白名单允许为空（可进入但不能执行 actor 操作）。白名单只表达六项 actor 操作（[能力层](./capability.md#2-actor-操作目录)）；root 能力与任务进程的阶段调用都不经白名单。白名单是 Agent 操作权限的唯一来源：Agent Profile 不承载权限，内置 OMNI_DOLL 的默认能力就是默认用户级记录的全部六项；它与 Profile 原子是否存在无关，也适用于没有 Profile 原子的 actor。`session_id`、run/frame 和调用协议不进入权限键；两个 owner 使用相同 `workspace_id` 时记录不串扰。
- **用户级记录**：省略 `agent_id` 的记录覆盖该用户的所有具体 Agent，但不覆盖保留的 `system`；`system` 必须单独登记。查询时精确记录优先，精确记录被禁用即拒绝，不回落到用户级记录；每个 `(owner, workspace, user)` 至多一条用户级记录。
- **装载规则**：两个文件都拒绝未知字段；装载期拒绝重复键、跨 owner 记录（W0 基线：`user_id` 必须等于 `owner_user_id`）、“禁用却配置白名单”的矛盾配置，以及为保留的 `system` 配置 operation 的记录；未知 operation 值（包括已删除的旧名称）装载期失败并指出值与记录坐标。默认路径的文件缺失时按空登记装载并告警，网关随后拒绝一切认证（fail closed）；文件路径可用 `HIVEMEMORY_PRINCIPALS_PATH` / `HIVEMEMORY_WORKSPACE_ACTORS_PATH` 覆盖，显式指定的路径缺失或内容非法则显式失败。登记在运行实例内不可变，修改经重启生效；登记中没有 context 有效期。
- **随仓库发布的默认登记**：

| 文件 | 登记 | 内容 |
|:---|:---|:---|
| principals | server 的 principal `hivememory:http-server` | adapter `http`；需与 `config.yaml` 的 `system.server_principal_id` 一致 |
| principals | Profile 注入脚本的 principal `hivememory:profile-injector` | kind `local-process`，adapter `local`；脚本以它和 `system` 声明经认证网关取得请求级 context，再调用 root API |
| workspace_actors | 用户 `default` 在 `main_workspace` 的用户级记录（所有具体 Agent） | 六项 actor 操作：`memory.read`、`memory.search`、`memory.run`、`memory.write`、`memory.update`、`agent.call` |
| workspace_actors | 用户 `default` 的 `system` | 只登记准入，不配置 operation |

`system` 记录只回答“能否进入”：root 授权不读取白名单，`system` 也因白名单为空而不能经操作入口执行任何 actor 操作，管理员的直接通道不会走到带缓存的 actor 读取。

## 3. 访问 context：密封的运行时凭据

`WorkspaceAccessContext`（`core/access.py`）在第 2 阶段签发时写入授予内容 `AccessGrant`：

| 内容 | 说明 |
|:---|:---|
| actor | 已验证的 `ActorIdentity` |
| 驻留 Workspace | 准入的 `WorkspaceIdentity` |
| principal | 认证经由的 `CallerPrincipal` |
| 运行绑定 | 运行类型（任务进程、执行线程或请求）与运行标识（`process_id`，或 server 为本次请求生成的标识）；执行线程沿用所属进程的 `process_id`，另带非空的线程标识（主线程为 `main`），其他运行类型不带线程标识 |

行为白名单不进入授予内容：每次授权都按访问登记即时查询，不缓存授权结论。

context 是**密封的**凭据：

- 没有公开字段，repr 不显示内容；签发（`_seal`）、读取（`_unseal`）与撤销（`_revoke`）是私有接口，架构测试限定签发与撤销只在认证一侧调用，读取只在操作授权者与认证一侧（派生、线程撤销校验与诊断查询）调用；
- 直接构造被拒绝——只有认证一侧能签发，不存在“未签发的 context”；
- 复制与序列化被拒绝，修改被拒绝：撤销状态随凭据对象本身，副本不能逃过撤销。

为什么这样设计：

- **不公开字段**：如果 context 公开身份字段，持有它的代码就会直接读身份往下传，“驻留 Workspace”被当作操作身份使用，第 3 阶段的目标检查与白名单被绕过。身份只在授权点经操作授权者取得，其余代码只能传递 context。
- **内容密封在凭据内，而不是存在签发方**：如果内容存在签发方的表里，读取内容就必须回到签发方，第 3 阶段因此依赖第 2 阶段的对象；把内容密封在凭据内，认证一侧与操作授权者互不依赖，只经 context 这个类型发生联系。
- **信任模型**：私有接口与架构测试维护的是可信进程内的调用纪律，不隔离刻意违规的进程内代码。

## 4. 进程 context 与线程 context

任务进程内有两类 context，分别服务两类操作——进程 context 用于进程替 actor 做的系统操作（阶段调用），线程 context 用于执行线程发起的 actor 操作：

| | 进程 context | 主线程的线程 context | 子线程的线程 context |
|:---|:---|:---|:---|
| actor | 进程的 Agent | 进程的 Agent | CALL 的目标 Agent（同一用户） |
| 签发 | 注册时认证签发 | 进入 Actor 阶段时由线程表从进程 context 派生 | 主线程的 CALL 派生到达线程表时，以被调用方准入后从进程 context 派生 |
| 运行绑定 | 任务进程，`process_id` | 所属进程的 `process_id` 加线程标识 `main` | 所属进程的 `process_id` 加进程分配的线程标识（`child-N`） |
| 持有者 | 进程记录 | 线程表登记；CPU 驱动绑定为主帧的线程句柄 | 线程表登记；CPU 驱动绑定为子帧的线程句柄 |
| 用途 | 全部四个阶段的阶段授权、操作快照读取与进程控制比对 | 主线程发起的 actor 操作 | 子线程发起的 actor 操作，按被调用方的白名单与可见性授权 |
| 撤销 | 进程以任何结局关闭时由注册入口最后撤销 | 线程结束时：Actor 阶段收尾兜底结束全部线程，关闭流程再兜底 | 线程结束时：CALL 返回时由 CPU 经操作入口结束；否则同主线程 |

**派生**：两种派生都只接受未撤销的进程 context，沿用其中的驻留 Workspace 与 principal，运行绑定换为执行线程；新 context 记入已签发集合，可以单独撤销。请求级、线程级与已撤销的 context 不能派生，认证入口也拒绝直接签发执行线程绑定。派生不是认证：不重新检查接入登记，也不受网关关闭影响；准入与白名单仍由每次授权即时检查。

- `WorkspaceAuthenticator.derive_thread_context` 派生主线程：复制原 actor，不改变身份。
- `derive_child_thread_context(access, agent_id=..., thread_id=...)` 派生子线程：actor 换为同一用户的被调用方，签发前做被调用方的第 2 阶段准入（owner 与准入记录），准入失败以 `AdmissionDeniedError`（`actor_not_admitted`）拒绝；被调用方不能是保留的 `system`（`system_actor_not_allowed`），线程标识不能是 `main`。

进程侧经 `core.access` 的 `ThreadContextLifecycle` 端口使用这两项派生与线程撤销（撤销拒绝非线程 context），组合根注入认证网关的实现；这些调用只由任务进程的线程管理 `ProcessThreads`（`workspace/process/threads.py`）发出，执行器与 CPU 分配仍不导入认证一侧。线程何时派生、何时结束，以及唯一的存活判定见[执行线程](./execution-threads.md)。

为什么分成两类：context 本身已是密封、可撤销的凭据，线程身份的令牌就是线程 context，不另设一层指向进程 context 的执行凭据；两类 context 各自撤销，线程结束后迟到的 actor 请求即被拒绝，而 finalize、话题池读取与 cleanup 仍以进程 context 完成。

## 5. context 的生命周期

context 只属于一次运行，没有固定有效期，只在以下时点失效：

- 任务进程的进程 context 在注册时签发并绑定本进程，由进程记录持有，进程以任何结局关闭时由注册入口撤销；
- 线程 context 随线程结束撤销：子线程随 CALL 返回由 CPU 结束，其余在 Actor 阶段收尾时由执行器兜底结束，进程关闭流程再兜底；
- 请求级 context 在请求开始时签发并绑定本次请求，请求结束（含失败）时撤销；
- System 停止时先关闭认证网关、拒绝新认证，停止流程最后撤销全部已签发 context（包括派生的线程 context）。关闭网关不影响已签发的 context。

## 6. 认证网关与认证一侧

| 类 | 负责 | 依赖 | 调用方 |
|:---|:---|:---|:---|
| `ActorAuthenticationGateway` | 唯一对外的认证入口：`authenticate(adapter, principal, actor, workspace, binding)` 依次完成第 1、2 阶段并签发 context；转交单个撤销（`revoke_context`）、撤销全部与诊断查询；实现进程侧的线程 context 生命周期端口（派生主线程与子线程、撤销线程 context）；关闭（拒绝新认证） | `PrincipalAuthenticator` 端口、`WorkspaceAuthenticator` | 运行持有者：server（请求级 context）、注册入口（进程 context）；System；进程线程管理只经生命周期端口 |
| `WorkspaceAuthenticator` | 第 2 阶段：检查 actor 用户等于要进入的 Workspace 的 owner，准入记录存在且启用；签发写有授予内容的 context；从进程 context 派生主线程与（经被调用方准入的）子线程 context；撤销（单个、只限线程 context 的撤销、全部）；诊断查询（授予内容摘要，只用于日志与观测标签） | Workspace Actor 访问注册表 | 只有认证网关 |

认证与授权互不依赖：两者都只读同一份访问注册表，经 context 这个类型发生联系；操作授权者见[身份与授权](./identity-and-authorization.md#4-操作授权)。

管理操作与取消的请求级 context 由 server 的 `get_request_access` 依赖取得：以 `(user, system)` 声明经网关认证，一次请求一个，请求结束（含失败）时撤销。Profile 注入脚本以登记的本地来源与 `system` 声明经同一网关取得请求级 context，结束时撤销。

## 7. 装配

`SystemAssembler` 从两个登记文件装载两类注册表，构造 `WorkspaceAuthenticator` 与 `WorkspaceOperationAuthorizer`（两者都只接收 Workspace Actor 访问注册表），再以 `SystemPrincipalAuthenticator` 与 `WorkspaceAuthenticator` 组装认证网关：

- 认证网关注入任务进程的注册入口，并经 `HiveMemorySystem.access_gateway` 暴露给 server；
- 操作授权者注入 root API 与 actor API 的各服务、操作入口、注册入口、执行器与 CPU 分配；CPU 分配另注入读取视图的 Profile 读取端口；
- 进程表由组合根在构建 actor API 之前创建，同时交给注册入口与进程线程管理 `ProcessThreads`；`ProcessThreads` 以认证网关作为线程 context 生命周期端口、以操作授权者读取运行绑定与快照，注入执行器（主线程的派生与兜底结束）、操作入口与 actor API（线程端口 `ExecutionThreadTable`，位于 `workspace/thread_ports.py`，不属于公共契约），能力层与任务进程仍互不导入；
- `WorkspaceOperationEntry` 由组合根以 actor API 与线程端口装配，公共 `OperationEntry` 注入 `AliceCPU`；actor API 只由操作入口持有，门面只向 server 暴露 root API 与任务进程服务；
- Patchouli 与 Gateway 不注入任何认证或授权对象。

停止时认证网关最先关闭、全部 context 最后撤销，见 [Workspace](./README.md#52-停止)。

## 8. 已知限制

- 访问登记只有进程内本地配置：无远程凭据协议、通用 IAM、管理 API 或热更新；HTTP 请求头中的用户身份不做证明，只适用于本地单用户部署；
- 用户级访问记录让同一用户的所有具体 Agent 共享一份白名单；按 Agent 区分需要逐个登记精确记录；
- Agent 操作白名单只能通过修改 `workspace_actors.yaml` 并重启来变更，前端没有权限编辑入口；运行时编辑并入 [ROADMAP](../ROADMAP.md) 的“配置热更新”，届时需要处理线程描述中的快照与即时检查之间的差异。

## 9. 代码与测试入口

- 认证：[`workspace/authentication.py`](../../src/hivememory/workspace/authentication.py)（认证网关与 `WorkspaceAuthenticator`）、[`system/access/`](../../src/hivememory/system/access/)（接入登记与 Principal authentication）、[`core/access.py`](../../src/hivememory/core/access.py)（密封的访问 context、授予内容与运行绑定、端口）；
- 登记：[`workspace/registry.py`](../../src/hivememory/workspace/registry.py)、[`config/access.py`](../../src/hivememory/config/access.py)；请求级访问：[`server/deps.py`](../../src/hivememory/server/deps.py)；
- 测试：[`tests/unit/workspace/test_access.py`](../../tests/unit/workspace/test_access.py)、[`tests/unit/workspace/test_registry.py`](../../tests/unit/workspace/test_registry.py)、[`tests/unit/system/access/test_gateway.py`](../../tests/unit/system/access/test_gateway.py)、[`tests/unit/core/test_access_binding.py`](../../tests/unit/core/test_access_binding.py)、[`test_thread_context.py`](../../tests/integration/workspace/test_thread_context.py)（主线程与子线程的派生、被调用方准入、单独撤销与运行绑定）、[`test_registration_configuration.py`](../../tests/integration/workspace/test_registration_configuration.py)（旧 operation 与 `system` 白名单的装载失败）、[`test_published_registration_chain.py`](../../tests/integration/workspace/test_published_registration_chain.py)（随仓库发布的登记文件驱动的 HTTP 链路）。
