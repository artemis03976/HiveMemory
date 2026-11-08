---
title: 认证与访问登记
status: current
owner: workspace
scope: two-phase-authentication-access-registration-and-working-actor
code_paths:
  - src/hivememory/workspace/authentication.py
  - src/hivememory/workspace/registry.py
  - src/hivememory/core/access.py
  - src/hivememory/config/access.py
  - src/hivememory/system/access/
  - src/hivememory/server/deps.py
  - src/hivememory/server/management_credential.py
  - src/hivememory/server/routers/import_bus.py
  - src/hivememory/mcp_server/
  - configs/system_principals.yaml
  - configs/workspace_actors.yaml
related_contracts:
  - docs/contracts/error-model.md
  - docs/contracts/mcp.md
related_docs:
  - docs/workspace/identity-and-authorization.md
  - docs/workspace/execution-threads.md
  - docs/system/configuration.md
  - docs/system/import-bus.md
  - docs/governance/security/identity-and-execution-safety.md
last_reviewed: 2026-11-06
---

# 认证与访问登记

本文说明 actor 如何进入 Workspace：两类访问登记、统一认证网关编排的两阶段认证，以及认证得到的工作 actor。四个访问阶段的总览与第 3 阶段的操作授权见[身份与授权](./identity-and-authorization.md#2-访问边界两阶段认证与两阶段授权)。

## 1. 两阶段认证

| 阶段 | 检查 | 负责者 |
|:---|:---|:---|
| 1 Principal 认证 | 调用来源已登记并启用、可经该 adapter 接入；登记了凭据的来源提供一致的凭据；可代表声明中的用户；声明保留的 `system` 时登记允许它这样做 | System：`core.access.PrincipalAuthenticator` 端口，由 `system/access/` 的 `SystemPrincipalAuthenticator` 实现 |
| 2 Workspace 认证 | actor 用户等于要进入的 Workspace 的 owner，准入记录存在且启用 | workspace：`WorkspaceAuthenticator` |

两个阶段由唯一对外的认证入口 `ActorAuthenticationGateway.authenticate(adapter, principal, credential, actor, workspace)` 依次完成，通过后返回工作 actor（第 3 节）。调用认证的是入口与系统组件：server 与 MCP 入口、任务进程的注册入口与 Import Bus 服务；它们只面对认证网关，不直接持有 `WorkspaceAuthenticator`。第 1 阶段属于 System，是因为“哪个调用来源可以接入”是部署的配置，与任何 Workspace 无关；第 2 阶段属于 workspace，是因为“谁能进入哪个 Workspace”是 Workspace 的配置。workspace 以端口声明对第 1 阶段的需要，由组合根注入实现，因此不导入 System。

入口分为两类。**进程内入口**以自己登记的 principal 接入，不携带凭据：server 的前端路由以 server 的 principal 与 `http` adapter，Profile 注入脚本以自己的 principal 与 `local` adapter，系统组件以系统自身的 principal 与 `local` adapter（第 5 节）。**外部入口**接收外部 harness 的请求，principal 取自请求中的声明（`X-Principal-ID`），只是声明，入口同时把 `Authorization: Bearer` 中的接入凭据交给网关，由第 1 阶段核对后才成立；入口只携带证据，不自行判定身份。请求头中的用户身份不做证明，这是本地单用户部署的信任假设（第 8 节）。与 Workspace 相关的 HTTP 入口都经网关认证：

| 入口 | principal 与 adapter | 声明 | 认证结果 |
|:---|:---|:---|:---|
| Chat | server 的 principal，`http` | (user, 请求体中的具体 Agent) | 注册入口认证得到执行者，另以系统自身的 principal 认证系统的 root 工作 actor，两者存进进程记录（[任务进程](./task-process.md#2-注册)） |
| 管理路由与 `/chat/stop` | server 的 principal，`http` | (user, `system`) | 请求的工作 actor，经 `get_request_actor` 依赖取得 |
| 导入 `POST /api/v1/import/turns` | 请求声明的 principal，`import`，携带凭据 | (user, 请求体中的具体 Agent) | 请求的工作 actor，路由经 `authenticate_request_actor` 取得；导入本身由 Import Bus 服务以系统的 root 工作 actor 执行（[Import Bus](../system/import-bus.md)） |
| MCP `POST /mcp` | 请求声明的 principal，`mcp`，携带凭据 | (user, `X-Agent-ID` 中的具体 Agent) | 每次工具调用认证一次，得到请求的工作 actor（[MCP 记忆访问](../contracts/mcp.md#2-身份交接)） |

**adapter 标识**表示调用来源经哪个入口接入，由入口代码固定传入，从不取自请求：`http`、`local` 是进程内入口，`mcp`、`import` 是外部入口，四个标识都定义在 `system/access/`。一个外部 harness 只登记一份接入登记，它插件中的各部分共用：Claude Code 的连接器经 `import` 导入、MCP server 经 `mcp` 检索与读取，以同一个 principal `harness:claude-code` 与同一份凭据认证。

**接入凭据**：

- 登记只写环境变量名（`credential_env`），凭据本身不进入仓库。组合根装配接入登记时按与其他配置相同的来源读取：进程环境变量优先，其次是 `.env` 文件；去掉首尾空白，凭据只存在于内存中的登记里，不进入日志与 `repr`。
- 第 1 阶段在 adapter 检查之后以常量时间比较核对凭据：登记了 `credential_env` 的 principal 必须提供一致的凭据，否则以 `invalid_credential` 拒绝。环境变量为空时启动时记录一条警告，此后该 principal 的每次认证都被拒绝；HiveMemory 照常启动，不使用这个来源的部署不受影响。凭据错误与未登记分开报告，用户才能区分“没登记”和“凭据不对”。
- 装载时校验：列出外部 adapter 的登记必须声明 `credential_env`，列出进程内 adapter 的登记不能声明凭据，否则装载失败。没有凭据的 principal（server、Profile 注入脚本）因此只能经进程内 adapter 接入，外部请求无法借用它们的身份；进程内入口总是不带凭据认证，与凭据同登记会使这些入口一律被拒绝。新增的 adapter 默认属于外部入口，必须声明凭据。

**`system` 的声明范围**：登记的 `allow_system_actor` 默认为 `false`。第 1 阶段在用户检查之后拒绝不允许的 principal 声明保留的 `system`（`system_actor_not_allowed`）；能声明 `system` 的只有用户经 HTTP 入口的管理访问、Profile 注入脚本与系统自身的 principal，三者的登记显式设为 `true`，外部 harness 取默认值。

**HTTP 管理面的管理凭据**：上表的 server 的 principal 不携带凭据，HTTP 管理面另由管理凭据证明调用方，检查发生在认证网关之前。

- 范围：server 的一个 ASGI 中间件（`server/management_credential.py`）只按路径判定、默认拒绝。路径以 `/api/` 开头的 HTTP 请求与 WebSocket 连接都要求管理凭据，导入 `POST /api/v1/import/turns` 除外；`/health`、`/health/ready`、`/mcp` 与前端页面不要求。以后在 `/api/` 下新增的路由自动受保护，前端使用的 `config`、`providers`、`models`、运行时事件与日志这些不经认证网关的接口也在其中。
- 校验：凭据放在 `Authorization: Bearer` 请求头或名为 `hivememory_management` 的 cookie 中，任一与后端持有的凭据一致即可，以常量时间比较。缺少或不一致时，HTTP 以 401 `request.management_credential_invalid` 拒绝，WebSocket 在接受连接之前以策略违规（1008）关闭，路由处理函数都不会执行。
- 凭据保存在 `configs/management.token`，与 `providers.secrets.yaml` 同目录，被 `.gitignore` 与 `.dockerignore` 排除。后端启动时读取它：不存在就生成随机凭据并原子写入，POSIX 上文件只有当前用户可读写；Windows 没有这类权限位，文件沿用所在目录的 ACL，本机其他账户能否读到它取决于目录权限。文件为空时启动失败。删除文件后重启即得到新凭据，没有环境变量覆盖与轮换。读入之前，中间件把要求凭据的请求一律按凭据无效拒绝。
- 前端源码不处理凭据：Vite 开发服务器的 `/api` 代理每次转发 HTTP 与 WebSocket 请求时读这个文件并附上请求头；后端提供页面时，返回页面的同时下发 `HttpOnly`、`SameSite=Strict`、`Path=/` 的 cookie，页面的同源请求、SSE 与 WebSocket 自动携带它，页面脚本读不到它。
- 凭据属于 server，不进入接入登记：它保护的是 HTTP 管理面，不参与 Workspace 准入。通过检查的请求照旧以 server 的 principal 经网关认证，请求头中的用户身份仍是声明。它只防误用，边界见第 8 节。

## 2. 两类访问登记

两类登记各用一个 YAML 文件，对应各自的配置所有者，由 `config/access.py` 的 `load_access_registration()` 在启动时装载：

| 文件 | 内容 | 配置所有者 | 装载为 |
|:---|:---|:---|:---|
| `configs/system_principals.yaml` | `principals`：稳定来源标识 `principal_id`、连接方式 `kind`、是否启用、可用 adapter、可选的 `allowed_user_ids` 收紧、能否声明 `system` 的 `allow_system_actor`（默认 `false`）、接入凭据的环境变量名 `credential_env` | System | `system/access/registry.py` 的接入注册表（`SystemActorAccessRegistry.from_config` 装配并读取凭据） |
| `configs/workspace_actors.yaml` | `workspace_actors`：`(owner_user_id, workspace_id, user_id, agent_id)` 坐标、是否启用、`allowed_operations` | Workspace | `workspace/registry.py` 的 Workspace Actor 访问注册表 |

- principal 登记不授予任何 Workspace operation；一条 Workspace 访问记录同时承载准入状态与 actor 操作白名单，白名单允许为空（可进入但不能执行 actor 操作）。白名单只表达六项 actor 操作（[能力层](./capability.md#2-actor-操作目录)）；root 授权的系统级操作都不经白名单。白名单是 Agent 操作权限的唯一来源：Agent Profile 不承载权限，内置 OMNI_DOLL 的默认能力就是默认用户级记录的全部六项；它与 Profile 原子是否存在无关，也适用于没有 Profile 原子的 actor。`session_id`、run/frame 和调用协议不进入权限键；两个 owner 使用相同 `workspace_id` 时记录不串扰。
- **用户级记录**：省略 `agent_id` 的记录覆盖该用户的所有具体 Agent，但不覆盖保留的 `system`；`system` 必须单独登记。查询时精确记录优先，精确记录被禁用即拒绝，不回落到用户级记录；每个 `(owner, workspace, user)` 至多一条用户级记录。
- **装载规则**：两个文件都拒绝未知字段；装载期拒绝重复键、跨 owner 记录（W0 基线：`user_id` 必须等于 `owner_user_id`）、“禁用却配置白名单”的矛盾配置，以及为保留的 `system` 配置 operation 的记录；未知 operation 值（包括已删除的旧名称）装载期失败并指出值与记录坐标。默认路径的文件缺失时按空登记装载并告警，网关随后拒绝一切认证（fail closed）；文件路径可用 `HIVEMEMORY_PRINCIPALS_PATH` / `HIVEMEMORY_WORKSPACE_ACTORS_PATH` 覆盖，显式指定的路径缺失或内容非法则显式失败。登记在运行实例内不可变，修改经重启生效。
- **随仓库发布的默认登记**：

| 文件 | 登记 | 内容 |
|:---|:---|:---|
| principals | server 的 principal `hivememory:http-server` | adapter `http`，`allow_system_actor: true`；需与 `config.yaml` 的 `system.server_principal_id` 一致；只用于前端的管理、chat 与取消路由 |
| principals | 系统自身的 principal `hivememory:system` | kind `local-process`，adapter `local`，`allow_system_actor: true`；任务进程的注册入口与 Import Bus 服务以它认证系统的 root 工作 actor（第 5 节） |
| principals | Profile 注入脚本的 principal `hivememory:profile-injector` | kind `local-process`，adapter `local`，`allow_system_actor: true`；脚本以它和 `system` 声明经认证网关得到工作 actor，再调用 root API |
| principals | Claude Code 的 principal `harness:claude-code` | kind `harness`，adapter `mcp` 与 `import`，`credential_env: HIVEMEMORY_CLAUDE_CODE_TOKEN`；不能声明 `system` |
| principals | DeepSeek Harness 的 principal `harness:dsh` | kind `harness`，adapter `mcp` 与 `import`，`credential_env: HIVEMEMORY_DSH_TOKEN`；不能声明 `system` |
| workspace_actors | 用户 `default` 在 `main_workspace` 的用户级记录（所有具体 Agent） | 六项 actor 操作：`memory.read`、`memory.search`、`memory.run`、`memory.write`、`memory.update`、`agent.call` |
| workspace_actors | 用户 `default` 的 `system` | 只登记准入，不配置 operation |
| workspace_actors | 用户 `default` 的 `claude_code`（Claude Code 导入与 MCP 访问声明的 Agent） | 精确记录，`allowed_operations` 为 `memory.read`、`memory.search`、`memory.write` 与 `memory.update` |
| workspace_actors | 用户 `default` 的 `dsh`（DeepSeek Harness 插件导入与记忆工具声明的 Agent） | 精确记录，`allowed_operations` 同 `claude_code` |

`system` 记录只回答“能否进入”：root 授权不读取白名单，`system` 也因白名单为空而不能经操作入口执行任何 actor 操作，管理员的直接通道不会走到带缓存的 actor 读取。管理路由、chat 的系统阶段与 Import Bus 都以这条记录准入 `system`，因此任何要对话、导入或经 MCP 写入的 Workspace 都需要它；用户级记录不覆盖 `system`。`claude_code` 与 `dsh` 的精确记录获准检索、读取、写入与更新：导入与写入时的确定话题由系统执行、不读取白名单（[身份与授权](./identity-and-authorization.md#4-操作授权)），MCP 的每次工具调用按这份白名单逐次授权；精确记录优先于用户级记录，这两个 Agent 因此不继承用户级的 `memory.run` 与 `agent.call`。

## 3. 工作 actor

认证的结果是工作 actor `WorkingActor`（`core/access.py`），一个冻结的值：

| 字段 | 说明 |
|:---|:---|
| `principal` | 认证经由的 `CallerPrincipal`：第 1 阶段核对通过的调用来源 |
| `actor` | 已验证的 `ActorIdentity` |
| `workspace` | 准入的驻留 `WorkspaceIdentity`；它不是某次操作的目标，目标由每次授权单独给出（[身份与授权](./identity-and-authorization.md#2-访问边界两阶段认证与两阶段授权)） |

工作 actor 不带运行绑定、有效期、adapter 或操作上限，也不带“是否经过证明”：行为白名单不进入它，每次授权都按访问登记即时查询，不缓存授权结论；证明分别发生在入口（HTTP 管理面的管理凭据、外部 principal 的接入凭据），之后没有使用方需要再区分。

为什么是普通值：进程内的伪造并不发生在这里。外部请求到达时，伪造只能发生在进程边界上，而边界上的证明已由接入凭据与管理凭据承担；进程内的代码无论拿到的是密封凭据还是普通值，区分特权最终都靠结构纪律。因此进程内的防误用改由构造与持有位置的限定承担，由架构测试守护：

- 工作 actor 只由认证一侧构造（`workspace/authentication.py`）；
- Alice、agent_runtime、Patchouli、Gateway 与引擎不引用它，CPU 可见的对象（输入清单、线程描述与提交函数）不含它，公开记录与物化任务不保存它；
- 操作授权者只读取它，与认证一侧互不导入，两者只经这个类型发生联系。

## 4. 工作 actor 的角色与持有位置

同一类型有四种角色，区别只在来源与持有位置：

| 角色 | 来源 | 持有位置 | 用途 |
|:---|:---|:---|:---|
| 执行者（主工作 actor） | 注册入口认证 (user, 具体 Agent) | 进程记录与主线程记录 | 主线程的 actor 操作（含预检索）；交互提交中作为交互的参与者 |
| 子工作 actor | 线程表派生子线程时的准入 | 子线程记录 | 子线程的 actor 操作，按被调用方的白名单与可见性授权 |
| 系统的 root 工作 actor | 系统组件以系统自身的 principal 认证 (user, `system`) | 进程记录；Import Bus 的调用链 | 任务进程的系统阶段（含线程表为这一轮确定话题）；导入与写入时确定话题 |
| 请求的工作 actor | 管理路由、`/chat/stop`、导入与 MCP 入口的认证 | 入口调用链 | 管理与取消（`system`）；证明导入的来源与参与者；MCP 的 actor 操作 |

持有工作 actor 的记录不撤销：进程记录随进程注销被丢弃，线程记录的存活由线程表判定（[执行线程](./execution-threads.md#3-生命周期与唯一的存活判定)），请求的工作 actor 是入口调用链内的局部值，调用返回即不再被引用。

**子工作 actor 的准入**：`admit_sub_actor(main, agent_id)` 是 `core.access.SubActorAdmission` 端口，由认证网关实现、转交 `WorkspaceAuthenticator`。子工作 actor 沿用主工作 actor 的 principal 与驻留 Workspace，只换成同一用户的被调用 Agent，再做第 2 阶段准入（owner 与准入记录），失败以 `AdmissionDeniedError`（`actor_not_admitted`）拒绝；被调用方不能是保留的 `system`（`system_actor_not_allowed`）。准入不是重新认证：不检查接入登记，也不受网关关闭影响，已有进程在关闭期间仍可派生子线程。这项准入只由任务进程的线程管理 `ProcessThreads`（`workspace/process/threads.py`）调用，执行器、CPU 分配与能力层不导入认证一侧。线程何时派生、何时结束见[执行线程](./execution-threads.md)。

## 5. 系统自身的 principal

系统级操作以保留的 `system` 执行（[身份与授权](./identity-and-authorization.md#4-操作授权)）。系统组件取得这个身份的方式与 Profile 注入脚本相同：以自己登记的 principal 经进程内 adapter 认证 (user, `system`)。

- 随仓库发布的登记包含 `hivememory:system`（`local` adapter、允许声明 `system`、不带凭据）；它的标识与 adapter 标识一样定义为 `system/access/` 中的常量 `SYSTEM_PRINCIPAL_ID`，只由组合根引用，注入任务进程的注册入口与 Import Bus 服务（架构测试守护）。
- 任务进程注册时认证一次，得到存进进程记录的系统 root 工作 actor；Import Bus 每次导入或确定话题时认证一次，认证的是参与者驻留 Workspace 的 `system`。
- (user, `system`) 的准入沿用访问登记中已有的 `system` 记录。缺少系统自身的 principal 时，chat 注册、导入与 MCP 写入都以 `unknown_principal` 拒绝；缺少该用户在对应 Workspace 的 `system` 记录时以 `actor_not_admitted` 拒绝。
- 能声明 `system` 的来源由登记逐个限定，外部 harness 不能；外部请求声明 `system` 作为参与者时，入口的声明解析先以 `request.agent_reserved` 拒绝。

## 6. 认证网关与认证一侧

| 类 | 负责 | 依赖 | 调用方 |
|:---|:---|:---|:---|
| `ActorAuthenticationGateway` | 唯一对外的认证入口：`authenticate(adapter, principal, credential, actor, workspace)` 依次完成第 1、2 阶段并返回工作 actor；实现子工作 actor 的准入端口；关闭（拒绝新认证） | `PrincipalAuthenticator` 端口、`WorkspaceAuthenticator` | server 与 MCP 入口、任务进程的注册入口、Import Bus 服务；Profile 注入脚本；进程线程管理只经准入端口 |
| `WorkspaceAuthenticator` | 第 2 阶段：检查 actor 用户等于要进入的 Workspace 的 owner，准入记录存在且启用，通过后构造工作 actor；子工作 actor 的准入 | Workspace Actor 访问注册表 | 只有认证网关 |

两个类都无状态：工作 actor 不登记、不撤销。认证与授权互不依赖：两者都只读同一份访问注册表，经工作 actor 这个类型发生联系；操作授权者见[身份与授权](./identity-and-authorization.md#4-操作授权)。

管理操作与取消的工作 actor 由 server 的 `get_request_actor` 依赖取得：以 `(user, system)` 声明经网关认证，一次请求认证一次。导入的 Agent 来自请求体，路由以 `resolve_request_identity_claims(require_agent=True)` 解析的 (user, 具体 Agent) 声明调用同一个 `authenticate_request_actor`；认证网关关闭后，新的导入请求以 `authentication_gateway_closed`（503）拒绝。System 停止时最先关闭认证网关、拒绝新认证；已有进程照常收尾，在途请求由随后各组件的关闭拒绝（[Workspace](./README.md#52-停止)）。

## 7. 装配

`SystemAssembler` 从两个登记文件装载两类注册表（接入注册表经 `SystemActorAccessRegistry.from_config` 装配，同时读取各来源的接入凭据），构造 `WorkspaceAuthenticator` 与 `WorkspaceOperationAuthorizer`（两者都只接收 Workspace Actor 访问注册表），再以 `SystemPrincipalAuthenticator` 与 `WorkspaceAuthenticator` 组装认证网关：

- 认证网关注入任务进程的注册入口与 Import Bus 服务，并经 `HiveMemorySystem.access_gateway` 暴露给 server 与 MCP 入口；系统自身的 principal 与 `local` adapter 注入注册入口与 Import Bus 服务；
- 操作授权者注入 root API 与 actor API 的各服务、注册入口、执行器、CPU 分配与进程线程管理；Import Bus 服务只注入认证网关，需要授权的步骤经 root API；CPU 分配另注入读取视图的 Profile 读取端口；
- 进程表由组合根在构建 actor API 之前创建，同时交给注册入口与进程线程管理 `ProcessThreads`；`ProcessThreads` 以认证网关作为子工作 actor 的准入端口、以操作授权者读取操作快照、以 root 话题 API 为进程这一轮确定话题，注入执行器（主线程的派生与兜底结束）、操作入口与 actor API（线程端口 `ExecutionThreadTable`，位于 `workspace/thread_ports.py`，不属于公共契约）；能力层不导入任务进程，任务进程在能力层中只导入 root API；
- `WorkspaceOperationEntry` 由组合根以 actor API 与线程端口装配，公共 `OperationEntry` 注入 `AliceCPU`，同一个入口经门面的 `operation_entry` 交给 MCP 入口，只用其请求级提交；MCP 入口把经门面的 `import_bus_service` 为这一轮确定话题的回调交给写入与更新，actor API 在授权之后才调用它；actor API 只由操作入口持有，门面不暴露它；
- Patchouli 与 Gateway 不注入任何认证或授权对象。

## 8. 已知限制

- 访问登记只有进程内本地配置：无远程凭据协议、通用 IAM、管理 API 或热更新；HTTP 请求头中的用户身份不做证明，只适用于本地单用户部署；
- 管理凭据只防误用：它挡住 agent 在工作中顺手调用管理接口，以及读不到凭据文件的其他机器；以同一系统用户运行、专门读取 `configs/management.token` 的程序仍能通过。后端提供页面时，能请求到页面的程序都能取得 cookie，因此正式入口默认只监听本机地址（[安装与启动](../help/setup.md#43-启动后端)）。凭据不区分用户，没有轮换与过期；
- 接入凭据是每个外部 principal 一份本地共享密钥，存在于两侧的环境变量中：只能防止本机其他程序冒用 harness 身份，不防止能读取用户环境的程序；凭据由用户生成，没有签发、轮换、撤销与过期，修改经重启生效；
- 用户级访问记录让同一用户的所有具体 Agent 共享一份白名单；按 Agent 区分需要逐个登记精确记录；
- Agent 操作白名单只能通过修改 `workspace_actors.yaml` 并重启来变更，前端没有权限编辑入口；运行时编辑并入 [ROADMAP](../ROADMAP.md) 的“配置热更新”，届时需要处理线程描述中的快照与即时检查之间的差异。

## 9. 代码与测试入口

- 认证：[`workspace/authentication.py`](../../src/hivememory/workspace/authentication.py)（认证网关与 `WorkspaceAuthenticator`）、[`system/access/`](../../src/hivememory/system/access/)（接入登记、Principal authentication、adapter 标识与系统自身的 principal）、[`core/access.py`](../../src/hivememory/core/access.py)（工作 actor、调用来源身份与端口）；
- 登记：[`workspace/registry.py`](../../src/hivememory/workspace/registry.py)、[`config/access.py`](../../src/hivememory/config/access.py)；请求的工作 actor：[`server/deps.py`](../../src/hivememory/server/deps.py)；
- 测试：[`tests/unit/workspace/test_access.py`](../../tests/unit/workspace/test_access.py)、[`tests/unit/workspace/test_registry.py`](../../tests/unit/workspace/test_registry.py)、[`tests/unit/system/access/test_gateway.py`](../../tests/unit/system/access/test_gateway.py)（认证结果、拒绝原因与子工作 actor 的准入）、[`test_process_threads.py`](../../tests/integration/workspace/test_process_threads.py)（子线程沿用主工作 actor 的来源与驻留 Workspace、被调用方准入）、[`test_registration_configuration.py`](../../tests/integration/workspace/test_registration_configuration.py)（旧 operation 与 `system` 白名单的装载失败）、[`test_published_registration_chain.py`](../../tests/integration/workspace/test_published_registration_chain.py)（随仓库发布的登记文件驱动的 HTTP 链路）、[`tests/unit/system/access/test_registry.py`](../../tests/unit/system/access/test_registry.py)（凭据与 adapter 类别的装载校验、从环境变量与 `.env` 读取凭据）、[`test_import_bus.py`](../../tests/unit/server/routers/test_import_bus.py)（导入以请求声明的 principal、凭据与具体 Agent 认证，缺少声明、凭据不对、未准入与网关关闭的拒绝）、[`test_import_bus_submission.py`](../../tests/integration/system/test_import_bus_submission.py)（系统自身的 principal 未登记时导入被拒）、[`test_mcp_memory_access.py`](../../tests/integration/system/test_mcp_memory_access.py)（MCP 入口经真实网关的凭据核对与借用 server principal 的拒绝）、[`test_import_turns.py`](../../tests/e2e/system/test_import_turns.py)（白名单为空的 Agent 经 HTTP 导入一轮交互）。
