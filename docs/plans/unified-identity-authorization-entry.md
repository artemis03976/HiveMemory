---
title: 统一身份授权入口
status: planned
owner: workspace
target: 版本之外的工作（身份与访问主线，owner 2026-11-05 定为不绑定版本的主线方向）
scope: working-actor-two-authorization-methods-root-system-operations-and-runs-without-identity
updates:
  - docs/workspace/authentication.md
  - docs/workspace/identity-and-authorization.md
  - docs/workspace/execution-threads.md
  - docs/workspace/task-process.md
  - docs/workspace/capability.md
  - docs/workspace/read-view-and-write-intents.md
  - docs/workspace/README.md
  - docs/patchouli/README.md
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/mcp.md
  - docs/contracts/error-model.md
  - docs/system/import-bus.md
  - docs/system/composition.md
  - docs/system/application-services.md
  - docs/system/configuration.md
  - docs/system/README.md
  - docs/help/configuration.md
  - docs/architecture/overview.md
  - docs/architecture/boundaries.md
  - docs/architecture/data-model.md
  - docs/governance/security/identity-and-execution-safety.md
  - docs/PROJECT.md
  - AGENTS.md
code_paths:
  - src/hivememory/core/access.py
  - src/hivememory/core/models/workspace.py
  - src/hivememory/workspace/authentication.py
  - src/hivememory/workspace/authorization.py
  - src/hivememory/workspace/process/
  - src/hivememory/workspace/capability/
  - src/hivememory/workspace/resolution/profile.py
  - src/hivememory/workspace/contracts/threads.py
  - src/hivememory/workspace/contracts/operations.py
  - src/hivememory/workspace/thread_ports.py
  - src/hivememory/patchouli/service.py
  - src/hivememory/patchouli/contracts/public_routes.py
  - src/hivememory/system/access/
  - src/hivememory/system/application/import_bus_service.py
  - src/hivememory/system/services/import_bus/
  - src/hivememory/system/system.py
  - src/hivememory/server/deps.py
  - src/hivememory/server/routers/
  - src/hivememory/mcp_server/server.py
  - src/hivememory/alice/interface/cpu.py
  - configs/system_principals.yaml
  - tests/unit/architecture/test_access_boundaries.py
related_docs:
  - docs/ideas/identity-and-access-model.md
  - docs/governance/baselines/identity-access-current-state-2026-11-05.md
  - docs/workspace/authentication.md
  - docs/workspace/execution-threads.md
  - docs/contracts/mcp.md
---

# 统一身份授权入口

> **计划状态**：planned。方向来自[身份与访问体系 Idea](../ideas/identity-and-access-model.md#访问-context-与授权入口的收拢)第 5 节的 owner 决定（2026-11-05，选项 B）。2026-11-05 计划评审中 owner 又做了一组决定，标为“owner，2026-11-05”，集中列在第 4.1 节；其余是候选设计，不是 owner 决定，评审时可调整，相对 Idea 候选设计的调整列在第 4.10 节。

## 1. 背景和问题

身份现在随运行类型分化（[身份与访问体系现状调查](../governance/baselines/identity-access-current-state-2026-11-05.md)第 4、6、10 节）：

- 认证网关签发密封的访问 context，按运行绑定分为进程、执行线程（主线程与子线程）和请求三种（[认证与访问登记](../workspace/authentication.md#3-访问-context密封的运行时凭据)第 3、4 节）。
- 操作授权者有七个授权方法和一个只读查询；actor 操作的授权另带 `request_level` 参数，MCP 的写入与更新另带派发话题。
- 回答“某个外部调用方能做什么”，要依次查 principal 登记、Workspace 访问登记、入口开放的请求类型与它经过的授权方法。

owner 对这些层层概念不满意，代码可读性差到无法理解现状（Idea 第 5 节的 owner 观察）。Idea 的分析：

- 发起者与目标在运行时确认是本质，身份随运行类型分化是偶然。
- 进程内的防伪造并不完整，伪造真正发生在进程边界上；边界上的证明已由外部 principal 的接入凭据（P-1c）与 HTTP 管理面的管理凭据承担。

计划评审中又看清了两个混淆：

1. **操作授权与能力层的入口检查混在一起**（owner，2026-11-05）。操作授权只在 actor 执行期间才有意义。root 管理、任务进程的阶段调用、导入与进程控制之所以也要检查，是因为所有 API 都收拢到了 Workspace 能力层。它们各有一个 `authorize_*` 方法，看上去像操作授权，实际做的都是“确认准入与目标，再组装 scope”。
2. **执行者与参与者混在一起**。`IdentityScope` 的 actor 既表示“谁在执行这项操作”，又被当作交互的参与者：
   - Patchouli 的 prepare 与 finalize 直接取 scope 的 actor 作为 `from_actor`；
   - Import Bus 的编排器用它推导外部会话键、交互标识与事件中的 `agent_id`。

   为了让参与者是那个 Agent，系统级操作只能以 Agent 的身份授权，于是有了“阶段授权”与“导入授权”。这也带来了悖论：

   - Profile 以 Agent 自己的 `agent_id`、按 Agent 自己的可见性读取。指向其他 Agent 的 PRIVATE 图纸，它自己的进程读不出来；TEAM 图纸在生产中任何进程都读不出来，因为 `team_id` 从未被设置。
   - 用户选择的附件，也以 Agent 的身份租借。

   生产中所有记忆都是 PUBLIC，所以至今没有触发。

本计划把身份与运行分开，并让每个操作以真正执行它的身份进行：

- 认证得到工作 actor，留在入口调用链或运行记录中；
- 授权只有两个方法：actor 执行期间的操作授权，以及系统级操作的 root 授权；
- 任务进程与执行线程只负责存活与派发话题。

## 2. 目标与非目标

### 2.1 目标

1. 认证网关产出工作 actor，不再签发访问 context。各类工作 actor 只留在入口调用链或运行记录中。
2. 授权只有 `authorize_agent` 与 `authorize_root` 两个方法，取代现有的七个授权方法、`request_level` 与导入授权。
3. 能力层 API 代表谁在执行这项操作（owner，2026-11-05）：
   - actor 操作（执行线程、MCP 请求、prepare 的检索）以 Agent 的身份、按白名单授权；
   - 其余都是系统级操作，以系统的 root 身份执行：root API、取消、任务进程的系统阶段、交互提交；
   - 交互中的 Agent 作为参与者显式传递。
4. 任务进程与执行线程不承载身份：CPU 只拿到不透明的线程句柄，存活只由线程表判定。
5. `IdentityScope` 与工作 actor 的构造和持有位置由架构测试限定。

### 2.2 非目标

- 跨 Workspace 的目标（Idea 第 7 节）：目标仍只能等于驻留 Workspace。
- P-4a（进程能否持有白名单的子集）、P-1b（controller 回调的归属校验）。
- P-7 中取消以外的部分：取消由 root 执行（owner，2026-11-05），状态查询，以及 controller 模式下由谁控制进程，仍未决。
- WorkspaceAsset 的归属拆分：资产桥接自行构造 `IdentityScope` 的例外保留（[WorkspaceAsset 归属拆分 Todo](../todo/workspace-asset-ownership-identity-split.md)）。
- 两阶段认证的检查内容、登记的生效时点（E-1）与接入凭据。
- 历史对话导入本身：本计划只让交互提交成为 root 能力，历史导入将来经它提交。
- MCP 写入与更新的顺序：先落位、再授权写入，保持 [MCP 记忆访问](../contracts/mcp.md#4-落位操作入口与授权)第 4 节的现状。
- 后台任务：以 `system` 发起的 SETTLE 等任务只携带 `belong_to` 与 `from_actor`，不在这条主线上，不变。

## 3. 当前状态和缺口

### 3.1 现有概念

| 概念 | 位置 | 作用 |
|:---|:---|:---|
| `WorkspaceAccessContext`、`AccessGrant` | `core/access.py` | 密封的凭据，授予内容为 actor、驻留 Workspace、principal 与运行绑定；不能复制、序列化，可撤销 |
| `RunBinding`、`AccessRunType` | `core/access.py` | 运行绑定：任务进程、执行线程、请求 |
| `ThreadContextLifecycle` | `core/access.py` | 进程侧的线程 context 派生与撤销端口，由认证网关实现 |
| 网关的派生、撤销与 `revoke_all_contexts` | `workspace/authentication.py` | 线程 context 的派生与撤销；System 停止时撤销全部已签发 context |
| `WorkspaceOperationAuthorizer` 的七个授权方法 | `workspace/authorization.py` | `authorize_operation`、`authorize_actor_operation`（带 `request_level`）、`authorize_stage`、`authorize_import`（返回 `ImportAuthorization`）、`authorize_root`、`authorize_process_control`、`get_thread_binding` |
| `get_stage_operations` | `workspace/authorization.py` | 线程描述中的操作快照 |
| `ProcessRecord.access`、`ProcessThread.access`、`ExecutionThread.access` | `workspace/process/`、`workspace/contracts/threads.py` | 进程与线程持有的 context；CPU 驱动把线程 context 绑定进 `ThreadOperationSubmitter` |
| `RequestAccess`、`request_access_for_claims` | `server/deps.py` | 请求级 context 的取得与撤销，供管理路由、`/chat/stop`、导入与 MCP 使用 |

### 3.2 现有 context 保证的行为与去处

Idea 要求实施前逐条列出现有 context 保证的行为，并确认每一条的去处。下表就是这份清单，实施与评审时逐行核对。

| # | 行为 | 现在由什么保证 | 本计划之后由什么保证 |
|:---:|:---|:---|:---|
| 1 | 阶段调用（Gateway、prepare、Profile、附件、finalize、cleanup）每次重新确认准入与目标 | 对每次阶段调用执行 `authorize_stage(进程 context, 目标)`，scope 的 actor 是进程的 Agent | 按执行者拆开（第 4.6 节）：prepare 的检索以执行者的 `authorize_agent(memory.search)`；其余由系统以 `authorize_root` 执行，系统的 root 工作 actor 取自进程记录 |
| 2 | 进程关闭后不能再做阶段调用 | 关闭时撤销进程 context | 阶段调用都在执行器的运行与关闭流程之内，二者在进程注销之前结束（现有顺序）。生产中没有关闭之后的阶段调用路径，撤销从未被触发 |
| 3 | 停止或线程结束后，迟到的 actor 请求以 `ThreadEndedError` 拒绝 | 两道判定：线程 context 已撤销，线程表的存活判定 | 线程表的存活判定一道（线程未结束且进程仍在运行） |
| 4 | 线程只能由运行中的进程派生；子线程只能由存活的主线程派生，不能递归 CALL | 派生检查进程 context 的种类；线程表检查调用方是主线程 | 线程表按进程记录派生；主线程检查不变 |
| 5 | 子线程以被调用方身份运行：被调用方须在本 Workspace 准入，不能是 `system`，不能使用主线程标识 | 子线程 context 的派生经第 2 阶段准入 | 认证一侧的子工作 actor 准入；线程记录持有子工作 actor |
| 6 | CPU 不能以另一个线程的身份提交 | CPU 驱动只持有绑定了 context 的句柄；线程表按对象身份核对登记的 context（`thread.access is access`） | 线程句柄是不透明对象，线程表按对象身份核对（第 4.5 节）；句柄不含身份 |
| 7 | 派生或登记失败不留下凭据与登记 | 失败时撤销刚签发的 context | 失败时不登记，没有凭据需要撤销 |
| 8 | UPDATE 的基础冷读跨越线程结束或停止时不登记意图 | actor API 冷读之后，在登记前的同步段重新解析线程 | 不变：actor API 收到线程句柄，冷读之后在同步段重新解析 |
| 9 | 进程 context 不能做 actor 操作 | `_actor_grant` 检查运行种类，以 `ScopeRequiredError` 拒绝 | 类型不同：线程提交只接收线程句柄，请求级提交只接收工作 actor，系统阶段不经操作入口 |
| 10 | 线程不能借请求级提交绕过存活判定与派发话题 | `request_level=True` 只接受请求级 context | CPU 只持有线程句柄，取不到工作 actor（架构测试），无法调用请求级提交 |
| 11 | 请求级身份只在一次调用内有效 | 请求结束时撤销 | 工作 actor 是入口调用链内的局部值，调用返回即不再被引用 |
| 12 | 请求级提交只开放检索、读取、写入与更新；写入与更新须带派发话题 | 操作入口的两个请求集合 | 不变 |
| 13 | root API 只接受保留的 `system` | 每个 root 方法调用 `authorize_root` | 不变：每个 root 方法在入口调用 `authorize_root`，输入换成工作 actor |
| 14 | 导入与 MCP 写入前的落位不检查白名单、拒绝 `system`，并给出 principal 标识 | `authorize_import` 以 Agent 身份授权，返回 `ImportAuthorization` | 交互提交属于 root（owner，2026-11-05）：Import Bus 以系统的 root 工作 actor 授权，参与的 Agent 与 principal 标识作为数据。白名单不参与，空白名单的 Agent 仍可导入；`system` 作为参与者已由入口的声明解析以 `request.agent_reserved` 拒绝 |
| 15 | 进程控制只允许同一 Workspace 的请求方，不匹配时与进程不存在一样呈现为 `not_found` | `authorize_process_control` 比对两份 context 的驻留 Workspace；记录侧 context 已撤销或准入已失效时也返回不匹配 | 取消属于 root（owner，2026-11-05）：`authorize_root(请求方, 进程的 Workspace)`，拒绝呈现为 `not_found`。记录侧两个分支没有生产触发（注销流程中撤销与注销之间没有挂起点，访问登记在运行中不可变），删除 |
| 16 | 准入与白名单不缓存，每次授权即时查询 | `_redeem` 每次查访问登记 | 两个授权方法的共同前提，不变 |
| 17 | 网关关闭只拒绝新认证，已有进程仍可派生子线程 | 派生不检查网关关闭 | 子工作 actor 的准入不检查网关关闭，不变 |
| 18 | System 停止后不留下仍可使用的凭据 | 停止流程最后一步撤销全部 context；此时读取视图与资产库已关闭 | 删除：没有凭据；在此之前各组件已停止，在途请求由各组件的关闭拒绝 |
| 19 | 身份不被复制、序列化，不写入记录、事件与 DTO | 密封与架构测试 | 工作 actor 是普通值；架构测试限定它的构造与持有位置，并把“公开记录不保存身份”扩展到工作 actor |
| 20 | 授予内容只由授权者与认证一侧读取 | 架构测试 | 删除这一层，由第 19 行的持有限制代替 |
| 21 | 各授权方法限定调用模块，不能成为白名单的旁路 | 每个授权方法一条架构测试 | 两个方法各自限定调用模块，系统自身的 principal 只由任务进程与 Import Bus 使用（第 4.8 节） |
| 22 | 线程描述中的操作快照以该线程自己的身份读取 | `get_stage_operations(线程 context, 目标)` | 以线程记录的工作 actor 读取，不变 |
| 23 | 交互记录的发起者（`from_actor`）是参与的 Agent | Patchouli 与 Import Bus 编排器取 scope 的 actor | 参与者作为显式参数交给 Patchouli 与 Import Bus 编排器；scope 只提供归属（第 4.7 节） |

### 3.3 现有缺口

- **执行者与参与者混在一起**（第 1 节）：系统级操作以 Agent 身份授权，Profile 与附件受 Agent 可见性约束。
- **`IdentityScope(...)` 的构造位置没有限制**：
  - 授权者之外，生产代码中有资产桥接一处，是已登记的例外；
  - `core/models/workspace.py` 中两个辅助函数（`resolve_default_identity_scope`、`build_internal_identity_scope`）没有生产调用方，只有 4 个测试文件使用。
- **第 2、15、18 行的保证没有生产触发**，却各自带着撤销、记录侧检查与停止步骤。

## 4. 目标设计

### 4.1 owner 决定（2026-11-05，计划评审）

1. **只有两个授权方法**：`authorize_agent` 与 `authorize_root`。入口检查与操作授权做的是同一件事，与其在各个 root API 中手写检查，不如统一起来。不设一个带操作枚举的通用授权方法，也不设 `ReservedOperation`。
2. **操作授权只在 actor 执行期间生效**。其余检查之所以存在，是因为所有 API 都收拢到了 Workspace 能力层。能力层的 API 代表谁在执行这项操作。
3. **取消由 root 执行**：取消是用户下达的指令。
4. **Profile 与附件由 root 执行**：两者是用户的选择，对用户可见，而不是对 Agent 可见，不能以错误的身份执行。
5. **Gateway 与 cleanup 由 root 执行**：它们面对的话题属于 Workspace，与 Agent 无关。
6. **prepare 的检索以执行者的 actor 操作进行**：要按进程 actor 的可见性检索。
7. **交互提交是系统级操作，属于 root**。三个来源都在这一范畴：
   - controller 模式的交互提交，即现在 chat 的 finalize；
   - plugin 模式的导入；
   - 将来的历史对话导入。

   Agent 不能拥有这项能力。外部 harness 不能声明 `system`，管的是边界上的声明，与系统执行交互提交无关。
8. **系统组件以自己登记的 principal 经进程内 adapter 认证 `(user, system)`**，取得执行 root 操作的身份，与 Profile 注入脚本的做法相同（选项 a）。
9. **保留驻留身份与单次操作 scope 的区分，以及能力层的入口检查**。“认证直接产出 `IdentityScope`、删除入口检查”的最小版本被否决：它会退掉 v0.7.0 的大半成果。

### 4.2 工作 actor

- 新类型 `WorkingActor`（`core/access.py`）是冻结的值，字段为：
  - `principal`（`CallerPrincipal`）；
  - `actor`（`ActorIdentity`）；
  - `workspace`（驻留的 `WorkspaceIdentity`）。

  它就是现在的授予内容去掉运行绑定。只由认证一侧构造。
- 同一类型有四种角色：

  | 角色 | 来源 | 持有位置 |
  |:---|:---|:---|
  | 执行者（主工作 actor） | 注册入口认证 `(user, 具体 Agent)` | 进程记录与主线程记录 |
  | 子工作 actor | 线程表派生子线程时的准入 | 子线程记录 |
  | 系统的 root 工作 actor | 系统组件以系统自身的 principal 认证 `(user, system)` | 进程记录；Import Bus 的调用链 |
  | 请求的工作 actor | 管理路由、`/chat/stop`、导入与 MCP 入口的认证 | 入口调用链 |

- 不带“是否经过证明”：没有使用方。HTTP 管理面由管理凭据在入口证明，外部 principal 由网关第 1 阶段核对接入凭据。
- 不带 adapter、运行绑定与操作上限。

### 4.3 认证与系统自身的 principal

- `ActorAuthenticationGateway.authenticate(adapter, principal, credential, actor, workspace)` 返回 `WorkingActor`，不再接收运行绑定。两个阶段的检查与拒绝语义不变。
- **系统自身的 principal**（owner 决定选项 a）：
  - 随仓库发布的 `configs/system_principals.yaml` 增加一条登记：`local` adapter，允许声明 `system`，不带凭据。暂名 `hivememory:system`。
  - 它的标识与 `local` 等 adapter 标识一样，定义为 `system/access/` 中的常量。
  - 任务进程注册时用它认证一次，得到系统的 root 工作 actor；Import Bus 每次导入或落位时同样认证一次。
  - `(user, system)` 的准入沿用访问登记中已有的 `system` 记录，不新增 Workspace 访问登记。
- **子工作 actor 的准入**：
  - `core/access.py` 中的端口 `admit_sub_actor(main: WorkingActor, agent_id) -> WorkingActor` 取代 `ThreadContextLifecycle`。认证网关实现它，转交 `WorkspaceAuthenticator`。
  - 子工作 actor 沿用主工作 actor 的 principal 与 Workspace，只换 Agent。它拒绝 `system`，不检查网关关闭。
- **删除**：签发登记（`WeakSet`）、派生、撤销与 `revoke_all_contexts`。

### 4.4 两个授权方法

两个方法都属于 workspace 的操作授权者 `WorkspaceOperationAuthorizer`（`workspace/authorization.py`）。这个类保留现有的定位：

- 它无状态，只依赖 Workspace Actor 访问登记；
- 它不导入认证一侧、任务进程与能力层；
- 组合根把它注入 root API、actor API、任务进程与 Import Bus 服务。

本计划只替换它的方法：现有的七个授权方法与一个查询，换成下表的两个授权方法与一个只读查询 `allowed_operations`。

两个方法都返回 `IdentityScope`，共用三项前提，按顺序检查：

1. 准入记录有效，否则 `ScopeRequiredError`（`actor_not_admitted`）；
2. 目标等于驻留 Workspace，否则 `OperationDeniedError`（`target_workspace_not_resident`）；
3. actor 的用户是目标 Workspace 的 owner，否则 `OperationDeniedError`（`target_owner_mismatch`）。

| 方法 | 规则 | 拒绝 | 调用位置 |
|:---|:---|:---|:---|
| `authorize_agent(actor, operation, target)` | `operation` 是六项 `WorkspaceOperation` 之一，且在访问登记的白名单中 | `operation_not_allowed` | actor API 的方法入口；任务进程中 prepare 的检索 |
| `authorize_root(actor, target)` | actor 是 `system` | `root_required` | root API 的方法入口；任务进程的系统阶段与取消；Import Bus |

- `WorkspaceOperation` 的成员与用法不变：每个 actor API 方法仍在自己的入口写明所需操作。
- `allowed_operations(actor)` 取代 `get_stage_operations`，是只读查询，供线程描述与 prepare 判断是否检索，不授予权限。
- 删除：
  - `authorize_operation`，并入 `authorize_agent`；
  - `authorize_actor_operation` 与 `request_level`；
  - `authorize_stage`、`authorize_import` 与 `ImportAuthorization`；
  - `authorize_process_control`、`get_thread_binding`；
  - 导入的 `agent_required` 拒绝原因。

### 4.5 运行：进程、线程与线程句柄

- **进程记录**：`ProcessRecord.access` 拆为两个字段：
  - `ProcessRecord.actor`：执行者；
  - `ProcessRecord.system_actor`：系统的 root 工作 actor。

  两者都在注册时认证，任一失败都不创建进程。
- **线程句柄**：`ThreadHandle`（`workspace/contracts/threads.py`）是线程表派生时创建的不透明对象。
  - 它携带 `process_id` 与 `thread_id` 用于定位，按对象身份比较，不含身份。
  - 线程表按标识找到线程后核对 `thread.handle is handle`；标识相同的另一个句柄以接线缺陷拒绝。
- **线程对象**：`ProcessThread` 持有句柄、工作 actor、状态、父线程与描述；`ExecutionThread` 改为 `(descriptor, handle)`。
- **解析**：`resolve(handle)` 判定存活，返回 `ResolvedThread`。在现有的进程标识、线程标识与派发话题之外，增加线程的工作 actor。
- **结束与存活**：结束线程只标记已结束。存活判定不变：线程未结束，且进程仍在运行。
- **派生**：登记成功前不留下任何状态；子线程的准入经第 4.3 节的端口。

### 4.6 能力层与任务进程

- **操作入口**：
  - 线程提交：`execute`、`derive_child_thread`、`end_thread` 接收线程句柄。`ThreadOperationSubmitter` 持有句柄，替代 context。
  - `execute` 先解析句柄，再把线程的工作 actor 与句柄交给 actor API。
  - 请求级提交：`execute_request(request, *, actor, dispatch_topic_id=None)` 把请求的工作 actor 交给 actor API。开放的请求类型与派发话题规则不变。
- **actor API**：
  - 六个方法接收工作 actor，在入口调用 `authorize_agent`，目标取驻留 Workspace。
  - 写入与更新另接收线程句柄或派发话题，二者恰有其一，并保留登记前同步段的重新解析（第 3.2 节第 8 行）。
- **CALL**：
  - 入口解析调用方句柄，actor API 以调用方授权 `agent.call`，按调用方的可见性解析图纸。调用方只能唤起它看得见的 Agent，这是 Agent 自己的操作。
  - 线程表随后重新判定调用方存活、确认它是主线程，再派生子线程。
- **root API**：20 个方法以工作 actor 取代访问 context，在入口调用 `authorize_root`。目标仍由请求给出。
- **任务进程的系统阶段**：

| 阶段 | 授权 | 说明 |
|:---|:---|:---|
| Gateway | `authorize_root(system_actor, 目标)` | Gateway 只用 scope 中的 Workspace；分析上下文中的 `identity` 字段没有读取方 |
| prepare | 话题：`authorize_root(system_actor, 目标)`。检索：执行者持有 `memory.search` 时 `authorize_agent(actor, memory.search, 目标)`，否则不检索 | prepare 路由接收 root scope，另接收一个可为空的检索 scope；话题与归属取 root scope，检索与可见性取检索 scope |
| Profile 解析 | `authorize_root(system_actor, 目标)` | 加载哪份图纸取自执行者的 `agent_id`，来自认证结果而不是请求输入；读取按管理语义，只检查 Workspace 归属，不受 Agent 可见性约束（第 4.9 节） |
| 附件租借 | `authorize_root(system_actor, 目标)` | 资产库只核对 Workspace |
| finalize | `authorize_root(system_actor, 目标)` | 参与者为执行者，显式传给 Patchouli（第 4.7 节） |
| cleanup | `authorize_root(system_actor, 目标)` | 补偿 prepare 新建且仍为空的话题 |
| 取消 | `authorize_root(请求方, 进程的 Workspace)` | 在注册入口的取消方法中；拒绝呈现为 `not_found` |

### 4.7 交互提交

- **执行者是系统，参与者是数据**：
  - Patchouli 的 finalize 路由与 `INTERACTION_SUBMIT` 路由都增加显式的参与者参数；
  - `from_actor` 取自参与者，scope 只提供归属；
  - prepare 的检索可见性取自检索 scope。
- **root API 增加“提交交互”**（候选设计）：
  - 它以 `authorize_root` 授权，再调用 Patchouli 的 `INTERACTION_SUBMIT`。
  - Import Bus 与将来的历史导入经它提交，由组合根注入编排器。
  - 主动链路的 finalize 依赖本轮 prepare 的结果，走 Patchouli 的 finalize 路由，仍由任务进程发起（进程不导入能力层），同样以 `authorize_root` 授权。
- **Import Bus**：
  - 导入与落位都先以请求声明的 principal、凭据与 `(user, 具体 Agent)` 认证。这一步证明请求来自哪个 principal，参与者是哪个 Agent。
  - 再以系统自身的 principal 认证 `(user, system)`，经 `authorize_root` 授权。
  - 编排器的 Gateway 路由、封口与提交都在 root scope 下进行。外部会话键、交互标识与事件中的 Agent 取自显式的参与者与 principal 标识。

### 4.8 入口与结构守护

- **入口**：
  - server：
    - 管理路由的依赖以 server 的 principal 与 `(user, system)` 认证，把工作 actor 与目标交给 root API；
    - `/chat/stop` 的依赖交出工作 actor；
    - `RequestAccess` 与请求结束时的撤销删除。
  - MCP 入口：每次工具调用认证一次，得到局部的工作 actor，交给操作入口的请求级提交；写入与更新先由 Import Bus 落位，顺序不变。
  - 导入路由：认证后交给 Import Bus 服务。
  - System 停止：删除 `revoke_all_contexts` 这一步。
- **新增的架构测试**：
  - `IdentityScope(...)` 只在授权者与资产桥接（已登记的例外）中构造；删除第 3.3 节的两个辅助函数，使用它们的测试改为直接构造。
  - `WorkingActor(...)` 只由认证一侧构造。Alice、`agent_runtime`、Patchouli、Gateway 与 CPU 可见的对象都不引用它。
  - `authorize_agent` 只在 actor API 与任务进程中调用；`authorize_root` 只在 root API、任务进程与 Import Bus 服务中调用。
  - 系统自身的 principal 常量只由任务进程的注册入口与 Import Bus 服务使用。
- **改写的架构测试**：
  - “CPU 可见的对象不含 context”改为不含工作 actor 与 `IdentityScope`；
  - “只有 CPU 驱动接触线程凭据”改为只有 CPU 驱动接触线程句柄；
  - “公开记录与物化任务不保存 `IdentityScope`”扩展到工作 actor；
  - “能力层、任务进程与 Import Bus 的公共方法不接收调用方组装的 `IdentityScope`”保留本意：能力层接收工作 actor，由自己授权。
- **`test_access_boundaries.py` 现有 23 条测试的去向**，逐条映射在实施时写进 PR 描述：
  - 保留约 11 条：包间导入方向、actor API 只经操作入口、请求级提交只供 MCP 入口、Alice 不引用资源路由与 `IdentityScope` 等；
  - 改写约 10 条：阶段授权与导入授权的两条并入上面按方法限定的测试，其余如上；
  - 删除 2 条：线程绑定查询只供线程表、授予内容只由授权者与认证一侧读取。两者守护的对象已不存在。

### 4.9 Profile 的管理读取

- CPU 分配以 root scope 解析执行者的图纸时，`ProfileResolver` 按管理语义交付：只检查 Workspace 归属，不按 scope 的 actor 检查可见性。缓存命中与冷读两条路径都要如此。
- CALL 解析被调用方的图纸仍按调用方的可见性交付。
- 管理语义与 root API 读取记忆时相同（`memory_is_readable` 的 `enforce_actor_visibility=False`）。

### 4.10 相对 Idea 候选设计的调整

以下都是候选设计，评审时可调整。

1. **一种类型**：工作 actor 是同一类型 `WorkingActor` 的四种角色，不设多个类型。各角色字段相同，区别只在来源。
2. **不带证明字段**：工作 actor 不带“是否经过证明”。它没有使用方，管理凭据已在入口完成 HTTP 管理面的证明。
3. **不透明句柄**：线程句柄是按对象身份比较的不透明对象，不是 `(process_id, thread_id)` 的值引用。值引用会让 CPU 一侧以另一个线程的标识提交，例如子线程冒用主线程的身份与白名单；现在这一点由按对象身份比较 context 守住（第 3.2 节第 6 行）。
4. **MCP 写入顺序不变**：保持先落位、后授权写入。落位由系统执行，与 Agent 的许可无关；现状已写入 MCP 契约。
5. **删除进程控制的记录侧分支**：两个分支没有生产触发（第 3.2 节第 15 行）。
6. **授权留在能力层方法的入口**：root API 与 actor API 的方法接收工作 actor，在入口授权，与现在授权点的位置相同。不把授权提前到入口适配层，也不需要“请求类型 → 操作”的对应表。
7. **prepare 带可为空的检索 scope**：prepare 仍是一次调用，AGENTS.md 中“Patchouli prepare（话题与检索）”的流程不变量不变。执行者没有 `memory.search` 时不检索，对话照常进行。

### 4.11 主要流程的变化

以一个场景推演，场景来自 Idea 第 5 节：

1. 前端用 coder 对话，coder 先检索，再派生子线程运行 reviewer；
2. reviewer 写入一条记忆；
3. 用户中途停止时，reviewer 还有一个读取请求在路上；
4. 同时，dsh 经 MCP 写入一条记忆，Claude Code 导入一轮对话。

| 流程 | 现在 | 本计划之后 |
|:---|:---|:---|
| chat 注册 | 网关签发进程 context，存进进程记录 | 以 server 的 principal 认证执行者 `(default, coder)`，以系统自身的 principal 认证 `(default, system)`，两者存进进程记录 |
| Gateway、话题准备、Profile、附件、cleanup | `authorize_stage(进程 context, 目标)`，以 coder 的身份 | `authorize_root(系统的 root 工作 actor, 目标)`；Profile 按管理语义读取 coder 的图纸 |
| prepare 的检索 | 同一个阶段 scope，按 coder 的可见性 | coder 持有 `memory.search`，以 `authorize_agent(coder, memory.search, 目标)` 取得检索 scope |
| 进入 Actor 阶段 | 线程表从进程 context 派生主线程 context，连同描述交给 CPU | 线程表登记主线程记录（执行者、运行中），把描述与线程句柄交给 CPU |
| coder 检索 | 入口判定存活；actor API 以 `authorize_actor_operation` 确认 context 种类，兑现并授权 | 入口按句柄判定存活并取出 coder，actor API 在入口 `authorize_agent(coder, memory.search, 目标)` |
| 派生 reviewer | 以 coder 授权 `agent.call` 并读取图纸；从进程 context 派生子线程 context，为 reviewer 做准入 | 以 coder 授权 `agent.call`，按 coder 的可见性读取图纸；线程表为 reviewer 做准入，登记子线程记录 |
| reviewer 写入 | 子线程 context 授权 `memory.write`，同步段再解析一次线程取得话题 | 按句柄取出 reviewer，actor API 授权 `memory.write`，同步段以句柄再解析一次线程取得话题 |
| 子线程结束 | 撤销子线程 context | 线程记录标为已结束 |
| 用户停止 | 以 `(default, system)` 取得请求级 context，比对两份 context 的驻留 Workspace；进程记为收到停止，全部线程即视为已结束，线程 context 到 Actor 阶段收尾时才撤销 | 以 `(default, system)` 认证得到工作 actor，`authorize_root(…, 进程的 Workspace)`；进程记为收到停止，全部线程即视为已结束 |
| reviewer 迟到的读取 | 存活判定看到进程已收到停止而拒绝；Actor 阶段收尾之后，已撤销的 context 也会拒绝 | 存活判定看到进程已收到停止而拒绝，一道判定 |
| finalize | `authorize_stage`，Patchouli 取 scope 中的 coder 作为 `from_actor` | `authorize_root(系统的 root 工作 actor, 目标)`，coder 作为参与者显式交给 Patchouli |
| 进程关闭 | 撤销进程 context，注销进程 | 注销进程 |
| dsh 的 `memory_write` | 请求级 context：先以 `authorize_import` 落位，再以 `request_level=True` 授权 `memory.write` | 认证得到局部的工作 actor `(harness:dsh, (default, dsh), main_workspace)`；Import Bus 以系统的 root 工作 actor 落位，dsh 作为参与者；actor API 以 `authorize_agent(dsh, memory.write, 目标)` 授权并登记 |
| Claude Code 导入 | 请求级 context；`authorize_import` 以 claude_code 的身份 | 认证 claude_code 证明来源与参与者；Import Bus 以系统的 root 工作 actor 经 root API 的“提交交互”提交 |
| 管理路由 | 请求级 context；`authorize_root` | 认证得到工作 actor，root API 在入口 `authorize_root` |

## 5. 受影响的系统、契约与数据

- **core**：
  - `access.py` 改写：增加 `WorkingActor` 与子工作 actor 准入端口；删除 context、授予内容、运行绑定与线程 context 生命周期端口。
  - `models/workspace.py` 删除两个辅助函数。
- **workspace**：
  - 认证一侧、授权者；
  - 任务进程：注册入口、进程记录、线程表、执行器、CPU 分配；
  - 能力层：操作入口、actor API 的 6 个方法、root API 的 20 个方法与新增的“提交交互”；
  - `resolution/profile.py` 的管理读取；
  - `contracts/threads.py`、`contracts/operations.py`、`thread_ports.py`。
- **Patchouli**：
  - prepare 路由增加可为空的检索 scope；
  - finalize 与 `INTERACTION_SUBMIT` 增加显式参与者；
  - 公共边界的拆分改为归属取自 scope、发起者取自参与者。
- **Gateway**：不变，收到的 scope 的 actor 变为 `system`。
- **system**：
  - `system/access/` 增加系统自身 principal 的常量；
  - Import Bus 服务与编排器改用 root 授权与显式参与者；
  - 组合根的装配；
  - 停止流程删除一步。
- **server**、**mcp_server**：取得身份的方式（第 4.8 节）。
- **alice**：`interface/cpu.py` 以执行线程构造线程句柄的方式不变，类型随契约变化；frame、CALL 编排与 MTP 适配器不变。
- **配置**：随仓库发布的 `configs/system_principals.yaml` 增加系统自身的 principal。
- **契约**：
  - HTTP、MCP 与导入的请求和响应不变。
  - 错误模型：`ScopeRequiredError` 的 `context_not_issued`、导入的 `agent_required` 两个原因不再出现，其余错误码、原因与 HTTP 状态不变。
  - 运行时事件 `system.stopped` 的步骤列表少了 `workspace_access_contexts.revoke`。
- **数据**：不涉及持久化格式；访问 context 从未持久化，交互记录的 `from_actor` 仍是参与的 Agent。

## 6. 兼容与迁移

- **配置迁移**：自定义过 `configs/system_principals.yaml` 的部署要补上系统自身的 principal。否则 chat 注册与导入在认证时以 `unknown_principal` 拒绝。帮助文档写明这一点，做法与此前 `allow_system_actor` 的升级说明相同。
- **行为变化**（生产中看不出差别）：
  - 非 PUBLIC 的图纸可以被自己的进程加载；
  - 没有 `memory.search` 的 Agent 不再自动检索。现在所有 chat 的 Agent 都有这项许可。
- **一次性切换**：内部接口一次性切换，不保留新旧两套身份并存的过渡期，因为并存正是本计划要消除的“层层概念”。
- **实施顺序**：按以下顺序组织提交，便于评审；中间提交可以不全绿，合并时全绿。
  1. 认证与授权：工作 actor、两个授权方法、子工作 actor 准入、系统自身的 principal；
  2. Patchouli 的契约：prepare 的检索 scope，finalize 与交互提交的显式参与者；
  3. 运行与能力层：进程记录、线程表与线程句柄、操作入口、actor API 与 root API 的签名、系统阶段、Profile 的管理读取；
  4. 入口：server、MCP、Import Bus 与 System 停止；
  5. 结构守护与测试；
  6. 文档收尾。

## 7. 测试和验收标准

需要守住的不变量与归属层：

| 不变量 | 归属层 |
|:---|:---|
| 两个授权方法的规则与三项共同前提，以稳定原因拒绝 | `tests/unit/workspace/test_access.py`（改写，删除密封、复制、序列化与撤销的用例） |
| 认证返回工作 actor；网关关闭拒绝新认证；子工作 actor 的准入拒绝 `system` 与未准入的 Agent，网关关闭后仍可准入；系统自身的 principal 未登记时注册与导入被拒 | `tests/unit/system/access/test_gateway.py`、`tests/unit/workspace/test_access.py` |
| 存活只有一个判定；标识相同的另一个句柄被拒；只有主线程能派生子线程；派生失败不留登记 | `tests/integration/workspace/test_process_threads.py`（并入两个以线程 context 命名的测试文件中仍然成立的用例） |
| UPDATE 冷读跨越线程结束或停止时不登记意图 | 现有用例保留 |
| 系统阶段以 root scope 执行；finalize 与导入提交的交互，`from_actor` 是参与的 Agent | 任务进程与 Patchouli 的现有测试改写，Import Bus 的集成测试 |
| 执行者没有 `memory.search` 时 prepare 不检索、对话照常进行 | 任务进程的单元测试 |
| 指向其他 Agent 的 PRIVATE 图纸能被自己的进程加载；CALL 仍按调用方的可见性解析被调用方 | `tests/integration/workspace/` 中 Profile 解析的现有测试 |
| 空白名单的 Agent 仍可导入 | 现有 e2e 用例保留 |
| 请求级提交只开放四类请求；写入与更新须带派发话题；MCP 每次工具调用认证一次 | `tests/integration/workspace/test_operation_requests.py`、`tests/integration/system/test_mcp_memory_access.py` |
| 取消对其他 Workspace 的请求方呈现为 `not_found` | `tests/unit/workspace/process/test_chat_run_control_contract.py` |
| 第 4.8 节的结构守护 | `tests/unit/architecture/test_access_boundaries.py`、`test_resource_identity_boundaries.py` |
| HTTP、导入与 MCP 的对外行为不变 | 现有路由测试与集成测试随签名调整，断言不变 |

- 只守护被删除概念的测试随契约删除，不写“某类型已不存在”的墓碑测试。涉及：密封、复制与序列化；撤销与 `revoke_all`；`request_level`；context 种类检查；导入授权；`tests/unit/core/test_access_binding.py`。
- 手动验收：用本分支重启正在使用的后端，逐项确认：
  1. 前端对话；有 `agent.call` 的 Agent 能派生子线程；
  2. 中途停止有效；
  3. 管理页面正常；
  4. dsh 与 Claude Code 的检索、写入与对话回流正常。

验证命令：

```bash
pytest tests/unit/workspace tests/unit/architecture tests/integration/workspace -q
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85
ruff check . && black --check .
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
```

`tests/e2e/system` 在 Qdrant 与模型可用时运行一次。

## 8. 风险与待决问题

- **改动面大**：
  - 能力层 26 个方法加 1 个新方法；Patchouli 的三个路由；约 28 个测试文件；约 40 份文档提到访问 context。
  - 缓解：以第 3.2 节的清单逐行核对实现与评审，文档收尾按第 9 节的清单进行。
- **Patchouli 契约变更**：prepare、finalize 与交互提交的签名变化，影响 Patchouli 的公共边界与 Import Bus 编排器。
  - 交互记录的内容不变：`from_actor` 仍是参与的 Agent。
  - 用例：以“`from_actor` 是参与者”的断言守住。
- **认证次数增加**：任务进程注册与每次导入各多一次认证。认证是进程内的登记查询，没有 I/O。
- **密封的取消**：工作 actor 是普通值，进程内防误用改由持有与构造位置的架构测试承担。在单个 Python 进程内，区分特权最终都要靠结构纪律（Idea 第 5 节的分析）。
- **待决**：无阻塞问题。可在评审时调整的有：
  - `WorkingActor`、`ThreadHandle`、`hivememory:system` 三个名称；
  - root API 中“提交交互”的方法形态（第 4.7 节）。

## 9. 完成时必须更新的当前文档

- **workspace**：
  - [认证与访问登记](../workspace/authentication.md)：第 1–7 节改写为工作 actor 与系统自身的 principal，删除 context 的密封、种类与生命周期；
  - [身份与授权](../workspace/identity-and-authorization.md)：第 2–4、6–8 节，包括执行者、参与者与两个授权方法；
  - [执行线程](../workspace/execution-threads.md)：线程句柄与存活判定；
  - [任务进程](../workspace/task-process.md)：注册时的两次认证与系统阶段；
  - [能力层](../workspace/capability.md)：两个授权方法、root API 的“提交交互”与操作入口；
  - [读取视图与写入意图](../workspace/read-view-and-write-intents.md)：Profile 的管理读取；
  - [Workspace](../workspace/README.md)：停止流程。
- **Patchouli**：[Patchouli](../patchouli/README.md)：prepare 与 finalize 的身份。
- **契约**：
  - [子系统公共契约](../contracts/subsystem-contracts.md)：访问上下文与 Patchouli 公共路由；
  - [路由与事件契约](../contracts/routes-and-events.md)第 6 节；
  - [MCP 记忆访问](../contracts/mcp.md)第 2、4 节；
  - [错误模型](../contracts/error-model.md)第 4.4 节。
- **system**：
  - [Import Bus](../system/import-bus.md)：root 授权与显式参与者；
  - [组合根](../system/composition.md)、[应用服务](../system/application-services.md)、[System](../system/README.md)；
  - [System 配置](../system/configuration.md)：系统自身的 principal。
- **帮助**：[配置指南](../help/configuration.md)第 5.1 节：系统自身的 principal 与升级说明。
- **架构**：[系统架构概览](../architecture/overview.md)、[边界](../architecture/boundaries.md)、[数据模型](../architecture/data-model.md)。
- **治理与项目**：[身份治理](../governance/security/identity-and-execution-safety.md)、[PROJECT](../PROJECT.md)。
- **Alice 文档**：凡提到线程 context 的段落，例如 MTP 运行时、Agent 运行时与编排。
- **`AGENTS.md`**：
  - 第 3 节“能力层”“任务进程与执行线程”“身份数据”三条约束；
  - 第 4 节 Import Bus 链路中的“导入授权”。
- **收尾时**：本计划归档；Idea 第 5 节、[Plans 导航](./README.md)与 ROADMAP 第 4.2 节“身份与访问体系”一行同步。
