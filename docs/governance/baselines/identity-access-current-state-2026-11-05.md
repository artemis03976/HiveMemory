---
title: 身份与访问体系现状调查（2026-11-05）
status: baseline
owner: project
scope: identity-and-access-current-state-entries-authorization-points-resource-visibility-and-patches
snapshot_at: 2026-11-05
source_commit: 8c66f594b094d6132c4434d207354ed878d2c929
code_paths:
  - src/hivememory/core/access.py
  - src/hivememory/core/models/identity.py
  - src/hivememory/core/models/workspace.py
  - src/hivememory/core/memory_access.py
  - src/hivememory/system/access/
  - src/hivememory/workspace/authentication.py
  - src/hivememory/workspace/authorization.py
  - src/hivememory/workspace/registry.py
  - src/hivememory/workspace/capability/
  - src/hivememory/workspace/process/
  - src/hivememory/workspace/resolution/
  - src/hivememory/workspace/intents/
  - src/hivememory/server/deps.py
  - src/hivememory/server/routers/
  - src/hivememory/mcp_server/server.py
  - src/hivememory/system/application/import_bus_service.py
  - src/hivememory/system/services/
  - src/hivememory/patchouli/application/
  - src/hivememory/alice/orchestration/verbs.py
  - frontend/src/services/identity.ts
  - configs/system_principals.yaml
  - configs/workspace_actors.yaml
  - scripts/inject_profiles.py
related_docs:
  - docs/ideas/identity-and-access-model.md
  - docs/governance/security/identity-and-execution-safety.md
  - docs/workspace/authentication.md
  - docs/workspace/identity-and-authorization.md
  - docs/workspace/capability.md
  - docs/workspace/execution-threads.md
  - docs/contracts/mcp.md
  - docs/system/import-bus.md
  - docs/archive/todo/local-http-management-access.md
  - docs/todo/workspace-asset-ownership-identity-split.md
  - docs/governance/baselines/task-process-concept-inventory-2026-10-28.md
last_reviewed: 2026-11-05
---

# 身份与访问体系现状调查

本文冻结 `source_commit` 上身份与访问体系的现状：每个入口怎样取得身份，每个授权点检查什么，资源授权在生产数据下实际起什么作用，以及 v0.7.0 以来留下了哪些补丁与例外。最新行为仍以代码、测试与当前设计为准。

**起因**：owner（2026-11-05）把身份与访问体系定为单独的主线，见[身份与访问体系 Idea](../../ideas/identity-and-access-model.md)。这条线上的设计分布在多份事实文档里，例外散落在 Idea、Todo 与治理文档中。没有一份清单能说明体系现在的样子，也无法区分哪些是设计、哪些是补丁。本文对应 owner 候选清单中的 A-01。

**与旧基线的关系**：原 Phase S0 身份与威胁模型清单冻结于 2026-08-07，早于 W0 Workspace（2026-09-01）。它记录的 L1 共享缓存、进程级 PendingAtom store、裸 `user_id` 与服务层硬编码均已不存在，因此整体删除，最后版本见 commit `8c66f594`。本文不是跨用户威胁清单，而是为收敛与补全这条线提供现状证据。

## 1. 范围与方法

- **范围**：调用方证明与 Workspace 准入（第 1、2 阶段），操作授权（第 3 阶段），资源授权与可见性（第 4 阶段），后台任务携带的身份，进程级共享状态，前端身份，以及两份访问登记。
- **使用方**：只认 `src/`、`frontend/src/` 与 `scripts/` 中的生产调用，测试不算使用方。
- **方法**：只读代码，不以事实文档为证据。HTTP 与 MCP 入口、能力层与任务进程、Patchouli 与 Gateway 公共路由、共享状态与前端四部分并行盘点，关键结论逐条回到代码核对。
- **类别**（第 11 节使用）：

  | 类 | 含义 |
  |:---|:---|
  | 设计 | 承重，有明确的设计理由，事实文档已描述 |
  | 补丁 | 为后来的需求加进已有模型，没有改写模型本身 |
  | 例外 | 已登记、与通用规则不一致的做法 |
  | 休眠 | 结构存在，但在当前部署与数据下从不改变结果 |
  | 无生产使用方 | 只有定义或只有测试调用 |
  | 不一致 | 注释、契约说明与代码不符 |

## 2. 结论

- **内部纪律很严，外部边界很松**。进程内有四个访问阶段、三种运行绑定、七个授权方法，由 40 条架构测试守护（第 4、6、10 节）。HTTP 管理接口却不证明调用方，主配置、提供商密钥、模型注册表、日志与运行时事件的接口完全不经认证网关（第 5.2 节）。
- **能区分结果的检查只有少数几项**。生产中只有一个用户、一个 Workspace，所有记忆都以 PUBLIC 创建，没有任何路径给 actor 赋 `team_id`。owner 检查、目标等于驻留 Workspace 的检查和 PRIVATE/TEAM 可见性在生产中恒通过。真正起作用的是外部 harness 的凭据、`system` 的声明限制、root 与 actor 两个调用面的分离，以及线程存活判定（第 7.2 节）。
- **Agent 身份是声明，没有唯一来源**。Agent 可以是访问登记里的字符串、Workspace 里的 Profile 原子，或者只是 harness 的标签。用户级记录会接纳任意字符串并给它全部六项操作，外部凭据也不与 Agent 绑定（第 4.3 节）。以 Agent 为粒度的规则（PRIVATE 目标、按 Agent 的白名单、意图的发起者）都建立在未经证明的字符串上。
- **碎片化来自“加而不改”**。约九周内有 16 份计划改动过这条线（第 3 节）。新需求多以新增一种 context、一个授权方法或一个参数的方式进入，例如导入授权、请求级 context、`request_level` 参数，已有模型很少被改写。`system` 因此有五种用法（第 4.2 节）。
- **补丁与例外 11 项，休眠结构 6 项，无生产使用方的身份代码 9 处**（第 11 节）。已登记的例外只有两项：WorkspaceAsset 的旧 scope 桥接与 Profile 注入脚本。

## 3. 演进：约九周、16 份计划

以下计划都已归档（[Archived Plans](../../archive/plans/README.md)），按完成日期排列。

| 完成 | 计划 | 对这条线的改动 |
|:---|:---|:---|
| 2026-09-01 | [W0 Workspace MVP](../../archive/plans/v0.6.2-workspace-mvp.md) | `WorkspaceIdentity` 与 `main_workspace`；资源按 Workspace 复合键寻址 |
| 2026-09-06 | [用户身份上下文与裸 user_id 收敛](../../archive/plans/v0.6.2-identity-projection-cleanup.md) | 身份只从请求头解析，删除裸 `user_id` 兼容投影 |
| 2026-09-13 | [Workspace Runtime 与缓存所有权迁移](../../archive/plans/v0.6.2-workspace-runtime-cache-migration.md) | 派生缓存按 Workspace 分区并迁入 workspace |
| 2026-09-19 | [A1 Workspace 访问边界与授权](../../archive/plans/v0.7.0-a1-workspace-access-boundary.md) | 统一认证网关、两类登记、逐次操作授权；没有接上生产入口 |
| 2026-10-04 | [A1 访问边界返工](../../archive/plans/v0.7.0-a1-access-boundary-rework.md) | 第一版暴露出 context 与 `IdentityScope` 的边界错误；改为两阶段认证加两阶段授权、密封 context |
| 2026-10-04 | [身份与访问体系第二批](../../archive/plans/v0.7.0-identity-access-batch-2.md) | Patchouli 内部拆为 `belong_to` 与 `from_actor`；WorkspaceAsset 当场暂缓，成为例外 |
| 2026-10-07 | [写入意图登记迁入 workspace](../../archive/plans/v0.7.0-intent-registry-and-read-cache.md) | 意图按 Workspace 硬边界回读，不设 policy |
| 2026-10-09 | [Alice 的能力层调用迁移](../../archive/plans/v0.7.0-alice-capability-migration.md) | Alice 的资源操作改经能力层，删除过渡执行身份 |
| 2026-10-10 | [能力层分层](../../archive/plans/v0.7.0-capability-surface-split.md) | root API 与 actor API 分开；阶段调用不再借用 actor 操作 |
| 2026-10-11 | [Agent 操作白名单](../../archive/plans/v0.7.0-agent-operation-whitelist.md) | 白名单成为 Agent 权限唯一来源，Profile 不再承载权限；系统工具不设权限 |
| 2026-10-11 | [线程 context 取代执行凭据](../../archive/plans/v0.7.0-thread-context.md) | 进程 context 与线程 context 两分；同期引入的执行凭据被删除 |
| 2026-10-11 | [执行线程层](../../archive/plans/v0.7.0-execution-thread-layer.md) | CALL 子线程以被调用方身份运行 |
| 2026-10-19 | [执行线程的正式定义](../../archive/plans/v0.7.0-execution-thread-definition.md) | 计划自述：执行线程“是为 actor 的操作权限逐步加进来的，从未被定义成一个对象” |
| 2026-10-21 | [Import Bus 重设计](../../archive/plans/v0.7.1-import-bus-redesign.md) | `import` adapter 与导入授权，成为能力层之外的第四个授权面 |
| 2026-10-29 | [记忆访问（MCP）](../../archive/plans/v0.7.1-memory-access-mcp.md) | 外部 principal 以共享密钥证明（P-1c）；`allow_system_actor`；不建进程的请求级 context（P-9c） |
| 2026-10-31 | [plugin 模式开放 WRITE 与 UPDATE](../../archive/plans/v0.7.2-plugin-write-update.md) | 请求级提交开放写入与更新；派发话题与 `request_level` 参数；写入前经导入授权落位 |

10 月 10 日到 11 日两天内完成了四份计划（能力层分层、白名单、线程 context、线程层）。10 月 9 日 Alice 能力层迁移引入的执行凭据，两天后被线程 context 取代。

## 4. 概念清单

### 4.1 身份形态

| 概念 | 定义 | 生产中的实际情况 | 类 |
|:---|:---|:---|:---|
| `ActorIdentity(user, agent, team?)` | [`identity.py`](../../../src/hivememory/core/models/identity.py) | 只在三处构造：声明解析、子线程派生、`system_actor_for_workspace`；`team_id` 从不赋值 | 设计；`team_id` 休眠 |
| `WorkspaceIdentity(owner, key, id)` | 同上 | 只有 `default` 的 `main_workspace`；HTTP 由用户推出 Workspace | 设计 |
| `IdentityScope(actor, workspace)` | 同上 | 只由授权者组装，另有资产桥接一处自行组装 | 设计；资产桥接是例外 |
| `CallerPrincipal` | [`core/access.py`](../../../src/hivememory/core/access.py) | 4 个登记：server、Profile 注入脚本、两个 harness | 设计 |
| `AccessGrant` 与密封的 `WorkspaceAccessContext` | 同上 | 授予内容：actor、驻留 Workspace、principal、运行绑定 | 设计 |
| `RunBinding`：任务进程、执行线程、请求 | 同上 | 进程 context 服务阶段调用，线程 context 服务 Alice 的 actor 操作，请求级 context 服务管理路由、取消、导入、MCP 与注入脚本 | 设计；请求级 context 的用途逐步扩展 |
| adapter：`http`、`local`、`mcp`、`import` | [`system/access/registry.py`](../../../src/hivememory/system/access/registry.py) | 进程内 adapter 不带凭据，外部 adapter 必须带 | 设计 |
| `ImportAuthorization` | [`authorization.py`](../../../src/hivememory/workspace/authorization.py) | 导入与写入时落位共用 | 补丁（第四个授权面） |
| 线程描述中的操作快照 | `workspace/contracts/threads.py` | 只用于提示词与执行策略，不授权 | 设计 |
| `ExecutionLabels(agent_id, workspace_id)` | `core/models/execution.py` | 只用于观测与展示 | 设计 |
| `belong_to` 与 `from_actor` | Patchouli 内部、记录与后台任务 | 公共边界拆分后独立传递 | 设计 |
| 来源记录 `source_agent_id` 与贡献者集合 | `core/models/provenance.py` | 只用于展示，不参与授权（已核对，没有发现例外） | 设计 |

### 4.2 `system` 的五种用法

保留值 `system` 定义为“没有具体 Agent 作为操作来源主体”，现在承担五种角色：

1. **root API 的唯一 actor**：HTTP 管理路由与取消以 `(user, system)` 声明认证，`authorize_root` 只认它。
2. **Profile 注入脚本的 actor**：[`inject_profiles.py`](../../../scripts/inject_profiles.py) 以 `local` adapter 和 `system` 进入 root API。
3. **SETTLE 生成任务的发起者**：`system_actor_for_workspace` 构造，也是查重的可见性主体，只看得到 PUBLIC。
4. **结算记忆的来源**：`system_settlement_provenance` 把 `source_agent_id` 记为 `system`。
5. **资产物化桥接的 actor**：[`AssetMaterializationReader`](../../../src/hivememory/system/services/asset_materialization_reader.py) 在一次租借内以它组装 scope。

另有四处禁止它：访问记录不得给它配置操作，记忆策略不得以它为目标，导入授权与子线程派生拒绝它。第 1、2 种是授权角色，第 3、5 种是内部任务的发起者，第 4 种是数据。

### 4.3 Agent 身份的三个来源

| 来源 | 内容 | 谁依赖它 |
|:---|:---|:---|
| Workspace 访问登记 | `agent_id` 字符串，或省略 `agent_id` 的用户级记录 | 准入与白名单 |
| Agent Profile | Workspace 中的记忆原子，`agent_id` 即其 alias；内置 `omni_doll` 没有原子 | chat 的 CPU 分配与 CALL 目标；找不到自定义 alias 时显式失败 |
| harness 标签 | `claude_code`、`dsh`，只有精确访问记录，没有 Profile | 导入与 MCP |

- 用户级记录接纳任意 Agent 字符串，并给它全部六项操作。chat 会因为找不到 Profile 而失败，导入与 MCP 却不查 Profile。
- principal 不与 Agent 绑定：持有 harness 凭据的调用方可以在 `X-Agent-ID` 中声明任何 Agent，包括另一个 harness 的标签或某个 Profile 的 alias。
- server 的 principal 同样不限制 Agent：前端 chat 可以选择 `claude_code` 或 `dsh` 作为执行 Agent。

## 5. 入口与认证

### 5.1 经认证网关的入口

| 入口 | principal 与 adapter | 凭据 | 声明 | context | 授权面 | 生产中的实际约束 |
|:---|:---|:---|:---|:---|:---|:---|
| 管理路由 20 个：agents、memories、memory-tasks、sessions、topics、workspace/assets | server，`http` | 无 | (请求头用户，`system`) | 请求级，`get_request_access` | root | 能连到端口即以 `system` 进入；缺省用户回退 `default` |
| `POST /chat` | server，`http` | 无 | (请求头用户，请求体 Agent) | 进程，由 `TaskProcessService.register_process` 认证 | 阶段；线程的 actor 操作 | 任意 Agent 字符串经用户级记录取得六项操作；需要存在对应 Profile |
| `POST /chat/stop` | server，`http` | 无 | (用户，`system`) | 请求级 | 进程控制 | 只比对 Workspace，不比对 Agent |
| `POST /import/turns` | 请求声明，`import` | Bearer | (用户，请求体 Agent) | 请求级 | 导入（不读白名单） | 凭据有效即可声明任意 Agent |
| `POST /mcp` 的 4 个工具 | 请求声明，`mcp` | Bearer | (用户，`X-Agent-ID`) | 每次调用一份请求级 | actor；写入与更新另经导入授权落位 | 只开放检索、读取、写入、更新；可声明任意 Agent |
| `scripts/inject_profiles.py` | `hivememory:profile-injector`，`local` | 无 | (用户，`system`) | 请求级 | root | 在脚本进程内组装完整系统 |

- 声明解析的唯一入口是 [`server/deps.py`](../../../src/hivememory/server/deps.py) 的 `resolve_request_identity_claims`。`x-workspace-id` 只校验取值（只接受 `main_workspace`），从不参与选择；Workspace 总是由用户推出。缺省的 `x-user-id` 回退 `default`，外部入口也适用这条回退。
- chat 在任务进程的注册入口内认证，其余 HTTP 入口在 `deps.py` 认证，同一个 server 有两个认证位置。
- 拒绝映射：第 1、2 阶段失败为 403（网关关闭为 503），第 3 阶段失败为 403，`ScopeRequiredError` 为 500。准入在认证之后失效时报的是 `ScopeRequiredError`；登记在运行中不可变，这条路径暂时走不到。

### 5.2 不经认证网关的接口

| 接口 | 能做什么 |
|:---|:---|
| `GET/POST /config`、`GET /config/defaults` | 读取打码后的主配置；写盘并替换运行时配置，包括 `system.server_principal_id`。所有经网关的 HTTP 路由每次请求都读取这个值 |
| `/models` 增删改查 | 读写模型注册表 |
| `/providers` 列出、写入、删除 | 写入 `configs/providers.secrets.yaml` 中的提供商密钥 |
| `/runtime-events/status`、`/runtime-events/stream` | 回放最近的运行时事件，不按 Workspace 过滤 |
| `WS /ws/logs` | 广播 Patchouli 与引擎日志，其中有用户消息、检索文本与 WRITE 内容的片段 |
| `/health`、前端静态文件 | 无身份需求 |

- server 只有 CORS 与请求日志两个中间件，没有认证中间件。CORS 只约束浏览器，不约束本机进程；WebSocket 不受 CORS 约束。
- `hivememory-server` 入口固定监听 `0.0.0.0:8769`（[`__main__.py`](../../../src/hivememory/server/__main__.py)）；开发脚本默认 `127.0.0.1`。
- 只有 `/mcp` 由 MCP SDK 做了 Host/Origin 检查。

## 6. 授权点

[`WorkspaceOperationAuthorizer`](../../../src/hivememory/workspace/authorization.py) 共七个方法。前四个的公共前缀是：兑现 context、确认准入仍有效、目标等于驻留 Workspace、actor 用户等于 owner。

| 方法 | 前缀之后的检查 | 生产调用 | 是否检查运行类型 |
|:---|:---|:---|:---|
| `authorize_actor_operation` | 白名单包含该操作 | actor API 的 6 个方法 | 只接受线程与请求级；`request_level=True` 时只接受请求级 |
| `authorize_root` | actor 是 `system` | root API 的 20 个方法 | 否，靠架构测试限定调用模块 |
| `authorize_stage` | 无 | 3 个调用点，覆盖 Gateway、Profile 解析、prepare、附件租借、finalize、cleanup | 否，同上 |
| `authorize_import` | actor 不是 `system` | Import Bus 服务 2 处：导入、写入时落位 | 否，同上 |
| `authorize_process_control` | 进程记录的 context 仍有效，两者驻留 Workspace 相同 | 取消 1 处 | 否；不比对 Agent 与 principal |
| `get_stage_operations` | 返回白名单 | 线程表 1 处 | 否 |
| `get_thread_binding` | 无 | 线程表 2 处 | 只接受线程 |

- **一次 MCP 写入经过两次授权**：先以导入授权为这一轮落位（不读白名单），再以 `memory.write` 授权登记意图。落位因此发生在白名单检查之前。
- **请求级读取不限运行类型**：检索与读取以 `request_level=False` 授权，线程 context 也能通过。只有 MCP 入口调用请求级提交，这一点由架构测试保证。
- **存活判定重复**：写入、更新与 CALL 在操作入口与能力方法中各解析一次线程。docstring 说明这是跨越 await 后的有意复查。
- **任务进程内的会话读写**以裸 `WorkspaceIdentity` 调用会话服务，不经阶段授权；写入会话发生在 finalize 的阶段授权之前。

## 7. 资源授权与可见性

### 7.1 各资源的检查

| 资源 | owner | 归属检查 | actor 可见性 |
|:---|:---|:---|:---|
| MemoryAtom | Patchouli | Qdrant 归属过滤，解码后重验；point id 由归属与 ID 派生 | PUBLIC、PRIVATE、TEAM，只表达读取。root 的列表与读取跳过；更新、删除、反馈只查归属 |
| Agent Profile（MemoryAtom） | Patchouli；workspace 的 Profile 缓存 | 缓存键含 Workspace；命中时不重验归属 | 命中时按 policy 重验；root 列表跳过 |
| 写入意图 | workspace | `belong_to` | WRITE 意图没有；UPDATE 意图跟随基础原子 |
| Topic | Patchouli | `(owner, workspace, topic_id)` 键 | 无 |
| Artifact | Patchouli | `belong_to` | 无 |
| 交互提交 | Patchouli | 状态与等待比对 `belong_to`，不一致报 `WorkspaceMismatchError`，与不存在可以区分；提交本身不查 | 无 |
| 记忆任务 | Patchouli | 读出全部后按 `belong_to` 过滤；返回对象含其他 actor 的 `from_actor` | 无 |
| ConversationSession | workspace | `belong_to` | 无；任何 Agent 都能续写同一会话 |
| WorkspaceAsset | workspace | 旧 `IdentityScope` 中的 Workspace | 无 |
| 冷存储 | Patchouli | `load` 重验；`remove` 与 `is_archived` 在 `main_workspace` 下回退到裸 `memory_id` | 无 |

- 授权谓词只有一份实现：[`core/memory_access.py`](../../../src/hivememory/core/memory_access.py)，Patchouli 存储边界与 workspace 读取视图共用。
- 可见性主体：actor API 用线程的 actor（主线程是进程 Agent，子线程是被调用方）；CALL 目标 Profile 用调用方；阶段调用用进程 Agent；SETTLE 与查重用 `system`。
- 生命周期维护的 `get_all_memories_for_maintenance` 跨全部 Workspace 读取，属于系统维护。
- UPDATE 的基础原子在准入时以发起者读取一次，快照随任务入队；执行与重试时只比对快照的归属，不再检查可见性。

### 7.2 生产数据下的实际效果

- **所有记忆都是 PUBLIC**。生成引擎与手动创建固定使用 `MemoryAccessPolicy.public()`，合并沿用保留方的策略。`meta.access_policy` 在补丁白名单中，但没有生产调用方写它；HTTP 的记忆契约也不接受策略字段。
- **TEAM 不可达**。没有生产路径给 `ActorIdentity` 赋 `team_id`，子线程派生也会丢弃它。
- **owner 检查与目标检查恒通过**。两份登记只有 `default` 用户与 `main_workspace`，HTTP 只接受 `main_workspace`，第 3 阶段只接受目标等于驻留 Workspace。
- **白名单几乎不拒绝**。按随仓库发布的登记，chat 的 Agent 落到用户级记录，拥有全部六项操作；harness 的 MCP 入口本来就不提供 RUN 与 CALL，导入不读白名单。唯一能被白名单拒绝的组合是：在 chat 中选择 `claude_code` 或 `dsh`，再使用 RUN 或 CALL。
- **RUN 总是可用**。`mtp_verbs_for_operations` 总会加入 RUN（[`verbs.py`](../../../src/hivememory/alice/orchestration/verbs.py)）。`sys_python_repl`、`sys_web_search`、`sys_read_file`、`sys_write_file` 不受任何操作约束，白名单为空的 Agent 也能执行；文件读写共用一个 `./workspace` 目录，不按 Workspace 划分。

因此在生产中真正改变结果的检查是：外部 harness 的凭据，adapter 与 `allow_system_actor`（外部来源不能声明 `system`），root 与 actor 两个调用面的分离，入口开放的请求类型，以及线程存活判定。

## 8. 后台任务与进程级共享状态

- **后台任务**：`InteractionSubmission`、`MemoryGenerationTaskSpec`、`MemoryGenerationTask`、`PendingAtomMaterializeTask` 携带 `belong_to` 与 `from_actor`；`TopicMaterializeTask` 与话题压缩只带 `belong_to`。重试从 payload 恢复身份，不重新组装 scope。通用 `WorkItem` 与调度任务不带身份。
- **运行时事件**只带 `workspace_id` 标签，没有用户字段；记忆任务事件不设 `workspace_id`。没有代码读取事件的 `workspace_id` 做判断。

只以单个 ID 为键、没有 Workspace 维度的共享状态：

| 状态 | 键 | 说明 |
|:---|:---|:---|
| 写入意图登记 | alias | 回读时比对 `belong_to` |
| 会话服务 | `session_id` | 访问时比对 `belong_to` |
| 进程表 | `process_id` | 控制授权在调用方 |
| Import Bus 的串行键 | (user, agent, source, 外部会话) | 不含 principal 与 Workspace |
| Import Bus 的 `interaction_id` | principal、user、agent、source、外部会话与轮次的哈希 | 不含 Workspace |
| Import Bus 的落位登记 | `interaction_id` | 命中时直接返回话题，不比对 Workspace |
| 交互提交队列、`_active_finalizations` | `interaction_id` | 状态查询时比对归属 |
| `_topic_write_memory_ids` | `topic_id` | 依赖 `topic_id` 全局唯一 |
| 结算终态 `_terminals` | pending alias | — |
| 资产租借 | `lease_id` | 只有内部调用方 |
| 运行时事件缓冲、日志缓冲 | 无 | 经第 5.2 节的接口不加过滤地交出 |

这些在单用户单 Workspace 下都不出错，因为 Workspace 由用户唯一确定、各种 ID 由服务端生成。出现第二个 Workspace 时，Import Bus 的三个键会首先需要补上 Workspace。

## 9. 前端

- [`services/identity.ts`](../../../frontend/src/services/identity.ts) 固定为 `default` 与 `main_workspace`，不持久化；`setIdentitySelection` 没有调用方。前端不发送任何凭据、`Authorization` 或 principal。
- 带身份请求头的：chat 与停止、记忆、话题、会话、`agentApi` 中的 Agent 请求、附件。
- 不带的：记忆任务、`useAgents` 中的 `GET /agents`，以及 config、providers、models、运行时事件与日志。前两者落到后端的缺省用户。
- 切换 Agent 只修改 `currentAgentId`，消息、会话、已检索的记忆与进行中的流都不清理。前端的运行时事件类型没有 `workspace_id`，任务视图接收全部事件。

## 10. 守护方式

- **架构测试 40 条**：`test_access_boundaries.py` 23 条、`test_workspace_boundaries.py` 12 条、`test_resource_identity_boundaries.py` 3 条、`test_identity_fields.py` 2 条。多数是调用点白名单，例如“阶段授权只由任务进程调用”“请求级提交只由 MCP 入口调用”，另有签名规则（公共方法不接收 `identity_scope`）与导入方向规则。
- **运行类型只在 actor 与线程路径上由代码检查**。root、阶段、导入、进程控制与快照读取接受任何有效 context，限制靠架构测试。
- **密封 context 是进程内纪律**：它的 docstring 写明“维护可信进程内调用纪律，不隔离任意恶意 Python 代码”。线程 context 经 `ThreadOperationSubmitter` 进入 Alice 的 CPU 驱动，这个文件在架构测试中被单独豁免。
- **类型检查重复 33 处**：Patchouli 与 Gateway 的公共处理函数各自以 `require_identity_scope` 检查参数类型，不做授权。

## 11. 补丁、例外与无生产使用方

| 项 | 证据 | 类 | 已有追踪 |
|:---|:---|:---|:---|
| 资产 Store、解析服务与端口仍收 `IdentityScope`，System 的桥接以 `system` 自行组装 scope | `workspace/assets/`、`asset_materialization_reader.py` | 例外 | [Todo](../../todo/workspace-asset-ownership-identity-split.md)；Idea 第 8 节 |
| Profile 注入脚本以 `system` 经网关进入 root API | `scripts/inject_profiles.py` | 例外 | Idea 第 8 节 |
| 导入授权：能力层之外的第四个授权面 | `authorize_import` | 补丁 | — |
| 请求级 context 从管理路由扩展到导入、MCP 读取、MCP 写入；写入另加派发话题与 `request_level` | `operations.py`、`actor/memory.py` | 补丁 | — |
| MCP 写入先落位、后检查白名单 | `mcp_server/server.py` 的 `_submit_intent` | 补丁 | — |
| chat 在注册入口认证，其余 HTTP 入口在 `deps.py` 认证 | `process/service.py`、`server/deps.py` | 补丁 | — |
| `system` 承担五种角色（第 4.2 节） | 多处 | 补丁 | — |
| `x-workspace-id` 只校验不使用；缺省用户回退也适用于外部入口 | `server/deps.py` | 补丁 | — |
| 冷存储在 `main_workspace` 下回退裸 `memory_id`，`remove` 与 `is_archived` 不重验归属 | `adapters/long_term.py` | 补丁 | — |
| 进程控制只比对 Workspace | `authorize_process_control` | 补丁 | Idea P-7 |
| RUN 与 `sys_*` 工具不受操作约束 | `verbs.py`、`mtp/verbs/run.py` | 补丁 | 治理 Phase S3；ROADMAP 第 5.4 节暂不考虑 |
| PRIVATE 与 TEAM 可见性、`team_id` | 第 7.2 节 | 休眠 | Idea P-8；[Memory 可见性策略前端 Todo](../../todo/memory-visibility-policy-ui.md) |
| owner 检查与目标等于驻留 Workspace 的检查 | `_check_target`、`admit` | 休眠 | Idea 第 7 节 |
| 可见性主体的区分（进程 Agent、被调用方、调用方、`system`） | 第 7.1 节 | 休眠 | — |
| 白名单 | 第 7.2 节 | 休眠（随仓库发布的登记下） | Idea E-1、P-4a |
| SETTLE 查重只看 PUBLIC | `coordinator.py` | 休眠 | — |
| `IdentityScope` 的目标参数 | 授权点签名 | 休眠，为跨 Workspace 预留 | Idea T-5b |
| `normalize_user_id`、`resolve_default_identity_scope`、`build_internal_identity_scope`、`ISOLATION_WORKSPACE_ID` | `core/constants.py`、`core/models/workspace.py` | 无生产使用方 | — |
| `meta.access_policy` 的补丁路径 | `adapters/mid_term.py` | 无生产使用方 | — |
| `ShortTermMemoryStore.list_all`、`LongTermMemoryStore.query`、`MEMORY_REVIVE` 路由（已绑定，没有请求方）、`RetrievalFamiliar.is_archived`（以裸 `memory_id` 调用需要复合键的端口） | Patchouli | 无生产使用方 | — |
| 前端 `setIdentitySelection` | `identity.ts` | 无生产使用方 | 前端身份 Todo（2026-11-09 删除，最后版本见 commit `3cfdec9d`） |
| `AGENT_PROFILE_LIST` 的契约说明写“当前发起者可见”，实现跳过可见性 | `patchouli/contracts/public_routes.py` | 不一致 | — |
| 前端停止注释写“复用创建时冻结的原始 scope”，后端只比对 Workspace | `frontend/src/services/chatApi.ts` | 不一致 | — |
| `resolve_default_identity_scope` 的 docstring 含计划代号 “P2.5” | `core/models/workspace.py` | 不一致（注释规则） | — |
| `RootSessionService` 不在 `root/__init__.py` 的导出列表中 | `workspace/capability/root/` | 不一致 | — |

## 12. 分析

以下是对现状的分析，不是 owner 决定，也不预排计划。

### 12.1 体系现在是什么样

这是一套为多用户、多 Workspace、多执行者设计的进程内访问模型，运行在单用户、单 Workspace、记忆全部 PUBLIC、HTTP 调用方不证明身份的部署上。

模型本身的骨架是清楚的，第 4.1 节标为“设计”的部分都承重：两阶段认证加两阶段授权，密封 context，root 与 actor 分面，白名单作为 Agent 权限的唯一来源，资源归属硬边界，Patchouli 的归属与发起者拆分，以及外部凭据。问题不在骨架，而在三处：骨架之外的入口没有被纳入，骨架之上叠加了补丁，骨架的大部分在当前数据下不产生可观察的效果。

### 12.2 碎片化的形态

- **加而不改**。第 3 节的 16 份计划各自服务当时的主线：Workspace 准入、Patchouli 拆分、Alice 的 CALL 与权限、Claude Code 导入、MCP 读取、plugin 写入。新需求多以新增一个授权方法、一种 context 用途或一个参数进入。A1 交付的是逐次操作授权，现在授权者有七个方法，`system` 累积了五种角色。每一步在局部都合理。
- **回答一个问题要拼四处**。“某个外部调用方到底能做什么”，要依次看 principal 登记、Workspace 访问登记、入口开放的请求类型，以及入口经过哪几个授权方法。以 MCP 写入为例：凭据决定能否接入，导入授权决定能否落位，`memory.write` 白名单决定能否登记，请求级提交决定开放哪几种操作。
- **规则一半在代码，一半在架构测试**。运行类型只在 actor 与线程路径上由代码检查，其余授权方法的调用范围由 40 条测试限定。这些测试守住了结构，但读代码的人需要同时读测试，才知道某个方法“只能”被谁调用。

### 12.3 复杂度与暴露面错位

多数复杂度用来防范可信代码内部的接线错误：密封与撤销、三种运行绑定、线程存活判定、调用点白名单。真实的暴露面却在没有身份的地方：

- HTTP 管理路由不证明调用方，能连到端口的程序都能以 `system` 读写记忆；
- config、providers、models 的写接口不经网关，可以写入提供商密钥，也可以改写 `server_principal_id`；
- 日志与运行时事件不加过滤地广播内容片段；
- `sys_*` 工具不受白名单约束，白名单为空的 Agent 也能执行 Python 与读写文件；
- 正式入口监听 `0.0.0.0`。

v1.0 让四个 harness 共用一个常驻后端。威胁从“自家代码接错线”变为“harness 中带 shell 的 agent 绕过白名单”。密封 context 对后者不起作用，调用方证明才起作用。2026-11-01 dsh 中的 agent 取得明文密钥，就是这一类情形（[管理接口 Todo](../../archive/todo/local-http-management-access.md)）。

### 12.4 Agent 身份是最弱的一环

用户这一层有回退值，Workspace 这一层由用户推出，两者都只有一个取值，所以暂时无害。Agent 这一层不同，它有真实的多个取值（`omni_doll`、自定义 Profile、`claude_code`、`dsh`，以后还有 Codex 与 pi），却没有任何一层证明它：

- 用户级记录接纳任意字符串；
- harness 凭据不绑定 Agent；
- chat 依赖 Profile，导入与 MCP 不依赖；
- `team_id` 从不赋值。

以 Agent 为粒度的规则因此都建立在声明上：PRIVATE 的 `target_agent_id`、按 Agent 区分的白名单、意图与生成任务的 `from_actor`、来源记录。现在所有记忆都是 PUBLIC，后果不可见。四个 harness 共用一个记忆库、开始需要 Agent 之间的可见性时（Idea 第 6 节，A-10），这会直接成为问题：某个 harness 只要声明另一个 Agent，就能读到写给那个 Agent 的 PRIVATE 记忆。

### 12.5 与候选清单 A-01 至 A-15 的关系

owner 候选清单中的待决问题是：v1.0 之前这条线做到哪一层，只做盘点与调用方证明（A-01、A-02），还是把登记生效时点、凭据与 Agent 绑定、记忆可见性也一并收敛。从现状看：

| 层次 | 条目 | 现状对它的意义 |
|:---|:---|:---|
| 外部边界 | A-02，以及第 5.2 节的无网关接口与 `0.0.0.0` | 暴露面集中在这里；第 5.2 节的接口已在管理接口 Todo 的范围内 |
| Agent 身份 | A-07 | 以 Agent 为粒度的规则都依赖它；A-10 在它之前没有可执行的意义 |
| 记忆可见性 | A-10、A-11 | 当前只有 PUBLIC，可见性机制完整但休眠；需要先有 A-07 |
| 登记生效时点 | A-03、A-04 | 运维便利性问题；每接入一个 harness 改两份配置并重启，接入数越多成本越高 |
| 已登记例外 | A-05、A-13 | 两项都已定位；A-13 涉及资产端口与全部消费者 |
| 小项 | A-08、A-09、A-14 | 各自独立，规模小 |
| controller 与多 Workspace | A-06、A-12、A-15 | 对应结构已经预留（目标参数、进程控制、运行绑定）；单 Workspace 的 plugin 模式用不到 |

本次调查发现、清单中没有单独成条的：

- RUN 与 `sys_*` 工具不受操作约束（第 7.2 节）；治理文档把它归入 Phase S3，ROADMAP 第 5.4 节暂不考虑可执行资产沙箱；
- `POST /config` 可以改写 `server_principal_id`，日志与运行时事件广播内容（第 5.2 节）；
- `system` 的五种角色（第 4.2 节）；
- Import Bus 的三个键不含 Workspace，冷存储的 `main_workspace` 回退（第 8、11 节），出现第二个 Workspace 时才会成为问题；
- UPDATE 执行时不重验基础原子的可见性（第 7.1 节），在只有 PUBLIC 时没有后果。

### 12.6 可以简化的地方

以下只说明可能性与它原本保护什么，由 owner 决定是否处理。

- **无生产使用方的身份代码**（第 11 节）：删除只影响测试；它们不保护任何生产行为。
- **`system` 的角色**：第 3、5 种（内部任务的发起者）与第 1、2 种（root 授权）共用一个值。分开命名可以让“root 只认 `system`”与“内部任务以 `system` 发起”不再互相牵连；保护的行为（SETTLE 查重只看 PUBLIC、root 不经白名单）可以由各自的规则保证。
- **运行类型检查**：可以在授权方法中统一检查，也可以维持由架构测试保证。现在是两种方式混用。
- **休眠结构**：不建议删除，它们是为多用户与多 Workspace 准备的地基。但在 A-07、A-10 让它们产生可观察的效果之前，继续细分可见性主体或增加授权面，都无法由真实使用验证。
