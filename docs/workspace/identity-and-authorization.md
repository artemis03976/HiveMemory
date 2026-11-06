---
title: 身份与授权
status: current
owner: workspace
scope: identity-coordinates-access-stages-operation-authorization-and-resource-ownership
code_paths:
  - src/hivememory/core/models/identity.py
  - src/hivememory/core/models/workspace.py
  - src/hivememory/core/access.py
  - src/hivememory/workspace/authorization.py
  - src/hivememory/workspace/process/
  - src/hivememory/server/deps.py
  - src/hivememory/system/application/import_bus_service.py
  - src/hivememory/mcp_server/
  - src/hivememory/patchouli/memory_library/stores.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/error-model.md
related_docs:
  - docs/workspace/authentication.md
  - docs/workspace/capability.md
  - docs/system/import-bus.md
  - docs/architecture/data-model.md
  - docs/governance/security/identity-and-execution-safety.md
last_reviewed: 2026-11-06
---

# 身份与授权

本文说明 workspace 的身份坐标、四阶段访问模型中的授权部分、各层持有的身份数据，以及资源归属如何在操作、记录与后台任务之间交接。第 1、2 阶段（认证）的登记、工作 actor 与认证网关见[认证与访问登记](./authentication.md)；授权在哪些调用面上发生见[能力层](./capability.md)。

## 1. 身份坐标

### 1.1 三个模型回答三个不同问题

| 模型 | 当前职责 | 不承担的职责 |
|:---|:---|:---|
| `ActorIdentity` | 谁在执行：`user_id`、`agent_id` 和可选 `team_id`。 | 不表示资源归属，也不单独授权 Workspace-owned 资源。 |
| `WorkspaceIdentity` | 资源归属于哪个用户和 Workspace：`owner_user_id`、`workspace_key`、`workspace_id`。W0 要求 key 与 ID 相同且非空。 | 不表示登录 session、grant 或永久 capability。 |
| `IdentityScope` | 一次操作的发起者与目标 Workspace，由 `ActorIdentity + WorkspaceIdentity` 组成；由授权点在操作授权通过后组装（第 4 节）。 | 不携带 `interaction_id`、generation、run、frame、request 或 trace 等关联 ID；不校验 owner 规则（owner 约束属于第 2、3 阶段）；不表示 actor 驻留在哪里（那是工作 actor 的内容）。 |

`interaction_id`、`topic_id`、`memory_id`、`artifact_id` 和 work/task ID 仍由各自领域载体持有。`ActorIdentity` 不含 `session_id`；Chat 请求的 `session_id` 选择会话资源，但不进入身份；finalize 仍以进程的 `interaction_id` 关联。这样可以避免把局部关联 ID 误当成公共身份，或在队列、缓存中派生出第二套 scope 模型。

### 1.2 默认入口和内部 seam

HTTP 层的用户导向身份选择（`user_id + workspace_id`）由统一请求头 `x-user-id` / `x-workspace-id` 承载，并在 `server/deps.py` 的 `resolve_request_identity_claims` 解析为**身份声明**：actor 声明与请求进入的 Workspace。声明不是身份结论——认证之前 server 只持有声明，`IdentityScope` 要到第 3 阶段才由授权点组装。该函数是默认身份解析的唯一入口：

1. 用户与 Workspace 的选择只取自请求头，路由不接受 body 或 query 中的这两项身份字段；Agent action 的具体 `agent_id` 由请求体给出（第 3 项）；
2. `workspace_id` 只允许已声明的公共入口 `main_workspace`，其余值按不存在拒绝（404）；全部缺省时在此唯一回退 `DEFAULT_USER_ID` 与默认 Workspace；
3. Agent action 必须显式提供具体 `agent_id`，缺失（`request.agent_required`）或显式给出保留的 `system`（`request.agent_reserved`）都返回 400。Agent action 有两种：由具体 Agent 执行的 Chat，以及代表具体 Agent 导入已发生交互的 `POST /api/v1/import/turns`。非 Agent action（管理操作、Topic 管理、取消等）由 server 使用保留 `SYSTEM_AGENT_ID = "system"`，其语义是“没有具体 Agent 作为操作来源主体”。`system` 是 root 授权唯一接受的 actor（第 4 节），但它不是 Agent：不持有任何 actor 操作、不能作为任务进程的执行者或交互的参与者、不是前端可选 Agent，也不得写入 `MemoryAccessPolicy.target_agent_id`。

声明只作为认证输入：server 把它交给统一认证网关，认证通过后经过验证的身份是工作 actor，server 不把声明当作已确认的身份继续使用，也不交给路由处理函数。W0 不提供 Workspace 创建或切换产品入口。`isolation_workspace` 只在隔离测试中显式构造，用于验证同一用户和 Agent 在两个资源域中的访问不会串扰。

下游不能读取进程级 `current_workspace`，也不能在 retry 时重新执行默认解析；后台任务必须从自己的领域 payload 分别恢复资源归属与发起者，并在最终访问 Workspace-owned 资源时重新执行归属检查（第 6.2 节）。

所有与 Workspace 相关的 HTTP 入口都经统一认证网关认证：Chat 由任务进程的注册入口认证，管理操作、取消、导入与 MCP 工具调用在入口认证得到请求的工作 actor（第 6.1 节）。server 不在认证前组装 `IdentityScope`，这条规则没有例外。

## 2. 访问边界：两阶段认证与两阶段授权

Workspace 的访问控制分四个阶段。前两个阶段在 actor 进入 Workspace 时完成（认证），后两个阶段在每次操作时进行（授权）：

| 阶段 | 回答的问题 | 负责者 | 产出 |
|:---|:---|:---|:---|
| 1 Principal 认证 | 调用来源是否已登记、能否经这个 adapter 代表这个 actor | System 接入登记：`core.access.PrincipalAuthenticator` 端口，由 `system/access/` 的 `SystemPrincipalAuthenticator` 实现 | 已验证的 `ActorIdentity` |
| 2 Workspace 认证 | 这个 actor 能否进入 Workspace W | `WorkspaceAuthenticator`（`workspace/authentication.py`） | 工作 actor：经哪个来源、以哪个 actor 驻留在 W |
| 3 操作授权 | 这次操作（执行者 → 目标 Workspace T）是否被允许 | `WorkspaceOperationAuthorizer`（`workspace/authorization.py`），在授权点调用 | `IdentityScope(actor, T)` |
| 4 资源授权 | 目标资源是否允许这次操作 | 资源 owner（Patchouli 等） | 允许，或按不可见处理 |

前两个阶段由唯一对外的认证入口 `ActorAuthenticationGateway` 编排（[认证与访问登记](./authentication.md)）。拆成四个阶段，是因为它们由不同的所有者负责，产出不同形态的身份数据：

- **接入登记与准入登记分属两个配置所有者**：哪个调用来源可以经哪些 adapter 接入是 System 的配置；哪个 actor 可以进入哪个 Workspace、能执行哪些 operation 是 Workspace 的配置（[认证与访问登记](./authentication.md#2-两类访问登记)）。所有者不同，但调用方只面对一个认证入口。
- **驻留 Workspace 与操作目标是两件事**：actor 进入 Workspace 时确定的是它驻留在哪里；每次操作时确定的是这次操作作用于哪个 Workspace。例如 coder 在 WA 中运行，想去 WB 查找资源：这次操作的发起者是 coder，目标是 WB，与 coder 驻留在 WA 无关。因此第 2 阶段的产出（工作 actor）与第 3 阶段的产出（`IdentityScope`）由一个 actor 和一个 Workspace 组成，含义却不同，不能互相代替。W0 的简化：第 3 阶段只接受目标等于驻留 Workspace，跨 Workspace 的授权模型尚不存在。
- **执行者与参与者是两件事**：每个操作以真正执行它的身份授权（第 4 节）。交互提交、话题准备、Profile 与附件由系统执行，交互中的 Agent 只是参与者，作为数据显式交给资源 owner，记为交互的发起者（`from_actor`）。如果让参与者兼任执行者，系统级操作就只能以 Agent 的身份授权：用户为一轮对话选择的 Profile 与附件会受 Agent 的可见性约束，Agent 也会获得提交交互的能力。
- 资源状态、输入有效性、版本冲突等仍是正常领域处理，不是第五层授权。

## 3. 身份数据的形态与流向

| 形态 | 回答的问题 | 何时确定 | 寿命 | 可以出现在哪里 |
|:---|:---|:---|:---|:---|
| `ActorIdentity` | 谁将要执行接下来的任务 | 认证前只是声明；第 1 阶段通过后成为已验证的身份 | 长期 | 任何地方 |
| 工作 actor（`WorkingActor`） | 这个 actor 经由哪个来源接入、驻留在哪个 Workspace | 第 2 阶段通过时，或子线程准入时 | 与持有它的进程记录、线程记录或入口调用链相同 | 进程记录、线程记录、入口与 Import Bus 的调用链，以及授权点 |
| `IdentityScope` | 这一次操作由谁执行、作用于哪个 Workspace | 每次操作授权时组装 | 只在这一次操作的调用链内 | 授权点、资源 owner 的公共边界与 Gateway |
| `belong_to: WorkspaceIdentity` | 资源、记录或任务属于哪个 Workspace | 公共边界拆分，或资源创建时写入 | 随资源、记录或任务保存 | Patchouli 内部、引擎、存储及领域载体 |
| `from_actor: ActorIdentity` 与来源记录 | 操作或生成由谁发起、哪些 Agent 贡献内容 | 公共边界拆分或显式的参与者，或生成链明确构造 | 随记录或需要它的任务保存 | 领域调用与来源记录；不代替资源归属 |

资源被创建时，归属取自操作的目标 Workspace，来源取自操作的发起者（交互提交时是显式的参与者）；此后归属与来源就是独立的数据。资源 owner 先比对目标 Workspace 与资源归属，再按 policy 判断发起者是否可见，资源的来源字段不参与授权（第 5 节）。

各层持有的身份数据：

| 层 | 持有 | 不持有 |
|:---|:---|:---|
| server（入口 adapter） | 认证前的声明（只作为认证输入）；自身的 principal 与 adapter；导入请求声明的 principal 与凭据（只交给网关）；管理操作、取消与导入在调用链内的请求的工作 actor；chat 的不透明进程句柄 | 认证前组装的 `IdentityScope`；进程记录及其中的工作 actor |
| MCP 入口（`mcp_server`，操作适配器） | 认证前的声明与凭据（只交给网关）；一次工具调用的工作 actor，只在本次调用内交给 Import Bus 服务的落位与操作入口的请求级提交 | `IdentityScope`；actor API 本身（只经门面取得的操作入口提交） |
| 授权点：注册入口、能力层、任务进程的执行器与 CPU 分配、Import Bus 服务 | 交来的工作 actor、操作授权者（注册入口与 Import Bus 服务另持认证网关与系统自身的 principal；进程线程管理另经子工作 actor 的准入端口） | 调用方另行传入的 `IdentityScope` |
| Import Bus 编排器（`ImportBusIngestor`） | 系统的 root 工作 actor 与以它授权的 `IdentityScope`、参与的 Agent 与 principal 标识，只在本次导入或落位请求的调用链内使用：推导外部会话键与 `interaction_id`，经公共路由交给 Gateway 与 Patchouli，经 root API 提交交互；落点登记只保存 prepare 的结果（话题与 Workspace 归属），不保存身份 | 认证网关、操作授权者；不把 scope 或工作 actor 写入记录、事件或后台任务 |
| Gateway 与 Patchouli 公共边界 | 授权点组装后传入的 `IdentityScope`（系统阶段是 root scope）；显式的参与者；Patchouli 立即拆分归属与发起者 | 工作 actor、认证网关、操作授权者 |
| Patchouli 内部、引擎、存储 | 独立的 `WorkspaceIdentity` 与需要时的 `ActorIdentity` | `IdentityScope`、工作 actor、认证网关、操作授权者 |
| 进程记录 | 执行者与系统的 root 工作 actor、进程元数据；事件绑定载体中的只读观测标签 | — |
| 进程的线程表（`TaskProcess.threads`） | Actor 阶段内本进程的执行线程：线程句柄、工作 actor、状态、父线程与线程描述 | — |
| CPU 驱动（`AliceCPU`） | 本次执行的主线程与 CALL 派生的子线程（`ExecutionThread`：线程描述与不透明的线程句柄），各自与操作入口绑定为提交函数（`workspace.contracts.ThreadOperationSubmitter`） | 工作 actor、`IdentityScope` |
| Alice 运行（frame、Agent loop、MTP、CALL 编排）与 agent_runtime | `ExecutionLabels`、线程描述 `ThreadDescriptor`、每个 frame 所属线程的提交函数 `OperationSubmitter` | 线程句柄、`ExecutionThread`、工作 actor、`IdentityScope`、可用于授权的 actor / Workspace 身份 |

必须持续成立的不变量：

1. 第 3 阶段之前不存在 `IdentityScope`：入口在认证前只持有 actor 声明与请求进入的 Workspace。
2. 工作 actor 只由认证一侧构造，只出现在进程记录、线程记录、入口与 Import Bus 的调用链和授权点；不进入资源 owner、Gateway、引擎或存储，不进入 CPU 可见的对象，不写入任何记录、事件、交互记录或 DTO。
3. 交互记录与后台任务独立保存资源归属和需要的发起者，不保存操作 scope 或工作 actor。
4. 授权点只接收工作 actor 与目标 Workspace，`IdentityScope` 只由操作授权者组装（资产桥接是登记的例外，见第 8 节），不由调用方另行传入；公共 route 接收 scope，Patchouli 公共边界以下只接收拆分后的归属与发起者。
5. 资源 owner 先校验目标 Workspace 等于资源归属，再按资源 policy 检查发起者；管理视角也不跳过 Workspace 硬边界，资源来源不能充当权限依据。归属检查常以分区键或查询预过滤的形式完成、看似多余，但缓存与资产索引是进程级共享设施，存储的预过滤也不是授权事实，因此不能省略。
6. 授权规则不是身份类型的约束：“actor 用户等于 Workspace owner”属于第 2、3 阶段，`IdentityScope` 不校验它。

不变量 2、4 由架构测试守护：认证与授权模块只允许 workspace、System（组合根与 Import Bus 服务）与 server 导入；能力层、任务进程的执行器、CPU 分配与进程线程管理不导入认证一侧，子工作 actor 的准入只由进程线程管理调用；`WorkingActor(...)` 只在认证一侧构造，`IdentityScope(...)` 只在授权者与资产桥接中构造；Alice、agent_runtime、Patchouli、Gateway 与引擎不引用工作 actor；能力层、任务进程与 Import Bus 服务的公共方法不接收调用方组装的 `IdentityScope`，显式的目标 Workspace 必须与工作 actor 配对出现。身份拆分边界测试另限定 Patchouli 的 scope 引用只能出现在公共 handler，五个引擎与记录/任务模型不保留 scope 或工作 actor。

执行标签 `ExecutionLabels(agent_id, workspace_id)` 不是上表中的身份形态：它是注册认证通过后创建的冻结观测标签，输入清单与线程描述各带一份，只服务历史角色匹配、展示回退与观测，不能授权或选择资源目标，也不得据此重建访问身份（[任务进程](./task-process.md#2-注册)、[执行线程](./execution-threads.md#5-主线程交给-cpu)）。

## 4. 操作授权

`WorkspaceOperationAuthorizer` 是第 3 阶段的唯一实现，无状态，只读 Workspace Actor 访问注册表。它只有两个授权方法，按“谁在执行这项操作”区分：

| 方法 | 规则 | 拒绝 | 调用位置 |
|:---|:---|:---|:---|
| `authorize_agent(actor, operation, target)` | `operation` 是六项 `WorkspaceOperation` 之一，且在访问登记的白名单中 | `operation_not_allowed` | actor API 的方法入口；任务进程中 prepare 的检索 |
| `authorize_root(actor, target)` | actor 是保留的 `system` | `root_required` | root API 的方法入口；任务进程的系统阶段与取消；Import Bus 服务 |

两个方法共用三项前提，按顺序检查：准入记录仍然有效，否则 `ScopeRequiredError`（`actor_not_admitted`）；目标等于驻留 Workspace，否则 `OperationDeniedError`（`target_workspace_not_resident`）；actor 用户是目标 Workspace 的 owner，否则 `OperationDeniedError`（`target_owner_mismatch`）。全部通过才返回 `IdentityScope(actor, target)`。owner 检查因此分在两处：第 2 阶段检查要进入的 Workspace，第 3 阶段检查操作的目标。行为白名单不进入工作 actor：每次授权都按访问登记即时查询，不缓存授权结论。另有只读查询 `allowed_operations(actor)`：返回白名单，只供线程表生成线程描述的操作快照，不授予权限。

**操作授权只在 actor 执行期间生效。** 其余检查之所以存在，是因为所有 API 都收拢到了 Workspace 能力层：能力层的 API 代表谁在执行这项操作，每个方法在入口以执行者授权。各调用面的执行者：

| 操作 | 执行者 | 授权 |
|:---|:---|:---|
| actor API 的六项操作（执行线程、MCP 请求） | 执行线程或请求的工作 actor | `authorize_agent`，目标取驻留 Workspace |
| prepare 的检索 | 任务进程的执行者 | 执行者持有 `memory.search` 时 `authorize_agent`，否则不检索 |
| root API（管理、会话、上传与提交交互） | 请求的 `system` | `authorize_root`，目标由请求给出 |
| 任务进程的 Gateway、话题准备、Profile、附件、finalize 与 cleanup | 系统的 root 工作 actor | `authorize_root` |
| 取消 | 请求方（`system`） | `authorize_root(请求方, 进程的 Workspace)`，拒绝呈现为 `not_found` |
| Import Bus 的导入与写入前的落位 | 系统的 root 工作 actor | `authorize_root` |

各项归属的理由：

- **取消**是用户下达的指令，不是 Agent 的操作。拒绝（请求方驻留在其他 Workspace、不是 `system`）与进程不存在一样返回 `not_found`，不泄露进程是否存在。
- **Profile 与附件**是用户为这一轮作出的选择，对用户可见，而不是对 Agent 可见，不能以 Agent 的身份执行。Profile 按管理语义读取，只检查 Workspace 归属：加载哪份图纸取自认证得到的执行者，指向其他 Agent 的 PRIVATE 图纸也能被它自己的进程加载（[读取视图与写入意图](./read-view-and-write-intents.md)）。CALL 解析被调用方的图纸是调用方自己的 actor 操作，仍按调用方的可见性。
- **Gateway 与 cleanup** 面对的话题属于 Workspace，与 Agent 无关。
- **prepare 的检索**要按执行者的可见性返回记忆，因此是执行者的 actor 操作；执行者没有 `memory.search` 时不检索，对话照常进行。
- **交互提交**是系统级操作，Agent 不能拥有这项能力。三个来源都在这一范畴：chat 的 finalize、plugin 模式的导入，以及将来的历史对话导入。交互中的 Agent 作为参与者显式传给 Patchouli。导入不检查白名单：白名单为空、只登记了准入的 Agent 也能导入；外部 harness 不能声明 `system`，管的是边界上的声明，与系统执行交互提交无关。

**系统自身的 principal**：系统组件以自己登记的 principal 经 `local` adapter 认证 (user, `system`)，取得执行系统级操作的身份（[认证与访问登记](./authentication.md#5-系统自身的-principal)）。任务进程在注册时认证一次，Import Bus 每次导入或落位时认证一次，认证的是参与者驻留 Workspace 的 `system`：目标不是参与者驻留的 Workspace 时，root 授权以 `target_workspace_not_resident` 拒绝，交互不会落到参与者所在之外的 Workspace。

两个授权方法的调用位置与系统自身的 principal 的使用位置都由架构测试锁定：`authorize_agent` 只在 actor API 与任务进程中调用，`authorize_root` 只在 root API、任务进程与 Import Bus 服务中调用。认证与授权互不依赖：两者都只读同一份访问注册表，经工作 actor 这个类型发生联系（[认证与访问登记](./authentication.md#3-工作-actor)）。

## 5. 资源归属与寻址

### 5.1 Workspace-owned 资源

Workspace 资源的最终寻址同时包含 WorkspaceIdentity 和资源 ID。复合键承载的是归属校验，而不是允许每个 Workspace 重新定义一套局部 ID：

| 资源 | 当前寻址结构 | 主要所有者 |
|:---|:---|:---|
| ConversationSession | `WorkspaceIdentity + session_id`，管理入口先做 root 授权 | workspace `ConversationSessionService`（[会话资源](./sessions.md)） |
| Topic | 公共边界为 `IdentityScope + topic_id`；内部为 `belong_to + topic_id`（adapter 内部为 `WorkspaceTopicKey`） | Patchouli Perception / `ShortTermMemoryStore` |
| Memory | `WorkspaceMemoryKey(workspace_identity, memory_id)` | Patchouli `MidTermMemoryStore` / 长期存储 |
| Artifact | `WorkspaceArtifactKey(workspace_identity, artifact_id)`；`ArtifactRef` 同时带 WorkspaceIdentity | Patchouli `ArtifactStore` 及其适配器 |
| WorkspaceAsset | `WorkspaceAssetKey(workspace_identity, asset_id)`；外部使用当前 Store 的 opaque `WorkspaceAssetRef` | workspace 持有的 `WorkspaceAssetStore`（[WorkspaceAsset](./assets.md)） |

`topic_id` 在领域上是全局唯一身份。正常创建路径由统一流程生成新的 UUID，两个 Workspace 可以使用相同标题，但不能把同一个 `topic_id` 作为两个合法 Topic 并存。公共调用方以 `IdentityScope + topic_id` 访问，Patchouli 内部只传 `belong_to + topic_id`；`WorkspaceTopicKey` 仅由短期 adapter 在内部构造，用于归属校验和物理索引。

### 5.2 Memory 的归属

Memory 在 `MetaData.workspace_identity` 中保存唯一持久化归属；读取时先限定 owning Workspace，再由 Patchouli 的记忆策略判断 actor 是否可读。具体 policy 与检索规则属于 [MemoryLibrary](../patchouli/memory-library.md) 与 [Retrieval](../patchouli/retrieval.md)，本文只确认 Workspace 是 Memory 的唯一归属边界；存储层的 legacy 兼容解释分支已随存量数据迁移完成而删除，`workspace_identity` 是归属字段的唯一权威。

### 5.3 共享基础设施不分区

work queue、ordering/idempotency key、task/run registry、scheduler、runtime container 和 EventBus 维持进程级共享语义。领域 TaskSpec 独立携带 `belong_to` 与需要的 `from_actor`，通用 WorkItem、WorkRecord 和 RuntimeEvent infrastructure 不解释这些身份字段。`RuntimeEvent.workspace_id` 只是可选观测标签，不参与路由、订阅、sequence、授权或缓存分组。派生缓存按派生源坐标键控的规则见[读取视图与写入意图](./read-view-and-write-intents.md#11-派生缓存的键控)。

## 6. 操作身份与资源归属交接

### 6.1 Chat、管理请求与导入

当前入口的 Workspace 交接可以概括为：

```text
用户导向身份选择（user_id + workspace_id；Agent action 附加 agent_id）
  -> server/deps.py 解析为身份声明（actor 声明 + 请求进入的 Workspace）
  -> 统一认证网关：第 1、2 阶段，得到工作 actor
       chat：注册入口认证执行者与系统的 root 工作 actor，存进进程记录
       管理操作、取消、导入与 MCP 工具调用：入口认证，得到请求的工作 actor
  -> 授权点：第 3 阶段，以执行者组装 IdentityScope（authorize_agent / authorize_root）
  -> public route / 资源 owner 公共边界（接收 IdentityScope，交互提交另收显式的参与者）
  -> Patchouli 拆为 belong_to / from_actor，内部独立传递
  -> 在 Workspace-owned 资源边界校验归属与 actor policy
```

主动 Chat 是 Agent action，必须携带具体 `agent_id`；注册入口认证执行者与系统的 root 工作 actor 并立即创建进程，进程的阶段调用以注册时通过认证的 Workspace 为目标，按第 4 节的执行者授权（[任务进程](./task-process.md#2-注册)）。`/chat/stop` 不是 Agent action：server 以 `(user, system)` 声明认证，注册入口对进程的 Workspace 执行 root 授权。

导入 `POST /api/v1/import/turns` 也是 Agent action：server 以 (user, 请求体中的具体 Agent) 声明、请求声明的 principal、凭据与 `import` adapter 经网关认证，这一步证明请求来自哪个调用来源、参与者是哪个 Agent；Import Bus 服务再以系统的 root 工作 actor 授权，经公共路由交给 Gateway 与 Patchouli，经 root API 提交交互。提交的交互归属取目标 Workspace，发起者是参与的 Agent。外部来源与外部会话标识只用于按会话串行与推导 `interaction_id`，不进入身份，也不因此建立第二套 Workspace 资源状态（[Import Bus](../system/import-bus.md)）。下游不使用进程当前 Workspace 推断资源归属。

MCP 工具调用同样是 Agent action：MCP 入口以 (user, `X-Agent-ID` 中的具体 Agent) 声明、请求声明的 principal、凭据与 `mcp` adapter 经网关认证一次，得到的工作 actor 经操作入口的请求级提交进入 actor API，actor API 以 `authorize_agent` 组装本次调用的 `IdentityScope`；写入与更新之前，Import Bus 服务以系统的 root 工作 actor 为这一轮落位。工具参数与连接器补上的轮次信息都不作为身份来源（[MCP 记忆访问](../contracts/mcp.md)）。

### 6.2 后台任务和重试

`InteractionSubmission`、`MemoryGenerationTaskSpec`、`MemoryGenerationTask` 与 `PendingAtomMaterializeTask` 独立保存必需的 `belong_to` 和 `from_actor`，并携带所需的 interaction、intent、topic 或 task ID；`PreparedAgentRun`、`TopicMaterializeTask` 和 Topic working set 的 lease 只保存归属。Topic working set 的驻留记录为 `(WorkspaceIdentity, topic_id) → 最近访问时间`，idle/LRU/shutdown 候选只返回归属与 Topic ID，不保存最近访问者。

Interaction codec 为 v7，generation codec 的 `schema_version` 为字符串 `"1.1"`，分别完整编码归属与发起者；当前队列是进程内队列，未注册旧 codec。Work Queue 只运输编码后的 payload 和执行状态，不解释 Workspace 领域模型。retry 从 payload 分别恢复身份与领域 ID，再到真正的 Workspace-owned resource 边界执行检查；它不重跑默认 resolver、不读取进程当前 Workspace，也不重新组装 scope。任务 list/get/cancel 以 task 的 `belong_to` 检查目标 Workspace，越域与不存在统一按 not found 处理。

手动、idle、LRU 与 shutdown 结算都由 coordinator 创建 SETTLE 生成任务，发起者统一为 `system_actor_for_workspace(belong_to)`（Workspace owner 用户、保留的 `system`、无 Team）。参与内容的 Agent 只进入贡献者集合；结算查重按这个 system actor 的普通读取规则只看到本 Workspace 的 PUBLIC 记忆。WRITE/UPDATE 则保留提交 Agent 的可见性与来源。这是 `system` 的第二种用法：非主动路径的发起者只经资源 owner 的普通可见性规则读取，从不经过 root API，与 `system` 作为 root 分属不同入口。队列的状态机和重试策略见[运行时机制：总线、调度器与 Work Queue](../components/runtime-and-bus.md)。

## 7. 拒绝语义

四个阶段的失败以稳定错误类型与 reason 区分：第 1、2 阶段的失败是 `AdmissionDeniedError`，第 3 阶段的失败是 `OperationDeniedError`，准入记录在授权时已不再有效是 `ScopeRequiredError`（`actor_not_admitted`）。线程已结束是生命周期拒绝，以 `ThreadEndedError` 区分：线程已结束、进程已收到停止或已经关闭时，actor 请求与 CALL 派生得到它；登记的不是这一个线程句柄、子线程试图派生或 CPU 结束主线程，属于接线缺陷，以 `ScopeRequiredError` 拒绝（[执行线程](./execution-threads.md#3-生命周期与唯一的存活判定)）。reason、错误码与 HTTP 映射的完整表格只维护在[错误模型](../contracts/error-model.md)第 4.4 节。访问错误必须沿 System/application 以原语义传播，不能被通用 `RuntimeError` 捕获包装成服务不可用。

会话和写入意图的异常定义由 `workspace.contracts.errors` 拥有；访问、共享资源与线程结束类型仍在 `core.errors`。读取视图的缺失/不可见为中立资源失败，Profile 类型与配置失败消费 Patchouli 的公共错误。CALL 的共享引用需要 READ，授权拒绝不能静默跳过。线程结束有稳定 code，Agent 出口明确提示停止操作；共同呈现与兼容规则见[错误模型](../contracts/error-model.md)。

## 8. 已知限制

- root 只由 actor 是否为保留的 `system` 判定；能声明 `system` 的来源由接入登记的 `allow_system_actor` 逐个限定（默认不能），当前只有 HTTP server、系统自身的 principal 与 Profile 注入脚本可以，外部 harness 不能（[认证与访问登记](./authentication.md#1-两阶段认证)）。HTTP 管理路由要求管理凭据（[认证与访问登记](./authentication.md#1-两阶段认证)第 1 节），但请求头中的用户身份仍不做证明，管理凭据也只防误用：能读到凭据文件的本机程序仍能经管理路由以 `system` 声明进入 root API，这沿用本地单用户部署的信任假设；
- principal 标识接入来源，一个外部 harness 对应一个 principal（[认证与访问登记](./authentication.md#1-两阶段认证)第 1 节），不与 Agent 绑定：同一个 harness 可以以不同的具体 Agent 运行，Agent 由经过凭据证明的来源声明，权限与记忆可见性只跟随 Agent，与经哪个来源接入无关。仍然存在的限制是可声明的 Agent 不要求已登记：用户级访问记录接纳该用户的任意具体 Agent，chat 需要对应的 Agent Profile，导入与 MCP 不要求。请求级提交只能检索、读取、写入与更新，RUN 与 CALL 仍需要执行线程；
- 工作 actor 是普通值，进程内的防误用由构造、持有与调用位置的架构测试承担，不隔离刻意违规的进程内代码；
- Patchouli 的交互提交（含状态查询与等待）与主动记忆意图提交路由只接收 `IdentityScope`、不做操作授权。交互提交路由唯一的生产调用方是 root API 的提交交互，意图提交路由唯一的生产调用方是 actor API，在 `memory.write`/`memory.update` 授权之后调用。新的调用方必须在自己的授权点先完成授权；
- WorkspaceAsset 的存储、解析服务与既有资产端口仍以 `IdentityScope` 做归属检查，System 的资产桥接 `AssetMaterializationReader` 为此在一次租借内自行组装 scope，是 `IdentityScope` 构造位置的登记例外；
- 第 3 阶段只接受目标等于驻留 Workspace，跨 Workspace 的授权模型尚不存在；后续方向见[身份与访问体系 Idea](../ideas/identity-and-access-model.md)。

## 9. 代码与测试入口

- 模型：[`identity.py`](../../src/hivememory/core/models/identity.py)、[`workspace.py`](../../src/hivememory/core/models/workspace.py)、[`core/access.py`](../../src/hivememory/core/access.py)（操作目录、工作 actor、调用来源身份与端口）；
- 授权：[`workspace/authorization.py`](../../src/hivememory/workspace/authorization.py)；系统阶段的授权：[`workspace/process/`](../../src/hivememory/workspace/process/)；导入与落位的 root 授权：[`import_bus_service.py`](../../src/hivememory/system/application/import_bus_service.py)；声明解析与请求的工作 actor：[`server/deps.py`](../../src/hivememory/server/deps.py)；
- 资源边界：[`topic.py`](../../src/hivememory/core/models/topic.py)、[`memory.py`](../../src/hivememory/core/models/memory.py)、[`artifact.py`](../../src/hivememory/core/models/artifact.py)、[`ShortTermMemoryStore`](../../src/hivememory/patchouli/memory_library/stores.py)、[`PerceptionFamiliar`](../../src/hivememory/patchouli/services/perception.py)；
- 测试：[`tests/unit/workspace/test_access.py`](../../tests/unit/workspace/test_access.py)、[`tests/unit/architecture/test_access_boundaries.py`](../../tests/unit/architecture/test_access_boundaries.py)、[`test_identity_resolution.py`](../../tests/unit/server/routers/test_identity_resolution.py)、[`test_import_bus.py`](../../tests/unit/server/routers/test_import_bus.py)（导入的声明、准入与 `system` 拒绝）、[`test_import_bus_submission.py`](../../tests/integration/system/test_import_bus_submission.py)（交互的发起者是参与者，目标不是参与者驻留的 Workspace 时被拒）、[`test_chat_run_control_contract.py`](../../tests/unit/workspace/process/test_chat_run_control_contract.py)（注册的两次认证与取消的 root 授权）、[`test_gateway_chat_flow.py`](../../tests/unit/workspace/process/test_gateway_chat_flow.py)（系统阶段以 root scope 执行、检索按执行者授权）、[`tests/unit/core/models/test_workspace.py`](../../tests/unit/core/models/test_workspace.py)、[`test_application_access_boundary.py`](../../tests/integration/workspace/test_application_access_boundary.py)、[`test_workspace_access_propagation.py`](../../tests/integration/system/test_workspace_access_propagation.py)、[`test_memory_workspace_isolation.py`](../../tests/integration/patchouli/test_memory_workspace_isolation.py)、[`test_topic_access_chain.py`](../../tests/integration/patchouli/test_topic_access_chain.py)。
