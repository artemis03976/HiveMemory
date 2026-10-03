---
title: 身份与访问体系
status: idea
horizon: current
serves_version: v0.7.0
owner: project
scope: actor-identity-access-context-identity-scope-and-resource-identity
code_paths:
  - src/hivememory/core/models/identity.py
  - src/hivememory/core/access.py
  - src/hivememory/workspace/access.py
  - src/hivememory/workspace/authentication.py
  - src/hivememory/workspace/registry.py
  - src/hivememory/system/access/
  - src/hivememory/server/deps.py
  - src/hivememory/workspace/capability/
  - src/hivememory/workspace/process/
  - src/hivememory/patchouli/application/access_consumption.py
related_docs:
  - docs/ideas/workspace-network-task-process-architecture.md
  - docs/plans/v0.7.0-a1-access-boundary-rework.md
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/ideas/external-session-and-topic-projection.md
  - docs/architecture/workspace.md
last_reviewed: 2026-10-03
---

# 身份与访问体系

**文档状态**：Idea，未形成实施承诺
**记录日期**：2026-10-03

## 0. 文档性质

owner 于 2026-10-03 决定把身份与访问作为一个独立的体系，在本文集中讨论。起因是 A1 返工的实现暴露出两个现象：访问 context 只剩 `IdentityScope` 一个字段；代码里 `identity_scope` 与 `access` 两个身份字段随意混用。这说明 A1 交付时建立的权限体系与身份之间的边界是错的。这本应是一个独立且复杂的体系，但鉴于 A1 的错误实现，需要在 v0.7.0 内正式建立，否则后续漂移会更严重。

- **与总 Idea 第三部分的分工**：[总 Idea](./workspace-network-task-process-architecture.md)第三部分讨论认证与授权的流程（两阶段认证、能力层操作授权、管理员直接通道、P-1–P-10）；本文界定在这些流程中流动的身份数据：actor 身份、访问 context、`IdentityScope` 与资源身份分别是什么、在哪里产生、可以流向哪里。
- **与 A1 返工计划的关系**：[A1 返工计划](../plans/v0.7.0-a1-access-boundary-rework.md)的第一版实现已提交（commit `37f800e`），但计划第 4 节的设计以“访问 context 携带 `IdentityScope`、资源 owner 校验 context”为前提，与本文冲突。A1 返工计划已于 2026-10-03 按本文第一批改写（第 8 节）。
- 现状事实按 commit `37f800e` 的代码核对（第 6 节）。待决问题只列出选项及其影响，不替 owner 作出选择；选项顺序不代表倾向。

## 1. 前提（owner 提出，2026-10-03）

1. `IdentityScope` 由 `ActorIdentity` 与 `WorkspaceIdentity` 组成，表达一次 workspace 能力层调用中**操作的发起者**与**操作的目标 workspace**，即“一次顶层操作冻结的执行者与 Workspace 访问硬边界”。这样设计，是为了以后 actor 的一次操作可以跨过它当前所在的 workspace，对其他 workspace 做有限访问。
2. 资源本身的身份不由 `IdentityScope` 决定。
3. 访问 context 是纯粹的运行时对象。actor 的行为结束，本次访问也就结束，等同于进程结束；不应出现 actor 运行结束后还使用或尝试访问它的情况。如果出现这种需求，说明需要的不是访问 context。
4. `IdentityScope` 应当在两阶段认证通过之后才组装。通过认证、正式进入 workspace 之前，只有 actor 身份是确定的；通过之后才能确认 workspace 身份。
5. actor 进入 workspace 与 actor 真正进行操作，两者的 workspace 可以不同。这是原本的四阶段权限体系拆分为两阶段认证加两阶段授权的原因。例如：coder actor 要在 WA 里执行任务，创建任务进程时完成两阶段认证，确认 coder 能进入 WA；执行过程中 coder 想去 WB 看有没有需要的资源，对这次操作而言，发起者是 coder，操作对象是 WB，发起者当前在哪里并不重要。
6. 当前的简化：资源的受限穿透访问暂不实现，因此发起者只能对它所在的 workspace 发起操作。
7. `IdentityScope` 的名称保留，逐步分批修正项目中现有的使用点。

## 2. 概念与边界

以下界定由第 1 节整理而来，owner 于 2026-10-03 确认。

| 概念 | 回答的问题 | 何时确定 | 寿命 | 可以出现在哪里 |
|:---|:---|:---|:---|:---|
| `ActorIdentity` | 谁将要执行接下来的任务 | 认证前只是声明；Principal 认证通过后成为已验证的身份 | 长期 | 任何地方 |
| 访问 context | 这个 actor 驻留在哪个 workspace、经由哪个来源接入、属于哪一次运行 | Workspace 认证（准入）通过时签发 | 只在本次运行内（任务进程或请求） | 本次运行的持有者，以及 workspace 内的授权点 |
| `IdentityScope` | 这一次操作由谁发起、作用于哪个 workspace | 每次操作授权时组装 | 只在这一次操作的调用链内 | 授权点以下：资源 owner、引擎、存储 |
| 资源身份 | 资源属于哪个 workspace、由谁产生 | 资源创建时写入 | 随资源持久保存 | 资源本身及其记录 |

几组容易混淆的区别：

- **访问 context 与 `IdentityScope`**：两者都由一个 actor 和一个 workspace 组成，但含义不同。context 里是 actor **驻留**的 workspace，是认证的结果；`IdentityScope` 里是这次操作的**目标** workspace，是授权的结果。当前目标只能是驻留 workspace，两者的值相同，但不能互相代替。
- **`IdentityScope` 与资源身份**：`IdentityScope` 描述一次操作，资源身份描述资源本身。资源被创建时，归属取自操作的目标 workspace，来源取自操作的发起者；此后资源身份就是独立的数据，不随后续操作改变。资源授权比对的是两者，而不是用其中一个代替另一个。
- **actor 身份与访问 context**：同一个 actor 可以同时持有多份访问 context（多个进程或请求）；每份 context 只属于一个 actor 和一次运行。

## 3. 两阶段认证与两阶段授权

| 阶段 | 回答的问题 | 输入 | 产出 | 在哪里做 |
|:---|:---|:---|:---|:---|
| 1 Principal 认证 | 调用来源是否已登记、能否代表这个 actor | principal、adapter、actor 声明 | 已验证的 `ActorIdentity` | System 接入登记（经端口） |
| 2 Workspace 认证 | 这个 actor 能否进入 workspace W | 已验证的 actor、请求进入的 W | 访问 context：actor 驻留在 W，属于本次运行 | workspace guard |
| 3 操作授权 | 这次操作（发起者 → 目标 workspace T）是否被允许 | 访问 context、operation、目标 T | `IdentityScope(actor, T)` | 授权点：能力层、任务进程的阶段检查 |
| 4 资源授权 | 目标资源是否允许这次操作 | `IdentityScope`、资源身份与资源 policy | 允许，或按不可见处理 | 资源 owner（Patchouli 等） |

- 前两阶段在进入 workspace 时完成（任务进程在创建前完成）；后两阶段在每次操作时进行。经网络接入的 actor 每次请求都重新认证（总 Idea 15.3，P-1a）。
- coder 的例子：创建进程时完成第 1、2 阶段，得到“coder 驻留在 WA”的 context。读 WA 的资源时，第 3 阶段取 T = WA，组装 `IdentityScope(coder, WA)`，第 4 阶段由资源 owner 校验。去 WB 查看时，第 3 阶段取 T = WB，只看 coder 能否对 WB 执行这个 operation，与 coder 驻留在 WA 无关。
- 当前的简化（前提第 6 条）：第 3 阶段只接受 T 等于驻留 workspace。跨 workspace 的授权模型（谁能对哪个非驻留 workspace 做什么）不在 v0.7.0。

## 4. 不变量（分析，由第 1–3 节得出）

1. 第 3 阶段之前不存在 `IdentityScope`。入口在认证前只持有 actor 声明与请求进入的 workspace。
2. 访问 context 只出现在本次运行的持有者手里和 workspace 的授权点；不进入资源 owner、Gateway、引擎或存储，不写入任何记录、事件、交互记录或 DTO。
3. 运行结束后不再读取访问 context。运行结束后的需求（交互记录、后台任务）使用 actor 身份或资源身份。进程记录与进程同寿：进程关闭时从进程表移除，此后取消与状态查询都返回 `not_found`；按任务进程 Idea Q-3a，访问 context 进入进程记录，与本条一致。
4. 授权点只接收访问 context，`IdentityScope` 由授权点组装，不由调用方另行传入；授权点以下只接收 `IdentityScope`。
5. 资源 owner 用 `IdentityScope` 与资源身份做资源授权；资源身份只在资源创建时从 `IdentityScope` 写入。
6. 授权规则（例如 W0 的“actor 用户等于 workspace owner”）属于第 2、3 阶段，不作为身份类型本身的约束。

## 5. 访问 context 的内容

按第 2 节，访问 context 承载以下内容：

| 内容 | 说明 | 用途 |
|:---|:---|:---|
| actor | 已验证的 `ActorIdentity` | 第 3 阶段组装 `IdentityScope` 的发起者；查访问登记 |
| 驻留 workspace | 准入的 `WorkspaceIdentity` | 当前是第 3 阶段唯一允许的目标；查访问登记 |
| 来源 | `CallerPrincipal`（I-2：暂时保存 principal） | 认证经由哪个调用来源；关联 P-1b 与错误模型要求可观测的 principal；不参与 operation 授权 |
| 运行绑定 | 运行类型（任务进程或请求）与运行标识 | 失效时点（P-6、P-9b）；两类 context 的区分（P-9c）；其余进程控制操作的授权（P-7）与审计 |

明确不包含：

- **行为白名单快照**：每次授权都按访问登记查询，不缓存授权结论。只有 P-4a 采用“进程持有白名单子集”时，才需要本次运行的 operation 上限；
- **有效期**：失效由绑定的运行结束决定（总 Idea 15.6）；
- **网络凭据**：context 不是远端凭据；
- **会话、trace、交互等关联 ID**：它们不是授权要素，属于运行或交互本身。

以上内容由签发 context 的 guard 内部保存，context 对外是不透明凭据（I-1）；运行绑定在签发时写入（I-3）。

## 6. 现状事实（代码核对，2026-10-03，commit `37f800e`）

### 6.1 访问 context 与认证

- `WorkspaceAccessContext`（`core/access.py`）是不可变对象，唯一的字段是 `identity_scope`，按对象身份判断是否由 guard 签发。
- guard（`workspace/access.py`）以 `WeakSet` 记录已签发的 context，`invalidate` 使单个 context 失效，`close()` 使全部失效；`verify_context` 与 `authorize_operation` 返回 context 里的 `IdentityScope`。
- 认证网关的 `authenticate(adapter, principal, actor, workspace)` 一次完成两阶段认证并签发 context，没有运行绑定参数。
- server 在认证之前就组装完整的 `IdentityScope`（`server/deps.py` 的 `resolve_request_identity_scope`），再把其中的 actor 与 workspace 交给网关。
- chat 链路的顺序（`server/routers/chat.py`、`workspace/process/`）：server 路由生成 `process_id` → 组装 `IdentityScope` → server 调用网关取得 context（未绑定）→ 调用注册入口 `run_process`，它在流开始迭代时才校验 context、创建进程并登记到进程表。两阶段认证由 server 完成，注册入口只做校验。
- 进程记录在进程关闭时从进程表移除（`ProcessTable.close`），此后取消与状态查询返回 `not_found`。
- 两个登记文件、用户级访问记录、请求级 context 与绑定进程的 context 已按 A1 返工计划实现。

### 6.2 `IdentityScope` 的定义

- 注释为“一次顶层操作冻结的执行者与 Workspace 访问硬边界”，即前提第 1 条的含义。
- 校验器 `_require_same_owner` 要求 actor 用户等于 workspace owner：W0 的准入规则写进了身份类型；guard 准入时也检查同一规则。
- `ActorIdentity` 带有兼容字段 `session_id`（[外部会话 Idea](./external-session-and-topic-projection.md) 第 8 节第 3 项已记录）。

### 6.3 `IdentityScope` 实际承担的角色

`src/` 中有 20 个类把 `IdentityScope` 作为字段保存。按第 2 节分类（分析）：

| 实际角色 | 对应第 2 节的概念 | 类 |
|:---|:---|:---|
| 驻留与运行身份 | 访问 context，或运行自身的记录 | `WorkspaceAccessContext`、`ProcessRecord`、`ProcessRequest`、`RuntimeScope`、`AgentRunContext`、`CPUInputManifest` |
| 一次操作的发起者与目标 | `IdentityScope` | `RetrievalRequest`、`RetrievalQuery`、`GatewayExecutionState`、`GatewayStateSnapshot`、`CandidateTopicsInput`、`RoutedTopicInput` |
| 记录与后台任务的归属和来源 | 资源身份 | `PreparedAgentRun`、`InteractionSubmission`、`FlushEvent`、`TopicMaterializeTask`、`MemoryGenerationTask`、`MemoryGenerationTaskSpec`、`PendingAtomMaterializeTask`、`LeaseToken` |

### 6.4 两个身份字段的混用

- 54 个函数同时接收 `identity_scope` 与 `access`：Patchouli 21 个、能力层 12 个、server 路由 9 个、任务进程 6 个、Gateway 6 个。
- Patchouli application 经 `access_consumption` 校验 context，并用 `_assert_scope_consistency` 核对两个字段是否一致。`core.access.WorkspaceAccessVerifier` 端口只有 Patchouli 使用。
- 例：能力层的 `create_memory` 先调用 `authorize_operation(access, …)`，但没有使用它返回的 scope，而是用调用方传入的 `identity_scope` 写原子的 `workspace_identity` 与来源；两者不一致时，要到 Patchouli 的一致性检查才会被拒绝。附件上传的 P1 缺陷是同一形态，`37f800e` 中以一致性检查修复。
- 任务进程的 `ProcessRequest`、`ProcessRecord` 同时保存两个字段；同一个 context 被 `ProcessRequest.access`、`TaskProcess._access` 与 `ProcessRecord.access` 三处持有。Gateway 处理路由与 Patchouli 阶段路由同时接收两个字段。

### 6.5 治理规则

AGENTS.md 第 3 节写有“`IdentityScope`（Actor + Workspace）必须沿应用服务、公共 route、Interaction 和后台任务传播，并在资源 owner 处再次校验”。这条规则把 `IdentityScope` 用于交互记录与后台任务，对应 6.3 的第三类。

## 7. 待决问题

### I-1 访问 context 的对外形态

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 不透明凭据：没有公开字段，guard 内部保存第 5 节的内容；授权点经 guard 换得 actor 与驻留 workspace | 从 context 读身份在结构上不可能，不变量 2–4 由结构保证；调试与日志需经 guard 查询 |
| B | 保留只读字段（actor、驻留 workspace 等） | 实现简单；字段仍可能被当作身份读取，不变量 2–4 只能靠约定与审查 |
| C | 其他 | —— |

**owner 决定（2026-10-03）**：选项 A。访问 context 改为不透明凭据，对外没有公开字段，第 5 节的内容由签发它的 guard 内部保存。

- 身份只在授权点经 guard 兑现：授权点把 context 交给 guard，由 guard 返回所需的身份，例如第 3 阶段组装的 `IdentityScope`；
- guard 只注入给授权点（注册入口、进程的阶段检查、能力层、取消入口）；Gateway、Patchouli、CPU 不持有 guard，即使拿到 context 也取不出身份；
- server 使用认证前自己持有的声明（actor 声明与请求进入的 workspace），认证成功即确认了这份声明，不从 context 读回；
- guard 可以提供只用于日志与诊断的查询，它不能成为第二个兑现入口。

### I-2 来源（principal、adapter）是否随 context 保存

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 保存 | 可支持 P-1b（运行期间的请求须来自注册时的 principal）与拒绝记录 |
| B | 不保存 | 与 A1 交付时“context 不持有来源 principal”一致；P-1b 若采用，需要另找依据 |

**owner 决定（2026-10-03）**：暂时保存 principal。adapter 是否一并保存本次未涉及。

### I-3 运行绑定何时写入

**背景**：认证与进程创建的先后已有以下决定：

- 总 Idea 前提第 2、3 条：借由唯一的任务请求注册入口进行两阶段认证；未通过两阶段认证的请求不创建任务进程；
- 总 Idea 15.3（P-1a，2026-09-27）：注册前经认证网关验证身份，未通过不予注册；
- 任务进程 Idea 1.2（2026-09-28）：进程在最开始创建，完成两阶段认证后立即创建，认证信息由进程携带；
- 任务进程 Idea Q-3（2026-09-28）：入口只管理任务进程的生命周期，按选项 A 的方向，入口负责进程标识、准入认证、请求方、状态、取消与停机收尾；Q-3a：访问 context 进入进程记录；
- 总 Idea 14.1 的流程图：认证通过后“创建任务进程，签发 context 并绑定进程”；
- 任务进程 Idea Q-16：`process_id` 由 server 入口在进入编排服务前生成。

分析：这些决定合起来指向“由注册入口完成两阶段认证，签发 context 的同时绑定本进程，并立即创建进程”，即下表的选项 A；`process_id` 在进入注册入口前已经存在，签发时就能绑定。现状与之不同（6.1）：chat 链路的认证由 server 路由完成，注册入口只校验 context；context 签发时没有绑定，进程要到流开始迭代时才创建。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 签发时写入：server 在进入服务前已生成 `process_id`（任务进程 Idea Q-16），可随认证一起交给网关 | 注册入口可以校验 context 绑定的就是本进程；网关认证接口增加绑定参数 |
| B | 进程创建时再绑定 | 认证接口不变；签发到绑定之间存在未绑定的 context |
| C | 不记录绑定，由持有者在运行结束时使其失效 | 即现状；两类 context 无法区分（P-9c） |

**owner 决定（2026-10-03）**：选项 A，与上述已有决定一致：由注册入口完成两阶段认证，签发 context 时写入运行绑定，并立即创建进程。

- 由此，chat 链路的认证从 server 移到注册入口：server 把自身的 principal、adapter，以及 actor 声明与请求进入的 workspace 交给注册入口（分析）；
- 不建进程的请求级 context 仍由 server 按请求认证，绑定本次请求（分析）。

**注册入口的形态（owner，2026-10-03）**：先注册、后运行。注册入口提供立即执行的注册步骤：认证、签发即绑定、创建进程并登记到进程表，失败直接抛出，HTTP 入口据此返回 403；流式响应只负责运行。注册成功但流一直没有开始时，由注册入口负责关闭进程。原 [TaskProcess 容器 Todo](../todo/task-process-container-ownership.md)中“登记与注销由入口负责”一项随之并入 A1 返工计划。

### I-4 第 3 阶段如何指定目标 workspace

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 授权点接口显式接收目标 workspace，当前只接受等于驻留 workspace | 将来实现受限穿透访问时只需放宽规则；现有调用点都要传目标 |
| B | 当前省略，授权点默认取驻留 workspace | 当前改动小；将来实现受限穿透访问时要改所有授权点的签名 |

**owner 决定（2026-10-03）**：选项 A。授权点接口改为显式接收目标 workspace；当前只接受等于驻留 workspace 的目标，其余一律拒绝。

### I-5 `_require_same_owner` 的去向

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 移到第 2、3 阶段 | 身份类型不再承担授权规则；准入时已检查同一规则，当前行为不变；直接构造 `IdentityScope` 的测试会受影响 |
| B | 保留到受限穿透访问实现时再移 | 当前不改；与不变量 6 暂时不一致 |

**owner 决定（2026-10-03）**：选项 A。owner 校验从 `IdentityScope` 移到第 2、3 阶段：准入时检查 actor 用户与要进入的 workspace 的 owner，操作授权时检查 actor 用户与目标 workspace 的 owner（W0 基线）；`IdentityScope` 不再带 `_require_same_owner` 校验器。

### I-6 资源身份的表达

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 记录与后台任务改用明确的归属（`WorkspaceIdentity`）与来源（`ActorIdentity`）字段 | 语义直接；改动面覆盖 6.3 第三类的全部类 |
| B | 定义专门的资源身份类型 | 只在一处定义；需要新类型与迁移 |
| C | 其他 | —— |

### I-7 `ActorIdentity.session_id`

与外部会话 Idea 第 8 节第 3 项关联。选项：去掉兼容字段 / 保留但不参与相等性与缓存键 / 随 ConversationSession 方向一并处理。

### I-8 进程记录如何持有身份

按任务进程 Idea Q-3a，访问 context 进入进程记录；进程记录与进程同寿（不变量 3）。现状是同一个 context 有三处持有者，记录还同时保存调用方传入的 `IdentityScope`（6.1、6.4）。进程记录里的身份现在有三个读取点：

| 读取点 | 现状 |
|:---|:---|
| 取消与状态查询时比对请求方与进程 | 进程表比对两者的 workspace 身份（`ProcessTable.get`） |
| 阶段调用的目标 workspace | 各阶段调用直接传 `request.identity_scope`；按 I-4，授权点要显式接收目标 workspace |
| 运行时事件的标签 | `for_process` 在进程创建时绑定一次 `workspace_id`、`agent_id`，取自 `record.identity_scope` |

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 记录只持有访问 context；三个读取点需要身份时都经 guard 即时解析 | 只有一个来源；事件发布器与进程表也要持有 guard，与 I-1“guard 只注入给授权点”冲突；所有读取都必须发生在 context 失效之前 |
| B | 记录持有访问 context，并在创建时经 guard 取得发起者与驻留 workspace 写入记录 | 读取方便；记录与 guard 各存一份身份，记录中的字段只能由 guard 写入一次，不能来自调用方 |
| C | 记录只持有访问 context 与进程自身的元数据，不保存身份；三个读取点按各自性质处理（见下） | 记录没有身份字段，也就没有重复；只有授权判断经过 guard |

选项 C 中三个读取点的处理：

- **取消与状态查询**：这是一次授权判断（P-7：谁可以控制这个进程），由取消入口这个授权点交给 guard，比对请求方的 context 与进程记录中的 context；不匹配时返回 `not_found`，不泄露进程是否存在；
- **阶段调用的目标**：I-4 把“要操作哪个 workspace”（调用方的意图）与“被准入到哪个 workspace”（凭据）分开。阶段调用的目标是任务注册时声明并通过认证的 workspace，作为任务参数由进程传给授权点，guard 核对目标是否允许；不从凭据读回驻留 workspace 作为目标。当前两者的值相同，含义不同，实现受限穿透访问后会出现不同；
- **事件标签**：它是观测标签（字符串），不是身份；AGENTS.md 规定 `workspace_id` 观测标签不等于授权或分区。进程创建时用通过认证的注册声明绑定一次，此后不再改变。

**owner 决定（2026-10-03）**：选项 C。进程记录的形态如下（示意）：

```python
@dataclass
class ProcessRecord:                     # 进程元数据，与进程同寿
    process_id: str
    access: WorkspaceAccessContext       # 凭据：只交给 guard 用于授权
    phase: ProcessPhase
    outcome: ProcessOutcome
    stop_reason: str | None
    # 不保存 actor、驻留 workspace 等身份字段
```

- context 只由进程记录持有：注册请求只携带声明（actor 声明、请求进入的 workspace、server 的 principal 与 adapter），不携带 context；注册入口认证成功后直接把 context 写入记录（I-3）；`TaskProcess` 经记录使用它，不另存；
- `TaskProcess` 反向持有进程表与各类组件的结构问题单独登记为 [Todo](../todo/task-process-container-ownership.md)，不在本 Idea 内处理。

### I-9 CPU 在过渡期的身份

Alice 改经能力层调用之前（总 Idea 15.5），CPU 输入清单携带一个 `IdentityScope`，Alice 用它直接调用 Patchouli。现状是它直接取自调用方传入的 scope（`workspace/process/allocation.py`）。按不变量 4，`IdentityScope` 只由授权点经 guard 组装，但 CPU 执行本身没有对应的 operation。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | guard 提供过渡专用的组装方法：只做第 3 阶段的目标 workspace 与 owner 检查，不检查 operation；进程在 CPU 分配时调用 | 新增一个过渡接口；Alice 的直接调用没有 operation 授权，与现状相同；Alice 迁移完成后删除 |
| B | 借用某次 operation 授权的结果，例如进入执行前 `interaction.submit` 检查返回的 scope | 不新增接口；CPU 的执行身份挂在一个不相关的 operation 上 |
| C | 用注册时通过认证的声明直接组装 | 最简单；违反不变量 4 |

**owner 决定（2026-10-03）**：选项 A。该方法只供任务进程在 CPU 分配时使用，Alice 的能力层调用迁移完成后删除。

## 8. 分批

owner 于 2026-10-03 决定：建立独立 Idea；`IdentityScope` 名称保留，逐步分批修正项目中的使用点。各批范围为当前设想，形成计划时细化。

| 批次 | 范围 | 关系 |
|:---|:---|:---|
| 第一批 | workspace 边界：入口在认证前只持有 actor 声明与请求进入的 workspace；访问 context 按第 5 节重新定义为不透明凭据（I-1、I-2）；注册入口完成认证、签发即绑定并立即创建进程（I-3）；授权点显式接收目标 workspace 并组装 `IdentityScope`，当前目标只能是驻留 workspace（I-4）；owner 校验移到第 2、3 阶段（I-5）；注册入口先注册、后运行，并负责进程的登记与注销（I-3）；CPU 输入清单的 `IdentityScope` 由 guard 的过渡方法组装（I-9）；按不变量 2，资源 owner 与 Gateway 只接收 `IdentityScope`，Patchouli 不再消费访问 context（`access_consumption` 与 `WorkspaceAccessVerifier` 端口随之失去用途）；进程记录只持有 context 与进程元数据，不保存身份：取消经 guard 比对、阶段调用的目标取自任务参数、事件标签在创建时绑定（I-8） | 即 A1 返工计划，已于 2026-10-03 按此改写；`37f800e` 的实现按该计划第 7 节保留或调整 |
| 第二批 | 记录与后台任务改用资源身份（6.3 第三类）；修订 AGENTS.md 第 3 节的相应规则（修改 AGENTS.md 需 owner 同意） | 在第一批之后 |
| 不在 v0.7.0 | 资源的受限穿透访问（跨 workspace 的授权模型） | 模型在第 3 阶段的目标 T 处预留 |

第一批对已记录决定的影响（分析）：

- 总 Idea 15.5 的“放行分支分两步去掉”：资源 owner 不再接收访问 context 后，Patchouli 一侧不再有“缺少 access 时放行”的分支，Alice 绕过能力层的问题改由 Alice 的能力层调用迁移解决（前提第 4 条）；
- 总 Idea 15.6 的“两个提交路由的检查暂留在 Patchouli”：失去前提，这两个路由的 operation 检查在能力层出现对应方法时进行；
- 总 Idea 15.6 的“阶段调用的 operation 检查放在进程内”、“去掉 context 的固定有效期”、“两类登记各用一个配置文件”以及 15.5 的其余决定仍然成立。

## 9. 与其他文档的关系

| 文档 | 关系 |
|:---|:---|
| [总 Idea](./workspace-network-task-process-architecture.md)第三部分 | 认证与授权的流程与待决问题（P-1、P-4a、P-9 等）仍在那里；本文界定其中流动的身份数据 |
| [A1 返工计划](../plans/v0.7.0-a1-access-boundary-rework.md) | 第一批的实施计划，已于 2026-10-03 改写 |
| [任务进程 Idea](./task-process-table-and-registration-entry.md) | Q-3a 已决定访问 context 进入进程记录，进程记录与进程同寿，与不变量 3 一致；进程记录如何持有身份见 I-8；认证与进程创建的顺序见 I-3 |
| [外部 Actor Idea](./external-actor-registration-and-runtime-access.md) | principal 与 adapter 的登记（I-2）；plugin 模式下不建进程的访问同样遵循本文的边界 |
| [外部会话 Idea](./external-session-and-topic-projection.md) | `session_id` 的处理（I-7） |
| [Workspace 架构](../architecture/workspace.md)第 4 节 | A1 交付时的事实描述；各批完成后按最终代码改写 |

## 10. 形成计划的条件

- 满足 [Ideas 升级规则](./README.md#升级规则)与[文档治理规范](../DOCUMENTATION.md)第 8.3 节；
- 第一批所需的 I-1 至 I-5、I-8 与 I-9 已于 2026-10-03 决定；第二批需要 I-6、I-7 的决定；
- 第一批不另开计划，改写 A1 返工计划（同一目标方向只有一份生效计划）。
