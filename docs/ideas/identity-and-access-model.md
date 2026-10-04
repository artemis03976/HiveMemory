---
title: 身份与访问体系
status: idea
horizon: current
serves_version: v0.7.0
owner: project
scope: actor-identity-access-context-identity-scope-and-resource-ownership
code_paths:
  - src/hivememory/core/models/identity.py
  - src/hivememory/core/access.py
  - src/hivememory/core/memory_access.py
  - src/hivememory/workspace/authentication.py
  - src/hivememory/workspace/authorization.py
  - src/hivememory/workspace/registry.py
  - src/hivememory/system/access/
  - src/hivememory/server/deps.py
  - src/hivememory/workspace/capability/
  - src/hivememory/workspace/process/
  - src/hivememory/patchouli/
  - src/hivememory/engines/
related_docs:
  - docs/ideas/workspace-network-task-process-architecture.md
  - docs/archive/plans/v0.7.0-a1-access-boundary-rework.md
  - docs/plans/v0.7.0-identity-access-batch-2.md
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/ideas/external-session-and-topic-projection.md
  - docs/ideas/pending-intent-migration.md
  - docs/architecture/workspace.md
last_reviewed: 2026-10-04
---

# 身份与访问体系

**文档状态**：Idea，未形成实施承诺；第一批已实施，第二批已建立计划（[v0.7.0 第二批](../plans/v0.7.0-identity-access-batch-2.md)），尚未实施
**记录日期**：2026-10-03；2026-10-04 按“已完成 / 未完成”重新整理，同日记录 I-6、I-6a、I-7、I-11 的决定，按 I-6 把资源一侧的概念改为“资源归属”，并把第二批扩大为 Patchouli 全系统的重构

## 0. 文档性质

owner 于 2026-10-03 决定把身份与访问作为一个独立的体系，在本文集中讨论。起因是 A1 返工的实现暴露出两个现象：访问 context 只剩 `IdentityScope` 一个字段；代码里 `identity_scope` 与 `access` 两个身份字段随意混用。这说明 A1 交付时建立的权限体系与身份之间的边界是错的。这本应是一个独立且复杂的体系，但鉴于 A1 的错误实现，需要在 v0.7.0 内正式建立，否则后续漂移会更严重。

- **与总 Idea 第三部分的分工**：[总 Idea](./workspace-network-task-process-architecture.md)第三部分讨论认证与授权的流程（两阶段认证、能力层操作授权、管理员直接通道、P-1–P-10）；本文界定在这些流程中流动的身份数据：actor 身份、访问 context、`IdentityScope` 与资源归属分别是什么、在哪里产生、可以流向哪里。两者冲突时，身份数据的界定以本文为准。
- **阅读方式**：
  - 第 1–5 节是身份模型（owner 于 2026-10-03 确认），已随第一批实施，当前事实见 [Workspace 架构](../architecture/workspace.md)第 4 节；其中资源一侧与资源 owner 内部的界定（第 2 节、第 3 节的第 4 阶段、不变量 3、4、5、7）于 2026-10-04 随 I-6、I-11 修订，尚未实施；
  - 第 6 节是第一批之后、与第二批相关的现状；
  - 第 7 节是问题：7.1 是已完成的问题，按“问题—实际设计”叙述，注明决定日期与实施状态；7.2 是未完成的问题，只列选项及其影响，选项顺序不代表倾向；
  - 第 8 节是分批。
- **2026-10-04 的整理**：问题编号 I-1–I-10 不变。原先在决定之后陆续追加的补充（I-1 的机制修订、I-8 的两次补充、I-10 的补充）已并入各问题的最终设计，被取代的中间方案只在各问题的“演进”中简述；第一批实施前的代码快照（原第 6 节）已删除。整理前的最后版本见 commit `2daa332`。

## 1. 前提（owner 提出，2026-10-03）

1. `IdentityScope` 由 `ActorIdentity` 与 `WorkspaceIdentity` 组成，表达一次 workspace 能力层调用中**操作的发起者**与**操作的目标 workspace**，即“一次顶层操作冻结的执行者与 Workspace 访问硬边界”。这样设计，是为了以后 actor 的一次操作可以跨过它当前所在的 workspace，对其他 workspace 做有限访问。
2. 资源本身的身份不由 `IdentityScope` 决定。
3. 访问 context 是纯粹的运行时对象。actor 的行为结束，本次访问也就结束，等同于进程结束；不应出现 actor 运行结束后还使用或尝试访问它的情况。如果出现这种需求，说明需要的不是访问 context。
4. `IdentityScope` 应当在两阶段认证通过之后才组装。通过认证、正式进入 workspace 之前，只有 actor 身份是确定的；通过之后才能确认 workspace 身份。
5. actor 进入 workspace 与 actor 真正进行操作，两者的 workspace 可以不同。这是原本的四阶段权限体系拆分为两阶段认证加两阶段授权的原因。例如：coder actor 要在 WA 里执行任务，创建任务进程时完成两阶段认证，确认 coder 能进入 WA；执行过程中 coder 想去 WB 看有没有需要的资源，对这次操作而言，发起者是 coder，操作对象是 WB，发起者当前在哪里并不重要。
6. 当前的简化：资源的受限穿透访问暂不实现，因此发起者只能对它所在的 workspace 发起操作。
7. `IdentityScope` 的名称保留，逐步分批修正项目中现有的使用点。

注（2026-10-04，I-6）：项目中没有独立的“资源身份”概念，第 2 条的“资源本身的身份”指资源归属：它在资源创建时取自操作的目标 workspace，此后不随后续操作改变。

## 2. 概念与边界

以下界定由第 1 节整理而来，owner 于 2026-10-03 确认；资源一侧于 2026-10-04 按 I-6 修订。

| 概念 | 回答的问题 | 何时确定 | 寿命 | 可以出现在哪里 |
|:---|:---|:---|:---|:---|
| `ActorIdentity` | 谁将要执行接下来的任务 | 认证前只是声明；Principal 认证通过后成为已验证的身份 | 长期 | 任何地方 |
| 访问 context | 这个 actor 驻留在哪个 workspace、经由哪个来源接入、属于哪一次运行 | Workspace 认证（准入）通过时签发 | 只在本次运行内（任务进程或请求） | 本次运行的持有者，以及 workspace 内的授权点 |
| `IdentityScope` | 这一次操作由谁发起、作用于哪个 workspace | 每次操作授权时组装 | 只在这一次操作的调用链内 | 授权点以下：资源 owner 的公开路由、Gateway；资源 owner 内部拆为归属与发起者，不再传递 `IdentityScope`（I-6，2026-10-04 修订） |
| 资源归属 | 资源属于哪个 workspace | 资源创建时取自操作的目标 workspace | 随资源持久保存 | 资源本身及其记录 |

资源一侧没有“身份”的概念，只有归属（`WorkspaceIdentity`，I-6）。资源还可能带有两类数据，都不是身份：

- **资源 policy**：资源自身的授权数据，第 4 阶段据此判断发起者能否看到资源；目前只有 MemoryAtom 带 policy（`MemoryAccessPolicy`，6.6）；
- **来源**：资源由谁产生，是历史信息，对访问没有约束力。

几组容易混淆的区别：

- **访问 context 与 `IdentityScope`**：两者都由一个 actor 和一个 workspace 组成，但含义不同。context 里是 actor **驻留**的 workspace，是认证的结果；`IdentityScope` 里是这次操作的**目标** workspace，是授权的结果。当前目标只能是驻留 workspace，两者的值相同，但不能互相代替。
- **`IdentityScope` 与资源归属**：`IdentityScope` 描述一次操作，资源归属描述资源属于哪个 workspace。归属在资源创建时取自操作的目标 workspace，此后是资源自己的数据，不随后续操作改变。第 4 阶段比对的是 `IdentityScope` 与资源的归属和 policy，不用其中一个代替另一个。
  - 2026-10-04 随 I-6 修订：原先的“资源身份”由归属与来源组成，现取消这一概念，演进见 I-6。
- **资源的来源字段与后台任务的发起者**：来源字段是资源上的历史信息（例如记忆的 `MemoryProvenance` 与贡献者集合），对访问没有约束力；后台任务的发起者回答“这项后台操作由谁发起”，任务执行中的读取以它为可见性主体，与一次普通操作的发起者相同。后台任务携带发起者，并不意味着来源字段成为访问条件。非主动生成路径没有任何 agent 的主动意图，发起者是 `system`（I-11）。
- **actor 身份与访问 context**：同一个 actor 可以同时持有多份访问 context（多个进程或请求）；每份 context 只属于一个 actor 和一次运行。

## 3. 两阶段认证与两阶段授权

| 阶段 | 回答的问题 | 输入 | 产出 | 在哪里做 |
|:---|:---|:---|:---|:---|
| 1 Principal 认证 | 调用来源是否已登记、能否代表这个 actor | principal、adapter、actor 声明 | 已验证的 `ActorIdentity` | System 接入登记（经端口） |
| 2 Workspace 认证 | 这个 actor 能否进入 workspace W | 已验证的 actor、请求进入的 W | 访问 context：actor 驻留在 W，属于本次运行 | `WorkspaceAuthenticator`，由认证网关调用（I-10） |
| 3 操作授权 | 这次操作（发起者 → 目标 workspace T）是否被允许 | 访问 context、operation、目标 T | `IdentityScope(actor, T)` | 授权点：能力层、任务进程的阶段检查，均经 `WorkspaceOperationAuthorizer`（I-10） |
| 4 资源授权 | 目标资源是否允许这次操作 | `IdentityScope`、资源归属与资源 policy | 允许，或按不可见处理 | 资源 owner（Patchouli 等） |

- 前两阶段在进入 workspace 时完成（任务进程在创建前完成）；后两阶段在每次操作时进行。经网络接入的 actor 每次请求都重新认证（总 Idea 15.2，P-1a）。
- **第 4 阶段分两步**（I-6；代码现状见 6.6）：
  1. **归属**：资源归属等于 `IdentityScope` 的目标 workspace。这是硬边界，任何读取视角都不跳过。它常以分区键或查询过滤的形式完成，看起来像不需要检查；但缓存、资产索引等是进程级共享设施，存储的预过滤也不是授权事实，“另一个 workspace 的资源不会出现在当前 workspace”正是这一步在资源 owner 处成立的结果，因此不能省略。实现受限穿透访问后，同一进程内还会同时出现多个目标 workspace 的资源。
  2. **可见性**：资源 policy 是否允许发起者。没有 policy 的资源只按归属授权。这一步随读取视角变化：owner 视角的管理读取跳过它（总 Idea 15.7）。

  来源不参与第 4 阶段。
- coder 的例子：创建进程时完成第 1、2 阶段，得到“coder 驻留在 WA”的 context。读 WA 的资源时，第 3 阶段取 T = WA，组装 `IdentityScope(coder, WA)`，第 4 阶段由资源 owner 校验。去 WB 查看时，第 3 阶段取 T = WB，只看 coder 能否对 WB 执行这个 operation，与 coder 驻留在 WA 无关。
- 当前的简化（前提第 6 条）：第 3 阶段只接受 T 等于驻留 workspace。跨 workspace 的授权模型（谁能对哪个非驻留 workspace 做什么）不在 v0.7.0。

## 4. 不变量（分析，由第 1–3 节得出）

1. 第 3 阶段之前不存在 `IdentityScope`。入口在认证前只持有 actor 声明与请求进入的 workspace。
2. 访问 context 只出现在本次运行的持有者手里和 workspace 的授权点；不进入资源 owner、Gateway、引擎或存储，不写入任何记录、事件、交互记录或 DTO。
3. 运行结束后不再读取访问 context。运行结束后的需求（交互记录、后台任务）以独立字段携带归属与（需要时）发起者，不保存 `IdentityScope`（I-6、I-6a，2026-10-04 修订）。进程记录与进程同寿：进程关闭时从进程表移除，此后取消与状态查询都返回 `not_found`；按任务进程 Idea Q-3a，访问 context 进入进程记录，与本条一致。
4. 授权点只接收访问 context，`IdentityScope` 由授权点组装，不由调用方另行传入；授权点以下的公开边界（资源 owner 的公开路由、Gateway）只接收 `IdentityScope`。资源 owner 在内部把它拆为归属与发起者分别传递，内部不再传递或重新组装 `IdentityScope`（I-6，2026-10-04 修订）。
5. 资源 owner 用 `IdentityScope` 与资源的归属、policy 做资源授权（第 3 节第 4 阶段）；归属只在资源创建时取自 `IdentityScope` 的目标 workspace；来源不参与资源授权（I-6，2026-10-04 修订）。
6. 授权规则（例如 W0 的“actor 用户等于 workspace owner”）属于第 2、3 阶段，不作为身份类型本身的约束。
7. 与创建者相关的权限（例如将来 P-8 可能出现的“只有创建者能修改”）在资源创建时写入资源 policy，授权时不读取来源；否则来源会重新成为访问条件（2026-10-04 新增，由 I-6 得出）。

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

以上内容在签发时密封在 context 内，context 没有公开字段（I-1）；运行绑定在签发时写入（I-3）。

## 6. 现状事实（代码核对，2026-10-04，第一批实施之后）

第一批落地的部分（访问 context、认证与授权的划分、注册入口、进程记录与句柄）是当前事实，见 [Workspace 架构](../architecture/workspace.md)第 4 节与 [System 应用服务](../system/application-services.md)第 3、4 节，本节不重复。以下是与第二批相关的现状。

### 6.1 仍以 `IdentityScope` 作为字段的类

`src/` 中有 17 个类把 `IdentityScope` 作为类字段保存（2026-10-04 复核，此前记为 18 个）。按第 2 节分类（分析）：

| 实际角色 | 对应第 2 节的概念 | 类 |
|:---|:---|:---|
| 运行自身的身份 | 一次运行（Alice run、CPU 执行）的执行身份 | `RuntimeScope`、`AgentRunContext`、`CPUInputManifest`（由过渡方法组装，I-9） |
| 一次操作的发起者与目标 | `IdentityScope` | `RetrievalRequest`、`RetrievalQuery`、`GatewayExecutionState`、`GatewayStateSnapshot`、`CandidateTopicsInput`、`RoutedTopicInput` |
| 记录与后台任务的归属和发起者 | 归属与发起者（两个独立字段） | `PreparedAgentRun`、`InteractionSubmission`、`FlushEvent`、`TopicMaterializeTask`、`MemoryGenerationTask`、`MemoryGenerationTaskSpec`、`PendingAtomMaterializeTask`、`LeaseToken` |

第三类是第二批的直接对象：这些记录与任务在产生它们的那次操作结束之后仍然存在，保存的却是“一次操作”的身份。第二批同时重构 Patchouli 内部对 `IdentityScope` 的使用（6.5，I-6）；第二类中由 Patchouli 内部使用的类（如 `RetrievalQuery`）随之按 I-6 重新审视（分析）。

### 6.2 Patchouli 应用服务的签名

Patchouli 的公开路由已只接收 `IdentityScope`（第一批），但 `patchouli/application/` 中仍有 20 个方法的签名写作 `identity_scope: IdentityScope | None = None`，运行时由 `require_identity_scope` 拒绝缺省值。这是去掉 `access` 参数后留下的签名形状，行为上已是必填。

### 6.3 `ActorIdentity.session_id`

`ActorIdentity` 仍带兼容字段 `session_id`（`core/models/identity.py`），[外部会话 Idea](./external-session-and-topic-projection.md) 第 8 节第 3 项已记录。

- 写入方只有 server 的请求身份解析（[`server/deps.py`](../../src/hivememory/server/deps.py)）：chat 请求体中的 `session_id` 随 actor 声明进入认证；
- 读取方只有 Patchouli 门面的 finalize（[`patchouli/service.py`](../../src/hivememory/patchouli/service.py)）：把它写入交互提交的关联字段；
- `ActorIdentity` 是 frozen 的 Pydantic 模型，`session_id` 参与相等判断与 hash。写入意图的回读按整个 `IdentityScope` 相等判断可见性（[`agent_runtime/aliases/resolver.py`](../../src/hivememory/agent_runtime/aliases/resolver.py)），`session_id` 也参与这一比较；
- 感知把完整的 `ActorIdentity` 写入 `TurnRecord.identity`，随 Topic 保存，并作为 `InteractionTurnSnapshot.actor_identity` 写入交互 Artifact；`ActorIdentity` 没有禁止额外字段。

按 I-7，第二批移除该字段。

### 6.4 治理规则

AGENTS.md 第 3 节原先写有“`IdentityScope`（Actor + Workspace）必须沿应用服务、公共 route、Interaction 和后台任务传播，并在资源 owner 处再次校验”。这条规则把 `IdentityScope` 用于交互记录与后台任务，对应 6.1 的第三类。

2026-10-04 经 owner 同意，在第二批的实施分支 `refactor/identity-access-batch-2` 上改写为三条规则：身份数据按形态区分，`IdentityScope` 只由授权点组装、只在一次操作的调用链内到达资源 owner 与 Gateway；资源 owner 在公共边界拆为归属与发起者，记录与后台任务以独立字段携带，非主动生成路径的发起者是 `system`；资源授权先校验归属、再按 policy 判断发起者的可见性，来源字段不参与授权。改写后的规则是第二批代码须满足的约束，在第二批完成前与代码存在差异（6.1、6.5、6.7）。

### 6.5 Patchouli 与引擎内部对 `IdentityScope` 的使用

`patchouli/` 与 `engines/` 中，从 `IdentityScope` 读取 actor 的地方只有 6 处，分为两类：

| 类别 | actor 的角色 | 位置 |
|:---|:---|:---|
| 第 4 阶段的读取授权 | 操作的发起者，按资源 policy 判断可见性 | 中期库冷读的可见性判断（[`memory_library/adapters/mid_term.py`](../../src/hivememory/patchouli/memory_library/adapters/mid_term.py) 的 `_readable`）；检索的可见性过滤（[`engines/retrieval/filter_adapter.py`](../../src/hivememory/engines/retrieval/filter_adapter.py)） |
| 来源记录 | 写入资源的来源字段或交互的关联字段，是历史信息，不约束访问 | 生成的 `provenance_from_actor`（[`engines/generation/models.py`](../../src/hivememory/engines/generation/models.py)，只取 `agent_id`、`team_id` 写入 `MemoryProvenance`）；感知写入 `TurnRecord.identity`（[`engines/perception/memory_perception_engine.py`](../../src/hivememory/engines/perception/memory_perception_engine.py)）；`PreparedAgentRun.agent_id`；finalize 写入关联字段的 `session_id`（6.3） |

第一类中的中期库检索也被后台生成的查重调用，此时发起者取自任务携带的 `IdentityScope`（6.7）。

其余读取只使用 workspace 归属，例如：`MemoryAtom` 与 Artifact 的 `from_identity_scope` 只取 `WorkspaceIdentity`；记忆任务的观察检查只比对 workspace（[`patchouli/application/memory_task_management_service.py`](../../src/hivememory/patchouli/application/memory_task_management_service.py)）；`memory_library` 的端口、store 与 adapter、lifecycle 等内部方法接收 `IdentityScope`，除上述冷读授权外只使用其中的 workspace。

### 6.6 资源授权的现状

- 记忆的授权谓词 `memory_is_readable`（[`core/memory_access.py`](../../src/hivememory/core/memory_access.py)）按固定顺序执行两步：先 `memory_belongs_to_workspace`（归属），再 `access_policy_permits`（policy 对 actor 的可见性）。整个过程不读取来源 `MemoryProvenance`。
- 归属检查在任何路径上都不跳过：owner 视角的管理读取（`enforce_actor_visibility=False`）只跳过可见性；内部可信路径 `get_by_key` 读取后仍检查归属；检索在存储层按 `meta.workspace_id` 预过滤，命中后仍按归属与 policy 重新检查（[`memory_library/adapters/mid_term.py`](../../src/hivememory/patchouli/memory_library/adapters/mid_term.py) 注明“存储预过滤不是授权事实”）。
- 进程级共享设施依靠归属检查分区：WorkspaceAsset 的 token 索引由整个进程共用，读取时显式比对 workspace，跨 workspace 与未知 token 返回同一结果（[`workspace/assets/store.py`](../../src/hivememory/workspace/assets/store.py) 的 `_entry_for_read`）；`AtomCache` 以 `(workspace, memory_id)` 作键。
- 只有 MemoryAtom 带 policy：`MemoryAccessPolicy` 分 PUBLIC、PRIVATE（指定 agent）与 TEAM（指定 team），与 actor 的 `agent_id`、`team_id` 比较；`user_id` 不参与，由第 2、3 阶段的 owner 检查保证。Topic、Artifact、WorkspaceAsset 与记忆任务只按归属授权。
- 来源不进入 policy：生成新记忆时 policy 固定为 `MemoryAccessPolicy.public()`（[`engines/generation/engine.py`](../../src/hivememory/engines/generation/engine.py)），不由创建者推导。
- 唯一以创建者作为访问条件的地方是写入意图的回读，它按 `RuntimeScope` 中的整个 `IdentityScope` 相等判断可见性（6.3）。

### 6.7 非主动生成路径的发起者

PR #96（`Refactor/identity cleanup`，commit `37a5329`，对应[已归档的记忆溯源 Todo](../archive/todo/memory-provenance-vs-authorship.md)）把 SETTLE 的来源字段统一为 `system`：记忆的 `source_agent_id` 取 `SYSTEM_AGENT_ID`（`system_settlement_provenance`），参与内容的 agent 从 block 身份聚合进 `contributing_agent_ids`。该 Todo 的“结算来源的当前偏差”同时记录了结算沿用话题最后一次 touch 时冻结的 `IdentityScope`，这一部分没有修改：SETTLE 任务的发起者仍不是 `system`。

- 结算任务规范直接沿用 `TopicMaterializeTask` 携带的 `IdentityScope`（[`patchouli/control/memory_generation/coordinator.py`](../../src/hivememory/patchouli/control/memory_generation/coordinator.py) 的 `submit_settlement`）；`TopicData` 只保存归属，不保存执行者。
- 四种触发的发起者：

  | 触发 | 发起者 |
  |:---|:---|
  | 手动结算 | HTTP 管理路由经能力层授权得到的 `system` |
  | 空闲超时、关闭 | 话题最后一次被访问时冻结的 scope（[`patchouli/services/topic_working_set.py`](../../src/hivememory/patchouli/services/topic_working_set.py) 的 `list_idle_candidates`、`list_shutdown_candidates`） |
  | LRU 驱逐 | 触发驱逐的调用方的 scope，来自另一个话题的请求（[`patchouli/services/perception.py`](../../src/hivememory/patchouli/services/perception.py) 的 `_maybe_evict_lru`） |

- 结算路径上，这个发起者只被查重使用：生成在查重时检索中期库（[`engines/generation/engine.py`](../../src/hivememory/engines/generation/engine.py) 的 `_dedup_and_resolve`），按发起者的可见性过滤（`enforce_actor_visibility` 为默认值 `True`）；交互 Artifact 的来源取自 block 身份，记忆的来源字段已是 `system`。
- 后果：非主动路径生成的新记忆都是 PUBLIC，但查重可能命中最后访问的 agent 可见的 PRIVATE 或 TEAM 记忆。判为 UPDATE 时，`_apply_update` 就地修改已有记忆并保留其 policy，结算内容进入非 PUBLIC 的记忆；判为 TOUCH 时，结算内容不产生 PUBLIC 记忆。
- 以 `system` 为发起者时，查重只看得到 PUBLIC 记忆：policy 拒绝把 `system` 作为 PRIVATE 的 target（`MemoryAccessPolicy` 的校验），管理路由组装的 `system` actor 的 `team_id` 为 `None`。

## 7. 问题

### 7.1 已完成的问题

以下问题均已由 owner 决定。I-1–I-5、I-8–I-10 已随第一批（[A1 访问边界返工](../archive/plans/v0.7.0-a1-access-boundary-rework.md)，2026-10-04 归档）实施，其中 I-9 是过渡设计，随 Alice 的能力层调用迁移删除；I-6、I-6a、I-7、I-11 于 2026-10-04 决定，随第二批实施。

#### I-1 访问 context 的对外形态

**状态**：已完成。2026-10-03 决定，2026-10-04 修订机制；已实施。

**问题**：如果访问 context 带有公开字段（actor、驻留 workspace 等），任何拿到它的代码都能把它当作身份读取并往下传，不变量 2–4 只能靠约定与审查维持。

**设计**：访问 context 是密封凭据。

- 第 5 节的授予内容在签发时密封在 context 内；context 没有公开字段，`repr` 只显示 `<sealed>`；
- 只能由 `WorkspaceAuthenticator` 签发，直接构造被拒绝；context 拒绝复制、序列化与属性写入，撤销状态随凭据对象本身；
- 签发、读取授予内容与撤销都是凭据上的私有接口，由架构测试限定调用方所在的模块：签发与撤销只在认证一侧，读取只在操作授权者（授权）与认证一侧的诊断查询（只用于日志与观测标签）；
- 身份只在授权点取得：授权点把 context 交给操作授权者，由它返回所需的身份，例如第 3 阶段组装的 `IdentityScope`；Gateway、Patchouli、CPU 不持有操作授权者，即使拿到 context 也取不出身份；
- server 使用认证前自己持有的声明（actor 声明与请求进入的 workspace），认证成功即确认了这份声明，不从 context 读回。

这维护的是可信进程内的调用纪律，不隔离刻意读取私有属性的代码，与项目一贯的信任模型一致。

**演进**：2026-10-03 选择不透明凭据，当时的机制是“内容保存在签发方，授权点经签发方兑现”。2026-10-04 实施审查发现，这种引用式凭据迫使授权者依赖签发方，与“认证与授权是两个分开的行为”冲突（I-10），于是改为内容密封在凭据内，意图不变。与 commit `37f800e` 的区别：那时 context 持有公开的 `IdentityScope`，即第 3 阶段的产物；现在持有第 2 阶段的结果，不公开，读到之后仍须经第 3 阶段的检查才组装 `IdentityScope`。

#### I-2 来源（principal、adapter）是否随 context 保存

**状态**：已完成。2026-10-03 决定；已实施。

**问题**：A1 交付时 context 不持有来源 principal；P-1b（运行期间的请求须来自注册时的 principal）与拒绝记录需要知道 context 经由哪个来源签发。

**设计**：context 暂时保存 principal（`CallerPrincipal`），不参与 operation 授权。adapter 是否一并保存未涉及，需要时随 P-1b（总 Idea 第 16 节）一并决定。

#### I-3 运行绑定何时写入

**状态**：已完成。2026-10-03 决定；已实施。

**问题**：A1 交付时 chat 链路的认证由 server 路由完成，签发的 context 不绑定运行，进程要到流开始迭代时才创建；签发到绑定之间存在未绑定的 context，绑定进程的 context 与请求级 context 也无法区分。

**设计**：签发时写入运行绑定。

- **任务进程**：由注册入口完成两阶段认证，签发 context 的同时绑定本进程，并立即创建进程、登记到进程表（“先注册、后运行”）。server 只把自身的 principal、adapter，以及 actor 声明与请求进入的 workspace 交给注册入口；`process_id` 由 server 在进入注册入口前生成（任务进程 Idea Q-16），签发时就能绑定。注册失败直接抛出，HTTP 入口据此返回 403，不创建进程；注册成功但流一直没有开始时，由注册入口负责关闭进程（实现上由 SSE 响应的收尾兜底调用关闭路径）。
- **不建进程的请求**：请求级 context 由 server 按请求认证，绑定本次请求，请求结束即撤销（总 Idea 15.6）。

**依据**：这与此前已有的决定一致：总 Idea 前提第 2、3 条（经唯一注册入口两阶段认证，未通过不创建进程）、15.2（P-1a）、任务进程 Idea 1.2（两阶段认证后立即创建进程）与 Q-3a（访问 context 进入进程记录）。原 TaskProcess 容器 Todo 中“登记与注销由入口负责”一项随之并入第一批。

#### I-4 第 3 阶段如何指定目标 workspace

**状态**：已完成。2026-10-03 决定；已实施。

**问题**：若授权点默认取驻留 workspace 作为目标，将来实现受限穿透访问时要改所有授权点的签名，而且“要操作哪个 workspace”（调用方的意图）与“被准入到哪个 workspace”（凭据）会继续混在一起。

**设计**：授权点接口显式接收目标 workspace；当前只接受等于驻留 workspace 的目标，其余一律拒绝（`target_workspace_not_resident`）。实现受限穿透访问时只需放宽这条规则。

#### I-5 `_require_same_owner` 的去向

**状态**：已完成。2026-10-03 决定；已实施。

**问题**：A1 把 W0 的准入规则“actor 用户等于 workspace owner”写进了 `IdentityScope` 的校验器，身份类型承担了授权规则，与不变量 6 冲突。

**设计**：owner 校验移到第 2、3 阶段：准入时检查 actor 用户与要进入的 workspace 的 owner（`actor_not_owner`），操作授权时检查 actor 用户与目标 workspace 的 owner（`target_owner_mismatch`）；`IdentityScope` 不再带 `_require_same_owner` 校验器。

#### I-6 资源身份的表达

**状态**：已完成。2026-10-04 决定，同日两次修订；随第二批实施，尚未实施。发起者的形态见 I-6a，非主动生成路径的发起者见 I-11。

**问题**：6.1 第三类的记录与后台任务保存的是一次操作的 `IdentityScope`，而它们的寿命长于那次操作。问题的核心是 `IdentityScope` 的滥用，而不是它与访问 context 的混淆：`IdentityScope` 携带操作的发起者与目标 workspace，Patchouli 内部用不到这么多（6.5）。

**设计**：

- **没有“资源身份”的概念，只有资源归属**（`WorkspaceIdentity`）。资源授权（第 4 阶段）先比对归属，再按资源 policy 判断发起者的可见性；policy 是资源的授权数据，不是身份（第 2、3 节）。
- **发起者与归属分开携带**：归属回答资源（以及记录、任务）属于哪个 workspace，发起者回答一次操作（包括延后执行的后台任务）由谁发起。两者性质不同，不组合成 `IdentityScope` 携带。
- **Patchouli 全系统重构**：Patchouli 的公开路由仍只接收 `IdentityScope`（第一批）；进入 Patchouli 之后拆为归属与发起者。内部的应用服务、控制面、服务、记忆库端口与存储，以及它驱动的 engines，不再传递或组装 `IdentityScope`：只用到归属的方法只接收 `WorkspaceIdentity`，需要可见性判断的读取另外接收发起者（不变量 4）。
- **记录与后台任务**不再保存 `IdentityScope`，以独立字段携带归属与（需要时）发起者。`MemoryGenerationTask` 拆出 `from_actor: ActorIdentity` 与 `belong_to: WorkspaceIdentity`（owner 指定）；其余类按同样方式拆分，字段名与是否需要发起者在计划中逐个确定。
- **资源上的来源字段**（`MemoryProvenance` 与贡献者集合）是历史信息，不约束访问（不变量 7），与后台任务的发起者是两回事（第 2 节）。

**依据**（owner，2026-10-04）：“资源属于哪个 workspace、由谁产生”本身没错，但“由谁产生”对资源访问不构成约束，与 `WorkspaceIdentity` 组合在一起并没有对等的作用。所谓资源身份只用于资源授权，真要构造一个，应当是 `WorkspaceIdentity` 加资源 policy；而 policy 显然不是身份，所以只有资源归属的概念。资源授权的过程也说明这一点：先看资源归属是否等于 `IdentityScope` 的目标 workspace，再看 policy 是否允许发起者。

- owner 指出归属这一步一般不用检查，因为另一个 workspace 的资源不会出现在当前 workspace。按代码核对（6.6），这一结果正是由资源 owner 处的归属检查保证的：检查常以分区键和查询过滤的形式完成，但不能省略（第 3 节第 4 阶段，分析）。
- 同日补充：Patchouli 内部不需要组装 `IdentityScope` 才能携带发起者与资源归属的信息，两者性质不同；既然收紧已不能局限于顶层（记录、后台任务与公开签名），本批直接对 Patchouli 全系统重构。

**取舍**：

- 定义专门的资源身份类型（原选项 B）不采用：把归属与来源打包成一个类型，本质上与使用 `IdentityScope` 没有变化；
- 归属加 policy 也不构成身份：policy 是授权数据；
- 原选项 A（明确的归属与来源两个字段）的字段形态保留，含义改变：两个字段是归属与发起者，不合称资源身份；
- 在 Patchouli 内部以发起者与归属重新组装 `IdentityScope`：不采用，内部不需要它；
- 第二批先只清理记录与后台任务、Patchouli 内部的收紧在实施一轮之后再评估：同日的先前安排，被 Patchouli 全系统重构取代。

**影响**（分析）：

- 改动面覆盖 6.1 第三类、6.5 列出的 Patchouli 内部使用点及其在 engines 中的对应部分；6.2 的应用服务签名一并收紧。
- 记录只按归属比对，与几处现有行为一致：记忆任务的观察检查只比对 workspace，`MemoryAtom` 与 Artifact 的归属只取 workspace（6.5），WorkspaceAsset 只按归属授权（6.6）。
- 写入意图的回读目前按 `RuntimeScope` 中的整个 `IdentityScope` 相等判断可见性（6.3、6.6），与本决定不一致；它属于[写入意图迁移 Idea](./pending-intent-migration.md#01-owner-的决定2026-09-28) 的范围，该 Idea 0.1 已决定写入意图在落库前对全 workspace 可回读。
- 后台生成的查重以任务的发起者为可见性主体；非主动生成路径的发起者是 `system`，查重因此只在 PUBLIC 内进行（I-11）。

**演进**：2026-10-04 先决定资源身份只剩归属、来源降为辅助信息；同日 owner 指出这样的“资源身份”只是归属，于是取消资源身份的概念，第 2–4 节改用“资源归属”；之后 owner 又指出发起者与归属性质不同，Patchouli 内部不需要组装 `IdentityScope`，范围扩大为 Patchouli 全系统的重构，并给出 `MemoryGenerationTask` 的拆分方式。

#### I-6a 发起者的形态

**状态**：已完成。2026-10-04 决定；随第二批实施，尚未实施。

**问题**：记录与后台任务单独携带 actor 时，携带完整的 `ActorIdentity`，还是只携带消费方用到的字段？现有消费方（6.5）：生成只取 `agent_id`、`team_id` 写入 `MemoryProvenance`；查重的可见性判断比较 `agent_id` 与 `team_id`；感知把完整的 `ActorIdentity` 写入 `TurnRecord`，并随交互 Artifact 保存。

**设计**：发起者以完整的 `ActorIdentity` 单独成字段，例如 `MemoryGenerationTask.from_actor`（owner 指定）。I-7 之后 `ActorIdentity` 只含 `user_id`、`agent_id`、`team_id`。

**演进**：原先以“来源辅助信息的形态”列为未完成问题，选项为完整的 `ActorIdentity` 与只携带所用字段；owner 指定 `from_actor: ActorIdentity` 后按前者处理，字段的含义也从“来源”改为“发起者”（I-6）。

#### I-7 `ActorIdentity.session_id`

**状态**：已完成。2026-10-04 决定；随第二批实施，尚未实施。

**问题**：兼容字段 `session_id` 仍在 `ActorIdentity` 中（6.3），会参与身份的相等性与 hash；会话的承载已决定由 ConversationSession 负责（任务进程 Idea Q-9）。原选项：去掉兼容字段；保留但不参与相等性与缓存键；随 ConversationSession 方向一并处理。

**设计**：在第二批的计划中从 `ActorIdentity` 移除 `session_id`，身份只表达 actor；不等待外部会话方向。这承接[外部会话 Idea](./external-session-and-topic-projection.md) 第 8 节第 3 项中“移除 identity.session_id 对 equality/hash/cache key 的影响”。

**影响**（分析，代码依据见 6.3）：

- 请求体中的 `session_id` 不再进入身份；交互提交的关联字段是否保留 `session_id`、经由什么途径传递，在计划中确定；chat 请求体中 `session_id` 字段本身的语义仍归外部会话方向；
- 写入意图的回读比较不再受 `session_id` 影响；
- 已保存的 Topic 与交互 Artifact 中可能带有 `session_id` 键，移除字段后需要确认读取旧数据时该键被忽略。

#### I-8 进程记录如何持有身份

**状态**：已完成。2026-10-03 决定，同日与 2026-10-04 两次补充；已实施。

**问题**：A1 交付时同一个 context 有三处持有者，进程记录还同时保存调用方传入的 `IdentityScope`；进程记录里的身份有三个读取点（取消与状态查询时比对请求方、阶段调用的目标 workspace、运行时事件的标签）。如果记录只持有 context，而三个读取点都经授权者即时解析身份，事件发布器与进程表也要持有授权者；如果创建时把身份另抄一份写入记录，就有两份身份来源。

**设计**：

- **进程记录**只持有访问 context 与进程自身的元数据（`process_id`、阶段、终态、停止原因、事件通道），不保存 actor、驻留 workspace 等身份字段。context 只由进程记录持有：注册请求只携带声明，不携带 context；注册入口认证成功后直接把 context 写入记录。
- **三个读取点按各自性质处理**：
  - 取消与状态查询是一次授权判断（P-7）：由取消入口把请求方的 context 与进程记录中的 context 交给操作授权者比对；不匹配时与进程不存在一样返回 `not_found`，不泄露进程是否存在；
  - 阶段调用的目标是任务注册时声明并通过认证的 workspace，作为任务参数由进程传给授权点（I-4），不从凭据读回驻留 workspace；
  - 事件标签是观测标签，不是身份（AGENTS.md：`workspace_id` 观测标签不等于授权或分区），进程创建时用通过认证的注册声明绑定一次，此后不再改变。
- **进程句柄**：注册入口交给入口 adapter 的是不透明的进程句柄，只暴露 `process_id`；入口 adapter 不接触进程记录、进程容器与其中的 context。句柄按对象身份判定有效：它只由注册入口签发，私下记着对应的进程对象，注册入口解析时要求进程表中登记的正是这一个；按 `process_id` 重新构造的对象、进程关闭后的旧句柄都不是有效句柄。句柄不离开进程，不提供序列化。
- **唯一的取消方法**，取消的依据作为参数：
  - 传入句柄：调用方是进程生命周期的所有者（例如客户端断开时的 chat 路由），不经进程控制授权；句柄已失效时只返回不存在，不发布事件；
  - 传入 `process_id` 与请求级 context：控制请求（`/chat/stop`），经进程控制授权；找不到或无权控制时返回不存在，并发布带请求方观测标签的事件。

  只有 `process_id` 而没有 context 不能取消；stop 记录、终态判定与运行时事件只有一份实现。
- **客户端断开是一次取消，不并入关闭**：断开时先同步取消（记录断开原因，使进程以已取消的终态结束），再取消并等待正在拉取事件的任务，最后关闭进程；若把取消并入关闭，断开原因与对应的终态事件将不再记录。

**演进**：

- 2026-10-03 第一批实现审查：注册入口把整个进程容器交给入口 adapter，server 由此读取进程记录中的 context，以进程自己的 context 作为取消的请求方，进程控制授权变成自己与自己比对。由此引入不透明的进程句柄。
- 2026-10-04：句柄最初是只含 `process_id` 的值对象，而 `process_id` 会经 SSE 事件发给客户端、随 `/chat/stop` 传回，任何代码都能现造一个句柄绕过进程控制授权。由此改为按对象身份判定有效；同时把原先作用相同、只在依据上不同的停止与取消两个方法合并为一个。

进程表登记的对象见[任务进程 Idea](./task-process-table-and-registration-entry.md) 1.2。进程容器的职责（状态容器与所有进程共用的执行器分开）见[已归档的 TaskProcess 容器 Todo](../archive/todo/task-process-container-ownership.md)。

#### I-9 CPU 在过渡期的身份

**状态**：已完成。2026-10-03 决定；过渡设计已实施，随 Alice 的能力层调用迁移删除。

**问题**：Alice 改经能力层调用之前（总 Idea 15.5），CPU 输入清单要携带一个 `IdentityScope` 供 Alice 直接调用 Patchouli。按不变量 4，`IdentityScope` 只由授权点组装，但 CPU 执行本身没有对应的 operation；借用某次不相关的 operation 授权结果，或直接用注册声明组装，都会破坏这条不变量。

**设计**：操作授权者提供过渡专用的 `cpu_execution_identity`：只做第 3 阶段的目标 workspace 与 owner 检查，不检查 operation。只有任务进程的 CPU 分配调用它（架构测试限定调用面）。Alice 的直接调用因此仍没有 operation 授权，与过渡前相同；Alice 的能力层调用迁移完成后删除该方法。

#### I-10 workspace 一侧的认证与操作授权如何划分

**状态**：已完成。2026-10-03 决定，2026-10-04 补充；已实施。

**问题**：A1 让认证网关只编排两项认证，签发跟踪、生命周期与行为授权都由一个 guard 负责（理由是两类配置的所有者不同，位于 System 的网关不应持有 Workspace 的签发状态）。总 Idea D-6 把网关移入 `workspace` 之后，这个理由消失，划分却保留下来：网关靠跨类调用私有方法完成第 2 阶段，guard 同时负责签发与授权，失效接口在网关与 guard 两处都有，注册入口同时依赖网关与 guard。

**设计**：第 2 阶段与第 3 阶段各有一个负责者，认证与授权是两个分开的行为：

| 类 | 负责 | 状态 | 调用方 |
|:---|:---|:---|:---|
| `ActorAuthenticationGateway` | 唯一对外的认证入口：依次调用两个认证者，完成第 1、2 阶段；为运行持有者提供 context 的撤销与诊断查询；为 System 提供关闭（只拒绝新的认证，已签发的 context 照常可用直到被撤销） | 网关自身的关闭状态 | 运行持有者：server（请求级 context）、注册入口（进程 context）；System |
| `WorkspaceAuthenticator` | 第 2 阶段：检查 actor 用户等于要进入的 workspace 的 owner（I-5），准入记录存在且启用；签发 context；单个撤销，System 停止时撤销全部；诊断查询 | 已签发 context 的弱引用集合 | 只有认证网关 |
| `WorkspaceOperationAuthorizer` | 第 3 阶段：读取凭据内容并按访问登记检查目标 workspace（I-4）、目标的 owner（I-5）与白名单，组装 `IdentityScope`；进程控制授权（P-7）；CPU 执行身份的过渡方法（I-9） | 无状态，只依赖访问登记 | 授权点：能力层、任务进程的执行器与 CPU 分配、注册入口（进程控制） |

- 命名与 Principal 一侧对应：第 1 阶段由 `PrincipalAuthenticator` 端口完成（System 的 `SystemPrincipalAuthenticator` 实现），第 2 阶段由 `WorkspaceAuthenticator` 完成，认证网关编排两者。
- 认证一侧与操作授权者互不依赖，只经凭据类型发生联系（I-1）；架构测试守护两者互不导入。
- 各调用方的依赖：server 只依赖认证网关；能力层、任务进程的执行器 `TaskProcessRunner` 与 CPU 分配只依赖操作授权者；注册入口同时依赖两者，因为它既是进程 context 的运行持有者，又是授权点（进程控制）。
- 拒绝语义见[错误模型](../contracts/error-model.md)第 4.4 节。

**取舍**：保留 A1 的划分、只消除重复接口并把私有调用改为包内正式接口，改动最小，但 guard 仍同时负责签发与授权；合并为一个访问服务、对外提供认证与授权两个角色协议，只需装配一个对象，但能授权的对象同时也能签发，边界只靠类型约束。

**演进**：2026-10-03 决定按阶段拆成两个类，当时授予记录由 `WorkspaceAuthenticator` 持有，操作授权者经它的只读兑现接口读取。2026-10-04 实施审查发现，这样操作授权者在结构上仍依赖一个也能签发 context 的对象；只读接口只能收窄依赖，去不掉依赖。根源在 I-1 当时的引用式凭据，于是改为授予内容密封在 context 内，`WorkspaceAuthenticator` 不再提供兑现接口，也不再有自身的关闭状态。

#### I-11 非主动生成路径的发起者

**状态**：已完成。2026-10-04 决定；随第二批实施，尚未实施。补完 PR #96 未改完的部分（6.7）。

**问题**：SETTLE（手动、空闲超时、LRU、关闭四种触发）的记忆来源字段已在 PR #96 统一为 `system`，但结算任务的发起者仍沿用话题最后一次访问或触发驱逐的 actor；查重按这个 actor 的可见性进行，可能把结算内容并入非 PUBLIC 的记忆（6.7）。

**设计**（owner）：

- 所有非主动写入的路径都以 `system` 作为发起者结算：这些路径中本来就没有任何 agent 的主动意图；参与过工作的 agent 只记录在 `contributing_agent_ids` 中。
- 当前实现中，非主动路径得到的记忆都以 PUBLIC 开放，查重同样只在 PUBLIC 内进行。以 `system` 为发起者即满足这一点：policy 拒绝把 `system` 作为 PRIVATE 的 target，`system` 的 `team_id` 为 `None`（6.7，分析）。
- 发起者作为后台任务的字段携带（I-6 的 `from_actor`），不在 Patchouli 内部组装 `IdentityScope`。

主动的 WRITE、UPDATE 仍以提交写入意图的 actor 为发起者（现状）。管理写入由用户经直接通道显式指定 policy，不属于非主动生成路径（分析）。

**演进**：2026-10-04 初稿把它列为未完成问题“后台任务中后续读取的视角”，并把后台任务携带发起者误判为“来源重新成为访问条件”；owner 指出资源的来源字段与后台操作的发起者是两回事，问题在于非主动路径的发起者应当是 `system`。

### 7.2 未完成的问题

暂无。第二批的前置决定均已作出（2026-10-04）。

## 8. 分批

owner 于 2026-10-03 决定：建立独立 Idea；`IdentityScope` 名称保留，逐步分批修正项目中的使用点。2026-10-04 决定：收紧不再局限于顶层，第二批直接重构 Patchouli 全系统（I-6），同时补完非主动生成路径的发起者（I-11）并移除 `ActorIdentity.session_id`（I-7）。

| 批次 | 范围 | 状态 |
|:---|:---|:---|
| 第一批 | workspace 边界：入口在认证前只持有声明；访问 context 为密封凭据并暂存 principal（I-1、I-2）；注册入口完成认证、签发即绑定、先注册后运行（I-3）；授权点显式接收目标 workspace（I-4）；owner 校验移到第 2、3 阶段（I-5）；进程记录只持有 context、进程句柄与唯一的取消方法（I-8）；CPU 过渡身份（I-9）；认证一侧与操作授权者分开且互不依赖（I-10）；资源 owner 与 Gateway 只接收 `IdentityScope` | 已完成：随 [A1 访问边界返工](../archive/plans/v0.7.0-a1-access-boundary-rework.md)实施，2026-10-04 归档；当前事实见 [Workspace 架构](../architecture/workspace.md)第 4 节 |
| 第二批 | Patchouli 全系统重构：公开路由仍接收 `IdentityScope`，内部拆为归属与发起者，不再传递或组装 `IdentityScope`（6.5，I-6）；记录与后台任务不再保存 `IdentityScope`，以独立字段携带归属与发起者，如 `MemoryGenerationTask` 的 `from_actor` 与 `belong_to`（6.1 第三类，I-6、I-6a）；非主动生成路径以 `system` 为发起者（I-11）；从 `ActorIdentity` 移除 `session_id`（I-7）；收紧 Patchouli 应用服务的签名（6.2）；修订 AGENTS.md 第 3 节的相应规则（6.4） | 计划已建立：[v0.7.0 身份与访问体系第二批](../plans/v0.7.0-identity-access-batch-2.md)（2026-10-04），尚未实施；AGENTS.md 已在实施分支上修订 |
| 不在 v0.7.0 | 资源的受限穿透访问（跨 workspace 的授权模型） | 模型在第 3 阶段的目标 T 处预留 |

第一批使总 Idea 中两项早先的决定失去前提：“放行分支分两步去掉”与“两个提交路由的检查暂留在 Patchouli”。资源 owner 不再接收访问 context 后，这两项按最终设计记录在总 Idea 15.8。

## 9. 与其他文档的关系

| 文档 | 关系 |
|:---|:---|
| [总 Idea](./workspace-network-task-process-architecture.md)第三部分 | 认证与授权的流程与待决问题（P-1、P-4a、P-9 等）在那里；本文界定其中流动的身份数据 |
| [A1 访问边界返工](../archive/plans/v0.7.0-a1-access-boundary-rework.md)（已归档） | 第一批的实施计划 |
| [v0.7.0 身份与访问体系第二批](../plans/v0.7.0-identity-access-batch-2.md) | 第二批的实施计划 |
| [任务进程 Idea](./task-process-table-and-registration-entry.md) | Q-3a 决定访问 context 进入进程记录，与不变量 3 一致；进程记录如何持有身份见 I-8；认证与进程创建的顺序见 I-3 |
| [外部 Actor Idea](./external-actor-registration-and-runtime-access.md) | principal 与 adapter 的登记（I-2）；plugin 模式下不建进程的访问同样遵循本文的边界 |
| [外部会话 Idea](./external-session-and-topic-projection.md) | `ActorIdentity.session_id` 随第二批移除（I-7），承接该文第 8 节第 3 项；chat 请求体中 `session_id` 的语义仍归该文 |
| [写入意图迁移 Idea](./pending-intent-migration.md) | `PendingAtomMaterializeTask` 属于第二批的范围（6.1）；写入意图回读的可见范围由该文决定，落库前对全 workspace 可回读，与 I-6 一致 |
| [已归档的记忆溯源 Todo](../archive/todo/memory-provenance-vs-authorship.md) | PR #96 把 SETTLE 的来源字段统一为 `system`；结算任务的发起者未随之修改，由 I-11 补完 |
| [Workspace 架构](../architecture/workspace.md)第 4 节 | 第一批的当前事实；第二批完成后按最终代码更新 |

## 10. 形成计划的条件

- 满足 [Ideas 升级规则](./README.md#升级规则)与[文档治理规范](../DOCUMENTATION.md)第 8.3 节；
- 第一批已完成；
- 第二批：I-6、I-6a、I-7、I-11 已决定（2026-10-04）；AGENTS.md 第 3 节已经 owner 同意，在实施分支上修订（6.4）；计划已建立（[v0.7.0 第二批](../plans/v0.7.0-identity-access-batch-2.md)）。
