---
title: 身份与访问体系
status: idea
horizon: candidate
owner: project
scope: identity-and-access-mainline-from-principal-proof-to-workspace-admission-operation-authorization-and-resource-visibility
code_paths:
  - src/hivememory/core/models/identity.py
  - src/hivememory/core/access.py
  - src/hivememory/core/memory_access.py
  - src/hivememory/system/access/
  - src/hivememory/config/access.py
  - src/hivememory/workspace/authentication.py
  - src/hivememory/workspace/authorization.py
  - src/hivememory/workspace/registry.py
  - src/hivememory/server/deps.py
  - src/hivememory/server/routers/
related_docs:
  - docs/workspace/identity-and-authorization.md
  - docs/workspace/authentication.md
  - docs/workspace/capability.md
  - docs/governance/security/identity-and-execution-safety.md
  - docs/ideas/fifth-generation-architecture.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/ideas/pending-intent-migration.md
  - docs/ideas/execution-environment.md
  - docs/ideas/ae2-hivememory-architecture-analogy.md
  - docs/archive/ideas/execution-thread-definition.md
  - docs/todo/local-http-management-access.md
  - docs/todo/workspace-asset-ownership-identity-split.md
  - docs/todo/memory-visibility-policy-ui.md
  - docs/todo/frontend-identity-ownership.md
  - docs/todo/run-target-executability-check.md
last_reviewed: 2026-11-05
---

# 身份与访问体系

## 0. 文档性质

- **主线定位**（owner，2026-11-05）：身份与访问体系作为单独的主线方向维护。它是 Workspace 体系的基础，不只服务 controller 模式，覆盖从 principal 的登记与调用方证明，到进入 Workspace 的身份、操作调用的授权，再到记忆等资源的可见性。这条方向最好在发布前为系统打好地基，否则 controller 与 plugin 两种接入都难以继续。
- 本文收集这条线上尚未决定或尚未实施的部分，不描述当前行为，也不代表排期。理念见[第五代架构 Idea](./fifth-generation-architecture.md)第 3.5、3.6 节；当前的身份模型见[身份与授权](../workspace/identity-and-authorization.md)与[认证与访问登记](../workspace/authentication.md)，三个调用面见[能力层](../workspace/capability.md)。
- **2026-11-05 重写**：原文只覆盖跨 Workspace 方向。散落在[第五代架构 Idea](./fifth-generation-architecture.md)第 5 节（P-1、P-4a、P-5、P-7、P-8）、[外部 Actor Idea](./external-actor-registration-and-runtime-access.md)（E-1）与[写入意图后续方向 Idea](./pending-intent-migration.md)第 5 节的身份问题移入本文，沿用原编号，原位置改为链接；本文不新增问题编号。重写前的版本见 commit `3cfdec9d` 中的本文；2026-10-13 的精简删除了已实施的 I-1–I-11，删除前的版本见 commit `a384688`。
- 问题只列选项及其影响，选项顺序不代表倾向；标为“分析”的内容不是 owner 决定。

## 1. 为什么单独成线

身份与访问自 v0.7.0 起经过多轮返工与调整，每一轮都挂在当时的主线方向下进行：[A1 访问边界返工](../archive/plans/v0.7.0-a1-access-boundary-rework.md)、[身份与访问体系第二批](../archive/plans/v0.7.0-identity-access-batch-2.md)、[能力层分层](../archive/plans/v0.7.0-capability-surface-split.md)、[执行线程层](../archive/plans/v0.7.0-execution-thread-layer.md)、[记忆访问](../archive/plans/v0.7.1-memory-access-mcp.md)中 P-1c 与 P-9c 的复核，以及 [plugin 模式开放写入与更新](../archive/plans/v0.7.2-plugin-write-update.md)时的请求级授权（均为归档计划）。结果是：

- 已落地的模型分布在多份事实文档中：身份与授权、认证与访问登记、能力层、执行线程与 MCP 记忆访问；
- 未决的部分散落在第五代架构 Idea、外部 Actor Idea、写入意图 Idea、五份 Todo 与身份治理中；
- 例外与补丁逐渐累积：WorkspaceAsset 仍以旧 `IdentityScope` 做归属检查（AGENTS.md 登记的唯一例外），Profile 注入脚本以 `system` 进入 root API，HTTP 管理接口沿用本地单用户部署的信任假设。

已经具备、本文不再讨论的地基：三个身份模型与两阶段认证、两阶段授权；密封的访问 context 及进程、线程、请求三种运行绑定；资源归属与发起者的拆分；root API、actor API 与阶段调用三个调用面；外部 principal 以本地共享密钥证明身份（P-1c）；`allow_system_actor` 限定能声明 `system` 的来源；不建进程的访问每次操作一份请求级 context（P-9c）。

**对发布的影响**（分析）：owner 定下的 v1.0 范围包括桌面客户端，以及 Claude Code、dsh、Codex 与 pi 四个 harness 的 plugin 模式接入（owner，2026-11-04）。桌面端常驻后端之后，本机进程、各 harness 中的 agent 与前端都是调用方，本地单用户的信任假设与只能静态修改的登记都会被直接触及。

## 2. 全链路与缺口

| 环节 | 已有（当前事实） | 缺口与去向 |
|:---|:---|:---|
| Principal 登记 | 两份配置文件在启动时装载（[认证与访问登记](../workspace/authentication.md#2-两类访问登记)第 2 节） | 运行时登记、修改与撤销：第 3 节 |
| 调用方证明 | 统一认证网关；外部 principal 以共享密钥证明；任务进程注册、Import Bus 与 MCP 入口经网关 | HTTP 管理入口不证明调用方；controller 模式下请求与进程的归属校验：第 4 节 |
| 访问 context 与操作授权 | context 只含 actor、驻留 Workspace、principal 与运行绑定；白名单逐次授权；阶段授权；root 只认 `system` | context 是否带 adapter、本次运行的操作上限、进程控制的授权主体：第 5 节 |
| 资源可见性 | 只有 MemoryAtom 带资源 policy，且只表达读取；其余资源只按归属授权 | 修改与执行授权、写入意图的可见性：第 6 节 |
| 进入其他 Workspace | 第 3 阶段只接受目标等于驻留 Workspace | 受限穿透访问：第 7 节 |
| 例外、暂缓与已知限制 | — | 第 8 节 |

## 3. Principal 登记的来源与生效时点（E-1）

2026-11-05 由[外部 Actor Idea](./external-actor-registration-and-runtime-access.md) 移入。

**背景**：System 接入登记（`configs/system_principals.yaml`）与 Workspace Actor 访问登记（`configs/workspace_actors.yaml`）只在启动时装载，运行中不可变；除配置文件外，没有登记、修改或撤销接入的入口（[认证与访问登记](../workspace/authentication.md#8-已知限制)第 8 节）。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 维持启动时从配置装载 | 新增或撤销外部 Actor 需要修改配置并重启 |
| B | 提供运行时登记入口（例如经管理员直接通道） | 需要定义登记、修改、撤销的授权与生效时点，以及已签发 context 的处理 |
| C | 登记数据来自外部身份系统 | 需要定义与外部系统的信任关系与同步方式 |
| D | 其他 | —— |

**owner 决定**（2026-09-27）：接入登记维持启动时从配置装载（选项 A）。运行时登记与其他配置文件的热更新一并由未来单独的计划实现。

- **补充**（owner，2026-09-30）：harness 登记的执行侧面同样从配置装载。登记使用单独的配置文件（owner，2026-10-01），2026-10-02 起 `principals` 与 `workspace_actors` 各用一个文件。
- **并入**（2026-10-10）：Workspace 访问登记中 Agent 操作白名单的运行时编辑（含前端开关）并入同一份未来计划（[ROADMAP](../ROADMAP.md) 第 5.5 节“配置热更新”）。届时需要处理线程描述中的操作快照与即时检查之间的差异。

**E-1a**：System 接入登记与 Workspace Actor 访问登记是否经同一入口维护：同一入口 / 分开维护 / 其他。owner 决定随运行时登记的计划一并决定。

**分析**：按 v1.0 范围，Codex 与 pi 接入时各需登记一个 principal 与一份访问记录；维持选项 A 时，每接入一个 harness 都要修改两份配置并重启后端。

## 4. 调用方证明

### P-1 经网络接入的执行者如何证明身份

2026-11-05 由[第五代架构 Idea](./fifth-generation-architecture.md) 移入。外部 principal 以接入凭据证明身份；HTTP 入口仍以 server 登记的 principal 接入，请求头中的用户、Workspace 与 Agent 只是声明（[认证与访问登记](../workspace/authentication.md#8-已知限制)第 8 节）。

- **P-1b 请求与进程的归属校验**：请求的 principal 必须与进程注册时一致 / 其他规则。只涉及 controller 模式，即外部 harness 经绑定执行线程的操作适配器回调时（[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#controller-模式的步骤分析退回候选方向) 的后续步骤）。
- **P-1c 身份证明的实现位置**：owner 决定（2026-10-29）由统一认证网关证明，每个外部 principal 一份本地共享密钥作为接入证据，一个 harness 对应一个 principal；已实施，决定与复核记录在[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#p-1c-与-p-9c-身份证明与不建进程的访问-context)。

### HTTP 管理入口的调用方证明

问题、证据与影响见[本机 HTTP 管理接口 Todo](../todo/local-http-management-access.md)：前端使用的 Workspace 路由以 server 的 principal 经网关接入，`config`、`providers`、`models`、`logs`、`runtime_events` 不经网关，本机任何进程（包括 harness 中的 agent）都能调用，绕过 Workspace 访问登记的白名单，以 `system` 声明进入 root API。owner（2026-11-01）决定下一步修复；它会改变整个 HTTP API 的信任模型，Todo 要求实施前先在 Idea 或计划中定下认证方式，本节承接这一设计问题。

| 选项 | 内容 | 影响（分析） |
|:---|:---|:---|
| A | HTTP 管理接口增加本机管理凭据，前端随页面取得，外部进程拿不到 | 管理路由与前端使用的 Workspace 路由都要核对凭据；页面由同源后端提供时可以注入，Vite 开发模式与桌面壳需要各自的取得方式（桌面壳可由主进程下发） |
| B | 管理接口单独监听另一个端口或 Unix socket，与 `/mcp`、`/api/v1/import/turns` 分开 | 端口分离本身不阻止同机进程连接，需要配合 socket 的文件权限或凭据；WSL2 会把 Windows 上的本机地址转发到 WSL 中的服务 |
| C | 收紧为只读，或移除不经认证网关的写接口 | 可写面变小，但 Settings、Provider 与模型管理依赖这些写接口，需要另找路径 |
| D | 其他 | —— |

### P-5 被动请求的认证

2026-11-05 由[第五代架构 Idea](./fifth-generation-architecture.md) 移入。被动请求只存在于 controller 模式（[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#11-两种接入模式owner2026-09-27) 1.1）。

- **P-5b 触发器的 principal**：触发器的登记者 / 系统内置 principal / 其他。owner 倾向（2026-09-27）由登记的 Agent 反推 principal；
- **P-5c 触发器的准入检查时点**：登记触发器时 / 每次触发时 / 两者都查。

## 5. 访问 context 与操作授权

访问 context 的授予内容只包含 actor、驻留 Workspace、principal 与运行绑定（[认证与访问登记](../workspace/authentication.md#3-访问-context密封的运行时凭据)第 3 节）。以下方向可能改变这一点，各自随相应问题决定：

- **adapter**：context 暂存签发时的 principal（I-2），adapter 是否一并保存未决，随 P-1b 一并决定；
- **本次运行的操作上限**：只有 P-4a 采用“进程持有白名单的子集”时，context 才需要携带它。

context 没有固定有效期，随绑定的运行结束失效。原先列为开放方向的“plugin 模式需要长连接时是否加有效期”已不再成立：plugin 模式的 MCP 访问每次工具调用签发一份请求级 context，调用结束即撤销（P-9c，[MCP 记忆访问](../contracts/mcp.md)），没有长连接持有 context。

### P-4a 进程能否持有白名单的子集

2026-11-05 由[第五代架构 Idea](./fifth-generation-architecture.md) 移入。

- 支持：注册时声明所需操作集合，入口校验它是白名单子集，能力层按进程的集合授权；访问 context 需要携带本次运行的操作上限；
- 不支持：进程一律继承 actor 在该 Workspace 的白名单。

### P-7 进程控制操作的授权主体

2026-11-05 由[第五代架构 Idea](./fifth-generation-architecture.md) 移入。取消由持有进程句柄的入口与进程控制授权处理；状态查询等其余控制操作的授权主体尚未决定。

选项：仅进程的注册 principal / 注册 principal 与管理员 / Workspace 内任何获准者 / 其他。

## 6. 资源可见性与资源授权

现在只有 MemoryAtom 带资源 policy（PUBLIC、PRIVATE、TEAM，只表达读取）；Topic、Artifact、WorkspaceAsset 与记忆任务只按归属授权（[身份与授权](../workspace/identity-and-authorization.md#5-资源归属与寻址)第 5 节）。

约束（2026-10-04，由“资源上的来源字段不参与授权”得出）：与创建者相关的权限，例如“只有创建者能修改”，应在资源创建时写入资源 policy，授权时不读取来源；否则来源会重新成为访问条件。

### P-8 资源边界是否增加修改与执行授权

2026-11-05 由[第五代架构 Idea](./fifth-generation-architecture.md) 移入。`MemoryAccessPolicy` 只表达读取。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 资源策略扩展到修改与执行 | 需要扩展 `MemoryAccessPolicy` 或新增策略模型，并确定默认值与存量迁移 |
| B | 不增加 | 修改权限只由操作白名单与读取可见性共同约束 |
| C | 只对部分资源类型增加 | 需要划分资源类型 |

### 写入意图的可见性与 policy

2026-11-05 由[写入意图后续方向 Idea](./pending-intent-migration.md) 第 5 节移入。

**问题**：意图不设 policy，同 Workspace 内持有别名即可回读（UPDATE 意图跟随基础原子的可读性）。别名由标题或内容片段加 4 位十六进制后缀组成，“只有写入者知道别名”是惯例，不是访问控制，真正的边界是 Workspace。`WriteFocus` 只有 content、reason、title，不携带目标 policy。

需要回答：

- WRITE 若能声明目标 policy，落库前的意图可见性是否跟随该 policy；
- 增加意图 policy 时，不要求外部构造 RuntimeScope，也不把整个 scope 相等重新变成权限条件；
- 读取正式原子时，是否提示调用者自己仍在进行的 UPDATE 修订；不提示时，Pending 正文不能静默叠加到已提交版本上。

管理页查看与编辑记忆策略的前端组件见 [Memory 可见性策略前端组件 Todo](../todo/memory-visibility-policy-ui.md)，它依赖 Memory CRUD 契约先支持 `access_policy` 的读写。

## 7. 受限穿透访问

owner（2026-10-03）：`IdentityScope` 由操作的发起者与目标 Workspace 组成，正是为了以后 actor 的一次操作可以跨过它所在的 Workspace，对其他 Workspace 做有限访问。actor 进入 Workspace 与 actor 真正进行操作，两者的 Workspace 可以不同：coder 在 WA 中执行任务，创建任务进程时确认 coder 能进入 WA；执行过程中 coder 想去 WB 查找资源，这次操作的发起者是 coder，目标是 WB，与 coder 驻留在 WA 无关。

现在第 3 阶段只接受目标等于驻留 Workspace；授权点接口已经显式接收目标 Workspace，实现受限穿透访问时只需放宽这条规则，不必改动各授权点的签名。

需要回答：

- **授予模型**：谁能对哪个非驻留 Workspace 做什么。分析（2026-10-10）：一次跨 Workspace 的操作要同时满足 agent 自身的白名单与目标一侧的授予；
- **agent 的身份**：Agent Profile 是某个 Workspace 中的记忆原子，`agent_id` 即其 alias，只在该 Workspace 内唯一，不同 Workspace 中的同名 agent 是两条独立的原子。“同一 agent 在不同 Workspace 有不同能力”只会出现在穿透访问中，那时 actor 身份需要带上驻留 Workspace（分析，2026-10-10）；
- **目标的表达**：执行一侧如何给出目标，见本节末尾的 T-5b；
- **资源边界**：资源 owner 仍按目标 Workspace 校验归属；同一进程内会同时出现多个目标 Workspace 的资源，派生缓存按派生源坐标键控，不能从 scope 的存在推导隔离；
- **与子网的关系**：Mount、Bridge 与能力子网是更完整的形态（[AE2 Idea](./ae2-hivememory-architecture-analogy.md)，GitHub Epic #92 的主子网部分），穿透访问的授予模型应能向它扩展。

以下前提与问题于 2026-10-19 从原《执行单元、执行线程与执行环境》Idea（现[执行环境 Idea](./execution-environment.md)）移入，沿用原编号。

**前提 2**（owner，2026-10-05）：一次操作的发起者与目标分开看待。

- **发起者**是客观事实，由执行单元一侧补全，agent 不在工具调用中声明；当前由发起操作的执行线程给出，见 [执行线程](../workspace/execution-threads.md)。
- **目标 Workspace** 是 agent 的意图：现在由补全的一层给出，只接受驻留 Workspace；实现跨 Workspace 的受限穿透访问后，agent 需要能够自己带上操作目标（T-5b）。

### T-5b 跨 Workspace 目标的表达形式

- **选项**：按指令给出目标，一条指令一个目标 / 按引用限定，alias 带上 Workspace 限定 / 其他。
- **影响**：
  - alias 按 (Workspace, alias) 分区，跨 Workspace 的引用必须带限定；
  - agent 只能给出 Workspace key 一类名称，`WorkspaceIdentity` 中的 owner 由补全的一层解析，解析失败与无权访问需要对 agent 表现一致；
  - （分析）MTP 文本不能指定 Workspace 的现有约束，随穿透访问设计另行评审；发起者不能由工具调用指定（前提 2）长期成立。

## 8. 例外、暂缓与已知限制

以下各项不是本文的开放问题，按各自的去向跟踪，这里只列出它们属于这条线。

**已登记的例外与暂缓**

- WorkspaceAsset 的 Store、解析服务与旧端口仍接收 `IdentityScope`，归属拆分暂缓：[WorkspaceAsset 归属拆分 Todo](../todo/workspace-asset-ownership-identity-split.md)；
- chat 请求体中 `session_id` 的未来语义：[外部会话与 Topic 投影 Idea](./external-session-and-topic-projection.md)；
- **root 的声明范围**（owner，2026-10-20）：能以保留的 `system` 进入 root API 的只有用户经 HTTP 入口的管理访问，其他接入来源不能声明 `system`。已落实（2026-10-29，[记忆访问计划](../archive/plans/v0.7.1-memory-access-mcp.md)）：接入登记的 `allow_system_actor` 逐个声明能否声明 `system`，默认不能，server 与 Profile 注入脚本显式允许，外部 harness 不能（[认证与访问登记](../workspace/authentication.md#1-两阶段认证)）。Profile 注入脚本（`hivememory:profile-injector`）不在这条规则的范围内（owner，2026-10-20）：它用于向默认 Workspace 快速注入预设 Agent，理想情况下只在项目初次部署时使用，不应经过权限系统；现状仍作为登记来源经网关以 `system` 进入 root API，项目接入部署流程时再调整。

**相关 Todo 与治理工作包**

- RUN 的错误可以探测记忆的类型与是否存在，不可执行的目标也会记录一次引用：[RUN 目标的可执行性检查 Todo](../todo/run-target-executability-check.md)；
- 前端身份切换时的持久化、失效与清理：[前端身份状态所有权 Todo](../todo/frontend-identity-ownership.md)，对应[身份治理](../governance/security/identity-and-execution-safety.md) Phase S4；
- 显式 Profile 解析失败、权限拒绝与未指定 Profile 的稳定结果，以及失败事件不泄漏不可见正文：身份治理 Phase S1 第 4、5 项；
- 身份与威胁模型清单：身份治理 Phase S0，尚未开始。

**事实文档登记的已知限制**（[认证与访问登记](../workspace/authentication.md#8-已知限制)第 8 节、[身份与授权](../workspace/identity-and-authorization.md#8-已知限制)第 8 节）

- 接入凭据没有签发、轮换、撤销与过期，修改经重启生效；
- principal 不与 Agent 绑定：持有某个外部 principal 凭据的调用方可以声明该用户的任意具体 Agent；
- 用户级访问记录让同一用户的所有具体 Agent 共享一份白名单；
- Patchouli 的交互提交与主动记忆意图提交路由只接收 `IdentityScope`、不做操作授权，依赖唯一的生产调用方先完成授权。

## 9. 地基整理的起点（分析）

以下是分析，不是 owner 决定，也不预排计划。

- 这条线目前没有一份统一的清单，能逐项列出各入口、路由、actor 操作、缓存、后台任务与资源操作的身份输入与授权所有者，也无法据此区分哪些是设计、哪些是补丁。身份治理 Phase S0 定义的正是这份清单，适合作为整理的第一步，第 8 节的例外与已知限制可以据此判断去留。
- 按发布影响，第 4 节的 HTTP 管理入口与第 3 节的登记生效时点直接关系桌面端与多个 harness 同机运行；第 5 节的 P-1b、P-5、P-7 与第 7 节主要服务 controller 模式与多 Workspace，可以留到对应方向重启时再决定。

## 10. 相关文档

| 文档 | 关系 |
|:---|:---|
| [身份与授权](../workspace/identity-and-authorization.md)、[认证与访问登记](../workspace/authentication.md)、[能力层](../workspace/capability.md) | 身份模型、访问边界与调用面的当前事实 |
| [身份治理](../governance/security/identity-and-execution-safety.md) | 跨版本的身份与执行安全目标、成熟度与工作包 S0–S4 |
| [第五代架构 Idea](./fifth-generation-architecture.md) | 3.5、3.6 的理念；P-1、P-4a、P-5、P-7、P-8 的原所在（2026-11-05 移入本文） |
| [外部 Actor Idea](./external-actor-registration-and-runtime-access.md) | E-1 的原所在（2026-11-05 移入本文）；P-1c、P-9c 的决定与复核；controller 模式的身份需求（E-3b、绑定线程的操作适配器） |
| [写入意图后续方向 Idea](./pending-intent-migration.md) | 意图可见性与 policy 的原所在（2026-11-05 移入本文） |
| [执行线程](../workspace/execution-threads.md)（已归档的[执行线程 Idea](../archive/ideas/execution-thread-definition.md)） | 操作的发起者由执行线程给出；线程的目标现为驻留 Workspace，跨 Workspace 目标的表达见第 7 节 T-5b |
| [执行环境 Idea](./execution-environment.md) | 前提 2 与 T-5b 的原所在（2026-10-19 移入本文） |
| [AE2 Idea](./ae2-hivememory-architecture-analogy.md) | 子网与跨 Workspace 的更完整形态 |
