---
title: 外部 Actor 的接入登记与运行时访问
status: idea
horizon: current
serves_version: v0.7.1
owner: project
scope: external-actor-registration-runtime-access-and-adapter-interface
code_paths:
  - src/hivememory/system/access/
  - src/hivememory/workspace/registry.py
  - src/hivememory/workspace/authentication.py
  - src/hivememory/workspace/authorization.py
  - src/hivememory/workspace/capability/
  - src/hivememory/config/access.py
  - src/hivememory/server/deps.py
  - src/hivememory/server/routers/ingest.py
  - src/hivememory/workspace/contracts/
  - integrations/claude-code/connector/claude_code/reader.py
  - src/hivememory/prompts/transcript/actions.py
related_docs:
  - docs/ideas/fifth-generation-architecture.md
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/external-session-and-topic-projection.md
  - docs/ideas/pending-intent-migration.md
  - docs/ideas/identity-and-access-model.md
  - docs/workspace/README.md
  - docs/VISION.md
  - docs/ideas/execution-environment.md
  - docs/archive/ideas/execution-thread-definition.md
last_reviewed: 2026-10-29
---

# 外部 Actor 的接入登记与运行时访问

**文档状态**：Idea，外部 harness 接入未形成实施承诺

**实施关联（2026-10-09）**：Alice 的操作请求、执行凭据与观测标签已通过 [迁移计划验收](../archive/plans/v0.7.0-alice-capability-migration.md)，当前接口见 [子系统契约](../contracts/subsystem-contracts.md)。本 Idea 的 controller/plugin 模式、传输会话到凭据的映射和独立子线程身份仍是后续方向，不能从 Alice 迁移完成推断它们已实现。
**记录日期**：2026-09-27，由原 v0.7.0 计划 B 退回

## 0. 文档性质

> 2026-10-13：文中的“总 Idea”指《Workspace 网络与任务进程架构》Idea，所引节号与问题编号对应其 commit `6add90f` 版本；该 Idea 已重写为[第五代架构 Idea](./fifth-generation-architecture.md)，仍开放的架构级问题（P-1、P-3、P-4a、P-5、P-7–P-10a）保留原编号。
>
> 2026-10-13：文中引用的任务进程 Idea、执行单元 Idea 与身份 Idea 的节号和已决定问题对应 commit `a384688` 版本；三文已精简为后续方向，仍开放的问题（Q-3b、Q-5a、Q-5b、Q-7、Q-8、T-5b、T-7、T-8）保留原编号，已实施的决定见 [Workspace](../workspace/README.md)与[任务进程](../workspace/task-process.md)。
>
> 2026-10-19：执行单元 Idea 改名为[执行环境 Idea](./execution-environment.md)，只保留执行环境的选择与输出词汇；文中引用的执行线程相关前提与问题（前提 1、3、4、7，T-1、T-3 等）以 commit `a384688` 版本为准，执行线程的正式定义已实施，当前事实见 [执行线程](../workspace/execution-threads.md)，设计推导见已归档的[执行线程 Idea](../archive/ideas/execution-thread-definition.md)。

本文由原 v0.7.0 计划 B（外部记忆服务与 Actor 交互契约）于 2026-09-27 退回 Idea。计划的最后版本见 commit `dda9d9d` 中的 `docs/plans/v0.7.0-external-memory-service-and-actor-interaction.md`。

- owner 判定计划 B 作为计划已经作废：原被动输入（现称 Import Bus）的设计已经更新（[Import Bus Idea](./import-bus.md)第 1 节：退化为与记忆系统零主动交互的纯输入），计划中的内容也大多过时；但它要解决的问题仍然存在。
- 本文只保留这个问题以及仍然成立的设计材料。没有保留的内容：阶段划分（EMS-0–6）、交付与验收出口、A/B 交接与依赖顺序、测试矩阵、代码落点与迁移切片、文档收口清单；以 Passive Ingress 返回记忆 context 为前提的实时被动模式；以“System 应用服务 → GlobalSystemBus → Patchouli application”为主干的目标结构（已被总 Idea 第 12 节“能力层是 Actor 唯一可见的 API”这一前提取代）。
- 现状事实按 2026-09-27 的代码核对。
- 待决问题只列出选项及其影响，不替 owner 作出选择；选项顺序不代表倾向。
- **版本归属**（owner，2026-09-27）：外部 Actor 的真实接入（真正的 adapter 接口与外部服务身份等）不在 v0.7.0。v0.7.0 不对外部 Actor 所需的基建作承诺，只验证 Alice 在新架构下跑通、各流程协作无误，使之后的 adapter 不需要再大改系统拓扑结构（[ROADMAP](../ROADMAP.md) 的 v0.7.0 范围）。外部 Actor 分为两种接入模式（1.1）：v0.7.1 的版本目标为 plugin 模式的真实接入，首个 harness 为 Claude Code；controller 模式退回候选方向，版本待定（1.1 的版本安排，owner，2026-10-20；2026-09-27 原定先做 controller 模式）。2026-10-20 起本文的 horizon 为 `current`，服务 v0.7.1，目标窗口见 [ROADMAP](../ROADMAP.md) 第 4.3 与 4.4 节。
- **术语**（owner，2026-10-06）：本文的 adapter 即**操作适配器**（operation adapter），定义见 3.4；1.2 中执行侧面的实现，即 CPU 端口的实现，称 **CPU 驱动**。两者的分层见[第五代架构 Idea](./fifth-generation-architecture.md)第 2 节的执行一侧用语。2026-10-20 用语修订后，执行侧面由执行单元（CPU 的常驻实例）承担，实现执行单元的代码称驱动。

## 1. 问题

owner 表述（2026-09-27）：计划 B 的核心议题是**如何把外部 Actor 的信息注册到系统里，以及外部 Actor 在运行时如何访问系统**，属于 adapter 的接口设计。

它包含三件事：

1. **接入登记**：外部 Actor 及其背后的调用来源以什么形式、由谁、在什么时点登记进系统；
2. **运行时访问**：已登记的外部 Actor 每次调用时经什么传输、如何证明身份、可以调用哪些操作、如何观察结果；
3. **adapter 接口**：协议适配层与能力层之间的边界，adapter 可以做什么、不可以做什么。

### 1.1 两种接入模式（owner，2026-09-27）

外部 Actor 如何接入现有系统，此前从未描述过；它决定了系统能在多大程度上管理外部 agent harness 的行为。owner 表述：这个议题的核心一直没变，就是如何兼容现在主流的 agent harness 生态。owner 采纳以下两种接入模式：

| 模式 | 形态 | 对话的控制权 | 取舍 |
|:---|:---|:---|:---|
| plugin 模式 | HiveMemory 以插件形式挂载在某个外部 harness 中。该 harness 一般有完整的生态（包括 GUI），用户直接使用它，而不是 HiveMemory；HiveMemory 本质上是监听者，同时通过提供 MCP 工具响应 harness 对记忆域的任何工具调用 | 在外部 harness 手中，HiveMemory 对用户的对话没有控制权 | 没有侵入性，用户继续使用自己喜欢的 harness，把 HiveMemory 作为辅助；对用户更友好 |
| controller 模式 | 外部 harness 作为类似 agent 的存在（类似 Agent Client Protocol 的概念）：所有请求统一走 HiveMemory 自己的请求入口，由用户指定想使用的 actor（如同选择 agent），系统处理请求后转发给对应的外部 harness | 流程包裹在 HiveMemory 的系统控制内，只是利用外部 harness 的完备性来干活 | 更突出项目自身的特色 |

- **命名**：此前称为“被动 / 主动服务状态”，是历史遗留叫法，改称 plugin 模式与 controller 模式。两种模式与[任务进程 Idea](./task-process-table-and-registration-entry.md#1-请求方的分类owner2026-09-27)第 1 节的主动请求 / 被动请求是两个维度：controller 模式同时接受两类请求；plugin 模式下的对话不经注册入口。
- **与现有实现的关系**：plugin 模式很像一直以来的 Passive Ingress 模式；controller 模式本质上就是如今的 Alice。任务进程表与注册入口更偏向 controller 模式，这与现有 chat 链路服务 Alice 同源。
- **共同基建**：两种模式对进程的控制能力不同，但都使用 workspace 能力层。到能力层为止是真正的 workspace 架构基建，再往上两种模式各有各的样式。
- **版本安排**：v0.7.0 不能同时完成两者（2026-09-27 原定先做 controller 模式，作为 v0.7.1 的首个真实外部 harness 接入，plugin 模式在其后的 v0.7.x 完善）。
  - **owner 决定（2026-10-20）**：v0.7.1 的版本目标为 plugin 模式的真实外部 harness 接入，首个接入的 harness 为 Claude Code；controller 模式退回候选方向，版本待定。同日先定的 controller 模式目标及其首份计划的方向（首个外部驱动的纯文本闭环）随之撤回，分析保留在第 4 节的“controller 模式的步骤分析”。这是整个版本的目标，按小步分别建立计划，各步的范围与阻塞问题见第 4 节的“v0.7.1 的步骤与各步的阻塞问题”。
  - 理由（owner）：各家 harness 都在完善自己的生态，接口开放程度不一，controller 模式能达到的控制效果取决于被接入方，工程上难以统一（[接口调研快照](../research/harness-interface-survey-snapshot.md)）；非侵入的 plugin 模式更容易得到支持，用户也继续在 harness 自己的界面中工作。目标始终是让外部 harness 使用 Patchouli 记忆，并在真实的日常高强度使用中检验记忆系统的闭环。controller 模式的机制在 plugin 模式下多数用不上，但 Workspace 网络仍是整体的系统设计，任务进程、执行线程与 CPU 端口继续由 Alice 使用。
  - **owner 决定（2026-10-20）**：首份计划的方向为 Import Bus 的重设计，以 Claude Code 的对话回流作为第一条真实路径，让记忆在真实使用中积累（第 4 节的第一步）；记忆访问在其后。Import Bus 的概念改为只向网络输入（[Import Bus Idea](./import-bus.md)前提 1），plugin 模式的设计因此需要完善。
  - **规模控制（owner，2026-10-20）**：v0.7.1 期间不新增只服务 controller 模式的机制；每份计划写明它针对的已观察问题或待验证的核心假设（[VISION](../VISION.md) 第 10.10 节）。

两种模式在各层的对应如下（分析，不是已决定的设计）：

| 层面 | plugin 模式 | controller 模式 |
|:---|:---|:---|
| 能力层（资源操作与授权） | 共用 | 共用 |
| 任务进程与注册入口 | 不使用：外部 harness 的任务对 HiveMemory 不可见 | 使用：外部 harness 与 Alice 同为进程中的 CPU |
| 访问 context | 不绑定进程，形状与管理员直接通道相同（[能力层](../workspace/capability.md#1-三个调用面)第 1 节；[第五代架构 Idea](./fifth-generation-architecture.md) P-9） | 绑定进程 |
| 记忆域访问 | 外部 harness 经 MCP 调用能力层 | 外部 harness 同样可经 MCP 调用能力层；例如 ACP 在创建会话时允许客户端提供 MCP server，HiveMemory 可以提供绑定进程的操作适配器端点（这类由执行单元回到进程的通道见执行单元 Idea T-4） |
| 交互记录的回流 | 经 Import Bus（2026-10-21 实施，见 [Import Bus](../system/import-bus.md)） | 任务进程自行提交（[任务进程](../workspace/task-process.md)） |
| HiveMemory 能否发起任务 | 不能 | 能；被动请求（定时任务、队列任务）只能在此模式下存在 |
| 对外部 harness 的控制 | 无 | 任务边界：actor 选择、输入、生命周期与取消、执行轨迹回流；harness 内部的 loop、工具、上下文压缩与子 agent 派生仍由其自身管理（子 agent 不被识别为执行线程，见 E-5） |
| 现有代码 | Import Bus 与 `/api/v1/import/turns`、Claude Code 连接器、能力层 | Alice 的 chat 链路与 chat run 注册表（内部 CPU） |
| [VISION](../VISION.md) 中的对应 | 兼容轨（第 8.1 节）；四级对照基线的第 3 组“外部 harness + Patchouli”（第 13.1 节） | 以 Alice 为内部 CPU 时即原生轨（第 8.2 节）；以外部 harness 为 CPU 的形态在 VISION 中没有对应 |

两种模式并存，对以下尚未决定的共用设计构成约束（分析，决定时一并考虑）：

- 写入意图（[读取视图与写入意图](../workspace/read-view-and-write-intents.md)）：plugin 模式经 MCP 提交的写入意图没有所属进程（2026-09-28 已消解：写入意图在 workspace 登记，与进程解耦）；
- 任务进程 Idea Q-14：两条回流路径进入同一提交队列；同一 harness 同时以两种模式使用时，同一交互可能被记录两次（2026-09-28：Q-14 选 A，Import Bus 排除在现有系统之外，此问题留待 plugin 模式设计时处理）；
- [外部会话与 Topic 投影](./external-session-and-topic-projection.md)：plugin 模式的会话归外部 harness 所有，Alice 与 controller 模式的会话由 HiveMemory 发起；
- [第五代架构 Idea](./fifth-generation-architecture.md) P-9：不建进程的访问不只来自管理员；
- Import Bus：同时是 plugin 模式的对话回流通道，不只承担对话导入。

此前列入“外部 Actor 形态”单独审议的问题，在两种模式下的归属：

| 问题 | 归属 |
|:---|:---|
| [Q-8 外部 CPU 的进程](./task-process-table-and-registration-entry.md#q-8-外部执行单元的进程) | 只涉及 controller 模式（plugin 模式不建进程）；Q-8a–Q-8c 仍待决 |
| E-4 结果观察的方式 | 主要涉及 plugin 模式，controller 模式可经任务进程观察结果；仍待决 |
| [P-5b 被动请求的 principal](./fifth-generation-architecture.md) | 被动请求只存在于 controller 模式；owner 倾向由登记的 Agent 反推；仍待决 |
| 请求方类型与认证 principal 的关系 | 按两种模式的定义：controller 模式下，用户直接向 HiveMemory 的入口发出请求；plugin 模式下，没有请求进入注册入口，外部 harness 以不建进程的方式访问能力层。外部 harness 连接方的身份如何证明见 P-1c |

### 1.2 harness 登记的两个侧面（owner，2026-09-30）

owner 采纳：一个外部 harness 只登记一份，由这份登记派生两个侧面。

| 侧面 | 回答的问题 | 运行时去向 | 使用方 |
|:---|:---|:---|:---|
| 接入侧面 | 该 harness 以哪个调用来源（principal）接入、经哪种操作适配器（3.4）、可以声称哪些用户身份 | System 接入登记（`SystemActorAccessEntry`，2.1） | 认证网关第一阶段；harness 回调能力层时的认证 |
| 执行侧面 | 任务进程如何驱动该 harness 执行（驱动类型及其配置） | 执行单元：CPU 端口的一个常驻实现，由驱动代码配出 | 任务进程的 Actor 执行阶段 |

- 运行时仍是两个注册表，按方向分开：入站认证属于 System，出站执行属于任务进程。System 装载登记后分别构造两者并注入使用方，与总 Idea D-5 的做法一致。
- 由此，任务进程调用 CPU 采用对象端口：端口由 workspace 定义，CPU 实现，组合根注入，不采用总线路由契约（[子系统公共契约](../contracts/subsystem-contracts.md)第 4.1 节）。
- 登记仍在启动时从配置装载（E-1）。
- 并非每个参与者都有两个侧面（分析）：plugin 模式的 harness 与管理员直接通道只有接入侧面；Alice 两个侧面都有，执行侧面是内置的；测试 CPU 只有执行侧面。
- 驱动与执行单元（分析，2026-10-20 按修订后的用语改写）：驱动是代码，按协议而不是按 harness 编写，例如一个 ACP 驱动可以覆盖所有支持 ACP 的 harness；执行单元由登记数据配出，登记包括选用的驱动、驱动配置与绑定的 principal。能否不写代码就加入一个新 harness 取决于 E-2a：采用通用协议时，只需新增一条登记；按各家 SDK 或 CLI 适配时，每家至少需要一个驱动。入站方向的 `adapters` 字段已经是“代码定义种类、登记数据引用种类”的形态。
- 后续设计（owner，2026-09-30）：actor 对应哪个 CPU 的映射放在哪里（执行侧面 / Agent Profile 的字段 / 独立的映射表 / 其他）；按 actor 或按请求选择 CPU 的机制。v0.7.0 只需要 Alice 接入 CPU 端口。
  - 2026-10-06：按[第五代架构 Idea](./fifth-generation-architecture.md)第 2 节的用语，CPU 专指执行单元，即 CPU 驱动加上它驱动的 harness 实例，actor 是执行单元中的执行线程；本表的执行侧面对应 CPU 驱动，接入侧面对应操作适配器。只有接入侧面的参与者不称 CPU（执行单元 Idea T-6）。
  - 2026-10-20：第五代架构 Idea 第 2 节修订用语：执行单元改指 CPU 的常驻实例，原“CPU 驱动加 harness 实例”合并为执行单元，驱动只指实现它的代码；本表的执行侧面对应执行单元。
  - 2026-10-06：按执行单元 Idea前提 9、10，agent 与执行环境解耦，不存在固定的“actor 对应哪个 CPU”的映射：用户在对话开始前分别选择 agent 与执行环境，选择执行环境不是 workspace 的能力，倾向由系统层级控制、在 CPU 分配时决定 backend（E-6a）。选择的承载、默认值与可用范围见执行单元 Idea T-8。
- 仍待决：登记记录的结构，以及它与 Workspace Actor 访问登记的关系（E-1a）。

## 2. 现状事实（代码核对，2026-09-27）

### 2.1 接入登记

- 两类登记在启动时从配置装载，运行中不可变，修改需要重启：
  - System 接入登记（[`system/access/registry.py`](../../src/hivememory/system/access/registry.py)）：`SystemActorAccessEntry` 包含 `principal_id`、`kind`、`enabled`、`adapters`（该来源可经哪些 adapter 接入）与可选的 `allowed_user_ids`；
  - Workspace Actor 访问登记（[`workspace/registry.py`](../../src/hivememory/workspace/registry.py)）：`WorkspaceActorAccessRecord` 按 (owner, workspace, user, agent) 登记 `enabled` 与 `allowed_operations`；
  - 配置模型位于 [`config/access.py`](../../src/hivememory/config/access.py)（`AccessControlConfig`：`principals`、`workspace_actors`、`context_ttl_seconds`）。`configs/config.yaml` 中没有 access 配置段，按设计 fail closed。
- 除配置文件外，没有登记、修改或撤销接入的入口。

### 2.2 认证

- [`ActorAuthenticationGateway`](../../src/hivememory/workspace/authentication.py) 执行两阶段认证：Principal authentication 由 System 的 `SystemPrincipalAuthenticator` 经 `core.access.PrincipalAuthenticator` 端口实现，Workspace 准入由 `WorkspaceAccessGuard` 完成；两步都通过才签发 `WorkspaceAccessContext`。
- context 是进程内对象，只携带 `IdentityScope`，不作为可序列化的远端凭据。
- 认证网关没有生产调用方。principal 的身份证明应由 adapter 依据接入证据构造，目前没有 adapter 实现这一证明（总 Idea P-1）。

### 2.3 HTTP 入口

（2026-10-21：Import Bus 重设计删除了下述 `/ingest` 入口与记忆上下文的返回，现状见 [Import Bus](../system/import-bus.md)。）

- `/api/v1/ingest` 与 `/api/v1/ingest/flush` 是现有的外部对话事件入口：要求显式 `agent_id`（参与外部会话命名空间），body 中的身份与请求头身份冲突时拒绝。
- 其他资源路由的身份取自请求头 `x-user-id` / `x-workspace-id`，不经认证网关。
- `PassiveIngressService.ingest_event` 在 user 事件上返回编译后的记忆 context，与 Import Bus 的新前提不一致（总 Idea 2.3、3.3）。

### 2.4 能力层

- `workspace/capability/` 下有 memory、agent_profiles、topic、memory_tasks、assets 五类能力服务。读取方法在 backing 调用前执行 `authorize_operation`；写入与管理路径的 operation 检查仍在 Patchouli application，见 [A1 访问边界返工](../archive/plans/v0.7.0-a1-access-boundary-rework.md)。workspace 包的现有实现需要重新调查（总 Idea 5.5）。
- `WorkspaceOperation` 共 11 项，没有代码执行、工具调用、CALL 或创建任务类操作（总 Idea 13.1）。

## 3. 仍然成立的设计材料（来自原计划 B）

### 3.1 三条使用路线

| 路线 | 调用方需要什么 | 必须保留的差异 |
|:---|:---|:---|
| Import Bus（原 Passive Ingress，2026-10-21 重设计） | 外部 bot/harness 自己对话，把已发生的交互交给系统（完整记录或流式事件） | 实时 turn 边界、buffer、flush、背压与重发 |
| 主动资源交互 | 外部 Actor 按需搜索/读取、提交写入或修订意图、观察结果 | 结构化资源引用、显式授权、提交与结果语义；无需先发一条 user 消息 |
| 历史对话导入 | 把已有消息与工具轨迹保真导入，再按策略形成记忆 | 批次、历史发生时间、来源、重复导入、分支/编辑与恢复，不重放实时副作用（与总 Idea Q-13 相关） |

原计划中的 Import Bus 路线还包括“在 user turn 开始时取得可选记忆 context”；按总 Idea 前提 2.3，这一点已不成立，取得记忆属于主动资源交互。

按两种接入模式（1.1）：plugin 模式使用 Import Bus 与主动资源交互两条路线，不建进程；controller 模式下，外部 harness 作为进程中的 CPU 经能力层进行主动资源交互。历史对话导入的范围见总 Idea Q-13。

三条路线可以共享身份映射、资源引用、来源模型和 Patchouli 领域能力，但不共享全部会话规则、状态机或成功条件；消息能够序列化并不能证明处理流程等价。

### 3.2 主动调用与自动交接是两个维度

本节主要针对 plugin 模式（1.1）；controller 模式下，交互记录由任务进程一侧处理（任务进程 Idea Q-14）。

使用路线与“Actor 主动工具调用 / adapter 自动交接”是不同维度。search/read/write/update 可由工具触发；完整交互结束后提交记录由 adapter 自动触发，不交给模型自行决定是否调用保存工具。同一检索或提交操作不因触发方式不同而走不同的能力。

2026-10-06 注：按 3.4，操作适配器只归一化收到的操作请求，不自行发起操作；上文“由 adapter 自动触发”的自动交接因此不属于操作适配器，由 plugin 模式的接入组件随 Import Bus 另行设计。

只提供工具协议（例如 MCP）的接入，仍需另有完成回调、消息流或 flush 集成，才能声称支持自动交接。仅有零散消息或工具调用而没有可靠结束信号时，协议必须能表达“尚未完整”或显式 flush 的边界；取消、中断、idle/shutdown flush 不能伪装为完整成功的回复。

### 3.3 身份的五个概念

区分系统接入登记确认的 `CallerPrincipal`、外部软件/connector source、本次 `ActorIdentity`、Workspace 归属以及消息说话者：

- source、agent_id、user_id 字符串不能自行构成认证结果；同一 principal 可以服务多个 Actor，但不能接受未经接入规则确认的身份声明；
- 输入消息中“某用户说过”属于 provenance 声明，不自动成为已认证的 Actor 身份；
- 读取结果的使用权、提交意图的权限、管理删除权限分别判断；
- 一个仍有效的 context 可以用于多个操作，各操作分别授权；失效或切换 Actor/Workspace 时重新认证，不能在 adapter 内替换 context 的身份字段。

### 3.4 操作适配器的定义与边界

**定义**（owner，2026-10-06；E-3a 的决定）：操作适配器（operation adapter，原称 adapter）把某种协议中的一次操作请求，归一化为 workspace 统一能力层 API 的一次调用，并把结果与错误映射回该协议。

- 例：现有的 MTP READ 与将来经 MCP 提供的记忆读取，都归一化为 `MemoryApplicationService.read` 的调用；管理员操作同样遵循归一化原则，经 HTTP 操作适配器映射到能力层。管理员与 agent 因身份与操作意图不同，最终映射到不同的方法（总 Idea 15.7），归一化的逻辑相同。
- 职责：协议翻译；一次操作映射到一个能力方法；把结果与错误映射回协议。
- 细化（owner，2026-10-09）：操作适配器不直接调用能力层，而是把协议请求解析为固定类型的操作请求，交回 CPU，由 CPU 经 workspace 的单一操作入口调用能力层；请求与能力方法一一对应，对 actor 暴露的操作、操作请求、能力层 API 与 `WorkspaceOperation` 是同一套操作列表。HTTP 管理侧理论上也采用这一形态，暂不迁移。见[子系统公共契约](../contracts/subsystem-contracts.md)第 4.6 节。
- 细化（owner，2026-10-10）：能力层分为 actor API 与 root API。外部 Actor（含 plugin 模式的 harness）只能使用 actor API，其操作列表翻译为 MCP 工具时不含 CALL；管理员（唯一的 root 角色 `system`）经 HTTP 操作适配器调用 root API。上文例子中“管理员操作映射到不同的方法”应理解为映射到不同的 API 面。见[能力层](../workspace/capability.md#1-三个调用面)第 1 节。
- 与连接器的区别（分析，2026-10-29）：两者分处网络两侧。
  - 连接器是运行在外部 harness 一侧的客户端程序：读取会话记录、切分轮次、维护游标与重发，再调用 HiveMemory 的 HTTP 接口；不导入 HiveMemory 包。Claude Code 的连接器见 [Import Bus](../system/import-bus.md#8-claude-code-连接器)第 8 节。
  - 操作适配器与接入登记中的 adapter 都在 HiveMemory 一侧：前者把协议中的一次操作翻译为能力层调用；后者是接入登记 `adapters` 字段中的标识（现有 `http` 与 `local`），表示调用来源从哪个入口进入认证网关。
  - 按 E-2 的候选设计（HiveMemory 以 HTTP 提供 MCP 端点），记忆访问路径上没有连接器：harness 按插件中的声明直接连接该端点，连接器只承担对话回流。

**不属于操作适配器的职责**：

- **身份**：操作适配器不产生身份。入口一侧的认证由认证网关完成；进程内的发起者由执行单元与进程补全（执行单元 Idea前提 1、4），身份如何随调用到达授权点见执行单元 Idea T-1、T-4。
- **触发时机**：交互结束时自动提交记录（3.2）是自行发起操作，不是对收到的操作请求的归一化。controller 模式下交互记录由任务进程封口（[任务进程 Idea](./task-process-table-and-registration-entry.md) Q-14），plugin 模式随 Import Bus 另行设计。
- **任务请求**：HTTP 上的任务请求进入唯一注册入口、创建任务进程，不是能力层的操作，由请求入口承担（执行单元 Idea前提 7）。

**边界的判据**：归一化的目标必须是能力层已有的方法。外部操作无法表达为一次能力调用时，扩展能力层的方法，而不是在操作适配器中实现领域语义；例如 MTP READ 的 pending、redirect 等结果需要由能力层提供（总 Idea 15.5）。

- 分析（2026-10-06）：操作适配器可以持有与协议相关的状态，例如绑定进程时会话到进程的对应关系；是否持有状态不是判据。

**演进**：原先收录的是原边界宪章 §5.3（2026-09-25 裁定）的两条候选判据：

- “adapter 只做五件事：协议翻译；认证交接（经认证网关取得或复用 access context）；一次操作恰好调用一个能力方法；把错误映射为 wire 格式；决定触发时机”；
- “出现以下任一情况即属于能力层代码而非 adapter：需要 resolver、cache、lease 或 registry；组合多个领域步骤；解释资源 policy；持有平面状态”。

2026-10-06 owner 指出这两条不符合归一化的设计意图：五件事中的“认证交接”与“决定触发时机”不是归一化；后一条以机制而不是目的界定 adapter，实际回答的是“什么代码属于能力层”，用语也来自宪章时期的 workspace runtime。两条由上述定义取代。

原计划 B 中与此一致的约束：

- 外部 Actor 不为自己新造一套资源 API，同一资源操作与 Alice 使用同一能力；
- 管理能力由独立 operation 授权，主动 Actor 不能借管理接口获得 owner bypass；
- 同一领域规则不在 Import Bus、主动操作或 connector 中分别重写；首版不增加完全通用的 resource registry；
- “与 Alice 基本相同的资源能力”指相同资源操作遵守同样的授权与领域语义；不要求外部 Actor 采用 Alice 的 frame、CALL、PendingAtomRuntime、prompt history 或 MTP 语法，也不替代外部 harness 的执行 loop。

### 3.5 外部 Actor 需要的操作

原计划 B 的最小能力集（去掉已过时的 Import Bus 一行）：

| 能力 | 内容 | 限制 |
|:---|:---|:---|
| Memory search/read | 结构化引用、内容或编译视图、已有的版本与来源信息 | 使用 Actor 可见读取，不使用管理 bypass |
| Profile read | Agent Profile 定义与授权语义 | 读取不等于实例化 Agent、改变调用方身份或已应用 system prompt；客户端可以不采用定义（与总 Idea P-10 相关；2026-09-28 决定 Profile 的两个 allow 字段演变为能力层的 operation 控制，对所有 CPU 生效，见总 Idea 15.4） |
| Attachment read | 解析已有授权 ref、READY 检查、必要的内容快照 | lease 在服务内管理，不把 lease 对象跨网络传递 |
| Write/update intent | 显式提交内容、目标、理由与来源引用 | Patchouli 决定生成、合并、修订或丢弃；不把日志自动当作写入命令 |
| Pending read/resolve | 登记后读取意图与状态，结算后解析原引用 | 当前事实见 [读取视图与写入意图](../workspace/read-view-and-write-intents.md)，外部提交与回读契约见[写入意图后续方向](./pending-intent-migration.md)第 7 节；记忆任务的观察（原 `task.observe`，2026-10-10 起为只供 `system` 的 root API）不自动授予内容读取权 |
| Result query | 提交类型及真实领域结果或任务投影 | 查询状态不授予读取最终 Memory 的权限；Patchouli 的记忆任务不暴露给 Agent（[能力层](../workspace/capability.md#1-三个调用面)第 1 节），涉及任务投影的部分需按此重新审视 |

能力层是否覆盖执行类操作见总 Idea P-3。

### 3.6 协议必须说明的语义

无论采用哪种传输，外部协议都需要说明：

| 类别 | 必须说明的语义 |
|:---|:---|
| 请求关联 | request、source event、conversation、turn、领域 interaction、operation 的区别及稳定关联 |
| 触发与完整交互 | 区分主动工具请求和自动交接；规定完成回调、seal/flush、取消/中断的消息完整性与重复完成信号处理，不把缺少结束消息推断为成功完成 |
| 资源寻址 | scope 下的 ID/alias/ref；版本是现有事实或显式不可用，不伪造版本和精确 locator |
| 意图提交 | 操作类型、内容/修订目标、已知基版本、来源 refs、客户端去重键、何时视为明确提交 |
| Pending | 登记/物化接纳的区别、稳定引用、读取授权与可见范围、focus/状态/结算投影、保留期；不要求 Alice run/frame |
| 来源 | source、外部 ID、occurred_at 与 received_at、role、工具调用关联；未知与缺失可表达 |
| 读取响应 | 结构化资源与可选渲染 context；未读取全部资源时不能宣称全部引用均被使用 |
| 结果 | 操作种类、其阶段与领域结果、canonical refs、必要错误和查询期限 |
| 错误 | invalid/unsupported、invisible/not-found、conflict、capacity、unavailable、expired/unknown；重试条件 |
| 版本 | 旧客户端兼容、新模型差异、弃用条件与示例 |

这些不是一个所有字段都可空的通用消息包；按操作分为判别模型，复用明确的公共值类型。

### 3.7 结果不能合并为一种“成功”

| 对象 | 权威来源 | 对外成功能说明什么 |
|:---|:---|:---|
| ingress event | 实时入口及其去重/buffer | 已接收、缓冲、重复或忽略；不代表记忆落库 |
| 封口的交互记录 | 会话记录或 buffer 的封口结果（见[外部会话与 Topic 投影](./external-session-and-topic-projection.md)） | 记录已封口并具有明确的结束状态；不表示提交已接纳 |
| interaction submission | Patchouli application / InteractionSubmissionQueue / apply 结果 | 已接纳或已应用，路由后可查询 Topic 关联；不保证生成正式 Memory |
| 写入意图 | 意图的权威持有者（workspace 登记，见 [读取视图与写入意图](../workspace/read-view-and-write-intents.md)） | 已登记内容可按权限读回，并可关联物化状态；不表示任务已接纳或 Memory 已生成 |
| memory intent work | Patchouli 生成/修改服务与实际任务记录 | 该意图处理终态及 created/updated/discarded 等实际结果 |
| canonical resource | Patchouli 与资源读取 | 已存在且当前调用方可读的资源 |

外部 receipt 可以统一“如何定位结果”，不能统一这些对象的状态机。结果按操作种类带各自的阶段与结果；领域所有者继续维护状态，查询服务不通过 best-effort RuntimeEvent 推进第二套业务状态。如需外部 ID 到领域句柄的关联记录，它只持有映射、scope 与必要的去重信息，保留期、容量和缺失行为需要明确，不能复制领域状态成为新的权威记录。

**幂等和模糊失败**：同一调用方、Workspace、操作种类下的去重键绑定规范化的载荷身份。相同键相同请求在承诺窗口内定位同一提交；相同键不同内容显式 conflict；event、interaction 和 intent 不混用一个无类型 ID。接收方提交成功但响应丢失时，客户端用同一键查询或受控重试，不能生成新键盲目重放；超出保留范围返回 unknown/expired 等明确结果，不猜测此前未提交，也不自动再次写入。首版不承诺跨重启 exactly-once。

**取消、失败与读取**：取消只暴露领域实际支持的范围，无法撤销已提交的副作用时返回真实结果，不把“客户端断开”写成“资源已回滚”。参数/授权失败、容量拒绝、已接纳工作失败和查询不可用分开表达。结果查询也校验 scope，知道 ID 不等于有访问权；写入意图的内容读取、任务观察和 canonical 读取分别授权，即使查询返回 canonical ref，读取时仍重新判断当前可见性。调用方回传的来源使用情况是报告，不能把“服务返回过 ref”当作实际使用的证明。

### 3.8 外部会话与 Topic

外部会话 ID 不能直接当作 topic_id 或授权身份；外部客户端不必实例化内部 Python 类型，也不必构造 AgentAction、TurnRecord、Alice run 或 Patchouli 队列 envelope。外部会话与 Topic 的数据交接见[外部会话与 Topic 投影](./external-session-and-topic-projection.md)；外部摘要或 checkpoint 只作为带来源的报告，不接管 Topic 的 canonical summary 或 harness 的 prompt history。

### 3.9 历史导入的设计验证

用含用户更正、助手推测、工具结果、旧发生时间及分支标识的历史样例，检查来源、时间与关联模型能否无损表达，并记录与实时模型不兼容的字段和处理边界。不为每条历史 user 消息自动检索当前 context，不按到达时间把旧偏好当作最新偏好，不把整个批次混入当前活动 Topic，也不因能够解析消息 schema 就声称已经完成历史导入。

## 4. 待决问题

每个问题只列出选项及其影响，不作选择；选项顺序不代表倾向。

### v0.7.1 的步骤与各步的阻塞问题

问题本身仍在各自的位置，这里只按步骤汇总。v0.7.1 的版本目标为 plugin 模式接入 Claude Code（1.1，owner，2026-10-20），第一步对话回流已于 2026-10-21 实施，第二步记忆访问的计划已于 2026-10-29 建立。

plugin 模式在 Claude Code 中的形态（分析，2026-10-20 按 [Claude Code 文档](https://code.claude.com/docs/en/hooks)核对）：插件可以把 hooks 与 MCP server 打包为一个单元；`UserPromptSubmit` hook 可以在模型处理用户消息之前追加上下文，默认时限 30 秒，超时后追加的上下文被丢弃、消息照常送达；`Stop` hook 的输入带有会话记录文件的路径。由此插件有两条通路：对话回流由连接器经 hooks 在轮次结束时把交互交给 Import Bus；记忆访问经 MCP 工具，或者在用户提交消息时自动检索并注入（E-7）。

**第一步：对话回流（Import Bus 重设计）**——已实施（2026-10-21，[归档计划](../archive/plans/v0.7.1-import-bus-redesign.md)）。Claude Code 插件在轮次结束时把该轮的完整记录交给 Import Bus，经统一认证网关与导入授权、Gateway 落位与共用封口进入 Patchouli，应用完成后才确认；不读取记忆，也不返回记忆上下文。当前事实见 [Import Bus](../system/import-bus.md)；真实使用验收随日常使用持续观察（[Todo](../todo/import-bus-real-use-observation.md)）。

**第二步：记忆访问**——计划已建立（2026-10-29，[计划](../plans/v0.7.1-memory-access-mcp.md)）。以下是建立计划时的范围、验收方向与阻塞问题；各项决定之后的候选设计在计划中采纳或调整，以计划为准。

- 范围：Claude Code 插件提供 MCP 工具，经能力层检索与读取记忆，访问不建进程、不绑定进程；结果带有可回查的记忆引用。
- 验收方向：Claude Code 在会话中检索到已有记忆（包括对话回流积累与 Alice 会话形成的记忆）并实际使用；访问按 Workspace 访问登记逐次授权，越权与跨 Workspace 的访问被拒绝。与第一步合起来达到 [ROADMAP](../ROADMAP.md) 第 4.4 节的验收闭环。
- 阻塞问题：已于 2026-10-29 全部决定。本文 E-2（传输承载）、E-3b（操作适配器的代码位置）与 E-7（记忆到达 harness 的方式），见各节；[第五代架构 Idea](./fifth-generation-architecture.md) 的 P-1c（身份证明的实现位置）与 P-9c（不建进程的 actor 访问如何进入 actor API），见下文“[P-1c 与 P-9c](#p-1c-与-p-9c-身份证明与不建进程的访问-context)”。计划为 Claude Code 登记自己的 principal，并在同一计划中落实已决定的 root 声明范围（身份与访问体系 Idea 第 4 节）。
- 不阻塞本步（分析）：只读取时没有异步结果需要观察，E-4 不涉及。
- 候选设计（分析，2026-10-29）：取回的记忆不经对话回流再次提取（回声），检索结果的优先级低于仓库代码与 harness 自己的指令和记忆文件；见 E-7 之后的“与对话回流及 harness 自带记忆的衔接”。

**后续步骤**（候选，未决定）

- 写入意图经 MCP 提交：需要 E-4 与外部 Actor 的提交与回读契约（[写入意图 Idea](./pending-intent-migration.md)）。

### controller 模式的步骤分析（退回候选方向）

2026-10-20 曾定为 v0.7.1 首份计划方向的“首个外部驱动的纯文本闭环”随 controller 模式退回候选方向（1.1）。以下分析在重新排期时仍可使用。

- 范围：选一个 harness，编写驱动并配出第一个外部执行单元。执行单元把输入清单中已经编译好的提示词（人设、检索记忆与附件材料、环境通知）连同用户消息交给 harness，把正文流式回传，结束时返回最终正文；停止转为对 harness 的取消。同时提供最小的执行环境选择：harness 登记的执行侧面从配置装载（1.2、E-1），组合根装配多个执行单元，用户在新建会话时选择执行环境。
- 验收方向：在 HiveMemory 前端选择该 harness 完成多轮对话，停止有效，交互封口后进入记忆生成；新会话中 prepare 检索到的记忆作为材料交给 harness，形成被动注入的跨会话闭环。
- 不包含：harness 主动调用记忆（操作适配器）、工具调用的事件呈现、Profile 应用结果的报告、harness 的子 agent、第二个 harness。
- 阻塞问题：本文 E-2a；[任务进程 Idea](./task-process-table-and-registration-entry.md#q-8-外部执行单元的进程) Q-8；[执行环境 Idea](./execution-environment.md) T-7、T-8、T-11。候选 harness 的程序化接口见[接口调研需求](../research/harness-interface-survey.md)与 [2026-10-20 资料快照](../research/harness-interface-survey-snapshot.md)（模型实测待验证）。
- 不阻塞这一步（分析）：harness 不回调 HiveMemory，也不需要登记外部来源，E-2 主问题、E-3b、[第五代架构 Idea](./fifth-generation-architecture.md) P-1b 与 P-1c、[身份与授权](../workspace/identity-and-authorization.md#8-已知限制)第 8 节的 `system` 声明规则都不涉及；只回传正文与终态字段时，T-10 的三个选项行为相同。
- 后续步骤（候选）：harness 经绑定线程的操作适配器主动使用记忆，需要 E-2 主问题、E-3b、P-1b 与 `system` 声明规则，T-10 需要在此之前决定；Profile 应用结果的报告（P-10a，与 T-7b 相互约束）与前端呈现；E-2a 若选通用协议，接入第二个 harness 验证驱动的通用性。Q-3b 只在外部协议直接发起任务时才需要；P-3 随执行基座考虑（暂不考虑，见 ROADMAP 第 5.4 节）。验收方向见 ROADMAP 第 5.6 节。

### E-1 接入登记的来源与生效时点

**背景**：两类登记只能在启动时从配置装载，运行中不可变（2.1）。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 维持启动时从配置装载 | 新增或撤销外部 Actor 需要修改配置并重启 |
| B | 提供运行时登记入口（例如经管理员直接通道，见总 Idea 15.1 方案 C） | 需要定义登记、修改、撤销的授权与生效时点，以及已签发 context 的处理 |
| C | 登记数据来自外部身份系统 | 需要定义与外部系统的信任关系与同步方式 |
| D | 其他 | —— |

**子问题 E-1a**：System 接入登记与 Workspace Actor 访问登记是否经同一入口维护：同一入口 / 分开维护 / 其他。

**owner 决定（2026-09-27）**：接入登记维持启动时从配置装载（选项 A）。运行时登记与其他配置文件的热更新一并由未来单独的计划实现；E-1a 随该计划决定。

**补充（owner，2026-09-30）**：启动时从配置装载的决定仍然成立，harness 登记（1.2）同样从配置装载。目前 `configs/config.yaml` 没有 access 配置段（2.1），需要先建立承载登记的配置文件。owner 于 2026-10-01 决定：登记使用单独的配置文件，在 [A1 返工](../archive/plans/v0.7.0-a1-access-boundary-rework.md)中建立。2026-10-02：`principals` 与 `workspace_actors` 移出 `config.yaml`，改用单独的配置文件；2026-10-03 细化为两类登记各用一个文件，harness 登记的执行侧面在 principals 文件中留出位置，具体结构不在 A1 返工中定义（总 Idea 15.6）。

### E-2 运行时访问的传输承载

**背景**：现有外部入口只有 HTTP 的 ingest 与 flush（2.3）；原计划 B 的首版选择是 HTTP/JSON 版本化接口加参考客户端。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | HTTP/JSON 版本化接口 | 其他协议由各自的 adapter 另行适配 |
| B | 以 MCP 为主 | 自动交接需要另有完成回调或消息流（3.2） |
| C | 多种传输并存，各由 adapter 适配到同一能力层 | 各 adapter 需要共享 3.6 的语义 |
| D | 其他 | —— |

与任务进程 Idea [Q-3b](./task-process-table-and-registration-entry.md#q-3b-唯一入口与传输入口)（唯一入口是否同时要求唯一的传输入口）相关。

- **owner 决定（2026-10-29）**：选项 C。多种传输各由操作适配器归一到同一能力层，这是建立统一 Workspace 能力层的核心理念；对支持 MCP 的 harness，记忆访问以 MCP 为传输。子问题 E-2a 随 controller 模式退回候选方向，仍待决。
- 候选设计（分析，不是 owner 决定，可在计划评审时调整）：MCP 端点由 HiveMemory 以 HTTP 提供，harness 按插件中的声明直接连接；对话回流继续使用已实施的 Import Bus。

**子问题 E-2a controller 模式下驱动外部 harness 的方式**：上述选项是外部 harness 调用 HiveMemory 的传输；controller 模式还需要 HiveMemory 驱动外部 harness（1.1）。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 采用 ACP 一类的通用协议 | 一个 adapter 可以覆盖多个 harness；受限于各 harness 对该协议的支持程度 |
| B | 按 harness 分别使用其 SDK 或非交互 CLI | 每个 harness 需要一个 adapter；可使用各 harness 的专有能力 |
| C | 其他 | —— |

### E-3 adapter 的判据与代码位置

- **E-3a 判据**：采用 3.4 的候选判据 / 其他划分。
  - **owner 决定（2026-10-06）**：不采用原候选判据，改以归一化定义操作适配器：把一次操作请求归一化为能力层 API 的一次调用，判据是归一化的目标必须是能力层已有的方法（3.4）。
- **E-3b 代码位置**：入口层（`server`） / workspace 包内 / 独立的 adapter 包 / 其他。位置决定操作适配器在包分层中的层级，以及它能导入哪些包（分层规则见[系统架构概览](../architecture/overview.md)）。
  - **owner 决定（2026-10-29）**：MCP 的操作适配器放在与 `server` 同层的独立入口包；MTP 的操作适配器留在 `agent_runtime`，不一起迁移。本决定只涉及 plugin 模式下不绑定进程的 MCP 适配器；controller 模式下绑定线程的适配器（T-4）随该模式另行考虑。
  - 两者不放在一起的理由（分析）：
    - 分层：MTP 的动词由 Alice 的 agent loop 调用，放到 `server` 同层后 Alice 无法导入；MCP 适配器要经认证网关与门面取得访问 context，放不进子系统层。
    - 两者位于执行线程的两侧：MTP 适配器在执行线程内运行，只持有线程句柄 `OperationSubmitter`，读不出 context；MCP 适配器位于网络入口，没有线程，每次调用经网关签发一份请求级 context（P-9c）。
    - MTP 的动词还承担执行单元的行为：RUN 在沙箱中执行用户工具，`sys_*` 系统工具在本地执行，CALL 是 Alice 的编排，SEARCH 把结果编译为 Agent 可读的检索上下文。MCP 适配器不需要这些。
  - 能力层统一的是归一化的目标，不是适配器的位置。两个适配器共用的部分已经位于更低的层：同一套操作请求（`workspace.contracts`）与 actor API、中立的错误类型，以及 `engines.memory_compiler` 的记忆渲染。各适配器只各自实现协议两端：解析请求，把结果与错误映射回协议（MTP 一侧见 [`agent_runtime/mtp/error_mapping.py`](../../src/hivememory/agent_runtime/mtp/error_mapping.py)）。
  - 候选设计（分析，不是 owner 决定，可在计划评审时调整）：
    - 入口包只导入 actor API 的读取方法与认证网关的公开接口，由组合根装配；不放进 `server`（能力层的三个调用面互不借用，server 在能力层中只调用 root API），也不放进 `workspace`（适配器属于传输层）。包名在计划中确定。
    - 用语：当前的[子系统公共契约](../contracts/subsystem-contracts.md)第 4.6 节把 MTP 一侧称为“执行适配器”，与 E-3a 定义的操作适配器是同一概念；MCP 适配器落地时事实文档第一次出现两个适配器，随记忆访问的计划统一为“操作适配器”。

### E-4 结果观察的方式

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 按操作类型分别查询 | 每类操作各有查询入口与保留规则 |
| B | 统一的 receipt 定位，再按类型返回 | 需要一个只持有映射的关联记录，保留期与容量需要定义（3.7） |
| C | 推送或回调 | 外部方需要提供可达端点；推送不能作为唯一的终态真相 |
| D | 组合 | —— |

**owner 决定（2026-09-27）**：本问题与 Q-8 联动，列入外部 Actor 形态的单独审议。同日审议为两种接入模式（1.1）；本问题主要涉及 plugin 模式，仍待决。

### E-5 外部执行单元的执行线程

**背景**（2026-10-06）：按执行单元 Idea前提 3、4，执行单元认识其全部执行线程，并为每个线程的工具调用补上发起人。Alice 的线程（frame）在进程内，可以直接观察；主流外部 harness 大多自带子 agent，controller 模式下 HiveMemory 只能经驱动与回调通道（执行单元 Idea T-4）看到 harness 交出的信息。本问题只涉及 controller 模式，plugin 模式不建进程（1.1）。

原列出的子问题：E-5a 子线程的可见性（外部执行单元一律只有主线程 / 驱动或协议能上报子 agent 时按执行单元 Idea T-3 的派生流程登记为子线程）；E-5b 线程标记跨网络（进程内的线程句柄按对象身份判定有效、不提供序列化，经网络回调时无法沿用）。

**owner 决定（2026-10-06）**：作为最坏打算，忽视外部 harness 的子线程派生行为。一个进程内的所有 actor 共享主线程的 context，外部执行单元发起的全部操作都记在主线程的 actor 名下。E-5a 取第一项，E-5b 随之不再需要。

- **理由**（owner）：外部 harness 的子 agent 派生难以控制，至少在现阶段实现难度远大于收益。这在 owner 的预期之内，也是坚持自行实现一套 agent 执行环境（Alice）的原因之一：自己的执行环境可以在派生时确定子线程的身份（执行单元 Idea T-1）。
- **依据**（分析，2026-10-06，按 [ACP 仓库文档](https://github.com/agentclientprotocol/agent-client-protocol)核对）：
  - 驱动认不出派生调用：ACP 的 `tool_call` 通知带 `toolCallId`、`title`、`kind`、`status` 等字段，不标识所调用的工具；标识工具的 `name` 字段仍是 [RFD 提案](https://github.com/agentclientprotocol/agent-client-protocol/blob/main/docs/rfds/tool-call-name.mdx)。该提案指出从 `title` 或 `rawInput` 推断工具“脆弱且依赖实现”，并且 ACP 不赋予工具名任何授权语义。判断“这是一次派生、目标是谁”只能按 harness 分别编写规则。
  - 来不及拦截：`tool_call` 通知是事后报告；只有 harness 对该工具发出 `session/request_permission`（协议中为 MAY）时，驱动才能在派生发生前介入。
  - 无法签发 context：harness 自己的子 agent 类型不是 HiveMemory 登记的 agent，没有 Agent Profile，也没有访问登记，无法通过 Workspace 准入（E-6a）。
- **影响**（分析）：
  - 外部 harness 子 agent 发起的读取与写入意图，以主线程 actor 的权限授权，来源也记为主线程的 actor；这与 Alice 目前的情形相同（总 Idea 18.2），差别是 Alice 将按执行单元 Idea T-1、T-2 改为每个线程一份 context。
  - 在 HiveMemory 看来，外部执行单元只有主线程一个 actor，身份 Idea 第 2 节“一份 context 只属于一个 actor”仍然成立；执行单元 Idea T-1a、T-1b、T-2a 只涉及 Alice。
  - 外部执行单元不经过派生授权（执行单元 Idea T-3）；子 agent 能做什么只受主线程 actor 的白名单与资源 policy 约束。
  - 失效条件：出现能在协议中可靠标识派生调用、并允许子 agent 使用独立端点的 harness 或协议版本，且接入它的收益足够时，重新讨论。
- **后续**（2026-10-19）：Alice 的子线程随 CALL 返回结束；外部 harness 既无法探测子 agent 的结束时点，也不一定能取得创建时点，子线程可能根本不会创建，这一点与停止时它在收到取消通知前的行为一起，留到接入外部 harness 时考虑，见已归档的[执行线程 Idea](../archive/ideas/execution-thread-definition.md)第 3.1、3.3 节；Alice 一侧的现状见 [Alice 编排](../alice/orchestration.md#8-失败取消与降级)第 8 节。

与[任务进程 Idea](./task-process-table-and-registration-entry.md#q-8-外部执行单元的进程) Q-8（外部 CPU 的进程，尤其 Q-8c 的粒度）相关。

### E-6 actor 与执行单元的对应

**背景**（2026-10-06）：1.2 把“actor 对应哪个 CPU 的映射”列为后续设计。现状是 `agent_id` 取自 Agent Profile 的 alias，执行单元只有 Alice；1.1 中 controller 模式“由用户指定想使用的 actor（如同选择 agent）”。按执行单元 Idea前提 3，actor 是执行线程，CPU 是执行单元。

按 E-5 的决定，外部执行单元只有主线程；本问题只涉及外部执行单元的主线程以哪个 actor 运行，以及 Alice 的 CALL 目标能否落在外部执行单元。

- **E-6a 外部 harness 的定位**：原选项为 harness 实例是一个执行单元，其中运行的 actor 由登记或 Agent Profile 决定 / harness 本身作为一个 actor（有自己的 `agent_id` 与 Profile），由驱动构成执行单元 / 其他。
  - **owner 决定（2026-10-06）**：取第一项，并且 agent 与执行环境分别由用户选择（执行单元 Idea前提 9、10）。外部 harness 最初被视为一个单独的 agent，现在是一个执行环境；可以在其上应用 Agent Profile，外部协议是否支持另当别论。主线程以用户选择的 agent 运行。
  - 影响（分析）：Profile 的执行参数在外部 harness 上只能尽力应用，应用结果的承载见总 Idea P-10a；按 ACP 仓库文档（2026-10-06 核对），`session/new` 未见系统提示词字段，模型只能在 harness 声明的配置项中选择（执行单元 Idea前提 9 的分析）。组合的有效性见执行单元 Idea T-7。
- **E-6b CALL 的目标位于另一个执行单元**：前提 3 中子线程与主线程在同一执行单元内；若 CALL 的目标 actor 映射到另一个执行单元，同一进程内会出现跨执行单元的线程。选项：允许，一个进程内可以有多个执行单元 / 不允许，CALL 只能派生本执行单元能够运行的 actor / 其他。影响：任务进程 Idea 1.2 的 CPU 分配目前是每个进程一次；Q-10 只规定子线程在父进程内执行。
  - 分析（2026-10-06）：E-6a 决定后，agent 不再绑定执行环境，“CALL 的目标 actor 映射到另一个执行单元”不再自然发生，被调用方可以在父线程所在的执行环境中以自己的 Profile 运行；同时执行单元 Idea 前提 10 规定执行环境只由用户在对话开始前选择，CALL 无法为子线程选择另一执行环境。第一项（允许）与之冲突，待 owner 确认。
  - **owner 决定（2026-10-13）**：不允许。CALL 只能派生本执行单元能够运行的 actor，被调用方在父线程所在的执行环境中以自己的 Profile 运行。影响（分析）：一个任务进程只对应一个执行单元，与执行单元 Idea 前提 10“执行环境只由用户在对话开始前选择”一致，CPU 分配仍是每个进程一次。

### E-7 plugin 模式下记忆到达 harness 的方式

**背景**（2026-10-20）：plugin 模式下 HiveMemory 不控制对话，记忆只能经 harness 的扩展点到达模型。Import Bus 只向网络输入、不返回记忆上下文（[Import Bus Idea](./import-bus.md)前提 1，2026-10-21 已实施），原 `/ingest` 在 user 事件上返回记忆上下文的行为已删除。本问题阻塞第 4 节中记忆访问一步。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 只由 harness 经 MCP 工具主动调用 | 是否检索由模型决定，可能该用的记忆没有被用上；与 [VISION](../VISION.md) 假设 H3（主动记忆交互优于纯预检索注入）的检验方式一致 |
| B | 连接器在用户提交消息时自动检索并注入上下文（经能力层，不经 Import Bus） | 每条消息都承担一次检索的延迟与成本，经 Gateway 分析时还多一次模型调用；受 harness hook 时限约束，超时则本轮没有记忆；自动检索与 MCP 工具相同，以请求级访问 context 发起（P-9c，2026-10-29 已决定） |
| C | 两者并存 | 需要避免同一轮重复检索与重复注入 |
| D | 其他 | —— |

- **owner 决定（2026-10-29）**：选项 A。plugin 模式下暂时不再提供主动预检索的能力，记忆只经 MCP 工具到达 harness，由 agent 自己选择使用。
- 候选设计（分析，不是 owner 决定，可在计划评审时调整）：MCP server 的说明引导 agent 何时检索；真实使用中记录该用而未用记忆的情况，作为 [VISION](../VISION.md) 假设 H3 的检验样本。

**与对话回流及 harness 自带记忆的衔接**（候选设计，分析，2026-10-29，不是 owner 决定，可在计划评审时调整）：以下两点不随 E-7 的选项而变，来自 Claude Code 对话回流的真实运行观察，不另立问题。

- **回声**：选项 A、C 中，经 MCP 工具取回的记忆作为普通工具结果留在会话记录里，本轮结束时随对话回流再次导入。
  - 生成视图会渲染外部工具的结果（[生成转写的动作段](../../src/hivememory/prompts/transcript/actions.py)），提取模型可能把已有记忆再提取一遍，形成重复记忆，并在之后的检索中自我强化。
  - Alice 经 MTP READ、SEARCH 读回的记忆不渲染进生成，正是为了避免这一点；但外部 MCP 调用在连接器看来只是普通工具调用。
  - 候选做法：连接器按 MCP server 名识别 HiveMemory 自己的工具调用，并在导入事件上标明来源；生成视图对这类动作只保留一行摘要（工具名与查询），不渲染结果，与 MTP READ、SEARCH 同样处理。标注用哪个字段在计划中确定。
  - 选项 B 没有这条回路：经 `UserPromptSubmit` hook 注入的上下文以 `attachment` 条目写入 Claude Code 会话记录，而连接器只读取 `user` 与 `assistant` 条目（[会话记录读取器](../../integrations/claude-code/connector/claude_code/reader.py)），不会再次导入。助手据此作出的回答照常进入生成。
- **优先级与时效**：取回的记忆描述的是生成当时的状态，常含“某分支待推送”“某项进行中”这类一两天内就过时的内容。
  - 2026-10-29 核对：对话回流生成的 5 条记忆中，2 条标为 WORK_IN_PROGRESS，所述工作在记忆生成之前已经合并。
  - harness 侧还有自己的指令与记忆：Claude Code 的项目指令文件（CLAUDE.md、AGENTS.md），以及按项目保存的记忆文件。记忆文件的索引在每次会话自动加载，记录的是与用户协作的偏好与纠正。
  - 两边已经有重叠：写入 harness 记忆文件的决定，作为工具调用留在会话记录中，经对话回流也被提取进 HiveMemory 的记忆（例如任务进程不回退的决定）。
  - 候选做法：MCP 工具的说明与每条结果写明记忆的更新时间与来源，并声明优先级——仓库中的代码与文档、harness 自己的指令与记忆文件，优先于 HiveMemory 的检索结果；检索结果作为带日期的历史证据，引用前先核实。
  - 本步不做两套记忆之间的同步或去重：HiveMemory 不读取、不写入 harness 的记忆文件，两边冲突时按上述优先级处理。

### P-1c 与 P-9c 身份证明与不建进程的访问 context

**背景**：两个问题位于[第五代架构 Idea](./fifth-generation-architecture.md) 第 5 节，阻塞第 4 节中记忆访问一步，都不易回退：P-1c 决定外部调用方的身份由谁证明，P-9c 决定 actor API 是否接受进程之外的访问。

#### P-1c 身份证明的实现位置

- **owner 决定（2026-10-29）**：由统一认证网关证明。网关第一阶段验证 principal，第二阶段做 Workspace 准入；这是 Workspace 架构的基础，现有的 HTTP 入口、任务进程注册与 Import Bus 都经网关认证（[认证与访问登记](../workspace/authentication.md#1-两阶段认证)第 1 节）。
- **复核（owner，2026-10-29）**：同日先写的候选设计是“沿用 server 的 principal 与 `http` adapter，不为 Claude Code 另行登记 principal，也不做凭据证明”，已作废。
  - 作废的理由：它重复了 Import Bus 落地时的偏离，与 1.2“一个外部 harness 只登记一份，接入侧面回答以哪个 principal、经哪种操作适配器接入”相悖，也与 Import Bus Idea D-8a（2026-10-20 选“经统一认证网关”，影响写明“输入来源需要作为 principal 登记”）相悖。
  - 一个 harness 对应一个 principal。principal 是接入的合法身份标识，不建进程的 plugin 模式下同样生效；adapter 标识同理，表示调用来源经哪个入口接入，外部入口不复用 `http`。
  - **owner 决定**：接入证据采用每个外部 principal 一份本地共享密钥，由网关第一阶段核对，入口只携带证据。
  - **owner 决定**：对话回流（Import Bus）在记忆访问的计划中一并迁移到 harness 自己的 principal。
- 具体设计见[记忆访问计划](../plans/v0.7.1-memory-access-mcp.md)第 5.2、5.6 节，包括：
  - Claude Code 的接入登记；
  - `mcp` 与 `import` 两个 adapter 标识；
  - 凭据只以环境变量名登记；
  - 登记逐个声明能否声明 `system`，落实身份与访问体系 Idea 第 4 节的 root 声明范围。

#### P-9c 不建进程的 actor 访问如何进入 actor API

- **owner 决定（2026-10-29）**：与 root 的直接通道相同，不建进程的 actor 访问只有操作、没有进程，每次操作对应一份访问 context，由入口经统一认证网关签发，操作结束时撤销。
- 现状核对（代码，2026-10-29）：
  - 这种 context 已经存在。root 管理、Import Bus 与 Profile 注入脚本使用的都是 `WorkspaceAccessContext`，运行绑定为请求级（`AccessRunType.REQUEST`）：server 在请求开始时经网关签发，请求结束时撤销（[`server/deps.py`](../../src/hivememory/server/deps.py) 的 `request_access_for_claims`）。任务进程、执行线程与请求是同一种密封凭据的三种运行绑定，不需要新的 context 类型。
  - actor API 目前只接受线程 context：授权经 `authorize_thread_operation`，运行绑定不是执行线程即拒绝（[`workspace/authorization.py`](../../src/hivememory/workspace/authorization.py)）；actor API 唯一的调用方 `WorkspaceOperationEntry` 在分派前先经线程表判定线程存活。底层的 `authorize_operation`（准入、目标与白名单检查）不检查运行绑定。
  - READ、SEARCH、RUN 授权后只使用 scope；WRITE、UPDATE 还要从线程取得 `process_id`、`thread_id` 与派发话题。
- 候选设计（分析，不是 owner 决定，可在计划评审时调整）：
  - 授权：actor 授权接受请求级绑定，目标取驻留 Workspace（与线程相同，请求不能切换 Workspace），白名单照常逐次授权；线程绑定的路径不变。
  - 调用路径（已作废，见下方复核）：请求级访问不经线程专用的 `WorkspaceOperationEntry`，MCP 端点经门面取得 actor API 的读取方法。
  - 写入：目前不开放写入类 MCP 工具。写入意图的登记需要进程关联与派发话题，不建进程的写入如何取得它们是后续必须解决的问题，随第 4 节后续步骤“写入意图经 MCP 提交”设计。
- **复核（owner，2026-10-29，记忆访问计划阶段二实施后）**：上面的调用路径让 actor API 有了两个出口，MCP 绕过操作入口的调用约束，违背 3.4 的细化（2026-10-09）——操作适配器经 workspace 的单一操作入口调用能力层。
  - **owner 决定**：actor API 只经操作入口，门面不交出 actor API；
  - **owner 决定**：操作入口新增请求级提交，只开放检索与读取，门面交出操作入口供 MCP 入口使用；
  - **owner 决定**：actor 操作的授权合并为一个方法，接受线程或请求级 context。
  - 具体设计见[记忆访问计划](../plans/v0.7.1-memory-access-mcp.md)第 5.3、5.4 节。

## 5. 相关问题（位于其他文档）

| 问题 | 位置 | 关系 |
|:---|:---|:---|
| P-1 经网络接入的 Actor 每次请求如何证明身份 | [第五代架构 Idea](./fifth-generation-architecture.md) P-1 | 运行时访问的身份证明；P-1a 已决定：注册前经认证网关验证身份，此后每次请求重新校验（总 Idea 15.2）；P-1c 已决定（2026-10-29）：由统一认证网关证明，每个外部 principal 一份本地共享密钥作为接入证据，见第 4 节“P-1c 与 P-9c” |
| P-3 能力层是否覆盖执行类操作 | 同上 | 外部 Actor 可调用的操作范围（3.5） |
| P-9c 不建进程的 actor 访问如何进入 actor API | 同上 | plugin 模式的访问不建进程（1.1）；已决定（2026-10-29）：每次操作一份请求级访问 context，见第 4 节“P-1c 与 P-9c” |
| P-10 Agent Profile 的能力描述是否与 MTP 解耦 | 同上 | 外部 Actor 如何理解 Profile；已决定：allow 字段并入能力层的 operation 控制（总 Idea 15.4），P-10a 仍待决 |
| Q-8 外部 CPU 的进程 | [任务进程 Idea](./task-process-table-and-registration-entry.md#q-8-外部执行单元的进程) | 外部 Actor 的调用与进程的关系、回收与取消；只涉及 controller 模式（1.1） |
| Import Bus 后续方向 | [Import Bus Idea](./import-bus.md)第 2 节 | 流式监听、历史导入另设通道、第二个 harness 的连接器；完整交互记录的接收已实施 |
| T-1–T-6 执行单元与执行线程 | [执行线程](../workspace/execution-threads.md) | T-1（子线程身份在派生时确定）、T-2（每个线程一份 context）与 T-6（只有接入侧面的参与者不称 CPU）已决定；外部执行单元不识别子线程，见 E-5；回调通道与操作适配器的绑定（T-4）关系到 E-3b；T-1a、T-1b、T-2a 只涉及 Alice |
