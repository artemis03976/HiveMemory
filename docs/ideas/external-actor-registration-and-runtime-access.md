---
title: 外部 Actor 的接入登记与运行时访问
status: idea
horizon: long-term
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
  - src/hivememory/server/routers/import_bus.py
  - src/hivememory/workspace/contracts/
  - integrations/claude-code/connector/claude_code/reader.py
  - src/hivememory/engines/perception/material.py
  - src/hivememory/mcp_server/server.py
  - src/hivememory/workspace/intents/registry.py
related_docs:
  - docs/ideas/fifth-generation-architecture.md
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/external-session-and-topic-projection.md
  - docs/archive/ideas/pending-intent-migration.md
  - docs/ideas/identity-and-access-model.md
  - docs/workspace/README.md
  - docs/VISION.md
  - docs/ideas/execution-environment.md
  - docs/archive/ideas/execution-thread-definition.md
last_reviewed: 2026-11-20
---

# 外部 Actor 的接入登记与运行时访问

## 0. 文档性质

本文是外部 Actor（外部 agent harness）接入 HiveMemory 的设计：如何登记、运行时如何访问、操作适配器的边界，以及仍开放的问题。plugin 模式已经实现，当前事实见 [MCP 记忆访问](../contracts/mcp.md)与 [Import Bus](../system/import-bus.md)；controller 模式是候选方向。正文是已定的设计，候选设计标“候选”。

- 问题编号沿用原编号（E-x、P-x、Q-x、T-x）。文中的“总 Idea”指已重写为[第五代架构 Idea](./fifth-generation-architecture.md)的《Workspace 网络与任务进程架构》Idea（commit `6add90f`），“执行单元 Idea”的前提与 T-x 编号指[执行环境 Idea](./execution-environment.md)在 commit `a384688` 中的版本；执行线程的当前事实见 [执行线程](../workspace/execution-threads.md)。
- **术语**：本文的 adapter 即**操作适配器**（operation adapter），定义见 3.4；1.2 中执行侧面由执行单元（CPU 的常驻实例）承担，实现执行单元的代码称**驱动**。两者的分层见[第五代架构 Idea](./fifth-generation-architecture.md)第 2 节的执行一侧用语。

## 1. 问题

核心议题是如何把外部 Actor 的信息注册到系统里，以及外部 Actor 在运行时如何访问系统，属于 adapter 的接口设计。它包含三件事：

1. **接入登记**：外部 Actor 及其背后的调用来源以什么形式、由谁、在什么时点登记进系统；
2. **运行时访问**：已登记的外部 Actor 每次调用时经什么传输、如何证明身份、可以调用哪些操作、如何观察结果；
3. **adapter 接口**：协议适配层与能力层之间的边界，adapter 可以做什么、不可以做什么。

### 1.1 两种接入模式（owner，2026-09-27）

接入方式决定了系统能在多大程度上管理外部 agent harness 的行为。这个议题的核心是如何兼容主流的 agent harness 生态，接入分两种模式：

| 模式 | 形态 | 对话的控制权 | 取舍 |
|:---|:---|:---|:---|
| plugin 模式 | HiveMemory 以插件形式挂载在某个外部 harness 中。该 harness 一般有完整的生态（包括 GUI），用户直接使用它，而不是 HiveMemory；HiveMemory 本质上是监听者，同时通过提供 MCP 工具响应 harness 对记忆域的任何工具调用 | 在外部 harness 手中，HiveMemory 对用户的对话没有控制权 | 没有侵入性，用户继续使用自己喜欢的 harness，把 HiveMemory 作为辅助；对用户更友好 |
| controller 模式 | 外部 harness 作为类似 agent 的存在（类似 Agent Client Protocol 的概念）：所有请求统一走 HiveMemory 自己的请求入口，由用户指定想使用的 actor（如同选择 agent），系统处理请求后转发给对应的外部 harness | 流程包裹在 HiveMemory 的系统控制内，只是利用外部 harness 的完备性来干活 | 更突出项目自身的特色 |

- 两种模式与[任务进程 Idea](./task-process-table-and-registration-entry.md#1-请求方的分类)第 1 节的主动请求 / 被动请求是两个维度：controller 模式同时接受两类请求；plugin 模式下的对话不经注册入口。
- 与现有实现的关系：plugin 模式对应 Import Bus 与 MCP 记忆访问；controller 模式本质上就是如今的 Alice。任务进程表与注册入口更偏向 controller 模式，这与现有 chat 链路服务 Alice 同源。
- 共同基建：两种模式对进程的控制能力不同，但都使用 workspace 能力层。到能力层为止是真正的 workspace 架构基建，再往上两种模式各有各的样式。
- **先做 plugin 模式，controller 模式是候选方向**。理由：各家 harness 都在完善自己的生态，接口开放程度不一，controller 模式能达到的控制效果取决于被接入方，工程上难以统一（[接口调研快照](../research/harness-interface-survey-snapshot.md)）；非侵入的 plugin 模式更容易得到支持，用户也继续在 harness 自己的界面中工作。目标始终是让外部 harness 使用 Patchouli 记忆，并在真实的日常高强度使用中检验记忆系统的闭环。controller 模式的机制在 plugin 模式下多数用不上，但 Workspace 网络仍是整体的系统设计，任务进程、执行线程与 CPU 端口继续由 Alice 使用。

两种模式在各层的对应（候选分析）：

| 层面 | plugin 模式 | controller 模式 |
|:---|:---|:---|
| 能力层（资源操作与授权） | 共用 | 共用 |
| 任务进程与注册入口 | 不使用：外部 harness 的任务对 HiveMemory 不可见 | 使用：外部 harness 与 Alice 同为进程中的 CPU |
| 身份 | 入口每次调用认证得到工作 actor，只在本次调用内，与管理员直接通道相同（[认证与访问登记](../workspace/authentication.md#4-工作-actor-的角色与持有位置)；[第五代架构 Idea](./fifth-generation-architecture.md) P-9） | 进程记录持有执行者与系统的 root 工作 actor |
| 记忆域访问 | 外部 harness 经 MCP 调用能力层 | 外部 harness 同样可经 MCP 调用能力层；例如 ACP 在创建会话时允许客户端提供 MCP server，HiveMemory 可以提供绑定进程的操作适配器端点（这类由执行单元回到进程的通道见执行单元 Idea T-4） |
| 交互记录的回流 | 经 Import Bus（[Import Bus](../system/import-bus.md)） | 任务进程自行提交（[任务进程](../workspace/task-process.md)） |
| HiveMemory 能否发起任务 | 不能 | 能；被动请求（定时任务、队列任务）只能在此模式下存在 |
| 对外部 harness 的控制 | 无 | 任务边界：actor 选择、输入、生命周期与取消、执行轨迹回流；harness 内部的 loop、工具、上下文压缩与子 agent 派生仍由其自身管理（子 agent 不被识别为执行线程，见 E-5） |
| 现有代码 | Import Bus 与 `/api/v1/import/turns`、各 harness 的连接器、MCP 入口、能力层 | Alice 的 chat 链路与任务进程（内部 CPU） |
| [VISION](../VISION.md) 中的对应 | 兼容轨（第 8.1 节）；四级对照基线的第 3 组“外部 harness + Patchouli”（第 13.1 节） | 以 Alice 为内部 CPU 时即原生轨（第 8.2 节）；以外部 harness 为 CPU 的形态在 VISION 中没有对应 |

两种模式并存时，共用的设计受到以下约束：

- 写入意图在 workspace 登记，与进程解耦，plugin 模式经 MCP 提交的写入意图因此不需要所属进程（[读取视图与写入意图](../workspace/read-view-and-write-intents.md)）；
- 同一 harness 同时以两种模式使用时，同一交互可能被记录两次（[任务进程 Idea](./task-process-table-and-registration-entry.md#plugin-模式与进程模型的交界)，仍开放）；
- [外部会话与 Topic 投影](./external-session-and-topic-projection.md)：plugin 模式的会话归外部 harness 所有，Alice 与 controller 模式的会话由 HiveMemory 发起；
- 不建进程的访问不只来自管理员（P-9c，见第 4 节“[P-1c 与 P-9c](#p-1c-与-p-9c-身份证明与不建进程的访问-context)”）；
- Import Bus 同时是 plugin 模式的对话回流通道，不只承担对话导入。

“外部 Actor 形态”相关问题在两种模式下的归属：

| 问题 | 归属 |
|:---|:---|
| [Q-8 外部 CPU 的进程](./task-process-table-and-registration-entry.md#q-8-外部执行单元的进程) | 只涉及 controller 模式（plugin 模式不建进程）；Q-8a–Q-8c 仍待决 |
| E-4 结果观察的方式 | 已关闭：项目中不存在“结果观察”这一概念，见 E-4 |
| [P-5b 被动请求的 principal](./identity-and-access-model.md#p-5-被动请求的认证) | 被动请求只存在于 controller 模式；owner 倾向由登记的 Agent 反推；仍待决 |
| 请求方类型与认证 principal 的关系 | controller 模式下，用户直接向 HiveMemory 的入口发出请求；plugin 模式下，没有请求进入注册入口，外部 harness 以不建进程的方式访问能力层。外部 harness 连接方的身份如何证明见 P-1c |

### 1.2 harness 登记的两个侧面（owner，2026-09-30）

一个外部 harness 只登记一份，由这份登记派生两个侧面。

| 侧面 | 回答的问题 | 运行时去向 | 使用方 |
|:---|:---|:---|:---|
| 接入侧面 | 该 harness 以哪个调用来源（principal）接入、经哪种操作适配器（3.4）、可以声称哪些用户身份 | System 接入登记（`SystemActorAccessEntry`，2.1） | 认证网关第一阶段；harness 回调能力层时的认证 |
| 执行侧面 | 任务进程如何驱动该 harness 执行（驱动类型及其配置） | 执行单元：CPU 端口的一个常驻实现，由驱动代码配出 | 任务进程的 Actor 执行阶段 |

- 运行时仍是两个注册表，按方向分开：入站认证属于 System，出站执行属于任务进程。System 装载登记后分别构造两者并注入使用方，与总 Idea D-5 的做法一致。
- 由此，任务进程调用 CPU 采用对象端口：端口由 workspace 定义，CPU 实现，组合根注入，不采用总线路由契约（[子系统公共契约](../contracts/subsystem-contracts.md)第 4.1 节）。
- 登记的来源与生效时点见[身份与访问体系 Idea](./identity-and-access-model.md#3-principal-登记的来源与生效时点e-1)第 3 节（E-1）；现状见 2.1。
- 并非每个参与者都有两个侧面（分析）：plugin 模式的 harness 与管理员直接通道只有接入侧面；Alice 两个侧面都有，执行侧面是内置的；测试 CPU 只有执行侧面。只有接入侧面的参与者不称 CPU（执行单元 Idea T-6）。
- 驱动与执行单元（分析）：驱动是代码，按协议而不是按 harness 编写，例如一个 ACP 驱动可以覆盖所有支持 ACP 的 harness；执行单元由登记数据配出，登记包括选用的驱动、驱动配置与绑定的 principal。能否不写代码就加入一个新 harness 取决于 E-2a：采用通用协议时，只需新增一条登记；按各家 SDK 或 CLI 适配时，每家至少需要一个驱动。入站方向的 `adapters` 字段已经是“代码定义种类、登记数据引用种类”的形态。
- actor 不固定对应某个 CPU：agent 与执行环境解耦（执行环境 Idea 前提 9、10），用户在对话开始前分别选择 agent 与执行环境，选择执行环境不是 workspace 的能力，倾向由系统层级控制、在 CPU 分配时决定 backend（E-6a）；选择的承载、默认值与可用范围见执行环境 Idea T-8。
- 仍待决：登记记录的结构，以及它与 Workspace Actor 访问登记的关系（E-1a，同在身份与访问体系 Idea 第 3 节）。

## 2. 现状（以代码为准）

plugin 模式已经实现。本节只列与本文开放问题相关的现状，完整事实以链接的当前文档为准。

### 2.1 接入登记

- 两类登记在启动时从配置装载，运行中可以经管理接口编辑并写回：System 接入登记来自 `configs/system_principals.yaml`（[`system/access/registry.py`](../../src/hivememory/system/access/registry.py)），Workspace Actor 访问登记来自 `configs/workspace_actors.yaml`（[`workspace/registry.py`](../../src/hivememory/workspace/registry.py)），字段与运行时编辑见[认证与访问登记](../workspace/authentication.md)。
- 外部来源由插件服务在为 harness 安装插件时登记；接入登记的管理接口只能列出与修改外部来源（启停、adapter、可声明的 Agent），进程内来源只读。

### 2.2 认证

- 统一认证网关执行两阶段认证，任务进程注册、Import Bus 与 MCP 入口都经它认证得到工作 actor；外部 principal 以各自的本地共享密钥证明身份（P-1c，[认证与访问登记](../workspace/authentication.md)）。
- HTTP 管理面以管理凭据防误用，前端以外的调用方没有凭据时被拒绝（[认证与访问登记](../workspace/authentication.md#1-两阶段认证)第 1 节）；防刻意随桌面客户端，见[身份与访问体系 Idea](./identity-and-access-model.md#http-管理入口的调用方证明)第 4 节。

### 2.3 外部入口

- 对话回流：`POST /api/v1/import/turns` 每次接收一整轮，认证证明来源与参与的 Agent 后，由系统以 root 授权执行 Gateway 路由与共用封口提交，应用完成后才确认；不读取记忆，也不返回记忆上下文（[Import Bus](../system/import-bus.md)）。
- 记忆访问：MCP 端点由 HiveMemory 以 HTTP 提供，有 `memory_search`、`memory_read`、`memory_write` 与 `memory_update`，每次工具调用认证一次；写入与更新先经 Import Bus 为这一轮确定话题（[MCP 记忆访问](../contracts/mcp.md)）。记忆访问路径上没有连接器，harness 按插件中的声明直接连接端点。
- 已接入的 harness：Claude Code、DeepSeek Harness、Codex 与 pi，见[接入 Claude Code](../help/claude-code-plugin.md)、[接入 DeepSeek Harness](../help/deepseek-harness-plugin.md)、[接入 Codex](../help/codex-plugin.md)与[接入 pi](../help/pi-plugin.md)。

### 2.4 能力层

- 能力层分为 root API、actor API 与任务进程的阶段调用三个调用面；actor API 的六项 actor 操作与 MTP 的六个动词一一对应，包括 `agent.call`（[能力层](../workspace/capability.md)）。
- 操作入口是 actor API 唯一的调用方：执行线程经线程提交，MCP 入口经请求级提交，后者只开放检索、读取、写入与更新。

## 3. 设计材料

### 3.1 三条使用路线

| 路线 | 调用方需要什么 | 必须保留的差异 |
|:---|:---|:---|
| Import Bus | 外部 bot/harness 自己对话，把已发生的交互交给系统（完整记录或流式事件） | 实时 turn 边界、buffer、flush、背压与重发 |
| 主动资源交互 | 外部 Actor 按需搜索/读取、提交写入或修订意图、观察结果 | 结构化资源引用、显式授权、提交与结果语义；无需先发一条 user 消息 |
| 历史对话导入 | 把已有消息与工具轨迹保真导入，再按策略形成记忆 | 批次、历史发生时间、来源、重复导入、分支/编辑与恢复，不重放实时副作用（与总 Idea Q-13 相关） |

Import Bus 不在 user turn 开始时返回记忆 context，取得记忆属于主动资源交互（总 Idea 前提 2.3）。plugin 模式使用 Import Bus 与主动资源交互两条路线，不建进程；controller 模式下，外部 harness 作为进程中的 CPU 经能力层进行主动资源交互。历史对话导入的范围见总 Idea Q-13。

三条路线可以共享身份映射、资源引用、来源模型和 Patchouli 领域能力，但不共享全部会话规则、状态机或成功条件；消息能够序列化并不能证明处理流程等价。

### 3.2 主动调用与自动交接是两个维度

本节主要针对 plugin 模式（1.1）；controller 模式下，交互记录由任务进程一侧处理（任务进程 Idea Q-14）。

使用路线与“Actor 主动工具调用 / 自动交接”是不同维度。search/read/write/update 可由工具触发；完整交互结束后提交记录自动触发，不交给模型自行决定是否调用保存工具。同一检索或提交操作不因触发方式不同而走不同的能力。按 3.4，操作适配器只归一化收到的操作请求，不自行发起操作，自动交接因此不属于操作适配器，由 plugin 模式的连接器随 Import Bus 承担。

只提供工具协议（例如 MCP）的接入，仍需另有完成回调、消息流或 flush 集成，才能声称支持自动交接。仅有零散消息或工具调用而没有可靠结束信号时，协议必须能表达“尚未完整”或显式 flush 的边界；取消、中断、idle/shutdown flush 不能伪装为完整成功的回复。

### 3.3 身份的五个概念

区分系统接入登记确认的 `CallerPrincipal`、外部软件/connector source、本次 `ActorIdentity`、Workspace 归属以及消息说话者：

- source、agent_id、user_id 字符串不能自行构成认证结果；同一 principal 可以服务多个 Actor，但不能接受未经接入规则确认的身份声明；
- 输入消息中“某用户说过”属于 provenance 声明，不自动成为已认证的 Actor 身份；
- 读取结果的使用权、提交意图的权限、管理删除权限分别判断；
- 一个仍有效的认证结果可以用于多个操作，各操作分别授权；失效或切换 Actor/Workspace 时重新认证，不能在 adapter 内替换身份字段。

### 3.4 操作适配器的定义与边界

**定义**（E-3a）：操作适配器（operation adapter）把某种协议中的一次操作请求，归一化为 workspace 统一能力层 API 的一次调用，并把结果与错误映射回该协议。

- 例：MTP READ 与 MCP 的 `memory_read` 都归一化为同一项 actor 操作；管理员操作同样遵循归一化原则，经 HTTP 映射到能力层。管理员与 agent 因身份与操作意图不同，映射到不同的 API 面（总 Idea 15.7），归一化的逻辑相同。
- 职责：协议翻译；一次操作映射到一个能力方法；把结果与错误映射回协议。
- 操作适配器不直接调用能力层，而是把协议请求解析为固定类型的操作请求，经 workspace 的单一操作入口调用能力层；请求与能力方法一一对应，对 actor 暴露的操作、操作请求、能力层 API 与 `WorkspaceOperation` 是同一套操作列表。HTTP 管理侧理论上也采用这一形态，暂不迁移。见[子系统公共契约](../contracts/subsystem-contracts.md)第 4.6 节。
- 能力层分为 actor API 与 root API。外部 Actor（含 plugin 模式的 harness）只能使用 actor API，其操作列表翻译为 MCP 工具时不含 CALL；管理员（唯一的 root 角色 `system`）经 HTTP 调用 root API。见[能力层](../workspace/capability.md#1-三个调用面)第 1 节。
- 与连接器的区别（分析）：两者分处网络两侧。
  - 连接器是运行在外部 harness 一侧的客户端程序：读取会话记录、切分轮次、维护游标与重发，再调用 HiveMemory 的 HTTP 接口；不导入 HiveMemory 包。Claude Code 的连接器见 [Import Bus](../system/import-bus.md#8-claude-code-连接器)第 8 节。
  - 操作适配器与接入登记中的 adapter 都在 HiveMemory 一侧：前者把协议中的一次操作翻译为能力层调用；后者是接入登记 `adapters` 字段中的标识，表示调用来源从哪个入口进入认证网关。

**不属于操作适配器的职责**：

- **身份**：操作适配器不产生身份。入口一侧的认证由认证网关完成；进程内的发起者由执行单元与进程补全（执行单元 Idea 前提 1、4），身份如何随调用到达授权点见执行单元 Idea T-1、T-4。
- **触发时机**：交互结束时自动提交记录（3.2）是自行发起操作，不是对收到的操作请求的归一化。controller 模式下交互记录由任务进程封口（[任务进程 Idea](./task-process-table-and-registration-entry.md) Q-14），plugin 模式由连接器随 Import Bus 承担。
- **任务请求**：HTTP 上的任务请求进入唯一注册入口、创建任务进程，不是能力层的操作，由请求入口承担（执行单元 Idea 前提 7）。

**边界的判据**：归一化的目标必须是能力层已有的方法。外部操作无法表达为一次能力调用时，扩展能力层的方法，而不是在操作适配器中实现领域语义；例如 MTP READ 的 pending、redirect 等结果需要由能力层提供（总 Idea 15.5）。操作适配器可以持有与协议相关的状态，例如绑定进程时会话到进程的对应关系；是否持有状态不是判据。

不采用以机制界定 adapter 的判据（“adapter 只做五件事：协议翻译、认证交接、一次操作调用一个能力方法、错误映射、决定触发时机”；“需要 resolver、cache、lease 或 registry，组合多个领域步骤，解释资源 policy，持有平面状态，即属于能力层”）：其中的认证交接与决定触发时机不是归一化；后一条以机制而不是目的界定 adapter，实际回答的是“什么代码属于能力层”。

约束：

- 外部 Actor 不为自己新造一套资源 API，同一资源操作与 Alice 使用同一能力；
- 管理能力由独立 operation 授权，主动 Actor 不能借管理接口获得 owner bypass；
- 同一领域规则不在 Import Bus、主动操作或 connector 中分别重写；首版不增加完全通用的 resource registry；
- “与 Alice 基本相同的资源能力”指相同资源操作遵守同样的授权与领域语义；不要求外部 Actor 采用 Alice 的 frame、CALL、PendingAtomRuntime、prompt history 或 MTP 语法，也不替代外部 harness 的执行 loop。

### 3.5 外部 Actor 需要的操作

外部 Actor 需要的最小能力集（候选；其中 Profile read 已定保留，见表下）：

| 能力 | 内容 | 限制 |
|:---|:---|:---|
| Memory search/read | 结构化引用、内容或编译视图、已有的版本与来源信息 | 使用 Actor 可见读取，不使用管理 bypass |
| Profile read | AgentProfileStore 中的 Agent Profile 定义与授权语义 | 读取不等于实例化 Agent、改变调用方身份或已应用 system prompt；客户端可以不采用定义（与总 Idea P-10 相关）；Profile 不承载权限，操作由访问登记的白名单控制，对所有 CPU 生效（总 Idea 15.4） |
| Attachment read | 解析已有授权 ref、READY 检查、必要的内容快照 | lease 在服务内管理，不把 lease 对象跨网络传递 |
| Write/update intent | 显式提交内容、目标、理由与来源引用 | Patchouli 决定生成、合并、修订或丢弃；不把日志自动当作写入命令 |
| Pending read/resolve | 登记后读取意图与状态，结算后解析原引用 | 当前事实见 [读取视图与写入意图](../workspace/read-view-and-write-intents.md)；记忆任务的观察是只供 `system` 的 root API，不自动授予内容读取权 |

**设计**：外部 Actor 保留读取 Agent Profile 的能力。Profile 移出中期库、成为 AgentProfileStore 中的文件之后（[MemoryLibrary 与存储层](../patchouli/memory-library.md#15-agent-profile人偶图纸)第 1.5 节），读取的对象是这份定义，而不再是一条记忆原子；它不经记忆检索返回，要有自己的读取方式。

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

外部会话 ID 不能直接当作 topic_id 或授权身份；外部客户端不必实例化内部 Python 类型，也不必构造 TurnRecord、Alice run 或 Patchouli 队列 envelope。外部会话与 Topic 的数据交接见[外部会话与 Topic 投影](./external-session-and-topic-projection.md)；外部摘要或 checkpoint 只作为带来源的报告，不接管 Topic 的 canonical summary 或 harness 的 prompt history。

### 3.9 历史导入的设计验证

用含用户更正、助手推测、工具结果、旧发生时间及分支标识的历史样例，检查来源、时间与关联模型能否无损表达，并记录与实时模型不兼容的字段和处理边界。不为每条历史 user 消息自动检索当前 context，不按到达时间把旧偏好当作最新偏好，不把整个批次混入当前活动 Topic，也不因能够解析消息 schema 就声称已经完成历史导入。

## 4. 待决问题

每个问题只列出选项及其影响，选项顺序不代表倾向；已决定的问题写成设计。

### v0.7.1 的步骤与各步的阻塞问题

已实现：plugin 模式接入 Claude Code 的两步，对话回流（Import Bus）与记忆访问（MCP），当前事实见 [Import Bus](../system/import-bus.md)与 [MCP 记忆访问](../contracts/mcp.md)；当时的范围、阻塞问题与候选设计在已归档的[对话回流计划](../archive/plans/v0.7.1-import-bus-redesign.md)与[记忆访问计划](../archive/plans/v0.7.1-memory-access-mcp.md)中。Claude Code 插件把 hooks 与 MCP server 打包为一个单元：对话回流由连接器经 hooks 在轮次结束时交给 Import Bus，记忆访问经 MCP 工具（E-7）。

### v0.7.2：plugin 模式开放 WRITE 与 UPDATE

**设计**：

- plugin 模式下 agent 对记忆的 WRITE 与 UPDATE 与主动链路共享同一记忆更新逻辑。
- 记忆能力属于 workspace 的 actor API，不为 MTP 独占。plugin 模式开放的是 actor API 中写入与更新这两项 actor 操作；MTP 的 WRITE、UPDATE 只是 Alice 对同一组操作的协议映射。
- WRITE 与 UPDATE 始终产生写入句柄（PendingAtom）并交还前台 actor，作为读写一致性的保障；记忆生成完成后由结算事件触发重定向。这条链路任何一个环节断开，都说明迁移没有到位，需要返工，而不是另行设计。
- 记忆编译的命名与文案不再按 MTP 编写。

**现状**：已实现，见 [MCP 记忆访问](../contracts/mcp.md)、[Import Bus](../system/import-bus.md#47-写入时确定话题)第 4.7 节与[读取视图与写入意图](../workspace/read-view-and-write-intents.md)；当时的范围与候选设计在[已归档的计划](../archive/plans/v0.7.2-plugin-write-update.md)中。

### controller 模式的步骤分析（退回候选方向）

首个外部驱动的纯文本闭环（候选）：

- 范围：选一个 harness，编写驱动并配出第一个外部执行单元。执行单元把输入清单中已经编译好的提示词（人设、检索记忆与附件材料、环境通知）连同用户消息交给 harness，把正文流式回传，结束时返回最终正文；停止转为对 harness 的取消。同时提供最小的执行环境选择：harness 登记的执行侧面从配置装载（1.2、E-1），组合根装配多个执行单元，用户在新建会话时选择执行环境。
- 验收方向：在 HiveMemory 前端选择该 harness 完成多轮对话，停止有效，交互封口后进入记忆生成；新会话中 prepare 检索到的记忆作为材料交给 harness，形成被动注入的跨会话闭环。
- 不包含：harness 主动调用记忆（操作适配器）、工具调用的事件呈现、Profile 应用结果的报告、harness 的子 agent、第二个 harness。
- 阻塞问题：本文 E-2a；[任务进程 Idea](./task-process-table-and-registration-entry.md#q-8-外部执行单元的进程) Q-8；[执行环境 Idea](./execution-environment.md) T-7、T-8、T-11。候选 harness 的程序化接口见[接口调研需求](../research/harness-interface-survey.md)与[资料快照](../research/harness-interface-survey-snapshot.md)（2026-10-20，模型实测待验证）。
- 不阻塞这一步（分析）：harness 不回调 HiveMemory，也不需要登记外部来源，E-2 主问题、E-3b、[身份与访问体系 Idea](./identity-and-access-model.md#p-1-经网络接入的执行者如何证明身份) P-1b 与 P-1c、[身份与授权](../workspace/identity-and-authorization.md#8-已知限制)第 8 节的 `system` 声明规则都不涉及；只回传正文与终态字段时，T-10 的三个选项行为相同。
- 后续步骤（候选）：harness 经绑定线程的操作适配器主动使用记忆，需要 E-3b 中绑定线程的适配器（T-4）与 P-1b（E-2 主问题已决定，`system` 声明规则已以 `allow_system_actor` 落实），T-10 需要在此之前决定；Profile 应用结果的报告（P-10a，与 T-7b 相互约束）与前端呈现；E-2a 若选通用协议，接入第二个 harness 验证驱动的通用性。Q-3b 只在外部协议直接发起任务时才需要；P-3 随执行基座考虑（暂不考虑，见 ROADMAP 第 5.4 节）。验收方向见 ROADMAP 第 5.6 节。

### E-1 接入登记的来源与生效时点

在[身份与访问体系 Idea](./identity-and-access-model.md#3-principal-登记的来源与生效时点e-1)第 3 节随身份与访问主线维护，含子问题 E-1a。

### E-2 运行时访问的传输承载

**设计**：多种传输各由操作适配器归一到同一能力层，这是建立统一 Workspace 能力层的核心理念；对支持 MCP 的 harness，记忆访问以 MCP 为传输。不采用的选项：只用 HTTP/JSON 版本化接口（其他协议由各自的 adapter 另行适配）；以 MCP 为主（自动交接需要另有完成回调或消息流，3.2）。与任务进程 Idea [Q-3b](./task-process-table-and-registration-entry.md#q-3b-唯一入口与传输入口)（唯一入口是否同时要求唯一的传输入口）相关。

**现状**：MCP 端点由 HiveMemory 以 HTTP 提供，对话回流使用 Import Bus 的 HTTP 接口（2.3）。

**子问题 E-2a controller 模式下驱动外部 harness 的方式**：上面是外部 harness 调用 HiveMemory 的传输；controller 模式还需要 HiveMemory 驱动外部 harness（1.1）。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 采用 ACP 一类的通用协议 | 一个 adapter 可以覆盖多个 harness；受限于各 harness 对该协议的支持程度 |
| B | 按 harness 分别使用其 SDK 或非交互 CLI | 每个 harness 需要一个 adapter；可使用各 harness 的专有能力 |
| C | 其他 | —— |

### E-3 adapter 的判据与代码位置

- **E-3a 判据**：以归一化定义操作适配器，判据是归一化的目标必须是能力层已有的方法（3.4）。
- **E-3b 代码位置**：MCP 的操作适配器放在与 `server` 同层的独立入口包 `mcp_server`；MTP 的操作适配器留在 `agent_runtime`。controller 模式下绑定线程的适配器（T-4）随该模式另行考虑。
  - 两者不放在一起的理由：
    - 分层：MTP 的动词由 Alice 的 agent loop 调用，放到 `server` 同层后 Alice 无法导入；MCP 适配器要经认证网关取得工作 actor、经门面取得操作入口，放不进子系统层。
    - 两者位于执行线程的两侧：MTP 适配器在执行线程内运行，只持有提交函数 `OperationSubmitter`，取不到线程句柄或工作 actor；MCP 适配器位于网络入口，没有线程，每次调用经网关认证一次（P-9c）。
    - MTP 的动词还承担执行单元的行为：`sys_*` 系统工具在本地执行，CALL 是 Alice 的编排，SEARCH 把结果编译为 Agent 可读的检索上下文。MCP 适配器不需要这些。
  - 能力层统一的是归一化的目标，不是适配器的位置。两个适配器共用的部分位于更低的层：同一套操作请求（`workspace.contracts`）与 actor API、中立的错误类型，以及 `engines.memory_compiler` 的记忆渲染。各适配器只各自实现协议两端：解析请求，把结果与错误映射回协议（MTP 一侧见 [`agent_runtime/mtp/error_mapping.py`](../../src/hivememory/agent_runtime/mtp/error_mapping.py)）。
  - 现状：`mcp_server` 只经门面取得操作入口与认证网关，不放进 `server`（能力层的三个调用面互不借用，server 在能力层中只调用 root API），也不放进 `workspace`（适配器属于传输层）；[子系统公共契约](../contracts/subsystem-contracts.md)第 4.6 节把 MTP 适配器与 MCP 入口统称为操作适配器。

### E-4 结果观察的方式

**已关闭**：问题的定义不准确，项目中不存在“结果观察”这一概念。WRITE 与 UPDATE 始终产生写入句柄并交还前台 actor，记忆生成完成后由结算事件触发重定向（第 4 节的 v0.7.2 一步）；按操作类型查询、统一的 receipt 定位、推送或回调这些选项不再讨论。

### E-5 外部执行单元的执行线程

**背景**：按执行单元 Idea 前提 3、4，执行单元认识其全部执行线程，并为每个线程的工具调用补上发起人。Alice 的线程（frame）在进程内，可以直接观察；主流外部 harness 大多自带子 agent，controller 模式下 HiveMemory 只能经驱动与回调通道（执行单元 Idea T-4）看到 harness 交出的信息。本问题只涉及 controller 模式，plugin 模式不建进程（1.1）。

**设计**：作为最坏打算，忽视外部 harness 的子线程派生行为。一个进程内的所有 actor 共享主线程的身份，外部执行单元发起的全部操作都记在主线程的 actor 名下。子问题 E-5a（子线程的可见性）因此取“外部执行单元一律只有主线程”，E-5b（线程标记跨网络）不再需要。

- **理由**：外部 harness 的子 agent 派生难以控制，至少在现阶段实现难度远大于收益。这也是坚持自行实现一套 agent 执行环境（Alice）的原因之一：自己的执行环境可以在派生时确定子线程的身份（执行单元 Idea T-1）。
- **依据**（分析，按 [ACP 仓库文档](https://github.com/agentclientprotocol/agent-client-protocol)核对，2026-10-06）：
  - 驱动认不出派生调用：ACP 的 `tool_call` 通知带 `toolCallId`、`title`、`kind`、`status` 等字段，不标识所调用的工具；标识工具的 `name` 字段仍是 [RFD 提案](https://github.com/agentclientprotocol/agent-client-protocol/blob/main/docs/rfds/tool-call-name.mdx)。该提案指出从 `title` 或 `rawInput` 推断工具“脆弱且依赖实现”，并且 ACP 不赋予工具名任何授权语义。判断“这是一次派生、目标是谁”只能按 harness 分别编写规则。
  - 来不及拦截：`tool_call` 通知是事后报告；只有 harness 对该工具发出 `session/request_permission`（协议中为 MAY）时，驱动才能在派生发生前介入。
  - 无法签发身份：harness 自己的子 agent 类型不是 HiveMemory 登记的 agent，没有 Agent Profile，也没有访问登记，无法通过 Workspace 准入（E-6a）。
- **影响**（分析）：
  - 外部 harness 子 agent 发起的读取与写入意图，以主线程 actor 的权限授权，来源也记为主线程的 actor。Alice 则不同：每个线程持有自己的工作 actor，CALL 子线程以被调用方的身份运行（[执行线程](../workspace/execution-threads.md#63-子线程以被调用方身份运行)）。
  - 在 HiveMemory 看来，外部执行单元只有主线程一个 actor；执行单元 Idea T-1a、T-1b、T-2a 只涉及 Alice。
  - 外部执行单元不经过派生授权（执行单元 Idea T-3）；子 agent 能做什么只受主线程 actor 的白名单与资源 policy 约束。
  - 失效条件：出现能在协议中可靠标识派生调用、并允许子 agent 使用独立端点的 harness 或协议版本，且接入它的收益足够时，重新讨论。
- **接入外部 harness 时再考虑**：Alice 的子线程随 CALL 返回结束；外部 harness 既无法探测子 agent 的结束时点，也不一定能取得创建时点，子线程可能根本不会创建，这一点与停止时它在收到取消通知前的行为一起考虑，见已归档的[执行线程 Idea](../archive/ideas/execution-thread-definition.md)第 3.1、3.3 节；Alice 一侧的现状见 [Alice 编排](../alice/orchestration.md#8-失败取消与降级)第 8 节。

与[任务进程 Idea](./task-process-table-and-registration-entry.md#q-8-外部执行单元的进程) Q-8（外部 CPU 的进程，尤其 Q-8c 的粒度）相关。

### E-6 actor 与执行单元的对应

按 E-5，外部执行单元只有主线程；本问题只涉及外部执行单元的主线程以哪个 actor 运行，以及 Alice 的 CALL 目标能否落在外部执行单元。按执行单元 Idea 前提 3，actor 是执行线程，CPU 是执行单元。

- **E-6a 外部 harness 的定位**：harness 实例是一个执行单元，其中运行的 actor 由用户选择；agent 与执行环境分别由用户选择（执行环境 Idea 前提 9、10）。外部 harness 不是一个单独的 agent，而是一个执行环境；可以在其上应用 Agent Profile，外部协议是否支持另当别论。主线程以用户选择的 agent 运行。
  - 不采用：harness 本身作为一个 actor（有自己的 `agent_id` 与 Profile），由驱动构成执行单元。
  - 影响（分析）：Profile 的执行参数在外部 harness 上只能尽力应用，应用结果的承载见总 Idea P-10a；ACP 的 `session/new` 未见系统提示词字段，模型只能在 harness 声明的配置项中选择（执行环境 Idea 前提 9 的分析）。组合的有效性见执行环境 Idea T-7。
- **E-6b CALL 的目标位于另一个执行单元**：不允许。CALL 只能派生本执行单元能够运行的 actor，被调用方在父线程所在的执行环境中以自己的 Profile 运行。理由：agent 不再绑定执行环境，“CALL 的目标映射到另一个执行单元”不再自然发生；执行环境只由用户在对话开始前选择，CALL 无法为子线程选择另一执行环境。影响（分析）：一个任务进程只对应一个执行单元，CPU 分配仍是每个进程一次。

### E-7 plugin 模式下记忆到达 harness 的方式

**背景**：plugin 模式下 HiveMemory 不控制对话，记忆只能经 harness 的扩展点到达模型。Import Bus 只向网络输入、不返回记忆上下文。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 只由 harness 经 MCP 工具主动调用 | 是否检索由模型决定，可能该用的记忆没有被用上；与 [VISION](../VISION.md) 假设 H3（主动记忆交互优于纯预检索注入）的检验方式一致 |
| B | 连接器在用户提交消息时自动检索并注入上下文（经能力层，不经 Import Bus） | 每条消息都承担一次检索的延迟与成本，经 Gateway 分析时还多一次模型调用；受 harness hook 时限约束（Claude Code 的 `UserPromptSubmit` 默认 30 秒，超时后追加的上下文被丢弃），超时则本轮没有记忆 |
| C | 两者并存 | 需要避免同一轮重复检索与重复注入 |

**设计**：取选项 A。plugin 模式下不提供主动预检索，记忆只经 MCP 工具到达 harness，由 agent 自己选择使用。真实使用中该用而未用记忆的情况，作为 VISION 假设 H3 的检验样本（候选）。

**现状**：MCP 的工具说明与服务说明写明何时检索、结果是带日期的历史证据、仓库中的代码与文档以及 harness 自己的指令与记忆文件优先于检索结果；连接器把本插件的记忆工具调用写成能力调用（回声标记），生成视图对检索与读取只渲染一行摘要，取回的记忆不会随对话回流被再提取（[MCP 记忆访问](../contracts/mcp.md)第 3、6 节）。HiveMemory 不读取、不写入 harness 的记忆文件，两边冲突时按上述优先级处理。

**用户与偏好的全局部分怎样到达 harness**：类型体系的用户一侧中，USER_PROFILE 与 Preference 的全局部分对所有 agent 生效，适合常驻（[v0.8 类型体系 Idea](./memory-type-system.md)第 7.3 节）。这一块是 harness 插件的问题，不是类型体系的问题：harness 也可能有自己的记忆，强制注入要看插件能接受多大程度的提示词干预；不能注入时，引导 agent 在对话开始时主动获取，用主动阅读绕开注入的麻烦。

- **设计**：先走主动读取，仍在选项 A 之内，不开新的例外。
- 现状：MCP 的 `memory_search` 只接受一个检索字符串（[MCP 入口](../../src/hivememory/mcp_server/server.py)），不能按类型列出记忆，agent 只能靠查询碰巧命中，主动读取现在还不可靠。注入的通道已经存在，并且不碰 harness 自己的记忆文件：Claude Code 插件有 SessionStart hook（现在只登记会话），Claude Code 允许 SessionStart hook 向上下文追加内容；Codex 插件也有 SessionStart hook，能否追加内容待核实；dsh 与 pi 插件在系统提示词中注入 MCP 的使用说明（[hooks 配置](../../integrations/hivememory/hooks/hooks.json)）。
- 候选设计：
  - 主动读取的前提：MCP 能按类型列出 USER_PROFILE 与 Preference 的全局部分（给检索加类型参数，或单独的读取入口，在计划中确定）。
  - 读取到的内容写明：与项目指令冲突时以项目指令为准，并向用户指出冲突。
  - 真实使用中 agent 常常不去读时，再考虑放行会话开始时追加一段固定内容的窄例外：它不是选项 B 那样每条消息都检索，没有逐条消息的延迟，也不调用 Gateway，注入内容记为 attachment 时也没有回声（UserPromptSubmit 已核实，SessionStart 待核实）；代价是占用常驻的上下文，要有预算与溢出提示。

### E-8 plugin 模式写入意图的话题归属

**问题**：Import Bus 在一轮交互完整结束后才做 Gateway 路由，交互进行中没有目标话题，而写入意图恰好在这段时间内提交；写入意图派发到 Patchouli 时必须携带话题（`MemoryIntent.topic_id` 必填，[意图契约](../../src/hivememory/patchouli/contracts/memory_intent.py)）。plugin 模式没有进程与线程，MCP 请求头是插件配置中的静态值，不携带外部会话或轮次标识。

**设计**：写入时就地确定话题（选项 D）。连接器以 `PreToolUse` hook 为 HiveMemory 的写入与更新工具补上外部会话与本轮的标识；本轮第一次写入时，以轮次结束时相同的输入对这一轮做话题路由并登记得到的话题，写入以该话题派发；轮次结束回流时，Import Bus 沿用登记的话题，不再路由。写入的话题因此就是本轮最终所在的话题，与主动链路中派发话题同义，生成材料同样不含本轮。现状见 [Import Bus](../system/import-bus.md#47-写入时确定话题)第 4.7 节与 [MCP 记忆访问](../contracts/mcp.md)第 6 节。

不采用的选项：

| 选项 | 内容 | 不采用的原因 |
|:---|:---|:---|
| A | 不绑定话题，只以提交内容生成 | 影响最小，但也不完全合理：材料没有过去几轮的背景，显式写入的记忆不能回查到对话证据 |
| B | 绑定外部会话最近一轮所在的话题 | 不稳妥：最近一轮所在的话题不一定是本轮所在的话题，会话的第一轮没有可绑定的话题 |
| C | 推迟到本轮回流之后再派发 | 不考虑：ACK 不再表示已被生成队列接纳，本轮始终没有回流时意图需要一个终态 |
| E | 用户消息时点确定话题 | 每一轮都提前路由；同步 hook 让每条用户消息先等一次模型调用，异步 hook 下写入可能早于路由完成 |
| F | 按写入内容路由 | 每次写入多一次模型调用；选出的话题可能不是本轮所在的话题，来源 Artifact 会取自另一段对话 |

### P-1c 与 P-9c 身份证明与不建进程的访问 context

两个问题位于[第五代架构 Idea](./fifth-generation-architecture.md) 第 5 节，都不易回退：P-1c 决定外部调用方的身份由谁证明，P-9c 决定 actor API 是否接受进程之外的访问。

#### P-1c 身份证明的实现位置

**设计**：由统一认证网关证明。网关第一阶段验证 principal，第二阶段做 Workspace 准入；这是 Workspace 架构的基础，HTTP 入口、任务进程注册、Import Bus 与 MCP 入口都经网关认证。

- 一个 harness 对应一个 principal。principal 是接入的合法身份标识，不建进程的 plugin 模式下同样生效；adapter 标识同理，表示调用来源经哪个入口接入，外部入口不复用 `http`（现有 `mcp` 与 `import`）。对话回流与记忆访问使用同一个 harness 的 principal。
- 接入证据是每个外部 principal 一份本地共享密钥，由网关第一阶段核对，入口只携带证据；凭据只以环境变量名登记；登记逐个声明能否声明 `system`。
- 理由：沿用 server 的 principal 与 `http` adapter、不为 harness 另行登记的做法，与 1.2“一个外部 harness 只登记一份，接入侧面回答以哪个 principal、经哪种操作适配器接入”相悖，也与 Import Bus 接入时“输入来源需要作为 principal 登记”的决定相悖。
- 现状：见[认证与访问登记](../workspace/authentication.md)。

#### P-9c 不建进程的 actor 访问如何进入 actor API

**设计**：与 root 的直接通道相同，不建进程的 actor 访问只有操作、没有进程。入口经统一认证网关认证得到工作 actor，经操作入口的请求级提交进入 actor API，actor 操作只以 `authorize_agent` 授权，目标取驻留 Workspace（请求不能切换 Workspace），白名单逐次授权。actor API 只经操作入口，门面不交出 actor API：MCP 绕过操作入口会让 actor API 有两个出口，违背 3.4 的单一操作入口。

- 现状：见[认证与访问登记](../workspace/authentication.md#3-工作-actor)与[能力层](../workspace/capability.md#6-操作入口)。

## 5. 相关问题（位于其他文档）

| 问题 | 位置 | 关系 |
|:---|:---|:---|
| P-1 经网络接入的 Actor 每次请求如何证明身份 | [身份与访问体系 Idea](./identity-and-access-model.md#p-1-经网络接入的执行者如何证明身份) P-1 | 运行时访问的身份证明；P-1a 已决定：注册前经认证网关验证身份，此后每次请求重新校验（总 Idea 15.2）；P-1c 已决定，见第 4 节“P-1c 与 P-9c” |
| P-3 能力层是否覆盖执行类操作 | [第五代架构 Idea](./fifth-generation-architecture.md) P-3 | 外部 Actor 可调用的操作范围（3.5） |
| P-9c 不建进程的 actor 访问如何进入 actor API | 同上 | plugin 模式的访问不建进程（1.1）；已决定，见第 4 节“P-1c 与 P-9c” |
| P-10 Agent Profile 的能力描述是否与 MTP 解耦 | 同上 | 外部 Actor 如何理解 Profile；已决定：权限由能力层的操作控制（总 Idea 15.4），P-10a 仍待决 |
| Q-8 外部 CPU 的进程 | [任务进程 Idea](./task-process-table-and-registration-entry.md#q-8-外部执行单元的进程) | 外部 Actor 的调用与进程的关系、回收与取消；只涉及 controller 模式（1.1） |
| Import Bus 后续方向 | [已归档的 Import Bus Idea](../archive/ideas/import-bus.md)第 2 节 | 流式监听、历史导入另设通道；完整交互记录的接收已实现，各方向的去向见其归档说明 |
| T-1–T-6 执行单元与执行线程 | [执行线程](../workspace/execution-threads.md) | T-1（子线程身份在派生时确定）、T-2（每个线程一份身份）与 T-6（只有接入侧面的参与者不称 CPU）已决定；外部执行单元不识别子线程，见 E-5；回调通道与操作适配器的绑定（T-4）关系到 E-3b；T-1a、T-1b、T-2a 只涉及 Alice |
