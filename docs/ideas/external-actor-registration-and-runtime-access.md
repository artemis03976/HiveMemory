---
title: 外部 Actor 的接入登记与运行时访问
status: idea
horizon: candidate
owner: project
scope: external-actor-registration-runtime-access-and-adapter-interface
code_paths:
  - src/hivememory/system/access/
  - src/hivememory/workspace/registry.py
  - src/hivememory/workspace/authentication.py
  - src/hivememory/workspace/capability/
  - src/hivememory/config/access.py
  - src/hivememory/server/deps.py
  - src/hivememory/server/routers/ingest.py
related_docs:
  - docs/ideas/workspace-network-task-process-architecture.md
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/external-session-and-topic-projection.md
  - docs/ideas/pending-intent-migration.md
  - docs/architecture/workspace.md
  - docs/VISION.md
last_reviewed: 2026-09-28
---

# 外部 Actor 的接入登记与运行时访问

**文档状态**：Idea，未形成实施承诺
**记录日期**：2026-09-27，由原 v0.7.0 计划 B 退回

## 0. 文档性质

本文由原 v0.7.0 计划 B（外部记忆服务与 Actor 交互契约）于 2026-09-27 退回 Idea。计划的最后版本见 commit `dda9d9d` 中的 `docs/plans/v0.7.0-external-memory-service-and-actor-interaction.md`。

- owner 判定计划 B 作为计划已经作废：原被动输入（总 Idea 现称 Import Bus）的设计已经更新（[总 Idea](./workspace-network-task-process-architecture.md)前提 2.3：退化为与记忆系统零主动交互的纯输入），计划中的内容也大多过时；但它要解决的问题仍然存在。
- 本文只保留这个问题以及仍然成立的设计材料。没有保留的内容：阶段划分（EMS-0–6）、交付与验收出口、A/B 交接与依赖顺序、测试矩阵、代码落点与迁移切片、文档收口清单；以 Passive Ingress 返回记忆 context 为前提的实时被动模式；以“System 应用服务 → GlobalSystemBus → Patchouli application”为主干的目标结构（已被总 Idea 第 12 节“能力层是 Actor 唯一可见的 API”这一前提取代）。
- 现状事实按 2026-09-27 的代码核对。
- 待决问题只列出选项及其影响，不替 owner 作出选择；选项顺序不代表倾向。
- **版本归属**（owner，2026-09-27）：外部 Actor 的真实接入（真正的 adapter 接口与外部服务身份等）不在 v0.7.0。v0.7.0 不对外部 Actor 所需的基建作承诺，只验证 Alice 在新架构下跑通、各流程协作无误，使之后的 adapter 不需要再大改系统拓扑结构（总 Idea [6.1](./workspace-network-task-process-architecture.md#61-已决定事项)）。外部 Actor 分为两种接入模式（1.1）：先做 controller 模式，作为 v0.7.1 的首个真实外部 harness 接入；plugin 模式在 v0.7.1 之后的 v0.7.x 版本完善。本文的 horizon 相应为 `candidate`，目标窗口见 [ROADMAP](../ROADMAP.md) 第 4.2 与 4.4.2 节。

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

- **命名**：此前称为“被动 / 主动服务状态”，是历史遗留叫法，改称 plugin 模式与 controller 模式。两种模式与[任务进程 Idea](./task-process-table-and-registration-entry.md#11-请求方的分类owner2026-09-27) 1.1 的主动请求 / 被动请求是两个维度：controller 模式同时接受两类请求；plugin 模式下的对话不经注册入口。
- **与现有实现的关系**：plugin 模式很像一直以来的 Passive Ingress 模式；controller 模式本质上就是如今的 Alice。任务进程表与注册入口更偏向 controller 模式，这与现有 chat 链路服务 Alice 同源。
- **共同基建**：两种模式对进程的控制能力不同，但都使用 workspace 能力层。到能力层为止是真正的 workspace 架构基建，再往上两种模式各有各的样式。
- **版本安排**：v0.7.0 不能同时完成两者，先做 controller 模式，作为 v0.7.1 的首个真实外部 harness 接入；plugin 模式已有一定代码基础，在 v0.7.1 之后的 v0.7.x 版本完善，不在 v0.7.1 中完成。

两种模式在各层的对应如下（分析，不是已决定的设计）：

| 层面 | plugin 模式 | controller 模式 |
|:---|:---|:---|
| 能力层（资源操作与授权） | 共用 | 共用 |
| 任务进程与注册入口 | 不使用：外部 harness 的任务对 HiveMemory 不可见 | 使用：外部 harness 与 Alice 同为进程中的 CPU |
| 访问 context | 不绑定进程，形状与管理员直接通道相同（总 Idea [第 15 节](./workspace-network-task-process-architecture.md#15-第三部分已决定事项)、P-9） | 绑定进程 |
| 记忆域访问 | 外部 harness 经 MCP 调用能力层 | 外部 harness 同样可经 MCP 调用能力层；例如 ACP 在创建会话时允许客户端提供 MCP server，HiveMemory 可以提供绑定进程的端点 |
| 交互记录的回流 | 经 Import Bus（现有 Passive Ingress 链路；已排除在现有系统之外，随 plugin 模式另行设计） | 任务进程自行提交（[任务进程 Idea](./task-process-table-and-registration-entry.md#q-14-主动进程的交互记录去向) Q-14 选项 A） |
| HiveMemory 能否发起任务 | 不能 | 能；被动请求（定时任务、队列任务）只能在此模式下存在 |
| 对外部 harness 的控制 | 无 | 任务边界：actor 选择、输入、生命周期与取消、执行轨迹回流；harness 内部的 loop、工具与上下文压缩仍由其自身管理 |
| 现有代码 | Passive Ingress 与 `/api/v1/ingest`、能力层 | Alice 的 chat 链路与 chat run 注册表（内部 CPU） |
| [VISION](../VISION.md) 中的对应 | 兼容轨（第 8.1 节）；四级对照基线的第 3 组“外部 harness + Patchouli”（第 13.1 节） | 以 Alice 为内部 CPU 时即原生轨（第 8.2 节）；以外部 harness 为 CPU 的形态在 VISION 中没有对应 |

两种模式并存，对以下尚未决定的共用设计构成约束（分析，决定时一并考虑）：

- [任务进程 Idea](./task-process-table-and-registration-entry.md#q-2-写入意图中间产物的可见范围) Q-2：plugin 模式经 MCP 提交的写入意图没有所属进程（2026-09-28 已消解：写入意图在 workspace 登记，与进程解耦）；
- 任务进程 Idea Q-14：两条回流路径进入同一提交队列；同一 harness 同时以两种模式使用时，同一交互可能被记录两次（2026-09-28：Q-14 选 A，Import Bus 排除在现有系统之外，此问题留待 plugin 模式设计时处理）；
- [外部会话与 Topic 投影](./external-session-and-topic-projection.md)：plugin 模式的会话归外部 harness 所有，Alice 与 controller 模式的会话由 HiveMemory 发起；
- 总 Idea [P-9](./workspace-network-task-process-architecture.md#p-9-方案-c-的后续问题)：不建进程的访问不只来自管理员；
- Import Bus：同时是 plugin 模式的对话回流通道，不只承担对话导入。

此前列入“外部 Actor 形态”单独审议的问题，在两种模式下的归属：

| 问题 | 归属 |
|:---|:---|
| [Q-8 外部 CPU 的进程](./task-process-table-and-registration-entry.md#q-8-外部-cpu-的进程) | 只涉及 controller 模式（plugin 模式不建进程）；Q-8a–Q-8c 仍待决 |
| E-4 结果观察的方式 | 主要涉及 plugin 模式，controller 模式可经任务进程观察结果；仍待决 |
| [P-5b 被动请求的 principal](./workspace-network-task-process-architecture.md#p-5-call-与触发器的认证) | 被动请求只存在于 controller 模式；owner 倾向由登记的 Agent 反推；仍待决 |
| 请求方类型与认证 principal 的关系 | 按两种模式的定义：controller 模式下，用户直接向 HiveMemory 的入口发出请求；plugin 模式下，没有请求进入注册入口，外部 harness 以不建进程的方式访问能力层。外部 harness 连接方的身份如何证明见 P-1c |

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
- 认证网关没有生产调用方。principal 的身份证明应由 adapter 依据接入证据构造，目前没有 adapter 实现这一证明（总 Idea 13.1 与 P-1）。

### 2.3 HTTP 入口

- `/api/v1/ingest` 与 `/api/v1/ingest/flush` 是现有的外部对话事件入口：要求显式 `agent_id`（参与外部会话命名空间），body 中的身份与请求头身份冲突时拒绝。
- 其他资源路由的身份取自请求头 `x-user-id` / `x-workspace-id`，不经认证网关。
- `PassiveIngressService.ingest_event` 在 user 事件上返回编译后的记忆 context，与 Import Bus 的新前提不一致（总 Idea 2.3、3.3）。

### 2.4 能力层

- `workspace/capability/` 下有 memory、agent_profiles、topic、memory_tasks、assets 五类能力服务。读取方法在 backing 调用前执行 `authorize_operation`；写入与管理路径的 operation 检查仍在 Patchouli application，见 [A1 访问边界返工](../todo/a1-access-boundary-rework.md)。workspace 包的现有实现需要重新调查（总 Idea 第 6.1 节）。
- `WorkspaceOperation` 共 11 项，没有代码执行、工具调用、CALL 或创建任务类操作（总 Idea 13.2）。

## 3. 仍然成立的设计材料（来自原计划 B）

### 3.1 三条使用路线

| 路线 | 调用方需要什么 | 必须保留的差异 |
|:---|:---|:---|
| Import Bus（现有 Passive Ingress） | 外部 bot/harness 自己对话，把已发生的交互交给系统（完整记录或流式事件） | 实时 turn 边界、buffer、flush、背压与重发 |
| 主动资源交互 | 外部 Actor 按需搜索/读取、提交写入或修订意图、观察结果 | 结构化资源引用、显式授权、提交与结果语义；无需先发一条 user 消息 |
| 历史对话导入 | 把已有消息与工具轨迹保真导入，再按策略形成记忆 | 批次、历史发生时间、来源、重复导入、分支/编辑与恢复，不重放实时副作用（与总 Idea Q-13 相关） |

原计划中的 Import Bus 路线还包括“在 user turn 开始时取得可选记忆 context”；按总 Idea 前提 2.3，这一点已不成立，取得记忆属于主动资源交互。

按两种接入模式（1.1）：plugin 模式使用 Import Bus 与主动资源交互两条路线，不建进程；controller 模式下，外部 harness 作为进程中的 CPU 经能力层进行主动资源交互。历史对话导入的范围见总 Idea Q-13。

三条路线可以共享身份映射、资源引用、来源模型和 Patchouli 领域能力，但不共享全部会话规则、状态机或成功条件；消息能够序列化并不能证明处理流程等价。

### 3.2 主动调用与自动交接是两个维度

本节主要针对 plugin 模式（1.1）；controller 模式下，交互记录由任务进程一侧处理（任务进程 Idea Q-14）。

使用路线与“Actor 主动工具调用 / adapter 自动交接”是不同维度。search/read/write/update 可由工具触发；完整交互结束后提交记录由 adapter 自动触发，不交给模型自行决定是否调用保存工具。同一检索或提交操作不因触发方式不同而走不同的能力。

只提供工具协议（例如 MCP）的接入，仍需另有完成回调、消息流或 flush 集成，才能声称支持自动交接。仅有零散消息或工具调用而没有可靠结束信号时，协议必须能表达“尚未完整”或显式 flush 的边界；取消、中断、idle/shutdown flush 不能伪装为完整成功的回复。

### 3.3 身份的五个概念

区分系统接入登记确认的 `CallerPrincipal`、外部软件/connector source、本次 `ActorIdentity`、Workspace 归属以及消息说话者：

- source、agent_id、user_id 字符串不能自行构成认证结果；同一 principal 可以服务多个 Actor，但不能接受未经接入规则确认的身份声明；
- 输入消息中“某用户说过”属于 provenance 声明，不自动成为已认证的 Actor 身份；
- 读取结果的使用权、提交意图的权限、管理删除权限分别判断；
- 一个仍有效的 context 可以用于多个操作，各操作分别授权；失效或切换 Actor/Workspace 时重新认证，不能在 adapter 内替换 context 的身份字段。

### 3.4 adapter 的职责边界（候选判据）

以下判据来自原边界宪章 §5.3（2026-09-25 裁定；宪章已拆分，能力面入口的其余裁定见[总 Idea](./workspace-network-task-process-architecture.md) 7.1.6），在此作为候选收录：

- adapter 只做五件事：协议翻译；认证交接（经认证网关取得或复用 access context）；一次操作恰好调用一个能力方法；把错误映射为 wire 格式；决定触发时机；
- 出现以下任一情况即属于能力层代码而非 adapter：需要 resolver、cache、lease 或 registry；组合多个领域步骤；解释资源 policy；持有平面状态。

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
| Pending read/resolve | 登记后读取意图与状态，结算后解析原引用 | 见[写入意图体系迁移](./pending-intent-migration.md)；`task.observe` 不自动授予内容读取权 |
| Result query | 提交类型及真实领域结果或任务投影 | 查询状态不授予读取最终 Memory 的权限；Patchouli 的记忆任务不暴露给 Agent（[任务进程 Idea](./task-process-table-and-registration-entry.md#q-7-记忆库内部工作与进程表) Q-7，2026-09-27），涉及任务投影的部分需按此重新审视 |

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
| 写入意图 | 意图的权威持有者（见[写入意图体系迁移](./pending-intent-migration.md)） | 已登记内容可按权限读回，并可关联物化状态；不表示任务已接纳或 Memory 已生成 |
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

### E-1 接入登记的来源与生效时点

**背景**：两类登记只能在启动时从配置装载，运行中不可变（2.1）。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 维持启动时从配置装载 | 新增或撤销外部 Actor 需要修改配置并重启 |
| B | 提供运行时登记入口（例如经管理员直接通道，见总 Idea 第 15 节方案 C） | 需要定义登记、修改、撤销的授权与生效时点，以及已签发 context 的处理 |
| C | 登记数据来自外部身份系统 | 需要定义与外部系统的信任关系与同步方式 |
| D | 其他 | —— |

**子问题 E-1a**：System 接入登记与 Workspace Actor 访问登记是否经同一入口维护：同一入口 / 分开维护 / 其他。

**owner 决定（2026-09-27）**：接入登记维持启动时从配置装载（选项 A）。运行时登记与其他配置文件的热更新一并由未来单独的计划实现；E-1a 随该计划决定。

### E-2 运行时访问的传输承载

**背景**：现有外部入口只有 HTTP 的 ingest 与 flush（2.3）；原计划 B 的首版选择是 HTTP/JSON 版本化接口加参考客户端。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | HTTP/JSON 版本化接口 | 其他协议由各自的 adapter 另行适配 |
| B | 以 MCP 为主 | 自动交接需要另有完成回调或消息流（3.2） |
| C | 多种传输并存，各由 adapter 适配到同一能力层 | 各 adapter 需要共享 3.6 的语义 |
| D | 其他 | —— |

与任务进程 Idea [Q-3b](./task-process-table-and-registration-entry.md#q-3-唯一注册入口的职责边界)（唯一入口是否同时要求唯一的传输入口）相关。

**子问题 E-2a controller 模式下驱动外部 harness 的方式**：上述选项是外部 harness 调用 HiveMemory 的传输；controller 模式还需要 HiveMemory 驱动外部 harness（1.1）。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 采用 ACP 一类的通用协议 | 一个 adapter 可以覆盖多个 harness；受限于各 harness 对该协议的支持程度 |
| B | 按 harness 分别使用其 SDK 或非交互 CLI | 每个 harness 需要一个 adapter；可使用各 harness 的专有能力 |
| C | 其他 | —— |

### E-3 adapter 的判据与代码位置

- **E-3a 判据**：采用 3.4 的候选判据 / 其他划分。
- **E-3b 代码位置**：入口层（`server`） / workspace 包内 / 独立的 adapter 包 / 其他。位置决定 adapter 在包分层中的层级，以及它能导入哪些包（分层规则见[系统架构概览](../architecture/overview.md)）。

### E-4 结果观察的方式

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 按操作类型分别查询 | 每类操作各有查询入口与保留规则 |
| B | 统一的 receipt 定位，再按类型返回 | 需要一个只持有映射的关联记录，保留期与容量需要定义（3.7） |
| C | 推送或回调 | 外部方需要提供可达端点；推送不能作为唯一的终态真相 |
| D | 组合 | —— |

**owner 决定（2026-09-27）**：本问题与 Q-8 联动，列入外部 Actor 形态的单独审议。同日审议为两种接入模式（1.1）；本问题主要涉及 plugin 模式，仍待决。

## 5. 相关问题（位于其他文档）

| 问题 | 位置 | 关系 |
|:---|:---|:---|
| P-1 经网络接入的 Actor 每次请求如何证明身份 | [总 Idea](./workspace-network-task-process-architecture.md#p-1-经网络接入的-actor每次请求如何证明身份)第三部分 | 运行时访问的身份证明；P-1a 已决定：注册前经认证网关验证身份，此后每次请求重新校验（总 Idea 15.3） |
| P-3 能力层是否覆盖执行类操作 | 同上 | 外部 Actor 可调用的操作范围（3.5） |
| P-9a 直接通道的适用范围 | 同上 | 外部 Actor 的哪些请求不建进程；plugin 模式的访问不建进程（1.1） |
| P-10 Agent Profile 的能力描述是否与 MTP 解耦 | 同上 | 外部 Actor 如何理解 Profile；已决定：allow 字段并入能力层的 operation 控制（总 Idea 15.4），P-10a 仍待决 |
| Q-8 外部 CPU 的进程 | [任务进程 Idea](./task-process-table-and-registration-entry.md#q-8-外部-cpu-的进程) | 外部 Actor 的调用与进程的关系、回收与取消；只涉及 controller 模式（1.1） |
| Q-11–Q-13 Import Bus | [总 Idea](./workspace-network-task-process-architecture.md#5-待决问题import-bus)第 5 节 | Import Bus 路线的 Topic 落位、价值信号与历史导入；不在 v0.7.0 范围 |
