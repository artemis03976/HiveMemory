---
title: 外部 Actor 的接入登记与运行时访问
status: idea
horizon: current
serves_version: v0.7.0
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
last_reviewed: 2026-09-27
---

# 外部 Actor 的接入登记与运行时访问

**文档状态**：Idea，未形成实施承诺
**记录日期**：2026-09-27，由原 v0.7.0 计划 B 退回

## 0. 文档性质

本文由原 v0.7.0 计划 B（外部记忆服务与 Actor 交互契约）于 2026-09-27 退回 Idea。计划的最后版本见 commit `dda9d9d` 中的 `docs/plans/v0.7.0-external-memory-service-and-actor-interaction.md`。

- owner 判定计划 B 作为计划已经作废：Passive 的设计已经更新（[总 Idea](./workspace-network-task-process-architecture.md)前提 2.3：Passive 退化为与记忆系统零主动交互的纯输入），计划中的内容也大多过时；但它要解决的问题仍然存在。
- 本文只保留这个问题以及仍然成立的设计材料。没有保留的内容：阶段划分（EMS-0–6）、交付与验收出口、A/B 交接与依赖顺序、测试矩阵、代码落点与迁移切片、文档收口清单；以 Passive Ingress 返回记忆 context 为前提的实时被动模式；以“System 应用服务 → GlobalSystemBus → Patchouli application”为主干的目标结构（已被总 Idea 第 12 节“能力层是 Actor 唯一可见的 API”这一前提取代）。
- 现状事实按 2026-09-27 的代码核对。
- 待决问题只列出选项及其影响，不替 owner 作出选择；选项顺序不代表倾向。

## 1. 问题

owner 表述（2026-09-27）：计划 B 的核心议题是**如何把外部 Actor 的信息注册到系统里，以及外部 Actor 在运行时如何访问系统**，属于 adapter 的接口设计。

它包含三件事：

1. **接入登记**：外部 Actor 及其背后的调用来源以什么形式、由谁、在什么时点登记进系统；
2. **运行时访问**：已登记的外部 Actor 每次调用时经什么传输、如何证明身份、可以调用哪些操作、如何观察结果；
3. **adapter 接口**：协议适配层与能力层之间的边界，adapter 可以做什么、不可以做什么。

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
- `PassiveIngressService.ingest_event` 在 user 事件上返回编译后的记忆 context，与新的 Passive 前提不一致（总 Idea 3.3）。

### 2.4 能力层

- `workspace/capability/` 下有 memory、agent_profiles、topic、memory_tasks、assets 五类能力服务。读取方法在 backing 调用前执行 `authorize_operation`；写入与管理路径的 operation 检查仍在 Patchouli application，见 [A1 访问边界返工](../todo/a1-access-boundary-rework.md)。workspace 包的现有实现需要重新调查（总 Idea 第 6.1 节）。
- `WorkspaceOperation` 共 11 项，没有代码执行、工具调用、CALL 或创建任务类操作（总 Idea 13.2）。

## 3. 仍然成立的设计材料（来自原计划 B）

### 3.1 三条使用路线

| 路线 | 调用方需要什么 | 必须保留的差异 |
|:---|:---|:---|
| 被动输入 | 外部 bot/harness 自己对话，把已发生的交互交给系统（完整记录或流式事件） | 实时 turn 边界、buffer、flush、背压与重发 |
| 主动资源交互 | 外部 Actor 按需搜索/读取、提交写入或修订意图、观察结果 | 结构化资源引用、显式授权、提交与结果语义；无需先发一条 user 消息 |
| 历史对话导入 | 把已有消息与工具轨迹保真导入，再按策略形成记忆 | 批次、历史发生时间、来源、重复导入、分支/编辑与恢复，不重放实时副作用（与总 Idea Q-13 相关） |

原计划中的被动路线还包括“在 user turn 开始时取得可选记忆 context”；按总 Idea 前提 2.3，这一点已不成立，取得记忆属于主动资源交互。

三条路线可以共享身份映射、资源引用、来源模型和 Patchouli 领域能力，但不共享全部会话规则、状态机或成功条件；消息能够序列化并不能证明处理流程等价。

### 3.2 主动调用与自动交接是两个维度

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
- 同一领域规则不在被动输入、主动操作或 connector 中分别重写；首版不增加完全通用的 resource registry；
- “与 Alice 基本相同的资源能力”指相同资源操作遵守同样的授权与领域语义；不要求外部 Actor 采用 Alice 的 frame、CALL、PendingAtomRuntime、prompt history 或 MTP 语法，也不替代外部 harness 的执行 loop。

### 3.5 外部 Actor 需要的操作

原计划 B 的最小能力集（去掉已过时的被动输入一行）：

| 能力 | 内容 | 限制 |
|:---|:---|:---|
| Memory search/read | 结构化引用、内容或编译视图、已有的版本与来源信息 | 使用 Actor 可见读取，不使用管理 bypass |
| Profile read | Agent Profile 定义与授权语义 | 读取不等于实例化 Agent、改变调用方身份或已应用 system prompt；客户端可以不采用定义（与总 Idea P-10 相关） |
| Attachment read | 解析已有授权 ref、READY 检查、必要的内容快照 | lease 在服务内管理，不把 lease 对象跨网络传递 |
| Write/update intent | 显式提交内容、目标、理由与来源引用 | Patchouli 决定生成、合并、修订或丢弃；不把日志自动当作写入命令 |
| Pending read/resolve | 登记后读取意图与状态，结算后解析原引用 | 见[写入意图体系迁移](./pending-intent-migration.md)；`task.observe` 不自动授予内容读取权 |
| Result query | 提交类型及真实领域结果或任务投影 | 查询状态不授予读取最终 Memory 的权限 |

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

### E-2 运行时访问的传输承载

**背景**：现有外部入口只有 HTTP 的 ingest 与 flush（2.3）；原计划 B 的首版选择是 HTTP/JSON 版本化接口加参考客户端。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | HTTP/JSON 版本化接口 | 其他协议由各自的 adapter 另行适配 |
| B | 以 MCP 为主 | 自动交接需要另有完成回调或消息流（3.2） |
| C | 多种传输并存，各由 adapter 适配到同一能力层 | 各 adapter 需要共享 3.6 的语义 |
| D | 其他 | —— |

与任务进程 Idea [Q-3b](./task-process-table-and-registration-entry.md#q-3-唯一注册入口的职责边界)（唯一入口是否同时要求唯一的传输入口）相关。

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

## 5. 相关问题（位于其他文档）

| 问题 | 位置 | 关系 |
|:---|:---|:---|
| P-1 经网络接入的 Actor 每次请求如何证明身份 | [总 Idea](./workspace-network-task-process-architecture.md#p-1-经网络接入的-actor每次请求如何证明身份)第三部分 | 运行时访问的身份证明 |
| P-3 能力层是否覆盖执行类操作 | 同上 | 外部 Actor 可调用的操作范围（3.5） |
| P-9a 直接通道的适用范围 | 同上 | 外部 Actor 的哪些请求不建进程 |
| P-10 Agent Profile 的能力描述是否与 MTP 解耦 | 同上 | 外部 Actor 如何理解 Profile |
| Q-8 外部 CPU 的进程 | [任务进程 Idea](./task-process-table-and-registration-entry.md#q-8-外部-cpu-的进程) | 外部 Actor 的调用与进程的关系、回收与取消 |
| Q-11–Q-13 被动输入 | [总 Idea](./workspace-network-task-process-architecture.md#5-待决问题被动输入)第 5 节 | 被动输入路线的 Topic 落位、价值信号与历史导入 |
