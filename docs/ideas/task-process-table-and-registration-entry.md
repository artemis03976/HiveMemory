---
title: 任务进程表与任务请求唯一注册入口
status: idea
horizon: current
serves_version: v0.7.0
owner: project
scope: task-process-table-unique-registration-entry-and-process-lifecycle
code_paths:
  - src/hivememory/workspace/process/
  - src/hivememory/alice/runtime/core.py
  - src/hivememory/agent_runtime/runtime.py
  - src/hivememory/agent_runtime/pending_atom/runtime.py
related_docs:
  - docs/ideas/workspace-network-task-process-architecture.md
  - docs/ideas/ae2-hivememory-architecture-analogy.md
  - docs/ideas/chat-run-lifecycle-follow-ups.md
  - docs/ideas/external-session-and-topic-projection.md
  - docs/ideas/pending-intent-migration.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/ideas/identity-and-access-model.md
last_reviewed: 2026-10-04
---

# 任务进程表与任务请求唯一注册入口

**文档状态**：Idea，未形成实施承诺；本方向在 v0.7.0 内的批次已全部实施
**记录日期**：2026-09-27，内容自 [Workspace 网络与任务进程架构](./workspace-network-task-process-architecture.md)第一部分拆出；2026-10-04 按“已完成 / 未完成”重新整理

## 0. 文档性质

owner 于 2026-09-27 将“任务进程表与任务请求唯一注册入口”定为 v0.7.0 的首个计划方向，本文是它的集中讨论载体，本身不是 Plan。v0.7.0 的范围（M-5）、迁移方式（M-1）与首条迁移流程（M-3，Alice 的 chat 链路）见总 Idea [第 5 节](./workspace-network-task-process-architecture.md#5-第一部分已完成的问题)。

- **来源**：内容自总 Idea 第一部分拆出（原前提 2.2、现状事实 3.1/3.2/3.5、流程图 4.2–4.4 与问题 Q-1–Q-10、Q-14），问题编号沿用原编号。全局拓扑、Import Bus（Q-11–Q-13）、迁移问题（M-1–M-7）与认证授权（第三部分）仍在总 Idea，关联见第 5 节。
- **阅读方式**：
  - 第 1 节是前提，以及 owner 已定的请求方分类（1.1）与任务进程结构（1.2），各部分的实施状态写在 1.2 开头；
  - 第 2 节是迁移前的代码快照，只作为问题的背景；
  - 第 3 节流程图画的是 1.2 决定的目标形态，其中尚未实施的部分已标出；
  - 第 4 节是问题：4.1 是已完成的问题，按“问题—实际设计或实现”叙述，注明决定日期与实施状态；4.2 是未完成的问题，只列选项及其影响，选项顺序不代表倾向。
- **2026-10-04 的整理**：已决定问题的选项表与逐次追加的决定已改写为最终设计；部分决定、部分未决的问题（Q-3、Q-5、Q-7）拆成两半，已决定的部分留在 4.1，剩余部分移入 4.2。整理前的最后版本见 commit `2daa332`。

**实施进度**：

| 批次 | 内容 | 依据 |
|:---|:---|:---|
| 第一批（2026-09-28） | 进程表与 chat 四阶段骨架迁入 `workspace.process`；取消只在 Gateway 与 Actor 执行阶段响应（Q-15）；`process_id` 作为唯一进程标识（Q-16） | [归档计划](../archive/plans/v0.7.0-task-process-table.md) |
| 第二批（2026-09-29） | Patchouli prepare 只做 Topic 与检索；进程完成 CPU 分配（Profile 解析、附件租借与编译、记忆编译、`CPUInputManifest`） | [归档计划](../archive/plans/v0.7.0-task-process-prepare-split.md) |
| 第三批（2026-09-30） | 进程封口交互记录 `InteractionPayload`；finalize 不再接收 `AgentRunResult` | [归档计划](../archive/plans/v0.7.0-task-process-finalize-neutral-input.md) |
| 第四批（2026-10-01） | CPU 端口 `CPUPort` 与中立执行结果；Alice 统一流式与非流式入口并实现端口；测试 CPU 跑通任务进程 | [归档计划](../archive/plans/v0.7.0-task-process-cpu-port.md) |
| 第五批（2026-10-01） | 命令只解析不执行，内置命令暂时不可用 | [归档计划](../archive/plans/v0.7.0-task-process-command-parse-only.md) |
| A1 访问边界返工（2026-10-04） | 注册入口完成两阶段认证、先注册后运行；进程记录持有访问 context；进程表登记任务进程；不透明的进程句柄与唯一的取消方法 | [归档计划](../archive/plans/v0.7.0-a1-access-boundary-rework.md) |
| TaskProcess 容器（2026-10-04） | 四阶段骨架拆为所有进程共用的执行器 `TaskProcessRunner`，`TaskProcess` 只作状态容器；注册入口只持有生命周期依赖 | [归档的 Todo](../archive/todo/task-process-container-ownership.md) |

当前事实见 [System 应用服务](../system/application-services.md)第 3、4 节、[子系统公共契约](../contracts/subsystem-contracts.md)与 [Gateway 全局命令](../gateway/commands.md)。1.2 中尚未实施的部分归其他方向：Topic 按需创建与 cleanup 路由的移除归[外部会话与 Topic 投影](./external-session-and-topic-projection.md)，写入意图的实时提交归[写入意图迁移](./pending-intent-migration.md)，Profile 权限并入 operation 控制归 Alice 的能力层调用迁移（总 Idea 15.4、15.5）。

## 1. 前提（owner 提出）

类比映射见总 Idea [第 2.1 节](./workspace-network-task-process-architecture.md#21-类比映射)：Workspace 对应 ME 网络，Patchouli 对应存储系统，任意 Actor 对应合成 CPU，一次任务请求对应一个任务进程。

1. 运行一个任务，需要为它保留一个“合成进程”；CPU 在进程内工作。
2. CPU 必须在 Workspace 的一个集中区域里工作，但这件事不由单一的 runtime 环境承担。原先把 Workspace runtime 当作 CPU 真实工作区的观念是半对半错。
3. 任意任务请求从**唯一入口**注册为一个进程，直到任务结束进程才关闭。这是新架构下 Workspace 网络的核心运作逻辑。
4. 请求方不只有用户主动下单；满足指定条件的被动请求方（类比合成卡、ME 请求器）也可以向网络创建任务。两类请求方的定义与区分标准见 1.1。
5. 现有实现中与此最接近的是 chat application service 的 run 注册表：用户发出指令后，注册一个独立的 chat generation run。

### 1.1 请求方的分类（owner，2026-09-27）

请求方只有两类：

| 类别 | 定义 | 例子 |
|:---|:---|:---|
| 主动请求 | 用户发出的指令主动且即时地驱动接下来的任务进程 | 经 Alice 的 chat 链路；未来以 controller 模式接入的外部 harness |
| 被动请求（Passive） | 由条件触发；用户设定之后不再主动发出请求 | 最常见的是定时任务与队列任务 |

- **区分标准是“即时”**：进程在用户指令到达的那一刻创建，即为主动请求；否则为被动请求。队列任务虽由用户提交，但进程要等条件满足时才创建，因此属于被动请求。
- **术语**：“被动（Passive）”一词保留给被动请求体系。现有的 Passive Ingress 链路不是请求方，总 Idea 称其为 Import Bus；它已排除在现有系统之外（总 Idea [5.8](./workspace-network-task-process-architecture.md#58-import-bus-移出现有系统)）。
- 请求方类别与执行者（CPU）是两个维度：controller 模式下，请求方是在 HiveMemory 入口发出指令的用户，外部 harness 是 CPU；plugin 模式下的对话不经注册入口（[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#11-两种接入模式owner2026-09-27) 1.1）。
- 现状：注册入口只接收主动请求（Alice 的 chat 链路）；被动请求这一阶段不考虑（Q-6）。

以下对象不是请求方：

| 对象 | 定位 |
|:---|:---|
| CALL 派生的子执行单元 | 在父 Agent 的任务进程内执行，不能请求新进程（Q-10） |
| 管理员直接通道 | 不是请求；性质更接近 CPU，直接执行 operation，不经注册入口、不建进程（总 Idea 15.1） |
| Patchouli 的记忆任务 | 给后台系统的任务，不暴露给 Agent（Q-7） |
| plugin 模式下的外部 harness | 对话由外部 harness 管理，不经注册入口；以不建进程的方式经能力层访问（外部 Actor Idea 1.1） |

被动请求的 principal 由谁承担仍待决（总 Idea P-5b；owner 倾向由登记的 Agent 反推）；被动请求只存在于 controller 模式。

### 1.2 任务进程的结构（owner，2026-09-28）

本节适用于 controller 模式；plugin 模式不建进程（[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#11-两种接入模式owner2026-09-27) 1.1）。

**实施状态**（2026-10-04 核对）：

| 部分 | 状态 |
|:---|:---|
| 创建时机与入口 | 已实施：注册入口完成两阶段认证后立即创建进程，入口只管理生命周期（A1 返工、TaskProcess 容器） |
| 四阶段通用骨架 | 已实施：所有进程共用的执行器 `TaskProcessRunner`（第一批、TaskProcess 容器） |
| CPU 分配、输入清单与 CPU 端口 | 已实施（第二、四批）；CPU 的选择机制与 actor 对应 CPU 的映射后置 |
| 进程记录与工作集 | 部分实施：进程记录与工作集的现有内容见下文“进程记录与工作集”；请求方式与 CPU 分配未记入进程记录，GatewayDecision 与执行记录未成为工作集槽位 |
| Patchouli prepare 与结算的拆分 | prepare 退化、进程编译、附件、Profile 解析、交互记录由进程组装已实施（第二、三批）；Topic 按需创建与 Profile 解析回到 prepare 之后未实施（外部会话方向）；Profile 权限并入 operation 控制未实施（Alice 的能力层调用迁移） |
| v0.7.0 版本目标 | 第 1、2、4 条已达成；第 3 条只剩 Patchouli cleanup 路由，随 Topic 按需创建移除 |

**创建时机与入口**

- 入口只管理任务进程的生命周期（Q-3）。
- 进程在最开始创建：完成两阶段认证后立即创建。此后 Gateway 的分析、记忆上下文等都由进程携带。这样取消在进程容器上响应，完整覆盖后续流程；中间产物与认证信息也都有携带者。
- 不存在“任务类型”。主动请求与被动请求只是触发方式不同，在入口注册这一步就收敛到一起，对入口没有影响。

**四阶段通用骨架**

- controller 模式下，所有任务进程共用四个阶段：Gateway 分析 → Patchouli 预检索 → Actor 执行 → Patchouli 结算。这一划分不受全局拓扑影响（Q-4）。
- 设置阶段最初主要是为了取消：四个阶段相对独立，取消需要各自处理，不同阶段的策略也不同（迁移前现状见 2.4）。阶段划分同时反映了 chat 链路的实际结构。
- 由此：Gateway 是每个进程的第一阶段，在进程内执行（Q-5）；命令解析在 Gateway 内进行，命令的实际运行不在 Gateway 中（Q-5a）。
- 只有 Gateway 与 Actor 执行两个阶段可以取消；进入 Actor 执行前统一检查一次取消请求（Q-15）。
- 失效条件（分析）：出现装不进四阶段的任务，例如 v0.7.4 Deep Research 需要多轮检索与执行、长期保留研究状态，或被动请求保存的指令无法作为用户消息交给 Gateway 分析。届时需要重新讨论任务类型。

**CPU 分配与输入清单**

- 进入 Actor 执行阶段前完成 CPU 分配，并同步记录到进程中。
- 前面阶段的产出作为面向 CPU 的输入清单，让任意 Actor 都能接收。
- **CPU 端口**（owner，2026-09-30）：进程经对象端口调用 CPU：端口由 workspace 定义（`workspace.contracts`），CPU 实现，组合根注入；不采用总线路由契约。依据是外部 harness 的一份登记派生接入与执行两个侧面（[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#12-harness-登记的两个侧面owner2026-09-30) 1.2）：外部 harness 的驱动多数不是子系统，按路由契约接入需要为每种驱动增加路由常量，或在总线后面再建一层分派。v0.7.0 只需要 Alice 接入；CPU 的选择机制与 actor 对应 CPU 的映射后续设计。
  - **CPU 端口的输出**（owner，2026-10-01）：CPU 的事件流把终态结果与其余交互事件分开，交互事件原样透传；执行结果去掉 `mtp_iterations` 与 `total_iterations` 两个冗余字段。
  - **Alice 的统一入口**（owner，2026-10-01）：Alice 的流式与非流式两条执行路径合并为一个入口，与注册入口的 `run_process` 一样以参数控制是否流式。

**进程记录与工作集**

任务进程是运行时的状态容器，容纳任务状态与中间产物。它分为两部分，按“进程关闭时有没有义务”划分：

| 部分 | 内容 | 性质 |
|:---|:---|:---|
| 进程记录（控制面，经进程取得） | 进程标识 `process_id`（Q-16）、访问 context、请求方式（主动或被动）、当前阶段、待处理的取消请求、终态、CPU 分配 | 字段固定，有类型 |
| 工作集中的值 | GatewayDecision（含 Topic 路由决定）、预检索得到的记忆原子、执行记录 | 不可变，关闭时无需处理 |
| 工作集中的资源 | 附件租借 | 进程关闭时必须释放或补偿，无论从哪个阶段结束 |

- 槽位可以动态，值必须有类型：哪些槽位被填上由进程自己决定（例如命令进程只有 GatewayDecision），但每个槽位都有声明的类型与所有者；不使用 `dict[str, Any]` 这类无类型载体。
- 进程关闭时释放工作集中登记的资源，取代迁移前分散的清理：`finally` 中的 prepared run 清理、finalize 中的租借释放、临时话题的补偿（2.5）。
- 写入意图不在工作集中：它在 workspace 登记，生命周期与进程完全解耦（Q-1、Q-2，[写入意图迁移 Idea](./pending-intent-migration.md#01-owner-的决定2026-09-28) 0.1）；进程最多在执行记录中保留意图的引用。
- **进程表登记的对象**（owner，2026-10-04）：进程表是唯一的进程注册表，登记 `process_id → TaskProcess`（进程容器），进程记录作为进程的控制面经进程取得，不单独登记。上表“进程记录（控制面）”最初沿用的是 `ChatGenerationRunRegistry` 演化为进程表、`ChatGenerationRun` 演化为进程记录的路线；进程容器出现后，进程表登记容器本身。
- **现有实现**（2026-10-04 核对）：
  - 进程记录（`ProcessRecord`）有 `process_id`、访问 context、事件通道、当前阶段、停止请求与终态，不保存身份字段（[身份与访问体系 Idea](./identity-and-access-model.md) I-8）。请求方式与 CPU 分配尚未记入：被动请求这一阶段不考虑（Q-6），CPU 的选择机制后置。
  - 工作集（`ProcessWorkingSet`）登记 prepare 结果、附件租借、附件编译得出的实际使用引用与 Actor 执行期间打开的 CPU 输出流；它只登记、不释放，资源由取得它的一方释放（租借由 `CPUAllocator`，CPU 输出流与 prepare 结果的 cleanup 由执行器）。GatewayDecision 与 CPU 执行结果由骨架以局部值持有，尚未成为工作集槽位；Topic 路由决定跨阶段传到结算是 Topic 按需创建的一部分。
  - 进程容器 `TaskProcess` 只持有进程记录、任务参数与工作集；四阶段骨架是所有进程共用的执行器 `TaskProcessRunner`。

迁移前对象的去向（均已完成）：

| 迁移前对象 | 去向 |
|:---|:---|
| `ChatGenerationRunRegistry` | 进程表 `ProcessTable`，位于 workspace 的 `process` 子包（总 Idea [D-9](./workspace-network-task-process-architecture.md#d-9-chat-编排与-chat-run-注册表的最终归属)） |
| `ChatGenerationRun` | 进程记录 `ProcessRecord` |
| `interaction_id` / `generation_id` 作为进程标识的用法 | 由 `process_id` 取代（Q-16） |
| `PreparedAgentRun` | 拆散：附件租借进入工作集；记忆原子、编译后的记忆与附件文本进入 CPU 输入清单；`AgentRunContext` 中编译好的记忆文本改由进程编译；`StreamPrelude` 改由进程从自身状态推导；`PreparedAgentRun` 不再回传用户消息与 GatewayDecision，两者由进程自己持有 |

**Patchouli prepare 与结算的拆分**

- prepare 退化为只执行一轮预检索，返回未编译的记忆原子；产出 `AgentRunContext` 的步骤移到新流程。
  - **编译由进程完成**（owner，2026-09-29；修订原记录中的“由 CPU 一侧编译”）：进程在分配 CPU 之后，调用共享引擎 `MemoryCompiler` 编译本轮检索结果，把编译后的文本与原始记忆原子一起放入输入清单；CPU 只决定文本在提示词中的位置。理由：编译算法本就是共享的 L2 引擎，已有 Patchouli prepare、Passive Ingress、Alice MTP、Alice CALL、向量存储与 reranker 等调用方；在 controller 模式下，由进程调用可以让外部 harness 的 adapter 不必各自重复调用逻辑。
  - 原记录的理由“编译结果依赖 Alice 的 alias 体系”有误：`engines/memory_compiler` 对 `agent_runtime.aliases` 的依赖只在编译 Alice 的 MTP 解析结果（`ResolveResult`，即 MTP READ 的返回）时出现；编译检索结果（`RETRIEVAL_CONTEXT`）使用记忆自身的 alias，与 Alice 无关。MTP READ 在执行循环中的返回仍由 Alice 的 MTP runtime 编译。
  - 编译的预算与目标将来由 CPU 在分配时声明；在只有 Alice 一个 CPU 时，沿用现有配置。
- **附件由进程取得租借并编译**（owner，2026-09-29）：附件脱离 Patchouli，租借作为工作集中的资源；进程据编译结果得知本轮实际用到的附件，交给结算阶段。
- **Topic 不再预先创建**：Topic 将与 conversation session 解耦，不再承担上下文（Q-9），因此不再需要“先建临时话题”：Gateway 的 Topic 路由决定作为工作集中的值，跨阶段传递到结算，提交后依此按需创建 Topic。现有临时话题的补偿（Patchouli 的 prepared run 清理路由）随之不再需要。
  - 实际使用的对话上下文由 ConversationSession 提供，原样积累，不再由外界干涉；Topic 作为内部记忆生成的资料，Gateway 话题路由与 Topic 只为记忆生成服务（Q-9）。conversation session 不是记忆的材料来源（owner，2026-09-28）。
  - 影响：前端以 SSE 事件 `topic_info` 确认本轮的 Topic，该事件目前在 prepare 之后、Actor 执行之前发出。Topic 改为提交后创建、进程在交互被提交队列接纳后就结束（Q-1）之后，进程结束时新 Topic 可能还没有路由确定。2026-09-28 已决定：取消“当前 Topic”概念，`topic_info` 改为进程结束后的异步记忆标注，前端回归 session 模型（[外部会话与 Topic 投影 Idea](./external-session-and-topic-projection.md#01-会话模型与-topic-池owner2026-09-28) 0.1）。
- **Profile 在 CPU 分配时由进程解析**（owner，2026-09-29）：prepare 不再解析 Profile。Profile 暂时直接调用 Patchouli 的公开路由，不经能力层：能力层的 `get_agent_profile` 依赖的 Profile 缓存还没有失效机制（`ProfileCache.evict_source` 没有调用方）。失效机制具备后再改经能力层（总 Idea 15.5）。
  - **解析时点的中间态**（owner，2026-09-29）：Profile 暂时在 Gateway 之后、Patchouli prepare 之前解析。当前 prepare 会按路由决定预先新建 Topic，话题池已满时还会先按 LRU 结算一个已有话题；若 Profile 在 prepare 之后才失败，失败的请求会留下这些不可逆的副作用。Topic 的新建与驱逐改到 interaction 提交之后以后，Profile 可以回到 prepare 之后的 CPU 分配步骤。
  - Agent Profile 的 `allowed_mtp_verbs` 与 `allowed_sys_tools` 演变为 workspace 能力层的 operation 控制，对所有 CPU 生效（总 Idea 15.4）。
- **交互记录由进程组装**（owner，2026-09-29）：进程组装最终的交互记录 `InteractionPayload` 交给结算阶段，结算阶段不再接收 `AgentRunResult`（版本目标第 1 条）。ActionReducer 与 TraceReducer 位于 core，进程调用它们归约轨迹不形成第二套规则；被动链路也由提交方（System 的 turn buffer）自行组装 `InteractionPayload`。`PendingAtomMaterializeTask` 继续存在，在写入意图的实时派发实现之前，`InteractionPayload` 仍携带 `materialize_tasks`。
- 写入意图改为经能力层实时提交之后，结算阶段只提交交互记录与相应的衍生内容，随后关闭进程（Q-1）。

**v0.7.0 版本目标**（owner，2026-09-28；记录见总 Idea [5.4](./workspace-network-task-process-architecture.md#54-m-5-v070-的范围与版本目标)）：

| 目标 | 状态 |
|:---|:---|
| 1. Patchouli 的公开路由既不产出、也不接收 Alice 专属的类型，包括 `AgentRunContext`、`StreamPrelude`、`AgentRunResult` 与编译好的记忆文本 | 已达成（第二、三批）；`InteractionPayload` 过渡期仍携带 `materialize_tasks` |
| 2. 一个非 Alice 的 CPU（测试中的替身即可）能跑完整个任务进程，不需要改动进程与入口的代码 | 已达成（第四批） |
| 3. 取消与清理都经过进程容器：进程关闭时释放已登记的资源，取代 Patchouli 的清理路由与 chat 编排中的补偿；每个阶段的取消都能通过容器接口测试 | 取消与关闭已收口到进程容器；剩余的 Patchouli cleanup 路由随 Topic 按需创建移除（外部会话方向） |
| 4. 命令、主动请求与被动请求经同一入口注册 | 这一阶段收口（第五批）：命令与主动请求经同一入口登记为进程，命令只解析不执行；被动请求这一阶段不考虑（Q-6） |

## 2. 迁移前的代码快照（2026-09-27，部分于 2026-09-28 核对）

本节描述迁移前的实现，2026-09-28 起已由 `workspace.process` 取代，只作为第 4 节问题的背景。

### 2.1 Chat run 注册表

`ChatGenerationRunRegistry`（原 `src/hivememory/alice/application/chat_control.py`，已随第一批实施删除）以 `interaction_id` 为键登记 run，重复登记直接拒绝；提供 get / cancel / status，控制请求只比较 Workspace 身份。

- `interaction_id` 由 server 的 chat 路由在进入服务前生成（`interaction_{uuid}`），`generation_id` 与它取值相同；
- SSE 的第一个事件 `generation_id` 在 Gateway 阶段之前发出；前端据此发起停止请求，请求体携带 `generation_id`；
- 阶段枚举 `ChatRunPhase` 把 chat 编排写死：`CREATED → GATEWAY → PREPARE → ALICE → FINALIZE → TERMINAL`；
- 注册表本身不持有工作状态：附件租借在 Patchouli prepare 返回的 `PreparedAgentRun` 中，写入意图在 Alice 的 `PendingAtomRuntime` 中，执行事件在 Alice run 中；
- 注册与编排都在 `chat_service.py` 内完成；run 记录终态后由 `close` 移出注册表。

```mermaid
flowchart LR
    REQ["POST /chat"] --> REG["ChatGenerationRunRegistry<br/>登记 run（阶段与终态）"]
    REG --> GW["GATEWAY<br/>Gateway 分析"]
    GW --> PREP["PREPARE<br/>Patchouli prepare"]
    PREP --> AL["ALICE<br/>Alice run"]
    AL --> FIN["FINALIZE<br/>Patchouli finalize"]
    FIN --> CLOSE["移出注册表"]

    PREP -. "附件租借" .-> S1[("PreparedAgentRun")]
    AL -. "写入意图" .-> S2[("Alice PendingAtomRuntime<br/>进程级共享")]
    AL -. "执行事件" .-> S3[("Alice run 内 turn events")]
    PREP -. "Profile / Topic / 检索" .-> S4[("Patchouli 内部读取")]
```

### 2.2 写入意图的寿命

本节内容至今未变，写入意图迁移实施后改变。

- 每个 AliceRuntime 只创建一个 `PendingAtomRuntime` 实例（[`alice/runtime/core.py`](../../src/hivememory/alice/runtime/core.py)），主帧与子帧共享；
- 任何一个根 run 以 COMPLETED 收尾时，[`AgentRuntime.finalize_run`](../../src/hivememory/agent_runtime/runtime.py) 都会调用 `evict_by_run`：删除上一次已标为 EXPIRED 的意图，并把**其他** run 中已离开 in-flight 的意图标为 EXPIRED；非 COMPLETED 的 run 调用 `cancel_run`；
- 因此一条已结算意图还能保留多久，取决于其后有多少个根 run 完成，其中也包括其他用户的 run。意图在事实上比产生它的 run 活得更久，但没有明确的持有者与保留期限。

### 2.3 记忆库内部工作

记忆生成（独立业务 lane）、Topic 的空闲/LRU 结算，以及在全局维护调度器上注册的维护任务，都在 Patchouli 或 System 调度设施内部运行，不由任何 Actor 执行。

记忆任务的暴露面：能力层的 [`MemoryTaskApplicationService`](../../src/hivememory/workspace/capability/memory_tasks.py) 为 HTTP 管理路由 `/api/v1/memory-tasks` 提供列表、查询与取消；任务进程在 finalize 返回后把本轮产生的记忆任务标识放入 SSE 事件（`memory_task_ids`）。MTP 没有与记忆任务相关的动词。

### 2.4 各阶段的取消行为（迁移前）

| 阶段 | 取消行为 |
|:---|:---|
| Gateway | 可以中断：直接取消阶段 task |
| Prepare | 不可中断：在它前后各检查一次停止请求，事后经 Patchouli 的 prepared run 清理路由补偿临时话题 |
| Alice | 可以中断：Alice 以 `CANCELLED` 状态结束 |
| Finalize | 拒绝取消（`already_finalizing`）；Patchouli 内部以 shield 保证继续完成已接管的工作 |
| 断流 | 由 `finally` 统一兜底：关闭 Alice 子流、prepare 成功而 finalize 未成功时清理 prepared run、移出注册表 |

客户端断开时，server 的 chat 路由以同一标识发起取消（reason 为 `client_disconnected`）。

命令由 Gateway 的 `GATEWAY_PROCESS` 路由内的命令分派器解析并执行，返回执行结果；Gateway 识别出命令时，chat 直接以命令结果结束，不进入 prepare。内置命令为 help、commands、clear（由客户端执行）与 runtime.status，没有服务端副作用。

### 2.5 Patchouli prepare 与 finalize（迁移前）

[`PatchouliService`](../../src/hivememory/patchouli/service.py) 的 `prepare_agent_run` / `finalize_agent_run`（2026-09-28 核对）：

| 阶段 | 内容 |
|:---|:---|
| prepare | 解析 Agent Profile；解析或新建 Topic，读取候选话题与话题上下文；检索并用 `MemoryCompiler` 编译记忆文本；取得附件租借并由 AttachmentCompiler 编译附件；组装 `AgentRunContext`、`StreamPrelude` 与 `PreparedAgentRun` |
| finalize | 接收 `AgentRunResult`，归约 MTP 轨迹并组装 `InteractionPayload`；提交交互并等到 applied（排序键为 `topic:{topic_id}`）；释放附件租借；派发写入意图的物化并记录检索命中 |
| cleanup | 清理 prepare 新建且仍为空的临时话题 |

Alice 的提示词以 Topic 的 `state_summary` 与最近 5 个 block 作为对话上下文（[`prompts/assembler.py`](../../src/hivememory/prompts/assembler.py)）。

## 3. 流程图

### 3.1 任务进程的生命周期

“任务结束”的判定见 Q-1：交互被提交队列接纳后关闭。

```mermaid
flowchart LR
    A["任务请求"] --> B["唯一入口注册<br/>两阶段认证后登记为进程"]
    B --> C["CPU 在进程内工作"]
    C --> D["任务结束<br/>交互被提交队列接纳（Q-1）"]
    D --> E["进程关闭"]
```

### 3.2 任务进程的四阶段骨架（controller 模式）

依据 1.2，画的是目标形态。尚未实施的部分：写入意图的实时提交（写入意图迁移第 2 步；第 1 步期间派发仍在结算阶段）、携带 Topic 路由决定提交并按需创建 Topic（外部会话方向）、CPU 分配记入进程记录（CPU 选择机制后置）。

```mermaid
sequenceDiagram
    autonumber
    participant R as 请求方
    participant E as 唯一注册入口
    participant P as 任务进程
    participant G as Gateway
    participant L as Patchouli
    participant S as 能力层
    participant C as Actor（CPU）
    R->>E: 任务请求（主动；被动请求见 Q-6）
    E->>E: 两阶段认证
    E->>P: 创建进程（进程记录与工作集）
    E-->>R: 进程句柄 / process_id（Q-16）
    P->>G: 阶段 1：分析
    G-->>P: GatewayDecision（含 Topic 路由决定）
    Note over P,G: 可取消；Gateway 只解析命令，不执行（Q-5a）
    P->>L: 阶段 2：预检索
    L-->>P: 未编译的记忆原子
    P->>S: 取得附件租借（工作集中的资源）
    P->>P: 分配 CPU，组装输入清单
    P->>P: 检查一次取消请求（Q-15）
    P->>C: 阶段 3：交付 CPU 输入清单（可取消）
    loop 执行期间
        C->>S: 读取 / 检索
        C->>S: 写入 / 修订意图：在 workspace 登记并实时提交给 Patchouli（Q-2）
        C->>P: 执行事件（进入执行记录）
    end
    C-->>P: 执行结束
    P->>L: 阶段 4：结算，提交交互记录与衍生内容，携带 Topic 路由决定（Q-14）
    L->>L: 按需创建 Topic
    P->>E: 关闭进程，释放工作集中的资源（Q-1）
```

### 3.3 一次 chat 请求在骨架中的走向

```mermaid
flowchart TB
    A["用户发送消息"] --> B["两阶段认证后创建进程"]
    B --> C["阶段 1：Gateway 分析"]
    C -. "解析出命令" .-> K["命令终态：暂不可用<br/>命令的运行随命令系统后置（4.2）"]
    C -- "对话" --> D["阶段 2：Patchouli 预检索<br/>返回未编译的记忆原子"]
    D --> E["CPU 分配：附件租借与编译、记忆编译"]
    E --> X{"检查一次取消请求（Q-15）"}
    X -- "有" --> I
    X -- "无" --> F["阶段 3：CPU 执行<br/>消费进程编译的输入清单<br/>写入意图经能力层实时提交"]
    F --> G{"执行结果"}
    G -- "完成" --> H["阶段 4：Patchouli 结算<br/>提交交互记录与衍生内容（Q-14）<br/>按需创建 Topic"]
    G -- "取消 / 失败" --> I["不提交交互记录（Q-14）<br/>已提交的写入意图照常生成"]
    H --> J["进程关闭，释放工作集中的资源<br/>（Q-1）"]
    I --> J
    F -. "CALL 子 Agent（Q-10）" .-> L["子执行单元<br/>在本进程内执行"]
```

## 4. 问题

### 4.1 已完成的问题

#### Q-1 进程何时关闭

**状态**：已完成。2026-09-28 决定；进程在 finalize 返回后关闭已实施，“接纳后即退出”尚未实施。

**问题**：AE2 中合成任务结束即产物回到网络存储，是同步、可确认的。HiveMemory 中交互要等 applied 才进入 Topic，写入意图的物化可能需要数十秒以上，结果也可能被丢弃（迁移前的意图寿命见 2.2）。进程是在执行结束时关闭，还是保留一个“结算中”的阶段，等所有交接到达终态再关闭？

**设计**：进程在结算结束的时点关闭，不设结算期。

- 写入意图的提交成为 workspace 能力层的一个方法，实时响应 Actor 的请求，不必等到一轮对话结束；写入意图的生命周期与进程完全解耦，登记位于 workspace，生成与结算由 Patchouli 的 memory generation controller 单独管理（[写入意图迁移 Idea](./pending-intent-migration.md#01-owner-的决定2026-09-28) 0.1）。收尾只需提交对话记录与相应的衍生内容，这一步基本没有开销，随后关闭进程。
- 收尾阶段等到交互被提交队列（`InteractionSubmissionQueue`）成功接纳，进程即可退出，不再等待 applied。附件租借因此在接纳时随进程关闭释放；Artifact promotion 在生成时按 `binding.asset_ref` 重新取得内容（[Chat 附件链路](../system/attachments.md)第 4 节），不依赖本轮的租借（分析）。
- 实时提交时当前一轮的交互记录还没有进入 Topic，生成材料只能“舍弃当前一轮”；取消与失败不再丢弃已提交的写入意图。
- 对 `topic_info` 的影响见 1.2（改为异步的记忆标注）。

**取舍**：曾考虑两阶段关闭（运行中 → 结算中 → 关闭，意图在关闭前归该进程），它需要在进程表中保留不占用 CPU 的进程，并定义保留窗口、容量与停机处理；写入意图改由 workspace 登记后，这些都不再需要。

**实现现状**：进程在 finalize 返回后关闭，finalize 仍等到交互 applied，因为写入意图的物化仍在结算阶段派发、以 applied 为边界（总 Idea 3.4）。“接纳后即退出”要等写入意图迁移第 2 步把派发移出结算阶段之后才能实施（分析）。

#### Q-2 写入意图（中间产物）的可见范围

**状态**：已完成。2026-09-28 决定；随写入意图迁移实施，尚未实施。

**问题**：AE2 中合成 CPU 存储的中间材料不出现在网络存储视图中。迁移前意图在同一 AliceRuntime 内共享，可见性按 `identity_scope` 相等判断；外部 Actor 与后续 run 都需要读回意图。写入意图应只对本进程可见，还是对更大范围可见？

**设计**：写入意图的读写一致性不因实时提交而改变。在记忆正式落库之前，PendingAtom 仍是替代正式记忆的唯一机制，因此直到落库之前都必须对后续进程可回读。

- 第一版采用简单实现：PendingAtom 不设 policy，默认对全 workspace 开放；
- PendingAtom 不参与检索，能拿到其别名的一般只有写入它的 agent，别人几乎无法访问；狭义上能做到“中间产物归进程”；
- 登记与进程解耦后，plugin 模式（不建进程，经 MCP 提交的写入意图没有所属进程）与 CALL 子执行单元的写入同样适用；
- 结算后句柄的生命周期需要重新设计，兼容期内暂不回收；可见范围放宽与别名强度的分析见[写入意图迁移 Idea](./pending-intent-migration.md#01-owner-的决定2026-09-28) 0.1。

**取舍**：“仅本进程可见”会让后续任务在结算前读不到上一个任务的意图别名，与迁移前在同一 AliceRuntime 内可读不同；“显式声明前驱进程链可见”需要定义前驱的声明、校验与链长。

#### Q-3 唯一注册入口的职责边界

**状态**：已完成。2026-09-28 决定；已实施（2026-10-04 TaskProcess 容器完成后完全一致）。子问题 Q-3b 未决，见 4.2。

**问题**：迁移前的注册表把 chat 阶段写死在枚举中，且不持有工作状态（2.1）。入口只负责注册与通用生命周期，还是同时承担编排？哪些工作状态进入进程（Q-3a）？

**设计**：

- 入口只管理任务进程的生命周期：进程标识、两阶段认证、登记与注销、取消、停机收尾；不存在任务类型（1.2），因此不需要任务类型的登记与分派机制。
- 实现：注册入口 `TaskProcessService` 只持有生命周期依赖（认证网关、操作授权者、事件发布器、进程表与执行器）；编排依赖只由所有进程共用的执行器 `TaskProcessRunner` 持有（[System 应用服务](../system/application-services.md)第 3 节）。
- **Q-3a 哪些工作状态进入进程**：访问 context 进入进程记录（已实施）；附件租借作为工作集中的资源（已实施）；执行轨迹作为工作集中的值，即执行记录（未实施：CPU 执行结果由骨架以局部值持有）；写入意图不进入进程，在 workspace 登记（Q-1、Q-2）。

#### Q-4 “合成树”在 HiveMemory 中的形态

**状态**：已完成。2026-09-28 决定；已实施。

**问题**：AE2 的合成计划由样板确定性推导；Agent 任务是在线的“生成—行动—观察”循环（见 [AE2 类比 §1](./ae2-hivememory-architecture-analogy.md#1-结论摘要)）。任务是只定义固定骨架、由 Actor 在线决定内部，还是在进程启动时计算显式执行计划（任务图）？

**设计**：controller 模式下只有一套固定骨架，即四阶段通用骨架，骨架内部由 Actor 在线决定；失效条件见 1.2。显式执行计划需要计划表示、规划器与计划失败语义，接近 AE2 类比中的 Job Graph，ROADMAP 中通用 workflow / DAG 为 Unscheduled。

#### Q-5 Gateway 在新架构中的位置

**状态**：已完成。2026-09-28 决定，2026-10-01 补充；已实施。命令在进程中何时、由谁运行，以及 Q-5b，见 4.2。

**问题**：Gateway 有 `ACTIVE_CHAT`（命令、查询分析、检索计划、话题路由）与 `PASSIVE_MEMORY`（分析、话题路由、价值信号）两种模式。它是特定任务类型的入口分析步骤，还是注册入口对所有请求的通用前置步骤？命令（Q-5a）是注册为任务进程，还是作为直接的网络操作执行？

**设计**：

- **Q-5**：Gateway 是每个任务进程的第一阶段，在进程内执行（1.2）。不存在任务类型，所以它不“只属于特定任务类型”；它也不在注册之前执行。
- **Q-5a**：命令所在的请求同样注册为进程。命令解析在 Gateway 内执行，但命令的实际运行不在 Gateway 中：后续指令会与用户请求同时出现，不能让 Gateway 实际执行命令。
- **命令系统后置**（2026-09-28）：命令系统不在 v0.7.0 计划内完整接回；现有四个内置命令（help、commands、clear、runtime.status）在 v0.7.0 内暂时不可用。
- **解析与执行分开**（2026-10-01）：“暂时不可用”指命令仍由 Gateway 解析，但不会执行；Gateway 只输出解析结果，原有的命令分发与执行（dispatcher 与 handler）删除；任务进程按解析状态产生命令终态（第五批实施）。

#### Q-6 被动请求的范围

**状态**：已完成。2026-10-01 决定；无需实施。

**问题**：被动请求由条件触发，最常见的是定时任务与队列任务（1.1），代码中没有面向用户的定时任务或队列任务设施；VISION 阶段 E 把“记忆事件触发 Agent 唤醒”放在前序阶段取得证据之后。当前阶段是否保留或实现被动请求？

**设计**：这一阶段先不考虑被动请求，避免现在做好的接口被后续的设计重新推翻；入口形状以后可能需要调整。版本目标第 4 条中的被动请求部分随之不在这一阶段实施。

#### Q-7 记忆库内部工作与进程表

**状态**：已完成一半。2026-09-27 决定记忆任务不暴露给 Agent；进程表是否收录这类后台任务未决，见 4.2。

**问题**：记忆生成、Topic 结算与维护任务在 Patchouli 或调度设施内部运行，不由 Actor 执行（2.3）；在 AE2 中，存储系统的存取不经过合成 CPU。这些工作是否进入进程表、是否对 Agent 可见？

**设计**：Patchouli 的记忆任务是给后台系统的任务，不暴露给 Agent。这排除了让记忆任务对 Agent 可见的形态；[外部 Actor Idea](./external-actor-registration-and-runtime-access.md) 3.5 中涉及任务投影的结果查询需按此重新审视。

#### Q-9 对话连续性的承载

**状态**：已完成。2026-09-28 决定；随外部会话与 Topic 投影方向实施，尚未实施。

**问题**：进程按任务划分后，多轮对话的连续性不在单个进程内。迁移前 chat 的连续性来自 Gateway 的话题路由与 Topic 工作集，`ChatRequest.session_id` 只是兼容字段。连续性由 Topic、独立的会话记录还是进程链承载？

**设计**：保留独立的会话记录。实际使用的对话上下文由 ConversationSession 提供，原样积累，不再由外界干涉；Topic 作为内部记忆生成的资料，Gateway 话题路由与 Topic 只为记忆生成服务。这是[外部会话与 Topic 投影](./external-session-and-topic-projection.md) Idea 一开始就定下的前提（该 Idea 第 0、3 节）。

- 现状下 Alice 的对话上下文来自 Topic 的 `state_summary` 与最近 5 个 block（2.5），迁移后改由 ConversationSession 提供；
- Topic 不再预先创建，见 1.2；
- 原占位计划“话题折叠、Actor 上下文与原始证据统一改造”的背景（“话题折叠同时影响 Alice 使用的上下文”）按本决定不再成立；该计划已于 2026-10-01 退回 Idea 后删除（删除前最后版本见 commit `74b5056`），内容按前台与后台分别并入[Turn 内上下文折叠 Idea](./long-running-agent-intra-turn-context-folding.md)与 [Page Folding Raw Evidence Idea](./PatchouliPageFoldingRawEvidenceDesign.md)。

#### Q-10 CALL 子 Agent

**状态**：已完成。2026-09-27 决定；与现状一致（CALL 在父 Agent 的 run 内执行）。子执行单元的认证方式见总 Idea P-5a（未决）。

**问题**：被调用方有不同的 agent_id、Profile 与 MTP 权限；迁移前子帧与主帧共享 `PendingAtomRuntime`，子帧写入主帧可见；CALL 只能从根 frame 发起。子 Agent 是建子进程，还是共用父进程？

**设计**：CALL 派生的子执行单元在父 Agent 的任务进程内执行，不能请求新进程，即不建子进程。父进程的访问上下文因此需要容纳被调用方的身份与权限，子执行单元的认证方式见总 Idea P-5a。

**取舍**：子进程需要定义父子间意图可见性，以及“只有顶层进程可派生子进程”等规则。

#### Q-14 主动进程的交互记录去向

**状态**：已完成。2026-09-28 决定；进程封口并只在 completed 时提交已实施（第三批），携带 Topic 路由决定提交未实施（外部会话方向）。

**问题**：Active finalize 与 Passive Ingress 共用同一提交队列（总 Idea [3.4](./workspace-network-task-process-architecture.md#34-统一的交互提交队列)）。进程是自行封口并提交交互记录，还是把执行轨迹交给 Import Bus、自身只读取与提交写入意图？

**设计**：进程自行封口并提交交互记录。

- 提交的交互记录采用 `InteractionPayload`，它是 owner 提出的共用提交模型（[外部会话与 Topic 投影](./external-session-and-topic-projection.md#22-interactionpayload共同封口交互)第 2.2 节）；其 `interaction_id` 字段始终取 `process_id` 的值（Q-16）；
- 只有 completed 的进程才提交交互记录；以取消或失败结束时不提交，已提交的写入意图照常生成，它们的材料本就不含当前一轮（Q-1），两者一致；
- 这一规则只针对提交给 Patchouli 的交互记录；取消或失败的一轮是否记入 ConversationSession，见外部会话与 Topic 投影 Idea 第 8 节第 6 条；
- 写入意图改为实时提交后，结算阶段只提交交互记录与衍生内容（Q-1），`materialize_tasks` 随写入意图迁移第 2 步移出交互载荷；
- 提交时携带 Topic 路由决定，Topic 在提交后按需创建（1.2）；现有排序键以 `topic:{topic_id}` 生成，新建 Topic 的交互在提交时还没有 topic_id，需要随外部会话方向处理。

**取舍**：经 Import Bus 提交的方案依赖 Import Bus 向提交方返回可等待的 applied 回执；Import Bus 已排除在现有系统之外（总 Idea [5.8](./workspace-network-task-process-architecture.md#58-import-bus-移出现有系统)），该方案不成立。“同一 harness 同时以 plugin 模式上报对话、又以 controller 模式运行任务进程时重复记录”的问题，留待 plugin 模式设计时处理。

#### Q-15 各阶段取消策略的声明方式

**状态**：已完成。2026-09-28 决定；已实施（第一批）。

**问题**：各阶段的取消策略不同（2.4）；取消改在进程容器上响应（1.2）。策略是由各阶段进入时向容器声明，还是由骨架静态定义？

**设计**：只有 Gateway 与 Actor 执行两个阶段可以取消，因为这两块有前台调用 LLM 的行为；进程负责这两块的取消管理。其余地方不设取消响应点，相当于不允许取消；不设置额外的取消策略。

- 统一在 Actor 执行开始前检查一次取消请求并响应：在预检索、取得附件租借、CPU 分配期间收到的取消请求，在这里生效；结算阶段不可取消（`already_finalizing`）；
- 外部取消请求必须带明确的 `process_id`，指明取消哪个任务进程；
- 只有用户有权取消，入口是唯一的 HTTP server 入口（总 Idea [15.10](./workspace-network-task-process-architecture.md#1510-进程控制取消与记忆任务路由)）；客户端断开时 server 路由发起的取消也经这一入口；
- 分析：系统停机时，asyncio task 在任何阶段都会被取消，这不属于外部取消请求；进程容器在这条路径上仍要释放工作集中的资源（AGENTS.md 对 `CancelledError` 传播与资源释放的要求）。

#### Q-16 进程标识与交互标识

**状态**：已完成。2026-09-28 决定；已实施（第一批）。

**问题**：迁移前的注册表以 `interaction_id` 为键，`generation_id` 只是它的兼容投影（2.1）；命令进程不产生交互记录（1.2），却仍要有标识。

**设计**：删除 `interaction_id` 与 `generation_id` 作为进程标识的用法，改用 `process_id` 作为任意任务进程的唯一标识，向下兼容 `interaction_id` 原先的位置。

- server 生成 `process_{uuid}`；前后端与 server 契约（SSE 首个事件、停止请求体、done 事件、RuntimeEvent、内核终端的分组）统一使用 `process_id`，不再使用 `generation_id`，也不设原拟的 `active_request_id`；
- `InteractionPayload` 单独保留 `interaction_id` 这一字段名，但始终赋 `process_id` 的值；
- 一个进程最多提交一次交互（Q-14），`process_id` 可以直接作为交互提交的幂等键；
- Qdrant 中的持久数据不保存 `interaction_id`（交互 Artifact 以 `turn_id` 与 `block_id` 记录），改名不需要数据迁移；
- plugin 模式不建进程，它提交的 `InteractionPayload` 中 `interaction_id` 如何取值，在 plugin 模式设计时处理。

### 4.2 未完成的问题

#### Q-3b 唯一入口与传输入口

**背景**：Q-3 已决定入口只管理任务进程的生命周期。剩下的问题是“唯一入口”的含义。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 唯一入口只指唯一的注册点 | HTTP、外部协议、触发器可以各自有对外端点，最终都调用同一个注册入口 |
| B | 同时要求唯一的传输入口 | 所有任务请求经同一个对外端点；外部协议与触发器需要适配到该端点 |

与[外部 Actor Idea](./external-actor-registration-and-runtime-access.md)的 controller 模式接入相关。

#### Q-5a（剩余）命令在进程中何时、由谁运行

**背景**：命令所在的请求注册为进程，Gateway 只解析命令（Q-5）；命令系统后置，v0.7.0 内内置命令暂时不可用。命令的运行位置随命令系统一并决定，候选尚未列出。[会话 `/compact` 指令 Todo](../todo/conversation-compact-command.md) 依赖这一决定。

#### Q-5b `PASSIVE_MEMORY` 模式的去留

**背景**：Gateway 的 `PASSIVE_MEMORY` 模式只服务于 Import Bus（现有 Passive Ingress 链路）。它的去留取决于总 Idea 的 [Q-11](./workspace-network-task-process-architecture.md#q-11-import-bus-交互的-topic-落位) 与 [Q-12](./workspace-network-task-process-architecture.md#q-12-import-bus-交互的价值信号worth_saving)；Import Bus 不在 v0.7.0 范围（总 Idea 5.8）。

#### Q-7（剩余）进程表是否收录后台任务

**背景**：记忆任务不暴露给 Agent 已决定（4.1 Q-7）。剩下的问题是进程表是否收录这类后台任务（收录时同样对 Agent 不可见）。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 不进入进程表，保留在 Patchouli 内部队列与维护调度 | 进程表只包含由 Actor 执行的任务 |
| B | 全部进入统一进程表 | 进程表成为全局任务表，承担调度职责 |
| C | 部分进入（例如由模型驱动的生成任务进入，纯维护不进入） | 需要明确划分标准 |

#### Q-8 外部 CPU 的进程

**背景**：Alice 在进程内运行，注册表能取消它，也知道它何时结束。外部 harness 不受网络控制，更接近 AE2 中“处理样板 → 外部机器”：网络送出材料，等待产物回流。owner 于 2026-09-27 决定本问题单独审议，v0.7.0 不对外部 Actor 所需的基建作承诺（总 Idea 5.4）；同日审议为 plugin 与 controller 两种接入模式（[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#11-两种接入模式owner2026-09-27) 1.1）：本问题只涉及 controller 模式，plugin 模式不建进程；controller 模式作为 v0.7.1 的首个真实外部 harness 接入。

- **Q-8a 回收方式**：请求方显式关闭 / TTL / 心跳 / 组合。
- **Q-8b 取消语义**：网络对外部 CPU 的取消能做到什么程度（例如吊销访问、丢弃未提交的意图、通知外部方），各项分别是否纳入。
- **Q-8c 粒度**：外部 harness 的一轮对应一个进程 / 一个外部会话对应一个进程 / 由接入适配层决定。

## 5. 相关问题（位于其他文档）

| 问题 | 位置 | 与本文的关系 |
|:---|:---|:---|
| P-1 经网络接入的 Actor 如何证明身份 | 总 Idea [15.2](./workspace-network-task-process-architecture.md#152-每次请求重新校验身份p-1a)、[第 16 节](./workspace-network-task-process-architecture.md#p-1-经网络接入的-actor每次请求如何证明身份) | P-1a 已完成：每次请求重新校验身份；P-1b、P-1c 未决，与 Q-8 相关 |
| P-2、P-10 Agent Profile 的权限 | 总 Idea 15.4 | 已完成：Profile 的两个 allow 字段演变为能力层的 operation 控制，随 Alice 的能力层调用迁移实施 |
| P-4 进程级权限收窄与创建进程的授权 | 总 Idea 15.9、[第 16 节](./workspace-network-task-process-architecture.md#p-4-进程级权限收窄与创建进程的授权) | P-4b 已完成：不开放创建任务进程；P-4a 未决 |
| P-5 CALL 与触发器的认证 | 总 Idea [第 16 节](./workspace-network-task-process-architecture.md#p-5-call-与触发器的认证) | 未决；P-5a 与 Q-10、P-5b/c 与 Q-6 相关；被动请求只存在于 controller 模式 |
| P-6 进程绑定 context 的失效时点 | 总 Idea 15.6 | 已完成并实施：context 与进程绑定，随进程关闭失效 |
| P-7 进程控制操作的授权主体 | 总 Idea 15.10、[第 16 节](./workspace-network-task-process-architecture.md#p-7-进程控制操作的授权主体) | 取消已完成并实施（Q-15）；其余控制操作未决 |
| P-9d 进程的定义 | 总 Idea 第 16 节 | 未决：管理员直接通道（方案 C）与前提第 3 条的关系 |
| Q-11–Q-13 Import Bus | 总 Idea [6.2](./workspace-network-task-process-architecture.md#62-import-bus-的问题不在-v070) | 不在 v0.7.0；Q-14 进程不经 Import Bus 提交交互记录 |
| M-1–M-7 迁移问题 | 总 Idea [第 5 节](./workspace-network-task-process-architecture.md#5-第一部分已完成的问题) | M-4 之外均已完成：按流程纵切，首条迁移流程为 Alice 的 chat 链路 |
| 会话记录的设计 | [外部会话与 Topic 投影](./external-session-and-topic-projection.md) | Q-9 的承载者；Topic 不再承担上下文、不再预先创建（1.2） |
| 写入意图的迁移 | [写入意图体系迁移](./pending-intent-migration.md) | Q-1、Q-2：登记位于 workspace、与进程解耦、第一版不设 policy；v0.7.0 内分两步实施（该 Idea 0.1） |
| 外部 Actor 的接入与运行时访问 | [外部 Actor 的接入登记与运行时访问](./external-actor-registration-and-runtime-access.md) | 两种接入模式：controller 模式（v0.7.1）使用本文的进程模型，plugin 模式（其后的 v0.7.x）不建进程；Q-8、Q-3b |
| 进程记录如何持有身份 | [身份与访问体系 Idea](./identity-and-access-model.md) I-3、I-8 | 已完成：注册入口签发即绑定；进程记录只持有 context；进程句柄与唯一的取消方法 |

## 6. 形成 Plan 的条件

- 满足 [Ideas 升级规则](./README.md#升级规则)，并遵守[文档治理规范](../DOCUMENTATION.md)第 8.3 节的计划约束；
- 本方向在 v0.7.0 内的五个批次均已实施归档（第 0 节“实施进度”）；
- 1.2 中尚未实施的部分分别由外部会话与 Topic 投影、写入意图迁移与 Alice 的能力层调用迁移各自建立计划，依据本文已完成的问题；
- 4.2 的问题都不阻塞 v0.7.0：Q-3b 与 Q-8 随外部 Actor 的 controller 模式（v0.7.1）决定，Q-5a 的剩余部分随命令系统决定，Q-5b 随 Import Bus 决定，Q-7 的剩余部分在需要统一后台任务视图时决定；
- owner 对问题的决定记录在对应问题下，并注明日期。
