---
title: HiveMemory Roadmap
status: current
owner: project
scope: releases-and-planned-capabilities
updates:
  - docs/PROJECT.md
  - docs/plans/
  - docs/governance/
  - docs/ideas/
  - docs/todo/
  - docs/archive/plans/
last_reviewed: 2026-10-30
---

# HiveMemory 开发路线图

本文只维护版本阶段、当前完成度、依赖关系和相关入口。已经生效的详细设计进入 Architecture、System 或子系统文档；跨版本质量目标进入 Governance；未承诺方向进入 Ideas；绑定版本的实施方案进入 Plans；完成后的实施文档进入 Archive。

路线图是一条能力依赖链，不是功能愿望清单。排序首先回答“下一项能力需要哪些已经可信的状态、契约和证据”，其次才考虑它是否显眼或易于演示。HiveMemory 的长期方向涉及异步任务、文件、文档、来源和研究，如果这些能力各自建立一套临时状态与失败语义，规模越大反而越难保持记忆闭环。因此近期路线优先补齐可复用底座，再让上层能力逐步消费它们。

## 1. 状态口径

| 状态 | 含义 |
|:---|:---|
| Released | 已有对应 Git tag 的发布版本 |
| Current Development | 当前开发基线，主体可能已合并但尚未发布 |
| Completed | 版本或事项内容已收尾；是否已发布仍以 Git tag 为准 |
| Planned | 已形成目标和大致边界，尚未成为当前事实 |
| Candidate | 候选排期，范围和顺序仍可调整 |
| Unscheduled | 已确认问题或方向，但尚未进入具体版本承诺 |
| Partially Landed | 阶段中的一部分已进入当前实现，其余仍未完成 |
| Deferred | 已明确后置，不属于近期承诺 |

当前版本事实如下：

- 最近已发布基线：`v0.7.0`（Git tag，2026-10-20）；[收尾审计](./archive/plans/v0.7.0-release-closeout-audit.md)记录版本目标判定、内容核对与验证结果；
- 当前开发版本：`v0.7.1`，版本目标为 plugin 模式的真实外部 harness 接入，首个接入的 harness 为 Claude Code（owner，2026-10-20），版本目标已完成（Completed，2026-10-29）：对话回流（[Import Bus 重设计](./archive/plans/v0.7.1-import-bus-redesign.md)，2026-10-21）与记忆访问（[记忆访问](./archive/plans/v0.7.1-memory-access-mcp.md)，2026-10-29）两份计划均已归档；版本内另有推理模型的空闲卸载计划（Planned，第 4.3 节）。

当前规范代码版本为 `0.7.0`，由 `src/hivememory/_version.py` 唯一声明并供构建与运行时复用。Python 包、前端清单与锁文件保持一致，与最近已发布标签 `v0.7.0` 对应；v0.7.1 开发期间合入的内容在版本收尾时升版。

## 2. 发布历史

| 版本 | 状态 | 核心结果 | 当前依据 |
|:---|:---:|:---|:---|
| `v0.1.0-beta` / `v0.1.1` | Released | 记忆 MVP、API、基础前端与早期冷热路径 | Git tag；后续设计已演进 |
| `v0.2.0` | Released | 多 Agent 隔离、Agent Profile 记忆化 | Git tag；[Patchouli 当前设计](./patchouli/README.md)；[Alice 当前设计](./alice/README.md) |
| `v0.3.0` | Released | CALL、PendingAtom、Alice Orchestrator、MemoryCompiler | Git tag；[Alice 当前设计](./alice/README.md)；[MTP](./contracts/mtp.md) |
| `v0.4.0` | Released | chat run / memory task 取消控制与 RuntimeEvent | Git tag；[路由与事件](./contracts/routes-and-events.md) |
| `v0.5.0` | Released | artifact/provenance、MemoryLibrary、async-native、模型注册 | Git tag；[System 当前设计](./system/README.md)；[Patchouli 当前设计](./patchouli/README.md) |
| `v0.6.0` | Released | System Gateway、全局命令、主动/被动入口契约、Passive Ingress 串行与 outbox | Git tag；[Gateway 当前设计](./gateway/README.md)；[Import Bus 当前设计](./system/import-bus.md)（2026-10-21 取代 Passive Ingress） |
| `v0.6.1` | Released | Local Work Queue Runtime、Active/Passive Interaction Submission 统一接入、Memory Generation queue 与进程内可靠生命周期 | Git tag；[System Runtime 当前设计](components/runtime-and-bus.md#3-local-work-queue-runtime)；[归档实施计划](./archive/plans/v0.6.1-local-work-queue-runtime.md) |
| `v0.6.2` | Released | Workspace MVP（WorkspaceIdentity、端到端 scope、双 Workspace 隔离、进程级 WorkspaceAssetStore）、Identity 投影收敛、V1 Memory 迁移、Chat Attachments、MTP scope 修复与 Topic/短期存储边界整理 | Git tag；[Workspace](./workspace/README.md)；[附件链路](./system/attachments.md)；[收尾审计](./archive/plans/v0.6.2-release-closeout-audit.md) |
| `v0.7.0` | Released | Workspace 网络与任务进程：唯一注册入口与任务进程、两阶段认证与授权、能力层与 Agent 操作白名单、执行线程、会话资源与自动压缩、写入意图实时派发、记忆内容版本 | Git tag；[Workspace](./workspace/README.md)；[收尾审计](./archive/plans/v0.7.0-release-closeout-audit.md) |

过去文档中的 `v0.5.1`、`v0.5.2`、`v0.5.3` 是 v0.5 开发期的内部工作批次，不是当前仓库中的独立发布标签。它们的已实现事实应按模块并入当前文档，而不是继续作为平行版本入口。

## 3. 最近已发布基线：v0.7.0

主题：**Workspace 网络与任务进程**。Git tag `v0.7.0` 创建于 2026-10-20。逐项内容、版本目标判定、收尾时搁置的事项与验证结果见[收尾审计](./archive/plans/v0.7.0-release-closeout-audit.md)，各方向的归档计划见 [Plans 导航](./plans/README.md#v070)。

### 3.1 范围与版本目标

- **范围**（2026-09-27 决定）：完全完成项目向新架构的演进，使 workspace 体系在项目架构中稳定存在。首条迁移流程为 Alice 的 chat 链路，验收口径是 Alice 在新架构下跑通、各流程协作无误，使之后的 adapter 不需要再大改系统拓扑结构；v0.7.0 不对外部 Actor 所需的基建作承诺。架构层面的后续问题见 [第五代架构 Idea](./ideas/fifth-generation-architecture.md)第 5 节。
- **版本目标**（2026-09-28 决定）与收尾判定：(1) Patchouli 的公开路由既不产出、也不接收 Alice 专属的类型——达成；(2) 非 Alice 的 CPU（测试替身即可）能跑完整个任务进程，不改动进程与入口的代码——达成；(3) 取消与清理都经过进程容器——达成，2026-10-13 决定保留话题预建后，cleanup 路由作为预建空话题的补偿保留，只由进程容器在关闭时发起；(4) 命令、主动请求与被动请求经同一入口注册——命令与主动请求达成，被动请求按 [Q-6](./ideas/task-process-table-and-registration-entry.md#26-被动请求q-6) 这一阶段不考虑，内置命令在 v0.7.0 内不可用。

### 3.2 发布内容

| 方向 | 结果 | 当前事实 |
|:---|:---|:---|
| 任务进程 | 任务进程表与唯一注册入口（五批）：进程表与四阶段编排落位 `workspace.process`，prepare 拆分与 CPU 输入清单，进程封口交互记录，CPU 端口与测试 CPU，命令只解析不执行；随后进程拥有自己的 task 与输出通道，交付改为订阅，断连按停止处理 | [任务进程](./workspace/task-process.md)、[CPU 契约](./contracts/subsystem-contracts.md#4-cpu-端口与-alice-实现) |
| 访问边界与身份 | A1 访问边界及其返工：统一认证网关、两阶段认证与两阶段授权、密封的访问 context；身份第二批：Patchouli 内部与后台任务分开携带归属与发起者 | [认证与访问登记](./workspace/authentication.md)、[身份与授权](./workspace/identity-and-authorization.md) |
| 能力层 | Alice 的资源操作经单一操作入口；capability 分为只供 `system` 的 root API 与 actor API，六项 actor 操作对齐 MTP 动词；访问登记的 actor 操作白名单成为 Agent 权限的唯一来源 | [能力层](./workspace/capability.md)、[MTP 契约](./contracts/mtp.md#4-权限) |
| 执行线程 | 进程 context 与线程 context 取代执行凭据；CALL 子线程以被调用方身份运行；线程表成为派生、结束与解析的唯一入口，子线程随 CALL 返回结束 | [执行线程](./workspace/execution-threads.md) |
| 写入意图与读取视图 | 写入意图登记迁入 workspace，canonical 变更同步失效读取缓存；WRITE/UPDATE 在操作内派发，生成队列接纳后返回 ACK | [读取视图与写入意图](./workspace/read-view-and-write-intents.md) |
| 会话 | Workspace 会话资源、Alice 全量历史、Gateway 最近 N 轮、有输出的取消与失败保存；Alice 会话压缩与检查点 | [会话资源](./workspace/sessions.md)、[Agent Runtime](./alice/agent-runtime.md#31-会话上下文视图) |
| 记忆模型 | 记忆内容版本与 Lifecycle 状态重构、Memory 域时间语义与可控时钟 | [数据模型](./architecture/data-model.md)、[Lifecycle](./patchouli/lifecycle.md) |
| 结构收敛 | 包分层；Alice 与 agent_runtime 结构重构；提示词编译前移至任务进程；异常体系与边界错误呈现；LLM 访问收敛到 infrastructure；总线路由类型化与 mypy 门禁；代码收敛清单 | [系统架构概览](./architecture/overview.md)、[Alice](./alice/README.md)、[错误模型](./contracts/error-model.md)、[System 组合根](./system/composition.md) |

### 3.3 发布范围边界

- 外部 Actor 的真实接入分为两种接入模式（[外部 Actor Idea](./ideas/external-actor-registration-and-runtime-access.md#11-两种接入模式owner2026-09-27) 1.1）：plugin 模式是 v0.7.1 的版本目标（第 4.3 节），controller 模式退回候选方向（第 5.6 节）；执行环境的选择（用户在对话开始前分别选择 agent 与执行环境）随 controller 模式接入一并考虑；
- Import Bus（原 Passive Ingress 链路）排除在现有系统之外，随 plugin 模式另行设计（v0.7.1；已于 2026-10-21 实施，见 [Import Bus](./system/import-bus.md)）；
- 收尾时搁置的 Todo 与仍未决的 Gateway 路由上下文矛盾见收尾审计第 4 节；
- 记忆库的归属判据与独立工作契约见 [ADR-0006](./architecture/decisions/0006-memory-library-custody-criteria-and-independence-contract.md)。

上一个发布版本 v0.6.2（Workspace 身份隔离与 Chat 附件）的内容见第 2 节与[收尾审计](./archive/plans/v0.6.2-release-closeout-audit.md)。

## 4. 近期计划

### 4.1 排序原则与版本调整

本轮规划依据 [VISION](./VISION.md) 的双轨策略：Workspace/Patchouli 提供可独立使用的资源与记忆基础设施；Alice 是 memory-native reference runtime，外部 harness 可以通过适配器成为其他 Actor。先验证资源可复用、执行可替换，再推进文档、研究和桌面产品化。

“脱离 Actor”在此指资源身份、授权、证据、结果和任务状态不依赖某个特定 Actor。确定性解析可由普通服务完成；搜索、浏览或代码执行可以调用工具 worker 或 Actor。Deep Research 的执行仍需要执行者，但研究状态和结果应在执行者离开后继续存在。外部 harness 无需采用 Alice 的 loop、run/frame 工作集或完整 MTP 语法；主动写入的 Pending 读取与结算属于共同资源能力。

版本重新安排如下；这些是目标与候选排期，不改变当前规范代码版本、Git tag 或已完成阶段的归属：

| 原排期 | 新排期 | 调整原因 |
|:---|:---|:---|
| `v0.6.2` Workspace 资源体系与 Agent 执行边界 | `v0.7.0`，已发布（第 3 节） | 独立于 W0/W1 收口；2026-09-27 起按 Workspace 网络与任务进程架构重新规划 |
| `v0.7.0` Document Ingestion & Provenance | `v0.7.2`，Candidate | 建立在新资源边界上，并纳入冷启动、历史对话导入和证据资产化 |
| `v0.7.1` MTP READ Provenance | `v0.7.3`，Candidate | 消费已经稳定的文档表示、来源和版本契约 |
| `v0.7.2` Deep Research MVP | `v0.7.4`，Candidate | 等待资源、证据、读取编译和可靠执行的闭环 |
| 整体后置的 Executable Asset Sandbox | 曾前置为 `v0.7.1` 执行基座；2026-10-20 起暂不考虑（第 5.4 节） | 原意是先解决 MTP RUN 可复现的可靠性问题和可兑现的能力边界；v0.7.1 改以外部 harness 接入为目标 |

旧 Idea、W1 实施记录及历史材料中出现的上述旧版本号，按本表解释其后续排期；不重写历史使其看起来曾采用新顺序。新建 Plan 必须使用新目标版本。

除已定为 v0.7.1 版本目标的 plugin 模式接入（Completed，2026-10-29）外，下列新增工作仍为 Candidate，实施前分别建立范围、迁移、测试与验收方案。版本号表达交付顺序；互不依赖的小切片可以并行验证，不能以并行开发跳过契约冻结。

### 4.2 目标总览

| 目标版本或工作流 | 状态 | 目标结果 | 依赖与实施入口 |
|:---|:---:|:---|:---|
| `v0.7.1` 外部 harness 的 plugin 模式（首个：Claude Code） | Completed / v0.7.1 版本目标（2026-10-20）：对话回流（2026-10-21）与记忆访问（2026-10-29）均已完成 | HiveMemory 以插件形式挂载在 Claude Code 中，不接管对话：对话经 Import Bus 回流进入记忆生成，Claude Code 经 MCP 工具检索与读取记忆；在用户的日常使用中形成“外部 harness + Patchouli”的跨会话闭环 | v0.7.0 的能力层；对话回流见 [Import Bus](./system/import-bus.md)（[归档计划](./archive/plans/v0.7.1-import-bus-redesign.md)）；记忆访问见 [MCP 记忆访问](./contracts/mcp.md)（[归档计划](./archive/plans/v0.7.1-memory-access-mcp.md)） |
| `v0.7.2` Cold Start, Historical Import & Document Ingestion | Candidate | 冷启动种子、历史对话导入、文档解析、证据与候选记忆分层，冻结 provenance | v0.7.0 的资源边界、外部 Actor 接入契约的身份/来源/提交语义、记忆价值策略最小切片；[Artifacts](./patchouli/artifacts.md)、[附件链路](./system/attachments.md)；导入与文档分别建 Plan |
| `v0.7.3` MTP READ 专项编译与来源表达 | Candidate | 按资源类型、版本、定位和预算编译 READ 输出，给出可核验引用 | [v0.7.0 A2-P 完整记忆历史](archive/plans/v0.7.0-a2-pre-memory-version-and-lifecycle.md)、v0.7.2 provenance；[MTP](./contracts/mtp.md)、[MemoryCompiler](./patchouli/memory-compiler.md)；正式 Plan 待建立 |
| `v0.7.4` Deep Research MVP | Candidate | 研究状态、来源、证据、发现和报告闭环，执行提供者可替换 | 资源边界、Document/READ、实际执行能力、明确恢复范围；正式 Plan 待建立 |
| 外部 harness 的 controller 模式 | Candidate / 版本待定（2026-10-20 由 v0.7.1 版本目标退回） | 外部 harness 作为执行环境运行用户选择的 agent，请求登记为任务进程并交给该 harness 执行 | v0.7.0 的任务进程表与唯一注册入口；第 5.6 节；[接口调研快照](./research/harness-interface-survey-snapshot.md)；正式 Plan 待建立 |
| 记忆价值策略重设计 | Candidate / 跨版本 | 先冻结入口信号与持久化决策边界，再用真实样本校准 | v0.7.1 起启动分析，v0.7.2 批量物化前交付最小策略；[Gateway](./gateway/analysis.md)、[Perception](./patchouli/perception.md)、[Lifecycle](./patchouli/lifecycle.md) |
| Frontend Reliability & Resource UX | Partially Landed / 后续 Candidate | 先修身份、数据来源和状态可信性，再完善资源操作、来源和导入体验 | 可与 v0.7 并行；[Frontend](./frontend/README.md)与第 4.9 节 Todo；正式 Plan 待建立 |
| Electron 桌面客户端 | Candidate / `v0.8.x` 产品化窗口 | 单一客户端管理本地服务、数据目录、连接、升级与诊断 | 前端传输与资源生命周期稳定；[状态与传输](./frontend/state-and-transports.md)、[配置](./system/configuration.md)；正式 Plan 待建立 |

### 4.3 v0.7.1：版本目标

状态：Completed（版本目标 owner，2026-10-20；完成于 2026-10-29）。首份计划 [Import Bus 重设计](./archive/plans/v0.7.1-import-bus-redesign.md)已完成并归档（2026-10-21），当前事实见 [Import Bus](./system/import-bus.md)；真实使用验收随日常使用持续观察（[Todo](./todo/import-bus-real-use-observation.md)）。第二份计划[记忆访问](./archive/plans/v0.7.1-memory-access-mcp.md)已完成并归档（2026-10-29），当前事实见 [MCP 记忆访问](./contracts/mcp.md)：Claude Code 以自己的接入登记与凭据认证，对话回流随之改用这一身份；检索与读取的真实使用样本记入同一观察 Todo。

- **版本目标**：plugin 模式的真实外部 harness 接入，首个接入的 harness 为 Claude Code（第 4.4 节）。目的是让外部 harness 在用户真实的日常使用中使用 Patchouli 记忆，以此开始检验记忆系统的闭环；它对应 [VISION](./VISION.md) 的兼容轨，以及第 6 节评估中的“外部 harness + Patchouli”对照组。
- **排序依据**（owner，2026-10-20）：同日先定的 controller 模式目标改为 plugin 模式。各家 harness 的程序化接口开放程度不一（[接口调研快照](./research/harness-interface-survey-snapshot.md)），controller 模式能达到的控制效果取决于被接入方；非侵入的 plugin 模式更容易得到各家生态的支持，用户也继续在 harness 自己的界面中工作，真实的日常使用才会发生。记忆系统的核心机制自 MVP 以来进展较少，当前优先以真实使用检验它。controller 模式退回候选方向（第 5.6 节）。
- **分步实施**：版本目标按小步分别建立计划。plugin 模式由对话回流（经 Import Bus 进入记忆生成）与记忆访问（经 MCP 工具调用能力层）两部分组成。对话回流已由首份计划完成（2026-10-21）：harness 一轮结束后提交完整记录，经统一认证网关与导入授权，Gateway 只做落位与价值信号，应用完成后才确认；话题路由失败时续接上一轮的话题；生成视图纳入思考与外部工具的三元组；Claude Code 连接器以插件形式提供，详见 [Import Bus](./system/import-bus.md)。记忆访问在其后，阻塞问题已于 2026-10-29 全部决定：多种传输各由操作适配器归一到同一能力层，支持 MCP 的 harness 以 MCP 访问记忆（外部 Actor Idea E-2）；MCP 的操作适配器放在与 server 同层的独立入口包，MTP 的留在 agent_runtime（E-3b）；plugin 模式暂不提供主动预检索，记忆只经 MCP 工具由 agent 自己选择使用（E-7）；由统一认证网关证明身份，每次操作使用一份请求级访问 context（[第五代架构 Idea](./ideas/fifth-generation-architecture.md) P-1c、P-9c）；复核后，Claude Code 以自己的 principal 与本地共享密钥接入，对话回流一并迁移，不再借用 server 的 principal。能以保留的 `system` 进入 root API 的只有 HTTP 入口的管理访问（owner，2026-10-20），登记外部来源之前在接入登记中落实（[身份与访问体系 Idea](./ideas/identity-and-access-model.md#4-已登记的暂缓与例外)第 4 节）。各步的范围见[外部 Actor Idea](./ideas/external-actor-registration-and-runtime-access.md#v071-的步骤与各步的阻塞问题)。
- **规模控制**（owner，2026-10-20）：v0.7.1 期间不新增只服务 controller 模式的机制；每份计划写明它针对的已观察问题或待验证的核心假设（[VISION](./VISION.md) 第 10.10 节）。真实使用开始后，后续主线转向由使用中观察到的问题驱动 Patchouli 核心机制的改进，样本同时供第 4.8 节的价值策略校准。
- **常驻内存**（owner，2026-10-27；第 1、3 项已完成，int8 评估待开始）：plugin 模式下后端与 Qdrant 随日常使用常驻。观察到后端进程峰值常驻内存约 3.0 GB，主要来自启动后一直驻留的嵌入与重排模型，对 16 GB 内存的机器不友好。按以下顺序处理：
  1. 先把模型推理移出事件循环（已完成，2026-10-27，[归档记录](./archive/todo/inference-off-event-loop.md)）；
  2. 重排模型 int8 先做实验评估，与 ONNX 内存池、截断长度一并处理（[Todo](./todo/reranker-int8-and-onnx-memory.md)；内存池、截断、不生效的配置与 Docker 预下载已于 2026-10-27 完成，int8 评估待开始）；
  3. 模型的空闲卸载、按需加载与预取（已完成，2026-10-29，[归档计划](./archive/plans/v0.7.1-model-idle-unload.md)）：两个模型默认空闲卸载、使用时重新加载，chat 在 Gateway 之前预取，重排模型未就绪时这一次跳过重排；当前事实见 [Patchouli](./patchouli/README.md#41-模型驻留与空闲卸载)。实测重新加载期间其他请求最长等待约 1～3.7 秒，记为已知限制，之后单独优化（owner，2026-10-30，[Todo](./todo/model-reload-event-loop-stall.md)）。
- **工具解析链路重写**（owner，2026-10-30；设计中）：Claude Code 接入后发现 MTP 与外部 MCP 的工具调用写出的事件不同形，事件、动作、轨迹三份表达并存，下游解析来源不一致（[调查基线](./governance/baselines/tool-call-data-shapes-2026-10-30.md)）。重写以 `TurnEvent` 作为唯一的交互原子事件，分两部分：工具调用如何统一进入 `TurnEvent`，感知层如何解析这一份事件集；设计讨论见 [Idea](./ideas/tool-call-parsing-chain.md)，实施见[计划](./plans/v0.7.1-tool-call-parsing-chain.md)（Planned，2026-10-30）。
- **执行基座与沙箱基线**：原排期在 v0.7.1，暂不考虑（owner，2026-10-20），见第 5.4 节。plugin 模式下 HiveMemory 不执行代码，harness 的工具由其自身管理。

### 4.4 v0.7.1：plugin 模式接入 Claude Code

状态：Completed，v0.7.1 的版本目标（owner，2026-10-20）；对话回流已完成（2026-10-21），记忆访问已完成（2026-10-29，[归档计划](./archive/plans/v0.7.1-memory-access-mcp.md)，当前事实见 [MCP 记忆访问](./contracts/mcp.md)），真实使用验收已逐项核对，检索质量等长期表现随日常使用观察（[Todo](./todo/import-bus-real-use-observation.md)）；各步的范围与阻塞问题见[外部 Actor Idea](./ideas/external-actor-registration-and-runtime-access.md#v071-的步骤与各步的阻塞问题)。本节的验收针对整个版本目标。

外部 Actor 分为 plugin 与 controller 两种接入模式（[外部 Actor Idea](./ideas/external-actor-registration-and-runtime-access.md#11-两种接入模式owner2026-09-27) 1.1，2026-09-27 决定）。plugin 模式下 HiveMemory 以插件形式挂载在外部 harness 中，用户直接使用该 harness，HiveMemory 不接管对话：对话经 Import Bus 回流，harness 经 MCP 工具响应记忆域调用；外部 harness 的任务对 HiveMemory 不可见，不建任务进程，记忆访问经能力层、不绑定进程。首个接入的 harness 为 Claude Code（owner，2026-10-20）：用户日常使用它，它的插件可以把 hooks 与 MCP server 打包为一个单元。对话回流由 [Import Bus](./system/import-bus.md) 与 Claude Code 连接器实现（2026-10-21）：Import Bus 只向网络输入、不返回记忆上下文，记忆到达 harness 由记忆访问一步承担。

验收必须形成真实闭环：用户在 Claude Code 中的对话回流进入记忆生成，形成的记忆来源可查；Claude Code 在之后的会话中检索并实际使用这些记忆；记忆访问经能力层逐次授权，越权与跨 Workspace 的访问被拒绝。仅采集日志、仅发送事件，或者只有回流、只有读取，都不足以证明 plugin 模式接入完成。

接入需把 Claude Code 的会话标识、轮次边界、重发与来源映射到 Import Bus 的接收语义；HiveMemory 服务不可用时不阻塞用户在 Claude Code 中的工作，并明确披露失败。HiveMemory 不控制 Claude Code 的 loop、工具与上下文压缩。交互应用和记忆物化状态继续分开；连接器与操作适配器不重写领域状态机，也不扩大已有幂等与持久化承诺。

### 4.5 v0.7.2：冷启动、历史导入与外源文档资产化

状态：Candidate，建议拆成两个可独立验收的 Plan，共享资源归属、来源和物化策略；确定性解析可以提前并行验证。

**冷启动与历史对话导入**解决“空库如何首次产生价值”和“既有会话如何保真迁入”。冷启动允许用户显式选择少量项目事实、偏好、Profile 或资料作为种子；不得自动把默认示例当作用户事实。历史导入保留 source、conversation/turn 标识、发生时间、说话者及分支/编辑关系；助手的推测或旧结论不能直接升级成用户当前事实。

导入过程支持预览、来源与 Workspace 选择、稳定导入标识、去重、进度/失败报告，以及只保留证据、稍后提炼记忆。它承接外部 Actor 接入契约的身份、来源和领域提交语义及历史样例缺口，另行定义批次、历史时间、冲突和恢复语义；复用 Patchouli 的物化入口，不简单按实时顺序把全部历史文本重放进当前活动话题，避免改写当前偏好或丢失历史时间关系。

**外源文档摄入**先形成 RAW/Artifact，再产生 representation、chunk、locator 和 evidence，最后按策略提炼候选与正式 Memory。复用 W1 的附件与 provenance 基础，并明确进程内 WorkspaceAsset 与持久化 Artifact 的区别。来源获取、确定性解析、模型辅助理解和记忆物化各自有成功/失败边界。

本地已给定文档的解析不依赖 Alice 或通用代码执行；需要网页获取、本地目录浏览或复杂转换时使用相应 provider。语义提炼可以调用模型或 Actor，但不能让文档注册、证据保存和读取依赖 Alice 的 run。

验收至少包含重复导入、部分失败、来源回查、跨 Workspace 隔离、旧信息与当前事实冲突，以及中断后的可解释处理。若承诺重启续跑，先交付导入 checkpoint 和所需任务持久化切片；现有进程内队列不足以承担该承诺。批量自动物化先完成第 4.8 节最小价值策略，避免一次导入大量低质量或相互冲突的记忆。

### 4.6 v0.7.3：MTP READ 专项编译与来源表达

状态：Candidate。基于 [v0.7.0 A2-P](archive/plans/v0.7.0-a2-pre-memory-version-and-lifecycle.md)的完整记忆历史，以及 v0.7.2 的来源、文档表示与版本契约，扩展 MemoryCompiler 对文档、代码、历史证据等资源的定向读取、片段定位、token 预算和引用呈现。先冻结实际需要的 READ 模式与错误语义，再扩展协议。完整历史存储、内容版本与 lifecycle 更新边界由 A2-P 提前交付，本阶段不再反向定义记忆写入格式。

结果应说明读到哪个资源和版本、选择了哪个片段、引用如何回到原始证据，以及截断、缺失来源、旧裁剪历史和无权限如何表达。历史记录中的 lifecycle 是捕获时状态，额外展示当前评估时必须区分时点，不能用当前原子补造历史。不能仅在现有文本末尾追加一个链接，也不能建立第二套 provenance 或检索状态。其编译能力可由外部 adapter 复用，MTP 负责自己的协议呈现。已登记的缺陷 [MTP READ 截断正式记忆的正文](./todo/mtp-read-content-truncation.md)预计在本阶段一并修复。

### 4.7 v0.7.4：Deep Research MVP

状态：Candidate。研究领域持有 request、source、evidence、finding 和 report；协调器负责推进、重试与进度；执行 adapter 提供搜索、抓取、文件读取或分析。资源和研究状态不依赖 Alice 的 frame，Actor 可替换也不表示各 harness 的工具与策略完全等价。

首个 MVP 选择一个有预算和停止条件的真实研究任务，交付来源采集、证据定位、发现记录、报告 Artifact、取消和失败披露。以 v0.7.2/3 的证据与读取结果作为报告依据，并复用既有队列；仅为验证过的研究流程增加最小协调状态，不预先建设通用 DAG 平台。

可先使用一个可靠的外部 harness 或工具 provider。若该路径不运行不可信本地代码，不把完整本地沙箱当作前置条件；若需下载并执行未知代码，先满足相应隔离门槛。执行提供者缺失时任务应明确等待或失败，不阻止已有研究资产读取。

研究状态跨 Actor 存续与跨进程恢复是两个承诺。恢复范围必须在正式 Plan 中冻结：若要在应用重启后继续，先持久化来源、进度与 checkpoint，并定义重放和重复副作用；在此之前只承诺实际支持的进程内暂停/继续。

### 4.8 记忆价值策略：跨版本重设计

状态：Candidate。v0.7.1 起可启动分析与样本收集，v0.7.2 大规模自动物化前完成最小策略，后续根据真实使用反馈迭代。

当前已有 Gateway 的 WRITE/SKIP/UNKNOWN 预判与 worth_saving 投影，但它们只供观测：Perception settlement 原先按 worth_saving 整轮筛选材料，Claude Code 真实导入中这一预判只看到用户消息，把“请提交”“继续”背后的整轮实现丢出了生成，2026-10-29 起不再筛选（证据见 [Todo](./todo/generation-material-filtering-and-truncation.md)）；正式 Memory 的生命周期评分另有依据。不能把现状简化成“缺少三态布尔值”，也不能用一个总分替代所有层次。事实入口见 [Gateway Analysis](./gateway/analysis.md)、[Perception](./patchouli/perception.md)和 [Lifecycle](./patchouli/lifecycle.md)。

规划分两个切片：

1. **决策边界与最小策略**：区分原始证据保留、候选提炼、正式记忆写入、长期维护价值；明确用户显式选择、不同内容类型、来源置信度、时效和重复冲突的处理，以及 UNKNOWN 的保守语义。入口信号不能替代 Patchouli 最终决定，也不应单独成为删除来源证据的依据。
2. **样本与反馈校准**：使用真实外部会话、历史导入和日常使用样本，评估关键事实漏存、错误/重复记忆、过时信息、用户修正成本、后续召回收益和模型开销。记录决策理由与策略版本，允许对同一批证据重评，避免每次策略调整都重新导入。

权责与评估明确后再决定是否需要多维评分或新字段。新策略不追溯性批量删除旧记忆；任何存量重评或迁移另有预览、审计和回滚范围。正式 Plan 待建立。

### 4.9 前端与 Todo 并行工作流

Frontend 已部分落地，后续优化按可用性和资源操作体验推进，详细事实见 [Frontend](./frontend/README.md)。本轨不阻塞后端基础重构，但与受影响 API 一起验收。

- **近期可靠性**：统一身份状态，明确真实/mock/stale/error，修复失败披露和乐观更新，覆盖 HTTP/SSE/WebSocket 的一致连接配置与请求终态。
- **资源操作体验**：Memory Garden 真实语义检索、可见性策略、附件/来源检查、导入预览与失败重试。Workspace 选择 UI 只在对应后端能力真实可用后开放。
- **交互和视觉完善**：布局、导航、窄屏、可访问性、空状态及主题；避免将未接线的 Terminal 或研究页面显示成已有功能。

Todo 排期按已核对状态和实际依赖吸收，不能把目录中所有事项视为新需求：

| Todo | 排期方式 | 范围约束 |
|:---|:---|:---|
| [MTP 缓存作用域重验（归档）](./archive/todo/mtp-cache-scope-revalidation.md) | Completed / Archived；保留为 v0.7.0 回归基线 | 已修复的 L0/L1/L2 隔离语义不重排为待开发功能 |
| [总线路由类型化与 mypy 门禁（归档）](./archive/plans/v0.7.0-typed-bus-routes.md) | Completed / Archived（2026-10-12）：三阶段实现、验收与审查完成，`mypy src` 清零；属于 v0.7.0 内容 | 提供方 contracts 声明类型化 route；运行时参数检查保留，CI 按 `uv.lock` 检查源码和类型回归样例 |
| [Alice 健康探针](./todo/alice-health-probes.md) | v0.7.0 收尾时搁置（2026-10-19），未绑定版本 | 反映执行器实际就绪/故障；不令资源 readiness 依赖 Alice 在线 |
| [RuntimeEvent 生产端迁移](./todo/runtime-event-producer-migration.md) | v0.7.0 收尾时搁置；按受影响生产域迁移，未绑定版本 | 保持事件只做观测，不以事件投递成功控制业务状态 |
| [前端身份所有权](./todo/frontend-identity-ownership.md) | 前端可靠性优先项 | 不把固定 main_workspace 选择冒充认证或 Workspace 管理 |
| [Mock fallback 披露](./todo/frontend-mock-fallback-disclosure.md) | 前端可靠性优先项 | 可辨认的数据来源与可兑现的写操作 |
| [Memory Garden 语义检索](./todo/frontend-memory-semantic-search.md) | 前端资源体验 | 复用真实后端能力，保留失败和空结果 |
| [Memory visibility policy UI](./todo/memory-visibility-policy-ui.md) | 后端契约稳定后进入资源体验 | UI 编辑策略不能替代后端授权 |
| [会话 /compact 指令](./todo/conversation-compact-command.md) | 会话与 Alice 自动压缩已完成，仍依赖命令运行位置与命令系统接回 | compact 只作用于 CPU 的会话上下文，不改写 Session 记录与 Topic；后台 Topic 的 page folding 是另一个问题 |
| [Work Queue 多 lane 拓扑](./todo/work-queue-runtime-lane-topology.md) | 保持 Deferred | 只有共享 Store、跨 lane 调度或可复现故障触发才重评 |

上述排期摘要不代替 Todo 本体的完成条件。实现时先复核其是否已解决；跨系统范围扩大则建立 Plan，不通过“清理 Todo”引入未评审的新状态机。

### 4.10 依赖与验收门槛

```text
v0.6.2 已实现基础 -> v0.7.0：Workspace 网络与任务进程（已发布）
                          |
                          +-> 外部 Actor 接入契约（Idea）
                          |       +-> v0.7.1 plugin 模式（Claude Code）
                          |       |          (Import Bus 回流 + 不绑定进程的能力层访问)
                          |       +-> controller 模式（候选，版本待定）
                          |       +-> v0.7.2 历史导入契约与批次能力
                          +-> v0.7.2 冷启动 / 文档资源与证据

v0.7.0 A2-P 完整记忆历史 + v0.7.2 来源 / 文档版本 -> v0.7.3 READ 专项编译
可靠执行提供者 + 证据 / READ + 研究状态恢复契约 -> v0.7.4 Deep Research

价值策略最小切片 -> v0.7.2 批量自动物化
前端可靠性与资源体验 -> 随 v0.7 各能力并行交付
服务生命周期 / 传输 / 数据升级稳定 -> Electron 产品化
```

各版本的计划顺序见 [Plans 导航](./plans/README.md)。确定性导入可以提前并行，依赖复杂获取或转换的切片需等对应 provider。真实外部接入越早形成样本，越能帮助价值策略和导入计划减少猜测。

每项候选进入实施前必须冻结范围、权威状态、数据来源、失败/取消/恢复承诺、幂等边界和观察指标。任何跨重启恢复、强隔离或 Actor 替换承诺都必须有对应实现证据。仍未具备的能力不能通过 UI、accepted 响应或事件日志伪装成完成。

## 5. 产品化与后置方向

### 5.1 Electron 桌面客户端

状态：Candidate，目标窗口为 `v0.8.x`，正式 Plan 待建立；不是 Deep Research 或 Alice 高级编排的强制下游。v0.7 期间可以做有限启动/连接原型，完整发布以资源生命周期、前端传输和数据管理稳定为门槛。

桌面化覆盖的工作多于 Web 页面外壳：本地 Python 服务及存储依赖的分发/连接、启动就绪、端口冲突、关闭 drain、数据目录、备份与升级、凭据、日志与崩溃诊断。HTTP、SSE、WebSocket 都需可配置地连接同一服务，不能仅迁移日志 URL。平台先选实际使用的一种完成闭环，再扩展安装包和签名发布。

Electron renderer 使用公开资源 API，main process 只持有必要的原生桥接与服务管理权限；远程内容和 Agent 代码不在特权主进程执行。Electron 的 renderer sandbox 不能替代 Agent 执行沙箱。

验收是可安装、启动、使用、关闭、重启和升级，用户知识资产持续可读；卸载/删除数据必须有明确语义。桌面生命周期引出的最小持久化需求按 [恢复治理](./governance/reliability/durability-and-recovery.md) 拆出，不依靠 localStorage 承担权威资源恢复。

### 5.2 Alice 高级编排

状态：Deferred，`v0.8.0+` 后置窗口，不是已承诺的单一版本主题。plan-and-execute、多角色协作、parallel specialists、review loop 和任务图需经过 VISION 中 Alice 存在价值的判定，并与“外部 harness + 同一 Patchouli 记忆库”比较。

优先验证能证明记忆作为状态与控制输入有价值的能力，不因外部 Actor 接入而复制一套通用 harness 功能清单。

### 5.3 高级记忆生命周期

状态：Deferred，保留 `v0.9.0+` 方向。Memory split/merge、完整 rollback、branch invalidation、provenance reverse index 和 L3 复活等待真实演化与恢复需求。记忆价值策略的最小重设计不等于提前实现全部生命周期能力。

### 5.4 执行基座与不可信代码沙箱

状态：Deferred。执行基座与沙箱基线原排期在 v0.7.1，2026-10-20 起暂不考虑（owner）；完整不可信代码沙箱是其后的剩余工作包。

**执行基座与沙箱基线**：用户实测 MTP RUN 经常失败是启动调查的依据，正式 Plan 首先收集复现样例，区分协议解析、alias/权限、参数编译、环境、依赖、工具进程与取消链路的问题；不预设所有故障都由缺少沙箱引起。

目标是统一执行 provider 的环境绑定、stdout/stderr、退出码、结构化结果、超时、取消和清理语义。Workspace 声明资源和可用能力，执行 provider 兑现文件、网络、进程与资源限制，Actor 选择调用；工具执行不能反向占有资源系统。

验收覆盖正常结束、工具异常、启动失败、超时、取消、子进程清理和不同 Workspace 环境隔离。受信任执行、工具 API 限制和 OS 级强隔离必须明确区分：工作目录、prompt 或 Python 包装层不足以限制任意代码。若第一版无法实施相应隔离，不接纳要求该隔离等级的任务。完整不可信代码沙箱保留独立的实现与验证门槛。

**完整不可信代码沙箱**：按实际威胁与负载选择隔离机制，验证宿主文件、网络、进程、资源配额、子进程终止和跨 Workspace 逃逸边界。不以目录配置或工具包装宣称强隔离。

### 5.5 未排期的其他工作包

| 事项 | 状态与进入条件 | 分类入口 |
|:---|:---|:---|
| 复合意图分解 | Unscheduled；先有 C0 指标和真实样本，再冻结 composite envelope | [Idea](./ideas/composite-intent-decomposition.md) |
| 自定义入口拦截规则 | Unscheduled；需真实外部接入场景、配置所有者和验收样本 | 正式 Idea/Plan 待建立 |
| 领域持久化与恢复 | Unscheduled 剩余范围；导入、Research、桌面化按各自承诺提取最小切片，通用 checkpoint/saga 平台不提前展开 | [Durability Governance](./governance/reliability/durability-and-recovery.md) |
| 领域幂等与 reconciliation | Unscheduled 剩余范围；新的提交/导入入口先明确所需业务幂等，再独立设计存量领域操作补偿 | [Idempotency Governance](./governance/reliability/idempotency-and-retry.md) |
| 数据模型可变性治理 | Unscheduled；按具体 DTO/聚合风险分批，不混入完整 PendingAtom 重写 | [Mutability Governance](./governance/data-model/mutability.md) |
| 跨用户认证及权限治理 | Unscheduled 的完整范围；外部 connector 先落实所需身份映射，不能把 scope 字段当作认证 | [Identity Governance](./governance/security/identity-and-execution-safety.md) |
| 通用长期 workflow / DAG | Unscheduled；Research 先验证最小真实流程，再决定是否提炼通用机制 | [VISION](./VISION.md)；独立 Plan 待建立 |
| Conversation Branching | Unscheduled；依赖来源、生命周期和真实编辑需求 | [聊天运行后续 Idea](./ideas/chat-run-lifecycle-follow-ups.md) |
| 配置热更新 | Unscheduled；包括接入登记的运行时登记、Workspace 访问登记中 Agent 操作白名单的运行时编辑（含前端开关，2026-10-10 决定并入）与其他配置文件的持久化与热更新，未来单独建立 Plan | [外部 Actor Idea](./ideas/external-actor-registration-and-runtime-access.md) E-1；正式 Plan 待建立 |
| 其他历史存储转换 | Unscheduled；若发现仍有未迁移存量，先列证据再建迁移 Plan | 已完成的 [V1 迁移历史](./archive/plans/v0.6.2-v1-memory-legacy-migration.md)不重新列为待办 |

### 5.6 controller 模式的外部 harness 接入

状态：Candidate，版本待定。2026-10-20 曾定为 v0.7.1 的版本目标，同日改为 plugin 模式（第 4.3 节）；首份计划的方向（首个外部驱动的纯文本闭环）随之撤回，计划未建立。步骤分析与阻塞问题保留在[外部 Actor Idea](./ideas/external-actor-registration-and-runtime-access.md#controller-模式的步骤分析退回候选方向)，Codex、Claude Code、pi 与 ACP 的接口资料见[接口调研快照](./research/harness-interface-survey-snapshot.md)（2026-10-20，模型实测待验证）；重新排期时先复核其中的未知项与待实测项。

controller 模式把 Codex、Claude Code、pi 一类成熟的 agent harness 作为执行环境与 CPU 类型纳入兼容范围：用户在对话开始前分别选择 agent 与执行环境（[执行环境 Idea](./ideas/execution-environment.md)前提 9、10；外部 Actor Idea E-6a），外部 harness 作为执行环境运行所选 agent，请求经唯一注册入口登记为任务进程，处理后交给该 harness 执行。它建立在 v0.7.0 的任务进程表与注册入口之上，消费外部 Actor 的接入与运行时访问契约（原计划 B），不强依赖 HiveMemory 本地沙箱；外部 harness 继续使用自己的工具和执行环境。采用 ACP 一类的通用协议还是各 harness 的 SDK 或非交互 CLI，见 Idea 的 E-2a。

验收必须形成真实闭环：请求登记为任务进程并由外部 harness 执行；外部 harness 经能力层检索已有记忆并实际用于任务，写入/修订意图经能力层提交；进程的交互记录进入记忆库（去向见 [任务进程](./workspace/task-process.md)）；下一次会话能召回新形成的知识并定位来源。取消、失败与进程回收按外部执行单元的进程问题（[任务进程 Idea](./ideas/task-process-table-and-registration-entry.md#q-8-外部执行单元的进程) Q-8）的决定呈现。仅转发请求、仅采集日志、仅发送事件或仅返回 memory context 均不足以证明 Actor 接入完成。

controller 模式下 HiveMemory 控制任务边界（actor 选择、输入、生命周期与取消、执行轨迹回流），不控制外部 harness 内部的 loop、工具与上下文压缩；外部 harness 默认保有自己的 prompt history，HiveMemory 不隐式接管上下文压缩。

## 6. 长期方向与评估

长期取舍遵循 [VISION](./VISION.md)：先证明开放记忆基础设施的实际价值，再验证 Alice 的 memory-native 运行能否带来额外收益。模型与 harness 的更换不应迫使用户重建长期知识资产；Actor 可替换性也不表示其执行过程或能力可以完全等价迁移。

评估逐步使用同一任务和记忆库比较无记忆、普通检索、外部 harness + Patchouli、Alice + Patchouli。记录跨会话复用、来源可查、错误与重复记忆、用户修正成本、执行可靠性、延迟和资源开销。新增功能、抽象层数和目录重排本身不作为成功指标。

## 7. 路线图维护规则

1. 只有 Git tag 对应版本标为 Released；本次重排不修改规范代码版本或构建产物。
2. 新候选先在 Roadmap 明确目标、依赖与验收出口，实施前建立绑定版本的 Plan；详细设计只维护一份。
3. 仅已完整完成实现、测试、迁移与审查的内容晋升到当前事实文档，并归档 Plan。
4. 部分落地保持 Partially Landed，已完成 Todo 不重新排期，Deferred 项只有满足触发条件才启动。
5. 排期变化同步 Plan 的 target、文件名、索引和反向引用；已完成历史文档保留原版本，当前路线提供迁移映射。
6. PROJECT、Architecture、System 和子系统索引的行为说明按文档治理收口门禁更新；单纯调整候选排期不提前向事实层推广设计。
7. 定期依据真实外部接入、导入样本和运行故障复核顺序；范围变化显式记录，不把前置能力缺口隐藏在上层功能中。
