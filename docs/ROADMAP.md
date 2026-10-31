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
last_reviewed: 2026-10-31
---

# HiveMemory 开发路线图

本文只维护版本阶段、当前完成度、依赖关系和相关入口。已经生效的详细设计进入 Architecture、System 或子系统文档；跨版本质量目标进入 Governance；未承诺方向进入 Ideas；绑定版本的实施方案进入 Plans；完成后的实施文档进入 Archive。

路线图回答当前版本围绕什么功能、为什么是它，以及哪些方向仍是候选。自 v0.7.0 之后，计划之间不预先排定顺序：一份计划只判断能否立即执行，能够立即执行时绑定当前版本建立；不能立即实施的方向只留在 Idea 候选池，不预先分配版本号（owner，2026-10-30，第 4.1 节）。版本功能优先由真实使用中观察到的问题决定（[VISION](./VISION.md) 第 10.10 节）。能力之间的依赖仍然成立：异步任务、文件、文档、来源和研究如果各自建立一套临时状态与失败语义，规模越大越难保持记忆闭环，依赖未成为事实的计划不能立即执行（第 4.9 节）。

## 1. 状态口径

| 状态 | 含义 |
|:---|:---|
| Released | 已有对应 Git tag 的发布版本 |
| Current Development | 当前开发基线，主体可能已合并但尚未发布 |
| Completed | 版本或事项内容已收尾；是否已发布仍以 Git tag 为准 |
| Planned | 已形成目标和大致边界，尚未成为当前事实 |
| Candidate | 候选方向，范围仍可调整；v0.7.0 之后不绑定版本 |
| Unscheduled | 已确认问题或方向，但尚未进入具体版本承诺 |
| Partially Landed | 阶段中的一部分已进入当前实现，其余仍未完成 |
| Deferred | 已明确后置，不属于近期承诺 |

当前版本事实如下：

- 最近已发布基线：`v0.7.1`（Git tag，2026-10-30）；[收尾审计](./archive/plans/v0.7.1-release-closeout-audit.md)记录版本目标判定、内容核对与验证结果；
- 当前开发版本：`v0.7.2`，生成质量，状态为 Planned（版本目标 owner，2026-10-30，第 4.3 节）；[生成材料修正与原始证据保存](./archive/plans/v0.7.2-generation-material-and-raw-evidence.md)已于 2026-10-31 完成归档，其余方向的阻塞问题待决定。

当前规范代码版本为 `0.7.1`，由 `src/hivememory/_version.py` 唯一声明并供构建与运行时复用。Python 包、前端清单与锁文件保持一致，与最近发布标签 `v0.7.1` 对应。

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
| `v0.7.1` | Released | plugin 模式接入 Claude Code：Import Bus 对话回流与 MCP 记忆访问；推理模型空闲卸载；`TurnEvent` 统一的工具解析链路 | Git tag；[Import Bus](./system/import-bus.md)；[MCP 记忆访问](./contracts/mcp.md)；[收尾审计](./archive/plans/v0.7.1-release-closeout-audit.md) |

过去文档中的 `v0.5.1`、`v0.5.2`、`v0.5.3` 是 v0.5 开发期的内部工作批次，不是当前仓库中的独立发布标签。它们的已实现事实应按模块并入当前文档，而不是继续作为平行版本入口。

## 3. 最近已发布基线：v0.7.1

主题：**plugin 模式接入 Claude Code**。Git tag `v0.7.1` 创建于 2026-10-30。逐项内容、版本目标判定、未纳入的事项与验证结果见[收尾审计](./archive/plans/v0.7.1-release-closeout-audit.md)，各计划见 [Plans 导航](./plans/README.md#v071)。

### 3.1 版本目标

- **版本目标**（owner，2026-10-20）：plugin 模式的真实外部 harness 接入，首个接入的 harness 为 Claude Code。目的是让外部 harness 在用户真实的日常使用中使用 Patchouli 记忆，以此开始检验记忆系统的闭环；它对应 [VISION](./VISION.md) 的兼容轨，以及第 6 节评估中的“外部 harness + Patchouli”对照组。
- **接入形态**：plugin 模式下 HiveMemory 以插件形式挂载在外部 harness 中，不接管对话：对话经 Import Bus 回流，harness 经 MCP 工具检索与读取记忆；外部 harness 的任务对 HiveMemory 不可见，不建任务进程（[外部 Actor Idea](./ideas/external-actor-registration-and-runtime-access.md#11-两种接入模式owner2026-09-27) 1.1）。
- **排序依据**（owner，2026-10-20）：同日先定的 controller 模式目标改为 plugin 模式。各家 harness 的程序化接口开放程度不一（[接口调研快照](./research/harness-interface-survey-snapshot.md)），controller 模式能达到的控制效果取决于被接入方；非侵入的 plugin 模式更容易得到各家生态的支持，用户也继续在 harness 自己的界面中工作，真实的日常使用才会发生。记忆系统的核心机制自 MVP 以来进展较少，当前优先以真实使用检验它。controller 模式退回候选方向（第 5.6 节）。
- **规模控制**（owner，2026-10-20）：v0.7.1 期间不新增只服务 controller 模式的机制；每份计划写明它针对的已观察问题或待验证的核心假设（[VISION](./VISION.md) 第 10.10 节）。真实使用开始后，后续主线转向由使用中观察到的问题驱动 Patchouli 核心机制的改进；v0.7.2 由此确定（第 4.3 节）。
- **验收**：用户在 Claude Code 中的对话回流进入记忆生成、来源可查；之后的会话检索并实际使用这些记忆；记忆访问经能力层逐次授权，越权与跨 Workspace 的访问被拒绝；HiveMemory 不可用时不阻塞 Claude Code。收尾审计第 1 节逐条判定为达成。

### 3.2 发布内容

| 方向 | 结果 | 当前事实 |
|:---|:---|:---|
| 对话回流 | Import Bus 重设计：一次接收 harness 一轮已结束交互的完整记录，经统一认证网关与导入授权，Gateway 只做话题落位与价值信号，应用完成后才确认；话题路由失败时续接上一轮的话题；Claude Code 连接器以插件形式提供 | [Import Bus](./system/import-bus.md) |
| 记忆访问 | Claude Code 经 MCP 工具检索与读取记忆：MCP 入口是与 server 同层的独立入口包，每次工具调用取得一份请求级访问 context，经操作入口的请求级提交逐次授权；Claude Code 以自己的接入登记与本地共享密钥认证，对话回流随之改用这一身份；能声明保留的 `system` 的只有登记了 `allow_system_actor` 的来源 | [MCP 记忆访问](./contracts/mcp.md)、[认证与访问登记](./workspace/authentication.md) |
| 常驻内存 | 模型推理移出事件循环；ONNX 会话关闭 CPU 内存池，嵌入输入截断到 1024 token，删除从未生效的配置，Docker 预下载与运行时读取同一配置；两个模型默认空闲卸载、使用时重新加载，chat 在 Gateway 之前预取 | [Patchouli](./patchouli/README.md#41-模型驻留与空闲卸载) |
| 工具解析链路 | `TurnEvent` 是唯一的交互原子事件：MTP 与外部 MCP 的工具调用写出同形事件，一次工具调用与它的结局是同一个事件；交互载荷只以事件表达内容，感知层对事件集只解析一次，生成视图、LLM 接力摘要与 token 统计共用唯一的渲染 | [数据模型](./architecture/data-model.md)、[感知与短期话题](./patchouli/perception.md)、[Patchouli 生成](./patchouli/generation.md) |
| 真实使用中的修正 | 话题折叠移出交互应用，作为提交队列中的独立工作项；结算不再按 `worth_saving` 筛选材料；gardening 不归档 Profile 记忆；只有 Profile 记忆可以经编辑设置 `agent_config` | [感知与短期话题](./patchouli/perception.md)、[Lifecycle](./patchouli/lifecycle.md)、[Patchouli 生成](./patchouli/generation.md) |
| 结构收敛 | 任务进程概念盘点冻结为治理基线，AGENTS.md 加入复杂度控制与可读性约束；删除任务进程中无生产使用方的代码并合并重复表达 | [任务进程](./workspace/task-process.md)、[概念盘点基线](./governance/baselines/task-process-concept-inventory-2026-10-28.md) |

### 3.3 发布范围边界

- **不在 v0.7.1**：重排模型 int8 现在不是必要项，未来再考虑（owner，2026-10-30，[Todo](./todo/reranker-int8-and-onnx-memory.md)）；推理模型重新加载时的事件循环停顿记为已知限制，之后单独优化（owner，2026-10-30，[Todo](./todo/model-reload-event-loop-stall.md)）；controller 模式退回候选方向（第 5.6 节），执行基座与沙箱基线暂不考虑（第 5.4 节）；写入意图经 MCP 提交 2026-10-30 起是 v0.7.2 的方向之一（第 4.3 节）。
- **真实使用观察**：Import Bus 真实使用观察 Todo 随收尾归档（owner，2026-10-30，[归档记录](./archive/todo/import-bus-real-use-observation.md)），之后正常使用；观察期间需要改动的项目登记为[生成查重的决策信号](./todo/generation-dedup-decision-signal.md)与[生成材料的筛选与截断](./archive/todo/generation-material-filtering-and-truncation.md)，现为 v0.7.2 的输入（后者已随 v0.7.2 的生成材料修正归档）。

上一个发布版本 v0.7.0（Workspace 网络与任务进程）的内容见第 2 节与[收尾审计](./archive/plans/v0.7.0-release-closeout-audit.md)，各方向的归档计划见 [Plans 导航](./plans/README.md#v070)。

## 4. 近期计划

### 4.1 排序原则与版本调整

本轮规划依据 [VISION](./VISION.md) 的双轨策略：Workspace/Patchouli 提供可独立使用的资源与记忆基础设施；Alice 是 memory-native reference runtime，外部 harness 可以通过适配器成为其他 Actor。先验证资源可复用、执行可替换，再推进文档、研究和桌面产品化。

“脱离 Actor”在此指资源身份、授权、证据、结果和任务状态不依赖某个特定 Actor。确定性解析可由普通服务完成；搜索、浏览或代码执行可以调用工具 worker 或 Actor。Deep Research 的执行仍需要执行者，但研究状态和结果应在执行者离开后继续存在。外部 harness 无需采用 Alice 的 loop、run/frame 工作集或完整 MTP 语法；主动写入的 Pending 读取与结算属于共同资源能力。

**计划的准入**（owner，2026-10-30）：

- v0.7.0 之后，计划之间没有预先排定的顺序约束。一份计划只判断能否立即执行：阻塞问题已经决定、依赖已经成为事实，就绑定当前版本建立；
- 不能立即实施的方向只留在 Idea 候选池，不预先分配版本号，路线图也不为它们排先后；
- 一个版本围绕一个功能，由多份计划完成，不以单份计划完成整个版本。

版本号的调整如下；这些调整不改变规范代码版本、Git tag 或已完成阶段的归属：

| 原排期 | 现状 | 调整原因 |
|:---|:---|:---|
| `v0.6.2` Workspace 资源体系与 Agent 执行边界 | `v0.7.0`，已发布（第 2 节） | 独立于 W0/W1 收口；2026-09-27 起按 Workspace 网络与任务进程架构重新规划 |
| `v0.7.0` Document Ingestion & Provenance | 2026-09-27 改排 `v0.7.2`（冷启动、历史导入与文档资产化）；2026-10-30 撤销版本号，改为候选方向（第 4.4 节） | v0.7.2 改以生成质量为版本目标；候选方向不预先分配版本 |
| `v0.7.1` MTP READ Provenance | 2026-09-27 改排 `v0.7.3`；2026-10-30 撤销版本号（第 4.5 节） | 同上 |
| `v0.7.2` Deep Research MVP | 2026-09-27 改排 `v0.7.4`；2026-10-30 撤销版本号（第 4.6 节） | 同上 |
| 整体后置的 Executable Asset Sandbox | 曾前置为 `v0.7.1` 执行基座；2026-10-20 起暂不考虑（第 5.4 节） | 原意是先解决 MTP RUN 可复现的可靠性问题和可兑现的能力边界；v0.7.1 改以外部 harness 接入为目标 |
| Electron `v0.8.x` 产品化窗口、Alice 高级编排 `v0.8.0+`、高级记忆生命周期 `v0.9.0+` | 2026-10-30 撤销版本窗口（第 5.1–5.3 节） | 候选方向不预先分配版本 |

旧 Idea、W1 实施记录及历史材料中出现的上述旧版本号，按本表解释；不重写历史使其看起来曾采用新顺序。新建 Plan 必须使用当前版本。

### 4.2 目标总览

| 版本或候选方向 | 状态 | 目标结果 | 依赖与入口 |
|:---|:---:|:---|:---|
| `v0.7.2` 生成质量 | Planned（版本目标 owner，2026-10-30）；一份计划已完成 | 真实使用中生成的记忆可信：生成材料不被单轮上限截断，Topic 之后的原始证据可以回查；一个话题能提取多条记忆；查重与更新按新机制处理同一件事的后续阶段；Claude Code 能显式写入与更新记忆 | 第 4.3 节 |
| 冷启动、历史导入与外源文档资产化 | Candidate / 不绑定版本 | 冷启动种子、历史对话导入、文档解析、证据与候选记忆分层，冻结 provenance | 第 4.4 节；[Artifacts](./patchouli/artifacts.md)、[附件链路](./system/attachments.md) |
| MTP READ 专项编译与来源表达 | Candidate / 不绑定版本 | 按资源类型、版本、定位和预算编译 READ 输出，给出可核验引用 | 第 4.5 节；[MTP](./contracts/mtp.md)、[MemoryCompiler](./patchouli/memory-compiler.md) |
| Deep Research MVP | Candidate / 不绑定版本 | 研究状态、来源、证据、发现和报告闭环，执行提供者可替换 | 第 4.6 节 |
| 外部 harness 的 controller 模式 | Candidate / 不绑定版本（2026-10-20 由 v0.7.1 版本目标退回） | 外部 harness 作为执行环境运行用户选择的 agent，请求登记为任务进程并交给该 harness 执行 | 第 5.6 节；[接口调研快照](./research/harness-interface-survey-snapshot.md) |
| 记忆价值体系重新设计 | Candidate / 不绑定版本 | owner 重新设计价值体系，不在现有信号上增补 | 第 4.7 节 |
| Frontend Reliability & Resource UX | Partially Landed / 后续 Candidate | 先修身份、数据来源和状态可信性，再完善资源操作、来源和导入体验 | 随受影响的 API 并行；[Frontend](./frontend/README.md)与第 4.8 节 Todo |
| Electron 桌面客户端 | Candidate / 不绑定版本 | 单一客户端管理本地服务、数据目录、连接、升级与诊断 | 第 5.1 节；[状态与传输](./frontend/state-and-transports.md)、[配置](./system/configuration.md) |

### 4.3 v0.7.2：生成质量

状态：Planned。版本目标（owner，2026-10-30）是生成质量：v0.7.1 让 Claude Code 的日常使用开始产生记忆，真实使用随即暴露出生成链路的问题，v0.7.2 由这些问题驱动 Patchouli 核心机制的改进。下列方向各自在阻塞问题决定、能够立即执行时建立计划，彼此不预设先后，计划导航见 [Plans](./plans/README.md#v072)；生成材料修正一份已于 2026-10-31 完成归档，实测随日常使用进行；plugin 模式开放 WRITE 与 UPDATE 一份已于同日完成实测验收并归档。

真实使用暴露的问题（证据见各 Todo 与 Idea）：

- 生成视图只剩导入原文的 8%：单轮上限截断了大部分工具过程；结算曾按 `worth_saving` 整轮丢弃 39% 的材料，2026-10-29 已停用筛选（已随生成材料修正解决）；
- 折叠裁剪的原文没有留存，结算冻结的来源只覆盖未折叠的轮次（已随生成材料修正解决）；
- 一个话题至多产出一条记忆，只写在正文中的内容检索不到；
- 查重分不开同一项目中相关与无关的记忆；同项目记忆增多后，查重判出的 UPDATE 整体覆盖已有记忆，已有两条记忆的主要内容在当前版本中被冲掉；
- plugin 模式只开放了检索与读取，Claude Code 的显式写入与更新从未发生；owner 判断上述生成问题约有一半源于此（已随 plugin 模式开放 WRITE 与 UPDATE 解决）。

| 方向 | 内容 | owner 决定（2026-10-30） | 设计讨论与阻塞问题 |
|:---|:---|:---|:---|
| 生成材料修正与原始证据保存 | 生成视图只保留单次工具调用结果的上限；移除 `worth_saving`；同步完成从 Topic 往后全链路的原始证据保存覆盖面（Topic 完整资料、来源冻结、来源 Artifact、Memory 的来源关联） | 一轮交互的内容不得被任何上限数字截断，只允许单次工具调用的结果受限，现行的单轮、思考与参数上限属于实现漂移；`worth_saving` 只做移除；计划同步完成 Topic 往后的原始证据覆盖面 | R-1、R-2 已决定（2026-10-30）：不分片，生成继续读工作视图；Topic 按顺序保存全部交互记录与折叠记录，作为来源 Artifact 的数据。已完成（2026-10-31）：[归档计划](./archive/plans/v0.7.2-generation-material-and-raw-evidence.md)；证据见[生成材料 Todo](./archive/todo/generation-material-filtering-and-truncation.md)（已归档）；当前事实见[感知与短期话题](./patchouli/perception.md)与 [Patchouli 生成](./patchouli/generation.md) |
| plugin 模式开放 WRITE 与 UPDATE | Claude Code 经 MCP 工具显式写入与更新记忆，经操作入口的请求级提交进入 actor API，派发到 Patchouli 的 Mode B、Mode C；记忆编译的命名与文案不再按 MTP 编写 | 同步开放两项能力，与主动链路共享同一记忆更新逻辑；计划同时清理 MTP 命名与文案残留，包括记忆编译文本（2026-10-31）；交互进行中的话题挂载采用写入时就地落位（E-8，2026-10-31） | [外部 Actor Idea](./ideas/external-actor-registration-and-runtime-access.md#v072plugin-模式开放-write-与-update)的 v0.7.2 一步；E-8 已决定。已完成（2026-10-31）：[归档计划](./archive/plans/v0.7.2-plugin-write-update.md)，以 Claude Code 的真实会话完成实测验收；当前事实见 [MCP 记忆访问](./contracts/mcp.md)与 [Import Bus](./system/import-bus.md#47-写入时落位) |
| 一个话题提取多条记忆 | 话题结算从整段材料中提取多条记忆，逐条查重与写入 | 版本内完成 | [多条提取与查重 Idea](./ideas/settlement-multi-memory-extraction-and-dedup.md)的 G-1 |
| 查重与更新机制 | 查重的判定信号，以及命中已有记忆后的写入方式 | 设计未决定 | 同一 Idea 的 G-2、G-3；证据见[查重 Todo](./todo/generation-dedup-decision-signal.md) |

验收方向：以 Claude Code 的真实会话验收，不只依靠离线测试。生成材料没有单轮截断，结算来源可以回查到已折叠的轮次；一个话题中的多项决定分别形成可检索的记忆；[多条提取与查重 Idea](./ideas/settlement-multi-memory-extraction-and-dedup.md#22-查重分不开相关记忆命中后整体覆盖)第 2.2 节的两条覆盖链等真实样本，在新的查重与更新机制下得到可解释的结果；Claude Code 显式写入与更新的结果经 Patchouli 生成链结算，能被之后的会话检索到。各计划的验收由计划自己冻结。

不在 v0.7.2：记忆价值体系的重新设计（第 4.7 节）；冷启动与历史导入（第 4.4 节）；重排模型 int8 与推理模型重新加载的停顿（Todo，未排期）；正文进入检索索引（搁置，见多条提取与查重 Idea 第 6 节）。

### 4.4 冷启动、历史导入与外源文档资产化（候选）

状态：Candidate，不绑定版本（2026-10-30 撤销原 `v0.7.2` 的版本号）。建议拆成两个可独立验收的 Plan，共享资源归属、来源和物化策略；确定性解析可以提前并行验证。

**冷启动与历史对话导入**解决“空库如何首次产生价值”和“既有会话如何保真迁入”。冷启动允许用户显式选择少量项目事实、偏好、Profile 或资料作为种子；不得自动把默认示例当作用户事实。历史导入保留 source、conversation/turn 标识、发生时间、说话者及分支/编辑关系；助手的推测或旧结论不能直接升级成用户当前事实。Claude Code 插件安装之前的会话从未导入，MCP 记忆访问的第一个检索落空样本即由此而来（[已归档的观察 Todo](./archive/todo/import-bus-real-use-observation.md)）。

导入过程支持预览、来源与 Workspace 选择、稳定导入标识、去重、进度/失败报告，以及只保留证据、稍后提炼记忆。它承接外部 Actor 接入契约的身份、来源和领域提交语义及历史样例缺口，另行定义批次、历史时间、冲突和恢复语义；复用 Patchouli 的物化入口，不简单按实时顺序把全部历史文本重放进当前活动话题，避免改写当前偏好或丢失历史时间关系。

**外源文档摄入**先形成 RAW/Artifact，再产生 representation、chunk、locator 和 evidence，最后按策略提炼候选与正式 Memory。复用 W1 的附件与 provenance 基础，并明确进程内 WorkspaceAsset 与持久化 Artifact 的区别。来源获取、确定性解析、模型辅助理解和记忆物化各自有成功/失败边界。

本地已给定文档的解析不依赖 Alice 或通用代码执行；需要网页获取、本地目录浏览或复杂转换时使用相应 provider。语义提炼可以调用模型或 Actor，但不能让文档注册、证据保存和读取依赖 Alice 的 run。

验收至少包含重复导入、部分失败、来源回查、跨 Workspace 隔离、旧信息与当前事实冲突，以及中断后的可解释处理。若承诺重启续跑，先交付导入 checkpoint 和所需任务持久化切片；现有进程内队列不足以承担该承诺。批量自动物化依赖第 4.7 节的价值体系，避免一次导入大量低质量或相互冲突的记忆。

### 4.5 MTP READ 专项编译与来源表达（候选）

状态：Candidate，不绑定版本（2026-10-30 撤销原 `v0.7.3` 的版本号）。基于 [v0.7.0 A2-P](archive/plans/v0.7.0-a2-pre-memory-version-and-lifecycle.md)的完整记忆历史，以及外源文档摄入（第 4.4 节）的来源、文档表示与版本契约，扩展 MemoryCompiler 对文档、代码、历史证据等资源的定向读取、片段定位、token 预算和引用呈现。先冻结实际需要的 READ 模式与错误语义，再扩展协议。完整历史存储、内容版本与 lifecycle 更新边界由 A2-P 提前交付，本方向不再反向定义记忆写入格式。

结果应说明读到哪个资源和版本、选择了哪个片段、引用如何回到原始证据，以及截断、缺失来源、旧裁剪历史和无权限如何表达。历史记录中的 lifecycle 是捕获时状态，额外展示当前评估时必须区分时点，不能用当前原子补造历史。不能仅在现有文本末尾追加一个链接，也不能建立第二套 provenance 或检索状态。其编译能力可由外部 adapter 复用，MTP 负责自己的协议呈现。已登记的缺陷 [MTP READ 截断正式记忆的正文](./todo/mtp-read-content-truncation.md)预计随本方向一并修复。

### 4.6 Deep Research MVP（候选）

状态：Candidate，不绑定版本（2026-10-30 撤销原 `v0.7.4` 的版本号）。研究领域持有 request、source、evidence、finding 和 report；协调器负责推进、重试与进度；执行 adapter 提供搜索、抓取、文件读取或分析。资源和研究状态不依赖 Alice 的 frame，Actor 可替换也不表示各 harness 的工具与策略完全等价。

首个 MVP 选择一个有预算和停止条件的真实研究任务，交付来源采集、证据定位、发现记录、报告 Artifact、取消和失败披露。以第 4.4、4.5 节的证据与读取结果作为报告依据，并复用既有队列；仅为验证过的研究流程增加最小协调状态，不预先建设通用 DAG 平台。

可先使用一个可靠的外部 harness 或工具 provider。若该路径不运行不可信本地代码，不把完整本地沙箱当作前置条件；若需下载并执行未知代码，先满足相应隔离门槛。执行提供者缺失时任务应明确等待或失败，不阻止已有研究资产读取。

研究状态跨 Actor 存续与跨进程恢复是两个承诺。恢复范围必须在正式 Plan 中冻结：若要在应用重启后继续，先持久化来源、进度与 checkpoint，并定义重放和重复副作用；在此之前只承诺实际支持的进程内暂停/继续。

### 4.7 记忆价值体系：重新设计（候选）

状态：Candidate，不绑定版本。owner（2026-10-30）将重新设计价值体系，而不是在现有信号上增补。

现状：入口不再产生价值信号。Gateway 的 WRITE/SKIP/UNKNOWN 预判与由它派生的 `worth_saving` 已随 v0.7.2 的生成材料修正移除（第 4.3 节），`PASSIVE_MEMORY` 也不再运行查询分析。此前 Perception settlement 曾按 `worth_saving` 整轮筛选材料，Claude Code 真实导入中这一预判只看到用户消息，把“请提交”“继续”背后的整轮实现丢出了生成，2026-10-29 起不再筛选（证据见已归档的[生成材料 Todo](./archive/todo/generation-material-filtering-and-truncation.md)）。正式 Memory 的生命周期评分另有依据，事实入口见 [Gateway Analysis](./gateway/analysis.md)、[Perception](./patchouli/perception.md)和 [Lifecycle](./patchouli/lifecycle.md)。

已有的约束：入口信号不能替代 Patchouli 的最终决定，也不应单独成为删除来源证据的依据；新体系不追溯性批量删除旧记忆，任何存量重评或迁移另有预览、审计和回滚范围。正式的 Idea 与 Plan 待建立。

### 4.8 前端与 Todo 并行工作流

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

### 4.9 依赖与验收门槛

```text
v0.7.1（已发布）：plugin 模式接入 Claude Code，真实使用开始产生样本
        |
        +-> v0.7.2 生成质量：生成材料与原始证据 / plugin 模式的写入与更新 / 多条提取 / 查重与更新

候选方向之间的依赖（用于判断一份计划能否立即执行，不表示排期）：
外部 Actor 的身份 / 来源 / 提交语义 -> 历史导入契约与批次能力
记忆价值体系 -> 冷启动与历史导入的批量自动物化
v0.7.0 A2-P 完整记忆历史 + 文档来源 / 版本 -> MTP READ 专项编译
可靠执行提供者 + 证据 / READ + 研究状态恢复契约 -> Deep Research
任务进程表与唯一注册入口 + 外部 Actor 接入契约 -> controller 模式
服务生命周期 / 传输 / 数据升级稳定 -> Electron 产品化
前端可靠性与资源体验 -> 随受影响的 API 并行交付
```

当前版本的计划见 [Plans 导航](./plans/README.md)。确定性导入可以提前并行，依赖复杂获取或转换的切片需等对应 provider。

每项候选进入实施前必须冻结范围、权威状态、数据来源、失败/取消/恢复承诺、幂等边界和观察指标。任何跨重启恢复、强隔离或 Actor 替换承诺都必须有对应实现证据。仍未具备的能力不能通过 UI、accepted 响应或事件日志伪装成完成。

## 5. 产品化与后置方向

### 5.1 Electron 桌面客户端

状态：Candidate，不绑定版本（2026-10-30 撤销原 `v0.8.x` 产品化窗口），正式 Plan 待建立；不是 Deep Research 或 Alice 高级编排的强制下游。完整发布之前可以做有限启动/连接原型，完整发布以资源生命周期、前端传输和数据管理稳定为门槛。

桌面化覆盖的工作多于 Web 页面外壳：本地 Python 服务及存储依赖的分发/连接、启动就绪、端口冲突、关闭 drain、数据目录、备份与升级、凭据、日志与崩溃诊断。HTTP、SSE、WebSocket 都需可配置地连接同一服务，不能仅迁移日志 URL。平台先选实际使用的一种完成闭环，再扩展安装包和签名发布。

Electron renderer 使用公开资源 API，main process 只持有必要的原生桥接与服务管理权限；远程内容和 Agent 代码不在特权主进程执行。Electron 的 renderer sandbox 不能替代 Agent 执行沙箱。

验收是可安装、启动、使用、关闭、重启和升级，用户知识资产持续可读；卸载/删除数据必须有明确语义。桌面生命周期引出的最小持久化需求按 [恢复治理](./governance/reliability/durability-and-recovery.md) 拆出，不依靠 localStorage 承担权威资源恢复。

### 5.2 Alice 高级编排

状态：Deferred，不绑定版本（2026-10-30 撤销原 `v0.8.0+` 后置窗口），不是已承诺的单一版本主题。plan-and-execute、多角色协作、parallel specialists、review loop 和任务图需经过 VISION 中 Alice 存在价值的判定，并与“外部 harness + 同一 Patchouli 记忆库”比较。

优先验证能证明记忆作为状态与控制输入有价值的能力，不因外部 Actor 接入而复制一套通用 harness 功能清单。

### 5.3 高级记忆生命周期

状态：Deferred，不绑定版本（2026-10-30 撤销原 `v0.9.0+` 方向）。Memory split/merge、完整 rollback、branch invalidation、provenance reverse index 和 L3 复活等待真实演化与恢复需求。记忆价值体系的重新设计不等于提前实现全部生命周期能力。

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

状态：Candidate，不绑定版本。2026-10-20 曾定为 v0.7.1 的版本目标，同日改为 plugin 模式（第 3.1 节）；首份计划的方向（首个外部驱动的纯文本闭环）随之撤回，计划未建立。步骤分析与阻塞问题保留在[外部 Actor Idea](./ideas/external-actor-registration-and-runtime-access.md#controller-模式的步骤分析退回候选方向)，Codex、Claude Code、pi 与 ACP 的接口资料见[接口调研快照](./research/harness-interface-survey-snapshot.md)（2026-10-20，模型实测待验证）；重新排期时先复核其中的未知项与待实测项。

controller 模式把 Codex、Claude Code、pi 一类成熟的 agent harness 作为执行环境与 CPU 类型纳入兼容范围：用户在对话开始前分别选择 agent 与执行环境（[执行环境 Idea](./ideas/execution-environment.md)前提 9、10；外部 Actor Idea E-6a），外部 harness 作为执行环境运行所选 agent，请求经唯一注册入口登记为任务进程，处理后交给该 harness 执行。它建立在 v0.7.0 的任务进程表与注册入口之上，消费外部 Actor 的接入与运行时访问契约（原计划 B），不强依赖 HiveMemory 本地沙箱；外部 harness 继续使用自己的工具和执行环境。采用 ACP 一类的通用协议还是各 harness 的 SDK 或非交互 CLI，见 Idea 的 E-2a。

验收必须形成真实闭环：请求登记为任务进程并由外部 harness 执行；外部 harness 经能力层检索已有记忆并实际用于任务，写入/修订意图经能力层提交；进程的交互记录进入记忆库（去向见 [任务进程](./workspace/task-process.md)）；下一次会话能召回新形成的知识并定位来源。取消、失败与进程回收按外部执行单元的进程问题（[任务进程 Idea](./ideas/task-process-table-and-registration-entry.md#q-8-外部执行单元的进程) Q-8）的决定呈现。仅转发请求、仅采集日志、仅发送事件或仅返回 memory context 均不足以证明 Actor 接入完成。

controller 模式下 HiveMemory 控制任务边界（actor 选择、输入、生命周期与取消、执行轨迹回流），不控制外部 harness 内部的 loop、工具与上下文压缩；外部 harness 默认保有自己的 prompt history，HiveMemory 不隐式接管上下文压缩。

## 6. 长期方向与评估

长期取舍遵循 [VISION](./VISION.md)：先证明开放记忆基础设施的实际价值，再验证 Alice 的 memory-native 运行能否带来额外收益。模型与 harness 的更换不应迫使用户重建长期知识资产；Actor 可替换性也不表示其执行过程或能力可以完全等价迁移。

评估逐步使用同一任务和记忆库比较无记忆、普通检索、外部 harness + Patchouli、Alice + Patchouli。记录跨会话复用、来源可查、错误与重复记忆、用户修正成本、执行可靠性、延迟和资源开销。新增功能、抽象层数和目录重排本身不作为成功指标。

## 7. 路线图维护规则

1. 只有 Git tag 对应版本标为 Released；本次重排不修改规范代码版本或构建产物。
2. 当前版本的功能在 Roadmap 写明目标、方向与验收方向；计划只在能够立即执行时绑定当前版本建立，不能立即实施的方向留在 Idea 候选池，不预先分配版本号或先后（owner，2026-10-30）；详细设计只维护一份。
3. 仅已完整完成实现、测试、迁移与审查的内容晋升到当前事实文档，并归档 Plan。
4. 部分落地保持 Partially Landed，已完成 Todo 不重新排期，Deferred 项只有满足触发条件才启动。
5. 排期变化同步 Plan 的 target、文件名、索引和反向引用；已完成历史文档保留原版本，当前路线提供迁移映射。
6. PROJECT、Architecture、System 和子系统索引的行为说明按文档治理收口门禁更新；单纯调整候选排期不提前向事实层推广设计。
7. 定期依据真实使用、导入样本和运行故障复核候选方向与下一个版本的功能；范围变化显式记录，不把前置能力缺口隐藏在上层功能中。
