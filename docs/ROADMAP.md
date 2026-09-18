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
last_reviewed: 2026-09-18
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

- 本次发布标签：`v0.6.2`（合并后创建）；
- 最近已发布基线：`v0.6.1`；
- 当前内容基线：`v0.6.2`，状态为 Completed（版本内容已收尾、相关计划与修复记录已归档）；[收尾审计](./archive/plans/v0.6.2-release-closeout-audit.md)记录内容核对与验证结果；
- 下一计划版本：`v0.7.0`，Workspace 资源平面重构计划组（A 协调入口，A1/A2/A5 Active，A3/A4/A6 Planned）、外部记忆服务与 Actor 交互契约（B，Planned）。

当前规范代码版本为 `0.6.2`，由 `src/hivememory/_version.py` 唯一声明并供构建与运行时复用。Python 包、前端清单与锁文件保持一致；本次发布标签为 `v0.6.2`，待合并后创建，最近已发布基线为 `v0.6.1`。

## 2. 发布历史

| 版本 | 状态 | 核心结果 | 当前依据 |
|:---|:---:|:---|:---|
| `v0.1.0-beta` / `v0.1.1` | Released | 记忆 MVP、API、基础前端与早期冷热路径 | Git tag；后续设计已演进 |
| `v0.2.0` | Released | 多 Agent 隔离、Agent Profile 记忆化 | Git tag；[Patchouli 当前设计](./patchouli/README.md)；[Alice 当前设计](./alice/README.md) |
| `v0.3.0` | Released | CALL、PendingAtom、Alice Orchestrator、MemoryCompiler | Git tag；[Alice 当前设计](./alice/README.md)；[MTP](./contracts/mtp.md) |
| `v0.4.0` | Released | chat run / memory task 取消控制与 RuntimeEvent | Git tag；[路由与事件](./contracts/routes-and-events.md) |
| `v0.5.0` | Released | artifact/provenance、MemoryLibrary、async-native、模型注册 | Git tag；[System 当前设计](./system/README.md)；[Patchouli 当前设计](./patchouli/README.md) |
| `v0.6.0` | Released | System Gateway、全局命令、主动/被动入口契约、Passive Ingress 串行与 outbox | Git tag；[Gateway 当前设计](./gateway/README.md)；[Passive Ingress 当前设计](./system/passive-ingress.md) |
| `v0.6.1` | Released | Local Work Queue Runtime、Active/Passive Interaction Submission 统一接入、Memory Generation queue 与进程内可靠生命周期 | Git tag；[System Runtime 当前设计](./system/runtime-and-bus.md#3-local-work-queue-runtime)；[归档实施计划](./archive/plans/v0.6.1-local-work-queue-runtime.md) |

过去文档中的 `v0.5.1`、`v0.5.2`、`v0.5.3` 是 v0.5 开发期的内部工作批次，不是当前仓库中的独立发布标签。它们的已实现事实应按模块并入当前文档，而不是继续作为平行版本入口。

## 3. 最近已发布基线：v0.6.1

主题：**Reliable Local Work Runtime**。

### 3.1 发布内容

- Local Work Queue 已建立不可变 `WorkItem`、权威 `WorkRecord`、状态机、lane、versioned codec 与 handler registry；
- `InMemoryWorkStore`、Runtime 与 Supervisor 统一提供 enqueue、claim、retry wait、timeout、cancel、backpressure 和 shutdown drain；
- Passive Interaction Submission 已迁移到通用 submission lane，admission 成功后才 commit/reset accumulator；
- Active finalize 已复用同一 submission queue，并以同步 applied gate 作为继续后续副作用与返回成功的边界；
- Active/Passive 使用稳定 `interaction_id`、canonical payload 与 topic/conversation ordering key；
- Memory Generation 已接入独立业务 lane，保留 list/get/wait/cancel 与领域事件，通过 typed handle 投影 WorkRecord；
- Interaction Submission 与 Memory Generation 保持独立 payload、成功条件、retry classifier、capacity 与取消策略；
- RuntimeEvent 只投影状态转换，sink 失败不改变业务结果；
- Durability D0、Idempotency I0、Identity S0 和数据模型 Phase I 四项前置基线已经建立；
- 相关单元、集成、模糊失败、capacity、取消与 shutdown 行为均有测试证据。

当前事实入口：

- [系统架构概览](./architecture/overview.md)
- [系统边界与所有权](./architecture/boundaries.md)
- [子系统公共契约](./contracts/subsystem-contracts.md)
- [公开路由与事件](./contracts/routes-and-events.md)
- [System 当前设计](./system/README.md)
- [System Runtime 与 Work Queue](./system/runtime-and-bus.md#3-local-work-queue-runtime)
- [Passive Ingress 当前设计](./system/passive-ingress.md)
- [Patchouli 记忆生成](./patchouli/generation.md)
- [Gateway 当前设计](./gateway/README.md)
- [Patchouli 当前设计](./patchouli/README.md)
- [Alice 当前设计](./alice/README.md)
- [Frontend 当前设计](./frontend/README.md)
- [Help](./help/README.md)
- [Applications](./applications/README.md)
- [v0.6.1 归档实施计划](./archive/plans/v0.6.1-local-work-queue-runtime.md)

### 3.2 发布范围边界

- v0.6.1 的可靠性承诺限定为单进程、单 event loop 的进程内执行生命周期，不构成跨重启可靠交付；
- SQLite WorkStore、claim ownership、lease recovery 和数据库级唯一 idempotency key 不属于本版本，已进入持久化治理；
- Runtime 多 lane 抽象当前保留，但生产组件仍按业务 queue 分别装配 Runtime/Store；拓扑重构等待真实触发条件；
- priority、用户任务 API、定时/hook workflow、DAG 和 outcome artifact 不属于本版本；
- Memory Generation 含领域副作用的数据面固定单次 attempt，不因通用 Runtime 支持 retry 就自动重放；
- queue FIFO、topic append order 与 Agent causal order 是不同保证，v0.6.1 不宣称已经解决因果排序；
- v0.6.1 未回溯改写 v0.6.0 Passive Ingress 的公共契约。

### 3.3 v0.6.1 发布验收

- Work Queue 公共协议与状态机不依赖 Patchouli、Alice 或 server 业务模型；
- Interaction Submission 与 Memory Generation 使用独立 lane，不共享 payload、成功条件或 retry classifier；
- capacity 满时明确拒绝，不静默丢弃已接纳 work；
- Active finalize 只有在 interaction work `SUCCEEDED` 后才执行 materialization/HIT 等后续副作用；
- Passive admission 失败保留 payload，重复提交与 retry 复用稳定 `interaction_id`；
- Memory Generation 的 concurrency、queued/running cancel、timeout、wait/list/get 和 shutdown drain 保持一致投影；
- at-least-once、业务幂等、模糊失败、RuntimeEvent isolation 和进程内 accepted 边界均有测试与文档；
- 当前设计、README、Python 包、前端清单和 `v0.6.1` tag 口径一致；
- Release workflow 同时校验并发布 backend wheel、sdist、frontend archive 和合并校验文件。

## 4. 近期计划

### 4.1 排序原则与版本调整

本轮规划依据 [VISION](./VISION.md) 的双轨策略：Workspace/Patchouli 提供可独立使用的资源与记忆基础设施；Alice 是 memory-native reference runtime，外部 harness 可以通过适配器成为其他 Actor。先验证资源可复用、执行可替换，再推进文档、研究和桌面产品化。

“脱离 Actor”在此指资源身份、授权、证据、结果和任务状态不依赖某个特定 Actor。确定性解析可由普通服务完成；搜索、浏览或代码执行可以调用工具 worker 或 Actor。Deep Research 的执行仍需要执行者，但研究状态和结果应在执行者离开后继续存在。外部 harness 无需采用 Alice 的 loop、run/frame 工作集或完整 MTP 语法；主动写入的 Pending 读取与结算属于共同资源能力。

版本重新安排如下；这些是目标与候选排期，不改变当前规范代码版本、Git tag 或已完成阶段的归属：

| 原排期 | 新排期 | 调整原因 |
|:---|:---|:---|
| `v0.6.2` Workspace 资源体系与 Agent 执行边界 | `v0.7.0` A 计划组 / B，状态见第 4.4 节 | 内部重构拆为 A1–A6，外部服务与协议由 B 承接，共同构成本版本，独立于 W0/W1 收口 |
| `v0.7.0` Document Ingestion & Provenance | `v0.7.2`，Candidate | 建立在新资源边界上，并纳入冷启动、历史对话导入和证据资产化 |
| `v0.7.1` MTP READ Provenance | `v0.7.3`，Candidate | 消费已经稳定的文档表示、来源和版本契约 |
| `v0.7.2` Deep Research MVP | `v0.7.4`，Candidate | 等待资源、证据、读取编译和可靠执行的闭环 |
| 整体后置的 Executable Asset Sandbox | `v0.7.1` 执行基座前置；完整强隔离继续按条件推进 | 先解决 MTP RUN 可复现的可靠性问题和可兑现的能力边界 |

旧 Idea、W1 实施记录及历史材料中出现的上述旧版本号，按本表解释其后续排期；不重写历史使其看起来曾采用新顺序。新建 Plan 必须使用新目标版本。

除已有正式计划组的 `v0.7.0` 外，下列新增工作仍为 Candidate，实施前分别建立范围、迁移、测试与验收方案。版本号表达交付顺序；互不依赖的小切片可以并行验证，不能以并行开发跳过契约冻结。

### 4.2 v0.6.2：内容已收尾

状态：Completed。W0、Identity 收敛、V1 Memory 迁移、W1、Topic 存储边界整理与 MTP scope 修复已完成内容核对；相关计划和修复记录均已归档，规范版本已调整至 0.6.2，本次发布标签为 v0.6.2（合并后创建）。资源体系重构归属 v0.7.0。

| 切片 | 已形成的基础 | 事实与历史入口 |
|:---|:---|:---|
| W0 Workspace MVP | WorkspaceIdentity、默认 main_workspace、端到端 scope、双 Workspace 隔离、进程级 WorkspaceAssetStore、两级状态机与 TopicAssetBinding | [Workspace 架构](./architecture/workspace.md)、[W0 归档 Plan](./archive/plans/v0.6.2-workspace-mvp.md) |
| Identity 投影收敛 | 服务入口统一 IdentityScope、身份解析入口收口、actor 值对象化、管理与检索可见性分离 | [System 应用服务](./system/application-services.md)、[Identity 归档 Plan](./archive/plans/v0.6.2-identity-projection-cleanup.md) |
| V1 Memory Legacy 迁移 | 已有 V1 记录迁入 canonical v2，移除 legacy 解释分支 | [数据模型](./architecture/data-model.md)、[迁移归档 Plan](./archive/plans/v0.6.2-v1-memory-legacy-migration.md) |
| W1 Chat Attachments | 上传、确定性解析、READY/FAILED、选择与 lease、AttachmentCompiler、Topic binding、按需 Artifact promotion | [附件链路](./system/attachments.md)、[W1 归档 Plan](./archive/plans/v0.6.2-w1-chat-attachments.md) |
| MTP scope 修复 | L0 pending 与 L1/L2 atom 查找重验 scope，越权按不可见处理 | [MTP 契约](./contracts/mtp.md)、[归档修复记录](./archive/todo/mtp-cache-scope-revalidation.md) |
| Topic 与短期存储边界 | TopicWorkingSet、lease 与短期 CRUD/快照责任收敛 | [Perception](./patchouli/perception.md)、[归档 Plan](./archive/plans/short-term-memory-store-boundary-cleanup.md) |
| 旧缓存迁移及后续所有权调整 | Workspace-aware cache key 已形成；随后按 ADR-0004 由 AliceRuntime 持有，WorkspaceRuntime 聚合解体 | [旧迁移归档 Plan](./archive/plans/v0.6.2-workspace-runtime-cache-migration.md)、[ADR-0004](./architecture/decisions/0004-execution-path-derived-caches.md) |

本次收尾已核对上述范围的测试、事实文档、归档和版本产物一致性，结果见 [收尾审计](./archive/plans/v0.6.2-release-closeout-audit.md)。W0/W1 不因此重新承担外源文档全文摄入、强沙箱或外部 harness 完整接入。已完成的 V1 数据转换也不同于未来的历史对话导入。

### 4.3 目标总览

| 目标版本或工作流 | 状态 | 目标结果 | 依赖与实施入口 |
|:---|:---:|:---|:---|
| `v0.7.0` A：Workspace Resource Plane Refactor | Active | 以协调计划统筹 A1–A6：访问边界、资源读取与 WorkspaceRuntime/cache、Session/Topic 投影、共享 Pending、统一 API 收敛、Actor 适配与集成收口；不新增平行业务 port/provider 层 | v0.6.2 基础；[计划 A 协调入口](./plans/v0.7.0-workspace-resource-system-and-agent-execution-boundaries.md) |
| `v0.7.0` B：External Memory Service & Actor Interaction | Planned | 被动对话与主动资源交互协议、身份与来源、领域提交和结果查询；参考客户端闭环与历史样例验证 | A 的访问边界与公开 application 契约；[计划 B](./plans/v0.7.0-external-memory-service-and-actor-interaction.md) |
| `v0.7.1` Execution Substrate & Sandbox Baseline | Candidate | MTP RUN 可靠执行、工具 provider、超时取消、显式文件/网络/进程能力边界 | v0.7.0 A 的访问契约与工具执行适配边界；[MTP 当前设计](./alice/mtp-runtime.md)、[执行安全治理](./governance/security/identity-and-execution-safety.md)；正式 Plan 待建立 |
| `v0.7.1` 首个真实外部 harness 接入 | Candidate / 可独立交付 | 外部 harness 实际使用记忆并把结果送回 Patchouli，验证无 Alice 的跨会话闭环 | v0.7.0 B 的外部交互协议；[Passive Ingress 当前设计](./system/passive-ingress.md)；正式 Plan 待建立 |
| `v0.7.2` Cold Start, Historical Import & Document Ingestion | Candidate | 冷启动种子、历史对话导入、文档解析、证据与候选记忆分层，冻结 provenance | A 的资源边界、B 的身份/来源/提交契约、记忆价值策略最小切片；[Artifacts](./patchouli/artifacts.md)、[附件 Idea](./ideas/workspace-mvp-chat-attachments-design.md)；导入与文档分别建 Plan |
| `v0.7.3` MTP READ 专项编译与来源表达 | Candidate | 按资源类型、版本、定位和预算编译 READ 输出，给出可核验引用 | v0.7.2 provenance；[MTP](./contracts/mtp.md)、[MemoryCompiler](./patchouli/memory-compiler.md)；正式 Plan 待建立 |
| `v0.7.4` Deep Research MVP | Candidate | 研究状态、来源、证据、发现和报告闭环，执行提供者可替换 | 资源边界、Document/READ、实际执行能力、明确恢复范围；正式 Plan 待建立 |
| 记忆价值策略重设计 | Candidate / 跨版本 | 先冻结入口信号与持久化决策边界，再用真实样本校准 | v0.7.0 期间启动分析，v0.7.2 批量物化前交付最小策略；[Gateway](./gateway/analysis.md)、[Perception](./patchouli/perception.md)、[Lifecycle](./patchouli/lifecycle.md) |
| Frontend Reliability & Resource UX | Partially Landed / 后续 Candidate | 先修身份、数据来源和状态可信性，再完善资源操作、来源和导入体验 | 可与 v0.7 并行；[Frontend](./frontend/README.md)与第 4.10 节 Todo；正式 Plan 待建立 |
| Electron 桌面客户端 | Candidate / `v0.8.x` 产品化窗口 | 单一客户端管理本地服务、数据目录、连接、升级与诊断 | 前端传输与资源生命周期稳定；[状态与传输](./frontend/state-and-transports.md)、[配置](./system/configuration.md)；正式 Plan 待建立 |

### 4.4 v0.7.0：Workspace 资源平面与外部交互计划组

本版本由计划 A 的六个内部子计划和计划 B 共同承接。A 的协调入口只维护依赖和共同出口，各子计划有独立的目标、迁移、测试和验收；以下为规划目标，未表示全部实现已完成。

- [计划 A 协调入口](./plans/v0.7.0-workspace-resource-system-and-agent-execution-boundaries.md)将六份计划按交付依赖排序：A1 建立四阶段授权和 operation 绑定基线；A2 交付 canonical/Profile 读取、快照、WorkspaceRuntime/cache 和失效；A3 交付 Session/Segment/Part、Topic 生命周期与交互/资料公共路由；A4 交付共享 Pending、主动提交及完整引用解析；A5 在前置模型与路由齐备后收敛整体 API、补检索/使用报告差额并明确旧服务职责；A6 切换真实消费者、稳定装配和 shutdown。各计划继续使用同一 GlobalSystemBus 领域链，不增加 Workspace 业务 port/provider 层。
- 默认实施顺序为 [A1 访问边界](./plans/v0.7.0-a1-workspace-access-boundary.md) → [A2 资源读取与缓存](./plans/v0.7.0-a2-workspace-resource-reads-and-caches.md) → [A3 Session/Topic](./plans/v0.7.0-a3-conversation-session-and-topic-projection.md) → [A4 Pending/主动写入](./plans/v0.7.0-a4-pending-memory-intents.md) → [A5 API 收敛](./plans/v0.7.0-a5-patchouli-unified-api.md) → [A6 消费者集成](./plans/v0.7.0-a6-actor-adapters-and-integration.md)。A2/A3 可在 A1 后并行，A4 消费二者成果。领域计划各自公开并验证 API，靠前计划不以靠后计划的契约或实现作为完成门禁。A3 仍不包含完整折叠算法，后者由[话题折叠专项](./plans/topic-folding-context-and-raw-evidence.md)承接。

- [计划 B：外部记忆服务与 Actor 交互契约](./plans/v0.7.0-external-memory-service-and-actor-interaction.md)消费 A 的访问边界、公开 API 及共享 Pending，定义被动对话与主动资源交互协议，补齐身份、来源、提交关联、物化前读取和结算解析，用参考客户端完成无 Alice 的闭环。Passive 保留被动摄入职责；外部 Actor 无需创建 Alice Runtime 或运行 frame，MCP 等协议适配不另建 Pending 状态机。历史样例用于验证后续导入契约，完整批次导入仍后置。

A 系列的共同验收是“无 Alice 可使用资源服务”“Workspace 与执行状态不串扰”“合法 mutation 后缓存一致”“Session 与 Topic 各自承担正确生命周期”“Pending 写后可读、结算可解析”“保留最后 settlement 的 shutdown drain”。A1–A5 不等待完整外部协议；A6 使用真实内部组件完成集成。B 的验收覆盖公开接口的读取、提交、结果观察和再次读取，不能以 fake 代替集成证据。A/B 可分别收口，共同发布 v0.7.0 前两组出口均须完成。A 系列与 B 均不包含完整沙箱、研究编排或特定厂商连接器，也不承诺所有资源与任务已持久化。

统一 API 以 System、Alice、外部 Actor 的共同操作为依据，详细映射见[A5](./plans/v0.7.0-a5-patchouli-unified-api.md)；三方 adapter 对相同操作调用相同总线路由，Patchouli application 继续按处理领域实现。A3 解决外部确定性 Session 与内部 Topic 的数据边界，A4 解决主动写入的 Pending 一致性；B 区分主动工具调用和自动交接，完整交互由 adapter 自动提交，不依赖模型选择保存工具，也不因外部接入新增另一套资源 API。

2026-09-17 规划裁定：按 [A2 第 1.1 节](./plans/v0.7.0-a2-workspace-resource-reads-and-caches.md)明确 ADR-0004 的继承/替换范围，通用缓存从 Alice 客户端加速转为 Patchouli 读取链内部使用的基础设施，由单一 WorkspaceRuntime 聚合。A2 交付 canonical 读取，A4 扩展统一 Pending/canonical 引用解析，A6 删除 Alice 独立 resolver 路径；Profile 定义读取留在资源侧，执行配置和 system prompt 应用留在各 Actor。外部 harness 无需支持动态 Profile 即可使用记忆服务。ADR-0004 仍记录 v0.6.2 已落地基线，正式后继 ADR 在 A6 联合验收收尾后建立；该安排不改变后续版本排期。

### 4.5 v0.7.1：执行基座与真实外部 Actor 两个独立切片

#### 4.5.1 执行基座与沙箱基线

状态：Candidate。用户实测 MTP RUN 经常失败是启动调查的依据，正式 Plan 首先收集复现样例，区分协议解析、alias/权限、参数编译、环境、依赖、工具进程与取消链路的问题；不预设所有故障都由缺少沙箱引起。

目标是统一执行 provider 的环境绑定、stdout/stderr、退出码、结构化结果、超时、取消和清理语义。Workspace 声明资源和可用能力，执行 provider 兑现文件、网络、进程与资源限制，Actor 选择调用；工具执行不能反向占有资源系统。

验收覆盖正常结束、工具异常、启动失败、超时、取消、子进程清理和不同 Workspace 环境隔离。受信任执行、工具 API 限制和 OS 级强隔离必须明确区分：工作目录、prompt 或 Python 包装层不足以限制任意代码。若第一版无法实施相应隔离，不接纳要求该隔离等级的任务。完整不可信代码沙箱保留独立的实现与验证门槛。

#### 4.5.2 首个真实外部 harness 接入

状态：Candidate，目标窗口为 v0.7.1。它消费 v0.7.0 计划 B 已验证的外部交互协议，不强依赖 HiveMemory 本地沙箱；外部 harness 可继续使用自己的工具和执行环境。优先选一个用户实际使用且有可用接入方式的 harness，具体通过 API、MCP、skill、hook 或 connector 由样本与支持接口决定。

验收必须形成真实闭环：已有记忆被检索并实际用于外部任务；对话经 Passive Ingress 回流，显式写入/修订经主动领域提交入口处理；下一次会话能召回新形成的知识并定位来源。仅采集日志、仅发送事件或仅返回 memory context 均不足以证明 Actor 接入完成。

接入需把具体平台的身份、事件顺序、重发、turn 结束和来源映射到 B 的契约，验证平台能力不足时的失败披露。外部 harness 默认保有自己的 prompt history；HiveMemory 不隐式接管上下文压缩。B 定义的 ingress、交互应用和记忆物化状态继续分开；connector 不重写领域状态机，也不扩大已有幂等与持久化承诺。

### 4.6 v0.7.2：冷启动、历史导入与外源文档资产化

状态：Candidate，建议拆成两个可独立验收的 Plan，共享资源归属、来源和物化策略；确定性解析可在 v0.7.1 执行基座建设期间并行验证。

**冷启动与历史对话导入**解决“空库如何首次产生价值”和“既有会话如何保真迁入”。冷启动允许用户显式选择少量项目事实、偏好、Profile 或资料作为种子；不得自动把默认示例当作用户事实。历史导入保留 source、conversation/turn 标识、发生时间、说话者及分支/编辑关系；助手的推测或旧结论不能直接升级成用户当前事实。

导入过程支持预览、来源与 Workspace 选择、稳定导入标识、去重、进度/失败报告，以及只保留证据、稍后提炼记忆。它承接 v0.7.0 B 的身份、来源和领域提交契约及历史样例缺口，另行定义批次、历史时间、冲突和恢复语义；复用 Patchouli 的物化入口，不简单按实时顺序把全部历史文本重放进当前活动话题，避免改写当前偏好或丢失历史时间关系。

**外源文档摄入**先形成 RAW/Artifact，再产生 representation、chunk、locator 和 evidence，最后按策略提炼候选与正式 Memory。复用 W1 的附件与 provenance 基础，并明确进程内 WorkspaceAsset 与持久化 Artifact 的区别。来源获取、确定性解析、模型辅助理解和记忆物化各自有成功/失败边界。

本地已给定文档的解析不依赖 Alice 或通用代码执行；需要网页获取、本地目录浏览或复杂转换时使用相应 provider。语义提炼可以调用模型或 Actor，但不能让文档注册、证据保存和读取依赖 Alice 的 run。

验收至少包含重复导入、部分失败、来源回查、跨 Workspace 隔离、旧信息与当前事实冲突，以及中断后的可解释处理。若承诺重启续跑，先交付导入 checkpoint 和所需任务持久化切片；现有进程内队列不足以承担该承诺。批量自动物化先完成第 4.9 节最小价值策略，避免一次导入大量低质量或相互冲突的记忆。

### 4.7 v0.7.3：MTP READ 专项编译与来源表达

状态：Candidate。基于 v0.7.2 的来源、表示与版本契约，扩展 MemoryCompiler 对文档、代码、历史证据等资源的定向读取、片段定位、token 预算和引用呈现。先冻结实际需要的 READ 模式与错误语义，再扩展协议。

结果应说明读到哪个资源和版本、选择了哪个片段、引用如何回到原始证据，以及截断、缺失来源和无权限如何表达。不能仅在现有文本末尾追加一个链接，也不能建立第二套 provenance 或检索状态。其编译能力可由外部 adapter 复用，MTP 负责自己的协议呈现。

### 4.8 v0.7.4：Deep Research MVP

状态：Candidate。研究领域持有 request、source、evidence、finding 和 report；协调器负责推进、重试与进度；执行 adapter 提供搜索、抓取、文件读取或分析。资源和研究状态不依赖 Alice 的 frame，Actor 可替换也不表示各 harness 的工具与策略完全等价。

首个 MVP 选择一个有预算和停止条件的真实研究任务，交付来源采集、证据定位、发现记录、报告 Artifact、取消和失败披露。以 v0.7.2/3 的证据与读取结果作为报告依据，并复用既有队列；仅为验证过的研究流程增加最小协调状态，不预先建设通用 DAG 平台。

可先使用一个可靠的外部 harness 或工具 provider。若该路径不运行不可信本地代码，不把完整本地沙箱当作前置条件；若需下载并执行未知代码，先满足相应隔离门槛。执行提供者缺失时任务应明确等待或失败，不阻止已有研究资产读取。

研究状态跨 Actor 存续与跨进程恢复是两个承诺。恢复范围必须在正式 Plan 中冻结：若要在应用重启后继续，先持久化来源、进度与 checkpoint，并定义重放和重复副作用；在此之前只承诺实际支持的进程内暂停/继续。

### 4.9 记忆价值策略：跨版本重设计

状态：Candidate。v0.7.0 期间可启动分析与样本收集，v0.7.2 大规模自动物化前完成最小策略，后续根据真实使用反馈迭代。

当前已有 Gateway 的 WRITE/SKIP/UNKNOWN 预判、worth_saving 投影，以及 Perception settlement 对材料的筛选；正式 Memory 的生命周期评分另有依据。不能把现状简化成“缺少三态布尔值”，也不能用一个总分替代所有层次。事实入口见 [Gateway Analysis](./gateway/analysis.md)、[Perception](./patchouli/perception.md)和 [Lifecycle](./patchouli/lifecycle.md)。

规划分两个切片：

1. **决策边界与最小策略**：区分原始证据保留、候选提炼、正式记忆写入、长期维护价值；明确用户显式选择、不同内容类型、来源置信度、时效和重复冲突的处理，以及 UNKNOWN 的保守语义。入口信号不能替代 Patchouli 最终决定，也不应单独成为删除来源证据的依据。
2. **样本与反馈校准**：使用真实外部会话、历史导入和日常使用样本，评估关键事实漏存、错误/重复记忆、过时信息、用户修正成本、后续召回收益和模型开销。记录决策理由与策略版本，允许对同一批证据重评，避免每次策略调整都重新导入。

权责与评估明确后再决定是否需要多维评分或新字段。新策略不追溯性批量删除旧记忆；任何存量重评或迁移另有预览、审计和回滚范围。正式 Plan 待建立。

### 4.10 前端与 Todo 并行工作流

Frontend 已部分落地，后续优化按可用性和资源操作体验推进，详细事实见 [Frontend](./frontend/README.md)。本轨不阻塞后端基础重构，但与受影响 API 一起验收。

- **近期可靠性**：统一身份状态，明确真实/mock/stale/error，修复失败披露和乐观更新，覆盖 HTTP/SSE/WebSocket 的一致连接配置与请求终态。
- **资源操作体验**：Memory Garden 真实语义检索、可见性策略、附件/来源检查、导入预览与失败重试。Workspace 选择 UI 只在对应后端能力真实可用后开放。
- **交互和视觉完善**：布局、导航、窄屏、可访问性、空状态及主题；避免将未接线的 Terminal 或研究页面显示成已有功能。

Todo 排期按已核对状态和实际依赖吸收，不能把目录中所有事项视为新需求：

| Todo | 排期方式 | 范围约束 |
|:---|:---|:---|
| [MTP 缓存作用域重验（归档）](./archive/todo/mtp-cache-scope-revalidation.md) | Completed / Archived；保留为 v0.7.0 回归基线 | 已修复的 L0/L1/L2 隔离语义不重排为待开发功能 |
| [全局路由签名校验](./todo/global-route-signature-consistency-check.md) | v0.7.0 契约迁移时纳入质量切片 | 检测参数不匹配；不重做路由系统 |
| [Alice 健康探针](./todo/alice-health-probes.md) | v0.7.0 生命周期切片 | 反映执行器实际就绪/故障；不令资源 readiness 依赖 Alice 在线 |
| [RuntimeEvent 生产端迁移](./todo/runtime-event-producer-migration.md) | v0.7.0/1 按受影响生产域迁移 | 保持事件只做观测，不以事件投递成功控制业务状态 |
| [前端身份所有权](./todo/frontend-identity-ownership.md) | 前端可靠性优先项 | 不把固定 main_workspace 选择冒充认证或 Workspace 管理 |
| [Mock fallback 披露](./todo/frontend-mock-fallback-disclosure.md) | 前端可靠性优先项 | 可辨认的数据来源与可兑现的写操作 |
| [Memory Garden 语义检索](./todo/frontend-memory-semantic-search.md) | 前端资源体验 | 复用真实后端能力，保留失败和空结果 |
| [Memory visibility policy UI](./todo/memory-visibility-policy-ui.md) | 后端契约稳定后进入资源体验 | UI 编辑策略不能替代后端授权 |
| [Page Folding 跨入口后续](./todo/page-folding-cross-ingress-follow-ups.md) | 由[话题折叠、Actor 上下文与原始证据计划](./plans/topic-folding-context-and-raw-evidence.md)集中承接，Planned / 占位；发布版本待详细设计确定 | 统筹折叠算法、原文证据与长 turn 两份 Idea；本次仅建立独立里程碑，不扩大 v0.7.0 A/B 的发布范围 |
| [Topic /compact](./todo/topic-compact-command-ingress.md) | 独立小切片，可并行 | 不作为 Workspace、文档或 Research 的统一前置 |
| [Work Queue 多 lane 拓扑](./todo/work-queue-runtime-lane-topology.md) | 保持 Deferred | 只有共享 Store、跨 lane 调度或可复现故障触发才重评 |

上述排期摘要不代替 Todo 本体的完成条件。实现时先复核其是否已解决；跨系统范围扩大则建立 Plan，不通过“清理 Todo”引入未评审的新状态机。

### 4.11 依赖与验收门槛

```text
v0.6.2 已实现基础 -> v0.7.0 A：Workspace 资源与内部执行边界
                          |
                          +-> v0.7.0 B：外部记忆服务与 Actor 交互
                          |       +-> v0.7.1 真实外部 harness 闭环
                          |       |          (不强依赖本地沙箱)
                          |       +-> v0.7.2 历史导入契约与批次能力
                          +-> v0.7.1 本地执行基座
                          +-> v0.7.2 冷启动 / 文档资源与证据

v0.7.2 来源 / 证据 / 版本契约 -> v0.7.3 READ 专项编译
可靠执行提供者 + 证据 / READ + 研究状态恢复契约 -> v0.7.4 Deep Research

价值策略最小切片 -> v0.7.2 批量自动物化
前端可靠性与资源体验 -> 随 v0.7 各能力并行交付
服务生命周期 / 传输 / 数据升级稳定 -> Electron 产品化
```

A1 先完成访问基线，A2/A3 交付读取和交互/Topic，A4 交付主动意图与完整读取，A5 再核对完整 API；B 可先调查场景，随后按这些交付分批冻结外部协议并以真实领域能力验收。A6 负责既有消费者及稳定生产装配。A 系列不等待 B 的完整协议；v0.7.0 发布同时核对 A1–A6 与 B 的出口。确定性导入可与执行基座并行，依赖复杂获取或转换的切片需等对应 provider。真实外部接入越早形成样本，越能帮助价值策略和导入计划减少猜测。

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

### 5.4 完整不可信代码沙箱

状态：Deferred 的剩余工作包；v0.7.1 已提取执行可靠性和可实施的限制基线。后续按实际威胁与负载选择隔离机制，验证宿主文件、网络、进程、资源配额、子进程终止和跨 Workspace 逃逸边界。不以目录配置或工具包装宣称强隔离。

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
| 其他历史存储转换 | Unscheduled；若发现仍有未迁移存量，先列证据再建迁移 Plan | 已完成的 [V1 迁移历史](./archive/plans/v0.6.2-v1-memory-legacy-migration.md)不重新列为待办 |

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
