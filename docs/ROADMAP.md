---
title: HiveMemory Roadmap
status: current
owner: project
scope: releases-and-planned-capabilities
updates:
  - docs/PROJECT.md
  - docs/plans/
  - docs/archive/plans/
last_reviewed: 2026-08-06
---

# HiveMemory 开发路线图

本文只维护版本阶段、当前完成度、依赖关系和相关设计入口。已经生效的详细设计进入 Architecture、System 或子系统文档；尚未实现的详细方案进入 Plans；完成后的实施文档进入 Archive。

路线图是一条能力依赖链，不是功能愿望清单。排序首先回答“下一项能力需要哪些已经可信的状态、契约和证据”，其次才考虑它是否显眼或易于演示。HiveMemory 的长期方向涉及异步任务、文件、文档、来源和研究，如果这些能力各自建立一套临时状态与失败语义，规模越大反而越难保持记忆闭环。因此近期路线优先补齐可复用底座，再让上层能力逐步消费它们。

## 1. 状态口径

| 状态 | 含义 |
|:---|:---|
| Released | 已有对应 Git tag 的发布版本 |
| Current Development | 当前开发基线，主体可能已合并但尚未发布 |
| Planned | 已形成目标和大致边界，尚未成为当前事实 |
| Candidate | 候选排期，范围和顺序仍可调整 |
| Unscheduled | 已确认问题或方向，但尚未进入具体版本承诺 |
| Partially Landed | 阶段中的一部分已进入当前实现，其余仍未完成 |
| Deferred | 已明确后置，不属于近期承诺 |

当前版本事实如下：

- 最新已发布标签：`v0.6.0`；
- 当前发布基线：`v0.6.0`；
- 下一计划版本：`v0.6.1`，状态为 Planned。

当前规范代码版本同为 `0.6.0`，由 `src/hivememory/_version.py` 唯一声明并供构建与运行时复用。发布基线必须与完全匹配的 Git tag、Python 包、前端清单和构建检查保持一致；后续开发工作进入下一版本时，再在新的提交中更新代码版本。

## 2. 发布历史

| 版本 | 状态 | 核心结果 | 当前依据 |
|:---|:---:|:---|:---|
| `v0.1.0-beta` / `v0.1.1` | Released | 记忆 MVP、API、基础前端与早期冷热路径 | Git tag；后续设计已演进 |
| `v0.2.0` | Released | 多 Agent 隔离、Agent Profile 记忆化 | Git tag；[Patchouli 当前设计](./patchouli/README.md)；[Alice 当前设计](./alice/README.md) |
| `v0.3.0` | Released | CALL、PendingAtom、Alice Orchestrator、MemoryCompiler | Git tag；[Alice 当前设计](./alice/README.md)；[MTP](./contracts/mtp.md) |
| `v0.4.0` | Released | chat run / memory task 取消控制与 RuntimeEvent | Git tag；[路由与事件](./contracts/routes-and-events.md) |
| `v0.5.0` | Released | artifact/provenance、MemoryLibrary、async-native、模型注册 | Git tag；[System 当前设计](./system/README.md)；[Patchouli 当前设计](./patchouli/README.md) |
| `v0.6.0` | Released | System Gateway、全局命令、主动/被动入口契约、Passive Ingress 串行与 outbox | Git tag；本路线图 3.1；[Passive Ingress 当前设计](./system/passive-ingress.md) |

过去文档中的 `v0.5.1`、`v0.5.2`、`v0.5.3` 是 v0.5 开发期的内部工作批次，不是当前仓库中的独立发布标签。它们的已实现事实应按模块并入当前文档，而不是继续作为平行版本入口。

## 3. 当前发布：v0.6.0

主题：**System Gateway、全局命令与 Passive Ingress**。

### 3.1 发布内容

- Gateway 已从旧引擎角色上移为 `GatewaySystem` 同级子系统；
- `gateway.public.process` 成为主动/被动入口的统一决策路由；
- 固定 Gateway workflow、request deadline、取消和 step-local fallback 已实现；
- 全局命令 Registry / Parser / Dispatcher 和 chat 短路已实现；
- 候选话题、话题路由、查询分析与保守降级已实现；
- 主动 chat 已收敛为 Gateway -> Patchouli prepare -> Alice -> Patchouli finalize；
- Passive Ingress 已归属 System 应用层，具备去重、顺序缓冲、封口、outbox 重试和 drain；
- Passive Memory 禁止命令、Alice、MTP 和回复生成；
- Agent-facing 错误 payload 已完成 XML escaping；
- 同一 `PassiveConversationKey` 在单进程内串行处理，不同会话仍可并发；
- 取消、超时、Profile 失败、Passive degradation 和 outbox retry 均有测试与观测；
- 浅色主题主体已合并，但它属于 v0.6.3 工作流的部分提前交付。

当前事实入口：

- [系统架构概览](./architecture/overview.md)
- [系统边界与所有权](./architecture/boundaries.md)
- [子系统公共契约](./contracts/subsystem-contracts.md)
- [公开路由与事件](./contracts/routes-and-events.md)
- [System 当前设计](./system/README.md)
- [Gateway 当前设计](./gateway/README.md)
- [Patchouli 当前设计](./patchouli/README.md)
- [Alice 当前设计](./alice/README.md)
- [Frontend 当前设计](./frontend/README.md)
- [Help](./help/README.md)
- [Applications](./applications/README.md)

### 3.2 发布范围边界

- 完整复合意图分解已经移出 v0.6.0 发布范围，暂列为 **Unscheduled**；
- 自定义入口拦截规则不属于当前能力，只有出现明确接入需求并形成 Plan 后再排期；
- 浅色主题属于 v0.6.3 工作流的部分提前交付，不改变 v0.6.0 的后端发布范围；
- `v0.6.1` 负责通用 Local Work Queue Runtime，不回溯改写 v0.6.0 Passive Ingress 的公共契约。

其 C0 样本与指标工作可以作为非阻塞研究继续推进，但在证据门槛、公共 envelope 和下游消费协议成立前，不得把它写成当前能力。

### 3.3 v0.6.0 发布验收

- Gateway 的公开决策覆盖发布范围内的所有入口模式；
- 命令、普通 chat 和 Passive Memory 的契约边界明确；
- 局部分析失败不会破坏可安全降级的主流程；
- 子 Agent 终态、并发 frame/cancel 状态和显式 Profile 失败不会被错误包装为成功或静默放大权限；
- 取消、超时和不可恢复不变量失败均有测试与观测；
- 当前设计文档、README、包版本和 `v0.6.0` tag 口径一致；
- 未完成方案已移入后续 Plan，不在当前文档中使用现在时；
- Release workflow 同时校验并发布 backend wheel、sdist、frontend archive 和合并校验文件。

## 4. 近期计划

以下顺序表示当前依赖关系，不是已经发布的承诺。

| 目标 | 状态 | 目标结果 | 依赖/计划入口 |
|:---|:---:|:---|:---|
| `v0.6.1` Reliable Local Work Runtime | Planned | 统一后台 work/job 生命周期，并建立最小耐久性、幂等、身份 scope 与 SQLite 恢复门槛 | [Local Work Queue Runtime](./plans/v0.6.1-local-work-queue-runtime.md)及三项可靠性 Plan 的前置切片 |
| `v0.6.2` Chat Attachments | Candidate | 文件先成为受身份约束、可幂等写入的原始 artifact，再编译为当前 chat 上下文；大文件走 Job | 依赖 v0.6.1、Artifact provenance、Identity scope；正式 Plan 待建立 |
| Frontend Reliability | Partially Landed / Parallel | 统一 identity、真实/mock 来源、Settings 契约以及 loading/error/waiting 状态，不把视觉个性化作为后端能力前置条件 | [Frontend 当前设计](./frontend/README.md)与相关 Todo；正式 Plan 待建立 |
| `v0.7.0` Document Ingestion & Provenance Contract | Candidate | document artifact -> chunk/evidence -> 可审核候选记忆，并在该阶段冻结 provenance 数据契约 | 依赖 v0.6.1/v0.6.2 与 Patchouli provenance；正式 Plan 待建立 |
| `v0.7.1` MTP READ Provenance | Candidate | 将已经稳定的版本、来源 artifact 和检索证据暴露给 READ | [MTP 当前契约](./contracts/mtp.md)；正式 Plan 待建立 |
| `v0.7.2` Deep Research MVP | Candidate | 可取消、可观测、可追溯的研究 Job 与报告 artifact | 依赖 Job、Document、READ provenance |

### 4.1 未排期事项

下表集中维护所有已经确认、但尚未进入具体版本承诺的事项。可靠性与模型治理 Plan 中被 v0.6.1 采用的最小前置切片不再视为未排期；表中记录的是其余完整领域阶段。

| 事项 | 范围 | 暂不排期原因或进入条件 | 计划入口 |
|:---|:---|:---|:---|
| 复合意图分解 | Gateway / Contracts / Patchouli / Alice | 先完成 C0 指标和脱敏样本门禁，证明单主意图路径的真实缺口，再冻结 composite envelope 与消费协议 | [Composite Intent Decomposition](./plans/composite-intent-decomposition.md) |
| 自定义入口拦截规则 | Gateway | 当前固定入口链已可运行；只有出现明确外部接入需求、配置所有者和验收样本后才建立 Plan | 待建立 |
| 领域状态持久化与恢复 | System / Patchouli / Alice | v0.6.1 只承担 work、lease 和已承诺操作的最小恢复；Artifact/Memory saga、Agent checkpoint、反馈与维护恢复另行排期 | [Runtime State Durability and Recovery](./plans/runtime-state-durability-and-recovery.md) |
| 领域幂等与 reconciliation | Patchouli / MemoryLibrary / Lifecycle | v0.6.1 先建立 operation identity 与 WorkStore 记录；Memory update、archive/revive、HIT/CITATION 等领域副作用后续推进 | [Cross-Subsystem Idempotency and Retry](./plans/cross-subsystem-idempotency-and-retry.md) |
| 执行资产安全与外部身份对齐 | Alice / MTP / Frontend | run/cache 隔离与最小 identity scope 前置；强沙箱、可信资产、资源限制和完整外部认证需要独立证据与方案 | [Identity Isolation and Execution Safety](./plans/identity-isolation-and-execution-safety.md) |
| 数据模型可变性治理后续阶段 | Cross-system | v0.6.1 只前置模型/边界清单；深不可变原语、Memory/PendingAtom 聚合重构和公共 DTO 迁移需按风险分批 | [Data Model Mutability Governance](./plans/data-model-mutability-governance.md) |
| RuntimeEvent 生产端抽象重构 | System / Cross-system | 当前 wire format、总线与消费语义已生效；除新 Work Runtime 所需最小事件契约外，不阻塞近期功能 | [RuntimeEvent Publishing Refactor](./plans/runtime-event-publishing-refactor.md) |
| Frontend 视觉个性化 | Frontend | 主题覆盖和自定义背景可并行探索，但必须后于真实状态、identity 和错误披露，不阻塞后端版本 | 待建立 |
| Conversation Branching | Chat / Memory / Lifecycle | 等 provenance、生命周期和真实编辑需求稳定后，再设计分支所有权与已沉淀记忆的失效语义 | 待建立 |

### 4.2 v0.6.1 Reliable Local Work Runtime

近期最优先的新增底座。目标是把当前分散的后台 memory task、未来 ingestion/research 和定时/hook 任务收敛到可持久化、可查询的统一生命周期。

第一阶段只要求单机可靠语义，不提前引入分布式队列或通用任务图。

这里的核心矛盾是“统一生命周期”与“过早抽象成万能任务框架”。验收重点不是支持尽可能多的调度形式，而是让任务身份、状态、取消、重试、超时和 outcome artifact 对调用方有一致含义，并能覆盖下一阶段真实负载。若 Document 或 Research 仍需自建第二套任务状态，Job Queue 就没有完成其底座职责。

进入 Queue 实现前，必须先完成四项轻量门槛：Durability D0 的状态分级、Idempotency I0 的业务操作身份清单、Identity S0 的身份/威胁模型，以及数据模型治理 Phase I 的 payload/所有权边界清单。它们用于冻结“什么可以被接受、序列化、恢复和重放”，不要求提前完成四份 Plan 的全部后续阶段。

v0.6.1 的发布顺序应为：

1. 冻结 Queue 契约并完成 in-memory runtime 的机械状态机验证；
2. 建立 SQLite WorkStore、lease recovery、唯一 idempotency key 和最小 identity scope；
3. 通过 feature flag 迁移 Interaction Submission 与 Memory Generation，验证重复消费、模糊失败和状态投影；
4. 最后开放用户可见 Runtime Job API 和 outcome artifact。

In-memory runtime 是内部实现里程碑，不构成“可靠 Job 已交付”。任何对外返回 durable accepted 的入口都必须在 SQLite 恢复和业务幂等门槛完成后开放。

### 4.3 v0.6.2 Attachments

上传文件先成为原始 artifact 和解析 artifact，再按当前对话需要编译为上下文。附件上传不应默认直接污染长期记忆；大文件解析交给 Job Queue。

Artifact 先于 Document Ingestion，是为了先保存“用户实际提供了什么”，再讨论系统从中理解出什么。原始文件、解析结果和候选记忆具有不同真实性和生命周期；如果上传后直接生成正式记忆，后续无法可靠区分证据、解析错误与系统结论。该阶段的验收重点是身份、来源和失败边界，而不是提前完成完整文档知识化。

附件还必须复用 v0.6.1 的 operation identity、重试结果和 identity scope：同一上传重试不能产生多份原始 Artifact，也不能因为拿到 artifact id 就跨身份读取。大文件只有在 Job 已经 durable accepted 后，才可向页面报告后台解析已接受。

### 4.4 Frontend Reliability（并行工作流）

浅色主题、Chat 的结构化 MTP/子 Agent 卡片、Kernel Vision 与部分状态持久化已经合并，因此前端工作不能整体标为“未开始”。当前实现和缺口以 [Frontend 当前设计](./frontend/README.md)为准。

近期优先级是可靠性而不是视觉个性化：建立单一 identity context、区分真实后端/缓存/mock、修复 Settings 配置结构偏差，并让 loading/error/waiting 状态来自可观察事实。Terminal 空入口也应删除、接通或明确标记。主题覆盖和自定义背景进入未排期表，可并行探索但不阻塞后端能力链；界面不得展示或暗示不可观测的模型内部推理。

### 4.5 v0.7.x 能力链

```text
Runtime Job Queue
  -> Attachment / Document Artifacts
  -> Document Ingestion + Provenance Contract
  -> MTP READ Provenance
  -> Deep Research
```

这条顺序优先复用现有 artifact、provenance、MemoryCompiler 和 RuntimeEvent，避免每个长任务建立独立状态系统。

Document Ingestion 必须建立在 Job 与 Artifact 之上，因为解析可能耗时、失败、取消，也必须保留原始证据；provenance 数据契约必须在摄入阶段同步冻结，`v0.7.1` 只负责把已有证据链稳定暴露给 MTP READ。Deep Research 又依赖可取消 Job、可追溯输入和可引用输出，否则报告只是另一段无法审计的生成内容。Conversation Branching 不再占用固定版本号，进入未排期表等待 provenance、生命周期边界和真实编辑需求稳定。

### 4.6 排序与验收检查

路线项目进入实现前，应确认：

- 它是否复用上一阶段已经形成的权威状态，而不是建立第二套 Job、Artifact 或 provenance 模型；
- 它的成功条件是否包含取消、失败、来源和恢复语义，而不只是 happy path 可演示；
- 新生成的信息是否仍能回到原始证据，并区分 artifact、候选记忆和正式记忆；
- 计划是否把强沙箱、自动回滚或分布式可靠性等尚未成立的能力误写成当前依赖；
- 完成后应更新哪些当前设计与契约，哪些实施文档应进入 Archive。

如果一项候选功能绕过这些前置条件才能落地，应优先调整范围或顺序，而不是用局部实现制造新的平行真相源。

## 5. 后置方向

### v0.8.0+：Alice 高级编排

状态：Deferred。

候选范围包括 plan-and-execute、多角色协作、parallel specialists、review loop 和可观测任务图。它依赖稳定的 Gateway task spec、Job Queue 与真实长任务负载，在这些前置能力完成前不扩大 Alice 框架。

### v0.9.0+：高级记忆生命周期

状态：Deferred。

候选范围包括 Memory split/merge、完整 rollback、branch invalidation、provenance reverse index和 L3 复活。

这些能力涉及不可逆数据修改，不能以当前简单历史字段替代正式设计。

### Executable Asset Sandbox

状态：Deferred，且与高级记忆生命周期分离。

候选范围包括 MTP RUN executable asset 的来源与信任等级、强隔离沙箱、文件/网络/进程限制、资源配额和真取消。在这些能力落地前，MTP RUN 只面向受信资产；不受信任代码必须拒绝执行或降级为展示/提议，不能用 prompt 约束代替安全边界。

## 6. 长期方向

- **记忆系统**：从 CRUD 走向可审计、可回放、可演化的知识资产；
- **运行时**：从单次 run/task 走向可持久化 Job 与受控任务图；
- **多 Agent**：从有限 CALL 走向有真实工作负载支撑的编排策略；
- **编译系统**：让 Retrieval、READ、Profile、Document、Research 与可执行资产共享 MemoryCompiler IR；
- **开放接入**：在保持身份和 provenance 边界的前提下服务外部 Agent harness。

长期理念、可证伪假设和决策门槛见[VISION.md](./VISION.md)。

## 7. 路线图维护规则

1. 只有 Git tag 对应版本可以标为 Released；
2. 已实现事实合并进当前设计，Roadmap 只保留阶段摘要；
3. 新增完整功能前先在 `plans/` 建立目标、非目标和验收条件；
4. 计划完成后更新当前文档，再移入 `archive/plans/`；
5. 部分提前落地的能力必须标为 Partially Landed，不能把整个阶段写成完成；
6. 任何版本范围变化都应同步更新[PROJECT.md](./PROJECT.md)和相关子系统索引；
7. Ideas 不自动进入 Roadmap，只有形成明确依赖和验收口径后才升级为 Plan。
