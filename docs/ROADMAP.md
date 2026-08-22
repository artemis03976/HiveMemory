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
last_reviewed: 2026-08-22
---

# HiveMemory 开发路线图

本文只维护版本阶段、当前完成度、依赖关系和相关入口。已经生效的详细设计进入 Architecture、System 或子系统文档；跨版本质量目标进入 Governance；未承诺方向进入 Ideas；绑定版本的实施方案进入 Plans；完成后的实施文档进入 Archive。

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

- 最新已发布标签：`v0.6.1`；
- 当前发布基线：`v0.6.1`；
- 下一计划版本：`v0.6.2`，整体状态为 Candidate；[W0 Workspace MVP](./plans/v0.6.2-workspace-mvp.md)已经形成 Planned 实施方案，W1 Chat Attachments 以其完成为硬前置。

当前规范代码版本为 `0.6.1`，由 `src/hivememory/_version.py` 唯一声明并供构建与运行时复用。`v0.6.1` Git tag、Python 包、前端清单和构建检查使用完全一致的版本口径。

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

## 3. 当前发布：v0.6.1

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

以下顺序表示当前依赖关系，不是已经发布的承诺。

| 目标 | 状态 | 目标结果 | 依赖/计划入口 |
|:---|:---:|:---|:---|
| `v0.6.2 W0` Workspace MVP | Planned | 建立 `WorkspaceIdentity`、默认 `main_workspace`、端到端 scope、双 Workspace 隔离、System-owned WorkspaceAssetStore、两级状态机和 SemanticBuffer binding | 依赖 v0.6.1 与 Identity scope；[正式 Plan](./plans/v0.6.2-workspace-mvp.md)，设计推导见 [Workspace MVP Idea](./ideas/workspace-mvp-chat-attachments-design.md) |
| `v0.6.2 W1` Chat Attachments | Candidate | 在已经验收的 Workspace 公共契约上实现上传、文本解析、asset refs、Context Compiler 与按需 Artifact promotion | 硬依赖 `v0.6.2 W0` Workspace MVP 与 Artifact provenance；独立正式 Plan 待建立 |
| Frontend Reliability | Partially Landed / Parallel | 统一 identity、真实/mock 来源、Settings 契约以及 loading/error/waiting 状态，不把视觉个性化作为后端能力前置条件 | [Frontend 当前设计](./frontend/README.md)与相关 Todo；正式 Plan 待建立 |
| `v0.7.0` Document Ingestion & Provenance Contract | Candidate | document artifact -> chunk/evidence -> 可审核候选记忆，并在该阶段冻结 provenance 数据契约 | 依赖 v0.6.1/v0.6.2 与 Patchouli provenance；正式 Plan 待建立 |
| `v0.7.1` MTP READ Provenance | Candidate | 将已经稳定的版本、来源 artifact 和检索证据暴露给 READ | [MTP 当前契约](./contracts/mtp.md)；正式 Plan 待建立 |
| `v0.7.2` Deep Research MVP | Candidate | 可取消、可观测、可追溯的研究过程与报告 artifact | 依赖真实长任务负载、Document、READ provenance；后台执行机制待独立设计 |

### 4.1 未排期事项

下表集中维护所有已经确认、但尚未进入具体版本承诺的事项。治理主题中被 v0.6.1 采用的最小前置切片不再视为未排期；表中记录的是其余治理工作包、Ideas 与候选方向。它们不因出现在路线图中就自动成为 Plan。

| 事项 | 范围 | 暂不排期原因或进入条件 | 分类入口 |
|:---|:---|:---|:---|
| 复合意图分解 | Gateway / Contracts / Patchouli / Alice | 先完成 C0 指标和脱敏样本门禁，证明单主意图路径的真实缺口，再冻结 composite envelope 与消费协议 | [Composite Intent Decomposition Idea](./ideas/composite-intent-decomposition.md) |
| 自定义入口拦截规则 | Gateway | 当前固定入口链已可运行；只有出现明确外部接入需求、配置所有者和验收样本后才建立 Plan | 待建立 |
| 领域状态持久化与恢复 | System / Patchouli / Alice | v0.6.1 只冻结进程内 work 契约与持久化门槛；SQLite、claim ownership/lease、Artifact/Memory saga、Agent checkpoint、反馈与维护恢复均按真实恢复需求另行排期 | [Durability and Recovery Governance](./governance/reliability/durability-and-recovery.md) |
| 领域幂等与 reconciliation | Patchouli / MemoryLibrary / Lifecycle | v0.6.1 先建立 operation identity 与 WorkStore 记录；Memory update、archive/revive、HIT/CITATION 等领域副作用后续推进 | [Idempotency and Retry Governance](./governance/reliability/idempotency-and-retry.md) |
| 执行资产安全与外部身份对齐 | Alice / MTP / Frontend | run/cache 隔离与最小 identity scope 前置；强沙箱、可信资产、资源限制和完整外部认证需要独立证据与方案 | [Identity and Execution Safety Governance](./governance/security/identity-and-execution-safety.md) |
| 数据模型可变性治理后续阶段 | Cross-system | v0.6.1 只前置模型/边界清单；深不可变原语、Memory/PendingAtom 聚合重构和公共 DTO 迁移需按风险分批 | [Data Model Mutability Governance](./governance/data-model/mutability.md) |
| RuntimeEvent 生产端迁移 | System / Cross-system | 当前 wire format、总线与消费语义已生效；仅剩各生产域 emitter/publisher 接驳，不阻塞近期功能 | [RuntimeEvent Producer Migration Todo](./todo/runtime-event-producer-migration.md) |
| 用户可见长期任务与后台 Agent workflow | System / Gateway / Alice | 当前缺少稳定的复杂任务自主执行能力和已经验证的长任务负载；先用现有 interaction/memory lane 跑通真实业务，再根据 Document/Research 需求重新设计 | 待建立 |
| Frontend 视觉个性化 | Frontend | 主题覆盖和自定义背景可并行探索，但必须后于真实状态、identity 和错误披露，不阻塞后端版本 | 待建立 |
| Conversation Branching | Chat / Memory / Lifecycle | 等 provenance、生命周期和真实编辑需求稳定后，再设计分支所有权与已沉淀记忆的失效语义 | 待建立 |

### 4.2 v0.6.2 Workspace MVP 与 Chat Attachments

`v0.6.2` 使用两份独立开发 Plan。[W0 Workspace MVP 正式 Plan](./plans/v0.6.2-workspace-mvp.md)是当前首先实施的基础计划，W1 Chat Attachments 是其下游计划；W1 不得通过私有兼容字段或局部容器绕过尚未完成的 Workspace scope、资源归属和隔离验收。

#### 4.2.1 W0 Workspace MVP

Workspace 以不可变 `WorkspaceIdentity(owner_user_id, workspace_key, workspace_id)` 统一持有身份。MVP 不启用独立 ID 生成器，固定 `workspace_id == workspace_key`；默认身份使用 `workspace_key=workspace_id="main_workspace"`。完整资源坐标是 `(owner_user_id, workspace_id)`，所有对外协议、store、cache、filter、event 和 work payload 只使用非空 `workspace_id` 寻址，不允许在内部执行 `workspace_id or workspace_key` fallback。

普通请求可以不传 Workspace，但只允许在最外层入口解析一次默认 `WorkspaceIdentity`。一次 Chat run 使用唯一 `interaction_id`，并由 `IdentityScope` 将 actor identity 与 WorkspaceIdentity 一起冻结；Gateway、Patchouli、Alice、MemoryLibrary、MTP、finalize 和后台 work 必须复用同一 scope。同一个 Agent 在两个后端 Workspace 并发运行时不得串扰。第一版不开放 Workspace 创建、切换或通信，只要求后端显式构造第二个 Workspace 验证隔离。

W0 还负责由 System runtime 建立一个进程级、按 WorkspaceIdentity 逻辑分区的 WorkspaceAssetStore；实现 WorkspaceAsset/AssetRepresentation 两级状态机、READY-only 使用、删除与进程内 lease，并把 TopicAssetBinding 放入 Patchouli 的 SemanticBuffer/ShortTermMemoryStore。MVP 可以用极薄的单例 WorkspaceRuntime 聚合 Store，也可以先由 `_RuntimeBundle` 直接持有，但不得为每个 Workspace 创建 Runtime 或保存 `current_workspace`。W0 不实现真实附件上传、解析、Context Compiler、现有 cache 迁移或 Artifact promotion；WorkspaceAsset 只承诺当前进程内生命周期。

现有持久化数据不在 W0 落地前批量改写。W0 先为关键模型增加 Workspace 字段，并通过受控兼容投影把历史缺字段记录解释为对应用户的 `main_workspace`；待系统基本落地且双 Workspace 隔离验证通过后，再提供独立转换脚本，应用新的 visibility 枚举值并补齐 Memory、Artifact、Topic 等关键记录的 WorkspaceIdentity 投影字段。

#### 4.2.2 W1 Chat Attachments

W1 把上传文件注册到 W0 已建立的 System-owned `WorkspaceAsset` working set。原始内容、提取文本和 metadata 是同一资产内的 runtime representations；只有 required representation 解析 READY 后，asset ref 才能进入 Chat。解析失败直接向用户返回稳定错误且不自动重试，用户重新上传创建新的逻辑资产。`WorkspaceAssetRef` 在 Chat 中显式选择，再按当前对话需要编译为上下文。WorkspaceAsset 继续只承诺当前进程内可用，不承诺跨重启恢复；这一口径与当前 Topic 仍为内存态一致。

上传和 UI 选择本身都不创建 Topic 关系或 Artifact。只有用户显式选择 READY ref 且本轮 Interaction 成功完成，系统才把 block 与 `TopicAssetBinding` 原子提交；binding 是 Topic 真实使用过该资产的权威事实，不存在“只绑定但未使用”的第二状态。Topic settlement 在清理 SemanticBuffer 前把全部 binding refs 冻结进 Materialization task；可选 `ContextAttachmentUse` 只补充实际 representation revision/hash、locator、token 与 compile 诊断。当 Topic Materialization 得到 Memory CREATE/UPDATE 时，consumer 用 task ref 反查 WorkspaceAssetStore、持有 lease，并将对应内容提升为不可变来源 Artifact；`DISCARD` 不执行 promotion。提升是创建独立证据快照，不是把 WorkspaceAsset 原地转换；task/ref 只在当前进程和 Store 存活期内结算，已提升 Artifact 才按自身持久化契约存在。

当前只支持的文档型附件在提升时复用 `DocumentArtifact`，并通过 `origin=CHAT_ATTACHMENT`、源 asset/revision、parser version 和 content hash 等 metadata 与 `v0.7.0` Document Ingestion 区分入口。Artifact 类型按内容语义而不是入口选择；未来非文档附件不能被强塞进 DocumentArtifact。附件还必须复用 v0.6.1 的 operation identity 与重试语义：同一进程内相同 upload operation 只返回一个逻辑资产，同一 materialization retry 不重复生成来源 Artifact。

W1 只消费 W0 已经稳定的 `WorkspaceIdentity`、`IdentityScope`、WorkspaceAssetStore 和 Topic binding 契约，不重新定义 Workspace 所有权或 fallback。大文件异步解析需要在出现真实负载后独立设计，不预设复用 v0.6.1 的业务 lane。

### 4.3 Frontend Reliability（并行工作流）

浅色主题、Chat 的结构化 MTP/子 Agent 卡片、Kernel Vision 与部分状态持久化已经合并，因此前端工作不能整体标为“未开始”。当前实现和缺口以 [Frontend 当前设计](./frontend/README.md)为准。

近期优先级是可靠性而不是视觉个性化：建立单一 identity context、区分真实后端/缓存/mock、修复 Settings 配置结构偏差，并让 loading/error/waiting 状态来自可观察事实。Terminal 空入口也应删除、接通或明确标记。主题覆盖和自定义背景进入未排期表，可并行探索但不阻塞后端能力链；界面不得展示或暗示不可观测的模型内部推理。

### 4.4 v0.7.x 能力链

```text
WorkspaceAsset + on-materialization Artifact promotion
  -> Document Ingestion + Provenance Contract
  -> MTP READ Provenance

真实长任务负载 + 稳定 Agent 执行能力
  -> 独立设计后台执行机制
  -> Deep Research
```

这条顺序优先复用 Workspace scope、Artifact、provenance、MemoryCompiler 和 RuntimeEvent，同时避免在真实长任务出现前冻结一套用户任务抽象。

Document Ingestion 必须建立在 Artifact 之上，因为解析必须保留原始证据；当解析规模确实需要后台执行时，再根据真实耗时、取消和恢复要求建立对应机制。provenance 数据契约应在摄入阶段同步冻结，`v0.7.1` 只负责把已有证据链稳定暴露给 MTP READ。Deep Research 还依赖可追溯输入、可引用输出和已经证明可行的 Agent 长任务执行，否则报告只是另一段无法审计的生成内容。Conversation Branching 不再占用固定版本号，进入未排期表等待 provenance、生命周期边界和真实编辑需求稳定。

### 4.5 排序与验收检查

路线项目进入实现前，应确认：

- 它是否复用上一阶段已经形成的权威状态，而不是建立第二套工作状态、Artifact 或 provenance 模型；
- 它的成功条件是否包含取消、失败、来源和恢复语义，而不只是 happy path 可演示；
- 新生成的信息是否仍能回到原始证据，并区分 artifact、候选记忆和正式记忆；
- 计划是否把强沙箱、自动回滚或分布式可靠性等尚未成立的能力误写成当前依赖；
- 完成后应更新哪些当前设计与契约，哪些实施文档应进入 Archive。

如果一项候选功能绕过这些前置条件才能落地，应优先调整范围或顺序，而不是用局部实现制造新的平行真相源。

## 5. 后置方向

### v0.8.0+：Alice 高级编排

状态：Deferred。

候选范围包括 plan-and-execute、多角色协作、parallel specialists、review loop 和可观测任务图。它依赖稳定的 Gateway task spec、真实长任务负载和为这些负载验证过的后台执行契约，在这些前置能力完成前不扩大 Alice 框架。

### v0.9.0+：高级记忆生命周期

状态：Deferred。

候选范围包括 Memory split/merge、完整 rollback、branch invalidation、provenance reverse index和 L3 复活。

这些能力涉及不可逆数据修改，不能以当前简单历史字段替代正式设计。

### Executable Asset Sandbox

状态：Deferred，且与高级记忆生命周期分离。

候选范围包括 MTP RUN executable asset 的来源与信任等级、强隔离沙箱、文件/网络/进程限制、资源配额和真取消。在这些能力落地前，MTP RUN 只面向受信资产；不受信任代码必须拒绝执行或降级为展示/提议，不能用 prompt 约束代替安全边界。

## 6. 长期方向

- **记忆系统**：从 CRUD 走向可审计、可回放、可演化的知识资产；
- **运行时**：在真实负载证明需要后，从单次 run/task 走向可持久化后台工作与受控任务图；
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
