---
title: HiveMemory Roadmap
status: current
owner: project
scope: releases-and-planned-capabilities
updates:
  - docs/PROJECT.md
  - docs/plans/
  - docs/archive/plans/
last_reviewed: 2026-07-28
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
| Partially Landed | 阶段中的一部分已进入当前实现，其余仍未完成 |
| Deferred | 已明确后置，不属于近期承诺 |

当前必须同时使用两个版本事实：

- 最新已发布标签：`v0.5.0`；
- 当前未发布开发基线：`v0.6.0`。

`pyproject.toml` 中的历史版本不用于判断路线图完成度。

## 2. 发布历史

| 版本 | 状态 | 核心结果 | 当前依据 |
|:---|:---:|:---|:---|
| `v0.1.0-beta` / `v0.1.1` | Released | 记忆 MVP、API、基础前端与早期冷热路径 | Git tag；后续设计已演进 |
| `v0.2.0` | Released | 多 Agent 隔离、Agent Profile 记忆化 | Git tag；[Patchouli 当前设计](./patchouli/README.md)；Alice 文档待 P1 收敛 |
| `v0.3.0` | Released | CALL、PendingAtom、Alice Orchestrator、MemoryCompiler | Git tag；[MTP](./contracts/mtp.md) |
| `v0.4.0` | Released | chat run / memory task 取消控制与 RuntimeEvent | Git tag；[路由与事件](./contracts/routes-and-events.md) |
| `v0.5.0` | Released | artifact/provenance、MemoryLibrary、async-native、模型注册 | Git tag；[System 当前设计](./system/README.md)；[Patchouli 当前设计](./patchouli/README.md) |

过去文档中的 `v0.5.1`、`v0.5.2`、`v0.5.3` 是 v0.5 开发期的内部工作批次，不是当前仓库中的独立发布标签。它们的已实现事实应按模块并入当前文档，而不是继续作为平行版本入口。

## 3. 当前开发：v0.6.0

主题：**System Gateway、全局命令与 Passive Ingress**。

### 3.1 已经落地

- Gateway 已从旧引擎角色上移为 `GatewaySystem` 同级子系统；
- `gateway.public.process` 成为主动/被动入口的统一决策路由；
- 固定 Gateway workflow、request deadline、取消和 step-local fallback 已实现；
- 全局命令 Registry / Parser / Dispatcher 和 chat 短路已实现；
- 候选话题、话题路由、查询分析与保守降级已实现；
- 主动 chat 已收敛为 Gateway -> Patchouli prepare -> Alice -> Patchouli finalize；
- Passive Ingress 已归属 System 应用层，具备去重、顺序缓冲、封口、outbox 重试和 drain；
- Passive Memory 禁止命令、Alice、MTP 和回复生成；
- 浅色主题主体已合并，但它属于 v0.6.3 工作流的部分提前交付。

当前事实入口：

- [系统架构概览](./architecture/overview.md)
- [系统边界与所有权](./architecture/boundaries.md)
- [子系统公共契约](./contracts/subsystem-contracts.md)
- [公开路由与事件](./contracts/routes-and-events.md)
- [System 当前设计](./system/README.md)
- [Gateway 当前设计](./gateway/README.md)
- [Patchouli 当前设计](./patchouli/README.md)

### 3.2 发布前剩余工作

- 完成 Alice 的 P1 当前设计文档迁移；System、Gateway 与 Patchouli 旧入口已停止维护；
- 将复合意图从分析结果扩展为下游可消费的稳定任务/决策契约；
- 补齐自定义入口拦截规则的配置与运行时注入；
- 继续稳定 Gateway 降级、命令和 Passive Ingress 的端到端行为；
- 统一包版本元数据、README 和发布检查；
- 明确未完成项是否移出 v0.6.0，避免发布范围继续膨胀。

复合意图的详细方案仍位于迁移中的计划文档：[V0.6.0CompositeIntentDecompositionDesign.md](./mod/V0.6.0CompositeIntentDecompositionDesign.md)。其中未实现内容不得作为当前能力引用。

### 3.3 v0.6.0 完成条件

- Gateway 的公开决策能够覆盖发布范围内的所有入口模式；
- 命令、普通 chat、Passive Memory 和复合意图的契约边界明确；
- 局部分析失败不会破坏可安全降级的主流程；
- 取消、超时和不可恢复不变量失败均有测试与观测；
- 当前设计文档、README、包版本和 tag 口径一致；
- 未完成方案已经移入后续 Plan，不在当前文档中使用现在时。

## 4. 近期计划

以下顺序表示当前依赖关系，不是已经发布的承诺。

| 目标 | 状态 | 目标结果 | 依赖/计划入口 |
|:---|:---:|:---|:---|
| `v0.6.1` Runtime Job Queue | Planned | 统一后台 Job 状态、取消、重试、超时、触发和 outcome artifact | [迁移中的 Plan](./mod/V0.6.1LocalWorkQueueRuntimePlan.md) |
| `v0.6.2` Chat Attachments | Candidate | 文件先 artifact 化，再作为当前 chat 上下文；大文件走 Job | 依赖 v0.6.1；正式 Plan 待建立 |
| `v0.6.3` Frontend Experience | Partially Landed | 完成主题系统覆盖、自定义背景与明确运行状态反馈 | 浅色主题已落地，其余待形成 Plan |
| `v0.7.0` Document Ingestion | Candidate | document artifact -> chunk/evidence -> 可审核候选记忆 | 依赖 v0.6.1/0.6.2 与 Patchouli provenance |
| `v0.7.1` MTP READ Provenance | Candidate | READ 可访问版本、来源 artifact 和检索证据 | [MTP 当前契约](./contracts/mtp.md)；正式 Plan 待建立 |
| `v0.7.2` Deep Research MVP | Candidate | 可取消、可观测、可追溯的研究 Job 与报告 artifact | 依赖 Job、Document、READ provenance |
| `v0.7.3` Conversation Branching | Candidate / Optional | 编辑消息后创建安全分支，不承诺自动回滚旧记忆 | 正式 Plan 待建立 |

### 4.1 v0.6.1 Runtime Job Queue

近期最优先的新增底座。目标是把当前分散的后台 memory task、未来 ingestion/research 和定时/hook 任务收敛到可持久化、可查询的统一生命周期。

第一阶段只要求单机可靠语义，不提前引入分布式队列或通用任务图。

这里的核心矛盾是“统一生命周期”与“过早抽象成万能任务框架”。验收重点不是支持尽可能多的调度形式，而是让任务身份、状态、取消、重试、超时和 outcome artifact 对调用方有一致含义，并能覆盖下一阶段真实负载。若 Document 或 Research 仍需自建第二套任务状态，Job Queue 就没有完成其底座职责。

### 4.2 v0.6.2 Attachments

上传文件先成为原始 artifact 和解析 artifact，再按当前对话需要编译为上下文。附件上传不应默认直接污染长期记忆；大文件解析交给 Job Queue。

Artifact 先于 Document Ingestion，是为了先保存“用户实际提供了什么”，再讨论系统从中理解出什么。原始文件、解析结果和候选记忆具有不同真实性和生命周期；如果上传后直接生成正式记忆，后续无法可靠区分证据、解析错误与系统结论。该阶段的验收重点是身份、来源和失败边界，而不是提前完成完整文档知识化。

### 4.3 v0.6.3 Frontend Experience

浅色主题已经合并，因此该阶段不能再整体标为“未开始”。剩余目标是补齐主要页面覆盖、自定义背景和基于可观察状态的等待反馈，不展示或暗示不可观测的模型内部推理。

### 4.4 v0.7.x 能力链

```text
Runtime Job Queue
  -> Attachment / Document Artifacts
  -> Document Ingestion
  -> MTP READ Provenance
  -> Deep Research
  -> Conversation Branching
```

这条顺序优先复用现有 artifact、provenance、MemoryCompiler 和 RuntimeEvent，避免每个长任务建立独立状态系统。

Document Ingestion 必须建立在 Job 与 Artifact 之上，因为解析可能耗时、失败、取消，也必须保留原始证据；MTP READ Provenance 必须建立在文档证据链之上，否则“查看来源”只能返回没有稳定身份的临时文本；Deep Research 又依赖可取消 Job、可追溯输入和可引用输出，否则报告只是另一段无法审计的生成内容。Conversation Branching 被放在更后，是因为编辑历史与已沉淀记忆之间不存在天然可逆关系，不能在 provenance 和生命周期边界尚不稳定时承诺自动回滚。

### 4.5 排序与验收检查

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

### v0.9.0+：高级生命周期与安全执行

状态：Deferred。

候选范围包括 Memory split/merge、完整 rollback、branch invalidation、provenance reverse index、L3 复活，以及 MTP RUN executable asset、强隔离沙箱、资源限制和真取消。

这些能力涉及不可逆数据修改与不受信任代码执行，不能以当前本地执行器或简单历史字段替代正式设计。

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
