---
title: Patchouli Artifacts and Provenance
status: current
owner: patchouli
scope: immutable-evidence-and-memory-versioning
code_paths:
  - src/hivememory/core/models/artifact.py
  - src/hivememory/engines/artifacts/
  - src/hivememory/patchouli/memory_library/adapters/artifact.py
  - src/hivememory/patchouli/services/memory_generation.py
related_contracts:
  - docs/architecture/boundaries.md
  - docs/contracts/subsystem-contracts.md
related_docs:
  - docs/architecture/workspace.md
  - docs/system/attachments.md
last_reviewed: 2026-09-24
---

# Artifacts 与来源追踪

记忆不是原始证据本身。`MemoryAtom` 会被提炼、合并、修订和降权，如果系统只保存当前正文，就无法回答它来自哪轮交互、某次更新改了什么、当时引用了哪些外部材料。Artifact 层因此保存与当前记忆头部相邻、但不随头部一起改写的历史事实。

这一设计遵循两个原则：第一，原始材料与派生结论必须拥有不同身份；第二，历史应该以可独立验证的快照存在，而不是藏在不断追加的 `history_summary` 字符串中。

## 1. Artifact 不是另一种 MemoryAtom

`MemoryAtom` 是当前可检索、可演化的知识头部；Artifact 是 append-oriented 的证据或版本记录。两者通过轻量 `ArtifactRef` 关联：

```text
MemoryAtom.payload.artifacts
  ├─ refs[]   -> ArtifactRef(id, type, uri, sha256, summary)
  └─ events[] -> CREATED | VERSIONED | ARCHIVED | REVIVED

ArtifactStore
  └─ immutable JSON snapshots
```

Artifact 不进入普通向量检索，不承担 alias，也不因为某条记忆被合并就随之改写。相反，一份原始 InteractionArtifact 可以成为记忆创建或更新的 source artifact；一个 MemoryVersionArtifact 又可以记录某次完整状态变化。

`TopicAssetBinding.asset_ref` 不是 `ArtifactRef`，也不代表已经生成了一份 Artifact。它只是 Topic 对已使用 WorkspaceAsset 的不透明关系事实，随 `InteractionArtifactInput` 进入生成任务；Artifact 链不会把 WorkspaceAsset 原地转换为 Artifact，也不会在本层复制资产状态机或可见性策略。WorkspaceAsset 的所有权和 ref 生命周期以[Workspace 架构](../architecture/workspace.md)为准。

Chat 附件的来源 promotion 已接入本链路（见[Chat 附件链路](../system/attachments.md)）：记忆生成确实产生 CREATE/UPDATE 时，生成数据面对 task 中的每个 binding 按 `asset_ref` acquire READY representation，创建独立的 `DocumentArtifact` 证据快照后释放 lease；产物的 `source_uri` 钉住源 asset/representation 标识、revision 与 parser producer/version，`content_hash` 保存 representation 哈希。ref 已 remove、Store 已关闭或写入失败时按 best-effort 跳过并记录 warning，已提交的 binding 与 Memory 结果不变。

## 2. 当前四种类型

### 2.1 InteractionArtifact

InteractionArtifact 是一个话题材料快照，保存 `topic_id/title/summary` 和多个 `InteractionTurnSnapshot`。每个 turn 从 `LogicalBlock.turn` 冻结得到，包括：

- user/agent/team 三轴的 `actor_identity`；
- 原始与重写后的用户问题；
- assistant final text；
- `turn_events`、`actions` 与 `semantic_traces` 的 dict 快照。

它刻意不保存 memory id、alias、source intent、capture policy，也不设置顶层 Agent 来源字段：来源 provenance 由每个 block 冻结的 `actor_identity` 承载，因为同一 Workspace 话题内可以切换不同 Agent，话题级单值来源无法表达这一事实。原始交互先保持中立，哪条记忆由它派生则由 MemoryCreation/Version Artifact 表达。这样同一份证据不会因为生成出不同记忆而被重复解释为不同“原文”。

当前生成任务在执行前根据 `InteractionArtifactInput` 构建它。被动 settlement 使用本次结算的 blocks；主动 WRITE/UPDATE 使用话题最近五个 blocks 作为背景。若一次 finalize 产生多个主动任务，每个 task 目前会独立捕获自己的 InteractionArtifact，并不共享单一 task-group artifact。

### 2.2 DocumentArtifact

DocumentArtifact 表达某一时点的外源文档引用，可保存 source/canonical URI、MIME、retrieved time、etag、last-modified、原始快照地址、提取文本地址和页码/标题路径/行号/quote 等定位符。

当前已经有 model、builder 和 filesystem persistence，但完整 Document Ingestion 尚未接入当前主流程。因此它是已落地的数据基础，不等于系统已经能够抓取、切分、审核和生成文档记忆。

DocumentArtifact 是完整来源快照；某条 Memory 精确使用了哪些页码、行号或 quote，长期更适合由 Memory/Artifact 之间的细粒度 `SourceEvidenceRef` 表达。该 locator 归属会在 `v0.7.0` provenance contract 中最终裁定，当前模型字段保持兼容。

### 2.3 MemoryCreationArtifact

MemoryCreationArtifact（schema `"2"`）是一条记忆的 genesis record，记录：

- `memory_id`；
- `source_intent`：`ARCHIVE / WRITE / IMPORT / MANUAL / SYSTEM`；
- 结构化 `provenance`（`core.models.MemoryProvenance`：`source_agent_id`、`source_team_id` 与 `contributing_agent_ids`，与 MemoryAtom 语义一致，见下文“来源与归属语义”）；
- 当时的结构化 `GenerationContext`；
- source artifacts 与 source memories；
- 指向初始 `MemoryVersionArtifact(v1)` 的引用。

它不复制 title、alias、tags 等可变字段。初始完整状态由 v1 version snapshot 保存，避免 genesis 和 version 链出现两份互相漂移的“初始值”。

### 2.4 MemoryVersionArtifact

`MemoryVersionArtifact`（schema `"2"`，独立于 Memory 的 `"2.1"` 版本轴）保存 `memory_id`、`version_number`、`update_source`、结构化 `provenance`、`snapshot_before`、`snapshot_after`、changelog、source artifacts/memories 与 `changed_at`。两个快照字段直接嵌入捕获时点完整 `MemoryAtom` 的 canonical JSON（含 `meta.provenance`、`meta.lifecycle`、`payload.agent_config`、`payload.artifacts` 与 relations），经结构约束校验（缺键、未知内嵌 schema 拒绝），因此单个版本足以完整重建当时的原子，不依赖从 v1 顺序重放。

捕获时点在 Familiar 提交边界：完成内容与 lifecycle 合并、分配版本与内容时间之后，追加本次版本记录自身的 refs/events 之前——快照因此不含自身引用。创建时先写 v1（`snapshot_before` 为 `null`），再写 creation artifact；更新、去重合并和手工编辑分别使用 `UPDATE / MERGE / MANUAL_EDIT / SYSTEM_REWRITE` 来源。`history_summary` 字段已从 Memory schema 删除，正式来源与版本事实只由 artifacts 承载。

### 2.5 来源与归属语义

Artifact 是 Workspace 资产，归属只由 `workspace_identity` 表达，不存在 Agent owner 字段。来源 provenance 按类型定义：InteractionArtifact 以 block 内 `actor_identity` 记录来源；MemoryCreation/VersionArtifact 复用 core 的 `MemoryProvenance`，记录与 MemoryAtom 一致的 `source_agent_id`（操作来源，允许保留 `system` 表示"没有具体 Agent 作为操作来源主体"）与 `contributing_agent_ids`（实际贡献内容的 Agent 集合，去重、保持首次出现顺序、不含 `system`）。这些字段只记录 provenance 事实，不参与读取授权；DocumentArtifact 等其他类型的来源粒度按其生产入口单独裁定。

## 3. 生成链中的写入顺序

MemoryGenerationFamiliar 当前执行：

```text
capture InteractionArtifact (best effort)
  -> GenerationEngine.process()（纯计算；内容一致的重复更新降级 TOUCH）
  -> 提交边界：一次取 now；分配 version / updated_at / decay_anchor_at / confidence
  -> build MemoryCreation/Version Artifacts（mandatory；changed_at 使用同一 now）
  -> attach ArtifactRef + MemoryEventLog to MemoryAtom
  -> MidTermMemoryStore.upsert(MemoryAtom, recompute_vectors=<embedding 输入是否变化>)
```

CREATE 会挂载 v1、creation 和 interaction refs，并追加 CREATED event；UPDATE 会挂载 version 与 interaction refs，并追加 VERSIONED event；TOUCH 不创建版本 artifact、不改内容时间与版本、不挂载 interaction ref，只经 `patch_payload` 推进访问统计；本次 InteractionArtifact 仍按 best-effort 捕获留存为证据，但不与该记忆建立新引用。DISCARD 不写 MemoryAtom。

**版本记录是内容提交成功的前置条件**：版本存储未启用或写入失败时，Familiar 直接抛错，不发布无历史的新 canonical；canonical 写入失败时错误传播，已写入的版本记录保留为孤立 Artifact、不构成已提交版本。手工创建/编辑同样经过 MemoryGenerationFamiliar：手工创建使用 `MANUAL` creation intent，手工编辑生成 `MANUAL_EDIT` version artifact；传入值与当前值相同的编辑不创建版本。

## 4. 存储与完整性

默认 filesystem adapter 使用布局：

```text
{root}/{owner_digest}/{workspace_digest}/{artifact_type}/{YYYY}/{MM}/{DD}/{artifact_id 摘要}.json
```

目录段为 owner 与 Workspace 标识的摘要，文件名为 artifact_id 摘要；存储位置不含明文身份，归属由 `workspace_identity` 字段与索引共同校验。

写入时先把 `content_hash` 置空并计算规范 JSON 的 SHA-256，再把 hash 写回文件，同时返回携带 URI 与 sha256 的 `ArtifactRef`。读取会验证文件内 hash；通过 ref 读取时还会验证 ref hash。`verify()` 可独立返回 stored/actual hash 比较结果。

写入时持久化 artifact 索引（`.artifact_index.json`）支撑 `list_by_memory()`；同一 `(workspace, artifact_id)` 已存在且规范内容不同时，`put()` 直接拒绝（append-only 硬校验），内容相同则幂等返回既有 ref。Artifact 的“不可变”由该 append-only 校验、随机 artifact id、版本模型与只追加调用方式共同保证。

## 5. 可选旁路与失败语义

Artifacts 可整体或按 interaction/document/memory builder 关闭；关闭后 Runtime 注入 NoOp builders，主链无需布满条件判断。Artifact storage 在健康报告中是 optional。

A2-P 起版本记录是内容提交的强制前置条件：memory 组件关闭（NoOp builder）或构建异常会让内容 create/update 直接失败，不再有“记录 warning 后继续持久化”的分支。剩余的非原子性边界是：

- 版本 Artifact 已写入但后续 Qdrant upsert 失败时，该记录成为孤立文件，不构成已提交版本，需运维清理；
- interaction/document 等可选来源 Artifact 仍按各自开关独立运行，其失败不阻断内容提交；
- ref 与 artifact 的全库一致性当前没有后台扫描器。

`payload.artifacts.refs` 非空说明已挂载引用可以继续验证；孤立 Artifact 不在此列，也不得被当作 canonical head。

## 6. 双视图与结构化事实

同一轮 Agent 事实以 `TurnEvent -> AgentAction -> TraceItem -> TurnRecord` 保存。消费方随后产生两个不同视图：

- 历史重放视图保留工具调用和对话顺序，服务于 Agent 上下文；
- Generation 视图保留 user query、final text 和语义 trace，服务于记忆提取；
- InteractionArtifact 则冻结底层结构化事实，服务于来源追踪。

三者共享同一 `TurnRecord`，但不共享一段提前压扁的 `context_messages` 字符串。新增交互事实应先进入结构化事件或 turn 模型，再由各视图渲染，不能重新引入字符串反解析主路径。

## 7. 当前限制

- DocumentArtifact builder 尚未接入完整文档摄入用例；
- 没有 orphan/ref consistency scanner、保留策略或垃圾回收；
- 没有把 MTP trace 中每条 READ/RUN 证据自动提升为细粒度 source memory/document refs；
- artifacts 与 Qdrant 写入不是原子事务；
- 完整版本重建模型已经存在，但公开回档、diff 和 UI 浏览仍未完成。

这些缺口不否定 Artifact 的当前价值，但它们限定了系统现在能声称的是“具有可验证来源与版本基础”，而不是“已经拥有完整审计账本”。
