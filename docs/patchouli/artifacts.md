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
last_reviewed: 2026-09-01
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

`TopicAssetBinding.asset_ref` 不是 `ArtifactRef`，也不代表已经生成了一份 Artifact。它只是 Topic 对已使用 WorkspaceAsset 的不透明关系事实，随 `InteractionArtifactInput` 进入生成任务；当前 Artifact 链不会把 WorkspaceAsset 原地转换为 Artifact，也不会在本层复制资产状态机或可见性策略。WorkspaceAsset 的所有权和 ref 生命周期以[Workspace 架构](../architecture/workspace.md)为准。

## 2. 当前四种类型

### 2.1 InteractionArtifact

InteractionArtifact 是一个话题材料快照，保存 `topic_id/title/summary` 和多个 `InteractionTurnSnapshot`。每个 turn 从 `LogicalBlock.turn` 冻结得到，包括：

- user/agent/team identity；
- 原始与重写后的用户问题；
- assistant final text；
- `turn_events`、`actions` 与 `semantic_traces` 的 dict 快照。

它刻意不保存 memory id、alias、source intent 或 capture policy。原始交互先保持中立，哪条记忆由它派生则由 MemoryCreation/Version Artifact 表达。这样同一份证据不会因为生成出不同记忆而被重复解释为不同“原文”。

当前生成任务在执行前根据 `InteractionArtifactInput` 构建它。被动 settlement 使用本次结算的 blocks；主动 WRITE/UPDATE 使用话题最近五个 blocks 作为背景。若一次 finalize 产生多个主动任务，每个 task 目前会独立捕获自己的 InteractionArtifact，并不共享单一 task-group artifact。

### 2.2 DocumentArtifact

DocumentArtifact 表达某一时点的外源文档引用，可保存 source/canonical URI、MIME、retrieved time、etag、last-modified、原始快照地址、提取文本地址和页码/标题路径/行号/quote 等定位符。

当前已经有 model、builder 和 filesystem persistence，但完整 Document Ingestion 尚未接入当前主流程。因此它是已落地的数据基础，不等于系统已经能够抓取、切分、审核和生成文档记忆。

DocumentArtifact 是完整来源快照；某条 Memory 精确使用了哪些页码、行号或 quote，长期更适合由 Memory/Artifact 之间的细粒度 `SourceEvidenceRef` 表达。该 locator 归属会在 `v0.7.0` provenance contract 中最终裁定，当前模型字段保持兼容。

### 2.3 MemoryCreationArtifact

MemoryCreationArtifact 是一条记忆的 genesis record，记录：

- `memory_id`；
- `source_intent`：`ARCHIVE / WRITE / IMPORT / MANUAL / SYSTEM`；
- 当时的结构化 `GenerationContext`；
- source artifacts 与 source memories；
- 指向初始 `MemoryVersionArtifact(v1)` 的引用。

它不复制 title、alias、tags 等可变字段。初始完整状态由 v1 version snapshot 保存，避免 genesis 和 version 链出现两份互相漂移的“初始值”。

### 2.4 MemoryVersionArtifact

每个版本 artifact 保存 `version_number`、`update_source`、`snapshot_before`、`snapshot_after`、changelog、source artifacts/memories 与 changed time。`snapshot_after` 包含 content、alias、title、summary、tags 和 memory type，因此单个版本足以重建当时的可变字段，不依赖从 v1 顺序重放每个 patch。

创建时先写 v1，再写 creation artifact；更新、去重合并和手工编辑分别使用 `UPDATE / MERGE / MANUAL_EDIT / SYSTEM_REWRITE` 来源。MemoryAtom 的轻量 `history_summary` 目前仍保留为兼容展示，但正式来源与版本事实应以 artifacts 为准。

## 3. 生成链中的写入顺序

MemoryGenerationFamiliar 当前执行：

```text
capture InteractionArtifact (best effort)
  -> GenerationEngine.process()
  -> build MemoryCreation/Version Artifacts (best effort)
  -> attach ArtifactRef + MemoryEventLog to MemoryAtom
  -> MidTermMemoryStore.upsert(MemoryAtom)
```

CREATE 会挂载 v1、creation 和 interaction refs，并追加 CREATED event；UPDATE 会挂载 version 与 interaction refs，并追加 VERSIONED event；TOUCH 不创建版本 artifact，但仍可能挂载本次 interaction ref。DISCARD 不写 MemoryAtom。

手工创建/编辑也经过 MemoryGenerationFamiliar：手工创建使用 `MANUAL` creation intent，手工编辑生成 `MANUAL_EDIT` version artifact，而不是绕过 provenance 直接写 Qdrant。

## 4. 存储与完整性

默认 filesystem adapter 使用布局：

```text
{root}/{artifact_type}/{YYYY}/{MM}/{DD}/{artifact_id}.json
```

写入时先把 `content_hash` 置空并计算规范 JSON 的 SHA-256，再把 hash 写回文件，同时返回携带 URI 与 sha256 的 `ArtifactRef`。读取会验证文件内 hash；通过 ref 读取时还会验证 ref hash。`verify()` 可独立返回 stored/actual hash 比较结果。

Artifact 的“不可变”目前主要由随机 artifact id、版本模型和只追加调用方式保证。Adapter 的 `put()` 本身没有 compare-and-set，也不会拒绝用相同 id 覆盖已有路径，因此调用方仍必须把 artifact id 视为一次性身份。

## 5. 可选旁路与失败语义

Artifacts 可整体或按 interaction/document/memory builder 关闭；关闭后 Runtime 注入 NoOp builders，主链无需布满条件判断。Artifact storage 在健康报告中是 optional。

更重要的是，当前 generation 把 artifact 构建异常记录为 warning 后继续持久化 MemoryAtom。这保证证据存储故障不会阻断所有记忆写入，却也意味着 provenance 不是强一致事务：

- MemoryAtom 可能存在但没有 creation/version artifact；
- artifact 可能已经写入，但后续 Qdrant upsert 失败而成为未引用文件；
- ref 与 artifact 的全库一致性当前没有后台扫描器。

任何调用方都不能把 `payload.artifacts.refs` 非空当作“来源绝对完整”的证明；它只能说明已挂载的引用可以继续验证。

## 6. 双视图与结构化事实

同一轮 Agent 事实以 `TurnEvent -> AgentAction -> TraceItem -> TurnRecord` 保存。消费方随后产生两个不同视图：

- 历史重放视图保留工具调用和对话顺序，服务于 Agent 上下文；
- Generation 视图保留 user query、final text 和语义 trace，服务于记忆提取；
- InteractionArtifact 则冻结底层结构化事实，服务于来源追踪。

三者共享同一 `TurnRecord`，但不共享一段提前压扁的 `context_messages` 字符串。新增交互事实应先进入结构化事件或 turn 模型，再由各视图渲染，不能重新引入字符串反解析主路径。

## 7. 当前限制

- `ArtifactStoragePort.list_by_memory()` 的 filesystem 实现仍返回空列表，尚无 artifact index；
- DocumentArtifact builder 尚未接入完整文档摄入用例；
- 没有 orphan/ref consistency scanner、保留策略或垃圾回收；
- 没有把 MTP trace 中每条 READ/RUN 证据自动提升为细粒度 source memory/document refs；
- artifacts 与 Qdrant 写入不是原子事务；
- 完整版本重建模型已经存在，但公开回档、diff 和 UI 浏览仍未完成。

这些缺口不否定 Artifact 的当前价值，但它们限定了系统现在能声称的是“具有可验证来源与版本基础”，而不是“已经拥有完整审计账本”。
