---
title: Memory Provenance vs Authorship
status: archived
owner: patchouli
scope: separate-multi-agent-provenance-from-single-valued-authorship
code_paths:
  - src/hivememory/core/models/memory.py
  - src/hivememory/core/models/artifact.py
  - src/hivememory/engines/generation/engine.py
  - src/hivememory/engines/retrieval/filter_adapter.py
  - src/hivememory/engines/retrieval/memory_codec.py
  - src/hivememory/engines/perception/models.py
  - src/hivememory/engines/artifacts/
  - src/hivememory/patchouli/control/memory_generation/
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/patchouli/services/topic_working_set.py
  - src/hivememory/core/models/topic.py
  - src/hivememory/core/models/identity.py
related_docs:
  - ../plans/perception-topic-buffer-boundary-refactor.md
  - ../plans/v0.6.2-identity-projection-cleanup.md
  - ../../plans/v0.6.2-v1-memory-legacy-migration.md
  - ../../todo/page-folding-cross-ingress-follow-ups.md
  - ../../patchouli/perception.md
  - ../../patchouli/artifacts.md
  - ../../architecture/workspace.md
  - ../../architecture/decisions/0002-unique-identities-and-minimal-concurrency.md
last_reviewed: 2026-09-06
completed_at: 2026-09-06
archived_at: 2026-09-06
implemented_by: branch refactor/identity-cleanup
superseded_by:
  - docs/patchouli/artifacts.md
  - docs/patchouli/generation.md
---

# 记忆溯源与责任主体的分离

## 实施记录

**本事项已在 `refactor/identity-cleanup` 分支完成（2026-09-06）。**

落地内容：`MetaData.contributing_agent_ids` 贡献者集合；manual / idle / LRU / shutdown 四类 SETTLE 来源统一为保留 `SYSTEM_AGENT_ID` 并聚合 block 贡献者；`BaseArtifact.owner_agent_id` 删除，MemoryCreation/VersionArtifact 改为记录 `source_agent_id` 与 `contributing_agent_ids`；来源 Agent 查询匹配贡献者集合；`TopicData.current_agent_id` 删除。本文只保留问题证据、目标裁定与完成条件，当前系统归属语义以[Artifacts 与来源追踪](../../patchouli/artifacts.md)与[记忆生成](../../patchouli/generation.md)为准。

## 事项定位

`MetaData.source_agent_id` 是单值字段，但历史上同时承载了三种不同语义：

- **溯源**：哪些 Agent 的工作产出了内容；
- **作者/责任主体（历史解释）**：曾被误解为审计链上的单一 Agent owner；
- **V1 兼容授权**：旧记录曾从来源 Agent 推断可见目标。

Workspace 落地后，一个 Topic 属于一个 Workspace，用户可以在同一话题内切换不同 Agent。一个话题因此可能包含多个顶层交互者的贡献，单值 `source_agent_id` 无法完整表达事实。

identity/workspace 收敛已经解决了身份作用域的另一类问题：后端现在以唯一的 `IdentityScope` 传递 actor 与 Workspace ownership，W0 强制 `actor.user_id == workspace.owner_user_id`，并由 `TopicWorkingSet` 冻结和复用最后一次 touch 的完整作用域。本 todo 不再追踪旧的 scope 重建错误，而只处理来源记录与历史作者/owner 解释仍然混用的问题。

本事项是数据语义和审计准确性技术债，不是当前 V2 授权漏洞。授权边界和待解决的 legacy 兼容边界见下文。

## 当前身份与 `system` 语义

`IdentityScope` 是运行时身份不变量的唯一载体，当前不为 provenance 另造第二套身份对象。对于没有具体 Agent 作为操作来源主体的操作，来源记录使用保留值 `SYSTEM_AGENT_ID = "system"`。这里的 `system` 表示：

> 没有具体 Agent 作为操作来源主体。

它不表示一个可被授权的 Agent，也不应出现在 `MemoryAccessPolicy.target_agent_id` 或 `target_team_id`。Memory 管理页面等不具备 Agent 来源的用户操作可以使用该值补全来源记录；这与“该用户仍然通过 IdentityScope 选择并校验 user/workspace”的要求相互独立。

当前 W0 的强 owner 校验仍然有效。未来如果允许 actor user 与 Workspace owner 不同，应另行定义跨用户访问策略，不是本 todo 的实现范围。

MemoryAtom 与 Artifact 都是 Workspace 资产，持久化归属由 `workspace_identity`（以及存储层的 Workspace 复合键）表达。Artifact 当前 `BaseArtifact.owner_agent_id` 仍把 Agent 写成 owner，这与 V2 的 ownership 模型不一致；该字段不能继续被解释为资产所有权。

## 授权已经与 V2 来源字段分离

新写入的 V2 Memory 中，读取授权只由 `access_policy` 决定：

- [memory_visible_to_actor](../../../src/hivememory/engines/retrieval/policy.py) 只读取 `memory.meta.access_policy`；
- [filter_adapter](../../../src/hivememory/engines/retrieval/filter_adapter.py) 的 V2 分支匹配 `meta.access_policy.visibility`、`target_agent_id` 和 `target_team_id`；
- `MemoryAccessPolicy` 已拒绝把保留的 `system` 作为可见性 target；
- V2 新写入的 `source_agent_id` 和 `source_team_id` 只记录来源，不参与可见性决定。

Memory 管理页面在通过 Workspace ownership 校验后可以观测该 Workspace 的全部记忆；Agent retrieval 仍执行 `MemoryAccessPolicy`。用户主动创建 PRIVATE/TEAM 记忆时，策略 target 必须由前端填写具体 Agent/Team，不能使用 `system`。

V1 兼容读取仍保留旧耦合：对缺少 V2 schema 信息的记录，legacy 分支可能以 `meta.source_agent_id` 推断 PRIVATE 目标，[memory_codec](../../../src/hivememory/engines/retrieval/memory_codec.py) 也保留 V1 policy 适配。这段行为只能作为历史兼容层存在，不能复制到 V2 写入或新的授权判断中。V1→V2 的历史迁移由独立计划跟踪，不能在本 todo 中假定已经完成。

因此，为 V2 增加多值 provenance 不会自动扩大权限；实现时仍须测试“改变来源记录不改变 V2 可见性”。

## Memory 与 Artifact 中仍然挤着来源和归属两个语义

| 语义 | 需要的基数 | 当前形态 | 后续处理 |
|:---|:---:|:---|:---|
| 溯源：哪些 Agent 的工作产出了内容 | 多值 | Memory 只有 `source_agent_id`；Memory Artifact 当前只有 `owner_agent_id` | 增加贡献者集合，并统一来源字段语义 |
| 操作来源：没有具体 Agent 时如何记录 | 单值或缺省 | Memory 允许保留 `system`，Artifact 尚无统一约定 | MemoryCreation/Version 使用 `source_agent_id`，允许 `system`；其他 Artifact 按类型决定是否需要来源 |
| 资产归属：谁拥有该对象 | Workspace 单值 | Memory 使用 `workspace_identity`；Artifact 仍有误导性的 `owner_agent_id` | 删除 Agent owner 语义，统一由 Workspace 持有 |
| 授权：谁能读 | 策略结构 | 已由 `access_policy` 承载 | 保持独立 |

`contributing_agent_ids` 不能替代单值来源记录，但也不再需要一个 Agent owner 字段来表达资产归属。MemoryCreationArtifact 与 MemoryVersionArtifact 应和 MemoryAtom 一样记录 `source_agent_id` 与 `contributing_agent_ids`；Artifact 的 Workspace 所有权始终由 `workspace_identity` 表达。

## 当前父子 Agent 范围

当前 Interaction/Topic 写入路径中，一次顶层 run 生成一个 `InteractionPayload`，子 Agent 在父 frame 内工作并通过 IPC return 回到父 Agent 的 working history，不单独摄入，也不单独形成 `LogicalBlock`。因此本 todo 所说的多身份 block，主要来自同一 Workspace 内用户切换不同 Agent 的多个顶层交互；如果未来要把子 Agent 的贡献也纳入 provenance，需要另行扩展交互和 block 契约。

## 结算来源的当前偏差

主动 WRITE/UPDATE 的身份语义相对清楚：

- WRITE 创建的 Memory 当前使用提交该操作的 actor Agent；
- UPDATE 针对已有 Memory 生成新版本，保留已有 metadata 的来源字段，并把本次生成的贡献者并入已有集合（主动操作先记录发起 Agent，再合并上下文贡献者；SETTLE 演化同样合并本轮贡献者）；版本 Artifact 同步记录来源与合并后的贡献者，不引入 Agent owner 语义。

被动结算（SETTLE）则存在明确的不一致：

1. [TopicWorkingSet](../../../src/hivememory/patchouli/services/topic_working_set.py) 按 `(WorkspaceIdentity, topic_id)` 保存最后一次 touch 的完整 `IdentityScope`；
2. idle、LRU、shutdown 等维护路径以及手动 settle 复用该冻结作用域，并将其带入 [TopicMaterializeTask](../../../src/hivememory/engines/perception/models.py)；
3. [MemoryGenerationCoordinator](../../../src/hivememory/patchouli/control/memory_generation/coordinator.py) 将任务标为 `MemoryGenerationSource.SETTLE`；
4. `SETTLE.creation_artifact_intent` 已经返回 `"SYSTEM"`，但 [MemoryGenerationEngine](../../../src/hivememory/engines/generation/engine.py) 构造 `Memory.meta.source_agent_id` 时仍写入冻结作用域中的最后访问 Agent；
5. artifact builder 又以 `memory.meta.source_agent_id` 填充两个 Artifact 的 `owner_agent_id`，使“系统触发的结算”与“最后访问 Agent 的单值来源/owner”同时出现。

这意味着 B1 已消除了“从最后一个 block 猜身份”和“无 block 时回落 `omni_doll`”的旧路径，但尚未消除 settle 的来源语义偏差。后续设计应裁定：

- SETTLE 的 `source_agent_id` 统一写入 `SYSTEM_AGENT_ID`，并由 `contributing_agent_ids` 保存实际参与内容的 Agent；
- `source_team_id` 在没有具体来源 Agent 时是否为空，按 Memory V2 的来源记录规则处理；
- MemoryCreationArtifact/MemoryVersionArtifact 删除 `owner_agent_id` 的资产归属语义，改为记录 `source_agent_id` 与 `contributing_agent_ids`；
- InteractionArtifact 不增加顶层 `source_agent_id`，每个 block 继续通过 `InteractionTurnSnapshot.actor_identity` 记录来源；
- manual、idle、LRU、shutdown 四类 settle 共享上述来源裁定。

上述裁定已于 2026-09-06 落地：SETTLE 的 `source_agent_id` 统一写入 `SYSTEM_AGENT_ID`，贡献者集合从 block identity 聚合；两个 Memory Artifact 记录 `source_agent_id` 与 `contributing_agent_ids`，`owner_agent_id` 已删除。

## 建议的多值 provenance 字段

在 `MetaData`、`MemoryCreationArtifact` 和 `MemoryVersionArtifact` 增加：

```python
contributing_agent_ids: tuple[str, ...] = ()
```

生成时从参与该 Memory 内容的 `LogicalBlock` identity 聚合，去重并保持首次出现顺序。该字段表达“哪些具体 Agent 贡献过内容”，不表达 Agent 所有权、历史作者解释或授权目标。

实现后（2026-09-06）：

- 单 Agent 内容的集合应为 `(source_agent_id,)`（SETTLE 使用 system 时，集合仍应只包含实际贡献者，不应把 `system` 当作内容贡献者）；
- 没有具体 Agent 作为操作来源的外部或管理操作，可以让 `source_agent_id = "system"`，但这不改变内容中实际贡献者的集合；
- 查询侧的 [`QueryFilters.source_agent_id`](../../../src/hivememory/engines/retrieval/models.py) 和 MTP `agent:` token 已改为匹配贡献者集合（保留 `meta.source_agent_id` 分支兼容无贡献者集合的历史记录），多 Agent 场景测试已补齐；
- Qdrant 当前没有为这些 payload 建立专用 index，[vector_store](../../../src/hivememory/infrastructure/storage/vector_store.py) 的现有索引策略无需因字段增加而改变，仍需以实际存储验证数组 MatchValue 行为。

Artifact 按类型采用不同的来源粒度：`InteractionArtifact` 的每个 block 已由 `InteractionTurnSnapshot.actor_identity` 记录来源，不再添加顶层 Agent 来源字段；`MemoryCreationArtifact` 和 `MemoryVersionArtifact` 需要记录与 MemoryAtom 一致的 `source_agent_id` 和 `contributing_agent_ids`。`DocumentArtifact` 等其他类型是否需要来源字段，应依据其实际生产入口单独裁定，不能由 `BaseArtifact.owner_agent_id` 继承出默认语义。

该字段属于向后兼容的增量模型扩展。历史记录缺少字段时应按空集合解码；是否为历史 V1/V2 记录补写贡献者，留给 [V1→V2 迁移计划](../../plans/v0.6.2-v1-memory-legacy-migration.md) 和单独的数据证据裁定，不能在本 todo 中凭 `source_agent_id` 猜测并回填。

在确认序列化兼容性后，预期保持 `schema_version = 2`；如果实现发现新增集合改变了持久化契约，再单独升级版本并更新迁移计划，不在本 todo 中隐式改变版本含义。

## Summary-only Topic 的已知限制

当前 `TopicData.has_content` 允许 `blocks == ()` 且 `state_summary != ""` 的 summary-only Topic。它可以驻留、参与路由和生命周期处理，但 [`TopicMaterializeTask.from_topic_data`](../../../src/hivememory/engines/perception/models.py) 在没有可保存 block 时返回 `None`，因此该 Topic 不会独立生成一个 summary-only Memory。

这已经不再是“无 block 时伪造 `omni_doll`”的问题。后续是否支持从 summary-only Topic 生成记忆，应作为独立能力设计，并明确其 provenance、来源主体和测试；本 todo 只记录边界，不把它与默认 Agent fallback 混为一谈。

## `current_agent_id` 是仍未删除的死字段

[`TopicData.current_agent_id`](../../../src/hivememory/core/models/topic.py) 仍以 `"default"` 为默认值，仓库中没有发现其业务消费者。感知/提示渲染中其他名为 `current_agent_id` 的函数参数用于当前渲染上下文，不能据此认定它们消费了 `TopicData.current_agent_id`。

该字段不应被重新解释为 Memory provenance；后续应从 `TopicData` 删除，并清理序列化、fixture 和相关测试。

## 影响范围

- [core/models/memory.py](../../../src/hivememory/core/models/memory.py)：新增 `contributing_agent_ids`，并保持 V2 access policy 与 provenance 分离；
- [core/models/artifact.py](../../../src/hivememory/core/models/artifact.py)：将 Artifact 归属收敛为 Workspace 资产，移除或重命名 `BaseArtifact.owner_agent_id` 的 Agent owner 语义；按 Artifact 类型定义来源字段；
- [engines/generation/engine.py](../../../src/hivememory/engines/generation/engine.py)：按生成模式写入 source 与贡献者集合；
- [engines/retrieval/filter_adapter.py](../../../src/hivememory/engines/retrieval/filter_adapter.py)：让 `source_agent_id` 查询匹配贡献者集合，同时保留 legacy V1 分支；
- [engines/artifacts/memory.py](../../../src/hivememory/engines/artifacts/memory.py)：让 MemoryCreationArtifact/MemoryVersionArtifact 记录 `source_agent_id` 与 `contributing_agent_ids`，不再把 Agent 写作 owner；
- [engines/artifacts/interaction.py](../../../src/hivememory/engines/artifacts/interaction.py)：保持 InteractionArtifact 以 block 内 `actor_identity` 记录来源，不增加顶层 Agent 来源字段；
- [core/constants.py](../../../src/hivememory/core/constants.py)：使用现有 `SYSTEM_AGENT_ID`，避免再引入含义不同的默认值；
- [patchouli/services/perception.py](../../../src/hivememory/patchouli/services/perception.py) 与 [patchouli/services/topic_working_set.py](../../../src/hivememory/patchouli/services/topic_working_set.py)：验证所有 settle 入口都复用冻结 scope，且不重新引入默认身份重建；
- [engines/perception/models.py](../../../src/hivememory/engines/perception/models.py)：明确 summary-only Topic 的 no-material 行为；
- [core/models/topic.py](../../../src/hivememory/core/models/topic.py)：删除无消费者的 `current_agent_id`；
- 已写入的历史 MemoryAtom：不在本 todo 中盲目回填。迁移按独立 V1→V2 计划执行。

## 明确非目标

- 不改变 `MemoryAccessPolicy` 的可见性语义，不让 provenance 字段参与 V2 授权；
- 不恢复 V1 的 `source_agent_id → target_agent_id` 推断到新写入路径；
- 不把 `system` 作为 MemoryAccessPolicy 的可见 target；
- 不引入“贡献度”权重或单值 tie-break 启发式；
- 不改变 `IdentityScope` 的字段集，不为 provenance 另造第二套运行时身份类型；
- 不允许未来跨用户 actor/Workspace 访问在本事项中提前落地；
- 不把 summary-only Topic 的生成能力混入本次字段拆分；
- 不通过修改 Page Folding 保留已不存在的 block 身份；
- 不覆盖或回写无法可靠证明来源的历史 MemoryAtom。

## 完成条件

- [x] 明确并实现 `MetaData.contributing_agent_ids`：按 block identity 去重、保持首次出现顺序；
- [x] manual、idle、LRU、shutdown 的 SETTLE 来源语义一致，并明确“没有具体 Agent 作为操作来源主体”时使用 `SYSTEM_AGENT_ID`；
- [x] Artifact 的 Workspace 归属只由 `workspace_identity` 表达，不再使用 `owner_agent_id` 表示 Agent 所有权；
- [x] MemoryCreationArtifact/MemoryVersionArtifact 的 `source_agent_id` 与 `contributing_agent_ids` 和 MemoryAtom 语义一致，SETTLE 允许 `source_agent_id = SYSTEM_AGENT_ID`；
- [x] InteractionArtifact 不增加顶层 `source_agent_id`，block 内来源记录保持完整；
- [x] Mode B/C 的真实 Agent 来源与 SETTLE 的 system 来源不互相污染；
- [x] 维护路径不再通过 Pydantic 默认值产生 `agent_id` / `team_id`，且不再出现 `omni_doll` 作为 settle 来源；
- [x] 明确 summary-only Topic 的行为并覆盖其 no-material 或生成路径测试；
- [x] `filters.source_agent_id`（包括 MTP `agent:` token）匹配贡献者集合，能检出“参与过但未收尾”的多 Agent 记忆；
- [x] provenance 字段不参与 V2 授权；测试改变来源记录不会改变 MemoryAccessPolicy 的可见性；
- [x] 历史 V1/V2 记录在新字段缺省下可正常解码，V1 source→visibility 兼容分支仅保留到迁移门槛；
- [x] 删除 `TopicData.current_agent_id`，并清理其序列化和测试引用；
- [x] 相关当前文档在实现收尾时再同步系统归属语义；本 todo 不提前把未实现设计写入 current 文档。

## 相关事项

- [v0.6.2 identity projection cleanup（已归档计划）](../plans/v0.6.2-identity-projection-cleanup.md)：B1 identity/workspace 收敛的实现依据；
- [V1→V2 memory legacy migration](../../plans/v0.6.2-v1-memory-legacy-migration.md)：历史记录迁移与兼容门槛；
- [page folding cross-ingress follow-ups](../../todo/page-folding-cross-ingress-follow-ups.md)：折叠态 Topic 和跨入口行为的后续问题；
- [Topic shutdown 逐 Topic 失败隔离（已归档 todo）](../todo/topic-shutdown-per-topic-failure-isolation.md)：同样作用于 `flush_all_for_shutdown`，可能触及相同维护路径；
- [ADR-0002：全局唯一身份与按需并发保护](../../architecture/decisions/0002-unique-identities-and-minimal-concurrency.md)。
