---
title: Workspace 文档收口现状与待办
status: todo
owner: project
scope: workspace-documentation-current-state
code_paths:
  - src/hivememory/core/models/
  - src/hivememory/system/runtime/workspace/
  - src/hivememory/system/assembler.py
  - src/hivememory/system/system.py
  - src/hivememory/patchouli/memory_library/
  - src/hivememory/patchouli/services/
  - src/hivememory/patchouli/runtime/
  - src/hivememory/engines/retrieval/
  - src/hivememory/engines/artifacts/
  - src/hivememory/engines/perception/
  - src/hivememory/engines/generation/
  - src/hivememory/gateway/
  - src/hivememory/server/
related_docs:
  - docs/DOCUMENTATION.md
  - docs/plans/v0.6.2-workspace-mvp.md
  - docs/architecture/overview.md
  - docs/architecture/boundaries.md
  - docs/architecture/data-model.md
  - docs/system/composition.md
  - docs/system/runtime-and-bus.md
  - docs/patchouli/memory-library.md
  - docs/patchouli/perception.md
  - docs/patchouli/generation.md
  - docs/patchouli/artifacts.md
  - docs/patchouli/retrieval.md
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
  - docs/governance/security/identity-and-execution-safety.md
  - docs/ideas/ae2-hivememory-architecture-analogy.md
  - docs/ideas/workspace-mvp-chat-attachments-design.md
last_reviewed: 2026-09-01
---

# Workspace 文档收口现状与待办

本文记录 Workspace MVP 当前实现和文档分布，服务于一次性的文档收口工作。它是
`docs/todo/` 下的现状清单，不是 Workspace 当前设计的唯一来源；运行行为以代码和
测试为准，实施细节以 [v0.6.2 Workspace MVP Plan](../plans/v0.6.2-workspace-mvp.md)
为准，探索性内容仍保留在 `docs/ideas/`。

## 1. 调查范围

现状核对覆盖核心身份模型、Topic/Memory/Artifact、WorkspaceAssetStore、Patchouli
SemanticBuffer 与 binding、System shutdown、Gateway/Server 入口以及共享 runtime。
测试按照 `tests/unit/`、`tests/integration/` 和 `tests/e2e/` 的目录分类；代码路径和
测试路径只作为核对入口，不构成另一套架构说明。

## 2. P0–P6 实现现状

| 阶段 | 当前代码与行为 | 主要测试入口 |
|:---|:---|:---|
| **P0 核心模型与失败语义** | `ActorIdentity`、`WorkspaceIdentity`、`IdentityScope`、默认 Workspace resolver、owner/scope 校验、WorkspaceAsset 状态和 opaque ref 模型已存在。 | [`tests/unit/core/models/test_workspace.py`](../../tests/unit/core/models/test_workspace.py)、[`tests/unit/core/models/test_topic.py`](../../tests/unit/core/models/test_topic.py) |
| **P1 入口、Run 与传播** | Chat、Gateway、Alice、Patchouli、Passive ingress 和 child frame 传播 `IdentityScope`；普通入口解析 `main_workspace`，非默认 Workspace 使用内部 seam。 | [`tests/integration/system/test_workspace_access_propagation.py`](../../tests/integration/system/test_workspace_access_propagation.py)、[`tests/integration/patchouli/test_active_interaction_submission.py`](../../tests/integration/patchouli/test_active_interaction_submission.py) |
| **P2 资源硬隔离** | Memory schema v2、Workspace ownership、actor read policy、检索 hard filter、Artifact 复合寻址和历史缺字段兼容读取已实现。 | [`tests/unit/core/models/test_memory_v2.py`](../../tests/unit/core/models/test_memory_v2.py)、[`tests/unit/engines/retrieval/test_memory_policy.py`](../../tests/unit/engines/retrieval/test_memory_policy.py)、[`tests/integration/patchouli/test_memory_workspace_isolation.py`](../../tests/integration/patchouli/test_memory_workspace_isolation.py) |
| **P2.5 字段迁移与兼容收敛** | `IdentityScope` 只包含 actor/workspace；Interaction、Generation TaskSpec 和 codec 使用唯一 scope，`WorkScopeSnapshot` 已删除。 | [`tests/unit/patchouli/control/test_interaction_submission.py`](../../tests/unit/patchouli/control/test_interaction_submission.py)、[`tests/unit/patchouli/control/test_memory_generation_models.py`](../../tests/unit/patchouli/control/test_memory_generation_models.py)、[`tests/integration/patchouli/test_workspace_interaction_retry.py`](../../tests/integration/patchouli/test_workspace_interaction_retry.py) |
| **P3 共享组件命名域返工** | cache、queue、registry、scheduler、runtime 和 EventBus 保持既有共享 key；`RuntimeEvent.workspace_id` 只作为观测投影。 | [`tests/unit/agent_runtime/aliases/test_cache.py`](../../tests/unit/agent_runtime/aliases/test_cache.py)、[`tests/unit/alice/runtime/test_runtime_events.py`](../../tests/unit/alice/runtime/test_runtime_events.py)、[`tests/unit/system/runtime/test_runtime_events.py`](../../tests/unit/system/runtime/test_runtime_events.py) |
| **P4 System-owned AssetStore** | `_RuntimeBundle` 持有进程级唯一 Store；资产两级状态、READY-only、opaque ref、lease、REMOVED 和 shutdown `close_and_clear()` 已实现。 | [`tests/unit/system/runtime/workspace/test_store.py`](../../tests/unit/system/runtime/workspace/test_store.py)、[`tests/integration/system/test_workspace_asset_runtime.py`](../../tests/integration/system/test_workspace_asset_runtime.py) |
| **P5 binding 与 Topic 生命周期** | SemanticBuffer/ShortTermMemoryStore 提供原子 interaction apply、单写者状态、settle/compact/evict 分离、binding 真实使用事实和 shutdown drain 行为。 | [`tests/unit/patchouli/memory_library/test_buffer.py`](../../tests/unit/patchouli/memory_library/test_buffer.py)、[`tests/unit/patchouli/memory_library/test_binding_and_reservation.py`](../../tests/unit/patchouli/memory_library/test_binding_and_reservation.py)、[`tests/integration/patchouli/test_perception_flush_chain.py`](../../tests/integration/patchouli/test_perception_flush_chain.py)、[`tests/integration/patchouli/test_asset_binding_lifecycle.py`](../../tests/integration/patchouli/test_asset_binding_lifecycle.py) |
| **P6 双 Workspace walking skeleton** | 公开入口使用 `main_workspace`，内部测试 seam 使用 `isolation_workspace`；Workspace-owned 资源隔离，cache、queue、registry、runtime 和 EventBus 不按 Workspace 分区。 | [`tests/integration/patchouli/test_topic_access_chain.py`](../../tests/integration/patchouli/test_topic_access_chain.py)、[`tests/integration/patchouli/test_memory_workspace_isolation.py`](../../tests/integration/patchouli/test_memory_workspace_isolation.py)、[`tests/integration/system/test_workspace_access_propagation.py`](../../tests/integration/system/test_workspace_access_propagation.py)、[`tests/integration/system/test_workspace_asset_runtime.py`](../../tests/integration/system/test_workspace_asset_runtime.py) |

当前 Plan 正文记录 P2.5–P6 已落地，但 Plan 第 12 节仍有未勾选项，P3 章节仍保留
“返工中”字样，`docs/plans/README.md` 与 `docs/ROADMAP.md` 仍把 W0 标为 Planned。
这是文档状态尚未同步的现状，不改变代码和测试已经呈现的行为。

## 3. 当前 Workspace 语义

| 术语 | 当前含义 |
|:---|:---|
| `ActorIdentity` | 谁在执行；不承担 Workspace ownership。 |
| `WorkspaceIdentity` | `owner_user_id + workspace_key + workspace_id` 的不可变资源归属；MVP 中 key 与 ID 相同。 |
| `IdentityScope` | `ActorIdentity + WorkspaceIdentity` 的一次操作硬边界；不携带 interaction、generation、frame、request 或 trace ID。 |
| `main_workspace` | 普通产品入口的默认 Workspace。 |
| `isolation_workspace` | 仅供内部服务和隔离测试构造的第二 Workspace。 |
| `topic_id` | 领域上的全局唯一 Topic 身份；`WorkspaceTopicKey` 只负责带归属寻址和校验。 |
| Workspace-owned resource | 通过 owner/workspace 与资源 ID 共同寻址；Memory 的 actor read policy 只在 owning Workspace 内生效。 |
| `WorkspaceAsset` | 不保存 `visibility`、`created_by_agent_id`、`created_by_team_id` 或 actor-policy target；同一 Workspace 内不区分 Agent/Team 访问。 |
| `WorkspaceAssetStore` | 进程级唯一内存 Store；asset、representation、ref 和 lease 只在当前 Store 存活期内有效，shutdown 后清空。 |
| `TopicAssetBinding` | 成功 Interaction 按交接约定先明确选择 READY ref、持有 lease 并完成本轮写入后形成的使用事实；当前无真实附件入口，未使用资产可以是 orphan。 |
| `settle` / `compact` / `evict` | 分别表示 Topic 结算、buffer 压缩和从 Topic pool 移除；`archive` 保留给中期记忆进入长期记忆库。 |
| `RuntimeEvent.workspace_id` | 可选观测标签，不参与 EventBus 路由、订阅、sequence、授权或分区。 |

## 4. 当前测试验证记录

在项目内可写临时目录下运行：

```text
pytest tests/unit tests/integration -q --tb=short -m "not live_llm and not e2e and not slow" --basetemp .pytest-basetemp-d0
```

结果为 **2112 passed，1 个 pytest cache 权限 warning**。未运行 `tests/e2e/` 中依赖
真实 Qdrant、Embedding/Reranker 或 live LLM 的测试，也未运行被 `slow` 排除的测试。
默认 pytest 临时目录曾因 Windows 权限得到 `1990 passed, 122 errors`；错误均来自创建
`AppData/Local/Temp/pytest-of-29305` 时的 `PermissionError (WinError 5)`，不是已观察到的
Workspace 代码失败。

## 5. 文档分布现状

- [`docs/architecture/workspace.md`](../architecture/workspace.md) 已成为 Workspace W0
  当前事实入口；`docs/system/`、`docs/patchouli/`、`docs/contracts/` 和
  `docs/governance/security/` 中仍只需要补充各自边界内的必要链接或局部摘要。
- [`docs/plans/v0.6.2-workspace-mvp.md`](../plans/v0.6.2-workspace-mvp.md) 仍是 W0
  实施细节和阶段验收的工作依据，尚未归档。
- [`docs/ideas/ae2-hivememory-architecture-analogy.md`](../ideas/ae2-hivememory-architecture-analogy.md)
  与 [`docs/ideas/workspace-mvp-chat-attachments-design.md`](../ideas/workspace-mvp-chat-attachments-design.md)
  仍保存探索内容、W1 附件设计和 AE2 类比，不应被当作当前实现说明。
- `docs/governance/baselines/` 保留 Durability、Idempotency、Identity 和 Data Model 等
  可持续治理主题的时间点调查；本次 Workspace 文档收口记录不属于该目录。
- `docs/archive/` 中内容仅用于历史追溯，不作为 Workspace 当前实现依据。

## 6. D1 已完成与待处理事项

1. **D1 已完成**：[`Workspace 架构`](../architecture/workspace.md) 已建立，并已加入
   Architecture 索引、总体架构入口和 PROJECT 当前架构列表；文档承接了代码和测试已经
   证明的 W0 资源归属、Topic 寻址、AssetStore 生命周期、binding 和共享组件边界，并与
   [`AE2 与 HiveMemory 的架构同构性`](../ideas/ae2-hivememory-architecture-analogy.md)
   建立双向链接，明确当前初步“ME 网络”与未来完整主/子网设想的边界。
2. 在 System、Patchouli、Contracts、Security 及 PROJECT 的其他相关位置补充必要的短链接或
   局部摘要，避免复制完整模型形成平行真相源。
3. 将 Plan 的阶段状态和验收清单与最终代码、测试结果同步，并在 W0 完成后归档 Plan。
4. 将 Idea 中仍然属于 W1、AE2 或其他未来方向的内容保留在 Idea，同时修正其对 W0 当前
   实现状态的引用。
5. 完成上述整理后，检查受影响 Markdown 链接、索引和文档状态字段。

## 7. 后续收口完成条件

- Workspace 当前事实有一个明确的 Architecture 入口，且与代码和测试一致；
- 受影响的 System、Patchouli、Contracts、Security、PROJECT、Plan、Idea 和索引链接已
  完成必要同步，没有重复维护整套 Workspace 模型；
- 本文不再承担当前设计说明，完成后可删除或移入 Archive，并保留必要的追溯链接。
