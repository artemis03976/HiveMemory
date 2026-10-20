---
title: WorkspaceAsset
status: current
owner: workspace
scope: workspace-asset-store-ownership-states-binding-and-close
code_paths:
  - src/hivememory/workspace/assets/
  - src/hivememory/core/models/workspace_asset.py
  - src/hivememory/core/ports/workspace_assets.py
  - src/hivememory/system/services/asset_materialization_reader.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
related_docs:
  - docs/system/attachments.md
  - docs/patchouli/perception.md
  - docs/patchouli/artifacts.md
last_reviewed: 2026-10-19
---

# WorkspaceAsset

WorkspaceAsset 是 workspace 的进程内资源：用户上传的附件先成为 Workspace 的工作资产，上传本身不让它成为记忆。本文说明 AssetStore 的所有权、状态与交接边界；从上传、解析到 Chat 选择、编译与 Artifact promotion 的完整链路见 [Chat 附件链路](../system/attachments.md)。

## 1. 所有权和生命周期

组合根在运行时装配中只创建一个 `InMemoryWorkspaceAssetStore`（`workspace/assets/store.py`）。Store 是当前进程内 WorkspaceAsset、representation、opaque ref、幂等记录和 lease 的权威真相源，通过窄化的 Reader/Command port（`core/ports/workspace_assets.py`）提供给业务消费者。它不查询 Topic，也不负责 binding 或 settlement。lease 的业务消费者有两个：任务进程在 CPU 分配时为 Chat 选择的附件取得 lease，登记在进程工作集中，进程结束时由取得它的 `CPUAllocator` 释放（[任务进程](./task-process.md#6-cpu-分配)）；Patchouli 在 Artifact promotion 时按 binding 自行取得并释放。

资产、表示和引用均为当前 Store 存活期内的运行时对象。`close_and_clear()` 进入不可逆关闭状态后清空 asset、representation、ref、operation token、幂等记录、REMOVED 记录和 lease bookkeeping；关闭后的 System 不能重新打开该 Store，必须重新装配进程并重新上传资源。

## 2. 两级状态和命名命令

资产聚合状态为 `PROCESSING`、`READY`、`FAILED`、`REMOVED`；representation 状态为 `PENDING`、`PROCESSING`、`READY`、`FAILED`。状态只能由 Store 的命名 command 推进：

1. 创建操作按 `(WorkspaceIdentity, client_operation_id)` 幂等；相同操作的 metadata 不一致时报告冲突。
2. RAW representation 可以原子注册为 READY；其他 representation 先 PENDING，再由 `start_representation()` 签发 operation token 进入 PROCESSING。
3. `complete_representation()` 或 `fail_representation()` 校验 revision/token，并在同一临界区更新 representation 与 required representation 对应的资产聚合状态。
4. 只有资产 READY 且 preference 选中的 representation READY 时，`resolve_asset()` 和 `acquire_ready_representation()` 才成功。文档资产的 required representation 是 `EXTRACTED_TEXT`，RAW READY 本身不足以表示文档可用。
5. `REMOVED` 是不可逆终态。重复 remove 幂等，晚到的 parser callback、representation command 或 lease acquire 不能复活资产；活跃 lease 自己持有冻结内容，直到消费者显式 release。

上传经 root API 的 `RootWorkspaceAssetService.upload_asset` 进入（[能力层](./capability.md#3-root-api)）；解析交接由 `AttachmentParseService` 完成。

## 3. 身份与归属

WorkspaceAsset 以 `WorkspaceAssetKey(workspace_identity, asset_id)` 寻址，外部使用当前 Store 的 opaque `WorkspaceAssetRef`。它不保存 `visibility`、`created_by_agent_id`、`created_by_team_id` 或 actor-policy target：同一 Workspace 内不同 Agent/Team 的资产访问结果一致；跨 Workspace 的 ref、asset key 或 URI 均不能绕过归属校验。

Store、解析流程与 `WorkspaceAssetReaderPort` 仍接收旧的 `IdentityScope` 做归属检查。System 的 `AssetMaterializationReader` 兼容适配器接收 Patchouli 提交的 `belong_to`，仅在一次 representation 租借调用内以归属和保留 `system` actor 组装旧接口所需的 scope，不保存到后台任务、不返回给 Patchouli；Patchouli 只使用归属读取端口。此例外仅限该适配器，不得扩散，拆分见 [WorkspaceAsset 归属拆分 Todo](../todo/workspace-asset-ownership-identity-split.md)。

## 4. TopicAssetBinding 交接

`TopicAssetBinding` 是 Workspace 与 Patchouli Topic 之间的交接事实。它只记录 `asset_id`、opaque `asset_ref`、首次使用它的 `interaction_id` 和时间，不复制 WorkspaceAsset snapshot、representation 内容、actor-policy 字段或第二份 Topic/Workspace 坐标。

绑定只能在一轮成功 Interaction 中建立：上游消费者先根据用户明确选择的 READY ref 取得 representation lease，完成本轮 Interaction 后，再由 Patchouli 的 Topic 所有者提交首次 binding。调用方只应交接已经完成前置校验的 `(asset_id, asset_ref)`；Topic 写入口不会自行反查 AssetStore 或复制附件内容。重复使用同一资产只命中既有关系，不覆盖首次 Interaction 或首次绑定时间；上传、最近资产列表和 UI selection cache 不会产生 binding，没有 binding 的 WorkspaceAsset 是合法 orphan。

Asset remove 不回调 Patchouli，也不清理 binding。AssetStore 与 Topic 所有者之间不使用共同控制器、两阶段提交或额外协调器；binding 随所属 Topic 的 settle/evict 生命周期完成清理。Topic buffer 的结构、并发保护、compact/settle/evict 矩阵和 shutdown 批处理属于[Perception 与短期话题](../patchouli/perception.md)及 [MemoryLibrary](../patchouli/memory-library.md)，不在本文重复定义。

## 5. 关闭与失败边界

System 停止时，Alice 卸载运行入口、Patchouli 完成已接纳工作的 drain 之后，最后才清空 WorkspaceAssetStore（[Workspace](./README.md#52-停止)）。这样 settlement consumer 可以在 drain 期间按既有交接约定用 task 中的 asset ref 反查 Store、持有 lease 并在完成后 release；Store 不调用 Patchouli controller 的 `wait_all`，也不查询 Topic 或 binding。`close_and_clear()` 幂等。

WorkspaceAssetStore 的清理不是队列可靠性或跨 Store 事务的替代品。若上游 shutdown 尚未完成，System 不应以提前清空 Store 来掩盖活跃 lease；如果关闭过程失败，System 报告失败事件而不是把未完成的消费者工作伪装成正常的 `SYSTEM_STOPPED`。

## 6. 已知限制

- WorkspaceAssetStore、opaque ref 和 lease 只承诺当前进程生命周期，不提供跨重启恢复；
- Store、解析流程与旧 Reader 端口仍使用 `IdentityScope`（第 3 节）。

## 7. 代码与测试入口

- [`InMemoryWorkspaceAssetStore`](../../src/hivememory/workspace/assets/store.py)、[`parse_service.py`](../../src/hivememory/workspace/assets/parse_service.py)、[`upload.py`](../../src/hivememory/workspace/assets/upload.py)、[`workspace ports`](../../src/hivememory/core/ports/workspace_assets.py)、[`workspace_asset.py`](../../src/hivememory/core/models/workspace_asset.py)（含 `TopicAssetBinding`）、[`AssetMaterializationReader`](../../src/hivememory/system/services/asset_materialization_reader.py)；
- 测试：[`tests/unit/workspace/assets/test_store.py`](../../tests/unit/workspace/assets/test_store.py)、[`test_workspace_asset_runtime.py`](../../tests/integration/system/test_workspace_asset_runtime.py)、[`tests/integration/workspace/capability/test_assets.py`](../../tests/integration/workspace/capability/test_assets.py)、[`test_workspace_asset_upload_api.py`](../../tests/integration/system/test_workspace_asset_upload_api.py)、[`test_workspace_asset_chat_selection.py`](../../tests/integration/system/test_workspace_asset_chat_selection.py)。
