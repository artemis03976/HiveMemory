---
title: WorkspaceAsset 的归属拆分
status: archived
archived_at: 2026-11-06
implemented_by: 分支 docs/workspace-asset-ownership-plan（尚未提交）
superseded_by: docs/workspace/assets.md
owner: workspace
target: 版本之外的工作（身份与访问主线；候选清单 A-13）
scope: workspace-asset-owner-takes-belong-to-and-asset-bridge-removed
updates:
  - AGENTS.md
  - docs/workspace/assets.md
  - docs/workspace/identity-and-authorization.md
  - docs/workspace/capability.md
  - docs/system/attachments.md
  - docs/system/composition.md
  - docs/governance/security/identity-and-execution-safety.md
  - docs/ideas/identity-and-access-model.md
  - docs/archive/todo/workspace-asset-ownership-identity-split.md
code_paths:
  - src/hivememory/core/ports/workspace_assets.py
  - src/hivememory/workspace/assets/store.py
  - src/hivememory/workspace/assets/parse_service.py
  - src/hivememory/workspace/capability/root/assets.py
  - src/hivememory/workspace/process/allocation.py
  - src/hivememory/system/assembler.py
  - src/hivememory/patchouli/system.py
  - src/hivememory/patchouli/services/memory_generation.py
  - tests/unit/architecture/test_access_boundaries.py
  - tests/unit/architecture/test_resource_identity_boundaries.py
related_docs:
  - docs/archive/todo/workspace-asset-ownership-identity-split.md
  - docs/todo/workspace-asset-store-unused-methods.md
  - docs/archive/plans/v0.7.0-identity-access-batch-2.md
  - docs/archive/plans/unified-identity-authorization-entry.md
  - docs/ideas/identity-and-access-model.md
  - docs/workspace/assets.md
---

# WorkspaceAsset 的归属拆分

> **状态**：Archived（2026-11-06，已实施，实施记录见第 10 节）。当前事实见 [WorkspaceAsset](../../workspace/assets.md#3-身份与归属)第 3 节与[身份与授权](../../workspace/identity-and-authorization.md)第 3 节。以下为实施时的计划原文，链接已按归档位置改正。owner 在 2026-11-06 要求为候选清单的 A-13 制定计划，承接 [WorkspaceAsset 归属拆分 Todo](../todo/workspace-asset-ownership-identity-split.md)（已归档）；评审中决定没有生产调用方的五个 Store 方法保留，登记为待检查的缺陷（第 8.1 节）。

## 1. 背景和问题

身份与访问体系第二批（[归档计划](./v0.7.0-identity-access-batch-2.md)）把 Patchouli 的内部调用拆为资源归属 `belong_to: WorkspaceIdentity` 与必要的发起者 `from_actor`，当时 owner 把 `WorkspaceAssetStore` 的调整排除在外（2026-10-04）。于是：

- 资产 Store、附件解析服务 `AttachmentParseService`，以及读取端口 `WorkspaceAssetReaderPort` 与命令端口 `WorkspaceAssetCommandPort`，仍以 `IdentityScope` 为参数。
- Patchouli 的记忆物化只传归属，经另一个端口 `WorkspaceAssetMaterializationReaderPort` 读取。System 的适配器 `AssetMaterializationReader`（`system/services/asset_materialization_reader.py`，已随本计划删除） 在一次租借内，用归属和保留的 `system` 自行构造一个 `IdentityScope`，再转调旧端口。

统一身份授权入口落地之后（[归档计划](./unified-identity-authorization-entry.md)），`IdentityScope` 只在两处构造：操作授权者，以及这个适配器。架构测试把适配器登记为例外，AGENTS.md 第 3 节也把它列为唯一的已知例外。它与“资源 owner 在授权点以下只接收归属与必要的发起者”的规则不一致，也是“scope 只由授权点组装”唯一的反例。

## 2. 目标与非目标

### 2.1 目标

1. 资产 Store、解析服务与资产端口在授权点以下只接收资产归属 `belong_to: WorkspaceIdentity`，不接收、不保存、也不构造 `IdentityScope`。
2. 删除 `AssetMaterializationReader`，Patchouli 直接使用资产 Store 的读取端口。
3. `IdentityScope` 只由操作授权者构造，架构测试与 AGENTS.md 去掉这个例外。
4. 资产的对外行为不变：上传与重放、未知与跨 Workspace 引用的统一隐藏、解析终态、租借与释放、关闭清理。

### 2.2 非目标

- 资产的可见性、修改与执行授权（候选清单 A-11，[身份与访问体系 Idea](../../ideas/identity-and-access-model.md) P-8）：资产仍只按归属授权。
- 上传与附件租借的授权位置：上传仍在 root API 入口授权，租借仍在 CPU 分配中以系统的 root 工作 actor 授权，只是交给资产 Store 的参数改变。
- 资产 Store 的持久化、状态机与租借生命周期。
- 没有生产调用方的 Store 方法的去留：保留并随签名一起切换（owner，2026-11-06），检查与处置见[待检查缺陷 Todo](../../todo/workspace-asset-store-unused-methods.md)。

## 3. 当前状态和缺口

- **Store**（[`store.py`](../../../src/hivememory/workspace/assets/store.py)）：每个公开方法的第一个参数是 `identity_scope`，内部先以 `require_identity_scope` 检查类型，之后只读取 `scope.workspace_identity`，用于比对资产键的归属，以及组成幂等键 `(workspace_identity, client_operation_id)`。actor 从未被读取。
- **解析服务**（[`parse_service.py`](../../../src/hivememory/workspace/assets/parse_service.py)）：把 scope 原样转交给 Store 的命令。
- **端口**（[`workspace_assets.py`](../../../src/hivememory/core/ports/workspace_assets.py)）：读取端口有 `resolve_asset`、`list_workspace_assets`、`acquire_ready_representation`、`release_representation_lease` 四个方法；物化端口只有后两个，参数是归属。拆分之后，两个端口中租借与释放的签名完全相同。
- **生产调用方**：
  - root API 的上传 [`RootWorkspaceAssetService.upload_asset`](../../../src/hivememory/workspace/capability/root/assets.py)：`authorize_root` 之后，把授权得到的 scope 交给 Store 注册与解析服务；
  - CPU 分配 [`CPUAllocator`](../../../src/hivememory/workspace/process/allocation.py) 租借附件：`authorize_root` 之后，以 scope 调用读取端口的租借，按租借标识释放；
  - Patchouli 的附件提升（[`memory_generation.py`](../../../src/hivememory/patchouli/services/memory_generation.py)）：以 `belong_to` 经物化端口租借，组合根注入适配器。
- **没有生产调用方的 Store 方法**（2026-11-06 核对，只在测试中调用）：`resolve_asset`、`list_workspace_assets`、`create_asset`、`register_raw_representation`、`remove_asset`。
- **架构守护**：`IdentityScope` 的构造位置限定为授权者与适配器；资源 owner 的身份边界测试只覆盖 Patchouli，没有覆盖资产 owner。

## 4. 目标设计

以下都是候选设计。

### 4.1 资产 owner 只接收归属

- Store、解析服务与命令端口的每个方法把第一个参数 `identity_scope: IdentityScope` 改为 `belong_to: WorkspaceIdentity`，参数位置不变。
- 归属检查保持原样：资产键的 Workspace 不等于 `belong_to`、或引用未知时，同样以 `AssetNotFoundError` 拒绝，不泄露资源是否存在；幂等键改为 `(belong_to, client_operation_id)`，值与现在相同。
- 不保留运行时的类型检查：`require_identity_scope` 在 Store 中删除，缺少归属不是生产中会出现的输入，类型由 mypy 守住。这与统一身份授权入口中授权者、线程表不检查参数类型的做法一致。

### 4.2 一个读取端口

- 拆分之后，物化端口与读取端口中租借与释放的签名相同，是同一能力的两份声明。合并为一个 `WorkspaceAssetReaderPort`，只含 `acquire_ready_representation(belong_to, asset_ref, preference=None)` 与 `release_representation_lease(lease_id)`，由 CPU 分配与 Patchouli 共用；删除 `WorkspaceAssetMaterializationReaderPort`。
- `resolve_asset` 与 `list_workspace_assets` 没有生产调用方，不进入端口，作为 Store 自身的方法保留（第 8.1 节）。

### 4.3 调用方

- **上传**：root API 授权之后，以授权得到的目标 Workspace 调用 Store 注册与解析服务；串行门的键不变。
- **附件租借**：CPU 分配照常以系统的 root 工作 actor `authorize_root`，再以目标 Workspace 租借；版本不一致时释放租借、拒绝整轮的规则不变。
- **附件提升**：Patchouli 的参数类型改为合并后的读取端口，行为不变；组合根直接注入资产 Store。
- 删除 `system/services/asset_materialization_reader.py`。`system_actor_for_workspace` 仍被生成协调器使用，保留。

### 4.4 架构守护

- `IdentityScope` 的构造位置只剩操作授权者。
- 资源 owner 的身份边界测试扩展到资产 owner：`workspace/assets/` 与资产端口不引用 `IdentityScope` 与 `require_identity_scope`，与 Patchouli 内部的同类约束并列。

## 5. 受影响的系统、契约与数据

| 范围 | 变化 |
|:---|:---|
| core 端口 | 读取端口与命令端口改收归属；读取端口只保留租借与释放；删除物化端口 |
| workspace | Store 与解析服务改收归属；root API 的上传与 CPU 分配的租借改为交出目标 Workspace |
| Patchouli | 附件提升依赖的端口改名，行为不变 |
| System | 删除资产桥接适配器；组合根直接把资产 Store 交给 Patchouli |
| 架构守护 | 第 4.4 节 |
| HTTP、Patchouli 公开路由、事件、错误码与配置 | 不变 |

## 6. 兼容、迁移与实施分批

- 资产状态只在进程内，没有持久化数据，不需要数据迁移；对外契约不变。
- 端口与 Store 签名是进程内接口，提供方与全部调用方在同一次改动中切换，不保留旧签名或适配器。

实施分批提交，每批保持全量门槛通过：

1. **签名切换**：端口、Store、解析服务、三个调用方与组合根一起切换，删除适配器；测试与测试辅助随之改用归属。签名变化是原子的，不能再拆。
2. **架构守护**：第 4.4 节。
3. **文档收尾**：按第 9 节更新当前文档，Todo 与计划归档。

## 7. 测试和验收标准

测试写入归属层的现有文件，不建以计划命名的测试集合。本计划不改变行为，现有测试改用归属后应原样通过；只有架构守护是新增的约束。

| 不变量 | 归属层 |
|:---|:---|
| 未知与跨 Workspace 引用以同一错误拒绝；上传幂等按 Workspace 与操作标识；租借、释放与关闭清理 | `tests/unit/workspace/assets/test_store.py` |
| 解析的终态与失败投影 | `tests/integration/system/services/attachments/test_parse_service.py` |
| 上传在接收与注册之前完成 root 授权，拒绝时没有资产副作用 | `tests/integration/workspace/test_root_api_authorization.py` |
| 附件租借先授权；版本不一致时释放租借并拒绝整轮 | `tests/unit/workspace/process/test_cpu_allocation.py` |
| 附件提升按任务的归属租借并释放 | `tests/unit/patchouli/services/test_memory_generation_promotion.py` |
| `IdentityScope` 只由操作授权者构造；资产 owner 不引用 `IdentityScope` | `tests/unit/architecture/test_access_boundaries.py`、`tests/unit/architecture/test_resource_identity_boundaries.py` |

验证命令：

```bash
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85
ruff check . && black --check .
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
git diff --check
```

需要真实 Qdrant 与模型的 e2e 在环境可用时运行，其中 `tests/e2e/system/test_chat_operation_whitelist.py` 经测试辅助使用资产 Store，随签名切换。手动验收在重启后的 live 后端上进行：上传一个附件，在一轮 chat 中选用它，产生记忆后确认附件提升出 DocumentArtifact。

## 8. 风险与待决问题

### 8.1 没有生产调用方的 Store 方法（owner 已决定）

`resolve_asset`、`list_workspace_assets`、`create_asset`、`register_raw_representation`、`remove_asset` 只在测试中调用。评审时有两个选项：随签名一起切换并保留；或者一并删除，但删除 `remove_asset` 后 REMOVED 状态不再有生产来源，状态机中相关的检查、`AssetRemovedError` 与 410 错误投影也要处理，超出归属拆分的范围。

**owner 决定**（2026-11-06）：保留，随签名一起切换；登记为待检查的缺陷，逐个决定补上生产使用方还是删除，见 [WorkspaceAsset Store 中没有生产调用方的方法](../../todo/workspace-asset-store-unused-methods.md)。

### 8.2 其他风险

- **类型检查的移除**：Store 不再在入口拒绝错误类型的参数。生产调用方只有三处，都由授权点或任务的归属提供，类型由 mypy 守住；这与第 4.1 节的理由相同。

## 9. 完成时必须更新的当前文档

- **AGENTS.md** 第 3 节：删除 WorkspaceAsset 的已知例外。
- **workspace**：[WorkspaceAsset](../../workspace/assets.md) 第 3 节（身份与归属）；[身份与授权](../../workspace/identity-and-authorization.md) 第 3 节不变量 4 与第 8 节已知限制；[能力层](../../workspace/capability.md) 第 8 节结构守护。
- **system**：[Chat 附件链路](../../system/attachments.md)中附件提升的读取端口；[组合根](../../system/composition.md)中 CPU 分配与 Patchouli 注入的资产读取端口。
- **治理**：[身份与执行安全](../../governance/security/identity-and-execution-safety.md)中 WorkspaceAsset 一行与例外说明。
- **Idea 与 Todo**：[身份与访问体系 Idea](../../ideas/identity-and-access-model.md) 第 8 节的例外列表；[WorkspaceAsset 归属拆分 Todo](../todo/workspace-asset-ownership-identity-split.md) 归档，[待检查缺陷 Todo](../../todo/workspace-asset-store-unused-methods.md)中指向本计划的链接改为归档位置，[Todo 索引](../../todo/README.md)同步。
- **索引**：[Plans 导航](../../plans/README.md)、[归档计划索引](./README.md)。

## 10. 实施记录

**实现**（2026-11-06，分支 `docs/workspace-asset-ownership-plan`，尚未提交）。第 6 节的三批在同一工作区完成：

1. 签名切换：
   - 资产 Store、解析服务与命令端口的第一个参数改为 `belong_to: WorkspaceIdentity`，Store 中删除 `require_identity_scope`。
   - 读取端口合并为只含租借与释放的 `WorkspaceAssetReaderPort`，删除 `WorkspaceAssetMaterializationReaderPort`。
   - root API 的上传与 CPU 分配的附件租借在授权之后只交出目标 Workspace；Patchouli 的参数类型改为合并后的端口。
   - 组合根把资产 Store 直接交给 Patchouli，删除 `system/services/asset_materialization_reader.py`。
2. 架构守护：`IdentityScope` 的构造位置只剩 `workspace/authorization.py`；资源 owner 的身份边界测试新增资产 owner 一项，覆盖 `workspace/assets/` 与资产端口。新增的守护在 Store 重新引入 `IdentityScope` 时失败（已核对）。
3. 文档收尾：第 9 节所列文档，另加 [Patchouli](../../patchouli/README.md) 第 3 节与 [Patchouli 生成](../../patchouli/generation.md)中指向原 Todo 的链接。

与第 4、7 节的出入：

- 资产 Store 的两条测试原本断言“不同 actor 共用同一幂等键”“同 Workspace 的其他 actor 能读取”。Store 不再接收 actor，这两项性质由签名保证，相应测试改写为只断言同一 Workspace 内的幂等与跨 Workspace 隐藏。
- CPU 分配的附件测试原本断言租借收到系统的 root scope；现在断言资产 Store 只收到任务的目标 Workspace。“由系统而不是执行者授权”由 `authorize_agent` 调用位置的架构测试保证：CPU 分配不在其中。

**测试**：没有新增行为测试；资产 Store、解析、上传、chat 选用附件、附件提升、进程收尾与 root 边界的现有测试改用归属，测试辅助 `make_ready_text_asset` 与 `ScriptedAttachmentParser` 随之改为接收归属。

**验证**：

- `pytest tests/unit tests/integration -m "not live_llm and not e2e and not slow" -n auto --cov=hivememory --cov-fail-under=85`：3197 通过，覆盖率 94.74%；
- `ruff check .`、`black --check .`、`mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py`（445 个文件）、`git diff --check` 通过；`pytest tests/e2e --collect-only` 全部收集成功。

**未完成**：

- e2e 需要 Qdrant 与模型，未运行；
- 第 7 节的手动验收要用这个分支重启正在运行的后端，在日常使用中进行。
