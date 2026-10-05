---
title: WorkspaceAsset 内部归属与操作身份拆分
status: todo
owner: workspace
scope: workspace-asset-store-parser-and-materialization-reader-ownership-boundary
code_paths:
  - src/hivememory/workspace/assets/store.py
  - src/hivememory/workspace/assets/parse_service.py
  - src/hivememory/core/ports/workspace_assets.py
  - src/hivememory/system/services/asset_materialization_reader.py
related_docs:
  - docs/architecture/workspace.md
  - docs/system/attachments.md
  - docs/contracts/subsystem-contracts.md
  - docs/archive/plans/v0.7.0-identity-access-batch-2.md
last_reviewed: 2026-10-04
---

# WorkspaceAsset 内部归属与操作身份拆分

## 问题与证据

身份与访问体系第二批已将 Patchouli 的内部调用、交互记录和后台任务拆为 `belong_to: WorkspaceIdentity` 与必要的 `from_actor: ActorIdentity`。用户明确将 `WorkspaceAssetStore` 的调整排除在本次范围外：资产 Store、解析服务和既有 Reader/Command 端口仍接收 `IdentityScope`，虽然资源归属与幂等键只使用其中的 Workspace。

这与资源 owner 内部不继续传播或组装操作 scope 的通用规则存在一个已登记的偏差。资产 owner 仍执行原有 hard boundary：未知与跨 Workspace token 均不可读；本 Todo 不表示该隔离检查被绕过。

Patchouli 的记忆物化现接收 `WorkspaceAssetMaterializationReaderPort`，只传 `belong_to`。组合根注入的 `AssetMaterializationReader` 在一次租借调用内构造归属对应的 `system` scope，转调旧 Reader；scope 不返回 Patchouli，也不进入后台记录。该桥接是临时接口适配，不代表通用的授权点或新认证路径。

## 影响范围

- 资产 Store、解析服务及其 Reader/Command 端口的内部签名；
- workspace 能力层在完成操作授权后的资产交接；
- 上传、解析、representation 租借与关闭清理的调用方和测试；
- System 的记忆物化 reader 适配器及其在组合根的装配。

Alice/CPU 的执行身份、PendingAtom 回读范围与 Import Bus `/ingest` 不由本 Todo 承接。资产 ownership、状态机和租借生命周期也不因参数拆分另设所有者。

## 完成条件

- 公开授权边界以下只使用资产归属和确有必要的发起者，不保存或重新组装 `IdentityScope`；
- Store/解析服务、端口与全部消费者一起切换，资产 owner 的归属检查保持生效；
- 移除 `AssetMaterializationReader` 的旧 scope 桥接，Patchouli 继续只接收归属读取端口；
- 上传幂等、未知/跨 Workspace token 隐藏、解析终态、租借释放和关闭清理回归通过；
- 当前架构、附件契约与 AGENTS.md 同步移除这一临时例外。

## 追踪

- 2026-10-04：用户在身份第二批任务中明确暂缓 `WorkspaceAssetStore` 调整，本地阶段收尾时登记；实施历史见 [第二批归档计划](../archive/plans/v0.7.0-identity-access-batch-2.md)。目前未绑定后续版本或 Issue。
