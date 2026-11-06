---
title: WorkspaceAsset Store 中没有生产调用方的方法
status: todo
owner: workspace
scope: workspace-asset-store-methods-without-production-callers
code_paths:
  - src/hivememory/workspace/assets/store.py
  - src/hivememory/core/ports/workspace_assets.py
  - src/hivememory/core/models/workspace_asset.py
  - src/hivememory/server/error_handlers.py
related_docs:
  - docs/workspace/assets.md
  - docs/plans/workspace-asset-ownership-split.md
last_reviewed: 2026-11-06
---

# WorkspaceAsset Store 中没有生产调用方的方法

## 问题与证据

制定 [WorkspaceAsset 的归属拆分](../plans/workspace-asset-ownership-split.md)计划时核对（2026-11-06）：资产 Store [`InMemoryWorkspaceAssetStore`](../../src/hivememory/workspace/assets/store.py) 有五个公开方法，生产代码里没有调用方，只在测试与测试辅助中使用。

| 方法 | 做什么 | 现状 |
|:---|:---|:---|
| `resolve_asset` | 按引用读取资产快照 | 读取端口中有声明，生产调用方只用租借与释放 |
| `list_workspace_assets` | 列出一个 Workspace 中未移除的资产 | 没有 HTTP 列表路由，前端不经它取得资产 |
| `create_asset` | 只凭元数据创建资产 | 生产只经 `register_uploaded_asset` 一次完成注册与原始内容登记 |
| `register_raw_representation` | 为 `create_asset` 创建的资产补登原始内容 | 同上，只服务 `create_asset` 的两步路径 |
| `remove_asset` | 把资产推进到不可逆的 REMOVED 并释放内容引用 | 没有删除资产的 HTTP 路由，也没有保留期清理 |

`remove_asset` 是 REMOVED 状态唯一的来源。生产中资产因此不会进入 REMOVED，Store 中针对 REMOVED 的检查、`AssetRemovedError`，以及它在 HTTP 上的 410 `workspace.asset.removed`（上传重放遇到已移除的资产）都不会在生产中出现。

## 影响范围

- 资产 Store 与资产端口的公开方法；
- REMOVED 状态、`AssetRemovedError` 与对应的 HTTP 错误投影；
- 以这些方法搭建状态的测试与测试辅助（资产 Store、解析、上传、chat 选用附件与附件提升的测试）。

这些方法不影响现有行为，归属拆分计划按 owner 决定（2026-11-06）保留它们，随签名一起切换。

## 完成条件

逐个检查后，每个方法二选一：

- 有真实需要的，补上生产使用方，并写明由谁调用。可能的需要：用户删除已上传的附件、资产的保留期清理、附件选择列表；
- 没有需要的，删除它们与只为它们存在的测试；删除 `remove_asset` 时，一并处理 REMOVED 状态、`AssetRemovedError` 与 410 错误投影的去留。

## 追踪

- 2026-11-06：owner 在归属拆分计划评审中决定保留这五个方法，登记为待检查的缺陷。未排期，没有对应 Issue。
