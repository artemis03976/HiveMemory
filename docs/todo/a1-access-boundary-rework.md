---
title: A1 访问边界返工
status: todo
owner: workspace-patchouli-system
scope: operation-check-relocation-compat-branch-exit-and-production-gateway-wiring
priority: v0.7.0
code_paths:
  - src/hivememory/patchouli/application/access_consumption.py
  - src/hivememory/workspace/capability/
  - src/hivememory/workspace/authentication.py
  - src/hivememory/server/
related_docs:
  - docs/archive/plans/v0.7.0-a1-workspace-access-boundary.md
  - docs/architecture/workspace.md
  - docs/ideas/workspace-network-task-process-architecture.md
  - docs/todo/workspace-asset-upload-access-scope-mismatch.md
last_reviewed: 2026-09-27
---

# A1 访问边界返工

## 状态与处理决定

**排期**（owner，2026-09-27）：在 v0.7.0 内、任务进程表与任务请求唯一注册入口计划完成之后接入，此时已有稳定的入口（[总 Idea](../ideas/workspace-network-task-process-architecture.md#61-已决定事项) 6.1）。本项仍不阻塞该计划的制定和实现；该计划不以本项完成为前提。

本项汇总 A1（[已归档](../archive/plans/v0.7.0-a1-workspace-access-boundary.md)）交付后仍未完成的三部分工作。它们此前分别交由 v0.7.0 A2 与 A6 计划收口；两份计划已于 2026-09-27 作废删除（删除前最后版本见 commit `dda9d9d`），相关事项改由本 Todo 承接。

## 问题与证据

1. **写入与管理路径的 operation 检查仍在 Patchouli application。** 读取路径（点读、alias 批读、语义检索、Profile 解析）已在 `workspace/capability/` 中于 backing 调用前执行 `authorize_operation`，Patchouli 侧经 `access_consumption.backing_scope` 只校验 context 有效性；其余公开方法（Memory 管理写入、`interaction.submit`、`memory_intent.submit`、Topic 与任务管理等）仍经 `access_consumption.verified_scope` / `required_scope` 在 Patchouli application 内检查行为白名单。
2. **迁移期裸 scope 兼容分支仍然存在。** `access_consumption.py` 冻结了一份兼容清单：管理 CRUD、alias/语义读取、Agent Profile、Memory 任务、Topic 管理、附件上传以及 `PatchouliService` 的 prepare/finalize/cleanup 等方法在缺少 access 时按裸 `IdentityScope` 受信处理。
3. **生产入口没有接入认证网关。** `ActorAuthenticationGateway` 已由组合根装配，但 `server/` 中没有调用方；生产 HTTP 请求实际不经过网关认证，依赖第 2 条的兼容分支运行。认证 context 在 System 停止时的关闭时机也尚未随生产接线确定。

## 影响

- 行为授权分散在两个位置：读取在能力层，写入与管理在 Patchouli application；
- 兼容分支存在期间，“请求 DTO 不得覆盖可信身份”只对带 access 的调用成立；附件上传的一个具体缺陷见 [WorkspaceAsset 上传 scope 不一致](./workspace-asset-upload-access-scope-mismatch.md)；
- 访问边界的事实文档（[Workspace 架构](../architecture/workspace.md)第 4 节等）描述的是 A1 交付时的状态，尚未反映读取路径检查点的迁移。

## 约束

- 目标检查位置可能随 [Workspace 网络与任务进程架构 Idea](../ideas/workspace-network-task-process-architecture.md)第三部分（认证与授权流程）的决定调整；开始实施前先按该 Idea 当时的已决定事项核对目标，不以作废计划的设计为准。
- 保留 Patchouli 存储边界对资源归属与资源 policy 的独立校验，不因检查点迁移而移除。

## 完成条件

- [ ] 所有公开方法的 operation 检查在确定的目标位置执行，Patchouli application 不再承担行为白名单检查；
- [ ] 裸 scope 兼容分支删除，缺少 access 的公开调用显式拒绝；
- [ ] 生产入口经认证网关取得 context，并确定 context 在停止时的关闭时机；
- [ ] 返工完成后，把 A1 以来的实际变化统一整理进入事实文档（[Workspace 架构](../architecture/workspace.md)第 4 节、[错误模型](../contracts/error-model.md)第 4.4 节、[子系统公共契约](../contracts/subsystem-contracts.md)第 3.5 节）。
