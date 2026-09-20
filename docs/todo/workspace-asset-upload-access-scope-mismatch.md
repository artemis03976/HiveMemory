---
title: WorkspaceAsset 上传的认证上下文与 scope 不一致
status: todo
owner: system-workspace
scope: workspace-asset-upload-access-scope-consistency
priority: P1
code_paths:
  - src/hivememory/system/application/workspace_asset_service.py
  - src/hivememory/workspace/access.py
  - src/hivememory/system/runtime/workspace/store.py
related_docs:
  - docs/archive/plans/v0.7.0-a1-workspace-access-boundary.md
  - docs/system/attachments.md
last_reviewed: 2026-09-19
---

# WorkspaceAsset 上传的认证上下文与 scope 不一致

## 状态与处理决定

2026-09-19 在 v0.7.0 A1 重新实现的代码审查中确认，记录为已知 bug，尚未修复。由于修复可能涉及 Asset 链路的身份传递及兼容行为，按用户决定单独留待后续处理，具体修复版本未定。本记录承接该审查发现，不改变 [A1（已归档）](../archive/plans/v0.7.0-a1-workspace-access-boundary.md) 中“请求 DTO 不得覆盖可信身份”的目标约束，也不将记录问题等同于验收通过。

证据基线为 `318d8f02` 之上的 A1 工作区实现；以下描述针对当时的带 access 上传路径，不代表 v0.6.2 的稳定基线。

## 问题与影响

[WorkspaceAssetApplicationService.upload_asset](../../src/hivememory/system/application/workspace_asset_service.py) 同时接收 `access` 与 `identity_scope`。提供 access 时，共享检查确认其具有 `management.asset` 权限，但方法未校验传入 scope 与 `access.identity_scope` 是否一致，随后直接使用传入 scope 选择串行门分区、注册资产并交给解析服务。

因此，调用方可以持有 W1 的合法上传权限，却通过另一个 scope 把资产写入未获准进入的 W2；这个 scope 也可以同时替换 Actor 与 owner。问题发生在授权结果与实际资源操作之间：认证上下文自身不需要被篡改，也不需要通过 W2 的准入。

已确认影响为应用服务入口的跨 Workspace 写入及身份一致性失效。复现直接调用真实应用服务，没有验证外部 HTTP/harness 能否构造同样请求，不将其扩大描述为已确认的远程入口利用。

## 复现证据

使用真实 `ActorAuthenticationGateway`、`WorkspaceAccessGuard`、上传应用服务、`AttachmentParseService` 和进程内 `InMemoryWorkspaceAssetStore`：

1. 只为用户 `review-user` 的 Actor `uploader` 在 Workspace `permitted` 中登记 `management.asset`，认证取得 context。
2. 构造另一份有效 scope：用户/owner 为 `other-user`、Actor 为 `other-actor`、Workspace 为 `forbidden`。注册表中没有对应准入记录，直接请求进入该 Workspace 被拒绝，reason 为 `actor_not_admitted`。
3. 调用 `upload_asset(access=context, identity_scope=另一份 scope, ...)`，上传普通文本附件。
4. 当前结果为 `created=True`，返回资产归属于 `other-user/forbidden`，该 Workspace 中资产数量变为 1。

预期应在读取上传内容、注册资产或启动解析之前拒绝 scope 冲突，目标 Workspace 不产生资产或解析状态。

当次相关测试共 178 项通过，但已有附件测试通过 [make_upload_service](../../tests/helpers/attachment_parsing.py) 使用无 access 的兼容路径，未覆盖上述带 access 的身份冲突。真实组件的独立复现确认了该缺陷，尚未加入回归测试。

## 后续处理范围与完成条件

修复时先核对上传入口到 Store/解析服务的 scope 传递，以及无 access 的既有 HTTP 兼容路径；具体签名和兼容调整随 Asset 链路方案确定。可信 context 是认证后操作范围的依据，另行传入的 scope 只能用于一致性校验。

- 带 access 且 scope 一致的上传正常完成；串行门、资产注册和解析均使用同一可信范围。
- 跨 Workspace、跨 owner，以及同 Workspace 下替换 Actor 的 scope 冲突均在副作用前拒绝；不消费上传源、不创建资产、不启动解析，并返回稳定的 scope 冲突错误。
- 缺少 `management.asset`、context 过期或已失效时继续拒绝，`asset.acquire` 不因此获得上传权限。
- 明确无 access 兼容入口的保留或迁移方式，不因本次修复意外改变已声明的兼容范围。
- 以真实网关、行为检查、上传服务和内存 Store 增加回归验证，并复核上传幂等与解析交接；通过后更新本记录并归档。
