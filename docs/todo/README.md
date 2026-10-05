---
title: Todo
status: current
owner: project
scope: small-defects-and-technical-debt
last_reviewed: 2026-10-04
---

# Todo

本目录用于范围较小、排期灵活的缺陷和技术债。跨系统功能或需要完整迁移、验收方案的工作应进入 `plans/`。

当前事项：

- [WorkspaceAsset 内部归属与操作身份拆分](./workspace-asset-ownership-identity-split.md)（身份第二批明确暂缓：Store/解析旧 scope 接口与 System 物化 reader 桥接）；
- [Memory alias 后续事项](./memory-alias-follow-ups.md)（未排期：无 alias 记忆的寻址、alias 查询索引）；
- [会话 `/compact` 指令](./conversation-compact-command.md)（2026-10-01 由原 Topic `/compact` 系统指令接入重写：compact 归属前台的 CPU 会话上下文，依赖 ConversationSession、命令运行位置与 Alice 会话压缩）；
- [Memory Garden 接入真实语义检索](./frontend-memory-semantic-search.md)；
- [建立前端身份状态所有权](./frontend-identity-ownership.md)；
- [Work Queue Runtime 多 lane 拓扑技术债](./work-queue-runtime-lane-topology.md)；
- [RuntimeEvent 生产端迁移后续](./runtime-event-producer-migration.md)；
- [统一前端 mock fallback 的状态披露](./frontend-mock-fallback-disclosure.md)；
- [补齐 Alice Runtime 健康探针](./alice-health-probes.md)；
- [全局路由 kwargs 与 handler 签名一致性校验](./global-route-signature-consistency-check.md)；

原“A1 访问边界返工”涉及 workspace、Patchouli、server 与配置的跨系统改动，于 2026-10-02 升级为计划，2026-10-04 实施归档：[v0.7.0 A1 访问边界返工](../archive/plans/v0.7.0-a1-access-boundary-rework.md)。原“WorkspaceAsset 上传的认证上下文与 scope 不一致”已随该计划修复并归档：[归档记录](../archive/todo/workspace-asset-upload-access-scope-mismatch.md)。原“TaskProcess 容器的职责与依赖方向”于 2026-10-04 完成并归档（编排骨架拆为执行器 `TaskProcessRunner`，进程只是状态容器）：[归档记录](../archive/todo/task-process-container-ownership.md)。

原“Page Folding 跨入口上下文与证据后续技术债”于 2026-10-01 按前台与后台拆分后删除（删除前最后版本见 commit `74b5056`）：后台 Topic 折叠的缺口并入 [Page Folding Raw Evidence Idea](../ideas/PatchouliPageFoldingRawEvidenceDesign.md) 第 9 节，前台上下文的缺口并入 [Turn 内上下文折叠 Idea](../ideas/long-running-agent-intra-turn-context-folding.md) 第 14 节。

已完成的 [MTP 缓存命中作用域重验](../archive/todo/mtp-cache-scope-revalidation.md) 已归档，继续作为隔离回归基线；[Memory Alias 重名缺陷](../archive/todo/memory-alias-uniqueness.md)与 [AgentProfile 模型演进](../archive/todo/agent-profile-model-evolution.md)于 2026-09-27 归档（后者的开放项转为 Idea 问题）；其他已完成事项见 [Archived Todo](../archive/todo/README.md)。

Todo 只保存问题、证据、影响和完成条件。若事项扩展为跨系统功能或身份架构，应升级为 Plan；若已有项目 Issue，则链接 Issue，避免维护两份详细状态。
