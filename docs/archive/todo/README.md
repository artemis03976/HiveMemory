---
title: Archived Todo
status: current
owner: project
scope: completed-todo-history
last_reviewed: 2026-09-27
---

# Archived Todo

本目录保存已完成的修复与技术债处置历史；当前行为从对应事实文档进入，活动待办见 [Todo](../../todo/README.md)。

- [WorkspaceAsset 上传的认证上下文与 scope 不一致](./workspace-asset-upload-access-scope-mismatch.md)：2026-10-04 随 v0.7.0 A1 访问边界返工结构性修复（上传只使用授权返回的 scope）；当前事实见 [Workspace 架构](../../architecture/workspace.md)第 4.5 节。
- [MTP 缓存命中作用域重验](./mtp-cache-scope-revalidation.md)：v0.6.2 修复与 READ/RUN 回归入口；当前事实见 [MTP 契约](../../contracts/mtp.md)。
- [ShortTermMemoryStore 边界收敛](./short-term-memory-store-boundary-cleanup.md)。
- [记忆来源与作者语义](./memory-provenance-vs-authorship.md)。
- [Topic 关闭时的失败隔离](./topic-shutdown-per-topic-failure-isolation.md)。
- [Memory Alias 重名缺陷](./memory-alias-uniqueness.md)：同 Workspace alias 唯一性的两层修复（PR #103）；当前事实见 [MemoryLibrary](../../patchouli/memory-library.md)第 1.2 节，后续事项见 [Todo](../../todo/memory-alias-follow-ups.md)。
- [AgentProfile 模型演进](./agent-profile-model-evolution.md)：模型自持 `agent_id`（PR #103）；当前事实见 [Alice](../../alice/README.md)，开放项转为 [Idea](../../ideas/workspace-network-task-process-architecture.md) P-10。
