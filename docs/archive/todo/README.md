---
title: Archived Todo
status: current
owner: project
scope: completed-todo-history
last_reviewed: 2026-10-12
---

# Archived Todo

本目录保存已完成的修复与技术债处置历史；当前行为从对应事实文档进入，活动待办见 [Todo](../../todo/README.md)。

- [kernel 遗留命名清理](./kernel-legacy-naming.md)：2026-10-12 完成，系统工具保留 syscall 命名只去掉 Kernel（错误键改为 `mtp.run.tool_not_found`），提示词移除 kernel 与 HiveOS，后端注释改用当前组件名，前端观测面板改称 Workspace Vision、Chat 页面与中间栏改称 Chat View 与 ChatPane；当前事实见 [MTP Runtime](../../alice/mtp-runtime.md)第 5 节与 [Chat 视图](../../frontend/chat-view.md)。
- [检索分数阈值与候选截取收敛](./retrieval-hybrid-threshold-with-reranker.md)：2026-10-12 完成，分数阈值只作用于 Dense 召回（默认 0.5），Sparse、融合与重排不设阈值；融合不再提前截取，截取只在重排之后按请求的 `top_k`（缺省为 `retrieval.default_top_k`）进行一次；当前事实见 [Patchouli 检索](../../patchouli/retrieval.md)第 4、5、9 节。
- [TaskProcess 容器的职责与依赖方向](./task-process-container-ownership.md)：2026-10-04 完成，编排骨架从进程容器拆为所有进程共用的执行器 `TaskProcessRunner`，注册入口只持有生命周期依赖；当前事实见 [System 应用服务](../../system/application-services.md)第 3 节。
- [WorkspaceAsset 上传的认证上下文与 scope 不一致](./workspace-asset-upload-access-scope-mismatch.md)：2026-10-04 随 v0.7.0 A1 访问边界返工结构性修复（上传只使用授权返回的 scope）；当前事实见 [Workspace 架构](../../architecture/workspace.md)第 4.5 节。
- [MTP 缓存命中作用域重验](./mtp-cache-scope-revalidation.md)：v0.6.2 修复与 READ/RUN 回归入口；当前事实见 [MTP 契约](../../contracts/mtp.md)。
- [ShortTermMemoryStore 边界收敛](./short-term-memory-store-boundary-cleanup.md)。
- [记忆来源与作者语义](./memory-provenance-vs-authorship.md)。
- [Topic 关闭时的失败隔离](./topic-shutdown-per-topic-failure-isolation.md)。
- [Memory Alias 重名缺陷](./memory-alias-uniqueness.md)：同 Workspace alias 唯一性的两层修复（PR #103）；当前事实见 [MemoryLibrary](../../patchouli/memory-library.md)第 1.2 节，后续事项见 [Todo](../../todo/memory-alias-follow-ups.md)。
- [AgentProfile 模型演进](./agent-profile-model-evolution.md)：模型自持 `agent_id`（PR #103）；当前事实见 [Alice](../../alice/README.md)，开放项转为 [Idea](../../ideas/workspace-network-task-process-architecture.md) P-10。
