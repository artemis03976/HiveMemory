---
title: Todo
status: current
owner: project
scope: small-defects-and-technical-debt
last_reviewed: 2026-09-01
---

# Todo

本目录用于范围较小、排期灵活的缺陷和技术债。跨系统功能或需要完整迁移、验收方案的工作应进入 `plans/`。

当前事项：

- [Topic shutdown 逐 Topic 失败隔离](./topic-shutdown-per-topic-failure-isolation.md)；
- [Topic `/compact` 系统指令接入](./topic-compact-command-ingress.md)；
- [Memory Garden 接入真实语义检索](./frontend-memory-semantic-search.md)；
- [建立前端身份状态所有权](./frontend-identity-ownership.md)；
- [Page Folding 跨入口上下文与证据后续技术债](./page-folding-cross-ingress-follow-ups.md)；
- [Work Queue Runtime 多 lane 拓扑技术债](./work-queue-runtime-lane-topology.md)；
- [RuntimeEvent 生产端迁移后续](./runtime-event-producer-migration.md)；
- [统一前端 mock fallback 的状态披露](./frontend-mock-fallback-disclosure.md)；
- [补齐 Alice Runtime 健康探针](./alice-health-probes.md)；
- [MTP 缓存命中作用域重验](./mtp-cache-scope-revalidation.md)；
- [ShortTermMemoryStore 职责边界与存储键封装](./short-term-memory-store-boundary-cleanup.md)。

Todo 只保存问题、证据、影响和完成条件。若事项扩展为跨系统功能或身份架构，应升级为 Plan；若已有项目 Issue，则链接 Issue，避免维护两份详细状态。
