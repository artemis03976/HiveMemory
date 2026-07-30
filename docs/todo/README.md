---
title: Todo
status: current
owner: project
scope: small-defects-and-technical-debt
last_reviewed: 2026-07-30
---

# Todo

本目录用于范围较小、排期灵活的缺陷和技术债。跨系统功能或需要完整迁移、验收方案的工作应进入 `plans/`。

当前事项：

- [Memory Garden 接入真实语义检索](./frontend-memory-semantic-search.md)；
- [建立前端身份状态所有权](./frontend-identity-ownership.md)；
- [Page Folding 跨入口上下文与证据后续技术债](./page-folding-cross-ingress-follow-ups.md)；
- [清理已声明但未接线的运行时配置](./runtime-configuration-wiring-drift.md)；
- [收紧 Alice 子帧终态与 CALL 结果判断](./alice-child-frame-terminal-status.md)；
- [Alice FrameScheduler 与取消状态的运行隔离](./alice-frame-scheduler-concurrency.md)；
- [区分 Agent Profile 缺失与显式加载失败](./agent-profile-explicit-failure.md)；
- [补齐 Agent-facing 错误 payload 的 XML escaping](./error-formatter-xml-escaping.md)；
- [统一前端 mock fallback 的状态披露](./frontend-mock-fallback-disclosure.md)；
- [补齐 Alice Runtime 健康探针](./alice-health-probes.md)。

Todo 只保存问题、证据、影响和完成条件。若事项扩展为跨系统功能或身份架构，应升级为 Plan；若已有项目 Issue，则链接 Issue，避免维护两份详细状态。
