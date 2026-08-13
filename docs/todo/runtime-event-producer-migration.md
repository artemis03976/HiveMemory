---
title: RuntimeEvent Producer Migration Follow-ups
status: todo
owner: system
scope: runtime-event-producer-emitter-migration
related_docs:
  - docs/system/observability.md
  - docs/contracts/routes-and-events.md
  - docs/archive/plans/runtime-event-publishing-refactor.md
last_reviewed: 2026-08-13
---

# RuntimeEvent 生产端迁移后续

## 问题与证据

RuntimeEventBus、`RuntimeEventPublisher`、scoped sink 和 `AgentRunEventEmitter` 已经落地，Memory Generation 也已建立独立领域 emitter；但 Chat、Gateway、Scheduler、System lifecycle 和 Passive Ingress 等生产点尚未全部收敛到相同的发布边界。

当前外部信封和 best-effort 语义已经由 [System 可观测性](../system/observability.md) 与 [公开路由和事件](../contracts/routes-and-events.md)定义。剩余问题是生产端重复的 envelope 组装、默认 severity、关联上下文、payload 白名单和异常隔离逻辑，而不是重新设计一套事件系统。

## 约束

- 事件发生时机继续由业务控制流显式决定；
- 领域 emitter 只投影事实，不修改业务状态；
- Publisher 统一 scope、关联上下文、payload 安全转换和 best-effort 边界；
- RuntimeEvent 不进入 `GlobalSystemBus`，不承担命令、重试、提交确认或持久化审计；
- 生产端迁移不得改变现有 wire format 和消费语义。

## 完成条件

- 逐域确认 Chat、Gateway、Scheduler、System lifecycle 与 Passive Ingress 是否仍有值得迁移的长 emit/envelope 组装；
- 高收益生产域使用窄领域 emitter 或 scoped publisher，业务主流程不再重复拼装稳定字段；
- 关键 payload 有明确白名单或类型化投影，不把原始 prompt、memory context、tool args、异常正文或路径写入公共事件；
- 删除迁移后无消费者的旧 helper 与重复测试；
- 保持 RuntimeEvent sink failure isolation、stream gap、关联 ID 和现有前端消费契约；
- 每个生产域可独立迁移和验收，不要求一次性完成全项目重构。

原完整重构稿已作为历史设计移入 [Archive](../archive/plans/runtime-event-publishing-refactor.md)，其中仍有价值的当前约束已经由 System 与 Contracts 文档承接。
