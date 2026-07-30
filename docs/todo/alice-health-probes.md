---
title: Alice Runtime Health Probes
status: todo
owner: alice
scope: runtime-health-readiness-probes
related_docs:
  - docs/alice/agent-runtime.md
  - docs/system/observability.md
  - docs/help/troubleshooting.md
last_reviewed: 2026-07-29
---

# 补齐 Alice Runtime 健康探针

## 问题与证据

`AliceRuntime.health()` 当前主要返回 loop/worker 的固定装配状态，不验证模型端点、syscall 可用性、cache 隔离、正在运行的 frame 或迭代耗尽率。服务可能报告 `ok`，但实际 Agent run 已无法完成。

## 影响

- readiness 与真实可用性不一致；
- 运维无法区分“未装配”“模型不可用”“执行资源耗尽”和“缓存隔离异常”；
- 排障需要依赖分散日志，而不是稳定诊断结果。

## 完成条件

- 区分装配健康、依赖 ready、当前运行统计和最近失败摘要；
- 探针不执行真实用户 MTP/RUN，不泄漏 prompt、memory context 或凭据；
- 明确 liveness/readiness/diagnostic 三类语义，避免把一次短暂模型故障误报为进程死亡；
- 增加模型未就绪、syscall 不可用、无运行/有运行和迭代耗尽的测试，并同步 Help 与 System observability 文档。
