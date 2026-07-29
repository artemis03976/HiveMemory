---
title: Agent Profile Explicit Failure Semantics
status: todo
owner: alice
scope: profile-resolution-fail-open
related_docs:
  - docs/alice/README.md
  - docs/alice/orchestration.md
  - docs/plans/identity-isolation-and-execution-safety.md
last_reviewed: 2026-07-29
---

# 区分 Agent Profile 缺失与显式加载失败

## 问题与证据

当前未指定主 Agent 时可以使用 `OMNI_DOLL_PROFILE` fallback；但“调用方没有指定 Profile”和“调用方明确指定的 Profile 不存在、无权限或加载失败”仍可能落入相似的降级路径。当前 Alice 文档已将其记录为 fail-open 风险。

## 影响

- 配置错误或权限错误可能被掩盖，调用方误以为使用了目标 Profile；
- fallback Profile 的全量 verb/tool 白名单可能扩大实际能力；
- 运行事件和用户反馈无法准确说明失败原因。

## 完成条件

- 未指定 Profile、明确指定但不存在、无权访问、模型未就绪分别返回稳定结果；
- 只有未指定 Profile 才允许使用 Omni-Doll fallback；
- fallback 的权限与安全边界不再默认为“所有允许”而没有显式记录；
- 增加 Profile resolver、CALL 和并发请求的回归测试，并更新 Alice 当前文档。
