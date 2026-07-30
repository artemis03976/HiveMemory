---
title: Alice Child Frame Terminal Status Guard
status: todo
owner: alice
scope: child-frame-result-and-call-status
related_docs:
  - docs/alice/orchestration.md
  - docs/alice/agent-runtime.md
  - docs/contracts/error-model.md
last_reviewed: 2026-07-29
---

# 收紧 Alice 子帧终态与 CALL 结果判断

## 问题与证据

Alice 编排文档已经记录：子帧的 `FrameExecutionResult` 当前没有被完整检查；子帧取消、达到循环预算、意外再次 `SUSPENDED` 时，Orchestrator 仍可能组装 success `MTPCallResponse`。另有异常路径把 `tool_result` 标为 error，但主 frame 的 CALL action 仍可能统一更新为 success。

## 影响

- 主 Agent 可能把未完成或失败的子任务当作成功结果继续生成；
- `TurnEvent` 中的 `tool_call/tool_result` 状态出现矛盾，影响重放、记忆生成和观测；
- 用户无法区分子 Agent 正常完成、取消、预算耗尽和异常失败。

## 完成条件

- Orchestrator 根据 `FrameExecutionResult` 明确映射 completed/cancelled/failed/suspended/budget-exhausted；
- 只有真正完成的子帧才组装 success CALL response；其他状态形成稳定 error/cancelled 结果；
- 主 frame 的 action 状态与对应 tool result 状态保持一致；
- 增加子帧取消、预算耗尽、再次挂起、异常和正常完成的回归测试；
- 更新 Alice Orchestration、Agent Runtime、error model 和流式事件说明。
