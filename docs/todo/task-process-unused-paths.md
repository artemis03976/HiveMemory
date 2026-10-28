---
title: 任务进程中无生产使用方的代码与可合并小项
status: todo
owner: workspace
scope: task-process-unused-production-paths
code_paths:
  - src/hivememory/workspace/process/service.py
  - src/hivememory/workspace/process/outputs.py
  - src/hivememory/workspace/process/runner.py
  - src/hivememory/workspace/process/events.py
  - src/hivememory/workspace/process/table.py
  - src/hivememory/workspace/authentication.py
  - tests/helpers/registered_chat.py
related_docs:
  - docs/governance/baselines/task-process-concept-inventory-2026-10-28.md
  - docs/workspace/task-process.md
  - docs/architecture/decisions/0002-unique-identities-and-minimal-concurrency.md
last_reviewed: 2026-10-28
---

# 任务进程中无生产使用方的代码与可合并小项

## 问题与证据

[任务进程概念盘点](../governance/baselines/task-process-concept-inventory-2026-10-28.md)第 3、4 节逐个核对了生产使用方（`src/` 与 `frontend/src/`）。下列代码只有测试在用，或者用两种形式表达同一个事实。测试调用点于 2026-10-28 用 AST 统计。

删除项：

1. **非流式交付**：`run_process(stream=False)` 的重载、`_deliver_once`、`NonStreamingResult`、`NonStreamingAgentOutcome`、`NonStreamingCommandOutcome` 与 `non_streaming_result`。`POST /chat` 只用流式交付。测试有 35 处调用，分布在 13 个文件。
2. **按句柄取消**：`cancel_process(ProcessHandle)` 的重载。断连经 `_detach` 直接请求停止，`POST /chat/stop` 用 `process_id` 加请求级 context。测试有 22 处调用，分布在 5 个文件。
3. **诊断查询**：`ActorAuthenticationGateway.describe_context`、`WorkspaceAuthenticator.describe` 与 `AccessGrantSummary`。唯一的生产用途是：按 `process_id` 取消、进程不存在或不可控时，给 `chat.run.cancel_requested` 补上请求方的 `workspace_id` 观测标签。测试有 34 处调用，分布在 11 个文件，多数用来断言 context 已撤销。
4. **重复 `process_id` 的注册预检**：server 以 uuid4 生成 `process_id`，进程表登记本身也拒绝重复。按 [ADR-0002](../architecture/decisions/0002-unique-identities-and-minimal-concurrency.md)，不为随机碰撞保留专门的业务分支。

合并项：

5. 停止原因 `stream_closed` 与 `client_disconnected` 都表示交付方离开，合并为一个。
6. `ExecutionLabels` 从 `BoundProcessEvents.labels` 移到进程记录；执行器、CPU 分配与线程表改从进程记录读取。
7. 删除 `_require_prepared_workspace`：prepare 以目标 scope 调用，Workspace 归属由资源 owner 校验。

## 影响范围

- **生产代码**：`workspace/process/` 的注册入口、输出与执行器，事件发布器，以及 `workspace/authentication.py`。约 170 行删除，合并项另有少量改动。
- **测试**：
  - 非流式交付的调用改为消费流式交付，可复用 `tests/helpers/registered_chat.py` 已有的 `stream_registered_chat`；
  - 按句柄取消的调用改为 `process_id` 加请求级 context；
  - 用 `describe` 断言撤销的测试，改为经操作授权者确认 context 已撤销（`context_not_issued`）；
  - 预检的回滚测试随预检一起删除。
- **观测**：进程不存在或不可控时，`chat.run.cancel_requested` 不再携带请求方的 `workspace_id`；交付方离开只剩一个停止原因名。两者都只影响 RuntimeEvent，不影响 SSE 交付与业务结果。
- **事实文档**（随实现同步）：
  - [任务进程](../workspace/task-process.md)：注册入口签名、非流式交付、取消依据、关闭时的停止原因；
  - [项目总览](../PROJECT.md)的能力列表；
  - [错误模型](../contracts/error-model.md)中的 `NonStreamingCommandOutcome`；
  - [路由与事件](../contracts/routes-and-events.md)中的非流式结果字段；
  - [认证](../workspace/authentication.md)中的诊断查询。

## 完成条件

- 删除项从生产代码中移除，合并项完成；`POST /chat` 与 `POST /chat/stop` 的公开行为不变；
- 测试经公开交付形态驱动进程，不保留只守护被删形态的测试；
- 上列事实文档同步更新；
- 定向测试、CI 等价门槛、ruff、black 与 mypy 通过。

## 追踪

- 2026-10-28：由任务进程概念盘点登记，未绑定版本或 Issue。盘点中的其余项（系统关闭超时后的硬取消、为 controller 预留的部分、主线程 context）不属于本 Todo，处置见盘点第 9 节。
