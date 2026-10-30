---
title: 任务进程中无生产使用方的代码与可合并小项
status: archived
archived_at: 2026-10-30
implemented_by: refactor/task-process-unused-paths
superseded_by: docs/workspace/task-process.md
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
last_reviewed: 2026-10-30
---

# 任务进程中无生产使用方的代码与可合并小项

> 已完成并归档（2026-10-30）。当前事实见[任务进程](../../workspace/task-process.md)；处理方式见文末实施记录。以下正文保留登记时的原文。

## 问题与证据

[任务进程概念盘点](../../governance/baselines/task-process-concept-inventory-2026-10-28.md)第 3、4 节逐个核对了生产使用方（`src/` 与 `frontend/src/`）。下列代码只有测试在用，或者用两种形式表达同一个事实。测试调用点于 2026-10-28 用 AST 统计。

删除项：

1. **非流式交付**：`run_process(stream=False)` 的重载、`_deliver_once`、`NonStreamingResult`、`NonStreamingAgentOutcome`、`NonStreamingCommandOutcome` 与 `non_streaming_result`。`POST /chat` 只用流式交付。测试有 35 处调用，分布在 13 个文件。
2. **按句柄取消**：`cancel_process(ProcessHandle)` 的重载。断连经 `_detach` 直接请求停止，`POST /chat/stop` 用 `process_id` 加请求级 context。测试有 22 处调用，分布在 5 个文件。
3. **诊断查询**：`ActorAuthenticationGateway.describe_context`、`WorkspaceAuthenticator.describe` 与 `AccessGrantSummary`。唯一的生产用途是：按 `process_id` 取消、进程不存在或不可控时，给 `chat.run.cancel_requested` 补上请求方的 `workspace_id` 观测标签。测试有 34 处调用，分布在 11 个文件，多数用来断言 context 已撤销。
4. **重复 `process_id` 的注册预检**：server 以 uuid4 生成 `process_id`，进程表登记本身也拒绝重复。按 [ADR-0002](../../architecture/decisions/0002-unique-identities-and-minimal-concurrency.md)，不为随机碰撞保留专门的业务分支。

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
  - [任务进程](../../workspace/task-process.md)：注册入口签名、非流式交付、取消依据、关闭时的停止原因；
  - [项目总览](../../PROJECT.md)的能力列表；
  - [错误模型](../../contracts/error-model.md)中的 `NonStreamingCommandOutcome`；
  - [路由与事件](../../contracts/routes-and-events.md)中的非流式结果字段；
  - [认证](../../workspace/authentication.md)中的诊断查询。

## 完成条件

- 删除项从生产代码中移除，合并项完成；`POST /chat` 与 `POST /chat/stop` 的公开行为不变；
- 测试经公开交付形态驱动进程，不保留只守护被删形态的测试；
- 上列事实文档同步更新；
- 定向测试、CI 等价门槛、ruff、black 与 mypy 通过。

## 追踪

- 2026-10-28：由任务进程概念盘点登记，未绑定版本或 Issue。盘点中的其余项（系统关闭超时后的硬取消、为 controller 预留的部分、主线程 context）不属于本 Todo，处置见盘点第 9 节。

## 实施记录

2026-10-30 在分支 `refactor/task-process-unused-paths` 实施，删除项与合并项全部完成，`POST /chat` 与 `POST /chat/stop` 的公开行为不变。

**删除项**

- 非流式交付：`run_process` 只剩流式交付，签名为 `run_process(handle)`，`stream` 与 `detach_reason` 参数一并删除。执行器的 `run(process)` 原先把终态产出作为进程 task 的返回值交给非流式交付，现在只写入输出通道，进程 task 的类型随之改为 `Task[None]`。
- 按句柄取消：`cancel_process(process_id, *, access)` 是唯一形式。原先两种形式共用的 `reason` 参数只剩默认值一个生产用法，随重载一起删除，用户停止的原因固定为 `user_requested`；交付方离开与 `close_process` 仍经同一份停止实现。
- 诊断查询链：删除 `describe_context`、`describe` 与 `AccessGrantSummary`。按 `process_id` 取消找不到可控进程时，`chat.run.cancel_requested` 只带请求中的 `process_id`。
- 重复 `process_id` 的注册预检：删除后由进程表登记拒绝重复，失败时照常撤销已签发的 context、还原会话认领与新建。

**合并项**

- 停止原因合并为 `client_disconnected`（`CLIENT_DISCONNECTED_STOP_REASON`）：HTTP 断连在生产中一直使用这个名字，`close_process` 也改用它；server 路由不再自行声明断连原因。
- `ExecutionLabels` 移到进程记录的 `labels` 字段，`BoundProcessEvents` 不再持有标签；执行器、CPU 分配与线程表从进程记录读取。
- 删除 `_require_prepared_workspace`，prepare 结果写入工作集后不再比对归属；finalize 时 Patchouli 仍比对 prepared 的归属与 scope 的目标 Workspace。

**测试**

- 测试辅助：`tests/helpers/process.py` 新增 `collect_events`、`done_data` 与 `error_code`，经流式交付把进程跑到结束；`AccessTestComposition.is_revoked` 经操作授权者判断 context 是否已撤销（`context_not_issued`），取代断言撤销的 `describe_context`；`register_main_thread` 改为显式接收 `process_id` 与 `agent_id`。
- 断言授予内容的测试改为可观察行为：运行绑定看派生线程的 `get_thread_binding`，principal 看 `authorize_import`，actor 看阶段授权或 root 授权组装的 scope；请求级认证的测试记录交给网关的认证输入。
- 非流式用例改为消费流式交付后，与原有流式用例重复的合并为一条：封口交互记录、命令短路、CPU 自报取消、CPU 自报失败、prepare 期间停止、cleanup 期间交付方取消。只守护旧契约的测试删除：句柄取消、参数组合校验、诊断查询摘要、`detach_reason` 透传、prepare 归属漂移。认证后登记失败的回滚测试改由会话忙碌触发。
- e2e 的 `run_registered_chat` 改为返回流式事件；MTP 调用次数与 WRITE ACK 改从 `mtp_start`、`mtp_result` 事件取得。

**验证**

- `pytest tests/unit tests/integration -m "not live_llm and not e2e and not slow" -n auto`：3100 passed（同一命令在 master 上为 3114 passed，减少的是上面删除与合并的用例），警告与 master 相同；
- `pytest tests/e2e -o addopts="" -m "not live_llm and not real_infra"`：22 passed；依赖真实 LLM 的 e2e（含改动的 `test_agent_loop_e2e.py`、`test_chat_run_e2e.py`）只做了收集，未运行；
- `ruff check .`、`black --check .`、`mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py` 通过。
