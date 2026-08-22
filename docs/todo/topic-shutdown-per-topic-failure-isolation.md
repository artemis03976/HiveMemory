---
title: Topic Shutdown Per-Topic Failure Isolation
status: todo
owner: patchouli-system
scope: topic-shutdown-settlement-failure-isolation-and-aggregate-failure
related_docs:
  - docs/patchouli/perception.md
  - docs/system/runtime-and-bus.md
  - docs/plans/v0.6.2-workspace-mvp.md
  - docs/governance/testing/test-design-standards.md
last_reviewed: 2026-08-22
---

# Topic shutdown 逐 Topic 失败隔离

## 当前结论

本事项独立处理 shutdown settlement 的异常隔离，不改变已经落地的正常 skip 语义：`generation_skipped_topic_ids` 只表示 Topic 已正常结束生命周期、但没有建立 generation task；它不是失败列表，也不能用于吞掉异常。

逐 Topic 失败隔离的目标是：一个 Topic 失败时仍尽量处理后续 Topic，并等待此前已经接纳的 generation task 收敛；全部可继续的工作完成后，shutdown 仍必须以聚合失败向上游结束，不能把部分结算伪装成成功，也不能因此提前关闭 WorkspaceAssetStore。

本 TODO 不阻塞当前正常 skip 修复，也不授权修改 memory generation controller 的 `wait_all()` 语义。

## 当前链路

正常停止顺序为：

```text
System scheduler stop
  -> Passive ingress shutdown drain
  -> Alice stop
  -> Patchouli interaction submission / active finalize drain
  -> PerceptionFamiliar.flush_all_for_shutdown()
  -> MemoryGenerationTaskController.wait_all()
  -> generation timeout cancel
  -> generation queue stop
  -> Patchouli bridge / local routes unmount
  -> Gateway stop
  -> WorkspaceAssetStore.close_and_clear()
```

`flush_all_for_shutdown()` 当前按快照顺序逐个调用
`perception_layer.settle_topic(..., SHUTDOWN)`。SHUTDOWN 决策矩阵会执行
`settle=True / compact=False / evict=True`；有结算材料时再通过 local bus 提交
generation admission。

当前任一阶段抛出普通异常都会立即终止整个循环。其后果包括：

- 后续 Topic 不再结算或驱逐；
- 此前已经接纳的 generation task 不会进入本次 `wait_all()`；
- `_shutdown_drain_started` 已经置为 `True`，同一 Runtime 的再次调用会进入
  reentrant no-op，无法真实重试失败步骤；
- `PatchouliSystem.stop()` 在 queue stop 和 bus unmount 之前退出；
- 顶层 System 正确地报告 stop failure，并且不会继续关闭 WorkspaceAssetStore，
  但 Patchouli 内部已经形成部分完成状态。

## 必须区分的三类结果

### 1. 正常 settle + generation admission

Topic 生命周期结束，generation task 已建立。该 Topic 进入
`settled_topic_ids`，不进入 skip 或 failure。

### 2. 正常 settle + generation skip

Topic 生命周期结束，但没有可提交材料，或 generation coordinator 正常返回无任务。
该 Topic 同时进入 `settled_topic_ids` 与 `generation_skipped_topic_ids`。这是成功结果，
不降低 shutdown success，也不应产生 error severity。

### 3. Topic settlement failure

冻结材料、generation admission 或 evict 任一阶段抛出异常。它必须使用独立的失败
事实表达，不能写入 `generation_skipped_topic_ids`。本事项需要决定最小失败模型，例如：

```text
topic_id
stage: prepare | admission | evict
error_type
error_message
```

错误详情只用于进程内聚合与 RuntimeEvent 诊断，不进入 Topic HTTP 响应，也不得保存
traceback、WorkspaceAsset 内容或其他敏感载荷。

## 关键设计问题

### 1. 当前 SHUTDOWN 在 admission 前已经 evict

自动 shutdown 当前调用 `settle_topic()`；TriggerManager 在返回 settlement payload 前
已经清 blocks 并 evict。若随后 generation admission 失败，Topic 内容已经无法重试。

逐 Topic 隔离不能只在外层增加 `try/except`，否则虽然可以继续处理后续 Topic，却会
把“admission 失败且原文已清除”记录成一个无法恢复的普通失败。需要评估是否将
SHUTDOWN 调整为与 manual settle 相同的顺序：

```text
prepare settlement payload
  -> generation admission
  -> evict Topic
```

若复用现有 `prepare_settlement()`，必须允许显式传入 `FlushReason.SHUTDOWN`，不能把
shutdown payload 伪装为 `MANUAL_SETTLE`。这项调整只改变 Topic settlement 编排，
不得扩散到 controller、queue 或 Workspace identity。

### 2. 失败后的 Topic 是否保留

建议冻结以下原则：

- prepare 或 admission 失败时保留 Topic 内容，供本次聚合诊断或明确的 shutdown
  retry 使用；
- admission 成功后 evict 失败时，generation task 已拥有不可变材料，不能重复 admission；
- 顶层 shutdown 最终仍失败，因此 WorkspaceAssetStore 不进入正常
  `close_and_clear()`；
- 若进程被外部强制退出，进程内 Topic 与资产自然消失，本事项不引入跨重启恢复承诺。

是否在聚合失败前强制 evict 失败 Topic，需要在实现前明确裁定。强制 evict 会减少内存
残留，但会取消同进程重试能力；保留则要求修复 reentrant 状态机，不能继续使用单一
`_shutdown_drain_started` 布尔值。

### 3. drain 与最终失败顺序

即使部分 Topic 失败，也应先：

1. 继续处理其余 Topic；
2. 调用现有 controller `wait_all()` 等待已经接纳的 generation task；
3. 对超时任务沿用现有 cancel 策略；
4. 形成完整的成功、正常 skip、失败和 generation drain 摘要；
5. 最后抛出聚合 shutdown 异常。

不得在第一个 Topic 失败时跳过 generation drain，也不得为了实现该顺序修改
controller `wait_all()` 的返回模型或通用语义。

### 4. shutdown reentrant 状态

当前 `_shutdown_drain_started: bool` 无法区分 RUNNING、FAILED 与 COMPLETED。逐 Topic
失败隔离至少需要裁定以下一种方案：

- 用显式状态机记录 `NOT_STARTED / RUNNING / FAILED / COMPLETED`，FAILED 可使用冻结的
  per-topic 结果重试未完成项；
- 或保留 fail-once 语义，但 reentrant 调用必须重新抛出先前聚合失败，不能返回成功
  no-op。

在没有稳定 task identity 与 Topic settlement 幂等键的前提下，不应直接自动重跑已经
admit 成功的 Topic。

## 建议的最小实现边界

生产代码预计只涉及：

- `patchouli/services/perception.py`：逐 Topic 捕获普通异常、记录阶段、继续循环；
- `patchouli/runtime/models.py`：增加最小失败事实与失败集合；
- `patchouli/runtime/core.py`：无论 per-topic 失败与否都执行 generation drain，随后聚合失败；
- `patchouli/runtime/shutdown_drain.py`：RuntimeEvent 只投影计数和稳定阶段，不泄漏载荷；
- 必要时参数化 perception `prepare_settlement(reason=...)`，实现 admission-before-evict；
- `PatchouliSystem.stop()`：确保 queue/bridge 的必要清理顺序与聚合失败传播一致。

明确不在范围内：

- 修改 controller `wait_all()`、`wait_many()` 或 task snapshot；
- 把 Topic ID 加入 IdentityScope、queue lane 或 cache key；
- 让 WorkspaceAssetStore 查询 Topic、binding 或 controller；
- 跨进程 Topic/asset 恢复；
- 将失败降级为正常 generation skip；
- 为 shutdown settlement 新增 HTTP API。

## 测试计划

所有测试遵循 `docs/governance/testing/test-design-standards.md`，以 Topic 状态、任务终态、
RuntimeEvent 和 System shutdown 顺序为可观察结果，不以 mock 调用次数替代行为验证。

### Unit

- 首个 Topic prepare 失败时，后续 Topic 仍被处理，并记录稳定的 failure stage；
- admission 失败不进入 `generation_skipped_topic_ids`；
- per-topic failure 存在时，RuntimeEvent 显示 partial failure 计数并最终抛出聚合异常；
- reentrant 调用不会把先前 FAILED 状态报告为成功；
- `asyncio.CancelledError` 原样传播，不进入 per-topic failure 聚合。

### Integration

使用真实 `ShortTermMemoryStore + SemanticFlowPerceptionLayer + TriggerManager +
PerceptionFamiliar + PatchouliBus`，只替换 generation admission 外部边界：

- 三个 Topic 中第二个 admission 失败时，第一个和第三个仍建立任务；
- admission 失败 Topic 的 blocks/state summary 按裁定保留，不能被误清空；
- 已接纳任务全部经过真实 controller drain 后，shutdown 才抛出聚合失败；
- generation skip、generation failure 和 execution timeout 三类观测互不混淆。

这些测试不需要 `real_infra`、`live_llm` 或 `slow` 标记。

### System ordering

- Patchouli 聚合失败时，System 不报告 `STOPPED`，也不调用
  `WorkspaceAssetStore.close_and_clear()`；
- 已成功接纳的 generation consumer 完成前，WorkspaceAssetStore 保持可读；
- 正常 skip 不阻止 System 完成 shutdown 和最后的 Store 清理。

## 完成条件

- 一个 Topic 的普通异常不会阻止后续 Topic 尝试 settlement；
- 已接纳 generation task 总会进入既有 shutdown drain；
- generation skip 与 per-topic failure 使用不同字段、日志和 RuntimeEvent 计数；
- prepare/admission 失败不在无诊断的情况下提前丢失 Topic 内容；
- 全部可继续工作完成后，存在任一 per-topic failure 时 shutdown 仍向上抛出聚合失败；
- FAILED reentrant 不会返回成功 no-op，也不会重复 admission 已成功的 Topic；
- System 不会在 Patchouli 聚合失败后关闭 WorkspaceAssetStore；
- controller `wait_all()` 与 Workspace/identity 分区保持不变；
- unit、integration 与 System ordering 测试通过，且无新增真实基础设施依赖。
