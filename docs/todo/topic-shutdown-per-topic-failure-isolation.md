---
title: Topic Shutdown Per-Topic Failure Isolation
status: todo
owner: patchouli-system
scope: topic-shutdown-settlement-failure-isolation-and-drain-continuation
related_docs:
  - docs/patchouli/perception.md
  - docs/system/runtime-and-bus.md
  - docs/archive/plans/v0.6.2-workspace-mvp.md
  - docs/governance/testing/test-design-standards.md
last_reviewed: 2026-09-01
---

# Topic shutdown 逐 Topic 失败隔离

## 当前裁定

本事项只处理 automatic SHUTDOWN settlement 的逐 Topic 失败隔离与后续 drain，
不改变正常 skip 语义，也不为 queue admission 之外再建立一套可靠性机制。

automatic settlement 统一采用以下顺序：

```text
冻结 settlement payload（blocks / state summary / TopicAssetBinding refs）
  -> 驱逐 Topic buffer
  -> 尝试 generation queue admission
```

queue 成功接纳任务后，重试、持久化与幂等性由既有任务队列负责。若 admission 本身失败：

- 不恢复已经驱逐的 Topic；
- 不在 queue 外重放冻结 payload；
- 不新增外层 retry、outbox 或 journal；
- 不把失败降级成正常 generation skip；
- 继续尝试后续 Topic，并最终 drain 已经成功接纳的任务。

这是自动生命周期机制的有意取舍：automatic IDLE/LRU/SHUTDOWN 都不能因为一个待清理
Topic 的 admission 失败而留下阻塞后续处理的驻留数据。manual settle 保持独立的用户可重试
语义，继续使用 `prepare -> admission -> evict`，admission 失败时保留 Topic 并向用户抛错。

本 TODO 不授权修改 memory generation controller 的 `wait_all()` 通用语义，也不阻塞
Workspace P5 的 TopicAssetBinding 实现。

## 当前链路与缺口

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
`perception_layer.settle_topic(..., SHUTDOWN)`。SHUTDOWN 决策矩阵执行
`settle=True / compact=False / evict=True`；有结算材料时，再通过 local bus 尝试
generation admission。

当前一个普通 admission 异常会直接跳出循环，造成：

- 后续 Topic 不再结算或驱逐；
- 此前已经接纳的 generation task 无法进入本次 `wait_all()`；
- queue stop、bus unmount 与 Store teardown 被连带中断；
- `_shutdown_drain_started` 已置为 `True`，同一 Runtime 的再次调用可能被误报为成功 no-op；
- Patchouli 内部形成难以解释的半停止状态。

最小修复的重点不是挽救 admission 失败的单个自动 Topic，而是把异常隔离在该 Topic，
让其余生命周期清理和已接纳任务 drain 得以继续。

## 必须区分的四类结果

### 1. 正常 generation skip

Topic 已按自动生命周期完成 settlement/eviction，但没有建立 generation task，例如没有可生成
材料，或 generation coordinator 正常返回无任务。该 Topic 可进入
`generation_skipped_topic_ids`。这是正常结果，不降低 shutdown success。

### 2. 已接纳 generation task

Topic 已被驱逐，generation task 已由 queue 接纳。该任务进入既有 controller drain，
后续成功、失败、超时或取消由任务执行结果表达。

### 3. Generation admission failure

Topic 的 settlement payload 已冻结、buffer 已驱逐，但向 queue 提交时抛出普通异常。
它必须使用独立的 admission failure 事实表达，不能写入
`generation_skipped_topic_ids`，也不能据此恢复 Topic 或再次提交 payload。

最小诊断事实可以包含：

```text
topic_id
stage: admission
error_type
error_message
```

错误详情只用于进程内聚合与 RuntimeEvent 诊断，不进入 Topic HTTP 响应，也不得保存
traceback、asset ref、WorkspaceAsset 内容或其他敏感载荷。

### 4. 已接纳任务的 execution failure

queue 已接纳任务后发生的执行失败、timeout 或 cancel，属于既有 generation task 终态。
它与 admission failure 分开统计和投影，不得借逐 Topic admission 隔离修改 controller
任务状态机或 `wait_all()` 返回模型。

## Settlement payload 与资产引用

`TopicAssetBinding` 是 Topic 内真实使用过资产的权威事实。在 automatic SHUTDOWN 驱逐
buffer 前，必须把该 Topic 的全部 binding refs 连同 blocks/state summary 冻结进
settlement task payload。任务至少携带：

```text
asset_id
asset_ref
```

冻结完成后，buffer 与 binding 可以一起移除；task 拥有独立生命周期。generation consumer
在创建 artifact 时通过 `asset_ref` 反查 WorkspaceAssetStore，并在读取期间持有 lease。

本事项不提供跨进程恢复：若 admission 未成功，冻结 payload 不再由外层持久化；若进程退出，
未被 queue 接纳的自动 settlement 与进程内 asset/ref 一起消失。RuntimeEvent、日志和聚合错误
只投影失败计数与稳定阶段，不得泄漏 asset ref 或资产内容。

## 目标编排

`flush_all_for_shutdown()` 应按 Topic 快照执行：

1. 尝试冻结该 Topic 的 settlement payload 与 binding refs；
2. 按 automatic settlement 语义驱逐该 Topic；
3. 若存在 generation payload，尝试 queue admission；
4. 普通 admission 异常记录为该 Topic 的独立失败，然后继续下一个 Topic；
5. `asyncio.CancelledError` 原样传播，不纳入普通失败聚合；
6. 全部 Topic 尝试完成后，调用既有 controller `wait_all()`；
7. 对已接纳任务沿用既有 timeout/cancel 策略；
8. 继续 queue/consumer、bus、Gateway 与 WorkspaceAssetStore 的最终 teardown；
9. 依据最终 lifecycle 投影报告完整成功或带错误完成。

一个 Topic 的 admission failure 不能阻止后续 Topic 尝试，也不能跳过已经接纳任务的 drain。
同样，它不应阻止最终 teardown，使 System 长期停留在半停止状态。

## 最终失败投影待裁定

仍需在实现前冻结最终 lifecycle 对外投影，二选一：

- teardown 全部完成后进入显式 `completed_with_errors`；
- teardown 全部完成后抛出聚合异常，并由顶层记录“清理完成但部分 admission 失败”。

无论选择哪一种，都必须满足：

- admission failure 不伪装成 skip 或完整成功；
- 已接纳任务先完成 drain；
- queue、consumer、bus 与 Store 最终完成 teardown；
- reentrant 调用不能把先前带错误完成误报为首次成功；
- 不恢复失败 Topic，不重复 admission，也不重放冻结 payload。

## 建议的最小实现边界

生产代码预计只涉及：

- `patchouli/services/perception.py`：逐 Topic 捕获普通 admission 异常、记录失败并继续；
- `patchouli/runtime/models.py`：增加最小 admission failure 事实或计数；
- `patchouli/runtime/core.py`：确保 per-topic failure 后仍执行 generation drain 与最终 teardown；
- `patchouli/runtime/shutdown_drain.py`：RuntimeEvent 只投影计数和稳定阶段，不泄漏载荷；
- 必要的顶层 shutdown lifecycle 投影与 reentrant 结果修正。

明确不在范围内：

- 将 automatic SHUTDOWN 改成 admission-before-evict；
- admission 失败后保留或恢复 Topic；
- 自动重试、重复 admission 或 reentrant payload replay；
- 新增 queue 外 outbox、journal 或持久化 payload；
- 修改 controller `wait_all()`、`wait_many()` 或 task snapshot 通用语义；
- 把 Topic ID 加入 IdentityScope、queue lane 或 cache key；
- 让 WorkspaceAssetStore 查询 Topic、binding 或 controller；
- 跨进程 Topic/asset/ref 恢复；
- 将失败降级为正常 generation skip；
- 为 shutdown settlement 新增 HTTP API。

## 测试计划

所有测试遵循 `docs/governance/testing/test-design-standards.md`，以 Topic 状态、任务终态、
RuntimeEvent 和 System shutdown 顺序为可观察结果，不以 mock 调用次数替代行为验证。

### Unit

- 三个 Topic 中第二个 admission 失败时，第三个仍被处理；
- 全部三个 Topic 最终都已离开 Topic pool；
- admission 失败不进入 `generation_skipped_topic_ids`；
- 已接纳任务仍进入既有 controller drain；
- settlement task 在 buffer pop 前已经冻结全部 TopicAssetBinding refs；
- failure 记录与 RuntimeEvent 不包含 asset ref、内容或 traceback；
- reentrant 调用不会把先前带错误完成报告为首次成功；
- `asyncio.CancelledError` 原样传播，不进入 per-topic failure 聚合。

### Integration

使用真实 `ShortTermMemoryStore + SemanticFlowPerceptionLayer + TriggerManager +
PerceptionFamiliar + PatchouliBus`，只替换 generation admission 外部边界：

- 三个 Topic 中第二个 admission 失败时，第一个和第三个仍建立任务；
- 三个 Topic 的 buffer 与 binding 均按 automatic 语义移除，不恢复失败 Topic；
- 第一和第三个任务全部经过真实 controller drain；
- generation skip、admission failure、execution failure 与 timeout 四类观测互不混淆；
- 单个 admission failure 不跳过 queue、bus、Gateway 与 Store 的最终 teardown；
- generation consumer 通过 task 中冻结的 ref 反查 Store，并在读取期间持有 lease。

这些测试不需要 `real_infra`、`live_llm` 或 `slow` 标记。

## 完成条件

- 一个 Topic 的普通 admission 异常不会阻止后续 Topic 尝试 settlement；
- automatic admission failure 后不恢复 Topic、不重放 payload、不增加 queue 外重试；
- 所有已经成功冻结 payload 的自动 settlement Topic 均按既定生命周期离开 Topic pool，
  admission 失败不会将其恢复；
- 已接纳 generation task 总会进入既有 shutdown drain；
- generation skip、admission failure 与 execution failure 使用不同事实和观测；
- settlement task 在 buffer/binding 清理前冻结所需 asset refs；
- RuntimeEvent 与错误投影不泄漏 ref 或资产内容；
- failure 不跳过 queue、consumer、bus、Gateway 与 Store 的最终 teardown；
- reentrant 调用不把先前带错误完成误报为成功，也不重复 admission；
- controller `wait_all()` 与 Workspace/identity 分区保持不变；
- unit、integration 与 System ordering 测试通过，且无新增真实基础设施依赖。
