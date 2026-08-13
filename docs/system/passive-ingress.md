---
title: Passive Conversation Ingress
status: current
owner: system
scope: external-conversation-memory-ingress
code_paths:
  - src/hivememory/system/application/passive_ingress_service.py
  - src/hivememory/system/services/passive/
  - src/hivememory/system/config/passive.py
  - src/hivememory/patchouli/control/interaction_submission.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
last_reviewed: 2026-08-08
---

# 被动对话摄入

Passive Ingress 是外部对话的记忆中间件。它接收已经在其他 harness、Bot 或工具链中发生的 user、assistant、tool call 和 tool result 事件，为 user 事件准备可选的记忆上下文，并在 turn 结束后把完整交互移交 Patchouli 的 `InteractionSubmissionQueue`。

它不是第二套 active chat，也不会触发命令、Alice 或 MTP。Active `finalize_agent_run()` 同样通过共享 `InteractionSubmissionQueue` 提交，并由自己的 applied gate 决定最终成功；两条链路只共享 submission adapter，不共享入口时序或领域成功条件。

## 1. 外部事件与会话身份

`PassiveIngressEvent` 的主要字段包括：

- `source` 与 `external_conversation_id`：构成外部会话命名空间；
- `external_event_id`：与 `source` 共同构成进程内幂等键；
- 可选 `turn_id`、`sequence`、`occurred_at`；
- `role`：`user`、`assistant`、`tool_call`、`tool_result`；
- `content`、tool metadata、`is_final`。

内部 `PassiveConversationKey` 还叠加 HiveMemory 的 `user_id`、`agent_id` 和可选 `team_id`。同一 key 的事件路由、Gateway/retrieval、accumulator 修改和 queue admission 由 keyed async lock 串行化；不同会话仍可并发。

`is_final` 表示当前 turn 结束，与 role 无关。下一条 user、idle timeout、手动 flush 和 shutdown drain 也可以结束当前 turn。`sequence` 只用于关联和观测，当前实现不等待缺口或重排乱序事件。

dedup 是有界的进程内 TTL registry，不承诺跨进程 exactly-once。若新 user 在写入 accumulator 前因为上一轮 admission 失败而被拒绝，本次新事件的 dedup 占位会撤销，connector 可以使用同一 `external_event_id` 重试。

## 2. 事件路由与记忆上下文

```text
user event
  -> 若上一 turn 尚未移交，先尝试 queue admission
  -> Gateway.process(PASSIVE_MEMORY)
  -> Patchouli memory.retrieve（按 GatewayDecision）
  -> 初始化当前 turn accumulator
  -> 返回 accepted + 可选 memory context

assistant / tool_call / tool_result
  -> 只追加到当前 turn accumulator
  -> is_final 时尝试 queue admission
  -> 不调用 Gateway，不执行工具
```

Gateway 与 retrieval 的可恢复失败由 `MemoryContextProvider` 收敛为降级结果：

- Gateway 失败：当前 user 仍进入 accumulator，但没有 topic/检索上下文；
- retrieval 失败：保留已获得的 Gateway decision，只缺少 memory context；
- route 缺失、类型违约等契约问题不降级，直接向上抛出。

公共响应仍只使用 `accepted`、`buffered`、`duplicate`、`ignored` 以及可选 memory context；降级细节只进入安全摘要化的 RuntimeEvent。

## 3. Accumulator 与 submission queue

Passive 只维护当前 turn 的可变 accumulator，不再维护 `SealedTurn`、Passive outbox 或专属 submitter。完成一轮时采用两阶段交接：

```text
mutable accumulator
  -> prepare payload snapshot（不清空）
  -> 构建 InteractionSubmission
  -> InteractionSubmissionQueue.submit()
  -> admission 成功
  -> commit/reset accumulator
```

每轮 user 开始时生成稳定的 `interaction_id`，直到 queue admission 成功才清除。`InteractionSubmission` 在队列边界经 versioned codec 编码为 canonical JSON bytes；进入 `WorkItem` 后不再持有可被调用方修改的 DTO 引用。

queue admission 失败会直接向调用方施加背压：

- accumulator、payload 和 `interaction_id` 保持不变；
- 下一条 user 不能覆盖旧 turn；
- 后续重试仍使用同一 `interaction_id`；
- 不另建 pending store 兜底，否则会重新形成第二个 outbox。

admission 成功只表示 work 已由通用队列接受，不表示 Patchouli apply 已完成。之后的 FIFO、retry、capacity、幂等 apply、dead letter 和通用 `WORK_*` 观测都由 Work Queue Runtime 负责。同一会话使用 `PassiveConversationKey.ordering_key` 保证提交顺序。

完整 turn 始终保留为 `InteractionPayload` 原始交互事实；Gateway 的 `memory_write_signal` 只是写入预判，不能用来删除外部经历。

## 4. 维护与关闭

`PassiveIngressService.start()` 在全局 scheduler 中注册 `observer_idle_flush`。扫描逐会话取得同一串行门，在门内重新检查 idle 时间，再调用统一 finalize 入口。

关闭时，System 先停 scheduler，再执行：

```text
PassiveIngressor.shutdown_drain
  -> 把全部当前 accumulator 移交 submission queue
InteractionSubmissionQueue.drain_all
  -> 等待已接受 work 进入终态
```

Ingressor 返回 `finalized_turns` 与 `accepted_submissions`；应用服务再查询 queue pending 数生成 shutdown 摘要。若 admission 本身失败，异常向上传播，未接收的 accumulator 仍保留在进程内，不会被假报为已提交。

## 5. 设计不变量

- `PASSIVE_MEMORY` 只能获得 Gateway decision，不得产生 command outcome；
- assistant/tool event 不调用 Gateway、Alice、MTP 或工具 runtime；
- queue 接收成功前不得 reset accumulator；
- admission 失败后不得开启或覆盖下一 turn；
- retry admission 必须复用同一 `interaction_id`；
- `source + external_event_id` 的已处理事件不能再次检索或追加；
- 同一 `PassiveConversationKey` 不能并发修改 accumulator，不同会话不能被一把全局锁串行化；
- `is_final` 不等价于 assistant role；
- `memory_write_signal=SKIP` 不得删除 raw interaction 或 provenance；
- RuntimeEvent 失败不能改变业务结果。

Passive turn admission 后与主动交互共享 Patchouli 的短期 topic working set 和 Page Folding。当前公共响应只返回 retrieval memory，不返回 `state_summary + recent_blocks`；相关开放项见 [Page Folding 跨入口后续技术债](../todo/page-folding-cross-ingress-follow-ups.md)。

## 6. 配置与验证入口

配置归属 `HiveMemoryConfig.passive_ingress` 与 `HiveMemoryConfig.scheduler.tasks`：

- `dedup_ttl_seconds`、`max_dedup_entries`；
- `max_buffered_events_per_turn`；
- observer idle flush 的 interval、timeout 和启用开关。

验证入口：

- `tests/unit/system/services/passive/test_passive_gateway_mode.py`
- `tests/unit/system/services/passive/test_passive_degradation.py`
- `tests/unit/system/services/passive/test_passive_ordering_and_submission.py`
- `tests/unit/system/services/passive/test_passive_submission_queue.py`
- `tests/unit/system/services/passive/test_passive_runtime_events.py`
