---
title: Passive Conversation Ingress
status: current
owner: system
scope: external-conversation-memory-ingress
code_paths:
  - src/hivememory/system/application/passive_ingress_service.py
  - src/hivememory/system/services/passive/
  - src/hivememory/system/config/passive.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
last_reviewed: 2026-08-06
---

# 被动对话摄入

Passive Ingress 是外部对话的记忆中间件。它接收已经在其他 harness、Bot 或工具链中发生的 user、assistant、tool call 和 tool result 事件，为 user 事件准备可选的记忆上下文，并在 turn 封口后把完整交互提交给 Patchouli。

它不是第二套 active chat，也不是把外部输入伪装成 HiveMemory 用户来触发命令、Alice 或 MTP。这个边界保留了外部系统的控制权：HiveMemory 可以吸收经历和提供记忆，但不替外部 harness 生成回复或执行其中声明的工具。

## 1. 外部事件与会话身份

`PassiveIngressEvent` 当前包含：

- `source` 与 `external_conversation_id`：构成外部会话命名空间；
- `external_event_id`：与 `source` 共同构成进程内幂等键；
- 可选 `turn_id`、`sequence`、`occurred_at`；其中 `sequence` 当前用于关联与观测，不触发缺口等待或乱序重排；
- `role`：`user`、`assistant`、`tool_call`、`tool_result`；
- `content`、tool metadata、`is_final`。

内部 `PassiveConversationKey` 还叠加 HiveMemory 的 `user_id`、`agent_id` 和可选 `team_id`。不能只用 user/agent 分桶，否则不同 connector 的同名会话会互相污染。

`is_final` 表示当前 turn 是否结束，与 role 无关。外部 assistant 可能分段输出，tool result 也可能是一个 turn 的最后事件，因此不能用“收到 assistant 就提交”替代显式 final、下一条 user、idle timeout、手动 flush 或 shutdown drain。

当前 dedup 是有界的进程内 TTL registry：默认窗口为 300 秒，最大 4096 条。重复事件在窗口内不会重复追加 buffer、重复 retrieval 或重复提交 interaction；这不是跨进程 exactly-once 保证。

同一 `PassiveConversationKey` 的事件路由与显式 flush 由进程内 keyed async lock 串行化。串行范围覆盖 dedup、Gateway/retrieval、turn buffer 修改、seal 和本次 drain，因此先进入服务并开始处理的 user 不会在等待 Gateway 时被同会话 assistant/tool 事件越过。不同会话使用不同锁，仍可并发处理；最后一个持有者或等待者退出后会移除对应 lock entry。

该保证只适用于单 event loop、单进程内的接收顺序。connector 必须按同一外部会话的因果顺序投递事件；如果 `sequence=3` 先于 `sequence=2` 到达，当前实现不会等待或重排。跨进程排序、缺口恢复和持久化事件 mailbox 不属于 v0.6.0 契约。

## 2. 事件路由与记忆上下文

当前路由时序如下：

```text
user event
  -> 先 seal 并尝试提交上一 turn
  -> Gateway.process(PASSIVE_MEMORY)
  -> Patchouli memory.retrieve（按 GatewayDecision）
  -> 初始化当前 turn buffer
  -> 返回 accepted + 可选 memory context

assistant / tool_call / tool_result
  -> 只追加到当前 turn buffer
  -> 返回 buffered
  -> 不调用 Gateway，不执行工具
```

新 user 到达时先处理上一 turn，是为了避免新请求的 Gateway 或 retrieval 失败连带阻塞已经完成的旧交互。旧 turn 提交失败会留在 outbox，但不会占用或覆盖当前 accumulator。

Gateway 与 retrieval 的可恢复失败由 `MemoryContextProvider` 收敛为内部 `MemoryContextAttempt(degraded=True)`：

- Gateway 失败：当前 user 仍进入 buffer，但没有 topic/检索上下文；
- retrieval 失败：保留已获得的 Gateway decision，只缺少 memory context；
- 解析错误、route 缺失、类型违约等契约问题不降级，直接向上抛出。

当前 `PassiveIngressService.ingest_event()` 对 user 事件仍返回 `status="accepted"`；是否发生 degraded 只出现在内部结果和 RuntimeEvent 中，不伪造一个新的对外状态。非 user 事件返回 `buffered`，重复事件返回 `duplicate`，无法路由的输入返回 `ignored`。

## 3. Turn buffer 与 sealed outbox

Passive runtime 明确分离两种状态：

| 状态 | 性质 | 所有者 |
|:---|:---|:---|
| 当前 turn accumulator | 可变，接收事件并维护当前交互 | `MessageTurnBufferManager` |
| sealed-turn outbox | 不可变，等待 Patchouli 提交 | `SealedTurnOutbox` / `SealedTurnSubmitter` |

封口原因目前包括 `next_user`、`explicit_final`、`idle_timeout`、`manual_flush` 和 `shutdown_drain`。封口后先生成不可变 `SealedTurn` 并进入 outbox，只有 `patchouli.public.submit_interaction` 成功才从队列移除。

每个外部会话的 drain 使用一把 `asyncio.Lock` 保持提交顺序。一次提交失败会把当前项和剩余项按原顺序放回队首，并停止该会话本轮 drain；其他会话和当前新 turn 不被阻塞。outbox 有界，超出上限时按当前实现淘汰最旧项，淘汰会记录 warning。这是当前进程内可靠性边界，不应被描述为持久化队列。

完整 turn 会保留为 InteractionPayload/raw interaction 事实；Gateway 的 `memory_write_signal` 只是写入预判，不能用来删除原始外部经历或直接决定正式记忆是否生成。

## 4. 维护与关闭

`PassiveIngressService.start()` 在全局 scheduler 中注册：

```text
owner = system.passive_ingress
name  = observer_idle_flush
callback = PassiveMessageIngressor.scan_idle_conversations_once
```

扫描会按配置的 idle timeout seal 超时 buffer，再 drain 全部 outbox。idle flush 会逐会话取得与事件路由相同的串行门，并在门内重新检查 idle 时间，避免扫描快照与刚到达的事件竞争。该维护任务由 System 的统一 `GlobalMaintenanceScheduler` 驱动，Passive Ingress 不自建线程或 event loop。

关闭时，System 先停 scheduler，再调用 `shutdown_drain()`：停止注册、封口所有活动 buffer、尽力 drain，并返回 `sealed_turns`、`submitted_turns` 和 `outbox_pending` 摘要。outbox 仍有 pending 不等价于 shutdown 失败已经被补偿；它表示当前进程在关闭边界仍有未提交事实。

## 5. 公开响应与观测

公共响应只包含外部调用方需要的接收状态、事件 ID 和可选编译后的 memory context：

```json
{
  "status": "accepted",
  "external_event_id": "evt-123",
  "memory": "<memory_context>...</memory_context>"
}
```

不返回 `GatewayExecutionState`、fallback 原因、完整 trace、command response、assistant response 或 tool 执行结果。重复、降级、seal、提交失败和重试通过 `RuntimeEventSink` 观测，避免把内部控制信息固化成外部 API。

## 6. 设计不变量与矛盾检查

- `PASSIVE_MEMORY` 必须只获得 Gateway decision，不得产生 command outcome；
- assistant/tool event 不调用 Gateway、Alice、MTP 或工具 runtime；
- 上一 turn 提交失败不能阻塞下一 turn accumulator；
- sealed turn 在 Patchouli 成功前不能从 outbox 删除；
- `source + external_event_id` 的重复事件不能再次检索或提交；
- 同一 `PassiveConversationKey` 的事件和 flush 不能并发修改 accumulator；不同会话不能被一把全局锁串行化；
- connector 必须按会话因果顺序投递；可选 `sequence` 不代表系统承诺乱序重排；
- outbox 的内存有界限制必须显式记录，不能宣称跨进程 exactly-once；
- `memory_write_signal=SKIP` 不得删除 raw interaction 或 provenance；
- 观测 sink 失败不能改变 accepted/buffered/duplicate 等业务响应。

评审新 connector 或新 flush 入口时，重点检查它是否创建了第二个会话键、绕过 outbox 直接提交、把 `is_final` 错当成 assistant role，或为了“简化”而把 Passive Ingress 改造成 active chat。

Passive turn 提交后与主动交互共享 Patchouli 的短期 topic working set 和 Page Folding。外部 Agent harness 即使拥有自己的对话 compact，也不会缩减 Patchouli 已经摄入的 blocks，因此 Passive Ingress 不按入口模式跳过内部 folding。当前公共响应只返回 retrieval memory，不返回 `state_summary + recent_blocks`；所以 folding 能约束内部话题路由与后续记忆生成输入，但不能替外部 bot 管理 prompt history。若未来需要 HiveMemory 承担外部上下文压缩，必须建立显式的上下文所有权和覆盖范围契约，不能根据 connector 名称推断。相关开放项见 [Page Folding 跨入口后续技术债](../todo/page-folding-cross-ingress-follow-ups.md)。

## 7. 配置与验证入口

配置归属 `HiveMemoryConfig.passive_ingress` 与 `HiveMemoryConfig.scheduler.tasks`：

- `dedup_ttl_seconds`、`max_dedup_entries`；
- `max_buffered_events_per_turn`、`max_outbox_items_per_conversation`；
- observer idle flush 的 interval、timeout 和启用开关。

验证入口：

- `tests/unit/system/services/passive/test_passive_gateway_mode.py`
- `tests/unit/system/services/passive/test_passive_degradation.py`
- `tests/unit/system/services/passive/test_passive_ordering_and_outbox.py`
- `tests/unit/system/services/passive/test_passive_service_degraded_response.py`
- `tests/unit/system/services/passive/test_passive_runtime_events.py`
