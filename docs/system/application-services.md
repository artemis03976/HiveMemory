---
title: System Application Services
status: current
owner: system
scope: application-use-cases-and-cross-subsystem-orchestration
code_paths:
  - src/hivememory/system/application/
  - src/hivememory/system/runtime/control.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
last_reviewed: 2026-07-28
---

# System 应用服务

System 应用服务是 transport 与子系统之间的用例层。它们回答“这次请求应该按什么顺序跨边界运行”，而不回答“记忆如何检索”“Agent 如何生成”或“Gateway 如何分析”。

这一层存在，是因为 HTTP、SSE、CLI 和未来外部 adapter 都需要共享同一条主动 chat、取消、被动摄入和管理 API 链路。若每个 router 自己拼装 Gateway、Patchouli 和 Alice，就会再次出现多套 prepare/finalize、错误处理和取消语义。

## 1. 共同边界

应用服务的共同规则是：

1. 通过 `GlobalSystemBus` 请求公开 route；
2. 在边界处构造 `Identity`、请求模型或面向 API 的结果；
3. 不持有另一个子系统的 Runtime、Service、Controller 或 local bus；
4. 不把内部 execution state、fallback 原因和观测事件直接返回给外部客户端；
5. 对取消、失败和 cleanup 保持与 Contracts 一致的终态。

应用服务可以保存一次用例的短期控制状态，例如 chat generation registry，但不能保存 Patchouli 的长期记忆状态或 Gateway 的请求级 workflow state。

## 2. 服务分工

| 服务 | 当前职责 | 主要依赖 |
|:---|:---|:---|
| `ChatApplicationService` | 主动非流式/流式 chat、command short-circuit、取消和 prepare/run/finalize 编排 | Gateway、Patchouli、Alice public routes；RuntimeEventSink |
| `PassiveIngressService` | 外部事件摄入、idle maintenance 注册、显式 flush、shutdown drain | Passive Ingressor、Gateway/Patchouli public routes、scheduler |
| `MemoryApplicationService` | Memory CRUD、feedback 和查询参数转换 | Patchouli memory routes |
| `MemoryTaskApplicationService` | 查询/取消 Patchouli 拥有的 memory generation task | Patchouli task routes |
| `AgentApplicationService` | 构造 Agent Profile atom 并调用 Patchouli profile routes | Patchouli profile routes |
| `TopicApplicationService` | 活跃话题查询、手动 settle、evict | Patchouli topic routes |
| `SystemReadinessService` | 模型 warmup、ready 和简短 readiness 状态 | Patchouli readiness routes |

这些服务的“拥有”只指顶层用例入口，不改变表中后端子系统的状态所有权。例如 `MemoryTaskApplicationService` 可以取消任务，但任务生命周期仍由 Patchouli 负责。

## 3. 主动 chat：唯一编排者

### 3.1 非流式链路

```text
ChatApplicationService.chat
  -> register ChatGenerationRun
  -> Gateway public process (ACTIVE_CHAT)
  -> command: return command outcome
  -> decision: Patchouli prepare_agent_run
  -> Alice run_agent
  -> completed: Patchouli finalize_agent_run
  -> cancelled/failed: Patchouli cleanup_prepared_agent_run
  -> close generation registry
```

Gateway 返回 command outcome 时，服务立即完成本次 run，不进入 topic、retrieval、Alice 或主动记忆生成。这是控制消息与普通对话之间的语义隔离，不是一个性能优化开关。

普通决策进入 prepare 后，如果调用方已经请求取消，则跳过 Alice 和 finalize，返回 cancelled 结果并在 finally 中请求 cleanup。Alice 只有在 `AgentRunResult.status == completed` 且 run 未取消时才允许进入 finalize；finalize 成功后才将 prepared 标记为已接管，不再 cleanup。

### 3.2 流式链路

`chat_stream()` 保持同一条阶段顺序，但把阶段事实以事件交给 transport：

```text
generation_id
  -> Gateway decision / command
  -> command_result + done
  -> 或 prepare
       topic_info
       memory_refs
       Alice stream events
       run_status(finalizing)
       done(completed + memory_task_ids + pool_topics)
```

流式生成必须收到 Alice 的最终 `done` 才能构造 `AgentRunResult`。若流在没有终态事件时结束，服务按协议错误处理；客户端提前关闭流则将 run 标记为 `cancelled`，关闭 Alice 子流，并对尚未 finalize 的 prepared run 执行 cleanup。

流式 `done`、`command_result` 和 `error` 是 transport 可消费的事件，不是新的跨子系统业务契约；它们的来源和调用顺序仍由本服务和 Contracts 共同约束。

## 4. ChatGenerationRun 与取消

`ChatGenerationRunRegistry` 是 System 应用层拥有的进程内控制表。每条 run 有：

- `generation_id`；
- `asyncio.Event` 取消信号；
- `created/preparing/streaming/finalizing/completed/cancelling/cancelled/failed` 状态；
- 首次取消原因。

`cancel_generation()` 是幂等的：不存在返回 `not_found`，已完成或已失败的 run 不再改变业务终态，仍然可以返回当前状态。取消信号通过 Gateway、Alice 和流式清理边界传递，不由 RuntimeEvent 或客户端断连事件直接替代。

当前 registry 是进程内短期控制状态，不是可恢复 Job。进程重启后不能据此恢复 run；长期 Job 生命周期属于后续 Runtime Job Queue 计划。

## 5. 管理类应用服务

### 5.1 Memory 与 Profile

`MemoryApplicationService` 将 API 字段构造成 `MemoryAtom`，再通过 Patchouli public route 执行 create/list/get/update/delete/feedback。列表查询会显式排除 `AGENT_PROFILE`，避免普通记忆管理 API 与 Profile 资产混为一类；不存在的 get/update 被翻译为 `MemoryNotFoundError`。

`AgentApplicationService` 用同样的 atom 结构创建 `AGENT_PROFILE`，Profile 的实际持久化和可见性仍由 Patchouli 负责。这里的 `agent_config` 是资产内容，不是 System 直接解释的运行时权限。

### 5.2 Task、Topic 与 Readiness

`MemoryTaskApplicationService` 只转发 task list/get/cancel；它不从 Patchouli task 对象推导第二套状态机。

`TopicApplicationService` 提供活跃话题、手动 settle 和 evict 入口；`SystemReadinessService` 提供模型 warmup、ready 查询和 `ready/warming_up` 摘要。它们都通过 route 访问所有者，不能根据 ID 或缓存自行判断可见性和生命周期。

## 6. 错误、观测与 cleanup

- 业务拒绝或资源不存在应使用服务定义的稳定结果/异常；
- Gateway cancel/timeout 和 Alice 取消不应被包装成普通 success；
- `RuntimeEventSink` 只记录 chat run、状态和失败，不改变返回值；
- cleanup 是 prepare 失败后的有限补偿，不是跨子系统 rollback；
- 应用服务捕获的错误应保留原始因果，不能通过“空列表/空话题”掩盖 route 缺失或契约违约。

## 7. 应用层矛盾检查

新增应用服务或新入口时，检查：

1. 是否复制了 `ChatApplicationService` 的 prepare/run/finalize 顺序？
2. 是否绕过 `GlobalSystemBus` 直接持有子系统 Runtime？
3. 是否把 command outcome 继续送进普通 chat？
4. 是否在取消后仍调用 finalize，或 finalize 失败后忘记 cleanup？
5. 是否把 memory task、topic 或 run registry 的临时状态写成另一个权威来源？
6. 是否把内部 RuntimeEvent、fallback 原因或 trace 当成公开 API 字段？

## 8. 验证入口

- `tests/unit/system/application/test_gateway_chat_flow.py`
- `tests/unit/system/test_cancel_hardening.py`
- `tests/unit/system/application/test_api_services.py`
- `tests/unit/system/application/test_memory_service.py`
- `tests/unit/system/application/test_memory_task_service.py`
- `tests/unit/system/application/test_agent_service.py`
- `tests/unit/system/application/test_topic_service.py`
- `tests/unit/system/application/test_readiness_service.py`
