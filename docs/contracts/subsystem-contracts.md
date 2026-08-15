---
title: Subsystem Contracts
status: current
owner: system
scope: subsystem-public-contracts
code_paths:
  - src/hivememory/system/contracts/
  - src/hivememory/gateway/contracts/
  - src/hivememory/patchouli/contracts/
  - src/hivememory/alice/contracts/
  - src/hivememory/core/protocol/
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
last_reviewed: 2026-07-28
---

# 子系统公共契约

本文定义 System、Gateway、Patchouli、Alice 跨边界可观察的输入、输出和不变量。路由字符串与事件名的完整清单见[routes-and-events.md](./routes-and-events.md)。

契约不是把公开函数逐一抄进文档，也不是要求每个子系统共享同一套内部对象。它描述的是一次能力交接：调用方必须提供哪些事实，所有者承诺返回什么，以及双方都不能偷偷改变哪些语义。只要这些交接保持稳定，Gateway 的分析流程、Patchouli 的记忆实现和 Alice 的执行循环就可以各自演进；一旦内部 workflow state 或引擎实体越过边界，局部重构便会重新变成全系统改造。

因此，公共模型倾向于使用 frozen、Pydantic 或依赖中立的 dataclass。不可变并不只是编码偏好，它要求上游先形成完整决定，再交给下游只读消费；依赖中立则阻止某个领域对象沿模型引用把存储、Runtime 或 Controller 一并泄漏出去。本文既记录字段和终态，也记录这些形态背后的所有权理由。

## 1. 生命周期契约

标准子系统实现 `SubsystemProtocol`：

```python
name: str
async start() -> None
async stop() -> None
async health() -> dict[str, Any]
```

当前 Gateway、Patchouli、Alice 均满足此契约。`start()` 挂载 local/public route，`stop()` 撤销 route 并释放自己拥有的运行时资源；重复启停的具体幂等能力由宿主实现保证，调用方不应绕过 `HiveMemorySystem` 随意改变单个子系统状态。

统一生命周期的目的，是让路由是否可用与资源是否已准备好保持同一顺序。如果调用方分别启动 Runtime、注册 route 或释放存储资源，就可能出现“路由仍在但所有者已停止”或“依赖尚未就绪却已接受请求”的半启动状态。因此启停属于 System 的组合职责，领域对象不自行组织全局生命周期。

## 2. Gateway 契约

### 2.1 Process

```python
process(
    message: str,
    *,
    identity: Identity,
    ingress_mode: GatewayIngressMode,
    request_timeout_ms: int | None = None,
) -> GatewayProcessResult
```

`GatewayProcessResult` 是不可变判别联合：

- `GatewayCommandOutcome(kind="command")`：包含 `CommandExecutionResult`；
- `GatewayDecisionOutcome(kind="decision")`：包含 `GatewayDecision`。

二者互斥。命令终态不能同时携带普通分析结果；普通决策不能携带命令执行结果。

这种互斥使命令成为真正的短路终态。系统指令已经完成、被拒绝或要求确认时，继续执行检索、Agent run 和记忆提交既浪费资源，也可能把一条控制消息误当成普通对话沉淀。判别联合让调用方必须显式选择一条链路，不能依赖多个可空字段猜测 Gateway 的意图。

### 2.2 GatewayDecision

| 字段 | 语义 |
|:---|:---|
| `target_topic_id` | 已有 topic id 或 `NEW_TOPIC` |
| `new_topic_title` / `new_topic_summary` | 新话题的可选初始元数据 |
| `rewritten_query` | 下游检索使用的完整查询 |
| `search_keywords` | 稀疏检索关键词元组 |
| `memory_write_signal` | `WRITE`、`SKIP` 或 `UNKNOWN` |
| `retrieval_plan` | 检索模式、`top_k` 和 dense/sparse 权重 |
| `intent_type` | `RAG`、`WRITE`、`CHAT`、`COMPOSITE` 或 `UNKNOWN` |

`worth_saving` 是从 `memory_write_signal` 派生的只读值，不是第二份状态。

`GatewayDecision` 只保留下游可以稳定依赖的决策结果，而不携带 step、snapshot 或分析器内部对象。它是 Gateway 与执行链之间的交接单，不是远程操纵 Gateway workflow 的句柄。

### 2.3 模式不变量

- `ACTIVE_CHAT` 可以识别并执行系统指令；
- `PASSIVE_MEMORY` 必须返回普通决策，绝不能返回 command outcome；
- `request_timeout_ms` 只能收紧配置的默认总超时，不能扩大它；
- 局部可恢复失败可以降级，但最终结果仍必须满足完整终态不变量。

## 3. Patchouli 契约

Patchouli 的公开面分为 chat 协作、记忆、任务、Agent Profile、话题与就绪状态。调用方不能直接调用 Patchouli local route。

这些能力虽然服务于不同用例，却共享一个核心约束：Patchouli 对长期状态和身份可见性拥有最终解释权。公开契约允许外部请求“创建、检索或提交”，但不会把 MemoryLibrary、Familiar 或生成 Controller 交给调用方直接操作。

### 3.1 PrepareAgentRun

```python
prepare_agent_run(
    user_message: str,
    user_id: str,
    *,
    gateway_decision: GatewayDecision,
    agent_id: str = "omni_doll",
    session_id: str | None = None,
    enable_memory_retrieval: bool = True,
    generation_options: dict[str, Any] | None = None,
) -> PreparedAgentRun
```

`PreparedAgentRun` 是不可变 dataclass，包含：

- `agent_run_context`：Alice 的完整中立输入；
- `gateway_decision`：本轮只读决策快照；
- `stream_prelude`：topic、是否新话题、话题池与 memory refs；
- `generation_options`：本轮生成覆盖参数。

`AgentRunContext` 至少包含 `Identity`、真实 topic id、用户消息、话题上下文、原始 `RetrievalResponse`、MemoryCompiler 编译文本、Agent Profile 和存储可用性。

prepare 的意义不只是拼装参数。它把 Gateway 的入口决定解析成 Alice 可以直接执行的本轮记忆视图，并由 Patchouli 在交出控制权前确认真实话题、可见性和 Profile。Alice 因而无需理解 Patchouli 内部存储，也不会在执行途中重新推导另一套记忆上下文。

### 3.2 FinalizeAgentRun

```python
finalize_agent_run(
    prepared_run: PreparedAgentRun,
    loop_result: AgentRunResult,
) -> list[MemoryGenerationTask]
```

Finalize：

1. 从 `turn_events` 归约 action 和 MTP trace；
2. 构造 `InteractionPayload`；
3. 将交互提交到目标话题；
4. 为 WRITE/UPDATE 形成的 materialize task 启动主动记忆生成；
5. 记录预检索命中。

System 只应对 `AgentRunStatus.COMPLETED` 的结果调用 finalize。Finalize 已成功后不能再 cleanup。

finalize 是执行事务与记忆事务的分界。Alice 负责声明“本轮发生了什么”，Patchouli 负责判断这些执行事实如何形成 Interaction、引用和延迟物化任务。把归约与提交留在 Patchouli，可以避免 System 或 Alice 各自维护第二套感知规则，也确保取消和失败的半完成 run 不会默认进入长期知识。

### 3.3 CleanupPreparedAgentRun

```python
cleanup_prepared_agent_run(prepared_run: PreparedAgentRun) -> bool
```

Cleanup 只尝试删除 prepare 阶段新建但仍为空的话题。已有话题或已经产生内容的话题不应被删除。调用方把 cleanup 当作失败补偿，不把返回 `False` 视为新的业务错误。

它不是 rollback，也不承诺撤销整个 prepare 之后发生的一切。跨子系统没有一项可以原子回滚的数据库事务；cleanup 只补偿明确由 prepare 创建、且仍可安全判断为空的临时副作用。将它描述为回滚会诱使调用方删除已经存在或已被其他流程使用的长期状态。

### 3.4 其他公开能力

| 能力组 | 当前公开行为 |
|:---|:---|
| Interaction | 提交 `InteractionPayload` 到指定或新话题 |
| Memory | create/list/get/update/delete、feedback、retrieve、retrieve_by_aliases |
| Memory Task | list/get/cancel |
| Agent Profile | create/list/get |
| Topic | list active、读取可见 topic data、manual settle、evict |
| Citation | 记录 MTP READ/RUN 等来源的记忆引用 |
| Readiness | 模型 warmup 与 ready 查询 |

Memory 与 Topic 的身份可见性由 Patchouli 执行，调用方不能仅凭拿到 id 就假设目标可见。

## 4. Alice 契约

### 4.1 非流式运行

```python
run_agent(
    agent_run_context: AgentRunContext,
    generation_options: dict[str, Any] | None = None,
) -> AgentRunResult
```

`AgentRunResult` 包含：

- `status`：`completed`、`cancelled` 或 `failed`；
- `final_text`：最终用户可见文本；
- `mtp_iterations` / `total_iterations`：执行统计；
- `turn_events`：结构化运行事实；
- `materialize_tasks`：本 run 产生的不可变物化请求；
- `model_used`：注册表解析出的展示名，空字符串表示未解析。

这个结果是 Alice 对一次执行的完整事实声明，而不是已经提交的长期记忆。Chat application 通过拥有的 task 控制用户 stop，Alice 只沿 await 传播原生 `asyncio.CancelledError`；System 据此决定是否进入 finalize，Patchouli 再归约其中的 turn events 和 materialize tasks；任何一方都不能仅凭流中的部分文本推断 run 已经完成。

### 4.2 流式运行

`run_agent_stream()` 接收相同输入，经全局 RPC 返回 async generator。流中包含增量事件，消费者关闭时由 Alice 取消并 join 自己创建的 runner；最终必须给 System 提供完整 `AgentRunResult`，只有拿到正常完成的最终结果才能进入 Patchouli finalize。

### 4.3 Alice 不变量

- Alice 不修改 `AgentRunContext` 所指向的长期记忆或话题；
- WRITE/UPDATE 只产生 PendingAtom 和 materialize task；
- 取消或失败结果不默认进入 Patchouli finalize；
- MTP 权限由 Agent Profile 的 `allowed_mtp_verbs` 与 `allowed_sys_tools` 控制；
- CALL 仅允许根 frame 发起，子 frame 不能继续递归 CALL。

## 5. 顶层主动链路契约

```text
Gateway command outcome
  -> System 返回命令结果
  -> 不调用 Patchouli prepare / Alice / Patchouli finalize

Gateway decision outcome
  -> Patchouli prepare
  -> Alice run
  -> completed: Patchouli finalize
  -> cancelled/failed/exception: Patchouli cleanup (若已 prepare)
```

该顺序由 `ChatApplicationService` 拥有。任何 transport adapter 都不能复制或调整此顺序。

顺序本身就是契约的一部分：Gateway 先收敛入口语义，Patchouli 再准备长期知识的本轮视图，Alice 只执行，最后由 Patchouli 提交。让 transport adapter 复制这条链路，会很快产生“HTTP 可以、其他入口不可以”或两条 finalize 规则不一致的问题。

## 6. 被动链路契约

Passive Ingress 由 System 拥有并调用 Gateway `PASSIVE_MEMORY`。它可以读取 Patchouli 记忆上下文并最终提交 interaction，但不调用 Alice、不执行命令、不运行 MTP，也不生成 assistant reply。

对外 `PassiveIngressOutcome` 只表达 accepted/buffered/duplicate/degraded 等业务结果；Gateway execution state、fallback 原因和 RuntimeEvent 不进入 API 响应。

同一 `PassiveConversationKey` 在单进程内按服务接收顺序串行处理，串行范围包含 Gateway/retrieval、accumulator 修改与 submission queue admission；不同会话仍可并发。admission 成功前 accumulator 不会清空，也不会被下一 user 覆盖。connector 负责按会话因果顺序投递，`sequence` 当前只用于关联和观测，不承诺对已经乱序到达的事件进行重排。该契约不扩展为跨进程排序或持久化 mailbox。

这条限制保护的是入口语义。Passive Memory 用于摄入已经发生的外部经历，并不等价于伪造一次用户与 Agent 的对话；如果它允许命令或 Alice 执行，外部内容便可能意外触发控制行为、工具调用和回复生成，也会让“谁发起了这次行动”失去可靠答案。

## 7. Interaction 与 Topic 时序契约

Active 与 Passive 的消息来源和入口流程不同，但二者最终都向 Topic 追加 Interaction，因此共享同一组时序职责：

1. **Interaction 内全序由生产者冻结。** `TurnEvent.sequence` 只在所属 interaction 内有效；payload 一旦进入 submission queue，retry、dedup、cleanup 和 handler 都不得改写既有事件顺序或生成新的语义身份。
2. **Topic append 顺序由 Patchouli 拥有。** 当前以成功 apply 的实际 append 顺序作为 topic-local 权威投影。若未来增加 `topic_position`，必须由 topic owner 在持久化提交时原子分配，调用方不能根据时间戳自行计算。
3. **Queue FIFO 只是执行约束。** ordering key 只串行化已经入队的 work，不代表源事件发生时间，也不表达 Agent 因果关系。idempotency journal 只防止重复副作用，不参与排序。
4. **Passive source sequence 不负责事后重排。** connector 应按会话因果顺序投递；源 `sequence` 当前用于关联和观测，不承诺缓存、等待或重排晚到消息。
5. **Active prepare 读取的是一个 topic snapshot。** finalize/apply 的先后位置只能证明提交顺序，不能证明某个 run 在生成 LLM input 时看见过中间提交。多个 run 可以基于同一 revision 并发执行。
6. **多 Agent 并发优先表达因果偏序。** 未来应保存 `base_topic_revision`、causal parent 或等价的 run/frame 关系，再由 prompt/display 层按需要确定性线性化；不得仅按完成时间伪造语义先后。

`occurred_at`、`received_at`、`enqueued_at` 与 `applied_at` 是不同阶段的观测时间，均不能替代 topic 内的权威顺序。当前契约要求的是 topic-local authoritative log 与可扩展的 causal relation，而不是全系统绝对时间线；本阶段不要求全局序列、向量时钟或完整事件溯源。

## 8. 契约矛盾检查

新增或修改公共能力时，应先回答以下问题：

1. 这个模型是否只包含一次交接需要的稳定事实，还是暴露了可变 workflow state、引擎对象或回调？
2. 接收方是否正在修改本应只读的决定，或重新推导一份与所有者可能分叉的状态？
3. command outcome 是否仍能立即短路？Passive Memory 是否可能通过新分支触发命令、Alice、MTP 或回复生成？
4. prepare、run、finalize 的顺序或资格是否被 transport、事件订阅者或兼容 fallback 悄悄改变？
5. cleanup 是否仍是对空话题的有限补偿，还是被当成可以撤销长期状态的事务回滚？
6. `AgentRunResult`、PendingAtom ACK 或流式片段是否被误认为 finalize 已成功？
7. 身份、可见性和权限检查是否仍由状态所有者执行，而不是由拿到 id 的调用方自行假设？
8. 是否把 enqueue/apply timestamp 或 queue FIFO 误当作业务发生顺序？
9. 是否把 topic append 顺序误当作 Agent 已观察到彼此结果的因果关系？
10. 是否声称 finalize ordering 已经解决 prepare/LLM input snapshot 的并发？

这些问题能帮助评审者从契约语义发现设计分叉，而不只是检查函数签名是否还能调用。

## 9. 兼容与变更

以下变化属于跨子系统破坏性变更，必须同步修改 route 常量、公共模型、调用方、契约测试和本文：

- 修改 route 字符串或 handler 参数；
- 修改判别联合的 `kind`；
- 新增必填公共字段或改变字段语义；
- 改变 prepare/finalize/cleanup 顺序；
- 允许 Passive Memory 返回命令；
- 改变 AgentRunResult 终态和 finalize 资格；
- 将 local route 或内部 workflow state 暴露为公共 API。

验证入口：`tests/unit/system/contracts/`、`tests/unit/system/application/`、`tests/unit/gateway/test_phase3b_contracts.py`、`tests/unit/patchouli/test_phase3f_gateway_decision.py`、`tests/unit/alice/test_service.py`。
