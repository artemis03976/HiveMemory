---
title: Patchouli Perception
status: current
owner: patchouli
scope: structured-ingestion-and-short-term-topics
code_paths:
  - src/hivememory/engines/perception/
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/patchouli/memory_library/buffer.py
  - src/hivememory/patchouli/memory_library/stores.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/system/passive-ingress.md
last_reviewed: 2026-08-19
---

# 感知与短期话题

Perception 负责把一轮已经发生的交互转化为 Patchouli 可以持续维护的短期事实。它不猜测用户下一步要做什么，也不直接决定某段文字应成为哪条正式记忆；它保存结构化 turn、维护相互隔离的话题 buffer，并在容量、空闲、驱逐、手动操作或停机边界上形成可交给 Generation 的材料。

“感知”在这里更接近记忆管理单元，而不是第二个 Gateway。Gateway 先解释入口，Alice 先完成 run，Perception 随后观察结果。这个先后关系防止记忆系统为了整理历史而重新取得入口决策或 Agent 执行权。

## 1. 模型分层

### 1.1 内容事实：TurnRecord

一轮交互的内容真相由 `core.models.TurnRecord` 表达，包括：

- identity、原始 user query 与 rewritten query；
- assistant final text；
- 有序 `TurnEvent[]`；
- 由事件归并的 `AgentAction[]`；
- 由 action 派生的 `TraceItem[]`。

当前 `TurnEvent.kind` 只接受 `user_message / assistant_message / thought / tool_call / tool_result`。工具事件通过 `action_id` 保持一次 action 的结构边界。Perception 不再接受旧 `assistant_message` fallback 字段，也不会从一段拼接文本逆向解析工具事实。

### 1.2 感知容器：LogicalBlock

`LogicalBlock` 在 TurnRecord 外增加感知层元数据：block id、created time、估算 token、gateway intent 与 `worth_saving`。它是不可变模型，表明同一轮内容事实进入短期话题后的感知视图。

可以简写为：

```text
LogicalBlock = TurnRecord + perception metadata
```

### 1.3 话题工作区：SemanticBuffer

SemanticBuffer 是短期话题的可变工作区，保存 blocks、展示 title/summary、Page Folding 产生的 `state_summary`、状态、token 总量、最近访问时间和最近模型名。ShortTermMemoryStore 持有它；外部只读取不可变 `TopicData` / `TopicSnapshot`。

## 2. 结构化摄入

`InteractionPayload` 是主动与被动入口共享的协议，当前主字段包括 identity、user message、rewritten query、assistant final text、turn events、MTP traces、materialize tasks、worth_saving 与 model_used。

```text
InteractionPayload
  -> require turn_events
  -> ActionReducer(turn_events)
  -> TurnRecord
  -> token estimate
  -> LogicalBlock
  -> ShortTermMemoryStore.add_block(topic_id)
  -> optional Page Folding
```

主动流程由 Patchouli finalize 构建 payload；被动流程由 System 的 turn buffer 构建 payload。两者进入同一 `PerceptionFamiliar.submit_interaction()` 与 `SemanticFlowPerceptionLayer.route_and_ingest()`，不存在被动专用的扁平文本主链。

结构化摄入的边界来自被动入口的真实事件形态：`MessageTurnBuffer` 接收 `user`、`assistant`、`tool_call`、`tool_result` 四类消息。只有自然语言 assistant 段落进入 `assistant_final_text`；工具调用与工具返回必须以 `TurnEvent` 保留，不能为了生成一段看似完整的 transcript 而提前丢弃。`ActionReducer`/`TraceReducer` 只能由 Perception 从这些事件派生，入口层不应另行构造第二套摘要，否则历史重放、主动生成与被动归档会拥有互相漂移的事实来源。

`target_topic` 则属于 user 到达时的 Gateway route 决策，而不是 flush 时重新计算的感知结果。Buffer 在接收 user 时保存目标话题；下一轮 user 到来时，先 flush 旧轮并完成旧轮的 payload，再初始化新轮。这样 `user2` 的 gaze 不会污染 `user1` 的归档归属。System Passive Ingress 拥有 session、idle 计时与事件提交；TheEye/Gateway 只负责入口判断，不重新拥有话题 buffer 或被动分析状态。

Perception 只把 `worth_saving` 写入 block，不在普通 ingest 时删除它。真正形成 settlement payload 时，`worth_saving is False` 的 blocks 被过滤，`None` 与 `True` 保留。这种保守语义确保 Gateway 缺失判断时不会默认丢弃材料。

## 3. 话题准备与 LRU

`prepare_topic()` 在 Agent run 之前确保目标话题存在：

- `NEW_TOPIC` 创建 UUID、title 和 summary；
- 已存在话题继续使用并刷新访问顺序；
- 指定话题不存在时记录 warning 并回退到新话题。

当需要创建新话题且活跃池达到 `max_resident_topics` 时，PerceptionFamiliar 选择 LRU topic，按 `LRU_EVICTION` 结算并提交后台 generation task，然后移除 buffer。已有话题命中不会为了“置顶”而误驱逐另一话题。

Prepare 与 ingest 都会再次确保话题存在，以覆盖 prepare 后异常清理或其他边界变化。若 prepare 已创建新话题但 Agent run 失败，Patchouli cleanup 只在它仍为空时删除，避免误删已经摄入内容的话题。

## 4. 结算矩阵

TriggerManager 把触发原因映射为三种原子动作：

- Settle：从 blocks 快照构建 `TopicMaterializeTask`；
- Compact：通过 RelayController 生成 `state_summary`；
- Evict：从短期 store 移除 buffer。

当前矩阵为：

| 原因 | Settle | Compact | Evict | 当前结果 |
|:---|:---:|:---:|:---:|:---|
| `TOKEN_OVERFLOW` | 否 | 是 | 否 | 折叠旧前缀并保留最近工作集，话题继续存活 |
| `IDLE_TIMEOUT` | 是 | 否 | 是 | 提交候选材料并移出活跃池 |
| `LRU_EVICTION` | 是 | 否 | 是 | 为新话题腾出位置 |
| `SHUTDOWN` | 是 | 否 | 是 | 停机前结算并驱逐（含真正空 Topic） |
| `MANUAL_SETTLE` | 是 | 否 | 是 | 手动结算为记忆资产并结束 Topic 生命周期 |
| `MANUAL_COMPACT` | 否 | 是 | 否 | 手动压缩工作集，不结算、不驱逐 |
| `MANUAL_DELETE` | 否 | 否 | 是 | 丢弃 Topic，不写记忆 |

`TOKEN_OVERFLOW` 是纯 Compact：TriggerManager 只把保留后缀之前的旧 blocks 交给 RelayController，再由 ShortTermMemoryStore 原子写入摘要、裁剪旧前缀并重算 token。其余触发原因维持原有 Settle/Evict 语义；发生 Settle 时旧 blocks 才会清空，需要 evict 的原因随后再删除 buffer。Settle 只是返回 payload，TriggerManager 不知道 local bus，也不直接触发 Generation。PerceptionFamiliar 才负责把 payload 提交给 Coordinator。

手动三个用例互不混杂：`MANUAL_SETTLE` 只结算（不再 compact）并在结算材料被可靠接纳后 evict；`MANUAL_COMPACT` 只压缩工作集，不结算、不驱逐；`MANUAL_DELETE` 只驱逐、不生成记忆。manual settle 的提交顺序固定为冻结 settlement payload → generation admission 成功 → evict；admission 失败抛出受控错误且 Topic、blocks 与 state_summary 保持完整可重试。无任务（真正空 Topic 或 blocks 均被过滤）时 settle 仍按契约结束生命周期并返回成功，以 `generation_submitted` 表达是否建立后台任务。

### 内容判空语义

Topic 是否为空由 `blocks` 与 `state_summary` 共同决定：`has_content = blocks OR 非空白 state_summary`，`is_empty = NOT has_content`。空白字符串不构成有效摘要。summary-only Topic（刚完成压缩、尚无新对话）仍可被列出、继续路由并免于空 Topic 误删。真正空 Topic 没有可 settle/compact 的内容，但仍按决策矩阵执行 `evict=True` 的生命周期动作。compact 不再支持“总结后清空全部 blocks”的 `retain_count=0` 语义：所有 compact 配置与内部入口都要求 `retain_recent_blocks >= 1`。

这层返回值边界很重要：底层感知算法可以决定“应交出哪些材料”，但后台任务、事件和取消属于上层控制面。

## 5. Page Folding

当话题总 token 超过 `fold_token_threshold`（默认 32768）时，Perception 触发 `TOKEN_OVERFLOW`。`fold_retain_recent_blocks`（默认 2，必须大于 0）定义 active buffer 中保留的最近 blocks 数；旧前缀与 previous summary 由 RelayController 合成为新的 `state_summary`，保留后缀继续参与下一轮上下文。Relay 不总结保留后缀，避免 `state_summary + recent_blocks` 重复承载同一轮事实。

Page Folding 的设计目标是把 context window 视为工作集，而不是无限日志：旧页可以折叠为接力摘要，新的交互继续发生在同一话题。摘要是工作视图，不应被误认为原始证据；需要长期保留的原始 turn 应由 settlement/artifact 链另行保存。

Page Folding 是 Patchouli 的内部 topic working-set compaction，主动与被动入口共享同一语义。主动 Agent 会消费 `state_summary + recent_blocks`；Passive Ingress 的公共响应当前只返回检索记忆，不把折叠上下文返回给外部 Agent，但 Gateway 话题分析与后续 settlement generation 仍会消费这份内部工作集。外部 harness 自行 compact 对话不会缩减 Patchouli 已摄入的 blocks，因此不能作为跳过内部 folding 的依据。

当前 overflow 仍不执行 Settle。被移除的旧前缀不会自动进入 InteractionArtifact 或长期记忆；开放的 raw-evidence folding 方案仍位于 Ideas，尚不能当作当前能力。当 blocks 数量不大于保留数时没有可折叠前缀，本轮 compact 会延后，因此阈值是工作集软水位线，而不是严格的模型 context 上限。

## 6. RelayController

RelayController 只有摘要职责，不拥有话题、存储或生成：

- SimpleRelayController 以确定性规则形成简要接力摘要；
- LLMRelayController 可调用 Librarian LLM 生成更丰富摘要；
- NoOpRelayController 用于关闭折叠能力。

Perception engine 由 Runtime 按配置创建并注入 ShortTermMemoryStore。关闭 perception 时使用 NullPerceptionLayer 保持接口形状，但它不会保存或结算内容。

## 7. 维护与关闭

全局 scheduler 定期调用 `scan_idle_buffers_once()`；Familiar 按 `idle_timeout_seconds` 选择话题并逐个结算。关闭时 `flush_all_for_shutdown()` 结算并驱逐所有活跃话题；真正空 Topic 没有可提交材料，但仍按 SHUTDOWN 矩阵执行 evict，不留在活跃池中。

手动 settle 返回 `TopicSettleResult`，通过可选 `generation_task_id` 与派生的 `generation_submitted` 表达是否建立后台任务；无任务不等于生命周期失败。手动 evict（删除话题）返回 `TopicEvictionResult`，明确“不触发结算、不写记忆”，适合用户主动丢弃短期话题；manual compact 只压缩工作集，不结算、不驱逐。三种手动操作互不混用。Patchouli 业务结果不再放在 `services/perception.py`，server 层只在 HTTP 边界投影为响应模型。shutdown 批处理使用 `runtime.models.TopicShutdownFlushReport` 记录已结算 Topic、未建立 generation task 的正常 skip 子集，以及结算前驻留 block 数量；该运行时报告不进入 HTTP 链路，异常也不会被归入正常 skip。

## 8. 当前限制

- 短期话题是进程内状态，异常退出可能丢失未结算 blocks；
- token 统计只覆盖 user/final text 与部分 trace 字段，不是模型级精确 tokenizer 预算；
- `fold_retain_recent_blocks` 只限制 block 数量，不保证保留后缀的 token 总量低于阈值；单个超大 block 也可能超过软水位线；
- 所有 compact 配置与内部入口（`apply_compaction`、`_compact_topic`、公开配置 `ge=1`）都拒绝小于 1 的 retain 值，至少保留一个最新 block；summary-only Topic 可被列出、路由并免于空 Topic 误删，但当前 generation 仍以 `state_summary + 至少一个 recent block` 作为可结算材料，独立的 summary-only memory/artifact 生成能力未定义；
- overflow 不产生 settlement/artifact，被折叠旧前缀目前只进入有损 `state_summary`；
- Relay 的摘要调用位于结算路径内，LLM relay 可能增加该操作的同步等待；
- `worth_saving=False` 在 settlement 时过滤，但原始 block 在此之前仍存在短期 buffer；
- 旧文档中的 `assistant_message` fallback、扁平 `context_messages` 和 Perception 私有 `InteractionPayload` 均已退出主路径。

调整这些语义时必须同时检查 Generation、Artifacts、Passive Ingress 与 shutdown drain，因为“何时清空 blocks”本质上是数据耐久性边界，而不只是一个摘要算法参数。

后续跨入口上下文所有权、token-aware 保留、summary-only 与折叠证据 checkpoint 统一记录在 [Page Folding 跨入口后续技术债](../todo/page-folding-cross-ingress-follow-ups.md)。
