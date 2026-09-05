---
title: Patchouli Perception
status: current
owner: patchouli
scope: structured-ingestion-and-short-term-topics
code_paths:
  - src/hivememory/engines/perception/
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/patchouli/services/topic_working_set.py
  - src/hivememory/patchouli/memory_library/stores.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/system/passive-ingress.md
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-09-05
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

### 1.3 话题工作区：TopicData 快照

Perception 以 `TopicData` 快照作为话题工作视图，保存 blocks、展示 title/summary、Page Folding 产生的 `state_summary`、token 总量与最近模型名。它只承载内容事实：执行占用不建模为记录字段（原 `BufferState` 状态机与 `last_accessed_at` 访问时间已删除），跨 await 的占用权由 `TopicWorkingSet` 的 lease 表达，访问顺序由 WorkingSet 的 LRU 索引维护。短期 adapter 直接存储 frozen `TopicData`，Store/Port 与 Perception 之间只交接不可变快照；生命周期写回由 PerceptionFamiliar 编排。

## 2. 结构化摄入

`InteractionPayload` 是主动与被动入口共享的协议，当前主字段包括 user message、rewritten query、assistant final text、turn events、MTP traces、materialize tasks、worth_saving 与 model_used。它不重复保存 Workspace 身份；主动和被动入口在进入提交队列时由外层 `InteractionSubmission.identity_scope` 携带唯一的 `IdentityScope`。

交互应用成功后，Perception 才会把用户明确使用的 `(asset_id, asset_ref)` 记录为 Topic 的 `TopicAssetBinding`。上传、候选列表或 UI 选择本身不会进入 Topic 事实。由此，感知层只接收已完成身份与资源前置校验的交接输入，不拥有 `WorkspaceAssetStore` 的生命周期。

```text
InteractionPayload
  -> require turn_events
  -> ActionReducer(turn_events)
  -> TurnRecord
  -> token estimate
  -> LogicalBlock
  -> ShortTermMemoryStore.put(TopicData snapshot)
  -> optional Page Folding
```

主动流程由 Patchouli finalize 构建 payload；被动流程由 System 的 turn buffer 构建 payload。两者先进入同一 `InteractionSubmissionQueue`，再由 `PerceptionFamiliar.apply_interaction()` 编排 `MemoryPerceptionEngine.build_block()`（纯算法）、占用检查与 Store 写入，不存在被动专用的扁平文本主链，也不存在中间感知层。

结构化摄入的边界来自被动入口的真实事件形态：`MessageTurnBuffer` 接收 `user`、`assistant`、`tool_call`、`tool_result` 四类消息。只有自然语言 assistant 段落进入 `assistant_final_text`；工具调用与工具返回必须以 `TurnEvent` 保留，不能为了生成一段看似完整的 transcript 而提前丢弃。`ActionReducer`/`TraceReducer` 只能由 Perception 从这些事件派生，入口层不应另行构造第二套摘要，否则历史重放、主动生成与被动归档会拥有互相漂移的事实来源。

`target_topic` 则属于 user 到达时的 Gateway route 决策，而不是 flush 时重新计算的感知结果。Buffer 在接收 user 时保存目标话题；下一轮 user 到来时，先 flush 旧轮并完成旧轮的 payload，再初始化新轮。这样 `user2` 的 gaze 不会污染 `user1` 的归档归属。System Passive Ingress 拥有 session、idle 计时与事件提交；TheEye/Gateway 只负责入口判断，不重新拥有话题 buffer 或被动分析状态。

Perception 只把 `worth_saving` 写入 block，不在普通 ingest 时删除它。真正形成 settlement payload 时，`worth_saving is False` 的 blocks 被过滤，`None` 与 `True` 保留。这种保守语义确保 Gateway 缺失判断时不会默认丢弃材料。

## 3. 话题准备与 LRU

`prepare_topic()` 在 Agent run 之前确保目标话题存在：

- `NEW_TOPIC` 创建 UUID、title 和 summary；
- 已存在话题继续使用并刷新访问顺序；
- 指定话题不存在时拒绝请求，不把跨 Workspace 的全局 ID 误当作新话题。

当需要创建新话题且活跃池达到 `max_resident_topics` 时，`TopicWorkingSet.needs_eviction()` 判定容量、`select_lru_candidate()` 从驻留索引（OrderedDict，O(k)）选出最久未访问且未被占用（lease）的话题，按 `LRU_EVICTION` 结算并提交后台 generation task，然后从 Store 与驻留集合移除。已有话题命中不会为了“置顶”而误驱逐另一话题；候选被占用时改选其他话题，全部被占用时抛出 `TopicBusyError` 留待重试。容量与候选以 WorkingSet 驻留索引为准，Store 只表达内容事实。

Prepare 与 ingest 都会再次确保话题存在，以覆盖 prepare 后异常清理或其他边界变化。若 prepare 已创建新话题但 Agent run 失败，Patchouli cleanup 只在它仍为空时删除，避免误删已经摄入内容的话题。

## 4. 触发原因与具名用例

settle / compact / evict 是三个概念独立的具名用例，由 `PerceptionFamiliar` 编排，不再通过决策矩阵把触发原因解释成动作组合。`TriggerReason` 只作为 `TopicMaterializeTask` 的 provenance 标签（记录“这次结算为什么发生”），不驱动分支：

| 原因（provenance） | 触达的用例 | 当前结果 |
|:---|:---|:---|
| `TOKEN_OVERFLOW` | compact | 折叠旧前缀并保留最近工作集，话题继续存活 |
| `IDLE_TIMEOUT` | settle | 维护扫描中提交候选材料并移出活跃池 |
| `LRU_EVICTION` | settle | 为新话题腾出位置 |
| `SHUTDOWN` | settle | 停机前结算并驱逐（含真正空 Topic） |
| `MANUAL_SETTLE` | settle | 手动结算为记忆资产并结束 Topic 生命周期 |
| `MANUAL_COMPACT` | compact | 历史触发原因；当前无公开手动 compact 入口，compact 仅在 token 溢出时自动触发 |
| `MANUAL_DELETE` | evict | 历史触发原因；手动删除通过 `evict_topic` 用例表达，不写记忆 |

三个用例的职责边界：

- **settle**（`_settle_topic` 统一时序，所有来源共用）：获取 lease → 从 blocks 快照构建 `TopicMaterializeTask` → 锁外 generation admission → 删除话题并移除驻留。admission 失败时话题内容原样保留可重试，无需 abort——记录从未被改动。
- **compact**（`_compact_topic_if_needed`，调用方持有 lease）：把保留后缀之前的旧 blocks 交给 RelayController 生成 `state_summary`，形成新快照并写回摘要、裁剪旧前缀和重算 token。
- **evict**（`evict_topic` / `discard_if_empty`）：从 Store 与驻留集合移除话题，不结算、不写记忆；话题被占用时报 `removed=False` 而不是阻塞。

历史上曾由 `TriggerManager`/`TRIGGER_PLANS` 决策矩阵与记录字段状态机（`BufferState`）承担这些流程，现已删除：占用权建模为 WorkingSet 的 lease，而不是记录状态，补偿校验随之消失。

`TopicMaterializeTask` 是 Perception 交给 Generation 的冻结快照，除 Topic 内容和 `state_summary` 外还携带原始 `identity_scope` 与本轮已确认的 `asset_bindings`。进入队列后，重试和后续处理只能使用这份交接事实，不能从当前进程状态重新推导 Workspace 或资产关系。

手动用例互不混杂：`manual_settle_topic` 只结算并在结算材料被可靠接纳后删除话题；token 溢出 compact 只压缩工作集，不结算、不驱逐（当前没有公开的手动 compact 入口，`MANUAL_COMPACT` 仅作为 provenance 保留）；`evict_topic` 只驱逐、不生成记忆。manual settle 的提交顺序固定为冻结 settlement payload → generation admission 成功 → 删除；admission 失败抛出受控错误且 Topic、blocks 与 state_summary 保持完整可重试。无任务（真正空 Topic 或 blocks 均被过滤）时 settle 仍按契约结束生命周期并返回成功，以 `generation_submitted` 表达是否建立后台任务。

### 内容判空语义

Topic 是否为空由 `blocks` 与 `state_summary` 共同决定：`has_content = blocks OR 非空白 state_summary`，`is_empty = NOT has_content`。空白字符串不构成有效摘要。summary-only Topic（刚完成压缩、尚无新对话）仍可被列出、继续路由并免于空 Topic 误删。真正空 Topic 没有可 settle/compact 的内容，但 settle 仍会正常结束其生命周期。compact 不支持“总结后清空全部 blocks”的 `retain_count=0` 语义：所有 compact 配置与内部入口都要求 `retain_recent_blocks >= 1`。

这层返回值边界很重要：底层感知算法可以决定“应交出哪些材料”，但后台任务、事件和取消属于上层控制面。

## 5. Page Folding

当话题总 token 超过 `fold_token_threshold`（默认 32768）时，Familiar 在当次 apply 的 lease 持有期间触发 compact。`fold_retain_recent_blocks`（默认 2，必须大于 0）定义保留的最近 blocks 数；旧前缀与 previous summary 由 RelayController 合成为新的 `state_summary`，保留后缀继续参与下一轮上下文。Relay 不总结保留后缀，避免 `state_summary + recent_blocks` 重复承载同一轮事实。由于 Relay 摘要生成的整个窗口内调用方持有该话题的 lease，其他调用无法同时取得占用权，写回不再需要旧的 expected_state 前缀补偿校验。

Page Folding 的设计目标是把 context window 视为工作集，而不是无限日志：旧页可以折叠为接力摘要，新的交互继续发生在同一话题。摘要是工作视图，不应被误认为原始证据；需要长期保留的原始 turn 应由 settlement/artifact 链另行保存。

Page Folding 是 Patchouli 的内部 topic working-set compaction，主动与被动入口共享同一语义。主动 Agent 会消费 `state_summary + recent_blocks`；Passive Ingress 的公共响应当前只返回检索记忆，不把折叠上下文返回给外部 Agent，但 Gateway 话题分析与后续 settlement generation 仍会消费这份内部工作集。外部 harness 自行 compact 对话不会缩减 Patchouli 已摄入的 blocks，因此不能作为跳过内部 folding 的依据。

当前 overflow 仍不执行 Settle。被移除的旧前缀不会自动进入 InteractionArtifact 或长期记忆；开放的 raw-evidence folding 方案仍位于 Ideas，尚不能当作当前能力。当 blocks 数量不大于保留数时没有可折叠前缀，本轮 compact 会延后，因此阈值是工作集软水位线，而不是严格的模型 context 上限。

## 6. MemoryPerceptionEngine 与 RelayController

`MemoryPerceptionEngine` 是无状态的短期记忆摄入与 compact 算法引擎，职责包括：

- `build_block`: 把 `InteractionPayload` 转化为不可变 `LogicalBlock`，归并结构化事件为 actions、估算 token；
- `should_compact`: 判断话题总 token 是否超过折叠阈值；
- `select_blocks_to_fold`: 选择待折叠的旧 blocks 前缀（保留最近 N 个）；
- `generate_fold_summary`: 生成 Page Folding 摘要（委托给持有的 `RelayController`）。

Engine 持有 `RelayController`，封装了完整的 compact 算法能力。`PerceptionFamiliar` 通过 Engine 的 `generate_fold_summary()` 获取摘要，不再直接持有 Relay 组件。

RelayController 有三种实现：

- **SimpleRelayController**: 以确定性规则形成简要接力摘要；
- **LLMRelayController**: 可调用 Librarian LLM 生成更丰富摘要；
- **NoOpRelayController**: 用于关闭折叠能力。

Relay 控制器由 Runtime 按配置创建并注入 Engine 构造。关闭 perception（`engine.enable=false`）时 Runtime 注入 `engine=None`，Familiar 的摄入路径变为无副作用 no-op（原 NullPerceptionLayer 语义），维护用例对空存储自然空转。

Engine 不持有 Store / Journal / Queue，也不导入 `hivememory.patchouli.*`，可被其他 runtime 复用、可纯单元测试。

## 7. 维护与关闭

全局 scheduler 定期调用 `scan_idle_buffers_once()`；候选来自 WorkingSet 的 idle 查询（访问时间超时且未被 lease 占用），逐个执行统一 settle 时序，admission 失败记录警告并留待下一轮。关闭时 `flush_all_for_shutdown()` 对全部驻留话题执行同一时序；真正空 Topic 没有可提交材料，但仍正常结束生命周期并计入 `generation_skipped_topic_ids`。单个话题的占用冲突或 admission 异常被隔离到报告的 `failed_topic_ids`，不阻止其余话题清理。

手动 settle 返回 `TopicSettleResult`，通过可选 `generation_task_id` 与派生的 `generation_submitted` 表达是否建立后台任务；无任务不等于生命周期失败。手动 evict（删除话题）返回 `TopicEvictionResult`，明确“不触发结算、不写记忆”，适合用户主动丢弃短期话题；manual compact 只压缩工作集，不结算、不驱逐。三种手动操作互不混用。Patchouli 业务结果不再放在 `services/perception.py`，server 层只在 HTTP 边界投影为响应模型。shutdown 批处理使用 `runtime.models.TopicShutdownFlushReport` 记录已结算 Topic、未建立 generation task 的正常 skip 子集，以及结算前驻留 block 数量；该运行时报告不进入 HTTP 链路，异常也不会被归入正常 skip。

## 8. 当前限制

- 短期话题是进程内状态，异常退出可能丢失未结算 blocks；
- token 统计只覆盖 user/final text 与部分 trace 字段，不是模型级精确 tokenizer 预算；
- `fold_retain_recent_blocks` 只限制 block 数量，不保证保留后缀的 token 总量低于阈值；单个超大 block 也可能超过软水位线；
- 所有 compact 配置与 Engine 的折叠选择都拒绝小于 1 的 retain 值，至少保留一个最新 block；summary-only Topic 可被列出、路由并免于空 Topic 误删，但当前 generation 仍以 `state_summary + 至少一个 recent block` 作为可结算材料，独立的 summary-only memory/artifact 生成能力未定义；
- overflow 不产生 settlement/artifact，被折叠旧前缀目前只进入有损 `state_summary`；
- Relay 的摘要调用位于 compact 路径内（lease 持有期间），LLM relay 可能增加该操作的同步等待；
- `worth_saving=False` 在 settlement 时过滤，但原始 block 在此之前仍存在于短期话题中；
- WorkingSet 的容量检查与候选选择运行在单事件循环内（方法同步、不内部 await），跨线程使用需调用方先行串行化；两个并发 NEW_TOPIC 的容量检查存在与旧实现等价的短暂竞态窗口；
- flush 触发器的历史 e2e 用例随感知层删除而移除，重建需要真实 LLM/Qdrant 栈；当前回归由 Familiar 集成测试（tests/integration/patchouli/test_perception_flush_chain.py）覆盖；
- 旧文档中的 `assistant_message` fallback、扁平 `context_messages` 和 Perception 私有 `InteractionPayload` 均已退出主路径。

调整这些语义时必须同时检查 Generation、Artifacts、Passive Ingress 与 shutdown drain，因为“何时清空 blocks”本质上是数据耐久性边界，而不只是一个摘要算法参数。

后续跨入口上下文所有权、token-aware 保留、summary-only 与折叠证据 checkpoint 统一记录在 [Page Folding 跨入口后续技术债](../todo/page-folding-cross-ingress-follow-ups.md)。
