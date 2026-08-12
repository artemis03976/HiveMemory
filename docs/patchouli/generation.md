---
title: Patchouli Memory Generation
status: current
owner: patchouli
scope: memory-materialization-and-background-tasks
code_paths:
  - src/hivememory/engines/generation/
  - src/hivememory/prompts/transcript/generation.py
  - src/hivememory/patchouli/services/memory_generation.py
  - src/hivememory/patchouli/control/memory_generation/
  - src/hivememory/patchouli/runtime/memory_tasks.py
related_contracts:
  - docs/contracts/mtp.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/subsystem-contracts.md
last_reviewed: 2026-08-12
---

# 记忆生成

记忆生成是把短期材料或显式保存意图转化为正式 `MemoryAtom` 的冷路径。它必须同时避免两种对称错误：把每段对话都机械保存，会让书库迅速被噪音淹没；把保存完全交给不透明的 LLM 判断，又可能让用户明确要求保留的内容悄悄消失。

当前设计因此保留三种生成模式，并把“构造任务”“执行生成”“写 artifacts 和持久化”“发布任务/settlement 终态”拆成不同层。生成不再是 Perception 的一个回调，也不再由一个 LibrarianCore 同时管理计算与后台任务。

提取阶段选择 LLM 而不是固定规则，是因为“是否值得长期保存”以及如何把自然语言重组为 title、summary、content 与 tags，本质上依赖跨表达方式的语义判断；只靠关键词或消息计数会把调用方措辞固化成记忆结构。代价是延迟、成本与非确定性，因此它被放在后台冷路径，并为显式 WRITE 保留确定性的 fallback；LLM 可以提出记忆草稿，但不能独自决定任务终态、持久化成功或用户承诺。

## 1. 控制面与数据面

```text
MemoryGenerationCoordinator
  -> MemoryGenerationTaskSpec
  -> MemoryGenerationTaskController
       -> immutable MemoryGenerationWork
       -> MemoryGenerationQueue / TaskHandle
       -> local route GENERATION_EXECUTE_SPEC
  -> MemoryGenerationFamiliar
       -> MemoryGenerationEngine.process
       -> ArtifactEngine
       -> MidTermMemoryStore.upsert
  -> PendingAtom settlement + RuntimeEvent
```

- Coordinator 把 settlement 或 PendingAtom materialize task 变成统一 spec；
- TaskController 创建不可变 work 并持有 typed handle，负责领域快照、取消、等待与事件时机；
- MemoryTaskEventEmitter 负责把领域快照投影为 `memory.task.*` RuntimeEvent；
- Familiar 是生成数据面，执行 compute -> artifact -> persist；
- GenerationEngine 负责提取、合并、去重和构造 outcome，不直接持久化或发布事件。

这条边界使 GenerationEngine 的返回值可以在落库前被补充来源信息，也让取消/失败只由 TaskController 写入一次终态。

## 2. 统一输入视图

Generation 不再消费拼接的 `context_messages`。`GenerationTranscriptBuilder` 从 `LogicalBlock[]` 构造：

```text
GenerationContext
  ├─ state_summary
  └─ GenerationTurn[]
       ├─ user_query
       ├─ assistant_final_text
       ├─ trace_summaries
       └─ identity
```

它保留对记忆提取有意义的语义，丢弃工具结果正文等执行噪音。历史重放与记忆生成共享底层 TurnRecord，却各自生成适合任务的视图；Generation 不能反向依赖 Alice 的消息渲染格式。

## 3. 三种生成模式

### 3.1 Mode A：被动观察 / ARCHIVE

当 idle、LRU、shutdown 或 manual settle 形成 `TopicMaterializeTask` 时，Coordinator 用其中的 blocks 与 state summary 构造 GenerationContext。没有 turns 时跳过任务。

GenerationEngine 渲染 transcript，调用 extractor 判断长期价值并生成 `ExtractedMemoryDraft`；没有草稿或 `has_value=false` 时不产生记忆。有效草稿随后进入 dense top-1 查重。

Mode A 的“被动”指没有显式 WRITE/UPDATE focus，而不是同步发生在用户响应内。它仍作为 Patchouli 后台 memory task 执行。

### 3.2 Mode B：主动 WRITE

Alice 的 MTP WRITE 先创建 PendingAtom 与 `PendingAtomMaterializeTask`。Patchouli finalize 把任务交给 Coordinator；Coordinator 使用话题最近五个 blocks 作为背景，并把 `WriteFocus` 作为保存核心。

Extractor 失败时，Mode B 会直接从 WriteFocus 构建 fallback draft，保证明确保存意图不会因为 Librarian LLM 暂时失败而无声丢失。Fallback 仍要经过去重，因而最终可能 CREATE、UPDATE、TOUCH 或 DISCARD；ACK 从未承诺“一定新建一条独立记忆”。

### 3.3 Mode C：主动 UPDATE

UPDATE spec 构建时，Coordinator 解析 `UpdateFocus.base_uuid` 并通过 local route 读取现有 MemoryAtom。UUID 非法或目标不存在只使该 active spec 失败，并发布对应 pending failure，不阻断同批其他任务。

Engine 用 extractor merge 生成新正文与 changelog；merge 失败时：

- focus 携带 content：把内容作为带日期的更新段追加到旧正文；
- 只有 instruction：保留旧正文，记录无内容变更的 fallback changelog。

更新前后快照交给 Artifact 层生成 MemoryVersionArtifact，MemoryAtom version 递增并写回中期库。

## 4. 去重与演化

Mode A/B 使用 `title + summary` 对中期库执行 dense top-1 搜索，Deduplicator 根据相似度与内容关系返回：

| 决策 | 当前行为 |
|:---|:---|
| `CREATE` | 从 draft 构造新 MemoryAtom 与 canonical alias |
| `UPDATE` | 用 draft 内容覆盖现有 head，合并索引 tags，递增 version |
| `TOUCH` | 增加 access count、刷新 updated time，并重新 upsert |
| `DISCARD` | 不写 MemoryAtom，返回低质量重复消息 |

去重 UPDATE 当前使用轻量覆盖，不调用强合并 prompt；这是刻意保守的当前实现，但也意味着 draft 质量直接决定新 head。历史由 MemoryVersionArtifact 保存，不能通过向正文不断追加旧版本来模拟版本控制。

Alias 由 memory type 前缀和 extractor 给出的 suffix 构造，例如 `code_* / fact_* / url_* / ref_* / user_* / wip_* / agent_*`。Pending alias 只在 Alice 运行期定位候选意图；settlement 后的 canonical alias 才属于正式 MemoryAtom。

## 5. Artifact 与持久化顺序

MemoryGenerationFamiliar 执行：

1. 尝试捕获 InteractionArtifact；
2. 调用 GenerationEngine 得到 outcomes；
3. 为 CREATE/UPDATE 尝试构建 creation/version artifacts，并把 refs/events 挂到 atom；
4. 对非 DISCARD 且 atom 非空的结果执行 MidTerm upsert；
5. 把 outcome 投影为 `MemoryGenerationResult` 与可选 PendingAtomSettlement。

Artifact 写入是 best effort，Qdrant upsert 失败则任务失败。详见[Artifacts 与来源追踪](./artifacts.md)。

手工 memory create/update 同样通过 Familiar：创建生成 MANUAL provenance，编辑捕获 before snapshot、递增 version 并生成 MANUAL_EDIT version artifact。

## 6. 后台任务与终态

每个 spec 对应一个不可变 `MemoryGenerationWork`；公开 API 使用同名的只读
`MemoryGenerationTask` 快照表达下列领域状态：

```text
PENDING -> RUNNING -> COMPLETED | FAILED | CANCELLED
```

`WorkRecord` 是执行状态的唯一真相源，Queue 的 `TaskHandle` 提供 snapshot/wait/cancel；
TaskController 只在查询时映射非终态快照，并在 work 终止后执行一次 settlement 与终态事件。
提交返回的 `MemoryGenerationTask` 不会原地更新，调用方需要通过 list/get/wait 获取新快照。

控制面提供 list/get/cancel/wait/wait_many/wait_all。等待使用 `asyncio.shield`，超时只返回快照，不会接管或自动取消后台任务；shutdown drain 在 wait_all 超时后才显式 cancel 那批任务。

任务状态通过 RuntimeEvent 发布。TaskController 只决定生命周期事件的发生时机，`MemoryTaskEventEmitter` 集中选择 `memory.task.*` 类型并组装稳定 payload；通用队列仍独立发布 `work.*` 基础设施事件，两者不互相替代。主动任务完成后，settlement 会通过 PatchouliBridge 转发为全局 PendingAtom event；发布失败不会把已持久化记忆回滚，但会 best-effort 发布 pending failure，使 Alice 不无限等待。PendingAtom settlement、failed、cancelled 仍属于功能事件并继续留在 TaskController。

## 7. Active finalize 的时序

Patchouli finalize 当前顺序为：

```text
AgentRunResult
  -> build InteractionPayload
  -> wait until current turn is applied to Perception
  -> lock Chat completed
  -> independently submit WRITE/UPDATE materialize tasks
  -> best-effort record prepared retrieval HITs
```

Interaction applied 是 Chat 的硬成功边界。之后 materialization admission 或执行失败由 `MemoryGenerationTask` 与 PendingAtom failed 独立表达，retrieval HIT 失败只记录 warning；两者都不能反向把 Chat 改写为 failed。Active continuation 由 Patchouli 进程级持有，因此 HTTP/SSE 调用方取消不会中止已经接管的 post-apply 义务。

主动生成因此可以从 topic 最近 blocks 中看到当前轮，而不会为了 WRITE/UPDATE 调用话题 settlement、summary 或 clear。当前交互仍留在被动话题链中，之后可在 idle/LRU/shutdown 时参与 Mode A；主动保存与被动归档是不同意图，不能通过清空 buffer 相互替代。

这条解耦边界是对旧链路实际问题的回应。旧实现曾把 MTP WRITE/UPDATE 当作话题结算理由，执行 `archive + compact + clear`：连续两次 WRITE 时，第二次只能看到已经被压扁的 summary；每次主动写还会额外触发一次 summary，使原始细节在明确保存意图下反复丢失。当前实现直接使用 topic 最近 context，并以 `WriteFocus`/`UpdateFocus` 驱动 Mode B/C；主动生成与被动归档保持两条正交流，主动写入后的 blocks 仍可在之后进入 Mode A。

这并不保证主动写出的正文与被动归档结果完全不重复。重复是当前可接受的成本，由 deduplicator 的 `CREATE`、`UPDATE`、`TOUCH`、`DISCARD` 决定最终落库动作；为了消除重复，不能重新让 Perception 感知 focus 或 pending 状态，否则会把控制意图污染为对话事实。当前每个主动 task 也会独立捕获 `InteractionArtifact`，因此 provenance 可能重复，这是已知的可追踪性成本，而不是重新合并任务输入的理由。

## 8. 当前限制

- memory work、typed result 与终态快照只存在于当前进程，重启后不可恢复；持久化与恢复边界见[运行时状态持久化与故障恢复计划](../plans/runtime-state-durability-and-recovery.md)；
- 终态快照与 Queue 使用同一 `terminal_retention` 上限；
- `submit_generation_many()` 逐个创建后台 task；active spec 的 I/O 构建并行，但没有持久化队列、并发额度或 backpressure；
- 运行中的 extractor/merge 调用不能保证在任意阻塞点立即响应 cancel；
- Mode A/B 去重只取 dense top-1 且没有 identity filter，跨用户隔离依赖存储/数据前提，仍需收紧；
- Dedup UPDATE 是直接覆盖 draft content，不是强语义 merge；
- Active tasks 复用相同 blocks 输入，但会各自写 InteractionArtifact；
- Artifact 失败不阻断主记忆，provenance 与 MemoryAtom 不是原子提交。

未来若引入 durable queue，首先要保持 `MemoryGenerationWork` codec、`WorkRecord` 单一状态源和 settlement 语义不变，避免基础设施升级重新把控制面塞回 GenerationEngine。
