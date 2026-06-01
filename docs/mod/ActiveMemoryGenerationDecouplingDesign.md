# 主动记忆生成脱离感知层设计

**文档状态**: Draft (草案)
**适用范围**: `engines/perception/trigger_manager.py`（DECISION_MATRIX / FlushReason）、`engines/perception/semantic_flow_perception_layer.py`（ingest_payload URGENT 分支）、`engines/perception/models.py`（FlushReason / LogicalBlock.write_focus/update_focus / ArchivePayload.focus）、`patchouli/services/librarian.py`（submit_interaction / _on_generate_memory）、`patchouli/service.py`（finalize_agent_run）、`core/protocol/models.py`（InteractionPayload）
**核心目标**: 让 MTP WRITE/UPDATE 触发的主动记忆生成脱离感知层 buffer 链路，由 finalize 阶段直接驱动 generation engine 的 mode b/c；感知层回归"短期对话 MMU"本分，不再感知 focus/pending；同时消除主动写蹭用"archive + compact + clear"导致的复数请求丢上下文与细节被反复 summary 碾碎两个问题。

---

## 1. 文档目标

本文是执行引擎边界裁定（[AgentRuntimeBoundaryDesign](AgentRuntimeBoundaryDesign.md)）与逻辑解耦（[AgentLoopDecouplingDesign](AgentLoopDecouplingDesign.md)）的后续工作，承接 [PendingAtomMaterializeTaskDesign](PendingAtomMaterializeTaskDesign.md) 中 `materialize_tasks` 进入 patchouli 后的路径定义。

它处理一个感知层早期设计遗留的耦合：为统一记忆生成流的来源，当初刻意让 WRITE/UPDATE 数据流入感知层、再触发归档生成。该设计造成两个实际问题（§2 详证）：复数主动请求互相丢失历史上下文；每次主动请求都触发 summary 导致细节飞速流失。

设计目标：

- **主动生成脱离 perception buffer**：WRITE/UPDATE 的记忆生成不再经 `submit_interaction → route_and_ingest → URGENT flush → _on_generate_memory` 链，改由 finalize 直接拿 `materialize_tasks` 调 generation engine 的 mode b/c。
- **感知层去 focus 化**：移除 `FlushReason.MTP_WRITE/MTP_UPDATE`、`ingest_payload` 的 URGENT 分支、`LogicalBlock`/`ArchivePayload` 的 focus 字段。perception 不再知道 focus/pending 的存在。
- **summary 回归正确语义**：state_summary 只在 `TOKEN_OVERFLOW` 时压缩，不再被主动写蹭触发。
- **三流分离**：被动对话感知（buffer，自然消亡）、主动记忆写入（materialize_tasks → mode b/c，即时）、上下文读取（buffer 只读，供两者共享）。

本文不改 MTP 协议、不改 generation engine 的 mode b/c 内部提取逻辑、不改 PendingAtom 状态机与 Settlement 回流。

---

## 2. 当前问题（代码核证）

### 2.1 主动写蹭用了"话题结算"动作组合

当前 WRITE/UPDATE 的链路：

```text
finalize → submit_interaction → route_and_ingest → ingest_payload
  → is_urgent (write_focus/update_focus 非空)
  → trigger_manager.resolve_topic(MTP_WRITE/MTP_UPDATE, mtp_focus=focus)
  → _on_generate_memory(ArchivePayload{blocks, state_summary, focus, reason})
  → generation_engine.process(mode b/c)
```

决策矩阵（[trigger_manager.py:64-73](../../src/hivememory/engines/perception/trigger_manager.py#L64)）对两个主动 reason 的定义：

| FlushReason | archive | compact | evict |
| :--- | :---: | :---: | :---: |
| `MTP_WRITE` | ✅ | ✅ | ❌ |
| `MTP_UPDATE` | ✅ | ✅ | ❌ |
| `TOKEN_OVERFLOW` | ❌ | ✅ | ❌ |

且 `resolve_topic` 在 archive/compact 之后**无条件清空 blocks**（[trigger_manager.py:209-211](../../src/hivememory/engines/perception/trigger_manager.py#L209)）：

```python
buffer.blocks.clear()
buffer.total_tokens = 0
```

即：**每次 WRITE/UPDATE = 打包归档 + 强制 compact(summary) + 清空 blocks。**

### 2.2 由此导出的两个问题

**问题 1：复数主动请求互相丢上下文。** 第一个 WRITE 触发后 `blocks.clear()`，buffer 只剩压缩后的 state_summary。第二个 WRITE 进来时，`_build_generation_context`（[librarian.py:179](../../src/hivememory/patchouli/services/librarian.py#L179)）只能基于"被压扁的 summary + 它自己那一个 block"生成——前面对话的原始细节已被 summary 吞掉。

**问题 2：summary 把细节碾碎。** `compact: True` 使每次 WRITE 都跑一遍 `relay_controller.generate_summary`（[trigger_manager.py:290](../../src/hivememory/engines/perception/trigger_manager.py#L290)）。一个每轮都 write 的 Agent，buffer 被反复 summary，原始 turn 细节飞速流失。而 `TOKEN_OVERFLOW` 才是"token 溢出才 compact"的正确语义——主动写蹭用这套机制纯属副作用。

### 2.3 根因

这与 [AgentRuntimeBoundaryDesign](AgentRuntimeBoundaryDesign.md) 揭示的模式同源：**主动写错误地复用了被动感知的管道。** archive+compact+clear 是为"话题结算"（一段对话告一段落、需要落库并留摘要接力）设计的动作组合，主动 WRITE 只是想生成一条记忆，却被迫触发了整套话题结算。

---

## 3. 目标设计

### 3.1 三条流分离

重构后三条流各行其道，perception 不再知道 focus/pending：

| 流 | 触发 | 路径 | 产物 |
| :--- | :--- | :--- | :--- |
| 被动对话感知 | 每轮交互 | `submit_interaction → ingest`（无 focus） | block 入 buffer，自然消亡时归档 |
| 主动记忆写入 | MTP WRITE/UPDATE | `finalize → materialize_tasks → mode b/c`（跳过 perception） | canonical 记忆原子，即时生成 |
| 上下文读取 | 两者共享 | `topic_context`（buffer 只读） | 历史对话作背景，不标记消费 |

### 3.2 主动生成走 finalize 直驱 mode b/c

`finalize_agent_run` 拿到 `AgentRunResult.materialize_tasks` 后，直接驱动 generation engine 的 mode b/c，不再经 perception buffer：

1. 先 `submit_interaction`（把本轮对话 block 写进 buffer，走被动流）。
2. 从 buffer 读 `topic_context`（含 blocks + state_summary）作 mode b/c 的只读历史背景。
3. 对每个 task 按 `source_verb` 调 generation engine 的 mode b（WRITE）/ mode c（UPDATE），focus 提供写参数、intent_id/pending_alias 用于组 Settlement。

### 3.3 时序约束（一致性关键）

主动生成现在发生在 finalize 阶段，而本轮对话 block 也在 finalize 才由 `submit_interaction` 写入 buffer。因此 finalize 内部顺序**必须钉死**：

```text
finalize:
  1. submit_interaction(payload)        # 本轮对话 block 进 buffer（被动流）
  2. topic_context = 读 buffer 上下文     # 此时已含「刚刚这轮」
  3. for task in materialize_tasks:      # 主动流：mode b/c 直驱
       generation_engine.process(mode b/c, context=topic_context, task=task)
```

若顺序颠倒（先跑 mode b/c 再 submit），mode b/c 会看不到当前这一轮对话，一致性反而比现状更差。"历史上下文、focus、最终记忆原子三者一致"依赖此顺序。

### 3.4 主动 block 作只读背景，不标记消费

mode b/c 把 `topic_context` 的对话 block 当**只读背景**，不把它们标记为已归档/已消费。这些 block 仍留在 buffer，将来因 token 溢出 compact、或话题更迭/idle/LRU 时正常走被动归档。

这意味着：**主动 WRITE 这一轮的对话，仍会被动归档。** 这是有意接受的（§4 决策）。

### 3.5 主动流程不再 summary / 不再 clear

`MTP_WRITE`/`MTP_UPDATE` 退出决策矩阵后，主动写不再触发 compact 与 blocks.clear。buffer 只会因：

- `TOKEN_OVERFLOW` → compact（留摘要接力，正确语义）
- 话题更迭 / `IDLE_TIMEOUT` / `LRU_EVICTION` / `MANUAL` / `SHUTDOWN` → 归档并自然消亡

即 state_summary 回归"仅 token 溢出时压缩"，buffer 因话题更迭与用户意志自然消亡。

---

## 4. 关键决策：主动 WRITE 这一轮对话仍走被动归档

§3.4 的结果是：一次主动 WRITE 的对话内容，既被 mode b 生成了 canonical 记忆，其对话 block 将来又会被被动归档（Mode A 提取）——存在内容重复的可能。

**决策：接受被动归档，不做去重特判。** 依据：

- **两层兜底**：① generation engine 的 deduplicator 对已有记忆原子查重（CREATE/UPDATE/TOUCH/DISCARD），主动写产出的 canonical 已在库，被动重复内容大概率被判 TOUCH/DISCARD 或 UPDATE 合并；② 未来的记忆分裂与合并机制会定期整合内容过近的原子。
- **稳定性优先**：被动归档重复主动 WRITE 内容，最多增加后续维护成本，**不影响已有链路稳定性**。作为已知小缺陷接受，避免为消除它在两条流之间引入特判耦合（特判会让 perception 重新感知"哪些 block 是主动写过的"，正是本文要消除的耦合）。

> 语义上这也更干净：对话归对话（被动流，自然归档），写入归写入（主动流，即时生成）。两条流彻底正交，代价是边界处的内容冗余——这笔交易划算。

---

## 5. 改动清单

### 5.1 感知层去 focus 化

| 文件 | 改动 |
| :--- | :--- |
| `engines/perception/trigger_manager.py` | DECISION_MATRIX 删除 `MTP_WRITE` / `MTP_UPDATE` 两项；`resolve_topic` 的 `mtp_focus` 参数移除 |
| `engines/perception/models.py` | `FlushReason` 删除 `MTP_WRITE` / `MTP_UPDATE`；`LogicalBlock` 删除 `write_focus` / `update_focus`；`ArchivePayload` 删除 `focus`（及 reason 中的主动分支） |
| `engines/perception/semantic_flow_perception_layer.py::ingest_payload` | 删除 URGENT 分支（[L216-236](../../src/hivememory/engines/perception/semantic_flow_perception_layer.py#L216)）、`write_focus`/`update_focus` 读取（[L201-202](../../src/hivememory/engines/perception/semantic_flow_perception_layer.py#L201)）；只保留普通 block 添加 + `_maybe_fold_pages`（TOKEN_OVERFLOW） |

### 5.2 librarian / 生成回调

| 文件 | 改动 |
| :--- | :--- |
| `patchouli/services/librarian.py::_on_generate_memory` | 移除 focus/reason 的 mode b/c 分支（[L167-219](../../src/hivememory/patchouli/services/librarian.py#L167)），回调只处理被动 Mode A 提取；Settlement 发布逻辑迁到主动生成路径（§5.3） |
| `patchouli/services/librarian.py::submit_interaction` | 日志去掉 write_focus/update_focus 字段；payload 不再带 focus |
| `patchouli/services/librarian.py` | 新增主动生成入口（如 `materialize_tasks(tasks, topic_id)`）：读 topic_context → 逐 task 调 mode b/c → 发布 Settlement |

### 5.3 finalize 控制流

| 文件 | 改动 |
| :--- | :--- |
| `patchouli/service.py::finalize_agent_run` | 按 §3.3 时序：先 `submit_interaction` → 读 `topic_context` → 调新增主动生成入口处理 `materialize_tasks`；Settlement 回流不变 |
| `core/protocol/models.py::InteractionPayload` | 移除 `write_focus` / `update_focus`（已由 [PendingAtomMaterializeTaskDesign](PendingAtomMaterializeTaskDesign.md) 改为 `materialize_tasks`，本文确保它不再流入 perception，而是流入主动生成入口） |

### 5.4 验证

- e2e：`test_write_chain` / `test_update_chain` / `test_active_mode_e2e` 全绿——主动 WRITE/UPDATE 仍正确生成 canonical 记忆、Settlement 正确回填。
- 专项回归：
  - **复数主动请求**：连续两次 WRITE，第二次的 mode b 上下文包含第一次的对话细节（不再被 summary 碾碎）。
  - **summary 不被主动写触发**：连续 WRITE 不增加 state_summary 压缩次数；仅 token 溢出时 compact。
  - **被动归档仍发生**：主动 WRITE 后，其对话 block 仍在 buffer，idle/LRU 时正常归档（接受重复，deduplicator 兜底）。
  - perception 代码中不再出现 focus/pending/write_focus/update_focus 任何引用。

---

## 6. 落地时间与依赖

本文**依赖 [PendingAtomMaterializeTaskDesign](PendingAtomMaterializeTaskDesign.md)**：主动生成入口消费的是 `materialize_tasks`（Task 携带 focus + 关联键），而非旧的 `write_focus`/`update_focus` 字段。因此排在 MaterializeTask 重组之后。

完整演进链顺序：

1. [AgentLoopDecouplingDesign](AgentLoopDecouplingDesign.md) — loop_executor 解耦，结果组装落到 AgentOrchestrator。
2. [PendingAtomMaterializeTaskDesign](PendingAtomMaterializeTaskDesign.md) — Focus 瘦身、Task 投影、`AgentRunResult` 重组。
3. **本文** — 主动生成脱离 perception，由 finalize 直驱 mode b/c。

本文不依赖 `agent_runtime/` 目录迁移，可在现结构内完成。改动集中在 patchouli + engines/perception，alice 侧不受影响（Task 组装在前序已完成）。

完成本文后，感知层回归纯"短期对话 MMU"本分，主动与被动两条记忆生成流彻底正交，summary 回归 token 溢出语义。

