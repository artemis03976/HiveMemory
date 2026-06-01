# PendingAtomMaterializeTask 与 AgentRunResult 重组设计

**文档状态**: Draft (草案)
**适用范围**: `core/models/pending.py`（WriteFocus / UpdateFocus / 新增 PendingAtomMaterializeTask）、`core/protocol/models.py`（ChatResult→AgentRunResult / InteractionPayload）、`alice/runtime/pending_atom/runtime.py`、`alice/runtime/koakuma.py`、`alice/runtime/orchestrator.py`（解耦后）、`patchouli/service.py`、`patchouli/services/librarian.py`、`engines/generation/{models,engine}.py`
**核心目标**: 把 WriteFocus / UpdateFocus 彻底退化为"只承载 Koakuma 从 MTP 指令解析出的参数"的不可变数据模型；将"生成关联键"从 Focus 中剥离、归位到 PendingAtom 单一真相源；引入 `PendingAtomMaterializeTask` 作为跨子系统的不可变物化请求；将 `ChatResult` 重组为 `AgentRunResult`，确立"每个字段由 Alice 组装且有明确下游流向"的不变量。

---

## 1. 文档目标

本文承接 [PendingAtomRuntimeDesign](PendingAtomRuntimeDesign.md) 与 [AgentLoopDecouplingDesign](AgentLoopDecouplingDesign.md)，处理一个历史遗留设计债：`WriteFocus` / `UpdateFocus` 最初只用于把 MTP WRITE/UPDATE 的请求内容从 Koakuma 传给 patchouli 生成域，但随 Phase 2 与 PendingAtom 演进，它逐渐同时承担了过多角色——既装"Agent 提交的参数"（content/title/reason/instruction），又装"生成关联键"（pending_alias/intent_id/identity），还被一路穿透到 `ChatResult` 三字段，由 loop_executor / harvest 反复维护。

设计目标：

- **Focus 回归初心**：只承载 Koakuma 从 MTP 指令解析出的参数，不可变，可一路无损传递到 patchouli 生成域。
- **关联键单一真相源**：intent_id / pending_alias / identity 等关联键由 `PendingAtom` 维护，不再复制进 Focus 穿透链路。
- **引入物化请求 `PendingAtomMaterializeTask`**：作为 `PendingAtom` 的不可变投影出境，与回流的 `PendingAtomSettlement` 构成一对请求/应答对偶。
- **`ChatResult` 重组为 `AgentRunResult`**：表达"上层一次 chat 调用后系统运行至自然中断的完整产出"，并确立字段不变量。
- **消除 `ChatResult` 三字段冗余**与 loop_executor 对 `write_foci`/`update_foci`/`pending_aliases` 的维护。

本文不改 MTP 协议语义、不改 PendingAtom 状态机、不改 Settlement 回流链路。

---

## 2. 当前问题

### 2.1 Focus 被劈成两半用，却塞在一个对象里穿透多层

Focus 当前的穿透路径（已核对源码）：

```text
Koakuma._handle_write/update          ← Focus 诞生（打包 Agent 参数 + 关联键）
  → pending_runtime.register_write      ← 同一份数据又进 PendingAtom（写缓冲已持有完整真相）
  → MTPResponse.write_focus + pending_alias
  → loop_executor: write_foci / update_foci / pending_aliases 三累积器
  → ChatResult.write_focus / update_focus / pending_aliases   ← 三字段冗余
  → PatchouliService.finalize 取出 → InteractionPayload.write_focus / update_focus
  → librarian.submit_interaction → GenerationRequest.write_focus / update_focus
  → engine._process_mode_b/c            ← 真正终点
```

终点 `engine._process_mode_b/c` 实际消费的字段，恰好天然分成下游不相交的两半：

| 字段 | 终点用途 | 性质 |
| :--- | :--- | :--- |
| `content` / `reason` / `title`（WRITE） | mode b 提取 + fallback 草稿（[engine.py:170-184](../../src/hivememory/engines/generation/engine.py#L170)） | **Agent 提交参数** |
| `instruction` / `content` / `base_alias` / `base_uuid`（UPDATE） | mode c 合并（[engine.py:235-256](../../src/hivememory/engines/generation/engine.py#L235)） | **Agent 提交参数** |
| `intent_id` / `pending_alias` | `_dedup_and_persist` / `_apply_update` 回填 Settlement（[engine.py:181-184](../../src/hivememory/engines/generation/engine.py#L181)） | **生成关联键** |
| `identity` | `GenerationRequest.identity` 派生（[models.py:156](../../src/hivememory/engines/generation/models.py#L156)） | **关联元数据** |

参数一侧只进提取/合并，关联键一侧只进 Settlement 组装，两者下游完全不相交——却被打包在同一个 Focus 对象里穿过 6 层。

### 2.2 关联键是 PendingAtom 已持有数据的副本

`register_write` / `register_update`（[runtime.py:84-96](../../src/hivememory/alice/runtime/pending_atom/runtime.py#L84)）已经把 `pending_alias` / `intent_id` / `identity` / `focus` / `runtime_scope` 全部存进了 `PendingAtom`。Focus 穿透链是在**并行搬运写缓冲已经持有的同一份真相**——这与 PCB 洞同根：本轮写状态存在两套真相源。

### 2.3 ChatResult 三字段冗余 + 死字段

- `ChatResult.pending_aliases`：仓库级搜索**零真实读者**，是死字段。loop_executor 维护它、harvest 往里塞，但下游无人消费。
- `ChatResult.write_focus` / `update_focus`：finalize 真正要的，但传的是去规范化的 Focus 副本。
- loop_executor 为此维护 `write_foci` / `update_foci` / `pending_aliases` 三个累积器（[loop_executor.py:206-208](../../src/hivememory/alice/runtime/agent/loop_executor.py#L206)），并在 CALL 路径合并子帧结果（[L388](../../src/hivememory/alice/runtime/agent/loop_executor.py#L388)）。一个"驱动生成的引擎"却在维护它同事（PendingAtomRuntime）的结果。

---

## 3. 目标设计

### 3.1 Focus 瘦身为纯参数 DTO

`WriteFocus` / `UpdateFocus` 只保留 Koakuma 从 MTP 指令解析出的参数，不可变，无关联元数据：

```python
class WriteFocus(BaseModel):
    content: str
    reason: Optional[str] = None
    title: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class UpdateFocus(BaseModel):
    instruction: str
    content: Optional[str] = None
    base_alias: str
    base_uuid: str
    model_config = ConfigDict(frozen=True)
```

移除字段：`pending_alias` / `intent_id`（关联键，归 PendingAtom）、`identity`（关联元数据，随 Task 出境，见 §3.3）。Focus 名副其实退回"Koakuma → generation 的请求 DTO"。

### 3.2 PendingAtomMaterializeTask：PendingAtom 的不可变投影

引入物化请求，作为 `PendingAtom` 剥掉 Alice 私有可变生命周期字段（status / settlement / runtime_scope）后的不可变投影：

```python
class PendingAtomMaterializeTask(BaseModel):
    """跨子系统的不可变物化请求。Alice 组装出境，进入 finalize 后才允许 patchouli 解析。"""
    pending_alias: str
    intent_id: str
    source_verb: Literal["WRITE", "UPDATE"]
    identity: Identity
    focus: WriteFocus | UpdateFocus
    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_pending_atom(cls, pa: "PendingAtom") -> "PendingAtomMaterializeTask":
        return cls(
            pending_alias=pa.pending_alias,
            intent_id=pa.intent_id,
            source_verb=pa.source_verb,
            identity=pa.identity,
            focus=pa.focus,
        )
```

字段下游流向（明确且不相交）：

- `pending_alias` / `intent_id` / `source_verb` → patchouli 组装 `PendingAtomSettlement`、分发 mode b/c。
- `focus` → 仅供 `engine._process_mode_b/c` 的提取/合并。
- `identity` → `GenerationRequest.identity`。

### 3.3 Task ↔ Settlement 对偶

`Task` 与已存在的 `PendingAtomSettlement` 构成这道缝上的一对请求/应答对偶，都不可变、都以 `(pending_alias, intent_id)` 为键、都归 `core/models/pending.py`：

| 方向 | 对象 | 语义 |
| :--- | :--- | :--- |
| Alice → patchouli（出境） | `PendingAtomMaterializeTask` | "请把这个写意图物化" |
| patchouli → Alice（回流） | `PendingAtomSettlement` | "物化结果如此" |

这把整条主动写链路收束为一个干净的请求/应答对，也是以后所有"跨缝传写意图"的范式。

### 3.4 A2 组装：Alice 编排层投影，Koakuma 纯化

**Koakuma 纯化**：`_handle_write` / `_handle_update` 执行 MTP 后只返回工具执行结果（ACK + pending_alias），不再在 `MTPResponse` / `MTPExecutionResult` 上携带 `write_focus` / `update_focus`。Focus 在 `pending_runtime.register_write/update` 时封装进 PendingAtom 即完成使命，Koakuma 不再维护写数据。

**编排层组装（A2）**：run 结束时，由 PendingAtomRuntime 的所有者——Alice 编排层（解耦后的 `AgentOrchestrator`）——按 scope 从 PendingAtomRuntime 取出本 run 产生的 PendingAtom，投影成 `Task` 列表装入 `AgentRunResult`。

> A2 不是被否决的"方案 B"。方案 B 的问题是 patchouli（下游）在 finalize（后处理）阶段**反向跨总线**回 Alice 取数。A2 是 PendingAtomRuntime 的所有者在**组装阶段、同子系统内**读自己持有的真相做投影，产出不可变 Task 后正向出境——符合单一真相源初衷，不引入反向依赖。

**按 scope 过滤**：PendingAtom 的 `runtime_scope` 已带 `run_id`（主帧创建、`for_child` 向子帧传播）。需给 `PendingAtomRuntime` 增补 scope 过滤查询：

- `tasks_by_run(run_id)` → 父子帧全部写意图，供 `AgentRunResult` 组装。
- `tasks_by_frame(frame_id)` → 单帧写意图（如需）。

因此引擎不必把 alias 一路 harvest 穿过帧栈来组装 run 级结果。

> 边界澄清：这里的 scope 过滤组装与 [AgentLoopDecouplingDesign](AgentLoopDecouplingDesign.md) 中**子帧 IPC 回复用的 harvest 是两件正交的事**。后者（`frame.harvested_aliases` → `_assemble_ipc_return` 的 `[Artifacts]`）只为给主 Agent 看的回复文本服务，保留显式收割不变。本文的"明确下游流向"不变量只约束 `AgentRunResult` 的数据条目，与 IPC harvest 无关。

### 3.5 ChatResult 重组为 AgentRunResult

`ChatResult` 这个名字已不再贴切——一次完整 Agent 执行的产出不只是"对话结果"。重命名为 `AgentRunResult`，代表"上层一次 chat 调用（用户的一个请求）后系统运行至自然中断的完整产出"。

```python
class AgentRunResult(BaseModel):
    final_text: str                                    # → 用户可见回复 / InteractionPayload.assistant_final_text
    mtp_iterations: int                                # → 统计
    total_iterations: int                              # → 统计
    turn_events: list[TurnEvent]                       # → ActionReducer → TraceReducer → 感知层
    materialize_tasks: list[PendingAtomMaterializeTask]  # → finalize 启动 mode b/c + 组 Settlement
```

字段变化：

- 删 `pending_aliases`（死字段，无下游）。
- `write_focus` + `update_focus` 两字段 → 塌缩为单一 `materialize_tasks`（WRITE/UPDATE 由 `task.source_verb` 区分）。

**字段不变量**：`AgentRunResult` 的每个字段都必须由 Alice 子系统组装，且有完全明确的下游消费者。无下游的字段不得进入（这条不变量天然挡住 `pending_aliases` 这类死字段）。

### 3.6 数据流（重组后）

```text
Koakuma._handle_write/update → register_write/update（Focus 封装进 PendingAtom，Koakuma 到此为止）
AgentOrchestrator（run 结束）→ pending_runtime.tasks_by_run(run_id) → 投影 Task[] → AgentRunResult.materialize_tasks
PatchouliService.finalize → InteractionPayload.materialize_tasks
librarian.submit_interaction → 按 task.source_verb 分发 → GenerationRequest{intent_id, pending_alias, focus, identity}
engine._process_mode_b/c → focus 用于提取/合并，intent_id/pending_alias 用于组 Settlement
```

### 3.7 Task 链路强化不变量

在 §3.1–3.6 的基础上，对 Task 链路追加两条强约束，防止职责回流与空键穿透。

**不变量 1：identity 只经 Task 注入 GenerationRequest，禁止 generation 读 `focus.identity`**

Focus 瘦身后已无 `identity` 字段（§3.1），identity 改由 `Task.identity` 携带出境。`GenerationRequest.identity` 必须从 `task.identity` 注入，generation 层**不得再读取 `focus.identity`**（该字段已不存在，且 identity 属关联元数据，不属 MTP 解析参数）。

落点：

- `engines/generation/models.py::GenerationRequest.get_identity()`（[models.py:150-160](../../src/hivememory/engines/generation/models.py#L150)）当前优先读 `write_focus.identity` / `update_focus.identity` → 改为读 `request.identity`（由 task 注入），Mode A 被动路径仍可回退 `context.turns[0].identity`。
- 删除 `write_focus.identity` / `update_focus.identity` 的派生分支。

**不变量 2：Task 链路内 `intent_id` 必填（非 Optional）**

`PendingAtomMaterializeTask.intent_id` 与 `PendingAtom.intent_id` 收紧为 `str`（非 `Optional`）。依据：`register_write/update` 总是生成 `intent_id`（[runtime.py:81-123](../../src/hivememory/alice/runtime/pending_atom/runtime.py#L81)），从不为 None——这是把既成事实写进类型，避免 Task → Settlement 组装时空键导致回填错配。`PendingAtomSettlement.intent_id` 已是 `str`，三者一致。

**边界限定**：本不变量**仅约束 Task 链路**（PendingAtom / Task / Settlement）。`engines/generation` 内部 helper（`_dedup_and_persist` / `_apply_update`，[engine.py:285/345/431](../../src/hivememory/engines/generation/engine.py#L285)）的 `intent_id: Optional` **保持不变**——因为被动观察（Mode A）无 WRITE/UPDATE、无 intent，正当地传 `intent_id=None`。收紧不得波及 Mode A。

---

## 4. 改动清单

### 4.1 core 层

| 文件 | 改动 |
| :--- | :--- |
| `core/models/pending.py` | `WriteFocus` / `UpdateFocus` 移除 `pending_alias` / `intent_id` / `identity`，加 `frozen=True`；新增 `PendingAtomMaterializeTask` + `from_pending_atom`；`PendingAtom.intent_id` 与 `Task.intent_id` 收紧为 `str` 非 Optional（§3.7 不变量 2）；导出 |
| `core/protocol/models.py` | `ChatResult` → `AgentRunResult`；删 `pending_aliases`，`write_focus`+`update_focus` → `materialize_tasks`；`InteractionPayload` 同步：`write_focus`/`update_focus` → `materialize_tasks` |

### 4.2 alice 层

| 文件 | 改动 |
| :--- | :--- |
| `pending_atom/runtime.py` | `register_write/update` 不再往 focus 写关联键（关联键已是 PendingAtom 字段）；新增 `tasks_by_run(run_id)` / `tasks_by_frame(frame_id)`（投影为 Task），底层 `_PendingAtomStore` 加 run_id/frame_id 索引或线性过滤 |
| `koakuma.py` | `_handle_write/update` 不再在 `MTPResponse` 带 `write_focus`/`update_focus`；构造 Focus 时去掉关联键参数 |
| `core/mtp` 或 `MTPResponse`/`MTPExecutionResult` 模型 | 移除 `write_focus`/`update_focus`/（评估 `pending_alias` 是否仍需用于 ACK 文案） |
| `orchestrator.py`（解耦后） | run 结束按 `run_id` 取 Task 列表组装进 `AgentRunResult`；不再维护 `write_foci`/`update_foci`/`pending_aliases` |

### 4.3 patchouli / engines 层

| 文件 | 改动 |
| :--- | :--- |
| `patchouli/service.py::finalize_agent_run` | 从 `AgentRunResult.materialize_tasks` 取，写入 `InteractionPayload.materialize_tasks` |
| `patchouli/services/librarian.py::submit_interaction` | 分发改为读 `task.source_verb`（替代"哪个 focus 非空 + FlushReason"判断） |
| `engines/generation/models.py::GenerationRequest` | 新增独立标量 `intent_id` / `pending_alias`；`get_identity()` 改读 `request.identity`（由 task 注入），删除 `focus.identity` 派生分支（§3.7 不变量 1）；`write_focus`/`update_focus` 仍承载纯参数 Focus |
| `engines/generation/engine.py` | `focus.intent_id` → `request.intent_id`，`focus.pending_alias` → `request.pending_alias`；mode b/c 提取逻辑不变；内部 helper 的 `intent_id: Optional` 保持不变（Mode A 正当传 None，§3.7 边界限定） |

### 4.4 验证

- 全量 `tests/unit/`（Focus / generation / mtp 链路）+ e2e（`test_write_chain` / `test_update_chain` / `test_sub_agent_call_e2e`）。
- 不变量回归：WRITE/UPDATE 经 mode b/c 落库结果一致；Settlement 回填的 `pending_alias`/`intent_id` 一致；最终 `final_text`/`turn_events` 不变。
- §3.7 专项：generation 不再出现 `focus.identity` 读取；Task 链路 `intent_id` 全程非空；Mode A 被动路径仍可 `intent_id=None` 正常落库。

---

## 5. 落地时间与依赖

本期**必须在 [AgentLoopDecouplingDesign](AgentLoopDecouplingDesign.md) 之后**，原因：`write_foci`/`update_foci`/`pending_aliases` 的维护与 harvest 正是解耦要从引擎搬到 `AgentOrchestrator` 的逻辑。若先做本期，会在引擎里改一遍、解耦时再搬一遍。

推荐顺序：

1. **解耦先行**（解耦文档 Phase 1–3）：把"结果组装"职责落到 `AgentOrchestrator`。此时 `AgentRunResult` 的组装方已就位。
2. **本期紧随**：Focus 瘦身 + Task 投影 + `ChatResult`→`AgentRunResult` 重组。组装方此时是唯一所在，Task 投影自然落位。

本期与解耦的交叉点（解耦阶段需预留）：

- `AgentRunResult` 重命名应在解耦 Phase 3 的"ChatResult 产出方从引擎改为编排"一并落地，避免二次改名。
- 解耦阶段编排聚合结果时，先按现字段（write_focus 等）聚合；本期再替换为 `materialize_tasks`。

本期**不依赖**目录迁移（`agent_runtime/`），可在 `alice/runtime/` 现结构内完成。

