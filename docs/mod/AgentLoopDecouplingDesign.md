---
title: Legacy Agent Loop Decoupling Plan
status: superseded
owner: alice
scope: completed-agent-loop-and-orchestrator-decoupling
archived_at: 2026-07-28
superseded_by:
  - docs/alice/agent-runtime.md
  - docs/alice/orchestration.md
---

> 本文保留 AgentLoopExecutor 与 Orchestrator 解耦的实施理由和迁移轨迹，已停止维护。当前 frame 执行、CALL trap 与恢复流程以 [Agent Runtime](../alice/agent-runtime.md)和[多 Agent 编排](../alice/orchestration.md)为准。

# Agent 执行循环解耦落地规划

**文档状态**: Draft (草案)
**适用范围**: `alice/runtime/agent/loop_executor.py`、`alice/runtime/agent/runtime.py`、`alice/runtime/agent/frame_scheduler.py`、`alice/runtime/agent/profile_resolver.py`、拟新建的编排驱动器、`core/protocol/models.py` 中的 `ChatResult`
**核心目标**: 把 `AgentLoopExecutor` 彻底打造成单 Agent 执行循环总控，将所有多智能体编排逻辑（CALL 派生、子帧调度、别名收割、IPC 组装、context_refs 继承、流式子帧合并）移出到 alice 侧的编排驱动器。本期是纯解耦重构，外部可观察行为零变化，且不迁移物理目录。

---

## 1. 文档目标

本文是 [AgentRuntimeBoundaryDesign](AgentRuntimeBoundaryDesign.md) 的下一步落地。边界文档已裁定：执行引擎与编排是两个层，三个 Runtime 属引擎，FrameScheduler / CALL 派发 / IPC 组装 / 别名收割属编排；并明确"逻辑先、目录后"——先在现目录解开缝，再谈迁移。

本期就是那个"逻辑先"：

- 把 `AgentLoopExecutor.execute_frame` 收敛为纯引擎循环：generate → MTP → 回填，遇 CALL 即挂起返回，**绝不自我编排**。
- 把编排逻辑上移到一个新的编排驱动器，承接 `run_agent` / `run_agent_stream`，由它驱动引擎。
- 完成边界文档 §5 的接口契约：引擎 `run_frame(frame) -> FrameExecutionResult`（收敛或挂起），编排 `run_agent(...)` 驱动引擎。

本期**不做**：

- 不新建 `agent_runtime/` 目录，不迁移任何文件（编排驱动器暂放 `alice/runtime/`）。
- 不改 MTP 协议、不改 PendingAtom 链路、不改 prompt 组装。
- 不改变任何外部可观察行为（最终文本、TurnEvent 序列、SSE 事件序列与 scope 区分必须逐字节一致）。

---

## 2. 现状调研

### 2.1 五个依赖的层归属

`AgentLoopExecutor.__init__` 当前注入五个依赖，按边界文档判定规则归层：

| 依赖 | 引擎需要 | 当前用途 | 归属 |
| :--- | :---: | :--- | :--- |
| `worker_agent` | ✅ | LLM 取指（generate / generate_stream） | 引擎 |
| `mtp_executor` | ✅ | MTP 执行（陷入记忆域） | 引擎 |
| `alias_resolver` | ◐ | 仅 `_fetch_context_refs_content` 用（[L681](../../src/hivememory/alice/runtime/agent/loop_executor.py#L681)） | 引擎对象，被编排方法借用 |
| `frame_scheduler` | ❌ | create_main_frame / suspend / fork_sub / resume | **编排** |
| `agent_profile_resolver` | ❌ | 仅 `_execute_call` 里 resolve 子 agent（[L586](../../src/hivememory/alice/runtime/agent/loop_executor.py#L586)） | **编排** |

五个依赖里两个纯编排、一个被编排方法借用——这量化了当前的缠绕程度。解耦后引擎应只剩 `worker_agent` + `mtp_executor`。

### 2.2 缠在 `execute_frame` 里的编排逻辑（病灶清单）

`execute_frame` 主循环（generate → MTP → 回填 → TurnEvent，[L212-453](../../src/hivememory/alice/runtime/agent/loop_executor.py#L212)）是干净的引擎职责。污染集中在 **SUSPEND 分支**（[L357-414](../../src/hivememory/alice/runtime/agent/loop_executor.py#L357)）及其调用的私有方法：

| # | 位置 | 内容 | 归属 |
| :-- | :--- | :--- | :--- |
| 1 | `_execute_call`（[L537-665](../../src/hivememory/alice/runtime/agent/loop_executor.py#L537)） | suspend → resolve profile → fetch context_refs → fork_sub_frame → **递归 `execute_frame(sub)`** → resume → 组 IPC | 编排（核心违规） |
| 2 | `_assemble_ipc_return`（[L710](../../src/hivememory/alice/runtime/agent/loop_executor.py#L710)） | 拼 `[Sub-Agent Reply]` + `[Artifacts]` XML | 编排 |
| 3 | `_fetch_context_refs_content`（[L667](../../src/hivememory/alice/runtime/agent/loop_executor.py#L667)） | context_refs 跨帧上下文继承 | 编排 |
| 4 | `_try_harvest_alias`（[L743](../../src/hivememory/alice/runtime/agent/loop_executor.py#L743)）+ 主循环调用（[L455](../../src/hivememory/alice/runtime/agent/loop_executor.py#L455)） | 子帧别名收割 | 编排 |
| 5 | `create_main_frame` 调用（[L116](../../src/hivememory/alice/runtime/agent/loop_executor.py#L116)、[L158](../../src/hivememory/alice/runtime/agent/loop_executor.py#L158)） | 造帧 | 编排 |
| 6 | `_sub_emit` 子帧流式合并（[L606](../../src/hivememory/alice/runtime/agent/loop_executor.py#L606)） | 把子帧事件并进父流 + `sub_agent_start/end` | 编排 |

### 2.3 关键约束：引擎与编排的递归是交错的

最棘手的一点：`_execute_call` 在编排中途**回调 `execute_frame(sub_frame)`**。调用栈实际是 `execute_frame(main) → 编排逻辑 → execute_frame(sub)` 交替下降。

因此解耦**不能简单把 SUSPEND 分支整块搬走**，必须把控制权**真正交还**给编排：引擎遇到 SUSPEND 就**返回**（携带挂起信息与已累积状态），由编排 fork 子帧、再调引擎跑子帧、再把 IPC 回填后**重入引擎续跑**。这是从"引擎递归调用编排"反转为"编排驱动引擎"——本期的支点。

### 2.4 跨缝契约

- **输入货币** `ExecutionFrame`：编排造帧、引擎消费。已定义在 [models.py](../../src/hivememory/alice/runtime/models.py)，无需改动。
- **输出货币** `ChatResult`：当前混装引擎产物（`final_text` / `turn_events` / `mtp_iterations` / `total_iterations`）与编排产物（`write_focus` / `update_focus` / `pending_aliases`，在 [L388](../../src/hivememory/alice/runtime/agent/loop_executor.py#L388) 被 `sub_result` 累加合并）。本期引入引擎级返回类型 `FrameExecutionResult` 表达"收敛 or 挂起"，`ChatResult` 保留为**编排级**对外返回，由编排从 `FrameExecutionResult` 聚合产出。

### 2.5 调用方与测试现状

- 调用方仅 [runtime.py:65/83](../../src/hivememory/alice/runtime/agent/runtime.py#L65)（`AgentRuntime.run_agent` / `run_agent_stream`），收口干净。
- [test_loop_executor_turn_events.py](../../tests/unit/patchouli/kernel/test_loop_executor_turn_events.py)：8 个用例直接调 `execute_frame` 验证 TurnEvent 序列——纯引擎测试，解耦后基本可留（仅适配新返回类型的取值方式）。
- [test_loop_executor_stream.py](../../tests/unit/patchouli/kernel/test_loop_executor_stream.py)：2 个用例走 CALL/子帧流式路径并 mock `frame_scheduler`（[L158-162](../../tests/unit/patchouli/kernel/test_loop_executor_stream.py#L158)）——编排测试，解耦后迁移到编排驱动器测试。
- e2e：[test_sub_agent_call_e2e.py](../../tests/e2e/pipeline/test_sub_agent_call_e2e.py)、[test_kernel_loop_e2e.py](../../tests/e2e/pipeline/test_kernel_loop_e2e.py) 是行为不变的回归基线。

---

## 3. 目标形态

### 3.1 引擎级返回类型 `FrameExecutionResult`

与输入货币 `ExecutionFrame` 对应，引擎单次执行返回 `FrameExecutionResult`。采用**单类型 + status 判别字段**形态（比 Union 更直观，单一命名）：

```python
class FrameExecutionStatus(str, Enum):
    COMPLETED = "completed"   # 自然收敛
    SUSPENDED = "suspended"   # 命中 CALL，等待编排派生子 agent

class FrameExecutionResult(BaseModel):
    """引擎单次执行的 trap/return 信号。

    它不再承载本帧累积产物——那些已下沉到 frame.progress（见 §3.1bis）。
    这里只表达"为什么停下来"以及挂起时编排需要的最小信息。
    """
    status: FrameExecutionStatus

    # ---- status == SUSPENDED 时填充 ----
    call_request: Optional[CallRequest] = None   # {target_alias, task, context_refs}
    suspend_assistant_text: Optional[str] = None # 触发 CALL 的 result.text（编排负责 append + ⟫）
    suspend_action_id: Optional[str] = None      # 供编排回填 tool_result 的 action_id
```

引擎语义：`run_frame(frame)` 读写传入的 `frame`，跑到自然收敛返回 `COMPLETED`，命中 CALL 返回 `SUSPENDED` 并把控制权交还编排，**自己不 fork、不 resume、不组 IPC**。两种 status 下本帧已产生的累积产物都已经写在 `frame.progress` 上（见 §3.1bis），不经由 `FrameExecutionResult` 传递。`ChatResult` 不再由引擎产出，改由编排在 `COMPLETED` 时从 `frame.progress` 读取聚合。

### 3.1bis ExecutionFrame 作为 PCB：执行状态重入模型

**这是反转能成立的前提，否则 CALL 后主 Agent 无法续跑。**

单体 `execute_frame` 把整轮累积状态放在函数局部变量里（`text_segments` / `turn_events` / `write_foci` / `update_foci` / `pending_aliases` / `iteration` / `_seq`）。CALL 走 `continue` 时局部变量不丢、循环继续，所以能工作。反转后引擎 `return`，局部变量全部蒸发——若不增补，会同时出现三个 bug：

| Bug | 现象 | 类别 |
| :--- | :--- | :--- |
| 续跑断裂 | 引擎在产出 IPC 前就 return，suspend 路径的两条 `working_history.append`（CALL 文本 + IPC 回填）无人执行，重入时 LLM 看不到 CALL 与子 Agent 回复 → 重发 CALL 死循环或幻觉 | correctness |
| 输出残缺 | 重入是全新一次执行，累积器从空开始，最终 `ChatResult` 只剩 CALL 之后那段 | completeness |
| 编号/预算断裂 | `iteration` 重置使主帧迭代预算翻倍；`_seq` / `action_id` 重置使 TurnEvent 序号冲突 | 破坏"逐字节不变" |

**解法：把 `ExecutionFrame` 定位为进程控制块（PCB）。** CALL = 陷入，引擎把 PCB 交还调度器（编排），编排处理完把**同一个 PCB** 交回引擎续跑；PCB 持有完整可恢复执行状态——这正是 OS 上下文切换的本质，与 [AgentRuntimeBoundaryDesign](AgentRuntimeBoundaryDesign.md) 的 OS 映射一致。

frame 上的状态分两类：

| 类别 | 字段 | 载体 | 说明 |
| :--- | :--- | :--- | :--- |
| 续跑输入 | `working_history` | frame（已有） | LLM 下一轮必须看到；天然持久 |
| 输出累积 | `text_segments` / `turn_events` / `write_foci` / `update_foci` / `pending_aliases` / `iteration` / `sequence` | **新增 `frame.progress`** | 从局部变量下沉；`harvested_aliases` 逻辑归此 |

引擎每段执行**读取并追加**到 `frame.progress`，而非局部变量。重入同一个 frame 自然续接：`iteration` / `sequence` / `action_id` 持久在 PCB 上 → **预算与编号天然连续，行为逐字节不变**。

为何选 PCB，而非"把 `FrameExecutionResult` 当入参喂回引擎 reseed"：

| 维度 | result 当入参喂回 | **frame 作 PCB（采用）** |
| :--- | :--- | :--- |
| 续跑 + 完整性 | 分散在 result↔engine 往返 | 统一在一个载体上 |
| 引擎是否"记得"被挂起过 | 是（编排概念泄漏进引擎） | 否（引擎只读写被给的 PCB，程序计数器本就该在 PCB 上） |
| 编号/预算连续 | 需额外 merge / renumber | 持久在 PCB 上，自动连续 |
| 契约对称性 | result 既是出参又是入参 | frame 进、result 出，职责清晰 |

PCB 方案把续跑、完整性、编号连续三个问题收敛到单一载体，与边界文档"引擎与编排通过共享可变状态紧耦合"哲学一致（`PendingAtomRuntime` 已是同一道缝上的共享可变状态）。

### 3.2 编排驱动器

暂名 `AgentOrchestrator`，本期放 `alice/runtime/`（后续随 `agent_runtime/` 迁移再移动），承接原 `AgentRuntime` 的 `run_agent` / `run_agent_stream`。它持有 `loop_executor` + `frame_scheduler` + `agent_profile_resolver` + `alias_resolver`，负责：

1. 造主帧（`create_main_frame`，frame 自带空 `progress`）。
2. 循环驱动引擎 `run_frame(main_frame)`：
   - 收到 `COMPLETED` → 从 `main_frame.progress` 聚合为 `ChatResult` 返回。
   - 收到 `SUSPENDED` → 按下列**重入序列**处理后 `continue`，重入**同一** `main_frame`：
     1. `main_frame.working_history.append(assistant: suspend_assistant_text + "⟫")`
     2. fork 子帧（独立 PCB）→ 递归驱动引擎跑子帧 → resume
     3. 从子帧 `progress` harvest 别名，选择性合并进 `main_frame.progress`（对应单体版 [L388](../../src/hivememory/alice/runtime/agent/loop_executor.py#L388)）
     4. 组 IPC → `main_frame.working_history.append(user: IPC)` + 追加对应 `tool_result` TurnEvent 到 `main_frame.progress`
3. 流式模式下负责把子帧事件并入父流、发 `sub_agent_start/end`。

> 关键：第 2.i / 2.iv 步的两条 `working_history.append` 正是单体版 [L394-400](../../src/hivememory/alice/runtime/agent/loop_executor.py#L394) 的逻辑，只是 append 主体从引擎搬到编排（因为 IPC 由编排产出）。这两条 append 是"续跑不断裂"的根本保证。

### 3.3 解耦后依赖对比

```text
现在:
  loop_executor → {worker_agent, mtp_executor, alias_resolver, frame_scheduler, profile_resolver}

之后:
  loop_executor → {worker_agent, mtp_executor}                                  # 纯引擎
  orchestrator  → {loop_executor, frame_scheduler, profile_resolver, alias_resolver}  # 编排
  AgentRuntime  → {orchestrator}                                                # 装配门面
```

---

## 4. Phase 改动清单

每个 Phase 自成一个可验证的提交，结束时全量回归测试通过、外部行为零变化。

### Phase 0 — 新增 `FrameExecutionResult` 类型 + `frame.progress`（纯新增，零行为变化）

| 文件 | 改动 |
| :--- | :--- |
| `alice/runtime/models.py` | 新增 `FrameExecutionStatus` 枚举、`CallRequest`、`FrameExecutionResult`（信号类，见 §3.1）；新增 `ExecutionProgress` 数据类（`text_segments` / `turn_events` / `write_foci` / `update_foci` / `pending_aliases` / `iteration` / `sequence`）；在 `ExecutionFrame` 上加 `progress: ExecutionProgress` 字段（`harvested_aliases` 逻辑并入或保留并存）；导出到 `__all__` |

- 不改任何调用路径，不接线。仅让类型与 PCB 字段先就位，为后续反转铺路。
- 验证：`import` 通过 + 现有测试全绿（应无影响）。

### Phase 1 — CALL 控制反转 + PCB 重入（核心，最高风险）

**引擎侧 `loop_executor.py`**

| 改动 | 说明 |
| :--- | :--- |
| 累积器下沉（[L204-209](../../src/hivememory/alice/runtime/agent/loop_executor.py#L204)） | 删除 `text_segments` / `turn_events` / `write_foci` / `update_foci` / `pending_aliases` / `iteration` / `_seq` 局部变量，全部改读写 `frame.progress.*`（PCB，见 §3.1bis），使重入续接、编号连续 |
| `execute_frame` SUSPEND 分支（[L357-414](../../src/hivememory/alice/runtime/agent/loop_executor.py#L357)） | 删除对 `_execute_call` 的调用与其后两条 `working_history.append` 及 IPC 回填；改为：把本段产物写入 `frame.progress` 后 `return FrameExecutionResult(status=SUSPENDED, call_request=..., suspend_assistant_text=result.text, suspend_action_id=action_id)`。**注意：suspend 时引擎不再 append 任何 working_history**（避免悬空 assistant 轮），append 由编排负责 |
| `execute_frame` 收敛出口（[L458-469](../../src/hivememory/alice/runtime/agent/loop_executor.py#L458)） | 返回 `FrameExecutionResult(status=COMPLETED)`；累积产物已在 `frame.progress`，不再构造 `ChatResult` |
| 删除 `_execute_call` / `_assemble_ipc_return` / `_fetch_context_refs_content` / `_try_harvest_alias` | 迁入编排驱动器 |
| `__init__` | 移除 `frame_scheduler`、`agent_profile_resolver` 参数（`alias_resolver` 暂留，Phase 3 处理） |
| 移除 `create_main_frame` 调用（[L116](../../src/hivememory/alice/runtime/agent/loop_executor.py#L116)、[L158](../../src/hivememory/alice/runtime/agent/loop_executor.py#L158)） | `execute_main_frame` / `execute_main_frame_stream` 整体迁入编排（造帧是编排职责）；引擎只保留 `execute_frame` / `execute_frame_stream` 接受现成 frame |

**编排侧（新建 `alice/runtime/orchestrator.py`）**

| 改动 | 说明 |
| :--- | :--- |
| 新建 `AgentOrchestrator` | 持有 loop_executor / frame_scheduler / profile_resolver / alias_resolver |
| `run_agent` | 造主帧（带空 `progress`）→ 循环调 `loop_executor.execute_frame(main_frame)`；`SUSPENDED` 时执行 §3.2 **重入序列**（append CALL 文本 → fork / 跑子帧 / resume / harvest 合并进主帧 progress / 组 IPC → append IPC + tool_result）→ `continue` 重入同一 main_frame；`COMPLETED` 时从 `main_frame.progress` 聚合 `ChatResult` |
| 迁入 `_execute_call` / `_assemble_ipc_return` / `_fetch_context_refs_content` / `_try_harvest_alias` | 子帧执行内部仍调 `loop_executor.execute_frame(sub_frame)`；子帧是独立 PCB，从其 `progress` 收割后合并进主帧 progress |

**装配侧 `runtime.py`**

| 改动 | 说明 |
| :--- | :--- |
| `AgentRuntime.__init__` | 构造 `AgentOrchestrator`，注入 loop_executor + frame_scheduler + profile_resolver + alias_resolver |
| `run_agent` / `run_agent_stream` | 委托给 `orchestrator` 而非 `loop_executor` |

- 验证：`test_loop_executor_turn_events.py`（适配返回类型 + 从 `frame.progress` 取累积值）+ `test_sub_agent_call_e2e.py` 全绿，CALL 链路行为不变。**重点回归 CALL 后主 Agent 续跑**：working_history 含 CALL 文本 + IPC，最终 ChatResult 含 CALL 前后全部 text/turn_events，iteration 预算不翻倍。

### Phase 2 — 流式反转（与 Phase 1 同构，单独提交降风险）

当前 `execute_frame_stream` 通过 `_runner` 在内部跑 `execute_frame`，并在 SUSPEND 分支用 `_sub_emit` 把子帧事件并进父队列（[L606](../../src/hivememory/alice/runtime/agent/loop_executor.py#L606)）。反转后职责切分：

| 侧 | 改动 |
| :--- | :--- |
| 引擎 `execute_frame_stream` | 只发**本帧**的 `token` / `mtp_start` / `mtp_result` 事件；遇 SUSPEND 时发出 `mtp_result(status=suspend)` 后**结束本段流**并通过返回值/哨兵交还 `FrameExecutionResult(SUSPENDED)`，不再内联跑子帧 |
| 编排 `run_agent_stream` | 消费引擎流；收到挂起后发 `sub_agent_start` → 驱动引擎跑子帧流（子帧事件透传，靠 `data.scope=sub` 区分）→ `sub_agent_end` → 回填 IPC → 重入引擎流续跑 |
| 引擎 `_namespace_for_frame`（[L78](../../src/hivememory/alice/runtime/agent/loop_executor.py#L78)） | 保留在引擎（每帧自带 scope/depth/frame_id 元数据），编排不需要改写事件名 |

- 难点：保持 SSE 事件序列、`scope` 区分、`sub_agent_start/end` 时机与现状完全一致。
- 验证：`test_loop_executor_stream.py` 两个用例迁移到编排测试后全绿，断言 `event_types` / `scope` / suspend 事件不变。

### Phase 3 — 依赖收敛与清理

| 文件 | 改动 |
| :--- | :--- |
| `loop_executor.py` | `_fetch_context_refs_content` 已随 Phase 1 迁出，其 `alias_resolver` 依赖一并移除 → 引擎依赖降至 `worker_agent` + `mtp_executor` 两个纯引擎依赖 |
| `loop_executor.py` 文件头 docstring / 类名语义 | 更新为"单 Agent 执行循环总控"，移除多智能体相关描述（Phase A→B→C→D 注释保留循环语义，删去 CALL/子帧派生措辞） |
| 测试归位 | 引擎测试（TurnEvent / 收敛 / MTP 回填）留 `test_loop_executor_*`；编排测试（CALL / 子帧 / 流式合并 / harvest / IPC）新建 `test_agent_orchestrator_*` |
| `core/protocol/models.py::ChatResult` | 字段不变（仍是编排级对外契约），仅确认产出方从 loop_executor 改为 orchestrator。**重命名 `ChatResult`→`AgentRunResult` 与字段重组（`materialize_tasks`）留给后续 [PendingAtomMaterializeTaskDesign](../archive/legacy-docs/agent_runtime/pending_atom/PendingAtomMaterializeTaskDesign.md)，本期不改名以收敛风险** |

- 验证：全量 `tests/unit/patchouli/kernel/` + `tests/unit/system/` + 相关 e2e 全绿。
- 与后续衔接：本期编排聚合结果仍按现字段（`write_focus` / `update_focus` / `pending_aliases`）进行；这些累积器一旦落到 `AgentOrchestrator`，即为 [PendingAtomMaterializeTaskDesign](../archive/legacy-docs/agent_runtime/pending_atom/PendingAtomMaterializeTaskDesign.md) 的 A2 组装（run_id 投影 Task）就位铺路。

---

## 5. 验证策略

本期是**纯重构**，外部可观察行为必须逐字节不变。每个 Phase 结束执行：

```bash
pytest tests/unit/patchouli/kernel/ tests/unit/system/ -n auto
pytest tests/e2e/pipeline/test_sub_agent_call_e2e.py tests/e2e/pipeline/test_kernel_loop_e2e.py tests/e2e/pipeline/test_active_mode_e2e.py -n auto
```

不变量断言（回归基线）：

- `ChatResult.final_text` / `turn_events` 序列与重构前一致。
- SSE 事件类型序列、`scope` 区分、`sub_agent_start/end` 时机一致。
- CALL 链路：子 agent 别名收割、IPC payload（`[Sub-Agent Reply]` / `[Artifacts]`）格式一致。
- `write_focus` / `update_focus` / `pending_aliases` 聚合结果一致（含子帧合并）。

---

## 6. 范围边界

本期严格限定为"逻辑解耦"，以下明确**不在范围内**，留待后续：

- 不新建 `agent_runtime/` 目录、不迁移任何文件（见 [AgentRuntimeBoundaryDesign](AgentRuntimeBoundaryDesign.md) §6 的"逻辑先、目录后"）。编排驱动器暂放 `alice/runtime/`。
- 不动 `frame_scheduler` / `profile_resolver` 的内部实现（仅改变持有者）。
- 不动 PendingAtom 链路、`_on_pending_atom_settled` 等 reconcile 逻辑（属边界文档 §6.1 第 2 步引擎聚合根工作）。
- 不引入引擎聚合根 `engine.py`（边界文档 §4.4 目标结构的一部分，属目录迁移阶段）。
- 不做 Focus 瘦身、`ChatResult`→`AgentRunResult` 重命名、`materialize_tasks` 重组（属后续 [PendingAtomMaterializeTaskDesign](../archive/legacy-docs/agent_runtime/pending_atom/PendingAtomMaterializeTaskDesign.md)，须在本期之后；本期仅把结果组装职责落到 `AgentOrchestrator` 为其铺路）。

完成本期后，`loop_executor` 即成为纯单 Agent 执行循环总控，编排逻辑全部归位 `AgentOrchestrator`，为后续目录迁移扫清依赖障碍。

