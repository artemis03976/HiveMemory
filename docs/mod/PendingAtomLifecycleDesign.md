# PendingAtom 生命周期与句柄回收设计

**文档状态**: Draft（草案）  
**适用范围**: `agent_runtime/pending_atom/runtime.py`、`agent_runtime/pending_atom/store.py`、`core/models/pending.py`、`alice/runtime/core.py`、`engines/memory_compiler/handlers/pending_atom.py`  
**核心目标**: 打通 L0 级 PendingAtom 从创建到落库后的完整生命周期，实现句柄的安全回收与内存释放，并在 MemoryCompiler 层提供状态感知的引导提示，消除 Agent 对 pending_alias 的持续依赖。

---

## 1. 背景

PendingAtom 是 Agent 发出 WRITE/UPDATE 意图后，在记忆落库完成前的临时占位结构。其 `pending_alias` 作为临时句柄，允许 Agent 在当前话题内持续引用尚未落库的记忆内容。

现状存在两个问题：

1. **句柄不失效**：SETTLED 后，pending_alias 的重导向路径永久存在，无任何回收机制，导致 `PendingAtomRuntime` 内存随运行时间无限累积。
2. **Agent 无感知**：settle 事件到来后，Agent 不知道 pending_alias 对应的正式记忆别名，只会继续使用临时名。即使 Resolver 能透明重导向，Agent 的 working history 里也永远是临时名，无法主动切换到正式名。

---

## 2. 完整数据流（含子 Agent）

```
用户指令
  └─ AgentOrchestrator 创建 ExecutionFrame(run_id=R, frame_id=F0, depth=0)
       └─ KoakumaRuntime MTP 循环
            ├─ WRITE → PA1(run_id=R, status=PENDING)
            ├─ CALL → fork_sub_frame(run_id=R, frame_id=F1, depth=1)  ← 共享 run_id
            │    └─ 子 Agent MTP 循环
            │         └─ WRITE → PA2(run_id=R, status=PENDING)
            │    └─ F1 销毁，IPC return 返回主 Agent
            └─ 主 Agent 自然结束
  └─ _assemble_agent_run_result()
       └─ collect_tasks_by_run(run_id=R)
            └─ 筛选 run_id=R && status=PENDING → [PA1, PA2]  ← 子 Agent 一并收集
            └─ claim_for_materialization → PA1, PA2: PENDING → MATERIALIZING
  └─ F0 销毁

PatchouliService.finalize_agent_run(tasks)
  └─ LibrarianCore.run_active_generation → Mode B/C
  └─ 发布 PENDING_ATOM_SETTLED(settlement) / PENDING_ATOM_FAILED(pending_alias)

AliceRuntime（异步，时间不定）
  └─ _on_pending_atom_settled → settle() → PA: MATERIALIZING → SETTLED，canonical 索引建立
  └─ _on_pending_atom_failed  → mark_failed() → PA: MATERIALIZING → FAILED
```

**关键性质**：
- 父帧与子帧共享 `run_id`，子 Agent 产生的 PendingAtom 在主 run 结算时一并收集，无需额外合并。
- settle 事件异步到来，与 run 结算时序无关；结算只关心当前状态，不等待 settle。

---

## 3. 状态机重设计

### 3.1 现状问题

`SETTLED / FAILED / CANCELLED` 当前均为终态（`_TRANSITIONS[x] = frozenset()`），导致无法从这些状态迁移到任何清理状态，既无法标记句柄失效，也无法触发回收。

### 3.2 目标：EXPIRED 为唯一终态

```
PENDING ──(run结算)──────────────→ MATERIALIZING
        └──(超时/取消)──→ EXPIRED ←─────────────────────┐
                                                          │
MATERIALIZING ──(settle成功)──→ SETTLED ──(x轮run后)──→ EXPIRED
              └──(settle失败)──→ FAILED  ──(x轮run后)──┘
              └──(取消)────────→ CANCELLED ─────────────┘
```

**新 `_TRANSITIONS`**：

```python
_TRANSITIONS = {
    PENDING:       {MATERIALIZING, EXPIRED, CANCELLED},
    MATERIALIZING: {SETTLED, FAILED, CANCELLED},
    SETTLED:       {EXPIRED},   # 解禁：句柄回收过期
    FAILED:        {EXPIRED},   # 解禁：错误清理
    CANCELLED:     {EXPIRED},   # 解禁：取消清理
    EXPIRED:       {},          # 唯一终态
}
```

**语义说明**：
- `PENDING → EXPIRED`：超时或系统取消，atom 从未进入物化流程，无 canonical 记录。
- `SETTLED → EXPIRED`：物化成功，正式记忆已落库，句柄到达安全回收窗口。
- `FAILED → EXPIRED`：物化失败，句柄已无意义，等待 Agent 知晓后清理。
- `CANCELLED → EXPIRED`：系统级中断，清理残留句柄。

CANCELLED 和 FAILED 的 `→ EXPIRED` 转移可在后续迭代中实现（含重试机制），当前优先落地 `SETTLED → EXPIRED` 主路径。

---

## 4. 回收时序

### 4.1 时序约束

```
Run N:   PA 创建 → run 结算 → MATERIALIZING
（异步）  settle 到来 → SETTLED，canonical 索引建立
Run N+1: Agent working history 里仍可能持有 pending_alias
Run N+1 结算完成后：PA 安全迁移至 EXPIRED，下次结算时删除
```

**为什么不在 settle 时立刻删除**：只要当前 topic 未结束，Agent 的 working history 就持有 pending_alias。settle 时上一个 ExecutionFrame 已销毁，但下一次 run 的 Agent 仍可能 READ 该 alias，因此句柄必须保留到至少一次 run 结算之后。

**为什么 x=1 足够**：settle 总在 run N 结束后异步到来；run N+1 结算时，Agent 在 run N+1 内已通过 SETTLED 提示（见 §5）得知正式名，此时将 SETTLED 迁移到 EXPIRED 不会造成困惑。

### 4.2 run 结算的双重职责

`collect_tasks_by_run(run_id)` 在完成本轮 PENDING → MATERIALIZING 收集后，顺序执行两步清理：

**步骤一（迁移）**：将上轮的 SETTLED atom 迁移至 EXPIRED。

条件：`status == SETTLED && runtime_scope.run_id != current_run_id`

**步骤二（删除）**：将已处于 EXPIRED 状态的 atom 从 store 中删除。

条件：`status == EXPIRED`（含本轮步骤一刚迁移的，以及更早轮次遗留的）

两步合并伪代码：

```python
def evict_by_run(self, current_run_id: str) -> None:
    for atom in self._store.all_atoms():
        if atom.status == SETTLED and atom.runtime_scope.run_id != current_run_id:
            self._set_status(atom, EXPIRED)
    for atom in self._store.all_atoms():
        if atom.status == EXPIRED:
            self._store.delete(atom.pending_alias)
```

---

## 5. MemoryCompiler 状态感知策略

| 状态 | 编译策略 |
|------|---------|
| PENDING / MATERIALIZING | 现有草稿内容，正常展示 |
| SETTLED | 内容正常展示 + 追加提示："此内容已落库，临时名 `<pending_alias>` 即将失效，后续请使用正式名 `<canonical_alias>` 访问" |
| FAILED | 不展示内容，显示错误信息："此记忆生成失败，临时名 `<pending_alias>` 已失效，可重新发起写入请求" |
| CANCELLED | 不展示内容，显示提示："此记忆请求已取消" |
| EXPIRED | 不展示内容，Resolver 返回 `kind="expired"`，显示句柄失效错误 |

SETTLED 的提示使 Agent 能在 working history 中感知到正式名，从而在后续轮次主动切换，降低对 pending_alias 的持续依赖。

---

## 6. Resolver 变更

现有 `_resolve_pending_hit` 对 EXPIRED/CANCELLED 返回 `kind="not_found"`，需新增 `kind="expired"` 以区分"从未存在"与"句柄已回收"两种语义，便于 MemoryCompiler 给出更精确的错误提示。

---

## 7. 改动范围汇总

| 文件 | 改动内容 |
|------|---------|
| `core/models/pending.py` | `_TRANSITIONS`：SETTLED/FAILED/CANCELLED 加 EXPIRED 出路 |
| `agent_runtime/pending_atom/store.py` | 新增 `delete(pending_alias)` 方法，清理三个索引 |
| `agent_runtime/pending_atom/runtime.py` | 新增 `evict_by_run(current_run_id)` |
| `alice/runtime/agent/runtime.py` | `collect_tasks_by_run` 完成后调用 `evict_by_run` |
| `engines/memory_compiler/handlers/pending_atom.py` | 补充 SETTLED / FAILED / CANCELLED / EXPIRED 四种渲染分支 |
| `agent_runtime/resolver.py` | EXPIRED 分支返回 `kind="expired"` 而非 `kind="not_found"` |

---

## 8. 暂不实现

- FAILED → EXPIRED 的自动迁移（含重试机制设计）
- CANCELLED 的触发逻辑（run 取消、系统级中断）
- x > 1 的多轮延迟回收（x=1 当前足够）