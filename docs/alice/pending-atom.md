---
title: Alice PendingAtom Runtime
status: current
owner: alice
scope: runtime-write-buffer-and-materialization-handoff
code_paths:
  - src/hivememory/agent_runtime/pending_atom/
  - src/hivememory/agent_runtime/aliases/resolver.py
  - src/hivememory/core/models/pending.py
  - src/hivememory/alice/runtime/core.py
  - src/hivememory/agent_runtime/runtime.py
  - src/hivememory/alice/orchestration/run_executor.py
  - src/hivememory/alice/orchestration/call_coordinator.py
related_contracts:
  - docs/contracts/mtp.md
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/patchouli/generation.md
last_reviewed: 2026-08-03
---

# PendingAtom：运行时写缓冲与物化交接

PendingAtom 解决的是 Agent 写入记忆时一道容易被忽略的时间缝隙：MTP `WRITE` / `UPDATE` 已经被 Alice 接收，但 Patchouli 还没有完成提取、去重、合并与持久化。在这段时间里，如果系统只返回一句“已收到”，Agent 无法在本轮继续检查或引用刚刚提出的内容；如果 Alice 直接创建正式 MemoryAtom，又会把尚未完成的运行、随后被修正的判断，甚至已经取消的任务提前写成长久事实。

当前设计把 PendingAtom 当作 Agent Runtime 的 **store buffer**。它保存“本轮希望写入或修订什么”的临时意图，立即给出可读的 pending alias；run 成功收敛后，Alice 再把仍然有效的意图投影为物化任务交给 Patchouli。Patchouli 是正式记忆事实的所有者，Alice 只维护异步交接期间的运行时视图。

因此，`ack` 的准确含义始终是“写入意图已登记”，而不是“记忆已落库”。这条区分既是用户可见语义，也是防止半完成运行污染长期记忆的事务边界。

## 1. PendingAtom 不是什么

PendingAtom 不是简化版 MemoryAtom，也不是尚未获得 UUID 的正式记录。二者分别属于不同阶段：

| 对象 | 所有者 | 表达的事实 | 生命周期 |
|:---|:---|:---|:---|
| `PendingAtom` | Alice Runtime | Agent 在一次 run 中提出的写入或修订意图 | 进程内、短暂、可失败或取消 |
| `PendingAtomMaterializeTask` | Alice -> Patchouli 交接 | 对一个有效意图的不可变物化请求 | 随 `AgentRunResult` 进入 finalize |
| `PendingAtomSettlement` | Patchouli -> Alice 交接 | Patchouli 对物化请求的处理结果 | 通过事件回填运行时视图 |
| `MemoryAtom` | Patchouli | 已进入记忆域的正式资产 | 由 MemoryLibrary 与 Store 持久化和演化 |

Alice 不自行决定一个 WRITE 应被创建、合并、触碰还是丢弃，也不自行更新 Qdrant。相反，Patchouli 不持有 Agent frame，也不依赖 Alice 的可变 PendingAtom 对象。请求和应答都上移到 `core.models.pending`，正是为了让两侧共享契约而不产生子系统反向依赖。

## 2. 当前数据模型

### 2.1 Focus：保留 Agent 真正提交的内容

`WriteFocus` 保存 `content`、可选 `title` 与 `reason`；`UpdateFocus` 保存修订 `instruction`、可选替换内容，以及原记忆的 `base_alias` 和 `base_uuid`。Focus 是冻结 DTO，只描述 Agent 希望发生什么，不携带跨系统关联键。

关联信息由 `PendingAtom` 自身保存：

- `pending_alias`：Agent 可见的临时句柄；
- `intent_id`：Alice 与 Patchouli 之间的系统关联键；
- `source_verb`：区分 `WRITE` 与 `UPDATE`；
- `identity`：物化时继续使用的调用方身份；
- `runtime_scope`：`run_id`、`frame_id` 与动作坐标；不保存父帧和深度拓扑；
- `status` 与可选 `settlement`：当前生命周期及其结算视图。

pending alias 服务于运行时可读性，`intent_id` 服务于异步关联。两者不能互相替代：alias 可能被 Agent 写进后续命令，intent_id 则不应成为 prompt 中的领域名词。

### 2.2 RuntimeScope：把写入意图放回执行现场

caller 与 callee frame 共享同一个 `run_id`，但拥有不同 `frame_id`；调用关系由 Alice 的 `CallRecord(caller_frame_id, action_id)` 保存，不再写入 `RuntimeScope` 的 `parent_frame_id` 或 `depth`。这样，被调用 Agent 创建的 PendingAtom 无需在 CALL 返回后复制或合并，根 run 收尾时可以按共同 `run_id` 一次性认领；`finalize_frame()` 则为当前 CALL 投影 `FrameProducts.artifact_aliases`。

这是一种运行时关联，不是持久化 provenance。正式记忆的来源仍由 Patchouli 在生成与落库阶段整理。

## 3. 状态机

当前状态迁移由 `PendingAtomRuntime` 集中检查：

```text
PENDING ------> MATERIALIZING ------> SETTLED ------> EXPIRED
   |                 |
   |                 +---------------> FAILED -------> EXPIRED
   |                 |
   +-----------------+---------------> CANCELLED ----> EXPIRED
   |
   +---------------------------------> EXPIRED

EXPIRED --x  不再允许迁移
```

精确的合法边为：

- `PENDING -> MATERIALIZING | EXPIRED | CANCELLED`；
- `MATERIALIZING -> SETTLED | FAILED | CANCELLED`；
- `SETTLED | FAILED | CANCELLED -> EXPIRED`；
- `EXPIRED` 没有后继状态。

`PENDING` 和 `MATERIALIZING` 是仍在飞行的状态。`SETTLED`、`FAILED` 与 `CANCELLED` 已经结束业务处理，但会保留一个短暂可解释窗口；只有 `EXPIRED` 是不可再迁移的永久终态。这里的“永久”指状态机语义，运行时对象仍会在下一次回收周期被删除。

非法迁移会抛出 `InvalidStateTransition`。这使晚到或重复的 settlement 不会静默重写历史，但当前事件处理层尚未把这种重复交付转换为幂等成功，见第 10 节。

## 4. WRITE、UPDATE 与临时句柄

### 4.1 WRITE

`WRITE` 创建 `WriteFocus`，状态从 `PENDING` 开始，并生成：

```text
draft_{slug}_{4hex}
```

slug 优先取 title，否则取 content 的前 20 个字符；它只保留小写 ASCII 字母、数字、空格和下划线，最长 30 个字符。中文等无法形成 ASCII slug 的标题会退化为 `untitled`。返回值是 `ack + pending_alias`，随后可立即通过 READ 检查本轮意图。

### 4.2 UPDATE

`UPDATE` 必须先把目标解析为正式 MemoryAtom，再以它的 alias 与 UUID 创建 `UpdateFocus`。新的临时句柄形如：

```text
rev_{base_alias}_{4hex}
```

Koakuma 同时使旧 alias 的 L1 缓存失效，避免后续迭代继续把待修订内容当作未变化的热缓存事实。UPDATE 仍然不原地修改 MemoryAtom；真正的修订、合并和 provenance 处理属于 Patchouli。

两类 PendingAtom 还会获得独立的 `intent_{12hex}`。当前 store 同时维护 alias、intent_id 和 canonical UUID 反查索引，以便 settlement 在 alias 之外仍能按 intent 找回原意图。

这些映射只是寻址索引，不是第二份业务状态。一个 PendingAtom 的生命周期真相始终在对象自身的 `status`，结算真相始终在对象自身的 `settlement`；snapshot 与 RuntimeAliasResolver 必须从同一个 PendingAtom 及其 settlement 派生。store 不应再维护平行的 `_resolution`、`_redirects` 或其他状态副本，否则 READ、snapshot、事件回填和回收会在更新顺序不同的时候给出互相矛盾的答案。换言之，索引回答“怎样找到它”，PendingAtom 回答“它现在是什么”。

## 5. 三级 alias 解析

PendingAtom 要对同一 run 可见，又要在结算后自然过渡到正式记忆，因此 `RuntimeAliasResolver` 按三层解析：

```text
alias
  -> L0 PendingAtomRuntime
       pending       : 返回尚在飞行的 Focus
       settled       : 返回 canonical redirect 或 discarded
       failed/expired: 返回明确终态
       cancelled     : 当前退化为 not_found
  -> L1 KoakumaAtomCache
       返回本进程已预热或曾冷查询的 MemoryAtom
  -> L2 Patchouli public retrieval
       使用调用上下文中的 Identity 冷查询，并把命中写回 L1
```

READ、RUN、UPDATE 与 CALL 的 `context_refs` 都消费这套解析结果，而不是各自维护 alias 规则。正式 atom、pending focus、redirect 与失败状态随后交给 MemoryCompiler 渲染，handler 不复制第二套记忆展示逻辑。

settlement 若给出 canonical alias 或 UUID，旧 pending alias 会继续作为 redirect 工作。这样 Agent 不需要在异步物化后立刻知道新名称，也能获得“原句柄已演化”的 warning，而不是突然读到完全无关的 not found。

## 6. 从 run 收尾到 Patchouli 结算

正常完成的主 run 按以下顺序交接：

```text
WRITE / UPDATE
  -> register PendingAtom(PENDING)
  -> Agent continues and may READ pending_alias
  -> RunExecutor receives terminal entry FrameExecutionResult
  -> AgentRuntime.finalize_run(run_id, result) once
  -> claim current run's PENDING atoms
  -> PendingAtom(MATERIALIZING)
  -> PendingAtomMaterializeTask[]
  -> Patchouli finalize / generation
  -> settled | failed | cancelled event
  -> AliceRuntime updates the process-local view
```

`PendingAtomMaterializeTask` 是冻结投影，只携带 `pending_alias`、`intent_id`、`source_verb`、`identity` 与 `focus`。它故意不把 Alice store 或可变状态暴露给 Patchouli。相反，`PendingAtomSettlement` 携带 resolution 与可选 canonical 引用，构成请求/应答对偶。

当前 resolution 有五种：

| Resolution | 含义 |
|:---|:---|
| `CREATED` | 创建了新的 canonical MemoryAtom |
| `MERGED` | 去重流程把内容合并进既有记忆 |
| `TOUCHED` | 命中既有事实，仅刷新或强化而未产生新正文 |
| `UPDATED` | UPDATE 修订已经应用 |
| `DISCARDED` | 判断为不应形成 canonical 记忆 |

除 `DISCARDED` 外，resolution 必须携带 canonical UUID；`DISCARDED` 不允许携带 canonical 引用。这个约束由 `PendingAtomSnapshot` 校验。

取消、失败或预算耗尽的 Agent run 不交出物化任务，而是把该 `run_id` 下仍在飞行的 PendingAtom 迁到 `CANCELLED`/清理状态。只有 `COMPLETED` 的根 run 由 `finalize_run()` claim materialize tasks；被调用 frame 只调用 `finalize_frame()`，不触发 run finalization。

## 7. 回收语义

PendingAtom 当前没有墙钟 TTL。回收发生在成功根 run 的 `finalize_run()` 中，并以 AliceRuntime 共享 store 中的 **成功根 run 周期** 为节拍：

1. 删除此前已经是 `EXPIRED` 的对象及其索引；
2. 把不属于当前 run 的 `SETTLED`、`FAILED`、`CANCELLED` 迁到 `EXPIRED`；
3. 新迁入 `EXPIRED` 的对象再保留一个回收周期，让 resolver 有机会解释“句柄已过期”。

这不是按用户、session 或 elapsed time 独立计算的保留期。取消或失败根 run 不推进 retention epoch；旧 run 中仍处于 `PENDING` / `MATERIALIZING` 的对象也不会被回收步骤强行删除，它们依赖正常取消或 Patchouli 事件结束生命周期。

## 8. 所有权与不变量

维护这条链路时，应保持以下约束：

1. `ack` 只确认 Alice 已登记意图，不能用于声称持久化成功；
2. 只有完成的 run 默认向 Patchouli 交出物化任务；失败与取消不应静默落库；
3. Alice 拥有可变 PendingAtom 与状态机，Patchouli 拥有正式 MemoryAtom 与物化决策；
4. 跨边界只传 `PendingAtomMaterializeTask` / `PendingAtomSettlement`，不传内部 store 引用；
5. caller/callee frame 以共同 `run_id` 汇总，以 `frame_id` 保留执行来源，调用关系由 Alice ledger 保存；
6. resolver 统一解释 pending、redirect 与 canonical atom，MTP handler 不各建一套 alias 语义；
7. settlement 的 intent 必须与原 PendingAtom 匹配；
8. `EXPIRED` 不得复活为 in-flight 状态。

完整的 WRITE/UPDATE 协议语义见 [MTP 契约](../contracts/mtp.md)，Patchouli 如何消费 materialize task 见[生成与物化](../patchouli/generation.md)。

## 9. 代码与验证入口

| 责任 | 当前入口 |
|:---|:---|
| 核心模型、状态机与校验 | `src/hivememory/core/models/pending.py` |
| 生命周期命令与查询 | `src/hivememory/agent_runtime/pending_atom/runtime.py` |
| alias / intent / canonical 索引 | `src/hivememory/agent_runtime/pending_atom/store.py` |
| L0/L1/L2 统一解析 | `src/hivememory/agent_runtime/aliases/resolver.py` |
| frame/run 收尾认领与取消 | `src/hivememory/agent_runtime/runtime.py`、`src/hivememory/alice/orchestration/run_executor.py`、`call_coordinator.py` |
| settlement 事件回填 | `src/hivememory/alice/runtime/core.py` |
| 状态机与回收测试 | `tests/unit/agent_runtime/pending_atom/` |
| WRITE/UPDATE/READ/RUN 链路 | `tests/unit/agent_runtime/mtp/test_*_chain.py` |

## 10. 当前限制与设计张力

- PendingAtomRuntime、三个索引与 settlement 视图全部在进程内；没有 durable ledger、重启恢复、事件重放或未结任务扫描。进程在 ACK 后、物化前退出时，Alice 无法恢复该意图；
- PendingAtomRuntime 和 KoakumaAtomCache 由 AliceRuntime 全局共享。L0/L1 命中目前不会重新检查调用方 `Identity`，只有 L2 冷查询会携带身份，因而尚未满足跨用户并发运行所需的强隔离；
- 回收以成功根 run 的收尾为节拍，而非用户/session TTL。当前个人本地服务允许一个成功 run 推进另一个已结束句柄进入 EXPIRED 或删除；取消/失败 run 不推进 epoch；
- alias 只使用 4 位十六进制随机后缀，store 写入前不检查碰撞。同名碰撞会覆盖 alias 对应对象，并可能留下不一致反查索引；
- 中文标题通常生成 `draft_untitled_*`，可读性有限；alias 也没有进程外唯一性承诺；
- `CANCELLED` 在当前 resolver 中映射为 `not_found`，因此 MemoryCompiler 已有的 cancelled 文案不会通过正常 READ 路径出现；取消与真正不存在尚未对 Agent 明确区分；
- failed/cancelled 事件只向 Alice 传 pending alias，失败细节和 cancel reason 没有持久化到 PendingAtom。`cancel(..., reason=...)` 的 reason 参数当前未被保存；
- settlement 只接受 `MATERIALIZING -> SETTLED`。重复或晚到 settlement 会触发非法迁移，而事件消费者没有独立的幂等投递账本；
- 子帧当前继承父帧 `Identity`，AgentProfile 又不保存解析出的 target alias，部分子 Agent WRITE 的 identity/provenance 会显示为父 Agent；
- 旧 run 中失联的 `PENDING` / `MATERIALIZING` 没有自动超时回收，长期运行进程可能积累悬挂对象。

这些限制意味着 PendingAtom 已经建立了清晰的异步写回边界，却还不是一个可恢复、按租户隔离的持久化任务系统。未来若引入 Job Queue 或 durable execution，应保留“运行时意图不等于正式记忆”这一核心语义，再替换其进程内存储、关联与回收机制。
