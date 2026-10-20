---
title: 执行线程
status: current
owner: workspace
scope: execution-thread-table-liveness-main-and-call-child-threads
code_paths:
  - src/hivememory/workspace/process/threads.py
  - src/hivememory/workspace/thread_ports.py
  - src/hivememory/workspace/contracts/threads.py
  - src/hivememory/workspace/contracts/operations.py
  - src/hivememory/workspace/capability/operations.py
  - src/hivememory/workspace/capability/actor/
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/mtp.md
  - docs/contracts/error-model.md
related_docs:
  - docs/workspace/task-process.md
  - docs/workspace/authentication.md
  - docs/workspace/capability.md
  - docs/alice/orchestration.md
last_reviewed: 2026-10-19
---

# 执行线程

执行线程是任务进程在 Actor 阶段内的执行主体，回答“这一步是谁在执行”。主线程在进入 Actor 阶段时派生，CALL 由主线程派生子线程；两者以同一种执行线程交给 CPU，各自持有一份线程 context。线程 context 作为凭据的签发、派生与撤销见[认证与访问登记](./authentication.md#4-进程-context-与线程-context)；进程如何进入与结束 Actor 阶段见[任务进程](./task-process.md#7-actor-阶段)。

## 1. 执行线程是什么

执行线程恰好属于一个进程，共享进程的资源（驻留 Workspace 与 principal 在进程 context 中，派发话题在工作集中，会话属于会话服务），有自己的身份（在线程 context 中）、能力（操作快照）与生命周期；CPU 在内部把它对应到自己的执行上下文（Alice 的帧）。

| 对象 | 内容 | 持有者 |
|:---|:---|:---|
| `ThreadDescriptor`（`workspace.contracts`） | 线程标识、观测标签、图纸与操作快照；冻结，CPU 可见 | 线程登记与执行线程 |
| `ExecutionThread`（`workspace.contracts`） | 线程描述与线程 context | CPU 驱动 |
| `ThreadOperationSubmitter`（`workspace.contracts`） | 执行线程与操作入口绑定成的线程句柄：交出线程描述，以泛型调用提交记忆请求，以 `derive_child` 返回子线程的句柄，以 `end` 同步结束本线程 | 帧（每个帧持有所属线程的句柄） |
| `ResolvedThread`（`workspace/thread_ports.py`） | 一次存活解析的结果：进程关联、线程标识与派发话题 | 能力层的一次调用 |

线程描述回答“谁来处理”，输入清单回答“处理什么”。CPU 读不出句柄中的 context，CPU 驱动也不持有进程 context；MTP 与 CALL 只提交操作参数，不能另选目标 Workspace 或 process_id。

## 2. 线程表

| | 进程 | 线程 |
|:---|:---|:---|
| 登记表 | `ProcessTable`：全系统一份，按 `process_id` 登记全部进程 | `ProcessThreadTable`：每个进程一份（`TaskProcess.threads`），按 `thread_id` 登记本进程的线程 |
| 被登记的对象 | `TaskProcess`：进程容器，不离开注册入口 | `ProcessThread`：可变的控制状态（线程 context、状态 `RUNNING`/`ENDED`、父线程）与冻结的线程描述，不离开线程管理 |
| 交给外部的视图 | `ProcessHandle`：交付方持有 | `ExecutionThread`：CPU 驱动持有；`ResolvedThread`：能力层的一次解析结果 |

线程表只登记派生成功的线程：派生的各步（签发线程 context、读取操作快照）全部成功后才登记，失败时撤销刚签发的 context，不留下登记，也不消耗编号。子线程标识按派生顺序分配（`child-1`、`child-2`……）；已结束的线程留在表中直到进程关闭，标识不复用。线程表不登记进程的资源、CPU 一侧的执行状态（帧与线程句柄归 CPU 驱动）或写入意图（意图只以 `thread_id` 归因）。线程表随进程创建，只在 Actor 阶段内有登记，随进程一起丢弃。

`ProcessThreads`（`workspace/process/threads.py`）是派生、结束与解析的唯一入口：`derive_main(process, profile)`、`derive_child`、`resolve`、`end_child`、`end_all`。线程表与线程对象的写方法只由它调用；执行器、CPU 分配与能力层都不直接读写线程表。它经 `ThreadContextLifecycle` 端口派生与撤销线程 context，经操作授权者读取运行绑定与操作快照；能力层通过组合根注入的线程端口 `ExecutionThreadTable` 使用它，能力层与任务进程因此互不导入。

## 3. 生命周期与唯一的存活判定

线程的生命周期是“派生 → 运行 → 结束”，结束即同步撤销线程 context 并记为 `ENDED`，重复结束是空操作，没有“停止中”状态。结束有三个时点：子线程随 CALL 返回结束（CPU 经操作入口提交，第 6.2 节）；进程的停止被接受时全部线程即已结束；Actor 阶段结束时全部兜底结束，关闭流程再兜底（第 7 节）。停止路径不通知线程表：存活判定直接读取进程结局，停止被接受到 Actor 阶段收尾撤销凭据之间的提交按线程已结束拒绝。

存活只有一个判定，由 `ProcessThreads.resolve` 一次完成：线程 context 未撤销且运行绑定是执行线程；进程在进程表中、线程登记在它的线程表中，且登记的就是这个 context（按对象身份比较）；线程状态为 `RUNNING`，且进程结局为 `RUNNING`。判定不读取阶段，因为线程只在 Actor 阶段内存在。操作入口的 `execute` 在分派前调用它，全部 actor 操作因此共用同一个存活判定（[能力层](./capability.md#6-操作入口)）。

拒绝分两类：线程的凭据已撤销、线程已结束或进程已收到停止，是生命周期拒绝 `ThreadEndedError`（`workspace.thread.ended`），actor 请求与 CALL 派生都得到它，Agent 出口明确提示停止操作；线程未登记、context 类型或运行绑定不对、子线程试图派生或 CPU 结束主线程，是接线缺陷，以 `ScopeRequiredError` 拒绝。错误码与呈现见[错误模型](../contracts/error-model.md)第 4.4 节。

## 4. 为什么这样设计

- **线程不另设记录一层**：进程单独设 `ProcessRecord`，是因为停止请求、事件发布、阶段 task 绑定与进程控制授权只读写控制状态，而容器还要容纳生命周期各不相同的任务参数、工作集、task 与输出通道。线程只有可变的控制状态与冻结的描述两部分，也没有只需要控制状态的使用方，所以由 `ProcessThread` 一个对象承担，它冻结的部分就是线程描述。
- **停止即结束**：停止被接受后 Actor 子 task 立即被取消，CPU 只用已经产生的内容组装结果。如果停止之后读取仍放行、写入与派生却被拒绝，拒绝就只能借用表示接线缺陷的错误；把停止并入“线程已结束”，六项 actor 操作得到同一种生命周期拒绝。
- **子线程随 CALL 返回结束**：子线程的凭据只为一次委派存在，CALL 返回后仍保留它，迟到的子帧操作就能在主线程继续运行期间以被调用方的身份执行。
- **一次解析**：能力层需要线程的进程关联、线程标识与派发话题，这些都属于线程；由线程表一次解析给出，能力层就不必先读进程关联、再回到进程表查话题。

## 5. 主线程交给 CPU

执行器把 prepare 之前解析的图纸交给 `ProcessThreads.derive_main(process, profile)`，后者派生主线程 context、以它读取操作快照、生成线程描述并登记，返回的 `ExecutionThread` 作为 CPU `execute` 的独立 `thread` 参数，不进入输入清单。CPU 驱动以 `ThreadOperationSubmitter` 把执行线程与入口绑定为主帧的线程句柄，帧与线程的对应关系由此建立。

**观测标签**：执行线程的标签取线程自己的身份，由线程表在派生时放进线程描述：主线程就是注册时创建的标签，CALL 子线程为 `ExecutionLabels(被调用方, 注册时的 workspace_id)`。Alice 的根帧与子帧各取所属线程描述中的标签，不读取访问 context 组装身份，也不从图纸推导。标签只服务历史角色匹配、展示回退与观测，不能授权或选择资源目标。

**操作快照**：线程表派生每个线程时，以该线程刚签发的 context 经 `get_stage_operations` 读取线程 actor 在访问登记中的 actor 操作集合（与阶段授权相同的准入与目标检查，空白名单得到空集合），写入线程描述的 `allowed_operations`。主线程的快照因此在进入 Actor 阶段时读取：准入在 prepare 与进入 Actor 之间被撤销时，主线程派生失败，进程在 Actor 阶段以失败收口。快照只描述执行者能做什么，供 CPU 生成提示词与执行策略（Alice 翻译为 MTP 动词，见 [MTP 契约](../contracts/mtp.md#4-权限)第 4 节）；它不是身份，也不是授权依据，每个操作请求仍由 actor API 即时授权。登记在运行实例内不可变，快照与即时检查因此一致。

## 6. CALL 派生与子线程

`agent.call` 与派生分开。actor API 的 `resolve_call_target` 只负责 actor 操作：以调用方的线程 context 授权 `agent.call`，再以调用方的 scope 读取目标 Profile（对调用方不可见与不存在相同），结果只有图纸、不含凭据。派生是线程的生命周期动作，由入口的 `derive_child_thread(request, *, access)` 在一次提交内依次完成：

1. `resolve` 确认调用方存活；
2. `resolve_call_target` 解析图纸；
3. 线程表的 `derive_child` 在冷读之后重新判定调用方存活，确认调用方是主线程（子线程派生以 `ScopeRequiredError` 拒绝），再分配线程标识，以图纸的 `agent_id`（内置图纸为 `omni_doll`）做被调用方准入后派生子线程 context，以它读取被调用方的操作快照，生成线程描述并登记；快照读取失败时撤销刚签发的 context，不登记。

结果是执行线程 `ExecutionThread`（线程描述与子线程 context），只返回给 CPU 驱动，由它绑定为子帧的线程句柄。`DeriveCallThreadRequest` 因此不是 `OperationRequest`，`execute` 不接受它。CALL 只允许根 frame 发起，子线程不能再 CALL。

### 6.1 派生在入口内组合

派生必须在入口内组合：线程 context 只能由认证一侧签发，被调用方是谁、能否进入 Workspace 必须来自 workspace 自己的解析。如果拆成“先取图纸、再凭图纸派生”两次提交，CPU 就能交回一份伪造的图纸，等于自报子线程的身份；入口依次调用 actor 操作与生命周期动作，两者在代码上分开，但在一次提交内完成。

### 6.2 子线程的结束

入口的 `end_thread(*, access)` 同步调用线程表的 `end_child`：撤销子线程凭据并记为结束，不检查进程是否仍在运行；已结束的线程是空操作，主线程与未登记的 context 以 `ScopeRequiredError` 拒绝，主线程只由 Actor 阶段收尾结束。Alice 在 CALL 收尾时经子线程句柄提交结束，子帧任何终态、取消清理与派生后的准备失败都会结束子线程（[Alice 编排](../alice/orchestration.md#8-失败取消与降级)第 8 节）；之后以子线程提交的请求按线程已结束拒绝。Alice 不结束主线程。

### 6.3 子线程以被调用方身份运行

每个 actor 请求按被调用方的白名单检查，`IdentityScope` 的发起者是被调用方，资源 owner 按被调用方的可见性过滤；写入意图的 `from_actor` 是被调用方、`thread_id` 是子线程，进程关联不变，在操作内立即派发到与主线程相同的话题。调用方持有 `agent.call` 即可让被调用方读取它自己的 PRIVATE 记忆，并经 CALL 的返回值交给调用方；这是委派的本义，闸门是 `agent.call` 与被调用方的准入。子 frame 的结果只影响 reply 与 alias 回填，不取消已提交的写入。

## 7. Actor 阶段结束与停止

执行器在 Actor 阶段结束时同步结束本进程仍在运行的全部线程（`end_all`），且只在这里执行一次：CPU 返回终态结果（含停止后返回的取消结果）之后、finalize 之前；Actor 阶段因异常、停止、交付方离开或硬取消提前结束时，也在交付终态（硬取消为传播取消）之前执行。关闭流程先于任何 await 兜底结束线程并释放附件租借，再等待 cleanup；进程 context 最后由注册入口撤销。

停止被接受时线程即已结束，到收尾撤销凭据之间的提交在存活判定处拒绝；CPU 内的在途读取随 Actor 子 task 的取消中断。结束不取消其他操作调用方的任务，在途只读请求可以自然完成，后续提交明确拒绝。CPU 取消后的收尾不得依赖 actor 操作。

WRITE/UPDATE 在基础校验之后，以无挂起点的同步段只调用一次 `resolve`，从结果取得进程关联、线程标识与派发话题并登记；UPDATE 的基础冷读在同步段之前，冷读跨越线程结束或停止时在同步段拒绝，不产生意图。派发开始后由 actor API 持有 shield 保护的任务直到完成，再传播调用方取消，不因线程结束、停止、进程终态或 CALL 子帧失败回滚（[读取视图与写入意图](./read-view-and-write-intents.md#5-派发与结算)）。

## 8. 派发话题

CPU 分配把与输入清单相同的真实 `topic_id` 写入 `ProcessWorkingSet.dispatch_topic_id`，先于派生主线程；`resolve` 在存活判定之后从工作集读取它，进程没有派发话题属于接线缺陷。CALL 子线程共享该话题，请求与 CPU 不能指定派发目标。

## 9. 已知限制

- 子线程不能再 CALL，没有“谁能调用谁”的细粒度策略（[能力层](./capability.md#9-已知限制)）；
- 外部 harness 的子线程随外部 Actor 接入一并考虑，见[外部 Actor Idea](../ideas/external-actor-registration-and-runtime-access.md#e-5-外部执行单元的执行线程) E-5；
- 操作快照在派生时读取一次；白名单的运行时编辑尚不存在，引入后需要处理快照与即时检查之间的差异。

## 10. 代码与测试入口

- [`workspace/process/threads.py`](../../src/hivememory/workspace/process/threads.py)（线程表与 `ProcessThreads`）、[`workspace/thread_ports.py`](../../src/hivememory/workspace/thread_ports.py)（线程端口与解析结果）、[`workspace/contracts/threads.py`](../../src/hivememory/workspace/contracts/threads.py)（线程描述与执行线程）、[`workspace/contracts/operations.py`](../../src/hivememory/workspace/contracts/operations.py)（入口与线程句柄协议）、[`WorkspaceOperationEntry`](../../src/hivememory/workspace/capability/operations.py)；
- 测试：[`test_process_threads.py`](../../tests/integration/workspace/test_process_threads.py)（线程表的登记与标识、唯一的存活判定、子线程结束与兜底结束、线程描述的标签与快照）、[`test_thread_context_process.py`](../../tests/integration/workspace/test_thread_context_process.py)（Actor 阶段结束即撤销全部线程，阶段收尾仍以进程 context 授权）、[`test_thread_context.py`](../../tests/integration/workspace/test_thread_context.py)、[`test_chat_mtp_operations.py`](../../tests/e2e/system/test_chat_mtp_operations.py)（CALL 子线程按被调用方授权与写入归属）。

背景与演进见已归档的[执行线程 Idea](../archive/ideas/execution-thread-definition.md)与[执行线程正式定义计划](../archive/plans/v0.7.0-execution-thread-definition.md)（历史记录）。
