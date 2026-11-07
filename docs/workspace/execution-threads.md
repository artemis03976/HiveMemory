---
title: 执行线程
status: current
owner: workspace
scope: execution-thread-table-handles-liveness-main-and-call-child-threads
code_paths:
  - src/hivememory/workspace/process/threads.py
  - src/hivememory/workspace/process/working_set.py
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
last_reviewed: 2026-11-06
---

# 执行线程

执行线程是任务进程从 PREPARE 到 Actor 阶段的执行主体，回答“这一步是谁在执行”。主线程在 PREPARE 开始时派生，先以执行者的身份提交预检索，进入 Actor 阶段后交给 CPU；CALL 由主线程派生子线程；两者以同一种执行线程交给 CPU，线程记录各自持有一个工作 actor，CPU 只拿到不透明的线程句柄。工作 actor 的角色与子工作 actor 的准入见[认证与访问登记](./authentication.md#4-工作-actor-的角色与持有位置)；进程如何进入与结束 Actor 阶段见[任务进程](./task-process.md#7-actor-阶段)。

## 1. 执行线程是什么

执行线程恰好属于一个进程，共享进程的资源（驻留 Workspace 与 principal 随执行者的工作 actor，派发话题在工作集中，会话属于会话服务），有自己的身份（线程记录中的工作 actor）、能力（操作快照）与生命周期；CPU 在内部把它对应到自己的执行上下文（Alice 的帧）。

| 对象 | 内容 | 持有者 |
|:---|:---|:---|
| `ThreadDescriptor`（`workspace.contracts`） | 线程标识、观测标签、图纸与操作快照；冻结，CPU 可见 | 线程登记与执行线程 |
| `ThreadHandle`（`workspace.contracts`） | 不透明的线程句柄：只带 `process_id` 与 `thread_id` 用于定位，按对象身份比较，不含身份 | 线程记录与 CPU 驱动 |
| `ExecutionThread`（`workspace.contracts`） | 线程描述与线程句柄 | CPU 驱动 |
| `ThreadOperationSubmitter`（`workspace.contracts`） | 执行线程与操作入口绑定成的提交函数：交出线程描述，以泛型调用提交记忆请求，以 `derive_child` 返回子线程的提交函数，以 `end` 同步结束本线程；句柄留在内部 | 帧（每个帧持有所属线程的提交函数） |
| `ResolvedThread`（`workspace/thread_ports.py`） | 一次存活解析的结果：进程关联、线程标识与线程的工作 actor；不含派发话题，写入经线程表另行取得（第 8 节） | 能力层的一次调用 |

线程描述回答“谁来处理”，输入清单回答“处理什么”。CPU 驱动只持有句柄，取不到工作 actor；帧只持有提交函数，取不到句柄。MTP 与 CALL 只提交操作参数，不能另选目标 Workspace 或 process_id。

## 2. 线程表

| | 进程 | 线程 |
|:---|:---|:---|
| 登记表 | `ProcessTable`：全系统一份，按 `process_id` 登记全部进程 | `ProcessThreadTable`：每个进程一份（`TaskProcess.threads`），按 `thread_id` 登记本进程的线程 |
| 被登记的对象 | `TaskProcess`：进程容器，不离开注册入口 | `ProcessThread`：可变的状态（`RUNNING`/`ENDED`）与登记时确定的句柄、工作 actor、父线程和线程描述，不离开线程管理 |
| 交给外部的视图 | `ProcessHandle`：交付方持有 | `ExecutionThread`：CPU 驱动持有；`ResolvedThread`：能力层的一次解析结果 |

线程表只登记派生成功的线程：派生的各步（子线程的被调用方准入、读取操作快照、创建句柄）全部成功后，登记是最后一步，失败时不留下任何状态，也不消耗编号。子线程标识按派生顺序分配（`child-1`、`child-2`……），不会与主线程的 `main` 重名；已结束的线程留在表中直到进程关闭，标识不复用。线程表不登记进程的资源、CPU 一侧的执行状态（帧与提交函数归 CPU 驱动）或写入意图（意图只以 `thread_id` 归因）。线程表随进程创建，主线程在 PREPARE 开始时登记，子线程在 Actor 阶段内登记，随进程一起丢弃。

`ProcessThreads`（`workspace/process/threads.py`）是派生、结束、解析与取得派发话题的唯一入口：`derive_main(process, profile)`、`derive_child`、`resolve`、`dispatch_topic`、`end_child`、`end_all`。线程表与线程对象的写方法只由它调用；执行器、CPU 分配与能力层都不直接读写线程表。它经 `core.access` 的 `SubActorAdmission` 端口为子线程的被调用方做准入（组合根注入认证网关），经操作授权者读取操作快照，经 root 话题 API 为进程这一轮确定话题；能力层通过组合根注入的线程端口 `ExecutionThreadTable` 使用它，因此不导入任务进程，任务进程在能力层中也只导入 root API。

## 3. 生命周期与唯一的存活判定

线程的生命周期是“派生 → 运行 → 结束”，结束只是同步记为 `ENDED`，重复结束是空操作，没有“停止中”状态，也没有需要撤销的凭据。结束有三个时点：子线程随 CALL 返回结束（CPU 经操作入口提交，第 6.2 节）；进程的停止被接受时全部线程即已结束；Actor 阶段结束时全部兜底结束，关闭流程再兜底，进程在 PREPARE 中失败或取消时由关闭流程结束（第 7 节）。停止路径不通知线程表：存活判定直接读取进程结局，停止被接受之后的提交按线程已结束拒绝。

存活只有一个判定，由 `ProcessThreads.resolve` 一次完成：线程未结束（状态为 `RUNNING`），且进程仍在运行（结局为 `RUNNING`）。在此之前，线程表按句柄的 `process_id` 与 `thread_id` 找到所属进程与线程，并要求登记的正是这一个句柄（按对象身份比较）。判定不读取阶段：线程从 PREPARE 存在到 Actor 阶段结束，此后全部结束。操作入口的 `execute` 在分派前调用它，线程提交的全部 actor 操作（含 PREPARE 中的预检索）因此共用同一个存活判定（[能力层](./capability.md#6-操作入口)）；取得派发话题也先做同一个判定。存活判定只覆盖线程提交：不建进程的外部入口经请求级提交 `execute_request` 以请求的工作 actor 提交，没有线程可判定，也不经线程表。

拒绝分两类：线程已结束、进程已收到停止或已经关闭（所属进程已不在进程表中），是生命周期拒绝 `ThreadEndedError`（`workspace.thread.ended`），actor 请求与 CALL 派生都得到它，Agent 出口明确提示停止操作；进程仍在而线程未登记、登记的不是这一个句柄、子线程试图派生或 CPU 结束主线程，是接线缺陷，以 `ScopeRequiredError` 拒绝。错误码与呈现见[错误模型](../contracts/error-model.md)第 4.4 节。

进程已注销的句柄按线程已结束处理，是因为句柄只由线程表在登记线程时创建：所属进程不在进程表中，只能是进程已经关闭，线程随之结束；进程关闭之后才到达的迟到提交因此得到与停止相同的生命周期拒绝。

## 4. 为什么这样设计

- **线程不另设记录一层**：进程单独设 `ProcessRecord`，是因为停止请求、事件发布、阶段 task 绑定与系统阶段的授权只读写控制状态，而容器还要容纳生命周期各不相同的任务参数、工作集、task 与输出通道。线程只有可变的控制状态与冻结的描述两部分，也没有只需要控制状态的使用方，所以由 `ProcessThread` 一个对象承担，它冻结的部分就是线程描述。
- **停止即结束**：停止被接受后 Actor 子 task 立即被取消，CPU 只用已经产生的内容组装结果。如果停止之后读取仍放行、写入与派生却被拒绝，拒绝就只能借用表示接线缺陷的错误；把停止并入“线程已结束”，六项 actor 操作得到同一种生命周期拒绝。
- **子线程随 CALL 返回结束**：子线程只为一次委派存在，CALL 返回后仍让它存活，迟到的子帧操作就能在主线程继续运行期间以被调用方的身份执行。
- **句柄不透明、按对象身份核对**：句柄若是 `(process_id, thread_id)` 的值引用，CPU 一侧就能以另一个线程的标识提交，例如子线程冒用主线程的身份与白名单。线程表要求登记的正是这一个句柄对象，标识相同的另一个句柄以接线缺陷拒绝；句柄本身不含身份，工作 actor 只留在线程记录中。
- **解析与取得派发话题分开**：每个操作都需要线程的进程关联、线程标识与工作 actor，由线程表一次解析给出，解析是同步的、不挂起。派发话题只有写入用到，而且可能要为这一轮第一次建立话题、需要 `await`，因此经 `dispatch_topic` 另行取得：READ、SEARCH、RUN 与 CALL 不触发话题确定，写入登记前的同步段也只做一次不挂起的解析。

## 5. 主线程交给 CPU

PREPARE 开始时，执行器把刚解析的图纸交给 `ProcessThreads.derive_main(process, profile)`，后者以进程记录的执行者读取操作快照、生成线程描述与句柄并登记。返回的 `ExecutionThread` 先由执行器以主线程提交预检索、取得已有话题的派发话题，进入 Actor 阶段后作为 CPU `execute` 的独立 `thread` 参数，不进入输入清单。主线程在 PREPARE 派生，是为了让预检索与 CPU 的 actor 操作走同一个操作入口、按同一个工作 actor 授权，并共用同一个存活判定：PREPARE 中收到停止时，尚未开始的预检索与话题确认同样以 `ThreadEndedError` 被拒绝。CPU 驱动以 `ThreadOperationSubmitter` 把执行线程与入口绑定为主帧的提交函数，帧与线程的对应关系由此建立。

**观测标签**：执行线程的标签取线程自己的身份，由线程表在派生时放进线程描述：主线程就是注册时创建的标签，CALL 子线程为 `ExecutionLabels(被调用方, 注册时的 workspace_id)`。Alice 的根帧与子帧各取所属线程描述中的标签，不据此组装身份，也不从图纸推导。标签只服务历史角色匹配、展示回退与观测，不能授权或选择资源目标。

**操作快照**：线程表派生每个线程时，以该线程的工作 actor 经只读查询 `allowed_operations` 读取它在访问登记中的 actor 操作集合（准入已失效时拒绝，空白名单得到空集合），写入线程描述的 `allowed_operations`。主线程的快照因此在 PREPARE 开始时读取：读取时准入已失效，主线程派生失败，进程在 PREPARE 以失败收口，先于预检索与任何话题副作用。快照只描述执行者能做什么，供 CPU 生成提示词与执行策略（Alice 翻译为 MTP 动词，见 [MTP 契约](../contracts/mtp.md#4-权限)第 4 节）；它不是身份，也不是授权依据，每个操作请求仍由 actor API 即时授权。登记在运行实例内不可变，快照与即时检查因此一致。

## 6. CALL 派生与子线程

`agent.call` 与派生分开。actor API 的 `resolve_call_target` 只负责 actor 操作：以调用方线程的工作 actor 授权 `agent.call`，再以调用方的 scope 读取目标 Profile（对调用方不可见与不存在相同），结果只有图纸、不含身份。调用方只能唤起它看得见的 Agent，这是调用方自己的操作。派生是线程的生命周期动作，由入口的 `derive_child_thread(request, *, thread)` 在一次提交内依次完成：

1. `resolve` 确认调用方存活，取得调用方的工作 actor；
2. `resolve_call_target` 以它解析图纸；
3. 线程表的 `derive_child` 在冷读之后重新判定调用方存活，确认调用方是主线程（子线程派生以 `ScopeRequiredError` 拒绝），再以图纸的 `agent_id`（内置图纸为 `omni_doll`）为被调用方做准入（[认证与访问登记](./authentication.md#4-工作-actor-的角色与持有位置)），得到子工作 actor，以它读取被调用方的操作快照，生成线程描述与句柄后登记；准入或快照读取失败时不登记。

结果是执行线程 `ExecutionThread`（线程描述与子线程句柄），只返回给 CPU 驱动，由它绑定为子帧的提交函数。`DeriveCallThreadRequest` 因此不是 `OperationRequest`，`execute` 不接受它。CALL 只允许根 frame 发起，子线程不能再 CALL。

### 6.1 派生在入口内组合

派生必须在入口内组合：子工作 actor 只能经认证一侧的准入得到，被调用方是谁、能否进入 Workspace 必须来自 workspace 自己的解析。如果拆成“先取图纸、再凭图纸派生”两次提交，CPU 就能交回一份伪造的图纸，等于自报子线程的身份；入口依次调用 actor 操作与生命周期动作，两者在代码上分开，但在一次提交内完成。

### 6.2 子线程的结束

入口的 `end_thread(*, thread)` 同步调用线程表的 `end_child`：把子线程记为结束，不检查进程是否仍在运行；已结束的线程（含所属进程已经关闭）是空操作，主线程与登记的不是这一个句柄时以 `ScopeRequiredError` 拒绝，主线程只由 Actor 阶段收尾结束。Alice 在 CALL 收尾时经子线程的提交函数提交结束，子帧任何终态、取消清理与派生后的准备失败都会结束子线程（[Alice 编排](../alice/orchestration.md#8-失败取消与降级)第 8 节）；之后以子线程提交的请求按线程已结束拒绝。Alice 不结束主线程。

### 6.3 子线程以被调用方身份运行

每个 actor 请求按被调用方的白名单检查，`IdentityScope` 的发起者是被调用方，资源 owner 按被调用方的可见性过滤；写入意图的 `from_actor` 是被调用方、`thread_id` 是子线程，进程关联不变，在操作内立即派发到与主线程相同的话题。调用方持有 `agent.call` 即可让被调用方读取它自己的 PRIVATE 记忆，并经 CALL 的返回值交给调用方；这是委派的本义，闸门是 `agent.call` 与被调用方的准入。子 frame 的结果只影响 reply 与 alias 回填，不取消已提交的写入。

## 7. Actor 阶段结束与停止

执行器在 Actor 阶段结束时同步结束本进程仍在运行的全部线程（`end_all`），且只在这里执行一次：CPU 返回终态结果（含停止后返回的取消结果）之后、封口与提交之前；Actor 阶段因异常、停止、交付方离开或硬取消提前结束时，也在交付终态（硬取消为传播取消）之前执行。关闭流程先于任何 await 兜底结束线程并释放附件租借，再等待进行中的话题确定结束；进程最后由注册入口注销。

停止被接受时线程即已结束，此后的提交在存活判定处拒绝；CPU 内的在途读取随 Actor 子 task 的取消中断。结束不取消其他操作调用方的任务，在途只读请求可以自然完成，后续提交明确拒绝。CPU 取消后的收尾不得依赖 actor 操作。

WRITE/UPDATE 由入口带着句柄交给 actor API，依次：以线程的工作 actor 授权；UPDATE 冷读并校验基础原子；经线程表的 `dispatch_topic` 取得派发话题（第 8 节）；以无挂起点的同步段再调用一次 `resolve`，取得进程关联与线程标识并登记；派发。基础冷读与取得派发话题都在同步段之前，跨越线程结束或停止时在同步段拒绝，不产生意图，已经建立的话题没有写入。派发开始后由 actor API 持有 shield 保护的任务直到完成，再传播调用方取消，不因线程结束、停止、进程终态或 CALL 子帧失败回滚（[读取视图与写入意图](./read-view-and-write-intents.md#5-派发与结算)）。

## 8. 派发话题

派发话题是进程这一轮唯一的话题，保存在工作集（`ProcessWorkingSet.dispatch_topic_id`）中，主线程与 CALL 子线程共享，请求与 CPU 不能指定派发目标。只有写入用到它：`resolve` 不涉及派发话题，写入经线程表的 `dispatch_topic(thread)` 取得——先做存活判定，派发话题已冻结时直接返回，否则加入正在进行的话题确定（没有、或上一次已失败时启动一次），等待它的结果。

话题确定由进程持有，任务保存在工作集中：线程表以进程记录的系统 root 工作 actor 调用 root API 的“为一轮确定话题”（`RootTopicService.determine_topic`），传入工作集保存的 Gateway 决定，指向已有话题时确认它并刷新驻留，指向新话题时以 Gateway 给出的标题与摘要建立。确定话题是系统级操作，授权在 root API 的方法入口；线程表只判定存活与取得派发话题，不授权。成功时冻结派发话题，此后的 `chat.run.*` 事件带这个话题；失败时不冻结，错误交回等待它的写入，下一次写入重新确定，一次暂时的失败不会让整轮无法写入。每个写入以 `join_shielded` 各自等待：一个写入被取消，不会取消话题确定，也不影响其他线程的写入；并发的写入只确定一次。

Gateway 指向已有话题时，执行器在 PREPARE 以主线程调用同一个方法，派发话题从 Actor 阶段开始即已冻结；指向新话题时由第一次写入建立，本轮没有写入时话题在交互应用时建立（[任务进程](./task-process.md#53-话题何时确定)）。提交任务与关闭流程都先等待进行中的话题确定结束。

## 9. 已知限制

- 子线程不能再 CALL，没有“谁能调用谁”的细粒度策略（[能力层](./capability.md#9-已知限制)）；
- 外部 harness 的子线程随外部 Actor 接入一并考虑，见[外部 Actor Idea](../ideas/external-actor-registration-and-runtime-access.md#e-5-外部执行单元的执行线程) E-5；
- 操作快照在派生时读取一次；白名单的运行时编辑尚不存在，引入后需要处理快照与即时检查之间的差异。

## 10. 代码与测试入口

- [`workspace/process/threads.py`](../../src/hivememory/workspace/process/threads.py)（线程表与 `ProcessThreads`）、[`workspace/thread_ports.py`](../../src/hivememory/workspace/thread_ports.py)（线程端口与解析结果）、[`workspace/contracts/threads.py`](../../src/hivememory/workspace/contracts/threads.py)（线程描述、线程句柄与执行线程）、[`workspace/contracts/operations.py`](../../src/hivememory/workspace/contracts/operations.py)（入口与提交函数协议）、[`WorkspaceOperationEntry`](../../src/hivememory/workspace/capability/operations.py)；
- 测试：[`test_process_threads.py`](../../tests/integration/workspace/test_process_threads.py)（线程表的登记与标识、唯一的存活判定含进程已关闭并同样作用于取得派发话题、解析不要求派发话题、并发取得派发话题只以 system 确定一次、一个等待者取消不影响确定、确定失败交回并由下一次重试、标识相同的另一个句柄被拒、只有主线程能派生、派生失败不留登记、子线程沿用主工作 actor 的来源与驻留 Workspace、子线程结束与兜底结束、线程描述的标签与快照、Actor 阶段结束后提交仍以 system 执行）、[`test_process_intents.py`](../../tests/integration/workspace/test_process_intents.py)（主线程与子线程共用一个话题、只有写入确定话题、确定话题期间停止时不登记）、[`test_chat_mtp_operations.py`](../../tests/e2e/system/test_chat_mtp_operations.py)（CALL 子线程按被调用方授权与写入归属）。

背景与演进见已归档的[执行线程 Idea](../archive/ideas/execution-thread-definition.md)与[执行线程正式定义计划](../archive/plans/v0.7.0-execution-thread-definition.md)（历史记录）。
