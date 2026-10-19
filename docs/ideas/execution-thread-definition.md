---
title: 执行线程的正式定义
status: idea
horizon: current
serves_version: v0.7.0
owner: workspace
scope: execution-thread-as-task-process-execution-subject-definition-lifecycle-and-observation
code_paths:
  - src/hivememory/core/access.py
  - src/hivememory/workspace/authentication.py
  - src/hivememory/workspace/authorization.py
  - src/hivememory/workspace/process/threads.py
  - src/hivememory/workspace/process/working_set.py
  - src/hivememory/workspace/thread_ports.py
  - src/hivememory/workspace/contracts/threads.py
  - src/hivememory/workspace/contracts/operations.py
  - src/hivememory/workspace/capability/actor/
  - src/hivememory/workspace/capability/operations.py
  - src/hivememory/alice/orchestration/call/
  - src/hivememory/alice/orchestration/executor.py
related_docs:
  - docs/architecture/workspace.md
  - docs/ideas/execution-environment.md
  - docs/ideas/identity-and-access-model.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/ideas/fifth-generation-architecture.md
  - docs/archive/plans/v0.7.0-execution-thread-layer.md
  - docs/archive/plans/v0.7.0-task-process-owned-execution.md
  - docs/governance/baselines/v0.7.0-code-structure-causes-2026-10-18.md
last_reviewed: 2026-10-19
---

# 执行线程的正式定义

## 0. 文档性质

本文是执行线程设计的 Idea 来源，也是当前版本的设计讨论，为下一份计划供料：把执行线程定义为任务进程中的正式概念。本文不描述当前行为，当前事实见 [Workspace 架构](../architecture/workspace.md)第 4.3、5.4 节。

**计划**：已据本文建立 [v0.7.0 执行线程的正式定义](../plans/v0.7.0-execution-thread-definition.md)（2026-10-19，Planned），第 2 节的合并方向与第 3 节各项候选设计在计划中取定。

**演变**：执行线程此前的方向与决定，原在《执行单元、执行线程与执行环境》Idea 中，包括执行单元认识其全部线程、每个线程一份 context、派生必须到达进程、子线程以被调用方身份运行等。这些决定实施后晋升为事实，已于 2026-10-13 从该 Idea 删除，删除前版本见 commit `a384688`。该 Idea 于 2026-10-19 改名为[执行环境 Idea](./execution-environment.md)，只保留执行环境的选择与输出词汇；执行线程的后续设计以本文为准。

**来由**：执行线程是为 actor 的操作权限引入的，目的是让每个执行者各持一份可撤销的访问凭据，见[执行线程层](../archive/plans/v0.7.0-execution-thread-layer.md)（历史）。之后任务进程改为[持有自己的执行流](../archive/plans/v0.7.0-task-process-owned-execution.md)（历史），线程反而成了进程内“谁在执行”的天然单位，但这个概念从未被正式定义。成因调查见[代码结构成因基线](../governance/baselines/v0.7.0-code-structure-causes-2026-10-18.md)第 3.3、3.4 节。

**仍然有效的前提**，均已实施，本文不重新讨论：
- 每个线程一份线程 context；
- 派生必须经过进程，由进程以被调用方做准入；
- 子线程以被调用方的身份授权；
- 只有主线程能派生，不得递归 CALL；
- 操作的发起者由线程给出，请求不携带身份；目标是驻留 Workspace，跨 Workspace 目标的表达见[身份与访问体系 Idea](./identity-and-access-model.md)第 1 节 T-5b；
- CPU 读不出 context；
- finalize 之前撤销全部线程 context；
- 白名单每次即时检查；
- 外部执行单元不识别子线程（[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#e-5-外部执行单元的执行线程) E-5）。

## 1. 现状调查（2026-10-19，代码 `84b52c20`）

### 1.1 线程现在由哪些部分拼成

| 方面 | 主线程 | 子线程 | 承载位置 |
|:---|:---|:---|:---|
| 身份（actor） | 进程的 Agent | CALL 目标 Agent（同一用户） | 只在密封的 context 内，只有授权者能读 |
| 凭据 | 线程 context | 同左 | `WorkspaceAccessContext`，运行绑定为 `EXECUTION_THREAD` |
| 标识 | `main` | `child-N` | `RunBinding.thread_id` 与 `ProcessWorkingSet.thread_contexts` 的键 |
| 图纸 | 输入清单的 `agent_profile`，prepare 之前由 CPU 分配解析 | actor API 经读取视图解析 | 一个在清单里，一个在 `ExecutionThread` 里 |
| 操作快照 | CPU 分配时用进程 context 读取 | 派生时用子线程 context 读取 | 一个在清单里，一个在 `ExecutionThread` 里 |
| 共享的进程资源 | 驻留 Workspace、principal、派发话题、进程关联 | 同左 | context 与工作集，能力层经两个端口读取 |
| 拓扑 | 唯一的根，只有它能派生 | 不能再派生 | `ProcessThreads` 以对象身份比较；Alice 的子帧策略去掉 CALL |
| CPU 侧对应物 | 根帧 | 被调用方的帧 | Alice 的 `RunSession`，与线程没有共同标识 |
| 生命周期 | 进入 Actor 阶段时派生 | CALL 时派生 | 只有“撤销”一个事件，Actor 阶段结束时全部一起撤销 |

### 1.2 问题

1. **没有线程对象**。线程的数据分散在凭据、`ExecutionThread`、输入清单、工作集字典与进程阶段五处。主线程与子线程都要得到“图纸、操作快照、凭据”三样，但创建路径不对称：主线程的三样分散在清单与 `CPUPort.execute` 的参数里，子线程的三样打包为 `ExecutionThread`，交给 CALL 编排时再换成去掉凭据的 `Callee`。
2. **只有撤销，没有结束**。
   - 子线程在它的 CALL 返回后仍然有效，直到 Actor 阶段结束。
   - 派生成功、但子帧准备失败的子线程，同样一直有效到阶段结束（`CallCoordinator._prepare_callee`）。
   - 执行线程层当初决定“与主线程一起撤销”，取舍理由是“CALL 返回时撤销更严格，但需要另设‘结束线程’的路径，而 Alice 是进程内的可信代码”。这是实现成本的理由，不是语义上的理由。
3. **存活判定分散，且大半冗余**。
   - 一次 WRITE 要经过 5 个条件：context 未撤销、进程存在、阶段为 ACTOR、结局为 RUNNING、context 在登记中。这些条件分布在 `WorkspaceOperationAuthorizer._thread_grant` 与 `ProcessThreads` 两处，同一个凭据被拆封 4 次以上。
   - 其中三个条件被“未撤销”蕴含：线程在离开 ACTOR 之前已经撤销；登记与未撤销同进同出；进程关联本身取自这个 context。
   - 唯一多出来的是“结局为 RUNNING”：进程收到停止后，READ、SEARCH、RUN 仍能通过，WRITE 与 CALL 被拒绝，而且拒绝时用的是表示接线缺陷的 `ScopeRequiredError`。这一语义从未定义，决定见第 3.3 节。
4. **能力层需要线程的进程属性，只能走回流**。进程关联、派发话题与派生动作都属于线程，现在的路径是：授权者从 context 读出 `process_id`，再经 `thread_ports.py` 的 `ChildThreadSpawner` 与 `IntentDispatchTopicReader` 回到 `ProcessThreads` 查进程表。
5. **派生被建模为“返回凭据的 actor 操作”**。`DeriveCallThreadRequest` 是 `OperationRequest[ExecutionThread]`，结果里含有 context，所以 `ThreadOperationSubmitter` 的通用入口必须拒绝它，另开 `derive_call_thread`。根源是 `agent.call` 的权限检查与线程派生这一生命周期动作被合进了一条请求。
6. **线程身份不可观测**。
   - `thread_id` 只在运行绑定与字典键中出现；`AccessGrantSummary` 有这个字段，但没有人读取。
   - 写入意图只记录 `from_actor` 与 `process_id`；RuntimeEvent 与 SSE 都没有线程标识。同一进程两次 CALL 同一个 Agent 时，无法区分各自的操作。
   - 子帧沿用注册时的观测标签（执行线程层的非目标），所以子帧的 `agent.context.budget_exceeded` 报告的是调用方的 agent_id，而 SSE 帧事件取的是被调用方 Profile 的 agent_id。
7. **线程与帧是同一棵树，却记了两份**。Alice 以 `frame_id` 与 `CallRecord(caller_frame_id, action_id)` 记录这棵树，workspace 以 `main`/`child-N` 记录，两者一一对应，但没有共同标识。

## 2. 候选定义

> **执行线程**：任务进程在 Actor 阶段内的一个执行主体，回答“这一步是谁在执行”。它恰好属于一个进程，共享进程的资源，有自己的身份、凭据、能力与生命周期；CPU 在内部把它对应到自己的执行上下文（Alice 的帧）。

| 方面 | 候选设计 |
|:---|:---|
| 归属 | 恰好属于一个任务进程，只存在于 Actor 阶段；进程以线程表登记自己的全部线程 |
| 身份与凭据 | 一个 actor（同一用户），以一份线程 context 表达。身份只在凭据内，线程记录不另存身份字段 |
| 共享资源 | 驻留 Workspace、principal、派发话题、进程关联与会话归进程所有；线程只引用，不持有 |
| 能力 | 由线程 actor 的白名单决定并即时授权；交给 CPU 的操作快照只用于提示词与执行策略 |
| 拓扑 | 一个主线程，以及主线程派生的子线程，深度为 1；不派生子线程的 CPU 只有主线程 |
| 生命周期 | 派生 → 运行 → 结束（撤销凭据），没有“停止中”状态。结束时点：子线程随 CALL 返回结束（第 3.1 节）；进程收到停止时全部结束（第 3.3 节）；Actor 阶段结束时全部兜底结束 |
| 标识 | 稳定的 `thread_id`，用于写入意图、事件与帧的归因 |
| 观测标签 | 取线程自己的身份（第 3.2 节） |
| 与 CPU 的对应 | 派生结果把 `thread_id` 交给 CPU 驱动，由 CPU 记入自己的执行上下文；对应关系由 CPU 维护 |

**与操作系统的对照**：
- Linux 的凭据本来就按线程存放；
- 线程共享进程的资源，相当于共享地址空间；
- 帧相当于线程的栈与寄存器，归 CPU 所有；
- 派生要经过内核，因为凭据只能由内核签发；
- CALL 返回相当于 `pthread_exit`；
- 进程退出时，全部线程终止。

**由定义推出的合并方向**（候选设计，不是 owner 决定，可在计划评审时调整）：

- **线程表**：进程侧持有本进程的线程记录。记录包含 `thread_id`、父线程、状态、线程 context、图纸、操作快照与观测标签，取代工作集中的凭据字典。主线程与子线程经同一条路径创建，得到同一种线程描述，`ExecutionThread` 与 `Callee` 随之收敛。
- **一次解析**：能力层凭线程 context 向线程表做一次解析，得到授权所需的线程、进程关联与派发话题。由此取代入口校验、`get_thread_process_id`、`get_dispatch_topic` 的多次调用，以及 `IntentDispatchTopicReader` 端口；存活条件化简为“线程未结束且进程仍在运行”这一个判定。
- **派生与 `agent.call` 分开**：`agent.call` 仍只是 actor 操作，负责权限检查与解析目标图纸；派生与结束是线程表的生命周期动作。CPU 驱动仍经 workspace 取得子线程，因为凭据只能由认证一侧签发。
- **守卫**：从“某个函数只能被某个模块调用”，改为“线程表是派生、结束与解析的唯一入口”。

## 3. owner 决定

### 3.1 子线程的结束时点（owner，2026-10-19）

- **Alice**：子线程随它的 CALL 返回而结束，不再等到 Actor 阶段结束。Actor 阶段结束时，全部线程仍兜底结束。
- **外部 harness**：无法探测子 agent 的结束时点；子 agent 的创建时点也不一定能取得，因此子线程可能根本不会被创建。这一点留到接入外部 harness 时考虑，与外部 Actor Idea E-5“外部执行单元不识别子线程”的现行决定一并处理。

候选设计（不是 owner 决定）：
- CALL 的三种结局（成功、失败、取消）都会结束子线程；
- 派生成功、但子帧准备失败时，子线程当场结束；
- 结束由 CPU 驱动在 CALL 收尾时经 workspace 提交；没有提交的线程，由 Actor 阶段的兜底结束覆盖；
- 结束之后，该线程的后续提交按生命周期拒绝；已经开始派发的写入不撤回，与现行规则一致。

### 3.2 子帧的观测标签（owner，2026-10-19）

子帧的观测标签取子线程自己的身份：`agent_id` 为被调用方，`workspace_id` 不变。这一决定推翻执行线程层“子帧沿用注册时的 `ExecutionLabels`”的非目标。标签仍只用于观测与提示词，不参与授权。

受影响的观测（2026-10-19 核对）：
- 子帧的 `agent.context.budget_exceeded`（`RunExecutor` 读取帧标签的 `agent_id`），改为报告被调用方；
- Alice 交互输出中帧事件的 `agent_id` 已经优先取 Profile 的来源 alias，只有回退值随之改变；
- 进程级的 `chat.run.*` 与运行级的 `agent.run.*` 仍使用注册标签。

候选设计（不是 owner 决定）：标签随派生结果一起交给 CPU 驱动，而不是由 Alice 自行从图纸推导，与 `thread_id` 同处一份线程描述。

### 3.3 停止时先停止生成，线程随停止结束（owner，2026-10-19）

停止请求的处理顺序是：先停止生成，再处理其他地方的取消后续。线程随停止结束，没有“停止中”状态。

**处理顺序**：
1. **停止生成**：停止请求同步记录停止，并立即取消 Actor 阶段的子 task。CPU 中断生成（Alice 关闭 LLM 流、打断正在执行的 MTP、标记未回填的 CALL），同时进程的全部线程结束，此后不再接受任何 actor 操作。
2. **CPU 收尾**：CPU 只用已经产生的内容组装 cancelled 结果，不依赖 actor 操作。
3. **取消后续**：撤销凭据、接受检查点、用户停止且有输出时封口并写入会话、finalize 与 cleanup，都经进程 context 完成，不需要线程。

**理由**：停止的语义是执行者不再产生新的行动。停止之后没有合法的 actor 操作需求：CPU 收尾只用手里已有的内容，后续步骤都走进程 context。所以线程不需要“停止中只读”这一态；“收尾时仍要读取”的前提是停止后执行者还在行动，这与停止的语义相悖。

**停止那一刻已经发生的事**：
- 在途的读取随子 task 的取消一起中断，不是被拒绝；
- 已经开始派发的写入受 shield 保护，照常完成。这是已经发生的事实，不是新的行动，与“已提交的写入不撤回”的规则一致。

**外部 harness**：我们无法直接停止它的生成，只能发出取消通知。结束它的线程、拒绝它此后发来的调用，是本地能做到的最强停止；它收到通知之前的行为，随接入外部 harness 一并考虑（与第 3.1 节相同）。

**候选设计**（不是 owner 决定）：
- **统一的拒绝条件**：每个 actor 操作都按“线程未结束且进程仍在运行”判定可否执行，现行“读取放行、写入与派生拒绝”的不一致随之消除；拒绝统一使用生命周期类错误，不再使用 `ScopeRequiredError`。
- **CPU 端口契约**：补充一句“收到取消后的收尾不得依赖 actor 操作，只能用已经产生的内容组装结果”。
- **线程结束的时机**：线程在停止请求被接受时就结束；凭据的撤销与线程表的清理仍在 Actor 阶段收尾时执行，两者之间的提交按线程已结束拒绝。对 Alice 而言，这段窗口里本来就没有新请求，可观察行为不变。

## 4. 相关文档

| 文档 | 关系 |
|:---|:---|
| [Workspace 架构](../architecture/workspace.md) | 第 4.3 节线程 context、第 5.4 节操作入口与 CALL 派生的当前事实 |
| [执行环境 Idea](./execution-environment.md) | agent 与执行环境的分别选择、不同执行环境的输出词汇；原《执行单元、执行线程与执行环境》Idea，执行线程部分以本文为准 |
| [身份与访问体系 Idea](./identity-and-access-model.md) | 前提 2 与 T-5b：操作的发起者与目标分开看待，跨 Workspace 目标的表达 |
| [外部 Actor Idea](./external-actor-registration-and-runtime-access.md) | E-5：外部执行单元不识别子线程；第 3.1、3.3 节的外部 harness 部分随接入一并考虑 |
| [第五代架构 Idea](./fifth-generation-architecture.md) | 第 2 节“执行线程（actor）”的用语与第 3.3 节“执行者可以替换”；本文是该用语的设计来源 |
| [执行线程层归档计划](../archive/plans/v0.7.0-execution-thread-layer.md) | 子线程的派生、身份与撤销的实施历史；本文第 3.2 节推翻其子帧标签的非目标 |
| [代码结构成因基线](../governance/baselines/v0.7.0-code-structure-causes-2026-10-18.md) | 线程概念逐片叠加（第 3.4 节）与能力层回流（第 3.3 节）的证据 |
