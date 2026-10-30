---
title: 任务进程概念盘点（2026-10-28）
status: baseline
owner: project
scope: task-process-concept-inventory
snapshot_at: 2026-10-28
source_commit: 3b6b8a3f476bd0c4bbc0e0a114ef98c9081702b5
code_paths:
  - src/hivememory/workspace/process/
  - src/hivememory/workspace/contracts/
  - src/hivememory/workspace/capability/operations.py
  - src/hivememory/workspace/thread_ports.py
  - src/hivememory/workspace/authentication.py
  - src/hivememory/workspace/authorization.py
  - src/hivememory/server/routers/chat.py
  - src/hivememory/alice/interface/cpu.py
related_docs:
  - docs/governance/baselines/v0.7.0-code-structure-causes-2026-10-18.md
  - docs/archive/todo/task-process-unused-paths.md
  - docs/workspace/task-process.md
  - docs/workspace/execution-threads.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/ideas/execution-environment.md
  - docs/architecture/decisions/0002-unique-identities-and-minimal-concurrency.md
last_reviewed: 2026-10-28
---

# 任务进程概念盘点

本文冻结 `source_commit`（v0.7.0 发布之后、v0.7.1 开发期间的 `master`）上对任务进程的逐个概念盘点。最新行为仍以代码、测试与当前设计为准。

**起因**：owner 担心代码复杂度与可读性：一部分功能提前做了用不上的工程化处理，另一部分为边界情况写了大量独立的处理代码。本文对任务进程中的每个概念回答三个问题：它保护的是什么真实行为，有几个生产使用方，能不能删。

**前提**（owner，2026-10-28）：controller 模式的第二个使用方落地之前，任务进程这块理论上是悬空的，但不考虑回退。因此本文不评估回退，只在现有结构内找可以删除与合并的部分。

**与上一份基线的关系**：[v0.7.0 代码结构成因调查](./v0.7.0-code-structure-causes-2026-10-18.md)回答“整体为什么显得乱”，其中执行流（3.1）与执行线程（3.3、3.4 的线程部分）已由两份计划处理并归档。本文在那之后逐个概念核对使用方，不重复成因分析。

## 1. 范围与方法

- **范围**：`workspace/process/`（11 个文件，2,768 行），以及它直接依赖的部分：`workspace/contracts/`、操作入口 `capability/operations.py`、`thread_ports.py`，以及认证与授权中与进程 context、线程 context 有关的方法。
- **使用方**：只认 `src/` 与 `frontend/src/` 中的生产调用。测试不算使用方；删除代价按测试中的调用点统计，用 AST 统计调用表达式。
- **分类**：

  | 类 | 含义 |
  |:---|:---|
  | A 承重 | 今天的 Alice chat 依赖它 |
  | B 预留 | 为 controller 模式的外部执行单元准备，当前只有 Alice 一个实现 |
  | C 无生产使用方 | 可以删除 |
  | D 可合并 | 同一事实有两种表达，或者检查的是自家已经保证的结果 |
  | E 罕见场景 | 只服务一个罕见场景，处理却分散在多处 |

- **规模**：进程包的非空行中，代码 1,758 行，docstring 537 行，注释 75 行。
- **判断边界**：只读调查，没有修改代码。

## 2. 结论

- 进程包大部分是承重的（第 7 节）。没有生产使用方、可以直接删除的生产代码约 170 行（第 3 节），删除代价主要在测试改写。
- 阅读难度主要来自两处，都不是死代码：系统关闭超时后的硬取消，其处理分散在 5 个位置（第 5 节）；进程、线程、请求三种 context 组成的凭据模型（第 8 节）。
- 为 controller 预留的部分成本不高，接缝的位置也对；但第二个实现出现之前，它的中立性无法验证（第 6 节）。

## 3. 没有生产使用方（C）

| 概念 | 代码 | 生产使用 | 测试依赖 |
|:---|:---|:---|:---|
| 非流式交付：`run_process(stream=False)` 的重载、`_deliver_once`、`NonStreamingResult`、`NonStreamingAgentOutcome`、`NonStreamingCommandOutcome`、`non_streaming_result` | `process/outputs.py`、`process/service.py` | 无。`POST /chat` 只用流式交付；[项目总览](../../PROJECT.md)已注明它“当前只供测试驱动” | 35 处调用，13 个文件，含 `tests/helpers/registered_chat.py` |
| 按句柄取消：`cancel_process(ProcessHandle)` 的重载 | `process/service.py` | 无。断连经 `_detach` 直接请求停止；`POST /chat/stop` 用 `process_id` 加请求级 context | 22 处，5 个文件 |
| 诊断查询：`ActorAuthenticationGateway.describe_context` → `WorkspaceAuthenticator.describe` → `AccessGrantSummary` | `authentication.py`、`process/service.py` | 只有一处：按 `process_id` 取消、进程不存在或不可控时，给 `chat.run.cancel_requested` 补上请求方的 `workspace_id` 观测标签 | 34 处，11 个文件，多数用来断言 context 已撤销 |
| 注册时对重复 `process_id` 的预检 | `process/service.py` 的 `register_process` | 无实际作用。server 以 uuid4 生成 `process_id`，生产中不会重复，进程表登记本身也拒绝重复。[ADR-0002](../../architecture/decisions/0002-unique-identities-and-minimal-concurrency.md) 不为随机碰撞建立专门的业务分支 | 1 个回滚测试 |

## 4. 可合并（D）

- **`stream_closed` 与 `client_disconnected`**：两者都表示交付方离开，同属丢弃输出的停止原因。HTTP 交付使用后者；前者只出现在 `close_process`（响应的流从未开始迭代时）与 `run_process` 的默认参数中。
- **观测标签挂在事件发布器上**：`ExecutionLabels` 由 `BoundProcessEvents.labels` 持有。执行器写入会话、CPU 分配组装清单、线程表生成主线程与子线程的标签，都从 `record.events.labels` 读取。标签是进程的属性，放在进程记录上更直接。
- **`_require_prepared_workspace`**：它校验 Patchouli prepare 返回的 `belong_to` 等于任务目标 Workspace。prepare 本身以阶段授权得到的目标 scope 调用，Workspace 归属由资源 owner 校验；这里检查的是自家子系统的返回值，执行器还因此必须“先写入工作集再校验”。

## 5. 罕见场景，处理分散（E）：系统关闭超时后的硬取消

- **唯一来源**：进程 task 只在一种情况下被取消：`TaskProcessService.shutdown` 等待超过 `shutdown_wait_seconds`（默认 10 秒）。用户停止与交付方离开只取消当前可中断阶段的子 task，不取消进程 task。
- **为它写的处理**：
  - `_run_interruptible` 记录进入时的 `cancelling()`，在子 task 正常结束后与抛出取消后各比较一次；
  - `_seal_and_finalize` 的 `CancelledError` 分支：硬取消期间提交仍然成功时，补记 CPU 的结局；
  - `close()` 的终态兜底与 `closed_before_terminal` 事件（这一事件的另一个来源是进程从未运行就被关闭）；
  - `_deliver_once` 在进程 task 没有终态产出时抛出 `RuntimeError`；
  - `tests/integration/workspace/test_session_completion.py` 中约 20 处相关代码与断言。
- **保护的行为**：某个进程不响应停止时，系统关闭不会无限等待；已经开始的提交仍由 `join_shielded` 等到结束。
- **未核实的前提**：停止只能中断 Gateway 与 Actor 两个阶段，prepare、CPU 分配与 FINALIZE 都不响应停止。这三段是否各有上限尚未核实。其中 finalize 等待交互提交队列的工作结果（`PatchouliService._wait_active_interaction`），这一层没有自己的超时，上限取决于队列的处理时限与重试。如果三段都有可以接受的上限，硬取消这一层可以去掉或收窄；否则应当保留。

## 6. 为 controller 模式预留（B）

| 概念 | 代码 | 当前实现 |
|:---|:---|:---|
| CPU 端口与中立结果：`CPUPort`、`CPUEventSink`、`CPUExecutionResult`、`CPUExecutionStatus`、`CPUFailureReason` | `contracts/cpu.py` | `AliceCPU` 与测试 CPU |
| 输入清单 `CPUInputManifest` | `contracts/process.py` | 同上 |
| 可交给任何 CPU 驱动的线程形态：`ThreadDescriptor`、`ExecutionThread`、`ThreadOperationSubmitter` | `contracts/threads.py`、`contracts/operations.py` | Alice |
| 提示词在进程侧编译，以不透明文本（`instructions`、`turn_context`）交给 CPU | `process/allocation.py` | Alice |

两个契约文件共 168 行，成本不高，接缝的位置也对。中立性目前只在类型上成立：`CPUFailureReason` 中的 `compaction_failed`、`turn_context_budget_exceeded` 描述的是 Alice 自己的会话压缩与轮内守卫，[执行环境 Idea](../../ideas/execution-environment.md) 已经记录 controller 模式下这类原因无从产生。上一份基线 3.2 的判断仍然成立：只有一个实现时，无法验证边界是否切对。

## 7. 承重（A）

| 概念 | 保护的行为 | 生产使用方 |
|:---|:---|:---|
| 注册入口 `TaskProcessService`：注册、流式运行、`close_process`、按 `process_id` 取消、`shutdown` | 认证失败不创建进程；注册成功即登记；响应的流从未开始时也能关闭进程 | `POST /chat`、`POST /chat/stop`、System 关闭 |
| 进程句柄 `ProcessHandle` | server 只持有不透明引用，取不到进程记录与 context | chat 路由 |
| 进程表 | 按 `process_id` 停止，关闭时遍历，线程表据此定位所属进程 | 注册入口、线程表 |
| 进程记录：阶段、结局、停止原因、当前可中断的 task | 只有 Gateway 与 Actor 可以中断；FINALIZE 拒绝停止；重复停止返回首次原因 | 执行器、注册入口 |
| 两类停止原因 | 用户停止保留部分输出并提交；交付方离开与系统关闭丢弃本轮 | 前端停止、SSE 断连、System 关闭 |
| 会话认领 | 同一会话同时只有一个进程 | 注册入口 |
| 工作集 | 附件租借随进程关闭释放；prepare 新建且仍为空的话题得到补偿 | 执行器、CPU 分配 |
| 四阶段骨架、`_run_interruptible`、`_ProcessCancelled`、`_ReturnedResult` | 停止立即中断 Gateway 或 Actor；CPU 把取消转换为返回值时保留部分输出 | 执行器 |
| 输出通道、阶段产出与 `stream_events` | 进程持有自己的 task，断连打断不了 finalize；交互事件有背压 | 流式交付；前端依赖 `process_id`、`memory_refs` 与 `run_status: finalizing` |
| 命令终态 `command_terminal` | 命令只解析、不执行，解析结果转换为“暂不可用”或拒绝的终态 | Gateway 命中命令时 |
| `CPUAllocator` | Profile 在 prepare 之前解析，缺少图纸的请求不留下预建话题；附件版本不一致时拒绝整轮 | 执行器 |
| 主线程与操作快照 | Alice 据快照决定提示词中的 MTP 动词与 frame 的执行策略 | `alice/run/service.py` |
| 子线程：`derive_child`、`end_child`、子线程 context、`child-N` 编号 | CALL 的子帧以被调用方身份运行：按被调用方的白名单授权，写入记在被调用方名下 | `alice/orchestration/call/coordinator.py` |
| `chat.run.*` 事件 | 运行时观测 | 前端运行时事件日志 |

## 8. 观察项：主线程 context

- 主线程 context 与进程 context 来自同一份授予内容，只是运行绑定不同（`WorkspaceAuthenticator.derive_thread_context`）。
- 停止被接受后，actor 操作被拒绝靠的是线程表在存活判定中检查进程结局（`ProcessThreads._resolve`），不依赖撤销主线程凭据。
- 独立凭据的作用：actor API 只接受线程 context，阶段调用只接受进程 context，两个调用面不能互相借用；主线程与子线程以同一种形态交给 CPU。
- 代价：同一份授予内容有两份凭据。“UPDATE 的基础冷读跨越线程结束”、派生子线程时在图纸冷读前后各判定一次存活，这类边界情况主要来自线程凭据的生命周期。
- 外部执行单元不识别子线程（[外部 Actor Idea](../../ideas/external-actor-registration-and-runtime-access.md#e-5-外部执行单元的执行线程) E-5），子线程的使用方会一直只有 Alice CALL。controller 模式落地时，可以依据外部 CPU 只使用主线程的实际情况，评估主线程是否需要独立凭据。本文不对此提出建议。

## 9. 后续处置

| 项 | 处置 |
|:---|:---|
| 第 3、4 节 | 登记为 Todo：[任务进程中无生产使用方的代码与可合并小项](../../archive/todo/task-process-unused-paths.md)（2026-10-30 完成并归档） |
| 第 5 节 | 未排期。修改系统关闭流程时，先核实第 5 节的前提再决定 |
| 第 6 节 | 保留。建议 controller 模式重新排期之前，不再往这一层增加中立抽象；上一份基线 3.2 的候选方向（收窄 CPU 边界）仍未排期 |
| 第 8 节 | 只记录，controller 模式落地时评估 |
| 上一份基线 3.5（结构靠规则维持） | 同日修订 [AGENTS.md](../../../AGENTS.md)：第 3、4 节压缩为所有权摘要与当前设计文档的索引，新增控制复杂度的工作原则 |
