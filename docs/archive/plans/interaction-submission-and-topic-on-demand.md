---
title: 交互提交收束与话题按需创建
status: archived
archived_at: 2026-11-06
implemented_by: 分支 docs/interaction-submission-plan（PR 待开）
superseded_by: docs/workspace/task-process.md
owner: workspace
target: 版本之外的工作（身份与访问主线的后续：controller 与 Import Bus 的交互提交统一经 root API）
scope: single-interaction-submission-through-root-api-topic-on-demand-and-prepare-split
updates:
  - AGENTS.md
  - docs/workspace/task-process.md
  - docs/workspace/execution-threads.md
  - docs/workspace/read-view-and-write-intents.md
  - docs/workspace/sessions.md
  - docs/workspace/capability.md
  - docs/workspace/identity-and-authorization.md
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/mcp.md
  - docs/contracts/mtp.md
  - docs/contracts/error-model.md
  - docs/system/import-bus.md
  - docs/system/application-services.md
  - docs/system/composition.md
  - docs/patchouli/README.md
  - docs/patchouli/perception.md
  - docs/patchouli/retrieval.md
  - docs/patchouli/lifecycle.md
  - docs/patchouli/generation.md
  - docs/patchouli/memory-library.md
  - docs/components/runtime-and-bus.md
  - docs/alice/pending-atom.md
  - docs/architecture/overview.md
  - docs/architecture/boundaries.md
  - docs/architecture/data-model.md
  - docs/frontend/chat-view.md
  - docs/governance/security/identity-and-execution-safety.md
  - docs/PROJECT.md
code_paths:
  - src/hivememory/workspace/process/
  - src/hivememory/workspace/thread_ports.py
  - src/hivememory/workspace/capability/operations.py
  - src/hivememory/workspace/capability/actor/memory.py
  - src/hivememory/workspace/capability/root/interaction.py
  - src/hivememory/workspace/capability/root/topic.py
  - src/hivememory/workspace/capability/root/memory.py
  - src/hivememory/workspace/sessions/service.py
  - src/hivememory/patchouli/service.py
  - src/hivememory/patchouli/system.py
  - src/hivememory/patchouli/contracts/public_routes.py
  - src/hivememory/patchouli/contracts/prepare.py
  - src/hivememory/patchouli/contracts/interaction_submission.py
  - src/hivememory/patchouli/application/interaction_submission_service.py
  - src/hivememory/patchouli/application/topic_management_service.py
  - src/hivememory/patchouli/runtime/bridge.py
  - src/hivememory/system/services/import_bus/ingestor.py
  - src/hivememory/system/assembler.py
  - src/hivememory/mcp_server/server.py
  - src/hivememory/alice/run/service.py
  - src/hivememory/core/protocol/models.py
  - src/hivememory/server/models/session.py
  - tests/unit/architecture/test_access_boundaries.py
related_docs:
  - docs/ideas/external-session-and-topic-projection.md
  - docs/ideas/unified-interaction-memory.md
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/chat-run-lifecycle-follow-ups.md
  - docs/archive/plans/import-bus-authorization-in-capability-layer.md
  - docs/archive/plans/unified-identity-authorization-entry.md
  - docs/workspace/task-process.md
  - docs/system/import-bus.md
---

# 交互提交收束与话题按需创建

> **状态**：Archived（2026-11-06，已实施，实施记录见第 10 节）。当前事实见[任务进程](../../workspace/task-process.md)、[能力层](../../workspace/capability.md)、[执行线程](../../workspace/execution-threads.md)与 [Import Bus](../../system/import-bus.md)。以下为实施时的计划原文，链接已按归档位置改正。owner 在 2026-11-06 作出的决定集中在第 4.1 节，背景见[外部会话与 Topic 投影 Idea](../../ideas/external-session-and-topic-projection.md#10-交互提交收束与话题按需创建)第 10 节；其余是当时的候选设计。本计划不拆分的理由见第 6.1 节。

## 1. 背景和问题

### 1.1 交互提交有两个入口

[`InteractionSubmissionQueue`](../../../src/hivememory/patchouli/control/interaction_submission.py) 是 Patchouli 内部的一条执行 lane，上面有两个互不相通的入口，各自提交、各自等待、各自归类失败：

| | controller（任务进程） | plugin（Import Bus） |
|:---|:---|:---|
| 入口 | `FINALIZE_AGENT_RUN`，Patchouli 内部直接入队 | root API 的 `submit_interaction`，另有 `interaction_status`、`wait_interaction` |
| 等待应用 | Patchouli 接管的续行，调用方离开也继续 | 编排器经 root API 等待，带时限 |
| 失败归类 | 续行里的 `_admit_active_interaction` / `_wait_active_interaction` | 编排器的 `_wait_applied` |
| 新话题 | prepare 预建，失败时补偿 | 提交携带标题与摘要，应用时建立；写入时确定话题经 `prepare_topic` 预建 |
| 收尾 | 补偿预建的空话题、核对实际所在的话题、记录检索命中 | 无 |

owner（2026-11-06）要求提交队列收束提交接口、完整封装提交逻辑，controller 的提交也走 root API，与 Import Bus 共用同一组方法。

### 1.2 预建话题把 finalize 绑在 prepare 上

finalize 中真正依赖 prepare 的只有两件事：补偿 prepare 新建的空话题，为 prepare 检索到的记忆记录命中。核对实际所在的话题与等待应用是“提交并等待”，Import Bus 也在做。耦合来自预建话题，而不是来自提交本身。

预建话题从 v0.7.0 起一直保留。2026-09-28/29 owner 为任务进程定的方向是 prepare 只做预检索、话题在提交后按需创建；2026-10-13 按需创建被暂缓，理由是将来流式同步需要先有容器（[任务进程 Idea](../../ideas/task-process-table-and-registration-entry.md#21-交互被接纳后即关闭进程q-1) 2.1）。同一天写入意图改为执行期间实时派发（#120），派发必须带一个已存在的话题，这个话题一直由 prepare 预建提供。owner（2026-11-06）重新打开按需创建：它唯一要解决的问题是如何响应模型的写入。

### 1.3 Import Bus 的临时 root 方法

[Import Bus 的授权移入能力层](./import-bus-authorization-in-capability-layer.md)把编排器需要 scope 的步骤做成了 root 方法：`interaction_status`、`wait_interaction`、`route_passive_turn`、`prepare_topic`。导入一轮最多做 5 次 root 授权（查进度、为续接再查上一轮、路由、提交、等待）。owner 决定其中查进度与 PASSIVE_MEMORY 路由是临时的，在本方向中移除，另外两个的去留由本方向决定。

### 1.4 预检索在 actor API 之外授权

预检索是执行者的 actor 操作，但现在由执行器自己调用 `authorize_agent` 取得检索 scope（`_retrieval_scope`），再交给 `PREPARE_AGENT_RUN`，不经 actor API。这是 `authorize_agent` 在 actor API 之外唯一的调用位置。

### 1.5 现有清理漏掉一种情形

Agent 先写入、CPU 随后没有输出时，关闭流程会把写入过的话题当空话题删除，因为 `Topic.is_empty` 只看是否接收过交互。

## 2. 目标与非目标

### 2.1 目标

1. **交互提交只有一个入口**：root API 的“提交交互”，controller 与 Import Bus 共用；Patchouli 提供“提交并等待应用”，接纳失败、应用失败与超时的归类只做一次。
2. **写入时确定话题只有一个入口**：root API 的“为一轮确定话题”，任务进程与 Import Bus 共用。
3. **话题按需创建**：任务进程不再预建话题；任何路径都不再补偿空话题。
4. **prepare 拆分**：预检索经 actor API 以主线程执行，检索命中与存储健康各自经 root API；prepare、finalize 与 cleanup 路由删除。
5. **写入先授权、再确定话题**：线程提交与请求级提交顺序一致，被拒的写入不触发路由、建立话题或 LRU 结算。
6. **其余对外行为不变**：停止、取消、交付方离开与系统关闭的语义，写入 ACK 的语义，导入的确认语义与 HTTP 契约不变；有意的变化只限第 4.10 节所列。

### 2.2 非目标

- 短期记忆的单位与形态：决定 9 只要求本计划不替它决定（第 8.8 节）。
- 交互被接纳后即关闭进程（[任务进程 Idea](../../ideas/task-process-table-and-registration-entry.md#21-交互被接纳后即关闭进程q-1) Q-1）：进程仍等交互应用完成，因为会话与 `done` 需要实际所在的话题。
- 提交队列的跨重启持久化：存储仍是 `InMemoryWorkStore`，归[持久化治理](../../governance/reliability/durability-and-recovery.md)。
- 对话流式同步到 Patchouli。
- Topic 的存储、settle、空闲与 LRU 规则本身。
- 检索命中扩展到 MCP 与 Alice 的 SEARCH：只记任务进程的预检索（第 8.1 节）。
- Gateway、Profile 与附件这几个系统阶段的调用方式，以及 Profile 的解析时点。
- 新话题提交的排序键（第 8.4 节）。

## 3. 当前状态和缺口

### 3.1 controller 链路

| 阶段 | 现状 | 代码 |
|:---|:---|:---|
| Gateway | 以 root scope 调用 Gateway（ACTIVE_CHAT），得到话题路由与检索计划 | [`runner.py`](../../../src/hivememory/workspace/process/runner.py) `_run_stages` |
| Profile | 在 prepare 之前以系统 root 工作 actor 解析 | [`allocation.py`](../../../src/hivememory/workspace/process/allocation.py) `resolve_agent_profile` |
| prepare | 执行器以 `authorize_agent` 取得检索 scope；`PREPARE_AGENT_RUN` 经 `TOPIC_PREPARE` 确认或新建话题（新建前按 LRU 驱逐）、检索、读取存储健康，返回 `PreparedAgentRun`；失败时清理自己新建的空话题 | `runner.py` `_prepare_and_allocate`、`_retrieval_scope`；[`patchouli/service.py`](../../../src/hivememory/patchouli/service.py) `prepare_agent_run` |
| CPU 分配 | 话题写入输入清单与工作集的 `dispatch_topic_id`；记忆编译进提示词，存储健康决定离线提示 | `allocation.py` `allocate` |
| Actor | 进入 Actor 时派生主线程；`resolve` 要求派发话题存在，否则以接线缺陷拒绝；WRITE/UPDATE 以派发话题派发 | `runner.py` `_run_actor`、[`threads.py`](../../../src/hivememory/workspace/process/threads.py) `derive_main`、`resolve` |
| 结算 | 封口后同步写入会话（话题取 prepare 的话题），进入 FINALIZE，以 root scope 调用 `FINALIZE_AGENT_RUN`；Patchouli 的续行接纳、等待应用、核对话题、记录命中，应用失败或调用方离开时补偿新话题 | `runner.py` `_seal_and_finalize`；`patchouli/service.py` `finalize_agent_run` |
| 关闭 | prepare 成功而 finalize 未成功时请求 `CLEANUP_PREPARED_AGENT_RUN` | `runner.py` `close`、[`working_set.py`](../../../src/hivememory/workspace/process/working_set.py) |
| 系统关闭 | 进程服务等待进程 task 结束；随后 Patchouli 先 `drain_active_finalizations` 再停止提交队列 | [`system.py`](../../../src/hivememory/system/system.py)、[`patchouli/system.py`](../../../src/hivememory/patchouli/system.py) |

### 3.2 plugin 链路

- **导入**（[`ingestor.py`](../../../src/hivememory/system/services/import_bus/ingestor.py) `ingest`）：在外部会话串行门内，经 root API 查进度（已应用的返回重复，应用中的继续等待，没有输出的跳过）；未提交或应用失败的，经 root API 做 PASSIVE_MEMORY 路由（为续接再查一次上一轮的进度），或沿用写入时登记的话题；经 root API 提交（新话题携带标题与摘要，应用时建立），再经 root API 等待应用。
- **写入时确定话题**（`place`）：登记里有的直接返回；经 root API 查进度，已应用的返回实际所在的话题，应用中的以 `apply_pending` 交回；否则路由后经 `prepare_topic`（背后是 `PREPARE_AGENT_RUN`，不检索）预建话题并登记。
- **MCP 写入**（[`server.py`](../../../src/hivememory/mcp_server/server.py) `_submit_intent`）：先经 Import Bus 确定话题，再以该话题经请求级提交 `execute_request`。确定话题发生在 actor API 授权之前：没有 `memory.write` 的 Agent 调用写入，也会先路由、建话题，可能触发 LRU 结算。

### 3.3 依赖“话题提早已知”的地方

| 使用方 | 用途 | 按需创建后 |
|:---|:---|:---|
| 输入清单 `CPUInputManifest.topic_id` | 只用于 Alice 根帧的 `topic_id` 与 `agent.run.*` 事件，不进入提示词 | 新话题时为空 |
| 工作集 `dispatch_topic_id` 与 `resolve` | 每次线程操作都要求存在；只有 WRITE/UPDATE 真正使用 | 只有写入需要，写入前确定 |
| `record.events.bind_topic` | 之后的 `chat.run.*` 运行事件带话题 | 确定话题或应用之后绑定 |
| 会话 `ConversationTurn.topic_id`（必填） | 下一轮 Gateway 路由失败时续接上一轮的话题；会话 HTTP 响应与前端历史 | 取实际所在的话题，可空 |
| 运行结果的 `topic_id` 与 SSE `done.topic_id` | 前端在消息上显示话题 | 取实际所在的话题 |
| `PreparedAgentRun.memories` | `memory_refs` 事件与记忆编译 | 取预检索结果 |
| `PreparedAgentRun.storage_available` | 提示词的存储离线提示 | 取存储健康调用的结果 |

### 3.4 缺口

1. 交互提交有两个入口，等待与失败归类分散在两边（1.1）。
2. 预建话题让 finalize 依赖 prepare，派生出清理路由、续行、补偿状态与话题核对（1.2）。
3. Import Bus 一轮做多次 root 授权，查进度与路由是临时的 root 方法（1.3）。
4. 预检索在 actor API 之外授权（1.4）。
5. 写入过的话题可能被当作空话题删除（1.5）。
6. MCP 写入在授权之前确定话题（3.2）。

## 4. 目标设计

### 4.1 owner 决定（2026-11-06）

1. **话题按需创建**：不预建话题；第一次写入时建立，否则推迟到接受交互时建立。推翻 2026-10-13 的“保留预建”。
2. **finalize 不做 cleanup**：发生写入之后话题就不再为空，不再有任何补偿。
3. **检索命中不必等到 finalize 才记录**。
4. **prepare 与 finalize 拆成单独的能力层调用**。
5. **交互提交队列收束提交接口、完整封装提交逻辑**，不能让两边各自管理。
6. **controller 的交互提交走 root API**，与 Import Bus 共用同一组方法。
7. **写入先授权、再建立话题**，MCP 同样。
8. **Import Bus 的进度查询与 PASSIVE_MEMORY 路由这两个临时 root 方法在本方向移除**；`wait_interaction` 与 `prepare_topic` 的去留由本方向决定。
9. **写入需要话题，允许第一次写入建立话题**：写入派发即生成的时点不变，话题何时、由谁建立收在一处；短期记忆的单位与形态不由本方向决定。
10. **交互提交与写入时确定话题收束为 root API 的两个方法**（“提交交互”与“为一轮确定话题”），入口各授权一次；查进度、路由与等待在方法内部完成；写入的顺序统一为授权 → UPDATE 的基础冷读 → 确定话题 → 同步段登记 → 派发，请求级提交的派发话题参数改为确定话题的回调。
11. **主线程在 PREPARE 开始时派生**：预检索经操作入口以主线程提交；请求级提交仍只给不建进程的入口。

### 4.2 话题何时建立

- **Gateway 指向已有话题**：PREPARE 中由任务进程经“为一轮确定话题”确认它存在并刷新驻留，与现在 prepare 对已有话题的处理相同；派发话题随即冻结。
- **Gateway 指向新话题**：PREPARE 中不建立。本轮第一次写入时，进程经“为一轮确定话题”以 Gateway 的标题与摘要建立话题并冻结派发话题；本轮没有写入时，提交携带 `NEW_TOPIC` 与标题、摘要，话题在交互应用时建立，与导入相同。
- **LRU 驱逐**跟随建立话题的时点，规则不变（[`PerceptionFamiliar.prepare_topic`](../../../src/hivememory/patchouli/services/perception.py) 与 `apply_interaction` 共用同一套驱逐与建立）。
- **不再补偿**：建立后最终没有接收交互的话题，由现有的空闲结算、LRU 或关闭结算按无材料删除（第 8.2 节）。

### 4.3 root API

root API 中与交互相关的方法只留两个，入口各以 `authorize_root` 授权一次；方法内部依次调用的公开路由不再各自授权。

**提交交互**：`RootInteractionService.submit_interaction`

- 输入：目标 Workspace、参与者、交互标识、封口后的交互、来源（`active_chat` / `import_bus`）、话题来源、可选的等待时限（controller 不设，导入设），以及执行者 `system` 的工作 actor。
- 话题来源三选一：
  - 已确定的话题：任务进程冻结的派发话题，或导入在写入时确定的话题；
  - Gateway 的新话题决定：`NEW_TOPIC` 与标题、摘要，交互应用时建立；
  - 一轮待路由的导入（候选名 `PassiveTurnRouting`）：用户消息、最近几轮，以及上一轮所在的话题。Import Bus 的写入时登记里有上一轮的话题时直接给出，否则给出上一轮的交互标识，由方法查进度取得。
- 提交前先查这一轮的进度，不论话题来源：已应用或超出保留窗口的直接返回“此前已接纳”的结局；应用中的等待到终态或时限，等待的结果（包括应用失败）作为这次的结局交回；只有未接纳或应用失败的才提交，带待路由的导入此时才以 PASSIVE_MEMORY 路由。导入以写入时确定的话题重发时也据此判定重复。查重在路由之前，所以提交队列的冲突规则不变：重新路由可能选中另一个话题，这只发生在应用失败之后，而失败的交互本来就允许换载荷重新提交。
- 然后交给 Patchouli 的“提交并等待应用”（4.4），返回结构化结局。
- Gateway 的异常原样传出，可恢复与否仍由 Import Bus 判断；PASSIVE_MEMORY 返回命令结果属于契约错误。

**为一轮确定话题**：`RootTopicService.determine_topic`（取代 `prepare_topic`）

- 输入：目标 Workspace、交互标识、话题来源（Gateway 的决定，或一轮待路由的导入）、执行者 `system` 的工作 actor。
- 带待路由的导入时，先查这一轮的进度：已应用的返回实际所在的话题；已提交仍在应用的不返回话题，由 Import Bus 以 `apply_pending` 交回；否则路由。
- 然后经 Patchouli 的话题准备公开路由（4.4）确认或建立话题（必要时先按 LRU 驱逐），返回话题标识与是否新建。

**检索命中与存储健康**：`RootMemoryService` 新增两个方法，供任务进程使用：

- 记录检索命中：为一批记忆逐条记录命中，同一批去重，尽力而为，失败只记日志；
- 读取存储健康：返回存储是否可用，供提示词的离线提示。

**删除**：`interaction_status`、`wait_interaction`、`route_passive_turn`、`prepare_topic`。

### 4.4 Patchouli

- **提交并等待应用**：`INTERACTION_SUBMIT` 改为接纳后等待应用到终态或时限，时限为空时等到终态（上限由队列自身的应用时限与重试决定）。结局 `InteractionSubmitResult` 改为携带：交互标识、进度（`succeeded` / `pending` / `failed`）、成功时实际所在的话题、失败时的异常类名，以及这次调用之前是否已经接纳。容量已满或队列已停止仍以 `InteractionAdmissionUnavailableError` 拒绝，可用同一标识重试。
- `INTERACTION_STATUS` 与 `INTERACTION_WAIT` 保留为公开路由，只由 root 的两个方法内部调用（查重、续接上一轮、应用中的等待）。
- **话题准备公开路由**：以 scope 的归属确认或建立目标话题，返回真实话题标识。挂载现有但没有生产调用方的 [`TopicManagementService.prepare_topic`](../../../src/hivememory/patchouli/application/topic_management_service.py)，背后仍是 `TOPIC_PREPARE`。
- **检索命中与存储健康公开路由**：分别包装本地路由 `MEMORY_RECORD_HIT` 与 `RUNTIME_STORAGE_HEALTH`。命中来源的取值改为反映新的记录时点。
- **删除**：`PREPARE_AGENT_RUN`、`FINALIZE_AGENT_RUN`、`CLEANUP_PREPARED_AGENT_RUN` 与 `PreparedAgentRun`；`PatchouliService` 中的 prepare、`retrieve_for_decision`、finalize、续行（`_active_finalizations`、`_detached_finalizations`、`drain_active_finalizations`）、话题核对、命中记录与空话题清理；`patchouli/system.py` 停止时不再 drain 续行。`PatchouliService` 只保留引用记录。

### 4.5 任务进程：PREPARE 阶段

阶段名与停止语义不变，顺序改为：

1. **解析 Profile**：不变。
2. **派生主线程**：`ProcessThreads.derive_main` 从 Actor 阶段移到这里，读取操作快照并复查准入；准入已失效时进程在任何话题副作用之前失败。
3. **预检索**：请求开启了检索、Gateway 的检索计划不是跳过且条数不为 0 时，按计划构造 `RetrieveRequest`（改写后的查询、关键词、条数），经操作入口 `execute` 以主线程提交，与 Alice 的 SEARCH 是同一条路。执行者没有 `memory.search` 时，把 `operation_not_allowed` 的拒绝译为不检索，与现在相同。结果经读取视图交付：逐条做归属校验与交付授权，并预热读取缓存。执行器的 `_retrieval_scope` 删除，`authorize_agent` 只剩 actor API 一个调用位置。
4. **已有话题**：Gateway 指向已有话题时，以主线程经线程表的 `dispatch_topic` 取得派发话题，与第一次写入走同一条路径（4.6）：经“为一轮确定话题”确认话题并冻结，绑定运行事件的话题。进程已收到停止时，预检索与这一步都以 `ThreadEndedError` 被拒绝，执行器按 PREPARE 阶段的取消收口。
5. **读取存储健康**：经 root API。
6. **CPU 分配**：附件租借与编译、记忆编译、提示词与清单，与现在相同；清单的 `topic_id` 取已冻结的派发话题，新话题时为空。
7. **停止检查**：不变。

进入 Actor 阶段后，把同一个主线程交给 CPU。线程从 PREPARE 存在到 Actor 阶段结束；存活判定本来就不读阶段，不变。`InputsAllocated` 改为携带预检索的记忆，`memory_refs` 事件照旧。

### 4.6 任务进程：第一次写入时确定话题

- 线程表端口 `ExecutionThreadTable` 增加异步方法 `dispatch_topic(thread) -> str`，由 `ProcessThreads` 实现：先做存活判定，再返回已冻结的派发话题；还没有时，加入或启动进程的建立任务，等它的结果。
- 建立任务由进程持有，保存在工作集中；同一进程只建立一次。它以进程记录中的系统 root 工作 actor，经“为一轮确定话题”传入工作集保存的 Gateway 决定。成功时冻结派发话题、绑定运行事件的话题；失败时不冻结，错误交回发起等待的写入，下一次写入重新建立。
- 每个写入以 `join_shielded` 各自等待：一个写入被取消，不会取消建立，也不影响其他线程的写入。
- 提交任务先等待进行中的建立结束；没有提交的丢弃路径由关闭流程等待它结束。
- `resolve` 不再要求派发话题，`ResolvedThread.dispatch_topic_id` 删除：READ、SEARCH、RUN、CALL 用不到它，写入改由 `dispatch_topic` 取得。
- `ProcessThreads` 注入 root 话题 API。`workspace.process` 因此开始依赖 `workspace.capability.root`（第 8.7 节）；线程表仍只判定存活与取得派发话题，授权在 root API 的方法入口。

### 4.7 写入的顺序

actor API 的 WRITE 与 UPDATE 统一为：

1. 以工作 actor 授权 `memory.write` / `memory.update`；
2. UPDATE 解析可读的正式基础原子；
3. 取得派发话题：线程提交调用 `dispatch_topic(thread)`，请求级提交调用入口给出的回调；
4. 无挂起点的同步段：线程提交再解析一次线程，登记意图；
5. 派发。

第 3 步的 `await` 位于同步段之前：期间线程结束或进程收到停止时，同步段照常拒绝登记，已经建立的话题没有写入（第 8.2 节）。

请求级提交 `execute_request(request, *, actor, dispatch_topic=None)`：`dispatch_topic` 是一个返回派发话题的异步回调，取代现在的 `dispatch_topic_id`。写入与更新必须给出，检索与读取不得给出，不符时在分派前以 `TypeError` 拒绝。MCP 入口给出的回调调用 `ImportBusService.place_turn`，因此被拒的写入不会触发路由、建立话题或 LRU 结算。

### 4.8 任务进程：提交与收尾

- 封口、进入 FINALIZE 与创建提交任务之间仍然没有 `await`；会话不再在封口时写入。提交任务由进程持有，以 `join_shielded` 等待，依次：
  1. 等待进行中的建立任务结束；
  2. 记录预检索到的记忆的检索命中（尽力而为，不依赖提交的结局）；
  3. 经 root API 提交交互：参与者为执行者，交互标识取 `process_id`，来源为 `active_chat`，不设时限；有派发话题时提交到它，否则提交 `NEW_TOPIC` 并携带 Gateway 的标题与摘要；
  4. 写入会话，话题取实际所在的话题；提交或应用失败时照样写入，话题为空；
  5. 运行事件尚未绑定话题时，绑定实际所在的话题。
- 失败沿用现有错误码 `patchouli.interaction.finalization_failed`：接纳被拒译为接纳阶段，结局为失败译为应用阶段，由进程以 `ActiveInteractionFinalizationError` 抛出，进程按失败收口。
- 核对实际所在的话题删除：提交给出真实话题时应用只写入该话题，给出 `NEW_TOPIC` 时由应用决定。
- 运行结果与 SSE `done.topic_id` 取实际所在的话题，没有提交或提交失败时为空。
- 关闭流程只结束线程、释放租借、等待进行中的建立任务，不再请求清理。
- 系统关闭：关闭顺序（Import Bus → 进程服务 → 会话 → … → Patchouli）已经保证进程的提交在提交队列停止前结束，Patchouli 自己的续行与 drain 不再需要。

### 4.9 Import Bus 与 MCP

- `ImportBusService` 不变：只认证系统自身。编排器仍持有 root 交互 API 与 root 话题 API，不再自己查进度、路由或等待。
- **导入**：在外部会话串行门内，没有输出的轮次直接跳过；写入时登记里有这一轮的话题时以它提交，否则给出待路由的导入；以导入的等待时限经 root API 提交交互，再把结局译为导入结果：
  - 此前已接纳且已应用或超出保留窗口：`duplicate`；
  - 已应用：`accepted`；
  - 等待超时：`apply_pending`；
  - 应用失败：`apply_failed`；
  - 接纳被拒：`queue_unavailable`，写入时登记保留；
  - Gateway 的可恢复异常：`gateway_unavailable`。
- **写入时确定话题**：登记里有的直接返回；否则经“为一轮确定话题”传入待路由的导入，不返回话题时以 `apply_pending` 交回，返回时登记话题标识与是否新建。登记不再保存 `PreparedAgentRun`。
- 导入一轮与写入时确定话题各做一次 root 授权。事件、串行与确认语义不变。
- **MCP**：写入与更新以确定话题的回调经请求级提交（4.7），缺少轮次信息时仍在认证前拒绝。

### 4.10 对外可观察的变化

1. 新话题在第一次写入或交互应用时才进入话题池；没有写入也没有输出的轮次不再产生空话题。
2. LRU 结算推迟到建立话题时，没有建立话题的轮次不再触发结算。LRU 驱逐失败（候选都被占用的 `TopicBusyError`、LRU 结算时生成队列拒绝接纳）从 PREPARE（进程失败、不耗 LLM）后移到两处：第一次写入时，Agent 中途看到写入错误；整轮结束后的应用，回复已经交付，提交以 `finalization_failed` 失败，会话里这一轮的话题为空。
3. 检索命中在 Actor 有输出后记录，不再要求交互应用成功；CPU 失败但有输出的轮次也记录。
4. 会话轮次的 `topic_id` 是交互实际所在的话题，提交失败时为空，会话 HTTP 响应的字段改为可空；FINALIZE 期间会话里还没有这一轮。
5. SSE `done.topic_id` 取实际所在的话题。
6. 新话题时，`chat.run.*` 运行事件在确定话题或应用之后才带话题；`agent.run.*` 事件与 Alice 根帧不带话题。
7. 写入过、但最终没有接收交互的话题不再被关闭流程删除，改由空闲结算或 LRU 回收。
8. 预检索结果逐条做归属校验与交付授权，并预热读取缓存。
9. 准入失效的进程在 PREPARE 开始时失败；PREPARE 期间收到停止时，尚未执行的预检索不再执行。
10. Import Bus：一轮导入的 root 授权从最多 5 次降到 1 次；没有输出的轮次不再先查进度。HTTP 契约与确认语义不变。
11. MCP：被拒的写入不再路由、建立话题或触发 LRU 结算；其余不变。

### 4.11 概念的增删

| 类别 | 内容 |
|:---|:---|
| 删除 | Patchouli 路由 `PREPARE_AGENT_RUN`、`FINALIZE_AGENT_RUN`、`CLEANUP_PREPARED_AGENT_RUN`；`PreparedAgentRun`；Patchouli 的 prepare、finalize 续行、drain、话题核对与空话题清理，以及随之没有使用方的本地路由 `TOPIC_DISCARD_IF_EMPTY` 与 `PerceptionFamiliar.discard_if_empty`；`ActiveInteractionFinalizationError.work_state`；`ImportBusContractError`（两处契约检查移入 root API）；root 方法 `interaction_status`、`wait_interaction`、`route_passive_turn`、`prepare_topic`；工作集的 `prepared`、`hand_off_prepared`、`take_prepared_for_cleanup`；关闭流程的补偿；执行器的 `_retrieval_scope`；`ResolvedThread.dispatch_topic_id`；`execute_request` 的 `dispatch_topic_id` |
| 新增 | root 方法 `determine_topic`、记录检索命中、读取存储健康；Patchouli 公开路由：话题准备、检索命中、存储健康；待路由的导入（`PassiveTurnRouting`）；线程表端口的 `dispatch_topic`；工作集中的 Gateway 决定与建立任务；`execute_request` 的 `dispatch_topic` 回调 |
| 改变 | root `submit_interaction` 接收话题来源、在内部查重与路由并等待应用；`INTERACTION_SUBMIT` 改为提交并等待应用，`InteractionSubmitResult` 改为结局；主线程的派生时点 |
| 改为可空 | `ConversationTurn.topic_id`、会话响应的 `topic_id`、`CPUInputManifest.topic_id` |

新增项都有生产使用方：root 的提交交互与确定话题由任务进程与 Import Bus 使用；检索命中与存储健康由任务进程使用；三条 Patchouli 公开路由由 root 方法使用；待路由的导入由 Import Bus 交给 root 方法；`dispatch_topic` 由 actor API 的写入使用；回调由 MCP 入口给出。

### 4.12 主要流程

```text
controller（之前）
Gateway → Profile → PREPARE_AGENT_RUN（建话题 + 检索 + 存储健康）→ CPU 分配（冻结话题）
  → Actor（派生主线程）→ 封口、写会话 → FINALIZE_AGENT_RUN（接纳、等待、核对、命中、补偿）
  → 关闭时 CLEANUP_PREPARED_AGENT_RUN

controller（之后）
Gateway → Profile → 派生主线程 → 预检索（主线程）→ 存储健康 → [已有话题] 确定话题 → CPU 分配
  → Actor（新话题：第一次写入时确定话题）
  → 封口 → 记录命中 → root.submit_interaction（等待应用）→ 写会话（实际所在的话题）

plugin 导入（之后）
认证 system → [串行门] 无输出则跳过 → root.submit_interaction（查重 → 路由或沿用登记 → 提交并等待）

plugin 写入（之后）
认证 → actor API 授权 → 回调：认证 system → [串行门] root.determine_topic（查重 → 路由 → 确认或建立）
  → 同步段登记 → 派发
```

## 5. 受影响的系统、契约与数据

| 范围 | 变化 |
|:---|:---|
| Patchouli 公开路由 | 删除 prepare、finalize、cleanup 三条；新增话题准备、检索命中、存储健康三条；`INTERACTION_SUBMIT` 改为提交并等待应用 |
| Patchouli 内部 | `PatchouliService` 收缩为引用记录；`patchouli/system.py` 的停止不再 drain；路由桥接随之调整 |
| root API | `submit_interaction` 改形；`determine_topic` 取代 `prepare_topic`；删除三个方法；`RootMemoryService` 新增两个方法 |
| workspace 进程 | 执行器、CPU 分配、工作集、线程表（派生时点、`dispatch_topic`）、输出通道、运行事件绑定；开始依赖 root API |
| workspace 契约 | `ExecutionThreadTable.dispatch_topic`；`ResolvedThread` 不带派发话题；`CPUInputManifest.topic_id` 可空 |
| 操作入口与 actor API | `execute_request` 的回调参数；WRITE/UPDATE 在授权后取得派发话题 |
| System | Import Bus 编排器简化；组合根为线程表注入 root 话题 API |
| MCP | 写入与更新以回调经请求级提交 |
| Alice | 根帧与 `agent.run.*` 事件的话题可空 |
| core 模型 | `ConversationTurn.topic_id` 可空 |
| HTTP 与前端 | 会话响应的 `topic_id` 可空，前端类型同步 |
| 架构守护 | `authorize_agent` 只在 actor API；`authorize_root` 的调用位置随之调整；`workspace.process` 只允许导入能力层中的 root API |

## 6. 兼容、迁移与实施分批

### 6.1 为什么不拆分

拆成“交互提交收束”与“话题按需创建与 prepare 拆分”两份计划时，前一份删除 finalize 后，prepare、cleanup 与 `PreparedAgentRun` 仍然存在：补偿要从 Patchouli 的续行改写到进程的关闭流程，命中要从续行移出，`determine_topic` 要暂时借 `PREPARE_AGENT_RUN` 建话题。这些改写都会在后一份计划里删除，而且中间状态下应用失败后的空话题补偿需要另行设计。两部分共享同一个结果模型与同一组路由，因此作为一份计划实施，按下面的批次分步提交。

### 6.2 兼容与迁移

- 涉及的状态都在进程内：话题工作集、提交队列、会话与工作集都不跨重启，没有数据迁移。
- Patchouli 路由是进程内总线契约，调用方与提供方在同一次改动中切换，不保留旧路由。
- HTTP 只有会话响应的 `topic_id` 改为可空，前端同步调整类型；导入与 MCP 的 HTTP 契约不变。

### 6.3 实施分批

每批保持全量门槛通过：

1. **Patchouli**：提交并等待应用与新的结局；话题准备、检索命中、存储健康三条公开路由。
2. **root API、Import Bus 与 MCP**：`submit_interaction` 改形、`determine_topic`、命中与存储健康两个方法；Import Bus 改用它们；actor API 的写入顺序与 `execute_request` 的回调；MCP 入口切换；删除四个旧 root 方法。
3. **任务进程**：PREPARE 的新顺序（主线程、预检索、存储健康、已有话题）；第一次写入时确定话题；提交改走 root API、会话写入后移、删除补偿；清单与 Alice 根帧的话题可空。
4. **删除**：prepare、finalize、cleanup 路由，`PreparedAgentRun`，Patchouli 续行；测试切换到新路径；架构守护更新。
5. **会话与前端**：会话模型、HTTP 响应与前端类型。
6. **文档收尾**：按第 9 节更新当前文档，计划归档。

## 7. 测试和验收标准

需要守住的不变量与归属层（测试写入归属层的现有文件，不建以计划命名的测试集合）：

| 不变量 | 归属层 |
|:---|:---|
| 提交并等待应用的结局：新接纳、此前已接纳、应用失败、超时；接纳被拒仍以可重试错误交回 | Patchouli 单元（`tests/unit/patchouli/application/test_submission_services.py`、`tests/unit/patchouli/control/test_interaction_submission.py`） |
| 话题准备路由确认已有话题、建立新话题（含 LRU 驱逐）；命中与存储健康路由 | Patchouli 单元（`test_topic_management_service.py` 等现有文件） |
| root 提交交互：带待路由的导入时查重先于路由，已接纳的不调用 Gateway，应用失败的重新路由；上一轮话题按登记或进度续接 | root API 测试（现有 Import Bus 提交集成测试所在文件） |
| root 确定话题：已应用返回实际所在的话题，应用中不返回话题 | 同上 |
| root 新方法只接受 `system`，目标不是驻留 Workspace 时先于路由拒绝 | `tests/integration/workspace/test_root_api_authorization.py` |
| Import Bus：结局到导入结果的映射、写入时登记的沿用与保留、无输出跳过 | `tests/unit/system/services/import_bus/test_ingestor.py`、`tests/integration/system/test_import_bus_submission.py` |
| 写入顺序：授权与基础冷读先于确定话题；被拒的写入不触发确定话题；确定话题期间线程结束时同步段拒绝登记 | actor API 与 MCP（`tests/integration/workspace/test_process_intents.py`、`tests/integration/system/test_mcp_memory_access.py`） |
| 第一次写入时确定话题：并发写入只建立一次；一个写入取消不影响建立与其他写入；失败交回这次写入，下一次重试；READ、SEARCH、CALL 不触发 | `tests/integration/workspace/test_process_threads.py`、`test_process_intents.py` |
| PREPARE：主线程在预检索之前派生；预检索经操作入口，没有 `memory.search` 时不检索；停止后预检索被拒绝并以取消收口；准入失效时在任何话题副作用之前失败 | 任务进程单元（`tests/unit/workspace/process/test_gateway_chat_flow.py`、`test_chat_run_control_contract.py`） |
| 已有话题在 CPU 分配前确定，派发话题从 Actor 开始可用；新话题且没有写入时，Actor 结束前话题池里没有这一轮的话题，提交携带 `NEW_TOPIC` 与 Gateway 的标题、摘要 | 任务进程单元（`test_gateway_chat_flow.py`、`test_cpu_allocation.py`） |
| 会话与 `done` 取实际所在的话题；提交失败时会话仍有这一轮且话题为空，进程以 `finalization_failed` 失败；任何路径都不请求清理 | 任务进程单元（`test_gateway_chat_flow.py`、`test_chat_run_control_contract.py`、`test_cpu_port.py`） |
| 检索命中在 Actor 有输出后记录，不依赖提交的结局；记录失败不影响本轮 | 任务进程单元 |
| 交付方离开后提交照常完成；系统关闭在进程的提交结束后才停止提交队列 | 任务进程单元与 System 生命周期测试（`tests/unit/system/test_lifecycle.py`） |
| 停止、取消与 FINALIZE 拒绝停止的语义不变 | 任务进程单元（现有用例） |
| `authorize_agent` 只在 actor API 调用；`authorize_root` 的调用位置符合新结构；System 包内没有授权调用 | 架构测试（`tests/unit/architecture/test_access_boundaries.py`） |

验证命令：

```bash
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85
ruff check . && black --check .
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
cd frontend && npm ci && npm run lint && npm run build
git diff --check
```

需要真实 Qdrant 与模型的 e2e（`tests/e2e/system/` 的 chat 与导入用例）在环境可用时运行。手动验收在重启后的 live 后端上进行：一轮有写入、一轮没有写入的新话题对话，一轮续接已有话题的对话，一次 dsh 导入与写入时确定话题，以及一次没有 `memory.write` 的 Agent 经 MCP 写入；分别核对话题池、会话的话题与 `done.topic_id`。

## 8. 风险与待决问题

### 8.1 检索命中的时点与范围

默认在 Actor 有输出后、提交之前记录，保留“本轮真正用上了”的含义，不依赖提交的结局；只记任务进程的预检索，保持现在“只有 chat 的预检索计命中”的范围。检索后立即记录会把失败、取消与重试的轮次也计入；由检索路由对所有检索记命中（含 MCP 与 Alice 的 SEARCH）会改变生命周期的强化信号，影响面超出本计划。评审时可以调整。

### 8.2 建立后没有接收交互的话题

以下窗口会留下没有交互的话题：确定话题之后线程结束或进程收到停止、派发接纳被拒或结果未知、写入后 CPU 没有输出、导入的轮次不再回流。这些话题不是 Gateway 的候选，由空闲结算、LRU 或关闭结算按无材料删除，不触发生成。建立后、应用前的新话题如果先被删除，应用会因话题不存在而失败；这个窗口比现在预建话题时短。

生成侧按话题登记的主动写入（`_topic_write_memory_ids`）只在该话题结算生成时取出，按无材料删除不会清除它，留下一条登记，是内存泄漏，不丢写入。现在的清理路径同样如此，本计划不处理，作为发现登记。

### 8.3 已有话题在长时间运行中被回收

已有话题在 PREPARE 刷新驻留，与现在相同。运行时间超过空闲时限时，话题仍可能在提交前被结算，应用会因话题不存在而失败，与现状一致。

### 8.4 新话题提交的排序键

提交的排序键是 `topic:<requested_topic_id>`，所有 `NEW_TOPIC` 提交共用 `topic:NEW_TOPIC`，按键串行；chat 的新话题提交以后会与导入的排在同一个键上。应用耗时很短，本计划不处理，需要时可改为按交互区分。

### 8.5 结算失败的错误类型

进程沿用 Patchouli 契约中的 `ActiveInteractionFinalizationError` 及其错误码，对客户端可见的契约不变。这个类型以后由 workspace 抛出；如果认为它应当归属 workspace，评审时可以改为 workspace 契约中的同码错误。

### 8.6 确定话题失败交回写入的错误

确定话题失败时，Patchouli 的结构化错误（例如 `TopicBusyError`）原样交回这次写入，与现在 MCP 写入时确定话题失败的处理相同。实施时核对 MTP 的工具结果与 MCP 的错误投影都能把它呈现为可读的结构化错误。

### 8.7 任务进程开始依赖 root API

`workspace.process` 现在不导入能力层；按决定 6，任务进程的提交、确定话题、命中与存储健康都改经 root API，进程随之依赖 `workspace.capability.root`。两个 API 目录互不导入的规则不受影响，root API 也不导入进程。

### 8.8 短期记忆的形态

决定 9 的前提是短期记忆的单位与形态尚未确定。将来改为按会话时间线切段时，需要重做的是 4.2 的建立时点与 4.6 的建立任务，以及 Import Bus 的写入时确定话题；root 的提交交互、Patchouli 的提交并等待应用与 prepare 的拆分不受影响。

### 8.9 发现但不处理

- `ExecutionFrame.topic_id` 只被没有调用方的 `is_transient()` 与 `__repr__` 读取。
- `NEW_TOPIC` 是散落在多个包里的字符串字面量，没有共享常量。
- 任务进程 Idea 2.2 等待的“Profile 回到 CPU 分配”的前提随本计划成立，移动时点不在本计划范围。

## 9. 完成时必须更新的当前文档

- **AGENTS.md** §3、§4：主动链路改为“Gateway → Profile → 派生主线程与预检索 → CPU 分配 → Actor → 提交交互”；Import Bus 链路经 root 的两个方法；MCP 写入先授权再确定话题；去掉 cleanup 与 prepare 建话题的表述。
- **workspace**：
  - [任务进程](../../workspace/task-process.md)：阶段与授权表、工作集、提交与关闭；
  - [执行线程](../../workspace/execution-threads.md)：主线程的派生时点、线程寿命、`resolve` 与派发话题；
  - [读取视图与写入意图](../../workspace/read-view-and-write-intents.md)：写入的顺序与派发话题的两个来源；
  - [会话资源](../../workspace/sessions.md)：轮次话题可空、写入时点；
  - [能力层](../../workspace/capability.md)：root 的两个交互方法与两个记忆方法、`execute_request` 的回调；
  - [身份与授权](../../workspace/identity-and-authorization.md)：授权调用位置。
- **contracts**：[子系统公共契约](../../contracts/subsystem-contracts.md)；[路由与事件](../../contracts/routes-and-events.md)（Patchouli 路由与 `done.topic_id`）；[MCP](../../contracts/mcp.md)（写入的顺序）；[MTP](../../contracts/mtp.md)（检索命中）；[错误模型](../../contracts/error-model.md)。
- **system**：[Import Bus](../../system/import-bus.md) 第 4 节的编排、进度查询与重发、等待应用与写入时确定话题；[应用服务](../../system/application-services.md)；[组合根](../../system/composition.md)。
- **patchouli**：[README](../../patchouli/README.md)；[感知](../../patchouli/perception.md)（建立话题的时点、不再清理）；[检索](../../patchouli/retrieval.md)、[生命周期](../../patchouli/lifecycle.md)、[生成](../../patchouli/generation.md)、[记忆库](../../patchouli/memory-library.md) 中的命中、prepare 与 finalize 表述。
- **其他当前文档**：[运行时与总线](../../components/runtime-and-bus.md)；[PendingAtom](../../alice/pending-atom.md)；[系统架构概览](../../architecture/overview.md) 第 5、6 节；[边界](../../architecture/boundaries.md)；[数据模型](../../architecture/data-model.md)；[Chat 视图](../../frontend/chat-view.md)；[身份与执行安全](../../governance/security/identity-and-execution-safety.md)；[PROJECT](../../PROJECT.md)。
- **Idea 与索引**：
  - [外部会话与 Topic 投影 Idea](../../ideas/external-session-and-topic-projection.md) §0、§4、§8、§10：记录实施；
  - [任务进程 Idea](../../ideas/task-process-table-and-registration-entry.md) §2.1、§2.2；
  - [Chat 运行生命周期 Idea](../../ideas/chat-run-lifecycle-follow-ups.md)：cleanup 已删除；
  - [外部 Actor Idea](../../ideas/external-actor-registration-and-runtime-access.md)、[长时间运行 Agent Idea](../../ideas/long-running-agent-intra-turn-context-folding.md) 中预建话题的表述；
  - [Ideas 索引](../../ideas/README.md)、[Plans 导航](../../plans/README.md)、[归档计划索引](./README.md)、[ROADMAP](../../ROADMAP.md) 身份与访问体系一行。

## 10. 实施记录

**实现**（2026-11-06，分支 `docs/interaction-submission-plan`）：第 6.3 节的六批在同一个工作区内完成，提交时再整理。

与第 4、5、7、9 节的出入：

- **提交前的查重**：提交交互在任何话题来源下都先查进度，不只是带待路由的导入，导入以写入时确定的话题重发时也能判定重复；第 4.3 节已同步改写。
- **已有话题的确认**：PREPARE 中以主线程经线程表的 `dispatch_topic` 确认，与第一次写入走同一条路径，顺序为预检索 → 已有话题 → 存储健康；第 4.5 节已同步改写。
- **CPU 清单**：`topic_id` 改为可空，新话题时为 `None`；Alice 根帧原样透传，不再转成空字符串。
- **随之删除的概念**：
  - 补偿删除后没有使用方的本地路由 `TOPIC_DISCARD_IF_EMPTY` 与 `PerceptionFamiliar.discard_if_empty`；
  - `ImportBusContractError`：它的两处契约检查移入 root API，改抛 `RuntimeError`；
  - `ActiveInteractionFinalizationError.work_state`；
  - 第 4.11 节已补上这些项。
- **架构守护**：
  - `test_workspace_process_does_not_import_capability` 改为 `test_workspace_process_imports_only_root_capability_api`，任务进程在能力层中只允许导入 root API，这是决定 6 的代价；
  - `authorize_agent` 的调用位置去掉执行器；
  - `scripts/typecheck_bus_routes.py` 的位置参数示例改用 `TOPIC_EVICT`。
- **实施后审计（owner，2026-11-06）**：交互提交队列中失去生产调用方的 `is_accepted`、`started`、`stopped`、`pending_count` 删除，按排序键排空的 `drain` 并入 `drain_all`；早于本计划、没有任何地方抛出的 `TransientInteractionSubmissionError` 一并删除，测试改用生产中真实出现的 `TopicBusyError` 触发重试。
- **写入时确定话题**：这一轮已经应用时，root API 返回它实际所在的话题，编排器同样登记。原实现不登记；这类登记与从未回流的轮次一样，留到进程重启。
- **第 9 节漏列、一并更新的文档**：
  - 根 README；
  - [Chat 附件链路](../../system/attachments.md)、[配置](../../system/configuration.md)；
  - [Gateway 分析](../../gateway/analysis.md)、[Gateway 工作流](../../gateway/workflow.md)；
  - Alice 的 [README](../../alice/README.md) 与[编排](../../alice/orchestration.md)；
  - [记忆编译](../../patchouli/memory-compiler.md)、[Artifact](../../patchouli/artifacts.md)；
  - [幂等与重试](../../governance/reliability/idempotency-and-retry.md)；
  - [MealAssistant 规格](../../applications/MealAssistantProductSpec.md)；
  - 前端 `chatStore.ts` 的一处注释。

**测试**：

- 删除：
  - `tests/unit/patchouli/test_service_gateway_decision.py`：守护 prepare/finalize；
  - `tests/integration/patchouli/test_active_interaction_submission.py`：守护 finalize 续行；
  - 两者中仍有效的不变量（队列停止、取消不撤回、应用失败、容量拒绝、命中尽力而为）移入 `tests/unit/patchouli/application/test_submission_services.py` 与新的 `tests/unit/patchouli/test_service.py`；队列的默认策略用例移入 `tests/unit/patchouli/control/test_interaction_submission.py`；
  - cleanup、`discard_if_empty` 与旧 root 方法的用例删除。
- 测试辅助：
  - `tests/helpers/process.py` 按新的组合根装配；
  - 新增 `PatchouliTurnStub`，作为进程一轮所用的 Patchouli 公开路由的替身；
  - `tests/helpers/chat_handoff.py` 删除 `make_prepared_run`。
- 第 7 节的不变量写入各归属层的现有文件：
  - root API：`tests/unit/workspace/capability/root/`、`test_root_api_authorization.py`；
  - Import Bus：`test_ingestor.py`、`test_import_bus_submission.py`；
  - MCP 与请求级提交：`test_mcp_memory_access.py`、`test_operation_requests.py`；
  - 线程与写入顺序：`test_process_threads.py`、`test_process_intents.py`；
  - 任务进程：`tests/unit/workspace/process/`；
  - 会话：`test_session_completion.py`、`test_conversation_sessions.py`；
  - 系统关闭顺序：`tests/unit/system/test_lifecycle.py`；
  - 话题准备：`tests/integration/patchouli/test_topic_access_chain.py`。
- 进程单元层与线程集成层用运行时替换实现的方式做过变异检查（没有改动源码）：每个注入的缺陷都至少让一个用例失败。

**验证**：

- `pytest tests/unit tests/integration -m "not live_llm and not e2e and not slow" -n auto --cov=hivememory --cov-fail-under=85`：3221 通过，覆盖率 94.81%；
- `ruff check .`、`black --check .`、`mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py`（445 个文件）通过；
- 前端 `npm run lint`、`npm run build` 通过；
- `git diff --check` 通过，改动文档的相对链接与锚点全部有效；
- `pytest tests/e2e -m e2e`：20 通过，1 失败。失败的 `test_agent_loop_python_repl_computes`（`live_llm`）断言模型调用工具，真实模型直接作答；改动前的代码同样失败两次，与本计划无关。

**未完成与遗留**：

- 第 7 节的手动验收要用这个分支重启正在运行的后端，未进行。
- 前端：收到 `finalizing` 后断流时，前端立即重新读取会话历史并提示“本轮已保存”。会话现在在提交结束后才写入，提交尚未结束时读到的历史可能还没有这一轮。文档与注释已改正，前端行为未改。
- `ActiveInteractionFinalizationError` 不是 `WorkspaceDomainError`，流式交付的 `error` 事件不带错误码，与改动前相同。
- 提交与关闭流程等待进行中的话题确定：只在 CPU 留下仍在运行的写入时才会出现，没有单元用例。
- `determine_topic` 遇到超出保留窗口（进度为 `unknown`）的一轮会重新路由并建立话题，而提交交互把它当作此前已接纳，与旧实现相同。
- 可合并的重复用例：`test_gateway_chat_flow.py` 与 `test_cpu_port.py` 中进程完成并提交执行结果的用例。
- 第 8 节的默认值（命中时点与范围、错误类型归属、确定话题失败的错误投影）未经 owner 评审。
