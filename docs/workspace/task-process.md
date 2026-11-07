---
title: 任务进程
status: current
owner: workspace
scope: task-process-registration-table-orchestration-delivery-cancellation-and-close
code_paths:
  - src/hivememory/workspace/process/
  - src/hivememory/workspace/capability/root/interaction.py
  - src/hivememory/workspace/capability/root/topic.py
  - src/hivememory/workspace/contracts/cpu.py
  - src/hivememory/workspace/contracts/process.py
  - src/hivememory/core/protocol/sealing.py
  - src/hivememory/server/routers/chat.py
  - src/hivememory/prompts/compilation.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
related_ideas:
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/chat-run-lifecycle-follow-ups.md
related_docs:
  - docs/workspace/identity-and-authorization.md
  - docs/workspace/capability.md
  - docs/workspace/execution-threads.md
  - docs/workspace/sessions.md
  - docs/system/application-services.md
  - docs/system/attachments.md
  - docs/system/import-bus.md
last_reviewed: 2026-11-06
---

# 任务进程

任务进程是 workspace 运行一次任务请求的单位：请求经唯一注册入口登记为进程，此后 Gateway 分析、预检索、CPU 分配、Actor 执行与交互提交都在进程内进行，停止与关闭因此覆盖整条链路，中间产物与工作 actor 也都有明确的持有者。当前只有主动 chat 经注册入口进入；外部 harness 中已经结束的交互经 System 的 Import Bus 导入，不创建任务进程，但与任务进程共用交互记录的封口函数（第 8 节、[Import Bus](../system/import-bus.md)）。

这一层存在，是因为 HTTP、SSE 与未来的外部 adapter 都需要共享同一条主动 chat、取消与提交链路。若每个入口自己拼装 Gateway、Patchouli 和 Alice，就会再次出现多套预检索与提交、错误处理和取消语义。任务进程回答“这次请求应该按什么顺序跨边界运行”，而不回答“记忆如何检索”“Agent 如何生成”或“Gateway 如何分析”。

## 1. 唯一注册入口与三个角色

`TaskProcessService` 是任务请求的唯一注册入口，注册与运行分开：`register_process()` 完成认证、认领会话并创建进程、返回进程句柄（第 2 节），`run_process(handle)` 以流式交付运行已注册的进程（第 11 节）；注册参数 `message` 是交给 Gateway 分析的指令文本。

任务进程由三个角色组成，按“谁持有什么”划分：

| 角色 | 持有 | 不持有 |
|:---|:---|:---|
| `TaskProcessService`（注册入口，`workspace/process/service.py`） | 生命周期依赖：认证网关与组合根注入的系统自身的 principal（注册时的两次认证）、操作授权者（取消的 root 授权）、事件 emitter、进程表（组合根创建，与进程线程管理共享）、执行器 | 总线、CPU 端口、CPU 分配器、asset reader、编译配置等编排依赖 |
| `TaskProcess`（进程状态容器，`workspace/process/task_process.py`） | 本进程独有的东西：进程记录（含执行者与系统的 root 工作 actor、观测标签与绑定了本进程关联字段的观测通道）、任务参数、工作集、线程表、`trace_id`、进程 task（本进程自己的执行流）与输出通道 | 任何跨进程共享的依赖 |
| `TaskProcessRunner`（执行器，`workspace/process/runner.py`） | 所有进程共用的编排依赖：全局总线、CPU 端口、会话服务、`CPUAllocator`、进程线程管理 `ProcessThreads`、操作入口（预检索）、root 交互 API 与 root 记忆 API（交互提交、检索命中与存储健康）、操作授权者（Gateway 的 root 授权）、Gateway 超时与会话轮数配置；唯一的编排骨架 `run(process)` 与唯一的关闭流程 `close(process)`，二者都运行在进程 task 中 | 任何进程的状态 |

这样划分是为了让注册入口只管理进程的生命周期，而不认识具体流程，并让进程本身只是状态容器：进程记录与工作集，由所有进程共用的四阶段骨架驱动（理念见[第五代架构 Idea](../ideas/fifth-generation-architecture.md)第 3.2 节）。执行器与 `CPUAllocator` 由组合根构建并注入注册入口，入口不转交编排依赖。任务进程在能力层中只导入 root API：话题确认与建立、存储健康、检索命中与交互提交都是系统级操作，由 root API 在方法入口授权；预检索是执行者的 actor 操作，经操作入口的公共契约 `OperationEntry` 以主线程提交，与 CPU 一侧走同一条路，进程不取得 actor API 或操作入口的实现（架构测试守护）。执行器、CPU 分配与进程线程管理不导入认证一侧，子线程的被调用方经 `core.access` 的准入端口做准入（[Workspace](./README.md#4-包结构与依赖规则)）。

## 2. 注册

注册在流式响应开始之前完成：

1. 拒绝保留的 `system` 作为任务请求的 actor：任务进程必须由具体 Agent 执行；
2. 经认证网关认证两次：以入口的 principal 认证执行者 (user, 具体 Agent)，再以组合根注入的系统自身的 principal 与 `local` adapter 认证同一用户的 `system`，得到系统的 root 工作 actor（[认证与访问登记](./authentication.md#5-系统自身的-principal)）；任一认证失败直接抛出，HTTP 入口据此返回 403，不创建、不登记进程；
3. 使用请求 `session_id` 查找本 Workspace 会话，缺省则自动创建；同步检查容量与单进程认领，失败时回滚本次自动创建的空会话（[会话资源](./sessions.md#3-认领与容量)）；然后以通过认证的声明创建一次冻结的 `ExecutionLabels(agent_id, workspace_id)`，创建进程记录（写入两个工作 actor 与标签，并以标签绑定 `chat.run.*` 的事件发布器）与 `TaskProcess`，登记到进程表；认证通过后、登记完成前失败时只还原会话的认领与新建，工作 actor 是局部值，不登记就不留下任何状态；
4. 签发并返回不透明的进程句柄（只读 `process_id/session_id`）；关闭时在进程注销后释放会话认领。

`process_id` 由 server 入口在进入编排服务前生成并冻结。执行者与系统分开认证，是因为进程里的操作分属两个执行者（第 5.2 节）：执行者只做 actor 操作（主线程的预检索与 CPU 提交的请求），系统执行 Gateway、Profile、附件、话题确认与建立、存储健康、检索命中与交互提交，交互中的 Agent 作为参与者显式传递（[身份与授权](./identity-and-authorization.md#4-操作授权)）。

执行标签由进程记录持有：事件发布器以它绑定关联字段，会话轮次的 `agent_id` 取自它，CPU 分配把同一标签对象传给 `CPUInputManifest.labels`，供运行级观测（Alice 的 `agent.run.*`）与历史角色匹配使用；执行线程的标签由线程表放进线程描述（[执行线程](./execution-threads.md#5-主线程交给-cpu)）。标签只服务历史角色匹配、展示回退与观测，不能授权或选择资源目标。

## 3. 进程表、进程记录与句柄

**进程表**（`ProcessTable`，`workspace/process/table.py`）是唯一的进程注册表，也是 workspace 的进程内共享设施：以 `process_id` 为键登记 `TaskProcess`，注销只移除同一个进程对象；进程记录作为进程的控制面经进程取得，不单独登记。进程记录（`ProcessRecord`）有：

- `process_id`：进程唯一标识；
- `actor`：执行者（主工作 actor），主线程以它执行 actor 操作（含预检索），交互提交以它作为交互的参与者；
- `system_actor`：系统的 root 工作 actor，系统阶段以它经 root 授权执行；
  两者只在进程内交给操作授权者、线程表与 CPU 分配，不进入 CPU 可见的对象、事件或会话记录；
- `labels`：注册时创建的冻结 `ExecutionLabels`，事件、会话记录、CPU 分配与线程描述共用；
- `events`：以 `labels` 绑定关联字段的事件发布器；
- `phase`：`created/gateway/prepare/actor/finalize/terminal`（`actor` 即 Actor 执行阶段）；
- `outcome`：`running/stop_requested/cancelled/completed/failed`；
- 首次接受的 `stop_reason`；
- 仅在 Gateway 或 Actor 执行阶段存在的 `active_task` 身份引用。

进程表不保存 `Event`、Token 或 waiter。进程单独设记录一层，是因为停止请求、事件发布、阶段 task 绑定与系统阶段的授权只读写控制状态，而容器还要容纳生命周期各不相同的任务参数、工作集、task 与输出通道。

**进程句柄**（`ProcessHandle`）是入口 adapter 与已注册进程之间唯一的引用：只暴露 `process_id` 与 `session_id`，不暴露进程记录、其中的工作 actor 或 `TaskProcess`。句柄只由注册入口签发，按对象身份判定有效：句柄私下记着它对应的进程，注册入口解析时要求进程表中登记的正是这一个。按 `process_id` 重新构造的对象、进程关闭后的旧句柄都不是有效句柄——运行时以 `process_handle_unknown` 拒绝，关闭是空操作。持有有效句柄即为该进程生命周期的所有者。

## 4. 执行流与输出通道

进程持有自己的执行流：交付方首次运行进程时，注册入口为它创建进程 task，阶段推进不依赖交付方读取。骨架只产出类型化的阶段产出（`workspace/process/outputs.py`），写入进程持有的输出通道 `ProcessOutputChannel`；交付方只订阅通道，由流式交付把阶段产出投影为 SSE 事件。编排途中的异常先记录失败终态，再由流式交付翻译为 `error` 事件（Workspace 领域错误携带安全文案与错误码，其余统一为系统错误）。

输出通道只允许一个订阅者，订阅者看到的顺序是阶段产出、终态产出、结束标记；结束标记在进程关闭完成之后才写入，因此正常结束时 SSE 响应在进程关闭之后结束。只有交互事件受容量限制（256 项）：订阅者在线而通道写满时进程等待，背压一直传到 CPU；终态产出与结束标记不受限，订阅者再慢也不会卡住进程收尾。订阅者暂停时，进程最多先行一个通道容量。订阅者离开后，通道唤醒阻塞的写入并丢弃此后的产出。

## 5. 四阶段骨架

### 5.1 主动链路

```text
TaskProcessService.register_process()：认证、创建进程、登记进程表，返回进程句柄
TaskProcessService.run_process(handle)：创建进程 task；流式交付订阅输出通道
进程 task -> TaskProcessRunner.run(process)
  -> Gateway 的 root 授权 -> Patchouli models.prefetch（立即返回；失败只记录日志）
  -> Gateway public process (ACTIVE_CHAT)
  -> command: return command outcome
  -> decision: PREPARE（不可中断）
       -> 解析执行者的 Agent Profile（workspace 读取视图，管理读取）  root 授权
       -> 以执行者与图纸派生主线程（读取操作快照、登记线程表）
       -> 预检索：按 Gateway 的检索计划，经操作入口以主线程提交 RetrieveRequest  actor API 授权 memory.search
       -> Gateway 指向已有话题：以主线程取得派发话题，root API 确认话题并冻结  root 授权
          Gateway 指向新话题：此时不建立话题
       -> root API 读取存储健康  root 授权
       -> CPU 分配：附件租借（root 授权）与编译、记忆编译、中立提示词编译、组装 CPUInputManifest
       -> 进入 Actor 之前检查停止
  -> Actor 执行：把主线程交给 CPU，在可中断子 task 中 await CPU 端口 execute
       （CPUInputManifest 与主线程的 ExecutionThread；交互事件经 events sink 写入输出通道，终态结果作为返回值）
       新话题的第一次写入：线程表经 root API 建立话题并冻结  root 授权
       -> Actor 阶段结束：同步结束全部线程，接受完成的历史压缩检查点
  -> 有输出的 completed/cancelled/failed：封口交互记录 -> 进入 FINALIZE -> 创建并持有提交 task
       -> 等待进行中的话题确定结束
       -> root API 为预检索的记忆记录检索命中（尽力而为）  root 授权
       -> root API 提交交互并等待应用（interaction_id 取 process_id 值，参与者为执行者）  root 授权
       -> 写入会话（话题取交互实际所在的话题，没有应用成功时为空）
  -> 无输出、异常、FINALIZE 前交付方离开：不提交、不写会话
  -> 终态产出写入输出通道
  -> 关闭：TaskProcessRunner.close(process)
       -> 同步结束仍在运行的线程（兜底），经 CPUAllocator 释放附件租借
       -> 等待进行中的话题确定结束
  -> 注册入口在进程 task 的 finally 中从进程表注销、释放会话认领，最后结束输出通道
```

调用 Gateway 之前，进程请求 Patchouli 在后台预取推理模型：路由只开始加载、立即返回，模型加载与 Gateway 的 LLM 调用并行，空闲卸载之后的重新加载因此尽量不落在预检索的路径上（[Patchouli](../patchouli/README.md#41-模型驻留与空闲卸载)）。预取在 Gateway 的 root 授权之后发出，准入已失效的进程不触发加载；Gateway 判定为命令或不需要检索时模型同样会加载，这部分内存由空闲卸载回收。预取不涉及 Workspace 资源，不是阶段调用，不携带 scope；预取失败只记录日志，不影响进程。

Gateway 阶段从会话取最近 `gateway.context_preparation.session_turn_limit` 轮的用户与助手文本，连同每轮在会话中记录的所在话题 `topic_id`（交互没有应用成功的轮次为空），随 `session_turns` 交给 Gateway（[Gateway 分析](../gateway/analysis.md)）。所在话题只在话题路由失败时使用：Gateway 续接上一轮的话题（该话题仍在本次的候选话题中时），否则新建话题；同一会话的相邻轮次大多仍在同一话题上，续接比每次失败都新建无标题话题归错的代价小。

### 5.2 阶段调用

阶段调用是进程在四个阶段中跨边界的调用，每次调用都以本进程注册时通过认证的 Workspace 为目标、按真正执行它的身份授权，授权返回的 `IdentityScope` 只交给紧随的路由或端口（规则与理由见[身份与授权](./identity-and-authorization.md#4-操作授权)）：

| 阶段调用 | 执行者 | 授权 |
|:---|:---|:---|
| Gateway 的话题读取 | 系统 | 执行器在调用前 `authorize_root(system_actor, 目标)`：话题属于 Workspace，与 Agent 无关 |
| Profile 解析、附件租借 | 系统 | CPU 分配在副作用前 `authorize_root(system_actor, 目标)`：两者是用户的选择，对用户可见 |
| 话题确认与建立、存储健康、检索命中、交互提交 | 系统；交互提交时执行者作为参与者 | 经 root API（`RootTopicService.determine_topic`、`RootMemoryService.storage_available` 与 `record_retrieval_hits`、`RootInteractionService.submit_interaction`），由方法入口 `authorize_root(system_actor, 目标)` |
| 预检索 | 执行者 | 经操作入口以主线程提交 `RetrieveRequest`，actor API 以 `authorize_agent(执行者, memory.search, 驻留 Workspace)` 授权；执行者没有 `memory.search`（`operation_not_allowed`）时不检索，其他授权失败（如准入已失效）照常使进程失败 |

是否预检索由执行器判定：请求关闭了记忆检索（`enable_memory_retrieval`）、Gateway 的检索计划为跳过或要求零条时不提交；否则按计划构造 `RetrieveRequest`（改写后的查询、关键词与条数），经操作入口以主线程提交，与 Alice 的 SEARCH 走同一条路。系统阶段不检查 actor 操作白名单，因此没有任何 actor 操作的 Agent 也能完成对话；它只是不预检索，对话照常进行。检索按执行者的可见性进行，结果经读取视图逐条做归属校验与交付授权，并预热读取缓存：预检索结果中不会出现其他 Agent 的 PRIVATE 记忆（[读取视图与写入意图](./read-view-and-write-intents.md#1-读取视图)）。预检索以主线程提交而不由执行器自行授权，是为了让 `authorize_agent` 只在 actor API 一处调用；预检索也因此共用线程的存活判定，停止被接受后以 `ThreadEndedError` 被拒绝（第 12 节）。

scope 不保存在工作集中等待使用，也不进入 CPU 输入清单、线程描述或任何 CPU 可见的对象。经 root API 的阶段调用由执行器或线程表交出进程记录的系统 root 工作 actor 与目标 Workspace，root 方法在入口授权一次，方法内部依次调用的公开路由不再各自授权；Workspace 归属由资源 owner 校验，进程不另行比对自家子系统的返回值。授权失败沿异常路径按进程失败收口；提交任务中的授权失败不影响已接纳写入。两个授权方法在 `workspace.process` 中的调用位置由架构测试锁定：`authorize_root` 只在执行器（Gateway）、CPU 分配（Profile 与附件）与注册入口（取消）中调用，`authorize_agent` 不在任务进程中调用。

### 5.3 话题何时确定

进程这一轮只有一个话题，即工作集中的派发话题。它不在 PREPARE 预先建立，而是在需要时由线程表确定，同一进程只成功确定一次（[执行线程](./execution-threads.md#8-派发话题)）：

- **Gateway 指向已有话题**：PREPARE 中，执行器在预检索之后以主线程向线程表取得派发话题，root API 确认话题存在并刷新驻留，派发话题随即冻结，此后的 `chat.run.*` 事件带这个话题；输入清单的 `topic_id` 取它。确认失败（例如话题在 Gateway 之后已被结算）时进程在 PREPARE 以失败收口。
- **Gateway 指向新话题**：PREPARE 不建立话题，输入清单的 `topic_id` 为空。本轮第一次写入（主线程或 CALL 子线程的 WRITE、UPDATE）时，线程表以 Gateway 给出的标题与摘要建立话题并冻结；本轮没有写入时，提交携带 Gateway 的新话题决定（`NEW_TOPIC` 与标题、摘要），话题在交互应用时建立，与导入相同。
- 建立话题时话题池已满，先按 LRU 结算一个已有话题（[感知](../patchouli/perception.md)）；只有真正建立话题的轮次才会触发 LRU 结算。

这样安排的理由：写入派发需要一个已经存在的话题，而话题只有被写入或接收交互之后才有内容。预先建立话题会让没有写入也没有输出的轮次留下空话题，提交还要依赖准备阶段的结果去补偿它们；按需确定之后，进行中的一轮只有线程表一处确定话题，没有写入的新话题交给交互应用建立，交互提交不依赖准备阶段，任何路径都不补偿或清理话题。确定之后最终没有接收交互的话题（确定话题之后线程结束或进程收到停止、派发被拒、写入后 CPU 没有输出）留在话题池中，由空闲结算或 LRU 回收。

新话题的建立失败（例如 LRU 结算时候选话题都被占用）随时点落在两处：第一次写入时，错误交回这次写入、不登记意图，Agent 在工具结果中看到它，下一次写入重新建立；交互应用时，回复已经交付，提交以失败结束（第 8 节），会话里这一轮的话题为空。

## 6. CPU 分配

CPU 分配（Profile 解析、附件租借与编译、记忆与中立提示词编译、清单组装）由 `CPUAllocator`（`workspace/process/allocation.py`）完成，它取得的附件租借也由它在进程关闭时释放。

**Profile 解析**：Agent Profile 属于 CPU 分配，但在 PREPARE 开始时解析：主线程以这份图纸派生，预检索与已有话题的确认都由主线程执行，Profile 缺失的请求因此在检索与刷新话题之前失败。加载哪份图纸取自注册时认证得到的执行者，不取自请求输入。Profile 经 root 授权后由组合根注入的读取视图（`ProfileResolver`）按管理语义解析，不经能力层：只检查 Workspace 归属，不按 Agent 的可见性裁剪，指向其他 Agent 的 PRIVATE 图纸也能被它自己的进程加载（[读取视图与写入意图](./read-view-and-write-intents.md#1-读取视图)）；正式原子变更会失效解析缓存。

**编译与清单**：CPU 分配接收预检索返回的未编译原子与经 root API 读取的存储健康。进程用共享引擎 `MemoryCompiler` 把检索结果编译为 `RETRIEVAL_CONTEXT` 文本，用 `AttachmentCompiler` 编译附件并得出实际使用的附件，再由 `prompts.compilation.compile_prompt_input` 按 Profile 语言（缺省时使用进程默认值）定稿中立提示词：环境通知（存储不可用时的离线提示由存储健康决定）与人设组成 `instructions`，记忆与附件组成 `turn_context`。两段文本、解析好的 `language`、话题 `topic_id`、会话全量历史 `session_history` 与已有 `compaction_checkpoint` 一起组装为输入清单，经 CPU 端口交给 CPU。清单字段见[子系统契约](../contracts/subsystem-contracts.md#41-cpu-端口)。

已解析的 Profile 只在编译人设时使用，不进入清单：图纸与操作快照随主线程的线程描述交给 CPU。清单回答“处理什么”，线程描述回答“谁来处理”。

编译放在进程一侧，是为了让不同执行者共用同一份已定稿的 HiveMemory 提示词内容，而不必各自调用引擎或复刻人格、环境通知与材料的拼装。中立编译只依赖 core、i18n 与提示词片段，不接收 Alice 或其他 CPU 的配置；工具教学、历史重放与压缩归执行者，Alice 只原样放置固定文本。进程不掌握 CPU 的模型窗口，固定输入超窗仍由 CPU 在生成前拒绝。

CPU 分配不确定话题：清单的 `topic_id` 只读取工作集中已冻结的派发话题，只用于根帧挂载与观测，Gateway 指向新话题时为空（第 5.3 节）。

## 7. Actor 阶段

Actor 执行经 CPU 端口完成：执行器由组合根注入一个 `CPUPort`（`workspace.contracts`，当前唯一的实现是 Alice 的 `AliceCPU`），进程只调用端口的 `execute(manifest, *, thread, model_selection, events)`，不出现任何具体 CPU 的路由名或结果类型。CPU 经 `events` sink 逐项推送交互事件，进程把它们写入输出通道；终态结果 `CPUExecutionResult` 是 `execute` 的返回值。端口与结果的契约见[子系统公共契约](../contracts/subsystem-contracts.md#4-cpu-端口与-alice-实现)第 4 节。端口定义在 workspace、由 CPU 实现，是为了让 CPU 可以替换而不改动进程：Alice 之外的 CPU（测试中的 `ScriptedCPU`）能跑完整个任务进程。

主线程在 PREPARE 开始时派生（Profile 解析之后、预检索之前）：执行器把图纸交给 `ProcessThreads.derive_main(process, profile)`，以进程记录的执行者登记主线程，取得主线程的 `ExecutionThread`（线程描述与不透明的线程句柄）；派生失败（如读取操作快照时准入已失效）时进程在 PREPARE 以失败收口，先于预检索与任何话题副作用。进入 Actor 阶段后，同一个主线程作为 `execute` 的独立 `thread` 参数交给 CPU。Actor 阶段在一个可中断子 task 中等待 `execute`，停止取消这个子 task，无论 CPU 在等模型还是阻塞在通道写入上都立即生效。

Actor 阶段结束时，执行器同步结束本进程仍在运行的全部线程（`end_all`），且只在这里执行一次：CPU 返回终态结果（含停止后返回的取消结果）之后、封口与提交之前；Actor 阶段因异常、停止、交付方离开或硬取消提前结束时，也在交付终态（硬取消为传播取消）之前执行。随后进程同步接受 CPU 返回结果中的历史压缩检查点，再判断本轮丢弃或无输出（[会话资源](./sessions.md#5-压缩检查点)）。执行线程的派生、CALL 子线程与唯一的存活判定见[执行线程](./execution-threads.md)。

## 8. 封口、部分输出与结算

交互记录由进程封口：Actor 有输出、进入 FINALIZE 之前，进程以 Actor 的执行结果（轮次事件、使用的模型、结局与回复正文）和实际使用的附件组装 `InteractionPayload`（core 的共用封口函数 `seal_interaction`，`core/protocol/sealing.py`）。封口只组装、不解析事件：交互内容只以轮次事件表达，序号 0 的用户消息事件由 CPU 写出（chat 入口因此拒绝空消息），入口消息与 Gateway 决定不进入交互记录，一轮事件集的不变量在载荷构造时校验；CPU 结果与交互记录都不携带物化任务；提交原样交出载荷，不派发写入，提交失败不影响已接纳写入。Import Bus 以同一个函数封口导入的轮次，经同一个 root 方法提交，两条链路交给 Patchouli 的载荷同形，Patchouli 不需要读懂执行者的运行结果。字段来源见[子系统公共契约](../contracts/subsystem-contracts.md)。

CPU 的三种结局只要有回复文本或执行事件就封口并进入 FINALIZE；初始 `user_message` 是输入，不算输出。只有正文而无助手事件时，封口处补齐规范 `assistant_message`。

进入 FINALIZE 后，进程持有的提交任务依次：

1. 等待进行中的话题确定结束（第一次写入启动的建立可能仍在进行）；
2. 经 root API 为预检索到的记忆记录检索命中（来源 `retrieval.pre_retrieval`）：只在 Actor 有输出后记录，不依赖提交的结局，CPU 失败但有输出的轮次同样记录；命中是生命周期的强化信号，尽力而为，失败只记日志；
3. 经 root API 提交交互并等待应用：交互标识取 `process_id`，来源为 `active_chat`，参与者为执行者，不设等待时限；有派发话题时提交到它，否则携带 Gateway 的新话题决定，话题在应用时建立。查重、提交与等待应用都在这一次调用之内，接纳、等待与结局的归类只在 Patchouli 做一次（[能力层](./capability.md#3-root-api)）；
4. 写入会话，话题取交互实际所在的话题；提交被拒或应用没有成功时照样写入，话题为空（[会话资源](./sessions.md#4-写入会话的时机)）；
5. 本轮话题在应用时才建立的，此时绑定 `chat.run.*` 事件的话题。

提交被拒（提交队列容量已满或已停止）或应用没有成功时，进程以 `ActiveInteractionFinalizationError`（`patchouli.interaction.finalization_failed`）按失败收口：接纳被拒记为接纳阶段，应用失败或队列停止使等待停在应用之前，都记为应用阶段。会话在提交之后才写入，是因为下一轮 Gateway 续接上一轮要用交互实际所在的话题；FINALIZE 期间会话里还没有这一轮，而进程仍持有会话认领，同一会话的下一轮不会在此期间开始。运行结果与 `done` 的 `topic_id` 都是交互实际所在的话题。

已输出的取消或失败轮次仍是后续会话的事实，因此同时保存到会话和记忆材料，并标明结局。这样既保留“内容够了就停止”的回复，又维持一轮对应一个 process/interaction；不引入待用户下一轮确认的中间状态。后续若支持编辑重发，需单独处理已提交轮次的分叉，不能以停止代替撤回。

用户停止仍只取消当前可中断 task，不引入取消 token、信号或轮询。CPU 收到原生取消后完成收尾，把已产生的输出与已完成的检查点作为 cancelled 结果返回；没有转换、直接传播取消的 CPU 按无输出处理。进程沿用 CPU 返回的结局：停止晚于 CPU 终态（CPU 已经返回 completed）时，本轮仍按 completed 结算，不把完整回复改记为中断。FINALIZE 前的交付方离开（客户端断连、交付流关闭）与 System 关闭都不是用户的停止，不保存部分轮次（`ProcessRecord.discards_output`）。

封口、进入 FINALIZE 与创建提交任务之间没有 await；提交任务在写入 `finalizing` 之前创建，由进程以局部状态持有并以 `join_shielded` 等待。交付方离开只退订，进入 FINALIZE 后的停止请求被拒绝，打断不了提交；只有系统关闭的硬取消会遇到这次等待，而它也要等提交结束，提交成功时照常记录 CPU 的结局再传播取消。因此在 `finalizing` 事件处离开不会跳过提交。有输出的 cancelled/failed 的 `chat.run.*` 终态晚于提交。停止后要等交互应用完成才结束，这是当前契约而非后台脱离运行：会话与 `done` 需要交互实际所在的话题。

## 9. 工作集与唯一的退出路径

进程工作集（`ProcessWorkingSet`，`workspace/process/working_set.py`）登记这一轮的话题与关闭时仍需释放或等待的资源：Gateway 为这一轮给出的决定（确定话题与提交新话题都按它进行）、确定之后冻结的派发话题 `dispatch_topic_id`、正在进行或最近一次的话题确定任务、附件租借与附件编译得出的实际使用引用。执行线程不在工作集：它们登记在进程的线程表中。输入清单在 CPU 分配后直接交给 CPU 端口，CPU 返回的执行结果与提交任务都是阶段内的局部状态，不进入工作集。工作集只登记、不释放，也不持有共享依赖：资源由取得它的一方处理，附件租借由 `CPUAllocator` 释放，话题确定由线程表启动、由提交任务与关闭流程等待。

进程只有一条退出路径，无论以何种结局结束都在进程 task 中完成：执行器的 `close(process)` 在骨架收尾时执行一次，先于任何 await 同步结束仍登记的线程并释放全部租借，再等待进行中的话题确定结束（停止、无输出或丢弃本轮时没有提交任务，写入启动的确定可能仍在进行）；注册入口的驱动协程随后在 `finally` 中从进程表注销、释放会话认领，最后结束输出通道，因此即使关闭中的等待被硬取消打断，租借、进程登记与会话认领也不会泄漏。系统阶段都在执行器的运行与关闭流程之内，在注销之前结束；注销之后，仍指向本进程的线程句柄按线程已结束拒绝（[执行线程](./execution-threads.md#3-生命周期与唯一的存活判定)）。终态兜底只在进程仍无终态时执行，即进程 task 被硬取消、或进程从未运行就被关闭两种情形；工作集中的每项资源只取出一次，关闭即使重复执行也不会重复释放，但正确性不依赖重复关闭。

关闭不补偿任何话题，也不回滚已提交的交互或已接纳的写入：话题只在写入或交互应用时建立，建立后没有接收交互的话题由空闲结算或 LRU 回收（第 5.3 节）。

## 10. 命令终态

Gateway 返回 command outcome 时，结果只携带命令解析结果，服务立即完成本次 run，不进入 topic、retrieval、Actor 执行或主动记忆生成。命令终态由进程按解析状态产生（`workspace/process/command_terminal.py`）：解析成功时命令暂不可用（`not_implemented`、`command.unavailable`），解析失败时拒绝（`rejected`、`command.parse.<状态>`），均不带客户端动作；进程仍以 completed 结束。这是控制消息与普通对话之间的语义隔离，不是一个性能优化开关。命令的解析规则见 [Gateway 全局命令](../gateway/commands.md)。

## 11. 交付

### 11.1 流式交付

`run_process(handle)` 运行骨架，阶段产出按以下顺序投影为 transport 事件：

```text
process_id（含 session_id）
  -> Gateway decision / command
  -> command_result + done
  -> 或 PREPARE（Profile 解析、预检索、已有话题的确认、CPU 分配）
       memory_refs
       CPU 交互事件（原样转交）
       run_status(finalizing)
       done(completed/cancelled/failed + topic_id)
```

`done` 由进程从 CPU 返回的终态结果投影，用户停止时 CPU 返回的取消结果同样经它交付。`done` 只携带交付方使用的执行结果字段（终态、最终回复、模型名与可选失败原因码）与进程字段：轮次事件只用于封口交互记录，不下发；检查点也不进入 done，由会话读取返回；记忆任务经 `memory.task.*` 事件与任务查询观察；也不含执行者专属的统计。`memory_refs` 在 PREPARE 成功后发出，列出预检索的记忆；有输出的三种 done 携带交互实际所在的话题 `topic_id`（提交没有成功时进程按失败收口，交付 `error` 而不是 `done`），无输出不携带 `final_text` 或话题标注。SSE 字段契约见[公开路由与事件](../contracts/routes-and-events.md#61-会话管理与-chat-sse)。

chat 路由订阅流式交付，不轮询断连，也不为每个事件创建 task。客户端断连时 sse-starlette 取消响应，交付生成器随之退订，进程尚未记录终态时经与 `cancel_process` 相同的实现以 `client_disconnected` 请求停止：生效时机与用户停止相同，发布 `chat.run.cancel_requested` 与带生效阶段的 `chat.run.cancelled`，本轮输出被丢弃。响应在等待 send 时被取消、生成器停在 `yield` 处时，响应收尾关闭生成器；生成器从未开始迭代时（例如关停信号在注册期间到达），响应收尾经 `close_process` 关闭进程。交付方离开后，进程在自己的 task 中完成关闭：SSE 响应不等待它，可能早于进程关闭结束；客户端此时立即对同一会话重发，会按会话忙碌规则被显式拒绝，不排队、不抢占（前端不自动重发，只有用户触发的重试会遇到）。

流式 `done`、`command_result` 和 `error` 是 transport 可消费的事件，不是新的跨子系统业务契约；它们的来源和调用顺序仍由本服务和 Contracts 共同约束。

## 12. 取消与停止

`cancel_process(process_id, actor=…)` 是唯一的取消方法，服务于 `POST /chat/stop` 的用户停止，停止原因为 `user_requested`。取消是用户下达的指令，属于 root：以请求方的工作 actor（server 以 (user, `system`) 认证）对进程的 Workspace 执行 `authorize_root`（[身份与授权](./identity-and-authorization.md#4-操作授权)）。进程不存在或授权拒绝（请求方驻留在其他 Workspace、不是 `system`）时都返回 `not_found`，并发布只带 `process_id` 的 `chat.run.cancel_requested`，不泄露进程是否存在；进程查无先于授权，准入失效等其他授权错误原样传播。

用户停止、交付方离开（`client_disconnected`）与 `close_process` 共用同一份停止实现：同步调用 `request_stop()`，首次 stop 固定 reason，重复 stop 返回同一判定且不重复取消 task，事件使用进程注册时绑定的观测标签。取消不新增 operation，不是 Agent action。注册入口不提供进程状态查询：流式交付以 `run_status`（进入 FINALIZE）与 `done` 交付阶段与结局。

取消响应点只有两处：只有 Gateway 与 Actor 执行两个阶段会取消当前 `active_task`，因为这两个阶段在前台调用 LLM。PREPARE 不可中断，正在进行的工作照常完成：停止被接受后主线程随之结束，此后才开始的预检索与已有话题的确认以 `ThreadEndedError` 被拒绝，进程随即按 PREPARE 阶段的取消收口；其余时刻（Profile 解析、正在进行的预检索、存储健康与 CPU 分配期间）收到的停止在进入 Actor 执行前统一检查一次（CPU 分配之后的检查与 `_run_interruptible` 的入口检查）。两种情形都跳过 Actor 执行与提交，返回 cancelled 结果，在关闭流程中释放租借；阶段交接窗口只记录 stop，下一阶段不会启动；FINALIZE 与 Terminal 拒绝 stop。停止在任何时刻都立即取消当前可中断阶段的子 task：Actor 中无论 CPU 在等模型还是阻塞在输出通道写入上。外部取消的入口是 `POST /chat/stop`（必须携带明确的 `process_id`）与交付方离开（HTTP 为客户端断连）。

**停止与执行线程**：停止请求同步记录停止（Actor 阶段同时取消 Actor 子 task），进程的全部执行线程随之结束：存活判定读取进程结局，此后的 actor 操作（含 PREPARE 中的预检索）、取得派发话题与 CALL 派生都以 `ThreadEndedError` 拒绝，停止路径不通知线程表。CPU 只用已产生的内容组装 cancelled 结果（CPU 端口要求取消后的收尾不依赖 actor 操作）；Actor 阶段收尾标记全部线程结束、接受检查点，之后的封口、检索命中、交互提交与会话写入都由系统经 root API 完成，不依赖任何线程。停止那一刻 CPU 内的在途读取随子 task 的取消中断，已开始派发的写入受取消保护照常完成；正在等待话题确定的写入不登记意图（随 Actor 子 task 一起取消，或在登记前的同步段被拒绝），话题确定本身照常完成，已经建立的话题没有写入。

用户 stop 在编排骨架内被翻译为私有 `_ProcessCancelled` 分支，下游只传播原生 `asyncio.CancelledError`。`_run_interruptible()` 区分“stop 取消 child task”和“进程 task 自身被取消”；后者只来自系统关闭的硬取消，必须原样向上传播。child task 可能把取消转为返回值（CPU 在最外层这样做），等待方的取消请求会随之丢失，因此 child task 正常结束后同样按进入时记录的 `cancelling()` 判断，重新抛出硬取消。资源所有者在 unwind 中关闭自己创建的 LLM 流与 provider response；收尾异常只记录日志，不能替换正在传播的 `CancelledError`。

## 13. 关闭与系统停止

进程关闭只在进程 task 中完成（第 9 节）。`close_process(handle)` 关闭句柄对应的进程并等待进程 task 结束：进程从未运行时创建只执行关闭流程的进程 task，仍在运行时以 `client_disconnected` 请求停止；这次等待可以被调用方取消，取消不影响进程自身的收尾；句柄已失效时是空操作。进程关闭后从进程表注销，此后取消按 `not_found` 处理。

当前进程表是进程内短期控制状态，不是可恢复的长期工作记录。进程重启后不能据此恢复进程；System 停止时 `TaskProcessService.shutdown()` 依次：直接关闭从未运行的进程，以 `system_shutdown` 停止已启动的进程，在 `workspace.process.shutdown_wait_seconds`（默认 10 秒）内等待全部进程 task 结束，超时仍未结束的进程 task 被取消（硬取消）并等待结束。硬取消时进程照常在自己的 task 中收尾，记录“终态前关闭”形态的 `chat.run.cancelled`；已进入 FINALIZE 的进程拒绝停止，仍等提交结束。返回之前全部进程都已关闭，之后才清空会话服务。System 的停止顺序中进程服务先于 Patchouli 停止（[System 组合根](../system/composition.md#3-停止顺序)），进程的提交因此在交互提交队列停止之前结束，Patchouli 不必另外等待任务进程。

## 14. 错误与观测

- `chat.run.*` 事件由领域 emitter `TaskProcessEventEmitter`（`workspace/process/events.py`）从进程记录投影阶段与终态，经 `RuntimeEventPublisher` best-effort 发布：发布失败不改变返回值，失败事件只携带 Workspace 领域错误码或固定摘要，不写入异常正文；
- Gateway 与 Actor 执行（CPU）的 task cancellation 不包装成普通 success 或 failed；Gateway timeout 仍按自己的 deadline/fallback 契约处理；
- failed done 保留 `reason=agent_run_failed`，CPU 有稳定原因时另带 `failure_reason`；失败原因在 done 与 `chat.run.failed` 中投影，字段见[事件契约](../contracts/routes-and-events.md#61-会话管理与-chat-sse)；
- 捕获的错误保留原始因果，不能通过“空列表/空话题”掩盖 route 缺失或契约违约。

新增入口或流程时，检查：是否绕过 root API 的提交交互另起一条提交路径；是否在写入与交互应用之外预先建立话题，或为话题补偿、清理；是否把 command outcome 继续送进普通 chat；是否在取消后仍提交；是否把进程表、memory task 或 topic 的临时状态写成另一个权威来源。

## 15. 已知限制

- 取消只检查请求方在进程的 Workspace 是 `system`：同一 Workspace 下的请求方可以停止其他请求方的进程；
- 断连后的 SSE 响应可能早于进程关闭结束，其间对同一会话的重发按会话忙碌规则拒绝；断连后继续运行、重连与多订阅者仍按 [Chat Run 生命周期 Idea](../ideas/chat-run-lifecycle-follow-ups.md) 处理；
- 已有话题只在 PREPARE 确认时刷新驻留；单轮运行超过空闲结算时限时，话题仍可能在提交之前被结算，交互应用随之失败，进程以 `patchouli.interaction.finalization_failed` 收口；
- 唯一注册入口只接收主动 chat：命令只解析不执行，内置命令暂不可用；被动请求按[任务进程 Idea](../ideas/task-process-table-and-registration-entry.md) 2.6 节的决定暂不经注册入口；
- 用户可见长期任务与后台 Agent workflow 当前未立项，需在真实负载和执行能力成立后独立设计。

## 16. 代码与测试入口

- [`workspace/process/`](../../src/hivememory/workspace/process/)：注册入口 `service.py`、进程表 `table.py`、进程状态容器 `task_process.py`、执行器 `runner.py`、CPU 分配 `allocation.py`、命令终态 `command_terminal.py`、输出通道与阶段产出 `outputs.py`、事件投影 `events.py`、工作集 `working_set.py`；
- 封口：与 Import Bus 共用的 [`core/protocol/sealing.py`](../../src/hivememory/core/protocol/sealing.py)；
- 公共契约：[`workspace/contracts/cpu.py`](../../src/hivememory/workspace/contracts/cpu.py)、[`workspace/contracts/process.py`](../../src/hivememory/workspace/contracts/process.py)；HTTP 入口：[`server/routers/chat.py`](../../src/hivememory/server/routers/chat.py)；
- 测试：
  - `tests/unit/workspace/process/test_gateway_chat_flow.py`（系统阶段以 root scope 执行、提交的参与者是执行者；预检索按 Gateway 的计划经读取视图交付，执行者没有 `memory.search` 或请求关闭检索时不检索；已有话题在 CPU 分配前确定，新话题且没有写入时由提交建立；提交没有成功时会话仍有这一轮、话题为空；检索命中在有输出后记录、记录失败不影响本轮；PREPARE 中停止拒绝预检索；Actor 中停止在等 CPU 与阻塞于通道写入时的响应，以及 Gateway 之前的模型预取）；
  - `tests/unit/workspace/process/test_chat_run_control_contract.py`（注册的两次认证与失败时不登记、取消的 root 授权、输出通道的容量与离开语义、订阅者暂停时进程仍推进、五种结局的唯一退出路径、交付方离开不等待关闭）；
  - `tests/unit/workspace/process/test_cpu_allocation.py`（Profile 的管理读取与附件租借以 root scope 执行、准入失效或 Profile 解析失败时先于预检索与话题失败、租借释放与关闭顺序）；
  - `tests/unit/workspace/process/test_cpu_port.py`（经 CPU 端口执行：测试 CPU 跑通任务进程、自报结局与执行异常）；
  - `tests/unit/workspace/process/test_command_terminal.py`、`test_process_events.py`（`chat.run.*` 投影与 best-effort 边界，交付方离开与硬取消的事件形态）；
  - `tests/unit/core/protocol/test_seal_interaction.py`（共用封口函数）；
  - `tests/unit/server/routers/test_chat.py`（注册后运行、断连关闭交付、流从未开始时的关闭）；
  - `tests/integration/workspace/test_session_completion.py`（部分输出、停止晚于终态、交付方离开、检查点接受与系统关闭的停止、等待与硬取消）；
  - `tests/integration/workspace/test_process_intents.py`（已接纳写入跨进程终态保留、主线程与子线程共用一个话题并各记线程标识、检索读取 RUN 与 CALL 不确定话题、被拒的写入不确定话题、确定话题失败交回这次写入并由下一次重试、确定话题期间停止时不登记、提交的授权失败不撤回已派发的意图、进程关闭拒绝等待中的更新）、`test_inflight_dispatch.py`（ACK 接纳时点、明确拒绝、未知结果与派发期间取消）；
  - `tests/integration/workspace/test_process_threads.py`（并发取得派发话题只以 system 确定一次、一个等待者取消不影响确定、CPU 结束或用户停止时线程先于终态结束，之后的提交仍以 system 执行）；
  - `tests/e2e/system/test_chat_operation_whitelist.py`（只有 `memory.search` 的 Agent 经 HTTP 完成 chat：Profile 与附件由系统加载，检索按执行者的可见性，交互的发起者是执行者）。
