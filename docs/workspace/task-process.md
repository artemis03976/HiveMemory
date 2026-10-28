---
title: 任务进程
status: current
owner: workspace
scope: task-process-registration-table-orchestration-delivery-cancellation-and-close
code_paths:
  - src/hivememory/workspace/process/
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
  - docs/workspace/execution-threads.md
  - docs/workspace/sessions.md
  - docs/system/application-services.md
  - docs/system/attachments.md
  - docs/system/import-bus.md
last_reviewed: 2026-10-21
---

# 任务进程

任务进程是 workspace 运行一次任务请求的单位：请求经唯一注册入口登记为进程，此后 Gateway 分析、预检索、CPU 分配、Actor 执行与结算都在进程内进行，停止与关闭因此覆盖整条链路，中间产物与访问 context 也都有明确的持有者。当前只有主动 chat 经注册入口进入；外部 harness 中已经结束的交互经 System 的 Import Bus 导入，不创建任务进程，但与任务进程共用交互记录的封口函数（第 8 节、[Import Bus](../system/import-bus.md)）。

这一层存在，是因为 HTTP、SSE 与未来的外部 adapter 都需要共享同一条主动 chat、取消与结算链路。若每个入口自己拼装 Gateway、Patchouli 和 Alice，就会再次出现多套 prepare/finalize、错误处理和取消语义。任务进程回答“这次请求应该按什么顺序跨边界运行”，而不回答“记忆如何检索”“Agent 如何生成”或“Gateway 如何分析”。

## 1. 唯一注册入口与三个角色

`TaskProcessService` 是任务请求的唯一注册入口，注册与运行分开：`register_process()` 完成认证、认领会话并创建进程、返回进程句柄（第 2 节），`run_process(handle, stream=…, detach_reason=…)` 运行已注册的进程，`stream`（默认 `True`）只决定交付形态，`detach_reason`（默认 `stream_closed`）是交付方离开时请求停止的原因；注册参数 `message` 是交给 Gateway 分析的指令文本。

任务进程由三个角色组成，按“谁持有什么”划分：

| 角色 | 持有 | 不持有 |
|:---|:---|:---|
| `TaskProcessService`（注册入口，`workspace/process/service.py`） | 生命周期依赖：认证网关、操作授权者（进程控制授权）、事件 emitter、进程表（组合根创建，与进程线程管理共享）、执行器 | 总线、CPU 端口、CPU 分配器、asset reader、编译配置等编排依赖 |
| `TaskProcess`（进程状态容器，`workspace/process/task_process.py`） | 本进程独有的东西：进程记录（含进程 context 与绑定了本进程标签的观测通道）、任务参数、工作集、线程表、`trace_id`、进程 task（本进程自己的执行流）与输出通道 | 任何跨进程共享的依赖 |
| `TaskProcessRunner`（执行器，`workspace/process/runner.py`） | 所有进程共用的编排依赖：全局总线、CPU 端口、`CPUAllocator`、进程线程管理 `ProcessThreads`、写入意图登记、操作授权者（阶段授权）、Gateway 超时配置；唯一的编排骨架 `run(process)` 与唯一的关闭流程 `close(process)`，二者都运行在进程 task 中 | 任何进程的状态 |

这样划分是为了让注册入口只管理进程的生命周期，而不认识具体流程，并让进程本身只是状态容器：进程记录与工作集，由所有进程共用的四阶段骨架驱动（理念见[第五代架构 Idea](../ideas/fifth-generation-architecture.md)第 3.2 节）。执行器与 `CPUAllocator` 由组合根构建并注入注册入口，入口不转交编排依赖。任务进程不导入能力层；执行器、CPU 分配与进程线程管理不导入认证一侧，线程 context 的派生与撤销经生命周期端口完成（[Workspace](./README.md#4-包结构与依赖规则)）。

## 2. 注册

注册在流式响应开始之前完成：

1. 拒绝保留的 `system` 作为任务请求的 actor：任务进程必须由具体 Agent 执行；
2. 经认证网关完成两阶段认证，运行绑定为本进程的 `process_id`（由 server 入口在进入编排服务前生成并冻结）；认证失败直接抛出，HTTP 入口据此返回 403，不创建、不登记进程；
3. 使用请求 `session_id` 查找本 Workspace 会话，缺省则自动创建；同步检查容量与单进程认领，失败撤销 context 并回滚本次自动创建的空会话（[会话资源](./sessions.md#3-认领与容量)）；然后创建进程记录（写入访问 context）与 `TaskProcess`，以通过认证的声明创建一次冻结的 `ExecutionLabels(agent_id, workspace_id)` 并绑定 `chat.run.*` 的事件发布器，登记到进程表；认证通过后、登记完成前失败时撤销已签发的 context；
4. 签发并返回不透明的进程句柄（只读 `process_id/session_id`）；关闭时在 context 撤销、进程注销后释放会话认领。

执行标签由事件发布器 `BoundProcessEvents` 持有，CPU 分配把同一标签对象传给 `CPUInputManifest.labels`，供运行级观测（Alice 的 `agent.run.*`）与历史角色匹配使用；执行线程的标签由线程表放进线程描述（[执行线程](./execution-threads.md#5-主线程交给-cpu)）。标签只服务历史角色匹配、展示回退与观测，不能授权或选择资源目标。

## 3. 进程表、进程记录与句柄

**进程表**（`ProcessTable`，`workspace/process/table.py`）是唯一的进程注册表，也是 workspace 的进程内共享设施：以 `process_id` 为键登记 `TaskProcess`，注销只移除同一个进程对象；进程记录作为进程的控制面经进程取得，不单独登记。进程记录（`ProcessRecord`）有：

- `process_id`：进程唯一标识；
- `access`：注册时签发、绑定本进程的访问 context，只交给操作授权者使用，不保存 actor、驻留 Workspace 等身份字段；
- `events`：注册时绑定冻结 `ExecutionLabels` 的事件发布器，CPU 分配沿用同一组标签；
- `phase`：`created/gateway/prepare/actor/finalize/terminal`（`actor` 即 Actor 执行阶段）；
- `outcome`：`running/stop_requested/cancelled/completed/failed`；
- 首次接受的 `stop_reason`；
- 仅在 Gateway 或 Actor 执行阶段存在的 `active_task` 身份引用。

进程表不保存 `Event`、Token 或 waiter。进程单独设记录一层，是因为停止请求、事件发布、阶段 task 绑定与进程控制授权只读写控制状态，而容器还要容纳生命周期各不相同的任务参数、工作集、task 与输出通道。

**进程句柄**（`ProcessHandle`）是入口 adapter 与已注册进程之间唯一的引用：只暴露 `process_id` 与 `session_id`，不暴露进程记录、其中的 context 或 `TaskProcess`。句柄只由注册入口签发，按对象身份判定有效：句柄私下记着它对应的进程，注册入口解析时要求进程表中登记的正是这一个。按 `process_id` 重新构造的对象、进程关闭后的旧句柄都不是有效句柄——运行时以 `process_handle_unknown` 拒绝，取消时返回 `not_found` 且不发布事件，关闭是空操作。持有有效句柄即为该进程生命周期的所有者。

## 4. 执行流与输出通道

进程持有自己的执行流：交付方首次运行进程时，注册入口为它创建进程 task，阶段推进不依赖交付方读取。骨架只产出类型化的阶段产出（`workspace/process/outputs.py`），写入进程持有的输出通道 `ProcessOutputChannel`；终态产出同时作为进程 task 的返回值。交付方只订阅或等待：流式交付订阅通道，把阶段产出投影为 SSE 事件；非流式交付不订阅，等待进程 task 交出终态产出。执行不区分交付形态，两种交付共用同一条阶段顺序、同一组取消响应点与同一个关闭流程，因此模型无文本时的收敛方式、停止时保留的部分输出等语义也一致。编排途中的异常在两种交付下都先记录失败终态，再由流式交付翻译为 `error` 事件（Workspace 领域错误携带安全文案与错误码，其余统一为系统错误），或由非流式交付在进程关闭后原样上抛。

输出通道只允许一个订阅者，订阅者看到的顺序是阶段产出、终态产出、结束标记；结束标记在进程关闭完成之后才写入，因此正常结束时 SSE 响应在进程关闭之后结束。只有交互事件受容量限制（256 项）：订阅者在线而通道写满时进程等待，背压一直传到 CPU；终态产出与结束标记不受限，订阅者再慢也不会卡住进程收尾。订阅者暂停时，进程最多先行一个通道容量。订阅者离开后，通道唤醒阻塞的写入并丢弃此后的产出。

## 5. 四阶段骨架

### 5.1 主动链路

```text
TaskProcessService.register_process()：认证、创建进程、登记进程表，返回进程句柄
TaskProcessService.run_process(handle, stream=…)：创建进程 task；交付方订阅输出通道或等待终态
进程 task -> TaskProcessRunner.run(process)
  -> Gateway public process (ACTIVE_CHAT)             阶段授权
  -> command: return command outcome
  -> decision: 解析 Agent Profile（workspace 读取视图）  阶段授权
  -> Patchouli prepare_agent_run（Topic 与检索；interaction_id 取 process_id 值）  阶段授权
  -> CPU 分配：附件租借（阶段授权）与编译、记忆编译、中立提示词编译、组装 CPUInputManifest
  -> Actor 执行：以图纸派生主线程（读取操作快照、登记线程表），在可中断子 task 中 await CPU 端口 execute
       （CPUInputManifest 与主线程的 ExecutionThread；交互事件经 events sink 写入输出通道，终态结果作为返回值）
       -> Actor 阶段结束：同步结束全部线程（撤销线程 context），接受完成的历史压缩检查点
  -> 有输出的 completed/cancelled/failed：封口交互记录 -> 同步写入会话 -> 进入 finalize
       -> 创建并持有提交 task，阶段授权 -> Patchouli finalize_agent_run
  -> 无输出、异常、finalize 前交付方离开：不写会话，转入关闭补偿
  -> 终态产出写入输出通道，并作为进程 task 的返回值
  -> 关闭：TaskProcessRunner.close(process)
       -> 同步结束仍在运行的线程（兜底），经 CPUAllocator 释放附件租借
       -> 若 prepare 成功且 finalize 未成功，阶段授权 -> await Patchouli cleanup_prepared_agent_run
  -> 注册入口在进程 task 的 finally 中撤销进程 context、从进程表注销、释放会话认领，最后结束输出通道
```

Gateway 阶段从会话取最近 `gateway.context_preparation.session_turn_limit` 轮的用户与助手文本，连同每轮在会话中标注的落位话题 `topic_id`，随 `session_turns` 交给 Gateway（[Gateway 分析](../gateway/analysis.md)）。落位话题只在话题路由失败时使用：Gateway 续接上一轮的话题（该话题仍在本次的候选话题中时），否则新建话题；同一会话的相邻轮次大多仍在同一话题上，续接比每次失败都新建无标题话题归错的代价小。

### 5.2 阶段调用

阶段调用是进程替 Agent 做的系统操作，不是 Agent 主动执行的操作：Gateway 话题读取、prepare（话题与检索）、cleanup 补偿、CPU 分配中的 Profile 解析与附件租借、finalize 与结算后的话题池读取，每次调用前执行器都以本进程注册时通过认证的 Workspace 为目标调用 `authorize_stage`，再把返回的 `IdentityScope` 传给对应路由或端口。阶段授权只校验 context 与准入仍然有效、目标与 owner 相符，不检查 actor 操作白名单，因此没有任何 actor 操作的 Agent 也能完成整条 chat 链路。返回的 scope 仍以进程的 Agent 为发起者，资源 owner 照常按该 Agent 过滤可见性：预检索结果中不会出现其他 Agent 的 PRIVATE 记忆，但预检索不检查 Agent 能否 READ（[能力层](./capability.md#1-三个调用面)）。

阶段授权的 scope 只交给紧随的阶段调用，不保存在工作集中等待使用，也不进入 CPU 输入清单、线程 context 或任何 CPU 可见的对象。`PreparedAgentRun` 只冻结 prepare 的 `belong_to`：进程在 prepare 返回后先把结果写入工作集，再比对它与任务目标 Workspace，不一致时抛 `WorkspaceMismatchError`（第 9 节）。finalize 的阶段授权在提交任务内、调用公开路由之前执行，Patchouli 在接纳交互之前再次比对 prepared 的归属与 scope 的目标 Workspace；授权失败时不提交交互，不影响已接纳写入。cleanup 重新执行阶段授权，拒绝时记录警告并完成进程关闭；资源 owner 对越域 prepared 返回 `False`，不执行删除。阶段授权只在 `workspace.process` 中调用（架构测试守护）。

## 6. CPU 分配

CPU 分配（Profile 解析、附件租借与编译、记忆与中立提示词编译、清单组装）由 `CPUAllocator`（`workspace/process/allocation.py`）完成，它取得的附件租借也由它在进程关闭时释放。

**Profile 解析**：Agent Profile 属于 CPU 分配，但目前在 prepare 之前解析：prepare 可能按路由决定新建 Topic，话题池已满时还会先按 LRU 结算一个已有话题，Profile 缺失的请求应在这些副作用发生之前失败。Profile 经阶段授权后由组合根注入的读取视图（`ProfileResolver`）解析，不经能力层，仍按进程 Agent 检查源原子可见性；正式原子变更会失效解析缓存。

**编译与清单**：Patchouli prepare 只返回话题准备结果与未编译的检索原子（`PreparedAgentRun`）；进程用共享引擎 `MemoryCompiler` 把检索结果编译为 `RETRIEVAL_CONTEXT` 文本，用 `AttachmentCompiler` 编译附件并得出实际使用的附件，再由 `prompts.compilation.compile_prompt_input` 按 Profile 语言（缺省时使用进程默认值）定稿中立提示词：环境通知与人设组成 `instructions`，记忆与附件组成 `turn_context`。两段文本、解析好的 `language`、会话全量历史 `session_history` 与已有 `compaction_checkpoint` 一起组装为输入清单，经 CPU 端口交给 CPU。清单字段见[子系统契约](../contracts/subsystem-contracts.md#41-cpu-端口)。

已解析的 Profile 只在编译人设时使用，不进入清单：图纸与操作快照随主线程的线程描述交给 CPU。清单回答“处理什么”，线程描述回答“谁来处理”。

编译放在进程一侧，是为了让不同执行者共用同一份已定稿的 HiveMemory 提示词内容，而不必各自调用引擎或复刻人格、环境通知与材料的拼装。中立编译只依赖 core、i18n 与提示词片段，不接收 Alice 或其他 CPU 的配置；工具教学、历史重放与压缩归执行者，Alice 只原样放置固定文本。进程不掌握 CPU 的模型窗口，固定输入超窗仍由 CPU 在生成前拒绝。

CPU 分配还把与输入清单相同的真实 `topic_id` 写入工作集的 `dispatch_topic_id`，先于派生主线程，供写入意图派发使用（[执行线程](./execution-threads.md#8-派发话题)）。

## 7. Actor 阶段

Actor 执行经 CPU 端口完成：执行器由组合根注入一个 `CPUPort`（`workspace.contracts`，当前唯一的实现是 Alice 的 `AliceCPU`），进程只调用端口的 `execute(manifest, *, thread, generation_options, events)`，不出现任何具体 CPU 的路由名或结果类型。CPU 经 `events` sink 逐项推送交互事件，进程把它们写入输出通道；终态结果 `CPUExecutionResult` 是 `execute` 的返回值。端口与结果的契约见[子系统公共契约](../contracts/subsystem-contracts.md#4-cpu-端口与-alice-实现)第 4 节。端口定义在 workspace、由 CPU 实现，是为了让 CPU 可以替换而不改动进程：Alice 之外的 CPU（测试中的 `ScriptedCPU`）能跑完整个任务进程。

进入 Actor 阶段时，执行器把 prepare 之前解析的图纸交给 `ProcessThreads.derive_main(process, profile)`，取得主线程的 `ExecutionThread`，作为 `execute` 的独立 `thread` 参数交给 CPU；派生失败（如准入在 prepare 之后被撤销）时进程在 Actor 阶段以失败收口。Actor 阶段在一个可中断子 task 中等待 `execute`，停止取消这个子 task，无论 CPU 在等模型还是阻塞在通道写入上都立即生效。

Actor 阶段结束时，执行器同步结束本进程仍在运行的全部线程（`end_all`），且只在这里执行一次：CPU 返回终态结果（含停止后返回的取消结果）之后、finalize 之前；Actor 阶段因异常、停止、交付方离开或硬取消提前结束时，也在交付终态（硬取消为传播取消）之前执行。随后进程同步接受 CPU 返回结果中的历史压缩检查点，再判断本轮丢弃或无输出（[会话资源](./sessions.md#5-压缩检查点)）。执行线程的派生、CALL 子线程与唯一的存活判定见[执行线程](./execution-threads.md)。

## 8. 封口、部分输出与结算

交互记录由进程封口：Actor 有输出、进入 finalize 之前，进程以入口消息、Gateway 决定、Actor 的执行结果与实际使用的附件组装 `InteractionPayload`（core 的共用封口函数 `seal_interaction`，`core/protocol/sealing.py`），MTP 轨迹由 core 的归约器从轮次事件得到；CPU 结果与交互记录都不携带物化任务；finalize 原样提交、等待 applied 并记录命中后返回 `None`，不派发写入。finalize 失败不影响已接纳写入。Import Bus 以同一个函数封口导入的轮次，两条链路交给 Patchouli 的载荷同形，Patchouli 不需要读懂执行者的运行结果。字段来源见[子系统公共契约](../contracts/subsystem-contracts.md#32-finalizeagentrun) 3.2。

CPU 的三种结局只要有回复文本或执行事件就封口、同步写入会话并进入 finalize；初始 `user_message` 是输入，不算输出。只有正文而无助手事件时，封口处补齐规范 `assistant_message`。finalize 成功后 prepared 标记为已接管，不再 cleanup。

已输出的取消或失败轮次仍是后续会话的事实，因此同时保存到会话和记忆材料，并标明结局。这样既保留“内容够了就停止”的回复，又维持一轮对应一个 process/interaction；不引入待用户下一轮确认的中间状态。后续若支持编辑重发，需单独处理已提交轮次的分叉，不能以停止代替撤回。

用户停止仍只取消当前可中断 task，不引入取消 token、信号或轮询。CPU 收到原生取消后完成收尾，把已产生的输出与已完成的检查点作为 cancelled 结果返回；没有转换、直接传播取消的 CPU 按无输出处理。进程沿用 CPU 返回的结局：停止晚于 CPU 终态（CPU 已经返回 completed）时，本轮仍按 completed 结算，不把完整回复改记为中断。finalize 前的交付方离开（客户端断连、交付流关闭）与 System 关闭都不是用户的停止，不保存部分轮次（`ProcessRecord.discards_output`）。

封口、会话追加、进入 FINALIZE 与创建提交任务之间没有 await；提交任务在写入 `finalizing` 之前创建，由进程以局部状态持有并以 shield 等待。交付方离开只退订，进入 FINALIZE 后的停止请求被拒绝，打断不了提交；只有系统关闭的硬取消会遇到这次等待，而它也要等提交结束，提交成功时照常记录 CPU 的结局再传播取消。因此在 `finalizing` 事件处离开不会跳过提交，也不会 cleanup 已接管交互。有输出的 cancelled/failed 的 `chat.run.*` 终态晚于 finalize。停止后要等交互 applied 才结束，这是当前契约而非后台脱离运行。

## 9. 工作集与唯一的退出路径

进程工作集（`ProcessWorkingSet`，`workspace/process/working_set.py`）只登记关闭时仍需释放或补偿的资源：本进程的 prepare 结果、CPU 分配时写入的派发话题 `dispatch_topic_id`、附件租借与附件编译得出的实际使用引用。执行线程不在工作集：它们登记在进程的线程表中，线程 context 随线程结束撤销。输入清单在 CPU 分配后直接交给 CPU 端口，CPU 返回的执行结果与 finalize 提交任务都是阶段内的局部状态，不进入工作集。工作集只登记、不释放，也不持有共享依赖：资源由取得它的一方释放，附件租借由 `CPUAllocator` 释放，prepare 结果的 cleanup 由执行器处理。

进程只有一条退出路径，无论以何种结局结束都在进程 task 中完成：执行器的 `close(process)` 在骨架收尾时执行一次，先于任何 await 同步结束仍登记的线程并释放全部租借，再对尚未交接的 prepare 结果重新执行阶段授权并请求 cleanup；注册入口的驱动协程随后在 `finally` 中撤销进程 context、从进程表注销、释放会话认领，最后结束输出通道，因此即使 cleanup 的 `await` 被硬取消打断，租借、context、进程登记与会话认领也不会泄漏。cleanup 的授权失败和调用失败都记录警告、跳过补偿，关闭照常完成。终态兜底只在进程仍无终态时执行，即进程 task 被硬取消、或进程从未运行就被关闭两种情形；工作集中的每项资源只取出一次，关闭即使重复执行也不会重复释放，但正确性不依赖重复关闭。prepare 返回的结果先写入工作集，再比对 `PreparedAgentRun.belong_to` 与任务目标 Workspace；校验失败时仍按本进程目标授权后交回 cleanup，Patchouli 拒绝清理越域结果，补偿不会删除其他 Workspace 的话题。

cleanup 是 prepare 之后的有限补偿，只补偿 prepare 新建且仍为空的临时话题，不是跨子系统 rollback。

## 10. 命令终态

Gateway 返回 command outcome 时，结果只携带命令解析结果，服务立即完成本次 run，不进入 topic、retrieval、Actor 执行或主动记忆生成。命令终态由进程按解析状态产生（`workspace/process/command_terminal.py`）：解析成功时命令暂不可用（`not_implemented`、`command.unavailable`），解析失败时拒绝（`rejected`、`command.parse.<状态>`），均不带客户端动作；进程仍以 completed 结束。这是控制消息与普通对话之间的语义隔离，不是一个性能优化开关。命令的解析规则见 [Gateway 全局命令](../gateway/commands.md)。

## 11. 交付

### 11.1 流式交付

`run_process(handle)`（默认 `stream=True`）运行同一个骨架，阶段产出按以下顺序投影为 transport 事件：

```text
process_id（含 session_id）
  -> Gateway decision / command
  -> command_result + done
  -> 或 Profile 解析、prepare、CPU 分配
       memory_refs
       CPU 交互事件（原样转交）
       run_status(finalizing)
       done(completed/cancelled/failed + topic_id)
```

`done` 由进程从 CPU 返回的终态结果投影，用户停止时 CPU 返回的取消结果同样经它交付。`done` 只携带交付方使用的执行结果字段（终态、最终回复、模型名与可选失败原因码）与进程字段：轮次事件只用于封口交互记录，不下发；检查点也不进入 done，由会话读取返回；记忆任务经 `memory.task.*` 事件与任务查询观察；也不含执行者专属的统计。`memory_refs` 在 CPU 分配成功后发出；有输出的三种 done 携带本轮 `topic_id`，无输出不携带 `final_text` 或话题标注。SSE 字段契约见[公开路由与事件](../contracts/routes-and-events.md#61-会话管理与-chat-sse)。

chat 路由以 `detach_reason="client_disconnected"` 订阅交付，不轮询断连，也不为每个事件创建 task。客户端断连时 sse-starlette 取消响应，交付生成器随之退订，进程尚未记录终态时经与 `cancel_process` 相同的实现以 `client_disconnected` 请求停止：生效时机与用户停止相同，发布 `chat.run.cancel_requested` 与带生效阶段的 `chat.run.cancelled`，本轮输出被丢弃。响应在等待 send 时被取消、生成器停在 `yield` 处时，响应收尾关闭生成器；生成器从未开始迭代时（例如关停信号在注册期间到达），响应收尾经 `close_process` 关闭进程。交付方离开后，进程在自己的 task 中完成关闭：SSE 响应不等待它，可能早于进程关闭结束；客户端此时立即对同一会话重发，会按会话忙碌规则被显式拒绝，不排队、不抢占（前端不自动重发，只有用户触发的重试会遇到）。

流式 `done`、`command_result` 和 `error` 是 transport 可消费的事件，不是新的跨子系统业务契约；它们的来源和调用顺序仍由本服务和 Contracts 共同约束。

### 11.2 非流式交付

非流式交付（`run_process(handle, stream=False)`）不订阅输出通道，等待进程 task 交出终态产出，再组装为 `NonStreamingCommandOutcome` 或 `NonStreamingAgentOutcome`；进程 task 结束时进程已经关闭，编排异常随后原样上抛。调用方在等待中被取消时，以 `detach_reason` 请求停止后传播取消，进程在自己的 task 中收尾；进程 task 被硬取消时没有终态，取消同样传给调用方。HTTP 只使用流式交付，非流式交付当前只有测试作为驱动使用；它的结果只含终态执行结果与 `session_id`，不含流式 `done` 的 `topic_id`、停止原因与记忆引用。

## 12. 取消与停止

`cancel_process` 是唯一的取消方法，取消的依据作为参数：

| 调用形式 | 依据 | 授权 | 找不到或无权时 |
|:---|:---|:---|:---|
| `cancel_process(handle, reason=…)` | 进程的所有者；交付方离开时的停止也经同一实现 | 不经进程控制授权；不接受另传的 access | 返回 `not_found`，不发布事件 |
| `cancel_process(process_id, access=…, reason=…)` | 以请求级 context 发起的控制请求（`POST /chat/stop`） | 经进程控制授权：请求方与进程记录驻留在同一 owner 与 Workspace（[身份与授权](./identity-and-authorization.md#4-操作授权)） | 返回 `not_found`，发布带请求方观测标签的事件 |

两种形式解析出进程后共用同一份实现：同步调用 `request_stop()`，首次 stop 固定 reason，重复 stop 返回同一判定且不重复取消 task，事件使用进程注册时绑定的观测标签。取消不新增 operation，不是 Agent action；跨 user/workspace 的取消得到 `not_found`，不泄露进程是否存在。注册入口不提供进程状态查询：流式交付以 `run_status`（进入 finalize）与 `done` 交付阶段与结局。

取消响应点只有两处：只有 Gateway 与 Actor 执行两个阶段会取消当前 `active_task`，因为这两个阶段在前台调用 LLM；进入 Actor 执行前统一检查一次停止请求（CPU 分配之后的检查与 `_run_interruptible` 的入口检查），Profile 解析、prepare 与 CPU 分配期间的停止请求都在这些工作照常完成后于此生效，命中即跳过 Actor 执行和 finalize，返回 cancelled 结果，并在关闭流程中释放租借、请求 cleanup；阶段交接窗口只记录 stop，下一阶段不会启动；Finalize 与 Terminal 拒绝 stop。停止在任何时刻都立即取消当前可中断阶段的子 task：Actor 中无论 CPU 在等模型还是阻塞在输出通道写入上。外部取消的入口是 `POST /chat/stop`（必须携带明确的 `process_id`）与交付方离开（HTTP 为客户端断连）。

**停止与执行线程**：停止请求同步记录停止并取消 Actor 子 task，进程的全部执行线程随之结束：存活判定读取进程结局，此后的 actor 操作与 CALL 派生以 `ThreadEndedError` 拒绝，停止路径不通知线程表。CPU 只用已产生的内容组装 cancelled 结果（CPU 端口要求取消后的收尾不依赖 actor 操作）；Actor 阶段收尾撤销凭据、接受检查点，之后的封口、会话写入、finalize 与 cleanup 都经进程 context 完成。停止那一刻 CPU 内的在途读取随子 task 的取消中断，已开始派发的写入受取消保护照常完成。

用户 stop 在编排骨架内被翻译为私有 `_ProcessCancelled` 分支，下游只传播原生 `asyncio.CancelledError`。`_run_interruptible()` 区分“stop 取消 child task”和“进程 task 自身被取消”；后者只来自系统关闭的硬取消，必须原样向上传播。child task 可能把取消转为返回值（CPU 在最外层这样做），等待方的取消请求会随之丢失，因此 child task 正常结束后同样按进入时记录的 `cancelling()` 判断，重新抛出硬取消。资源所有者在 unwind 中关闭自己创建的 LLM 流与 provider response；收尾异常只记录日志，不能替换正在传播的 `CancelledError`。

## 13. 关闭与系统停止

进程关闭只在进程 task 中完成（第 9 节）。`close_process(handle)` 关闭句柄对应的进程并等待进程 task 结束：进程从未运行时创建只执行关闭流程的进程 task，仍在运行时以 `stream_closed` 请求停止；这次等待可以被调用方取消，取消不影响进程自身的收尾；句柄已失效时是空操作。进程关闭后从进程表注销、绑定的 context 被撤销，此后取消按 `not_found` 处理。

当前进程表是进程内短期控制状态，不是可恢复的长期工作记录。进程重启后不能据此恢复进程；System 停止时 `TaskProcessService.shutdown()` 依次：直接关闭从未运行的进程，以 `system_shutdown` 停止已启动的进程，在 `workspace.process.shutdown_wait_seconds`（默认 10 秒）内等待全部进程 task 结束，超时仍未结束的进程 task 被取消（硬取消）并等待结束。硬取消时进程照常在自己的 task 中收尾，记录“终态前关闭”形态的 `chat.run.cancelled`；已进入 finalize 的进程拒绝停止，仍等提交结束。返回之前全部进程都已关闭，之后才清空会话服务。

## 14. 错误与观测

- `chat.run.*` 事件由领域 emitter `TaskProcessEventEmitter`（`workspace/process/events.py`）从进程记录投影阶段与终态，经 `RuntimeEventPublisher` best-effort 发布：发布失败不改变返回值，失败事件只携带 Workspace 领域错误码或固定摘要，不写入异常正文；
- Gateway 与 Actor 执行（CPU）的 task cancellation 不包装成普通 success 或 failed；Gateway timeout 仍按自己的 deadline/fallback 契约处理；
- failed done 保留 `reason=agent_run_failed`，CPU 有稳定原因时另带 `failure_reason`；失败原因在 done 与 `chat.run.failed` 中投影，字段见[事件契约](../contracts/routes-and-events.md#61-会话管理与-chat-sse)；
- 捕获的错误保留原始因果，不能通过“空列表/空话题”掩盖 route 缺失或契约违约。

新增入口或流程时，检查：是否复制了 prepare/run/finalize 顺序；是否把 command outcome 继续送进普通 chat；是否在取消后仍调用 finalize，或 finalize 失败后忘记 cleanup；是否把进程表、memory task 或 topic 的临时状态写成另一个权威来源。

## 15. 已知限制

- 进程控制授权只比对驻留的 owner 与 Workspace：同一 Workspace 下的请求方可以查询或停止其他请求方的进程；
- 断连后的 SSE 响应可能早于进程关闭结束，其间对同一会话的重发按会话忙碌规则拒绝；断连后继续运行、重连与多订阅者仍按 [Chat Run 生命周期 Idea](../ideas/chat-run-lifecycle-follow-ups.md) 处理；
- 唯一注册入口只接收主动 chat：命令只解析不执行，内置命令暂不可用；被动请求按[任务进程 Idea](../ideas/task-process-table-and-registration-entry.md) 2.6 节的决定暂不经注册入口；
- 用户可见长期任务与后台 Agent workflow 当前未立项，需在真实负载和执行能力成立后独立设计。

## 16. 代码与测试入口

- [`workspace/process/`](../../src/hivememory/workspace/process/)：注册入口 `service.py`、进程表 `table.py`、进程状态容器 `task_process.py`、执行器 `runner.py`、CPU 分配 `allocation.py`、命令终态 `command_terminal.py`、输出通道与阶段产出 `outputs.py`、事件投影 `events.py`、工作集 `working_set.py`；
- 封口：与 Import Bus 共用的 [`core/protocol/sealing.py`](../../src/hivememory/core/protocol/sealing.py)；
- 公共契约：[`workspace/contracts/cpu.py`](../../src/hivememory/workspace/contracts/cpu.py)、[`workspace/contracts/process.py`](../../src/hivememory/workspace/contracts/process.py)；HTTP 入口：[`server/routers/chat.py`](../../src/hivememory/server/routers/chat.py)；
- 测试：
  - `tests/unit/workspace/process/test_gateway_chat_flow.py`（含 Actor 中停止在等 CPU 与阻塞于通道写入时的响应）；
  - `tests/unit/workspace/process/test_chat_run_control_contract.py`（输出通道的容量与离开语义、订阅者暂停时进程仍推进、五种结局的唯一退出路径、交付方离开不等待关闭）；
  - `tests/unit/workspace/process/test_cpu_allocation.py`（CPU 分配、租借释放与关闭顺序）；
  - `tests/unit/workspace/process/test_cpu_port.py`（经 CPU 端口执行：测试 CPU 跑通任务进程、自报结局与执行异常）；
  - `tests/unit/workspace/process/test_command_terminal.py`、`test_process_events.py`（`chat.run.*` 投影与 best-effort 边界，交付方离开与硬取消的事件形态）；
  - `tests/unit/core/protocol/test_seal_interaction.py`（共用封口函数）；
  - `tests/unit/server/routers/test_chat.py`（注册后运行、断连以 `client_disconnected` 关闭交付、流从未开始时的关闭）；
  - `tests/integration/workspace/test_session_completion.py`（部分输出、停止晚于终态、交付方离开、检查点接受与系统关闭的停止、等待与硬取消）；
  - `tests/integration/workspace/test_process_intents.py`（已接纳写入跨进程终态保留、派发话题与线程 context 撤销）、`test_inflight_dispatch.py`（ACK 接纳时点、明确拒绝、未知结果与派发期间取消）；
  - `tests/e2e/system/test_chat_operation_whitelist.py`（没有任何 actor 操作的 Agent 经 HTTP 完成 chat，预检索与 Profile 仍按该 Agent 校验可见性）。
