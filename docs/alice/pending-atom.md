---
title: Alice PendingAtom Operations
status: current
owner: alice
scope: mtp-write-intent-adaptation-and-frame-artifacts
code_paths:
  - src/hivememory/agent_runtime/mtp/runtime.py
  - src/hivememory/agent_runtime/loop.py
  - src/hivememory/agent_runtime/runtime.py
  - src/hivememory/alice/orchestration/call/context.py
  - src/hivememory/workspace/contracts/operations.py
related_contracts:
  - docs/contracts/mtp.md
  - docs/contracts/subsystem-contracts.md
related_docs:
  - docs/architecture/workspace.md
  - docs/patchouli/generation.md
last_reviewed: 2026-10-18
---

# PendingAtom：Alice 的写入意图适配与回读

MTP `WRITE` / `UPDATE` 已被生成队列接纳、正式记忆尚未完成提取、去重、合并与持久化时，Agent 仍需要能够检查或引用刚提出的内容。PendingAtom 保留写入意图，以 pending alias 提供可读句柄，使 Agent 无需等待完整生成即可继续推理。ACK 表示登记与队列接纳都已成功，正式记忆仍由 Patchouli 异步生成；后续运行失败或取消不撤回已接纳的写入。

写入意图现在由 workspace 的进程级 `WriteIntentRegistry` 持有，正式 MemoryAtom 与物化决策由 Patchouli 持有。Alice 通过 CPU 驱动以线程 context 绑定的 `submit_operation` 提交和读取意图，负责把结果映射为 MTP 响应，不保存第二份登记、状态机、原子缓存或结算投影。登记模型、状态迁移、Workspace 回读范围、redirect 授权和缓存失效的唯一详细入口是 [Workspace 架构](../architecture/workspace.md)。

## 1. WRITE 与 UPDATE 的交付边界

`AliceCPU.execute(..., access=...)` 在 CPU 驱动内绑定主线程的线程 context，把 `submit_operation(request)` 交给统一执行路由。AgentRunService 将它交给 root frame 与 root 的每条 `MTPExecutionContext`；CALL 子帧使用 CPU 驱动为子线程绑定的另一个提交函数。请求只携带操作参数；`WorkspaceOperationEntry` 只接受线程 context，actor API 以其驻留 Workspace 为目标逐次授权，意图的 process ID 取自线程 context 的运行绑定。执行者不能选择目标或另交访问 context。

Koakuma 保留协议参数、frame 动词集合检查和错误格式化职责：

- WRITE 校验内容并构造 `WriteFocus`，提交 `SubmitWriteIntentRequest`；生成队列接纳后返回的 alias 进入 `ack` 文案。
- UPDATE 校验单 alias 与 instruction，提交 `SubmitUpdateIntentRequest`；可信基础 UUID 的解析、pending/缺失基础拒绝、登记、基础原子缓存失效与实时派发由能力层完成。
- 能力层的 pending 基础拒绝与缺失基础分别映射为既有 MTP 参数错误和 alias 不存在错误；操作授权拒绝映射为 MTP 权限错误。
- CPU 执行结束（或进程提前关闭）时同步撤销线程 context，后续请求以 `ThreadContextRevokedError` 拒绝，Alice 以结构化系统错误回填；task cancellation 继续原样传播。

这些动作均不在 Koakuma 内创建、修订或持久化正式 MemoryAtom。完整响应语义见 [MTP 契约](../contracts/mtp.md)。

## 2. 统一引用解析的消费

READ 与 CALL 的 `context_refs` 提交 `ResolveReferencesRequest`，RUN 的记忆目标提交 `ResolveRunTargetRequest`，UPDATE 的基础解析由对应能力方法完成，均使用 workspace 的同一引用解析。引用读取返回 core 中的 `ReferenceResolution`，由 MemoryCompiler 编译为既有 Agent 可读文本；Alice 不重新实现 L0/L1/L2 或资源授权。

同一 Workspace 的其他 Agent 或后续任务进程能够回读仍在飞行的意图；UPDATE 意图只对能读取其基础原子的 Agent 可见，否则在任何状态下都与不存在相同。结算后，旧 alias 可以解析为 canonical redirect；CALL 与 READ 只使用实际可读的 canonical 原子，不把无可读目标的 redirect 交给编译器。不同 Workspace 的意图与不存在相同。

SEARCH 提交 `RetrieveRequest`，经 `memory.search` 授权后检索；读取视图在代次未变时预热 workspace 原子缓存，随后的 READ 可直接命中。READ 与 RUN 在 actor API 返回前为交付的正式 atom 和可读 redirect 自动记录引用，同一请求按 UUID 去重，缓存命中也记录；pending、不可读目标及失败终态不记录。来源统一为 `workspace.reference_read`，记录失败只记日志，取消正常传播。UPDATE 基础解析不经此副作用。CALL 目标 Profile 也经操作请求与 workspace 读取视图取得，Alice 没有独立缓存。

## 3. frame 产物与进程收尾

Agent loop 只在收到 WRITE/UPDATE ACK 时把 alias 去重记入当前 frame 的 `harvested_aliases`。成功子帧的 `finalize_frame()` 投影这些句柄为 `FrameProducts.artifact_aliases`，供 caller 继续 READ 或通过 CALL 共享；失败、取消或预算耗尽的子帧不回填产物，但不撤回已提交的写入。UPDATE 的原基础 alias 不再被当作新产物补入。

CALL 的 alias 回填只服务于父 Agent 的当前认知。物化任务不从这些 alias 反推：workspace actor API 在每次 WRITE/UPDATE 内登记并提交 `MemoryIntent`，Patchouli 意图服务构造内部生成任务。`CPUExecutionResult` 与 `InteractionPayload` 都不承载物化任务，AgentRuntime 与 RunExecutor 不管理登记收尾。

CPU 执行结束时任务进程同步撤销全部线程 context；进程关闭先兜底撤销并释放附件租借，再等待 CPU 输出流关闭与 prepare cleanup。WRITE/UPDATE 在登记前同步复查线程；UPDATE 基础冷读跨越撤销时不登记。开始派发后，actor API 屏蔽调用方取消并持有提交任务至结束，再传播取消。主、子帧以各自提交函数授权，但都使用同一进程的真实话题，CPU 或 CALL 失败均不撤回已派发写入。明确拒绝得到结构化错误并标为 FAILED，未知结果保持 MATERIALIZING；结算事件由 workspace 登记订阅，Alice 不订阅它们。Patchouli 的生成、结算与持久化边界见[生成与物化](../patchouli/generation.md)。

## 4. 不变量与限制

- ACK、frame alias 回填和正式持久化是三个不同事实，不能互相替代。
- Alice 只消费操作提交函数与独立返回值，不持有 workspace 内部服务、登记或访问 context。
- 有输出的 completed/cancelled/failed 轮次均由任务进程封口、追加会话并进入 finalize；显式写入在操作内实时派发，不取决于进程终态。
- 登记属于进程内状态。EXPIRED 是 SETTLED、FAILED、CANCELLED 共同的永久终态，模型、引用解析种类与渲染文案都保留，但当前没有转入它的触发，终态句柄保留到重启。
- 当前没有持久化登记、重启恢复、事件补齐或终态句柄回收；派发结果未知或结算事件丢失时意图可能保持 MATERIALIZING。
- CALL 子帧以被调用方身份提交，意图的发起者是被调用方，进程关联与根帧相同；子帧沿用注册观测标签，标签不参与授权。

## 5. 代码与验证入口

- [MTP 动词处理器](../../src/hivememory/agent_runtime/mtp/verbs/)：提交意图、消费引用结果；[Koakuma Runtime](../../src/hivememory/agent_runtime/mtp/runtime.py) 统一映射错误；
- [Agent loop](../../src/hivememory/agent_runtime/loop.py)：记录 ACK alias；
- [AgentRuntime](../../src/hivememory/agent_runtime/runtime.py)：成功子帧产物投影；
- [操作请求与提交函数](../../src/hivememory/workspace/contracts/operations.py)：Alice 消费的稳定契约；
- [MTP 操作请求集成测试](../../tests/integration/mtp/test_process_operations.py)：跨轮回读、redirect、关闭、权限拒绝与 UPDATE 错误；
- [HTTP chat MTP 操作 E2E](../../tests/e2e/system/test_chat_mtp_operations.py)：真实 HTTP、任务进程与 Alice MTP 的确定性链路，覆盖跨轮回读上一进程的写入意图，以及同轮 SEARCH、READ、WRITE、CALL 共享引用、意图实时派发与引用记录。
