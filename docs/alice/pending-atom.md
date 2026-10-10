---
title: Alice PendingAtom Operations
status: current
owner: alice
scope: mtp-write-intent-adaptation-and-frame-artifacts
code_paths:
  - src/hivememory/agent_runtime/mtp/runtime.py
  - src/hivememory/agent_runtime/execution/loop.py
  - src/hivememory/agent_runtime/runtime.py
  - src/hivememory/alice/orchestration/sub_agent/call_context_provider.py
  - src/hivememory/workspace/contracts/operations.py
related_contracts:
  - docs/contracts/mtp.md
  - docs/contracts/subsystem-contracts.md
related_docs:
  - docs/architecture/workspace.md
  - docs/patchouli/generation.md
last_reviewed: 2026-10-09
---

# PendingAtom：Alice 的写入意图适配与回读

MTP `WRITE` / `UPDATE` 已被接收、正式记忆尚未完成提取、去重、合并与持久化时，Agent 仍需要能够检查或引用刚提出的内容。只返回一句“已收到”无法支持继续推理，而同步创建 MemoryAtom 又会把半完成或随后取消的运行提前变成长久事实。PendingAtom 因而保留写入意图，并以 pending alias 提供可读句柄；ACK 始终只表示登记成功。

写入意图现在由 workspace 的进程级 `WriteIntentRegistry` 持有，正式 MemoryAtom 与物化决策由 Patchouli 持有。Alice 通过 CPU 驱动交付的凭据绑定 `submit_operation` 提交和读取意图，负责把结果映射为 MTP 响应，不保存第二份登记、状态机、原子缓存或结算投影。登记模型、状态迁移、Workspace 回读范围、redirect 授权和缓存失效的唯一详细入口是 [Workspace 架构](../architecture/workspace.md)。

## 1. WRITE 与 UPDATE 的交付边界

`AliceCPU.execute(..., credential=...)` 在 CPU 驱动内绑定不透明执行凭据，把 `submit_operation(request)` 交给统一执行路由。AgentRunService 将同一个函数交给 root frame、每条 `MTPExecutionContext` 与 CALL 子帧。请求只携带操作参数；`WorkspaceOperationEntry` 从凭据表恢复访问 context、固定目标 Workspace 与 process ID，能力层再逐次授权。执行者不能选择目标或另交访问 context。

Koakuma 保留协议参数、Profile verb/tool 白名单和错误格式化职责：

- WRITE 校验内容并构造 `WriteFocus`，提交 `SubmitWriteIntentRequest`；登记返回的 alias 进入既有 `ack` 文案。
- UPDATE 校验单 alias 与 instruction，提交 `SubmitUpdateIntentRequest`；可信基础 UUID 的解析、pending/缺失基础拒绝、登记和基础原子缓存失效由能力层完成。
- 能力层的 pending 基础拒绝与缺失基础分别映射为既有 MTP 参数错误和 alias 不存在错误；操作授权拒绝映射为 MTP 权限错误。
- 进程关闭同步吊销凭据，后续请求以 `ExecutionCredentialRevokedError` 拒绝，Alice 以结构化系统错误回填；task cancellation 继续原样传播。

这些动作均不在 Koakuma 内创建、修订或持久化正式 MemoryAtom。完整响应语义见 [MTP 契约](../contracts/mtp.md)。

## 2. 统一引用解析的消费

READ、RUN 的记忆目标与 CALL 的 `context_refs` 提交 `ResolveReferencesRequest`，UPDATE 的基础解析由对应能力方法完成，均使用 workspace 的同一引用解析。引用读取返回 core 中的 `ReferenceResolution`，由 MemoryCompiler 编译为既有 Agent 可读文本；Alice 不重新实现 L0/L1/L2 或资源授权。

同一 Workspace 的其他 Agent 或后续任务进程能够回读仍在飞行的意图；UPDATE 意图只对能读取其基础原子的 Agent 可见，否则在任何状态下都与不存在相同。结算后，旧 alias 可以解析为 canonical redirect；CALL 与 READ 只使用实际可读的 canonical 原子，不把无可读目标的 redirect 交给编译器。不同 Workspace 的意图与不存在相同。

SEARCH 提交 `RetrieveRequest`，经 `resource.search` 授权后检索；读取视图在代次未变时预热 workspace 原子缓存，随后的 READ 可直接命中。引用读取在能力层返回前为交付的正式 atom 和可读 redirect 自动记录引用，同一请求按 UUID 去重，缓存命中也记录；pending、不可读目标及失败终态不记录。来源统一为 `workspace.reference_read`，记录失败只记日志，取消正常传播。UPDATE 基础解析不经此副作用。CALL 目标 Profile 也经操作请求与 workspace 读取视图取得，Alice 没有独立缓存。

## 3. frame 产物与进程收尾

Agent loop 只在收到 WRITE/UPDATE ACK 时把 alias 去重记入当前 frame 的 `harvested_aliases`。成功子帧的 `finalize_frame()` 投影这些句柄为 `FrameProducts.artifact_aliases`，供 caller 继续 READ 或通过 CALL 共享；失败、取消或预算耗尽的子帧不回填产物，CallCoordinator 还会提交 `CancelIntentsRequest` 撤回这些句柄中仍为 PENDING 的意图。UPDATE 的原基础 alias 不再被当作新产物补入。

CALL 的 alias 回填只服务于父 Agent 的当前认知。物化任务不从这些 alias 反推：任务进程只在 CPU completed 后从登记认领本进程的 PENDING 意图，并封入 `InteractionPayload.materialize_tasks`，随后进入 Patchouli finalize。`CPUExecutionResult` 不承载物化任务，AgentRuntime 与 RunExecutor 不再认领、取消或回收登记记录。

根进程关闭先同步吊销凭据、取消本进程仍为 PENDING 的记录并释放附件租借，再等待 CPU 输出流关闭与 prepare cleanup；若在途 UPDATE 冷读于关闭后返回，入口同步补偿新登记的 alias 并抛吊销错误，不取消调用方任务。已在途的只读请求自然完成。已经认领的 MATERIALIZING 记录保持原状。子帧与根帧共享操作提交函数；子帧未成功结束时撤回它已收到 ACK 的意图，作为取消语义的一部分，其余意图仍由根进程终态处理。结算事件由 workspace 登记订阅，AliceBridge 不再订阅它们。Patchouli 的生成、结算与持久化边界见[生成与物化](../patchouli/generation.md)。

## 4. 不变量与限制

- ACK、frame alias 回填和正式持久化是三个不同事实，不能互相替代。
- Alice 只消费操作提交函数与独立返回值，不持有 workspace 内部服务、登记或访问 context。
- 只有 completed 的任务进程默认进入 finalize；取消与失败不会把尚为 PENDING 的意图派发给 Patchouli。
- 登记属于进程内状态，终态句柄保留到重启，不再产生 EXPIRED；模型中的兼容枚举与渲染种类保留。
- 当前没有持久化登记、重启恢复、事件补齐或终态句柄回收；finalize 失败后已认领意图可能保持 MATERIALIZING。
- CALL 使用主线程凭据绑定的同一个提交函数，意图发起者仍为主线程 Actor；子帧沿用注册观测标签，不另行取得资源授权身份。

## 5. 代码与验证入口

- [Koakuma Runtime](../../src/hivememory/agent_runtime/mtp/runtime.py)：提交意图、消费引用结果与错误映射；
- [Agent loop](../../src/hivememory/agent_runtime/execution/loop.py)：记录 ACK alias；
- [AgentRuntime](../../src/hivememory/agent_runtime/runtime.py)：成功子帧产物投影；
- [操作请求与提交函数](../../src/hivememory/workspace/contracts/operations.py)：Alice 消费的稳定契约；
- [MTP 操作请求集成测试](../../tests/integration/mtp/test_process_operations.py)：跨轮回读、redirect、关闭、权限拒绝与 UPDATE 错误；
- [HTTP chat 跨轮回读 E2E](../../tests/e2e/system/test_intent_read_chat.py)：真实 HTTP、任务进程与 Alice MTP 的确定性链路。

- [HTTP 单轮能力迁移 E2E](../../tests/e2e/system/test_alice_capability_chat.py)：同轮 SEARCH、READ、WRITE、CALL 共享引用、意图认领与引用记录。
