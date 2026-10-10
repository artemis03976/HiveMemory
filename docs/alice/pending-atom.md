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

写入意图现在由 workspace 的进程级 `WriteIntentRegistry` 持有，正式 MemoryAtom 与物化决策由 Patchouli 持有。Alice 通过任务进程交付的 `ProcessOperations` 提交和读取意图，负责把结果映射为 MTP 响应，不保存第二份登记、状态机、原子缓存或结算投影。登记模型、状态迁移、Workspace 回读范围、redirect 授权和缓存失效的唯一详细入口是 [Workspace 架构](../architecture/workspace.md)。

## 1. WRITE 与 UPDATE 的交付边界

`AliceCPU.execute(..., operations=...)` 把进程绑定的操作端口传给统一执行路由。AgentRunService 将端口交给 root frame，单帧循环把同一个端口写入每条 `MTPExecutionContext`；CALL 子帧继续使用它。端口不允许执行者提交访问 context、目标 Workspace 或 process ID，这些可信值由进程绑定。

Koakuma 保留协议参数、Profile verb/tool 白名单和错误格式化职责：

- WRITE 校验内容并构造 `WriteFocus`，调用 `submit_write_intent()`；登记返回的 alias 进入既有 `ack` 文案。
- UPDATE 校验单 alias 与 instruction，调用 `submit_update_intent()`；可信基础 UUID 的解析、pending/缺失基础拒绝、登记和基础原子缓存失效由能力层完成。
- 能力层的 pending 基础拒绝与缺失基础分别映射为既有 MTP 参数错误和 alias 不存在错误；操作授权拒绝映射为 MTP 权限错误。
- 关闭后的通道拒绝资源操作，Alice 以结构化系统错误回填；task cancellation 继续原样传播。

这些动作均不在 Koakuma 内创建、修订或持久化正式 MemoryAtom。完整响应语义见 [MTP 契约](../contracts/mtp.md)。

## 2. 统一引用解析的消费

READ、RUN 的记忆目标、UPDATE 基础和 CALL 的 `context_refs` 均由操作端口进入 workspace 引用解析。端口返回 core 中的 `ReferenceResolution`，由 MemoryCompiler 编译为既有 Agent 可读文本；Alice 不重新实现 L0/L1/L2 或资源授权。

同一 Workspace 的其他 Agent 或后续任务进程能够回读仍在飞行的意图；UPDATE 意图只对能读取其基础原子的 Agent 可见，否则在任何状态下都与不存在相同。结算后，旧 alias 可以解析为 canonical redirect；CALL 与 READ 只使用实际可读的 canonical 原子，不把无可读目标的 redirect 交给编译器。不同 Workspace 的意图与不存在相同。

SEARCH 和 prepare 结果不再预热 Alice 原子缓存。SEARCH、引用记录和 CALL 目标 Profile 解析仍通过 Alice local bus 请求 Patchouli；CALL Profile 的独立运行时缓存保留，不能把它与 workspace 的 canonical 读取缓存混为一层。

## 3. frame 产物与进程收尾

Agent loop 只在收到 WRITE/UPDATE ACK 时把 alias 去重记入当前 frame 的 `harvested_aliases`。成功子帧的 `finalize_frame()` 投影这些句柄为 `FrameProducts.artifact_aliases`，供 caller 继续 READ 或通过 CALL 共享；失败、取消或预算耗尽的子帧不回填产物，CallCoordinator 还会经操作端口撤回这些句柄中仍为 PENDING 的意图。UPDATE 的原基础 alias 不再被当作新产物补入。

CALL 的 alias 回填只服务于父 Agent 的当前认知。物化任务不从这些 alias 反推：任务进程只在 CPU completed 后从登记认领本进程的 PENDING 意图，并封入 `InteractionPayload.materialize_tasks`，随后进入 Patchouli finalize。`CPUExecutionResult` 不承载物化任务，AgentRuntime 与 RunExecutor 不再认领、取消或回收登记记录。

根进程关闭时取消本进程仍为 PENDING 的记录，已经认领的 MATERIALIZING 记录保持原状。子帧与根帧共享进程端口；子帧未成功结束时撤回它已收到 ACK 的意图，作为取消语义的一部分，其余意图仍由根进程终态处理。结算事件由 workspace 登记订阅，AliceBridge 不再订阅它们。Patchouli 的生成、结算与持久化边界见[生成与物化](../patchouli/generation.md)。

## 4. 不变量与限制

- ACK、frame alias 回填和正式持久化是三个不同事实，不能互相替代。
- Alice 只消费端口与独立返回值，不持有 workspace 内部服务、登记或访问 context。
- 只有 completed 的任务进程默认进入 finalize；取消与失败不会把尚为 PENDING 的意图派发给 Patchouli。
- 登记属于进程内状态，终态句柄保留到重启，不再产生 EXPIRED；模型中的兼容枚举与渲染种类保留。
- 当前没有持久化登记、重启恢复、事件补齐或终态句柄回收；finalize 失败后已认领意图可能保持 MATERIALIZING。
- CALL 使用主线程绑定的同一个端口，意图发起者仍为主线程 Actor；子帧身份与权限策略不在本次迁移中改写。

## 5. 代码与验证入口

- [Koakuma Runtime](../../src/hivememory/agent_runtime/mtp/runtime.py)：提交意图、消费引用结果与错误映射；
- [Agent loop](../../src/hivememory/agent_runtime/execution/loop.py)：记录 ACK alias；
- [AgentRuntime](../../src/hivememory/agent_runtime/runtime.py)：成功子帧产物投影；
- [操作端口](../../src/hivememory/workspace/contracts/operations.py)：Alice 消费的稳定契约；
- [MTP 操作通道集成测试](../../tests/integration/mtp/test_process_operations.py)：跨轮回读、redirect、关闭、权限拒绝与 UPDATE 错误；
- [HTTP chat 跨轮回读 E2E](../../tests/e2e/system/test_intent_read_chat.py)：真实 HTTP、任务进程与 Alice MTP 的确定性链路。
