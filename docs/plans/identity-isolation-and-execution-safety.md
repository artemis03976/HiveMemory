---
title: Identity Isolation and Execution Safety
status: planned
owner: system
scope: cross-subsystem-identity-authorization-and-execution-safety
code_paths:
  - src/hivememory/core/models/identity.py
  - src/hivememory/agent_runtime/
  - src/hivememory/alice/runtime/
  - src/hivememory/patchouli/memory_library/
  - src/hivememory/server/
  - frontend/
related_docs:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/mtp.md
  - docs/alice/README.md
  - docs/alice/mtp-runtime.md
  - docs/alice/orchestration.md
  - docs/todo/frontend-identity-ownership.md
last_reviewed: 2026-08-03
---

# 身份隔离与执行安全计划

HiveMemory 已经把 Identity、MemoryVisibility、MTP permission、Agent Profile 和 MemoryLibrary 可见性写进了设计，但当前仍存在“拿到 alias/id 就可以继续使用”“进程级 cache 共享可变状态”“前端 user id 看起来像认证”“RUN 能执行代码但不是安全沙箱”等边界缺口。

这不是单纯的登录页面工作，也不是给每个 handler 再加一个 `if user_id`。它需要统一回答：一次请求、一次 Agent run、一个 PendingAtom、一个 Profile、一个 MemoryAtom 和一段可执行资产分别属于谁；哪个组件拥有最终授权权；缓存、子帧、重试和后台任务如何继承或缩小权限。

## 1. 当前问题证据

| 边界 | 当前实现基础 | 当前风险 |
|:---|:---|:---|
| Memory visibility | MemoryAtom 有 user/team/session/visibility，Patchouli 是可见性所有者 | 某些别名 L0/L1 cache 命中不会再次按调用 Identity 校验 |
| PendingAtom | atom 保存 identity，Alice 通过 alias/intent 解析 | 进程级 cache/store 与并发 run 共享，跨用户隔离尚未完全成立 |
| Agent Profile | Profile 作为 MemoryAtom，通过 retrieval/alias 发现 | alias cache 进程级、失效不完整；显式 Profile 加载失败与未指定 Profile 的 Omni-Doll fallback 语义混淆 |
| Agent run/frame | `ExecutionFrame`、`RunSession`、frame policy 与 `AgentRunStreamAdapter` | frame registry、CALL record、取消信号、输出队列和流序号已按 run 隔离；跨用户身份与缓存隔离仍待验证 |
| MTP permission | Prompt 与 Koakuma runtime 有双层权限设计 | prompt 教学不是硬安全保证，部分身份/权限重新校验仍需收紧 |
| MTP READ/RUN | READ 可访问记忆，RUN 可执行 memory code | RUN 没有强沙箱、资源限制、可信资产分级或强制审批边界 |
| Frontend identity | 前端已有默认 user id 和待办的 identity store 方向 | UI 字段不是认证/授权边界，多个请求可能使用不同默认身份；切换时 cache/stream 清理不完整 |
| Observability | RuntimeEvent 可携带 identity/run/frame/atom 关联 | 事件 payload 不能成为授权依据，也不能泄漏不应被当前身份看到的内容 |

## 2. 目标与非目标

### 2.1 目标

1. 建立从 request -> run/frame -> MTP -> Patchouli -> background work 的 identity propagation 和 scope 缩小规则；
2. 所有 Memory、Profile、PendingAtom、Artifact 和执行资产的读取/写入都由实际所有者重新校验 Identity；
3. 使 cache key、frame、cancel、task 和 retry 不会跨用户或 workspace 共享可变授权状态；
4. 区分“未指定 Profile”“指定 Profile 不存在”“Profile 无权访问”“Profile 已失效”四种结果；
5. 为 MTP RUN 建立可信资产、能力白名单、资源限制、取消和审计边界；
6. 让前端身份状态与后端认证/授权契约对齐，同时明确前端不是安全边界；
7. 为并发、越权、缓存污染、CALL 深度和执行资产逃逸提供测试与故障样本。

### 2.2 非目标

- 不在本计划中选择具体身份提供商、OAuth 产品或多租户商业方案；
- 不把 prompt 中的 verb 教学当作安全控制；
- 不让 Gateway、Alice 或前端自行决定 Patchouli 的长期可见性；
- 不承诺把任意 Python 代码变成安全可执行资产；
- 不为了实现身份隔离而把所有对象复制到每个用户的独立数据库；
- 不把 RuntimeEvent、日志或前端 localStorage 当作认证状态。

## 3. 目标权限模型

### 3.1 Identity 是不可变运行上下文

每个 request/run/frame/work item 都应携带不可变的身份快照和 scope：

```text
Identity
  -> user_id
  -> team/workspace scope
  -> agent/profile identity
  -> visibility capability
  -> request/run correlation
```

子 Agent 默认继承父 run 的用户和 workspace 边界，只能通过显式、经授权的 `context_refs` 缩小或选择可见资产；不能通过自然语言或 alias 自行扩大 scope。

### 3.2 所有者重新校验

Cache 命中、MTP READ/RUN、PendingAtom resolution、Artifact ref 读取、MemoryLibrary archive/revive 和 background retry 都必须由最终状态所有者重新验证 Identity。上游已检查不能替代下游检查，因为请求可能跨越重试、队列和子系统边界。

### 3.3 缓存不承载授权

缓存 key 至少需要包含实际 scope，或缓存值必须在命中后重新验证。失效不完整时宁可返回 miss，也不能返回另一个用户最近访问的 Profile、MemoryAtom、PendingAtom 或 compiled context。

### 3.4 可执行资产是更高风险能力

MTP RUN 应将“可读取的 Memory”与“可执行的 Memory”分开：

- 资产必须显式声明 executable capability、来源和信任级别；
- runtime 只能执行允许的 verb/tool 和受限资源；
- timeout、取消、输出大小、文件/网络访问和进程边界必须有硬限制；
- 执行结果与失败必须带 run/asset identity，不能只依赖 prompt；
- 无法提供强隔离时，默认拒绝不受信任资产，或明确降级为展示/提议而非执行。

## 4. 分阶段实施

### Phase S0：身份与威胁模型清单

1. 列出所有 request、route、MTP verb、cache、work item、Artifact 和 MemoryLibrary 操作的身份输入与授权所有者；
2. 画出主 Agent、子 Agent、后台 task、scheduler 和 frontend 的身份继承/缩小关系；
3. 建立越权、alias 污染、Profile stale、重试跨用户、stream 切换和执行逃逸的最小复现样本；
4. 明确当前只支持单用户/单 workspace 的地方，不在文档中暗示已经有完整租户隔离。

### Phase S1：Patchouli 与 Alice 身份收紧

1. 修复 L0/L1 alias 命中不重新校验 Identity；
2. 让 PendingAtom store/cache、Profile cache 和 compiled context 按 scope 隔离或命中后重验；
3. 为 MemoryLibrary、Artifact、archive/revive 和后台恢复入口统一 scope 检查；
4. 对显式 Profile 解析失败、权限拒绝和未指定 Profile 分别返回稳定结果；
5. 将失败 reason 和安全摘要写入可观察事件，但不泄漏不可见正文。

### Phase S2：Run-local 执行隔离（基础已完成）

1. 已完成：以 `RunSession` 替代共享 FrameScheduler stack，将 frame registry 和 CALL record 收敛为 run-local 状态；Chat application 在上层持有可取消阶段 task，流式输出队列与流序号由每次 run 独占的 `AgentRunStreamAdapter/QueueAgentRunOutput` 持有，运行预算由 frame policy 持有；
2. 继续验证并发 Agent run 的 CALL、READ、WRITE、citation、PendingAtom 和 cancel 不会跨用户或 workspace 交叉；
3. 已完成：被调用 frame 继承 caller Identity，CALL 权限由 `FrameExecutionPolicy` 和 Profile capability 硬检查，不再以 depth 作为控制依据；
4. 明确恢复/重试时不能复用已经失效的授权快照。

### Phase S3：执行资产安全

1. 为 executable Memory 建立来源、信任、审批和 capability 模型；
2. 先实现最小白名单与资源限制，再评估 subprocess/container/sandbox 方案；
3. 为文件、网络、环境变量、进程、输出和超时建立拒绝默认策略；
4. 对取消、超时、异常和部分副作用建立安全终态与 reconciliation；
5. 只有强隔离证据成立后，才允许在产品文案中将 RUN 描述为可执行能力。

### Phase S4：Frontend 与外部契约对齐

1. 前端身份 store 只作为请求上下文，不作为认证/授权来源；
2. 所有请求从同一 identity context 派生，切换/登出清理或隔离 chat、topic、memory cache 和 streams；
3. 与后端认证、session、workspace 和错误契约对齐；
4. UI 明确展示“当前身份/权限”与“后端拒绝”，不把默认 user id 当成登录状态。

## 5. 验收标准

- 任意 Memory/Artifact/Profile/PendingAtom alias 命中都经过实际 Identity scope 校验；
- 两个并发用户使用相同 alias、topic 或 Profile 名称不会读取对方状态；
- 子 Agent、后台 retry 和恢复任务不会扩大或错误继承身份权限；
- FrameScheduler 已删除，cancel、budget、frame registry 与 CALL record 按 run 隔离，并发 CALL/cancel/恢复测试稳定通过；PendingAtom 与 cache 的跨用户隔离仍需按本计划验证；
- 指定 Profile 失败不会静默加载全权限 Omni-Doll；未指定 Profile 的 fallback 仍有明确且可观察语义；
- MTP RUN 在未满足可信资产和硬限制时拒绝执行或明确降级，不能把 prompt 当安全边界；
- 前端身份切换不会留下旧用户的请求、缓存、stream 或页面状态；
- 越权、缓存污染、重试跨用户、CALL 深度和执行逃逸均有回归测试；
- Contracts、Alice、Patchouli、Frontend、Help 和 API 错误说明保持一致。

## 6. 依赖与风险

本计划依赖[运行时状态持久化与故障恢复](./runtime-state-durability-and-recovery.md)处理身份快照、任务恢复和 PendingAtom ledger，也依赖[跨子系统幂等性与重试语义](./cross-subsystem-idempotency-and-retry.md)防止重复操作跨用户复用。最大风险是把“身份字段已在模型中”误认为“安全边界已经成立”；完成判断必须以越权测试和失败路径为准。
