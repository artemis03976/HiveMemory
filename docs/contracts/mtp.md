---
title: Memory Tool Protocol
status: current
owner: alice
scope: mtp
code_paths:
  - src/hivememory/core/mtp/
  - src/hivememory/agent_runtime/mtp/runtime.py
  - src/hivememory/alice/orchestration/run_scheduler.py
  - src/hivememory/alice/orchestration/call_coordinator.py
  - src/hivememory/core/models/agent.py
related_contracts:
  - docs/contracts/error-model.md
  - docs/contracts/routes-and-events.md
last_reviewed: 2026-08-03
---

# Memory Tool Protocol (MTP)

MTP 是 Alice Agent 在生成循环中调用记忆、系统工具和子 Agent 的进程内文本协议。它延续了项目早期“Memory as a Tool”的核心判断：记忆不能只在回答开始前由系统预检索一次，还必须允许 Agent 在任务展开后主动发现、检查、使用和修订知识。

预检索适合提供一个低成本起点，却无法预知推理过程中出现的所有信息需求。一个 Agent 可能先读到摘要，随后才知道需要哪条原始证据；也可能在执行中形成值得长期保留的新事实，或需要把子任务委派给另一个 Agent。MTP 把这些动作放进生成循环，使记忆从静态上下文变成受权限和生命周期约束的可调用能力。

选择文本协议，是因为它可以直接出现在模型输出中，不绑定某一家模型供应商的 function calling 形态，也便于在消息历史中保留调用与响应。代价是文本可能不完整、参数可能含糊，且模型发出的指令不能天然视为可信调用。因此 parser 只负责把文本解析为结构化请求，KoakumaRuntime 仍必须执行权限、类型、身份、取消和错误约束；MTP 不是绕过 Runtime 的自由格式命令通道。

本文只描述当前 parser、runtime、formatter 和测试已经支持的行为。协议的理念不应被写成尚未实现的安全保证或自主能力。

## 1. 协议语法

```text
⟪ VERB | TARGET | ARGS ⟫
```

- 左右定界符：`⟪` / `⟫`；
- 分隔符：`|`，仅前两个用于分段，ARGS 内的 `|` 保留为内容；
- VERB 不区分输入大小写，解析后规范化为大写；
- TARGET 支持 `*` / `global`、单 alias、`[alias1, alias2]`；
- ARGS 支持 `key="value"`、可多行的 ``key=`raw content` `` 和 `key=["a", "b"]`；
- LLM stop sequence 是右定界符；`complete_and_parse` 可以补齐被 stop 截断的右定界符；
- parser 只解析文本中出现的第一条完整 MTP 指令。

`⟪` / `⟫` 的选择是为了降低与普通代码、Markdown、XML 以及自然语言括号冲突的概率；`VERB | TARGET | ARGS` 则把 action、object 与 details 显式分开，使 parser 可以保持小而确定。只取第一条完整指令也是有意的控制流串行化：每次执行的结果会改变 alias、权限可见性、PendingAtom 或 frame 状态，Agent 应看到结果后再决定下一步，而不是在一段文本中提交任意多条异构命令批处理。

示例：

```text
⟪ SEARCH | * | query="Gateway 的边界" filter="type:fact" ⟫
⟪ READ | [fact_gateway, fact_bus] | ⟫
⟪ WRITE | * | title="设计约束" content=`跨子系统只走公开路由。` reason="长期约束" ⟫
⟪ CALL | reviewer | task="检查这个方案" context_refs=["fact_gateway"] ⟫
```

## 2. 执行位置

`KoakumaRuntime` 负责 parse、权限检查、verb 分发、结果计时和格式化。它属于 Alice，使用 Alice local bus 映射的 Patchouli 公开路由访问记忆能力。

Agent loop 检测 MTP 文本后暂停自然语言生成，执行指令并把格式化结果回填到消息历史，再继续生成。CALL 的 `suspend` 由 Alice `RunScheduler`/`CallCoordinator` 消费，不直接回填为空结果；完成后由 `AgentRuntime.apply_call_response()` 一次性写回 caller history 和 `tool_result`。

## 3. 动词契约

六个动词共同覆盖 Agent 使用记忆时的基本闭环：SEARCH 用于发现，READ 用于检查，RUN 用于使用，WRITE 用于创建，UPDATE 用于修订，CALL 用于委派。它们不是六个任意工具名，而是刻意区分了“寻找什么”“确认内容”“执行能力”“提出长期变更”和“转交控制”等不同责任。

### 3.1 SEARCH

```text
⟪ SEARCH | * | query="..." filter="..." ⟫
```

- 必填 `query`；
- 可选 `filter`，当前支持 memory type、tags 和 vitality 等解析规则；
- filter 中的非法 token 不使整次搜索失败，而是忽略过滤并返回 warning；
- 经 `patchouli.public.memory.retrieve` 检索；
- 结果由 MemoryCompiler 编译为 Retrieval Context；
- 命中的完整 MemoryAtom 写入当前 Koakuma alias cache；
- 空结果仍为 `success`，并带 `no_memories_found` warning。

SEARCH 返回可继续消费的检索上下文，而不是把“没有找到”当成系统故障。检索本身具有不确定性，空结果只说明当前查询没有证据；Agent 仍可以改写查询、继续回答或明确告知信息不足。

### 3.2 READ

```text
⟪ READ | alias | ⟫
⟪ READ | [alias_a, alias_b] | ⟫
```

- 不支持 wildcard；
- alias 可以解析为正式 atom、pending、redirect 或 discarded/failed/expired 终态；
- 每类结果均经 MemoryCompiler 的 `MTP_READ` target 编译；
- 全部 alias 未命中时返回 error；
- 部分未命中时返回已解析内容，并把缺失 alias 放入 warnings；
- 正式 atom 和 redirect 命中会记录 `mtp.read` citation。

alias 是运行期稳定称呼，不等于永久 UUID。它让模型使用可读、短小的引用，又允许 Runtime 把同一名称解析为正式 atom、尚未结算的 PendingAtom 或修订后的 redirect。redirect 保留旧称呼的可追踪性，但通过 warning 提醒 Agent 目标已经演化；terminal 状态则防止一个失败或过期意图继续伪装成有效记忆。

READ 列表是协议中显式支持的批量读取，而不是多命令特例。Agent 在 SEARCH 后往往需要检查数条候选证据；一次 READ 多个 alias 可以减少额外的生成/执行轮次，同时仍让 Runtime 对每个 alias 独立解析并把部分失败表达为 warning。

### 3.3 RUN

```text
⟪ RUN | sys_tool_alias | key="value" ⟫
⟪ RUN | code_memory_alias | key="value" ⟫
```

两层分发：

1. `sys_` 工具或 Kernel Registry 中的工具走注册 syscall；
2. 其他 alias 经 PendingAtom/正式记忆解析，只允许 `CODE_SNIPPET` MemoryAtom。

系统工具还受 Agent Profile 的 `allowed_sys_tools` 限制。redirect 可以执行但会附带 warning；pending 和 terminal alias 不可执行。成功执行记忆工具会记录 `mtp.run` citation。

当前用户代码通过本地执行器运行，尚无可作为安全边界的强隔离沙箱、进程级资源限制和真取消。不得把 RUN 描述为安全执行不受信任代码的能力。

RUN 被保留在记忆协议中，是因为一部分记忆不仅需要被阅读，还可能代表可执行的代码资产或注册工具。但“能被调用”不等于“已被安全隔离”：权限白名单只约束可见的调用面，不能替代操作系统级沙箱、资源限制和可信来源审查。

### 3.4 WRITE

```text
⟪ WRITE | * | content=`...` title="..." reason="..." ⟫
```

- `content` 必填；`title`、`reason` 可选；
- 立即注册 PendingAtom 并返回 `ack + pending_alias`；
- 不在 Koakuma 内同步创建正式 MemoryAtom；
- PendingAtom materialize task 随 `AgentRunResult` 交给 Patchouli finalize；
- 只有完成 finalize 后，后续生成/结算流程才可能形成正式记忆。

ACK 表示意图已被运行时接收，不表示长期记忆已经持久化。

延迟物化保护了长期记忆免受半完成运行污染。WRITE 发生时，Agent 仍可能在后续迭代中失败、取消或修正自己的判断；如果 Koakuma 立即写入正式 MemoryAtom，执行事务尚未完成就会产生难以撤销的长期事实。PendingAtom 让本轮可以引用刚提出的内容，同时把正式生成、来源归约和持久化留给成功后的 Patchouli finalize。

### 3.5 UPDATE

```text
⟪ UPDATE | alias | instruction="..." content=`...` ⟫
```

- TARGET 必须是单 alias；`instruction` 必填，`content` 可选；
- 目标必须解析为正式 atom，pending alias 不能再次 UPDATE；
- 注册以原记忆 UUID 为基线的 pending revision；
- 使当前 alias cache 失效，防止后续脏读；
- 返回 `ack + pending_alias`，实际更新延迟到 Patchouli finalize 后处理。

UPDATE 同样不原地覆盖旧记忆。它以正式 atom 为基线创建 pending revision，使当前 run 能表达修订意图，又保留旧版本和来源链；alias cache 立即失效，是为了避免 Agent 在同一轮继续把待修订内容当成无变化的权威事实。

### 3.6 CALL

```text
⟪ CALL | agent_alias | task="..." context_refs=["alias_a"] ⟫
```

- TARGET 和 `task` 必填；`context_refs` 可选；
- Koakuma 返回 `suspend` 和结构化 `MTPCallRequest`；
- Alice RunScheduler 挂起 caller frame、解析共享上下文、运行 callee frame，再以 `MTPCallResponse` 回填；
- CALL 只允许 root frame 发起；callee 的 `FrameExecutionPolicy` 显式移除 CALL，防止递归爆炸；
- 只有 `COMPLETED` 子帧产生 success CALL response，并可以返回其 PendingAtom alias；
- `CANCELLED` 保持 cancelled 终态，`FAILED`、`BUDGET_EXHAUSTED` 和意外 `SUSPENDED` 会转换为结构化 error CALL response。

`suspend` 是控制流，不是“成功但没有正文”的普通工具结果。父 frame 必须停在一个可恢复位置，等待调度器建立子 frame、传递受控上下文并返回结构化响应；若直接把空结果写回模型，父 Agent 会在子任务尚未完成时继续生成，委派关系也无法被可靠观测和取消。

## 4. 权限

`AgentProfile` 定义两个白名单：

- `allowed_mtp_verbs`：`None` 表示全部允许，空列表表示全部禁止，其余为 verb 白名单；
- `allowed_sys_tools`：对 Kernel syscall 使用同样的三态语义。

权限在执行前检查。权限拒绝属于 `agent_fault`，Agent 可以调整方案，但不能通过换写法绕过 Profile。

MTP 是否应该出现，取决于当前行动门槛，而不是“能调用工具就调用”：信息存在缺口时先 SEARCH/READ；动作产生副作用或需要执行资产时才 RUN；内容确有跨会话长期价值时才 WRITE/UPDATE；委派能形成明确子任务时才 CALL。Agent 不得臆造 alias，也不得把错误响应理解为持续试探权限的邀请。错误应该驱动修正查询、参数或计划；权限拒绝则意味着停止该能力路径。

## 5. 响应

`MTPResponseStatus` 当前枚举：

| 状态 | 语义 |
|:---|:---|
| `success` | 指令完成；可以同时带 nonfatal warnings |
| `error` | 指令失败并携带结构化 `MTPErrorInfo` |
| `ack` | WRITE/UPDATE 已登记延迟意图 |
| `warning` | 兼容的警告终态；常规 nonfatal 警告优先放 `warnings` |
| `suspend` | CALL 要求 frame scheduler 接管 |
| `cancelled` | 执行在取消边界终止 |

Agent 可见格式为本地化 XML：

```xml
<mtp_response status="error" time="12ms">
<error code="mtp.argument.invalid" severity="agent_fault">
Localized message
</error>
</mtp_response>
```

Warning 放在 `<warnings><warning>...</warning></warnings>` 中。`pending_alias`、`call_request` 和内部 cause 不序列化到普通 Agent 响应正文，由运行时结构化消费。

CALL 取消使用 `<mtp_response status="cancelled">` 回填本地化的取消文案，不伪装为空 success，也不需要构造 error code。

当前 formatter 会把业务 `content`、reply 和 warning 文本直接嵌入 XML 容器，尚未对所有内容执行统一 XML escaping。若文本自身包含 `<`、`>` 或 `&`，Agent 可见结果可能不是严格可解析 XML；调用方当前应把它视为结构化文本信封，而不是承诺任意 payload 都能通过 XML parser。补齐 escaping 时必须同时验证代码片段与既有 prompt 行为。

错误结构详见[error-model.md](./error-model.md)。

## 6. 不变量

- parser、runtime 和 formatter 共用 `core/mtp/models.py` 的枚举与模型；
- MTP 错误必须以结构化 `MTPErrorInfo` 回填，不泄漏内部 exception cause；
- READ/SEARCH 输出通过 MemoryCompiler，不在 handler 内维护第二套记忆渲染；
- WRITE/UPDATE 的 ACK 不等同于持久化成功；
- CALL 的 suspend 只能由 Alice 编排层恢复；
- 记忆访问使用调用方 `Identity`，不能绕过可见性边界；
- cancellation 不能被转换成普通 success。

> **当前实现偏差**：L2 冷查询会携带调用方 `Identity`，但 AliceRuntime 进程级共享的 L0 PendingAtomRuntime 与 L1 KoakumaAtomCache 在命中时尚未重新校验身份。因此当前代码还没有完全满足上述可见性不变量；这是需要修复的隔离缺口，而不是放宽契约的理由。详见 [MTP Runtime](../alice/mtp-runtime.md)与 [PendingAtom](../alice/pending-atom.md)。

## 7. 设计矛盾检查

修改 MTP 时，应检查以下问题：

1. 新能力是帮助 Agent 发现、检查、使用、创建、修订或委派，还是把任意内部 API 暴露成了协议动词？
2. parser 是否只解析结构，权限、身份、类型和取消仍由 Runtime 强制执行？
3. alias 是否仍由运行时解析，还是模型或 handler 开始把可读名称直接当成正式记忆 id？
4. WRITE/UPDATE 的 `ack` 或 PendingAtom 是否被调用方、UI 或 Agent 文案描述为已经持久化成功？
5. 延迟意图是否只在完成的 run 中进入 finalize，失败和取消是否仍不会默认污染长期记忆？
6. CALL 的 `suspend` 是否仍交给 Alice 调度器恢复，还是被 formatter 当成普通 response 吞掉？
7. RUN 的权限检查是否被误写成强安全沙箱，redirect 或代码记忆来源是否失去可见提示？
8. error/warning 是否仍给 Agent 提供可执行的修正信息，同时不泄漏内部 cause？
9. formatter 是否安全处理 payload 中的 XML 特殊字符，还是新增内容扩大了当前 escaping 缺口？

## 8. 验证入口

- parser / filter：`tests/unit/core/mtp/`；
- verb 链路：`tests/unit/agent_runtime/mtp/test_*_chain.py`；
- alias / PendingAtom：`tests/unit/agent_runtime/mtp/test_alias_generation.py`、`test_read_chain.py`；
- CALL：`tests/unit/core/mtp/test_call_response_formatting.py`、Alice RunScheduler/CallCoordinator 测试；
- syscall：`tests/unit/agent_runtime/mtp/syscalls/`；
- i18n formatter：MTP formatter 和 i18n runtime 测试。
