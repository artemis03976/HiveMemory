---
title: Memory Tool Protocol
status: current
owner: alice
scope: mtp
code_paths:
  - src/hivememory/core/mtp/
  - src/hivememory/agent_runtime/mtp/runtime.py
  - src/hivememory/alice/orchestration/executor.py
  - src/hivememory/alice/orchestration/call/coordinator.py
  - src/hivememory/alice/orchestration/call/context.py
  - src/hivememory/core/models/agent.py
related_contracts:
  - docs/contracts/error-model.md
  - docs/contracts/routes-and-events.md
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-10-17
---

# Memory Tool Protocol (MTP)

MTP 是 Alice Agent 在生成循环中调用记忆、系统工具和子 Agent 的进程内文本协议。它延续了项目早期“Memory as a Tool”的核心判断：记忆不能只在回答开始前由系统预检索一次，还必须允许 Agent 在任务展开后主动发现、检查、使用和修订知识。

预检索适合提供一个低成本起点，却无法预知推理过程中出现的所有信息需求。一个 Agent 可能先读到摘要，随后才知道需要哪条原始证据；也可能在执行中形成值得长期保留的新事实，或需要把子任务委派给另一个 Agent。MTP 把这些动作放进生成循环，使记忆从静态上下文变成受权限和生命周期约束的可调用能力。

选择文本协议，是因为它可以直接出现在模型输出中，不绑定某一家模型供应商的 function calling 形态，也便于在消息历史中保留调用与响应。代价是文本可能不完整、参数可能含糊，且模型发出的指令不能天然视为可信调用。因此 parser 只负责把文本解析为结构化请求，KoakumaRuntime 仍必须执行 Profile 权限、类型、取消和错误约束，资源身份与操作授权由 workspace 处理；MTP 不是绕过 Runtime 的自由格式命令通道。

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

`KoakumaRuntime` 负责 parse、frame 执行策略的动词检查、verb 分发、结果计时和格式化。它属于 Alice；SEARCH、READ/RUN 记忆目标、WRITE/UPDATE 以及 CALL 目标 Profile 与 context_refs 全部提交公共操作请求，经绑定线程 context 的 `OperationSubmitter` 进入 workspace 能力层。Alice 不直接请求 Patchouli，不持有 local bus 代理或 Profile 读取缓存。

Agent loop 检测 MTP 文本后暂停自然语言生成，执行指令并把格式化结果回填到消息历史，再继续生成。CALL 的 `suspend` 由 Alice `RunExecutor`/`CallCoordinator` 消费，不直接回填为空结果；Koakuma 只产出结构化 `MTPCallRequest`，目标 Profile 与 `context_refs` 由 `CallContextProvider` 在 Alice 编排边界解析。Executor 递归等待被调用 frame，完成后由 `AgentRuntime.apply_call_response()` 一次性写回 caller history 和 `tool_result`。

每条指令的 `MTPExecutionContext` 从当前 frame 冻结得到，其中 `RuntimeScope` 只携带执行关联与只读 `ExecutionLabels`，不含 `IdentityScope`。任务进程进入 CPU 执行阶段时从进程 context 派生主线程的线程 context，CPU 驱动将它绑定为提交函数；CALL 派生的子 frame 得到以被调用方身份签发的子线程 context 与自己的提交函数，主、子 frame 沿用同一组注册标签。CPU 执行结束时全部线程 context 一起撤销，此后的提交被拒绝。请求不能自行指定目标或 process_id：actor API 的目标取线程的驻留 Workspace，意图的进程关联取自线程 context；READ、SEARCH、RUN 的记忆目标、WRITE、UPDATE 与 CALL 目标读取分别要求 actor 操作 `memory.read`、`memory.search`、`memory.run`、`memory.write`、`memory.update` 与 `agent.call`，各项互不隐含（RUN 不要求 READ，WRITE 不授予 UPDATE）；RUN 的 `sys_*` 系统工具在 Alice 本地执行，不提交请求；CPU 或 CALL 子帧失败均不撤回已接纳写入。资源归属与 actor policy 仍逐次校验。

## 3. 动词契约

六个动词共同覆盖 Agent 使用记忆时的基本闭环：SEARCH 用于发现，READ 用于检查，RUN 用于使用，WRITE 用于创建，UPDATE 用于修订，CALL 用于委派。它们不是六个任意工具名，而是刻意区分了“寻找什么”“确认内容”“执行能力”“提出长期变更”和“转交控制”等不同责任。

### 3.1 SEARCH

```text
⟪ SEARCH | * | query="..." filter="..." ⟫
```

- 必填 `query`；
- 可选 `filter`，当前支持 memory type、tags 和 vitality 等解析规则；
- filter 中的非法 token 不使整次搜索失败，而是忽略过滤并返回 warning；
- 提交 `RetrieveRequest` 经 workspace 语义检索能力读取，不指定 `top_k`，条数由检索引擎配置的 `retrieval.default_top_k`（默认 5）决定；
- 结果由 MemoryCompiler 编译为 Retrieval Context；
- 读取能力预热 workspace 完整原子缓存，后续交付仍重新检查当前 actor 的可见性；
- 空结果仍为 `success`，并带 `no_memories_found` warning。

SEARCH 返回可继续消费的检索上下文，而不是把“没有找到”当成系统故障。检索本身具有不确定性，空结果只说明当前查询没有证据；Agent 仍可以改写查询、继续回答或明确告知信息不足。

### 3.2 READ

```text
⟪ READ | alias | ⟫
⟪ READ | [alias_a, alias_b] | ⟫
```

- 不支持 wildcard；
- alias 经 workspace 读取视图解析为正式 atom、pending、redirect 或 discarded/failed 终态；模型与编译器保留 expired 兼容种类，但登记不再产生它；
- 每类结果均经 MemoryCompiler 的 `MTP_READ` target 编译；
- 全部 alias 未命中时返回 error；
- 部分未命中时返回已解析内容，并把缺失 alias 放入 warnings；
- 提交 `ResolveReferencesRequest`；workspace 对交付的正式 atom 与可读 redirect 目标自动记录 `workspace.reference_read` citation，同次请求按正式 UUID 去重，缓存命中同样计数。

alias 是运行期稳定称呼，不等于永久 UUID。它让模型使用可读、短小的引用，又允许 Runtime 把同一名称解析为正式 atom、尚未结算的 PendingAtom 或修订后的 redirect。redirect 保留旧称呼的可追踪性，但通过 warning 提醒 Agent 目标已经演化；terminal 状态则防止一个失败或过期意图继续伪装成有效记忆。

READ 列表是协议中显式支持的批量读取，而不是多命令特例。Agent 在 SEARCH 后往往需要检查数条候选证据；一次 READ 多个 alias 可以减少额外的生成/执行轮次，同时仍让 Runtime 对每个 alias 独立解析并把部分失败表达为 warning。

### 3.3 RUN

```text
⟪ RUN | sys_tool_alias | key="value" ⟫
⟪ RUN | code_memory_alias | key="value" ⟫
```

两层分发：

1. `sys_*` 工具走 syscall 注册表中注册的 syscall；未注册的 `sys_*` alias 直接返回 `mtp.alias.not_found`，不转为记忆解析；
2. 其他 alias 经 PendingAtom/正式记忆解析，只允许 `CODE_SNIPPET` MemoryAtom。

系统工具不设 Agent 权限，所有已注册的 `sys_*` 对所有 Agent 可用，在 Alice 本地执行、不提交操作请求。redirect 可以执行但会附带 warning；pending 和 terminal alias 不可执行。记忆目标通过 `ResolveRunTargetRequest` 读取，要求 actor 操作 `memory.run`（不要求 `memory.read`），引用计数在工具类型检查与执行前已完成；后续类型拒绝或工具失败仍保留这次 `workspace.reference_read` citation。系统 syscall 不读取正式原子，不因此计数。

当前用户代码通过本地执行器运行，尚无可作为安全边界的强隔离沙箱、进程级资源限制和真取消。不得把 RUN 描述为安全执行不受信任代码的能力。

RUN 被保留在记忆协议中，是因为一部分记忆不仅需要被阅读，还可能代表可执行的代码资产或注册工具。但“能被调用”不等于“已被安全隔离”：权限白名单只约束可见的调用面，不能替代操作系统级沙箱、资源限制和可信来源审查。

### 3.4 WRITE

```text
⟪ WRITE | * | content=`...` title="..." reason="..." ⟫
```

- `content` 必填；`title`、`reason` 可选；
- 提交 `SubmitWriteIntentRequest`，workspace 登记 MATERIALIZING 意图并立即经 Patchouli 意图路由派发，生成队列接纳后返回 `ack + pending_alias`；同 Workspace 的其他 Agent 可经读取许可回读；
- 不在 Koakuma 内同步创建正式 MemoryAtom；
- 派发话题来自任务进程，与 CPU 输入清单一致，生成材料取过去的 block 与摘要，不含当前轮；
- CPU 结果与 `InteractionPayload` 都不携带物化任务，finalize 只处理完成的交互与预检索命中。

ACK 表示意图已在 workspace 登记并被生成队列接纳，不表示长期记忆已经持久化。确定性拒绝返回 `mtp.memory.intent_rejected`（agent_fault），句柄为 FAILED；结果未知返回 `mtp.memory.intent_unknown`（system_fault），句柄保持 MATERIALIZING，若实际已接纳可继续收到结算。两类错误都不返回 ACK，详见[错误模型](./error-model.md)。

显式保存意图不再等待整轮完成：长时间运行的 Agent 可以及时提交阶段成果，随后 CPU 失败、取消或进程关闭都不撤回它。PendingAtom 支持在正式提取、去重与持久化之前引用刚提交的内容，正式 MemoryAtom 仍由 Patchouli 生成链创建。结算句柄保留到重启；队列接纳是进程内承诺，不是跨重启耐久承诺。

### 3.5 UPDATE

```text
⟪ UPDATE | alias | instruction="..." content=`...` ⟫
```

- TARGET 必须是单 alias；`instruction` 必填，`content` 可选；
- 目标必须解析为请求方可读的正式 atom；pending 与结算 redirect 句柄不能再次 UPDATE；
- 提交 `SubmitUpdateIntentRequest`，注册以原记忆 UUID 为基线的 pending revision；基础校验不属于引用交付、不记录 citation；pending revision 只对能读取基础原子的 Agent 可回读，读不到基础时在任何状态下都与不存在相同，以它为目标再次 UPDATE 也按不存在拒绝；
- 使 workspace 共享完整原子缓存中的基础原子及 alias 索引失效，防止后续脏读；
- 基础冷读之后同步复查线程、读取进程话题并登记，立即提交到 Patchouli；接纳后返回 `ack + pending_alias`，正式更新由后台生成链处理，失败和取消规则与 WRITE 一致。

UPDATE 同样不原地覆盖旧记忆。它以正式 atom 为基线创建 pending revision，使当前 run 能表达修订意图，又保留旧版本和来源链；workspace 共享完整原子缓存中的基础原子及 alias 索引立即失效，是为了避免 Agent 在同一轮继续把待修订内容当成无变化的权威事实。

### 3.6 CALL

```text
⟪ CALL | agent_alias | task="..." context_refs=["alias_a"] ⟫
```

- TARGET 和 `task` 必填；`context_refs` 可选；
- Koakuma 返回 `suspend` 和结构化 `MTPCallRequest`；
- Alice RunExecutor 通过协程调用栈挂起 caller frame，经提交函数的 `derive_call_thread` 提交 `DeriveCallThreadRequest` 读取目标并派生子线程、以调用方的 `ResolveReferencesRequest` 解析共享上下文，递归运行 callee frame，再以 `MTPCallResponse` 回填；
- 派生要求调用方的 `agent.call`，内置 Profile 也不跳过；被调用方须在本 Workspace 有准入，未准入或为保留的 `system` 时回填 `mtp.permission.denied`（文案键 `mtp.call.agent_unavailable`），不启动子 frame；context_refs 以 READ 请求解析，要求 `memory.read`，缺少时被跳过而不阻止 CALL；正式原子交付自动记录 `workspace.reference_read` citation，先前 READ 过的原子在这次独立请求中再次计数；
- callee frame 以被调用方身份运行：资源操作按被调用方的白名单与可见性授权，写入意图记在被调用方名下；
- CALL 只允许 root frame 发起；callee frame 的 `FrameExecutionPolicy` 由被调用方的操作快照翻译并显式移除 CALL，防止递归爆炸；被调用方 Profile 只提供 persona、模型与语言；
- 只有 `COMPLETED` 子帧产生 success CALL response，并可以返回其 PendingAtom alias；未成功结束的子帧不返回 alias，但其已提交写入继续生成；
- `CANCELLED` 保持 cancelled 终态，`FAILED`、`BUDGET_EXHAUSTED` 会转换为结构化 error CALL response；`SUSPENDED` 是 RunExecutor 继续消费的控制流 trap，不构造 CALL response。

`suspend` 是控制流，不是“成功但没有正文”的普通工具结果。父 frame 必须停在一个可恢复位置，等待调度器建立子 frame、传递受控上下文并返回结构化响应；若直接把空结果写回模型，父 Agent 会在子任务尚未完成时继续生成，委派关系也无法被可靠观测和取消。

## 4. 权限

Agent 的权限只由 Workspace 访问登记的 actor 操作白名单表达，Agent Profile 不承载权限。任务进程在 CPU 分配时把本进程 actor 的操作集合写入输入清单（`allowed_operations`），Alice 按下表翻译为 frame 的动词集合：

| actor 操作 | MTP 动词 |
|:---|:---|
| `memory.read` | READ |
| `memory.search` | SEARCH |
| `memory.write` | WRITE |
| `memory.update` | UPDATE |
| `agent.call` | CALL |
| `memory.run` | RUN 的记忆目标 |
| —（系统工具不设权限） | RUN 的 `sys_*` |

RUN 始终在动词集合中，因为系统工具不设权限；没有 `memory.run` 时，RUN 的记忆目标由 actor API 拒绝。CALL 的子 frame 使用被调用方的操作集合（派生时由进程读取），按同一张表翻译后去掉 CALL。同一集合驱动两层：

- **提示层**：只教学集合内的动词，同时裁剪提及集合外动词的行为准则、错误恢复与演示；系统工具菜单始终完整；
- **执行层**：Koakuma 在分发前以 frame 的 `FrameExecutionPolicy` 检查动词，集合外的动词在提交任何操作请求之前被拒绝。

资源操作仍由 actor API 按访问登记逐次授权；动词集合是 CPU 分配时的快照，只描述可用能力，不是授权依据。权限拒绝属于 `agent_fault`，稳定代码为 `mtp.permission.denied`（文案键 `mtp.permission.verb_denied`；CALL 目标未准入时为 `mtp.call.agent_unavailable`），Agent 可以调整方案，但不能通过换写法绕过许可。

MTP 是否应该出现，取决于当前行动门槛，而不是“能调用工具就调用”：信息存在缺口时先 SEARCH/READ；动作产生副作用或需要执行资产时才 RUN；内容确有跨会话长期价值时才 WRITE/UPDATE；委派能形成明确子任务时才 CALL。Agent 不得臆造 alias，也不得把错误响应理解为持续试探权限的邀请。错误应该驱动修正查询、参数或计划；权限拒绝则意味着停止该能力路径。

## 5. 响应

`MTPResponseStatus` 当前枚举：

| 状态 | 语义 |
|:---|:---|
| `success` | 指令完成；可以同时带 nonfatal warnings |
| `error` | 指令失败并携带结构化 `MTPErrorInfo` |
| `ack` | WRITE/UPDATE 已登记延迟意图 |
| `warning` | 兼容的警告终态；常规 nonfatal 警告优先放 `warnings` |
| `suspend` | CALL 要求 Alice RunExecutor 接管 |
| `cancelled` | 执行在取消边界终止 |

Agent 可见回填由本地化系统标题和随后的 XML 块组成；标题不属于 XML 文档，严格解析边界从 `<mtp_response>` 开始：

```xml
<mtp_response status="error" time="12ms">
<error code="mtp.argument.invalid" severity="agent_fault">
Localized message
</error>
</mtp_response>
```

Warning 放在 `<warnings><warning>...</warning></warnings>` 中。`pending_alias`、`call_request` 和内部 cause 不序列化到普通 Agent 响应正文，由运行时结构化消费。

内部 callee 自然产生 `CANCELLED` 时，CALL 可以使用 `<mtp_response status="cancelled">` exactly-once 回填本地化结果；这不是 Chat-level stop 协议。全局 run 被 task cancellation 取消时，CallCoordinator 只清理活跃 record 和 callee frame，不伪造 caller response，CancelledError 沿递归调用栈传播。

Formatter 把 handler、MemoryCompiler、i18n 和 CALL 提供的动态值都视为原始 Unicode 文本，并独占 XML 结构构造权。正文包含 `<`、`>` 或 `&` 时使用 CDATA，`]]>` 会拆成相邻 CDATA 段；因此已编码实体和嵌套 XML 样式 payload 会按字面文本保留，不会成为协议子节点。XML 属性单独执行实体转义。正文换行统一为 LF，XML 1.0 禁止的控制字符替换为 `U+FFFD`。

空 `content` 不产生正文，无 warnings 时不产生 `<warnings>`；成功 CALL 的空 reply 仍保留 reply label 和空正文。上述规则覆盖成功 content、MemoryCompiler 编译出的可选摘要、CALL reply、artifact alias、warning 和 error reason，不要求各 verb handler 自行转义。

错误结构详见[error-model.md](./error-model.md)。

## 6. 不变量

- parser、runtime 和 formatter 共用 `core/mtp/models.py` 的枚举与模型；
- MTP 错误必须以结构化 `MTPErrorInfo` 回填，不泄漏内部 exception cause；
- READ/SEARCH 输出通过 MemoryCompiler，不在 handler 内维护第二套记忆渲染；
- WRITE/UPDATE 的 ACK 不等同于持久化成功；
- CALL 的 suspend 只能由 Alice 编排层恢复；
- 记忆访问经线程 context 绑定的提交函数进入 workspace，授权点组装当次 `IdentityScope`，资源 owner 先执行 Workspace ownership hard boundary，再执行 actor 可见性策略；执行标签不能代替任一授权边界；
- cancellation 不能被转换成普通 success。

> **实现说明**：workspace 统一按 L0 写入意图登记、L1 完整原子缓存、L2 canonical 冷读解析。L0 只比较 Workspace 归属，不比较提交 Agent 或进程；UPDATE 意图另按基础原子的 actor policy 判断，读不到基础时与不存在相同；L1/L2 对正式原子逐次执行 ownership 与 actor policy。redirect 目标不可读时清空 canonical 引用和结算视图字段，并省略可能携带基础身份的 pending 记录。Patchouli 的 canonical 变更事件内联失效原子、Profile 与 Workspace 代次；UPDATE 登记成功后另失效基础原子。意图与原子结果均为独立副本。详见 [Workspace 架构](../architecture/workspace.md#54-写入意图线程-context-与操作入口)、[MTP Runtime](../alice/mtp-runtime.md)。

## 7. 设计矛盾检查

修改 MTP 时，应检查以下问题：

1. 新能力是帮助 Agent 发现、检查、使用、创建、修订或委派，还是把任意内部 API 暴露成了协议动词？
2. parser 是否只解析结构，Runtime 是否保留 Profile 权限、类型与取消检查，workspace 是否保留逐次操作授权和资源身份边界？
3. alias 是否仍由运行时解析，还是模型或 handler 开始把可读名称直接当成正式记忆 id？
4. WRITE/UPDATE 的 `ack` 或 PendingAtom 是否被调用方、UI 或 Agent 文案描述为已经持久化成功？
5. 延迟意图是否只在完成的 run 中进入 finalize，失败和取消是否仍不会默认污染长期记忆？
6. CALL 的 `suspend` 是否仍交给 Alice 调度器恢复，还是被 formatter 当成普通 response 吞掉？
7. RUN 的权限检查是否被误写成强安全沙箱，redirect 或代码记忆来源是否失去可见提示？
8. error/warning 是否仍给 Agent 提供可执行的修正信息，同时不泄漏内部 cause？
9. formatter 是否安全处理 payload 中的 XML 特殊字符，还是新增内容扩大了当前 escaping 缺口？

## 8. 验证入口

- parser / filter：`tests/unit/core/mtp/`；
- verb 链路：`tests/integration/mtp/`；操作入口、引用计数与撤销边界：`tests/integration/workspace/test_operation_requests.py`、`test_reference_citations.py`；
- HTTP 单轮 SEARCH/READ/WRITE/CALL 与真实记忆结算：`tests/e2e/system/test_chat_mtp_operations.py`；
- alias / PendingAtom：`tests/unit/workspace/intents/test_registry.py`、`tests/unit/workspace/resolution/test_alias_resolver.py` 与 `tests/integration/workspace/test_intent_registry_and_read_cache.py`；
- CALL：`tests/unit/core/mtp/test_call_response_formatting.py`、Alice RunExecutor/CallContextProvider/CallCoordinator 测试；
- syscall：`tests/unit/agent_runtime/tools/`；
- i18n formatter：MTP formatter 和 i18n runtime 测试。
