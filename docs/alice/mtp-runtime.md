---
title: Alice MTP Runtime
status: current
owner: alice
scope: mtp-parser-execution-permissions-and-syscalls
code_paths:
  - src/hivememory/agent_runtime/mtp/
  - src/hivememory/agent_runtime/resolver.py
  - src/hivememory/core/mtp/
  - src/hivememory/prompts/mtp.py
  - src/hivememory/system/config/alice.py
related_contracts:
  - docs/contracts/mtp.md
  - docs/contracts/error-model.md
  - docs/contracts/routes-and-events.md
  - docs/alice/pending-atom.md
  - docs/alice/orchestration.md
last_reviewed: 2026-08-03
---

# MTP Runtime：从文本指令到受控执行

Memory Tool Protocol 让 Agent 能在生成过程中主动发现、读取、执行、写入和修订记忆，也能把一项工作委派给子 Agent。协议选择文本形态，是为了不依赖某一家模型供应商的 function calling 接口，并让调用与回填自然保留在消息历史中；但文本可被模型随意生成，也可能残缺、越权或携带不可信参数，因此“看起来像 MTP”从来不等于“可以直接执行”。

KoakumaRuntime 就是这道执行边界。它把 WorkerAgent 截止在 MTP 右定界符附近的文本解析为结构化命令，建立本次 frame 的身份、权限与运行坐标，执行对应 handler，再把成功、warning、错误或控制流信号格式化为 Agent 可继续理解的本地化文本。它属于 Alice 消费的 Agent Runtime 执行层，不拥有多 Agent 拓扑，也不拥有长期记忆。

本文只解释当前 runtime 怎样兑现协议。`⟪ VERB | TARGET | ARGS ⟫` 的语法、六个动词的规范语义和响应状态只在 [MTP 契约](../contracts/mtp.md)维护，避免协议规范与实现说明形成两套真相源。

## 1. 执行链

一次普通 MTP 迭代沿以下路径推进：

```text
WorkerAgentService.generate(stop = MTP right delimiter)
  -> AgentLoopExecutor detects the last MTP opening delimiter
  -> KoakumaMTPExecutor.intercept_and_execute()
  -> KoakumaRuntime
       parse text -> MTPCommand
       resolve MTPExecutionContext(run_id, frame_id, action_id)
       check cancellation
       check verb permission
       dispatch handler
       check tool / identity / type constraints
       produce MTPResponse
       format localized XML
  -> tool_call + tool_result TurnEvents
  -> formatted response enters working history
  -> WorkerAgent continues generation
```

如果文本中没有 MTP 左定界符，Koakuma 返回 `None`，Agent loop 把本次生成当作自然语言收敛。解析错误不会抛回模型适配层，而会形成结构化 `MTPErrorInfo` 并格式化回填，让 Agent 有机会修正语法或参数。

CALL 是唯一不在 handler 内完成业务动作的动词。Koakuma 只产生 `suspend + MTPCallRequest`；AgentLoopExecutor 把 frame 状态交还 Alice 的 `RunScheduler`，由 `CallCoordinator.begin_call()` 解析目标并创建普通 callee frame，Scheduler 用同一个活动 frame 循环运行 callee，再由 `complete_call()` 通过 `AgentRuntime.apply_call_response()` 恢复 caller。把 CALL 留成 trap，保证单 Agent 执行器不偷偷取得多 Agent 调度权。

## 2. MTPExecutionContext

Runtime 不从命令文本相信身份或权限。每次执行都由 frame 构造 `MTPExecutionContext`，主要包含：

- `identity`：访问 Patchouli 公开记忆能力时使用的调用方身份；
- `agent_profile`：MTP verb 与系统工具白名单；
- `runtime_scope`：run、frame 与 action 坐标；不包含 parent/depth 拓扑信息；
- `language`：错误、warning 与普通响应的本地化选择。

WRITE/UPDATE 注册 PendingAtom 时会复制 identity 与 runtime scope；SEARCH、READ、UPDATE 和用户态 RUN 在 L2 冷查询时把 identity 传给 Patchouli；CALL 是否允许由 `FrameExecutionPolicy` 的 permitted verbs 决定。协议参数只描述“想做什么”，ExecutionContext 才回答“谁在做、在哪一帧做、允许做到哪里”。

## 3. 双层权限

Agent Profile 的权限同时作用于 prompt 和 runtime：

1. **提示层裁剪**：PromptBuilder 根据 `allowed_mtp_verbs` 与 `allowed_sys_tools` 缩减动词教学和工具菜单，降低模型生成不可用命令的概率；
2. **执行层强制**：Koakuma 在分发前调用 `is_verb_allowed()`，RUN 内核工具前再调用 `is_tool_allowed()`。拒绝以 `agent_fault` 结构化错误回填，模型不能通过改写 prompt 或手工生成 MTP 绕过白名单；
3. **控制流硬限制**：被调用 frame 的 policy 在 Profile 权限基础上显式移除 CALL；Koakuma 同时校验 Profile 与 frame policy，当前仍只支持根 run 的串行一层 CALL。

两个白名单都有三态语义：`None` 表示全部允许，空列表表示全部禁止，非空列表表示只允许列出的能力。提示层是可用性引导，runtime 检查才是授权判定；二者必须保持一致，但不能因为已有 runtime 检查就放任 prompt 持续教授被禁用能力。

## 4. 六个动词在 Runtime 中的兑现

| Verb | 当前 Runtime 行为 | 主要交接边界 |
|:---|:---|:---|
| `SEARCH` | 解析 query/filter，经 Alice local bus 请求 Patchouli retrieval；用 MemoryCompiler 编译结果并预热 L1 alias cache | Patchouli 拥有检索，Alice 拥有本帧缓存与回填 |
| `READ` | 通过 L0 PendingAtom、L1 atom cache、L2 Patchouli 冷查询解析一个或多个 alias；编译 pending、redirect、atom 与终态 | alias 语义由 RuntimeAliasResolver 统一 |
| `RUN` | 先匹配 Kernel syscall；否则解析 MemoryAtom，仅允许执行 `CODE_SNIPPET` | Profile 控制工具可见面，但不等于 OS 沙箱 |
| `WRITE` | 校验 content，注册 `WriteFocus` PendingAtom，返回 `ack + draft_*` | 正式物化延迟到 Patchouli finalize |
| `UPDATE` | 解析单个正式 atom，注册 `UpdateFocus` revision，使原 alias 的 L1 缓存失效，返回 `ack + rev_*` | 不在 Koakuma 内原地覆盖记忆 |
| `CALL` | 校验 target/task/policy，返回 `suspend + MTPCallRequest` | Alice `CallCoordinator` 负责 callee 与结果回流 |

SEARCH 空结果是带 `no_memories_found` warning 的 success，而不是基础设施错误；READ 列表中部分 alias 丢失时返回已有内容并附 warning，全部丢失才返回 error。这样的区别让“不确定检索没有证据”和“系统无法完成请求”保持不同语义。

READ/SEARCH 输出都经过 MemoryCompiler。MTP handler 不维护另一套记忆模板，以免 prompt 预检索、主动 READ 和 Generation 看到互相矛盾的字段与预算规则。redirect、citation 与 PendingAtom 的完整运行时语义见 [PendingAtom](./pending-atom.md)。

## 5. RUN 的两层执行面

### 5.1 Kernel syscalls

当前内核注册表提供五个固定工具：

- `sys_clock`：读取当前时间；
- `sys_python_repl`：在受限子进程执行短 Python 代码；
- `sys_web_search`：使用可选搜索依赖查询网络；
- `sys_read_file`：读取配置 workspace 内的文件；
- `sys_write_file`：写入配置 workspace 内的文件。

文件工具先解析 workspace 与目标的绝对路径，并拒绝逃逸 workspace 的路径；读写还受最大字节数限制。REPL 和代码记忆共用受限执行器：在独立 Python 子进程中使用 builtin allowlist，移除普通 import/open/exec/eval 等入口，并以 timeout 终止忙循环。

这些措施适合降低误操作和普通生成代码的风险，但不是不受信任代码的强安全边界。子进程没有容器、OS 权限降级、网络隔离、文件系统虚拟化、CPU/内存配额或来源签名。builtin 限制也不应被描述为经过对抗验证的沙箱。

### 5.2 `CODE_SNIPPET` 用户态工具

如果 target 没有命中 Kernel Registry，Koakuma 使用三级 alias resolver 查找记忆。只有 `MemoryType.CODE_SNIPPET` 能进入执行器；普通事实、pending alias、失败/过期句柄都不能作为代码运行。settled redirect 可以继续执行 canonical code atom，但会携带 alias 已重定向的 warning；成功执行后记录 `mtp.run` citation。

代码正文由 MemoryAtom 提供，命令 args 作为 `params` 注入受限 namespace。它仍使用与 Python REPL 相同的本地子进程限制，不能因为代码来自“记忆”就推断其来源可信。

## 6. 错误、warning、citation 与 i18n

Koakuma 把可预期失败统一转换为 `MTPResponse`：

- 参数、alias、类型或权限问题通常是 `agent_fault`，Agent 可修改命令后继续；
- route、存储或未知异常是 `system_fault`，对外不泄露内部 exception cause；
- nonfatal filter、partial miss、redirect 和空检索通过 warnings 保留主结果；
- formatter 根据 ExecutionContext language 输出本地化 XML；
- `pending_alias` 与 `call_request` 留在结构化 `MTPExecutionResult` 中供运行时消费，不序列化成普通响应内部字段。

READ 正式 atom/redirect 与 RUN code atom 会请求 Patchouli 记录 citation。citation 是 best-effort 的旁路行为，不应因为观测或强化记录失败而改写已经得到的业务响应。

完整错误分类和稳定 error code 见[错误模型](../contracts/error-model.md)。修改 handler 时应新增或复用结构化异常，而不是把 Python 异常文本直接交给 Agent。

## 7. 取消边界

Koakuma 在拦截命令前和 handler 分发前检查本次调用显式传入的 `cancel_event`；Agent loop 也在生成前后与 MTP 执行前后设置 checkpoint。命中取消时返回 `cancelled`，frame 随后由 Agent Runtime 结束，不应转换成普通 success。MTPExecutor 和 Koakuma 不再保存跨 run 的共享 cancel 字段。

这是一种协作式取消。同步 syscall 一旦开始执行，事件循环必须等待函数返回；REPL 自身的 subprocess timeout 能终止超时子进程，但普通 cancel_event 不能在运行中立即打断它。文件 I/O 和 web search 同样没有中途取消 checkpoint。

## 8. 配置与实际接线

Alice 配置当前分为两组：

| 配置 | 默认值 | 当前作用 |
|:---|:---:|:---|
| `runtime.max_loop_iterations` | `10` | 单 frame generate -> MTP 最大循环次数 |
| `koakuma.enabled` | `true` | 当前只影响是否向 prompt 注入 MTP 教学 |
| `koakuma.mtp_prompt.enabled` | `true` | 控制 MTP prompt section |
| `koakuma.mtp_prompt.include_demo` | `true` | 控制 one-shot 示例 |
| `koakuma.mtp_prompt.include_error_handling` | `true` | 控制错误恢复提示 |
| `koakuma.python_repl_timeout_seconds` | `10` | REPL 与 code snippet 子进程 timeout |
| `koakuma.workspace_path` | `./workspace` | 文件 syscall 可访问根目录 |
| `koakuma.file_read_max_bytes` | `102400` | 单次读取上限 |
| `koakuma.file_write_max_bytes` | `102400` | 单次写入上限 |
| `koakuma.web_search_timeout_seconds` | `15` | 传入 web search 注册函数，但当前实现忽略 |
| `koakuma.execution_timeout_seconds` | `30` | 当前未接入 MTP 总执行 timeout |
| `koakuma.tool_cache_size` | `64` | 当前未用于 Kernel Registry 或 atom cache 容量 |

`koakuma.enabled=false` 目前不会阻止 Runtime 解析和执行模型自行生成的 MTP 文本。若它被当作安全关闭开关，会形成错误预期；真正禁用执行面仍需 Profile 白名单或后续补齐 runtime gate。

## 9. 维护检查

改变 MTP Runtime 时，应同时回答：

1. 变化属于协议语义还是 Alice 的实现策略？前者更新 Contracts，后者更新本文；
2. 新能力是否同时经过 prompt 裁剪和 runtime 权限检查；
3. 记忆访问是否使用当前 frame 的 Identity，而不是全局缓存中的无主对象；
4. WRITE/UPDATE 是否仍然只登记意图，CALL 是否仍然只产生 `SUSPENDED` trap；
5. handler 是否保持结构化 error/warning，不泄露内部 exception；
6. 同步工作、timeout 与取消是否给出了与真实接线一致的保证；
7. RUN 的权限白名单是否被误写成安全沙箱；
8. 新配置是否真正进入执行路径，而不是只有 schema 字段或 prompt 文案。

## 10. 代码与验证入口

| 责任 | 当前入口 |
|:---|:---|
| parser、formatter 与协议模型 | `src/hivememory/core/mtp/` |
| Koakuma 分发与六个 handler | `src/hivememory/agent_runtime/mtp/runtime.py` |
| Agent Runtime 的窄 MTP port | `src/hivememory/agent_runtime/mtp/mtp_executor.py` |
| alias 解析与热缓存 | `src/hivememory/agent_runtime/resolver.py`、`cache.py` |
| syscall 注册与实现 | `src/hivememory/agent_runtime/mtp/syscalls/` |
| MTP prompt | `src/hivememory/prompts/mtp.py`、`i18n/prompts.py` |
| Alice 配置 | `src/hivememory/system/config/alice.py` |
| parser / formatter 测试 | `tests/unit/core/mtp/` |
| verb 与 syscall 链路测试 | `tests/unit/agent_runtime/mtp/` |

## 11. 当前限制与实现偏差

- `koakuma.enabled=false` 只移除 prompt 教学，不关闭 runtime 执行；它不是 kill switch；
- `execution_timeout_seconds` 没有包住 `execute_mtp()`，`tool_cache_size` 也未接入任何 cache；配置存在不代表能力已实现；
- `web_search_timeout_seconds` 虽被传进 syscall registry，但 `sys_web_search()` 明确忽略该参数；底层调用可能超过配置时间；
- Kernel syscalls 是同步函数，并在 async handler 中直接执行。文件、搜索和 subprocess 等工作会阻塞当前事件循环；
- cancel token 按 `intercept_and_execute(..., cancel_event=...)` 逐次传入，不再保存在共享 KoakumaRuntime 字段中；仍需通过并发回归测试持续验证 run-local 隔离；
- 同步 syscall 执行期间不会轮询 cancel_event，因此协作式取消不能立即中断文件或网络调用；
- PromptBuilder 会按白名单过滤主要动词说明和工具菜单，但 dense one-shot demo 没有完整按 denied verbs 裁剪。例如禁止 RUN 时，示例中仍可能出现 RUN；
- prompt 的默认工具菜单来自静态 `DEFAULT_RUNTIME_TOOLS`，不是从实际 Kernel Registry 动态生成。注册表与提示词可能漂移；
- RuntimeAliasResolver 的 L0 PendingAtom 和 L1 atom cache 命中不重新校验 Identity。当前代码尚未完全兑现 MTP 契约中“记忆访问使用调用方 Identity”的不变量；
- RUN 的受限子进程不是面向敌对输入的安全沙箱，也没有来源签名、资源配额与 OS 级隔离；
- Agent loop 达到 `max_loop_iterations` 后返回 `BUDGET_EXHAUSTED`，根 run 对外映射为 `AgentRunStatus.FAILED`，CALL callee 映射为稳定的 budget error；
- Koakuma、atom cache 与 PendingAtomRuntime 的共享服务仍属于 Alice 组合根，但 frame registry、CALL ledger、cancel event 与 stream sequence 已按 run 隔离。

当前 MTP Runtime 已经形成“文本协议、结构化解析、双层权限、受控 handler 与可恢复错误”的完整闭环，但它仍是面向单进程可信部署的实验性执行层。文档和上层产品都不应把它包装成强隔离插件平台、持久化工作流引擎或任意代码安全沙箱。
