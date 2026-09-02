---
title: Error Model
status: current
owner: system
scope: cross-boundary-errors-and-degradation
code_paths:
  - src/hivememory/core/mtp/exceptions.py
  - src/hivememory/core/mtp/models.py
  - src/hivememory/core/mtp/formatter.py
  - src/hivememory/core/protocol/gateway.py
  - src/hivememory/system/services/passive/exceptions.py
  - src/hivememory/system/runtime/bus/async_bus.py
related_contracts:
  - docs/contracts/mtp.md
  - docs/contracts/routes-and-events.md
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-09-01
---

# 跨边界错误模型

HiveMemory 当前没有一个覆盖所有 HTTP、子系统和运行时的统一 error envelope。现有契约按失败是否属于预期业务终态，分别使用结果模型、结构化协议错误、控制异常和观测事件。本文明确这些边界，避免调用方把不同机制混用。

错误不只是写给开发者看的日志。对 Agent 而言，它还是下一步行动的控制输入：语法错误可以修正，alias 未命中可以换一种检索方式，权限拒绝意味着不能继续尝试绕过，而存储离线则提示当前能力暂不可用。一个只有 traceback 或模糊“执行失败”的系统，即使记录了足够多的内部细节，也无法帮助 Agent 安全地调整计划。

与此同时，并非所有失败都应该进入同一种响应信封。业务终态需要稳定返回，取消和契约违约需要中断控制流，warning 要保留仍可消费的部分成果，RuntimeEvent 则只服务于旁路观测。强行统一这些机制，会让调用方分不清“任务已被拒绝”和“程序装配错误”，也可能让观测设施反向影响业务正确性。

## 1. 表达方式

| 机制 | 使用场景 | 调用方行为 |
|:---|:---|:---|
| 业务结果状态 | 命令、Agent run、Passive outcome 等预期终态 | 检查判别字段或 status，不依赖异常 |
| `MTPErrorInfo` | MTP 对 Agent 的可恢复/不可恢复错误 | 根据 severity 调整或停止 MTP 重试 |
| 控制异常 | cancel、timeout、契约违约、route 缺失 | 终止或由顶层应用服务翻译 |
| warning | 部分命中、无结果、容错解析等非致命降级 | 保留成功结果，同时呈现提示 |
| RuntimeEvent | 运行过程、失败原因和降级观测 | 仅观测，不改变业务结果 |

选择表达方式时，先问调用方接下来需要做什么。若结果属于正常状态空间，就用判别字段；若 Agent 能根据稳定类别改变行为，就用结构化协议错误；若继续执行会破坏不变量，就抛控制异常；若主要成果仍然有效，只需附带 warning；若信息只用于追踪，则发布 RuntimeEvent。这里不存在一种适合所有边界的“最统一”格式。

## 2. 业务终态

### 2.1 Gateway command

`CommandExecutionResult.status`：

- `completed`；
- `rejected`；
- `failed`；
- `requires_confirmation`；
- `not_implemented`。

这些是系统指令的可预期终态，仍通过 `GatewayCommandOutcome` 正常返回。`error_code` 可提供稳定机器标识，`message` 面向当前客户端展示。

`rejected` 或 `requires_confirmation` 并不表示 Gateway 自身发生异常。把它们建模为正常结果，可以让 transport 稳定展示命令终态，也避免顶层异常处理器误将一次业务拒绝记录成系统故障。

### 2.2 Agent run

`AgentRunResult.status` 为 `completed`、`cancelled` 或 `failed`。System 只对 `completed` 调用 Patchouli finalize；取消/失败触发本轮控制收尾和 prepared cleanup。

### 2.3 Passive Ingress

Passive Ingress 用 outcome 表达 accepted、buffered、duplicate、ignored 或 degraded 等对外结果。内部 Gateway fallback、提交重试次数和 RuntimeEvent 不直接泄漏到响应模型。

## 3. MTP 结构化错误

### 3.1 `MTPErrorInfo`

```text
code         稳定 dotted-path 机器错误码
message_key  i18n 文本 key
severity     agent_fault | system_fault
params       i18n 模板参数
cause        内部原因，只供调试，序列化时排除
```

`code` 标识错误类别，`message_key` 标识具体用户/Agent 文案。调用方不能解析本地化 message 来判断错误类型。

### 3.2 严重度

- `agent_fault`：语法、参数、alias、类型或权限等调用侧可修正问题，允许 Agent 修正后重试；
- `system_fault`：存储、总线路由、工具内部依赖或意外错误，同参数重试通常无意义，Agent 应继续完成不依赖该能力的回答。

严重度表达归因和重试指导，不代表日志等级或数据安全等级。

区分两者的目的，是给生成循环一个保守的重试边界。`agent_fault` 意味着改变请求有机会成功；`system_fault` 意味着继续用相同参数试探大多只会重复失败或放大负载。它不宣称系统故障永远不可恢复，也不授权 Agent 对权限错误持续换写法绕过。

### 3.3 稳定错误码

| Code | 类别 |
|:---|:---|
| `mtp.parse.syntax_error` | MTP 语法错误 |
| `mtp.alias.not_found` | alias 无法解析 |
| `mtp.memory.not_found` | 目标记忆不存在 |
| `mtp.memory.type_mismatch` | 记忆类型不适用于操作 |
| `mtp.argument.invalid` | 参数缺失或格式错误 |
| `mtp.permission.denied` | verb、tool 或 frame policy 禁止 CALL |
| `mtp.system.storage_offline` | 存储离线 |
| `mtp.system.storage_error` | 存储读取错误 |
| `mtp.system.service_unavailable` | 所需总线路由/服务不可用 |
| `mtp.system.tool_error` | 工具内部错误 |
| `mtp.syscall.invalid_argument` | syscall 参数错误 |
| `mtp.syscall.permission_denied` | syscall 权限拒绝 |
| `mtp.syscall.execution_error` | syscall 执行失败 |
| `mtp.syscall.timeout` | syscall 超时 |
| `mtp.syscall.unavailable` | syscall 依赖不可用 |
| `mtp.call_response.sub_agent_error` | CALL 子 Agent 失败 |
| `mtp.call_response.budget_exhausted` | CALL 子 Agent 耗尽执行迭代预算 |
| `mtp.call_response.unexpected_suspend` | CALL 子 Agent 意外再次返回挂起终态 |

新增具体场景通常应复用稳定 code 并使用更具体的 `message_key + params`；只有机器处理类别确实变化时才新增 code。

### 3.4 Warning

`MTPWarningInfo` 只包含 `message_key + params`。Warning 不改变 response status，适用于：

- SEARCH filter 的部分 token 无效；
- SEARCH 无命中；
- READ 列表部分 alias 未命中；
- RUN 使用 redirect alias；
- 其他结果仍可安全消费的局部降级。

全部目标都无法完成时必须返回 error，不能只给 warning 和空成功。

warning 的核心语义是“主要成果仍然成立”。例如 READ 多个 alias 时，部分缺失不应抹去已经找到的证据；SEARCH 的非关键过滤条件无法解析时，也可以返回未过滤但明确标注的结果。相反，如果调用的目标完全无法完成，空 payload 加 warning 会制造一种虚假的成功，Agent 也无法判断是否应改用其他方案。

### 3.5 构造与格式化边界

`MTPResponse.content` 只承载成功业务内容；error 和 warning 分别使用结构化字段，不能先拼进 content 再要求调用方解析。KoakumaRuntime 的 `_route_and_execute()` 是普通 verb handler 异常转换为 `MTPErrorInfo` 的集中边界，取消和 CALL suspend 等控制流继续按专用语义处理。

`MTPFormatter` 是普通 Agent-facing MTP 回填文本的唯一构造点：它根据 language 渲染 code/severity/message/warnings，并排除内部 cause。`response_content` 只是业务 payload，不等于完整回填；需要写回模型历史时应使用运行结果的 `formatted_response`。CALL 不经过旧的通用 IPC 文本拼接，而由 `RunExecutor`/`CallCoordinator` 消费 `MTPCallRequest`；`CallContextProvider` 在编排边界解析 Profile 和共享上下文。Executor 递归等待被调用 frame，CallCoordinator 先 finalize callee，再由 `call_response.py` 把最终 `FrameExecutionResult` 单向映射为一个 `MTPCallResponse`，最后交给 `AgentRuntime.apply_call_response()` 恢复 caller frame。response 提交早于可等待的结束事件发送；内部已结算的 `CANCELLED` 结果可以映射为 CALL response，但外层 task cancellation 不回填伪造 response，也不终止 caller 重入流程。`SUSPENDED` 是非终态控制流，不参与错误映射。

Formatter 把 content、CALL reply、artifact alias、本地化 error reason 和 warning 统一作为原始文本编码：含 XML 保留字符的正文使用 CDATA，属性使用实体转义，换行统一为 LF，XML 1.0 禁止的控制字符替换为 `U+FFFD`。已编码实体和嵌套 XML 样式 payload 保持字面语义，不由 handler 预转义。Agent 回填开头的本地化系统标题不属于 XML；需要严格解析时从 `<mtp_response>` XML 块开始。

## 4. 控制异常

### 4.1 GlobalSystemBus

请求未注册 route 时抛 `KeyError`。这表示装配或生命周期错误，不应伪装成空业务结果。Pub/Sub subscriber 异常则被隔离和记录，不传播给 publisher。

### 4.2 Gateway

- 外层 task cancellation：由 Chat application 取消 Gateway child task，原生 `asyncio.CancelledError` 直接传播；
- `GatewayTimeoutError`：总 deadline 无法形成完整终态；
- workflow 内部不变量或 step 投影失败：保留原异常并发布 failed RuntimeEvent。

局部 invoke 的 timeout 或 `RecoverableGatewayError` 可以使用该 step 声明的 fallback。其他异常、投影错误和状态不变量错误不属于可恢复调用失败。

### 4.3 Passive Ingress

`PassiveIngressContractError` 表示下游违反协议，例如 `PASSIVE_MEMORY` 返回 command outcome。它不是可重试基础设施错误。submission queue admission 失败会向调用方抛出明确异常，同时保留当前 accumulator；admission 后的 apply 失败由 Work Queue Runtime 按 policy 重试并通过通用 RuntimeEvent 观测。

### 4.4 Workspace 边界错误

Workspace 错误在资源所有者或身份交接边界产生，不能由 RuntimeEvent 或 HTTP 层伪装成空结果。缺少作用域、actor 与 owner 不一致、或资源不属于请求 Workspace 时，应分别保留稳定机器码；WorkspaceAsset 的运行时状态错误同样由其 Store 直接表达：

| Code | 语义 |
|:---|:---|
| `workspace.scope_required` | 缺少完整 `IdentityScope` |
| `workspace.owner_mismatch` | actor 用户与 Workspace owner 不一致 |
| `workspace.mismatch` | 资源与请求 Workspace 不一致 |
| `workspace.asset.not_found` | 当前作用域找不到资产 |
| `workspace.asset.expired` | ref 已随当前进程运行时失效 |
| `workspace.asset.not_ready` | 资产尚未达到可用状态 |
| `workspace.asset.failed` | 必要资产表示生成失败 |
| `workspace.asset.removed` | 资产已进入不可逆 removed 状态 |
| `workspace.asset.stale_result` | revision 或 operation token 已过期 |
| `workspace.asset.operation_conflict` | 相同幂等操作携带了不一致输入 |

这些错误表示跨边界拒绝或当前 WorkspaceAsset 生命周期状态，不改变 MTP error/warning 的表达规则。Workspace 资源归属、opaque ref 和 shutdown 清理的完整语义见[Workspace 架构](../architecture/workspace.md)。

## 5. 观测失败与业务失败

RuntimeEventSink 是 best-effort：

- 未配置时使用 `NullRuntimeEventSink`；
- emit 或 event sink 失败不得改变业务返回值；
- Gateway event sink 失败不能把成功 decision 改成 workflow failure；
- 业务失败可以产生 `*.failed` RuntimeEvent，但事件本身不是失败事实的唯一存储。

因此，业务调用方以返回值/异常为准，运维和 UI 观测以 RuntimeEvent 为辅助。

这也是错误机制不能合并的一个现实原因：观测流允许丢失、回放缺口和慢订阅者隔离，业务终态却必须在调用点得到确定答案。把 `*.failed` RuntimeEvent 当成唯一失败事实，会让断连后的客户端无法知道任务究竟失败还是只是漏掉了事件。

## 6. i18n 与信息泄漏

- 面向 Agent 的 MTP error/warning 由 formatter 根据 `message_key` 和当前 language 本地化；
- syscall key 使用 `syscall.*` 文本表，其他 MTP key 使用 `mtp.*` 文本表；
- 内部 exception `cause` 不进入 Agent 可见 XML；
- 不应把 traceback、文件系统绝对路径、密钥、完整 tool args 或 memory context 放进公共 message；
- RuntimeEvent 的 `data` 也应保持摘要化，尤其是 Passive Ingress 内容和工具参数。

隐藏 `cause` 不只是为了避免泄漏路径、密钥或内部数据，也是在保护协议稳定性。exception 文本随依赖版本和实现细节变化，既不适合作为 Agent 的机器判断依据，也可能诱导模型针对偶然实现编写脆弱的重试策略。稳定 `code`、本地化 message 和受控 params 才是公共契约；完整 cause 留给受保护的调试与日志通道。

## 7. 当前限制

- HTTP 层仍由各 router/exception handler 分别翻译错误，尚无项目级稳定 API error schema；
- RuntimeEvent 不是持久化审计记录；
- Command `error_code` 尚未形成与 MTP code 等价的全局注册表；
- 部分历史实现仍可能抛通用异常，调用方应在顶层边界记录并终止，不能猜测为可恢复错误。

这些限制应进入后续 System/API 计划，不通过扩大 MTP error 模型来掩盖。

## 8. 设计矛盾检查

评审错误处理改动时，应检查：

1. 这是预期业务终态、Agent 可修正错误、必须中断的控制异常、部分降级，还是纯观测信息？表达方式是否与调用方下一步行动一致？
2. `agent_fault` 是否真的可以通过修改请求解决，`system_fault` 是否避免鼓励无意义的同参数重试？
3. warning 是否保留了一份仍可安全消费的主要成果？若没有，是否应该返回 error？
4. fallback 是否只覆盖声明过的可恢复失败，还是正在掩盖 route 缺失、投影失败或不变量破坏？
5. RuntimeEvent 或日志是否被当成唯一业务事实，观测失败是否可能改变正常返回值？
6. 公共 message、params 或 event data 是否暴露了 cause、路径、工具参数或记忆正文？
7. HTTP 层是否仍按各入口翻译错误？在项目级 schema 尚未建立前，文档是否误称已有统一 envelope？
8. 新错误码是否真的改变机器处理类别，还是只需要新的 `message_key + params`？

## 9. 变更规则与验证

修改错误语义时必须：

1. 保持稳定 code 的已有含义；
2. 补齐中英文 `message_key`；
3. 验证 cause 不会序列化；
4. 区分 error、warning、业务终态和 RuntimeEvent；
5. 为重试/取消/降级边界增加测试；
6. 同步更新[MTP 契约](./mtp.md)和调用方文档。

验证入口：`tests/unit/core/mtp/`、`tests/unit/agent_runtime/mtp/`、`tests/unit/gateway/test_phase3c_workflow.py`、`tests/unit/system/services/passive/`、`tests/unit/system/runtime/test_runtime_events.py`。
