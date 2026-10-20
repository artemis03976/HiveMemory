---
title: Error Model
status: current
owner: system
scope: cross-boundary-errors-and-degradation
code_paths:
  - src/hivememory/core/errors/
  - src/hivememory/infrastructure/llm/litellm_service.py
  - src/hivememory/engines/generation/extractor.py
  - src/hivememory/workspace/contracts/errors.py
  - src/hivememory/patchouli/contracts/errors.py
  - src/hivememory/agent_runtime/mtp/error_mapping.py
  - src/hivememory/alice/orchestration/call/response.py
  - src/hivememory/core/mtp/exceptions.py
  - src/hivememory/core/mtp/models.py
  - src/hivememory/core/mtp/formatter.py
  - src/hivememory/core/protocol/gateway.py
  - src/hivememory/system/services/passive/exceptions.py
  - src/hivememory/components/bus/async_bus.py
  - src/hivememory/system/access/
  - src/hivememory/workspace/authentication.py
  - src/hivememory/workspace/authorization.py
  - src/hivememory/workspace/contracts/operations.py
  - src/hivememory/server/error_handlers.py
  - src/hivememory/server/models/errors.py
related_contracts:
  - docs/contracts/mtp.md
  - docs/contracts/routes-and-events.md
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-10-19
---

# 跨边界错误模型

HiveMemory 在产生失败的边界保留中立类型，在 HTTP、普通 MTP 与 CALL 出口分别投影安全响应。API 错误共用 `ErrorResponse`；业务结果、取消、warning 和观测事件仍各自表达原有语义，不强行套入同一种信封。本文维护分类、所有权与公开内容规则，代码是类型和映射的定义源。

错误不只是写给开发者看的日志。对 Agent 而言，它还是下一步行动的控制输入：语法错误可以修正，alias 未命中可以换一种检索方式，权限拒绝意味着不能继续尝试绕过，而存储离线则提示当前能力暂不可用。一个只有 traceback 或模糊“执行失败”的系统，即使记录了足够多的内部细节，也无法帮助 Agent 安全地调整计划。

与此同时，并非所有失败都应该进入同一种响应信封。业务终态需要稳定返回，取消和契约违约需要中断控制流，warning 要保留仍可消费的部分成果，RuntimeEvent 则只服务于旁路观测。强行统一这些机制，会让调用方分不清“任务已被拒绝”和“程序装配错误”，也可能让观测设施反向影响业务正确性。

## 1. 表达方式

| 机制 | 使用场景 | 调用方行为 |
|:---|:---|:---|
| 业务结果状态 | 命令、Agent run、Passive outcome 等预期终态 | 检查判别字段或 status，不依赖异常 |
| 中立异常 | 所有者判定的资源、访问、存储和子系统失败 | 按具体类型与受控原因处理，保留 cause，不依赖协议层 |
| `MTPErrorInfo` | MTP 对 Agent 的错误呈现 | 根据具体类别与文案调整行动，severity 不授予重试许可 |
| `ErrorResponse` | HTTP API 请求失败 | 检查 status/error/reason，展示安全 detail |
| 控制异常 | cancel、timeout、契约违约、route 缺失 | 终止或由顶层应用服务翻译 |
| warning | 部分命中、无结果、容错解析等非致命降级 | 保留成功结果，同时呈现提示 |
| RuntimeEvent | 运行过程、失败原因和降级观测 | 仅观测，不改变业务结果 |

选择表达方式时，先问调用方接下来需要做什么。若结果属于正常状态空间，就用判别字段；若 Agent 能根据稳定类别改变行为，就用结构化协议错误；若继续执行会破坏不变量，就抛控制异常；若主要成果仍然有效，只需附带 warning；若信息只用于追踪，则发布 RuntimeEvent。这里不存在一种适合所有边界的“最统一”格式。

### 1.1 定义归属与类型粒度

`core.errors` 提供 L0 的访问、资源、存储与模型错误（模型查找、提供商失败与上下文超窗）；`DomainError` 只携带稳定 `code` 和内部 `details`，不包含 HTTP status、MTP severity 或自动重试开关。会话与写入意图错误归 `workspace.contracts.errors`，Profile、Topic 与交互提交错误归 `patchouli.contracts.errors`。子系统只导入提供方公共契约；队列、codec、预算与私有控制异常保留在其机制或算法所有者处。

新增类型需要独立的处理分支、必需上下文、不变量或公共兼容义务。仅诊断原因不同、处理和上下文相同的失败使用受约束的 reason；已有业务终态直接构造响应，不为序列化再创建异常对象。容量不足、停止接纳、状态冲突，编码前拒绝与接纳后解码失败，以及写入拒绝与结果未知都有不同恢复语义，不能为减少类型而合并。

## 2. 业务终态

### 2.1 Gateway command

`CommandExecutionResult.status`：

- `completed`；
- `rejected`；
- `failed`；
- `requires_confirmation`；
- `not_implemented`。

这些是系统指令的可预期终态，作为正常结果返回给调用方（流式 `command_result`、非流式 `NonStreamingCommandOutcome`）。Gateway 只解析命令，`GatewayCommandOutcome` 只携带解析结果；命令终态由任务进程按解析状态产生，当前只会出现 `not_implemented`（解析成功，`command.unavailable`：内置命令暂时不可用）与 `rejected`（解析失败，`command.parse.<解析状态>`），其余取值随命令系统接回时使用。`error_code` 可提供稳定机器标识，`message` 面向当前客户端展示。

`rejected`、`not_implemented` 或 `requires_confirmation` 并不表示 Gateway 或任务进程自身发生异常。把它们建模为正常结果，可以让 transport 稳定展示命令终态，也避免顶层异常处理器误将一次业务拒绝记录成系统故障。

### 2.2 Agent run

CPU 执行结果（`CPUExecutionResult`）的 `status` 为 `completed`、`cancelled` 或 `failed`。任务进程对有输出的三种结局封口、写入会话并 finalize；无输出、异常或 finalize 前断连执行控制收尾与 prepared cleanup。已进入 finalize 后交付方取消也必须等待提交完成。

`CPUExecutionResult.failure_reason` 是可选的 `CPUFailureReason`，只在 failed 时有意义；未识别为以下类型的错误不填原因码，保留通用失败处理。Alice 按类型映射以下四种原因：

| 原因码 | 含义与失败边界 |
|:---|:---|
| `input_too_large` | 固定输入本身超限，或必须压缩时连固定输入与完整摘要预留都无法容纳 |
| `compaction_failed` | 历史压缩后仍超出输入上限，包括摘要拒绝后无法用减重视图继续；关闭压缩时历史超限也使用此码，不交给轮内守卫 |
| `turn_context_budget_exceeded` | frame 每次生成前的预算守卫拦截，不再调用模型；CALL 子帧沿 CALL error 返回父帧 |
| `model_context_window_exceeded` | 提供商仍报告上下文超窗；LLM 服务在 SDK 边界转换为 `ModelContextWindowExceededError`，Alice 按该类型识别，不重试 |

任务进程的 failed done 保留 `reason=agent_run_failed`，另投影非空 `failure_reason`；`chat.run.failed.message` 使用相同原因码。原因不含异常正文，不写入交互原文。可用的历史检查点可以随 failed 或无输出结果交回，接受与本轮是否封口独立；原生取消仍传播 `CancelledError`。

Alice CPU 宿主未启动或已停止时，await `execute()` 抛出 `AliceUnavailableError`（`RuntimeError`）。这是执行准入异常，任务进程按失败清理，不投影为四类预算原因。该类型定义在 Alice 接口层，不属于跨子系统可导入的契约类型；异常上的 `code=alice.unavailable`、`details.reason=not_running` 只供日志诊断，任务进程不据此分支，SSE 与 `chat.run.failed` 仍按未分类的进程异常投影。生命周期与交付边界见[CPU 契约](./subsystem-contracts.md#43-alice-的执行入口)。

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

严重度表达归因，不单独决定重试、日志等级或数据安全等级。参数错误可以修正；权限拒绝必须停止被拒操作；写入结果未知和线程结束遵循各自的停止/查询指导。

区分两者的目的，是给生成循环一个保守的重试边界。`agent_fault` 意味着改变请求有机会成功；`system_fault` 意味着继续用相同参数试探大多只会重复失败或放大负载。它不宣称系统故障永远不可恢复，也不授权 Agent 对权限错误持续换写法绕过。

### 3.3 稳定错误码

| Code | 类别 |
|:---|:---|
| `mtp.parse.syntax_error` | MTP 语法错误 |
| `mtp.alias.not_found` | alias 无法解析 |
| `mtp.memory.not_found` | 保留的兼容类别，当前无生产路径 |
| `mtp.memory.intent_rejected` | 写入意图明确未接纳 |
| `mtp.memory.intent_unknown` | 派发结果未知，原句柄可能继续结算 |
| `mtp.memory.type_mismatch` | 记忆类型不适用于操作 |
| `mtp.argument.invalid` | 参数缺失或格式错误 |
| `mtp.permission.denied` | frame 动词集合（由输入清单或被调用方的操作快照翻译，子帧移除 CALL）、Workspace actor 操作拒绝，或 CALL 目标不可用 |
| `mtp.system.storage_offline` | 存储离线 |
| `mtp.system.storage_error` | 存储读取错误 |
| `mtp.system.storage_write_error` | 存储写入未确认成功，不承诺没有副作用 |
| `mtp.system.service_unavailable` | 所需总线路由/服务不可用 |
| `mtp.system.tool_error` | 工具内部错误 |
| `mtp.system.fault` | 未分类的内部故障；不暴露 exception cause |
| `mtp.syscall.invalid_argument` | syscall 参数错误 |
| `mtp.syscall.permission_denied` | syscall 权限拒绝 |
| `mtp.syscall.execution_error` | syscall 执行失败 |
| `mtp.syscall.timeout` | syscall 超时 |
| `mtp.syscall.unavailable` | syscall 依赖不可用 |
| `mtp.call_response.sub_agent_error` | CALL 子 Agent 失败 |
| `mtp.call_response.budget_exhausted` | CALL 子 Agent 耗尽执行迭代预算 |
| `mtp.call_response.unexpected_suspend` | 保留的兼容类别；实际 SUSPENDED 终态违约抛 ValueError |

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

`MTPResponse.content` 只承载成功业务内容；error 和 warning 分别使用结构化字段，不能先拼进 content 再要求调用方解析。KoakumaRuntime 使用 `mtp_error_info()` 统一处理解析与普通 verb handler 异常；映射显式接收 verb/alias，按白名单构造 params，不复制内部 details。CALL 准备失败由 `preparation_error_response()` 统一投影，已定稿子帧结果直接构造响应。取消和 CALL suspend 继续按专用控制流处理。

`MTPFormatter` 是普通 Agent-facing MTP 回填文本的唯一构造点：它根据 language 渲染 code/severity/message/warnings，并排除内部 cause。`response_content` 只是业务 payload，不等于完整回填；需要写回模型历史时应使用运行结果的 `formatted_response`。CALL 不经过旧的通用 IPC 文本拼接，而由 `RunExecutor`/`CallCoordinator` 消费 `MTPCallRequest`；`CallContextProvider` 在编排边界解析 Profile 和共享上下文。Executor 递归等待被调用 frame，CallCoordinator 先 finalize callee，再由 `alice/orchestration/call/response.py` 把最终 `FrameExecutionResult` 单向映射为一个 `MTPCallResponse`，最后交给 `AgentRuntime.apply_call_response()` 恢复 caller frame。response 提交早于可等待的结束事件发送；内部已结算的 `CANCELLED` 结果可以映射为 CALL response，但外层 task cancellation 不回填伪造 response，也不终止 caller 重入流程。`SUSPENDED` 是非终态控制流，不参与错误映射。

Formatter 把 content、CALL reply、artifact alias、本地化 error reason 和 warning 统一作为原始文本编码：含 XML 保留字符的正文使用 CDATA，属性使用实体转义，换行统一为 LF，XML 1.0 禁止的控制字符替换为 `U+FFFD`。已编码实体和嵌套 XML 样式 payload 保持字面语义，不由 handler 预转义。Agent 回填开头的本地化系统标题不属于 XML；需要严格解析时从 `<mtp_response>` XML 块开始。

## 4. 控制异常

### 4.1 GlobalSystemBus

请求未注册 route 时抛 `KeyError`。这表示装配或生命周期错误，不应伪装成空业务结果。Pub/Sub subscriber 异常则被隔离和记录，不传播给 publisher。

### 4.2 Gateway

- 外层 task cancellation：由 Chat application 取消 Gateway child task，原生 `asyncio.CancelledError` 直接传播；
- `GatewayTimeoutError`：总 deadline 无法形成完整终态；
- workflow 内部不变量或 step 投影失败：保留原异常并发布 failed RuntimeEvent。

局部 invoke 的 timeout 或 `RecoverableGatewayError` 可以使用该 step 声明的 fallback。其他异常、投影错误和状态不变量错误不属于可恢复调用失败。

Gateway Context Provider 只包装 `StorageOfflineError`、`StorageReadError`、`ResourceUnavailableError`；超时直接交给 workflow。两个分析 Engine 只把 `ModelProviderError`（含超窗）与 `JSONParseError`/显式输出校验失败转换为能力错误；SDK 异常由 LLM 服务在边界转换（第 4.6 节），引擎不识别 SDK 类型。JSON 输出要求只在 SDK 明确报告不支持该参数时去掉格式参数再请求一次，其他失败直接传播。缺失 route、参数接线、投影和未知程序错误不会进入局部 fallback。

### 4.3 Passive Ingress

`PassiveIngressContractError` 表示下游违反协议，例如 `PASSIVE_MEMORY` 返回 command outcome。它不是可重试基础设施错误。submission queue admission 失败会向调用方抛出明确异常，同时保留当前 accumulator；admission 后的 apply 失败由 Work Queue Runtime 按 policy 重试并通过通用 RuntimeEvent 观测。

Passive 的可恢复集合只有 `TimeoutError`、`GatewayTimeoutError`、`StorageOfflineError`、`StorageReadError`、`ResourceUnavailableError`。允许降级时仍接收原始 user 事件；retrieval 失败保留已有 topic decision。未知错误、授权拒绝、契约违约与原生取消向上传播，不能被“尽力接收”掩盖。

### 4.4 Workspace 与访问边界错误

Workspace 错误在资源所有者、访问边界或身份交接边界产生，不能由 RuntimeEvent 或 HTTP 层伪装成空结果。缺少作用域、访问被拒绝、或资源不属于请求 Workspace 时，应分别保留稳定机器码；WorkspaceAsset 的运行时状态错误同样由其 Store 直接表达：

| Code | 语义 |
|:---|:---|
| `workspace.scope_required` | 缺少完整 `IdentityScope`，或访问 context 无效：传入的不是访问 context，或 context 已撤销（`reason=context_not_issued`），或授权时准入记录已不存在（`reason=actor_not_admitted`） |
| `workspace.mismatch` | 资源与请求 Workspace 不一致，例如 prepare 返回的 scope 与授权组装的 scope 不一致 |
| `workspace.admission_denied` | 统一认证网关的接入/准入拒绝（`details.reason` 区分阶段：`unknown_principal`、`adapter_mismatch`、`actor_not_allowed_for_principal`、`actor_not_owner`、`actor_not_admitted`、`authentication_gateway_closed`） |
| `workspace.operation_denied` | 第 3 阶段授权失败（`details.reason` 区分：`target_workspace_not_resident`、`target_owner_mismatch`、`operation_not_allowed`、`root_required`） |
| `workspace.resource.not_found` | 当前 Workspace 作用域内不存在目标资源 |
| `workspace.resource.not_visible` | 资源存在但当前 Actor 未通过可见性授权（与 not_found 有意区分；调用方可按契约合并呈现） |
| `workspace.resource.unavailable` | 资源 provider 暂时不可用，不代表资源不存在或被拒绝 |
| `workspace.memory.intent.pending_update_not_allowed` | UPDATE 的基础仍是 PENDING/MATERIALIZING 意图，不能作为正式 atom 更新（`PendingUpdateNotAllowedError`） |
| `workspace.asset.not_found` | 当前作用域找不到资产 |
| `workspace.asset.expired` | ref 已随当前进程运行时失效 |
| `workspace.asset.not_ready` | 资产尚未达到可用状态 |
| `workspace.asset.failed` | 必要资产表示生成失败 |
| `workspace.asset.removed` | 资产已进入不可逆 removed 状态 |
| `workspace.asset.stale_result` | revision 或 operation token 已过期 |
| `workspace.asset.operation_conflict` | 相同幂等操作携带了不一致输入 |

访问边界错误按两阶段认证与两阶段授权的阶段区分（模型见[Workspace 架构](../architecture/workspace.md)第 4 节）：

| 阶段与情形 | 错误 | `details.reason` | HTTP |
|:---|:---|:---|:---|
| 第 1 阶段：接入未登记或已禁用（不区分，避免泄漏配置） | `AdmissionDeniedError` | `unknown_principal` | 403 |
| 第 1 阶段：adapter 不匹配 | `AdmissionDeniedError` | `adapter_mismatch` | 403 |
| 第 1 阶段：principal 的身份解析规则不允许该用户 | `AdmissionDeniedError` | `actor_not_allowed_for_principal` | 403 |
| 第 2 阶段：actor 用户不是要进入的 Workspace 的 owner | `AdmissionDeniedError` | `actor_not_owner` | 403 |
| 第 2 阶段：准入记录缺失或已禁用 | `AdmissionDeniedError` | `actor_not_admitted` | 403 |
| 认证网关已关闭（System 停止中） | `AdmissionDeniedError` | `authentication_gateway_closed` | 503 |
| 第 3 阶段：目标 Workspace 不等于驻留 Workspace | `OperationDeniedError` | `target_workspace_not_resident` | 403 |
| 第 3 阶段：actor 用户不是目标 Workspace 的 owner | `OperationDeniedError` | `target_owner_mismatch` | 403 |
| 第 3 阶段（actor API）：白名单不含该 actor 操作 | `OperationDeniedError` | `operation_not_allowed` | 403 |
| 第 3 阶段（root API）：actor 不是保留的 `system` | `OperationDeniedError` | `root_required` | 403 |
| 访问 context 无效：不是访问 context，或已撤销（线程已结束见下文）；授权时准入记录已不存在；把进程 context、请求级 context 或未登记在线程表中的线程 context 交给操作入口；从非进程 context 派生线程 context；以子线程派生 CALL 子线程；CPU 结束主线程 | `ScopeRequiredError` | 无 / `context_not_issued` / `actor_not_admitted` | 500：生产入口经网关签发后不应出现，出现即为接线缺陷 |
| 资源与请求 Workspace 不一致 | `WorkspaceMismatchError` | —— | 409 |

- 缺少行为许可属于授权失败，不是身份认证失败；空白名单可以进入，但所有 actor 操作都被拒绝。未知 operation 在访问登记装载期失败。`operation_not_allowed` 的 `details.operation` 给出被拒绝的 actor 操作；root 授权与任务进程的阶段授权不检查 operation，它们的目标拒绝与 `root_required` 不附带 `operation`。阶段授权只校验准入与目标，因此不会出现 `operation_not_allowed`。
- 访问 context 没有固定有效期，不存在“过期”拒绝；认证网关关闭只拒绝新认证，已签发的 context 照常可用，直到被撤销。
- HTTP 映射由 server 的异常处理器统一完成（`server/error_handlers.py`），响应体包含稳定机器码 `error`、安全 `detail` 与可选 `reason`，不暴露接入凭据。
- 拒绝阶段可观测的最小记录为 principal 标识、Actor/Workspace 安全投影、operation、关联 ID 与拒绝阶段，不记录接入凭据和原始对话。

这些错误表示跨边界拒绝或当前 WorkspaceAsset 生命周期状态，不改变 MTP error/warning 的表达规则；访问错误必须沿 System/application 以原语义传播，不能被通用 `RuntimeError` 捕获包装成服务不可用。Workspace 资源归属、认证/授权模型、opaque ref 和 shutdown 清理的完整语义见[Workspace 架构](../architecture/workspace.md)第 4 节。

actor API 与操作入口的错误保持各自边界：每种 actor 请求缺少对应操作时抛 `OperationDeniedError`（`operation_not_allowed`）——READ 与 CALL 的 `context_refs` 缺少 `memory.read`、SEARCH 缺少 `memory.search`、RUN 的记忆目标缺少 `memory.run`、WRITE 缺少 `memory.write`、UPDATE 缺少 `memory.update`、CALL 目标读取缺少 `agent.call`（包括内置目标）；UPDATE 基础为 pending 时抛 `PendingUpdateNotAllowedError`，不存在、不可读或不是正式 atom 时抛 `ResourceNotFoundError`。Koakuma 分别映射为现有 `mtp.permission.denied`、`mtp.argument.invalid` 与 `mtp.alias.not_found`，provider 不可用映射为 `mtp.system.service_unavailable`，不暴露内部 cause。

CALL 派生子线程时，被调用方在本 Workspace 没有准入或为保留的 `system`，认证一侧抛 `AdmissionDeniedError`（`actor_not_admitted` / `system_actor_not_allowed`），不签发、不登记子线程；Alice 映射为 `mtp.permission.denied`，文案键 `mtp.call.agent_unavailable`（目标 agent 不可用），与调用方缺少 `agent.call` 时的 `mtp.permission.verb_denied` 区分。子线程试图再派生抛 `ScopeRequiredError`，属于接线缺陷，Koakuma 呈现为 `mtp.system.fault`；冷读期间调用方结束或进程收到停止抛 `ThreadEndedError`（见下文）。

线程已结束时，操作入口、CALL 派生与 actor API 的同步段抛 `ThreadEndedError`（`core.errors`，继承 `DomainError`，由 `workspace.contracts` 重新导出，code 为 `workspace.thread.ended`）：线程的凭据已撤销、线程已结束（子线程随 CALL 返回，或 Actor 阶段收尾），或所属进程已收到停止，都属于这一种生命周期拒绝，六项 actor 操作得到同一种错误。未知请求类型抛 `TypeError`，是接线违规；两者都与表示接线缺陷的 `ScopeRequiredError` 区分。它们没有独立 HTTP API，不新增 HTTP 映射。普通 MTP 与 CALL 将线程结束呈现为 `mtp.system.fault`，文案键 `mtp.system.thread_ended`：原线程已结束，停止操作，不重试或重新获取身份。未知入口异常使用固定系统故障文案，不暴露 cause。线程结束同步撤销线程 context，不取消其他在途能力调用任务，已开始的只读操作可以自然完成；WRITE、UPDATE 在基础校验之后、登记之前的同步段解析线程，跨越线程结束或停止时拒绝登记；派发开始后持有提交任务直到完成，再传播调用方取消。外部 task cancellation 仍以 `asyncio.CancelledError` 传播。

WRITE/UPDATE 登记后的派发错误携带 `pending_alias` 与 `intent_id`，不暴露内部 cause：

| Workspace 异常 / code | MTP 错误 / 严重度 | 句柄状态 |
|:---|:---|:---|
| `IntentDispatchRejectedError` / `workspace.memory.intent.dispatch_rejected` | `mtp.memory.intent_rejected` / agent_fault | 明确未接纳，同步标为 FAILED；失败通知可重复到达 |
| `IntentDispatchUnknownError` / `workspace.memory.intent.dispatch_unknown` | `mtp.memory.intent_unknown` / system_fault | 保持 MATERIALIZING，不假定未入队；若已接纳，结算仍可推进终态 |

两种错误均在 ACK 之前交给 Agent，中英文文案分别说明拒绝与未知。未知结果只在策略允许 READ 时建议查询已有句柄；没有 READ 能力则明确无法确认，任何情况下都不自动重复 WRITE/UPDATE。总线异常或接纳响应丢失按未知处理；队列停止、容量拒绝、UPDATE 基础丢失及本地意图参数非法按明确拒绝处理。由于 MTP 错误不携带内部 cause，根因只由 actor API 记录日志：总线异常导致的未知结果以 error 级别连同异常堆栈记录，并关联 `intent_id` 与 `pending_alias`，便于排查长期停在 MATERIALIZING 的句柄。

引用交付的 citation 使用 `source="workspace.reference_read"`；普通记录失败只记日志、不改变可读结果，取消继续传播，已记录的引用不回滚。因此 RUN 工具失败不会撤销执行前已经完成的正式原子读取计数。

会话错误由 `workspace.session.*` 表达：`not_found`（404，含越域隐藏）、`busy`（409，同一会话认领冲突或运行中删除）、`capacity_exceeded`（409，Workspace 会话数/会话轮数上限）、`closed`（503，服务已关闭）。这些错误在注册或管理操作时显式返回，不自动新建替代不存在的会话、不淘汰历史。

`SessionCheckpointError` 使用 `workspace.session.checkpoint_invalid`，表示检查点游标不属于会话、摘要边界晚于减重边界，或相对同一产生方与格式版本的已有检查点倒退；产生方或格式版本不同的检查点可以替换旧值。归属与认领错误沿用会话既有错误；任何拒绝都不改变原检查点。进程写回时仅记录告警、继续本轮收尾，没有公开的检查点写入 HTTP 入口。

### 4.5 Memory 输入校验与存储错误

| 异常 | 产生位置 | HTTP 呈现 | 语义 |
|:---|:---|:---|:---|
| `InvalidMemoryFieldError` | 接收外部输入的 Memory 构造/编辑点（管理创建、外部编辑），包装 Pydantic `ValidationError` | 422 | 调用方提交的字段取值不满足领域约束；内部保留字段诊断，HTTP 返回固定安全提示，与程序错误区分 |
| `StorageWriteError` | Qdrant 写入路径（`upsert_memory` / `patch_memory_payload` 等） | 5xx | 存储写入失败的结构化表达，与 `StorageOfflineError`/`StorageReadError` 同族；失败沿调用链传播，不能被吞成成功或空结果 |
| `MemoryAliasConflictError`（`workspace.memory.alias_conflict`） | 中期库 `upsert` 写入前校验、生成侧 `AliasGenerator`、精确 alias 查询 | 409（Memory 与 Agent Profile 管理 HTTP 入口） | 同一 Workspace 中期库内的 alias 唯一性不变量被破坏；`details.reason` 区分 `alias_occupied`（被其他 Memory 占用，本次不写入）、`alias_candidates_exhausted`（消歧候选耗尽）与 `ambiguous_alias`（精确查询命中多条，读取 fail closed） |
| `MemoryDecodeError` | Memory codec 读取边界 | 5xx | payload 无法安全归一化为 schema `"2.1"`（缺 `schema_version`、未知版本、投影不一致等），fail closed |

Memory 领域模型的 `validate_assignment` 使构造与就地赋值执行同一套约束；外部编辑入口把值相等判定为无变化时不产生版本（§3.2 语义见 Patchouli 生成文档）。

存储适配器仅将已知 SDK/连接故障转换为 `StorageOfflineError`（`storage.offline`）、`StorageReadError`（`storage.read_failed`）或 `StorageWriteError`（`storage.write_failed`），保留操作上下文与 cause。检索故障不返回空成功；写入故障不证明副作用未发生。局部模型、payload 解码与程序异常不冒充离线，取消原样传播。

Profile 缺失和不可见统一隐藏为 `ResourceNotFoundError`；已找到但类型/配置不适用时为 `ProfileResolutionError`，reason 仅为 `type_mismatch` / `invalid_config`，必须携带 `agent_alias`。CALL 在出口转为既有 MTP 类别，CPU 分配按失败收尾，两者都不回退默认 Profile。

Topic 缺失使用 `ResourceNotFoundError`，忙碌使用 `TopicBusyError`，明确的容量/停止接纳使用 `TopicSettleAdmissionError`。Active finalize 以 `ActiveInteractionFinalizationError` 保留 admission/apply 阶段；未知提交失败为 `admission_unknown`，类名放在诊断字段 `error_class`，不能据此声称可安全重试。生成任务的明确拒绝仅覆盖接纳前 codec、lane、容量与停止错误；任意 TypeError/ValueError、重复任务或未知队列失败保留 UNKNOWN 与原句柄结算权。

### 4.6 模型调用错误

`ModelProviderError` 与其子类 `ModelContextWindowExceededError` 定义在 `core.errors.model`，与 `ModelNotFoundError` 同处。它们只由 LLM 服务的实现（`LiteLLMService`）在 SDK 边界产生：完整调用、流式启动与流式拉取采用同一转换，超窗先于其他请求错误识别；由 SDK 异常转换而来时原始异常保留在原因链，异常文本是固定说明。提供商没有返回文本时同样抛出 `ModelProviderError`，此时没有原始异常。程序错误与取消不转换，原样传播。超窗作为提供商失败的子类，与 SDK 中超窗属于请求错误一致：按提供商失败处理的调用方无需单独捕获，需要区分超窗的调用方先判断子类。

| 使用方 | 处理 |
|:---|:---|
| Gateway 查询理解与话题路由 | 转为各自的能力错误，由 workflow 保守降级（第 4.2 节） |
| Patchouli 记忆抽取与合并 | 只对提供商失败重试，一次提取或合并最多两次请求；超窗与程序错误不重试。重试耗尽后抛出最后一次的 `ModelProviderError`；模型输出为空或无法解析是“没有产出”，不是失败 |
| Patchouli 生成引擎 | Mode A 不捕获提供商失败，生成任务以 `memory_generation_failed` 失败收尾，不再记录为没有产出；Mode B/C 对提供商失败使用既有 fallback，保证明确的保存与更新意图不因 Librarian LLM 暂时失败而丢失；程序错误不降级 |
| Page Folding 摘要 | 只在提供商失败时回退为规则摘要，程序错误沿 compact 路径传播 |
| Alice 帧生成 | frame 形成 `FAILED`；超窗投影为 `model_context_window_exceeded`，其他提供商失败不填原因码（第 2.2 节） |
| Alice 会话摘要器 | 以 `summary_model_context_exceeded` / `summary_provider_error` 拒绝本次摘要并保守回退 |

## 5. 观测失败与业务失败

RuntimeEventSink 是 best-effort：

- 未配置时使用 `NullRuntimeEventSink`；
- emit 或 event sink 失败不得改变业务返回值；
- Gateway event sink 失败不能把成功 decision 改成 workflow failure；
- 业务失败可以产生 `*.failed` RuntimeEvent，但事件本身不是失败事实的唯一存储。

因此，业务调用方以返回值/异常为准，运维和 UI 观测以 RuntimeEvent 为辅助。

这也是错误机制不能合并的一个现实原因：观测流允许丢失、回放缺口和慢订阅者隔离，业务终态却必须在调用点得到确定答案。把 `*.failed` RuntimeEvent 当成唯一失败事实，会让断连后的客户端无法知道任务究竟失败还是只是漏掉了事件。

观测中的 `error_class` 只作诊断，不是稳定分类或重试依据；类合并或迁移后可变化。Passive reason 使用 `gateway_unavailable` / `retrieval_unavailable`，成功为 None。Gateway 与 Work Queue 的具体原因码见[事件契约](./routes-and-events.md#44-失败与降级观测)，不携带异常正文；生成任务的公开 error 使用固定说明，原始 WorkOutcome.error 仅留在进程内。

## 6. i18n 与信息泄漏

- 面向 Agent 的 MTP error/warning 由 formatter 根据 `message_key` 和当前 language 本地化；
- syscall key 使用 `syscall.*` 文本表，其他 MTP key 使用 `mtp.*` 文本表；
- 内部 exception `cause` 不进入 Agent 可见 XML；
- 不应把 traceback、文件系统绝对路径、密钥、完整 tool args 或 memory context 放进公共 message；
- RuntimeEvent 的 `data` 也应保持摘要化，尤其是 Passive Ingress 内容和工具参数。

隐藏 `cause` 不只是为了避免泄漏路径、密钥或内部数据，也是在保护协议稳定性。exception 文本随依赖版本和实现细节变化，既不适合作为 Agent 的机器判断依据，也可能诱导模型针对偶然实现编写脆弱的重试策略。稳定 `code`、本地化 message 和受控 params 才是公共契约；完整 cause 留给受保护的调试与日志通道。

## 7. HTTP API 与客户端呈现

`server/error_handlers.py` 的 `project_error()` 是唯一 HTTP 错误映射入口，`install_error_handlers()` 由 app 装配。路由只保留传输解析与资源关闭；领域失败沿原类型传播。模型定义在 `server/models/errors.py`，OpenAPI 的 API 错误响应引用同一个 `ErrorResponse`：

```text
error   必需字符串，稳定机器码
detail  必需字符串，可公开的安全说明
reason  可选字符串，仅输出各错误允许的原因
issues  可选列表，每项仅 loc / type / message
```

请求校验为 422；配置路由手动校验仍为 400。issues 不包含原始 input、校验 ctx 或任意 validator 原文。显式传输拒绝用 `APIError` 的受控类别；未知 `HTTPException` 保留 status/headers，但 detail 替换为固定提示。未知程序错误为 500、`server.internal_error` 和固定说明。资源缺失/不可见继续隐藏为 404，Topic busy 为 409，确认的依赖或接纳不可用为 503；不能从 503 推断副作用尚未发生。

SSE 开始后由进程流内投影输出安全 code/message，不能改 HTTP status；CPU failure_reason、事件名、部分输出与 finalize 规则保持独立。readiness、静态文件、WebSocket 和正常业务 outcome 不强制使用错误信封。`/runtime-events` 的 disabled 503 仍返回其业务状态 DTO。

前端 `services/apiError.ts` 是响应解析入口，普通服务与 SSE 握手共用它；保留各服务的状态异常类。已知旧 `detail`/`message` 字符串、`detail.message` 字符串和 422 校验列表可读取；未知对象、HTML、空体回落为带 status 的安全默认提示，AbortError 继续传播。完整消费规则见[前端传输](../frontend/state-and-transports.md)。

## 8. 设计矛盾检查

评审错误处理改动时，应检查：

1. 这是预期业务终态、Agent 可修正错误、必须中断的控制异常、部分降级，还是纯观测信息？表达方式是否与调用方下一步行动一致？
2. `agent_fault` 是否真的可以通过修改请求解决，`system_fault` 是否避免鼓励无意义的同参数重试？
3. warning 是否保留了一份仍可安全消费的主要成果？若没有，是否应该返回 error？
4. fallback 是否只覆盖声明过的可恢复失败，还是正在掩盖 route 缺失、投影失败或不变量破坏？
5. RuntimeEvent 或日志是否被当成唯一业务事实，观测失败是否可能改变正常返回值？
6. 公共 message、params 或 event data 是否暴露了 cause、路径、工具参数或记忆正文？
7. HTTP 映射是否仍只有一个入口，校验响应、SSE 和前端是否遵守各自的公开结构？
8. 新错误码是否真的改变机器处理类别，还是只需要新的 `message_key + params`？

## 9. 变更规则与验证

修改错误语义时必须：

1. 保持稳定 code 的已有含义；
2. 补齐中英文 `message_key`；
3. 验证 cause 不会序列化；
4. 区分 error、warning、业务终态和 RuntimeEvent；
5. 为重试/取消/降级边界增加测试；
6. 同步更新[MTP 契约](./mtp.md)和调用方文档。

验证入口：`tests/unit/core/mtp/`、`tests/unit/agent_runtime/mtp/`、`tests/unit/infrastructure/test_litellm_service.py`、`tests/unit/engines/generation/test_extractor.py`、`tests/unit/gateway/test_workflow.py`、`tests/unit/system/services/passive/`、`tests/unit/components/events/test_bus.py`、`tests/unit/workspace/test_access.py`、`tests/unit/system/access/`、`tests/integration/workspace/test_application_access_boundary.py`。


## 10. v0.7.0 兼容迁移

本次保留 MTP wire code、XML、业务状态与已确认的 HTTP 状态语义，但 Python 异常导入及 HTTP 校验结构有意改变，不宣称完全兼容。类型的逐项处置与数量审计见[归档记录](../archive/plans/v0.7.0-error-model-convergence.md)。

| 原入口 | 当前入口与捕获语义 |
|:---|:---|
| `core.errors` 的五项 Session 错误、PendingUpdateNotAllowedError、IntentDispatchRejectedError/UnknownError | `workspace.contracts.errors`；仍表示本子系统的会话和意图失败，L0 不反向重导出 |
| `core.mtp` 的 StorageOfflineError/ReadError/WriteError | `core.errors`；中立领域失败，已不继承 SystemFault；协议出口继续使用原 wire code |
| root API 的 MemoryNotFoundError/MemoryLifecycleUnavailableError | `core.errors.ResourceNotFoundError` / `ResourceUnavailableError`；未知 ValueError/RuntimeError 不转换成资源错误 |
| `patchouli.errors` 与 service 内的 ActiveInteractionFinalizationError | `patchouli.contracts.errors`；跨子系统只从 contracts 导入 |
| MTP MemoryIntentRejectedError/UnknownError | `agent_runtime.mtp.error_mapping` 的具名信息构造；内部控制流消费 Workspace 意图错误 |
| SubAgentExecutionError/SubAgentBudgetExhaustedError/AgentModelUnavailableError | `alice.orchestration.call.response` 的具名信息构造，子帧终态无需异常对象 |
| MTP MemoryNotFoundError/SubAgentUnexpectedSuspendError | 删除无生产用途的 Python 类型；原 wire code 仅保留为兼容类别 |
| `agent_runtime` 的 ModelProviderError/ModelContextWindowExceededError | `core.errors`；`agent_runtime` 包根继续导出，超窗改为提供商失败的子类 |
| `infrastructure.llm.errors.LLM_PROVIDER_ERRORS` | 删除；SDK 分类是 `LiteLLMService` 的私有细节，调用方只捕获 `core.errors` 的中立错误 |

请求校验旧 `detail` 列表改为字符串，字段级说明位于 `issues`。未知异常及 HTTP detail 不再原样公开；客户端应依据 status/error/reason 分支。原 Passive 默认允许未知异常降级、Gateway 包装缺失路由、CALL 跳过 READ 授权失败的行为均已收窄。未建立无实际消费者的兼容异常别名；版本发布和仓库外调用方升级须显式采用这些入口。
