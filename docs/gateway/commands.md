---
title: Gateway Commands
status: current
owner: gateway
scope: system-command-registry-parsing-and-dispatch
code_paths:
  - src/hivememory/gateway/commands/
  - src/hivememory/engines/gateway/interceptors.py
  - src/hivememory/gateway/workflow/topology.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/error-model.md
last_reviewed: 2026-07-29
---

# Gateway 全局命令

系统命令是一类控制消息，而不是另一种自然语言意图。它们需要确定性匹配、显式参数、权限检查和可审计的执行终态；如果把 `/clear`、`/status` 或未知 slash 输入交给 LLM 猜测，不仅结果不稳定，还可能让本应短路的控制消息进入检索、Alice 和长期记忆生成。

因此命令系统归属 Gateway 的入口控制能力，但把“识别”和“执行副作用”严格分开：L1 只解析，Dispatcher 才能执行。

## 1. 组件分工

```text
CommandDefinition
  -> CommandRegistry
       -> RuleInterceptor.match
            -> CommandParseResult
                 -> Gateway command branch
                      -> SystemCommandDispatcher
                           -> CommandExecutionResult
                                -> GatewayCommandOutcome
```

- `CommandDefinition` 声明名称、别名、参数、路由目标、权限和展示信息；
- `CommandRegistry` 负责注册、冲突检查、列表和匹配，不产生副作用；
- Parser 负责 token 与最小参数 schema，不执行 shell；
- `RuleInterceptor` 只把解析结果放入 Gateway state；
- `SystemCommandDispatcher` 是唯一允许执行命令副作用的组件；
- workflow 在命令执行后立即 finalize，不进入普通 decision flow。

Definition 与 ParseResult 都是冻结模型，参数 mapping 会转为不可变结构。它们可以跨 Step 传递，但不能在执行中被 handler 回写。

## 2. 入口与短路

只有 `ACTIVE_CHAT` 允许系统命令。L1 收到 slash 输入时，即使命令未知、参数无效、Registry 未启用，也会形成 SYSTEM 分支，随后由 Dispatcher 返回 `REJECTED`；未知命令不会落入 LLM Query Analysis。

`PASSIVE_MEMORY` 调用 interceptor 时设置 `allow_system=false`，所以外部对话中的 `/help` 或 `/clear` 只被当作普通被动消息分析，不会控制 HiveMemory。这个限制是被动摄入作为“观察者而非参与者”的安全边界。

命令终态不会继续执行 topic routing、retrieval、Alice、MTP 或 active memory generation。即使执行结果是 `rejected`、`failed`、`requires_confirmation` 或 `not_implemented`，也仍然是已经完成的 command outcome，而不是回退到 chat。

## 3. Registry 与匹配规则

Registry 在 Gateway Runtime 装配时创建并注册内置命令。注册阶段会拒绝：

- 重复 `command_id`；
- 同一定义内重复的主名称/别名；
- 与已有命令冲突的标准化别名；
- 不以 `/` 开头的命令名称。

名称会去除多余空白并按大小写不敏感方式匹配。多个别名都可能匹配时，优先选择 token 更长的别名，其次选择更小的 `priority`；仍同级时返回 `AMBIGUOUS`，不猜测执行。

`Registry.list()` 按 priority 与主名称排序，并默认隐藏 hidden definition。Listing 的最终可见性还会由 handler 根据 debug/admin 策略过滤。

## 4. Parser 的确定性边界

命令使用 `shlex.split(..., posix=True)` 切分，因此支持带引号的文本，但不执行变量替换、通配符、管道、重定向或任何 shell expansion。当前参数形式包括：

```text
/command value
/command --key value
/command --key=value
/command --flag
```

位置参数进入 `_positional` 列表；孤立 flag 为布尔 `true`；其余值保持字符串。Parser 只实现最小 schema：检查 required 字段，以及 string、boolean、integer、number、array 的基本类型兼容性。

这不是完整 JSON Schema。当前实现不会统一转换数值类型，也不会拒绝 schema 未声明的额外参数；handler 必须把 ParseResult 视为已经完成最小入口校验，而不是可信业务对象。

解析状态与执行状态必须分开：

- `matched`、`invalid_args`、`unknown`、`ambiguous` 描述解析；
- `completed`、`rejected`、`failed`、`requires_confirmation`、`not_implemented` 描述执行终态。

## 5. Dispatcher 与副作用边界

Dispatcher 先检查 ParseResult、definition 是否仍在 Registry 中以及权限，然后按 `route_target.kind` 分发：

| Target | 当前行为 | 副作用所有者 |
|:---|:---|:---|
| `local_handler` | 调用 Gateway 进程内 handler，可同步或异步 | 注册的 handler |
| `global_route` | 通过 `GlobalSystemBus.request()` 调用公开 route | route 所有子系统 |
| `client_action` | 返回结构化 `client_action`，服务端不直接操作客户端 | transport / client |
| `future_job` | 返回 `NOT_IMPLEMENTED` | 尚未接入 Runtime Job Queue |

`global_route` 会传递 `command`、`identity`、解析参数和 definition payload；若 route 返回的不是 `CommandExecutionResult`，Dispatcher 将响应包进成功结果。route 或 handler 抛出的异常会被捕获并转为 `FAILED`，保留 `command.failed` error code。

`client_action` 尤其容易被误解：例如 `/clear` 只返回 `{type: clear_chat}`，Gateway 不清空 Patchouli 话题、服务端历史或长期记忆。客户端是否以及如何执行动作由 transport/client 契约决定。

## 6. 权限与确认

当前权限模型支持：

- `visibility`: `public`、`debug`、`admin`；
- `allowed_user_ids` 与 `allowed_agent_ids`；
- `requires_confirmation`；
- `destructive`。

用户或 Agent allowlist 不匹配时直接 `REJECTED`；debug 命令要求 runtime 开启 debug；admin 命令默认拒绝，除非身份命中至少一个显式 allowlist。标记为 destructive 或 requires_confirmation 的命令不会执行 target，而是返回 `REQUIRES_CONFIRMATION`。

Gateway 在这里只做基于 `Identity` 的授权判断，不负责认证该 Identity 的真实性。Transport 必须在进入 System 应用层之前建立可信身份，不能让客户端随意声明 user/agent ID。

当前尚没有确认 token、二次提交或确认过期协议，因此 `REQUIRES_CONFIRMATION` 只是安全终态，不代表用户可以沿同一命令链继续执行。这是有意选择的 fail-closed 行为，也是未来实现 destructive command 前必须补齐的契约。

## 7. 当前内置命令

| 命令 | 别名 | Target | 当前结果 |
|:---|:---|:---|:---|
| `/help` | `/start` | `local_handler: system.help` | 文本与结构化可见命令列表 |
| `/commands` | 无 | `local_handler: system.commands` | 结构化可见命令列表 |
| `/clear` | `/reset`、`/restart` | `client_action: clear_chat` | 请求客户端清空当前聊天状态 |
| `/status` | 无 | `local_handler: runtime.status` | Gateway 与命令可见性的最小进程摘要 |

`/status` 当前不是完整系统健康检查，也不包含 Patchouli/Alice 的深度状态。`/help` 和 `/commands` 是否暴露 listing 受 `expose_listing` 控制。

## 8. 结果与错误语义

所有命令都返回稳定 `CommandExecutionResult`：`command_id`、status、面向用户的 message、结构化 data、可选 client action 和稳定 error code。

机器消费者应判断 status、error code 和 client action，不能解析本地化 message。当前内置命令文案仍主要是中文硬编码，尚未完全接入 [System i18n](../system/i18n.md)；这不改变结构化字段语义。

Dispatcher 把可预期拒绝表示为结果而非异常，因为“未知命令”“无权限”“尚未实现”都是正常控制终态。相反，Gateway workflow 的状态不变量、取消和无法形成终态的 deadline 仍通过异常传播。

## 9. 当前限制

- Registry 与动态定义是进程内状态，重启后不会恢复，也没有配置热重载；
- `RuleInterceptor.add_system_command()` 只能注册隐藏的 `future_job` 目标，而该目标当前不执行；
- 没有命令审计存储、频率限制、跨进程幂等或后台 Job 生命周期；
- confirmation 只有拒绝执行的第一阶段，没有确认回路；
- 最小 schema 不提供完整类型转换、互斥参数、嵌套校验或未知字段拒绝；
- 身份认证和客户端动作执行都在 Gateway 边界之外；
- command handler 失败会变成结构化 `FAILED`，但不会自动补偿已经发生的外部副作用。

## 10. 设计矛盾检查

新增或修改命令时检查：

1. 该输入是否真的是系统控制消息，而不是应由普通 chat 处理的业务意图？
2. Registry、Parser 或 Interceptor 是否开始执行副作用？
3. 未知 slash 输入是否仍会短路，避免进入 LLM？
4. `PASSIVE_MEMORY` 是否仍然无法触发命令？
5. destructive command 是否在没有确认协议时被直接执行？
6. `client_action` 是否被误写成服务端已经完成的状态变更？
7. handler 是否把未认证的 Identity 当作可信权限来源？
8. 新 target 是否绕过 GlobalSystemBus 直接持有其他子系统 Runtime？
9. 调用方是否依赖 message 文本，而不是 status/error code？
10. 新命令是否声称由 `future_job` 执行，但 Runtime Job Queue 尚未存在？

## 11. 验证入口

- `tests/unit/engines/gateway/test_interceptors.py`
- `tests/unit/gateway/test_phase3b_contracts.py`
- `tests/unit/gateway/test_phase3c_workflow.py`
- `tests/unit/system/application/test_gateway_chat_flow.py`
- `tests/unit/server/routers/test_chat.py`
