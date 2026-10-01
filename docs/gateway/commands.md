---
title: Gateway Commands
status: current
owner: gateway
scope: system-command-registry-and-parsing
code_paths:
  - src/hivememory/gateway/commands/
  - src/hivememory/engines/gateway/interceptors.py
  - src/hivememory/gateway/workflow/state.py
  - src/hivememory/core/protocol/gateway.py
  - src/hivememory/workspace/process/command_terminal.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/error-model.md
related_docs:
  - docs/architecture/workspace.md
  - docs/system/application-services.md
last_reviewed: 2026-10-01
---

# Gateway 全局命令

系统命令是一类控制消息，而不是另一种自然语言意图。它们需要确定性匹配和显式参数；如果把 `/clear`、`/status` 或未知 slash 输入交给 LLM 猜测，不仅结果不稳定，还可能让本应短路的控制消息进入检索、Agent 执行和长期记忆生成。

因此命令识别归属 Gateway 的入口控制能力：Gateway 解析命令，但不执行命令。命令的实际运行不在 Gateway 中——后续指令会与用户请求同时出现，Gateway 不能在入口分析时产生命令副作用。命令在任务进程中何时、由谁运行尚未决定，命令系统整体后置；在此之前，内置命令只解析、暂时不可用。

## 1. 组件分工

```text
CommandDefinition
  -> CommandRegistry
       -> RuleInterceptor.match
            -> CommandParseResult
                 -> GatewayCommandOutcome（只携带解析结果）
                      -> 任务进程：command_terminal -> CommandExecutionResult（命令终态）
```

- `CommandDefinition` 声明名称、别名、参数、路由目标、权限和展示信息；
- `CommandRegistry` 负责注册、冲突检查、列表和匹配，不产生副作用；
- Parser 负责 token 与最小参数 schema，不执行 shell；
- `RuleInterceptor` 只把解析结果放入 Gateway state；
- Gateway workflow 在入口识别出命令后直接以 `GatewayCommandOutcome` 结束，不进入普通 decision flow，也不运行任何执行步骤；
- 任务进程（`workspace/process/command_terminal.py`）把解析结果转换为用户可见的命令终态。

`CommandParseResult` 与 `CommandParseStatus` 位于 `core.protocol.gateway`：它们是 Gateway 公开结果 `GatewayProcessResult` 的组成部分，任务进程只依赖 core 中的模型，不导入 Gateway。Definition 与 ParseResult 都是冻结模型，参数 mapping 会转为不可变结构。

## 2. 入口与短路

只有 `ACTIVE_CHAT` 允许系统命令。L1 收到 slash 输入时，即使命令未知、参数无效或 Registry 未启用，也会形成 SYSTEM 分支并带上相应的解析状态；未知命令不会落入 LLM Query Analysis。

`PASSIVE_MEMORY` 调用 interceptor 时设置 `allow_system=false`，所以外部对话中的 `/help` 或 `/clear` 只被当作普通被动消息分析，不会控制 HiveMemory。这个限制是被动摄入作为“观察者而非参与者”的安全边界；Gateway 执行状态的 `finalize()` 同样拒绝被动模式产生命令结果。

命令请求同样登记为任务进程。命令结果不会继续进入 topic routing、retrieval、Agent 执行、MTP 或 active memory generation；任务进程在 Gateway 阶段之后直接以命令终态结束，进程结局为 `completed`。

## 3. Registry 与匹配规则

Registry 在 Gateway Runtime 装配时创建并注册内置命令；`gateway.commands.builtin` 覆盖表可以按 `command_id` 关闭某个内置命令，被关闭的命令不再注册，输入它时按未知命令处理。注册阶段会拒绝：

- 重复 `command_id`；
- 同一定义内重复的主名称/别名；
- 与已有命令冲突的标准化别名；
- 不以 `/` 开头的命令名称。

名称会去除多余空白并按大小写不敏感方式匹配。多个别名都可能匹配时，优先选择 token 更长的别名，其次选择更小的 `priority`；仍同级时返回 `AMBIGUOUS`，不猜测。`Registry.list()` 按 priority 与主名称排序，并默认隐藏 hidden definition。

## 4. Parser 的确定性边界

命令使用 `shlex.split(..., posix=True)` 切分，因此支持带引号的文本，但不执行变量替换、通配符、管道、重定向或任何 shell expansion。当前参数形式包括：

```text
/command value
/command --key value
/command --key=value
/command --flag
```

位置参数进入 `_positional` 列表；孤立 flag 为布尔 `true`；其余值保持字符串。Parser 只实现最小 schema：检查 required 字段，以及 string、boolean、integer、number、array 的基本类型兼容性。

这不是完整 JSON Schema。当前实现不会统一转换数值类型，也不会拒绝 schema 未声明的额外参数；将来运行命令的一方必须把 ParseResult 视为已经完成最小入口校验，而不是可信业务对象。

解析状态与命令终态必须分开：

- `matched`、`invalid_args`、`unknown`、`ambiguous` 描述解析（`CommandParseStatus`）；
- `completed`、`rejected`、`failed`、`requires_confirmation`、`not_implemented` 描述命令终态（`CommandExecutionStatus`）。

## 5. 命令终态（暂时不可用）

任务进程按解析状态产生命令终态 `CommandExecutionResult`：

| 解析状态 | 命令终态 | error code | 文案 |
|:---|:---|:---|:---|
| `matched` | `not_implemented` | `command.unavailable` | “系统指令 <命令名> 暂不可用。” |
| `unknown`、`invalid_args`、`ambiguous` | `rejected` | `command.parse.<解析状态>` | 解析错误 |

`command_id` 取解析结果中的 `command_id`；解析失败时为空，回退为解析出的命令名。命令终态不携带客户端动作。流式交付仍发出 `command_result` 与 `done`，非流式返回 `NonStreamingCommandOutcome`；`chat.run.completed` 与 `gateway.workflow.completed` 事件都带 `command_id`。

命令不可用是命令自身的终态，不是进程失败，也不会回退到普通 chat。

## 6. 定义中保留的执行字段

`CommandDefinition` 仍声明路由目标（`local_handler`、`global_route`、`client_action`、`future_job`）与权限策略（`visibility`、用户与 Agent allowlist、`requires_confirmation`、`destructive`）。解析不使用这些字段；它们只在命令定义中保留，将来命令系统接回时随运行位置重新设计。因为命令不执行，当前不存在需要按这些字段授权的副作用。

## 7. 当前内置命令

| 命令 | 别名 | 定义中的路由目标 | 当前结果 |
|:---|:---|:---|:---|
| `/help` | `/start` | `local_handler: system.help` | 暂不可用 |
| `/commands` | 无 | `local_handler: system.commands` | 暂不可用 |
| `/clear` | `/reset`、`/restart` | `client_action: clear_chat` | 暂不可用；不返回 `clear_chat` |
| `/status` | 无 | `local_handler: runtime.status` | 暂不可用 |

`/clear` 不再返回 `clear_chat`，前端也就不再因它清空聊天；前端目前没有其他清空入口。

## 8. 结果与错误语义

机器消费者应判断 status、error code 和 client action，不能解析本地化 message。命令终态文案是中文硬编码，尚未接入 [System i18n](../system/i18n.md)；这不改变结构化字段语义。

可预期的解析失败表示为命令终态而非异常，因为“未知命令”“参数无效”“有歧义”都是正常控制终态。相反，Gateway workflow 的状态不变量、取消和无法形成终态的 deadline 仍通过异常传播。

## 9. 当前限制

- 内置命令全部暂时不可用；命令在任务进程中何时、由谁运行尚未决定；
- Registry 与动态定义是进程内状态，重启后不会恢复，也没有配置热重载；
- `RuleInterceptor.add_system_command()` 只能注册隐藏的 `future_job` 目标的定义；
- 最小 schema 不提供完整类型转换、互斥参数、嵌套校验或未知字段拒绝；
- 定义中的权限与确认字段当前没有执行方；destructive 命令接回前必须补齐确认协议。

## 10. 设计矛盾检查

新增或修改命令时检查：

1. 该输入是否真的是系统控制消息，而不是应由普通 chat 处理的业务意图？
2. Registry、Parser、Interceptor 或 Gateway workflow 是否开始执行副作用？
3. 未知 slash 输入是否仍会短路，避免进入 LLM？
4. `PASSIVE_MEMORY` 是否仍然无法产生命令结果？
5. Gateway 的公开结果是否仍只携带解析结果，而不是执行产物？
6. 命令终态是否被当作进程失败，或回退到普通 chat？
7. 调用方是否依赖 message 文本，而不是 status/error code？

## 11. 验证入口

- `tests/unit/gateway/commands/test_registry.py`、`tests/unit/gateway/commands/test_parser.py`
- `tests/unit/engines/gateway/test_interceptors.py`
- `tests/unit/gateway/test_phase3b_contracts.py`
- `tests/unit/gateway/test_phase3c_workflow.py`
- `tests/unit/workspace/process/test_command_terminal.py`
- `tests/unit/workspace/process/test_gateway_chat_flow.py`
- `tests/unit/server/routers/test_chat.py`
