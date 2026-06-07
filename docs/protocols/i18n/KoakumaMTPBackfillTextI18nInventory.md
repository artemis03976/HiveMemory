# KoakumaRuntime MTP 回填文本 i18n 清单

**文档状态**: Implemented  
**适用范围**: `agent_runtime/mtp/runtime.py`、`core/mtp/`、`agent_runtime/mtp/syscalls/`、`agent_runtime/loop_executor.py`、`alice/runtime/orchestrator.py`、`engines/memory_compiler/handlers/mtp.py`  
**相关文档**: [MTPErrorStructureDesign.md](../MTPErrorStructureDesign.md)、[I18nFoundationDesign.md](./I18nFoundationDesign.md)、[MemoryCompilerI18nMigrationPlan.md](./MemoryCompilerI18nMigrationPlan.md)

本文记录 KoakumaRuntime 及其外围链路中会回填给 Agent 的 MTP 文本，并说明这些文本当前如何被 i18n 模板、结构化错误、结构化 warning 与 formatter 统一管理。

---

## 1. 范围

本文只关注**会进入 Agent 运行上下文的 MTP 回填文本**。典型出口包括：

```text
[System MTP Execution Result]
<mtp_response status="...">
...
</mtp_response>
```

以及子代理调用返回：

```text
[System MTP Call Response]
<mtp_response status="..." type="call_response">
...
</mtp_response>
```

以下内容不纳入自然语言 i18n：

| 对象 | 示例 | 说明 |
| :--- | :--- | :--- |
| MTP 定界符 | `⟪`、`⟫` | 协议语法 |
| MTP 动词 | `SEARCH`、`READ`、`RUN`、`WRITE`、`UPDATE`、`CALL` | 协议命令 |
| XML 标签与属性 | `<mtp_response>`、`status`、`type` | Agent 与工具链依赖的结构 |
| status 值 | `success`、`error`、`ack`、`suspend` | 机器可消费枚举 |
| alias / pending alias | `fact_xxx`、`draft_xxx`、`rev_xxx` | 数据标识符 |
| 业务内容本体 | 记忆正文、工具 stdout、文件内容、URL | 用户或外部系统数据，不翻译 |

---

## 2. 当前回填链路

### 2.1 普通 MTP 执行

```text
KoakumaRuntime.execute_mtp()
  -> MTPParser.parse()
  -> KoakumaRuntime._route_and_execute()
  -> handler 返回 MTPResponse
  -> MTPFormatter.format_response()
  -> MTPExecutionResult.formatted_response
```

当前约束：

- `MTPResponse.content` 只承载成功业务内容。
- `MTPResponse.error` 承载结构化 `MTPErrorInfo`。
- `MTPResponse.warnings` 承载结构化 `list[MTPWarningInfo]`。
- `MTPExecutionResult.response_content == MTPResponse.content`。
- Agent 可见完整回填以 `formatted_response` 为准。

### 2.2 CALL / 子代理返回

`CALL` 指令成功解析后返回 `MTPCallRequest`，由编排层执行子代理。子代理结果不再伪装成普通 Koakuma 执行结果，而是构造 `MTPCallResponse`，再由 `MTPFormatter.format_call_response()` 渲染。

```text
AgentOrchestrator
  -> MTPCallResponse(status, agent_alias, reply, artifact_aliases, error)
  -> MTPFormatter.format_call_response()
  -> working_history / TurnEvent.content
```

旧命名 `ipc_return` / `system_ipc_return` 已移除，统一使用 `call_response` / `system_call_response`。

---

## 3. 模板模块划分

| 模块 | 职责 |
| :--- | :--- |
| `hivememory.i18n.mtp_runtime` | MTP 参数错误、权限错误、ACK、warning、formatter 标题、CALL response 标签 |
| `hivememory.i18n.syscall_runtime` | syscall 错误与成功/信息提示，key 使用 `syscall.*` namespace |
| `hivememory.i18n.memory_compiler` | MemoryAtom、PendingAtom、ResolveResult、READ 记忆对象渲染 |
| `hivememory.i18n.prompts` | MTP 教学 prompt 与系统 prompt |

边界原则：

- 执行语义、错误语义、warning 语义归 `mtp_runtime`。
- syscall handler 生成的系统提示归 `syscall_runtime`。
- 记忆对象如何展示归 `memory_compiler`。
- 协议教学和角色提示归 `prompts`。

---

## 4. Phase A：MTP 异常与 Koakuma 核心错误

**状态**: 已完成。

已迁移对象：

| 类型 | 当前承载方式 |
| :--- | :--- |
| 权限错误 | `PermissionDeniedError(message_key="mtp.permission.*")` |
| parse / unknown verb | `MTPParseError` 或 `InvalidArgumentError` + `message_key` |
| SEARCH 参数错误 | `mtp.search.missing_query` |
| READ 参数与 alias 错误 | `mtp.read.*` |
| RUN 参数、alias、类型错误 | `mtp.run.*` |
| WRITE / UPDATE 参数错误 | `mtp.write.*`、`mtp.update.*` |
| CALL 参数与权限错误 | `mtp.call.*`、`mtp.permission.call_depth_exceeded` |
| unexpected exception | `SystemFault` -> `mtp.system.unexpected` |
| WRITE / UPDATE ACK | `mtp.write.ack`、`mtp.update.ack` info 模板 |

当前实现要点：

- handler 不再手写错误响应文本。
- `_route_and_execute()` 是 MTP 语义异常到 `MTPResponse(error=...)` 的统一转换点。
- error 响应的 `content` 为空，文本只由 formatter 渲染。
- `cause` 仅供调试，`MTPErrorInfo.cause` 设置了 `exclude=True`，不会回填给 Agent。

---

## 5. Phase B：Parser Error 与 Filter Warnings

**状态**: 已完成。

parse error 使用结构化 `MTPParseError`：

| key | 用途 |
| :--- | :--- |
| `mtp.parse.no_command` | 未找到 MTP 指令 |
| `mtp.parse.unknown_verb` | 未知动词 |
| `mtp.parse.missing_separator` | 缺少 `|` 分隔符 |

filter warning 使用 `MTPWarningInfo`：

| key | 用途 |
| :--- | :--- |
| `mtp.filter.token_ignored` | 无法识别的 filter token |
| `mtp.filter.unknown_type` | 未知 memory type |
| `mtp.filter.confidence_out_of_range` | confidence 超出范围 |
| `mtp.filter.unknown_key` | 未知 filter key |
| `mtp.filter.parse_failed` | filter 解析失败，降级为宽搜索 |

当前约束：

- `MTPFilterParser.parse()` 不接收 language。
- parser 返回结构化 warning，不直接渲染文本。
- warning 最终由 `MTPFormatter._format_warnings()` 调用 `get_mtp_warning_text()` 渲染。

---

## 6. Phase C：Formatter、Loop 与 CALL Response

**状态**: 已完成。

### 6.1 MTPFormatter 原生渲染

`MTPFormatter.format_response(response, language=None)` 负责普通 MTP 回填：

```xml
<mtp_response status="success">
...业务内容...
<warnings>
<warning>...</warning>
</warnings>
</mtp_response>
```

错误响应：

```xml
<mtp_response status="error">
<error code="mtp.argument.invalid" severity="agent_fault">
[Invalid Argument] ...
</error>
</mtp_response>
```

### 6.2 LoopExecutor 回填

LoopExecutor 不再格式化原始 command，也不再把原始 MTP 指令拼回 user 消息。Agent 已经在上一轮输出过 MTP 指令，系统回填只需要 formatter 生成的执行结果。

### 6.3 CALL Response

CALL response 已结构化为：

```python
MTPCallResponse(
    status=...,
    agent_alias=...,
    reply=...,
    artifact_aliases=[...],
    error=...,
)
```

formatter 使用以下 info key：

| key | 用途 |
| :--- | :--- |
| `mtp.call_response.title` | `[System MTP Call Response]` |
| `mtp.call_response.reply_label` | 子代理回复标签 |
| `mtp.call_response.artifacts_label` | 产物列表标签 |
| `mtp.call_response.artifact_state` | pending alias 状态说明 |
| `mtp.call_response.sub_agent_error` | 子代理失败错误文本 |

---

## 7. Phase D：Syscall 文本

**状态**: 已完成。

syscall 失败路径已从 `SyscallResult(ok=False, content=..., error_code=...)` 改为 handler 直接抛结构化 `MTPError` 子类。`_handle_run()` 不再解析 `error_code`，只透传异常到 `_route_and_execute()`。

当前 syscall 异常类：

| 异常类 | 继承关系 | code |
| :--- | :--- | :--- |
| `SyscallInvalidArgumentError` | `InvalidArgumentError` | `mtp.syscall.invalid_argument` |
| `SyscallPermissionDeniedError` | `PermissionDeniedError` | `mtp.syscall.permission_denied` |
| `SyscallExecutionError` | `SyscallInternalError` | `mtp.syscall.execution_error` |
| `SyscallTimeoutError` | `SyscallInternalError` | `mtp.syscall.timeout` |
| `SyscallUnavailableError` | `SystemFault` | `mtp.syscall.unavailable` |

syscall i18n 使用独立 namespace：

```text
syscall.file_read.missing_path
syscall.file_read.path_denied
syscall.file_write.success
syscall.repl.stdout
syscall.repl.no_output
syscall.web_search.result_item
syscall.web_search.field_empty
```

成功/信息文本通过 `get_syscall_info_text()` 生成。数据本体不翻译，例如文件内容、stdout 值、URL、搜索结果标题与摘要值本身保持原样。

---

## 8. Phase E：MemoryCompiler 残留字段

**状态**: 已完成。

MTP READ 中的正式记忆、pending atom、redirect / terminal pending 结果均由 MemoryCompiler 渲染。以下字段已从 handler 拼接迁入 `i18n.memory_compiler` 模板：

| 字段 | 当前处理 |
| :--- | :--- |
| `canonical alias` / 正式名称 | `pending_read_settled` 模板 |
| `title` / 标题 | `pending_read_write` 模板 |
| `instruction` / 指令 | `pending_read_update` 模板 |
| `message` / 消息 | discarded / failed 模板 |
| `reason` / 原因 | discarded / failed 模板 |
| 空字段占位 | `memory_field_empty`，中文 `无`，英文 `None` |

RUN redirect notice 已从 MemoryCompiler 的 `MTP_REDIRECT_NOTICE` 目标移除，改为 `MTPWarningInfo(message_key="mtp.run.alias_redirected")`，由 formatter 统一渲染。

---

## 9. 当前验收口径

完成状态的判定标准：

- `src` 中不再使用 `MTPError.to_agent_prompt()` 作为 Agent 回填路径。
- error 响应不再向 `MTPResponse.content` 注入错误文本。
- warning 不再拼入 `content`，而是进入 `MTPResponse.warnings`。
- `MTPResponse.content` 只保留成功业务内容。
- `MTPExecutionResult.response_content` 不代表完整 Agent 回填；完整回填看 `formatted_response`。
- syscall 错误不再依赖 `SyscallResult.ok`、`error_code` 或 `"Error:"` 字符串嗅探。
- CALL response 不再使用 IPC 命名和手写文本组装。
- MemoryCompiler handler 不再散拼 READ 可见字段标签。

---

## 10. 后续可选优化

当前计划内项目已经完成。后续如果要继续提高一致性，可以考虑：

1. 将 syscall error 的 `detail` 参数进一步结构化，避免少量人为英文说明作为参数插入中文模板。
2. 为 warnings 扩展更通用的 `info / warning / error` 三层消息模型。
3. 为 formatter 输出增加 XML escaping 策略，明确内容与协议标签之间的边界。
4. 根据 Agent 行为评估是否继续保留英文 category token，例如 `[Invalid Argument]`、`[Tool Error]`。
