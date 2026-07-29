---
title: Legacy MTP Error Structure Design
status: archived
owner: alice
scope: legacy-mtp-error-design
archived_at: 2026-07-28
superseded_by:
  - docs/contracts/error-model.md
  - docs/contracts/mtp.md
---

> 本文已停止维护，只保留迁移期间的历史参考。当前错误、warning、CALL response 与回填语义见 [`docs/contracts/error-model.md`](../../../contracts/error-model.md) 和 [`docs/contracts/mtp.md`](../../../contracts/mtp.md)。

# MTP 错误与回填消息结构化设计

**文档状态**: Implemented  
**适用范围**: `core/mtp/models.py`、`core/mtp/exceptions.py`、`core/mtp/formatter.py`、`agent_runtime/mtp/runtime.py`、`agent_runtime/mtp/syscalls/`、`alice/runtime/orchestrator.py`  
**相关文档**: [KoakumaMTPBackfillTextI18nInventory.md](./i18n/KoakumaMTPBackfillTextI18nInventory.md)、[MemoryToolProtocol.md](./MemoryToolProtocol.md)

本文描述 MTP 错误、warning、CALL response 与 syscall 结果的当前结构化体系。该体系的目标是让 runtime 只产生结构化结果，由 `MTPFormatter` 统一构造 Agent-facing 回填文本。

---

## 1. 设计目标

MTP 回填文本曾经存在多个并行路径：

- handler 手写 `MTPResponse(content="[Invalid Argument] ...")`。
- 异常对象通过 `to_agent_prompt()` 生成文本，再塞回 `content`。
- warning 作为裸字符串拼入成功内容。
- syscall 通过 `SyscallResult(ok=False, content=..., error_code=...)` 返回错误，再由 `_handle_run()` 解析。
- 子代理返回由 orchestrator 手拼 IPC 文本。

这些路径导致错误文本难以 i18n、格式不稳定、错误语义分散。当前设计统一为：

```text
handler / parser / syscall
  -> raise MTPError 或返回 MTPResponse / MTPCallResponse
  -> MTPErrorInfo / MTPWarningInfo 结构化承载
  -> MTPFormatter 根据 language 渲染 Agent-facing 文本
```

核心原则：

1. `content` 只承载成功业务内容。
2. error 与 warning 不再通过 `content` 注入。
3. formatter 是 Agent-facing MTP 回填文本的唯一构造点。
4. i18n key 由 `message_key` 显式承载，不与错误类别 code 混用。
5. `cause` 只用于开发调试，不回填给 Agent。

---

## 2. 核心模型

### 2.1 MTPErrorSeverity

```python
class MTPErrorSeverity(str, Enum):
    AGENT_FAULT = "agent_fault"
    SYSTEM_FAULT = "system_fault"
```

语义：

- `agent_fault`: Agent 可修正输入后重试。
- `system_fault`: 系统、存储、工具或外部服务故障，不应使用相同输入重试。

当前不单独存储 `retryable` 字段。需要 retry 策略时，应从 `severity` 派生。

### 2.2 MTPErrorInfo

```python
class MTPErrorInfo(BaseModel):
    code: str
    message_key: str = ""
    severity: MTPErrorSeverity
    params: dict[str, Any] = Field(default_factory=dict)
    cause: str | None = Field(default=None, exclude=True)
```

字段职责：

| 字段 | 说明 |
| :--- | :--- |
| `code` | 稳定机器分类，用于追踪、监控、策略判断 |
| `message_key` | i18n 文本 key，用于 formatter 渲染 |
| `severity` | 错误归因与 retry 语义 |
| `params` | i18n 模板参数 |
| `cause` | 原始异常说明，仅调试使用，序列化时排除 |

### 2.3 MTPWarningInfo

```python
class MTPWarningInfo(BaseModel):
    message_key: str
    params: dict[str, Any] = Field(default_factory=dict)
```

warning 表示 nonfatal issue，不改变 `MTPResponse.status`。典型场景：

- SEARCH filter token 被忽略。
- SEARCH 无结果。
- READ 批量读取中部分 alias 未解析。
- RUN alias redirect notice。

### 2.4 MTPResponse

```python
class MTPResponse(BaseModel):
    status: MTPResponseStatus
    content: str = ""
    execution_time_ms: float = 0.0
    pending_alias: str | None = Field(default=None, exclude=True)
    call_request: MTPCallRequest | None = Field(default=None, exclude=True)
    error: MTPErrorInfo | None = None
    warnings: list[MTPWarningInfo] = Field(default_factory=list)
```

约束：

- `status=ERROR` 时，`content` 应为空，`error` 应非空。
- `warnings` 可以与成功 `content` 同时存在。
- `pending_alias` 与 `call_request` 是 runtime 内部结构，不直接序列化进 Agent 回填。

### 2.5 MTPCallResponse

```python
class MTPCallResponse(BaseModel):
    status: MTPResponseStatus
    agent_alias: str
    reply: str = ""
    artifact_aliases: list[str] = Field(default_factory=list)
    error: MTPErrorInfo | None = None
```

`MTPCallResponse` 用于子代理返回，与 `MTPCallRequest` 配对。它不经过 KoakumaRuntime 的普通执行链路，而是由 orchestrator 构造后交给 formatter 渲染。

---

## 3. 异常体系

### 3.1 基类

```python
class MTPError(Exception):
    code: str = "mtp.error"
    default_message_key: str = ""
    severity: MTPErrorSeverity = MTPErrorSeverity.AGENT_FAULT

    def __init__(
        self,
        message: str = "",
        *,
        message_key: str = "",
        params: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        ...

    def to_error_info(self) -> MTPErrorInfo:
        ...
```

`message` 仅作为 Python 异常的调试 fallback。Agent-facing 文本应通过 `message_key + params` 渲染。

旧方法 `to_agent_prompt()` 已移除，不再作为兼容路径。

### 3.2 主要错误类

| 类 | code | severity |
| :--- | :--- | :--- |
| `MTPParseError` | `mtp.parse.syntax_error` | `agent_fault` |
| `AliasNotFoundError` | `mtp.alias.not_found` | `agent_fault` |
| `MemoryNotFoundError` | `mtp.memory.not_found` | `agent_fault` |
| `MemoryTypeMismatchError` | `mtp.memory.type_mismatch` | `agent_fault` |
| `InvalidArgumentError` | `mtp.argument.invalid` | `agent_fault` |
| `PermissionDeniedError` | `mtp.permission.denied` | `agent_fault` |
| `StorageOfflineError` | `mtp.system.storage_offline` | `system_fault` |
| `StorageReadError` | `mtp.system.storage_error` | `system_fault` |
| `BusRouteUnavailableError` | `mtp.system.service_unavailable` | `system_fault` |
| `SystemFault` | `mtp.system.fault` | `system_fault` |
| `SubAgentExecutionError` | `mtp.call_response.sub_agent_error` | `system_fault` |

`code` 表示稳定错误类别；具体文案差异通过 `message_key` 区分。例如 `READ` 与 `RUN` 都可能使用 `InvalidArgumentError`，但分别传入 `mtp.read.missing_alias`、`mtp.run.missing_single_target` 等 key。

### 3.3 syscall 错误类

syscall 使用 MTP 异常体系，但 message key 使用独立 `syscall.*` namespace。

| 类 | 继承 | code | message_key 示例 |
| :--- | :--- | :--- | :--- |
| `SyscallInvalidArgumentError` | `InvalidArgumentError` | `mtp.syscall.invalid_argument` | `syscall.file_read.missing_path` |
| `SyscallPermissionDeniedError` | `PermissionDeniedError` | `mtp.syscall.permission_denied` | `syscall.file_read.path_denied` |
| `SyscallExecutionError` | `SyscallInternalError` | `mtp.syscall.execution_error` | `syscall.web_search.failed` |
| `SyscallTimeoutError` | `SyscallInternalError` | `mtp.syscall.timeout` | `syscall.repl.timeout` |
| `SyscallUnavailableError` | `SystemFault` | `mtp.syscall.unavailable` | `syscall.web_search.unavailable` |

---

## 4. Runtime 转换规则

### 4.1 execute_mtp()

`execute_mtp()` 负责：

1. 解析文本为 `MTPCommand`。
2. 调用 `_route_and_execute()`。
3. 调用 `MTPFormatter.format_response(response, language)`。
4. 构造 `MTPExecutionResult`。

parse error 在 `execute_mtp()` 捕获并转换为：

```python
MTPResponse(
    status=MTPResponseStatus.ERROR,
    content="",
    error=e.to_error_info(),
)
```

### 4.2 _route_and_execute()

`_route_and_execute()` 是普通 MTP 执行链路中唯一的异常转换点：

```python
try:
    return await handler(command, context)
except MTPError as e:
    return MTPResponse(status=ERROR, content="", error=e.to_error_info())
except Exception as e:
    fault = SystemFault(cause=e)
    return MTPResponse(status=ERROR, content="", error=fault.to_error_info())
```

handler 的职责是：

- 成功时返回 `MTPResponse(content=...)`。
- nonfatal issue 写入 `warnings`。
- 错误时抛 `MTPError` 子类，不手写 error response。

### 4.3 _handle_run() 与 syscall

syscall handler 当前只返回成功结果：

```python
class SyscallResult(BaseModel):
    content: str = ""
```

失败时由 syscall handler 自己抛结构化异常。`_handle_run()` 不再解析 `error_code`，也不再依赖 `"Error:"` 前缀判断 syscall 成败。

---

## 5. Formatter 渲染规则

### 5.1 普通响应

```python
MTPFormatter.format_response(response, language=None)
```

成功响应：

```xml
<mtp_response status="success">
...raw content...
</mtp_response>
```

成功但带 warning：

```xml
<mtp_response status="success">
...raw content...
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

渲染规则：

- `error.message_key.startswith("syscall.")` 时调用 `get_syscall_error_text()`。
- 其他错误调用 `get_mtp_error_text()`。
- warning 调用 `get_mtp_warning_text()`。
- 缺失 key 或参数时让 `KeyError` 暴露，避免生成空错误响应。

### 5.2 CALL response

```python
MTPFormatter.format_call_response(call_response, language=None)
```

成功：

```xml
<mtp_response status="success" type="call_response">
[Sub-Agent Reply]:
...

[Artifacts Generated / Updated]:
- draft_xxx (pending, readable now)
</mtp_response>
```

失败：

```xml
<mtp_response status="error" type="call_response">
<error code="mtp.call_response.sub_agent_error" severity="system_fault">
...
</error>
</mtp_response>
```

CALL response 不再使用 IPC 命名。

---

## 6. i18n 语言解析

语言解析统一由 i18n 模块完成：

```text
explicit language
> AgentProfile.language
> i18n config default_language
> fallback zh
```

runtime 与 formatter 只传递更高优先级的 explicit / context language，不在各层反复传递全局 default language。

---

## 7. 当前已移除的旧路径

以下路径已经不再使用：

- `MTPError.to_agent_prompt()`。
- error 文本写入 `MTPResponse.content`。
- warning 裸字符串拼接到 `content`。
- `MTPFormatter.format_command_with_response()`。
- `SyscallResult.ok` / `SyscallResult.error_code`。
- `_handle_run()` 解析 syscall error code。
- `_assemble_ipc_return()` 手拼子代理返回。
- `ipc_return` / `system_ipc_return` 命名。
- `MTP_REDIRECT_NOTICE` / `resolve_redirect_run_notice`。

---

## 8. 验收要点

后续修改 MTP handler、syscall 或 formatter 时，应保持以下约束：

1. 新错误必须提供稳定 `code` 与具体 `message_key`。
2. 不要为了临时方便向 `MTPResponse.content` 写错误文本。
3. warning 必须使用 `MTPWarningInfo`。
4. syscall 错误必须抛结构化 `MTPError` 子类。
5. 子代理返回必须使用 `MTPCallResponse`。
6. Agent-facing 文本必须由 formatter 渲染。
7. 业务数据本体不得被 i18n 模板改写。
