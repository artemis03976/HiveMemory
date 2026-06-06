# MTP 错误体系结构化设计

**状态**: Draft  
**范围**: `core/mtp/exceptions.py`、`core/mtp/models.py`、`agent_runtime/mtp/runtime.py`、`agent_runtime/mtp/syscalls/`  
**前置文档**: [I18nStatusAndRoadmap.md](i18n/I18nStatusAndRoadmap.md)、[KoakumaMTPBackfillTextI18nInventory.md](../mod/KoakumaMTPBackfillTextI18nInventory.md)  
**目标**: 在 KoakumaRuntime i18n 化之前，为错误信息提供结构化载体，消除 handler 内联拼装 MTPResponse.content 的畸形模式。

---

## 1. 问题现状

当前 MTP 错误产生路径有四条并行的机制，导致错误信息格式和语义散乱：

1. `raise MTPError 子类` → `to_agent_prompt()` — 规范路径，输出 `[Category] msg\nAction: ...`
2. handler 内联 `MTPResponse(status=ERROR, content="[Invalid Argument] ...")` — 手抄了 category token，绕过异常体系
3. handler 内联 `MTPResponse(status=ERROR, content="WRITE requires ...")` — 连 category 也省略了
4. syscall 返回裸字符串 `"Error: ..."`，由 `result.startswith("Error")` 判断成败

路径 2/3/4 的共同问题：错误信息只能靠字符串承载，i18n 无从下手；同一类错误（如 Alias Not Found）在多个 handler 里各写一遍，category 拼写靠人工保持一致。

---

## 2. 设计目标

1. 引入 `MTPErrorInfo` 作为结构化的错误载体，从 `MTPResponse.content` 中独立出来。
2. 给每个异常类一个稳定的 `code`，作为机器身份标识和 i18n join key。
3. 让 `_route_and_execute` 成为全系统唯一的"异常 → 响应"转换点，handler 只负责 `raise`。
4. 结构化 syscall 返回值，消除字符串嗅探。
5. 本阶段不迁移 i18n 文案，只建立结构；i18n 文案迁移见后续的 `mtp_runtime.py` 计划。

---

## 3. MTPErrorInfo

```python
class MTPErrorSeverity(str, Enum):
    AGENT_FAULT = "agent_fault"    # Agent 侧可修复，允许重试
    SYSTEM_FAULT = "system_fault"  # 系统故障，不可重试

class MTPErrorInfo(BaseModel):
    code: str                           # 稳定的 dotted-path 标识，同时作为 i18n join key
    severity: MTPErrorSeverity          # 决定重试语义，retryable 由此派生
    params: dict[str, Any] = Field(default_factory=dict)  # 用于参数化 i18n 模板
    cause: str | None = None            # 原始异常信息，仅供开发调试，不回填给 Agent
```

`retryable` 不单独存储，由消费方通过 `severity == AGENT_FAULT` 派生。

---

## 4. MTPError 异常体系

### 4.1 基类

```python
class MTPError(Exception):
    code: str = "mtp.error"
    severity: MTPErrorSeverity = MTPErrorSeverity.AGENT_FAULT

    def __init__(
        self,
        message: str,
        *,
        params: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.params = params or {}
        self.cause = cause

    def to_error_info(self) -> MTPErrorInfo:
        return MTPErrorInfo(
            code=self.code,
            severity=self.severity,
            params=self.params,
            cause=str(self.cause) if self.cause else None,
        )

    def to_agent_prompt(self) -> str:
        """过渡兼容方法，i18n 化后替换为从 error_info + language 渲染。"""
        ...
```

### 4.2 code 命名规范

code 使用 dotted-path 格式，同时作为异常身份和 i18n join key：

```
mtp.<domain>.<specific>
```

| 异常类 | code |
| :--- | :--- |
| `MTPParseError` | `mtp.parse.syntax_error` |
| `AliasNotFoundError` | `mtp.alias.not_found` |
| `MemoryNotFoundError` | `mtp.memory.not_found` |
| `MemoryTypeMismatchError` | `mtp.memory.type_mismatch` |
| `InvalidArgumentError` | `mtp.argument.invalid` |
| `PermissionDeniedError` | `mtp.permission.denied` |
| `SystemFault` | `mtp.system.fault` |
| `StorageOfflineError` | `mtp.system.storage_offline` |
| `StorageReadError` | `mtp.system.storage_error` |
| `BusRouteUnavailableError` | `mtp.system.service_unavailable` |
| `SyscallInternalError` | `mtp.system.tool_error` |

code 的点分层级有语义含义但不强制：`mtp.alias.*` 不代表这些异常在代码层级上有父子关系，只是命名空间管理。

**保留字段**：现有的 `category` 和 `suggestion` 字段在过渡期保留，供 `to_agent_prompt()` 继续使用，待 `mtp_runtime.py` i18n 化完成后移除。

---

## 5. MTPResponse 增加 error / warnings 字段

```python
class MTPResponse(BaseModel):
    status: MTPResponseStatus
    content: str = ""
    error: MTPErrorInfo | None = None            # 新增，status=error 时非空
    warnings: list[str] = Field(default_factory=list)  # 新增，nonfatal，不影响 status
    execution_time_ms: float = 0.0
    pending_alias: str | None = None
```

**error 与 warnings 的边界**：

`error` 和 `warnings` 对应两类性质完全不同的信息，不可混用：

| 字段 | 改变 status | 典型场景 |
| :--- | :--- | :--- |
| `error` | 是（→ ERROR） | 参数缺失、alias 不存在、权限拒绝、系统故障 |
| `warnings` | 否（status 不变） | filter token 被忽略、READ partial unresolved 中的缺失 alias |

nonfatal issue 不进入 `MTPError` 体系，handler 直接向 `MTPResponse.warnings` 追加字符串。formatter 统一决定是否把 warnings 追加到输出，handler 不再手工拼接 content。

`warnings` 现阶段为 `list[str]`，保留向 `list[MTPIssue]` 升级的空间，但不预先设计 `MTPIssue` 结构。

**error 过渡期约定**：
- `error` 字段非空时，`content` 仍然填充 `to_agent_prompt()` 的输出，保持 Agent 可见行为不变。
- 两个字段并行携带错误语义，直到 i18n 化完成后由 formatter 统一从 `error` 渲染。
- `content` 的错误信息来源由分散的 handler 内联字符串，收拢为 `MTPError.to_agent_prompt()`。

**MTPExecutionResult 不承载结构化错误**：

`MTPErrorInfo` 的生命周期止于 `MTPResponse`。`loop_executor` 等消费方只需要 `response_status`（字符串）、`response_content`（CALL payload）和 `formatted_response`（history 注入），没有任何消费方需要从 `MTPExecutionResult` 读取结构化错误。`MTPExecutionResult` 不新增 `error` 字段。

---

## 6. 错误收口：两层结构

`execute_mtp()` 中存在两个错误边界，各自负责不同阶段。

### 6.1 parse 边界（execute_mtp）

`MTPParseError` 在 `_route_and_execute` **之前**发生，此时尚无 `MTPCommand` 对象，formatter 也无法调用 `format_command_with_response`，只能调用 `format_response`。这一边界无法合并进 `_route_and_execute`，需单独处理：

```python
async def execute_mtp(self, text: str, context: ...) -> MTPExecutionResult:
    try:
        command = self._parser.complete_and_parse(text)
        response = await self._route_and_execute(command, context)
        ...
    except MTPParseError as e:
        # parse 边界：无 command，单独处理
        error_response = MTPResponse(
            status=MTPResponseStatus.ERROR,
            content=e.to_agent_prompt(),
            error=e.to_error_info(),
        )
        return MTPExecutionResult(
            command=None,
            formatted_response=self._formatter.format_response(error_response),
            ...
        )
```

`MTPParseError` 不应被 `_route_and_execute` 捕获，两者职责边界清晰：parse 边界处理协议语法错误，route 边界处理语义执行错误。

### 6.2 route 边界（_route_and_execute）

所有语义执行错误（参数、权限、alias、系统故障）集中在此转换：

```python
async def _route_and_execute(self, command: MTPCommand, context: ...) -> MTPResponse:
    ...
    try:
        self._check_verb_permission(command.verb.value, context=context)
        return await handler(command, context)

    except MTPError as e:
        _log_by_severity(e, command.verb)
        return MTPResponse(
            status=MTPResponseStatus.ERROR,
            content=e.to_agent_prompt(),
            error=e.to_error_info(),
        )

    except Exception as e:
        fault = SystemFault("An unexpected error occurred. Do NOT retry.", cause=e)
        logger.error(f"Unexpected error during {command.verb}", exc_info=True)
        return MTPResponse(
            status=MTPResponseStatus.ERROR,
            content=fault.to_agent_prompt(),
            error=fault.to_error_info(),
        )
```

`except MTPError` 捕获全部 AgentFault 和 SystemFault 子类，不再需要逐类分支。handler 内的参数错误、alias 错误、权限错误全部改为 `raise`，不再手工构建 `MTPResponse`。

---

## 7. SyscallResult：消除字符串嗅探

当前 `_execute_user_tool` 通过 `result.startswith("Error")` 判断 syscall 成败，syscall 均返回裸字符串。这一机制在 i18n 化后会直接失效（中文前缀不再是 `"Error"`），同时也是 [_handle_run 悬而未决的 TODO](../../src/hivememory/agent_runtime/mtp/runtime.py#L551) 的根因。

引入：

```python
@dataclass
class SyscallResult:
    ok: bool
    content: str
    error_code: str | None = None   # 失败时携带，对应 MTPErrorInfo.code 命名空间
```

`_execute_user_tool` 改为判断 `.ok`：

```python
def _execute_user_tool(self, alias: str, code: str, args: dict) -> MTPResponse:
    result: SyscallResult = execute_sandboxed(...)
    if not result.ok:
        raise SyscallInternalError(
            result.content,
            params={"alias": alias},
        )
    return MTPResponse(status=MTPResponseStatus.SUCCESS, content=result.content)
```

各 syscall handler（repl、file_io、web_search、clock）同步改为返回 `SyscallResult`，不再裸拼 `"Error: ..."` 字符串。

---

## 8. 变更范围总览

| 文件 | 变更类型 | 说明 |
| :--- | :--- | :--- |
| `core/mtp/exceptions.py` | 改造 | 各异常类新增 `code` 类属性；`MTPError.__init__` 新增 `cause`、`params`；新增 `to_error_info()` |
| `core/mtp/models.py` | 新增 | `MTPErrorSeverity`、`MTPErrorInfo` |
| `core/mtp/models.py` | 改造 | `MTPResponse` 新增 `error: MTPErrorInfo | None` |
| `agent_runtime/mtp/runtime.py` | 改造 | `_route_and_execute` 统一捕获 `MTPError`；handler 内联错误改为 `raise` |
| `agent_runtime/mtp/syscalls/` | 改造 | 各 syscall 返回 `SyscallResult` 替代裸字符串；`execute_sandboxed` 返回值同步 |
| `agent_runtime/mtp/runtime.py` | 改造 | `_execute_user_tool` 改判 `result.ok` |

**不在本阶段变更**：
- `to_agent_prompt()` 文本内容（过渡期保留现有英文文案）
- i18n 模板文件（在后续 `mtp_runtime.py` 阶段完成）
- `MTPExecutionContext` 语言传递（在 i18n 阶段建立）
- formatter 从 `error` 字段渲染文本（过渡期结束后清理 `content` 路径）

---

## 9. 过渡期结束条件

以下条件全部满足时，可以移除过渡期兼容逻辑：

1. `mtp_runtime.py` i18n 表建立，覆盖所有 `code` 对应的 category label / suggestion / message 模板。
2. `MTPError.to_agent_prompt()` 替换为从 i18n 表 + language 渲染。
3. formatter 改为从 `MTPResponse.error` 生成错误文本，`content` 在错误时置空。
4. 现有的 `category`、`suggestion` 类属性从异常类中移除。

---

## 10. 与 i18n 路线图的衔接

本文档对应 [I18nStatusAndRoadmap.md §3](../protocols/i18n/I18nStatusAndRoadmap.md#3-koakuma-mtp-运行时文本) 中"Koakuma MTP 运行时文本"的前置工作。

建议执行顺序：

```
本文档（结构化改造）
  → mtp_runtime.py i18n 表建立
  → MTPError.to_agent_prompt() 接入 language
  → MTPExecutionContext 语言传递
  → formatter 切换渲染路径
  → 过渡期字段清理
```

具体文案迁移计划见 [KoakumaMTPBackfillTextI18nInventory.md](../mod/KoakumaMTPBackfillTextI18nInventory.md)。
