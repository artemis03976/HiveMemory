# KoakumaRuntime MTP 回填文本 i18n 清单

**文档状态**: Draft  
**适用范围**: `agent_runtime/mtp/runtime.py`、`core/mtp/`、`agent_runtime/mtp/syscalls/`、`agent_runtime/loop_executor.py`、`alice/runtime/orchestrator.py`、`engines/memory_compiler/handlers/mtp.py`  
**核心目标**: 梳理 KoakumaRuntime 及其外围链路中会被回填到 Agent 上下文的 MTP 文本，识别 i18n 规范化的潜在对象，并给出分批迁移建议。

---

## 1. 范围界定

本文只关注**会进入 Agent 运行上下文的 MTP 回填文本**。这类文本通常最终出现在：

```text
<mtp_response status="...">
...
</mtp_response>
```

或由运行循环包装后注入 history：

```text
[System MTP Execution Result]
...
```

因此，本清单不把普通开发日志、代码注释、测试断言、文档说明作为 i18n 迁移对象。  
但如果某个错误字符串、提示语、ACK、syscall 结果会成为 `MTPResponse.content`、`formatted_response` 或 CALL response 内容，则纳入本文范围。

---

## 2. 回填链路概览

### 2.1 Koakuma 主链路

核心链路位于 `src/hivememory/agent_runtime/mtp/runtime.py`：

```text
execute_mtp()
  -> MTPParser.complete_and_parse()
  -> _route_and_execute()
  -> handler 返回 MTPResponse(content=...)
  -> MTPFormatter.format_response()
  -> MTPExecutionResult.formatted_response
```

格式化容器位于 `src/hivememory/core/mtp/formatter.py`：

```text
[System MTP Execution Result]
<mtp_response status="..." time="...ms">
{response.content}
</mtp_response>
```

### 2.2 Agent history 注入

普通 MTP 执行结果由 `src/hivememory/agent_runtime/loop_executor.py` 注入：

```text
[System MTP Execution Result]
{mtp_result.formatted_response}
```

CALL / 子代理返回由 `src/hivememory/alice/runtime/orchestrator.py` 构造 `MTPCallResponse`，再交给 `MTPFormatter.format_call_response()` 注入：

```text
[System MTP Call Response]
<mtp_response status="..." type="call_response">
...
</mtp_response>
```

---

## 3. 不建议翻译的协议结构

以下内容属于协议 token、结构化字段或机器可消费值，应保持稳定，不作为自然语言 i18n 文案处理：

| 对象 | 示例 | 说明 |
| :--- | :--- | :--- |
| MTP delimiter | `⟪`、`⟫` | 协议语法 |
| MTP verbs | `SEARCH`、`READ`、`RUN`、`WRITE`、`UPDATE`、`CALL` | 协议命令 |
| XML tag | `<mtp_response>`、`</mtp_response>` | Agent 教学和解析依赖 |
| XML attrs | `status`、`time`、`type` | 结构化字段 |
| status values | `success`、`error`、`ack`、`suspend` | 协议状态枚举 |
| CALL suspend request fields | `target_alias`、`task`、`context_refs` | `MTPCallRequest` 结构字段 |
| CALL response fields | `agent_alias`、`reply`、`artifact_aliases`、`error` | `MTPCallResponse` 结构字段 |
| alias / uuid / pending alias | `fact_xxx`、`draft_xxx`、`rev_xxx` | 数据标识符 |
| memory type values | `CODE_SNIPPET`、`FACT` 等 | 枚举值 |

迁移时应只替换自然语言说明、错误解释、Action 建议和字段标签，不改变上述结构。

---

## 4. KoakumaRuntime 内部文本清单

### 4.1 通用权限与异常回填

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `agent_runtime/mtp/runtime.py` | `_check_verb_permission()` | 权限错误 | `You do not have permission to use the '{verb}' command.` | P0 |
| `agent_runtime/mtp/runtime.py` | `_check_tool_permission()` | 工具权限错误 | `You do not have access to tool '{tool_alias}'.` | P0 |
| `agent_runtime/mtp/runtime.py` | `_route_and_execute()` unknown handler | 语法错误 | `[Syntax Error] Unknown verb... Valid verbs...` | P0 |
| `agent_runtime/mtp/runtime.py` | `_route_and_execute()` unexpected exception | 内部错误 | `[Internal Error] An unexpected error occurred. Do NOT retry...` | P0 |

建议：这些文本应迁入新的 `hivememory.i18n.mtp_runtime`，并通过 key 访问，例如：

```text
permission_verb_denied
permission_tool_denied
unknown_verb
internal_error_no_retry
```

### 4.2 SEARCH 回填

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `agent_runtime/mtp/runtime.py` | `_handle_search()` | 参数错误 | `[Invalid Argument] SEARCH requires a "query" argument.` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_search()` | 参数修复建议 | `Action: Provide a query argument and retry.` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_search()` | 空结果 | `No memories found. Try a different query.` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_search()` | 渲染缺失兜底 | `Search completed, but no rendered context was returned.` | P1 |

备注：SEARCH 命中结果主体通常来自 retrieval renderer / memory compiler，已有部分 i18n；本节只关注 Koakuma 自己拼装的错误、空状态与兜底文本。

### 4.3 READ 回填

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `agent_runtime/mtp/runtime.py` | `_handle_read()` wildcard | 参数错误 | `READ does not support wildcard target '*'. Use SEARCH instead.` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_read()` empty target | 参数错误 | `READ requires at least one target alias.` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_read()` all unresolved | alias 错误 | `[Alias Not Found] Alias '{a}' not found. Use SEARCH...` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_read()` partial unresolved | alias 错误 | `[{alias}]: [Alias Not Found]...` | P0 |

备注：READ 对已解析 memory / pending / redirect / terminal pending 的主体渲染已经主要进入 MemoryCompiler。Koakuma 仍保留 alias not found 和参数错误。

### 4.4 RUN 回填

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `agent_runtime/mtp/runtime.py` | `_handle_run()` empty/multiple target | 参数错误 | `RUN requires a single tool alias as target.` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_run()` missing kernel tool | alias 错误 | `[Alias Not Found] Kernel tool '{alias}' not found. Use SEARCH...` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_run()` syscall exception | 工具错误 | `[Tool Error] Tool '{alias}' execution failed. Do NOT retry...` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_run()` pending alias | 状态错误 | `[Pending Alias Not Runnable] Alias '{alias}' is a runtime pending atom...` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_run()` user tool not found | alias 错误 | `[Alias Not Found] Tool alias '{alias}' not found...` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_run()` type mismatch | 类型错误 | `[Type Mismatch] Alias '{alias}' is not a runnable tool...` | P0 |

备注：RUN 成功内容可能来自 syscall 或用户态工具执行结果，Koakuma 不应翻译工具输出本身；但 Koakuma 生成的错误和策略提示应 i18n。

### 4.5 WRITE 回填

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `agent_runtime/mtp/runtime.py` | `_handle_write()` | 参数错误 | `WRITE requires a "content" argument.` | P0 |
| `agent_runtime/mtp/runtime.py` | `_format_write_ack()` | ACK | `Memory accepted as pending atom '{pending_alias}'.` | P0 |
| `agent_runtime/mtp/runtime.py` | `_format_write_ack()` | ACK 说明 | `It is readable during this run via READ...` | P0 |

建议：WRITE ACK 是最明确的模板化对象，应拆成单一模板并保留 `{pending_alias}` 占位符。

### 4.6 UPDATE 回填

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `agent_runtime/mtp/runtime.py` | `_handle_update()` missing target | 参数错误 | `UPDATE requires a single alias as target.` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_update()` missing instruction | 参数错误 | `UPDATE requires an "instruction" argument.` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_update()` pending alias | 状态错误 | `[Pending Alias Not Updatable] Alias '{alias}' is a runtime pending atom...` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_update()` not found | alias 错误 | `[Alias Not Found] Alias '{alias}' not found...` | P0 |
| `agent_runtime/mtp/runtime.py` | `_format_update_ack()` | ACK | `Memory '{base_alias}' update accepted as pending revision...` | P0 |
| `agent_runtime/mtp/runtime.py` | `_format_update_ack()` | ACK 说明 | `It is readable during this run via READ...` | P0 |

建议：UPDATE ACK 应与 WRITE ACK 放在同一文案族，例如 `write_ack` / `update_ack`。

### 4.7 CALL 回填

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `agent_runtime/mtp/runtime.py` | `_handle_call()` depth denied | 权限错误 | `Sub-agents are not allowed to invoke CALL...` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_call()` missing target | 参数错误 | `CALL requires a single agent alias as target. Example: ...` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_call()` missing task | 参数错误 | `CALL requires a "task" argument. Example: ...` | P0 |
| `agent_runtime/mtp/runtime.py` | `_handle_call()` suspend content | JSON payload | `{"target_alias": "...", "task": "...", "context_refs": [...]}` | 不翻译 |

备注：CALL suspend 成功内容是编排层消费的结构化 JSON，不应翻译字段名或 JSON 结构。CALL 参数错误中的 example 文案可以翻译，但 MTP 命令语法本身保持不变。

---

## 5. core.mtp 文本清单

### 5.1 MTP 异常分类与 Action 建议

| 文件 | 对象 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `core/mtp/exceptions.py` | `MTPError.to_agent_prompt()` | 格式模板 | `[{category}] {message}\nAction: {suggestion}` | P0 |
| `core/mtp/exceptions.py` | `MTPParseError` | category / suggestion | `Syntax Error`、`Check your MTP command syntax...` | P0 |
| `core/mtp/exceptions.py` | `AliasNotFoundError` | category / suggestion | `Alias Not Found`、`Use SEARCH...` | P0 |
| `core/mtp/exceptions.py` | `MemoryNotFoundError` | category / suggestion | `Memory Not Found`、`The memory may have been archived...` | P0 |
| `core/mtp/exceptions.py` | `MemoryTypeMismatchError` | category / suggestion | `Type Mismatch`、`RUN only supports CODE_SNIPPET...` | P0 |
| `core/mtp/exceptions.py` | `InvalidArgumentError` | category / suggestion | `Invalid Argument`、`Check the required arguments...` | P0 |
| `core/mtp/exceptions.py` | `PermissionDeniedError` | category / suggestion | `Permission Denied`、`This operation is not allowed...` | P0 |
| `core/mtp/exceptions.py` | `SystemFault` | category / suggestion | `System Error`、`Do NOT retry...` | P0 |
| `core/mtp/exceptions.py` | storage / bus / syscall errors | category / suggestion | `Storage Offline`、`Service Unavailable`、`Tool Error` | P0 |

建议：`to_agent_prompt()` 应能接收或解析 language。异常内部可继续保存机器稳定的 `category_key`，展示文本从 i18n 获取。

### 5.2 Parser 解析错误与 filter warnings

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `core/mtp/parser.py` | `parse()` no command | 语法错误 | `No MTP command found. Expected '⟪...⟫'` | P0 |
| `core/mtp/parser.py` | `parse()` unknown verb | 语法错误 | `Unknown verb '{verb}'. Valid verbs: ...` | P0 |
| `core/mtp/parser.py` | `_split_segments()` | 语法错误 | `Missing separator '|' in MTP command` | P0 |
| `core/mtp/parser.py` | `MTPFilterParser.parse()` | filter warning | `Filter token '{token}' was ignored...` | P1 |
| `core/mtp/parser.py` | `MTPFilterParser.parse()` | filter warning | `Unknown filter type '{value}' was ignored...` | P1 |
| `core/mtp/parser.py` | `MTPFilterParser.parse()` | filter warning | `Filter confidence value ... is out of range...` | P1 |
| `core/mtp/parser.py` | `MTPFilterParser.parse()` | filter warning | `Unknown filter key '{key}' was ignored.` | P1 |
| `core/mtp/parser.py` | parse exception fallback | filter warning | `Filter parsing failed. Results may be broader than expected.` | P1 |

备注：filter warnings 会追加到 SEARCH 响应内容中，因此属于回填文本。

### 5.3 Formatter

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `core/mtp/formatter.py` | `format_response()` | 普通 MTP 回填标题 + XML 容器 | `[System MTP Execution Result]` + `<mtp_response status="...">` | 标题已模板化 |
| `core/mtp/formatter.py` | `format_call_response()` | CALL response 标题 + XML 容器 | `[System MTP Call Response]` + `<mtp_response status="..." type="call_response">` | 标题与标签已模板化 |

备注：formatter 中 XML tag、属性名、status 值和 `type="call_response"` 仍属于协议结构，不翻译；标题、reply label、artifact label 与 pending 状态说明已进入 `mtp_runtime` 的 `_INFO_TEXT`。

---

## 6. Syscall 回填文本清单

RUN 内核工具的返回值会原样成为 `MTPResponse.content`。其中 clock 多为数据输出，其余 syscall 有大量面向 Agent 的英文提示。

### 6.1 Python REPL

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `agent_runtime/mtp/syscalls/repl.py` | `_blocked_import()` | 沙箱错误 | `import is not allowed in the restricted REPL...` | P1 |
| `agent_runtime/mtp/syscalls/repl.py` | timeout | 执行错误 | `Error: Execution timed out after {timeout_seconds}s.` | P1 |
| `agent_runtime/mtp/syscalls/repl.py` | import error | 执行错误 | `Error: {e}` | P1 |
| `agent_runtime/mtp/syscalls/repl.py` | generic exception | 执行错误 | `Error: Python execution failed...` | P1 |
| `agent_runtime/mtp/syscalls/repl.py` | success with stdout | 成功标签 | `Stdout: {output}` | P2 |
| `agent_runtime/mtp/syscalls/repl.py` | success no output | 成功提示 | `Executed successfully (no output).` | P1 |
| `agent_runtime/mtp/syscalls/repl.py` | missing code | 参数错误 | `Error: 'code' argument is required.` | P1 |

### 6.2 File I/O

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `agent_runtime/mtp/syscalls/file_io.py` | safe path | 权限错误 | `Access denied: path '{path}' escapes workspace boundary.` | P1 |
| `agent_runtime/mtp/syscalls/file_io.py` | read missing path | 参数错误 | `Error: 'path' argument is required.` | P1 |
| `agent_runtime/mtp/syscalls/file_io.py` | read not found | 文件错误 | `Error: File not found: '{path}'` | P1 |
| `agent_runtime/mtp/syscalls/file_io.py` | read not file | 文件错误 | `Error: '{path}' is not a file.` | P1 |
| `agent_runtime/mtp/syscalls/file_io.py` | binary file | 文件错误 | `Error: '{path}' appears to be a binary file.` | P1 |
| `agent_runtime/mtp/syscalls/file_io.py` | read OSError | 文件错误 | `Error: Cannot read file...` | P1 |
| `agent_runtime/mtp/syscalls/file_io.py` | read success wrapper | 内容容器 | `<content>...</content>` | 不翻译 |
| `agent_runtime/mtp/syscalls/file_io.py` | truncation notice | 截断提示 | `[Truncated: showing first ... bytes...]` | P1 |
| `agent_runtime/mtp/syscalls/file_io.py` | write missing content | 参数错误 | `Error: 'content' argument is required.` | P1 |
| `agent_runtime/mtp/syscalls/file_io.py` | write invalid mode | 参数错误 | `Error: Invalid mode '{mode}'. Use 'overwrite' or 'append'.` | P1 |
| `agent_runtime/mtp/syscalls/file_io.py` | write too large | 参数错误 | `Error: Content too large... Maximum allowed...` | P1 |
| `agent_runtime/mtp/syscalls/file_io.py` | write OSError | 文件错误 | `Error: Cannot write file...` | P1 |
| `agent_runtime/mtp/syscalls/file_io.py` | write success | 成功提示 | `Success: File '{name}' saved...` | P1 |

### 6.3 Web Search

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `agent_runtime/mtp/syscalls/web_search.py` | missing query | 参数错误 | `Error: 'query' argument is required.` | P1 |
| `agent_runtime/mtp/syscalls/web_search.py` | dependency missing | 工具不可用 | `Error: Web search is not available...` | P1 |
| `agent_runtime/mtp/syscalls/web_search.py` | search exception | 工具错误 | `Error: Web search failed...` | P1 |
| `agent_runtime/mtp/syscalls/web_search.py` | empty result | 空结果 | `No results found for query: '{query}'` | P1 |
| `agent_runtime/mtp/syscalls/web_search.py` | result labels | 字段标签 | `Title:`、`Snippet:`、`URL:` | P2 |
| `agent_runtime/mtp/syscalls/web_search.py` | missing field fallback | 字段兜底 | `N/A` | P2 |

### 6.4 Clock

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `agent_runtime/mtp/syscalls/clock.py` | default output | 时间数据 | `YYYY-MM-DD HH:MM:SS (UTC+X)` | P3 |
| `agent_runtime/mtp/syscalls/clock.py` | iso/date/time output | 时间数据 | ISO / date / time | 不翻译 |

备注：clock 输出目前更接近数据格式，除非未来支持本地化时间展示，否则无需优先处理。

---

## 7. Agent loop 与 CALL response 包装文本清单

### 7.1 普通 MTP 回填包装

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `core/mtp/formatter.py` | `format_response()` | 系统包装标题 | `[System MTP Execution Result]` | 已完成 |
| `agent_runtime/loop_executor.py` | tool_result TurnEvent | `formatted_response` | XML block | 不翻译 |

备注：普通 MTP 回填标题现在由 `MTPFormatter` 拼接，`loop_executor` 只写入 `formatted_response`。

### 7.2 子代理 CALL response 回填

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `core/mtp/formatter.py` | `format_call_response()` | 系统包装标题 | `[System MTP Call Response]` | 已完成 |
| `core/mtp/formatter.py` | `format_call_response()` | reply label | `[Sub-Agent Reply]:` | 已完成 |
| `core/mtp/formatter.py` | `format_call_response()` | artifact label | `[Artifacts Generated / Updated]:` | 已完成 |
| `core/mtp/formatter.py` | `format_call_response()` | artifact state | `(pending, readable now)` / `(pending, 本次运行可读)` | 已完成 |
| `core/mtp/formatter.py` | `format_call_response()` | XML structure | `<mtp_response status="success" type="call_response">` | 不翻译 |
| `core/mtp/exceptions.py` | `SubAgentExecutionError` | 子代理异常错误文本 | `[Sub-Agent Error]: The sub-agent ... encountered an error...` | 已完成 |

备注：CALL response 不再由 orchestrator 手拼文本。orchestrator 只构造 `MTPCallResponse`，成功时携带 `reply` 与 `artifact_aliases`，失败时携带 `MTPErrorInfo`，最终统一由 formatter 渲染。

---

## 8. MemoryCompiler 相关剩余文本

MTP READ 的正式记忆、pending 记忆、redirect / terminal pending 渲染已经主要下沉到 MemoryCompiler i18n：

- `src/hivememory/i18n/memory_compiler.py`
- `src/hivememory/engines/memory_compiler/handlers/mtp.py`

但 `handlers/mtp.py` 仍存在少量字段标签硬编码：

| 文件 | 位置 / 来源 | 当前文本类型 | 示例 | 优先级 |
| :--- | :--- | :--- | :--- | :--- |
| `engines/memory_compiler/handlers/mtp.py` | settled pending | 字段标签 | `canonical alias: {canonical_alias}` | P1 |
| `engines/memory_compiler/handlers/mtp.py` | pending write | 字段标签 | `title: {title}` | P1 |
| `engines/memory_compiler/handlers/mtp.py` | pending update | 字段标签 | `instruction: {instruction}` | P1 |
| `engines/memory_compiler/handlers/mtp.py` | discarded pending | 字段标签 | `message: ...`、`reason: ...` | P1 |
| `engines/memory_compiler/handlers/mtp.py` | failed pending fallback | 错误兜底 | `Memory generation failed.` | P1 |

建议：这些字段标签仍应放在 `i18n/memory_compiler.py` 的 PendingAtom / ResolveResult 文本族中，而不是放进新的 `mtp_runtime.py`。原因是它们属于 memory target rendering，不属于 Koakuma 执行错误。

---

## 9. 已 i18n 化或不属于本清单主范围的对象

| 对象 | 当前状态 | 说明 |
| :--- | :--- | :--- |
| MTP system prompt | 已迁入 `i18n/prompts.py` | 属于协议教学 prompt，不是 runtime 回填 |
| MemoryAtom full/index/profile 模板 | 已迁入 `i18n/memory_compiler.py` | MTP READ 复用该路径 |
| PendingAtom READ 主体模板 | 大部分已迁入 `i18n/memory_compiler.py` | 仍有字段标签残留，见第 8 节 |
| ResolveResult redirect/terminal 模板 | 已迁入 `i18n/memory_compiler.py` | 仍需关注调用处 language 传递 |
| Retrieval context envelope | 已迁入 `i18n/memory_compiler.py` | SEARCH 命中结果主体通常来自这里 |
| MTPFormatter XML 容器 | 不建议翻译 | 协议结构 |
| CALL suspend JSON payload | 不建议翻译 | 编排层解析依赖 |

---

## 10. 建议的 i18n 模块划分

### 10.1 新增 `hivememory.i18n.mtp_runtime`

建议新增：

```text
src/hivememory/i18n/mtp_runtime.py
```

职责：

1. KoakumaRuntime 参数错误、权限错误、ACK、alias 提示。
2. `core.mtp.exceptions` 的 category / suggestion / `Action:` 模板。
3. `core.mtp.parser` 的解析错误与 filter warnings。
4. loop executor 的 MTP 回填包装标题。
5. syscall 的通用错误、成功提示和字段标签。

可选 getter：

```python
get_mtp_runtime_text(key, language=None)
get_mtp_error_category(key, language=None)
get_mtp_error_suggestion(key, language=None)
get_mtp_syscall_text(key, language=None)
```

### 10.2 继续使用 `hivememory.i18n.memory_compiler`

以下对象继续放在 `memory_compiler.py`：

1. MemoryAtom / PendingAtom / ResolveResult 的 READ 渲染字段。
2. MTP READ response envelope。
3. Shared context injection。
4. Retrieval context 文案。

不要把 memory rendering 字段标签塞入 `mtp_runtime.py`，否则会重新混淆“执行错误文本”和“记忆对象渲染文本”的边界。

---

## 11. 语言传递要求

当前要完成 Koakuma runtime i18n，关键前置不是模板本身，而是语言上下文传递。

建议语言来源优先级：

```text
MTPExecutionContext explicit language
> AgentProfile.language
> HiveMemoryConfig.i18n.default_language
> fallback zh
```

需要补齐的传递点：

1. `MTPExecutionContext` 增加或派生 runtime language。
2. `KoakumaRuntime.execute_mtp()` / `_route_and_execute()` 将 language 传给 handlers。
3. `MTPError.to_agent_prompt()` 支持 `language` 参数，或通过错误对象保存 `language`。
4. `MTPParser` / `MTPFilterParser` 若保持无状态，可在格式化错误时传 language；若要在 parser 内直接产出本地化错误，则 parser 需要可接收 language。
5. Syscall registry 的 handler 签名目前只接收 `args`，若要本地化 syscall 文案，需要引入运行时上下文或轻量包装。

---

## 12. 迁移优先级建议

### Phase A: MTP 异常与 Koakuma 核心错误

目标对象：

1. `core/mtp/exceptions.py`
2. `agent_runtime/mtp/runtime.py` 中 SEARCH / READ / RUN / WRITE / UPDATE / CALL 的参数错误和策略错误。
3. WRITE / UPDATE ACK。

理由：

- 全部直接进入 `<mtp_response>`。
- Agent 行为强依赖这些 Action 建议。
- 当前中英文混杂最明显。

### Phase B: Parser 与 filter warnings

目标对象：

1. `MTPParseError` message。
2. `MTPFilterParser` warnings。

理由：

- SEARCH filter warning 会混入成功响应。
- parse error 的修复建议会影响 Agent 是否重试。

### Phase C: Loop / CALL response 包装文本

目标对象：

1. `[System MTP Execution Result]`
2. `[System MTP Call Response]`
3. `[Sub-Agent Reply]`
4. `[Artifacts Generated / Updated]`
5. sub-agent error return。

理由：

- 这些是注入 Agent history 的系统提示。
- 与 CALL / 多智能体体验强相关。

当前状态：

- 普通 MTP 回填已由 `MTPFormatter.format_response()` 统一拼接标题。
- CALL response 已结构化为 `MTPCallResponse`，并由 `MTPFormatter.format_call_response()` 统一渲染。
- `system_ipc_return` / `ipc_return` 旧命名已替换为 `system_call_response` / `call_response`。
- CALL response 相关标题、reply label、artifact label、artifact state 与 sub-agent error 已进入 `hivememory.i18n.mtp_runtime`。

### Phase D: Syscall 文本

目标对象：

1. REPL 错误与成功提示。
2. File I/O 错误、截断、成功提示。
3. Web search 错误、空结果、字段标签。

理由：

- 范围较广，且 syscall handler 目前没有 language context。
- 需要先明确 handler 签名或包装策略。

### Phase E: MemoryCompiler 残留字段标签

目标对象：

1. `canonical alias:`
2. `title:`
3. `instruction:`
4. `message:`
5. `reason:`
6. `Memory generation failed.`

理由：

- 这些不属于 KoakumaRuntime 执行错误，但会在 MTP READ 中出现。
- 可与 MemoryCompiler i18n 收尾一起处理。

---

## 13. 验收检查清单

迁移完成后，建议至少覆盖以下检查：

- 默认中文配置下，Koakuma 参数错误、权限错误、ACK 和 alias not found 输出中文。
- AgentProfile language 为 `en` 时，同一 MTP 错误输出英文。
- `<mtp_response>` 结构、`status` 值、MTP verb、alias、pending alias 不发生变化。
- `CALL` suspend 成功路径的 JSON payload 字段不被翻译。
- `WRITE` / `UPDATE` ACK 保留 pending alias，可被后续 READ 使用。
- `MTPError.to_agent_prompt()` 的 category 与 Action 建议能随语言切换。
- SEARCH filter warning 能随语言切换，并仍追加在搜索结果后。
- syscall 错误不会破坏 Koakuma 对 `result.startswith("Error")` 的判断；若错误前缀也要本地化，应改为结构化 syscall result，而不是继续依赖字符串前缀。
- 现有 MTP prompt 教学中的错误类别说明与 runtime 实际输出保持一致。

---

## 14. 风险点

### 14.1 错误类别翻译可能影响 Agent 教学一致性

MTP prompt 中已经教学了 `[Syntax Error]`、`[Alias Not Found]` 等英文类别。  
如果 runtime category 改为中文，需要同步更新 MTP prompt 的 error handling 部分，或保留 category token 英文、只翻译说明和 Action。

建议短期保守策略：

```text
[Alias Not Found] 别名不存在...
Action: ...
```

即保留方括号 category token，翻译后续自然语言。

### 14.2 Syscall 目前用字符串判断错误

`KoakumaRuntime._execute_user_tool()` 通过：

```python
is_error = result.startswith("Error")
```

判断用户态工具执行是否失败。  
如果 syscall 或沙箱执行的错误前缀被翻译为中文，会改变行为。

建议在迁移 syscall 文本前，先把 syscall result 结构化为：

```text
SyscallResult(status="error" | "success", content="...")
```

或至少保持错误前缀 `Error:` 不翻译。

### 14.3 Parser 无 language context

`MTPParser` 当前无上下文，直接抛 `MTPParseError(message)`。  
如果 parse 阶段就生成自然语言 message，会遇到 language 传递问题。

建议保守策略：

1. Parser 抛结构化错误 key 和参数。
2. `execute_mtp()` 捕获后按当前 context language 渲染。

### 14.4 MemoryCompiler 与 MTPRuntime 文案边界

READ 渲染文本和 runtime 错误文本都进入 `<mtp_response>`，但来源不同。  
迁移时应保持：

- 记忆对象展示 -> `i18n.memory_compiler`
- 执行错误 / 参数错误 / ACK / syscall -> `i18n.mtp_runtime`

---

## 15. 参考文件

```text
src/hivememory/agent_runtime/mtp/runtime.py
src/hivememory/core/mtp/formatter.py
src/hivememory/core/mtp/exceptions.py
src/hivememory/core/mtp/parser.py
src/hivememory/agent_runtime/mtp/syscalls/repl.py
src/hivememory/agent_runtime/mtp/syscalls/file_io.py
src/hivememory/agent_runtime/mtp/syscalls/web_search.py
src/hivememory/agent_runtime/mtp/syscalls/clock.py
src/hivememory/agent_runtime/loop_executor.py
src/hivememory/alice/runtime/orchestrator.py
src/hivememory/engines/memory_compiler/handlers/mtp.py
src/hivememory/i18n/memory_compiler.py
src/hivememory/i18n/prompts.py
docs/protocols/i18n/I18nStatusAndRoadmap.md
docs/protocols/i18n/I18nFoundationDesign.md
```
