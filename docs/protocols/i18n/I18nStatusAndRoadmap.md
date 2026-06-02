# HiveMemory i18n 现状与路线图

**状态**：核心 i18n 基础设施以及主要面向 Agent 的 prompt 模板已基本完成。  
**最后更新**：2026-05-30

## 当前现状

项目现在在 `src/hivememory/i18n/` 下拥有统一的 i18n 基础设施。
语言解析以 `resolve_language()` 和全局的 `HiveMemoryConfig.i18n.default_language` 为核心。

已完成的部分：

1. 全局语言配置与语言规范化。
2. Agent prompt 的组装现在优先遵循 AgentProfile 的语言设置，其次是全局 i18n 配置。
3. MTP system prompt 模板已集中到 i18n 模块中，不再暴露组件级别的语言配置。
4. Gateway prompt 模板已集中到 i18n 模块中，不再暴露组件级别的 prompt/语言配置。
5. Relay 压缩 prompt 已集中到 i18n 模块中。
6. 用于 passive/write/update 模式的 Generation 提取器 prompt 已集中到 i18n 模块中。
7. 提取器不再接受外部的 `system_prompt` / `user_prompt` 配置覆盖。
8. Memory Compiler 的信封文本（envelope text）、检索渲染的空状态（empty states）、MemoryAtom 的 full/index/profile 模板，以及 TimeFormatter 文本已集中到 i18n 模块中。
9. 遗留的 prompt 兼容性包装器以及废弃的 Gateway/Relay/Generation prompt 路径在不再需要的地方已被清理。

## 当前的 i18n 模块

当前稳定的 i18n 入口：

```text
src/hivememory/i18n/
|-- __init__.py
|-- types.py
|-- resolver.py
|-- memory_compiler.py
|-- prompts.py
`-- time_formatter.py
```

推荐的职责划分：

1. `types.py` 和 `resolver.py`：语言规范化与回退（fallback）规则。
2. `prompts.py`：system prompt、MTP prompt、Gateway、Relay 以及 Generation 模板。
3. `memory_compiler.py`：Memory Compiler 输出的记忆对象及信封文本。
4. `time_formatter.py`：相对时间与过期警告（stale-warning）文本。

## 剩余工作

剩余的 i18n 工作目前集中在三个领域。

### 1. PendingAtom 的 i18n

`src/hivememory/engines/memory_compiler/handlers/pending_atom.py` 仍包含对 Agent 可见的纯英文输出：

1. 待处理草稿（Pending draft）的 READ 渲染。
2. 待处理修订（Pending revision）的 READ 渲染。
3. WRITE / UPDATE 的 ACK 消息。
4. 经过 pending 处理流程的结算重定向（settlement redirect）/未找到（not-found）类消息。
5. 针对无效 PendingAtom 状态的 TypeError 消息。

推荐的实现方式：

1. 在 `src/hivememory/i18n/memory_compiler.py` 中添加专属的 PendingAtom 文本组，与 MemoryAtom 文本分开。
2. 通过类似 `get_pending_atom_text(key, language)` 的 getter 方法暴露出来。
3. 将 `MemoryCompileOptions.language` 传递给 `compile_pending_atom()`。
4. 保持协议 token 不变，例如别名（alias）、状态（status）、`READ`、`WRITE` 和 `UPDATE`。

### 2. 剩余的 Memory Compiler 对象

MemoryAtom 已基本迁移完毕，但非 MemoryAtom 的编译路径仍需审查：

1. `handlers/resolve_result.py` 中的 `ResolveResult` 渲染。
2. `MTP_REDIRECT_NOTICE` 输出。
3. 未找到（Not-found）/ 不可用（unavailable）消息。
4. 任何剩余的将对 Agent 可见的特定对象元数据字段。

推荐的实现方式：

1. 在 i18n 字典中保持对象族（object families）的隔离：
   - 信封文本（envelope text）
   - MemoryAtom 文本
   - PendingAtom 文本
   - ResolveResult 文本
2. 避免将所有对象文本塞入 `_ENVELOPE_TEXT`；信封文本应仅保留包装/分节文本。
3. 为每个会输出对 Agent 可见文本的编译目标添加测试，覆盖中文默认值和英文覆盖值。

### 3. Koakuma MTP 运行时文本

Koakuma 和 MTP 运行时仍包含对 Agent 可见的英文错误和提示文本，主要集中在：

1. `src/hivememory/core/mtp/exceptions.py`
   - 错误分类（error categories）
   - 建议（suggestions）
   - `to_agent_prompt()`
2. `src/hivememory/core/mtp/parser.py`
   - 解析错误，例如缺少定界符或分隔符
3. `src/hivememory/alice/runtime/agent/mtp_executor.py`
   - 命令执行错误消息
4. `src/hivememory/alice/runtime/syscalls/`
   - 文件 I/O、Python REPL 以及网络搜索的错误字符串
5. 被重新注入 Agent 上下文的面向系统的 MTP 结果标签。

推荐的实现方式：

1. 添加专属的 `src/hivememory/i18n/mtp_runtime.py`。
2. 保持协议语法和结构化状态值不变。
3. 使 `MTPError.to_agent_prompt()` 能够接受或解析语言。
4. 将解析后的语言从 Agent 运行时 / profile 传递到 MTP 执行上下文中。
5. 小批量迁移运行时字符串：
   - MTP 异常分类和建议
   - 解析器错误
   - 执行器包装消息
   - 面向用户的 syscall 错误字符串

## 建议的顺序

1. Memory Compiler 中的 PendingAtom。
2. ResolveResult 和剩余的 Memory Compiler 编译对象。
3. MTP 异常分类/建议。
4. MTP 解析器和执行器消息。
5. Syscall 错误字符串。

这一顺序能将 Memory Compiler 的完善与更广泛的 Koakuma 运行时行为分开，降低了改变工具执行语义的风险，同时还能优先移除最显眼的纯英文文本。

## 测试期望

对于每个迁移的领域，添加测试以覆盖：

1. 默认的中文输出。
2. 通过 `MemoryCompileOptions.language` 或运行时语言显式指定的英文输出。
3. 未知语言的回退行为。
4. 稳定的协议 token 和别名。
5. 现有的结构化元数据保持不变。

推荐的回归测试命令：

```powershell
pytest tests\unit\i18n
pytest tests\unit\engines\memory_compiler
pytest tests\unit\alice tests\unit\patchouli\mtp
```

对于 Koakuma 运行时的修改，当相关 live/syscall 测试可用时，也应一并运行。
