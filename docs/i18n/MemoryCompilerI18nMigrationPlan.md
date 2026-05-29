# Memory Compiler i18n 逐步迁移计划

**状态**: 规划中  
**目标阶段**: 全局语言设置统一后，Memory Compiler 文案迁移第一阶段  
**范围**: Memory Compiler 生成的 Agent 注入文本、MTP 读响应、共享上下文包装文本

## 1. 背景

当前项目已经完成 i18n 基建的第一步：全局语言配置进入
`HiveMemoryConfig.i18n.default_language`，Agent prompt、MTP prompt 与 Gateway
等入口已逐步改为跟随全局语言或 Agent profile language。

下一阶段需要开始迁移实际文案。Memory Compiler 是最适合作为第一批落点的模块，
因为它是记忆到 Agent 上下文的统一表达通路，直接决定检索记忆、读取记忆、
共享上下文等文本在 Agent prompt 中的语言表现。

## 2. 当前问题

Memory Compiler 目前已经预留了语言参数：

```python
MemoryCompileOptions.language: Optional[str]
MemoryCompiler(default_language="zh")
```

但内部模板尚未系统使用该语言参数，仍存在以下问题：

1. `envelope_templates.py` 中的 `MEMORY_HEADER` / `MEMORY_FOOTER` 是硬编码中文。
2. `envelopes.py` 中的 section 标题、MTP READ 响应标题、shared context 提示
   是硬编码英文或中英混合。
3. `handlers/memory_atom.py` 中的字段标签、置信度文本、截断提示、时间格式仍
   硬编码中文。
4. `retrieval/renderer.py` 直接引用 `MEMORY_HEADER` / `MEMORY_FOOTER` 做 token
   预算，并包含硬编码中文空结果提示。

这些问题会导致当全局语言设置为英文时，Memory Compiler 输出仍然混杂中文。

## 3. 迁移原则

1. **先包装层，后条目层**  
   先迁移 envelope header/footer、section heading、空状态提示等外围文案；
   暂不一次性迁移 MemoryAtom 内部字段标签。

2. **先稳定结构，不改语义结构**  
   保持现有 XML/Markdown 结构、alias、字段顺序与协议提示不变，只替换自然语言文案。

3. **由 MemoryCompileOptions.language 驱动**  
   Memory Compiler 内部不读取全局 config。全局语言只在构造
   `MemoryCompiler(default_language=...)` 或调用 options 时进入。

4. **模板集中到 i18n 侧**  
   新增模板应放在 `src/hivememory/i18n/` 下，业务模块只通过函数获取模板，
   避免继续在业务模块中散落双语常量。

5. **保持小步可回滚**  
   每一批迁移都应有测试覆盖，避免一次性改动所有 prompt 文案造成 Agent 行为漂移。

## 4. 建议新增结构

第一阶段可在 i18n 包下新增一个轻量模板模块：

```text
src/hivememory/i18n/
  __init__.py
  types.py
  resolver.py
  memory_compiler.py
```

`memory_compiler.py` 负责 Memory Compiler 专属文案，例如：

```python
def get_memory_header(language: str | None = None) -> str: ...
def get_memory_footer(language: str | None = None) -> str: ...
def get_memory_section_title(kind: str, language: str | None = None) -> str: ...
def get_memory_envelope_text(key: str, language: str | None = None) -> str: ...
```

第一阶段不需要引入通用 catalog、外部 JSON/YAML 翻译文件或复杂插值框架。
Python 常量和小型 getter 已足够。

## 5. 分阶段计划

### Phase 1: Envelope 包装层

目标：让 Memory Compiler 的 envelope 输出跟随 `MemoryCompileOptions.language`。

迁移对象：

1. `MEMORY_HEADER`
2. `MEMORY_FOOTER`
3. retrieval context section 标题：
   - `Relevant Memories`
   - `Available Sub-Agents`
4. MTP READ response 标题：
   - `[MTP READ Result]`
5. shared context injection 文案：
   - `[Shared Context from Parent Agent]`
   - no shared artifacts 提示
   - parent agent shared artifacts 提示

需要修改：

1. `src/hivememory/i18n/memory_compiler.py`
2. `src/hivememory/i18n/__init__.py`
3. `src/hivememory/engines/memory_compiler/envelopes.py`
4. `src/hivememory/engines/memory_compiler/envelope_templates.py`
5. `tests/unit/engines/memory_compiler/test_compiler.py`

注意事项：

1. `compile_envelope()` 当前创建 `opts = options or MemoryCompileOptions()`；
   需要将 `opts.language` 传给模板选择函数。
2. `_compile_retrieval_context()`、`_compile_mtp_read_response()`、
   `_compile_shared_context_injection()` 需要接收 `options` 或 `language`。
3. 中文默认输出应保持与当前行为语义一致。

验收标准：

1. `MemoryCompiler(default_language="en").wrap(...)` 输出英文 header/footer。
2. `MemoryCompiler(default_language="zh").wrap(...)` 输出中文 header/footer。
3. `MemoryCompileOptions(language="en")` 能覆盖 compiler default。
4. MTP READ response 和 shared context injection 均有中英文测试。

### Phase 2: Retrieval Renderer 空状态与预算

目标：让 retrieval renderer 使用与 Memory Compiler 一致的 envelope 模板与语言。

迁移对象：

1. `_EMPTY_CONTEXT_NOTICE`
2. `_MEMORY_EMPTY_HINT`
3. `_AGENT_EMPTY_HINT`
4. header/footer token 预算来源

需要修改：

1. `src/hivememory/engines/retrieval/renderer.py`
2. renderer 构造或 render 调用路径中的语言传递
3. retrieval renderer 相关单元测试

开放问题：

当前 renderer 本身没有显式语言参数。可选方案：

1. 在 renderer 构造时注入 `default_language`。
2. 在 `render()` 增加可选语言参数。
3. 由上层统一持有 `MemoryCompiler(default_language=...)` 并注入 renderer。

建议先采用构造注入，保持 render 接口稳定。

验收标准：

1. 空检索结果能根据语言返回中文或英文提示。
2. token 预算使用同语言 header/footer 估算。
3. 不再从 renderer 直接引用旧的 `MEMORY_HEADER` / `MEMORY_FOOTER` 常量。

### Phase 3: MemoryAtom 条目模板

目标：迁移具体记忆条目的字段标签与状态文本。

迁移对象：

1. `FULL_ITEM_TEMPLATE`
2. `INDEX_ITEM_TEMPLATE`
3. `AGENT_PROFILE_ITEM_TEMPLATE`
4. 标签空值提示
5. `Change Log`
6. `_format_confidence()` 中的高/中/低、已验证、已废弃、未验证、幻觉警告
7. `_truncate_content()` 的截断提示
8. `TimeFormatter` 语言选择

需要修改：

1. `src/hivememory/engines/memory_compiler/handlers/memory_atom.py`
2. `src/hivememory/i18n/memory_compiler.py`
3. `tests/unit/engines/memory_compiler/test_compiler.py`
4. `tests/utils/test_time_formatter.py` 如需新增语言路径测试

验收标准：

1. `PROMPT_FULL` / `PROMPT_INDEX` / `MTP_READ` / `SHARED_CONTEXT` 输出标签随语言切换。
2. 英文输出使用 `TimeFormatter(Language.ENGLISH)`。
3. 中文默认行为保持兼容。

### Phase 4: 与其他 prompt 模板规范统一

目标：将 Gateway、MTP、Relay、Generation 等已有或潜在双语模板逐步纳入统一规范。

候选对象：

1. `src/hivememory/prompts/gateway.py`
2. `src/hivememory/prompts/mtp.py`
3. `src/hivememory/prompts/relay.py`
4. `src/hivememory/prompts/generation.py`
5. `src/hivememory/prompts/system_prompt.py`

建议顺序：

1. 先迁移 Gateway，因为已有双语模板且影响范围较小。
2. 再迁移 MTP，因为模板大、行为影响高，需要更完整测试。
3. 最后处理 generation/relay/system prompt，因为可能涉及 LLM 输出质量和提取结构。

## 6. 第一批实现清单

建议第一批只做 Phase 1：

1. 新增 `src/hivememory/i18n/memory_compiler.py`。
2. 将 `MEMORY_HEADER` / `MEMORY_FOOTER` 改为中英文模板 getter。
3. 修改 `compile_envelope()`，让 retrieval context、MTP READ response、
   shared context injection 使用 `opts.language`。
4. 保留 `envelope_templates.py` 作为兼容层，或直接让它 re-export 中文默认模板。
5. 补充 Memory Compiler envelope 中英文测试。

暂不处理：

1. `handlers/memory_atom.py` 条目字段标签。
2. retrieval renderer 空状态提示。
3. Gateway/MTP prompt 模板迁移。
4. 通用 catalog 或外部翻译文件。

## 7. 测试建议

第一批至少运行：

```powershell
pytest tests\unit\i18n tests\unit\engines\memory_compiler
```

如果修改影响 retrieval renderer，再追加：

```powershell
pytest tests\unit\engines\retrieval
```

建议新增测试覆盖：

1. 默认中文 envelope。
2. 英文 envelope。
3. options language 覆盖 compiler default。
4. MTP READ response 双语标题。
5. shared context injection 双语提示。

## 8. 风险与控制

### 8.1 Agent 行为变化

Memory Compiler 文案直接进入 Agent 上下文，修改措辞可能影响 LLM 行为。

控制方式：

1. 第一批只迁移 wrapper 文案，不改记忆条目结构。
2. 英文模板尽量忠实表达中文模板当前语义。
3. 保持 XML 标签与协议命令不变。

### 8.2 Token 预算不一致

不同语言 header/footer 长度不同，若 renderer 仍使用旧常量估算，可能导致预算偏差。

控制方式：

1. Phase 1 仅保证 Memory Compiler 自身输出正确。
2. Phase 2 专门处理 renderer 预算与空状态提示。

### 8.3 过早抽象

当前仅支持中英文，过早引入通用翻译 catalog 会增加维护负担。

控制方式：

1. 使用 Python 常量与小型 getter。
2. 等第三种语言或跨模块复用需求明确后，再评估 catalog 化。

## 9. 推荐下一步

下一步建议直接实施 Phase 1。它的改动边界清晰、测试成本低，并能建立后续
MemoryAtom 条目模板、Retrieval Renderer、Gateway/MTP 模板迁移的统一模式。
