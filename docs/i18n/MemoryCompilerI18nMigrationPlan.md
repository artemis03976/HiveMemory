# Memory Compiler i18n 逐步迁移计划

**状态**: Phase 1 与 Phase 2 已完成，下一阶段为 Phase 3  
**当前范围**: Memory Compiler envelope、MTP READ 包装、shared context 包装、retrieval renderer 空状态与预算  
**下一阶段范围**: MemoryAtom 条目模板与字段级文案

## 1. 背景

项目已经完成 i18n 基建的第一步：全局语言配置进入
`HiveMemoryConfig.i18n.default_language`，Agent prompt、MTP prompt 与 Gateway
等入口已逐步改为跟随全局语言或 Agent profile language。

Memory Compiler 是记忆到 Agent 上下文的统一表达通路，直接决定检索记忆、
读取记忆、共享上下文等文本在 Agent prompt 中的语言表现。因此，Memory
Compiler 是实际文案迁移的第一批落点。

## 2. 当前状态

Memory Compiler 已经通过以下入口接收语言：

```python
MemoryCompileOptions.language: Optional[str]
MemoryCompiler(default_language="zh")
```

当前已经完成：

1. `src/hivememory/i18n/memory_compiler.py` 作为 Memory Compiler 专属文案模块。
2. retrieval envelope header/footer 已迁移到 i18n。
3. retrieval section title 已迁移到 i18n。
4. MTP READ response 标题已迁移到 i18n。
5. shared context injection 标题、空状态和说明文本已迁移到 i18n。
6. retrieval renderer 空结果提示已迁移到 i18n。
7. retrieval renderer 的 header/footer 预算估算已改为使用当前语言模板。
8. renderer 构造侧支持 `default_language`，运行时可由全局语言传入。
9. 旧的 `envelope_templates.py` 兼容层已移除。

仍未完成：

1. `handlers/memory_atom.py` 中的字段标签、置信度文本、截断提示、时间格式仍待迁移。
2. Gateway、MTP、Relay、Generation 等非 Memory Compiler prompt 模板仍需后续逐步规范。
3. 暂未引入通用 catalog、外部 JSON/YAML 翻译文件或复杂插值框架。

## 3. 迁移原则

1. **先包装层，后条目层**  
   先迁移 envelope header/footer、section heading、空状态提示等外围文案；
   再迁移 MemoryAtom 内部字段标签。

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

## 4. 当前 i18n 结构

当前 i18n 包结构：

```text
src/hivememory/i18n/
  __init__.py
  types.py
  resolver.py
  memory_compiler.py
```

`memory_compiler.py` 负责 Memory Compiler 专属文案，对外提供：

```python
def get_memory_header(language: str | None = None) -> str: ...
def get_memory_footer(language: str | None = None) -> str: ...
def get_memory_section_title(kind: str, language: str | None = None) -> str: ...
def get_memory_envelope_text(key: str, language: str | None = None) -> str: ...
```

内部按 target 分组维护文案，例如 retrieval、MTP READ、shared context，再统一汇总到
envelope text 字典。当前仅支持中英文，Python 常量与小型 getter 足够满足需求。

## 5. 分阶段计划

### Phase 1: Envelope 包装层

**状态**: 已完成

目标：让 Memory Compiler 的 envelope 输出跟随 `MemoryCompileOptions.language`。

已迁移对象：

1. retrieval envelope header
2. retrieval envelope footer
3. retrieval context section 标题：
   - `memories`
   - `agent_profiles`
4. MTP READ response 标题：
   - `[MTP READ Result]`
5. shared context injection 文案：
   - shared context title
   - no shared artifacts 提示
   - parent agent shared artifacts 提示

已修改：

1. `src/hivememory/i18n/memory_compiler.py`
2. `src/hivememory/i18n/__init__.py`
3. `src/hivememory/engines/memory_compiler/envelopes.py`
4. `tests/unit/engines/memory_compiler/test_compiler.py`

实现结果：

1. `MemoryCompiler(default_language="en").wrap(...)` 输出英文 header/footer。
2. `MemoryCompiler(default_language="zh").wrap(...)` 输出中文 header/footer。
3. `MemoryCompileOptions(language="en")` 能覆盖 compiler default。
4. MTP READ response 和 shared context injection 均有中英文测试。
5. `envelope_templates.py` 兼容层已在 Phase 2 清理时移除。

### Phase 2: Retrieval Renderer 空状态与预算

**状态**: 已完成

目标：让 retrieval renderer 使用与 Memory Compiler 一致的 envelope 模板与语言。

已迁移对象：

1. `_EMPTY_CONTEXT_NOTICE`
2. `_MEMORY_EMPTY_HINT`
3. `_AGENT_EMPTY_HINT`
4. header/footer token 预算来源

已修改：

1. `src/hivememory/engines/retrieval/renderer.py`
2. `src/hivememory/patchouli/runtime/core.py`
3. `tests/unit/engines/retrieval/test_renderer.py`

实现结果：

1. 空检索结果能根据语言返回中文或英文提示。
2. token 预算使用同语言 header/footer 估算。
3. renderer 构造时接收 `default_language`，保持 `render()` 接口稳定。
4. runtime 创建 renderer 时传入 `self.config.i18n.default_language`。
5. renderer 不再引用旧的 `MEMORY_HEADER` / `MEMORY_FOOTER` 常量。
6. `envelope_templates.py` 已删除。

### Phase 3: MemoryAtom 条目模板

**状态**: 待实施

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

建议修改：

1. `src/hivememory/engines/memory_compiler/handlers/memory_atom.py`
2. `src/hivememory/i18n/memory_compiler.py`
3. `tests/unit/engines/memory_compiler/test_compiler.py`
4. `tests/utils/test_time_formatter.py`，如需新增语言路径测试

验收标准：

1. `PROMPT_FULL` / `PROMPT_INDEX` / `MTP_READ` / `SHARED_CONTEXT` 输出标签随语言切换。
2. 英文输出使用英文时间格式。
3. 中文默认行为保持兼容。
4. MemoryAtom 条目结构、字段顺序、alias 与协议提示不发生非必要变化。

### Phase 4: 与其他 prompt 模板规范统一

**状态**: 待实施

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

## 6. 已完成实现清单

Phase 1 与 Phase 2 已完成以下落地项：

1. 新增 `src/hivememory/i18n/memory_compiler.py`。
2. 将 retrieval header/footer、section title、MTP READ title、shared context 文案集中到 i18n。
3. 将 retrieval renderer 空状态提示集中到 i18n。
4. 修改 `compile_envelope()`，让 retrieval context、MTP READ response、
   shared context injection 使用 `opts.language`。
5. 修改 renderer，使预算估算使用当前语言的 retrieval header/footer。
6. runtime 创建 renderer 时传入全局 i18n default language。
7. 删除不再需要的 `envelope_templates.py` 兼容层。
8. 补充 Memory Compiler envelope 与 retrieval renderer 中英文测试。

暂未处理：

1. `handlers/memory_atom.py` 条目字段标签。
2. Gateway/MTP prompt 模板迁移。
3. 通用 catalog 或外部翻译文件。

## 7. 测试状态

当前已通过：

```powershell
pytest tests\unit\i18n tests\unit\engines\memory_compiler tests\unit\engines\retrieval\test_renderer.py
```

覆盖内容：

1. 默认中文 envelope。
2. 英文 envelope。
3. options language 覆盖 compiler default。
4. MTP READ response 双语标题。
5. shared context injection 双语提示。
6. retrieval renderer 空结果中英文提示。
7. retrieval renderer token 预算使用当前语言 header/footer。

## 8. 风险与控制

### 8.1 Agent 行为变化

Memory Compiler 文案直接进入 Agent 上下文，修改措辞可能影响 LLM 行为。

控制方式：

1. 已完成阶段只迁移 wrapper 与 retrieval 空状态文案，不改记忆条目结构。
2. 英文模板尽量忠实表达中文模板当前语义。
3. 保持 XML 标签与协议命令不变。

### 8.2 Token 预算不一致

不同语言 header/footer 长度不同，若 renderer 使用固定中文模板估算，可能导致预算偏差。

当前状态：

1. renderer 已改为使用当前语言 header/footer 估算。
2. 预算路径已有单元测试覆盖。

### 8.3 过早抽象

当前仅支持中英文，过早引入通用翻译 catalog 会增加维护负担。

控制方式：

1. 使用 Python 常量与小型 getter。
2. 等第三种语言或跨模块复用需求明确后，再评估 catalog 化。

## 9. 推荐下一步

下一步建议实施 Phase 3：迁移 `handlers/memory_atom.py` 的条目级模板。

建议仍保持小步推进：先迁移 `PROMPT_INDEX` / `PROMPT_FULL` 的字段标签与空值提示，
再处理 `_format_confidence()`、截断提示和时间格式，最后补齐 `MTP_READ` 与
`SHARED_CONTEXT` 的条目级输出测试。
