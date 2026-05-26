# MemoryCompiler Phase 1：渲染收敛设计

**文档状态**：草案  
**范围**：`MemoryAtom`、`PendingAtom`、运行时别名解析 (runtime alias resolution)、检索上下文渲染 (retrieval context rendering)、MTP `READ` / `WRITE` / `UPDATE` 响应、子智能体 `context_refs`、嵌入文本构造 (embedding text construction)  
**非范围 (不包含)**：MTP `RUN` 编译、Markdown/代码提取、可执行工具 schema 生成、沙箱策略变更  
**主要目标**：在引入可运行记忆编译 (runnable memory compilation) 之前，将所有“记忆到文本 (memory-to-text)”的表达逻辑收敛到 `MemoryCompiler` 中间层。

---

## 1. 背景

HiveMemory 已经不仅仅将记忆视为被动的文本。一条记忆可以是：

- 存储在长期记忆中的正式 `MemoryAtom`，
- 由 MTP `WRITE` / `UPDATE` 创建且在运行时可见的 `PendingAtom`，
- 已沉淀为规范记忆并被重定向的 pending 别名 (redirected pending alias)，
- 被渲染到 Prompt 上下文中的检索结果，
- 被 Koakuma 消费的 L0/L1/L2 缓存别名命中结果，
- 未来将被 MTP `RUN` 消费的可运行产物 (runnable artifact)。

在 `PendingAtomCache` 引入运行时影子记忆 (shadow memory) 之后，表达问题变得更加明显。相同的记忆意图现在可以以多种运行时状态出现，但每种状态都有其私有的渲染器或直接的字符串格式化逻辑。

这种碎片化对于 MVP 来说是可以接受的，但与长期的编译器方向不兼容。在 HiveMemory 能够将类似 README 的记忆编译成可执行工具之前，它首先需要一个权威层，专门负责为不同的消费者表达记忆资产。

---

## 2. 当前的碎片化现状

### 2.1 正式 MemoryAtom 渲染

当前实现：

```text
src/hivememory/utils/memory_atom_renderer.py
```

职责：

- 稠密嵌入文本 (dense embedding text)
- 稀疏嵌入文本 (sparse embedding text)
- 完整 Prompt 上下文 (full prompt context)
- 索引 Prompt 上下文 (index prompt context)
- 智能体画像菜单项 (agent profile menu item)
- 置信度/状态格式化
- 截断处理
- 共享记忆页眉/页脚模板

问题：

`MemoryAtomRenderer` 是一个通用工具，而不是编译器边界。业务模块直接导入它，导致记忆表达策略分散在检索、存储和重排序模块中。

### 2.2 检索上下文渲染

当前实现：

```text
src/hivememory/engines/retrieval/renderer.py
```

职责：

- `FullContextRenderer`
- `CascadeContextRenderer`
- `CompactContextRenderer`
- 列表级别的 Token 预算策略
- 常规记忆与 `AGENT_PROFILE` 记忆的分离
- 空结果提示/引导

问题：

检索渲染器仅覆盖检索路径。它们不包含：

- MTP `READ`
- 运行时 pending atoms
- 子智能体共享上下文
- 别名重定向 (alias redirects)
- 已沉淀的 pending 句柄
- L1/L2 缓存的 atom 读取

### 2.3 PendingAtom 渲染

当前实现：

```text
src/hivememory/alice/runtime/pending_renderer.py
```

职责：

- pending 状态的草稿 `READ`
- pending 状态的修订 `READ`
- `WRITE` / `UPDATE` 的 ACK 文本
- 重定向 `READ`
- 重定向 `RUN` 通知
- 已沉淀但未物化为 atom 的文本提示

问题：

`PendingAtomRenderer` 承担了本应由编译器负责的角色，造成了重复。它已经在格式化类似记忆的运行时产物，但它存在于 Alice 运行时内部，无法被检索或存储模块干净地复用。

### 2.4 Koakuma READ 格式化

当前实现：

```text
src/hivememory/alice/runtime/koakuma.py
```

相关路径：

- `_handle_read()` 根据 `pending`、`redirect`、`atom`、`discarded`、`failed` 和 `not_found` 状态来分发已解析的别名。
- `_format_cached_atoms()` 将正式的 atom 格式化为：

```text
[alias]:
payload.content
```

问题：

Koakuma 应该负责路由 MTP 命令、执行权限检查以及调用运行时服务。它不应该决定记忆状态的文本表示。

### 2.5 子智能体 context_refs 格式化

当前实现：

```text
src/hivememory/alice/runtime/agent/loop_executor.py
```

`_fetch_context_refs_content()` 解析别名并私有地格式化以下内容：

- pending atoms
- redirected atoms
- formal atoms

问题：

子智能体共享上下文可能会与 MTP `READ` 和检索上下文产生分歧，即使底层的别名解析为同一条记忆。

### 2.6 嵌入与重排序文本

当前实现：

```text
src/hivememory/infrastructure/storage/vector_store.py
src/hivememory/engines/retrieval/reranker.py
```

两者都使用了 `MemoryAtomRenderer.for_dense_embedding()` / `for_sparse_embedding()`。

问题：

嵌入文本同样是一个编译目标。如果它仍然留在 `MemoryCompiler` 之外，未来对记忆表达的修改可能会在检索时和索引时的行为之间产生隐式分歧。

---

## 3. 设计原则

第一阶段不应该让编译器变得“聪明”，而是要让它变得“权威”。

Phase 1 是一个收敛重构过程：

- 尽可能保留当前的输出行为。
- 将所有记忆表达的入口点移动到 `MemoryCompiler` 之后。
- 让检索渲染器仅负责列表策略和 Token 预算。
- 让 Koakuma 仅负责 MTP 路由和权限检查。
- 让缓存和别名解析器仅负责身份标识和查找。
- 将语义转换和可运行工具的编译推迟到后续阶段。

编译器应该成为唯一被允许将类记忆资产 (memory-like assets) 转换为面向消费者的产物的抽象层。

---

## 4. 建议的模块布局

添加专用包：

```text
src/hivememory/engines/memory_compiler/
  __init__.py
  models.py
  compiler.py
  adapters.py
  targets.py
```

长期文档应存放于：

```text
docs/memory_compiler/
```

这样可以将编译器的工作与 `docs/mod/` 中短期的迁移说明以及 `docs/components/` 中已建立的组件文档分离开来。

---

## 5. 核心抽象

### 5.1 类记忆输入 (Memory-Like Inputs)

Phase 1 编译器输入应支持：

```text
MemoryAtom
PendingAtom
ResolveResult
PendingAtomSettlement
list[MemoryAtom]
list[ResolveResult]
```

`ResolveResult` 非常重要，因为运行时别名解析已经编码了渲染所需的各种状态：

- `pending`
- `redirect`
- `discarded`
- `failed`
- `atom`
- `not_found`

如果调用者仅传递 `MemoryAtom` 或 `PendingAtom`，它们将会丢失重定向和沉淀的语义。

### 5.2 编译目标 (Compile Targets)

初始目标集合：

```python
class MemoryCompileTarget(str, Enum):
    PROMPT_FULL = "prompt_full"
    PROMPT_INDEX = "prompt_index"
    MTP_READ = "mtp_read"
    MTP_ACK = "mtp_ack"
    MTP_REDIRECT_NOTICE = "mtp_redirect_notice"
    SHARED_CONTEXT = "shared_context"
    DENSE_EMBEDDING = "dense_embedding"
    SPARSE_EMBEDDING = "sparse_embedding"
    AGENT_PROFILE_MENU = "agent_profile_menu"
    RUNNABLE_TOOL = "runnable_tool"
```

`RUNNABLE_TOOL` 是故意作为保留目标存在的，但在 Phase 1 中应对其返回“不支持的目标”错误或占位符结果。这保持了公共接口的稳定性，而无需过早引入复杂的 RUN 语义。

### 5.3 编译产物 (Compiled Artifact)

编译器的输出应该是一个结构化对象，而不仅仅是字符串：

```python
class CompiledMemoryArtifact(BaseModel):
    target: MemoryCompileTarget
    text: str
    source_kind: str
    alias: str | None = None
    memory_id: str | None = None
    status: str | None = None
    citations: list[dict] = []
    metadata: dict = {}
```

理由：

- MTP 响应现在需要文本，但未来的 RUN/工具编译需要元数据。
- 来源出处 (Provenance) 和别名重定向不应该隐藏在纯文本中。
- 测试可以对结构化的元数据进行断言，而不需要仅依赖字符串匹配。

### 5.4 编译选项 (Compile Options)

Phase 1 选项：

```python
class MemoryCompileOptions(BaseModel):
    max_content_length: int | None = None
    max_summary_length: int | None = None
    stale_days: int = 90
    include_header_footer: bool = False
    requested_alias: str | None = None
    canonical_alias: str | None = None
    format: Literal["xml", "markdown", "plain"] = "plain"
```

检索列表渲染器仍然可以负责全局预算计算。它们会将单项的选项参数传递给编译器。

---

## 6. 职责边界

### 6.1 MemoryCompiler

负责：

- 将 `MemoryAtom` 转换为目标特定的文本，
- 将 `PendingAtom` 转换为目标特定的文本，
- 将 `ResolveResult` 转换为感知运行时的文本，
- 格式化重定向和沉淀通知，
- 嵌入文本的构造，
- 目标特定的元数据/来源出处。

在 Phase 1 不负责：

- 检索排序，
- 列表预算分配，
- 别名查找，
- MTP 权限执行，
- 最终记忆的生成，
- 可执行工具的提取。

### 6.2 检索渲染器 (Retrieval Renderers)

负责：

- 选择 full/index/compact 策略，
- 对结果块进行排序，
- Token 预算决策，
- 空上下文引导，
- 章节包装器 (section wrappers)。

不负责：

- 单条记忆表达的具体细节。

它们应该调用：

```python
compiler.compile(atom, MemoryCompileTarget.PROMPT_FULL)
compiler.compile(atom, MemoryCompileTarget.PROMPT_INDEX)
compiler.compile(atom, MemoryCompileTarget.AGENT_PROFILE_MENU)
```

### 6.3 KoakumaRuntime

负责：

- 解析 MTP，
- 检查命令和工具的权限，
- 解析别名，
- 路由 `SEARCH` / `READ` / `RUN` / `WRITE` / `UPDATE` / `CALL`，
- 记录引用 (citations)。

不负责：

- 格式化已解析的记忆输出。

它应该调用：

```python
compiler.compile(resolve_result, MemoryCompileTarget.MTP_READ)
compiler.compile(pending, MemoryCompileTarget.MTP_ACK)
```

### 6.4 RuntimeAliasResolver

负责：

- L0 `PendingAtomCache`，
- L1 `KoakumaAtomCache`，
- L2 存储查找，
- 重定向和沉淀状态。

不负责：

- 渲染或编译输出。

### 6.5 存储与重排序 (Storage and Reranking)

负责：

- 存储的 upsert/检索，
- 向量生成的调用，
- 重排序服务的调用。

不负责：

- 嵌入文本的模板。

它们应该调用：

```python
compiler.compile(atom, MemoryCompileTarget.DENSE_EMBEDDING)
compiler.compile(atom, MemoryCompileTarget.SPARSE_EMBEDDING)
```

---

## 7. Phase 1 迁移计划

### 步骤 1：引入 Compiler 包

添加：

```text
src/hivememory/engines/memory_compiler/
```

将 `MemoryCompiler` 实现为现有渲染器之上的兼容外观 (facade)：

- 内部复用当前 `MemoryAtomRenderer` 的行为，
- 内部复用当前 `PendingAtomRenderer` 的行为，
- 暴露一个稳定的 compile API。

此步骤不应涉及任何行为变更。

### 步骤 2：将 Koakuma READ 输出移至编译器后方

替换：

- `PendingAtomRenderer.render_read(...)`
- `PendingAtomRenderer.render_redirect_read(...)`
- `PendingAtomRenderer.render_settled_without_atom(...)`
- `_format_cached_atoms(...)`

为：

```python
artifact = compiler.compile(resolve_result, MemoryCompileTarget.MTP_READ)
```

对于未解析的别名，要么：

- 编译一个轻量级的 `ResolveResult(kind="not_found")`，要么
- 将错误消息的构造保留在 Koakuma 中，直到 `ResolveResult` 支持足够的结构化数据。

首选方向是同样编译 `not_found`，因为它是记忆表达的一部分。

### 步骤 3：将 WRITE/UPDATE ACK 移至编译器后方

替换：

```python
PendingAtomRenderer.render_ack(pending)
```

为：

```python
compiler.compile(pending, MemoryCompileTarget.MTP_ACK)
```

### 步骤 4：将重定向 RUN 通知移至编译器后方

替换：

```python
PendingAtomRenderer.render_redirect_run_notice(...)
```

为：

```python
compiler.compile(resolve_result, MemoryCompileTarget.MTP_REDIRECT_NOTICE)
```

这并不会实现 `RUNNABLE_TOOL`；它只是集中管理了当前 RUN 行为所使用的重定向通知。

### 步骤 5：将子智能体共享上下文移至编译器后方

将 `_fetch_context_refs_content()` 中的直接格式化替换为：

```python
compiler.compile(resolve_result, MemoryCompileTarget.SHARED_CONTEXT)
```

包装器：

```text
[Shared Context from Parent Agent]
```

可以保留在 `LoopExecutor` 中，因为这是 IPC 打包，而不是记忆表达。

### 步骤 6：通过编译器路由检索项渲染

保留 `FullContextRenderer`、`CascadeContextRenderer` 和 `CompactContextRenderer`，但将对 `MemoryAtomRenderer.for_*` 的调用替换为对编译器的调用。

这保留了：

- full/cascade/compact 策略，
- 章节标题，
- Token 预算行为，
- 空上下文行为。

但集中化了以下内容：

- 单项模板，
- 置信度显示，
- 历史记录显示，
- 截断行为。

### 步骤 7：通过编译器路由嵌入文本

替换以下文件中的直接调用：

```text
src/hivememory/infrastructure/storage/vector_store.py
src/hivememory/engines/retrieval/reranker.py
```

替换为对 `DENSE_EMBEDDING` 和 `SPARSE_EMBEDDING` 的编译器调用。

这主要不是为了 Prompt 渲染。它确保编译器拥有所有记忆资产的表示权，包括索引和重排序的资产。

### 步骤 8：弃用直接的渲染器导入

在所有调用方迁移完成后：

- 暂时保留 `MemoryAtomRenderer` 和 `PendingAtomRenderer` 作为内部适配器，
- 从业务模块中移除公开的导入，
- 添加注释，说明新代码必须使用 `MemoryCompiler`，
- 后续删除它们或将其折叠到编译器目标处理程序 (target handlers) 中。

---

## 8. 兼容性要求

Phase 1 应尽可能保留当前行为：

- `SEARCH` 依然返回 `RetrievalResponse.rendered_context`。
- `READ` 依然支持多个别名。
- `READ` 依然支持 pending 别名。
- `READ` 依然报告已重定向的 pending 别名。
- `RUN` 依然拒绝 pending 别名。
- `RUN` 依然仅执行 `CODE_SNIPPET` payload 的内容。
- `WRITE` / `UPDATE` 依然返回包含 pending 别名的 ACK。
- 检索的 full/cascade/compact 行为保持不变。
- 嵌入文本应保持语义对等，除非测试明确允许更改。

预期的可接受变更：

- 在 `READ`、`context_refs` 和检索之间的用语可能会变得更加一致，
- 重复的私有格式化辅助函数可能会消失，
- 断言精确字符串的测试可能需要进行少量更新。

---

## 9. 测试策略

### 9.1 单元测试

为以下场景添加编译器单元测试：

- `MemoryAtom -> PROMPT_FULL`
- `MemoryAtom -> PROMPT_INDEX`
- `MemoryAtom -> MTP_READ`
- `MemoryAtom -> DENSE_EMBEDDING`
- `MemoryAtom -> SPARSE_EMBEDDING`
- `PendingAtom(WRITE) -> MTP_READ`
- `PendingAtom(UPDATE) -> MTP_READ`
- `PendingAtom -> MTP_ACK`
- `ResolveResult(redirect) -> MTP_READ`
- `ResolveResult(discarded/failed) -> MTP_READ`
- `ResolveResult(atom) -> SHARED_CONTEXT`

### 9.2 集成测试

更新现有的 MTP 测试以确保：

- 从 L0 pending 的 `READ` 正常工作，
- 从 L1 缓存的 `READ` 正常工作，
- 从 L2 冷查找的 `READ` 正常工作，
- 从已沉淀的重定向的 `READ` 正常工作，
- `WRITE` / `UPDATE` 的 ACK 依然包含 pending 别名，
- 子智能体 `context_refs` 包含与 `READ` 相同的记忆正文。

### 9.3 回归测试

保留检索测试：

- full 渲染器，
- cascade 渲染器，
- compact 渲染器，
- 被动模式 full 渲染，
- 别名检索渲染。

嵌入测试应验证编译器输出与现有的向量索引期望保持兼容。

---

## 10. 风险与缓解措施

### 风险：编译器变成上帝对象 (God Object)

缓解措施：

保持 `MemoryCompiler` 作为分发器的角色，并将特定于目标的行为移入小型处理程序/适配器中。检索预算逻辑和运行时路由必须留在外部。

### 风险：Prompt 输出改变过大

缓解措施：

Phase 1 应该在内部复用现有的渲染器。首先重构调用路径，稍后再改进模板。

### 风险：循环导入

缓解措施：

将编译器放置在 `engines/memory_compiler` 下，并保持模型导入的单向性：

```text
core models -> compiler models -> compiler handlers
Alice runtime -> compiler
Retrieval -> compiler
Storage -> compiler
```

除了 `PendingAtom` 类型外，避免从编译器导入 Alice 运行时模块。如果这会造成循环，则为 pending 输入定义一个轻量级的协议/适配器。

### 风险：编译目标模糊了运行时策略

缓解措施：

编译器可以描述并格式化记忆状态，但它绝不能决定权限、存储可见性或某项命令是否被允许。

---

## 11. 未来阶段

### Phase 2：记忆中间表示 (Memory IR)

引入结构化的中间表示 (Intermediate Representation)：

```text
MemorySource -> MemoryIR -> target artifact
```

潜在字段：

- 标题，
- 摘要，
- 正文章节，
- 代码块，
- 流程步骤，
- 参数，
- 前置条件，
- 副作用，
- 来源出处跨度 (provenance spans)，
- 置信度，
- 运行时要求。

### Phase 3：可运行记忆编译

实现 `RUNNABLE_TOOL`：

- 提取可运行代码或命令步骤，
- 推断参数 schema，
- 识别前置条件，
- 记录所需的权限，
- 生成执行计划或工具清单，
- 拒绝模棱两可的、类似 README 的记忆，并提供可操作的诊断信息。

### Phase 4：编译缓存与失效

使用以下维度持久化编译产物：

- 源 atom id，
- 源 hash，
- 编译器版本，
- 目标，
- 选项 hash。

失效条件：

- 记忆 payload 发生改变，
- pending atom 沉淀，
- 编译器版本改变，
- 目标 schema 发生改变。

### Phase 5：UI 与可观测性

暴露编译产物的视图：

- prompt 产物，
- 检索产物，
- 可运行产物，
- 来源出处，
- 编译器诊断。

这将使记忆编译变得可检查，而不再是隐藏的 Prompt 魔法。

---

## 12. Phase 1 完成定义 (Definition of Done)

当满足以下条件时，Phase 1 即告完成：

- 新代码拥有了 `MemoryCompiler` 包及 compile API，
- Koakuma `READ` 不再直接格式化 `MemoryAtom` / `PendingAtom`，
- Koakuma `WRITE` / `UPDATE` 的 ACK 通过编译器处理，
- 子智能体的 `context_refs` 通过编译器处理，
- 检索单项的渲染通过编译器处理，
- 稠密/稀疏嵌入文本通过编译器处理，
- 业务中对 `MemoryAtomRenderer` 和 `PendingAtomRenderer` 的直接导入被移除或被标记为仅供兼容使用，
- 测试覆盖了正式 atom、pending atom、重定向、已沉淀、检索和嵌入目标。

到那时，HiveMemory 将拥有一个稳定的表达边界。后续的 MTP `RUN` 编译器便可以作为一个新的目标来实现，而不是又成为另一条私有的 Koakuma 路径。
