# MemoryCompiler Phase 2 记忆中间表示设计

**文档状态**: Draft  
**适用范围**: `engines/memory_compiler/`、`engines/retrieval/renderer.py`、`agent_runtime/mtp/runtime.py`、`agent_runtime/resolver.py`、`core/models/pending.py`  
**核心目标**: 在 Phase 1 / Phase 1.5 已经收敛记忆渲染入口与 envelope 层的基础上，引入结构化 Memory IR，使 `MemoryAtom`、`PendingAtom`、`ResolveResult` 等不同来源先归一为稳定的中间表示，再面向 retrieval、MTP READ、shared context、未来 MTP RUN 等 target 进行编译。

---

## 1. 背景

当前 MemoryCompiler 已经完成第一阶段的入口收敛：

- `MemoryCompiler.compile()` 统一处理 `MemoryAtom`、`PendingAtom`、`ResolveResult`、`PendingAtomSettlement`。
- `MemoryCompiler.wrap()` 统一处理 retrieval context、MTP READ response、shared context injection 等 envelope。
- `MemoryAtom`、`PendingAtom`、`ResolveResult` 的文本模板已经逐步迁入 i18n。
- Retrieval renderer 主要保留策略职责：full / cascade / compact、token budget、agent profile 分组。

但是当前实现仍然是“source object -> target string”的直接渲染模式。每个 handler 都直接读取源对象字段并拼装 target 文本：

```text
MemoryAtom      -> memory_atom.py      -> CompiledMemoryArtifact.text
PendingAtom     -> pending_atom.py     -> CompiledMemoryArtifact.text
ResolveResult   -> resolve_result.py   -> CompiledMemoryArtifact.text
```

这在 Phase 1 足够，但随着 READ 版本溯源、RUN 可执行记忆编译、shared context 运行时提示、参数化记忆等能力进入系统，直接渲染会导致相同语义在不同 handler 中反复解释。

Phase 2 的目标不是继续搬迁模板，而是引入结构化 IR：

```text
MemorySource -> MemoryUnitIR -> target artifact
MemoryUnitIR list -> MemoryBundleIR -> envelope text
```

---

## 2. 当前问题

### 2.1 Source 语义没有统一

`MemoryAtom`、`PendingAtom`、`ResolveResult` 都可能被 READ、shared context、retrieval 或 RUN 使用，但它们当前以不同路径解释：

- `MemoryAtom` 直接渲染 title / summary / content / tags / confidence。
- `PendingAtom` 根据 status 与 source verb 渲染 runtime draft / revision / settled / failed / cancelled / expired。
- `ResolveResult` 根据 alias resolution kind 渲染 redirect / discarded / failed / expired / atom。

这些分支实际都在表达同一类语义：身份、别名、生命周期、正文、解析状态、来源和后续操作提示。

### 2.2 Target 渲染直接依赖源对象结构

例如 `MTP_READ` 现在短期复用 full body，但未来需要加入版本历史、canonical redirect、provenance 信息。如果继续直接从 `MemoryAtom` 或 `ResolveResult` 渲染，READ 专用逻辑会迅速扩散到多个 handler。

### 2.3 Envelope 输入仍然是 artifact list

Phase 1.5 的 `MemoryEnvelopeSection` 已经建立了 section 层，但其输入仍然是 `CompiledMemoryArtifact`。这意味着 envelope 只能拼接已经编译好的字符串，无法根据 bundle 语义统一处理：

- retrieval 的 section 空状态；
- MTP READ 多 alias 响应；
- shared context 注入来源；
- 后续 READ / RUN runtime hints。

### 2.4 RUN 编译缺少结构化前置层

当前 MTP RUN 仍直接执行 `atom.payload.content`，只检查 `MemoryType.CODE_SNIPPET`。未来 RUN 需要识别代码块、命令流水线、参数、前置条件、副作用和权限需求。这些都不适合直接写在字符串模板中，应先进入 IR。

---

## 3. 设计目标

Phase 2 应解决以下问题：

1. **统一记忆来源视图**  
   将正式记忆、pending 记忆、alias 解析结果归一为 `MemoryUnitIR`。

2. **分离 source normalization 与 target rendering**  
   handler 不再直接从源对象拼目标文本，而是先构建 IR，再由 target renderer 从 IR 读取字段。

3. **承接 Phase 1.5 的 body / envelope 分层**  
   `MemoryUnitIR` 面向单条记忆 body 编译，`MemoryBundleIR` 面向 retrieval、MTP READ、shared context 等 envelope 编译。

4. **为 READ 溯源和 RUN 编译预留结构**  
   初期不实现复杂版本系统和可执行编译，但 IR 应保留扩展点。

5. **保持外部 API 稳定**  
   `MemoryCompiler.compile()` 与 `MemoryCompiler.wrap()` 在 Phase 2A 中不改变签名和主要输出。

---

## 4. 非目标

Phase 2 初期不做以下事项：

- 不立即改变 MTP READ 的用户可见输出格式。
- 不立即实现 MTP RUN 的可执行记忆编译。
- 不立即引入完整 git-like 版本历史模型。
- 不立即替换 retrieval renderer 的 full / cascade / compact 策略。
- 不把权限判断、存储可见性、命令是否允许执行等 runtime 策略下沉到 compiler。

---

## 5. 核心 IR 模型

建议新增：

```text
src/hivememory/engines/memory_compiler/ir.py
```

Phase 2A 先实现最小可用模型。

### 5.1 MemoryIdentityIR

```python
class MemoryIdentityIR(BaseModel):
    source_kind: Literal["atom", "pending", "resolve_result"]
    alias: str | None = None           # 当前上下文展示给 Agent 的别名（主字段）
    redirected_from: str | None = None # 仅重定向时有值，表示原始请求别名
    memory_id: str | None = None
```

用于描述当前 IR 单元的身份与别名关系。`alias` 由 builder 负责填入：MemoryAtom builder 填 `canonical_alias`，PendingAtom builder 填 `pending_alias`，ResolveResult builder 在 `kind="redirect"` 时填 `canonical_alias` 并将 `requested_alias` 写入 `redirected_from`，其余 kind 委托给内部对象的 builder。

### 5.2 MemoryContentIR

```python
class MemoryContentIR(BaseModel):
    title: str | None = None
    summary: str | None = None
    content: str | None = None
    tags: list[str] = []
    memory_type: str | None = None
```

用于描述可渲染的正文与索引内容。各 source 类型的字段映射规则如下：

- **MemoryAtom**：`title` ← `index.title`，`summary` ← `index.summary`，`content` ← `payload.content`，`tags` ← `index.tags`，`memory_type` ← `index.memory_type.value`。
- **PendingAtom(WRITE)**：`title` ← `focus.title`，`content` ← `focus.content`，其余为 None / 空。
- **PendingAtom(UPDATE)**：`content` 优先填 `focus.content`（修改后的完整正文），若为空则填 `focus.instruction`（修改指令文本）；`title` 不填，UPDATE 不携带标题。
- **ResolveResult**：委托给内部 `atom` / `pending` 子对象的 builder，terminal 类型（discarded/failed/expired）`content` 为 None。

### 5.3 MemoryStatusIR

```python
class MemoryStatusIR(BaseModel):
    source_state: str | None = None      # PendingAtom.status，仅 pending 有；MemoryAtom 为 None
    resolve_state: str | None = None     # ResolveResult.kind：redirect/discarded/failed/expired/atom/pending
    settlement_state: str | None = None  # PendingAtomSettlement.resolution：CREATED/MERGED/TOUCHED/UPDATED/DISCARDED

    source_verb: Literal["WRITE", "UPDATE"] | None = None
    is_terminal: bool = False            # SETTLED/FAILED/CANCELLED/EXPIRED 时为 True

    message: str | None = None           # settlement.message 或 resolve 级别的说明文本
    reason: str | None = None            # settlement.reason 或 discarded/expired 的原因
    error: str | None = None             # settlement.error
```

三个 `*_state` 字段分别来自不同的源对象，语义互不重叠：

- `source_state`：描述 PendingAtom **当前处于哪个生命周期阶段**（来自 `PendingAtomStatus`）。
- `resolve_state`：描述 alias resolver **找到了什么**（来自 `ResolveResult.kind`）。
- `settlement_state`：描述 PendingAtom **最终以什么方式落盘**（来自 `PendingAtomSettlement.resolution`）。

对 `MemoryAtom` 而言，三个 `*_state` 均为 None，`MemoryStatusIR` 几乎为空，这是正常的。

### 5.4 MemoryUnitIR

```python
class MemoryUnitIR(BaseModel):
    identity: MemoryIdentityIR
    content: MemoryContentIR
    status: MemoryStatusIR
    metadata: dict[str, Any] = {}
```

这是单条记忆 body compiler 的核心输入。

### 5.6 Bundle IR

```python
class MemorySectionIR(BaseModel):
    kind: str
    units: list[MemoryUnitIR] = []
    empty_text: str | None = None


class MemoryBundleIR(BaseModel):
    purpose: MemoryEnvelopeTarget
    sections: list[MemorySectionIR]
    metadata: dict[str, Any] = {}
```

`MemoryBundleIR` 用于承接 retrieval context、MTP READ response、shared context injection 等批量交付语境。

---

## 6. Builder 分层

建议新增 builders 子模块：

```text
src/hivememory/engines/memory_compiler/builders/
  __init__.py
  memory_atom.py
  pending_atom.py
  resolve_result.py
```

### 6.1 MemoryAtom builder

职责：

- 提取 alias、memory id。
- 提取 title、summary、content、tags、memory type。
- 提取 confidence、verification status、updated_at 等 metadata。
- 不负责最终文本格式化。

### 6.2 PendingAtom builder

职责：

- 提取 pending alias、source verb、status。
- 根据 WRITE / UPDATE focus 填充 content。
- 将 `PENDING`、`MATERIALIZING`、`SETTLED`、`FAILED`、`CANCELLED`、`EXPIRED` 填入 `status.source_state`，terminal 状态设 `is_terminal = True`。
- 将 settlement 信息填入 `status.settlement_state`、`status.message`、`status.error`、`status.reason`。

### 6.3 ResolveResult builder

职责：

- 复制 requested alias、canonical alias、canonical uuid。
- 对 `atom` / `pending` 子对象复用对应 builder。
- 将 redirect、discarded、failed、expired、not_found 等解析结果填入 resolution。
- `not_found` 可继续不作为 MemoryCompiler 正常编译对象，由 MTP runtime 作为错误/警告提示处理。

---

## 7. Compiler 调用形态

Phase 2A 保持公开 API 不变：

```python
artifact = compiler.compile(source, MemoryCompileTarget.MTP_READ, options)
envelope = compiler.wrap(artifacts, MemoryEnvelopeTarget.RETRIEVAL_CONTEXT)
```

内部变为：

```text
compile(source, target, options)
  -> _build_unit_ir(source, options)
  -> _compile_unit_ir(unit, target, options)
  -> CompiledMemoryArtifact
```

`wrap()` 初期可以继续接收 `CompiledMemoryArtifact`，等 Phase 2C 再引入 bundle IR：

```text
wrap(...)
  -> build MemoryBundleIR
  -> compile envelope
```

---

## 8. 分阶段落地

### 8.1 Phase 2A: 最小 Unit IR

目标：引入 IR，但不改变用户可见输出。

实施内容：

1. 新增 `ir.py`。
2. 新增 `builders/`。
3. 在 `MemoryCompiler` 中新增内部 `_build_unit_ir()`。
4. 先迁移 `MemoryAtom -> PROMPT_FULL / PROMPT_INDEX / MTP_READ / SHARED_CONTEXT` 从 IR 渲染。
5. 保持 PendingAtom / ResolveResult handler 暂时不变。

验收标准：

- memory compiler 单测全部通过。
- retrieval renderer 单测全部通过。
- MTP READ / RUN 单测全部通过。
- `scripts/test_renderers.py` 输出结构不发生非预期变化。

### 8.2 Phase 2B: PendingAtom / ResolveResult IR 化

目标：统一运行时句柄和 alias 解析结果的结构语义。

实施内容：

1. 将 PendingAtom status 分支改为读取 `MemoryStatusIR.source_state`。
2. 将 settlement / failed / cancelled / expired 信息改为读取 `MemoryStatusIR.settlement_state` 及辅助字段。
3. 将 ResolveResult redirect / terminal 状态改为从 IR 渲染。
4. 保留 `not_found` 在 MTP runtime 中作为错误/警告提示。

验收标准：

- PendingAtom read / ack / terminal 状态测试通过。
- ResolveResult redirect / discarded / failed / expired 测试通过。
- context_refs shared context 测试通过。

### 8.3 Phase 2C: Bundle IR

目标：让 envelope 层接收结构化 section，而不是只拼接 artifact list。

实施内容：

1. 新增 `MemorySectionIR` / `MemoryBundleIR`。
2. retrieval renderer 产出 section plan 时可选择构建 bundle IR。
3. MTP READ 成功路径使用 `MTP_READ_RESPONSE` envelope。
4. shared context injection 使用 bundle IR 表达父子 agent 的共享语境。

验收标准：

- retrieval full / cascade / compact 输出兼容。
- MTP READ 多 alias 输出兼容或有明确迁移测试。
- shared context injection 输出兼容。

---

## 9. 与后续阶段的关系

### 9.1 READ 版本溯源

未来可新增：

```python
class MemoryVersionIR(BaseModel):
    version_id: str
    created_at: datetime | None = None
    summary: str | None = None
    diff_summary: str | None = None
    source: str | None = None
```

`MTP_READ` renderer 可根据 options 决定是否展示版本历史。

### 9.2 RUNNABLE_TOOL 编译

未来可新增：

```python
class ExecutableMemoryIR(BaseModel):
    language: str | None = None
    code_blocks: list[str] = []
    commands: list[str] = []
    parameters: dict[str, Any] = {}
    prerequisites: list[str] = []
    side_effects: list[str] = []
```

`RUNNABLE_TOOL` target 应消费 `ExecutableMemoryIR`，而不是直接执行 `payload.content`。

### 9.3 Provenance 与引用

未来可新增：

```python
class ProvenanceIR(BaseModel):
    source: str | None = None
    span: str | None = None
    confidence: float | None = None
```

READ 和 shared context 可使用 provenance 提供溯源信息。

---

## 10. 风险与缓解

### 风险：IR 过早膨胀

缓解：Phase 2A 只实现最小 Unit IR。version、executable、provenance 先作为未来扩展，不进入首轮实现。

### 风险：输出文本发生大幅变化

缓解：先让 IR renderer 复用现有 i18n 模板，测试继续断言关键结构和兼容文本。

### 风险：Compiler 变成运行时策略中心

缓解：Compiler 只描述和格式化记忆，不决定权限、存储可见性、命令是否允许执行。MTP runtime 仍负责 READ / RUN 的错误状态和权限策略。

### 风险：Bundle IR 与 envelope 边界混淆

缓解：Unit IR 表达“记忆是什么”；Bundle IR 表达“一组记忆如何被组织”；Envelope 表达“这组记忆以什么交付语境给 agent”。

---

## 11. 推荐下一步

建议下一步只实施 Phase 2A：

1. 新增 `ir.py`。
2. 新增 MemoryAtom IR builder。
3. 让 `MemoryAtom -> PROMPT_FULL / PROMPT_INDEX / MTP_READ / SHARED_CONTEXT` 从 `MemoryUnitIR` 渲染。
4. 保持 PendingAtom / ResolveResult 暂时不动。
5. 跑 memory compiler、retrieval renderer、MTP READ/RUN、context_refs 相关测试。

这样可以先验证 IR 插入点是否正确，同时避免一次性改动 PendingAtom 与 ResolveResult 的复杂状态逻辑。
