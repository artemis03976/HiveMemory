---
title: Legacy MemoryCompiler IR Design
status: superseded
owner: patchouli
scope: completed-memory-compiler-ir-design
archived_at: 2026-07-28
superseded_by: docs/patchouli/memory-compiler.md
---

> 本文保留 Memory IR 的形成背景，已停止维护。当前 builder、MemoryUnitIR、section/bundle IR 与 target dispatch 以 [MemoryCompiler 当前设计](../patchouli/memory-compiler.md)为准；未落地的 RUN 阶段仍不属于当前能力。

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

Phase 2 启动前的实现仍然是“source object -> target string”的直接渲染模式。每个 handler 都直接读取源对象字段并拼装 target 文本：

```text
MemoryAtom      -> memory_atom.py      -> CompiledMemoryArtifact.text
PendingAtom     -> pending_atom.py     -> CompiledMemoryArtifact.text
ResolveResult   -> resolve_result.py   -> CompiledMemoryArtifact.text
```

这在 Phase 1 足够，但随着 READ 版本溯源、RUN 可执行记忆编译、shared context 运行时提示、参数化记忆等能力进入系统，直接渲染会导致相同语义在不同 handler 中反复解释。当前实现已经转向 target-first handler：`prompt.py`、`mtp.py`、`embedding.py`、`agent_profile.py` 等文件按编译目标组织渲染逻辑，旧的 `memory_atom.py`、`pending_atom.py`、`resolve_result.py` handler 已不再作为渲染入口存在。

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

用于描述当前 IR 单元的身份与别名关系。`alias` 由 builder 负责填入：MemoryAtom builder 填 `canonical_alias`，PendingAtom builder 填 `pending_alias`。`requested_alias` 属于本次 compile 调用上下文，不进入 IR；透明展开 `ResolveResult(kind="atom" / "pending")` 时应保留到 `MemoryCompileOptions.requested_alias`。`redirected_from` 只用于显式 redirect，表示需要提示 Agent 从请求别名切换到 canonical alias；它不应承担普通请求别名记录职责。

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
    source_verb: Literal["WRITE", "UPDATE"] | None = None
    is_terminal: bool = False            # SETTLED/FAILED/CANCELLED/EXPIRED 时为 True
    is_redirect: bool = False            # resolver 已完成 canonical atom 预取（SETTLED + canonical alias/uuid）
    is_discarded: bool = False           # SETTLED + resolution == DISCARDED，唯一需要主动通知 Agent 的结算状态
    message: str | None = None
    reason: str | None = None
    error: str | None = None
```

`source_state` 是状态分支的单一事实来源，来自 `PendingAtomStatus`。`is_redirect` 和 `is_discarded` 是两个需要在渲染层显式区分的布尔标记：

- `is_redirect`：resolver 在 SETTLED + 有 canonical 的路径上预取了 canonical atom，IR 内容来自该 atom，`identity.redirected_from` 保存原始请求别名。
- `is_discarded`：`PendingAtomSettlement.resolution == DISCARDED`，是唯一需要告知 Agent"此别名不会产生新记忆"的结算结果；其余 resolution 值（CREATED/MERGED/TOUCHED/UPDATED）对 Agent 无需区分，不进入 IR。

`PendingAtomResolution` 的其余值不投影进 IR，避免 IR 随枚举演进而失同步。对 `MemoryAtom` 而言所有字段均为默认值，这是正常的。

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
- 将 settlement 信息填入 `status.is_discarded`（仅 DISCARDED）、`status.message`、`status.error`、`status.reason`。

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
2. 将 discarded 状态改为读取 `MemoryStatusIR.is_discarded`；其余 terminal 状态从 `source_state` 分支。
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
2. `MemoryCompiler.wrap()` 保持外部签名不变，在内部把 `artifacts` / `MemoryEnvelopeSection` list 归一化为 `MemoryBundleIR`。
3. `compile_envelope()` 改为只接收 `MemoryBundleIR`，并通过 `bundle.purpose` 分发 retrieval、MTP READ、shared context 三类 envelope。
4. `MemoryEnvelopeSection` 继续作为外部兼容输入与 `CompiledMemoryEnvelope.sections` 返回类型；bundle section 会在 envelope 输出时转回该模型。
5. 本轮 `MemorySectionIR.artifacts` 仍保存 `CompiledMemoryArtifact`，暂不回退到 `MemoryUnitIR`，因为 envelope 当前消费的是已经按 target 编译完成的 artifact。

验收标准：

- retrieval full / cascade / compact 输出兼容。
- MTP READ 多 alias 输出兼容或有明确迁移测试。
- shared context injection 输出兼容。
- `sections=None` 与 `sections=[]` 的语义保持区分：前者可生成 default empty section，后者保留空 section list，用于 shared context empty case。

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

---

## 12. Phase 2B 增补设计决策

本节记录 Phase 2A / 2B 实现后，对后续收敛方向的补充约束。其核心目标是让 IR 真正成为 source normalization 与 target rendering 之间的边界，而不是只在各 handler 内部临时构建。

### 12.1 Builders 与 Handlers 的职责边界

理想编译链路应逐步收敛为：

```text
source object
  -> builders: source -> MemoryUnitIR / MemoryBundleIR
  -> handlers/renderers: IR -> target artifact text
  -> envelopes: artifact/bundle -> delivery text
```

其中：

- `builders` 只理解源对象结构，负责把 `MemoryAtom`、`PendingAtom`、`ResolveResult` 以及未来新的记忆来源归一化为 IR。builder 不应知道 `PROMPT_FULL`、`MTP_READ`、`SHARED_CONTEXT` 等 target 模板。
- `handlers` / `renderers` 只理解 target 需求，负责把 `MemoryUnitIR` 编译为 `CompiledMemoryArtifact`。handler 不应直接读取源对象字段。
- `envelopes` 只处理交付语境，负责把 artifact 或 bundle 包装为 retrieval context、MTP READ response、shared context injection 等最终文本。
- `compiler.py` 负责调度：`compile(source, target)` 应先调用 `_build_unit_ir(source)`，再调用 `_compile_unit_ir(unit, target)`。

这意味着当前“handler 内部自行调用 builder”的实现只能作为过渡形态。后续收尾时，应让 `MemoryCompiler._compile_single()` 成为唯一的 Unit IR 构建入口，使 handler 的输入从 source object 迁移为 `MemoryUnitIR`。

### 12.2 MTP_READ 仍是单一 Target

虽然 `MemoryAtom` 与 `PendingAtom` 都有 `MTP_READ` 需求，而且未来输出会逐渐分化，但不建议拆成 `MTP_READ_MEMORY_ATOM`、`MTP_READ_PENDING_ATOM` 这类 target。

原因是 target 表达的是使用场景，而不是 source 类型。`MTP_READ` 的含义始终是“Agent 请求读取一个记忆引用后的响应”。正式记忆、运行时 pending 记忆、redirect、discarded、failed、expired 等差异应由 IR 中的身份、状态、内容和元数据表达，再由 READ renderer 选择内部 view。

推荐在 `MTP_READ` handler 内部引入只对 READ 可见的 view 分发：

```python
class MemoryReadView(str, Enum):
    FORMAL_ATOM = "formal_atom"
    PENDING_WRITE = "pending_write"
    PENDING_UPDATE = "pending_update"
    SETTLED_FALLBACK = "settled_fallback"
    REDIRECT = "redirect"
    DISCARDED = "discarded"
    FAILED = "failed"
    EXPIRED = "expired"
```

```text
render_mtp_read(unit, options)
  -> select_mtp_read_view(unit)
  -> render selected view
```

这样可以同时满足两类扩展：

- 正式 `MemoryAtom` 的 READ view 未来可以读取版本历史、provenance、diff summary 等信息。
- `PendingAtom` 的 READ view 可以继续表达运行时待定原子、WRITE / UPDATE 草稿、失败或终止状态等特殊提示。

因此差异应落在 `MemoryReadView` 或 READ 专用 renderer 上，而不是通过拆分 target 来表达。

### 12.3 ResolveResult 的透明展开策略

为了兼容统一 IR，`ResolveResult` 不应总是作为独立 source 编译。建议按语义分为两类：

- `ResolveResult(kind="atom")` / `ResolveResult(kind="pending")` 是透明 wrapper，应展开为内部 `MemoryAtom` / `PendingAtom` 后使用对应 builder。
- `ResolveResult(kind="redirect")` / terminal kind 是语义路由结果，应继续走 `build_resolve_result_ir()`，因为它们需要携带 redirect、discarded、failed、expired 等额外状态。

推荐的入口逻辑：

```python
if isinstance(source, ResolveResult):
    if source.kind == "pending" and source.pending:
        requested_alias = source.requested_alias
        source = source.pending
        options.requested_alias = requested_alias
    elif source.kind == "atom" and source.atom:
        requested_alias = source.requested_alias
        source = source.atom
        options.requested_alias = requested_alias
    elif source.kind == "not_found":
        raise ValueError(...)
    # redirect / terminal 继续走 build_resolve_result_ir()
```

实际实现时，透明展开路径应保留 `ResolveResult.requested_alias` 到 `options.requested_alias`，这是 compile 调用的上下文参数，不需要进入 IR。`MemoryIdentityIR` 只保留 `alias`（主展示别名）和 `redirected_from`（redirect 专用），职责更单一。

`not_found` 目前仍可不作为 MemoryCompiler 的正常编译对象处理。Koakuma / MTP runtime 可以继续把它作为 READ 错误或警告提示语生成。

### 12.4 后续迁移顺序

建议后续按以下顺序收敛：

1. 让 `MemoryCompiler._compile_single()` 先统一构建 `MemoryUnitIR`，再把 IR 交给 handler。透明展开 `ResolveResult(kind="atom"/"pending")` 时，将 `requested_alias` 传递到 `options`。
2. 将 handler 入参从 source object 逐步迁移为 `MemoryUnitIR`。
3. 在 `MTP_READ` handler 中引入 READ 专用 view 分发，而不是新增 target。
