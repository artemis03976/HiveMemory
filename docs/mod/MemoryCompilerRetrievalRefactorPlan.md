---
title: Legacy MemoryCompiler and Retrieval Refactor Plan
status: superseded
owner: patchouli
scope: completed-retrieval-compiler-separation
archived_at: 2026-07-28
superseded_by:
  - docs/patchouli/retrieval.md
  - docs/patchouli/memory-compiler.md
---

> 本计划的 Retrieval/Compiler 分离已经进入当前实现，本文停止维护。当前召回与排序以[记忆检索](../patchouli/retrieval.md)为准，IR、target 与上下文策略以 [MemoryCompiler](../patchouli/memory-compiler.md) 为准。

# MemoryCompiler 与 Retrieval 链路改造计划

> 版本：v0.5.x → v0.6.0 前置基础  
> 优先级：中（独立 PR）  
> 状态：设计评审中

---

## 一、背景与问题

### 1.1 当前链路

```
RetrievalFamiliar.retrieve()
  └── RetrievalEngine.retrieve()
        ├── retriever.retrieve()     → SearchResults（atoms + scores）
        └── renderer.render()        → rendered_context（文本，已定型）

retrieve_async()（bus 入口）
  ├── retrieve()                     → RetrievalResponse（含 rendered_context）
  ├── _refresh_vitality_for_memories()
  └── _rerender_response()           → 覆盖 rendered_context（质量降级）
```

### 1.2 存在的问题

**问题一：renderer 定位错误**

renderer 被绑定在 `RetrievalEngine` 中，而其实现完全依赖 `MemoryCompiler`——三个 renderer
（`FullContextRenderer`、`CascadeContextRenderer`、`CompactContextRenderer`）的逻辑就是以不同策略调用
`MemoryCompiler.compile()` + `MemoryCompiler.wrap()`。renderer 是 MemoryCompiler 的策略包装层，
却被放置在了 retrieval 域，造成职责错位。

**问题二：bus 入口双重渲染导致质量降级**

`retrieve()` 内部用 `search_results.results`（含 score/rank 元数据）渲染，
`retrieve_async()` 再调 `_rerender_response()` 用裸 `memories`（无元数据）覆盖渲染结果。
bus 入口返回的是质量更低的渲染，`retrieve()` 内的渲染逻辑对 bus 调用者而言是死代码。

**问题三：`mode` 参数是破坏边界的补丁**

passive 模式（外部消息流）需要 `FullContextRenderer`，但由于渲染策略被锁死在 Familiar 内部，
只能通过 `mode: str = "active"` 参数在 Familiar 内部做分支，再加一次 `_rerender_response()` 覆盖。
这使 `RetrievalFamiliar` 感知到了本不该它知道的上层渲染目标。

**问题四：IR 断档——MemorySectionIR 持有已渲染文本**

```python
# ir.py MemorySectionIR 现状
artifacts: List[CompiledMemoryArtifact]  # 已编译为文本，envelope 层无法再做结构决策

# ir.py 注释（已预判）
# Phase 2C 过渡形态：后续迁移时应替换为 List[MemoryUnitIR]，
# 由 envelope 层直接消费结构化单元。
```

因为 `MemorySectionIR` 持有的是已定型文本，`Cascade` 策略需要"前 N 条 FULL、其余 INDEX、超预算截止"，
只能在 renderer 内自己循环处理 token 预算，无法将此决策交给 envelope 层。两阶段之间存在结构性断档。

---

## 二、设计目标

1. **`RetrievalFamiliar` 只管检索及其副作用**：不负责任何编译，`rendered_context` 由调用方填充
2. **renderer 退出 retrieval 域**：三个策略类迁移并内化为 MemoryCompiler 的 envelope 编译策略
3. **完成 IR 管道**：`MemorySectionIR` 持有 `MemoryUnitIR`，envelope 层统一负责渲染决策
4. **消除 `mode` 参数**：编译策略完全由调用方决定，Familiar 对此一无所知
5. **消除双重渲染**：`rendered_context` 不再在 Familiar 内产生，调用方单次编译
6. **收敛 MemoryCompiler 公开入口**：最终外部只调用 `MemoryCompiler.compile()`；`wrap()` 进入兼容退场期，
   envelope 编译成为正常 compile 流程的一部分

---

## 三、目标架构

### 3.1 完整 IR 管道

```
atoms（List[MemoryAtom]）
  │
  ▼ MemoryCompiler.compile(source, envelope target, opts)
    build_xxx_ir()（每个 atom 独立）
MemoryUnitIR（纯结构，无文本输出）
  │
  ├──[无 envelope 场景]──▶ compile_from_ir(unit, target, opts) ──▶ CompiledMemoryArtifact
  │
  └──[envelope 场景]─────▶ MemoryBundleIR（sections 持有 List[MemoryUnitIR]）
                                │
                                ▼ _compile_envelope(bundle, strategy_config)
                              envelope 层按策略逐 unit 调用 compile_from_ir()
                                │
                                ▼
                          CompiledMemoryEnvelope（最终文本）
```

Cascade 的 "前 N 条 FULL、其余 INDEX" 预算逻辑自然落入 `_compile_envelope` 循环，
不再需要 renderer 预先干预。

最终公开调用形态：

```python
compiler.compile(source, target, options)
```

其中 `target` 可以是单元编译目标，也可以是 envelope 编译目标。`wrap()` 不再作为推荐公开 API，
仅在兼容期作为旧调用壳存在，内部转调 `compile()`。

### 3.2 Retrieval 链路

```
RetrievalFamiliar.retrieve()
  └── RetrievalEngine.retrieve()   → SearchResults（atoms + scores，无 rendered_context）

retrieve_async()（bus 入口）
  ├── retrieve()                   → atoms
  └── _refresh_vitality_for_memories()
  # rendered_context 不填充，由调用方负责

active 路径（PatchouliService.retrieve_for_gaze 等）
  ├── retrieve_async()             → atoms
  └── 自持活跃路径 strategy，调用 MemoryCompiler 编译 rendered_context

passive 路径（PatchouliService.analyze_and_retrieve(mode="passive")）
  ├── retrieve_async()             → atoms
  └── 使用 FullContext strategy，调用 MemoryCompiler 编译 rendered_context

PassiveIngressService
  └── 仍只请求 PATCHOULI_PASSIVE_ANALYZE_AND_RETRIEVE，不持有 MemoryCompiler
```

---

## 四、具体变更

### Phase A — 完成 MemoryCompiler IR 管道（`engines/memory_compiler/`）

**A1：`ir.py` — `MemorySectionIR` 迁移**

```python
# Before
class MemorySectionIR(BaseModel):
    artifacts: List[CompiledMemoryArtifact]  # 已渲染文本
    empty_text: Optional[str] = None

# After（兼容期）
class MemorySectionIR(BaseModel):
    units: List[MemoryUnitIR] = Field(default_factory=list)  # 纯 IR
    artifacts: List[CompiledMemoryArtifact] = Field(default_factory=list)  # deprecated 兼容旧路径
    empty_text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

兼容期允许 `units` 与 `artifacts` 共存；新代码只写 `units`。旧 `artifacts` 路径仅用于迁移期
承接现有测试和调用方，后续单独清理。删除原 Phase 2C 注释，改为明确的 deprecation 注释。

`MemoryUnitIR.metadata` 需要能承载检索元数据。由 `SearchResult` 构建 unit 时注入：

```python
{
    "retrieval_score": result.score,
    "retrieval_rank": index,
    "match_reason": result.match_reason,
    "vector_score": result.vector_score,
    "boost_applied": result.boost_applied,
}
```

本 PR 不要求模板一定展示这些字段，但 IR 不能再丢失 score/rank 等检索上下文。

**A2：`envelopes.py` — envelope 编译接收策略配置**

`compile_envelope()` 改为内部能力，由 `MemoryCompiler.compile()` 调用。其接收 `strategy_config`
参数（兼容期复用 `FullRendererConfig | CascadeRendererConfig | CompactRendererConfig`），
在 `_compile_retrieval_context()` 内按策略逐 unit 调用 `compile_from_ir(unit, target, opts)`。

配置对象归属说明：本 PR 暂复用现有 `retrieval.renderer` 配置对象，不在本 PR 中迁移配置域。
`engines/memory_compiler` 可以临时 import 这些 config 类型，但不得 import `engines.retrieval.renderer`。
后续单独 PR 将 renderer 配置迁移到 memory compiler 配置域或重命名为 context strategy 配置。

三个策略逻辑从 renderer 类迁入此处：

| 策略 | 逻辑迁入位置 | 核心行为 |
|------|-------------|---------|
| `Full` | `_compile_retrieval_context_full()` | 所有 unit → `PROMPT_FULL`，字符上限截断 |
| `Cascade` | `_compile_retrieval_context_cascade()` | 前 N 条 `PROMPT_FULL`，余 → `PROMPT_INDEX`，token 预算截止 |
| `Compact` | `_compile_retrieval_context_compact()` | 所有 unit → `PROMPT_INDEX`，token 预算截止 |

`compile_from_ir` 已在 `handlers/__init__.py` 中存在，直接调用即可。

section 构建规则：

| 输入 unit | section | 单元 target |
|-----------|---------|-------------|
| 普通记忆 | `memories` | 按 Full/Cascade/Compact 策略选择 `PROMPT_FULL` 或 `PROMPT_INDEX` |
| `MemoryType.AGENT_PROFILE` | `agent_profiles` | `AGENT_PROFILE_MENU` |

现有 `_separate_agent_profiles()` 行为必须迁移到 bundle 构建阶段，不能丢失子代理菜单能力。

**A3：`compiler.py` — `compile()` 接管 envelope 编译，`wrap()` 兼容退场**

```python
# Before
def wrap(self, artifacts: CompiledMemoryArtifact | List[CompiledMemoryArtifact] | None, ...)

# After（最终公开形态）
def compile(
    self,
    source: MemoryAtom | List[MemoryAtom] | MemoryUnitIR | List[MemoryUnitIR] | MemoryBundleIR | ...,
    target: MemoryCompileTarget | MemoryEnvelopeTarget,
    options: MemoryCompileOptions | None = None,
) -> CompiledMemoryArtifact | List[CompiledMemoryArtifact] | CompiledMemoryEnvelope:
```

当 `target` 是 `MemoryEnvelopeTarget` 时，`compile()` 内部统一构建 `MemoryBundleIR`，
再进入 `_compile_envelope(bundle, strategy_config)`。`strategy_config` 可放入
`MemoryCompileOptions` 或 compiler 内部调用参数中，具体实现时以最小 API 破坏为准。

推荐实现：扩展 `MemoryCompileOptions`，新增可选字段：

```python
retrieval_strategy_config: FullRendererConfig | CascadeRendererConfig | CompactRendererConfig | None = None
```

这样 `compile()` 仍保持单一公开入口和三参数形态，不额外增加 `compile_for_retrieval()` 或
`strategy_config` 顶层参数。该字段仅在 `target == MemoryEnvelopeTarget.RETRIEVAL_CONTEXT`
时生效；其他 target 忽略。

兼容期保留：

```python
def wrap(...):
    """Deprecated: use compile(..., MemoryEnvelopeTarget.X)."""
    return self.compile(bundle_or_sources, envelope_target, options)
```

`wrap()` 不再是推荐公开入口，不新增 `context_compiler.compile_for_retrieval()` 这类模糊外观。
需要上下文的调用方直接调用 `MemoryCompiler.compile(..., MemoryEnvelopeTarget.RETRIEVAL_CONTEXT)`。

原有以 `sections: List[MemoryEnvelopeSection]` 传入已编译 artifact 的调用方（如 renderer 内部、
agent profile 渲染路径）在兼容期可继续工作，但新代码需改为传入原始 atoms / units / bundle，
不再预编译 artifact 后二次 wrap。

**A4：`models.py` — `MemoryEnvelopeSection` 评估是否保留**

`MemoryEnvelopeSection` 保留为 `CompiledMemoryEnvelope.sections` 的输出 DTO。
输入侧逐步迁移到 `MemorySectionIR` / `MemoryBundleIR`；`MemoryEnvelopeSection` 不再作为新代码的输入类型。

---

### Phase B — `RetrievalEngine` 剥离 renderer（`engines/retrieval/`）

**B1：`engine.py` — 删除 renderer 依赖**

```python
# Before
class RetrievalEngine:
    def __init__(self, retriever: BaseMemoryRetriever, renderer: BaseContextRenderer)

# After
class RetrievalEngine:
    def __init__(self, retriever: BaseMemoryRetriever)
    # 删除 render_memories()
```

`RetrievalResult.rendered_context` 字段从 `models.py` 中删除（engine 层模型不再承载渲染结果）。

**B2：`renderer.py` — 整体删除**

三个 renderer 类及 `create_renderer()` 工厂在 Phase A 完成后整体删除。

**B3：`interfaces.py` — 删除 `BaseContextRenderer`**

---

### Phase C — `RetrievalFamiliar` 重构（`patchouli/services/retrieval.py`）

**构造函数变化：**

```python
# Before
def __init__(self, engine, memory_library, passive_renderer=None, local_bus=None)

# After
def __init__(self, engine, memory_library, local_bus=None)
# 删除 passive_renderer，不注入任何 compiler
```

**`retrieve()` — 纯语义检索，无渲染，无 `mode` 参数：**

返回 `RetrievalResponse`，填充 `memories`、`memories_count`、`latency_ms`，
不填充 `rendered_context`。

**`retrieve_async()` — bus 入口，搜索 + 活力刷新：**

```python
async def retrieve_async(self, request: RetrievalRequest) -> RetrievalResponse:
    response = await self.retrieve(request)
    await self._refresh_vitality_for_memories(response.memories)
    return response
    # rendered_context 不填充，调用方负责
```

**删除：** `_rerender_response()`、`_passive_renderer` 字段、`mode` 参数（`retrieve`、`retrieve_async`、`retrieve_by_aliases`、`retrieve_by_aliases_async` 全部）。

`retrieve_by_aliases_async()` 同理——仅做检索 + 活力刷新，不渲染。

---

### Phase D — 配套更新

**`patchouli/runtime/core.py`：**

```python
def _build_retrieval_engine(self):
    return RetrievalEngine(retriever=retriever)  # 不再传入 renderer

def _register_services(self):
    self._services["retrieval"] = RetrievalFamiliar(
        engine=self._engines["retrieval"],
        memory_library=self.memory_library,
        local_bus=self._local_bus,
        # 删除 passive_renderer / context_compiler
    )
```

`_build_retrieval_engine()` 的 `create_renderer()` 调用和相关 import 删除。

**active 路径（`PatchouliService.retrieve_for_gaze()`）：**

`retrieve_async()` 返回的 `RetrievalResponse.rendered_context` 为空，
`retrieve_for_gaze()` 在拿到 atoms 后，直接调用 `MemoryCompiler.compile()` 填充：

```python
response = await self._bus.request(PatchouliLocalRoutes.MEMORY_RETRIEVE, request)
if response.memories:
    envelope = self._compiler.compile(
        response.memories,
        MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
        options=MemoryCompileOptions(retrieval_strategy_config=self._active_retrieval_strategy),
    )
    response.rendered_context = envelope.text
return response
```

`PatchouliService` 可持有 `MemoryCompiler` 与 active retrieval strategy config。这里不引入
`context_compiler` 新服务；它只是调用 compiler 的 public workflow 组件。

**passive 路径（`PatchouliService.analyze_and_retrieve(mode="passive")`）：**

`PassiveIngressService` 仍只调用 `PATCHOULI_PASSIVE_ANALYZE_AND_RETRIEVE`，不持有 MemoryCompiler。
Patchouli public workflow 根据 `mode="passive"` 选择 Full strategy，并在 `retrieve_for_gaze()` 或
`analyze_and_retrieve()` 内完成编译：

```python
response = await self.retrieve_for_gaze(gaze_result, mode="passive")
if response.memories:
    envelope = self._compiler.compile(
        response.memories,
        MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
        options=MemoryCompileOptions(retrieval_strategy_config=self._passive_full_strategy),
    )
    response.rendered_context = envelope.text
```

这避免将 memory compiler 暴露到 system application 层，保持 Patchouli 子系统边界。

**`agent_runtime/mtp/runtime.py`（SEARCH 处理器）：**

SEARCH 处理器目前直接使用 `retrieval_result.rendered_context`。
迁移后 MTP SEARCH 在 Alice/Koakuma 内拿到 atoms 后调用 `MemoryCompiler.compile()` 生成
`RETRIEVAL_CONTEXT`，或者通过 Patchouli public route 请求一个已编译的 search context。
本 PR 推荐前者：Koakuma 已经持有 `MemoryCompiler` 用于 READ，可复用同一 compiler，避免
让 Patchouli 为 MTP 专门感知输出形态。

---

## 五、影响分析

| 文件 | 变化类型 | 说明 |
|------|---------|------|
| `engines/memory_compiler/ir.py` | `MemorySectionIR` 字段扩展 | 新增 `units`，保留 `artifacts` 兼容期字段 |
| `engines/memory_compiler/envelopes.py` | envelope 内部编译 units | 策略逻辑迁入，支持从 UnitIR 到 artifact 再到 envelope |
| `engines/memory_compiler/compiler.py` | `compile()` 接管 envelope | `wrap()` 保留为 deprecated 兼容壳 |
| `engines/retrieval/engine.py` | 删除 renderer 依赖 | 内部变化 |
| `engines/retrieval/renderer.py` | 整体删除 | 所有 import 需清理 |
| `engines/retrieval/models.py` | 删除 `RetrievalResult.rendered_context` | retrieval.py 中的引用需清理 |
| `engines/retrieval/interfaces.py` | 删除 `BaseContextRenderer` | 所有 import 需清理 |
| `patchouli/services/retrieval.py` | 去 mode/去 passive_renderer/去编译 | 核心改动，构造函数简化 |
| `patchouli/runtime/core.py` | 更新 builder，删除 renderer 创建 | 配套改动 |
| `patchouli/service.py` | `retrieve_for_gaze()` 增加编译步骤 | 持有 MemoryCompiler + active/passive strategy |
| `passive_ingress_service.py` | 不持有 compiler | 继续消费 Patchouli 已编译的 `rendered_context` |
| `agent_runtime/mtp/runtime.py` | SEARCH 处理器增加编译步骤 | 复用 Koakuma 已持有的 MemoryCompiler |
| `prompts/assembler.py` | 不变 | `rendered_context` 由上游填充后传入，无需感知 |
| `tests/unit/engines/retrieval/test_renderer.py` | 迁移/重写 | 迁至 memory compiler retrieval context 策略测试 |
| `tests/unit/engines/retrieval/test_engine.py` | 更新 | 不再断言 `rendered_context` |
| e2e / integration 中直接构造 renderer 的测试 | 更新 | 改为通过 MemoryCompiler.compile envelope 路径验证 |

`core/protocol/models.py` 中的 `RetrievalResponse.rendered_context` **保留**，
语义变更为"由调用方填充的编译结果"，Familiar 不再负责此字段。
未来所有消费路径稳定后可评估是否废弃。

---

## 六、实施顺序与依赖关系

```
Phase A（MemoryCompiler IR 管道）
  └── Phase B（RetrievalEngine 剥离）
        └── Phase C（RetrievalFamiliar 重构）
              └── Phase D（配套更新）
```

Phase A 完成后，Phase B 和 Phase C 可并行推进，最后 Phase D 收尾。

Phase A 完成前，`renderer.py` 保持现状以避免 import 破坏。

---

## 七、验收标准

1. `RetrievalEngine` 不再 import `BaseContextRenderer`、renderer 或 `MemoryCompiler`。
2. `RetrievalFamiliar.retrieve_async()` / `retrieve_by_aliases_async()` 返回 atoms 并刷新活力，但不写入 `rendered_context`。
3. `PatchouliService.prepare_agent_run()` 仍返回带非空 `rendered_context` 的 `AgentRunContext`（有检索结果时）。
4. `PatchouliService.analyze_and_retrieve(mode="passive")` 使用 Full strategy 编译上下文，`PassiveIngressService` 不持有 compiler。
5. Koakuma `SEARCH` 在收到 atoms 后单次调用 `MemoryCompiler.compile()` 生成返回内容。
6. `MemoryCompiler.compile(..., MemoryEnvelopeTarget.RETRIEVAL_CONTEXT)` 能覆盖 Full / Cascade / Compact 三类策略。
7. `MemorySectionIR.units` 路径覆盖普通记忆与 `AGENT_PROFILE` section，子代理菜单行为保持不变。
8. `SearchResult` 的 score/rank/match metadata 能进入 `MemoryUnitIR.metadata`，不再因裸 memories 重渲染丢失。
9. `MemoryCompiler.wrap()` 仍能通过旧测试或兼容测试，但被标记为 deprecated；新生产代码不再新增 `wrap()` 调用。
10. 删除 `engines/retrieval/renderer.py` 后，生产代码无 `create_renderer` / `FullContextRenderer` / `BaseContextRenderer` import。

---

## 八、不在本 PR 范围内

- TheEye 上移至系统级网关（计划 v0.6.0，见 ROADMAP.md）
- `RetrievalResponse.rendered_context` 字段最终的废弃（需等 memory compiler 在所有消费路径稳定后）
- 检索配置域（`retrieval.renderer`）迁移至 memory_compiler 配置域（可后续单独 PR）
- `MemoryCompileTarget` 新 target 的扩展（由具体业务需求驱动）
