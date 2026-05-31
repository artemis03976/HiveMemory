​# PendingAtomRuntime 落地规划

**文档状态**: Draft (草案)
**适用范围**: `alice/runtime/cache.py` 中的 `PendingAtomCache`、`alice/runtime/core.py`、`alice/runtime/resolver.py`、`alice/runtime/koakuma.py`、`alice/runtime/pending_atom_state.py`
**核心目标**: 将散落在 PendingAtomCache、AliceRuntime 订阅器、KoakumaRuntime、Resolver 中的 PendingAtom 操作收敛到一个统一的运行时中心 `PendingAtomRuntime`，并借此根治状态真相源分裂与状态机失约束的问题。

---

## 1. 文档目标

本文是 [PendingAtomCacheDesign](PendingAtomCacheDesign.md)、[PendingAtomStatusUnificationDesign](PendingAtomStatusUnificationDesign.md) 之后的下一步收敛工作。前两份文档分别完成了：

- 运行时缓存 + 三级别名解析的物理结构（M1 已落地）
- 状态枚举 / Settlement 字段 / Resolver kind 的语义统一（Commit 1-3 已落地）

本期工作从"数据结构 + 类型定义"层面进一步推到"行为入口"层面：把 PendingAtom 的注册、状态迁移、结算、查询、清理统一到一个对象上。

设计目标：

- 把目前散落在 `PendingAtomCache.register_write/register_update/apply_settlement` 与外部直接读写 `atom.status` / `atom.settlement` 的路径，收敛为一个 `PendingAtomRuntime` 外观。
- 让状态机 `_TRANSITIONS` 真正起约束作用：`PendingAtomCache.apply_settlement` 当前直接把 `atom.status` 设为 SETTLED，跳过 MATERIALIZING，状态机定义形同摆设。
- 消除 cache 内部并存的 `_resolution` / `_redirects` / `atom.settlement` / `atom.status` 四份真相源带来的派生分歧风险（resolver 与 snapshot 当前各读一部分，存在结论不一致的可能）。
- 为后续 M3 run 级 cache、Lifecycle/GC、事件流接入预留干净的对象边界。

本期不动 alice 子系统的物理目录（`alice/runtime/pending_atom/` 子包属于未来 alice 拆分讨论的范围），仅在 `alice/runtime/` 内部做收敛。

---

## 2. 当前问题

### 2.1 状态机定义未生效

`pending_atom_state._TRANSITIONS` 已经定义好 PENDING → MATERIALIZING → SETTLED 的合法迁移，但实际写入路径没有人走它：

```python
# cache.apply_settlement() 当前实现
atom.status = PendingAtomStatus.SETTLED   # 直接跳到终态
self._resolution[atom.pending_alias] = settlement.resolution
```

带来三个副作用：

1. MATERIALIZING 阶段在生产路径中不可观测；
2. 非法迁移（如 SETTLED → 再次 settle）不会报错，会静默覆盖；
3. 后续 M2 想引入 fail / cancel / expire 命令时，需要先清掉这条直跳分支。

### 2.2 真相源四处分裂

`PendingAtomCache` 当前持有：

| 字段 | 含义 | 写入位置 | 读取位置 |
|---|---|---|---|
| `atom.status` | 生命周期阶段 | `apply_settlement` | resolver.`_resolve_pending_hit` |
| `atom.settlement` | 结算视图 | `apply_settlement` | resolver、snapshot |
| `_resolution[alias]` | 终结分类 | `apply_settlement` | snapshot |
| `_redirects[alias]` | redirect 视图 | `apply_settlement` | resolver、snapshot |

`snapshot()` 以 `_resolution` / `_redirects` 为派生源，`resolver._resolve_pending_hit` 同时读 `pending.status` 与 `pending.settlement or get_redirect()`。两条派生路径理论上等价，但物理上从不同字段拼装结果——这是隐含一致性约定，调试时容易踩坑，将来加新状态字段也会被迫两边改。

### 2.3 行为入口分散在多个对象

PendingAtom 的命令面被切散：

- `register_write` / `register_update` 在 `PendingAtomCache` 上
- `apply_settlement` 也在 `PendingAtomCache` 上，但订阅触发点在 `AliceRuntime._on_pending_atom_settled`
- 没有显式的 `start_materializing` / `fail` / `cancel` / `expire`——状态机定义里写了，但没有入口

后续 M2 要引入这些命令时，没有一个明确的对象作为它们的归属。

---

## 3. 设计原则

### 3.1 单一外观

所有 PendingAtom 命令与查询走同一个对象。资源所有权（字典、索引）也归该对象，外部不持有 cache 实例。

### 3.2 真相源唯一

`PendingAtom` 自己持有完整状态：`status` / `resolution` / `settlement` / `canonical_alias` / `canonical_uuid`。runtime 只保留反查索引（`_intent_index`、`_canonical_index`），不重复存放派生字段。

snapshot() 与 resolver 共用同一条派生路径（都从 PendingAtom 读），物理上无法分裂。

### 3.3 状态机约束生效

所有 status 变更走同一个私有 `_set_status(atom, target)` 闸门，闸门内部调用 `is_legal_transition()`。非法迁移直接抛 `InvalidStateTransition`。

`settle()` 内部走 PENDING → MATERIALIZING → SETTLED 两步迁移，让 MATERIALIZING 阶段在写入时序上真实存在（即便外部观察窗口很短）。

### 3.4 接口稳定

对外 API 形态保持与现 `PendingAtomCache` 等价（仅命令名扩展），调用方迁移成本只有改 import + 改属性名。`snapshot()` / `register_write` / `register_update` / `get` / `get_by_intent_id` / `has` / `clear` 签名不变。

---

## 4. 物理结构

```
alice/runtime/
  pending_atom/                ← 新增子包
    __init__.py                ← 仅暴露 PendingAtomRuntime
    runtime.py                 ← PendingAtomRuntime（外观 + 状态机闸 + 命令入口）
    store.py                   ← _PendingAtomStore（私有；接管原 PendingAtomCache 的字典与索引）
    state.py                   ← 由 pending_atom_state.py 迁过来
  cache.py                     ← 仅保留 KoakumaAtomCache、AgentProfileCache
  core.py                      ← 持有 PendingAtomRuntime 实例
  resolver.py                  ← 注入 PendingAtomRuntime
  koakuma.py                   ← 通过 alias_resolver.pending_runtime 访问
```

**职责切分**：

- `_PendingAtomStore`（store.py）：纯存储层，独占 `_atoms` / `_intent_index` / `_canonical_index` 字典。提供原子级 CRUD 与索引查询（`put` / `get` / `pop` / `index_by_intent` / `index_by_canonical_uuid`），不做状态机校验、不发事件。访问级别为子包私有（命名带下划线前缀，`__init__.py` 不导出）。
- `PendingAtomRuntime`（runtime.py）：外观 + 状态机闸 + 命令入口。所有外部调用走 runtime，runtime 内部委托给 store 做存取。状态机校验、命令组合、与 `PendingAtomSnapshot` 派生都在这一层。
- `PendingAtomCache` 类整体迁移到 store.py 并改名为 `_PendingAtomStore`，内部字典结构与原实现一致；原 `register_write` / `register_update` / `apply_settlement` 等方法拆为两类：纯存储动作留在 store（如 `put_atom`、`bind_intent`），状态机相关动作移到 runtime（如 `register_write`、`settle`）。

`pending_atom_state.py` 整体搬到 `pending_atom/state.py`；保留旧路径作为兼容 re-export 一个版本，避免一次性触动所有引用方。

---

## 5. PendingAtomRuntime API

```python
class PendingAtomRuntime:
    """PendingAtom 全生命周期管理中心。

    所有 PendingAtom 的注册、状态迁移、结算、查询都通过本对象。
    资源所有权（_atoms、_intent_index、_canonical_index）由本对象独占。
    """

    # ---- 命令（写入路径） ----

    def register_write(...) -> PendingAtom:
        """注册 WRITE pending atom。status=PENDING。"""

    def register_update(...) -> PendingAtom:
        """注册 UPDATE pending revision。status=PENDING。"""

    def start_materializing(pending_alias: str) -> None:
        """PENDING → MATERIALIZING。生产端在 GenerationEngine 接手时调用。"""

    def settle(settlement: PendingAtomSettlement) -> None:
        """应用 settlement。
        内部走 PENDING → MATERIALIZING → SETTLED 两步迁移
        （兼容生产端尚未调用 start_materializing 的场景）。"""

    def fail(pending_alias: str, error: str) -> None:
        """MATERIALIZING → FAILED。"""

    def cancel(pending_alias: str, reason: str) -> None:
        """in-flight → CANCELLED。"""

    def expire(pending_alias: str) -> None:
        """PENDING → EXPIRED。M3 run 结束清扫使用。"""

    # ---- 查询（读取路径） ----

    def snapshot(alias: str) -> Optional[PendingAtomSnapshot]:
        """统一查询入口；resolver / compiler / 视图层均走此接口。"""

    def get(pending_alias: str) -> Optional[PendingAtom]:
        """返回 PendingAtom 原始引用（仅用于持有 focus / runtime_scope 等数据）。"""

    def get_by_intent_id(intent_id: str) -> Optional[PendingAtom]: ...
    def get_redirect(pending_alias: str) -> Optional[PendingAtomSettlement]: ...
    def get_pending_aliases_for_canonical_uuid(canonical_uuid: str) -> List[str]: ...

    def has(alias: str) -> bool: ...
    def all_aliases() -> List[str]: ...
    def all_atoms() -> List[PendingAtom]: ...

    # ---- 生命周期 ----

    def clear() -> None: ...

    @property
    def size(self) -> int: ...
```

### 5.1 状态机闸门

```python
def _set_status(self, atom: PendingAtom, target: PendingAtomStatus) -> None:
    if not is_legal_transition(atom.status, target):
        raise InvalidStateTransition(
            f"PendingAtom '{atom.pending_alias}': "
            f"{atom.status.value} -> {target.value} is not a legal transition"
        )
    atom.status = target
```

`InvalidStateTransition` 新增到 `core.mtp.exceptions` 或 `pending_atom/state.py`，作为 `RuntimeError` 子类。

---

## 6. 字段收敛

### 6.1 PendingAtom 字段扩展

```python
class PendingAtom(BaseModel):
    pending_alias: str
    intent_id: Optional[str]
    status: PendingAtomStatus
    source_verb: Literal["WRITE", "UPDATE"]
    focus: WriteFocus | UpdateFocus
    identity: Identity
    runtime_scope: RuntimeScope
    created_at: datetime

    settlement: Optional[PendingAtomSettlement] = None
    # ↓ 新增字段（迁移自 cache._resolution / _redirects）
    resolution: Optional[PendingAtomResolution] = None
    canonical_alias: Optional[str] = None
    canonical_uuid: Optional[str] = None
    error: Optional[str] = None        # FAILED 时填充
    cancel_reason: Optional[str] = None  # CANCELLED 时填充
```

### 6.2 Runtime 与 Store 的字段切分

```python
# pending_atom/store.py
class _PendingAtomStore:
    """内部存储层。仅负责字典与索引；不感知状态机。"""

    def __init__(self) -> None:
        self._atoms: dict[str, PendingAtom] = {}
        self._intent_index: dict[str, str] = {}            # intent_id -> pending_alias
        self._canonical_index: dict[str, list[str]] = {}    # canonical_uuid -> [pending_alias]

    # 纯存储动作
    def put(self, atom: PendingAtom) -> None: ...
    def get(self, alias: str) -> Optional[PendingAtom]: ...
    def get_by_intent(self, intent_id: str) -> Optional[PendingAtom]: ...
    def aliases_by_canonical(self, canonical_uuid: str) -> list[str]: ...
    def bind_intent(self, intent_id: str, alias: str) -> None: ...
    def bind_canonical(self, canonical_uuid: str, alias: str) -> None: ...
    def all_aliases(self) -> list[str]: ...
    def all_atoms(self) -> list[PendingAtom]: ...
    def clear(self) -> None: ...

    @property
    def size(self) -> int: ...


# pending_atom/runtime.py
class PendingAtomRuntime:
    """外观 + 状态机闸 + 命令入口。所有外部访问走本类。"""

    def __init__(self) -> None:
        self._store = _PendingAtomStore()

    # 命令调用 store.put + 维护索引 + 走 _set_status
    # 查询直接读 store
```

> 删除：原 `PendingAtomCache._resolution` / `_redirects` 字典在 PR2 后并入 `PendingAtom` 自身字段（见 §6.1），store 内部不再保留这两个字段。

### 6.3 派生路径合并

`snapshot()` 与 `resolver._resolve_pending_hit` 共用：

```python
def snapshot(self, alias: str) -> Optional[PendingAtomSnapshot]:
    atom = self._atoms.get(alias)
    if atom is None:
        return None
    return PendingAtomSnapshot(
        pending_alias=alias,
        status=atom.status,
        resolution=atom.resolution,
        canonical_alias=atom.canonical_alias,
        canonical_uuid=atom.canonical_uuid,
    )
```

resolver 改成只消费 snapshot：

```python
async def _resolve_pending_hit(self, pending, alias, context):
    snap = self._pending_runtime.snapshot(alias)
    if snap.status.is_in_flight:
        return ResolveResult(kind="pending", ...)
    if snap.status == PendingAtomStatus.FAILED:
        return ResolveResult(kind="failed", ...)
    if snap.status == PendingAtomStatus.SETTLED:
        if snap.resolution == PendingAtomResolution.DISCARDED:
            return ResolveResult(kind="discarded", ...)
        if snap.canonical_alias or snap.canonical_uuid:
            atom = self._resolve_cached_canonical(pending.settlement)
            ...
            return ResolveResult(kind="redirect", ...)
    return ResolveResult(kind="not_found", ...)
```

resolver 不再读 `pending.status` 与 `get_redirect()`——所有派生数据走 snapshot。

---

## 7. 迁移顺序

切成 3 个独立 PR，每一步都独立可测、独立可回滚。

### 7.1 PR1 — 拆分外观 / 存储 + 改名

- 新建 `pending_atom/` 子包（`runtime.py` / `store.py` / `state.py` / `__init__.py`）。
- `PendingAtomCache` 类内容拆分为两层：
  - 纯存储动作（字典读写、索引维护）下沉到 `_PendingAtomStore`，访问级别为子包私有（命名带下划线、`__init__.py` 不导出）。
  - 状态机相关动作（`register_write` / `register_update` / `apply_settlement → settle` / `snapshot`）保留在 `PendingAtomRuntime` 上，作为外观 + 命令入口；内部委托给 `_PendingAtomStore`。
  - 字段（`_resolution` / `_redirects`）暂时保留在 store 上不动（与现有 `PendingAtomCache` 等价），由 PR2 统一收敛。
- `pending_atom_state.py` 内容迁到 `pending_atom/state.py`，旧路径保留为 re-export 兼容入口（一行 `from .pending_atom.state import *`）。
- 调用方改动：
  - `core.py`: `PendingAtomCache()` → `PendingAtomRuntime()`，属性 `_pending_cache` → `_pending_runtime`；订阅器 `apply_settlement` → `settle`（接口名同步改）。
  - `resolver.py`: 构造参数 `pending_cache` → `pending_runtime`，私有字段 `_pending_cache` → `_pending_runtime`，property 同步改。SETTLED 分支已开始读 `pending_runtime.snapshot()` 派生 resolution（`DISCARDED` 判定走 snapshot）；`settlement` 字段仍从 `pending.settlement or get_redirect()` 取，由 PR2 把字段并入 PendingAtom 后整条派生路径才会完全收敛。
  - `koakuma.py`: property `pending_cache` → `pending_runtime`，调用点 `register_write` / `register_update` 不变。
  - 测试：批量替换 fixture 名与导入路径。

风险：低。无行为变化，是"封装动作"——把原 cache 的方法按职责切到两层，对外行为与字段语义完全保持。

### 7.2 PR2 — 字段收敛 + 真相源统一

- `PendingAtom` 加 `resolution` / `canonical_alias` / `canonical_uuid` / `error` / `cancel_reason` 字段。
- `PendingAtomRuntime` 删除 `_resolution` / `_redirects` 字典，改读写 PendingAtom 上的对应字段。
- `snapshot()` 改为直接从 PendingAtom 派生。
- `resolver._resolve_pending_hit` 改为只消费 `snapshot()`，删掉直接读 `pending.status` + `get_redirect()` 的混合分支。
- `get_redirect()` / `get_pending_aliases_for_canonical_uuid()` 保留（仅测试断言使用），内部实现改读 PendingAtom 字段。

风险：中。触碰 redirect 派生路径，需重点回归 `test_runtime_alias_resolver.py` 与 `test_pending_atom_state.py` 的 snapshot 断言。

### 7.3 PR3 — 状态机闸门 + 命令扩展

- `PendingAtomRuntime._set_status()` 私有方法，所有 status 变更走它。
- `apply_settlement` → `settle`：内部走 PENDING → MATERIALIZING → SETTLED 两步迁移。
- 新增命令：`start_materializing` / `fail` / `cancel` / `expire`。本 PR 不连入生产路径（GenerationEngine 等暂不调用），仅暴露 + 单测覆盖。
- 引入 `InvalidStateTransition` 异常。

风险：低-中。新命令暂不被生产端使用，主要是单测覆盖；`settle` 走两步迁移会让"重复结算"立刻报错（现状是静默覆盖），需要确认订阅链路没有重发。

---

## 8. 调用方影响面

| 文件 | 改动 | PR |
|---|---|---|
| `alice/runtime/cache.py` | 删除 `PendingAtomCache` 类，保留 `KoakumaAtomCache` / `AgentProfileCache` | PR1 |
| `alice/runtime/pending_atom/runtime.py` | 新增 `PendingAtomRuntime`（外观 + 状态机闸 + 命令入口） | PR1 |
| `alice/runtime/pending_atom/store.py` | 新增 `_PendingAtomStore`（接管字典与索引；子包私有） | PR1 |
| `alice/runtime/pending_atom/state.py` | 由 `pending_atom_state.py` 迁入 | PR1 |
| `alice/runtime/pending_atom_state.py` | re-export 兼容入口 | PR1 |
| `alice/runtime/core.py:39-78` | 持有 `PendingAtomRuntime`，订阅器调 `settle()` | PR1 |
| `alice/runtime/resolver.py:60-160` | 注入 `pending_runtime`；PR2 改为只读 snapshot | PR1 / PR2 |
| `alice/runtime/koakuma.py:293-776` | property `pending_cache` → `pending_runtime` | PR1 |
| `alice/runtime/models.py` | `PendingAtom` 加 5 个新字段 | PR2 |
| `tests/unit/alice/runtime/test_pending_atom_state.py` | fixture / import 改名 | PR1-3 |
| `tests/unit/patchouli/kernel/test_runtime_alias_resolver.py` | fixture 改名；PR2 同步 redirect 断言 | PR1-2 |
| 其他测试 | 通过 `register_write` / `apply_settlement` / `snapshot` 调用，仅需改方法名 | PR1 |

`MemoryGenerationEngine` 不改：它产出 settlement 后通过事件流交给 `AliceRuntime._on_pending_atom_settled` 转调 `runtime.settle()`，与生产端无直接耦合。

---

## 9. 与后续工作的衔接

- **M2 PendingAtomRuntime 命令扩展（生产端接入）**: 本期 PR3 引入的 `start_materializing` / `fail` / `cancel` / `expire` 命令暂不被生产端调用。后续在 `MemoryGenerationEngine` 接手时调用 `start_materializing`，在 LLM/Storage 失败时调用 `fail`，在 run 取消时调用 `cancel`。
- **M3 Run 级 cache + GC**: 本期保持 runtime 级作用域。M3 引入 `expire_run(run_id)` 后，PendingAtomRuntime 内部按 RuntimeScope.run_id 索引并对超龄项调用 `expire()`。
- **MemoryCompiler 边界**: compiler handler 已经全部走 `pending` 与 `settlement` 字段，不读 cache。本期 PendingAtomRuntime 改名对 compiler 透明。
- **alice 子系统拆分**: `pending_atom/` 子包结构已经按"将来可能整体迁出 alice"的方向放置，物理上贴近 `agent_kernel/` 候选目录。该拆分由后续 alice 边界讨论独立决策，不阻塞本期。

---

## 10. 风险与回滚

- 全部改动集中在 `alice/runtime/` 内部，不涉及 patchouli、engines、core 子系统，回滚边界清晰。
- 每个 PR 独立可测、独立可回滚。
- 主要回归风险点：
  - PR1: 调用方 import path 改动较多，需要 IDE 全局搜索校验（无运行时风险）。
  - PR2: redirect / canonical 派生路径改动，需重点回归 resolver 与 snapshot 单测；多 alias 指向同一 canonical_uuid 的场景需补充用例。
  - PR3: `settle()` 内部状态机变严，需确认 `AliceRuntime._on_pending_atom_settled` 不会在同一 alias 上重发事件。

---

## 11. 待办

- [x] PR1：新建 `pending_atom/` 子包（外观 `PendingAtomRuntime` + 私有 `_PendingAtomStore` + `state.py`），`pending_atom_state.py` 转为 re-export 兼容入口，调用方同步改名（`pending_cache`→`pending_runtime`、`apply_settlement`→`settle`）
- [ ] PR2：`PendingAtom` 加字段，删除 `_resolution` / `_redirects`，resolver 改为只读 snapshot
- [ ] PR3：状态机闸门 + 命令扩展（`start_materializing` / `fail` / `cancel` / `expire`），`settle` 走两步迁移
- [ ] 同步更新 [PendingAtomCacheDesign](PendingAtomCacheDesign.md) 第 18 节实现阶段规划
- [ ] 同步更新 [PendingAtomStatusUnificationDesign](PendingAtomStatusUnificationDesign.md) 第 11 节后续工作衔接
