# PendingAtom 状态体系统一设计

**文档状态**: Draft (草案)
**适用范围**: MTP `WRITE` / `UPDATE`、Koakuma Runtime、PendingAtomCache、MemoryGenerationEngine、MemoryCompiler、Resolver
**核心目标**: 收敛当前散落在多个模块的状态字符串与枚举，建立 PendingAtom 全生命周期唯一的状态真相源。

---

## 1. 文档目标

本文用于整理 PendingAtom 体系下"状态"层面的统一方案。该设计是 [PendingAtomCacheDesign](PendingAtomCacheDesign.md) 的延续，定位为 PendingAtom 生命周期 + MemoryCompiler 表达边界开发前的**前置收敛**。

设计目标：

- 把当前散落在 `PendingAtomCache.PendingAtomStatus` (旧)、`Settlement.status` (字符串)、`MemoryGenerationResult.operation` (字符串)、`ResolveResult.kind` (独立词汇) 中的状态语义合并到一套权威定义。
- 将"生命周期阶段"、"终结分类"、"对象种类"三件被混在一个字符串里的事，分到三个正交维度。
- 为后续 `PendingAtomRuntime`（M2）、运行级 cache（M3）和 MemoryCompiler 表达边界整理提供干净的类型基底。

本文不引入 PendingAtomRuntime 本身，也不动 cache 的作用域（仍是 runtime 级），仅完成枚举与 settlement 字段的语义统一。

---

## 2. 当前问题

### 2.1 三处状态字符串互相翻译

| 出处 | 字段 | 取值 | 形态 |
|---|---|---|---|
| `cache.PendingAtomStatus`（旧） | `status` | `pending` / `committed` / `merged` / `updated` / `touched` / `discarded` | enum (lowercase) |
| `PendingAtomSettlement.status` | `status` | `COMMITTED` / `MERGED` / `UPDATED` / `TOUCHED` / `DISCARDED` | str (UPPERCASE) |
| `MemoryGenerationResult.operation` | `operation` | `created` / `merged` / `touched` / `updated` / `discarded` | str (lowercase) |
| `ResolveResult.kind` | `kind` | `pending` / `redirect` / `discarded` / `failed` / `atom` / `not_found` | str |

`cache.apply_settlement()` 内部维护着一段手工 `status_map`，把上游的字符串再翻译回枚举；compiler handler 里出现 `pending.status.value if hasattr(pending.status, "value") else str(pending.status)` 这种防御代码，本质上是对类型不确定的运行期补救。

### 2.2 "阶段"与"终结分类"被混在同一维度

旧 `PendingAtomStatus` 把 `pending`（生命周期阶段）和 `committed/merged/touched/discarded`（终结分类）放在一个 enum 里。带来的副作用：

- 想表达"已结算但具体怎么结算"必须读 enum 值字符串本身，调用方需要硬编码字符串比较。
- 未来加 `MATERIALIZING`、`EXPIRED`、`CANCELLED` 等阶段，会和"终结分类"挤在一个维度，组合爆炸。

### 2.3 `Settlement.status` 与 `duplicate_decision` 语义重叠

当前 settlement 同时携带 `status="COMMITTED"` 与 `duplicate_decision="CREATE"`，两者一一对应却各自独立维护，订阅方需要关心保证一致性的隐含约定。

### 2.4 命名不一致

`Settlement.status="COMMITTED"` 对应 `MemoryGenerationResult.operation="created"`，同一件事在系统两处使用了不同动词。这种小不一致在跨模块调试时反复消耗注意力。

---

## 3. 设计原则

### 3.1 三个正交维度

把状态拆分到三个互不重叠的维度，每个维度只回答一个问题：

| 维度 | 关心什么 | 取值数 |
|---|---|---|
| **Status** | 这个 pending 处于生命周期的哪个阶段（不可变 → 可变流转） | 6 |
| **Resolution** | 如果到达 `SETTLED`，是怎么落地的（终结分类） | 5 |
| **Kind** | 这是 WRITE 产生的还是 UPDATE 产生的 | 2 |

`Kind` 当前靠 `draft_xxx` / `rev_xxx` 别名前缀隐式表达，本期不动，留给 M1（模型分层）。本期只统一 **Status × Resolution**。

### 3.2 一处定义、其余派生

- 持久态写入只发生在两处：`cache._status[alias] = Status` 与 `cache._resolution[alias] = Resolution`。
- `ResolveResult.kind`、`MemoryGenerationResult.operation`、视图层的状态文本，全部从 `(status, resolution)` 派生，不再独立维护。
- 这条原则保证未来再加状态时，只需要改一处枚举与一处派生函数。

### 3.3 强类型贯穿模块边界

跨模块传递（Engine → Cache → Compiler）一律使用枚举值；只有最终面向 LLM 的 prompt 文本和持久化 JSON 才做字符串化。

### 3.4 顺手修正命名

将 `COMMITTED` 重命名为 `CREATED`，与 `MemoryGenerationResult.operation="created"`、`DuplicateDecision.CREATE` 对齐。这是借本次窗口一次性完成的名称矫正，避免长期承担翻译成本。

---

## 4. 状态机

```
                   ┌─────────────────────────────┐
                   │           PENDING           │  ← register_write/register_update
                   └──────────────┬──────────────┘
                                  │ start_materializing()
                                  ▼
                   ┌─────────────────────────────┐
                   │        MATERIALIZING        │  ← GenerationEngine 接手
                   └──┬──────────────┬───────┬───┘
                      │              │       │
              settle()│       fail() │       │ cancel()
                      ▼              ▼       │
              ┌──────────────┐  ┌──────────┐ │
              │   SETTLED    │  │  FAILED  │ │
              │ (+resolution)│  └──────────┘ │
              └──────────────┘               │
                                             │
                    ┌──── PENDING ────┐      │
                    │                 │      │
              expire│            cancel│     │
                    ▼                 ▼      ▼
              ┌──────────┐      ┌────────────────┐
              │ EXPIRED  │      │   CANCELLED    │
              └──────────┘      └────────────────┘
```

- **非终态**: `PENDING`、`MATERIALIZING`
- **终态**: `SETTLED`、`FAILED`、`EXPIRED`、`CANCELLED`，不可再迁移

### 4.1 关于 `MATERIALIZING` 的取舍

单独保留 `MATERIALIZING` 有操作价值：

- 表示 GenerationEngine 已经接手，此时再来一个同 alias 的 register 应该被拒绝。
- `EXPIRED` 扫描应跳过 `MATERIALIZING`（正在跑的不能被超时清扫）。

如果一期想精简，可以合并进 `PENDING`，但代价是 cache 需要另开一个 `_inflight` 集合补充表达，反而比维护一个枚举值更乱。**建议一期就引入。**

### 4.2 关于 `EXPIRED` 的引入

本期只把枚举值定义出来，不实现转移逻辑。`EXPIRED` 的实际触发由 M3 run 级 cache + PendingAtomRuntime 的 `expire_run(run_id)` 统一驱动。本期预留枚举值，避免后续再做一轮"加状态值"的修改。

---

## 5. 枚举定义

建议落点 `src/hivememory/alice/runtime/pending_atom_state.py`（状态机本质属于 runtime 域；如果倾向 core，也可放 `core/models/pending.py`，但 runtime 路径更贴近 PendingAtomCache 的实际归属）。

```python
from enum import Enum


class PendingAtomStatus(str, Enum):
    """PendingAtom 生命周期阶段。"""
    PENDING        = "pending"
    MATERIALIZING  = "materializing"
    SETTLED        = "settled"
    FAILED         = "failed"
    EXPIRED        = "expired"
    CANCELLED      = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            PendingAtomStatus.SETTLED,
            PendingAtomStatus.FAILED,
            PendingAtomStatus.EXPIRED,
            PendingAtomStatus.CANCELLED,
        }

    @property
    def is_in_flight(self) -> bool:
        return self in {
            PendingAtomStatus.PENDING,
            PendingAtomStatus.MATERIALIZING,
        }


class PendingAtomResolution(str, Enum):
    """SETTLED 状态下的终结分类（其他状态此字段为 None）。"""
    CREATED    = "created"     # 新原子提交（dedup decision = CREATE）
    MERGED     = "merged"      # 合并到已有原子（dedup decision = UPDATE）
    TOUCHED    = "touched"     # 命中重复，仅更新访问信息（dedup decision = TOUCH）
    UPDATED    = "updated"     # Mode C UPDATE 应用完成
    DISCARDED  = "discarded"   # 低质量重复，未入库（dedup decision = DISCARD）

    @property
    def has_canonical(self) -> bool:
        """该 resolution 是否会产生 canonical_uuid。"""
        return self != PendingAtomResolution.DISCARDED
```

### 5.1 不变量

- `status != SETTLED` 时 `resolution` 必须为 `None`。
- `status == SETTLED` 时 `resolution` 必须非空。
- `resolution.has_canonical == True` 时 `canonical_uuid` 必须非空；反之 `canonical_uuid` 必须为空。

这三条不变量在 `cache.apply_settlement()` 中由断言 / Pydantic validator 强制。

---

## 6. Settlement 模型重塑

### 6.1 字段对比

**当前**：

```python
class PendingAtomSettlement(BaseModel):
    pending_alias: str
    intent_id: str
    status: str                              # "COMMITTED" | "MERGED" | "UPDATED" | "TOUCHED" | "DISCARDED"
    duplicate_decision: Optional[str]        # "CREATE" | "UPDATE" | "TOUCH" | "DISCARD"
    canonical_alias: Optional[str]
    canonical_uuid: Optional[str]
    message: str
```

**之后**：

```python
class PendingAtomSettlement(BaseModel):
    pending_alias: str
    intent_id: str
    resolution: PendingAtomResolution                  # ← 强类型
    duplicate_decision: Optional[DuplicateDecision]    # ← 复用现有枚举
    canonical_alias: Optional[str]
    canonical_uuid: Optional[str]
    message: str
```

### 6.2 字段命名修正

把 `status` 字段改名为 `resolution`。理由：它表达的就是"终结分类"，不是"阶段"。这个改名让调用方读到 `settlement.resolution` 时语义一目了然，并且让 `cache.apply_settlement` 不再需要 status_map 翻译。

```python
# cache.py 之前
status_map = {
    "COMMITTED": PendingAtomStatus.COMMITTED,
    "MERGED":    PendingAtomStatus.MERGED,
    ...
}
self._status[alias] = status_map[settlement.status]

# cache.py 之后
self._status[alias]     = PendingAtomStatus.SETTLED
self._resolution[alias] = settlement.resolution
```

---

## 7. 词汇映射表

| 旧出处 | 旧值 | 新值 |
|---|---|---|
| `MemoryGenerationResult.operation` | `"created"` | 字段删除，调用方读 `settlement.resolution` |
| 同上 | `"merged"` | 同上 |
| 同上 | `"touched"` | 同上 |
| 同上 | `"updated"` | 同上 |
| 同上 | `"discarded"` | 同上 |
| `Settlement.status` | `"COMMITTED"` | `resolution=CREATED` ⚠️ 顺便修正命名 |
| 同上 | `"MERGED"` | `resolution=MERGED` |
| 同上 | `"UPDATED"` | `resolution=UPDATED` |
| 同上 | `"TOUCHED"` | `resolution=TOUCHED` |
| 同上 | `"DISCARDED"` | `resolution=DISCARDED` |
| `Settlement.duplicate_decision` | `"CREATE"`/`"UPDATE"`/... | 复用现有 `DuplicateDecision` 枚举 |
| `cache.PendingAtomStatus`（旧） | `committed/merged/touched/updated/discarded` | 整体替换为新 `PendingAtomStatus` + `PendingAtomResolution` |

---

## 8. ResolveResult.kind 的处理

`ResolveResult.kind` 不应与 `(status, resolution)` 并列存在两个真相源。它是 resolver 视图，应当从 PendingAtom snapshot 派生：

```python
def kind_of(snapshot: PendingAtomSnapshot) -> ResolveKind:
    if snapshot.status.is_in_flight:
        return ResolveKind.PENDING
    if snapshot.status == PendingAtomStatus.FAILED:
        return ResolveKind.FAILED
    if snapshot.status in {PendingAtomStatus.EXPIRED, PendingAtomStatus.CANCELLED}:
        return ResolveKind.NOT_FOUND   # 或新增 EXPIRED/CANCELLED kind
    if snapshot.status == PendingAtomStatus.SETTLED:
        if snapshot.resolution == PendingAtomResolution.DISCARDED:
            return ResolveKind.DISCARDED
        return ResolveKind.REDIRECT    # canonical_uuid 一定存在
    raise UnreachableError
```

派生路径的具体落地建议放在 PendingAtomRuntime 引入之后再做（与 compiler handler 一并清理）；本期先把枚举与 settlement 字段统一，不强行牵连 resolver。

---

## 9. 迁移顺序

切成 3 个独立 commit，每一步都独立可测试，避免出现"既看新枚举又看旧字符串"的中间窗口。

### 9.1 Commit 1 — 加新枚举，不动旧路径

- 新建 `pending_atom_state.py`，定义 `PendingAtomStatus` 与 `PendingAtomResolution`。
- `PendingAtomCache` 内部新增 `_resolution: dict[str, PendingAtomResolution]` 字段。
- 提供新方法 `cache.snapshot(alias) -> PendingAtomSnapshot`（含 status / resolution / canonical_uuid），旧接口完整保留。
- 单元测试覆盖：合法状态迁移、非法迁移、不变量断言。

风险：低。新增类型与字段，不影响现有调用方。

### 9.2 Commit 2 — 切换 Settlement + Engine + Cache

- `PendingAtomSettlement.status: str` 改为 `.resolution: PendingAtomResolution`。
- `PendingAtomSettlement.duplicate_decision: Optional[str]` 改为 `Optional[DuplicateDecision]`。
- `engine._build_settlement` 直接传入枚举值（移除字符串字面量）。
- `cache.apply_settlement` 删掉手工 status_map，使用 resolution 字段直接落库。
- 删除 `MemoryGenerationResult.operation` 字段（调用方迁移到 `settlement.resolution`）。
- 删除 `pending_atom.py` handler 中 `hasattr(pending.status, "value")` 的防御代码。

风险：中。涉及 Settlement 序列化结构，需要检查事件订阅方与日志格式。

### 9.3 Commit 3 — 收敛 Resolver / Compiler 派生路径

- `ResolveResult.kind` 改为 property，从 snapshot 派生。
- compiler handlers 直接消费 `(status, resolution)`，不再伪造 ResolveResult。
- 删除 cache.py 旧 enum 定义与残留的 status_map。

风险：中低。Resolver 与 compiler 的视图统一，行为不变但内部路径变干净。

---

## 10. 几个值得提前对齐的小决定

1. **`COMMITTED` 重命名为 `CREATED`**: 强烈建议。一次性消除长期翻译成本。
2. **是否一期引入 `MATERIALIZING`**: 建议引入。占一行代码，但锁住 "是否可 cancel / 是否可 re-register" 的未来语义。
3. **`EXPIRED` 触发时机**: 本期只定义枚举值，转移逻辑由 M3 引入。
4. **是否为 `EXPIRED` / `CANCELLED` 新增 ResolveKind**: 倾向不加，统一映射到 `NOT_FOUND`，对 Agent 端无语义差别。
5. **新枚举的物理位置**: 倾向 `alice/runtime/pending_atom_state.py`；如果倾向 core 域，可考虑 `core/models/pending.py`，但与 cache 的物理距离会变远。

---

## 11. 与后续工作的衔接

- **M1 模型分层**: 本期统一的 `(status, resolution)` 将作为 `PendingAtom`（生命周期实体）的字段；`PendingCommand` 和 `PendingIntent` 不持有这两个字段。
- **M2 PendingAtomRuntime**: 运行时中心的 `register/start_materializing/settle/fail/expire/cancel` 直接对应本期的状态机迁移；本期已经把这些迁移点的语义锁死，runtime 出现时只是把"分散调用"汇总到一个对象上，不再需要二次设计状态机。
- **M3 Run 级 cache**: 本期不动 cache 作用域；但 `EXPIRED` 已经预留为 run 结束清扫的目标状态。
- **MemoryCompiler 边界整理**: handler 不再需要类型守卫；可以直接以 `(status, resolution)` 作为输入 schema。

---

## 12. 风险与回滚

- 全部改动集中在 PendingAtom 链路，不涉及向量存储、检索、感知层，回滚边界清晰。
- 每个 commit 独立可测、独立可回滚。
- 主要回归风险点：Settlement 事件订阅方对字段名的耦合（`status` → `resolution`）。需在 commit 2 同步排查 `patchouli.events.pending_atom.settled` / `alice.events.pending_atom.settled` 订阅链。

---

## 13. 待办

- [ ] Commit 1：新增枚举与 snapshot 接口
- [ ] Commit 2：切换 Settlement 字段与 Engine / Cache 调用
- [ ] Commit 3：收敛 Resolver / Compiler 派生路径
- [ ] 同步更新 [PendingAtomCacheDesign](PendingAtomCacheDesign.md) 中涉及状态枚举的章节
