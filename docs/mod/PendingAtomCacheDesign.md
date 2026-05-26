# PendingAtomCache 与运行时 Shadow Memory 设计草案

**文档状态**: Draft (草案)  
**适用范围**: MTP `WRITE` / `UPDATE`、Koakuma Runtime、Agent Loop、Sub-Agent CALL、Patchouli 异步记忆生成链路  
**核心目标**: 在不破坏异步记忆生成原则的前提下，为 Agent run 提供可立即读取、可传递、可结算的运行时记忆句柄。

---

## 1. 文档目标

本文用于整理 `PendingAtomCache` 及相关运行时流程的初始设计。该设计围绕一个已经显现的系统问题展开：

> Agent 在运行时通过 MTP `WRITE` / `UPDATE` 提交记忆生成意图后，系统会立即回填成功响应，但真正的记忆原子仍由帕秋莉在后台异步生成。此时 Agent 缺少一个可寻址入口，无法在同一 run 内读取刚刚提交的内容，也无法稳定收割子 Agent 产生的 artifacts。

本文的目标不是把记忆生成改回同步模式，而是为异步生成链路补上一层运行时可见性：

- `WRITE` / `UPDATE` 同步创建可读的 pending handle
- Agent 可以在同一 run 内立刻 `READ` 该 handle
- 子 Agent 产生的 pending artifacts 可以在 `CALL` 结束后返回给主 Agent，并在当前阶段转为主 Agent 可管理的运行时资产
- 后台生成完成后，pending handle 可以结算到正式 `MemoryAtom`
- 结算失败或去重合并时，Agent 仍能获得清晰状态

---

## 2. 当前问题

当前 MTP `WRITE` / `UPDATE` 的语义接近：

```text
Agent emits WRITE/UPDATE
  -> Koakuma captures WriteFocus/UpdateFocus
  -> returns ACK to Agent
  -> LoopExecutor records focus
  -> InteractionPayload enters Perception
  -> Flush triggers Librarian generation asynchronously
  -> GenerationEngine creates/updates MemoryAtom later
```

这条链路保留了冷路径异步生成的优点，但存在几个运行时缺口：

1. **ACK 与真实落库之间存在语义落差**
   - 回填文本可能暗示“记忆已保存”
   - 实际上此时只有 `WriteFocus` / `UpdateFocus` 被捕获，正式原子尚未生成

2. **Agent 无法立即读取新提交内容**
   - `READ` 只能解析已有 alias
   - 对于尚未生成的原子，没有 alias 可供读取

3. **子 Agent artifact 收割不稳定**
   - 子 Agent 内部 `WRITE` 触发后，主 Agent 返回时正式 atom 可能尚未生成
   - 当前 `WRITE` 分支无法可靠完成 artifact 归属转移

4. **多次 WRITE/UPDATE 可能被覆盖**
   - 当前 `ChatResult.write_focus` / `update_focus` 是单值语义
   - 长 run 内多次主动写入时，需要支持列表化聚合

---

## 3. OS 类比与设计启发

该问题与现代操作系统中的异步 IO、缓存写回和句柄可见性问题高度相似。

### 3.1 Page Cache / Write-back

OS 中应用调用 `write()` 后，数据通常先进入 page cache，随后由内核异步 flush 到磁盘。应用如果马上 `read()`，读到的是 page cache 中的最新内容，而不是磁盘上的旧内容。

HiveMemory 可借鉴该模型：

```text
OS write() -> page cache -> async disk flush
MTP WRITE  -> pending atom -> async memory generation/storage
```

这里的 pending atom 不是“假数据”，而是尚未写回长期存储的运行时可见数据。

### 3.2 File Descriptor / Handle

OS 通过 fd 向进程返回稳定句柄。进程后续读写依赖 fd，而不需要关心文件最终位于哪个磁盘 block。

HiveMemory 中，`WRITE` 应返回稳定的运行时句柄：

```text
draft_login_api_7f3a
rev_mem_login_api_spec_b921
```

Agent 使用该 alias 继续 `READ`。子 Agent 产生的 alias 会在 `CALL` 结束后回填给主 Agent，并在当前多智能体阶段下转为主 Agent 的运行时资产。

### 3.3 WAL / Intent Log

数据库与 journaling filesystem 通常先记录操作意图，再异步更新真实数据结构。

HiveMemory 中应区分：

- **控制面**: 同步记录 pending intent，生成可读句柄
- **数据面**: 异步执行帕秋莉生成、去重、合并、入库

### 3.4 Completion Queue

异步 IO 系统通常通过 completion event 通知请求最终状态。

HiveMemory 中需要一个 settlement 机制：

```text
pending_alias=draft_login_api_7f3a
status=COMMITTED
final_alias=mem_login_api_spec
final_uuid=...
```

---

## 4. 核心抽象

## 4.1 PendingAtom

`PendingAtom` 表示一个运行时可读的记忆生成句柄。

它不是正式 `MemoryAtom`，也不承诺最终一定落库。它的承诺是：

> 在其生命周期内，Agent 可以通过 pending alias 读取到本次 MTP 写入或更新意图的内容、元数据和当前生成状态。

建议语义定义：

```text
PendingAtom = a runtime-readable materialization handle for a memory intent.
```

建议字段：

```python
class PendingAtom:
    pending_alias: str
    intent_id: str
    source_verb: Literal["WRITE", "UPDATE"]
    status: PendingAtomStatus

    title: str | None
    content: str
    reason: str | None
    instruction: str | None

    target_alias: str | None
    target_uuid: str | None

    identity: Identity
    run_id: str
    frame_id: str
    parent_frame_id: str | None
    action_id: str | None
    agent_alias: str | None
    depth: int

    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None

    final_alias: str | None
    final_uuid: str | None
    error: str | None
```

## 4.2 PendingAtomCache

`PendingAtomCache` 是运行时快速查询层，职责类似 page cache：

- 同步注册 pending atom
- 通过 pending alias 查询 pending atom
- 支持 alias redirect / tombstone
- 支持按 run/frame 清理
- 支持结算后更新状态

当前阶段 `PendingAtomCache` 仅作为普通 runtime cache 实现，不引入持久化 ledger。它服务于单次 Agent run 及其子 frame 的运行时可见性，不承担跨进程恢复、审计追踪和完整事件溯源职责。

建议先实现为内存结构：

```text
pending_alias -> PendingAtom
final_alias -> pending_alias? (optional)
intent_id -> pending_alias
```

## 4.3 PendingAtomLedger

`PendingAtomLedger` 是未来更可靠的意图日志层，当前阶段不实现。

它用于解决：

- 服务重启后 pending intent 丢失
- Agent 已收到 pending alias，但后台任务未执行
- 生成链路需要可观测的状态追踪

Ledger 将在项目未来集中建设事件流与可观测性体系时再加入。当前设计只需避免把 `PendingAtomCache` 与长期审计、事件回放、跨重启恢复等职责绑定。

未来分层可以演进为：

```text
PendingAtomRegistry
  -> Runtime PendingAtomCache
  -> Durable PendingAtomLedger
```

## 4.4 Intent Identity 与 Focus 绑定

Phase 2 中，`intent_id` 用于标识一次系统内部的记忆生成意图，`pending_alias` 用于提供给 Agent 的运行时可寻址句柄。二者应在同一时刻生成并绑定，但语义上保持分离。

- `pending_alias` 面向 Agent，可出现在 MTP 回填、`READ`、`context_refs` 和子 Agent artifact transfer 中
- `intent_id` 面向系统内部，用于贯穿 Koakuma、ArchivePayload、LibrarianCore、GenerationEngine、Deduplicator 和 settlement 回填链路
- `intent_id` 不从 `pending_alias` 派生，避免 alias canonicalization 后形成不必要耦合
- `intent_id` 不属于 `RuntimeScope`；`RuntimeScope` 描述执行坐标，`intent_id` 描述记忆生成意图

生成责任应收束在 Alice runtime 的 pending 注册点：

```text
Koakuma handles WRITE/UPDATE
  -> PendingAtomCache.register_write/register_update()
  -> system generates pending_alias + intent_id
  -> PendingAtom stores pending_alias + intent_id
  -> Koakuma builds WriteFocus/UpdateFocus with the same pending_alias + intent_id
  -> Focus travels into Patchouli generation
  -> Generation result / settlement returns intent_id
  -> PendingAtomCache resolves intent_id back to pending_alias
```

Patchouli 侧只消费 `intent_id`，不负责创建或覆盖它。即使未来 MTP 参数中出现同名字段，也应忽略或拒绝，避免 Agent 越权伪造系统内部意图标识。

`WriteFocus` / `UpdateFocus` 应携带该绑定关系：

```python
class WriteFocus:
    content: str
    reason: str | None
    title: str | None
    identity: Identity
    pending_alias: str | None
    intent_id: str | None


class UpdateFocus:
    instruction: str
    content: str | None
    target_uuid: str
    target_alias: str
    existing_memory: MemoryAtom | None
    identity: Identity
    pending_alias: str | None
    intent_id: str | None
```

建议格式：

```text
intent_id = intent_{short_uuid_or_ulid}
```

该绑定使 Phase 2 的反向索引可以稳定表达为：

```text
pending_alias -> intent_id
intent_id -> pending_alias
intent_id -> settlement
canonical_uuid -> [pending_alias...]
```

---

## 5. Alias 策略

Pending alias 必须与正式 memory alias 区分，避免覆盖长期记忆。

建议命名：

```text
WRITE  -> draft_{slug}_{short_id}
UPDATE -> rev_{target_alias}_{short_id}
```

示例：

```text
draft_login_api_7f3a
rev_mem_login_api_spec_b921
```

设计约束：

- pending alias 由系统生成，不允许 Agent 自定义
- `title` 可参与 slug 生成，但不能直接作为 alias
- alias 应在当前 runtime 内唯一
- pending alias 不应覆盖正式 atom alias
- final alias 生成后，pending alias 至少在当前 run 内保留 redirect
- 禁止通过 MTP 参数指定 pending alias，以避免 alias 注入、越权覆盖和混淆正式 atom 的风险

---

## 6. 状态机

建议状态：

```text
PENDING
  -> MATERIALIZING
  -> COMMITTED
  -> MERGED
  -> UPDATED
  -> TOUCHED
  -> REDIRECTED
  -> DISCARDED
  -> FAILED
  -> EXPIRED
  -> CANCELLED
```

状态含义：

- `PENDING`: 已创建，可读，后台生成尚未开始
- `MATERIALIZING`: 帕秋莉正在处理
- `COMMITTED`: 已生成新的正式 `MemoryAtom`
- `MERGED`: 生成结果被去重合并到已有 atom
- `UPDATED`: `UPDATE` 已应用到目标 atom
- `TOUCHED`: 生成内容与已有 atom 高度重复，仅触达或强化已有 atom
- `REDIRECTED`: pending alias 已不再持有 pending 内容，只作为指向 canonical alias 的兼容入口
- `DISCARDED`: 生成链路判定无需物化，例如低价值或完全重复
- `FAILED`: 生成失败
- `EXPIRED`: pending 生命周期结束
- `CANCELLED`: run 取消或系统关闭导致任务终止

---

## 7. 生命周期与 Canonical Alias

PendingAtom 的生命周期需要拆成两层：

```text
内容生命周期:
PENDING -> MATERIALIZING -> COMMITTED / MERGED / UPDATED / TOUCHED / DISCARDED / FAILED

寻址兼容生命周期:
active pending alias -> redirect alias -> expired tombstone
```

理想情况下，pending 内容在正式记忆落地后即可结束。但 Agent 在读取过 pending alias 后，未必能立即意识到新的正式 alias 已经替代它。因此 pending alias 不应在物化完成后立刻失效，而应进入 redirect grace period。

确定策略：

- 正式记忆生成后，pending alias 不再持有 pending 内容
- pending alias 退化为轻量 redirect entry
- redirect entry 指向 canonical alias / canonical uuid
- 当前 run 内必须保留 redirect，保证旧 pending alias 仍可解析
- run 结束后可以保留短 TTL tombstone，便于前端、日志和迟到事件查看
- TTL 后可以清理，但 tombstone 如仍知道 final alias，应尽量返回 final alias 提示

示例：

```text
draft_login_api_7f3a -> mem_login_api_spec
```

此时 `draft_login_api_7f3a` 是 deprecated handle，`mem_login_api_spec` 是 canonical alias。

建议 resolver entry：

```python
class PendingAliasEntry:
    kind: Literal["pending", "redirect", "tombstone"]
    pending_alias: str
    canonical_alias: str | None
    canonical_uuid: str | None
    status: PendingAtomStatus
    expires_at: datetime | None
```

### 7.1 Redirect 后的访问行为

当 Agent 继续访问旧 pending alias：

```text
⟪ READ | draft_login_api_7f3a ⟫
```

系统应自动解析到 canonical alias，并返回正式记忆内容：

```text
[Alias Redirected]
Requested alias: draft_login_api_7f3a
Canonical alias: mem_login_api_spec
Status: committed

[mem_login_api_spec]:
...

Action: Use 'mem_login_api_spec' for future READ/RUN/UPDATE calls.
```

关键要求：

- 渲染主体必须使用正式 alias
- pending alias 只出现在 redirect header 中
- 响应中必须明确提示后续使用 canonical alias
- `SEARCH` 在正式记忆落地后只应返回 canonical alias，不再返回 pending alias

### 7.2 协议层 Canonicalization

不能只依赖 Prompt 让 Agent 自觉切换 alias，必须在协议层做 canonicalization。

规则：

- `READ draft_xxx`: 自动解析到 final atom，返回 canonical alias 提示
- `RUN draft_xxx`: 如果 canonical atom 是可运行工具，则执行 canonical atom，并提示 alias 已迁移
- `UPDATE draft_xxx`: 解析到 canonical alias 后，对 canonical atom 创建 revision，并提示后续使用 canonical alias
- `context_refs=["draft_xxx"]`: 注入 shared context 时自动替换为 canonical alias
- `SEARCH`: 只暴露 final alias，不暴露已 redirect 的 pending alias

这样即使 Agent 继续使用旧 alias，系统也会持续把它导向正式 alias。

### 7.3 过期 Tombstone

redirect grace period 结束后，pending alias 可以进入 tombstone 或被清理。

如果 tombstone 仍保留 final alias，访问时应返回：

```text
Temporary alias 'draft_login_api_7f3a' has expired.
Final alias was: mem_login_api_spec.
Action: Use 'mem_login_api_spec' for future access.
```

如果 tombstone 不再知道 final alias，访问时返回：

```text
Temporary alias 'draft_login_api_7f3a' has expired.
Action: Use SEARCH to locate the finalized memory.
```

---

## 8. MTP WRITE 流程

建议流程：

```text
Agent:
  ⟪ WRITE | * | title="Login API" content=`...` ⟫

Koakuma:
  1. 校验 WRITE 权限与参数
  2. 生成 pending_alias / intent_id
  3. 创建 PendingAtom(status=PENDING)，绑定 pending_alias / intent_id
  4. 写入 PendingAtomCache
  5. 构造 WriteFocus，附带 pending_alias / intent_id
  6. 返回 ACK
```

回填语义应避免暗示正式落库已经完成。推荐文本方向：

```text
Memory accepted as pending atom 'draft_login_api_7f3a'.
It is readable during this run. Final memory generation will complete asynchronously.
```

---

## 9. MTP UPDATE 流程

`UPDATE` 不应直接把目标 alias 伪装成已更新，而应创建 pending revision。

建议流程：

```text
Agent:
  ⟪ UPDATE | mem_login_api_spec | instruction="补充刷新 token 逻辑" content=`...` ⟫

Koakuma:
  1. 解析并校验目标 alias
  2. 读取目标 atom 当前版本
  3. 生成 rev_mem_login_api_spec_b921
  4. 创建 PendingAtom(source_verb=UPDATE, target_alias=...)
  5. 写入 PendingAtomCache
  6. 构造 UpdateFocus，附带 pending_alias / intent_id
  7. 返回 ACK
```

`READ rev_mem_login_api_spec_b921` 应返回 pending revision 视图。是否让 `READ mem_login_api_spec` 同时展示 pending overlay，需要单独决定。

MVP 建议：

- `READ target_alias` 返回正式旧版本
- `READ rev_xxx` 返回 pending revision
- `READ rev_xxx` 不返回旧 atom 主体内容，避免 Agent 误以为旧内容已经完成修改
- 回填文案明确 pending revision alias

---

## 10. RuntimeAliasResolver 与 READ 解析优先级

应引入统一的 `RuntimeAliasResolver` 管理运行时 alias 解析。它负责协调三级缓存命中路径，而不是让 `READ`、`RUN`、`UPDATE` 各自维护分散解析逻辑。

目标解析顺序：

```text
L0 PendingAtomCache
  -> L1 KoakumaAtomCache
  -> L2 storage retrieve_by_aliases
```

语义：

- L0 负责当前 run 内尚未物化的 pending atom / pending revision
- L1 负责当前会话中已解析过的正式 `MemoryAtom`
- L2 负责冷查询长期存储
- `READ`、`RUN`、`UPDATE` 和 `context_refs` 后续都应通过该 resolver 获取 alias 对应对象、pending 视图或 canonical redirect
- pending alias 与正式 alias 使用不同命名空间，不允许 L0 覆盖同名正式 atom

`READ pending_alias` 的输出应包含状态：

```text
[draft_login_api_7f3a]
status: pending_generation
source: WRITE
title: Login API

content:
...

note: This is a runtime pending atom. Final memory generation is asynchronous.
```

状态变化后的 READ 行为：

- `PENDING` / `MATERIALIZING`: 返回原始 draft 内容
- `COMMITTED`: 返回正式 atom 内容，并提示 canonical alias
- `MERGED`: 返回合并目标 atom，并提示 canonical alias
- `UPDATED`: 返回更新后的目标 atom，并提示 canonical alias
- `TOUCHED`: 返回被触达的已有 atom，并提示 canonical alias
- `REDIRECTED`: 返回 canonical atom 内容，pending alias 只作为 redirect header 出现
- `DISCARDED`: 返回原始 draft 与未物化原因
- `FAILED`: 返回原始 draft 与错误原因
- `EXPIRED`: 返回过期提示；如仍有 canonical alias，应提示使用 canonical alias

---

## 11. Agent Loop 与多次 WRITE/UPDATE 聚合

当前 `ChatResult.write_focus` / `update_focus` 是单值语义。PendingAtom 方案应同步修正为列表语义：

```python
class ChatResult:
    write_foci: list[WriteFocus]
    update_foci: list[UpdateFocus]
    pending_aliases: list[str]
```

兼容策略：

- 保留旧字段一段时间，语义为最后一个 focus
- 新链路优先消费列表字段
- `InteractionPayload` 同步支持列表化 focus

否则长 run 内多个 `WRITE` / `UPDATE` 仍可能丢失。

---

## 12. 子 Agent Artifact 归属转移

子 Agent 场景是 PendingAtomCache 的核心价值之一。当前阶段不再沿用“harvest alias”的抽象，而是将子 Agent 产生的 pending artifacts 在 `CALL` 结束时显式转移给主 Agent。

建议子 Agent 执行期间：

```text
Sub-frame starts with pending_artifacts=[]

Sub Agent WRITE
  -> returns draft_xxx
  -> frame.pending_artifacts.append(draft_xxx)

Sub-frame ends
  -> transfer pending artifacts to main frame
  -> IPC return includes pending aliases
```

CALL 返回示例：

```text
<mtp_response status="success" type="ipc_return">
[Sub-Agent Reply]:
登录接口逻辑已完成，关键实现已提交为运行时 artifact。

[Artifacts Generated / Pending]:
- draft_login_api_7f3a (pending, readable now)
</mtp_response>
```

主 Agent 继续运行时，可以直接：

```text
⟪ READ | draft_login_api_7f3a ⟫
```

可见性规则：

- 子 Agent pending alias 在 `CALL` 结束后返回给主 Agent
- 在当前多智能体系统阶段下，返回后的 pending atom 视为主 Agent 的运行时资产
- 主 Agent 可以继续 `READ` 这些 pending alias，并在后续流程中引用它们
- provenance 必须保留，避免主 Agent 混淆 artifact 来源
- 兄弟子 Agent 间是否直接传递 pending alias 暂不作为本阶段目标；如需传递，应由主 Agent 显式持有并再分发

---

## 13. 权限与安全

PendingAtom 不能绕过正式记忆权限。

确定规则：

- PendingAtom 创建仍遵循现有 Koakuma MTP 权限流程
- 没有 `WRITE` 权限的 Agent 不能创建 WRITE pending atom
- 没有 `UPDATE` 权限的 Agent 不能创建 revision pending atom
- 没有 `READ` 权限的 Agent 不能读取不可见 pending atom
- pending atom 继承创建时的 identity 与权限快照
- 子 Agent artifact 返回主 Agent 后，在当前阶段转为主 Agent 的运行时资产
- 子 Agent artifact 返回主 Agent 时仍保留 `created_by_agent` / `created_by_frame`

建议 provenance 字段：

```python
created_by_agent: str
created_by_frame: str
permissions_snapshot: dict
source_command: str
```

---

## 14. 与 GenerationEngine 的结算协议

后台生成完成后，需要把 pending alias 结算到最终状态。

结算协议必须保留 Deduplicator 的四个原生决策状态：

```text
CREATE
UPDATE
TOUCH
DISCARD
```

这些状态不应被简单压扁成 `success` / `failed`，因为它们分别表达了不同的长期记忆语义：

- `CREATE`: draft 被物化为一条新的正式记忆
- `UPDATE`: draft 与已有记忆发生知识演化合并
- `TOUCH`: draft 与已有记忆高度重复，仅强化或触达已有记忆
- `DISCARD`: draft 被判定为不应物化，例如低质量、无价值或不可用

建议模型：

```python
class PendingAtomSettlement:
    pending_alias: str
    intent_id: str
    status: Literal["COMMITTED", "MERGED", "TOUCHED", "DISCARDED", "FAILED"]
    duplicate_decision: Literal["CREATE", "UPDATE", "TOUCH", "DISCARD"] | None
    final_alias: str | None
    final_uuid: str | None
    target_alias: str | None
    target_uuid: str | None
    canonical_alias: str | None
    canonical_uuid: str | None
    message: str
    error: str | None
    reason: str | None
```

`PendingAtomSettlement` 表示 pending intent 的结算视图，不应替代 generation 的完整返回结果。被动对话整理链路没有 pending alias，也不需要 settlement；只有 MTP `WRITE` / `UPDATE` 触发的主动写入链路，在 `WriteFocus` / `UpdateFocus` 携带 `pending_alias` 与 `intent_id` 时，才会生成 settlement。

因此，GenerationEngine 建议重置返回模型为 `MemoryGenerationResult`，并返回 `list[MemoryGenerationResult]`：

```python
class MemoryGenerationResult:
    intent_id: str | None
    pending_alias: str | None

    atom: MemoryAtom | None
    canonical_alias: str | None
    canonical_uuid: str | None

    duplicate_decision: Literal["CREATE", "UPDATE", "TOUCH", "DISCARD"] | None
    operation: Literal["created", "merged", "touched", "discarded", "updated", "failed"]

    settlement: PendingAtomSettlement | None
    message: str | None
    error: str | None
```

语义约束：

- `GenerationEngine` 是最接近 Deduplicator 决策与持久化结果的组件，应负责产出结构化 `MemoryGenerationResult`
- `KoakumaRuntime` 不应在事后根据返回 atom 重新推断 settlement
- `settlement` 是 pending intent 存在时附加产生的结算视图；被动生成结果中 `settlement=None`
- 现有 `list[MemoryAtom]` 返回值意义有限，可在 Phase 2 中替换为 `list[MemoryGenerationResult]`
- 为避免与 Alice runtime 中表示 LLM 文本输出的 `GenerationResult` 混淆，生成引擎侧建议使用 `MemoryGenerationResult` 命名

### 14.1 Deduplicator 决策映射

`PendingAtomCache` 的 settlement 状态与 Deduplicator 决策建议按以下方式映射：

| Deduplicator 决策 | Pending settlement | canonical 指向 | Agent 说明 |
| --- | --- | --- | --- |
| `CREATE` | `COMMITTED` | 新建正式 atom | 已创建新的正式记忆，后续使用新 alias |
| `UPDATE` | `MERGED` | 被合并更新的已有 atom | 内容已合并到已有记忆，后续使用目标 alias |
| `TOUCH` | `TOUCHED` | 被触达的已有 atom | 内容与已有记忆重复，未创建新记忆，后续使用已有 alias |
| `DISCARD` | `DISCARDED` | 无，或可选指向相近 atom | 内容未物化，说明丢弃原因 |

说明：

- `TOUCH` 应作为独立 settlement 保留，而不是并入 `MERGED`
- `DISCARD` 不应表现为系统错误，它是 generation / dedup 链路的业务决策
- `FAILED` 只表示系统执行失败，例如 LLM、storage、异常中断等
- `canonical_alias` 是 Agent 后续应该使用的正式 alias
- 注意区分 MTP `UPDATE` 指令与 Deduplicator `UPDATE` 决策：前者表示 Agent 请求修改指定记忆，后者表示生成出的 draft 与已有记忆发生知识演化合并

### 14.2 多 Pending Intent 的反向索引

多个 pending intent 理论上可以通过 pending alias / intent id 与正式记忆建立反向索引关系，因为 pending alias 是系统生成且互不相同的。

建议索引关系：

```text
pending_alias -> intent_id
intent_id -> generation request item
generation result item -> intent_id
intent_id -> settlement
settlement -> canonical_alias / canonical_uuid
canonical_uuid -> [pending_alias...]
```

其中：

- `pending_alias` 是 Agent 可见句柄
- `intent_id` 是系统内部稳定关联键
- generation 请求进入引擎时必须携带 `intent_id` / `pending_alias`
- generation 输出或 dedup 决策必须能回传对应 `intent_id`
- 同一个正式 atom 可以对应多个 pending alias，例如多个重复 WRITE 最终都 `TOUCH` 同一条记忆
- 一个 pending alias 只能对应一个 settlement 结果

反向索引用于支持：

- `READ draft_xxx` 时找到 canonical atom
- 前端展示 `draft_xxx -> mem_xxx`
- 调试多个 pending intent 被一次 flush 处理后的结算结果
- 后续事件流中追踪 pending intent 的完整生命周期

### 14.3 结算回填说明

settlement 应包含面向 Agent 的说明信息 `message`。示例：

```text
CREATE:
Pending atom 'draft_login_api_7f3a' has been committed as 'mem_login_api_spec'.
Use 'mem_login_api_spec' for future access.

UPDATE:
Pending atom 'draft_login_api_7f3a' has been merged into existing memory 'mem_login_api_spec'.
Use 'mem_login_api_spec' for future access.

TOUCH:
Pending atom 'draft_login_api_7f3a' duplicated existing memory 'mem_login_api_spec'.
No new memory was created. Use 'mem_login_api_spec' for future access.

DISCARD:
Pending atom 'draft_login_api_7f3a' was not materialized.
Reason: content was too low-value or invalid for long-term memory.
```

### 14.4 Settlement 回填方式

Settlement 回填应通过系统总线完成，避免 Patchouli 与 Alice runtime 形成直接对象依赖。

现有 `AsyncSystemBus` 已支持 `subscribe()` / `publish()`，因此 Phase 2 可以使用全局事件广播：

```text
GenerationEngine
  -> returns list[MemoryGenerationResult]

LibrarianCore / PatchouliRuntime
  -> extracts result.settlement
  -> global_bus.publish(PENDING_ATOM_SETTLED, settlement)

AliceRuntime
  -> subscribes PENDING_ATOM_SETTLED
  -> PendingAtomCache.apply_settlement(settlement)
  -> optionally updates KoakumaAtomCache

RuntimeAliasResolver
  -> reads PendingAtomCache redirect / settlement state during resolve()
```

建议事件名：

```python
class GlobalEvents:
    PENDING_ATOM_SETTLED = "alice.events.pending_atom.settled"
```

职责划分：

- `GenerationEngine` 不直接持有 bus，只返回 `MemoryGenerationResult`
- `LibrarianCore` 或 Patchouli runtime 负责把 settlement publish 到 `GlobalSystemBus`
- `AliceRuntime` 作为 runtime 聚合根负责订阅、退订与异常隔离
- `RuntimeAliasResolver` 不直接订阅总线；它只在解析 alias 时读取 `PendingAtomCache` 中的 settlement / redirect 状态
- `PendingAtomCache` 负责根据 settlement 更新状态、反向索引与 canonical redirect

该事件是 runtime 回填事件，不是正式记忆落库的成功判定。正式状态仍以 storage 为准。当前 `publish()` 是内存广播，没有持久化与重放能力；这与 Phase 2 仍以 runtime cache 为主的目标一致。未来引入 `PendingAtomLedger` 或统一事件流后，可将同一 settlement event 接入持久化管道。

如果后续要求 Patchouli 必须确认 Alice 已处理 settlement，可以演进为 RPC route，例如 `alice.public.pending_atom.settle`。但 MVP 更推荐 pub/sub：Alice 不在线或订阅失败时，只影响运行时 redirect，不阻塞正式记忆生成与落库。

### 14.5 仍需明确的问题

- 如果 extraction 阶段从一个 pending intent 中提取出多条 draft，是否允许一个 pending alias 对应多个正式 atom

MVP 可以先做 best-effort settle：

- `WriteFocus.pending_alias` 随 request 进入 generation
- `intent_id` / `pending_alias` 随 generation request item 进入引擎
- GenerationEngine 返回 settlement 时保留 `duplicate_decision`
- 通过 `intent_id -> pending_alias` 反向索引更新 `PendingAtomCache`

---

## 15. Prompt 与 MTP 语义更新

MTP 教学 prompt 需要同步更新，否则 Agent 会误解 pending alias。

建议加入说明：

```text
WRITE returns a pending alias.
You may READ the pending alias immediately during the current run.
Pending aliases are runtime handles, not guaranteed permanent memory aliases.
After generation completes, the pending alias may redirect to a canonical memory alias.
When a canonical alias is shown, use it for future READ/RUN/UPDATE calls.
If you need a stable long-term alias, inspect READ status later or SEARCH after generation.
```

中文语义：

```text
WRITE 返回的是运行时 pending alias，不是最终长期记忆 alias。
你可以在当前 run 内立即 READ 它。
当后台生成完成后，该 alias 可能结算为正式记忆、合并到已有记忆、被丢弃或失败。
如果系统返回 canonical alias，后续应使用 canonical alias 进行 READ / RUN / UPDATE。
```

---

## 16. 可观测性

PendingAtom 横跨 Alice runtime、Koakuma、Librarian、Perception、Generation，必须具备清晰 trace。

建议记录状态迁移：

```text
created -> pending
pending -> materializing
materializing -> committed/merged/updated/touched/discarded/failed
committed/merged/updated/touched -> redirected
redirected -> expired
```

建议日志字段：

```text
pending_alias
intent_id
run_id
frame_id
action_id
source_verb
status_from
status_to
final_alias
final_uuid
error
```

前端 MTPCard 后续可展示：

```text
WRITE accepted -> draft_xxx pending -> mem_xxx committed
```

---

## 17. 术语建议

暂定命名：

- `PendingAtom`: 运行时待物化原子
- `PendingAtomCache`: 内存可读缓存
- `PendingAtomLedger`: 持久意图日志
- `PendingAtomRegistry`: 对 cache/ledger 的统一接口
- `pending_alias`: Agent 可见的运行时 alias
- `intent_id`: 系统内部写入意图 ID
- `settlement`: 后台生成结算结果
- `duplicate_decision`: Deduplicator 的原生决策结果，取值为 `CREATE` / `UPDATE` / `TOUCH` / `DISCARD`
- `materialization`: pending intent 物化为正式记忆的过程
- `RuntimeAliasResolver`: 统一 alias 解析层
- `canonical_alias`: 正式记忆的规范 alias
- `redirect entry`: pending alias 物化后的兼容寻址入口
- `tombstone`: redirect grace period 结束后的过期记录
- `PendingAtomRenderer`: MVP 阶段的 pending atom 简单渲染器，未来由 MemoryCompiler 接管

---

## 18. 实现阶段规划与演进流程

本设计建议按“先打通运行时闭环，再补全结算精度，最后接入事件流与可观测性”的顺序推进。

### 18.1 Phase 1: Runtime MVP

目标：让 Agent 在同一 run 内可以立即读取 `WRITE` / `UPDATE` 产生的 pending alias，并让子 Agent artifacts 能稳定回填给主 Agent。

范围：

1. 新增 `PendingAtom` / `PendingAtomStatus` / `PendingAtomSettlement` 基础模型
2. 新增内存态 `PendingAtomCache`
3. 新增 `RuntimeAliasResolver`，抽离三级记忆缓存命中管理逻辑
4. `WRITE` 创建 `draft_{slug}_{short_id}` pending atom
5. `UPDATE` 创建 `rev_{target_alias}_{short_id}` pending revision
6. `READ` 优先解析 L0 `PendingAtomCache`
7. 原地修改 `ChatResult` / `InteractionPayload` 的 focus 字段，使其支持列表形式
8. 子 Agent `WRITE` / `UPDATE` 产生的 pending artifact 通过归属转移替代现有 harvest 实现，并在 IPC return 中返回给主 Agent
9. `context_refs` 读取 pending alias 时能通过 resolver 获取 pending 内容
10. MTP prompt 更新 pending alias 与 canonical alias 语义
11. 新增简单 `PendingAtomRenderer`，负责 MVP 阶段的 pending atom 渲染；未来由 MemoryCompiler 接管

验收标准：

- Agent 执行 `WRITE` 后立即 `READ pending_alias` 能读到 draft 内容
- Agent 执行 `UPDATE` 后立即 `READ rev_alias` 能读到 pending revision 内容
- 子 Agent 产生的 pending alias 能出现在 `CALL` 返回 payload 中
- 主 Agent 能读取子 Agent 返回的 pending alias
- 一轮 run 中多个 `WRITE` / `UPDATE` 不再互相覆盖
- 权限仍由 Koakuma 原有 MTP 权限流程控制
- `run_id` 覆盖完整主/子 frame 生命周期
- `READ rev_xxx` 不返回旧 atom 主体内容

### 18.2 Phase 2: Settlement 与 Canonicalization

目标：让 pending alias 在正式记忆落地后自动指向 canonical alias，并保留 Deduplicator 决策信息。

范围：

1. `WriteFocus/UpdateFocus` 携带 `intent_id` / `pending_alias`
2. `GenerationEngine` 返回 `list[MemoryGenerationResult]`
3. settlement 保留 `duplicate_decision`
4. `LibrarianCore` / Patchouli runtime 通过 `GlobalSystemBus.publish()` 回填 settlement
5. `AliceRuntime` 订阅 settlement event，并调用 `PendingAtomCache.apply_settlement()`
6. `PendingAtomCache` 根据 settlement 更新状态
7. pending alias 物化后进入 redirect grace period
8. `READ` / `RUN` / `UPDATE` / `context_refs` 对 redirect alias 做 canonicalization
9. `SEARCH` 只暴露 canonical alias，不暴露已 redirect 的 pending alias
10. `READ redirected_alias` 渲染 canonical atom，并提示后续使用 canonical alias

验收标准：

- `CREATE` 结算后，`READ draft_xxx` 返回新建正式 atom，主体 alias 为 canonical alias
- `UPDATE` 结算后，`READ draft_xxx` 返回被合并的已有 atom
- `TOUCH` 结算后，`READ draft_xxx` 返回被触达的已有 atom
- `DISCARD` 结算后，`READ draft_xxx` 返回未物化说明，不表现为系统错误
- `RUN` / `UPDATE` 使用 redirect alias 时，会自动操作 canonical atom

### 18.3 Phase 3: Lifecycle 与 GC

目标：明确 run 结束后的 redirect / tombstone 清理策略。

范围：

1. 当前 run 内保留 redirect
2. run 结束后保留短 TTL tombstone
3. tombstone 中尽量保留 canonical alias
4. TTL 后清理 tombstone
5. cancellation / shutdown 时将未结算 pending atom 标记为 `CANCELLED` 或 `EXPIRED`

验收标准：

- run 内旧 pending alias 始终能 redirect 到 canonical alias
- run 结束后 tombstone 能给出 final alias 或 SEARCH 指引
- GC 不会删除仍被当前 frame 栈引用的 pending alias

### 18.4 Phase 4: Event Stream 与 Ledger

目标：在项目统一建设事件流与可观测性时，引入持久化 pending ledger。

范围：

1. 引入 `PendingAtomLedger`
2. pending 生命周期状态迁移写入事件流
3. 支持跨进程恢复 pending intent
4. 支持前端展示完整 pending -> settlement 轨迹
5. 支持离线调试和 replay

该阶段不属于当前 MVP。

### 18.5 Phase 5: 高级多智能体共享

目标：在多智能体系统进入更复杂拓扑后，扩展 pending alias 的跨 frame 共享策略。

可能方向：

- 主 Agent 显式把已接收的 pending artifact 分发给另一个子 Agent
- 支持 sibling sub-agent 间通过主 Agent 代理共享 pending alias
- 引入更细粒度的 pending artifact capability / lease
- 对不同 Agent 权限快照做更严格的 provenance 检查

该阶段暂不作为当前目标。

---
