---
title: Workspace 架构
status: current
owner: system
scope: workspace-identity-resource-ownership-and-runtime-lifecycle
code_paths:
  - src/hivememory/core/models/identity.py
  - src/hivememory/core/models/workspace.py
  - src/hivememory/core/models/topic.py
  - src/hivememory/core/models/memory.py
  - src/hivememory/core/models/artifact.py
  - src/hivememory/core/models/workspace_asset.py
  - src/hivememory/system/runtime/workspace/
  - src/hivememory/system/assembler.py
  - src/hivememory/system/system.py
  - src/hivememory/patchouli/memory_library/
  - src/hivememory/patchouli/system.py
  - src/hivememory/patchouli/services/perception.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
related_ideas:
  - docs/ideas/ae2-hivememory-architecture-analogy.md
related_docs:
  - docs/architecture/overview.md
  - docs/architecture/boundaries.md
  - docs/architecture/data-model.md
  - docs/system/composition.md
  - docs/system/runtime-and-bus.md
  - docs/patchouli/memory-library.md
  - docs/patchouli/artifacts.md
  - docs/patchouli/lifecycle.md
  - docs/governance/security/identity-and-execution-safety.md
last_reviewed: 2026-09-01
---

# Workspace 架构

本文是 Workspace 在当前系统架构中的事实入口，说明身份坐标、资源归属、运行时
生命周期以及与 System、Patchouli、Gateway、Alice 和共享基础设施的边界。具体路由、事件字段和错误类型以
[跨子系统契约](../contracts/subsystem-contracts.md)、[公开路由与事件](../contracts/routes-and-events.md)
和[错误模型](../contracts/error-model.md)为准。

Workspace 在 W0 中是资源归属和访问硬边界，不是一个独立的子系统或一组按 Workspace
复制的 Runtime。System 进程只装配一套 Gateway、Patchouli、Alice、缓存、队列、注册表、
调度器和 EventBus；需要隔离的资源在其最终寻址和授权处检查 WorkspaceIdentity。

## 1. 为什么建立 Workspace：初步的“ME 网络”边界

HiveMemory 引入 Workspace，不是为了给现有对象再增加一个筛选字段，而是为了回答
同一个问题的三个部分：谁在执行、资源归属于哪个稳定边界、一次后台或重试操作应当
沿用哪一份身份事实。只有把这三部分放进同一个不可变坐标系，Topic、Memory、Artifact、
WorkspaceAsset 以及它们的 binding/ref 生命周期才不会在共享进程运行时中相互串台。

Workspace 的架构意义是一个稳定的资源归属与访问边界，而不是 Agent 的永久身份。Agent、
一次 Chat Run、子 Frame 和后台任务都只是暂时进入 Workspace 的执行者；资源所有权、
访问硬边界和结算后的长期归属仍由 Workspace 及其领域 Store 负责。与此同时，Workspace
不会把所有基础设施复制成多套实例：cache、queue、registry、scheduler、runtime 和
EventBus 继续共享，只有已经裁定为 Workspace-owned 的资源在最终寻址和授权处使用
WorkspaceIdentity。

在这个意义上，当前 Workspace 已经形成一个初步的“ME 网络”概念。这里的“ME 网络”是
借用 AE2 的架构隐喻，不是代码中的独立类、网络进程或完整运行时；它指的是一片能够
被稳定寻址、由同一资源归属边界约束、并通过明确交接承载执行结果的最小资源网络：

1. `WorkspaceIdentity` 与 `IdentityScope` 提供网络入口和资源归属坐标；
2. Topic、Memory、Artifact、WorkspaceAsset 是当前已落地的 Workspace-owned 资源节点，
   各自 Store 在最终读写处执行 hard boundary；
3. settle、generation、artifact 和后台 retry 是跨节点交接，领域载体保留同一 scope，
   不从进程当前 Workspace 重新推断；
4. 共享 System runtime 是网络的公共骨架，但不因此变成某个 Workspace 的私有命名域。

这与[《AE2 与 HiveMemory 的架构同构性》](../ideas/ae2-hivememory-architecture-analogy.md)
形成正式的“当前事实—高阶设想”联系：Idea 文档解释 AE2 的网络、接口和子网为何能成为
审查 HiveMemory 所有权、能力和执行边界的语言；本文只承接其中已经落地的 Workspace
资源边界，并把它标记为未来主网/子网体系的最小基础。完整的主网/子网拓扑、具有独立
能力边界的子 Workspace、显式 Mount/Bridge、Capability Contract、独立工具与执行环境、配额/队列以及可恢复的子网生命周期尚未形成当前实现，仍以该 Idea 及后续独立 Plan 为准。

| 高阶架构维度 | 当前 Workspace 已形成的基础 | 完整主/子网系统尚缺的部分 |
|:---|:---|:---|
| 网络身份与资源寻址 | `IdentityScope`、Workspace 复合资源键、`main_workspace` 与内部隔离 seam | 用户可见的 Workspace 创建、切换和发现协议 |
| 网络存储与事实 | Topic、Memory、Artifact、WorkspaceAsset 的所有权和生命周期边界 | 跨 Workspace 的 Mount、Bridge、导入/导出及版本一致性 |
| 执行与结果回流 | Interaction、settlement、generation task 携带原始 scope，结果回到 Patchouli 边界 | 可持久化 Job Graph、子网内部执行器、恢复和 backpressure |
| 能力封装 | MTP、公开 Route 和窄 Asset port 提供现有交接基础 | 面向主网稳定暴露的 Capability Subnet 与版本化能力契约 |

因此，Workspace 当前应被理解为“初步 ME 网络边界”。后续若扩展 Workspace，必须先在 Idea/Plan 中明确所有权、Mount、能力和失败语义，再根据实际落地结果更新本文。

## 2. 在总体架构中的位置

`SystemAssembler` 是组合根。它创建全局运行时和注册表，再装配 Gateway、Patchouli、
Alice 以及应用服务；`HiveMemorySystem` 只持有这张组件图并负责启停。Workspace 语义
横跨这些边界，但不取得任何子系统的领域所有权：

```mermaid
flowchart TB
    IN["HTTP / Passive ingress / 内部测试入口"]
    SCOPE["IdentityScope\nActor + Workspace"]
    APP["System application services"]
    BUS["GlobalSystemBus"]
    GW["Gateway\n入口决策"]
    PA["Patchouli\nTopic / Memory / Artifact"]
    AL["Alice\nAgent run / MTP"]
    TOPIC["ShortTermMemoryStore\nSemanticBuffer"]
    ASSET["WorkspaceAssetStore\n进程级唯一"]
    SHARED["共享 Runtime\ncache / queue / registry / scheduler / EventBus"]

    IN --> SCOPE --> APP --> BUS
    BUS --> GW
    BUS --> PA
    BUS --> AL
    PA --> TOPIC
    APP --> ASSET
    SCOPE -. "最终资源边界重新校验" .-> TOPIC
    SCOPE -. "最终资源边界重新校验" .-> ASSET
    SCOPE -. "领域 payload 中传递；不建立分区" .-> SHARED
```

Workspace 只在资源所有者需要它的地方生效。共享组件收到领域 payload 中的
`IdentityScope` 时，把它作为调用所需的不可变事实传递，不因此自动产生 Workspace
命名域、缓存副本或独立调度分区。

## 3. 身份坐标

### 3.1 三个模型回答三个不同问题

| 模型 | 当前职责 | 不承担的职责 |
|:---|:---|:---|
| `ActorIdentity` | 谁在执行：`user_id`、`agent_id` 和可选 `team_id`。 | 不表示资源归属，也不单独授权 Workspace-owned 资源。 |
| `WorkspaceIdentity` | 资源归属于哪个用户和 Workspace：`owner_user_id`、`workspace_key`、`workspace_id`。W0 要求 key 与 ID 相同且非空。 | 不表示登录 session、grant 或永久 capability。 |
| `IdentityScope` | 一次操作的不可变硬边界，由 `ActorIdentity + WorkspaceIdentity` 组成，并校验 actor 用户与 owner 用户一致。 | 不携带 `interaction_id`、generation、run、frame、request 或 trace 等关联 ID。 |

`interaction_id`、`topic_id`、`memory_id`、`artifact_id` 和 work/task ID 仍由各自领域
载体持有。这样可以避免把一个局部关联 ID 误当成公共身份，或在队列、缓存中派生出
第二套 scope 模型。

### 3.2 默认入口和内部 seam

普通 Chat、Passive ingress 等顶层入口为当前用户解析一次 `main_workspace`，随后把
完整 `IdentityScope` 传入应用服务和领域交接。W0 不提供 Workspace 创建或切换产品入口。
`isolation_workspace` 仅由内部服务和隔离测试显式构造，用于验证同一用户和 Agent 在
两个资源域中的访问不会串扰。

下游不能读取进程级 `current_workspace`，也不能在 retry 时重新执行默认解析；后台
任务必须从自己的领域 payload 恢复原始 scope，并在最终访问 Workspace-owned 资源时
重新执行归属检查。

## 4. 资源归属与寻址

### 4.1 Workspace-owned 资源

Workspace 资源的最终寻址同时包含 WorkspaceIdentity 和资源 ID。复合键承载的是归属
校验，而不是允许每个 Workspace 重新定义一套局部 ID：

| 资源 | 当前寻址结构 | 主要所有者 |
|:---|:---|:---|
| Topic | `WorkspaceTopicKey(owner_user_id, workspace_id, topic_id)` | Patchouli `ShortTermMemoryStore` |
| Memory | `WorkspaceMemoryKey(workspace_identity, memory_id)` | Patchouli `MidTermMemoryStore` / 长期存储 |
| Artifact | `WorkspaceArtifactKey(workspace_identity, artifact_id)`；`ArtifactRef` 同时带 WorkspaceIdentity | Patchouli `ArtifactStore` 及其适配器 |
| WorkspaceAsset | `WorkspaceAssetKey(workspace_identity, asset_id)`；外部使用当前 Store 的 opaque `WorkspaceAssetRef` | System-owned `WorkspaceAssetStore` |

`topic_id` 在领域上是全局唯一身份。正常创建路径由统一流程生成新的 UUID，两个
Workspace 可以使用相同标题，但不能把同一个 `topic_id` 作为两个合法 Topic 并存。
`WorkspaceTopicKey` 只为读取、更新和拒绝跨 Workspace 访问提供明确的归属坐标。

### 4.2 Memory 的 Workspace ownership 与 actor read policy

Memory v2 在 `MetaData.workspace_identity` 中保存唯一持久化归属，在同一 Workspace
内部通过 `MemoryAccessPolicy` 执行 `PUBLIC`、`PRIVATE` 或 `TEAM` 的 actor 读取策略。
检索先执行 Workspace hard filter，再执行 actor policy；`PUBLIC` 是 owning Workspace
内公开，不是全局公开，也不授予写入、删除或 ownership 变更权限。历史缺少 Workspace
字段的记录只在对应用户的 `main_workspace` 受控兼容读取中解释。

### 4.3 共享基础设施不按 Workspace 分区

cache、work queue、ordering/idempotency key、task/run registry、scheduler、runtime
container 和 EventBus 维持进程级共享语义。领域 TaskSpec 可以携带唯一的
`IdentityScope`，但通用 WorkItem、WorkRecord 和 RuntimeEvent infrastructure 不把它
解释为资源分区字段。`RuntimeEvent.workspace_id` 只是可选观测标签，不参与路由、订阅、
sequence、授权或缓存分组。

## 5. WorkspaceAssetStore

### 5.1 所有权和生命周期

System 在 `_RuntimeBundle` 中只创建一个 `InMemoryWorkspaceAssetStore`。Store 是当前
进程内 WorkspaceAsset、representation、opaque ref、幂等记录和 lease 的权威真相源，
通过窄化的 Reader/Command port 提供给业务消费者。它不查询 Topic，也不负责 binding
或 settlement。

资产、表示和引用均为当前 Store 存活期内的运行时对象。`close_and_clear()` 进入不可逆
关闭状态后清空 asset、representation、ref、operation token、幂等记录、REMOVED 记录
和 lease bookkeeping；关闭后的 System 不能重新打开该 Store，必须重新装配进程并重新
上传资源。

### 5.2 两级状态和命名命令

资产聚合状态为 `PROCESSING`、`READY`、`FAILED`、`REMOVED`；representation 状态为
`PENDING`、`PROCESSING`、`READY`、`FAILED`。状态只能由 Store 的命名 command 推进：

1. 创建操作按 `(WorkspaceIdentity, client_operation_id)` 幂等；相同操作的 metadata
   不一致时报告冲突。
2. RAW representation 可以原子注册为 READY；其他 representation 先 PENDING，再由
   `start_representation()` 签发 operation token 进入 PROCESSING。
3. `complete_representation()` 或 `fail_representation()` 校验 revision/token，并
   在同一临界区更新 representation 与 required representation 对应的资产聚合状态。
4. 只有资产 READY 且 preference 选中的 representation READY 时，`resolve_asset()` 和
   `acquire_ready_representation()` 才成功。文档资产的 required representation 是
   `EXTRACTED_TEXT`，RAW READY 本身不足以表示文档可用。
5. `REMOVED` 是不可逆终态。重复 remove 幂等，晚到的 parser callback、representation
   command 或 lease acquire 不能复活资产；活跃 lease 自己持有冻结内容，直到消费者
   显式 release。

WorkspaceAsset 不保存 `visibility`、`created_by_agent_id`、`created_by_team_id` 或
actor-policy target。同一 Workspace 内不同 Agent/Team 的资产访问结果一致；跨 Workspace
的 ref、asset key 或 URI 均不能绕过归属校验。

## 6. Topic、SemanticBuffer 与 binding

### 6.1 SemanticBuffer 的所有权和保护

Patchouli 的 `ShortTermMemoryStore` 持有 Topic pool、最后活跃索引和可变
`SemanticBuffer`。Store 使用一把 `RLock` 覆盖这些状态的读取和写入；公开读取在锁内
投影为冻结的 `TopicData` 或 `TopicSnapshot`，调用方不会取得可变 buffer。锁内不执行
await、总线调用、生成、AssetStore 或 queue I/O。

同一 Topic 的写入以状态预约表达单写者窗口：

- Interaction 和 compact 预约 `PROCESSING`；窗口结束后释放回 `IDLE`；
- manual settle 预约 `FLUSHING`，在 generation admission 成功后提交驱逐，失败则恢复
  `IDLE` 并保留 Topic；
- automatic settle 使用一次 `freeze_and_evict`，只接受 `IDLE` Topic；
- LRU 维护只选择 `IDLE` 候选，busy Topic 被跳过；显式驱逐遇到 busy Topic 会失败。

W0 不为同一 Topic 增加 per-topic lock、content revision/CAS 或并行写入产品能力。

### 6.2 binding 是真实使用事实

`TopicAssetBinding` 位于 SemanticBuffer，只保存 `asset_id`、opaque `asset_ref`、首次
使用它的 `interaction_id` 和时间。它不保存 WorkspaceAsset snapshot、representation
内容、actor-policy 字段或第二份 Topic/Workspace 坐标。

绑定只能在一轮成功 Interaction 中建立：上游消费者先根据用户明确选择的 READY ref
取得 representation lease，完成本轮 Interaction 后，再由
`ShortTermMemoryStore.apply_interaction()` 在同一个临界区提交 block、首次 binding 和
metadata。按当前交接约定，调用方只应传入已经完成这项前置交接的
`(asset_id, asset_ref)`；`apply_interaction()` 不会自行反查 AssetStore 或复制内容。
当前 W0 尚无真实附件业务入口，lease 与 binding 的交接由现有集成测试边界验证。重复使用同一资产只命中既有关系，不覆盖首次
Interaction 或首次绑定时间。上传、最近资产列表和 UI selection cache 不会产生
binding；没有 binding 的 WorkspaceAsset 是合法 orphan。

Asset remove 不回调 Patchouli，也不清理 binding。AssetStore 与 ShortTermMemoryStore
之间不使用共同控制器、两阶段提交或额外协调器；binding 随 Topic 的 settle/evict 生命周期
完成清理。

### 6.3 settle、compact 和 evict

三者是不同操作：

- `compact` 压缩 buffer，保留 Topic 和已有 binding，并至少保留一个最近 block；
- `settle` 冻结 Topic 内容和 binding refs，形成记忆生成任务；manual settle 在任务被
  接纳后结束 Topic 生命周期，admission 失败时保留可重试的 Topic；
- `evict` 只从 Topic pool 移除运行时 buffer，不等于写入长期记忆。

`archive` 仍专用于中期记忆进入长期记忆库的生命周期，不作为 Topic settle 的别名。

## 7. 数据流与跨边界交接

### 7.1 主动和被动入口

```text
顶层入口
  -> 解析/构造 IdentityScope
  -> System application service
  -> GlobalSystemBus public route
  -> Gateway 形成入口决策
  -> Patchouli 读取或提交 Workspace-owned Topic/Memory/Artifact
  -> Alice 执行本轮 Agent run（主动链路）
  -> Patchouli finalize / settlement
```

Passive ingress 不运行 Alice，而是把外部经历按其会话键缓冲后提交 Patchouli Interaction。
两条入口都在最外层确定身份，随后由领域载体携带 scope；下游不使用进程当前 Workspace
推断资源归属。

### 7.2 后台任务和重试

`InteractionSubmission`、`MemoryGenerationTaskSpec` 等领域 DTO 各自只保存一份完整
`IdentityScope`，并独立保存所需的 interaction、intent、topic 或 task ID。codec 负责
完整 round-trip，Work Queue 只处理 payload bytes、状态和重试，不解释 Workspace 领域
模型。retry 从 payload 恢复原 scope 和领域 ID，再到真正的 Workspace-owned resource
边界执行授权；它不重跑默认 resolver、不读取进程当前 Workspace，也不改变身份坐标。

## 8. System 生命周期与 shutdown

### 8.1 启动

System 的启动顺序为：

```text
Gateway -> Patchouli -> Alice -> Scheduler -> Passive Ingress
```

WorkspaceAssetStore 在 System 装配阶段创建，但在启动阶段不单独复制或按 Workspace
启动。它的可用性由 System 生命周期承载。

### 8.2 停止

停止顺序为：

```text
Scheduler.stop
  -> PassiveIngress.shutdown_drain
  -> Alice.stop
  -> Patchouli.stop
       -> Interaction submission drain
       -> Active finalize drain
       -> Perception Topic settlement / generation drain
       -> Memory generation queue stop
  -> Gateway.stop
  -> WorkspaceAssetStore.close_and_clear
  -> SYSTEM_STOPPED
```

先停调度器和被动入口，避免 shutdown 期间继续接纳新的维护或摄入；Alice 和 Patchouli
完成已接纳的 interaction/finalize 及 Topic settlement/generation drain 后，才清空
WorkspaceAssetStore。这样 settlement consumer 可以在 drain 期间用 task 中的 asset ref
反查 Store、持有 lease 并在完成后 release；Store 不调用 Patchouli controller 的
`wait_all`，也不查询 Topic 或 binding。`close_and_clear()` 幂等，重复 stop 不会重新打开
或恢复任何 asset/ref。

### 8.3 失败边界

WorkspaceAssetStore 的清理不是队列可靠性或跨 Store 事务的替代品。若上游 shutdown
尚未完成，System 不应以提前清空 Store 来掩盖活跃 lease；如果关闭过程失败，System
报告失败事件而不是把未完成的消费者工作伪装成正常的 `SYSTEM_STOPPED`。

## 9. 当前边界与限制

- W0 只支持默认 `main_workspace` 的公开入口和内部 `isolation_workspace` 测试 seam，
  不提供用户可见的 Workspace 创建、切换、Mount、Bridge、Grant 或跨 Workspace sharing；
- 服务端当前没有完整认证/多租户安全沙箱，WorkspaceIdentity 是资源归属和业务硬过滤，
  不是独立的认证凭证；
- WorkspaceAssetStore、Topic buffer、opaque ref 和 lease 只承诺当前进程生命周期，
  不提供跨重启恢复；已持久化的 Memory/Artifact 按各自存储契约存在；
- W0 不包含真实附件上传、parser、Context Compiler 或基于附件的 Artifact promotion；
  当前只定义并测试 System-owned working set、窄化 Asset port 与 settlement ref 交接边界，
  尚无实际附件业务消费者；
- WorkspaceIdentity 的传播不意味着所有组件都参与隔离。任何新增资源都必须先明确其
  所有者，再决定是否使用 Workspace 复合键，不能从 scope 的存在自动推导隔离。

## 10. 代码与测试入口

核心模型和资源键：

- [`identity.py`](../../src/hivememory/core/models/identity.py)、[`workspace.py`](../../src/hivememory/core/models/workspace.py)；
- [`topic.py`](../../src/hivememory/core/models/topic.py)、[`memory.py`](../../src/hivememory/core/models/memory.py)、[`artifact.py`](../../src/hivememory/core/models/artifact.py)、[`workspace_asset.py`](../../src/hivememory/core/models/workspace_asset.py)。

运行时和生命周期：

- [`WorkspaceAssetStore`](../../src/hivememory/system/runtime/workspace/store.py)、[`workspace ports`](../../src/hivememory/system/runtime/workspace/ports.py)；
- [`SystemAssembler`](../../src/hivememory/system/assembler.py)、[`HiveMemorySystem`](../../src/hivememory/system/system.py)；
- [`ShortTermMemoryStore`](../../src/hivememory/patchouli/memory_library/stores.py)、[`SemanticBuffer`](../../src/hivememory/patchouli/memory_library/buffer.py)、[`PatchouliSystem`](../../src/hivememory/patchouli/system.py)。

代表性行为测试：

- [`tests/unit/core/models/test_workspace.py`](../../tests/unit/core/models/test_workspace.py)；
- [`tests/unit/system/runtime/workspace/test_store.py`](../../tests/unit/system/runtime/workspace/test_store.py)；
- [`tests/unit/patchouli/memory_library/test_buffer.py`](../../tests/unit/patchouli/memory_library/test_buffer.py)、[`test_binding_and_reservation.py`](../../tests/unit/patchouli/memory_library/test_binding_and_reservation.py)；
- [`tests/integration/patchouli/test_memory_workspace_isolation.py`](../../tests/integration/patchouli/test_memory_workspace_isolation.py)、[`test_topic_access_chain.py`](../../tests/integration/patchouli/test_topic_access_chain.py)；
- [`tests/integration/system/test_workspace_asset_runtime.py`](../../tests/integration/system/test_workspace_asset_runtime.py)、[`test_workspace_access_propagation.py`](../../tests/integration/system/test_workspace_access_propagation.py)。

相关入口：[总体架构](./overview.md)、[系统边界与所有权](./boundaries.md)、[数据模型与可变性边界](./data-model.md)、[System 组合根与生命周期](../system/composition.md)、[MemoryLibrary](../patchouli/memory-library.md) 和 [Workspace 文档收口待办](../todo/workspace-documentation-readiness.md)。
