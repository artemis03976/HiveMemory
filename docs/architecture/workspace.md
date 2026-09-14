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
  - src/hivememory/server/deps.py
  - src/hivememory/system/runtime/workspace/
  - src/hivememory/system/assembler.py
  - src/hivememory/system/system.py
  - src/hivememory/patchouli/memory_library/stores.py
  - src/hivememory/patchouli/system.py
  - src/hivememory/patchouli/services/perception.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/mtp.md
  - docs/contracts/error-model.md
related_ideas:
  - docs/ideas/ae2-hivememory-architecture-analogy.md
related_docs:
  - docs/architecture/overview.md
  - docs/architecture/boundaries.md
  - docs/architecture/data-model.md
  - docs/architecture/decisions/0004-execution-path-derived-caches.md
  - docs/system/composition.md
  - docs/system/runtime-and-bus.md
  - docs/patchouli/memory-library.md
  - docs/patchouli/perception.md
  - docs/patchouli/retrieval.md
  - docs/patchouli/artifacts.md
  - docs/governance/security/identity-and-execution-safety.md
  - docs/system/attachments.md
last_reviewed: 2026-09-13
---

# Workspace 架构

本文是 Workspace 在当前系统架构中的事实入口，说明身份坐标、资源归属、运行时生命周期以及与 System、Patchouli、Gateway、Alice 和共享基础设施的边界。具体路由、事件字段和错误类型以[跨子系统契约](../contracts/subsystem-contracts.md)、[公开路由与事件](../contracts/routes-and-events.md)和[错误模型](../contracts/error-model.md)为准。

Workspace 在 W0 中是资源归属和访问硬边界，不是一个独立的子系统或一组按 Workspace 复制的 Runtime。System 进程只装配一套 Gateway、Patchouli、Alice、队列、注册表、调度器和 EventBus；需要隔离的资源在其最终寻址和授权处检查 WorkspaceIdentity。派生自 Workspace-owned 资源的视图缓存（Alice 的 L1 atom cache 与 profile cache）按派生源的 Workspace 坐标键控，键控规则见 [ADR-0004](./decisions/0004-execution-path-derived-caches.md)。

## 1. 为什么建立 Workspace：初步的“ME 网络”边界

HiveMemory 引入 Workspace，不是为了给现有对象再增加一个筛选字段，而是为了回答同一个问题的三个部分：谁在执行、资源归属于哪个稳定边界、一次后台或重试操作应当沿用哪一份身份事实。只有把这三部分放进同一个不可变坐标系，Topic、Memory、Artifact、WorkspaceAsset 以及它们的 binding/ref 生命周期才不会在共享进程运行时中相互串台。

Workspace 的架构意义是一个稳定的资源归属与访问边界，而不是 Agent 的永久身份。Agent、一次 Chat Run、子 Frame 和后台任务都只是暂时进入 Workspace 的执行者；资源所有权、访问硬边界和结算后的长期归属仍由 Workspace 及其领域 Store 负责。与此同时，Workspace 不会把所有基础设施复制成多套实例：queue、registry、scheduler、runtime 和 EventBus 继续共享——它们是不拥有领域状态的处理管道，谈不上按 Workspace 分区。缓存按所有权分两类：跨子系统事实源（WorkspaceAssetStore）由 System 持有并在最终寻址处校验；执行路径的派生视图缓存由执行子系统（Alice）持有并按派生源坐标键控。已经裁定为 Workspace-owned 的资源在最终寻址和授权处使用 WorkspaceIdentity。

在这个意义上，当前 Workspace 已经形成一个初步的“ME 网络”概念。这里的“ME 网络”是借用 AE2 的架构隐喻，不是代码中的独立类、网络进程或完整运行时；它指的是一片能够被稳定寻址、由同一资源归属边界约束、并通过明确交接承载执行结果的最小资源网络：

1. `WorkspaceIdentity` 与 `IdentityScope` 提供网络入口和资源归属坐标；
2. Topic、Memory、Artifact、WorkspaceAsset 是当前已落地的 Workspace-owned 资源节点，各自 Store 在最终读写处执行 hard boundary；
3. settle、generation、artifact 和后台 retry 是跨节点交接，领域载体保留同一 scope，不从进程当前 Workspace 重新推断；
4. 共享 System runtime 是网络的公共骨架，但不因此变成某个 Workspace 的私有命名域。

这与[《AE2 与 HiveMemory 的架构同构性》](../ideas/ae2-hivememory-architecture-analogy.md)形成正式的“当前事实—高阶设想”联系：Idea 文档解释 AE2 的网络、接口和子网为何能成为审查 HiveMemory 所有权、能力和执行边界的语言；本文只承接其中已经落地的 Workspace 资源边界，并把它标记为未来主网/子网体系的最小基础。完整的主网/子网拓扑、具有独立能力边界的子 Workspace、显式 Mount/Bridge、Capability Contract、独立工具与执行环境、配额/队列以及可恢复的子网生命周期尚未形成当前实现，仍以该 Idea 及后续独立 Plan 为准。

| 高阶架构维度 | 当前 Workspace 已形成的基础 | 完整主/子网系统尚缺的部分 |
|:---|:---|:---|
| 网络身份与资源寻址 | `IdentityScope`、Workspace 复合资源键、`main_workspace` 与内部隔离 seam | 用户可见的 Workspace 创建、切换和发现协议 |
| 网络存储与事实 | Topic、Memory、Artifact、WorkspaceAsset 的所有权和生命周期边界 | 跨 Workspace 的 Mount、Bridge、导入/导出及版本一致性 |
| 执行与结果回流 | Interaction、settlement、generation task 携带原始 scope，结果回到 Patchouli 边界 | 可持久化 Job Graph、子网内部执行器、恢复和 backpressure |
| 能力封装 | MTP、公开 Route 和窄 Asset port 提供现有交接基础 | 面向主网稳定暴露的 Capability Subnet 与版本化能力契约 |

因此，Workspace 当前应被理解为“初步 ME 网络边界”。后续若扩展 Workspace，必须先在 Idea/Plan 中明确所有权、Mount、能力和失败语义，再根据实际落地结果更新本文。

## 2. 在总体架构中的位置

`SystemAssembler` 是组合根。它创建全局运行时和注册表，再装配 Gateway、Patchouli、Alice 以及应用服务；`HiveMemorySystem` 只持有这张组件图并负责启停。Workspace 语义横跨这些边界，但不取得任何子系统的领域所有权：

```mermaid
flowchart TB
    IN["HTTP / Passive ingress / 内部测试入口"]
    SCOPE["IdentityScope\nActor + Workspace"]
    APP["System application services"]
    BUS["GlobalSystemBus"]
    GW["Gateway\n入口决策"]
    PA["Patchouli\nTopic / Memory / Artifact"]
    AL["Alice\nAgent run / MTP"]
    TOPIC["Patchouli Topic Store"]
    ASSET["WorkspaceAssetStore\n进程级唯一"]
    SHARED["共享 Runtime\nqueue / registry / scheduler / EventBus"]

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

Workspace 只在资源所有者需要它的地方生效。共享组件收到领域 payload 中的 `IdentityScope` 时，把它作为调用所需的不可变事实传递，不因此自动产生 Workspace 命名域、缓存副本或独立调度分区；Alice 的两个派生视图缓存按派生源坐标键控，属于其执行路径的私有运行时状态（见 [Alice](../alice/README.md)）。

## 3. 身份坐标

### 3.1 三个模型回答三个不同问题

| 模型 | 当前职责 | 不承担的职责 |
|:---|:---|:---|
| `ActorIdentity` | 谁在执行：`user_id`、`agent_id` 和可选 `team_id`。 | 不表示资源归属，也不单独授权 Workspace-owned 资源。 |
| `WorkspaceIdentity` | 资源归属于哪个用户和 Workspace：`owner_user_id`、`workspace_key`、`workspace_id`。W0 要求 key 与 ID 相同且非空。 | 不表示登录 session、grant 或永久 capability。 |
| `IdentityScope` | 一次操作的不可变硬边界，由 `ActorIdentity + WorkspaceIdentity` 组成，并校验 actor 用户与 owner 用户一致。 | 不携带 `interaction_id`、generation、run、frame、request 或 trace 等关联 ID。 |

`interaction_id`、`topic_id`、`memory_id`、`artifact_id` 和 work/task ID 仍由各自领域载体持有。这样可以避免把一个局部关联 ID 误当成公共身份，或在队列、缓存中派生出第二套 scope 模型。

### 3.2 默认入口和内部 seam

HTTP 层的用户导向身份选择（`user_id + workspace_id`）由统一请求头 `x-user-id` / `x-workspace-id` 承载，并在 `server/deps.py` 的 `resolve_request_identity_scope` 一次性冻结为 `IdentityScope`。该函数是默认身份解析的唯一入口：

1. 同一请求的 header 与 body/query 携带同一身份字段且不一致时显式拒绝（409），不静默择一；
2. `workspace_id` 只允许已声明的公共入口 `main_workspace`，其余值按不存在拒绝（404）；全部缺省时在此唯一回退 `DEFAULT_USER_ID` 与默认 Workspace；
3. Chat 等由具体 Agent 执行的 Agent action 必须显式提供 `agent_id`（缺失显式失败）；非 Agent action（管理读取、Topic 管理等）由 server 注入保留 `SYSTEM_AGENT_ID = "system"`，其语义是"没有具体 Agent 作为操作来源主体"，不是超级 Agent、不是前端可选 Agent，也不得写入 `MemoryAccessPolicy.target_agent_id`；
4. `/chat/stop` 通过 generation registry 使用创建时冻结的原始 scope：请求方选择只做 owner/workspace 校验，取消与事件发布沿用 run 的身份坐标。

应用服务公共方法只接受冻结的 `IdentityScope`，不再解析裸 `user_id`，也不得再次执行默认解析。W0 不提供 Workspace 创建或切换产品入口。`isolation_workspace` 仅由内部服务和隔离测试显式构造（`build_internal_identity_scope`），用于验证同一用户和 Agent 在两个资源域中的访问不会串扰。

下游不能读取进程级 `current_workspace`，也不能在 retry 时重新执行默认解析；后台任务必须从自己的领域 payload 恢复原始 scope，并在最终访问 Workspace-owned 资源时重新执行归属检查。

## 4. 资源归属与寻址

### 4.1 Workspace-owned 资源

Workspace 资源的最终寻址同时包含 WorkspaceIdentity 和资源 ID。复合键承载的是归属校验，而不是允许每个 Workspace 重新定义一套局部 ID：

| 资源 | 当前寻址结构 | 主要所有者 |
|:---|:---|:---|
| Topic | `IdentityScope + topic_id`（adapter 内部为 `WorkspaceTopicKey`） | Patchouli Perception / `ShortTermMemoryStore` |
| Memory | `WorkspaceMemoryKey(workspace_identity, memory_id)` | Patchouli `MidTermMemoryStore` / 长期存储 |
| Artifact | `WorkspaceArtifactKey(workspace_identity, artifact_id)`；`ArtifactRef` 同时带 WorkspaceIdentity | Patchouli `ArtifactStore` 及其适配器 |
| WorkspaceAsset | `WorkspaceAssetKey(workspace_identity, asset_id)`；外部使用当前 Store 的 opaque `WorkspaceAssetRef` | System-owned `WorkspaceAssetStore` |

`topic_id` 在领域上是全局唯一身份。正常创建路径由统一流程生成新的 UUID，两个 Workspace 可以使用相同标题，但不能把同一个 `topic_id` 作为两个合法 Topic 并存。调用方以 `IdentityScope + topic_id` 访问；`WorkspaceTopicKey` 仅由短期 adapter 在内部构造，用于归属校验和物理索引。

### 4.2 Memory 的归属

Memory 在 `MetaData.workspace_identity` 中保存唯一持久化归属；读取时先限定 owning Workspace，再由 Patchouli 的记忆策略判断 actor 是否可读。具体 policy 与检索规则属于 [MemoryLibrary](../patchouli/memory-library.md) 与 [Retrieval](../patchouli/retrieval.md)，本文只确认 Workspace 是 Memory 的唯一归属边界；存储层的 legacy 兼容解释分支已随存量数据迁移完成而删除，`workspace_identity` 是归属字段的唯一权威。

### 4.3 共享基础设施与派生缓存的键控规则

work queue、ordering/idempotency key、task/run registry、scheduler、runtime container 和 EventBus 维持进程级共享语义。领域 TaskSpec 可以携带唯一的 `IdentityScope`，但通用 WorkItem、WorkRecord 和 RuntimeEvent infrastructure 不把它解释为资源分区字段。`RuntimeEvent.workspace_id` 只是可选观测标签，不参与路由、订阅、sequence、授权或缓存分组。

缓存按所有权适用两条键控规则（[ADR-0004](./decisions/0004-execution-path-derived-caches.md)）：

1. **事实源按 ownership 寻址与校验**：跨子系统的 WorkspaceAssetStore 等事实源以 `WorkspaceIdentity` 参与资源复合键，在最终边界校验归属；
2. **派生视图按派生源坐标键控**：Alice 执行路径的 L1 atom cache 与 profile cache 派生自 Workspace-owned 资源，alias 索引按 `(WorkspaceIdentity, alias)`、profile key 按 `(WorkspaceIdentity, Actor 投影, alias)` 分区。分区解决"错误命中、无效覆盖和不必要的冷查询"，不替代授权——L1 atom cache 命中后仍由 resolver 重验 Workspace ownership 与 actor policy，profile cache 只复用同授权坐标内已通过 Patchouli 校验的结果。

两个缓存的当前边界与生命周期见 [Alice](../alice/README.md)。

## 5. WorkspaceAssetStore

### 5.1 所有权和生命周期

System 在 `_RuntimeBundle` 中只创建一个 `InMemoryWorkspaceAssetStore`。Store 是当前进程内 WorkspaceAsset、representation、opaque ref、幂等记录和 lease 的权威真相源，通过窄化的 Reader/Command port 提供给业务消费者。它不查询 Topic，也不负责 binding 或 settlement。

资产、表示和引用均为当前 Store 存活期内的运行时对象。`close_and_clear()` 进入不可逆关闭状态后清空 asset、representation、ref、operation token、幂等记录、REMOVED 记录和 lease bookkeeping；关闭后的 System 不能重新打开该 Store，必须重新装配进程并重新上传资源。

### 5.2 两级状态和命名命令

资产聚合状态为 `PROCESSING`、`READY`、`FAILED`、`REMOVED`；representation 状态为 `PENDING`、`PROCESSING`、`READY`、`FAILED`。状态只能由 Store 的命名 command 推进：

1. 创建操作按 `(WorkspaceIdentity, client_operation_id)` 幂等；相同操作的 metadata 不一致时报告冲突。
2. RAW representation 可以原子注册为 READY；其他 representation 先 PENDING，再由 `start_representation()` 签发 operation token 进入 PROCESSING。
3. `complete_representation()` 或 `fail_representation()` 校验 revision/token，并在同一临界区更新 representation 与 required representation 对应的资产聚合状态。
4. 只有资产 READY 且 preference 选中的 representation READY 时，`resolve_asset()` 和 `acquire_ready_representation()` 才成功。文档资产的 required representation 是 `EXTRACTED_TEXT`，RAW READY 本身不足以表示文档可用。
5. `REMOVED` 是不可逆终态。重复 remove 幂等，晚到的 parser callback、representation command 或 lease acquire 不能复活资产；活跃 lease 自己持有冻结内容，直到消费者显式 release。

WorkspaceAsset 不保存 `visibility`、`created_by_agent_id`、`created_by_team_id` 或 actor-policy target。同一 Workspace 内不同 Agent/Team 的资产访问结果一致；跨 Workspace 的 ref、asset key 或 URI 均不能绕过归属校验。

## 6. TopicAssetBinding 交接事实

`TopicAssetBinding` 是 Workspace 与 Patchouli Topic 之间需要在本文保留的 Topic 交接事实。它只记录 `asset_id`、opaque `asset_ref`、首次使用它的 `interaction_id` 和时间，不复制 WorkspaceAsset snapshot、representation 内容、actor-policy 字段或第二份 Topic/Workspace 坐标。

绑定只能在一轮成功 Interaction 中建立：上游消费者先根据用户明确选择的 READY ref 取得 representation lease，完成本轮 Interaction 后，再由 Patchouli 的 Topic 所有者提交首次 binding。调用方只应交接已经完成前置校验的 `(asset_id, asset_ref)`；Topic 写入口不会自行反查 AssetStore 或复制附件内容。重复使用同一资产只命中既有关系，不覆盖首次 Interaction 或首次绑定时间；上传、最近资产列表和 UI selection cache 不会产生 binding，没有 binding 的 WorkspaceAsset 是合法 orphan。

Asset remove 不回调 Patchouli，也不清理 binding。AssetStore 与 Topic 所有者之间不使用共同控制器、两阶段提交或额外协调器；binding 随所属 Topic 的 settle/evict 生命周期完成清理。Topic buffer 的结构、并发保护、compact/settle/evict 矩阵和 shutdown 批处理属于[Perception 与短期话题](../patchouli/perception.md)及 [MemoryLibrary](../patchouli/memory-library.md)，不在本文重复定义。

## 7. Scope 传播与跨边界交接

### 7.1 主动和被动入口

当前入口的 Workspace 交接可以概括为：

```text
用户导向身份选择（user_id + workspace_id；Agent action 附加 agent_id）
  -> server/deps.py 一次性校验并冻结 IdentityScope
  -> System application service（只接受 identity_scope）
  -> public route / 领域所有者
  -> 在 Workspace-owned 资源边界校验 scope
```

主动 Chat 与 Passive ingest 都在最外层冻结身份（Chat 与被动接入是 Agent action，必须携带具体 `agent_id`；被动接入的 `agent_id` 参与外部会话分桶命名），随后由领域载体携带 scope。`/chat/stop` 不是 Agent action：服务端用请求方选择完成 owner/workspace 校验后，通过 generation registry 复用创建时冻结的原始 scope 执行取消。Passive ingress 仍按自己的外部会话键缓冲并提交 Patchouli Interaction，但不因此建立第二套 Workspace 资源状态。下游不使用进程当前 Workspace 推断资源归属。

### 7.2 后台任务和重试

`InteractionSubmission`、`MemoryGenerationTaskSpec` 等领域 DTO 各自保存一份完整 `IdentityScope`，并携带所需的 interaction、intent、topic 或 task ID。codec 负责 scope 的完整 round-trip；Work Queue 只运输编码后的 payload 和执行状态，不解释 Workspace 领域模型。retry 从 payload 恢复原 scope 和领域 ID，再到真正的 Workspace-owned resource 边界执行授权；它不重跑默认 resolver、不读取进程当前 Workspace，也不改变身份坐标。队列的状态机和重试策略见[System 运行时与总线](../system/runtime-and-bus.md)。

## 8. System 生命周期与 shutdown

### 8.1 启动

System 的启动顺序为：

```text
Gateway -> Patchouli -> Alice -> Scheduler -> Passive Ingress
```

WorkspaceAssetStore 在 System 装配阶段创建，但在启动阶段不单独复制或按 Workspace 启动。它的可用性由 System 生命周期承载。

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

先停调度器和被动入口，避免 shutdown 期间继续接纳新的维护或摄入；Alice 在自身停止时清空其派生缓存，Alice 和 Patchouli 完成各自已接纳工作的 drain 后，才清空 WorkspaceAssetStore。这样 settlement consumer 可以在 drain 期间按既有交接约定用 task 中的 asset ref 反查 Store、持有 lease 并在完成后 release；Store 不调用 Patchouli controller 的 `wait_all`，也不查询 Topic 或 binding。`close_and_clear()` 幂等，重复 stop 不会重新打开或恢复任何状态。Patchouli 内部的 drain 顺序见[System 组合根与生命周期](../system/composition.md)。

### 8.3 失败边界

WorkspaceAssetStore 的清理不是队列可靠性或跨 Store 事务的替代品。若上游 shutdown 尚未完成，System 不应以提前清空 Store 来掩盖活跃 lease；如果关闭过程失败，System 报告失败事件而不是把未完成的消费者工作伪装成正常的 `SYSTEM_STOPPED`。

## 9. 当前边界与限制

- W0 只支持默认 `main_workspace` 的公开入口和内部 `isolation_workspace` 测试 seam，不提供用户可见的 Workspace 创建、切换、Mount、Bridge、Grant 或跨 Workspace sharing；
- 服务端当前没有完整认证/多租户安全沙箱，WorkspaceIdentity 是资源归属和业务硬过滤，不是独立的认证凭证；
- WorkspaceAssetStore、opaque ref 和 lease 只承诺当前进程生命周期，不提供跨重启恢复；已持久化的 Memory/Artifact 按各自存储契约存在；
- Alice 的 L1 atom cache 与 profile cache 是执行路径的派生视图（见 [ADR-0004](./decisions/0004-execution-path-derived-caches.md)）：不跨重启恢复，`AliceSystem.stop()` 时清空；profile cache 没有 TTL、更新事件或显式失效入口，Profile 修改在 LRU 驻留期内可能 stale；
- atom cache 返回原始 `MemoryAtom` 引用，可变性语义遵循[数据模型 ADR-0001](./decisions/0001-data-model-mutability-and-boundary-projection.md)，未做深冻结；
- WorkspaceIdentity 的传播不意味着所有组件都参与隔离。任何新增资源都必须先明确其所有者，再决定是否使用 Workspace 复合键；新增派生缓存时按派生源坐标键控（[ADR-0004](./decisions/0004-execution-path-derived-caches.md)），不能从 scope 的存在自动推导隔离。

## 10. 代码与测试入口

核心模型和资源键：

- [`identity.py`](../../src/hivememory/core/models/identity.py)、[`workspace.py`](../../src/hivememory/core/models/workspace.py)；
- [`topic.py`](../../src/hivememory/core/models/topic.py)、[`memory.py`](../../src/hivememory/core/models/memory.py)、[`artifact.py`](../../src/hivememory/core/models/artifact.py)、[`workspace_asset.py`](../../src/hivememory/core/models/workspace_asset.py)。

运行时和生命周期：

- [`InMemoryWorkspaceAssetStore`](../../src/hivememory/system/runtime/workspace/store.py)、[`workspace ports`](../../src/hivememory/system/runtime/workspace/ports.py)；
- Alice 派生缓存：[`KoakumaAtomCache`](../../src/hivememory/agent_runtime/aliases/cache.py)（端口见 [`AtomCachePort`](../../src/hivememory/agent_runtime/aliases/ports.py)）、[`AgentProfileCache`](../../src/hivememory/alice/runtime/profile_cache.py)；消费侧 resolver 见 [`RuntimeAliasResolver`](../../src/hivememory/agent_runtime/aliases/resolver.py) 与 [`AgentProfileResolver`](../../src/hivememory/alice/runtime/profile_resolver.py)；
- [`SystemAssembler`](../../src/hivememory/system/assembler.py)、[`HiveMemorySystem`](../../src/hivememory/system/system.py)；
- [`TopicAssetBinding`](../../src/hivememory/core/models/workspace_asset.py)、[`ShortTermMemoryStore`](../../src/hivememory/patchouli/memory_library/stores.py) 和 [`PerceptionFamiliar`](../../src/hivememory/patchouli/services/perception.py)。

代表性行为测试：

- [`tests/unit/core/models/test_workspace.py`](../../tests/unit/core/models/test_workspace.py)；
- [`tests/unit/system/runtime/workspace/test_store.py`](../../tests/unit/system/runtime/workspace/test_store.py)；
- Alice 派生缓存：[`tests/unit/agent_runtime/aliases/test_cache.py`](../../tests/unit/agent_runtime/aliases/test_cache.py)、[`tests/unit/alice/runtime/test_profile_cache.py`](../../tests/unit/alice/runtime/test_profile_cache.py)、[`test_alice_runtime_caches.py`](../../tests/unit/alice/runtime/test_alice_runtime_caches.py)；
- [`tests/integration/patchouli/test_memory_workspace_isolation.py`](../../tests/integration/patchouli/test_memory_workspace_isolation.py)、[`test_topic_access_chain.py`](../../tests/integration/patchouli/test_topic_access_chain.py)；
- [`tests/integration/system/test_workspace_asset_runtime.py`](../../tests/integration/system/test_workspace_asset_runtime.py)、[`test_workspace_access_propagation.py`](../../tests/integration/system/test_workspace_access_propagation.py)；
- cache 串扰与授权重验：[`tests/unit/agent_runtime/aliases/test_resolver.py`](../../tests/unit/agent_runtime/aliases/test_resolver.py)、[`tests/unit/alice/runtime/test_profile_resolver.py`](../../tests/unit/alice/runtime/test_profile_resolver.py)；
- 附件链路：[`tests/integration/system/application/test_workspace_asset_service.py`](../../tests/integration/system/application/test_workspace_asset_service.py)、[`tests/integration/system/test_workspace_asset_upload_api.py`](../../tests/integration/system/test_workspace_asset_upload_api.py)、[`test_workspace_asset_chat_selection.py`](../../tests/integration/system/test_workspace_asset_chat_selection.py)；完整入口见[Chat 附件链路](../system/attachments.md)。

相关入口：[总体架构](./overview.md)、[系统边界与所有权](./boundaries.md)、[数据模型与可变性边界](./data-model.md)、[System 组合根与生命周期](../system/composition.md)、[MemoryLibrary](../patchouli/memory-library.md)、[Perception 与短期话题](../patchouli/perception.md)、[Artifacts 与来源追踪](../patchouli/artifacts.md)、[Chat 附件链路](../system/attachments.md)和[Workspace 文档收口历史审计](../archive/plans/documentation-migration-finalization-audit.md)。
