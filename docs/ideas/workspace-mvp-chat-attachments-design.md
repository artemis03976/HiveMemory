---
title: Workspace MVP for Chat Attachments
status: idea
owner: system-patchouli-alice
scope: workspace-resource-scope-and-chat-attachments-foundation
related_current:
  - ../ROADMAP.md
  - ../architecture/boundaries.md
  - ../patchouli/artifacts.md
  - ../patchouli/memory-library.md
  - ../governance/security/identity-and-execution-safety.md
  - ../governance/baselines/identity-s0-threat-model-inventory.md
related_ideas:
  - ./ae2-hivememory-architecture-analogy.md
related_plans:
  - ../plans/README.md
last_reviewed: 2026-08-18
---

# Workspace MVP 与 Chat Attachments 初步设计

本文同时记录 `v0.6.2 W0 Workspace MVP` 的前置设计和 `W1 Chat Attachments` 对其公共契约的依赖，但二者将形成两份独立实施 Plan。当前首先推进 W0；本文不是已经冻结的实施 Plan，也不代表系统已经具备 Workspace 或附件能力。

本文要解决的问题是：在不提前实现完整 Agent Workspace、Mount、Capability、工具环境和队列管理的前提下，如何建立一个足以承载 Chat Attachment 的资源作用域和资产引用基础，并让后续 `v0.7.0 Document Ingestion` 可以复用同一套来源链。

## 1. 背景与现状

路线图已经为附件和长文档划定了顺序：

```text
Chat Attachment
  -> WorkspaceAsset
  -> runtime representations
  -> 当前 Chat context
  -> 若实际参与记忆生成，再提升为 Artifact provenance

Document Ingestion
  -> document artifact
  -> chunk / evidence
  -> 可审核候选记忆
```

`v0.6.2` 的文件不应因为上传就直接变成正式记忆或 Artifact。系统先在 Workspace 的进程内工作集中保存原始内容、解析内容和 metadata，再根据当前轮次需要编译为 Agent 上下文。只有被实际使用的内容随后参与 Memory CREATE/UPDATE 时，Materialization 才把当时使用的 representation 提升为不可变 Artifact 证据。大文件异步解析、文档切块和候选记忆属于后续阶段。

当前代码存在几个会阻碍该能力的作用域缺口：

- `Identity` 只有 user、agent、team 和兼容性的 session 字段，没有 Workspace 资源作用域；
- `ChatRequest`、`AgentRunContext`、`ExecutionFrame` 和 `InteractionPayload` 没有独立的 Workspace 坐标；
- `SemanticBuffer`/`TopicData` 主要按 user/topic 管理，尚无资产绑定关系；
- `BaseArtifact` 没有 Workspace 归属，而且现有 Artifact 引用可以携带物理 URI；附件不能直接复用它作为用户选择协议；
- `KoakumaAtomCache` 的核心 key 只有 alias/UUID，不能避免不同作用域之间的命中污染；
- Artifact filesystem adapter 允许同 ID 覆盖，并通过全局目录递归查找 Artifact，尚无 Workspace-aware index。

因此，Workspace MVP 的目标不是先增加一个容器类，而是冻结资源身份、访问范围、资产生命周期和引用解析契约。

## 2. 核心判断

### 2.1 Workspace 的最小定义

Workspace MVP 定义为：

> 稳定的资源命名空间、访问作用域和资产目录。

它不成为 MemoryLibrary、Alice Runtime、Tool Registry、Work Queue、预算系统和所有缓存的超级管理器。

完整 Workspace 将来可以拥有 Memory、Artifact、Agent、工具、执行环境、策略、预算、队列和审计边界；但 AE2 方向文档建议首先明确 Workspace、Mount、Capability、Run 和 Artifact 的所有权，不立即新增大量运行时类。本稿遵循这一收敛原则。

### 2.2 三种不同生命周期

Workspace、Topic Working Set 和 Run 必须分开建模：

| 层级 | 生命周期 | 主要职责 |
|:---|:---|:---|
| Workspace | 长期 | 拥有资源、定义资源命名空间和访问边界 |
| Topic Working Set | 话题级 | 记录某个话题可重复引用的资产 |
| Run / Context Frame | 单次执行 | 记录本轮实际选择和编译后的上下文 |

附件属于 Workspace，但通常通过 Topic binding 在当前话题内反复使用；属于 Workspace 不等于每轮自动注入。

### 2.3 WorkspaceAsset 的 MVP 生命周期

Workspace 命名空间可以长期存在，但 MVP 中的 `WorkspaceAsset` 只承诺进程内生命周期。上传成功仅表示资产及其 runtime representations 已进入当前服务进程的 Workspace working set，并可在该进程仍存活时解析；它不表示原文件、解析结果、资产目录或引用已经跨重启持久化。

这一限制与当前 Topic/SemanticBuffer 只存在于内存中的事实一致。MVP 必须保证：

- `WorkspaceAssetRef` 在进程重启后允许失效，调用方应得到明确的 `ASSET_NOT_FOUND` 或 `ASSET_EXPIRED`，不能静默取得其他同名资产；
- 已被 Topic/Interaction 引用的 asset 和 representation，至少保持到相关 Materialization 完成；
- 零 Topic binding 的资产仍是合法的 Workspace 运行期资源，但可以随进程结束一起消失；
- 一旦 representation 已提升为 Artifact，Artifact 按自己的持久化和不可变契约独立存在，不再依赖源 WorkspaceAsset 是否仍在内存中。

MVP 可以使用内存对象保存小型文本内容，不要求先建设 durable Blob/Object Store。进程内生命周期是明确的产品契约，不应由实现细节含糊决定。

## 3. 推荐的资源链路

```mermaid
flowchart LR
    U["上传原始文件"] --> A["WorkspaceAsset<br/>稳定逻辑身份"]
    A --> R["RAW representation<br/>进程内对象"]
    R --> P["EXTRACTED_TEXT representation<br/>进程内解析结果"]
    A -->|"首次 Chat 引用被接纳"| B["TopicAssetBinding<br/>当前话题可引用"]
    B --> C["Context Compiler<br/>授权、选择、预算、定位"]
    R --> C
    P --> C
    C --> V["Run-local Context View"]
    V --> PA["Prompt Assembler"]
    PA --> AG["Agent"]
    C --> X["ContextAssetUse<br/>本轮实际使用记录"]
    X --> M["Topic Materialization"]
    M -->|"DISCARD"| D["不生成外源 Artifact"]
    M -->|"Memory CREATE / UPDATE"| E["提升实际使用的 representation<br/>为来源 Artifact 快照"]
    E --> I["Memory provenance"]
```

这条链路包含五个重要区分：

1. 用户上传的逻辑资源不是 Artifact 本身；
2. 原始文件和解析文本是同一 WorkspaceAsset 内不同的 runtime representation；
3. 话题绑定只提供可引用性，不代表自动注入；
4. `ContextAssetUse` 只记录本轮实际进入上下文的 representation revision/hash，不记录所有曾经绑定或选择的资产；
5. Artifact 在 Materialization 边界按需创建，不是把可变的 WorkspaceAsset 原地改名或搬运。

## 4. 必须在 Workspace MVP 中裁定的问题

### 4.1 Workspace 身份、默认 Workspace 与兼容策略

Workspace 直接使用不可变的 `WorkspaceIdentity` 作为统一身份字段，不在 Workspace 聚合上分散保存三个可以互相漂移的裸字段：

```text
Workspace
  identity: WorkspaceIdentity
  display_name = "Main Workspace"
  workspace_kind = MAIN
  status = ACTIVE

WorkspaceIdentity
  owner_user_id
  workspace_key
  workspace_id
```

MVP 不启用独立的 opaque Workspace ID 生成机制，而是冻结以下不变量：

```text
WorkspaceIdentity.owner_user_id is not empty
WorkspaceIdentity.workspace_key is not empty
WorkspaceIdentity.workspace_id is not empty
WorkspaceIdentity.workspace_id == WorkspaceIdentity.workspace_key
```

因此默认 Workspace 的规范身份为：

```text
WorkspaceIdentity(
  owner_user_id = current_user_id,
  workspace_key = "main_workspace",
  workspace_id = "main_workspace",
)
```

不采用 `workspace_id=None`。否则每个 store、cache、event、work payload 和授权检查都必须重新计算 `workspace_id or workspace_key`，形成两套有效寻址字段。兼容逻辑只允许存在于最外层入口：调用方可以暂不传 Workspace，入口据此构造默认 `WorkspaceIdentity`；一旦进入业务链，`workspace_id` 必须非空，任何下游都不得再用 `workspace_key` 补救缺失 ID。

`workspace_key` 是同一 owner user 下稳定的逻辑 key，用于默认解析和未来 Workspace registry lookup；`workspace_id` 是对外协议、资源归属、cache/store/filter 使用的规范 ID。即使二者在 MVP 中等值，下游也只能使用 `workspace_id` 做资源寻址。Workspace 的完整身份是 `(owner_user_id, workspace_id)`，因此不同用户拥有同名的 `main_workspace` 不会发生冲突。

后端隔离测试使用另一组同样确定性的身份，例如：

```text
WorkspaceIdentity(
  owner_user_id = current_user_id,
  workspace_key = "isolation_workspace",
  workspace_id = "isolation_workspace",
)
```

未来启用独立 ID 时，新 Workspace 可以使用 `workspace_key="project_alpha"` 与 `workspace_id="ws_..."`，而现有 `main_workspace` 无需强制迁移。届时若允许修改 `workspace_key`，资源仍由不变的 `workspace_id` 寻址。MVP 不实现 Workspace rename，因此当前的等值约束不会造成身份漂移。

MVP 的默认行为是：

- 普通请求不传 Workspace 时，入口解析为当前用户的 `main_workspace`；
- 默认解析只发生一次，进入内部链路后必须显式传播 Workspace scope；
- 第一版不提供 Workspace 创建、切换和通信的前端产品能力；
- 后端仍允许测试或内部服务显式构造第二个 Workspace；
- 每次 Run 只有一个 active Workspace；
- 不实现 Mount、Bridge 或跨 Workspace fallback，跨 Workspace 读取一律拒绝。

因此，“不支持切换”只表示没有对应产品入口，不表示下游可以继续不知道当前 Workspace。两个 Workspace 的后端并存测试用于验证隔离不变量，而不是提前开放多 Workspace 产品面。

### 4.2 Identity、Team 与 Workspace 的关系

Identity、Team 与 Workspace 是相互正交的轴，不应互相替代、覆盖或嵌套：

```text
user_id       = 用户硬边界
agent_id      = 当前执行者身份
team_id       = 当前执行者所属的协作域
workspace_id  = 资产命名空间
```

这允许同一个 Agent 在不同 Run 中进入多个 Workspace，也允许一个 Workspace 接纳多个 Agent/Team 身份。Agent 不成为 Workspace 的子对象，Workspace 也不永久拥有 Agent Runtime。

继续让 `Identity` 只描述执行者，并用完整的 `WorkspaceIdentity` 描述资产域：

```text
Identity
  user_id
  agent_id
  team_id

WorkspaceAccessContext
  actor_identity: Identity
  workspace_identity: WorkspaceIdentity
  request_id?
  interaction_id?
```

该上下文是单次请求/Run 的作用域坐标，不是可绕过资源授权检查的永久 capability token。`WorkspaceIdentity.owner_user_id` 描述 Workspace 的所有者，`Identity.user_id` 描述当前执行者；两者相等是访问条件，不是可以互相覆盖的别名。第一版不需要实现显式 `WorkspaceGrant`；可以暂定同一 user 下的 Agent 均可进入该 user 的 Workspace，但只能访问当前 active Workspace 中符合 actor visibility 的资产。未来如需限制 Agent/Team 可进入哪些 Workspace，再增加独立的多对多 grant，不改变 Identity 或资产归属模型。

需要特别区分运行时状态和持久化事实：

- “当前正在访问哪个 Workspace”是运行时状态；
- “某项资产属于哪个 Workspace”是必须持久化的资产事实；
- “哪个 Agent/Team 创建了资产”是 provenance；
- “哪个 Agent/Team 当前可以访问资产”是 authorization。

因此 MemoryAtom 等资产不能只持久化现有 Identity。概念上它们应持有 `WorkspaceIdentity` 作为资产归属，并保存 `source_agent_id/team_id` 作为行为来源；存储模型可以把它平铺为 `owner_user_id + workspace_id`，同时保留必要的 `workspace_key` metadata，但不能重新引入三套独立身份来源。创建者字段记录历史事实，不能单独代替当前访问策略。

推荐的访问判定结构为：

```text
Allowed
  = same_user
  AND same_workspace
  AND actor_visibility_policy
```

不得使用 `same_user AND (same_workspace OR same_agent OR same_team)`，否则 agent/team 条件会绕过 Workspace 的资产隔离。

当前 `MemoryVisibility.WORKSPACE` 实际通过 `team_id` 做过滤，这一历史名称会与新的 Workspace 概念冲突。Workspace MVP 必须把 Workspace 作为独立的硬过滤轴；现有 visibility 只作为进入当前 Workspace 后的 actor 域访问策略，具体兼容映射见 4.8。

### 4.3 子系统所有权

Workspace 是共享系统中的逻辑资源作用域，不是 Gateway、Patchouli、Alice 或存储运行时的物理实例边界。MVP 不为每个 Workspace 创建独立的子系统、Runtime、MemoryLibrary、队列、总线或数据库实例，而是在同一套共享基础设施中通过显式 scope、复合资源键、过滤器和授权检查实现隔离。

推荐运行形态：

```text
HiveMemorySystem
  ├─ one GatewaySystem
  ├─ one PatchouliSystem
  │    └─ one MemoryLibrary
  ├─ one AliceSystem
  ├─ one Work Queue Runtime
  ├─ one RuntimeEventBus
  └─ one Workspace Resolver（MVP 确定性构造，无持久化 Registry）

每次请求
  └─ immutable WorkspaceAccessContext
       ├─ actor_identity: Identity
       └─ workspace_identity: WorkspaceIdentity
```

建议使用“统一地址空间，分散权威所有权”：

| 状态 | 权威所有者 | Workspace MVP 中的职责 |
|:---|:---|:---|
| Workspace Resolver / 默认 Workspace | System | 确定性构造身份、解析默认值并建立访问上下文 |
| WorkspaceAsset / representation / Topic binding | Patchouli runtime | 进程内资产 working set、话题工作集和读取路由 |
| Artifact / Memory provenance | Patchouli / ArtifactStore | 持久化证据快照、创建与版本来源链 |
| Context Compiler 输入视图 | Patchouli | 解析引用并生成本轮只读上下文 |
| Run / Frame / PendingAtom | Alice | 运行期状态，不取得长期资产所有权 |
| Chat run registry / WorkRecord | System / Work Queue | 控制状态，不属于 Workspace cache |
| Prompt rendering | Alice prompt layer | 只渲染已编译上下文，不负责 I/O 和授权 |

共享组件与逻辑分区方式如下：

| 对象 | 是否共享单例 | Workspace 隔离方式 |
|:---|:---:|:---|
| GatewayRuntime | 是 | 请求状态携带 `WorkspaceAccessContext` |
| PatchouliRuntime | 是 | Memory/Topic/Artifact 操作强制 user + workspace filter |
| AliceRuntime | 是 | AgentRunContext、Frame 和 MTP context 携带 scope |
| MemoryLibrary | 是 | Store key、filter 和索引包含 Workspace |
| Artifact/Blob adapter | 是 | metadata/index 负责授权；Blob 可作为内部存储细节去重 |
| Work Queue Runtime | 是 | payload、idempotency、ordering 和 retry 携带 scope |
| RuntimeEventBus | 是 | event 投影可携带 Workspace，只用于观测 |
| Provider/Model/Parser registry | 是 | MVP 继续作为全局能力定义 |
| cache | 共享容器 | key 包含 scope，命中后由资源所有者重新授权 |
| Run/Frame | 每次 Run 独立 | 保存冻结 scope，不成为 Workspace 长期资产 |

“不进行物理分区”是 MVP 的组合与领域契约决策，不是永久禁止存储优化。未来 adapter 可以按 Workspace 分库、分 collection 或分 shard，只要调用方仍使用相同的逻辑 scope 契约，且物理布局不会成为授权依据。

#### 4.3.1 禁止进程级当前 Workspace

任何共享 Runtime、Service、Familiar 或 Store 都不得保存可变的 `current_workspace`：

```text
PatchouliRuntime.current_workspace_id = ...  # 禁止
AliceRuntime.current_workspace_id = ...      # 禁止
```

否则两个并发 Run 会互相覆盖进程级字段。Workspace scope 必须作为不可变参数或 DTO 显式传递，并由 `ExecutionFrame` 保存当前 Run 自己的 scope。ContextVar 可以辅助日志和 trace 关联，但不能成为业务授权的唯一来源。

默认 Workspace 只能在最外层应用入口解析一次。进入链路后，任何缺少 Workspace scope 的资产操作都应失败，不能在 Gateway、Patchouli、Alice、Store 或 retry 中重新回退到 `main_workspace`。

#### 4.3.2 逻辑分区不变量

Workspace MVP 冻结以下规则：

1. 每个进程只有一套 Gateway、Patchouli、Alice、MemoryLibrary、Work Queue 和总线；
2. Workspace 不拥有子系统实例，只拥有逻辑资源；
3. 共享 Runtime 不保存可变的当前 Workspace；
4. 默认 Workspace 只在应用入口解析一次；
5. 进入业务链后传递不可变 `WorkspaceAccessContext`，不重新拼装裸身份字段；
6. 同一 Run 的 Gateway、prepare、Alice、finalize 使用同一 user/workspace scope；
7. 子 Frame、MTP 和后台 work 只能继承或缩小 scope，不能扩大或切换；
8. 所有 Workspace 资产查询硬过滤 `owner_user_id + workspace_id`；
9. cache key、幂等键、ordering key 和 retry payload 包含 Workspace scope；
10. cache 命中、Artifact 读取、MTP、retry 和 finalize 由最终资源所有者重新授权；
11. RuntimeEvent 可以记录 Workspace，但不能参与授权；
12. 未来允许存储层透明物理分片，但不得改变以上领域契约。

这与[身份隔离与执行安全治理](../governance/security/identity-and-execution-safety.md)中“request/run/frame/work item 携带不可变身份快照和 scope”“资源所有者重验”“缓存不承载授权”的规则一致。Workspace MVP 需要把这些治理要求从后续目标提升为本功能的前置不变量，否则新增的资产轴会放大现有身份和缓存污染风险。

### 4.4 Workspace、Topic 和 Run 的边界

WorkspaceAsset、Topic 和单次 Chat Run 只通过显式引用建立关系。上传本身只创建 Workspace 资产，不创建 Topic 或 Run 关系。

需要区分四种状态事实：

| 事实 | 建立时机 | 生命周期 | 推荐载体 |
|:---|:---|:---|:---|
| 资产存在于 Workspace | 上传完成 | 当前进程 / Workspace 级 | `WorkspaceAsset` |
| 用户为本轮选择资产 | 发送 ChatRequest | Run/Interaction 级 | `ChatRequest.asset_refs` / `RunAssetSelection` |
| 资产与 Topic 关联 | 引用首次被该 Topic 接纳 | Topic 级 | `TopicAssetBinding` |
| 内容被本轮实际使用 | Context Compiler 成功选入上下文 | Run/Interaction 级 | `ContextAssetUse` |

刚上传的资产允许不存在任何 Topic binding：

```text
WorkspaceAsset
  topic_bindings = 0
```

“未绑定”是关系缺失，不是 WorkspaceAsset 的特殊生命周期状态。同一个资产可以始终没有 Topic binding，也可以同时与多个 Topic 建立独立关系。

#### 4.4.1 新上传附件与本轮选择

上传完成时，Chat Run 通常尚未创建，因此上传接口不能直接建立服务端 Run binding。推荐由 Chat Composer 持有当前草稿选择：

```text
upload
  -> 返回 WorkspaceAssetRef
  -> Chat Composer 自动加入当前草稿的 asset_refs
  -> 用户发送 message + asset_refs
  -> ChatRequest 显式提交本轮选择
```

“新附件自动挂载本轮”是一项前端便利行为，不是服务端根据上传时间推断的隐式规则。服务端不能维护 Workspace 级 `pending_attachments`，也不能把“最近上传的附件”自动塞入下一次请求，否则多标签页、并发 Chat 或上传后切换话题会把资产挂到错误的 Run。

后续轮次同样由 `ChatRequest.asset_refs` 显式表达选择。Topic Working Set 只提供可再次选择的资产列表，不默认把所有绑定资产注入 prompt；未来若需要长期自动注入，应另行增加 `pinned` policy。

#### 4.4.2 路由、校验与 Topic binding

首轮 Chat 可能还没有真实 `topic_id`，但附件绑定总是在路由完成后发生：

```text
chat(message, asset_refs, target=NEW_TOPIC)
  -> 预校验 asset refs 的 user/workspace 归属和资源状态
  -> Gateway/Patchouli 确定唯一 Topic
  -> ensure TopicAssetBinding
  -> Context Compiler 编译本轮资产
  -> Alice 执行 Agent run
```

路由前可以验证资产是否存在、是否属于当前 user/workspace、是否已删除，以及引用格式是否合法；但不建立 Topic binding，也不读取附件正文参与路由。

TopicAssetBinding 的建立边界定义为：

> 资产引用已经通过 user/workspace 校验，并被一个已完成路由的 ChatRequest 接纳。

不必等待 Agent 成功回复，也不必等待模型实际消费内容。解析失败、模型失败或 Run 被取消后，binding 默认保留，因为它记录用户已经把该资产带入这一 Topic 的意图；真正被模型消费的内容另由 `ContextAssetUse` 表达。无效、越权、已删除或完整性校验失败的引用不能建立 binding。

Topic binding 必须幂等：

```text
UNIQUE(owner_user_id, workspace_id, topic_id, asset_id)
```

两个并发 Chat Run 首次引用同一资产时只能产生一条 binding。同一 WorkspaceAsset 可以分别绑定多个 Topic；从某个 Topic detach 只删除对应 binding，不删除进程内资产、其他 Topic 的关系或已经独立生成的 Artifact。

#### 4.4.3 Run 选择与实际使用

Topic binding、用户本轮选择和模型实际使用是三种不同事实：

```text
TopicAssetBinding
  = 这个 Topic 以后可以继续选择该资产

RunAssetSelection
  = 用户本轮请求选择了该资产

ContextAssetUse
  = Context Compiler 最终使用了哪个 representation 的哪些部分
```

`RunAssetSelection` 应关联稳定的 `interaction_id` 或 Chat operation identity，而不是只依赖低层 `frame_id`。`ContextAssetUse` 记录实际 representation 的 revision/hash、locator、token 数和编译状态，并作为 Materialization 的来源输入；在提升完成前它不持有 Artifact 身份。

如果用户选择了附件，但解析失败或预算裁剪导致内容没有进入 Agent 上下文，该资产仍可以存在于 Topic Working Set，但不能被记录为本轮回答的有效来源。

#### 4.4.4 零 binding 资产与附件路由限制

用户上传后没有发送消息，或在发送前从 Composer 移除附件，会留下零 Topic binding 的 WorkspaceAsset。这是合法的进程内资产，不是 orphan Artifact：它可以在当前进程中以后从 Workspace 资产选择器再次使用，也可以由用户显式删除；无需为了零 binding 资产建立耐久保存或 provenance hold。

“绑定发生在 Topic 路由之后”成立的前提是 MVP 的 Topic 路由不依赖附件正文。对于 attachment-only Chat，第一版应要求使用当前显式 Topic，或者在没有当前 Topic 时直接创建新 Topic；MVP 不读取附件正文决定 Topic。未来若需要 attachment-aware routing，应单独设计 pre-route hint 或两阶段路由，不能让 Topic binding 与内容路由形成循环依赖。

### 4.5 WorkspaceAsset、Artifact 与内容对象的边界

`WorkspaceAsset` 与 `Artifact` 是两套用途正交的体系：

| 概念 | 用途 | MVP 生命周期 | 用户是否直接选择 |
|:---|:---|:---|:---:|
| `WorkspaceAsset` | Workspace 中可反复引用的逻辑工作资源 | 仅当前进程 | 否 |
| `WorkspaceAssetRef` | Chat/Topic 对逻辑资产的 opaque 引用 | 随所指资产有效 | 是 |
| `AssetRepresentation` | RAW、EXTRACTED_TEXT 等运行期内容对象 | 仅当前进程 | 由编译器选择 |
| `Artifact` / `ArtifactRef` | Memory 的不可变证据、创建与版本链 | 按 ArtifactStore 契约持久化 | 否 |

`WorkspaceAsset` 内部可以同时保存原始内容对象、解析后的内容对象和 metadata。MVP 不要求这些对象先转为 Blob 或 Artifact，也不承诺在进程重启后恢复。Artifact 则是已经进入持久化记忆资产证据链的独立、append-oriented 记录，不属于 Workspace 的可选择工作集，用户不能把现有 `ArtifactRef` 当成附件再次挂载。Artifact 不在物理上从属于某一个 Memory；同一来源 Artifact 可以被一次 Materialization 生成的多个 MemoryCreation/Version Artifact 引用。

两套体系只在 Topic Materialization 边界发生单向提升：

```text
WorkspaceAssetRef
  -> ContextAssetUse(asset_id, representation_id, revision, content_hash, locators)
  -> Generation Outcome
       DISCARD
         -> 不创建外源 Artifact
       CREATE / UPDATE
         -> snapshot actually used representation
         -> DocumentArtifact / other source Artifact
         -> MemoryCreationArtifact or MemoryVersionArtifact source refs
         -> MemoryAtom
```

这里的 promotion 不是把 WorkspaceAsset 原地“转换”为 Artifact，而是根据 `ContextAssetUse` 创建一份不可变证据快照。只提升实际进入 Agent 上下文并参与该次 Memory CREATE/UPDATE 的 representation；仅上传、仅 Topic binding、仅本轮选择，或因预算/解析失败未进入上下文的内容都不生成外源 Artifact。

Materialization 必须读取 `ContextAssetUse` 记录的确定 revision/hash，不能重新解析 WorkspaceAsset 的“最新内容”。如果记录的版本已经不可用，必须明确失败或标记来源不完整，不能悄悄使用另一版本。相同 materialization operation 的 retry 必须幂等，不能为同一份来源重复生成多个 Artifact；同一次 Materialization 产生多条 Memory 时，应复用同一个已提升来源 Artifact。Artifact 创建后即使源 WorkspaceAsset 随进程结束消失，证据链仍可独立读取。

当前 Artifact 写入是 best effort。正式 Chat Attachments Plan 必须裁定外源来源提升失败时的提交语义：要么阻止依赖该来源的 Memory 写入，要么显式保存 `provenance_incomplete`；不能静默丢失来源后仍声称记忆 provenance 完整。

### 4.6 Artifact 类型、资产引用和寻址规则

Artifact 类型应由被保存内容的语义决定，而不是由“聊天附件”或“文档摄入”入口决定：

- 当前 MVP 只支持文档型附件；当其实际参与 Memory CREATE/UPDATE 时，提升为 `DocumentArtifact`；
- 不把所有未来 WorkspaceAsset 一律映射为 `DocumentArtifact`；非文档来源应使用相应 Artifact 类型；
- 不用 `unknown` 文档类型承载任意附件。长期如出现多种外源载体，可以再抽取 `ExternalSourceArtifact` 基类，MVP 不提前实现；
- Chat Attachment 与 Document Ingestion 共享 DocumentArtifact 的外源快照性质，但前者由运行期实际使用触发，后者由正式摄入与多级文档溯源触发。

建议为外源 Artifact 补充来源 metadata：

```text
origin: CHAT_ATTACHMENT | DOCUMENT_INGESTION | URL_FETCH | REPO_IMPORT
source_asset_id?
source_asset_revision?
parser_id?
parser_version?
source_content_hash
```

其中 `source_asset_id` 只用于追踪来源，不使 Artifact 依赖 WorkspaceAsset 存活。页码、标题路径、行号、quote 等 locator 长期更适合作为 Memory/Artifact 间的 `SourceEvidenceRef` 关系信息，而不是把每条引用位置都塞入完整文档 Artifact 本体；该关系模型留给 `v0.7.0` provenance contract 最终裁定。

初版引用规则为：

- 文件名只是展示字段，同名文件可以并存；
- 用户侧只使用 opaque `WorkspaceAssetRef`，不暴露文件系统路径或 `ArtifactRef.uri`；
- 内部资源键至少包含 `(owner_user_id, workspace_id, asset_id)`；用户侧 opaque ref 不直接暴露这些字段；
- `ContextAssetUse` 必须锁定 `representation_id + revision + content_hash`，以支持可复现提升；
- asset/artifact ID 都不是授权凭证；
- 每次 resolve 都要重新验证 user、Workspace、Topic 关系和资源状态；
- 已有 `ArtifactRef` 只服务 Memory provenance 和内部历史读取，不进入 Chat 附件选择协议。

### 4.7 Scope 的端到端传播

Workspace 身份域必须在 Active Chat 进入业务链前锁定。推荐顺序为：

```text
HTTP / other adapter
  -> 构造 actor Identity
  -> Workspace Resolver 解析 requested workspace 或 main_workspace 为 WorkspaceIdentity
  -> 验证 Workspace owner/status
  -> 冻结 WorkspaceAccessContext
  -> register ChatGenerationRun
  -> Gateway
  -> Patchouli prepare
  -> Alice run
  -> Patchouli finalize / cleanup
```

“锁定”表示一次 Run 的 user/agent/team/workspace 坐标不可被下游替换，不表示后续可以省略授权。资源所有者仍需根据该上下文重新验证 Workspace 状态和资源归属。

Active Chat 至少需要经过以下载体：

```text
WorkspaceAccessContext
  -> ChatGenerationRun
  -> GatewayExecutionState
  -> Patchouli PreparedAgentRun
  -> AgentRunContext
  -> ExecutionFrame
  -> MTPExecutionContext
  -> InteractionPayload
  -> Memory / Artifact domain work payload
  -> RuntimeEvent projection
```

各阶段要求如下：

| 阶段 | Workspace scope 职责 |
|:---|:---|
| ChatApplicationService | 解析默认 Workspace，创建唯一不可变 AccessContext |
| ChatGenerationRun | 保存 scope 或 scope fingerprint，供 cancel/status/event 校验 |
| GatewayExecutionState | 将 AccessContext 作为初始化后不可覆盖的请求事实 |
| Patchouli prepare | 在该 scope 内路由 Topic、检索 Memory、绑定资产和编译上下文 |
| PreparedAgentRun | 冻结 prepare 结果与原始 AccessContext，供 finalize/cleanup 复用 |
| Alice AgentRunContext | 向主 Frame 提供相同 scope |
| ExecutionFrame / MTP | 继承相同或更窄的 scope，不允许更换 Workspace |
| Patchouli finalize | 只消费 PreparedAgentRun 中的 scope，不能接受另一组裸身份字段 |
| background work/retry | 序列化 scope snapshot，并在每次 attempt 重新授权 |
| RuntimeEvent | 投影 workspace_id 供观测，不作为授权输入 |

公共用例签名应优先接收完整上下文：

```text
prepare_agent_run(
  access_context,
  user_message,
  gateway_decision,
  ...
)
```

不应继续在跨子系统边界重复传递 `user_id + agent_id + workspace_id` 后由下游重新构造 Identity。当前 Active Chat 在 ChatApplicationService 中构造 Identity，随后又向 Patchouli prepare 传递裸 user/agent/session 字段；Workspace MVP 应消除这个身份漂移点。

同一 Run 必须满足：

```text
Gateway 使用的 scope
  == Patchouli prepare 读取的 scope
  == Alice/子 Frame 使用的 scope
  == Patchouli finalize 写入的 scope
```

`ChatGenerationRun` 虽然是 System 控制状态而不是 Workspace 资产，也应保存冻结 scope。cancel/status 查询不能只依赖 `generation_id`；调用者至少需要匹配 Run 的 user/workspace 边界，避免 generation ID 被另一个 Workspace 用来停止或观察运行。

子 Agent 默认继承父 Run 的 user/workspace hard boundary，只能通过经过授权的 context refs 缩小或选择资产。自然语言、alias、CALL 或默认值都不能扩大 scope。

通用 Work Queue 不必立即改变基础 `WorkItem` 结构，但所有附件、Interaction、Memory Generation 和 Lifecycle 相关领域 payload、ordering key 和 idempotency key 都必须包含 Workspace scope。后台任务使用可序列化的 scope snapshot，而不是依赖进程内 AccessContext 对象；每次 retry 重新解码并授权。全局 maintenance 如需处理多个 Workspace，应显式逐 Workspace 执行 scoped operation，不能以无 Workspace filter 的全库扫描替代所有权检查。

### 4.8 授权、opaque ID 与 cache 命中

当前系统仍处于单用户/单 Workspace 假设，尚无完整认证和多租户隔离。Workspace MVP 可以先建立内部访问契约，但不能将 opaque ID 误写成安全边界。

资源读取必须由资源所有者根据 `WorkspaceAccessContext` 执行三阶段判断：

```text
1. MUST access.workspace_identity.owner_user_id == access.actor_identity.user_id
2. MUST resource.workspace_identity == access.workspace_identity
3. MUST match_actor_visibility(access.actor_identity, asset_policy)
```

Workspace hard filter 不受 `agent_id`、`team_id` 或 visibility 结果影响，也不允许拿到 asset/artifact ID 后跳过。第一版没有跨 Workspace 通信，因此任何 workspace mismatch 都直接拒绝。

新的 Workspace 轴会使现有 `MemoryVisibility` 出现语义冲突。推荐的目标语义是：

```text
PRIVATE    -> 仅创建 Agent
TEAM       -> team_id 匹配
WORKSPACE  -> 当前 Workspace 内的任意合法 Agent
```

现有序列化值可以在迁移期解释为：

```text
现有 PRIVATE    -> PRIVATE
现有 WORKSPACE  -> TEAM（历史兼容值，未来应改名）
现有 PUBLIC     -> WORKSPACE
```

MVP 不一定要立即修改所有枚举值，但契约必须明确：现有 `WORKSPACE=team_id` 不是新的 Workspace 隔离语义，`PUBLIC` 也不能绕过 user/workspace hard filter。

缓存 key 至少以 `(owner_user_id, workspace_id, ...)` 开头。若缓存内容或 alias resolve 结果受 agent/team visibility 影响，还必须包含相应 actor scope，或者在每次命中后重新授权。缓存命中不能绕过资源所有者的三阶段检查。

### 4.9 幂等、并发和提交边界

推荐将以下身份分开：

- 上传幂等键：`owner_user_id + workspace_id + actor + client_operation_id`；
- 内容完整性：raw content hash；
- 解析结果身份：`raw_hash + parser_id + parser_version + options_hash`；
- Topic binding 唯一键：`workspace_id + topic_id + asset_id`；
- Artifact promotion 幂等键：`materialization_operation_id + representation_revision/hash + artifact_role`。

同一进程内，相同 upload operation 重试应返回同一个 WorkspaceAsset。相同内容 hash 可以用于运行期内容对象去重，但不能自动合并两个用户层面的逻辑资产，也不形成跨重启恢复承诺。

附件上传成功的必要条件是 WorkspaceAsset 及其 MVP 所需 runtime representations 已原子地进入当前进程 working set，并可立即通过返回的 ref 解析。上传路径不创建 Artifact，也不要求内容对象或资产目录耐久落盘。若注册资产后解析失败，应通过明确的 representation state 表达，不得返回一个看似可用但无法区分失败原因的引用。

Materialization promotion 是另一条提交边界。它必须以记录的 representation revision/hash 建立幂等 Artifact 快照；重试不得读取更新后的 runtime representation，也不得重复追加同义 Artifact。

### 4.10 Detach、Delete、Retention 与 Provenance Hold

至少需要区分：

- 取消本轮选择：只影响 Run；
- Topic detach：只删除 TopicAssetBinding；
- WorkspaceAsset delete：在当前进程内使资产及未被进行中 Materialization 持有的 runtime representations 不再可解析；
- 进程结束：所有未提升的 WorkspaceAsset 和 representations 都可以消失；
- Artifact retention：由持久化 provenance 自己的 append-oriented 保留策略管理。

MVP 不需要为 WorkspaceAsset 设计跨重启 soft delete、tombstone 或完整垃圾回收。它只需要一个 in-process lease/hold 语义：已经被 `ContextAssetUse` 锁定且 Materialization 尚未完成的 representation 不能在处理中被删除。若内容已提升为 Artifact，用户后来取消选择、Topic detach、删除 WorkspaceAsset 或进程结束都不能删除既有 provenance；若只参与回答而没有生成 Memory，则不因此承诺形成持久化 Artifact。

### 4.11 Context Compiler 的读取契约

当前 Patchouli 在 prepare 阶段编译 `memory_context`，Prompt Assembler 只负责将已准备内容渲染成消息。附件应复用相同分工。

Context Compiler 负责：

- 解析资产引用；
- 验证 Workspace 和 Topic 权限；
- 选择 RAW 或 EXTRACTED_TEXT 等 runtime representation；
- 应用 token/大小预算；
- 生成 locator/citation；
- 返回带有 representation revision/hash 的 `ContextAssetUse`。

Prompt Assembler 不负责文件 I/O、资源授权或解析器调用。

Workspace MVP 不必实现完整的长文档编译器，但至少应冻结以下读取契约：

```text
resolve_asset(access_context, asset_ref)
list_topic_assets(access_context, topic_id)
resolve_representation(access_context, asset_id, preference)
```

`resolve_representation` 返回的内容视图应在本轮编译期间保持稳定；Context Compiler 不能把“取最新”语义留给后续 Materialization。Prompt Assembler 只消费编译后的只读内容，不接收 WorkspaceAssetStore 或 ArtifactStore 句柄。

## 5. 最小数据模型草案

以下模型用于冻结概念和边界，字段可以在正式 Plan 中继续调整。

```text
Workspace
  identity: WorkspaceIdentity
  workspace_kind: MAIN | STANDARD
  display_name
  status
  created_at

WorkspaceIdentity
  owner_user_id
  workspace_key: "main_workspace" | ...
  workspace_id

WorkspaceAccessContext
  actor_identity: Identity
  workspace_identity: WorkspaceIdentity
  request_id?
  interaction_id?

AssetOwnership
  workspace_identity: WorkspaceIdentity

AssetProvenance
  created_by_agent_id
  created_by_team_id?

AssetAccessPolicy
  visibility: PRIVATE | TEAM | WORKSPACE

WorkspaceAsset
  asset_id
  ownership
  provenance
  access_policy
  kind
  display_name
  media_type
  size_bytes
  representations[]
  lifecycle_state
  created_at

AssetRepresentation
  representation_id
  ownership
  asset_id
  kind: RAW | EXTRACTED_TEXT | ...
  revision
  content_object
  content_hash
  derived_from_representation_id?
  producer
  producer_version
  state

TopicAssetBinding
  workspace_identity
  topic_id
  asset_id
  bound_by
  bound_at
  first_bound_interaction_id

RunAssetSelection
  interaction_id
  run_id?
  workspace_identity
  topic_id
  asset_id
  requested_representation_id?
  selection_source: NEW_UPLOAD | TOPIC_REUSE | WORKSPACE_PICKER

ContextAssetUse
  context_asset_use_id
  interaction_id
  run_id?
  workspace_identity
  topic_id
  asset_id
  representation_id
  representation_revision
  content_hash
  locators
  included_tokens
  compile_status

SourceArtifactPromotion
  materialization_operation_id
  interaction_id
  target_memory_ids[]
  context_asset_use_id
  artifact_ref
  promotion_status

MemoryMeta / ArtifactMeta / TopicMeta
  owner_user_id
  workspace_id
  workspace_key
  source_agent_id
  source_team_id?
  visibility
```

`WorkspaceIdentity` 是 Workspace 聚合及其资产归属的统一值对象，应不可变、可序列化并在构造时完成非空和 MVP 等值校验。`AssetOwnership`、`AssetProvenance` 和 `AssetAccessPolicy` 表达三个不同的时间语义：归属定义资产在哪个 user/workspace 命名空间，provenance 记录历史创建者，access policy 决定当前执行者能否访问。实现时可以为了现有存储格式平铺 `owner_user_id/workspace_id/workspace_key`，但领域代码必须从同一份 `WorkspaceIdentity` 投影，不能让各字段分别成为权威来源。

`RunAssetSelection` 表达用户请求选择，`ContextAssetUse` 表达系统实际送入 Agent 上下文的内容；二者不能合并。它们都应优先关联稳定的 `interaction_id`，低层 `run_id` 只作为可选运行坐标。MVP 中它们可以随当前 Topic/Interaction 保存在进程内，但 `ContextAssetUse` 至少要存活到 Materialization 完成，并冻结 representation revision/hash。

`SourceArtifactPromotion` 是 Materialization 的幂等结果映射，不是用户可操作资源。实现可以把它并入 generation operation record，但必须能够在 retry 时判断同一来源快照是否已经生成 Artifact。WorkspaceAsset/representation 不保存反向 `artifact_ref` 作为生命周期依赖；需要审计时从 Memory provenance 或 promotion record 读取。

## 6. 初版作用域与缓存迁移方向

### 6.1 需要携带 Workspace scope 的资源

至少包括：

- ChatRequest 入口和 ChatGenerationRun scope snapshot；
- GatewayExecutionState / Gateway context；
- PreparedAgentRun；
- Memory metadata 和 Retrieval filters；
- Artifact metadata 和 Artifact access route；
- SemanticBuffer / TopicData；
- AgentRunContext / ExecutionFrame / MTPExecutionContext；
- InteractionPayload；
- 附件、Interaction、Memory Generation 和 Lifecycle work payload；
- RuntimeEvent 的观测投影。

active Workspace 是请求和 Run 的运行时字段；资源在各自生命周期内必须保留明确的 Workspace 归属。持久化的 Memory/Artifact 将该归属写入存储，进程内的 WorkspaceAsset/Topic 将其保存在 authoritative runtime record。它们的读取基线统一为：

```text
MUST access.workspace_identity.owner_user_id == access.actor_identity.user_id
MUST resource.workspace_identity == access.workspace_identity
MUST match_actor_visibility(access.actor_identity, ...)
```

如果现有历史数据没有 Workspace 字段，应根据其 `user_id` 统一回填 `workspace_id="main_workspace"`，并在需要完整聚合时构造 `WorkspaceIdentity(owner_user_id=user_id, workspace_key="main_workspace", workspace_id="main_workspace")`，而不是长期保留可空字段。历史 `source_agent_id/team_id` 继续作为 provenance 和兼容访问策略输入，不能替代 Workspace 归属。

跨子系统公开方法应逐步从裸 `user_id/agent_id/workspace_id` 参数迁移为完整的 `WorkspaceAccessContext`。可持久化或可重试的 work item 使用其可序列化投影：

```text
WorkScopeSnapshot
  workspace_identity: WorkspaceIdentity
  actor_identity: Identity
  interaction_id / operation_id
```

scope snapshot 记录目标坐标，不缓存永久授权结果；handler 在每次 attempt 仍由资源所有者重验。

### 6.2 需要逻辑分区的缓存

候选 Workspace-partitioned cache：

- Memory alias/UUID cache；
- Workspace-sensitive Agent Profile cache；
- attachment representation/cache；
- Context Compiler 的切片或 token 估算 cache。

cache key 至少包含 `owner_user_id + workspace_id`。对 Private/Team visibility 敏感的 alias 或 resolve 结果，还应携带 actor scope，或在命中后重新执行授权检查。

需要区分资源对象缓存与授权结果缓存：

```text
Memory object cache key
  = owner_user_id + workspace_id + memory_id

Alias resolution cache key
  = owner_user_id + workspace_id + alias
```

共享缓存可以保存资源对象，但 `alias -> MemoryAtom` 的命中不能直接代表当前 Agent 已获授权。若结果依赖 Private/Team visibility，cache key 还需包含 agent/team scope，或者命中后交回 Patchouli 重新执行 actor policy。失效不完整时应返回 miss，而不是复用其他 scope 的最近授权结果。

仍属于 Run、System 或 Work Queue 的状态：

- ExecutionFrame；
- PendingAtom 运行时视图；
- Chat generation registry；
- WorkRecord。

这些状态不应因为引入 Workspace 就搬进 Workspace manager。

## 7. Workspace MVP 的范围

### 7.1 本阶段目标

- 引入稳定的 Workspace/default Workspace 作用域；
- 建立 WorkspaceAccessContext；
- 建立只承诺进程内生命周期的 WorkspaceAsset working set；
- 建立 RAW / EXTRACTED_TEXT runtime representation 与 revision/hash 契约；
- 建立 TopicAssetBinding 和 asset reference；
- 让两个 Workspace 在同一进程内资源、Topic、Memory 和 cache 不串扰；
- 为 Context Compiler、Materialization promotion 和 provenance 预留稳定读取契约。

### 7.2 本阶段不实现

- multipart 上传界面；
- 完整 MIME/parser registry；
- PDF、DOCX、OCR 或多模态解析；
- 长文档异步任务；
- chunk、embedding、evidence 检索和候选记忆；
- WorkspaceAsset、原始附件和解析结果的跨进程持久化与重启恢复；
- WorkspaceAsset 的完整 soft delete、tombstone、保留期和 GC；
- Workspace Mount/Bridge；
- 父子 Workspace 和 Capability Subnet；
- 独立 Tool Registry、执行环境、预算和队列；
- 完整 RBAC、跨组织协作和多租户认证产品面；
- 为每个 Workspace 实例化一套 Alice/Patchouli Runtime。

## 8. Walking Skeleton 验收

Workspace MVP 至少应通过以下纵向场景：

1. 为同一个 user 创建 `main_workspace` 和第二个仅供后端验证的 Workspace。
2. 两个 Workspace 各自建立一个 Topic。
3. 使用同一个 Agent，在两个 Workspace 分别注册同名的纯文本 WorkspaceAsset。
4. 验证上传完成后资产的 Topic binding 数量为零。
5. Chat Composer 自动把新上传的 asset ref 加入当前草稿，但服务端不存在 Workspace 级 pending attachment。
6. ChatRequest 显式提交 asset refs，系统完成 Topic 路由后才幂等建立 TopicAssetBinding。
7. 同一 Topic 的下一轮不自动注入已绑定资产；用户重新选择后才再次 resolve 内容。
8. 验证 RunAssetSelection 与 ContextAssetUse 分离：选择但未成功编译的资产不形成 `ContextAssetUse`，也不被声明为回答来源。
9. 同一 WorkspaceAsset 在另一个 Topic 首次使用时建立第二条独立 binding。
10. 使用 Workspace A 的上下文访问 Workspace B 的 asset ID 被拒绝且不产生 binding。
11. 两边使用相同 Agent、alias、文件名和 Topic 标题时，Memory 和 cache 不串扰。
12. 两个并发 Run 首次引用同一 Topic/Asset 时只产生一条 binding。
13. 同一进程内重复相同 upload operation 只产生一个逻辑资产，不在上传路径生成 Artifact。
14. Topic detach 后，当前进程内的 WorkspaceAsset 和其他 Topic binding 仍然存在；已经提升的 Artifact 也不受影响。
15. 零 binding 资产在当前进程中仍可再次选择，但不承诺跨重启存在。
16. WorkspaceAssetRef 在进程重启后允许明确失效，不能解析到同名或复用 ID 的其他资产。
17. `ContextAssetUse` 锁定 representation revision/hash；Materialization 不能改读最新内容，处理完成前相关对象不会被回收。
18. Generation `DISCARD` 不生成外源 Artifact；Memory CREATE/UPDATE 只提升实际使用的文档 representation 为 `DocumentArtifact`。
19. 同一 materialization retry 不重复创建来源 Artifact；promotion 完成后，源 WorkspaceAsset 消失不影响 Artifact 与 Memory provenance。
20. 两个 Workspace 的并发 Chat 共用同一套 Gateway、Patchouli、Alice、MemoryLibrary 和 Work Queue 实例。
21. 并发 Chat 的 Gateway、prepare、Alice、finalize 始终保持各自冻结的 AccessContext，不出现进程级 `current_workspace` 覆盖。
22. 使用 Workspace A 的身份请求取消或查询 Workspace B 的 generation 被拒绝。
23. 子 Frame、MTP 和 background retry 继承原 Run 的 user/workspace scope，不能回退到 `main_workspace` 或切换 Workspace。
24. 任意深层资产操作缺少 Workspace scope 时显式失败，不静默选择默认 Workspace。

权限和串扰矩阵至少包含：

| user | workspace | agent/team | 预期 |
|:---|:---|:---|:---|
| 相同 | 不同 | 相同 Agent | 拒绝跨 Workspace 访问 |
| 相同 | 相同 | 不同 Agent，Workspace-visible | 允许 |
| 相同 | 相同 | 不同 Agent，Private | 拒绝 |
| 相同 | 相同 | 相同 Team，Team-visible | 允许 |
| 不同 | `workspace_key` 同名 | 其他条件任意 | 用户硬边界拒绝 |
| 相同 | 不同 | 显式传入其他 Workspace 的 asset ID | 仍然拒绝，不允许 ID 绕过 scope |

缓存验证应复用同一矩阵，尤其需要证明同一个 Agent 先访问 Workspace A、再进入 Workspace B 时，不会因为 alias/UUID cache 命中而读取到 A 的记忆或资产。

核心成功条件是：

> 两个 Workspace 通过同一套单例子系统并发运行时，Active Chat、Topic、Memory、Artifact、alias cache、background work 和 asset reference 都不会发生串扰；WorkspaceAsset 只存在于当前进程，而被提升的 Artifact 独立遵循持久化 provenance 契约。

## 9. Roadmap 建议

Workspace MVP 与 Chat Attachments 应形成两份具有独立目标、非目标、迁移和验收出口的开发 Plan，而不是一份 Plan 内两个相互牵连的实现切片。两者构成严格的前置依赖：Workspace MVP 先建立资源身份、作用域传播、逻辑隔离和进程内资产基础；Chat Attachments 只能消费已经完成验收的 Workspace 公共契约。

```text
Plan W0: Workspace MVP
  -> scope、in-process working set、runtime representations、binding、幂等、隔离
  -> 独立完成并通过双 Workspace 验收

Plan W1: Chat Attachments
  -> hard dependency: Workspace MVP completed
  -> upload、文本解析、asset refs、Context Compiler、Materialization promotion

v0.7.0 Document Ingestion
  -> 长文档解析、chunk/evidence、citation、候选记忆
```

如果初版只提供 `main_workspace`，可以暂缓 Workspace 创建/切换界面和跨 Workspace 通信；但即使不开放多 Workspace 产品面，也应允许测试或内部服务显式构造第二个 Workspace，并用后端测试建立隔离不变量。

如果准备立即开放用户创建和切换多个 Workspace，则现有身份治理、Memory scope、Artifact access 和 cache isolation 必须成为 `v0.6.2` 的前置条件。

## 10. 后续升级为正式 Plan 的门槛

### 10.1 Workspace MVP Plan

当前首先建立的正式 Plan 只覆盖 Workspace MVP。`WorkspaceIdentity` 的字段、默认值和 MVP 等值不变量已经在 4.1 冻结，不再把独立 ID 生成、Workspace Registry、附件上传、Context Compiler 或 Artifact promotion 作为本 Plan 的完成条件。

Workspace MVP Plan 进入 `docs/plans` 前仍需补齐：

- 具体受影响模块和迁移顺序；
- `WorkspaceAccessContext = actor Identity + WorkspaceIdentity + request/run correlation` 的公共模型与 route 变更清单；
- 默认 `main_workspace` 的入口解析、历史数据回填和第二 Workspace 测试构造方案；
- Memory、Artifact、Topic、cache 与 Work payload 从裸字段迁移到规范 Workspace 身份投影的方案；
- 现有 `MemoryVisibility.WORKSPACE/PUBLIC` 与新 Workspace hard boundary 的兼容映射；
- WorkspaceAsset in-process store、representation revision/hash 与生命周期 lease 的具体选型；
- WorkspaceAsset 注册、Topic binding、删除和进程内重试的失败矩阵；
- 两 Workspace 隔离、cache 污染、幂等和 provenance hold 的测试计划；
- 完成后需要同步更新的当前设计文档、contracts 和 Roadmap 条目；
- 将 `identity-and-execution-safety.md` 中的身份结构明确为 `AccessContext = actor Identity + WorkspaceIdentity + request/run correlation`，避免继续把执行者域与资产域写成同一个 Identity 层级。

### 10.2 Chat Attachments Plan

Chat Attachments 使用独立 Plan，并把“Workspace MVP 已完成且公共契约稳定”列为硬前置。以下问题不阻塞 Workspace MVP Plan，但必须在 Chat Attachments Plan 中冻结：

- 上传 API、MIME/编码限制、解析器和内存资源预算；
- `ChatRequest.asset_refs`、Composer 默认选择和 Topic 复用协议；
- `ContextAssetUse` 如何随 Interaction/LogicalBlock 进入 Topic Materialization；
- Materialization promotion 的内容粒度、幂等和跨 Memory 复用；
- DocumentArtifact 原文/解析内容的耐久快照载体与来源 metadata；
- Artifact promotion 失败时阻止 Memory 写入或显式 `provenance_incomplete` 的一致性策略；
- 附件选择、编译、预算裁剪、解析失败、删除和重试的完整失败矩阵。

建议治理文档采用以下表达：

```text
AccessContext
  -> actor Identity
       -> user_id
       -> agent_id
       -> team_id
  -> WorkspaceIdentity
       -> owner_user_id
       -> workspace_key
       -> workspace_id
  -> request/run correlation
```

这不是推翻已有身份治理，而是把其中“不可变身份快照和 scope”原则具体化：Identity 描述执行者，WorkspaceIdentity 描述资产域，AccessContext 在一次 request/run 中冻结两者。

在这些内容冻结前，本稿只作为设计讨论和实现边界参考，不作为当前能力说明。
