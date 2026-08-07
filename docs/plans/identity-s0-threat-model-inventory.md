---
title: Identity Isolation Phase S0 Threat Model Inventory
status: current
owner: system
scope: cross-subsystem-identity-s0-threat-model-inventory
code_paths:
  - src/hivememory/core/models/
  - src/hivememory/agent_runtime/
  - src/hivememory/alice/runtime/
  - src/hivememory/patchouli/memory_library/
  - src/hivememory/patchouli/services/
  - src/hivememory/server/
  - frontend/
related_docs:
  - docs/plans/identity-isolation-and-execution-safety.md
  - docs/plans/idempotency-i0-operations-inventory.md
  - docs/plans/durability-d0-state-inventory.md
  - docs/contracts/mtp.md
  - docs/todo/frontend-identity-ownership.md
  - docs/frontend/state-and-transports.md
last_reviewed: 2026-08-07
---

# Phase S0 身份与威胁模型清单

本文是[身份隔离与执行安全计划](./identity-isolation-and-execution-safety.md) **Phase S0** 的交付物。S0 的目标不是实现任何隔离机制，而是冻结"每个入口的身份输入、授权所有者、身份继承关系、当前威胁面"的现状，并建立最小复现样本，为 S1（Patchouli/Alice 身份收紧）、S2（run-local 执行隔离验证）、S3（执行资产安全）提供输入。

Phase S0 的四项任务：

1. 列出所有 request、route、MTP verb、cache、work item、Artifact 和 MemoryLibrary 操作的身份输入与授权所有者；
2. 画出主 Agent、子 Agent、后台 task、scheduler 和 frontend 的身份继承/缩小关系；
3. 建立越权、alias 污染、Profile stale、重试跨用户、stream 切换和执行逃逸的最小复现样本；
4. 明确当前只支持单用户/单 workspace 的地方，不在文档中暗示已经有完整租户隔离。

## 1. 全局结论摘要

- **全局没有认证/鉴权层**：Server 仅装配 CORS + 请求日志中间件（[app.py](../../src/hivememory/server/app.py#L69-L92)），无 token、session、auth middleware；`x-user-id` header 只被 topics 一个路由读取且纯透传不校验（[deps.py](../../src/hivememory/server/deps.py#L82-L84)）。
- **身份输入碎片化，四种来源并存且默认值兜底**：请求体字段（chat/ingest）、header（topics）、query param（memories 列表）、服务层字面量硬编码（`POST /memories` 与 `POST /agents` 在服务层写死 `user_id="default"`，路由层无法覆盖，[memory_service.py](../../src/hivememory/system/application/memory_service.py#L60)、[agent_service.py](../../src/hivememory/system/application/agent_service.py#L50)）。全项目约 25 处 `DEFAULT_*` 常量引用 + 2 处绕过常量的字面量。
- **核心隔离缺口在进程级共享缓存**：L0 PendingAtom store（[store.py](../../src/hivememory/agent_runtime/pending_atom/store.py)）与 L1 KoakumaAtomCache（[cache.py](../../src/hivememory/agent_runtime/aliases/cache.py)）都是 AliceRuntime 进程级单例，**cache key 完全不含 user/team/scope**；READ/UPDATE/RUN 的 alias 解析在 L1 命中时**不做任何身份重验**（[resolver.py](../../src/hivememory/agent_runtime/aliases/resolver.py#L94-L97)）。
- **检索链是唯一有系统性身份过滤的路径**：SEARCH 携带 identity，服务端以 `meta.user_id` 为 must 基线、visibility 为 should 作用域（[filter_adapter.py](../../src/hivememory/engines/retrieval/filter_adapter.py#L75-L105)），且 MTP filter 无法覆盖该基线。但同一检索服务的 `get_memory`（[retrieval.py](../../src/hivememory/patchouli/services/retrieval.py#L126-L131)）与 `list_memories`（[:133-L153](../../src/hivememory/patchouli/services/retrieval.py#L133-L153)）无 identity 参数；archive/revive/artifact 读写全部无身份校验。
- **子 Agent 完整继承 caller 的 Identity**（user_id/agent_id/team_id 不变，[call_coordinator.py](../../src/hivememory/alice/orchestration/sub_agent/call_coordinator.py#L178)）；CALL 权限由 `FrameExecutionPolicy` 硬检查，且子 Agent 被禁止递归 CALL（[:L163-L167](../../src/hivememory/alice/orchestration/sub_agent/call_coordinator.py#L163-L167)）。这是"身份不扩大"的少数已成立机制。
- **RUN 无执行身份、无审计、沙箱不足**：syscall 签名不接收 identity（[types.py](../../src/hivememory/agent_runtime/mtp/syscalls/types.py#L23)）；`sys_python_repl` 用裸 `subprocess.run` 无容器隔离（[repl.py](../../src/hivememory/agent_runtime/mtp/syscalls/repl.py#L127-L134)）；`sys_web_search` 的 timeout 参数未消费（[web_search.py](../../src/hivememory/agent_runtime/mtp/syscalls/web_search.py#L20-L22)）；全库无 approval/trusted/审批边界概念。
- **前端无身份状态入口**：chat/topics 恒用 `DEFAULT_USER_ID='default'` 兜底（[chatApi.ts](../../frontend/src/services/chatApi.ts#L36-L37)），无 token/登录存储；`docs/todo/frontend-identity-ownership.md` 已声明该问题。
- **RuntimeEvent 无 user_id 字段**（[runtime_events.py](../../src/hivememory/system/contracts/runtime_events.py#L87-L93)），观测层无法按用户关联，也不应成为授权依据。
- **当前系统本质是"单用户/单 workspace"假设下的实现**：任何"完整租户隔离"的声明都不成立（详见 §5）。

## 2. 身份输入与授权所有者矩阵

口径：`身份输入`（当前入口从哪里拿到 identity）、`授权所有者`（谁有最终解释权）、`现状`（已满足 / 部分 / 缺失）。

### 2.1 HTTP route

| # | 路由（位置） | 身份输入 | 授权所有者 | 现状 |
|:--|:--|:--|:--|:--|
| R1 | `POST /api/v1/chat`（[chat.py](../../src/hivememory/server/routers/chat.py#L52-L53)） | 请求体 `user_id`/`agent_id`（默认 `default`/`omni_doll`） | 无（客户端任意指定） | 缺失：无校验、无认证 |
| R2 | `POST /api/v1/chat/stop` | 无（仅 generation_id） | — | 缺失 |
| R3 | `POST /api/v1/ingest`（[ingest.py](../../src/hivememory/server/routers/ingest.py#L24-L33)） | 请求体 `user_id`（必填）+ `agent_id`（默认 `omni_doll`） | 无 | 缺失：user_id 必填但可由任意调用方自报 |
| R4 | `POST /api/v1/ingest/flush` | 请求体 `user_id`/`agent_id` | 无 | 缺失 |
| R5 | `GET /api/v1/topics`（[deps.py](../../src/hivememory/server/deps.py#L82-L84)） | header `x-user-id`（默认 `default`） | 无 | 缺失：唯一使用 header 的路由，纯透传 |
| R6 | `GET /api/v1/memories`（[memories.py](../../src/hivememory/server/routers/memories.py#L48)） | query `user_id`（可选，默认 None = 不过滤） | Patchouli（仅当传入时） | 缺失：不传则列出全局 |
| R7 | `POST /api/v1/memories` | **无身份入参**；服务层硬编码 `user_id="default"` | Patchouli（写死） | 缺失：路由层无法覆盖 |
| R8 | `POST /api/v1/agents` | **无身份入参**；服务层硬编码 `user_id="default"` | Patchouli（写死） | 缺失 |
| R9 | memories get/update/delete/feedback、topics settle/delete、memory-tasks、config、models、providers | 无 | — | 缺失 |

全局：无认证/鉴权中间件；服务端唯一权威默认是 `DEFAULT_USER_ID="default"`（[constants.py](../../src/hivememory/core/constants.py#L9)），被四种方式零散消费。

### 2.2 MTP verb（[runtime.py](../../src/hivememory/agent_runtime/mtp/runtime.py)）

统一入口 `_route_and_execute` 对所有 verb 先做 `_check_verb_permission`（policy + profile 双通道，[:L372](../../src/hivememory/agent_runtime/mtp/runtime.py#L372)）。

| # | verb | 身份输入 | 授权所有者 | 现状 |
|:--|:--|:--|:--|:--|
| V1 | SEARCH（[:L397-L467](../../src/hivememory/agent_runtime/mtp/runtime.py#L397-L467)） | `context.identity` → `RetrievalRequest.identity` | Patchouli（filter_adapter 身份基线） | **已满足**（服务端基线过滤，MTP filter 不可覆盖） |
| V2 | READ（[:L469-L571](../../src/hivememory/agent_runtime/mtp/runtime.py#L469-L571)） | `context.identity` → alias resolver | L2 冷查询带 user_id；**L1 缓存命中无过滤**；读后无可见性重校验 | 部分（L1 缺口） |
| V3 | WRITE（[:L659-L709](../../src/hivememory/agent_runtime/mtp/runtime.py#L659-L709)） | `context.identity` → `PendingAtom.identity` | Alice（注册时记录） | 已满足（写侧记录身份；读侧见 V2） |
| V4 | UPDATE（[:L711-L782](../../src/hivememory/agent_runtime/mtp/runtime.py#L711-L782)） | `context.identity` → register_update | **base atom 无可见性校验**（resolve 走 L1 不校验） | 缺失 |
| V5 | RUN（[:L573-L657](../../src/hivememory/agent_runtime/mtp/runtime.py#L573-L657)） | 间接（syscall 不接收 identity）；Level 1 alias resolve 走 L1 | 无执行身份；仅 `MemoryType.CODE_SNIPPET` 类型门槛 | 缺失 |
| V6 | CALL（[:L784-L850](../../src/hivememory/agent_runtime/mtp/runtime.py#L784-L850)） | `execution_policy.allows("CALL")` 硬检查；identity 由 caller frame 继承 | Alice（frame policy + Profile capability） | **已满足**（子 Agent 完整继承，禁止递归 CALL） |

### 2.3 Cache 与进程内共享状态

| # | 组件（位置） | key 维度 | 命中后身份重验 | 现状 |
|:--|:--|:--|:--|:--|
| C1 | KoakumaAtomCache（L1，[cache.py](../../src/hivememory/agent_runtime/aliases/cache.py#L33-L54)） | alias / uuid，**无 user/team/scope** | **无** | **核心缺口**：alias 跨用户冲突，后写覆盖（[:L47/L54](../../src/hivememory/agent_runtime/aliases/cache.py#L47-L54)） |
| C2 | PendingAtom store（L0，[store.py](../../src/hivememory/agent_runtime/pending_atom/store.py)） | alias / intent 全局 dict | **无**；`get_by_intent` 不区分用户（[:L47-L52](../../src/hivememory/agent_runtime/pending_atom/store.py#L47-L52)） | **核心缺口**：进程级单例共享；settle/claim/cancel 均不校验身份 |
| C3 | AgentProfileCache（[profile_resolver.py](../../src/hivememory/alice/runtime/profile_resolver.py#L24-L52)） | `(user_id, agent_id, team_id, alias)` 四维 | key 已含 scope；`identity=None` 时直接抛 `PermissionDeniedError`（[:L73-L77](../../src/hivememory/alice/runtime/profile_resolver.py#L73-L77)） | **已满足（key）**，但无失效机制 → Profile stale（见 §4.3）；且 `default`/`omni_doll`/空 alias 直返内置 `OMNI_DOLL_PROFILE` 不经过缓存与身份检查（[:L70-L71](../../src/hivememory/alice/runtime/profile_resolver.py#L70-L71)） |
| C4 | ExternalEventDedupRegistry（[dedup.py](../../src/hivememory/system/services/passive/dedup.py#L27)） | `(source, external_event_id)`，**无 user_id** | n/a | 缺失：跨用户误判重复（见 §4.4） |
| C5 | MessageTurnBufferManager（[turn_buffer.py](../../src/hivememory/system/services/passive/turn_buffer.py#L294)） | `PassiveConversationKey`（source + external_conversation_id + user_id + agent_id + team_id） | key 含身份 | **已满足** |
| C6 | RuntimeAliasResolver（L0/L1/L2 三级，[resolver.py](../../src/hivememory/agent_runtime/aliases/resolver.py)） | L0/L1 无身份；L2 带 identity（context 为 None 时退化为默认 Identity，[:L197-L201](../../src/hivememory/agent_runtime/aliases/resolver.py#L197-L201)） | L2 结果回填共享 L1 → 跨身份投毒 | 缺失 |

### 2.4 Work item

| # | 工作项（位置） | 身份输入 | 现状 |
|:--|:--|:--|:--|
| W1 | MemoryGenerationTask（[memory_tasks.py](../../src/hivememory/patchouli/runtime/memory_tasks.py#L72-L82)） | **无独立 identity 字段**；主动链路显式注入 `task.identity`（[coordinator.py](../../src/hivememory/patchouli/control/memory_generation_coordinator.py#L153-L180)）；被动链路从 `context.turns[0].identity` 回退，可退化为默认 Identity（[generation/models.py](../../src/hivememory/engines/generation/models.py#L130-L137)） | 缺失 |
| W2 | Scheduler 任务（[async_scheduler.py](../../src/hivememory/system/runtime/scheduler/async_scheduler.py)） | **无身份概念**（spec 仅 owner/name/interval） | 缺失 |
| W3 | Passive drain 后台提交（[passive_ingress_service.py](../../src/hivememory/system/application/passive_ingress_service.py#L90-L102)） | 靠 `InteractionPayload.identity`（buffer 构建时写入，[turn_buffer.py](../../src/hivememory/system/services/passive/turn_buffer.py#L272)） | 部分 |

### 2.5 Artifact / MemoryLibrary / retrieval 服务端

| # | 操作（位置） | 身份输入 | 现状 |
|:--|:--|:--|:--|
| M1 | `RetrievalFamiliar.get_memory`（[retrieval.py](../../src/hivememory/patchouli/services/retrieval.py#L126-L131)） | **无**，仅按 UUID | 缺失：任何持 id 者可读 |
| M2 | `RetrievalFamiliar.list_memories`（[:L133-L153](../../src/hivememory/patchouli/services/retrieval.py#L133-L153)） | **无**（filters 由调用方传入，不传无基线过滤） | 缺失 |
| M3 | `get_by_alias`（[vector_store.py](../../src/hivememory/infrastructure/storage/vector_store.py#L255-L259)） | user_id 可选，不传则不过滤 | 缺失（调用方决定） |
| M4 | `MemoryLibrary.archive/revive`（[library.py](../../src/hivememory/patchouli/memory_library/library.py#L61-L91)） | **无**身份参数 | 缺失 |
| M5 | Artifact put/get/exists（[artifact.py](../../src/hivememory/patchouli/memory_library/adapters/artifact.py#L52-L99)） | **无**；get 用 `rglob(artifact_id)` 全局搜索 | 缺失 |
| M6 | `get_agent_profile`（[retrieval.py](../../src/hivememory/patchouli/services/retrieval.py#L155-L216)） | 有 `_is_memory_visible_to`（user_id + visibility + team_id + source_agent_id） | **已满足**（唯一完整的读侧校验） |

## 3. 身份继承 / 缩小关系图

### 3.1 当前关系（现状）

```text
HTTP request（身份由客户端自报/默认值）
  └─ ChatApplicationService ──> Identity(user_id, agent_id, session_id)  [chat_service.py:156-160]
       └─ Gateway.process ──> GATEWAY_PROCESS（携带 identity）
       └─ PrepareAgentRun ──> user_id/agent_id/session_id
            └─ ExecutionFrame.identity（agent_run_service.py:94/234）
                 └─ MTPExecutionContext(identity=frame.identity)  [loop.py:194-199]
                      ├─ SEARCH ──> RetrievalRequest(identity) ──> Patchouli 基线过滤 ✓
                      ├─ READ/UPDATE/RUN ──> RuntimeAliasResolver（L0/L1 无身份；L2 带身份）✗
                      ├─ WRITE ──> PendingAtom.identity ✓
                      └─ CALL ──> CallCoordinator ──> callee frame identity=caller_frame.identity（完整继承）✓
                                    └─ 子 Agent denied_verbs={"CALL"}（禁止递归）✓

后台工作（身份薄弱或无身份）：
  ├─ MemoryGenerationTask（无独立 identity，被动链路可退化为默认 Identity）✗
  ├─ Scheduler 任务（无身份概念）✗
  └─ Passive drain（靠 InteractionPayload.identity，间接携带）△

Frontend：
  └─ chatApi/topicApi 逐请求兜底 DEFAULT_USER_ID='default'（无单一身份入口）✗
```

### 3.2 身份继承规则速查

| 关系 | 规则 | 现状 |
|:--|:--|:--|
| 主 Agent frame | 从 chat 请求 identity 派生，贯穿整个 run | ✓ 已成立 |
| 子 Agent（CALL） | **完整继承** caller 的 user/agent/team identity；profile 换成子 Agent | ✓ 已成立（但不缩小，见 §4.6 风险） |
| 后台 task（memory generation） | 主动链路显式注入 task.identity；被动链路回退 | △ 被动链路可退化 |
| Scheduler 任务 | 无身份传播 | ✗ 缺失 |
| Passive 提交 | 随 InteractionPayload 携带 | △ 间接 |
| Frontend | 无继承关系，逐请求默认值 | ✗ 缺失 |

## 4. 威胁模型与最小复现样本

### 4.1 越权读取（READ/RUN/UPDATE 跨用户）

| 项 | 内容 |
|:--|:--|
| 场景 | 用户 B 读取/修改/执行用户 A 的私有记忆 |
| 复现 | ① 用户 A 执行 SEARCH（结果 `ingest_atoms` 写入进程级共享 L1，[runtime.py#L461](../../src/hivememory/agent_runtime/mtp/runtime.py#L461)）；② 同进程用户 B 用相同 alias 发 READ（[:L500](../../src/hivememory/agent_runtime/mtp/runtime.py#L500)）→ resolver L1 命中（[resolver.py#L94-L97](../../src/hivememory/agent_runtime/aliases/resolver.py#L94-L97)）→ 直接返回 A 的 atom；③ UPDATE（[:L740](../../src/hivememory/agent_runtime/mtp/runtime.py#L740)）对同一 atom 发起修订，无可见性校验；④ 若 A 的记忆是 `MemoryType.CODE_SNIPPET`，B 的 RUN 直接执行（[:L643-L652](../../src/hivememory/agent_runtime/mtp/runtime.py#L643-L652)） |
| 根因 | L1 cache key 无 scope（[cache.py#L33-L54](../../src/hivememory/agent_runtime/aliases/cache.py#L33-L54)）；`_is_memory_visible_to` 仅用于 SEARCH 服务端与 CALL profile 解析，未覆盖 READ/UPDATE/RUN 的 L1 命中路径 |
| 当前防御 | 无 |
| S1 方向 | L1 命中后按 identity 重验；cache key 含 scope 或按用户分缓存；READ/UPDATE 读后补可见性检查 |

### 4.2 alias 污染（L2 冷查询回填共享缓存）

| 项 | 内容 |
|:--|:--|
| 场景 | 用户 B 因 L1 缓存投毒读到用户 A 的 atom |
| 复现 | ① 用户 A 的 alias 被用户 B 的 L2 冷查询命中（L2 带 identity 过滤，但 PUBLIC/alias 撞车时仍可返回 A 的原子）；② 结果回填**共享 L1**（[resolver.py#L216](../../src/hivememory/agent_runtime/aliases/resolver.py#L216)）；③ 此后任何用户对该 alias 的 READ 直接 L1 命中 |
| 根因 | L1 无身份标记；L2 结果回填未按身份分桶；alias 是进程级全局字符串 key，后写覆盖（[cache.py#L47/L54](../../src/hivememory/agent_runtime/aliases/cache.py#L47-L54)） |
| 当前防御 | 无 |
| S1 方向 | L2 回填写入按身份分桶的缓存；或回填后不参与无身份命中 |

### 4.3 Profile stale

| 项 | 内容 |
|:--|:--|
| 场景 | Profile 被管理员更新后，缓存仍向后续 run 返回旧权限快照 |
| 复现 | ① `AgentProfileResolver.resolve` 命中 LRU 缓存（[profile_resolver.py#L79-L88](../../src/hivememory/alice/runtime/profile_resolver.py#L79-L88)）；② Profile 被 `PATCHOULI_GET_AGENT_PROFILE` 之外的管理接口 UPDATE；③ 缓存无失效入口、无 TTL → 后续 run 持续使用旧 Profile 的 verb/tool 白名单 |
| 根因 | AgentProfileCache key 已按身份隔离（好），但**无失效机制** |
| 当前防御 | 无（并发 miss 有 `_load_lock` 防污染，但更新后不失效） |
| S1 方向 | UPDATE 后 invalidate，或引入 TTL/版本 |

### 4.4 重试跨用户

| 项 | 内容 |
|:--|:--|
| 场景 | 幂等窗口跨用户误判 |
| 复现 | ① 用户 A 提交事件（`source`, `external_event_id`）进入 `ExternalEventDedupRegistry`（[dedup.py#L27](../../src/hivememory/system/services/passive/dedup.py#L27)）；② 外部系统复用同一事件 ID 空间，用户 B 的同 `(source, external_event_id)` 事件被 `register()` 判为重复而丢弃（[ingressor.py#L195-L206](../../src/hivememory/system/services/passive/ingressor.py#L195-L206)） |
| 根因 | dedup key 不含 user_id；后台 task 身份回退可退化为默认 Identity（见 W1） |
| 当前防御 | 无 |
| S1 方向 | dedup key 增加 user_id 维度；WorkStore/PendingAtom record 从第一天带身份（与 D0 §7 交叉） |

### 4.5 stream / 身份切换残留

| 项 | 内容 |
|:--|:--|
| 场景 | 前端切换身份后，旧用户的 chat/topic/缓存状态残留 |
| 复现 | ① chatStore 只持久化 `currentAgentId`（[chatStore.ts#L348-L361](../../frontend/src/stores/chat/chatStore.ts#L348-L361)），不存 user_id；② chatApi 恒 `DEFAULT_USER_ID='default'` 兜底（[chatApi.ts#L36-L37](../../frontend/src/services/chatApi.ts#L36-L37)）；③ 后端 `ChatGenerationRunRegistry`、`KoakumaAtomCache` 进程级共享，即使前端换身份，后端缓存仍残留旧 run 内容 |
| 根因 | 前端无单一身份入口（[todo/frontend-identity-ownership.md](../todo/frontend-identity-ownership.md#L16-L24)）；后端缓存/registry 无身份维度 |
| 当前防御 | 无 |
| S1/S4 方向 | 前端建立身份 store 并定义切换清理；后端按 §4.1/4.2 修缓存 |

### 4.6 执行逃逸（RUN 资产）

| 项 | 内容 |
|:--|:--|
| 场景 | MTP RUN 执行不可信代码/无限阻塞外部调用 |
| 复现 | ① `sys_python_repl` 用 `subprocess.run([sys.executable, "-c", runner])`（[repl.py#L127-L134](../../src/hivememory/agent_runtime/mtp/syscalls/repl.py#L127-L134)）——无容器/namespace/cgroup，子进程可访问全部文件系统与网络；② `sys_web_search` 的 `timeout_seconds` 参数**未消费**（[web_search.py#L20-L22](../../src/hivememory/agent_runtime/mtp/syscalls/web_search.py#L20-L22)），同步 `DDGS()` 可无限阻塞；③ `sys_read_file`/`sys_write_file` 无 timeout（仅 workspace 路径 + max_bytes 限制）；④ RUN Level 1 只查 `MemoryType.CODE_SNIPPET`（[runtime.py#L643](../../src/hivememory/agent_runtime/mtp/runtime.py#L643)），**不检查 `verification_status`、无审批、无审计** |
| 根因 | 无可信资产/审批边界（全库无 approval/trusted 概念）；syscall 层无 identity 透传（[types.py#L23](../../src/hivememory/agent_runtime/mtp/syscalls/types.py#L23)）；`SyscallResult` 仅含 content，无 run/asset identity |
| 当前防御 | `_check_tool_permission`（仅 Level 0 内核工具，[runtime.py#L605](../../src/hivememory/agent_runtime/mtp/runtime.py#L605)）+ Profile 白名单 + repl 的 safe_builtins/blocked_import（可绕过，仅防误用非防恶意） |
| S3 方向 | 先冻结现状（本清单）；S3 建立来源/信任/审批模型 + 资源硬限制 + 拒绝默认策略；在强隔离证据成立前，产品文案不得宣称 RUN 为可执行能力 |

## 5. 单用户 / 单 workspace 现状声明

以下场景当前**明确不成立**，文档与代码不得暗示已有完整租户隔离：

1. **无认证**：任何请求可自报任意 `user_id`（请求体/header），服务端不校验；`POST /memories` 与 `POST /agents` 甚至硬编码 `default`。
2. **进程级缓存跨用户共享**：`KoakumaAtomCache`、`PendingAtomRuntime`、`ChatGenerationRunRegistry`、`ExternalEventDedupRegistry` 均为进程级单例，身份维度缺失（§2.3）。
3. **前端无身份状态**：chat/topics 恒 `default`；memories 不发 user_id；无 token/localStorage 登录态。
4. **RuntimeEvent 无 user_id**：观测层无法按用户隔离或审计。
5. **同一 Qdrant collection 混存所有用户数据**：隔离完全依赖应用层查询过滤器（filter_adapter），且 `get_memory`/`list_memories`/artifact 读路径绕过该过滤器。

结论：当前系统在"单个默认用户 `default` + 单个 Agent `omni_doll`"的假设下运行。多用户并发时，隔离靠"无并发用户"或"检索层过滤"偶然成立，不能视为安全边界。

## 6. 结论与后续建议（S1 前置输入）

1. **S1 第一优先级是 L0/L1 两级缓存的身份维度**（3 个位置）：`KoakumaAtomCache` key 含 scope + L1 命中重验；`PendingAtom` store 的 alias/intent 查询按 identity 过滤；L2 回填不投毒共享缓存。这是 §4.1/4.2 越权与污染样本的共同根因。
2. **统一身份输入**：在 deps 层建立 per-request Identity 上下文；消除 `memory_service.py`/`agent_service.py` 两处字面量硬编码；让所有路由显式携带身份（对齐 I0 §4 的 accepted 语义修改窗口）。
3. **补齐读路径校验**：`get_memory`、`list_memories`、`archive/revive`、Artifact get 增加 identity 参数与可见性检查；`get_by_alias` 强制 user_id（不再可选）。
4. **Profile 缓存失效**：管理接口 UPDATE 后 invalidate 或引入 TTL；区分"未指定 Profile / 指定 Profile 不存在 / 无权访问 / 已失效"四种结果（主计划 §2.1 目标 4）。
5. **dedup key 与后台任务身份**：`ExternalEventDedupRegistry` key 增加 user_id（与 I0 的幂等键目录交叉）；memory generation task 补独立 identity 字段（与 D0 §7 的 WorkStore schema 交叉）。
6. **RUN 资产**：S0 已冻结现状（§4.6）；S3 之前不承诺"可执行能力"，syscall 至少补齐 timeout 消费与执行者身份记录。
7. **前端身份**：按 `docs/todo/frontend-identity-ownership.md` 完成条件推进，但明确前端不是安全边界（主计划 §2.2 非目标）。
8. **越权测试**：将 §4 六个样本固化为回归测试（主计划 §5 验收标准），S1 完成后以"样本全部失效"为完成判据。
