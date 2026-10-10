---
title: Identity Isolation and Execution Safety Governance
status: governance
owner: system
scope: cross-subsystem-identity-authorization-and-execution-safety
code_paths:
  - src/hivememory/core/models/identity.py
  - src/hivememory/agent_runtime/
  - src/hivememory/alice/runtime/
  - src/hivememory/workspace/intents/
  - src/hivememory/workspace/capability/
  - src/hivememory/workspace/cache/
  - src/hivememory/patchouli/memory_library/
  - src/hivememory/server/
  - frontend/
related_docs:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/mtp.md
  - docs/architecture/workspace.md
  - docs/alice/README.md
  - docs/alice/mtp-runtime.md
  - docs/alice/orchestration.md
  - docs/todo/frontend-identity-ownership.md
  - docs/archive/todo/mtp-cache-scope-revalidation.md
  - docs/todo/workspace-asset-ownership-identity-split.md
last_reviewed: 2026-10-10
---

# 身份隔离与执行安全治理

HiveMemory 已建立 Workspace 归属、MemoryVisibility、MTP permission、Agent Profile 和 MemoryLibrary 可见性边界，workspace 派生缓存（含 CALL 目标 Profile）已接入 canonical 变更失效，Alice 全部资源操作已经过凭据绑定的能力入口；PendingAtom 的耐久隔离、子线程独立授权、前端身份状态与 RUN 执行安全仍有缺口。`IdentityScope` 由 `ActorIdentity` 与 `WorkspaceIdentity` 组成，表达一次操作的发起者与目标 Workspace；它只沿该操作的公开调用链传播。Patchouli 在公开边界拆为 `belong_to` 与 `from_actor`，内部、交互记录和后台任务分开携带归属与必要的发起者。来源记录不参与资源授权。

这不是单纯的登录页面工作，也不是给每个 handler 再加一个 `if user_id`。它需要统一回答：一次请求、一次 Agent run、一个 PendingAtom、一个 Profile、一个 MemoryAtom 和一段可执行资产分别属于谁；哪个组件拥有最终授权权；缓存、子帧、重试和后台任务如何继承或缩小权限。

## 1. 当前问题证据

| 边界 | 当前实现基础 | 当前风险 |
|:---|:---|:---|
| Memory visibility | MemoryAtom 独立持有归属、`MemoryAccessPolicy` 与来源记录；Patchouli 是可见性所有者 | workspace resolver 在 L1 命中与 L2 交付时重验归属及 actor policy；canonical 变更失效原子、旧 alias 和 Profile 派生项，不提供通知未送达补偿 |
| PendingAtom | workspace registry 独立保存 `belong_to`、`from_actor` 与 `process_id`；能力层分别以 `memory_intent.submit` / `resource.read` 授权 | 第一版不设 Pending policy，同 Workspace 中获读授权的 Actor 可读回意图；跨 Workspace 不可读，无持久化 ledger；结算 redirect 仍逐次校验 canonical 可见性 |
| Agent Profile | Profile 作为 MemoryAtom，主线程分配经 workspace 能力层 `profile.read`，ProfileCache 按 Workspace/alias 定位，命中时以源 policy 对当前 actor 重验 | 主线程与 CALL 目标共用 workspace 读取视图，来源 UUID 失效与代次回填保护已生效；事件通知无持久重放或未送达补偿 |
| Agent run/frame | `ExecutionFrame`、`RunSession`、frame policy、Chat phase task 与 `AgentRunStreamAdapter` | frame registry/CALL record、可中断阶段 task、Alice runner、输出队列和流序号均按 run 隔离；执行模型只含观测标签，CALL 子帧共享主线程凭据，尚无独立子线程授权 |
| MTP permission | Prompt 与 Koakuma runtime 有双层权限设计 | prompt 教学不是硬安全保证；Profile verb/tool 与 workspace operation 分别校验，观测标签不能替代访问凭据 |
| MTP READ/RUN | READ 可访问记忆，RUN 可执行 memory code | RUN 没有强沙箱、资源限制、可信资产分级或强制审批边界 |
| Frontend identity | 前端已统一用户导向身份选择上下文（`user_id + workspace_id` 基础选择经 `services/identity.ts` 以请求头携带） | UI 字段不是认证/授权边界；尚无登录/Workspace 切换 UI，切换时 cache/stream 清理仍待完整设计 |
| Observability | RuntimeEvent 可携带 identity/run/frame/atom 关联 | 事件 payload 不能成为授权依据，也不能泄漏不应被当前身份看到的内容 |
| Background work | Interaction 与生成任务独立保存 `belong_to`、`from_actor`；四种 SETTLE 触发统一使用归属 owner 的 `system` actor，查重只读取 PUBLIC | Work Store 仍只有进程内实现；跨重启恢复不在当前承诺内 |
| WorkspaceAsset | Store 与解析链仍使用旧 `IdentityScope` 接口，但资源检查只比较归属；System 的 `AssetMaterializationReader` 为 Patchouli 提供归属读取端口 | 这是身份第二批明确暂缓的内部拆分，跟踪见 [WorkspaceAsset 归属拆分 Todo](../../todo/workspace-asset-ownership-identity-split.md) |

## 2. 目标与非目标

### 2.1 目标

1. 按寿命区分访问 context、单次操作的 `IdentityScope`、资源归属与后台任务发起者，建立跨边界传播和权限缩小规则；
2. 所有 Memory、Profile、PendingAtom、Artifact 和执行资产的读取/写入都由实际所有者重验归属与适用的资源 policy；
3. 使 cache key、frame、cancel、task 和 retry 不会因共享组件的复用而跨用户或 Workspace 共享未验证的可变授权状态；
4. 区分“未指定 Profile”“指定 Profile 不存在”“Profile 无权访问”“Profile 已失效”四种结果；
5. 为 MTP RUN 建立可信资产、能力白名单、资源限制、取消和审计边界；
6. 让前端身份状态与后端认证/授权契约对齐，同时明确前端不是安全边界；
7. 为并发、越权、缓存污染、CALL 深度和执行资产逃逸提供测试与故障样本。

### 2.2 非目标

- 不在本治理主题中选择具体身份提供商、OAuth 产品或多租户商业方案；
- 不把 prompt 中的 verb 教学当作安全控制；
- 不让 Gateway、Alice 或前端自行决定 Patchouli 的长期可见性；
- 不承诺把任意 Python 代码变成安全可执行资产；
- 不为了实现身份隔离而把所有对象复制到每个用户的独立数据库；
- 不把 RuntimeEvent、日志或前端 localStorage 当作认证状态。

## 3. 目标权限模型

### 3.1 身份形态与寿命

运行凭据、操作身份和资源记录分别承担不同职责：

```text
访问 context：本次运行的密封访问凭据，由运行持有者和 Workspace 授权点使用
ExecutionCredential：workspace 签发与吊销的不透明执行凭据，CPU 驱动绑定为操作提交函数
IdentityScope：操作授权后组装的 ActorIdentity + 目标 WorkspaceIdentity
ExecutionLabels：agent_id/workspace_id 字符串，仅用于提示词、展示与运行观测
资源 owner 内部、交互记录与后台任务：belong_to + 必要的 from_actor
```

子 Agent 当前继承父 run 的操作提交函数和观测标签，所有请求仍按主线程凭据授权；显式 `context_refs` 只选择共享内容，不签发更广或独立的资源权限。Alice/CPU 的 `RuntimeScope`、`AgentRunContext` 与 `CPUInputManifest` 不携带 `IdentityScope`，frame 与 MTP 上下文没有 `identity` 派生。标签取自注册入口并冻结，不能还原成资源归属或授权身份；提示词历史归并和运行事件使用注册标签，frame 流式角色展示优先使用 Profile 的源 alias。`request/run/frame` 等关联坐标属于运行模型，`ActorIdentity` 只含 user/agent/team，不含 `session_id`。

入口层现状：HTTP 的用户导向身份选择在 `server/deps.py resolve_request_identity_claims` 解析为身份声明，经统一认证网关认证后，由 workspace 授权点在操作授权时组装 `IdentityScope`（`/ingest` 仍在认证前组装，是既有例外；模型见[Workspace 架构](../../architecture/workspace.md)第 4 节），应用服务不再解析裸 `user_id`。非 Agent action 使用单独登记的保留 `system` actor；Memory 管理读取按 owner-management 语义在归属硬边界内跳过 actor 可见性过滤，Agent retrieval 与 SETTLE 查重仍执行普通 `MemoryAccessPolicy`，后者以 `system` 为可见性主体。

WorkspaceAsset 的 Store、解析服务与既有 Reader/Command 端口仍使用旧 `IdentityScope` 接口做归属检查；System 的 `AssetMaterializationReader` 只在一次 representation 租借内以归属和保留 `system` actor 组装适配所需 scope，Patchouli 使用归属读取端口，不保存该 scope。此例外不扩展到 Alice、CPU 或其他资源路径，内部拆分继续由 [WorkspaceAsset 归属 Todo](../../todo/workspace-asset-ownership-identity-split.md) 跟踪。

执行凭据按对象身份兑现，不能序列化，也不保存身份字段；只有 workspace 凭据表保存访问 context、固定目标和 process ID 绑定。`WorkspaceOperationEntry` 在分派前拒绝未知/吊销凭据，能力方法仍逐次授权。进程关闭同步吊销、取消 PENDING 与释放租借先于任何 await；在途只读可完成；WRITE、UPDATE 返回入口后都会复查凭据（当前 UPDATE 的基础冷读可能跨越关闭），由入口同步补偿关闭期间产生的登记，不取消请求调用方 task。新增副作用前有 await 的能力方法必须补上同类检查和补偿。

### 3.2 所有者重新校验

Cache 命中、MTP READ/RUN、PendingAtom resolution、Artifact ref 读取、MemoryLibrary archive/revive 和 background retry 都必须由最终状态所有者重新验证资源归属与适用的 policy。Patchouli 公开入口从 `IdentityScope` 拆出这两项输入，内部读取以 `from_actor` 为可见性主体；owner-management 只跳过 actor 可见性，不跳过 Workspace 归属。上游检查不能替代下游检查，因为请求可能跨越重试、队列和子系统边界。

生成后台任务的 `from_actor` 是执行该操作的发起者，不是记忆来源字段。WRITE/UPDATE 保留提交者，SETTLE 的手动、空闲、LRU 与关闭触发都由 `system_actor_for_workspace` 构造 `system` actor。`system` 不能成为 PRIVATE target 且无 team，因此查重只看得到 PUBLIC，不把结算内容并入相似的 PRIVATE/TEAM 记忆。四触发与真实生成链证据见 [SETTLE 集成回归](../../../tests/integration/patchouli/test_settlement_identity.py)。

### 3.3 缓存不承载授权

通用共享基础设施不自动按 Workspace 分区；缓存值必须在命中后由最终资源 owner 或 resolver 以调用方的归属和发起者重验。workspace 持有 canonical 原子与 alias 索引、全部 Profile 派生缓存；Alice 不保存资源读取缓存。分区消除错误命中与无效覆盖，但**不能替代命中后的 ownership/actor policy 重验**。workspace 在 canonical 提交尝试后依次失效原子、旧 alias、来源 Profile 并推进 Workspace 代次，在途点读跨代次时重读一次，仍变化则报资源不可用，不回填旧值；语义检索跨代次仍交付已授权结果，但不预热缓存。失败结果不进入缓存。调用方取得独立副本，不能修改缓存内授权事实。

PendingAtom 的全 Workspace 回读只适用于意图：解析到结算后的 canonical 时再次执行当前 actor policy；目标不可读时不交付正文、canonical alias/UUID，也不交付含 UPDATE 基础坐标的 Pending 副本。UPDATE 意图的回读跟随基础原子的可读性，基础不可读时在任何状态下都与不存在相同。UPDATE 只接受可读的正式 atom，不接受 pending 或结算 redirect。MTP READ、RUN、SEARCH、意图提交/撤回与 CALL 的目标 Profile/共享引用全部经操作请求进入能力层。引用记录由能力层为已交付的正式原子自动执行（含缓存命中），失败只记日志，取消正常传播；UPDATE 基础解析不记录。Alice 与 agent_runtime 不导入 `IdentityScope`，也不请求 Patchouli 路由。权限、redirect 防泄露与副本隔离证据见[引用解析测试](../../../tests/unit/workspace/resolution/test_alias_resolver.py)和[意图/缓存集成回归](../../../tests/integration/workspace/test_intent_registry_and_read_cache.py)。

### 3.4 可执行资产是更高风险能力

MTP RUN 应将“可读取的 Memory”与“可执行的 Memory”分开：

- 资产必须显式声明 executable capability、来源和信任级别；
- runtime 只能执行允许的 verb/tool 和受限资源；
- timeout、取消、输出大小、文件/网络访问和进程边界必须有硬限制；
- 执行结果与失败必须带 run/asset identity，不能只依赖 prompt；
- 无法提供强隔离时，默认拒绝不受信任资产，或明确降级为展示/提议而非执行。

## 4. 未排期治理工作包

### Phase S0：身份与威胁模型清单

1. 列出所有 request、route、MTP verb、cache、work item、Artifact 和 MemoryLibrary 操作的身份输入与授权所有者；
2. 画出主 Agent、子 Agent、后台 task、scheduler 和 frontend 的身份继承/缩小关系；
3. 建立越权、alias 污染、Profile stale、重试跨用户、stream 切换和执行逃逸的最小复现样本；
4. 明确当前只支持单用户/单 workspace 的地方，不在文档中暗示已经有完整租户隔离。

### Phase S1：Patchouli 与 Alice 身份收紧

1. 已完成（2026-10-07）：PendingAtom registry 与引用解析归 workspace，按 Workspace 硬边界回读，不再按整个执行 scope 相等判断。Pending 第一版在全 Workspace 开放（UPDATE 意图跟随基础原子的可读性，2026-10-09），canonical redirect 重新校验资源 policy；旧 scope 校验背景见 [MTP cache scope revalidation 历史记录](../../archive/todo/mtp-cache-scope-revalidation.md)；
2. 已完成（2026-10-07）：workspace 原子和 Profile 派生缓存按源坐标分区、命中重验、保存/读取副本，canonical 变更事件失效派生项并推进 Workspace 代次；CALL 目标 Profile 于 2026-10-09 纳入同一读取视图与 canonical 失效链。编译上下文等其余共享组件不因 Workspace 自动拆分；
3. 已完成（身份第二批）：MemoryLibrary、Artifact 与 lifecycle 内部按归属传递，涉及 actor 可见性的读取另外接收发起者；后台任务独立保存归属与发起者，架构测试限制 Patchouli 内部和五个引擎包使用 `IdentityScope`。跨重启恢复仍由可靠性治理处理，资产旧 scope 接口由 [Todo](../../todo/workspace-asset-ownership-identity-split.md) 跟踪；
4. 对显式 Profile 解析失败、权限拒绝和未指定 Profile 分别返回稳定结果；
5. 将失败 reason 和安全摘要写入可观察事件，但不泄漏不可见正文。

Alice 操作请求与过渡执行身份删除的实现证据见[操作请求集成测试](../../../tests/integration/workspace/test_operation_requests.py)、[CALL Profile 集成测试](../../../tests/integration/alice/orchestration/test_call_profile_requests.py)、[访问边界测试](../../../tests/unit/architecture/test_access_boundaries.py)与 [HTTP 系统 E2E](../../../tests/e2e/system/)。实施历史保留于 [v0.7.0 Alice 能力迁移归档计划](../../archive/plans/v0.7.0-alice-capability-migration.md)。

### Phase S2：Run-local 执行隔离（基础已完成）

1. 已完成：以 `RunSession` 替代共享 FrameScheduler stack，将 frame registry 和 CALL record 收敛为 run-local 状态；Chat application 在上层持有可取消阶段 task，流式输出队列与流序号由每次 run 独占的 `AgentRunStreamAdapter/QueueAgentRunOutput` 持有，运行预算由 frame policy 持有；
2. 继续验证并发 Agent run 的 CALL、READ、WRITE、citation、PendingAtom 和 cancel 不会跨用户或 workspace 交叉；
3. 已完成（2026-10-09）：被调用 frame 继承 caller 的凭据绑定提交函数与观测标签，不携带操作 scope；CALL 权限由 `FrameExecutionPolicy`、Profile capability 与 workspace `profile.read` 分别硬检查，不以 depth 或标签作为授权依据；
4. 明确恢复/重试时不能复用已经失效的授权快照。

### Phase S3：执行资产安全

1. 为 executable Memory 建立来源、信任、审批和 capability 模型；
2. 先实现最小白名单与资源限制，再评估 subprocess/container/sandbox 方案；
3. 为文件、网络、环境变量、进程、输出和超时建立拒绝默认策略；
4. 对取消、超时、异常和部分副作用建立安全终态与 reconciliation；
5. 只有强隔离证据成立后，才允许在产品文案中将 RUN 描述为可执行能力。

### Phase S4：Frontend 与外部契约对齐

1. 前端身份 store 只作为请求上下文，不作为认证/授权来源；
2. 所有请求从同一 identity context 派生，切换/登出清理或隔离 chat、topic、memory cache 和 streams；
3. 与后端认证、session、workspace 和错误契约对齐；
4. UI 明确展示“当前身份/权限”与“后端拒绝”，不把默认 user id 当成登录状态。

## 5. 治理成熟度目标

- 任意 Memory/Artifact/Profile/PendingAtom alias 命中都经过实际归属和适用的可见性校验；L0/L1/L2 已具备 owner/resolver 重验，Patchouli 内部不保存或重新组装操作 scope；
- 两个并发用户使用相同 alias、topic 或 Profile 名称不会读取对方状态；
- 子 Agent、后台 retry 和恢复任务不会扩大或错误继承身份权限；
- FrameScheduler 已删除，cancel、budget、frame registry 与 CALL record 按 run 隔离，并发 CALL/cancel/恢复测试稳定通过；workspace 派生缓存与 Pending registry 的 Workspace 硬边界、权限拒绝和并发回填已有回归，子线程独立身份与 durable ledger 仍按本治理主题验证；
- 指定 Profile 失败不会静默加载全权限 Omni-Doll；未指定 Profile 的 fallback 仍有明确且可观察语义；
- MTP RUN 在未满足可信资产和硬限制时拒绝执行或明确降级，不能把 prompt 当安全边界；
- 前端身份切换不会留下旧用户的请求、缓存、stream 或页面状态；
- 越权、缓存污染、重试跨用户、CALL 深度和执行逃逸均有回归测试；
- Contracts、Alice、Patchouli、Frontend、Help 和 API 错误说明保持一致。

## 6. 依赖与风险

本治理主题依赖[运行时状态持久化与故障恢复](../reliability/durability-and-recovery.md)处理身份快照、任务恢复和 PendingAtom ledger，也依赖[跨子系统幂等性与重试语义](../reliability/idempotency-and-retry.md)防止重复操作跨用户复用。最大风险是把“身份字段已在模型中”误认为“安全边界已经成立”；完成判断必须以越权测试和失败路径为准。具体安全切片只有在绑定版本和验收出口后才形成独立 Plan。
