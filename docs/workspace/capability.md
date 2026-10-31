---
title: 能力层
status: current
owner: workspace
scope: capability-surfaces-actor-operations-root-api-and-operation-entry
code_paths:
  - src/hivememory/workspace/capability/
  - src/hivememory/core/access.py
  - src/hivememory/workspace/contracts/operations.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/mtp.md
  - docs/contracts/mcp.md
  - docs/contracts/routes-and-events.md
related_docs:
  - docs/workspace/identity-and-authorization.md
  - docs/workspace/execution-threads.md
  - docs/workspace/read-view-and-write-intents.md
  - docs/system/attachments.md
last_reviewed: 2026-10-31
---

# 能力层

能力层是 workspace 对外提供资源能力的地方。它对外有两个互不导入的 API 面：只供保留 `system` 的 root API，与只经操作入口供 actor 使用的 actor API；任务进程另有自己的阶段调用。三者都是授权点，对应 `WorkspaceOperationAuthorizer` 的三种授权（[身份与授权](./identity-and-authorization.md#4-操作授权)）。授权者另有的导入授权（`authorize_import`）不是能力层的调用面：它只供 System 的 Import Bus 服务把外部已经结束的交互提交给 Patchouli，不提供任何资源能力，也不能代替 actor 操作或 root 授权。

## 1. 三个调用面

| 调用面 | 代码 | 授权 | 检查（共用的准入与目标检查之外） | 调用方 |
|:---|:---|:---|:---|:---|
| actor API | `workspace/capability/actor/` | `authorize_actor_operation`（确认是未撤销的执行线程或请求级 context 后，以驻留 Workspace 为目标执行 `authorize_operation`；`request_level=True` 时只接受请求级 context） | 访问登记白名单包含该 actor 操作 | 只有操作入口 `WorkspaceOperationEntry`（第 6 节）；执行线程经线程提交，MCP 入口经门面取得同一入口的请求级提交 |
| root API | `workspace/capability/root/` | `authorize_root` | actor 是保留的 `system`；不读取白名单 | 门面暴露给 server 的 HTTP 管理入口；Profile 注入脚本 |
| 阶段调用 | `workspace/process/`（`TaskProcessRunner`、`CPUAllocator`） | `authorize_stage` | 不检查 actor 操作 | 任务进程自身的编排（[任务进程](./task-process.md#52-阶段调用)） |

**为什么分成三个调用面**：能力层最初从 System 层的应用服务原样迁入，管理方法与 actor 方法混在同一批服务中，管理能力是可以登记给任何 actor 的普通 operation；actor 一侧的方法也与 actor 能做的事对不齐，任务进程的阶段调用还借用 actor 的 operation，没有检索许可的 Agent 在 prepare 阶段就会失败。现在的划分按“谁在做这件事”区分：

- **actor 操作**是 actor 在 Workspace 中主动执行的事，与 MTP 动词一一对应，白名单只表达这一类；
- **root 能力**是管理员（root）对 Workspace 的管理，类似操作系统中 root 账户与普通账户的关系：它不是 actor 操作，不能登记给任何 actor，root 只有保留的 `system` 一个；
- **阶段调用**是进程替 actor 做的系统操作（Gateway 读话题、预检索、CPU 分配、交互提交），actor 没有发起它们，因此不以 actor 的许可为前提。阶段授权返回的 scope 仍以进程 actor 为发起者，资源 owner 照常按该 actor 的身份过滤可见性：预检索不会让其他 Agent 的 PRIVATE 记忆进入输入清单，但它不检查 actor 能否 READ——没有 READ 许可的 actor 能否进一步读取预检索结果，不由预检索负责。

`system` 还有第二种用法：非主动路径（手动、idle、LRU、shutdown 结算）的发起者（[身份与授权](./identity-and-authorization.md#62-后台任务和重试)）。这种用法只经资源 owner 的普通可见性规则读取，从不经过 root API，两者分属不同入口。

## 2. actor 操作目录

`WorkspaceOperation`（`core.access`）只有六项，每项对应一种请求和一个 actor API 方法；某 actor 实际获准的集合只由访问注册表表达。各项互不隐含，新增操作不自动加入已有白名单：

| operation | MTP 动词 | 请求 | actor API 方法 | 说明 |
|:---|:---|:---|:---|:---|
| `memory.read` | READ（含 Agent Profile 原子；CALL 的 `context_refs` 也以此请求逐项解析） | `ResolveReferencesRequest` | `ActorMemoryService.resolve_references` | 交付正式原子时记录引用 |
| `memory.search` | SEARCH | `RetrieveRequest` | `ActorMemoryService.retrieve` | 结果预热原子缓存；不记录引用 |
| `memory.run` | RUN 的记忆中用户工具 | `ResolveRunTargetRequest` | `ActorMemoryService.resolve_run_target` | 交付执行目标时记录引用；`sys_*` 系统工具在 Alice 本地执行，不经能力层 |
| `memory.write` | WRITE | `SubmitWriteIntentRequest` | `ActorMemoryService.submit_write_intent` | ACK 表示已登记并被生成队列接纳 |
| `memory.update` | UPDATE | `SubmitUpdateIntentRequest` | `ActorMemoryService.submit_update_intent` | 基础原子按正式原子 policy 验证；不记录引用 |
| `agent.call` | CALL（只有 Alice 使用） | `DeriveCallThreadRequest` | `ActorAgentService.resolve_call_target` | 以调用方的可见性解析 CALL 目标 Profile，只交回图纸；子线程由操作入口在同一次提交内交给线程表派生（[执行线程](./execution-threads.md#6-call-派生与子线程)）；内置 Profile 同样授权 |

RUN 不要求 READ，WRITE 不授予 UPDATE；只持有 `agent.call` 的 actor 能创建子帧，但没有 `memory.read` 时 `context_refs` 解析失败并被跳过，共享上下文中不出现这些记忆。子帧的操作按被调用方自己的白名单授权，与调用方是否持有对应操作无关。Alice 把线程描述中的操作快照翻译为 MTP 动词（RUN 始终可用），只用于提示词与 frame 执行策略，见 [MTP 契约](../contracts/mtp.md#4-权限)第 4 节。写入意图的登记、派发与回读见[读取视图与写入意图](./read-view-and-write-intents.md)。

## 3. root API

方法与原管理用例一致，授权为 root 授权：只接受保留的 `system`，任何具体 Agent 即使持有全部 actor 操作也以 `root_required` 被拒绝，`system` 也不需要（不能配置）白名单。管理读取经 Patchouli 公开路由直接读取，不进入 actor 读取视图的缓存。

| 服务 | 方法 | 行为 |
|:---|:---|:---|
| `RootMemoryService` | `create_memory`、`list_memories`、`get_memory`、`update_memory`、`record_feedback`、`delete_memory` | 以授权返回的 `IdentityScope` 经 Patchouli public route 执行；创建与编辑只提交 API 字段，原子由 Patchouli 构造与校验，root API 不组装 `MemoryAtom`。列表查询显式排除 `AGENT_PROFILE`，避免普通记忆管理与 Profile 资产混为一类；不存在的 get/update 使用共享的 `ResourceNotFoundError` |
| `RootAgentProfileService` | `create_agent_profile`、`list_agent_profiles` | 只提交创建材料并列出 `AGENT_PROFILE`，Profile 原子的构造、持久化和可见性都由 Patchouli 负责；`agent_config` 是资产内容，不是 System 直接解释的运行时权限 |
| `RootSessionService` | `create_session`、`list_sessions`、`get_session`、`rename_session`、`delete_session` | 委托共享会话服务（[会话资源](./sessions.md)） |
| `RootTopicService` | `list_active_topics`、`settle_topic`、`evict_topic` | 授权后以返回的 scope 经 Patchouli topic route 交给 Topic 所有者，不根据 ID 或缓存自行判断 Workspace 可见性和生命周期 |
| `RootMemoryTaskService` | `list_memory_tasks`、`get_memory_task`、`cancel_memory_task` | Patchouli 将 scope 的目标 Workspace 与任务的 `belong_to` 比对；不存在与越域任务统一隐藏，HTTP 查询和取消都映射为同一 404。不从任务对象推导第二套状态机 |
| `RootWorkspaceAssetService` | `upload_asset` | 编排附件的接收、原子注册和请求内解析，保留首次创建/重放回执语义；在接收与注册副作用之前执行 root 授权，注册与解析交接只使用授权返回的 scope；链路事实见 [Chat 附件链路](../system/attachments.md) |

管理读写按 owner-management 语义执行：在 Workspace 归属硬边界之内访问该 Workspace 的全部 Memory 与 Profile，不执行 Agent 级 `MemoryAccessPolicy` 可见性过滤（[MemoryLibrary](../patchouli/memory-library.md)）。这些服务的“拥有”只指顶层用例入口，不改变后端状态的所有权：例如 `RootMemoryTaskService` 可以取消任务，但任务生命周期仍由 Patchouli 负责。记忆任务是后台系统的任务，不暴露给 Agent：任务的列表、查询与取消只在 root API，没有对应的 actor 操作或 MTP 动词。

附件上传同样遵守这一边界：`upload_asset()` 只是用例入口，文件名规则、受限读取和 SHA-256 计算由附件接收函数实现，解析接纳、来源校验和失败/取消收尾由 `AttachmentParseService` 实现。用例层决定上传串行门覆盖整个请求，并保留 Store 返回的 `created` 标记；它不实现解析算法，也不维护另一份资产状态。

Profile 注入脚本（`scripts/inject_profiles.py`）以登记的本地来源 `hivememory:profile-injector` 和 `system` 声明经认证网关取得请求级 context，再调用同一 root API，结束时撤销 context。

**System 层面的 API**：模型、Provider、配置、运行时事件与日志与 Workspace 资源无关，HTTP 路由直接经门面调用 System 的注册表与服务，不经认证网关、root API 或操作授权。workspace 只负责与 Workspace 资源相关的请求。

## 4. 两种读取视角

| | actor 读取（actor API） | 管理读取（root API） |
|:---|:---|:---|
| 方法 | `resolve_references`、`resolve_run_target`、`retrieve`、`resolve_call_target`（读取目标 Profile） | `get_memory`、`list_memories`、`list_agent_profiles`、`list_active_topics` |
| 可见性 | 按原子的 `MemoryAccessPolicy` 对当前 actor 授权，不可见与不存在都按不存在处理 | owner 管理语义：整个 Workspace 可见，只校验 Workspace 归属 |
| 读取路径 | workspace 读取视图（L0 意图登记、L1 原子/Profile 缓存、L2 冷读回填） | 经 Patchouli 公开路由直接读取，不进入缓存 |

行为许可与资源许可必须同时满足：有 `memory.read` 仍可能被 PRIVATE 记忆拒绝，PUBLIC 记忆也不会使没有读取许可的 actor 获得读取能力。读取视图的结构见[读取视图与写入意图](./read-view-and-write-intents.md#1-读取视图)。

## 5. 授权点的共同约定

root API 与阶段调用只接收访问 context、目标 Workspace 与业务参数；actor API 方法只接收请求对象与线程 context，目标取线程的驻留 Workspace（第 3 阶段只接受驻留目标）。两者都先调用对应授权，再用返回的 `IdentityScope` 构造领域对象并调用资源 owner，不向下传 context；授权失败时不触达资源后端。附件上传只使用授权返回的 scope，检索请求由 actor API 用授权返回的 scope 构造。HTTP 管理路由只调用 root API。

## 6. 操作入口

操作入口 `WorkspaceOperationEntry` 是 actor API 唯一的调用方，门面不暴露 actor API。它服务两类提交：执行线程的线程提交，与不建进程的外部入口的请求级提交。

执行者通过 `workspace.contracts` 的 `OperationRequest` 与线程句柄 `OperationSubmitter` 提交操作。`WorkspaceOperationEntry` 实现公共 `OperationEntry`，线程提交只接受存活线程的 context：`execute` 在分派前调用线程表的 `resolve`，全部 actor 操作因此共用同一个存活判定（[执行线程](./execution-threads.md#3-生命周期与唯一的存活判定)）；线程已结束以 `ThreadEndedError` 拒绝，进程 context、请求级 context 或未登记的线程以 `ScopeRequiredError` 拒绝，均没有副作用。随后 `execute` 只按请求类型把五项记忆请求与线程 context 直接交给对应的 actor API 方法，不拆参数；actor API 逐次授权并组装该次调用的 `IdentityScope`，资源 owner 仍执行归属与 actor policy 检查。

请求级提交 `execute_request` 只供 MCP 入口调用，门面以 `operation_entry` 交出同一个入口：外部 harness 的任务对 HiveMemory 不可见，没有进程与线程，每次工具调用取得一份请求级 context，调用结束即撤销（[MCP 记忆访问](../contracts/mcp.md)）。它接受检索、读取、写入与更新四种请求，与 `execute` 共用同一分派表；可提交的请求类型只写在这一处（[子系统公共契约](../contracts/subsystem-contracts.md#46-操作请求与线程-context)）。写入与更新登记意图后要以一个派发话题派发：线程提交取线程的派发话题，请求级提交没有线程，由调用方以参数 `dispatch_topic_id` 给出，即 MCP 入口经 Import Bus 为这一轮落位得到的话题（[Import Bus](../system/import-bus.md#47-写入时落位)）。检索与读取不得给出派发话题，写入与更新必须给出；其余请求类型、或派发话题与请求类型不符时在分派前以 `TypeError` 拒绝，没有副作用。它不做存活判定，撤销由 actor API 的授权检查。

actor API 的授权 `authorize_actor_operation` 接受线程与请求级 context，哪种提交方式能使用哪些操作由入口决定。写入与更新给出派发话题时以 `request_level=True` 授权，只接受请求级 context：线程 context 走这条路径会绕过线程的存活判定与线程的派发话题，以 `ScopeRequiredError` 拒绝。只有授权者能读出 context 的种类，能力层不解封 context。给出派发话题时不解析线程，意图登记的进程与线程关联为空；派发、ACK、拒绝与结果未知的语义与线程路径相同。请求级 context 到不了 RUN 与 CALL，由入口与线程解析保证：经 `execute` 提交时被线程表拒绝，经 `execute_request` 提交时被类型表拒绝，CALL 的派生先判定调用方存活。

请求只携带操作参数，不暴露访问 context，也不能切换目标 Workspace、指定 process_id 或派发话题；线程提交的意图登记的 `process_id`、`thread_id` 与派发话题取自线程表的解析结果。线程 context 的派生不授予操作许可：`system` 的进程即使派生出线程 context，也因白名单为空而在每个 actor 请求上被拒绝。

CALL 的派生与子线程的结束是入口的另外两个生命周期方法 `derive_child_thread` 与 `end_thread`，不经 `execute`：`DeriveCallThreadRequest` 不是 `OperationRequest`。两者见[执行线程](./execution-threads.md#6-call-派生与子线程)。六项 actor 请求的参数与结果契约见[子系统公共契约](../contracts/subsystem-contracts.md#46-操作请求与线程-context)。

## 7. 引用读取的副作用

引用读取的副作用由 actor API 拥有：READ（`resolve_references`）与 RUN（`resolve_run_target`）共享同一解析与引用记录路径，对实际交付的可读正式 atom 或 redirect 目标记录 `source="workspace.reference_read"` 的 citation，同次请求按正式 UUID 去重，缓存命中仍计数；pending、failed、discarded、not_found 与无可读目标的 redirect 不计数。RUN 在类型检查和工具执行前已交付目标，所以后续执行失败仍计数；CALL context_refs 经 READ 请求，是一次独立读取。UPDATE 基础校验与 SEARCH 不属于引用交付，不因此记录 citation。引用记录的普通失败只记录日志，不改变读取结果；取消继续传播，已经完成的 citation 不回滚。

## 8. 结构守护

架构测试限定：

- `authorize_stage` 只在 `workspace/process/` 的 runner 与 CPU 分配中出现，`get_stage_operations` 与 `get_thread_binding` 只在进程线程管理中出现，`authorize_root` 只在 root API 中出现，`authorize_actor_operation` 只在 actor API 中出现；派生与撤销线程 context 只由进程线程管理调用；
- Alice 除 CPU 驱动与组合入口外不引用访问 context、执行线程或操作入口；CPU 可见的对象（输入清单、线程描述、线程句柄交出的描述）不含访问 context；
- 入口的 `execute` 只把五项记忆请求直接分派到 actor API 方法，没有拆参 handler，请求级提交 `execute_request` 共用同一分派表，CALL 的派生与结束是入口的两个生命周期方法；
- actor API 只被操作入口导入（组合根装配除外）；`execute_request` 只由 MCP 入口调用；MCP 入口不导入 actor API、root API、任务进程与 Import Bus 的服务和编排器，在 Import Bus 中只导入在途轮次的输入模型，落位经门面取得的服务调用；
- 两个 API 目录互不导入，capability 父包只导出共享的读取 backing；server 只导入 root API，操作入口只导入 actor API；actor API 的公开方法、操作目录与请求类型保持一一对应。

不要新增可登记给 actor 的管理 operation，也不要让阶段调用借用 actor 操作。

## 9. 已知限制

- `sys_*` 系统工具不设 Agent 权限，所有 Agent 都可使用全部已注册系统工具（含文件写入与 Python 执行），执行安全的缺口见[身份与执行安全治理](../governance/security/identity-and-execution-safety.md)；
- 能否 CALL 某个 Agent 只取决于调用方的 `agent.call`、目标 Profile 对调用方可见以及被调用方的准入，没有“谁能调用谁”的细粒度策略；
- 只有 `memory.run` 的 actor 可经 RUN 的错误探测可见记忆的类型与存在，不可执行目标也记录引用，见 [RUN 目标的可执行性检查 Todo](../todo/run-target-executability-check.md)。

## 10. 代码与测试入口

- [`workspace/capability/root/`](../../src/hivememory/workspace/capability/root/)（root API）、[`workspace/capability/actor/`](../../src/hivememory/workspace/capability/actor/)（actor API）、[`WorkspaceOperationEntry`](../../src/hivememory/workspace/capability/operations.py)、[`backing.py`](../../src/hivememory/workspace/capability/backing.py)；
- 公共契约：[`workspace/contracts/operations.py`](../../src/hivememory/workspace/contracts/operations.py)；操作目录：[`core/access.py`](../../src/hivememory/core/access.py)；
- 测试：`tests/unit/workspace/capability/root/`、`tests/unit/workspace/capability/actor/`、[`test_root_api_authorization.py`](../../tests/integration/workspace/test_root_api_authorization.py)（逐方法的 root 授权：非 `system` 拒绝且不触达 owner，`system` 无白名单也能调用）、[`test_operation_requests.py`](../../tests/integration/workspace/test_operation_requests.py)（含请求级的检索、读取、写入与更新，以及派发话题与请求类型不符时的拒绝）、[`test_reference_citations.py`](../../tests/integration/workspace/test_reference_citations.py)、[`test_profile_injection_script.py`](../../tests/integration/workspace/test_profile_injection_script.py)、[`test_chat_mtp_operations.py`](../../tests/e2e/system/test_chat_mtp_operations.py)、[`test_chat_operation_whitelist.py`](../../tests/e2e/system/test_chat_operation_whitelist.py)（没有 actor 操作的 Agent 完成 chat，预检索仍按 actor 过滤；只授予 READ 与 SEARCH 的 Agent 输出 WRITE 时被拒绝且不登记意图）。
