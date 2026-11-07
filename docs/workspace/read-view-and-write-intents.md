---
title: 读取视图与写入意图
status: current
owner: workspace
scope: derived-read-caches-reference-resolution-and-write-intent-registry
code_paths:
  - src/hivememory/workspace/read_view.py
  - src/hivememory/workspace/cache/
  - src/hivememory/workspace/resolution/
  - src/hivememory/workspace/intents/
  - src/hivememory/workspace/capability/actor/memory.py
  - src/hivememory/workspace/capability/operations.py
  - src/hivememory/core/models/pending.py
  - src/hivememory/core/models/reference.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/mtp.md
related_ideas:
  - docs/ideas/pending-intent-migration.md
related_docs:
  - docs/workspace/capability.md
  - docs/workspace/execution-threads.md
  - docs/contracts/mcp.md
  - docs/patchouli/generation.md
  - docs/architecture/decisions/0006-memory-library-custody-criteria-and-independence-contract.md
last_reviewed: 2026-11-06
---

# 读取视图与写入意图

actor 读取经 workspace 读取视图完成，写入意图在正式落库之前由 workspace 登记。两者同在 workspace：pending 引用在读取器内本地查表，系统只有一个意图状态机。语义检索与正式状态仍在 Patchouli，workspace 不成为第二个记忆库。

## 1. 读取视图

`WorkspaceReadView`（`workspace/read_view.py`）聚合 actor 读取路径的三层：L0 写入意图登记、L1 完整原子缓存与 Profile 缓存、L2 经 `CanonicalReadBackend` 冷读并回填。读取视图提供两个读取器：`MemoryReader`（`.memory`）解析引用，`ProfileResolver`（`.profiles`）读取主进程执行者与 CALL 目标的 Profile。它只负责引用定位、交付授权与缓存，不做检索排序、去重或内容解释，也没有写路径。root API 的管理读取不经读取视图（[能力层](./capability.md#4-两种读取视角)）。

语义检索能力预热完整原子缓存：MTP SEARCH、MCP 的检索与任务进程的预检索都经 actor API 的同一个检索方法进入读取视图，结果逐条做归属校验与交付授权后交付，并获得同样的预热。

### 1.1 派生缓存的键控

派生视图按派生源坐标键控：完整原子缓存按 `(WorkspaceIdentity, memory_id)` 寻址并维护 alias 索引，Profile 缓存按 `(WorkspaceIdentity, agent_alias)` 寻址，随条目保存源原子 policy 与 UUID。命中后对当前发起者逐次授权，不缓存授权结论；保存与交付均复制嵌套可变对象。主进程与 CALL 目标经同一个 `ProfileResolver` 读取、共用同一份缓存，但交付语义不同：

- **CALL 目标**是调用方自己的 actor 操作：actor API 先检查 `agent.call`（内置 Profile 同样检查），再按调用方检查源原子可见性，调用方只能唤起它看得见的 Agent；
- **主进程执行者的 Profile** 是用户为这一轮作出的选择：CPU 分配以系统的 root scope 读取、加载哪份图纸取自认证得到的执行者，按管理语义交付（`enforce_actor_visibility=False`）。缓存命中与经 Patchouli `GET_AGENT_PROFILE` 的冷读都只检查 Workspace 归属，不按 scope 的 actor 检查可见性：指向其他 Agent 的 PRIVATE 图纸也能被它自己的进程加载，与 root API 读取记忆的管理语义相同。如果按 Agent 的可见性读取，用户为 Agent 选择的图纸反而可能对它不可见；以 `system` 的可见性读取则读不出任何 PRIVATE 图纸。

管理读取由一个参数切换而非另设读取器，是因为两种语义共用同一份按 `(WorkspaceIdentity, agent_alias)` 键控的缓存与同一条失效路径；缓存条目随存源原子 policy，管理读取填入的条目在之后的 CALL 命中时仍按调用方授权。跳过可见性的管理读取只由 CPU 分配发起，由架构测试守护。

这与事实源的规则不同：跨子系统的 WorkspaceAssetStore 等事实源以 `WorkspaceIdentity` 参与资源复合键，在最终边界校验归属（[身份与授权](./identity-and-authorization.md#5-资源归属与寻址)）。

### 1.2 canonical 变更与失效

canonical 变更经 Patchouli local bus 与 bridge 内联转发（只改 `meta.lifecycle` 动态状态的 patch 不发布，缓存原子的统计值可能较旧）；workspace 依次失效原子及 alias、失效源原子的 Profile 条目、推进 Workspace 代次。两个读取器共用一个 `ColdReadGuard`：冷读前记录代次，读取返回后比对，代次变化说明读取期间发生过 canonical 变更，旧值既不回填也不作为成功结果返回，有限次重试后显式失败，避免旧值在变更后重新进入缓存；读取视图关闭后拒绝新读，关闭前发出的冷读结果仍交还调用方但不再回填。通知只携带资源引用，不回放值，不做未送达补齐或重试。发布在提交调用内联完成，订阅者只做纯内存的失效与代次推进，不回填也不调用外部：提交返回时缓存已不可能命中旧值，订阅者也没有会失败而留下过期条目的步骤。组合根装配时即订阅，System 停止时先解除订阅再关闭读取视图（[Workspace](./README.md#52-停止)）。

## 2. 写入意图为什么由 workspace 持有

写入意图在正式落库之前，是 Actor 回读刚写内容的唯一句柄。它是尚未被书库接收的要约：没有入库承诺，也从未进入记忆库的管护，按 [ADR-0006](../architecture/decisions/0006-memory-library-custody-criteria-and-independence-contract.md) 的管护判据不归 Patchouli；它又必须活得过单次 run，供同 Workspace 的其他 Actor 与后续任务进程回读，所以也不归 Alice 的 run/frame 或任务进程的工作集。生成链只是消费意图的异步任务，不因消费取得所有权。两侧互相透明：Patchouli 只接收带登记标识的 `MemoryIntent`，不知道登记的存在；workspace 只看到接纳结果与结算事件，不参与生成决策。登记中的意图既不是 canonical Memory，也不是可淘汰的缓存。

## 3. 写入意图登记

`WriteIntentRegistry`（`workspace/intents/registry.py`）是进程级唯一登记，与任务进程生命周期解耦，记录 `belong_to`、`from_actor` 与只作观测关联的 `process_id`、`thread_id`（同一进程两次 CALL 同一个 Agent 时以线程区分）；不建进程的请求级提交没有进程与线程，二者为空。同 Workspace 的其他 Agent 可经 `memory.read` 回读；越界与不存在相同。UPDATE 意图携带基础原子的修改内容与坐标，回读跟随基础原子的可读性：读不到基础的 Actor 在任何状态下都得到 not_found。

WRITE 经 `memory.write`、UPDATE 经 `memory.update` 分别授权后登记为 MATERIALIZING，并经 `PatchouliRoutes.MEMORY_INTENT_SUBMIT` 派发；生成队列接纳后才返回 ACK；UPDATE 只接受可读的正式 atom，pending 与 redirect 句柄不能作为基础，登记成功后失效基础原子。登记的归属与发起者取自本次授权结果。

线程提交与请求级提交的写入顺序相同：

1. 以工作 actor 授权 `memory.write` 或 `memory.update`；
2. UPDATE 解析可读的正式基础原子；
3. 取得派发话题，可能要为这一轮第一次建立话题；
4. 无挂起点的同步段：线程提交再解析一次线程，登记意图；
5. 派发。

取得派发话题排在授权与基础冷读之后，被拒的写入因此不会触发话题路由、建立话题或 LRU 结算；它又在同步段之前，期间线程结束或进程收到停止时，同步段照常拒绝登记，已经建立的话题没有写入。取得派发话题失败时（例如建立话题时 LRU 结算的候选都被占用），错误原样交回这次写入，不登记意图。

派发话题是 Patchouli 读取生成材料的话题，有两个来源：

- **线程提交**（Alice 的 MTP WRITE、UPDATE）：经线程表的 `dispatch_topic` 取得进程这一轮的话题。Gateway 指向已有话题时它在 PREPARE 已经冻结；指向新话题时，第一次写入以 Gateway 的标题与摘要建立话题，主线程与 CALL 子线程共享（[执行线程](./execution-threads.md#8-派发话题)）；
- **请求级提交**（外部 harness 经 MCP）：没有线程，调用方给出一个确定派发话题的回调，actor API 在第 3 步才调用它；MCP 入口的回调经 Import Bus 为这一轮确定话题，这一轮回流时沿用同一话题（[Import Bus](../system/import-bus.md#47-写入时确定话题)）。授权取入口认证得到的工作 actor，登记不带进程与线程关联（[能力层](./capability.md#6-操作入口)）。

两种提交登记的是同一种意图，回读、派发、ACK 与结算的规则相同。

## 4. 引用解析

`MemoryReader`（`workspace/resolution/memory.py`）统一产出 core 的 `ReferenceResolution`：pending、redirect、discarded、failed、atom 与 not_found。SETTLED redirect 的正式目标仍按原子 policy 授权；不可读时清空 canonical 字段与结算视图中的引用，并不交付 pending 记录，避免 UPDATE focus 泄露基础身份。意图和原子结果都是独立副本。新登记不产生 expired，终态句柄保留到重启。

登记中存在的别名只按意图解析：取消、对当前 Actor 不可见或缺少结算载荷都得到 not_found，不回落为同名的正式原子。redirect 优先按结算的 `canonical_uuid` 读取，alias 被重新绑定后旧意图不会跟随到另一份资源，只有缺少 UUID 的结算载荷才按 `canonical_alias` 查找。读取正式原子始终返回已提交版本，进行中的 UPDATE 修订只经它自己的 pending 句柄可见。

UPDATE 的基础冷读在取得派发话题与登记的同步段之前完成，冷读跨越线程结束或停止时在同步段拒绝，不产生意图（[执行线程](./execution-threads.md#7-actor-阶段结束与停止)）。引用交付时的 citation 记录属于 actor API（[能力层](./capability.md#7-引用读取的副作用)）。

## 5. 派发与结算

派发开始后由 actor API 持有 shield 保护的任务直到完成，再传播调用方取消；CPU 失败、取消、停止、子线程结束、进程关闭、CALL 子帧失败或 MCP 调用方断开都不撤回已提交写入。ACK 表示意图已在 workspace 登记并被 Patchouli 生成队列接纳，不表示正式记忆已经持久化；正式持久化由 Patchouli 生成链结算（[Patchouli 生成](../patchouli/generation.md)）。

确定性拒绝同步以 `mark_failed` 把句柄推进为 FAILED，并向 agent 返回结构化错误；结果未知保持 MATERIALIZING。结算事件仍由 Patchouli 发布，同步失败与随后失败通知幂等；三个结算事件都携带 pending_alias 与 intent_id，登记只在两者都匹配时迁移状态。状态终结后不再被晚到通知重写。

句柄可以在结算之后改指一次。WRITE 的草稿在查重中与已有记忆 M 相似时，生成链先把它保存为新记忆 N，句柄以 `CREATED` 结算到 N；随后的合并任务把 N 并入 M 并删除 N，Patchouli 再发布改指事件（`pending_atom.merged`），携带指向 M 的 `MERGED` 结算与 N 的 ID（[记忆生成](../patchouli/generation.md#41-合并任务)）。登记只在 alias 与 `intent_id` 都匹配、句柄处于 SETTLED、现有结算为 `CREATED` 且正式身份仍是 N 时，用新结算替换旧结算，其余情况一律忽略，晚到或重复的改指因此不会改写其他句柄。改指不是状态迁移，句柄仍为 SETTLED。合并之前读取句柄得到 N，改指之后按新结算的 `canonical_uuid` 得到 M。

## 6. 已知限制

- 读取视图与写入意图登记均为进程内状态；失效通知不处理未送达补齐，意图句柄不回收也不跨重启恢复，实时派发的结果未知时可能长期保持 MATERIALIZING；
- 改指事件与结算事件同样没有补偿：改指丢失时句柄仍指向已删除的 N，SETTLED redirect 读不到正式目标，按不可读处理；
- 写入意图不设 policy，同 Workspace 内持有别名即可回读；别名由标题或内容片段（UPDATE 为基础 alias）加 4 位十六进制后缀组成，不能充当访问控制。结算对账、句柄保留期、意图 policy 与外部提交契约等后续方向见[写入意图后续方向 Idea](../ideas/pending-intent-migration.md)；
- 原子缓存与意图读取交付独立副本，但对象仍可变，不宣称递归冻结。

## 7. 代码与测试入口

- [`WorkspaceReadView`](../../src/hivememory/workspace/read_view.py)（[`cache/`](../../src/hivememory/workspace/cache/)、[`resolution/`](../../src/hivememory/workspace/resolution/)）、[`MemoryReader`](../../src/hivememory/workspace/resolution/memory.py)、[`ProfileResolver`](../../src/hivememory/workspace/resolution/profile.py)、[`WriteIntentRegistry`](../../src/hivememory/workspace/intents/registry.py)；
- 测试：[`test_operation_requests.py`](../../tests/integration/workspace/test_operation_requests.py)（线程与请求级提交的登记与派发；请求级的更新在基础冷读之后、登记之前确定话题，被拒的请求不确定话题）、[`test_process_intents.py`](../../tests/integration/workspace/test_process_intents.py)（线程提交只有写入确定话题、被拒的写入不确定话题、确定话题期间停止时不登记）、[`tests/unit/workspace/intents/test_registry.py`](../../tests/unit/workspace/intents/test_registry.py)、[`test_intent_registry_and_read_cache.py`](../../tests/integration/workspace/test_intent_registry_and_read_cache.py)（含进程加载 PRIVATE 图纸而 CALL 仍按调用方可见性拒绝）、[`test_memory_reader.py`](../../tests/unit/workspace/resolution/test_memory_reader.py)、[`test_profile_resolver.py`](../../tests/unit/workspace/resolution/test_profile_resolver.py)（缓存串扰与授权重验，管理读取在命中与冷读两条路径交付 PRIVATE 图纸）、[`test_workspace_intent_subscription.py`](../../tests/integration/system/test_workspace_intent_subscription.py)（组合根装配即订阅、关闭即取消订阅）、[`test_inflight_dispatch.py`](../../tests/integration/workspace/test_inflight_dispatch.py)。
