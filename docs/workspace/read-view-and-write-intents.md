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
  - docs/patchouli/generation.md
  - docs/architecture/decisions/0006-memory-library-custody-criteria-and-independence-contract.md
last_reviewed: 2026-10-19
---

# 读取视图与写入意图

actor 读取经 workspace 读取视图完成，写入意图在正式落库之前由 workspace 登记。两者同在 workspace：pending 引用在读取器内本地查表，系统只有一个意图状态机。语义检索与正式状态仍在 Patchouli，workspace 不成为第二个记忆库。

## 1. 读取视图

`WorkspaceReadView`（`workspace/read_view.py`）聚合 actor 读取路径的三层：L0 写入意图登记、L1 完整原子缓存与 Profile 缓存、L2 经 `CanonicalReadBackend` 冷读并回填。读取视图提供两个读取器：`MemoryReader`（`.memory`）解析引用，`ProfileResolver`（`.profiles`）读取主进程与 CALL 目标的 Profile。它只负责引用定位、交付授权与缓存，不做检索排序、去重或内容解释，也没有写路径。管理读取不经读取视图（[能力层](./capability.md#4-两种读取视角)）。

语义检索能力预热完整原子缓存，MTP SEARCH 经该能力获得同样行为；prepare 的检索结果不预热。

### 1.1 派生缓存的键控

派生视图按派生源坐标键控：完整原子缓存按 `(WorkspaceIdentity, memory_id)` 寻址并维护 alias 索引，Profile 缓存按 `(WorkspaceIdentity, agent_alias)` 寻址，随条目保存源原子 policy 与 UUID。命中后对当前发起者逐次授权，不缓存授权结论；保存与交付均复制嵌套可变对象。主进程与 CALL 目标都经同一个 `ProfileResolver` 读取并按当前发起者检查源原子可见性：主进程 Profile 由 CPU 分配在阶段授权后读取，CALL 目标经 actor API 先检查 `agent.call`，内置 Profile 同样检查。

这与事实源的规则不同：跨子系统的 WorkspaceAssetStore 等事实源以 `WorkspaceIdentity` 参与资源复合键，在最终边界校验归属（[身份与授权](./identity-and-authorization.md#5-资源归属与寻址)）。

### 1.2 canonical 变更与失效

canonical 变更经 Patchouli local bus 与 bridge 内联转发（只改 `meta.lifecycle` 动态状态的 patch 不发布，缓存原子的统计值可能较旧）；workspace 依次失效原子及 alias、失效源原子的 Profile 条目、推进 Workspace 代次。两个读取器共用一个 `ColdReadGuard`：冷读前记录代次，读取返回后比对，代次变化说明读取期间发生过 canonical 变更，旧值既不回填也不作为成功结果返回，有限次重试后显式失败，避免旧值在变更后重新进入缓存；读取视图关闭后拒绝新读，关闭前发出的冷读结果仍交还调用方但不再回填。通知只携带资源引用，不回放值，不做未送达补齐或重试。发布在提交调用内联完成，订阅者只做纯内存的失效与代次推进，不回填也不调用外部：提交返回时缓存已不可能命中旧值，订阅者也没有会失败而留下过期条目的步骤。组合根装配时即订阅，System 停止时先解除订阅再关闭读取视图（[Workspace](./README.md#52-停止)）。

## 2. 写入意图为什么由 workspace 持有

写入意图在正式落库之前，是 Actor 回读刚写内容的唯一句柄。它是尚未被书库接收的要约：没有入库承诺，也从未进入记忆库的管护，按 [ADR-0006](../architecture/decisions/0006-memory-library-custody-criteria-and-independence-contract.md) 的管护判据不归 Patchouli；它又必须活得过单次 run，供同 Workspace 的其他 Actor 与后续任务进程回读，所以也不归 Alice 的 run/frame 或任务进程的工作集。生成链只是消费意图的异步任务，不因消费取得所有权。两侧互相透明：Patchouli 只接收带登记标识的 `MemoryIntent`，不知道登记的存在；workspace 只看到接纳结果与结算事件，不参与生成决策。登记中的意图既不是 canonical Memory，也不是可淘汰的缓存。

## 3. 写入意图登记

`WriteIntentRegistry`（`workspace/intents/registry.py`）是进程级唯一登记，与任务进程生命周期解耦，记录 `belong_to`、`from_actor` 与只作观测关联的 `process_id`、`thread_id`（同一进程两次 CALL 同一个 Agent 时以线程区分）。同 Workspace 的其他 Agent 可经 `memory.read` 回读；越界与不存在相同。UPDATE 意图携带基础原子的修改内容与坐标，回读跟随基础原子的可读性：读不到基础的 Actor 在任何状态下都得到 not_found。

WRITE 经 `memory.write`、UPDATE 经 `memory.update` 分别授权后登记为 MATERIALIZING，并经 `PatchouliRoutes.MEMORY_INTENT_SUBMIT` 派发；生成队列接纳后才返回 ACK；UPDATE 只接受可读的正式 atom，pending 与 redirect 句柄不能作为基础，登记成功后失效基础原子。登记的归属与发起者取自本次授权结果，进程关联与线程标识取自线程表的一次解析（[执行线程](./execution-threads.md#7-actor-阶段结束与停止)）。

## 4. 引用解析

`MemoryReader`（`workspace/resolution/memory.py`）统一产出 core 的 `ReferenceResolution`：pending、redirect、discarded、failed、atom 与 not_found。SETTLED redirect 的正式目标仍按原子 policy 授权；不可读时清空 canonical 字段与结算视图中的引用，并不交付 pending 记录，避免 UPDATE focus 泄露基础身份。意图和原子结果都是独立副本。新登记不产生 expired，终态句柄保留到重启。

登记中存在的别名只按意图解析：取消、对当前 Actor 不可见或缺少结算载荷都得到 not_found，不回落为同名的正式原子。redirect 优先按结算的 `canonical_uuid` 读取，alias 被重新绑定后旧意图不会跟随到另一份资源，只有缺少 UUID 的结算载荷才按 `canonical_alias` 查找。读取正式原子始终返回已提交版本，进行中的 UPDATE 修订只经它自己的 pending 句柄可见。

UPDATE 的基础冷读在登记的同步段之前完成，冷读跨越线程结束或停止时在同步段拒绝，不产生意图（[执行线程](./execution-threads.md#7-actor-阶段结束与停止)）。引用交付时的 citation 记录属于 actor API（[能力层](./capability.md#7-引用读取的副作用)）。

## 5. 派发与结算

派发开始后由 actor API 持有 shield 保护的任务直到完成，再传播调用方取消；CPU 失败、取消、停止、子线程结束、进程关闭或 CALL 子帧失败都不撤回已提交写入。ACK 表示意图已在 workspace 登记并被 Patchouli 生成队列接纳，不表示正式记忆已经持久化；正式持久化由 Patchouli 生成链结算（[Patchouli 生成](../patchouli/generation.md)）。

确定性拒绝同步以 `mark_failed` 把句柄推进为 FAILED，并向 agent 返回结构化错误；结果未知保持 MATERIALIZING。结算事件仍由 Patchouli 发布，同步失败与随后失败通知幂等；三个结算事件都携带 pending_alias 与 intent_id，登记只在两者都匹配时迁移状态。状态终结后不再被晚到通知重写。

## 6. 已知限制

- 读取视图与写入意图登记均为进程内状态；失效通知不处理未送达补齐，意图句柄不回收也不跨重启恢复，实时派发的结果未知时可能长期保持 MATERIALIZING；
- 写入意图不设 policy，同 Workspace 内持有别名即可回读；别名由标题或内容片段（UPDATE 为基础 alias）加 4 位十六进制后缀组成，不能充当访问控制。结算对账、句柄保留期、意图 policy 与外部提交契约等后续方向见[写入意图后续方向 Idea](../ideas/pending-intent-migration.md)；
- 原子缓存与意图读取交付独立副本，但对象仍可变，不宣称递归冻结。

## 7. 代码与测试入口

- [`WorkspaceReadView`](../../src/hivememory/workspace/read_view.py)（[`cache/`](../../src/hivememory/workspace/cache/)、[`resolution/`](../../src/hivememory/workspace/resolution/)）、[`MemoryReader`](../../src/hivememory/workspace/resolution/memory.py)、[`ProfileResolver`](../../src/hivememory/workspace/resolution/profile.py)、[`WriteIntentRegistry`](../../src/hivememory/workspace/intents/registry.py)；
- 测试：[`tests/unit/workspace/intents/test_registry.py`](../../tests/unit/workspace/intents/test_registry.py)、[`test_intent_registry_and_read_cache.py`](../../tests/integration/workspace/test_intent_registry_and_read_cache.py)、[`test_memory_reader.py`](../../tests/unit/workspace/resolution/test_memory_reader.py)、[`test_profile_resolver.py`](../../tests/unit/workspace/resolution/test_profile_resolver.py)（缓存串扰与授权重验）、[`test_workspace_intent_subscription.py`](../../tests/integration/system/test_workspace_intent_subscription.py)（组合根装配即订阅、关闭即取消订阅）、[`test_inflight_dispatch.py`](../../tests/integration/workspace/test_inflight_dispatch.py)。
