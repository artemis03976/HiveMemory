---
title: 写入意图（PendingAtom）体系的迁移
status: idea
horizon: current
serves_version: v0.7.0
owner: project
scope: pending-intent-registry-read-consistency-and-materialization
related_docs:
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/external-session-and-topic-projection.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/architecture/decisions/0006-memory-library-custody-criteria-and-independence-contract.md
  - docs/alice/pending-atom.md
  - docs/ideas/workspace-network-task-process-architecture.md
  - docs/ideas/execution-unit-thread-and-environment.md
last_reviewed: 2026-10-09
---

# 写入意图（PendingAtom）体系的迁移

## 0. 文档性质

本文由原 v0.7.0 A4 计划（共享 Pending 与主动记忆写入）于 2026-09-27 退回 Idea：删除了阶段划分、验收门禁、跨计划依赖与文档更新清单，设计内容保留。计划的最后版本见 commit `dda9d9d` 中的 `docs/plans/v0.7.0-a4-pending-memory-intents.md`。

- 要解决的问题：PendingAtom 体系的迁移（owner 表述，2026-09-27）。现状下写入意图的寿命与持有者见[任务进程 Idea](./task-process-table-and-registration-entry.md)第 2.2 节。
- 原计划中的“决定”“冻结”在本文中均为候选设计；原计划留待 A4-0 冻结的事项汇总为第 5 节的开放问题。
- 本文的归属与可见性按原边界宪章 §6.2 的裁定写成（论证见第 4.2 节）：意图由 workspace runtime 的 registry 持有，按 policy 在 Workspace 内可见。2026-09-28 的决定见 0.1：登记位于 workspace，第一版不设 policy。这对应任务进程 Idea Q-2 的选项 C；是否采用取决于 Q-2，以及 Q-1（进程关闭时点）与 Q-3a（哪些工作状态进入进程工作区）。
- 原文依赖的 A2 读取能力与缓存（resolver、L1 cache、backing 读取）来自已作废删除的 A2 计划。2026-10-07 第 1 步已经按最终实现完成登记、解析与失效迁移；当前事实见 0.2 及链接的事实文档，实时派发等剩余内容仍是候选设计。

PendingAtom 解决的是所有 Actor 共有的资源问题：Actor 明确提出 WRITE/UPDATE，而正式 Memory 由后台异步生成时，如何在物化前读回意图、在结算后定位 canonical 结果，并避免读写不一致。它不是 Alice 专属机制。迁移前的 PendingAtomRuntime 曾混合 run/frame/action 关联（执行状态）与 intent 资源状态（按原边界宪章的裁定归 workspace registry，见第 4.2 节）；registry 不依赖 Alice，也不设在 Patchouli，更不给外部 Actor 复制状态机。

### 0.1 owner 的决定（2026-09-28）

以下决定优先于本文其余部分的候选设计；两者不一致时，以本节为准。

**版本与顺序**：写入意图迁移纳入 v0.7.0；放在外部会话与 Topic 投影改造之前或之后都可以。

**流程与两侧的解耦**：

1. Actor 发出主动写入请求；
2. 请求进入 workspace 能力层；
3. 在 workspace 的 PendingAtomRuntime 中登记；
4. 调用 Patchouli 的 API 提交；
5. 收到 Patchouli 一侧发出的全局事件（结算）。

PendingAtom 对 Patchouli 透明，记忆生成对 workspace 透明，两边完全解耦。这与 [ADR-0006](../architecture/decisions/0006-memory-library-custody-criteria-and-independence-contract.md) 一致：记忆库不持有写入意图的登记，写入意图经物化过线，结算回流时注销。

**代码位置**：写入意图登记位于 workspace 的共享设施子包；Alice 的 alias resolver 与缓存迁移到 workspace 的读取视图（[总 Idea](./workspace-network-task-process-architecture.md#d-9-chat-编排与-chat-run-注册表的最终归属) D-9，2026-09-28）。

**与任务进程解耦**：写入意图的生命周期与任务进程完全解耦；生成与结算由 Patchouli 的 memory generation controller 单独管理（[任务进程 Idea](./task-process-table-and-registration-entry.md) Q-1）。

**实时提交**：主动写入意图的提交是 workspace 能力层的一个方法，可以实时响应 Actor 的请求，不必等到一轮对话结束（operation 目录中已有 `memory_intent.submit`）。

**取消与失败**：不再丢弃已经提交的写入意图。

**生成材料**：记忆的历史材料来源只有 Topic 中的内容，conversation session 不是。实时提交时当前一轮的交互记录还没有进入 Topic，因此目前只能采用：

| 选项 | 内容 | 决定 |
|:---|:---|:---|
| A | 只用意图的 focus 与 Topic 中已有的内容，舍弃当前一轮的交互记录 | 采用 |
| B | 进程把当前一轮尚未闭合的执行记录作为快照，随意图一起提交 | 目前不采用 |
| C | 从 conversation session 的历史取材料 | 排除：session 不是记忆的材料来源 |

按选项 A：Gateway 路由到已有 Topic 时，材料是该 Topic 在提交时点已有的内容；路由到新 Topic 时，Topic 要到结算后才创建（任务进程 Idea 1.2），材料只有 focus。生成引擎支持只凭 focus 生成（第 3 节）。

**回读与可见性**：在记忆正式落库之前，PendingAtom 是替代正式记忆的唯一机制，因此直到落库之前，它都必须对后续进程可回读（任务进程 Idea Q-2）。第一版采用简单实现：PendingAtom 不设 policy，默认对全 workspace 开放。PendingAtom 不参与检索，能拿到其别名的一般只有写入它的 agent，狭义上能做到“中间产物归进程”。

- （owner，2026-10-09）UPDATE 意图除外：它携带基础原子的修改内容与坐标，回读跟随基础原子的可读性，读不到基础的 actor 在任何状态下都与不存在相同。起因是第 1 步的 code review 发现其他 actor 能读到私有基础记忆的修改内容。

**结算后的句柄**：结算后 PendingAtom 句柄的生命周期需要重新设计。这一项不阻塞现有计划；兼容期内暂不回收句柄。

**第 1 步的补充决定**（owner，2026-10-06，建立计划前接受的默认决定 W4、W5）：

- **W4 未以 completed 结束的进程**：第 1 步沿用现状，进程关闭时取消本进程仍为 PENDING 的意图；本节“取消与失败：不再丢弃”随第 2 步与实时派发一起实施。第 1 步仍只在 finalize 时派发物化，不取消的意图将永远不被派发。
  - （owner，2026-10-09）CALL 子 frame 未成功结束时，同样撤回它提交且仍为 PENDING 的写入意图，一并视为第 1 步取消语义的一部分；第 2 步改为实时派发后随取消语义重新设计，现阶段不做更复杂的处理。
- **W5 operation 与默认登记**：提交（WRITE、UPDATE）绑定 `memory_intent.submit`，读回绑定 `resource.read`；默认的用户级访问登记加入 `memory_intent.submit`。
- 第 1 步与读取缓存失效、Alice 引用解析的整体迁出合为一份计划，已于 2026-10-07 实施验收：[写入意图登记与读取缓存失效归档计划](../archive/plans/v0.7.0-intent-registry-and-read-cache.md)（历史实施记录，总 Idea 15.11 的补充）。

**分两步实施**：

1. 登记迁出 Alice：登记移到 workspace、对全 workspace 开放的回读、能力层的提交方法、生命周期与进程解耦、结算事件的接收；已实施（2026-10-07，见 0.2）；
2. 实时派发生成（材料按选项 A）；尚未实施，仍在本文讨论。

依赖实时派发的简化，必须与第 2 步在同一份计划中完成，包括：收尾阶段不再派发物化、`InteractionPayload.materialize_tasks` 移除、写入意图不再作为进程工作集中的资源。

**分析与遗留（未决定）**：

- 本文第 4 节要求结算能从权威的任务与领域结果核对，不能只依赖事件订阅者；结算事件丢失时 workspace 一侧的登记如何补齐（例如按 intent_id 查询结算结果），尚未决定；
- 可见范围比迁移前宽：旧 resolver 曾要求 IdentityScope 完全相同，第一版已放宽到整个 workspace。别名为 `draft_{slug}_{4 位十六进制}` 或 `rev_{base_alias}_{4 位十六进制}`，slug 取自标题或内容开头，后缀只有 16 位，所以“只有写入者知道别名”是惯例而不是强制；真正的边界是 workspace，读取时仍要校验 workspace；
- `WriteFocus` 目前只有 content、reason、title，不携带目标 policy；将来 WRITE 若能声明 policy，需要重新审视“pending 不设 policy”；
- 兼容期不回收句柄，意味着进程内的登记会一直增长到重启；
- 取消语义的变化在实施完成后，需要按晋升门禁同步到 AGENTS.md 第 4 节与相关契约。
- （2026-10-06）本方向与 Alice 的能力层调用迁移都涉及 resolver 的迁出。owner 已决定第 1 步先于 Alice 的能力层调用迁移完成：登记与 L0 先进入 workspace，resolver 由 Alice 迁移整体迁出（[总 Idea](./workspace-network-task-process-architecture.md#1511-写入意图迁移第-1-步先于-alice-的能力层调用迁移p-11) 15.11）。影响（分析）：第 1 步中 Alice 的写入意图提交与 pending 读回要经能力层，至少主线程的回调通道（[执行单元 Idea](./execution-unit-thread-and-environment.md#t-4-进程与执行单元之间的回调通道) T-4）需要在第 1 步的计划之前决定；pending 读回若经 workspace 的原子缓存跟随结算后的 canonical 引用，还需要读取缓存失效在前。能力层如何提供 pending 与结算状态的解析结果，见总 Idea P-12a，第 4.1 节的候选设计是其中一个选项；结算后的缓存维护由谁承担，见总 Idea P-11a。

### 0.2 第 1 步实施结果与剩余方向（2026-10-07）

登记、共同引用解析与读取缓存失效已同批形成稳定基线，实施与验收记录见[归档计划](../archive/plans/v0.7.0-intent-registry-and-read-cache.md)。当前事实以[Workspace 架构](../architecture/workspace.md)、[PendingAtom](../alice/pending-atom.md)、[MemoryLibrary](../patchouli/memory-library.md)及[公开路由与事件](../contracts/routes-and-events.md)为准，本文不复制其完整接口。

- workspace `WriteIntentRegistry` 是唯一状态机，PendingAtom 分开保存 `belong_to`、`from_actor` 与 `process_id`，不保存 RuntimeScope；意图在同 Workspace 内可回读（UPDATE 意图跟随基础原子的可读性，2026-10-09），读回及交接任务是独立副本。
- 任务进程把绑定主线程 context/目标的操作通道作为 CPU `execute` 独立参数交给 Alice；WRITE、UPDATE、READ、RUN 资源解析与 CALL 共享引用经过能力层逐次授权。主线程 Profile 解析也已经能力层；SEARCH、引用记录、CALL 目标 Profile 与其 Alice 本地缓存、过渡 `cpu_execution_identity` 尚未迁完。
- core `ReferenceResolution` 提供七种逐项状态。结算后优先按 UUID 读取 canonical，当前 actor 不可读时不交付目标坐标或含 UPDATE 基础坐标的 Pending 副本；UPDATE 继续只接受正式 atom，不接受结算 redirect。
- Store 的 canonical 变更事件内联失效 workspace 原子、旧 alias 与来源 Profile 派生项并推进代次；结算只推进 registry，不回填原子，通知无重试、replay 或未送达对账。
- completed 进程认领意图后仍经 `InteractionPayload.materialize_tasks` 和 finalize 派发；关闭取消尚未认领的 PENDING，不取消 MATERIALIZING。没有实时派发、保留期或 durable ledger，终态句柄保留到重启。

第 2 步仍需决定并实施独立物化派发、Topic 全部可用资料与预算、认领前意图的寿命、移除 interaction 隐式物化字段，以及权威结果对账和容量/保留期。下文有关实时提交、独立路由、完整 Topic 输入及持久化的要求继续作为候选约束；不能从第 1 步完成推断它们已落地。

## 1. 目标边界

| 能力 | 目标权威 | 说明 |
|:---|:---|:---|
| Pending 意图登记、内容、状态、结算关联 | workspace runtime registry（第 4.2 节） | 唯一状态机，不依赖 Alice，也不设在 Patchouli |
| Pending/canonical 引用分派与结算跟随 | workspace 读取能力面（resolver） | L0 registry 本地查表，L2 复用 canonical backing 读取；不留 Alice 专属三级 resolver |
| 物化任务与 canonical 生成 | Patchouli generation/domain | 任务进度和 Pending 可读内容分开 |
| run/frame/action 关联 | Alice run/frame | 只持有执行关联；意图状态不再由 Alice 持有 |
| completed 认领、延迟物化交接与进程关闭取消 | workspace task process | 第一批兼容行为；实时派发后再调整 |
| 外部输入/结果 wire | 外部 Actor adapter 或 MTP adapter | 不复制 Pending 状态 |

Pending 不是 canonical Memory、也不是可淘汰 cache。第一版 PendingAtom 不设 policy，默认对全 workspace 开放（0.1）；原候选设计中“registry 持有不等于整个 Workspace 的 Actor 默认可读”的约束随之不适用于第一版。全局 intent ID 负责定位，提交者、Workspace、operation 和内容可见性分别判断。

## 2. 共同流程与独立路由

```text
WRITE/UPDATE -> Pending 登记 -> 受权 Pending READ
  -> memory_intent.submit(PendingAtomMaterializeTask)
  -> 读取目标 Topic 的全部可用资料（允许空 blocks）
  -> 物化任务接纳 -> canonical revision / discarded / failed
  -> 失效派生 cache -> Pending 解析真实结果 -> 重新授权读取 canonical
```

`interaction.submit` 和 `memory_intent.submit` 是独立路由方法、独立 operation 和独立收据；可以由同一个 application service 承接，不要求两个类。旧 `InteractionPayload.materialize_tasks` 只作兼容迁移字段，最终不能让普通交互入口隐式执行主动物化。

共同输入继续使用 `PendingAtomMaterializeTask`：它的 pending alias、intent_id、source verb、`belong_to`、`from_actor` 和 WriteFocus/UpdateFocus 具有跨 Actor 语义；`from_pending_atom()` 不能成为外部客户端前置依赖。ACK 必须区分 Pending 已登记、物化任务已接纳和 canonical Memory 已产生。

登记和物化提交是两个业务阶段，不强制两个客户端往返。Alice 可先登记、返回 ACK，稍后按运行策略提交；外部工具可组合登记和提交，但接纳失败时需说明已登记的意图如何读回/重试，不能返回模糊的整体 success。输入 scope 必须同时匹配 access 和已登记 Pending 的归属；不可借提交方法挪用别人的 intent_id。模型名的 Task 表示请求，不表示已经入队。

## 3. Topic 资料和 Session 关系

主动生成需要读取对应 Topic 在共同契约确定的资料读取时点的全部可用 blocks、summary 和 provenance，再按领域预算编译；不能隐式退化为“最近五条”。Topic 为空时，有效 WRITE/UPDATE 仍可进入生成；Topic 读取失败、越权、scope 冲突和成功取得空资料必须区分。没有 Topic 或没有交互不等于没有有效主动意图。

本文只讨论资料消费与生成预算；Topic 输入分类、prepare handle 是否可用、省略 Topic 的行为、资料绑定/快照时点与重试保留见[外部会话与 Topic 投影](./external-session-and-topic-projection.md#44-向主动意图交接资料)第 4.4 节与第 8 节。生成侧只提供约束，不另行决定是否创建空 Topic；如契约要求准备/释放，消费共同领域能力，不复制 prepare/cleanup 算法或跟随 Session cursor 重选资料。UPDATE 仍须验证目标 Memory、授权和已知基版本；空交互不免除这些条件。

“全部可用”指 Topic 资料契约所选读取时点的实际资料快照；已折叠且未保存的原始 blocks 无法凭空恢复，原文保全由折叠专项负责。应先取得该时点全部可用材料，再按生成预算编译，不让 adapter 截取最近几条代替领域读取。

代码现状（2026-09-27 复核）：`patchouli/control/memory_generation/coordinator.py` 的 `submit_active()` 仍使用 `recent_blocks(5)`；MemoryGenerationEngine 在没有上下文且没有 WRITE/UPDATE focus 时才跳过。候选方向是保留生成引擎支持意图独立生成的能力，修正上游资料获取，而非让 adapter 自行拉 blocks。

交互是可选来源（conversation session 不是记忆的材料来源，见 0.1），交互资料交接见[外部会话与 Topic 投影](./external-session-and-topic-projection.md)第 4.4 节：需要纳入某次交互时，通过授权结果查询确认其 applied 和实际 topic_id；路由关联本身不证明内容已应用，不要求独立 TopicAssignment 实体。意图接纳后使用契约规定的资料绑定，不因物化失败回滚已应用交互，也不让 Topic 清理影响仍被已接纳任务依赖的资料；保留/释放方式见外部会话 Idea 第 8 节，尚未决定。

## 4. Pending 读取、结算和生命周期

| 阶段 | 对外语义 |
|:---|:---|
| WRITE 未物化 | 返回登记内容/焦点和 Pending 状态，不标成 Memory |
| UPDATE 未物化 | 返回修订意图、目标引用和已知内容；只有 instruction 时不伪造最终正文 |
| 接纳后 | 返回真实任务/领域阶段；Actor 断连不取消已接纳工作 |
| canonical 已产生 | 原 Pending 引用可解析真实结果，canonical 读取再次授权 |
| 原 canonical 引用 | 返回当前已提交版本；是否提示调用者自己的未完成修订待定，不静默叠加 Pending 正文 |
| discard/fail/cancel/expire | 明确区别，不伪造成功引用、不自动重建意图 |

`task.observe` 不自动授予 Pending 内容或最终 Memory 的读取权。稳定 intent identity 用于响应丢失后的查询/重试；相同 identity 不同 payload 返回 conflict。通知事件仅作观测，不能作为唯一终态真相。当前实现不承诺跨重启恢复；若要引入持久化需要另行规划迁移。

Pending snapshot、materialization task、task result 和 settlement projection 分别保存交接信息，不复制终态推进器。结算须可从权威任务/领域结果核对，不能等待事件订阅者才变正确。提交前取消与接纳后取消有不同承诺，cancel 只开放领域支持的范围。并发 UPDATE 的版本冲突仍由 canonical 更新规则判断；Pending 的原意图可读不等于所有 search 已能看到最终 Memory，也不默认加入全局搜索。

### 4.1 共同引用读取与 alias resolver 归属

2026-09-23 修订（取代 2026-09-17 裁定）：按原边界宪章重裁归属（论证见第 4.2 节）——Pending 资源权威状态机是 **workspace registry**（不依赖 Alice，也不设在 Patchouli）；Pending/canonical 分派、结算跟随、终态解释和 canonical 读取协作由 workspace 读取能力面（alias resolver）执行，L0 是 registry 本地查表，L2 复用 canonical backing 读取。共同读取对相同 access、引用和资源状态给出相同领域结果；MTP、管理 UI 和外部协议只改变输入转换与呈现。管理操作仍使用自己的 operation/policy，不能把结果一致理解为权限相同。

旧 `agent_runtime/aliases/resolver.py` 曾依赖 `MTPExecutionContext`、Alice Pending runtime、Atom cache 和 MTP 异常，现已删除。当前能力层接收可信 access 与资源引用，授权后委托 workspace resolver 返回 core `ReferenceResolution` 的 Pending 内容/状态及 canonical 读取结果；领域错误由 adapter 映射为 MTP 或外部错误。结果不暴露权威 Pending 对象或缓存内部的共享可变引用，也不要求调用方创建 RuntimeScope。canonical 分支保留完整 `MemoryAtom`（原 A2 读取契约） 的独立副本及其中已有的来源和版本，直接支持 MemoryCompiler，不强制转换为裁剪型 MemorySnapshot。

```text
System / Alice MTP / 外部 read adapter
    -> 相同的 workspace 读取能力边界（带 access，逐次 operation 授权）
    -> workspace alias resolver
         ├─ L0 Pending ref -> registry 本地查表 -> 授权读取意图/终态
         │                  └─ 已结算 -> 目标引用 -> L1/L2 当前授权下的 canonical 读取
         └─ canonical ref -> L1 缓存 / L2 backing 路由（冷读，库侧校验纵深防御）
    -> Pending 内容/状态快照或完整 canonical 原子副本 -> 各 adapter 编译/呈现
```

原 L0/L1/L2 是历史查找顺序，不是三层等价缓存：Pending 是尚未物化意图的权威状态，cache 才是可丢弃派生副本。resolver 与 registry 同在 workspace runtime（原宪章裁定）；L0 是本地查表，L2 经公共 backing 路由冷读（过线契约见[总 Idea](./workspace-network-task-process-architecture.md) 7.1.3，不是递归）。能力边界承接共同入口，内部复用 backing 读取即可；不强制增加独立 service 类。

| 读取对象/状态 | 共同语义 |
|:---|:---|
| canonical alias/UUID/ref | 返回当前可读的已提交完整 `MemoryAtom` 副本；命中缓存与冷读语义一致，不裁剪成 MemorySnapshot |
| Pending 未物化 | 返回授权范围内的原意图、已知内容和阶段；不伪造最终正文 |
| Pending 已结算 | 保留请求引用和 settlement 关系，经当前授权读取真实 canonical；知道 Pending 不授予目标内容读取权 |
| discarded/failed/cancelled/expired | 明确领域终态；只有通过相应可见性检查后才能披露，不伪装为已生成 Memory |
| 未知或不可见引用 | 遵循 A1 访问边界和防泄露错误投影；不回退到别的 Workspace，也不改查同名的另一种资源 |
| 原 Memory 有未完成 UPDATE | 仍读取已提交 Memory；读对应 Pending 才看到修订意图，不默认为正文叠加 Pending |

在 canonical 读取基线上需要确定的扩展部分：引用种类/命名空间、共同引用解析的固定状态结果、批量逐项结果、Pending 内容读取与 canonical 读取所需 operation 及资源权限、目标不可见时可披露的 settlement 字段。认证上下文复用 A1 统一网关的结果，不为不同引用另建认证凭据。原 A2 契约中的 `read -> MemoryAtom | None`、`retrieve_by_aliases/retrieve -> list[MemoryAtom]` 和 `get_agent_profile -> AgentProfile` 保持稳定，不把这些方法改成随调用方或引用种类改变的返回类型。

Pending 阶段与 settlement 关系确需状态结果时，由共同引用解析方法明确表达；canonical 分支的资源值仍是完整 MemoryAtom。第 1 步已采用 `resolve_references` 与 core `ReferenceResolution`，按输入顺序逐项关联；后续外部 wire 仍需决定，不能借兼容恢复 MemorySnapshot/ProfileSnapshot，或把 RetrievalResponse 当作领域状态容器。各主体对同一操作使用同一路由；引用解析内部复用 canonical 读取实现，不复制领域链，也不要求外部先查 Pending 再自行读取 Memory。旧 MTP/Passive 的输出包装留在 adapter。

READ、RUN、UPDATE 和 Profile 类型读取可以复用同一资源解析事实，但各自继续校验操作条件。解析到代码不授予执行权，解析到 Profile 不授予身份切换；UPDATE 的目标/基版本规则归主动写入。Profile 定义只从正式可读资源解析，未物化 Pending 不能被当作可执行 Profile；若接受结算后的引用，须沿共同跟随规则取得正式资源后再验证类型。不同操作的错误映射由兼容矩阵记录，不能把“统一解析”误作所有操作都接受所有 Pending 状态。

候选设计在同一读取能力面增加 Pending/settlement 分支（L0 分支在 resolver 内本地完成），负责状态结果、批量逐项关联和 adapter 映射；保持 canonical 返回类型与授权语义，不因跟随 Pending 结算引用重新引入正常命中的逐次回源查询。不存在必须由 Alice 拼装多个领域能力的中间业务层。

Pending 终态不被当作普通 Memory 负缓存淘汰；其保留与过期按权威状态生命周期处理。失效和结算推进不依赖 Alice 订阅者。settlement 经既有事件链（`pending_atom_settler`/bridge）加速 registry 终态回写，但事件是加速不是真相——终态以 intent_id 锚定的权威任务/领域结果对账核对（事件协作见总 Idea 7.1.3；本节"通知事件仅作观测"约束继续有效）。

旧 resolver 曾以 `pending.runtime_scope.identity_scope == context.identity_scope` 判断可见性；第 1 步已按 Workspace 硬边界及 `resource.read` 进行回读，Pending 不设 policy。后续若增加 Pending policy，仍不能要求外部构造 RuntimeScope，也不能重新引入整个 scope 相等而把兼容 session 字段变为权限条件。Pending 引用类型或命名空间必须明确，未知/不可见 Pending 不回落为同名 canonical；Pending 与 canonical 的可见性分别处理，不能通过 redirect 泄露不可读目标的内容或未获准披露的元数据。

结算指向应以权威 canonical identity 定位，alias 用于展示或兼容查找；若 alias 已被重新绑定，不得让旧 Pending 跟随到另一份资源。缓存丢失、失效或没有可用 alias 时，仍能按已结算的目标 identity 经授权读取；不存在和不可见不能被伪装为物化仍在进行。具体错误投影与 legacy settlement 缺失 identity 的处理待定（第 5 节）。

### 4.2 候选归属的论证（原边界宪章 §6.2）

原宪章把 pending registry 判给 workspace runtime，论证分四层：

1. **判据面**：pending 是未被接收的要约——可撤销（CANCELLED 是既有状态）、没有任何入库承诺、从未进入库的管护，所以它在 Actor 手里（[ADR-0006](../architecture/decisions/0006-memory-library-custody-criteria-and-independence-contract.md) 的管护权判据）。
2. **决断规则面**：intent 是 Actor 的即时工作产物——由 Actor 组成、需要同步读回；generation 是异步任务的消费者，不因消费取得所有权（ADR-0006 的工作产物规则）。这一条推翻了 2026-09-17 把 pending 判给“Patchouli 资源侧”的裁定。
3. **不增不减面**：generation 消费 materialize task 是既有交接，不变；settlement 的报告通道从“经 bridge/event 写回 Alice runtime”改为总线事件协作（registry 订阅加权威对账，见[总 Idea](./workspace-network-task-process-architecture.md) 7.1.3），属于出边通道的形态变化，不是领域扩张。
4. **结构收益**：resolver 的 L0 变为本地查表（registry 与 resolver 同在一处）；“外部 Actor 可读回意图”“意图活得过单次 run”“单一权威状态机”三项动机都得到满足，权威状态机即 registry。

执行拆分已于第 1 步完成：旧 Alice `PendingAtomRuntime` 已删除，intent/alias/内容/状态/settlement 关联归 registry；run/frame/action 仍是 Alice 执行状态，进程认领和取消关联归任务进程。旧实现是历史迁移起点，不是共享实现的依赖。

ADR-0006 的判据只裁定写入意图不归记忆库；上述论证进一步把它判给 workspace runtime。2026-09-28 已决定：登记位于 workspace 的共享设施，生命周期与任务进程解耦（0.1；[总 Idea](./workspace-network-task-process-architecture.md#d-9-chat-编排与-chat-run-注册表的最终归属) D-9）。

## 5. 开放问题

原计划中留待 A4-0 冻结的接口与迁移事项如下；0.1 的决定与 0.2 的第 1 步结果已经确定，其余部分保留为后续问题。

| 事项 | 需要回答的问题 |
|:---|:---|
| registry 的持有者与生命周期 | 持有者已决定为 workspace，生命周期与任务进程解耦（0.1）；结算后句柄的生命周期待重新设计，兼容期不回收；不含 Alice run/frame 管理，不新增 Workspace 业务转发层 |
| 登记/read/resolve/submit | 第 1 步的登记、读回与 operation 已定；实时派发与外部 wire 仍需决定，外部不依赖 RuntimeScope |
| 共同读取/引用与结果 | 第 1 步已采用中立逐项结果、UUID 跟随与目标不可读时的脱敏；外部协议投影及 legacy settlement 后续兼容仍需决定 |
| intent identity 与幂等 | ID 签发、客户端键映射、直接提交时登记关系、同键异载荷 conflict 与未知接纳查询 |
| 状态与收据 | 登记、任务接纳、结算的阶段，权威结果关联，内容读取与观察权的差异 |
| Topic 与预算 | 实时提交的材料按 0.1 选项 A；Topic 资料契约见外部会话 Idea 第 8 节；本文只讨论全部材料的预算编译，不另定省略 Topic、handle 或快照保留规则 |
| 运行清理与切换 | 第 1 步已切为单一 registry/resolver 并删除 Alice 旧调用方；第 2 步需改变 completed 认领与关闭取消策略 |
| 保留期与关闭 | 容量拒绝、期限、expired/unknown、关闭时未完成任务的结果；不声称跨进程耐久性 |

迁移按单一状态所有者逐步切换，必要的兼容 shim 只委托同一实现。`PendingAtomMaterializationTask` 是讨论时的泛称，实际共同模型沿用代码中的 `PendingAtomMaterializeTask`，不再创建同义类型。

## 6. 设计需满足的行为约束

以下约束来自原计划的验收要求，是候选设计必须满足的行为，不是测试计划：

- 无交互、无 Session 的主动写入可以进入生成；Topic 缺失、越权、为空与读取失败分别表达；多于五个 blocks 不被静默截断；UPDATE 仍验证目标与基版本；
- 无 Alice run/frame 也能读回 WRITE/UPDATE 意图；只存在一份权威状态机和领域解析实现，旧/新状态机与 resolver 不同时作为成功来源；
- 重复键同载荷定位同一提交、同键异载荷 conflict；响应丢失后可查询；过期可解释；重复提交不产生第二个 canonical；
- 原 Pending 引用在 cache 命中与未命中时定位同一真实结果，并重新授权 canonical 读取；Pending 与 canonical 权限可以不同；
- 失败、取消、丢弃终态可区分；alias 重绑定后旧 Pending 不跟随到另一份资源；缓存清空后按 UUID 冷读；同名引用不跨类型回落；
- 原 Memory 存在未完成的 UPDATE Pending 时，读取仍返回已提交版本；
- 接纳后断连或关闭不撤销已接纳工作，结果可解释；回滚只切换 adapter，不删除已接纳的 Pending。
