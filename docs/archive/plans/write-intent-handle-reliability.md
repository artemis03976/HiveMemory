---
title: 写入句柄的可靠性
status: archived
archived_at: 2026-11-12
implemented_by: PR #221
superseded_by: docs/workspace/read-view-and-write-intents.md
owner: workspace
target: v1.0 范围（候选清单 M-10，以及跨重启持久化 D-03 中写入句柄与生成任务记录的部分）
scope: persist-write-intent-handles-and-generation-task-history-across-restarts-expire-handles-after-seven-days-resolve-unknown-dispatch-by-idempotent-retry-and-record-update-base-version
updates:
  - docs/workspace/read-view-and-write-intents.md
  - docs/alice/pending-atom.md
  - docs/patchouli/generation.md
  - docs/contracts/mtp.md
  - docs/contracts/mcp.md
  - docs/frontend/chat-view.md
  - docs/frontend/desktop-client.md
  - docs/help/desktop-client.md
  - docs/system/configuration.md
code_paths:
  - src/hivememory/workspace/intents/registry.py
  - src/hivememory/workspace/read_view.py
  - src/hivememory/workspace/resolution/memory.py
  - src/hivememory/workspace/capability/actor/memory.py
  - src/hivememory/core/models/pending.py
  - src/hivememory/patchouli/contracts/memory_intent.py
  - src/hivememory/patchouli/control/memory_generation/
  - src/hivememory/engines/memory_compiler/
  - src/hivememory/system/assembler.py
  - src/hivememory/system/system.py
  - src/hivememory/server/__main__.py
  - src/hivememory/config/workspace.py
  - src/hivememory/config/patchouli.py
related_docs:
  - docs/ideas/pending-intent-migration.md
  - docs/governance/reliability/durability-and-recovery.md
  - docs/archive/plans/shutdown-pending-generation-persistence.md
---

# 写入句柄的可靠性

> **状态**：Archived（2026-11-12，已实施，实施记录见第 10 节）。当前事实见[读取视图与写入意图](../../workspace/read-view-and-write-intents.md#31-写盘读回与保留期)第 3.1、4、5 节、[Patchouli 生成](../../patchouli/generation.md#33-mode-c主动-update)第 3.3、6 节与[耐久性与故障恢复](../../governance/reliability/durability-and-recovery.md)。以下为实施时的计划原文，链接已按归档位置改正。依据[写入意图的后续方向 Idea](../../ideas/pending-intent-migration.md)第 2、3、6 节与候选清单 M-10。owner 的决定（2026-11-12）：
>
> - 先修复 M-10 与其他写入句柄的问题；
> - 写入句柄的登记表做持久化，正常关闭时写盘、启动时读回；
> - 句柄以 7 天为过期界限：登记表最多保留 7 天内的记录，超过的删除；
> - 记忆生成任务列表仍服务全部生成来源，已结束的保留最近 100 条，不再因重启丢失，同样只保留 7 天以内；登记表不接入这个页面；
> - 不新增按 `intent_id` 查询结局的只读路由，结局只经全局事件通知，workspace 不因此依赖 Patchouli；结局不明时改为重试已有的提交路由，事件靠持久化与启停顺序不丢，7 天到期兜底。
>
> 其余内容是候选设计，不是 owner 决定，可在评审时调整。

## 1. 背景和问题

写入与更新（MTP 与 MCP 的 WRITE、UPDATE）先在 workspace 的登记表中登记意图，交给 Patchouli 生成队列，ACK 返回 `draft_`/`rev_` 句柄；正式记忆生成后，Patchouli 发布结算事件，登记表把句柄改指正式记忆（[读取视图与写入意图](../../workspace/read-view-and-write-intents.md)）。2026-11-12 调查发现四个问题：

1. **重启后句柄失效**：
   - 登记表（`workspace/intents/registry.py`）只在内存中，重启即清空。
   - 关闭时保存的生成工作（[退出时保存待生成的记忆工作](./shutdown-pending-generation-persistence.md)）在下次启动后继续执行并结算，但登记表中已没有句柄，结算事件按“intent_id 不匹配”被忽略；已经结算的句柄同样丢失。
   - 结果是重启后再读这些句柄得到“不存在”，而记忆其实已经生成或仍在生成。桌面端常驻后重启频繁，harness 常把句柄留在上下文里稍后读取，这种情况会经常出现。
2. **结局不明时句柄停在“生成中”**：派发时总线异常、或接纳结果既不是接纳也不是拒绝时，ACK 报 `dispatch_unknown`，句柄停在 MATERIALIZING，没有任何机制让它离开。
3. **句柄不过期，登记表只增不减**：终态句柄保留到重启；`PendingAtomStatus.EXPIRED` 没有产生方；读取侧把 EXPIRED 与 CANCELLED 都按不存在交付，编译器中终态写入意图的渲染分支与 `expired` 引用种类走不到（Idea 第 3 节）。
4. **UPDATE 不记基础版本**：登记时只记基础原子的 UUID，生成时读取那一刻的最新版本；两个 UPDATE 并发，或排队期间用户在管理页修改了内容，后写回的会静默覆盖前者（管理页修改记忆可见性计划已把“排队中的 UPDATE 覆盖期间的手动编辑”记为已知限制）。

另外，对话页右栏的“记忆生成任务”列表读 Patchouli 生成队列的任务记录：已结束的只保留最近 100 条，重启即清空。

**已核对的事实**：

- 全局总线的 `publish` 依次 `await` 每个订阅者，没有队列，不丢弃；结算事件只在登记表没有订阅时丢失，也就是重启、关闭过程或崩溃。
- 登记表在装配时订阅结算事件（`SystemAssembler` 中 `workspace_read_view.subscribe`），早于 Patchouli 启动与恢复保存的工作；关闭时 `workspace_read_view.close` 排在 `patchouli.stop` 之后。
- 提交路由 `patchouli.public.memory_intent.submit` 以 `intent_id` 为生成任务的幂等键：同一意图重复提交命中原任务并返回它的接纳结果，不产生第二个任务。

## 2. 目标与非目标

### 2.1 目标

1. 正常重启前后，句柄照常可读：在途的继续等结算，已结算的照常改指正式记忆。
2. 句柄保留 7 天，超过的删除；之后读取得到明确的“句柄不存在或已过期”，不伪装成仍在生成。
3. 派发结局不明时，以有限次重试得到确定结果；仍不确定的句柄最晚 7 天后离开登记表。
4. UPDATE 的基础在排队期间被改过时，显式失败并说明原因，不静默覆盖。
5. 记忆生成任务列表在正常重启后保留已结束的任务（最近 100 条、7 天以内）。

### 2.2 非目标

- **崩溃恢复**：进程被强制结束时，登记表、任务历史与保存的工作一起丢失，与现在保存生成工作的承诺相同；每次变化即写盘的方式不在本计划。
- **按 `intent_id` 查询结局的只读路由**（owner 排除）。
- **启动时重放结算事件**：第 4.3 节的启停顺序已覆盖正常关闭，崩溃时任务历史本身也会丢失，重放价值有限。
- **生成材料不含当前一轮**（Idea 第 8 节）：依赖流式同步与长时间运行 Agent 的折叠，属于记忆核心。
- **句柄的可见性与 policy**：在身份与访问主线中。
- **会话、导入轮次、附件等其他状态的跨重启**：D-03 的其余部分。

## 3. 当前状态和缺口

| 环节 | 现在 | 本计划之后 |
|:---|:---|:---|
| 登记表 | 只在内存中，重启清空；只增不减 | 正常关闭时写盘、启动时读回；7 天以前的记录删除 |
| 重启后的结算事件 | 句柄已不在，事件被忽略 | 启动时先读回登记表，再恢复保存的工作，事件落到句柄上 |
| 派发结局不明 | 报 `dispatch_unknown`，句柄停在“生成中” | 用同一意图重试提交，得到确定的接纳或拒绝；仍不明的 7 天后删除 |
| 过期的句柄 | 永不过期；`EXPIRED` 无产生方 | 7 天后删除，读取得到 `expired` 解析结果 |
| UPDATE 基础 | 不记版本，静默以最新版本为底 | 登记时记版本，生成时不一致则以原因码失败 |
| 生成任务列表 | 已结束的最近 100 条，重启清空 | 同样最近 100 条且 7 天以内，正常重启后仍在 |
| 删除记忆数据 | 删 Artifact、长期归档、保存的工作、预设标记 | 另删登记表与任务历史两份文件 |

## 4. 目标设计

### 4.1 两份记录与各自的权威

| | 写入句柄登记表 | 生成任务历史 |
|:---|:---|:---|
| 所有者 | workspace | Patchouli |
| 内容 | 句柄的状态机：alias、`intent_id`、状态、意图正文、结算指向 | 生成工作的执行记录：四种来源（结算、写入、更新、合并）的任务快照 |
| 读取方 | Agent 读句柄（MTP READ、MCP `memory_read`） | 对话页右栏的“记忆生成任务”列表（`/api/v1/memory-tasks`） |
| 保留 | 登记后 7 天 | 已结束的最近 100 条，且结束后 7 天以内 |
| 持久化 | 正常关闭时写盘，启动时读回 | 同左 |

两份记录只在写入与更新的结局上重叠：任务 `active:{intent_id}` 的结局就是句柄的结局。Patchouli 的任务记录是执行结果的权威，登记表是唯一的句柄状态机，只经结算事件更新，不读取 Patchouli 的任务记录；前端只读任务历史，不读登记表。两边的 7 天各自起算，差异对用户不可见。

### 4.2 登记表的持久化与 7 天保留

- **文件**：新增配置项 `workspace.write_intents.record_file`（缺省 `.hivememory/write_intents.json`，相对数据根）与 `workspace.write_intents.retention_days`（缺省 7）。文件写法与保存的生成工作相同：带格式版本，原子写入；读不回时改名保留为 rejected 文件，登记表以空表启动并记录日志。
- **写盘**：系统关闭时，`workspace_read_view.close` 在解除订阅之后、Patchouli 停止之后写盘。关闭时正在写入的那几项此时已经结算，它们的事件先进登记表。
- **读回**：装配时读回，在订阅结算事件与 Patchouli 恢复保存的工作之前完成。读回时跳过 7 天以前的记录。
- **删除**：登记新意图时顺带删除登记满 7 天的记录（按 `created_at`），不论状态；登记表在 7 天内的写入量有限，扫描的代价可以接受。写盘前同样过滤一次。
- **不保存的内容**：线程句柄、工作 actor 与 `IdentityScope` 本来就不在登记中；`process_id`、`thread_id` 只供观测，照常写入。

### 4.3 结算事件不丢

- **启动**：登记表读回之后才订阅；Patchouli 恢复保存的工作排在其后，恢复的工作结算时，句柄已经在表中。
- **关闭**：结算事件的订阅直到 Patchouli 停止之后才解除，登记表在此之后写盘。
- **生成任务历史**：Patchouli 持久化任务历史时，不把“关闭时保存”的工作（取消原因 `shutdown_persisted`）记为已结束：它在下次启动后以同一任务 ID（写入与更新为 `active:{intent_id}`）继续执行，历史中留下“已取消”会与恢复的任务冲突。

### 4.4 派发结局不明：重试已有的提交路由

- 现在的 `_submit_intent` 在总线异常或接纳结果不明时立即报 `IntentDispatchUnknownError`。改为在同一个屏蔽取消的派发任务中，以同一个 `MemoryIntent` 重试提交，有限次数并退避，候选为最多 3 次、间隔 0.5、1、2 秒：
  - 得到“接纳”：照常交还 ACK；
  - 得到“拒绝”：句柄转为失败，报 `dispatch_rejected`；
  - 仍然不明：照旧报 `dispatch_unknown`，句柄留在“生成中”，最晚 7 天后随保留期删除。
- 重试依赖提交路由的幂等：原任务没有结束时不会被淘汰，重试紧跟在不明之后，命中的就是原任务。
- 重试期间调用方取消时，与现在一样只影响调用方，派发任务继续到有结果为止。

### 4.5 过期句柄的读取

- 句柄删除之后，读取 `draft_`、`rev_` 开头而登记表中没有的 alias，解析为现有的 `expired` 引用种类，文案说明“句柄不存在或已过期（保留 7 天）”，不再与普通的“找不到”混在一起。MCP `memory_read` 的未解析列表同样给出这句说明。
- `PendingAtomStatus.EXPIRED` 继续没有产生方：过期表达为解析结果，不存入登记表。实施时核对它与相关的状态迁移是否还有使用方，没有就删除（模型中的枚举值不进入持久化文件以外的契约）。
- 编译器中终态写入意图的渲染分支（`pending_read_settled`、`pending_read_failed`、`pending_read_cancelled` 等）：实施时确认在生产中仍然走不到之后删除，终态只经引用解析结果表达。

### 4.6 UPDATE 记录基础版本

- 登记 UPDATE 时，在 `UpdateFocus` 中记下基础原子的 `meta.version`，随 `MemoryIntent` 交给 Patchouli。
- 比较放在已有的重读处：Familiar 在提交修订之前已经按复合键重读基础记忆，现在只处理策略与存在性（[Patchouli 生成](../../patchouli/generation.md)第 3.3 节）。版本与记下的不一致时不写回，任务以领域错误失败，原因码候选为 `memory.update.base_changed`（第 6.1 节的领域错误规则）。
- 句柄需要知道失败原因：写入意图的失败结算现在不带原因。失败事件与句柄增加一个可选的原因码，登记表记下它；读取失败的句柄时，基础被修改的给出“基础记忆在排队期间被修改，请重新读取后再更新”，其余失败的说明不变。
- 版本相同照常写回。PR #217 记下的“排队中的 UPDATE 覆盖期间的手动编辑”这一已知限制随之消除。

### 4.7 生成任务历史的持久化

- **文件**：新增配置项 `patchouli.shutdown.task_history_file`（缺省 `.hivememory/memory_tasks.json`）；保留条数沿用生成队列已有的已结束保留数（100），保留天数新增 `patchouli.shutdown.task_history_retention_days`（缺省 7）。
- **写盘**：Patchouli 停止时，把已结束的任务快照（不含 `shutdown_persisted` 的取消）写盘，最近 100 条且结束于 7 天以内。
- **读回**：Patchouli 启动时、恢复保存的工作之前读回，作为已结束的历史。`MEMORY_TASK_LIST` 与单个任务的读取在队列中的任务之外合并这份历史；同一任务 ID 以队列中的为准。历史只读，不能取消，也不参与接纳时的幂等判重。
- **前端**：右栏列表不改结构，“显示已结束”在正常重启后仍能看到之前的任务。

### 4.8 删除记忆数据

`--delete-memory-data` 的删除范围加上登记表与任务历史两份文件：句柄与任务都指向记忆库中的内容，记忆删除后它们不能留下。

### 4.9 实施顺序

1. 核对：`PendingAtomStatus.EXPIRED` 与终态渲染分支的使用方；`meta.version` 在 UPDATE 生成路径上的读取位置。
2. 登记表：持久化、7 天删除、启停顺序；过期句柄的解析与文案。
3. 派发：结局不明时的重试。
4. UPDATE 基础版本：登记、传递、提交前比较与原因码；失败事件带上原因码。
5. 生成任务历史：持久化、读回与列表合并。
6. 删除记忆数据的范围；配置项。
7. 当前文档；归档本计划，同步 Idea 的对应各节。

## 5. 受影响的系统、契约与数据

| 位置 | 变化 |
|:---|:---|
| workspace | 登记表持久化与 7 天删除；读取过期句柄解析为 `expired`；派发结局不明时重试 |
| core 模型 | `UpdateFocus` 增加基础版本；`PendingAtomStatus.EXPIRED` 视核对结果删除 |
| Patchouli 契约 | `MemoryIntent` 带上基础版本；新增失败原因码；写入意图的失败事件带上可选原因码；任务列表包含读回的历史 |
| Patchouli | UPDATE 生成时比较基础版本；任务历史持久化 |
| 编译器 | 终态写入意图的渲染分支视核对结果删除；过期句柄的文案 |
| System 与 server | 启停顺序中的读回与写盘；删除记忆数据的范围 |
| 配置 | `workspace.write_intents.record_file`、`workspace.write_intents.retention_days`、`patchouli.shutdown.task_history_file`、`patchouli.shutdown.task_history_retention_days` |
| 数据 | 数据根 `.hivememory` 下新增两份文件 |

## 6. 兼容与迁移

- **升级时**：没有登记表与任务历史文件时以空表启动，升级前发出的句柄照旧在重启后失效，之后的不再失效。
- **MTP 与 MCP**：工具名、参数与 ACK 不变；读取过期句柄的文字由“找不到”变为“不存在或已过期”；UPDATE 多了一种失败原因。
- **删除记忆数据**：多删两份文件，其余不变。

## 7. 测试和验收标准

| 不变量 | 归属层 |
|:---|:---|
| 写盘后读回，句柄的状态、结算指向与正文不变；7 天以前的记录不读回、不写盘；登记时删除满 7 天的记录 | 登记表的单元测试 |
| 读回在订阅与恢复保存的工作之前：恢复的写入结算后，原句柄改指正式记忆 | System 生命周期的集成测试（替身生成侧，真实登记表与总线） |
| 关闭时正在写入的项先结算再写盘 | 同上 |
| 结局不明时按同一意图重试；得到接纳则 ACK，得到拒绝则失败，仍不明则报 `dispatch_unknown` | actor API 写入的单元测试 |
| `draft_`/`rev_` 开头而登记表中没有的 alias 解析为 `expired`；其他不存在的 alias 仍是 `not_found` | 读取视图解析的单元测试 |
| UPDATE 基础版本不一致时不写回、以原因码失败；失败事件带上原因码，句柄记下它 | Patchouli 生成与登记表的单元测试 |
| 任务历史写盘与读回：不含 `shutdown_persisted`、最多 100 条、7 天以内；列表合并历史，同一 ID 以队列为准 | 生成控制器的单元测试 |
| 删除记忆数据包含两份新文件 | `tests/unit/server/test_main.py` 中已有的删除测试 |

手动验收：在开发实例中经 MCP 写入一条记忆，在生成完成前正常重启后端；重启后读取原句柄，先看到“生成中”，生成完成后改指正式记忆；对话页右栏“显示已结束”仍能看到重启前的任务。

验证命令：

```bash
pytest tests/unit/workspace tests/unit/patchouli tests/unit/system tests/unit/server -q
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85
ruff check . && black --check .
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
uv run --no-sync mypy --warn-unused-ignores --platform linux src scripts/typecheck_bus_routes.py
cd frontend && npm run lint && npm test && npm run build
```

## 8. 风险与待决问题

- **崩溃**：强制结束时三份记录（登记表、任务历史、保存的工作）一起丢失，彼此一致，但这段时间的句柄与任务会消失。每次变化即写盘留到 D-03 的后续。
- **重试的前提**：提交路由的幂等依赖原任务记录还在。重试紧跟在不明之后，原任务没有结束，不会被淘汰；如果以后把重试推迟到很久以后，就要重新评估。
- **7 天删除不看状态**：极少数在途超过 7 天的句柄（例如生成侧一直积压）也会被删除，之后读取得到“不存在或已过期”，生成完成的记忆照常可以检索。
- **两份记录的 7 天各自起算**：句柄从登记起算，任务从结束起算；前端只显示任务，差异对用户不可见。
- **待决**：无阻塞问题。重试的次数与间隔、原因码与配置项的命名都是候选，可在评审时调整。

## 9. 完成时必须更新的当前文档

- [读取视图与写入意图](../../workspace/read-view-and-write-intents.md)：登记表的持久化、保留与启停顺序，派发重试，过期句柄的解析；
- [PendingAtom](../../alice/pending-atom.md)：状态与过期；
- [Patchouli 生成](../../patchouli/generation.md)：UPDATE 基础版本与原因码（第 6.1 节），任务历史的持久化（第 6 节）；
- [MTP 契约](../../contracts/mtp.md)与 [MCP 记忆访问](../../contracts/mcp.md)：读取过期句柄与 UPDATE 新的失败原因；
- [对话页](../../frontend/chat-view.md)：记忆生成任务列表在重启后保留；
- [桌面客户端](../../frontend/desktop-client.md)与[使用桌面客户端](../../help/desktop-client.md)：删除记忆数据的范围；
- [System 配置](../../system/configuration.md)：新增的四个配置项；
- [耐久性与故障恢复](../../governance/reliability/durability-and-recovery.md)中写入意图与生成任务两行的现状；
- [ROADMAP](../../ROADMAP.md) 的对应行；
- 本计划移入 `archive/plans/`，从 [Plans](../../plans/README.md) 中删除，在归档索引中登记；[写入意图 Idea](../../ideas/pending-intent-migration.md) 第 2、3、6 节改为指向本计划的已实施说明。

## 10. 实施记录

2026-11-12 在分支 `feat/write-intent-handles` 实施第 4 节，以 PR #221 合并。与计划不同，或计划没有写到的地方：

- **原因码**：基础被修改的失败是新的领域错误 `MemoryUpdateBaseChangedError`，原因码 `workspace.memory.update_base_changed`，与领域错误码的命名一致，不用第 4.6 节候选的 `memory.update.base_changed`。
- **保存工作的信封**（计划没有写到）：`UpdateFocus` 新增必填的 `base_version`，生成工作的信封升到 1.2。生成文档要求改动信封格式时新版本读得回上一版本保存的工作，因此另注册 1.1 的读取器：升级前保存的修订读回时，以任务构建时读到的基础记忆（`existing_memory`）的版本作为基础版本，修订以这一版为底计算，此后被修改过的基础照常拒绝写回。
- **读取失败的句柄**：所有失败的“原因”一栏都显示失败事件带来的原因码，原先这一栏为空；基础被修改的换成“请重新读取后再更新”的提示，其余失败的提示不变。
- **其他 Workspace 的句柄**：读取时与登记中没有的句柄一样按 expired 交付，原先是 not_found；两者都不透露句柄是否存在。
- **删除的代码**：核对后确认没有使用方，删除了 `PendingAtomStatus.EXPIRED`、只为它服务的 `is_terminal`，以及编译器中终态写入意图的渲染分支与文案；编译器收到终态的写入意图时显式拒绝，终态只经引用解析结果交付。
- **读不回的文件**：改名保留的命名函数移到 `config/files.py` 的 `rejected_path`，保存的生成工作、写入句柄登记与生成任务历史共用；登记因此导入 `config.files`，workspace 边界的架构测试为它加了一条例外。
- **派发重试的日志**：每次结局不明记一条 warning（带异常栈），重试用尽另记一条 `reason=dispatch_unknown` 的 error。
- **任务列表的合并**：按任务 ID 合并，本进程的快照覆盖历史，不另设判断；读回的历史与本进程已结束的任务共用 100 条上限，新任务结束时先淘汰最早的历史。
- **测试的数据文件**：根 `conftest` 把关闭时写盘的三份文件的默认路径改到每个测试自己的临时目录，测试不读写本机的数据目录。

实测（2026-11-12）：后端门槛 3398 项通过、1 项跳过，覆盖率 94.67%；ruff 与 black 通过；mypy 按 Windows 与 Linux 两个平台检查均通过；前端没有改动。

没有实测的是第 7 节的手动验收：在实例中经 MCP 写入一条记忆，在生成完成前正常重启后端，再读原句柄与对话页的任务列表。
