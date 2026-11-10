---
title: 退出时保存待生成的记忆工作
status: archived
archived_at: 2026-11-10
implemented_by: PR #213
superseded_by: docs/patchouli/generation.md
owner: patchouli
target: 版本之外的工作（v1.0 范围内的桌面客户端，候选清单 D-03 跨重启状态的第一步）
scope: shutdown-saves-unfinished-memory-generation-work-instead-of-running-it-and-resubmits-it-after-next-start
updates:
  - docs/patchouli/README.md
  - docs/patchouli/perception.md
  - docs/patchouli/generation.md
  - docs/components/runtime-and-bus.md
  - docs/system/composition.md
  - docs/help/setup.md
  - docs/governance/reliability/durability-and-recovery.md
code_paths:
  - src/hivememory/patchouli/system.py
  - src/hivememory/patchouli/runtime/core.py
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/patchouli/services/memory_generation.py
  - src/hivememory/patchouli/control/memory_generation/
  - src/hivememory/components/work_queue/
  - src/hivememory/config/patchouli.py
related_docs:
  - docs/governance/reliability/durability-and-recovery.md
  - docs/patchouli/generation.md
  - docs/patchouli/perception.md
  - docs/ideas/pending-intent-migration.md
---

# 退出时保存待生成的记忆工作

> **状态**：Archived（2026-11-10，已实施，实施记录见第 10 节）。当前事实见[记忆生成](../../patchouli/generation.md#62-提交阶段关闭时保存与启动时继续)第 6.2 节与[Patchouli](../../patchouli/README.md#42-维护任务与关闭)第 4.2 节。以下为实施时的计划原文，链接已按归档位置改正。方向由 owner 于 2026-11-10 决定，见第 1.1 节；第 4 节的做法是当时的候选设计，按原样落地。对应桌面客户端（[ROADMAP](../../ROADMAP.md) 第 5.1 节）中跨重启状态的第一步。不绑定版本。

## 1. 背景和问题

后端正常关闭时，Patchouli 依次：

1. 关闭话题折叠，排空交互提交队列（不等待 LLM）；
2. 结算全部驻留话题：冻结材料，建立 SETTLE 任务放进生成队列，任务随即开始执行；
3. 最多等待 `patchouli.shutdown.generation_wait_timeout_seconds`（默认 30 秒），等的是合并之外的全部生成任务，包括排队中的；
4. 到时仍未结束的任务取消（原因 `shutdown_timeout`），停止队列，释放模型。

由此带来的问题：

- **点了退出还要等。** 每个话题一次提取调用，加上查重与写入。owner 验收第一个桌面原型时，结算一个话题约 6.5 秒，整个退出 9.6 秒；话题多时还要更久。用户只看到点了退出、程序没有动静，有误导性。
- **等完仍会丢。**
  - 30 秒内没做完的任务被取消，其中可能有 agent 经 MCP 写入、已经拿到 ACK 的 WRITE 与 UPDATE。
  - 合并任务不等，尚未执行的合并在关闭时丢失，两条相似的记忆都留下。
- **桌面端随 Windows 关机或注销退出时，留给应用的时间很短**，等 LLM 多半来不及。

已有的基础：

- 结算已经把话题的全部交互记录与折叠记录冻结进任务规范，任务进入队列后不再依赖话题本身（[感知与短期话题](../../patchouli/perception.md#7-维护与关闭)）。
- 生成队列的工作信封（`patchouli.memory_generation`，`schema_version` 1.1）把完整的任务规范编码为规范 JSON，解码时拒绝缺字段、多字段与非规范的载荷（[记忆生成](../../patchouli/generation.md)）。
- 队列的存储经 `WorkStorePort`，目前只有进程内实现；版本化载荷只用于进程内冻结与一致性校验。

### 1.1 owner 决定（2026-11-10）

1. 任何退出都走同一条路：Ctrl+C、托盘退出，以及桌面端随 Windows 关机或注销退出。没有结算的话题仍然结算、推入生成队列，但退出时不做实际生成；未完成的生成工作保存到磁盘，下次启动完成后再开始。
2. 这项工作单独建计划，不并入桌面计划。

## 2. 目标与非目标

### 2.1 目标

1. 退出不再等待提取与合并的 LLM 调用：结算产生的任务、排队中的写入与合并、在提交之前被打断的任务都保存下来。
2. 下次启动后，这些工作重新进入生成队列执行，使用关闭时冻结的同一份材料、归属与发起者，结果与没有退出时相同。
3. 保存的内容读不回时（文件损坏、格式版本不认识），不悄悄丢掉：读不回的内容保留在旁边的文件中，日志与运行时事件报告原因，启动照常进行。

### 2.2 非目标

- **崩溃、强制结束进程与断电**：驻留话题与排队中的工作仍只在内存中，照旧丢失。随写随存属于 SQLite WorkStore（[恢复治理](../../governance/reliability/durability-and-recovery.md#46-sqlite-workstore-持久化门槛与设计约束)第 4.6 节），不在本计划。
- **写入句柄跨重启**：写入意图登记仍只在内存中，重启后旧句柄读不到，这与现在相同；恢复的任务生成的记忆照常落库，可以检索（[写入意图后续方向 Idea](../../ideas/pending-intent-migration.md)）。
- **话题跨重启继续积累**：退出时仍然结算话题，不保存驻留话题本身。
- **任务记录跨重启**：重启后旧的任务 ID 查不到，恢复的工作以新任务出现（主动写入的任务 ID 沿用 `active:{intent_id}`）。
- **其他关闭等待**：任务进程（最多 10 秒）、Import Bus 的在途导入与交互提交队列的排空不变。

## 3. 当前状态和缺口

| 环节 | 现在 | 本计划之后 |
|:---|:---|:---|
| 结算产生的 SETTLE 任务 | 立即执行，最多等 30 秒 | 入队但不执行，写进保存文件 |
| 排队中的 WRITE、UPDATE | 30 秒内没轮到或没做完的被取消 | 写进保存文件 |
| 合并任务 | 不等，丢失 | 写进保存文件 |
| 在跑的任务 | 等到 30 秒，之后取消 | 提交之前的立即中止并写进保存文件；已进入提交的写完再结束 |
| 本话题主动写入的记录 | 只在内存，随进程丢失 | 写进对应的 SETTLE 任务 |
| 下次启动 | 生成队列为空 | 读回保存文件，重新接纳 |
| 退出耗时 | 结算加最多 30 秒的生成 | 结算与本地写入，不含 LLM 调用 |

## 4. 目标设计

### 4.1 关闭顺序

Patchouli 停止时：

1. 关闭折叠，排空交互提交队列（不变）。
2. 生成队列停止取新工作：已在执行的继续，入队照常接受。这是通用工作队列的新能力，现在的停止把“不再取工作”与“不再接受入队”放在同一步。
3. 结算全部驻留话题（统一结算时序不变），SETTLE 任务入队、不执行。
4. 处理在跑的任务（第 4.2 节）。
5. 把队列中全部未执行的工作写进保存文件（第 4.3 节），再停止队列。
   - 这些任务在本进程内以取消结束，原因 `shutdown_persisted`。
   - 不为其中的 WRITE、UPDATE 发布写入意图的取消事件：它们在下次启动后还会结算。
6. 取消模型预取，释放推理线程（不变）。

配置项 `patchouli.shutdown.generation_wait_timeout_seconds` 删除。

### 4.2 在跑的任务

生成数据面分两段：

- **提交之前**：为结算保存来源交互 Artifact，调用 LLM 提取、修订或合并，查重。这一段除了交互 Artifact，不写任何东西。
- **提交**：写版本记录、upsert 中期存储、提升附件；合并任务是写回保留记忆、删除被吸收的记忆。只有本地写入（Artifact 文件、嵌入编码、Qdrant），一般在秒级。

退出时取消在跑的任务：

- 处在提交之前的立即中止，按未执行写进保存文件，下次从头执行。
- 已进入提交的不被打断，写完后正常结束，结果与结算照常发布，关闭等待它结束。
- 进入提交时由数据面登记这一事实，提交这一段不响应取消（例如以 shield 执行），关闭据此区分两种任务。

生成 lane 不自动重试（`max_attempts=1`），原因是失败的任务可能已经写入一部分。这里只重跑还没开始写入的任务，与这条理由一致。

代价：被中止的任务已经保存过一份交互 Artifact，重跑会再保存一份，前一份没有记忆引用。孤立 Artifact 是恢复治理中已有的缺口。

### 4.3 保存文件

- **位置**：数据根下的 `.hivememory/pending_generation.json`，与 Artifact 目录相同，按工作目录解析相对路径。由新配置项 `patchouli.shutdown.pending_work_file` 给出，放在被删除的等待时限所在的配置段。
- **内容**：文件格式版本，加上每项工作的 `kind`、`schema_version` 与规范载荷，即队列现在编码的 JSON。只复用队列已有的 codec，不另写序列化。
- **来源**：控制器持有每个已接纳任务的规范，由它把未执行的任务重新编码写出，不需要存储接口提供列出载荷的方法。
- **写入**：先写临时文件再原子替换。没有待保存的工作时不写文件。
- **主动写入的记录**：控制器在内存中按话题记下本话题主动写入结算到的记忆，SETTLE 执行时据此让被动提取不改写这些记忆。写出 SETTLE 任务之前，把该话题的记录补进任务规范的 `topic_write_memory_ids`；重启后执行时，与那时新记下的记录取并集。
- **不保存**：任务 ID 与状态、终态记录、写入意图登记。

### 4.4 下次启动

- **时机**：Patchouli 启动、生成队列启动之后读保存文件。只入队，不阻塞启动；执行在后台进行，嵌入模型还在加载时生成等待加载，与现在相同。
- **顺序**：读出后先删除文件，再逐项经控制器重新接纳，与新任务走同一条路（建立任务、发布 created 事件）。接纳过程中崩溃会丢失这些工作，与其他崩溃相同，但不会在下一次启动重复生成。
- **读不回的内容**：
  - 整个文件无法解析，或文件格式版本不认识时，文件改名保留在原目录（`pending_generation.rejected-<时间>.json`）。
  - 单项的 `kind` 或 `schema_version` 不认识、解码不是规范载荷、或接纳被拒绝时，这些项写进同样命名的文件，其余照常接纳。
  - 两种情况都记录错误日志，发布带稳定原因码的运行时事件，启动继续。
- **容量**：保存的工作在关闭时都已被队列接纳，数量不超过队列容量；启动时队列为空，可以全部接纳。
- **跨版本**：桌面升级正是“旧版退出、新版启动”，保存的工作一定会跨版本。以后改动工作信封的格式时，同一改动必须让新版本读得回上一版本保存的工作（为上一版本注册解码），否则升级时这些工作进入 rejected 文件。本计划不改信封格式。

### 4.5 观测

- **关闭**：`patchouli.shutdown_drain` 的完成事件与日志汇总报告：按来源分的保存数、等待提交结束的任务数、保存文件是否写出；不再报告超时取消数。
- **启动**：发布一次恢复事件，报告读回数、重新接纳数与写进 rejected 文件的数；失败带稳定原因码。
- 事件与原因码的命名遵循[日志与运行时事件](../../governance/observability/logs-and-events.md)第 2 节。

## 5. 受影响的系统、契约与数据

- **components 的工作队列**：运行时与 supervisor 增加“停止取新工作、仍接受入队”；现有的停止照旧，先停取工作，再按 lane 收尾。
- **Patchouli**：
  - 停止顺序（`system.py`、`runtime/core.py`），结算时序本身不变；
  - 生成数据面登记进入提交，提交这一段不响应取消；
  - 控制器写出与读回保存文件、补上主动写入的记录、以 `shutdown_persisted` 结束被保存的任务且不发布写入意图的取消事件。
- **配置**：删除 `patchouli.shutdown.generation_wait_timeout_seconds`，新增 `patchouli.shutdown.pending_work_file`。
- **数据**：数据根下新增保存文件与可能出现的 rejected 文件。
- **公共契约**：公共路由与全局事件不变。被保存的 WRITE、UPDATE 在本进程内不发布取消事件；关闭与恢复的运行时事件载荷有变化。
- **桌面端**：不需要改动，托盘退出随之变快。

## 6. 兼容与迁移

- **删除的配置键**：随仓库发布的 `config.yaml` 没有它，设置页也不暴露。手写进 `config.local.yaml`，或在 `.env` 中设置 `HIVEMEMORY__PATCHOULI__SHUTDOWN__GENERATION_WAIT_TIMEOUT_SECONDS` 的部署启动会失败，错误指出键名，升级说明写明删除。
- **第一次以新版本启动**：没有保存文件，行为与现在相同。
- **回退到旧版本**：旧版本不读保存文件，文件留在原处；再次升级后读回。
- **启动脚本**：`ps1` 与 `sh` 脚本停止后端时最多等 60 秒，这个上限不变，实际退出会快得多。

## 7. 测试和验收标准

| 不变量 | 归属层 |
|:---|:---|
| 停止取新工作之后，队列仍接受入队但不执行新工作，已在执行的工作照常结束；随后的停止照常收尾 | `tests/unit/components/work_queue/test_runtime.py` |
| 关闭后，排队中的 SETTLE、WRITE、UPDATE 与合并任务都出现在保存文件中，关闭期间没有一项开始执行；被保存的 WRITE、UPDATE 不发布写入意图的取消事件 | `tests/unit/patchouli/control/test_memory_generation_controller.py` |
| 提交之前被取消的在跑任务进入保存文件；已进入提交的任务写完、发布结果，不进入保存文件 | 同上；数据面进入提交的登记在 `tests/integration/patchouli/test_memory_generation_pipeline.py` 的现有用例中扩展 |
| 读回后重新接纳的任务规范与关闭时相同（同一规范载荷）；文件在接纳之前删除 | `tests/unit/patchouli/control/test_memory_generation_queue.py` 或控制器测试 |
| 读不回的项进入 rejected 文件，其余照常接纳，启动不失败 | 控制器测试 |
| 保存的 SETTLE 携带本话题主动写入的记录，重启后执行时与新记下的记录取并集 | 控制器中现有的主动写入记录测试 |
| shutdown drain 结算话题后不再等待生成，报告保存数 | `tests/unit/patchouli/runtime/test_shutdown_drain.py`（改写原有的超时取消用例） |
| 关闭时有一个驻留话题；退出过程不调用提取，新系统启动后由它生成记忆 | 一条集成用例，放在 `tests/integration/patchouli/test_perception.py` |

- 删除等待生成 30 秒、超时取消的旧测试，不写“配置键不存在”的测试。旧键被拒绝，由现有的未知键拒绝测试覆盖。
- 手动验收：
  - `ps1 serve` 下对话产生一个话题后按 Ctrl+C，数秒内退出，日志报告保存了 1 项；
  - 再次启动后保存文件被删除，记忆在后台生成；
  - 桌面端从托盘退出时同样数秒内退出。

验证命令：

```bash
pytest tests/unit/components/work_queue tests/unit/patchouli -q
pytest tests/integration/patchouli -q
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85
ruff check . && black --check .
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
```

## 8. 风险与待决问题

- **提交的边界**：要按 SETTLE、WRITE、UPDATE 与合并逐个核对提交之前确实只写交互 Artifact。合并先写回保留记忆、再删除被吸收的记忆，两步都在提交之内。
- **重跑留下孤立的交互 Artifact**：见第 4.2 节。
- **停止取新工作与入队之间的竞争**：停止取工作必须先于结算生效，否则结算的任务可能在保存之前被取走执行。由第 2 步的工作队列测试守护。
- **候选设计**：第 4 节的做法都不是 owner 决定，评审时可以调整，包括：
  - 用关闭时写出的文件，而不是 SQLite WorkStore；
  - 在跑的任务按提交边界取消重跑；
  - 写入意图登记不跨重启；
  - 读不回的内容进入 rejected 文件；
  - 文件位置与配置项名称。
- **待决**：无阻塞问题。

## 9. 完成时必须更新的当前文档

- [Patchouli](../../patchouli/README.md)：第 4.2 节关闭顺序；当前限制中“短期话题关闭时尽力结算”“生成任务进程内”两条。
- [感知与短期话题](../../patchouli/perception.md)：第 7 节关闭。
- [记忆生成](../../patchouli/generation.md)：
  - 合并任务在关闭时丢失的说明；
  - 工作信封的版本化载荷用途；
  - 关闭时取消任务的说明；
  - 当前限制中进程内工作与本话题主动写入记录两条。
- [运行时机制](../../components/runtime-and-bus.md)：第 3.5 节“Backpressure、取消与关闭”，停止取新工作。
- [System 组合根与生命周期](../../system/composition.md)：关闭顺序中 Patchouli 的部分。
- [安装与启动](../../help/setup.md)：停止后端的说明。
- [恢复治理](../../governance/reliability/durability-and-recovery.md)：第 1 节表中“Active topic”与“Memory generation task”两行。
- 收尾时：本计划归档，[ROADMAP](../../ROADMAP.md) 第 5.1 节与 [Plans 导航](../../plans/README.md)同步。

## 10. 实施记录

2026-11-10 实施，在分支 `feat/shutdown-saves-pending-generation` 以 PR #213 合并，第 4 节的设计按原样落地。与计划不同，或计划没有写到的地方：

- **进入提交的登记方式**：
  - 本地路由 `generation.execute_spec` 增加关键字参数 `on_commit_started`，数据面开始写入之前同步调用它，控制器记在任务句柄上（`commit_started`）。
  - 提交这一段用已有的 `join_shielded` 执行：期间收到的取消等写完再传出。
  - 合并任务从写回保留记忆起算提交，写回之前的重读与校验仍可中止。
- **取消与开始写入之间的竞争**：控制器先取消还没开始写入的在跑任务，等全部在跑任务结束，再按“是否开始写入”决定要不要保存。取消送达之前恰好开始写入的任务会写完、不保存，不会重复生成。
- **其他来源的取消也受影响**：用户取消任务或生成 lane 的 300 秒时限，如果恰好落在提交阶段，现在会等写入完成再以取消或超时结束，不再中途打断写入。代价是这种情况下写入意图报告取消或失败，记忆却已落库；写入本身出错时，错误随取消一起丢弃，不另记日志。
- **本话题主动写入的记录**：
  - SETTLE 原先在执行开始时取出并清除这条记录，现在执行成功后才清除，在关闭时被中止的 SETTLE 因此还能把它写进保存文件。
  - `GenerationRequest.topic_write_memory_ids` 改为按排序输出：保存的载荷在另一个进程里解码时要重新编码并比对，集合的迭代顺序随进程变化，不排序时读回会被判为非规范载荷。
- **`wait_all` 删除 `include_merges` 参数**：它只服务于原来关闭时的等待。
- **观测**：
  - `patchouli.shutdown_drain` 的完成事件改为报告保存数（按来源）、等待写完的任务数与文件是否写出；没写出时状态为 `completed_without_saved_work`，级别 error。
  - 启动时的读回作为一次 `patchouli.restore_saved_generation` 操作发布开始与完成事件；有内容写进 rejected 文件时状态为 `completed_with_rejections`，级别 error。
  - 读不回的文件改名为 `pending_generation.rejected-<UTC 时间>.json`。
- **测试隔离**：保存文件的默认路径相对工作目录，从仓库根运行测试时会写进本机的数据目录、被下一次真实启动读回。测试的公共 fixture 把默认路径改到每个测试自己的临时目录，指定了路径的测试照旧。
- **集成用例的位置**：第 7 节写在 `test_perception.py`，实际放在 `tests/integration/patchouli/test_memory_generation_pipeline.py`。那里已经装配了协调器、控制器、真实数据面与中期存储，覆盖从结算接纳、保存、读回到落库的整条链；结算时序本身没有改动，由感知的现有用例覆盖。
- **故意改坏验证**：分别去掉提交阶段的屏蔽、停止取新工作、按是否开始写入的判断、保存时补上主动写入的记录、保存任务不发布取消事件，对应的新测试都会失败。

验证：

- 后端单元与集成测试（不含 `live_llm`、`e2e`、`slow`）3342 通过、1 跳过，覆盖率 94.89%；
- `ruff check .`、`black --check .`、`mypy` 通过；
- 测试运行后仓库的 `.hivememory/` 中没有保存文件。

收尾时第 9 节列出的当前文档已经更新。第 7 节的手动验收（`ps1 serve` 下按 Ctrl+C 退出、再次启动后记忆在后台生成，以及桌面端从托盘退出）由 owner 在日常使用中进行，不阻塞归档。
