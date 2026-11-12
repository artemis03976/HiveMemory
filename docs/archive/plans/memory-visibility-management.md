---
title: 管理页修改记忆可见性
status: archived
archived_at: 2026-11-11
implemented_by: PR #217
superseded_by: docs/frontend/management-views.md
owner: project
target: 版本之外的工作（身份与访问主线）
scope: read-and-edit-memory-access-policy-through-management-crud-and-the-memory-library-ui
updates:
  - docs/frontend/management-views.md
  - docs/patchouli/memory-library.md
  - docs/patchouli/generation.md
  - docs/workspace/capability.md
  - docs/workspace/identity-and-authorization.md
  - docs/workspace/read-view-and-write-intents.md
  - docs/contracts/routes-and-events.md
code_paths:
  - src/hivememory/core/models/memory.py
  - src/hivememory/server/models/memory.py
  - src/hivememory/server/routers/memories.py
  - src/hivememory/workspace/capability/root/memory.py
  - src/hivememory/patchouli/contracts/memory_management.py
  - src/hivememory/patchouli/contracts/local_routes.py
  - src/hivememory/patchouli/application/memory_management_service.py
  - src/hivememory/patchouli/services/memory_generation.py
  - frontend/src/types/memory.ts
  - frontend/src/services/memoryApi.ts
  - frontend/src/components/memory/
related_docs:
  - docs/ideas/identity-and-access-model.md
  - docs/workspace/identity-and-authorization.md
  - docs/patchouli/memory-library.md
  - docs/frontend/management-views.md
---

# 管理页修改记忆可见性

> **状态**：Archived（2026-11-11，已实施，实施记录见第 10 节）。当前事实见[管理页面](../../frontend/management-views.md#111-可见范围)第 1.1.1 节、[MemoryLibrary](../../patchouli/memory-library.md#12-中期当前可检索书库)第 1.2 节与 [Patchouli 生成](../../patchouli/generation.md#33-mode-c主动-update)第 3.3 节。以下为实施时的计划原文，链接已按归档位置改正。方向来自[身份与访问体系 Idea](../../ideas/identity-and-access-model.md#6-资源可见性与资源授权)第 6 节，原 Todo“Memory 可见性策略前端组件”并入本计划（删除前的最后版本见 commit `3cfdec9d`）。本计划的前提由 owner 于 2026-11-11 决定：
>
> - 记忆的可见性只由用户在管理页修改；被动提取、Agent 写入、手动创建的默认值与 Agent Profile 仍然生成 PUBLIC；
> - TEAM 保留在模型中，HTTP 与页面不开放；
> - 新增的前端组件先在设计画布上出样例，owner 复核后再实施。
>
> 其余内容是候选设计，不是 owner 决定，可在评审时调整。不绑定版本。

## 1. 背景和问题

记忆的可见性机制是完整的，但没有任何入口能让它产生效果：

- **模型与授权已经就绪**：`MemoryAccessPolicy` 有 PUBLIC、PRIVATE、TEAM 三种，只表达读取。检索的存储预过滤、引用解析、Profile 缓存命中与合并守卫都已按它执行；管理读取跳过 Agent 可见性，只检查 Workspace 归属（[身份与授权](../../workspace/identity-and-authorization.md#5-资源归属与寻址)第 5 节、[MemoryLibrary](../../patchouli/memory-library.md)）。
- **产生记忆的路径全部写 PUBLIC**：被动提取与 Agent 写入经生成引擎落库，手动创建与新建 Agent 经 `MemoryGenerationFamiliar.create_external_memory`；手动编辑与合并沿用原有策略。
- **修改策略的存储路径没有生产调用方**：`patch_payload` 已能整体替换 `meta.access_policy`，并发布记忆变更、失效读取视图，但只有测试调用它。
- **生产中没有 Team**：`ActorIdentity.team_id` 从未被赋值，TEAM 记忆对任何 Agent 都不可见。
- **HTTP 与前端没有策略**：记忆接口的请求与响应都没有 `access_policy`，前端没有对应的字段或组件。

四个 harness 共用一个记忆库之后，Agent 之间的可见性开始有实际意义。接入登记的 `allowed_agent_ids` 让每个 harness 只能声明自己的 Agent（[认证与访问登记](../../workspace/authentication.md#1-两阶段认证)第 1 节），PRIVATE 因此能真正挡住其他 harness。现在缺的是用户查看与修改策略的入口。

## 2. 目标与非目标

### 2.1 目标

1. 管理页显示每条记忆的可见范围：所有 Agent，或只有某一个 Agent。
2. 手动创建时可以选择可见范围，不选时为所有 Agent（PUBLIC）。
3. 已有记忆可以修改可见范围，下一次检索与读取即按新策略执行，不推进内容版本。
4. 后台的 Agent 修订（UPDATE）写回时，不把用户设定的策略改回去。

### 2.2 非目标

- **Agent 写入时声明策略**：MCP 的 `memory_write` 与 MTP 的 WRITE 不变，写入意图不加 policy（第 4.6 节说明为什么不需要）。
- **被动提取按参与者设定策略**。
- **TEAM**：不开放，也不引入 Team 归属的来源。
- **Agent Profile 的策略**：Agent 页不加这项设置，Profile 继续为 PUBLIC；经记忆接口修改 Profile 的策略被拒绝。
- **目标 Agent 的存在性**：不校验 PRIVATE 指向的 Agent 是否存在，Agent 删除或改名时也不级联处理（第 8 节）。
- **读取正式记忆时提示调用者自己仍在进行的 UPDATE 修订**：与可见性无关，留在 Idea。
- **修改与执行授权**（P-8）。
- **UPDATE 与手动内容编辑的版本冲突**：排队中的 UPDATE 写回时仍会覆盖这期间的手动内容编辑，与现状相同（第 8 节）。

## 3. 当前状态和缺口

| 环节 | 现在 | 本计划之后 |
|:---|:---|:---|
| HTTP 响应 | `MemoryResponse` 没有策略 | 返回 `access_policy` |
| 手动创建 | 固定 PUBLIC | 可选 PUBLIC 或 PRIVATE，缺省 PUBLIC |
| 手动编辑 | 只能改内容字段 | 还能改策略；只改策略时不推进版本 |
| Agent 修订写回 | 以入队时读到的整条原子为底写回 | 写回前重读，沿用当前策略 |
| TEAM | 模型可以构造，没有来源 | 模型不变，HTTP 入口拒绝 |
| 前端 | 没有策略字段与组件 | 创建与详情弹层共用策略组件，卡片与列表标出 PRIVATE |

## 4. 目标设计

### 4.1 两种可见范围

- **所有 Agent**（PUBLIC）：Workspace 内获准进入的 Agent 都能检索与读取。
- **只有一个 Agent**（PRIVATE，带 `target_agent_id`）：只有这个 Agent 能检索与读取。

策略约束 Agent 的检索与读取，包括 chat 的预检索、MTP 与 MCP；它不影响管理页，管理读取照常看到全部记忆。Agent 能执行哪些操作仍只由访问登记的白名单决定，策略不授予任何操作。

### 4.2 HTTP 契约

- `MemoryResponse` 增加 `access_policy`，形状与 `MemoryAccessPolicy` 相同（`visibility`、`target_agent_id`、`target_team_id`）。
- `MemoryCreateRequest` 增加可选的 `access_policy`，缺省为 PUBLIC。
- `MemoryUpdateRequest` 增加可选的 `access_policy`，缺省为不修改。
- 两个请求中的策略按 `MemoryAccessPolicy` 校验：PUBLIC 不带目标，PRIVATE 必须且只能带 `target_agent_id`，目标不能为空白或保留的 `system`。请求模型另外拒绝 TEAM。校验失败沿用 `request.validation_failed`，逐项说明不符合要求的字段，不新增错误码。
- 不新增路由；Agent 的创建接口不变。

### 4.3 root API 与 Patchouli 契约

- `RootMemoryService.create_memory` 与 `update_memory` 各多接收一个策略参数，原样交给 Patchouli。root API 照旧不组装原子，也不校验目标 Agent。
- `MemoryCreation` 与 `MemoryEdit` 各增加 `access_policy: MemoryAccessPolicy | None`。局部路由 `MEMORY_CREATE`、`MEMORY_UPDATE` 同步增加参数，`MemoryManagementService` 透传。
- 编辑的目标是 Agent Profile 且带策略时，以 `InvalidMemoryFieldError` 拒绝，与普通记忆提交 `agent_config` 被拒绝的方式相同。Profile 创建不传策略，仍为 PUBLIC。

### 4.4 创建与修改策略的写入

所有写入仍由 `MemoryGenerationFamiliar` 完成，它是手动创建与编辑的唯一写入者。

- **创建**：`create_external_memory` 接收策略，缺省 PUBLIC，与原子一起写入。
- **修改**：`update_external_memory` 先按现在的方式处理内容字段（有变化才推进版本、写版本记录），再在策略有变化时以 `patch_payload` 整体替换 `meta.access_policy`。只改策略时只有后一步：不推进内容版本、不写版本记录、不改内容时间，与 [MemoryLibrary](../../patchouli/memory-library.md) 中“`patch_payload` 是动态状态与资源策略的唯一持久化路径”一致。策略与当前值相同时不写入。
- **先内容、后策略**：内容提交写回的是编辑开始时读到的整条原子，策略若先写，会被随后的内容提交改回旧值。两步之间不是原子的：策略更新失败时内容已经提交，请求按失败返回；重新提交同一份编辑时，内容没有变化，只补上策略。
- **生效**：策略的 patch 照常发布记忆变更，workspace 失效该原子并推进代次（[读取视图与写入意图](../../workspace/read-view-and-write-intents.md#12-canonical-变更与失效)第 1.2 节）；检索的存储预过滤读取的是持久化的策略。下一次检索与读取即按新策略执行，不需要另外处理失效。

### 4.5 Agent 修订写回时沿用当前策略

UPDATE 的生成任务在构建时读取基础记忆，并随任务保存（正常关闭时一起写盘）；写回时以这份快照为底提交整条原子（[Patchouli 生成](../../patchouli/generation.md#8-当前限制)第 8 节）。如果用户在这期间修改了策略，写回就会把它改回去。

- UPDATE 写回前按复合键重读基础记忆，确认它仍对发起者可读，再沿用重读得到的策略提交。这与合并写回前的重读相同。
- 基础记忆已被删除或对发起者不可读时不写回，任务以 `ResourceNotFoundError` 失败；缺失与不可读同样按不存在处理。原因码按已有的“其他领域错误取它的 `code`”规则为 `workspace.resource.not_found`，不新增错误类型（实施时的调整）。现在这种情况会把已删除的记忆重新写回。
- 只处理策略：不比对内容版本，这期间的手动内容编辑仍会被覆盖（非目标）。

### 4.6 写入意图不加 policy

Agent 写入与被动提取都生成 PUBLIC，意图结算出的正式记忆对 Workspace 内所有 Agent 可见；意图在结算前按 Workspace 归属回读，两者一致。用户修改策略之后，现有的引用解析已经能正确处理：

- **UPDATE 意图**：回读跟随基础记忆的可读性。用户把基础改为只对其他 Agent 可见后，看不到它的 Agent 回读该意图得到 not_found。
- **已结算的意图**：重定向到正式记忆时仍按正式记忆的策略授权，读不到的 Agent 得不到正式身份（[读取视图与写入意图](../../workspace/read-view-and-write-intents.md#4-引用解析)第 4 节）。

因此本计划不改写入意图的登记与引用解析。

### 4.7 前端

- **样例先行**（owner，2026-11-11）：实施前在设计画布“HiveMemory 界面样稿”上补一组样例（“记忆图书馆 · 可见范围”两行），覆盖创建弹层与详情弹层中的策略组件、Agent 的选择、卡片上的 PRIVATE 标记，以及目标 Agent 不在列表中时的显示。owner 复核定稿后再写前端代码。
- **组件**：创建弹层与详情弹层共用一个可见范围下拉框：第一项“所有 Agent”，其后按“外部接入”“对话 Agent”分组直接列出 agent_id（owner，2026-11-11：不加“只有”“仅”之类的前缀，卡片、列表与详情中的标记同样只显示锁形图标与 agent_id）。选择本身就带出目标，不存在选了“只有一个 Agent”却没选 Agent 的状态，不需要额外的必填校验（样例复核时由两个单选项加必选下拉框改来：现有元数据栏只有 256px）。
- **详情中的位置**（owner，2026-11-11，样例复核时选定）：元数据栏的标签卡片下面单独一张“可见范围”卡片，原有的元数据、时间线、标签三张卡片不动。查看时显示值与一句解释，指向的 Agent 不在列表中时加一条提示；编辑模式下卡片里放下拉框，随详情的“保存”一起提交，只在策略有变化时带上 `access_policy`。“不是保密”的完整说明在创建弹层中直接显示，在详情卡片中放进悬停提示。
- **候选 Agent**：取当前 Workspace 访问记录中的具体 `agent_id`（不含 `system` 与用户级记录）、Agent Profile 的别名与内置的 `omni_doll`，去重后列出。数据来自已有的访问记录接口与 Agent 列表接口，不新增后端接口。记忆的目标不在候选中时（Agent 已删除或改名），组件照样显示它并标明不在当前列表中，用户不改动时原样保留。
- **列表**：卡片与列表项标出 PRIVATE 与目标 Agent，PUBLIC 不加标记。
- **类型与接口**：`MemoryAtom` 增加 `access_policy`，创建与更新的 payload 增加可选的 `access_policy`。页面只呈现 PUBLIC 与 PRIVATE。

### 4.8 实施顺序

1. 后端：HTTP 契约、root API、Patchouli 契约，以及 Familiar 的创建与修改；
2. Agent 修订写回前的重读；
3. 前端样例，owner 复核；
4. 前端实现；
5. 收尾：更新当前文档，归档本计划。

后端两步不依赖样例，可以在样例复核期间进行。

## 5. 受影响的系统、契约与数据

| 位置 | 变化 |
|:---|:---|
| server | 请求与响应模型增加策略；memories 路由透传 |
| workspace root API | `RootMemoryService` 的创建与更新参数 |
| Patchouli 公开契约 | `MemoryCreation`、`MemoryEdit` 增加字段；`patchouli.public.memory.create/update` 的参数说明 |
| Patchouli 局部路由与 Familiar | 创建与编辑接收策略；UPDATE 写回前重读 |
| 生成失败原因码 | 不变：UPDATE 基础记忆缺失或不可读时以 `ResourceNotFoundError` 失败，沿用领域错误的 `code` |
| MemoryLibrary | 不变：`patch_payload` 已支持替换策略 |
| 前端 | 类型、`memoryApi`、两个弹层、卡片与列表、新的可见性组件 |
| 数据 | 不迁移：现有记忆都是 PUBLIC，存储格式不变 |

## 6. 兼容与迁移

- HTTP 只增加可选的请求字段与响应字段，现有调用方不受影响；Agent 页与各 harness 插件不变。
- 现有记忆全部为 PUBLIC，不需要迁移。
- 行为变化：UPDATE 写回时基础记忆已被删除的，任务改为失败，不再把记忆重新写回。

## 7. 测试和验收标准

需要守住的不变量与归属层：

| 不变量 | 归属层 |
|:---|:---|
| 请求中的策略按模型校验；TEAM、PUBLIC 带目标、PRIVATE 缺少目标或指向 `system` 都以 `request.validation_failed` 拒绝；响应带策略 | `tests/unit/server/routers/test_memories.py` |
| root API 在 root 授权之后透传策略 | `tests/unit/workspace/capability/root/test_memory.py` |
| 创建按给定策略落库，缺省为 PUBLIC；只改策略时不推进版本、不写版本记录；内容与策略同时修改时两者都生效；对 Profile 提交策略被拒绝 | `tests/unit/patchouli/services/test_memory_generation.py`、`tests/unit/patchouli/services/test_external_edit_validation.py` |
| 经公开路由改为 PRIVATE 后，目标 Agent 能检索与读取，其他 Agent 不能，已缓存的原子不再交付给其他 Agent；改回 PUBLIC 后恢复 | `tests/integration/patchouli/test_memory_management_edit_route.py`、`tests/integration/workspace/test_intent_registry_and_read_cache.py` |
| UPDATE 写回沿用写回时的策略；基础记忆已删除或不可读时不写回，以原因码失败 | `tests/unit/patchouli/services/test_memory_generation.py` |
| 前端：payload 与响应的映射；候选 Agent 的合并与去重放在纯函数中测试 | `frontend/tests/memory-api.test.mjs` 与候选逻辑对应的 `frontend/tests/*.test.mjs` |

手动验收：

- 样例经 owner 复核定稿；
- 在页面把一条记忆设为只对 `dsh` 可见：dsh 经 MCP 能检索并读取它，Claude Code 检索不到，按别名读取得到 not_found；改回所有 Agent 后两者都能检索到；
- 只改策略后，详情中的更新时间不变。

验证命令：

```bash
pytest tests/unit/server tests/unit/workspace tests/unit/patchouli -q
pytest tests/integration/patchouli tests/integration/workspace -q
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85
ruff check . && black --check .
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
cd frontend && npm run lint && npm test && npm run build
```

## 8. 风险与待决问题

- **PRIVATE 只限定检索范围，不是保密**：被动结算的查重以 `system` 读取，只看得到 PUBLIC。同一件事在之后的对话中被再次提取时，会成为一条新的 PUBLIC 记忆；两条的策略不同，也不会合并。页面的说明文字写明这一点。
- **把 harness 当作 Agent 的遗留**：`claude_code`、`dsh`、`codex`、`pi` 是“harness 即 Agent”看法的残留（[身份与访问体系 Idea](../../ideas/identity-and-access-model.md#8-例外暂缓与已知限制)第 8 节）。plugin 模式支持选择 Agent 时，指向这些标识的 PRIVATE 记忆需要一并迁移。
- **目标 Agent 删除或改名**：指向它的 PRIVATE 记忆对任何 Agent 都不可见，只在管理页可见。Agent 改名见 [Todo](../../todo/agent-alias-rename.md)。
- **不限定 Agent 的 server 来源**：前端可以以任意有 Profile 的具体 Agent 发起 chat，从而检索到指向它的 PRIVATE 记忆。这是用户本人，与管理页能看到全部记忆一致。
- **内容与策略分两步写入**：见第 4.4 节。
- **排队中的 UPDATE 仍会覆盖这期间的手动内容编辑**：既有限制，不在本计划内。
- **待决**：无阻塞问题。页面用语、标记样式与组件位置在样例复核时确定。

## 9. 完成时必须更新的当前文档

- [管理页面](../../frontend/management-views.md)第 1.1 节与第 4 节：可见范围的显示与修改、候选 Agent 的来源、失败语义；
- [MemoryLibrary](../../patchouli/memory-library.md)：策略由管理编辑经 `patch_payload` 修改；
- [Patchouli 生成](../../patchouli/generation.md)：UPDATE 写回前的重读与失败、第 8 节当前限制；
- [能力层](../../workspace/capability.md)：`RootMemoryService` 的创建与编辑字段；
- [路由与事件](../../contracts/routes-and-events.md)：`patchouli.public.memory.create/update` 的参数说明；
- [身份与授权](../../workspace/identity-and-authorization.md)：资源策略由谁设定、何时修改；
- [读取视图与写入意图](../../workspace/read-view-and-write-intents.md)第 6 节：写入意图不设 policy 的说明与指向；
- [身份与访问体系 Idea](../../ideas/identity-and-access-model.md)第 6 节与 [ROADMAP](../../ROADMAP.md) 的对应行；指向本计划的入链改指归档位置；
- 本计划移入 `archive/plans/`，从 [Plans](../../plans/README.md) 中删除，在[归档索引](./README.md)中登记。

## 10. 实施记录

2026-11-11 在分支 `feat/memory-visibility-management` 实施。后端按第 4.2–4.5 节落地；前端在样例复核之后实施，样例见设计画布“HiveMemory 界面样稿”的“记忆图书馆 · 可见范围”两行。与计划不同或计划没有写到的地方：

- **UPDATE 写回失败不新增错误类型**：基础记忆已删除或对发起者不可读时，以已有的 `ResourceNotFoundError` 失败，原因码按“其他领域错误取它的 `code`”的规则为 `workspace.resource.not_found`，失败原因码表不变（第 4.5 节已随之改写）。
- **一个下拉框代替两个单选项**（样例复核时调整）：现有详情的元数据栏只有 256px，两个单选卡片加必选的 Agent 下拉框放不下。改为一个下拉框，第一项“所有 Agent”，其后分组列出 agent_id；选中即确定目标，原计划中“选了只有一个 Agent 却没选 Agent 时不能提交”的校验随之不需要。
- **详情中的位置**（owner，2026-11-11）：样例对比了“并进元数据卡片的一行”与“标签下面单独一张卡片”两种放法，owner 选定后者。
- **文案**（owner，2026-11-11）：下拉框、详情与卡片标记直接显示 agent_id，不加“只有”“仅”之类的前缀；解释句与悬停提示保留完整的说法。
- **“已不在列表中”的判定**：只有访问记录与 Agent 列表都读取成功时才判定，任一来源读取失败时不作判断，避免误报；目标失效时详情只显示警示，不再同时显示“只有这个 Agent 能检索与读取”。
- **浏览器核对中修正**：卡片上较长的标记曾压住别名，改为放进标题行、占住悬停按钮的位置，悬停时让位。
- **新增的常量与文件**：前端增加 `SYSTEM_AGENT_ID`，用于从候选中排除保留的 `system`；新增 `utils/memoryVisibility.ts`（候选合并与取值互换）、`hooks/useVisibilityAgents.ts` 与 `components/memory/MemoryVisibilitySelect.tsx`。

验证：

- 后端：单元与集成测试（不含 `live_llm`、`e2e`、`slow`）3373 个通过，覆盖率 94.88%；`ruff`、`black --check`、`mypy` 通过。新增与扩展的测试位于第 7 节所列的归属层；5 个已有的 UPDATE 写回测试补上了库中的基础记忆，测试用的内存存储替身补上了策略的替换。去掉写回前的重读时，三个 UPDATE 写回测试都会失败。
- 前端：`npm run lint`、`npm test`（80 个）、`npm run build` 通过；新增 `frontend/tests/memory-visibility.test.mjs`。
- 浏览器：常驻后端运行旧代码且使用桌面客户端的管理凭据，核对改用页内桩数据回答记忆、Agent 与访问记录三个读接口，写请求一律拦截。核对了卡片标记与警示色、详情卡片的查看与编辑、候选分组与排除、改动后“保存”可用，以及创建请求带上 `access_policy`。

尚未完成的手动验收：以本分支重启后端，在页面把一条记忆设为只有 dsh 可见，确认 dsh 经 MCP 能检索并读取、Claude Code 检索不到且按别名读取得到 not_found，改回所有 Agent 后两者都能检索到；只改策略后详情中的更新时间不变。由 owner 进行。

发现但没有处理：排队中的 UPDATE 写回仍会覆盖这期间的手动内容编辑（第 8 节、[Patchouli 生成](../../patchouli/generation.md#8-当前限制)第 8 节），未登记 Todo，由 owner 决定是否跟踪。
