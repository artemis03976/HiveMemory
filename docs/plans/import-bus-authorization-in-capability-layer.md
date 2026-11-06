---
title: Import Bus 的授权移入能力层
status: planned
owner: system
target: 版本之外的工作（身份与访问主线，统一身份授权入口的后续）
scope: import-bus-authenticates-only-and-root-api-authorizes-each-step
updates:
  - AGENTS.md
  - docs/system/import-bus.md
  - docs/system/application-services.md
  - docs/system/composition.md
  - docs/workspace/capability.md
  - docs/workspace/identity-and-authorization.md
  - docs/architecture/overview.md
code_paths:
  - src/hivememory/system/application/import_bus_service.py
  - src/hivememory/system/services/import_bus/ingestor.py
  - src/hivememory/system/services/import_bus/events.py
  - src/hivememory/system/assembler.py
  - src/hivememory/workspace/capability/root/interaction.py
  - src/hivememory/workspace/capability/root/topic.py
  - tests/unit/architecture/test_access_boundaries.py
related_docs:
  - docs/archive/plans/unified-identity-authorization-entry.md
  - docs/ideas/identity-and-access-model.md
  - docs/ideas/external-session-and-topic-projection.md
  - docs/system/import-bus.md
  - docs/workspace/capability.md
---

# Import Bus 的授权移入能力层

> **状态**：Planned（2026-11-06 起草，待 owner 评审）。owner 在 2026-11-06 作出的决定集中在第 4.1 节，其余是候选设计，评审时可以调整。本计划以[统一身份授权入口](../archive/plans/unified-identity-authorization-entry.md)（已归档）实施后的代码为依据，该实施所在的分支 `refactor/unified-identity-authorization-entry` 尚未合并。
>
> 本计划由“交互提交收束与话题按需创建”草案拆出：身份访问的部分留在这里；话题按需创建、prepare/finalize 拆分与交互提交收束退回 Idea，见[外部会话与 Topic 投影 Idea](../ideas/external-session-and-topic-projection.md#10-交互提交收束与话题按需创建)第 10 节。

## 1. 背景和问题

### 1.1 System 的服务在做操作授权

[`ImportBusService`](../../src/hivememory/system/application/import_bus_service.py) 导入一轮或为在途轮次确定话题时，先以系统自身的 principal 认证参与者驻留 Workspace 的 `system`，再调用 `authorize_root` 取得 scope，把 scope 与系统 root 工作 actor 一起交给编排器。

AGENTS.md 的所有权表写明 System 不负责“Workspace 准入与行为授权”；owner（2026-11-06）指出从未允许在 service 上做操作授权。这次授权从 #137（2026-10-27）起就在服务里，当时叫 `authorize_import`；统一身份授权入口的计划把 Import Bus 服务列为 `authorize_root` 的调用位置，实施时照搬，架构测试的白名单也照此书写。

表现是导入一轮做两次 root 授权：服务入口一次，编排器经 root API 的 [`RootInteractionService.submit_interaction`](../../src/hivememory/workspace/capability/root/interaction.py) 提交时一次。两次授权本身不是缺陷，能力层的方法本来就在入口逐次授权；缺陷是第一次发生在 System。

### 1.2 为什么不能只删掉

编排器（[`ImportBusIngestor`](../../src/hivememory/system/services/import_bus/ingestor.py)）除了提交，还有四步直接以 scope 调用公开路由，scope 只能由授权点组装：

| 步骤 | 用途 | 现在的调用 |
|:---|:---|:---|
| 查询提交进度 | 查重；续接上一轮所在的话题；写入时确定话题前检查这一轮是否已应用 | `PatchouliRoutes.INTERACTION_STATUS` |
| PASSIVE_MEMORY 路由 | 取得话题路由与新话题的标题、摘要 | `GatewayPublicRoutes.PROCESS` |
| 写入时建立话题 | 为在途轮次确认或新建话题（不检索） | `PatchouliRoutes.PREPARE_AGENT_RUN` |
| 等待应用 | 应用完成后才确认这一轮 | `PatchouliRoutes.INTERACTION_WAIT` |
| 提交 | 已经经 root API | `RootInteractionService.submit_interaction` |

导入事件也取 scope 的归属作为 Workspace 标识。

## 2. 目标与非目标

### 2.1 目标

1. System 的代码不调用任何授权方法：`ImportBusService` 只认证系统自身，不再持有操作授权者。
2. 编排器不再持有 scope：第 1.2 节的每一步都经 root API，由 root API 在方法入口 `authorize_root`。
3. Import Bus 与 MCP 的对外行为、HTTP 契约、确认语义、外部会话串行与错误码不变。

### 2.2 非目标

- 话题按需创建、prepare/finalize 拆分、controller 的提交改走 root API、交互提交收束：退回 Idea（[外部会话与 Topic 投影 Idea](../ideas/external-session-and-topic-projection.md#10-交互提交收束与话题按需创建)第 10 节）。
- 任务进程的授权方式：任务进程的系统阶段仍在进程内授权。
- `ImportBusService.place_turn`、编排器的 `place` 与 `_placements` 等标识符的改名。
- Import Bus 的进程内控制状态（外部会话串行门、已确定话题的登记、关闭时的等待）的归属：仍属于 System。

## 3. 当前状态和缺口

- `ImportBusService._authorize_system` 认证 `system` 后调用 `authorize_root`；目标不是参与者驻留的 Workspace 时，在任何编排之前以 `target_workspace_not_resident` 拒绝。
- 编排器的 `ingest` 同时接收 `system_actor` 与 `identity_scope`，`place` 只接收 `identity_scope`；编排器持有全局总线，四步直接请求路由，提交经 root API 以 `system_actor` 再授权一次。
- [`test_access_boundaries.py`](../../tests/unit/architecture/test_access_boundaries.py) 的 `authorize_root` 调用位置白名单包含 `system/application/import_bus_service.py`。

## 4. 目标设计

### 4.1 owner 决定（2026-11-06）

1. **System 的服务不做操作授权**：`ImportBusService` 的 `authorize_root` 不应存在；授权只在能力层方法的入口与任务进程中进行。
2. **Import Bus 需要 scope 的步骤临时走 root 方法**。
3. **查询提交进度与 PASSIVE_MEMORY 路由这两个 root 方法是临时的**，在之后的交互提交重构中移除。

### 4.2 root API 新增的方法

| 方法 | 包装的路由 | 说明 |
|:---|:---|:---|
| `RootInteractionService.interaction_status` | `INTERACTION_STATUS` | 临时（第 4.1 节第 3 项） |
| `RootInteractionService.wait_interaction` | `INTERACTION_WAIT` | 时限由调用方给出，与现在相同 |
| `RootTopicService.route_passive_turn` | Gateway `PROCESS`（PASSIVE_MEMORY） | 临时（第 4.1 节第 3 项）；请求时限与会话轮次由调用方给出；能力层第一次调用 Gateway |
| `RootTopicService.prepare_topic` | `PREPARE_AGENT_RUN`（不给检索 scope） | 为在途轮次确认或新建话题，返回 Patchouli 的结果，与现在相同 |

每个方法都以 `actor`（系统 root 工作 actor）与 `target_workspace` 为参数，在入口 `authorize_root` 后调用路由，与现有 root 方法相同。四个方法都没有 HTTP 使用方，生产使用方是 Import Bus。另外两个方法在交互提交重构中的去留由那次重构决定（Idea 第 10 节）。

### 4.3 Import Bus

- `ImportBusService` 只做认证：以系统自身的 principal 认证参与者驻留 Workspace 的 `system`，把系统 root 工作 actor 与目标 Workspace 交给编排器；构造参数去掉操作授权者。
- 编排器的 `ingest` 与 `place` 接收系统 root 工作 actor、目标 Workspace、参与者与 principal 标识，不再接收 scope；它持有 `RootInteractionService` 与 `RootTopicService`，不再持有全局总线。
- 导入事件改为接收目标 Workspace 的标识。
- **目标校验**：目标不是参与者驻留的 Workspace 时，第一次 root 调用（查询进度）以 `target_workspace_not_resident` 拒绝，仍然先于路由与提交。
- **已确定话题的登记命中**：`place` 在串行门内先查进程内的登记，命中时直接返回，不经过任何 root 调用。登记项只由此前经过 root 授权的调用写入，键由同一 principal 与参与者推导；访问登记在进程内不可变，跳过的只是一次重复的准入核对。写入本身仍由 actor API 按请求的工作 actor 授权。
- 其余规则（外部会话串行门、查重与续接、确认语义、关闭时等待在途请求）不变。

### 4.4 架构守护

- `authorize_root` 的调用位置白名单去掉 `system/application/import_bus_service.py`。
- 新增守护：`system/` 包内的源文件不调用 `authorize_root` 与 `authorize_agent`，也不导入 `WorkspaceOperationAuthorizer`。

## 5. 受影响的系统、契约与数据

| 范围 | 变化 |
|:---|:---|
| root API | `RootInteractionService` 新增两个方法，`RootTopicService` 新增两个方法 |
| System | `ImportBusService` 只认证；编排器不持有 scope 与全局总线；导入事件只取 Workspace 标识；组合根的装配随之调整 |
| 架构守护 | 第 4.4 节 |
| HTTP、MCP、Patchouli、Gateway | 不变 |

没有持久化数据与配置变化。

## 6. 兼容、迁移与实施分批

- 改动都在进程内，没有数据迁移；HTTP 与 MCP 契约不变。

实施分批提交，每批保持全量门槛通过：

1. **root API**：新增第 4.2 节的四个方法及其授权测试。
2. **Import Bus 切换**：服务只认证，编排器改经 root API，事件改取 Workspace 标识，组合根装配调整；测试切换。
3. **架构守护与文档收尾**：第 4.4 节；按第 9 节更新当前文档，计划归档。

## 7. 测试和验收标准

测试写入归属层的现有文件，不建以计划命名的测试集合。

| 不变量 | 归属层 |
|:---|:---|
| 四个新方法只接受 `system`；目标不是驻留 Workspace 时拒绝 | `tests/integration/workspace/test_root_api_authorization.py` |
| 编排器经 root API 完成查重、路由、建立话题、提交与等待；查重与续接规则不变 | `tests/unit/system/services/import_bus/test_ingestor.py` |
| 目标不符时先于路由拒绝；导入与写入时确定话题的对外结果不变 | `tests/integration/system/test_import_bus_submission.py`、`tests/integration/system/test_mcp_memory_access.py` |
| `system/` 内没有授权调用，也不导入操作授权者 | `tests/unit/architecture/test_access_boundaries.py` |

验证命令：

```bash
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85
ruff check . && black --check .
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
git diff --check
```

需要真实 Qdrant 与模型的导入 e2e（`tests/e2e/system/test_import_turns.py`）在环境可用时运行。手动验收在重启后的 live 后端上进行：一次 dsh 或 Claude Code 的导入，一次带写入的轮次（写入时确定话题、回流沿用）。

## 8. 风险与待决问题

- **每轮 root 授权的次数**：一轮导入经 root API 授权三到五次（查进度、上一轮的进度、路由、提交、等待）。每次都是进程内的登记查找，成本可以忽略；这是能力层逐次授权的设计，不是重复。
- **能力层第一次调用 Gateway**：`route_passive_turn` 让 root API 包装 Gateway 的路由；按第 4.1 节第 3 项，它在交互提交重构中移除。
- **单一使用方的方法**：四个新方法都只服务 Import Bus；临时的两项由交互提交重构移除，其余两项的去留由那次重构决定。
- **依赖的分支**：本计划依据的统一身份授权入口尚在未合并的分支上；在同一分支继续实施，还是等它合并后另开分支，由 owner 决定。

## 9. 完成时必须更新的当前文档

- **AGENTS.md** §3、§4：系统级操作中 Import Bus 不再自行授权；Import Bus 链路改为“认证 `system` → 需要授权的步骤经 root API”。
- **system**：[Import Bus](../system/import-bus.md)（认证与授权、编排步骤）；[应用服务](../system/application-services.md)；[组合根](../system/composition.md)。
- **workspace**：[能力层](../workspace/capability.md)（root API 新方法）；[身份与授权](../workspace/identity-and-authorization.md)（`authorize_root` 的调用位置）。
- **architecture**：[系统架构概览](../architecture/overview.md)中的 Import Bus 链路。
- **索引**：[Plans 导航](./README.md)、[归档计划索引](../archive/plans/README.md)。
