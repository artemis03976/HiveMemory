---
title: 访问登记的运行时管理
status: archived
archived_at: 2026-11-09
implemented_by: PR #209
superseded_by: docs/workspace/authentication.md
owner: project
target: 版本之外的工作（身份与访问主线）
scope: runtime-editing-of-principal-registration-and-workspace-access-records-with-credentials-and-agent-binding
updates:
  - docs/workspace/authentication.md
  - docs/workspace/identity-and-authorization.md
  - docs/workspace/capability.md
  - docs/system/configuration.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
  - docs/frontend/management-views.md
  - docs/help/configuration.md
  - docs/help/claude-code-plugin.md
  - docs/help/deepseek-harness-plugin.md
  - docs/help/codex-plugin.md
  - docs/help/pi-plugin.md
  - docs/help/troubleshooting.md
code_paths:
  - src/hivememory/config/access.py
  - src/hivememory/system/access/
  - src/hivememory/system/assembler.py
  - src/hivememory/workspace/registry.py
  - src/hivememory/workspace/authorization.py
  - src/hivememory/workspace/capability/root/
  - src/hivememory/server/routers/
  - src/hivememory/server/error_handlers.py
  - configs/system_principals.yaml
  - configs/workspace_actors.yaml
  - frontend/src/components/settings/
  - .gitignore
  - .dockerignore
related_docs:
  - docs/ideas/identity-and-access-model.md
  - docs/workspace/authentication.md
  - docs/governance/baselines/identity-access-current-state-2026-11-05.md
---

# 访问登记的运行时管理

> **状态**：Archived（2026-11-09，已实施，实施记录见第 10 节）。当前事实见[认证与访问登记](../../workspace/authentication.md#2-两类访问登记)第 2 节与[管理页面](../../frontend/management-views.md#33-接入与权限)。以下为实施时的计划原文，链接已按归档位置改正。方向来自[身份与访问体系 Idea](../../ideas/identity-and-access-model.md#3-principal-登记的来源与生效时点e-1)第 3 节：接入登记维持启动时装载，运行时登记另立计划（owner，2026-09-27）；操作白名单的运行时编辑并入同一计划（2026-10-10）。本计划的三项前提由 owner 于 2026-11-09 决定：改动写回生效的登记文件；接入登记与 Workspace 访问记录分开维护（E-1a）；接入凭据由 HiveMemory 生成。范围对应候选清单的 A-03、A-04 与 A-07；主配置的热更新（D-09）与记忆可见性（A-10）不在本计划内。不绑定版本。

## 1. 背景和问题

两类访问登记都只在启动时装载，运行中不可变（[认证与访问登记](../../workspace/authentication.md#2-两类访问登记)第 2 节）：

- **接入登记**（`configs/system_principals.yaml`，System 所有）：哪些调用来源可以接入、经哪些 adapter、以什么凭据证明。
- **Workspace 访问记录**（`configs/workspace_actors.yaml`，workspace 所有）：哪个用户与 Agent 能进入 Workspace，进入后能执行哪些 actor 操作。

由此带来三个问题：

- **改登记要改文件并重启。** 新增或撤销一个 harness、给某个 Agent 收回写入权限，都要编辑两份 YAML 并重启后端。重启会丢掉会话、进程表、写入意图与已接纳的导入，这些状态都只在内存中。桌面客户端打包之后，用户也不该去改安装目录里的 YAML。
- **接入凭据没有生命周期**（A-07）。登记里只写环境变量名，凭据由用户自己生成，并写进两侧的环境变量；后端在启动时读入。没有签发、轮换与撤销，换凭据要改 `.env` 并重启。
- **principal 不限定可声明的 Agent**（A-07）。持有某个 harness 凭据的调用方，可以声明该用户下的任意 Agent，并取得那个 Agent 的白名单。harness 以 `HIVEMEMORY_AGENT_ID` 选择 Agent，后端不核对。

## 2. 目标与非目标

### 2.1 目标

1. 外部接入登记可以在运行时登记、修改、启停与删除，立即生效，并写回生效的登记文件。
2. Workspace 访问记录的准入与操作白名单可以在运行时编辑，立即生效，并写回生效的登记文件。
3. 外部接入的凭据由 HiveMemory 生成、保存与轮换；撤销即禁用或删除这条登记。环境变量照常可用，并优先于生成的凭据。
4. 接入登记可以限定该来源能声明哪些 Agent；随仓库发布的四个 harness 登记各自只能声明自己的 Agent。
5. 前端设置页提供管理入口。

### 2.2 非目标

- **主配置热更新**（D-09）：`config.yaml` 的生效方式不变。
- **记忆可见性**（A-10）：写入意图的 policy、记忆 CRUD 的 `access_policy` 与策略组件，都不在本计划内。
- **进程内来源的运行时编辑**：`hivememory:http-server`、`hivememory:system` 与 `hivememory:profile-injector` 只能经文件修改并重启（第 4.2 节说明原因）。
- **凭据过期、多用户登录与外部身份系统**。
- **`allowed_user_ids` 的编辑**：现在只有一个用户。
- **打断进行中的进程**：禁用一个来源不会终止它已经在跑的进程（第 4.5 节）。
- **Agent 管理页的 Active 开关**：它可以接到访问记录的 `enabled` 上，但这属于前端空入口的接线，不在本计划内。

## 3. 当前状态和缺口

| 环节 | 现在 | 本计划之后 |
|:---|:---|:---|
| 登记装载 | `config/access.py` 在启动时读两份文件，组合根构造两个不可变注册表 | 启动时照旧装载；注册表可在运行时整体替换内容 |
| 授权 | `WorkspaceOperationAuthorizer` 每次授权都即时查询访问记录，不缓存结论 | 不变：编辑在下一次授权时生效 |
| 认证 | `SystemPrincipalAuthenticator` 每次认证查询接入登记；凭据在装配时从环境变量读入 | 凭据来自环境变量或凭据文件；新增 Agent 限定检查 |
| 管理入口 | 无 | 接入登记经 System 的管理接口；访问记录经 root API |
| 凭据 | 只有环境变量名，值由用户生成 | HiveMemory 生成并保存在不受版本控制的凭据文件；环境变量优先 |
| Agent 限定 | 无 | 接入登记的 `allowed_agent_ids` |
| 前端 | 无 | 设置页的“接入与权限”分类 |

已有可以直接复用的部分：

- 装载时的校验都在注册表的构造函数中：重复键、跨 owner 记录、禁用却配置白名单、为 `system` 配置 operation、未知 operation、进程内 adapter 与凭据不能共存。运行时编辑走同一套校验。
- 授权本来就不缓存，所以访问记录一经替换，MCP 与导入的下一个请求、chat 的下一个操作都按新内容授权。
- `ModelRegistry` 与 `ProviderRegistry` 已经以“同目录临时文件加原子替换”写回 YAML，写回方式照此。

## 4. 目标设计

### 4.1 写回与生效

- **写回位置**：写回当前生效的登记文件，即 `HIVEMEMORY_PRINCIPALS_PATH` / `HIVEMEMORY_WORKSPACE_ACTORS_PATH` 指向的文件，未设置时是仓库 `configs/` 下的默认文件（owner，2026-11-09）。不想让运行时编辑改动仓库文件时，用这两个环境变量指向仓库外；桌面客户端指向用户数据目录。
- **顺序**：一次编辑先在内存中构造新的完整记录集，并走与装载相同的校验；校验通过后以原子替换写入文件；写入成功后才替换注册表的内容。校验或写入失败时，内存与文件都保持原样。校验、写入与替换之间没有 `await`，同一事件循环中的两次编辑不会交错。
- **外部修改**：注册表记住上次装载或写入时文件的内容摘要。写入之前文件已被外部改动时，拒绝这次编辑并说明原因，不静默覆盖手工修改；需要重启后端装载手工修改后，再在运行时编辑。
- **注释**：写回以 `yaml.safe_dump` 序列化，第一次运行时编辑会丢掉文件中的注释，与 `models.yaml` 现状相同。随仓库发布的两份登记文件中的说明在完成时移到文档，文件只保留一段说明文件用途与运行时编辑的注释（第 9 节）。

### 4.2 接入登记的管理（System）

- **所有者与入口**：接入登记属于 System。管理接口与提供商、模型接口相同：不经认证网关，由 HTTP 管理面的管理凭据保护；System 服务中不做授权（E-1a：与访问记录分开维护，owner，2026-11-09）。读写方法放在 `SystemActorAccessRegistry` 自身，与 `ProviderRegistry` 一样负责写回自己的文件，不新增服务层。
- **能编辑的范围**：只有外部来源，即 adapter 全部是外部 adapter（`mcp`、`import`）的登记：
  - 登记：`principal_id`、`adapters`（`mcp`、`import` 的非空子集）、`allowed_agent_ids`，`kind` 固定为 `harness`，`allow_system_actor` 固定为 `false`；登记时生成凭据；
  - 修改：启停、`adapters` 与 `allowed_agent_ids`；
  - 轮换凭据；
  - 删除：同时删除它的生成凭据。
- **进程内来源只读**：列出 `local` 或 `http` adapter 的登记不能经接口创建、修改或删除。它们关系到前端、chat 与导入本身能否工作：禁用 `hivememory:http-server` 或 `hivememory:system`，管理接口与 chat 会立即失效，而恢复它们恰恰需要管理接口。外部来源也不能经接口取得 `allow_system_actor`。
- **列表**：返回全部登记。进程内来源标为只读；凭据只返回来源（环境变量、生成、未配置）与打码值，不返回明文。

### 4.3 接入凭据

- **保存**：生成的凭据保存在生效的接入登记文件同目录的 `principals.secrets.yaml`，按 `principal_id` 存放明文，与 `providers.secrets.yaml`、`management.token` 一样不受版本控制（加入 `.gitignore` 与 `.dockerignore`），以原子替换写入；POSIX 上文件只有当前用户可读写。接入登记文件本身仍不含凭据。
- **生成与返回**：登记与轮换时以 `secrets.token_urlsafe(32)` 生成，明文只在这两个接口的响应中返回一次，供用户复制到 harness 一侧；列表与其他响应只给打码值。用户错过时可以再次轮换。
- **来源优先级**：登记了 `credential_env` 且该环境变量有值时，环境变量优先，规则与 Provider 凭证相同；此时界面显示为只读，不能轮换。环境变量仍在启动时读入，修改后重启生效。
- **装载规则调整**：外部登记的凭据来源是 `credential_env` 指向的环境变量，或凭据文件中的生成值，二者至少有一个可用；都没有时照常启动并告警，该来源的每次认证以 `invalid_credential` 拒绝，与现在环境变量为空时一致。“列出外部 adapter 的登记必须声明 `credential_env`”这条装载规则随之放宽为“可以不声明”。进程内 adapter 不能带凭据的规则不变。
- **撤销**：禁用或删除登记，下一次认证即被拒绝。

### 4.4 principal 的 Agent 限定

- 接入登记新增可选的 `allowed_agent_ids`：`None` 表示不限定，与 `allowed_user_ids` 对称。第 1 阶段在用户检查之后核对，声明的 Agent 不在集合中时以新的原因码 `agent_not_allowed_for_principal` 拒绝（403）。
- 这与“principal 标识执行环境、同一执行环境可以以不同 Agent 运行”的前提不冲突（Idea 第 8 节）：限定的是一个集合，不是一对一绑定。
- 随仓库发布的四个 harness 登记分别设为 `[claude_code]`、`[dsh]`、`[codex]`、`[pi]`；进程内来源不设。

### 4.5 Workspace 访问记录的管理（root API）

- **所有者与入口**：访问记录属于 workspace，经 root API 管理：新的 root 服务以请求的工作 actor 执行 `authorize_root`，目标是它驻留的 Workspace。注册表负责写回自己的文件，写回方式同第 4.1 节。
- **能编辑的范围**：目标 Workspace 坐标下的记录：
  - 列出；
  - 新增或修改一条记录，内容为 `agent_id`（省略表示用户级记录）、`enabled` 与 `allowed_operations`；
  - 删除一条记录。
- **`system` 记录不能编辑**：root 授权以这条记录准入 `system`，禁用或删除它会让管理接口立即失效，因此接口拒绝修改或删除它。它也不出现在可编辑列表中。
- **用户级记录可以编辑与删除**：删除后，没有精确记录的 chat Agent 不能再进入 Workspace；界面在删除前提示这一点。
- **生效时点**：
  - MCP 与导入每次请求都重新认证与授权，编辑在下一个请求生效；
  - chat 的每个操作都即时授权，撤销在下一个操作生效。但线程描述中的操作快照在派生线程时生成：撤销的操作仍会出现在本轮的提示词与动词集合中，调用时被拒绝；新授予的操作从下一轮开始出现。这是 Idea 第 3 节要求处理的差异，本计划把它定为上述语义，写进当前文档，不在本轮内同步快照；
  - 接入登记只在认证时检查：禁用一个来源只拒绝它之后的认证。外部来源只经 MCP 与导入接入：MCP 每次工具调用认证一次；导入在请求时认证，已经接纳的一轮照常完成。

### 4.6 前端

设置页新增“接入与权限”分类，读写第 4.2、4.5 节的接口：

- **外部接入**：列出接入登记，显示启用状态、可声明的 Agent 与凭据来源。可以登记新的 harness，登记后弹出凭据供复制，并提示写进 harness 一侧的哪个环境变量；也可以启停、轮换凭据、修改可声明的 Agent，以及删除。进程内来源只读显示。
- **Agent 操作权限**：列出访问记录，每条以六项 actor 操作的勾选与准入开关编辑；可以为某个 Agent 新增精确记录，Agent 可从已有 Profile 中选择，也可以直接填写；用户级记录标为“其余所有 Agent”。
- 登记一个新 harness 通常需要两步：先登记来源，再为它的 Agent 新增访问记录。界面在一个流程中依次调用两个接口，第二步失败时提示补做；只有来源、没有访问记录时，认证以 `actor_not_admitted` 拒绝，不会放宽任何权限。

### 4.7 路由与错误

- **建议路由**，名称在评审时确定：
  - 接入登记：`/api/v1/access/principals` 的列表与登记，`/api/v1/access/principals/{principal_id}` 的修改与删除，`/api/v1/access/principals/{principal_id}/credential` 的轮换；
  - 访问记录：`/api/v1/access/actors` 的列表、新增或修改与删除。

  两组都在 `/api/` 下，由管理凭据中间件自动保护。
- **新的拒绝类别**：
  - 修改进程内来源，或修改、删除 `system` 记录；
  - 登记键重复；
  - 登记文件在外部被改动；
  - 校验失败，沿用现有的 400 校验信封。

  错误码与状态码在实施时登记进错误模型，原因码 `agent_not_allowed_for_principal` 加入准入拒绝的白名单。

## 5. 受影响的系统、契约与数据

- **config**：`config/access.py` 的接入登记模型新增 `allowed_agent_ids`，`credential_env` 对外部登记改为可选；新增两份登记文件的写回函数与凭据文件的读写。
- **System**：
  - `system/access/registry.py` 增加读写方法、写回与凭据文件；
  - `system/access/principal.py` 增加 Agent 限定检查；
  - 组合根把注册表交给新的管理接口。
- **workspace**：
  - `workspace/registry.py` 增加整体替换与写回；
  - 新的 root 服务管理访问记录；
  - 授权者不变。
- **server**：两组新路由；错误映射登记新的错误码与原因码。
- **前端**：设置页新分类与对应的 API 客户端。
- **配置文件**：
  - 随仓库发布的 `system_principals.yaml` 为四个 harness 加上 `allowed_agent_ids`；
  - 两份文件的说明性注释移到文档，只留一段说明用途与运行时编辑的注释；
  - `.gitignore`、`.dockerignore` 加上 `principals.secrets.yaml`。
- **不变**：认证网关的两阶段结构、工作 actor、授权者的两个方法、MCP 与导入入口、harness 插件代码（插件仍从环境变量读凭据）。

## 6. 兼容与迁移

- **已在 `.env` 中配置凭据的部署照常工作**：环境变量优先于生成的凭据。
- **新部署**：在设置页登记或为已有登记生成凭据，再写进 harness 一侧的环境变量。
- **Agent 限定是收紧**：通过 `HIVEMEMORY_AGENT_ID` 让 harness 声明其他 Agent 的部署，升级后会以 `agent_not_allowed_for_principal` 被拒绝，需要在设置页或文件中把那个 Agent 加进该来源的 `allowed_agent_ids`。自定义过 `system_principals.yaml` 的部署不受影响：没有这个字段即不限定。
- **live checkout 中的写回**：在仓库内运行时，第一次运行时编辑会改动受版本控制的登记文件，并丢掉其中的注释。不想要这个 diff 时，在 `.env` 中用两个路径变量指向仓库外的副本。
- **数据格式**：只新增字段与一个凭据文件，没有数据迁移。

## 7. 测试和验收标准

需要守住的不变量与归属层：

| 不变量 | 归属层 |
|:---|:---|
| 运行时编辑与装载走同一套校验；校验或写入失败时，内存与文件都不变；写回的文件重新装载后与内存一致；文件在外部被改动时拒绝写回 | `tests/unit/system/access/test_registry.py`、`tests/unit/workspace/test_registry.py` |
| 进程内来源不能经接口创建、修改或删除；外部来源不能取得 `allow_system_actor` | `tests/unit/system/access/test_registry.py` |
| 生成的凭据可以通过认证，轮换后旧凭据被拒；环境变量有值时优先于生成值；列表与错误响应不含明文 | `tests/unit/system/access/` |
| `allowed_agent_ids` 不含声明的 Agent 时第 1 阶段以 `agent_not_allowed_for_principal` 拒绝；`None` 不限定 | `tests/unit/system/access/test_gateway.py` |
| root 服务入口执行 root 授权；只能编辑目标 Workspace 坐标下的记录；`system` 记录不能修改或删除 | `tests/unit/workspace/capability/` |
| 访问记录编辑后，下一次 `authorize_agent` 按新内容判定 | `tests/unit/workspace/` 中授权者的现有测试 |
| 经接口登记一个来源并为它的 Agent 新增记录后，持新凭据的 MCP 调用成功；轮换或禁用后被拒 | `tests/integration/system/test_mcp_memory_access.py` |
| 新路由在 `/api/` 下受管理凭据保护 | 由 `tests/unit/server/` 现有的路由表枚举测试覆盖，不另写 |

- 前端以手动验收为主：登记、复制凭据、启停、轮换、编辑白名单、删除，以及进程内来源与 `system` 记录不可编辑。
- 端到端手动验收：在设置页轮换 dsh 的凭据并更新 harness 一侧，dsh 的检索与回流恢复，旧凭据被拒；收回某个 harness 的 `memory.write`，它的下一次写入被拒，检索照常。

验证命令：

```bash
pytest tests/unit/system/access tests/unit/workspace tests/unit/server -q
pytest tests/integration/system -q
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85
ruff check . && black --check .
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
cd frontend && npm run lint && npm test && npm run build
```

## 8. 风险与待决问题

- **把自己锁在外面**：第 4.2、4.5 节把进程内来源与 `system` 记录排除在接口之外。仍然可能的情况是删掉用户级记录后 chat 不能进入 Workspace，这可以在设置页恢复，界面在删除前提示。
- **System 管理接口不做授权**：接入登记的读写只由管理凭据保护，与提供商、模型接口相同。管理凭据只防误用，能读到 `configs/management.token` 的本机程序可以登记新的来源。这与现有管理面的边界一致，防刻意随桌面客户端解决。
- **凭据明文落盘**：与 `providers.secrets.yaml` 相同，只防误用。
- **线程快照的滞后**：撤销的操作在本轮的提示词中仍然可见，调用时被拒。语义写进当前文档，不在本轮内同步快照。
- **写回丢注释、改动受控文件**：第 4.1 节与第 6 节已说明；说明性内容移到文档后，不再依赖文件注释。
- **待决**：无阻塞问题。路由名称、错误码名称与前端分类名称可在评审时调整。

## 9. 完成时必须更新的当前文档

- [认证与访问登记](../../workspace/authentication.md)：
  - 第 2 节：登记可在运行时编辑、写回与外部修改检测、`allowed_agent_ids`、凭据来源与凭据文件、随仓库发布的默认登记表；
  - 第 1 节：凭据规则；
  - 第 8 节：删除“没有管理 API 或热更新”“凭据没有签发、轮换与撤销”“白名单只能改文件”三条限制，写明快照滞后与进行中进程不被打断。
- [身份与授权](../../workspace/identity-and-authorization.md)：第 8 节相应的限制。
- [能力层](../../workspace/capability.md)：root API 的新服务。
- [System 配置](../../system/configuration.md)第 1.1 节：登记文件的写回与凭据文件。
- [公开路由与事件](../../contracts/routes-and-events.md)：两组新路由。
- [错误模型](../../contracts/error-model.md)：新的错误码与原因码。
- [管理页面](../../frontend/management-views.md)：设置页的新分类。
- 帮助：
  - [配置指南](../../help/configuration.md)第 5.1 节：设置页的管理入口与写回位置；
  - 四份插件文档第 2 节：在设置页生成凭据，环境变量作为另一种方式；
  - [排障](../../help/troubleshooting.md)：`agent_not_allowed_for_principal`。
- 随仓库发布的两份登记文件：说明性注释移到上述文档，文件只留一段说明用途与运行时编辑的注释。
- 收尾时：
  - 本计划归档；
  - Idea 第 2、3、8 节同步：E-1 与 E-1a 已实施，A-07 的两条已知限制移出；
  - [ROADMAP](../../ROADMAP.md) 第 5.5 节“配置热更新”一行改为只剩主配置；
  - [Plans 导航](../../plans/README.md)同步。

## 10. 实施记录

2026-11-09 在分支 `feat/runtime-access-registration` 实施，第 4 节的设计按原样落地，路由采用第 4.7 节的建议路径。与计划不同或计划没有写到的地方：

- **准入被撤销改为授权拒绝**：访问记录可以在运行时禁用或删除之后，授权时准入失效不再是接线缺陷。`WorkspaceOperationAuthorizer` 改以 `OperationDeniedError`（`actor_not_admitted`）拒绝，原为 `ScopeRequiredError`：后者在 HTTP 映射为 500，在 MTP 呈现为系统故障，被收回权限的 Agent 会看到“系统错误”而不是“没有权限”。预检索仍只把 `operation_not_allowed` 译为不检索，准入失效时进程照旧失败。
- **装载规则**：放宽“外部登记必须声明 `credential_env`”的同时，新增“一条登记不能混用进程内与外部 adapter”。凭据核对改为按是否外部来源判断，不再按是否声明了 `credential_env`：否则没有声明它的外部来源会跳过凭据核对。
- **个人配置不进入仓库**（owner，2026-11-09，实施期间决定）：第 4.1 节“写回受版本控制的文件”与第 6 节“live checkout 中的写回”不再成立。`configs/system_principals.yaml` 与 `configs/workspace_actors.yaml` 移出版本控制、加入 `.gitignore` 与 `.dockerignore`，仓库改为发布同目录的 `*.example.yaml` 模板（保留完整注释，含四个 harness 的 Agent 限定）；默认路径的生效文件缺失时，装载前从模板原样复制一份，路径变量指定的文件仍须已经存在。随仓库发布的登记链路测试与插件测试改为读取模板，不读取本机的生效登记。
- **访问注册表的依赖**：Workspace 访问注册表负责读写自己的登记文件，因此导入 `config.access`；workspace 包内依赖边界的架构测试为它登记了这一例外，与会话服务读取自己的配置段相同。
- **顺带修正**：前端管理页面文档中 Agent 列表读取失败的提示，按上一次改动后的行为更新。

验证：后端单元与集成测试（不含 `live_llm`、`e2e`、`slow`）全部通过；`ruff`、`black --check`、`mypy` 通过；前端 `npm run lint`、`npm test`、`npm run build` 通过。

尚未完成的手动验收：在浏览器中登记、复制凭据、启停、轮换、编辑白名单与删除，以及 dsh 轮换凭据后恢复、收回写入后被拒。这些需要以本分支重启后端，由 owner 进行。
