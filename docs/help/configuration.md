---
title: HiveMemory Configuration Guide
status: current
owner: system
scope: operator-configuration-providers-and-models
code_paths:
  - configs/config.yaml
  - configs/models.example.yaml
  - configs/.env.example
  - configs/providers.secrets.example.yaml
  - configs/system_principals.example.yaml
  - configs/workspace_actors.example.yaml
  - src/hivememory/config/access.py
  - src/hivememory/config/
  - src/hivememory/system/provider_registry.py
  - src/hivememory/system/model_registry.py
  - src/hivememory/system/running_config.py
  - src/hivememory/server/routers/config.py
  - src/hivememory/server/management_credential.py
related_contracts:
  - docs/architecture/boundaries.md
related_docs:
  - docs/system/import-bus.md
last_reviewed: 2026-11-12
---

# HiveMemory 配置指南

HiveMemory 把配置分为三类：主配置描述系统如何装配，Model Registry 描述可选择哪些模型，Provider Registry 提供模型访问凭证。分离这三类事实，是为了让受版本控制的模型与算法配置可以审查，而密钥保持在环境或 gitignored secret 文件中。

配置架构与所有权理由见 [System 配置与注册表](../system/configuration.md)；本文只说明当前怎样操作。

## 1. 文件与用途

| 文件/来源 | 用途 | 是否应提交真实值 |
|:---|:---|:---:|
| `configs/config.yaml` | System、Gateway、Patchouli、Alice、日志、调度等主配置的默认值，随版本更新 | 随仓库发布；个人改动不写这里 |
| `configs/config.local.yaml` | 主配置的个人覆盖：设置页保存的改动，只含与默认值不同的字段 | 不提交，已 gitignore |
| `configs/models.yaml` | 模型 ID、展示名、LiteLLM model、Provider 引用和默认采样参数；本机的个人配置，首次启动时从 `models.example.yaml` 生成 | 不提交，已 gitignore；模板随仓库发布；不要写密钥 |
| 根 `.env` 或 `configs/.env` | 环境覆盖与 Provider 密钥 | 不提交 |
| 环境变量 `HIVEMEMORY_DATA_ROOT` | 数据根：设置后，上面各个本机的个人配置文件都改在 `<数据根>/configs`，`.env` 与相对路径的数据目录按工作目录解析；桌面客户端设为用户选择的数据目录，开发方式不设置 | —— |
| `configs/providers.secrets.yaml` | 由 UI/API 管理的 Provider 凭证 | 不提交，已 gitignore |
| `configs/system_principals.yaml` | 访问登记：允许接入的调用来源（server 自身作为 `hivememory:http-server` 登记）；本机的个人配置，首次启动时从 `system_principals.example.yaml` 生成 | 不提交，已 gitignore；模板随仓库发布 |
| `configs/workspace_actors.yaml` | 访问登记：哪个用户/Agent 可以进入哪个 Workspace、能执行哪些操作；本机的个人配置，首次启动时从 `workspace_actors.example.yaml` 生成 | 不提交，已 gitignore；模板随仓库发布 |
| `configs/principals.secrets.yaml` | 插件页为外部 harness 生成的接入凭据 | 不提交，已 gitignore |
| `HIVEMEMORY_CONFIG_PATH` | 改用指定 YAML 主配置 | 由部署环境决定 |

`.env` 与 `configs/.env` 都会被配置加载器读取。为减少同一键在两个文件中重复，个人开发建议只维护根 `.env`；Docker Compose 也明确把根 `.env` 作为可选 `env_file`。容器同时挂载 `configs/`，因此其中的配置和 secret 文件可以持久化。

## 2. 配置来源优先级

当前优先级从高到低为：

```text
显式构造参数
  > 进程环境变量
  > .env / configs/.env
  > 旧环境变量 alias
  > Provider 动态凭证扫描
  > config.local.yaml（与主配置文件同目录）
  > config.yaml（或 HIVEMEMORY_CONFIG_PATH）
  > file secrets
```

各来源按层级合并：上层只替换自己写出的字段，同一段中没有写出的字段仍取下层的值。`.env` 与环境变量高于 `config.local.yaml`：被它们设置的字段，在设置页保存的值不生效。

嵌套环境变量以 `HIVEMEMORY__` 开头，用双下划线分段：

```env
HIVEMEMORY__PATCHOULI__STORAGE__HOST=127.0.0.1
HIVEMEMORY__PATCHOULI__STORAGE__PORT=6333
HIVEMEMORY__GATEWAY__WORKFLOW__DEFAULT_REQUEST_TIMEOUT_MS=8000
HIVEMEMORY__LOGGING__LEVEL=INFO
```

旧 `HIVEMEMORY__QDRANT__HOST` 等形式，以及 `HIVEMEMORY__LLM__GATEWAY__TEMPERATURE` 这类 Gateway、Librarian 生成参数的旧变量，仍由兼容 alias 接受，但新部署应使用当前配置树；直接指定这两个组件的模型、密钥或地址的旧变量不再生效。显式指定的 `HIVEMEMORY_CONFIG_PATH` 若不存在会阻止配置加载；默认 `configs/config.yaml` 不存在时则回退到环境变量和代码默认值。

## 3. 配置 Provider 凭证

### 3.1 环境变量，适合本地与部署注入

```env
HIVEMEMORY__PROVIDERS__DEEPSEEK__API_KEY=sk-...
HIVEMEMORY__PROVIDERS__DEEPSEEK__API_BASE=https://api.deepseek.com
```

`DEEPSEEK` 必须与 `models.yaml` 的 `provider: deepseek` 对应，匹配不区分大小写。环境变量层优先级最高，在 Provider 页面中显示为只读，不能通过 API 删除或覆盖。

### 3.2 Provider 页面/API，适合个人运行时管理

Settings -> Provider Credentials 通过 `/api/v1/providers` 写入 `configs/providers.secrets.yaml`。列表响应只返回脱敏 key；编辑时 key 留空会保留 YAML 层已有值。文件采用同目录临时文件加 `os.replace()` 原子更新。

也可以复制示例后手工编辑：

```text
configs/providers.secrets.example.yaml
  -> configs/providers.secrets.yaml
```

环境变量与 YAML 同名时，环境变量始终生效。删除 API 也只能删除 YAML 层。

## 4. 配置 Model Registry

`configs/models.yaml` 是模型定义的当前单一注册表。它是本机的个人配置，不受版本控制：首次启动时从随仓库发布的 `configs/models.example.yaml` 原样生成，此后模型页面的改动只写回 `models.yaml`，升级不会改动它；删掉它，下次启动从模板重新生成。每条记录主要包含：

- `id`：Agent Profile、Chat override 和组件配置引用的稳定 ID；
- `display_name`：前端展示名；
- `litellm_model`：交给 LiteLLM 的实际模型名；
- `provider`：用于解析凭证；
- temperature、top_p、max_tokens 默认值；
- 可选的 `context_window`：提供商支持的总上下文窗口 token 数，用于 Alice 计算输入预算；
- `is_default`：系统默认模型标记。

Settings -> Model Registry 或 `/api/v1/models` 可以运行时 CRUD，并原子写回 YAML。注册表会在新增/更新默认模型时清除其他默认标记；如果没有显式默认，解析 `default` 会回退到第一条记录。

模型定义不保存连接凭证：密钥与 API 地址只在 Provider Registry 中保存，模型按 `provider` 取得。`models.yaml` 中写入非空 `api_key` 或 `api_base` 的模型在加载时被跳过，日志指出模型 ID 与它们应在的位置；删除这两个字段、把值移到 Provider 后重启。单个模型需要不同的密钥或地址时（例如经代理访问），另登记一个 Provider 名称并让这个模型引用它。

Gateway 与 Librarian 的模型在 `configs/config.yaml` 的 `gateway.llm` 与 `patchouli.llm` 中以 `model_id` 引用这里的模型（不写或写 `default` 为默认模型），也可以在设置页的“内部引擎”中选择；整段写为 `null` 表示这个组件不使用 LLM。两段的 `temperature`、`top_p` 与 `max_tokens` 覆盖模型定义的默认值，不写时沿用；随仓库发布的配置写了组件任务要求的值，例如 Librarian 一次结算输出多条记忆，`max_tokens` 为 32768。这两个组件在启动时解析模型，修改模型、Provider 或这两段之后重启生效。

## 5. 主配置与生效边界

主配置接口不读写密钥。Provider 凭证不属于主配置，由 Provider 页面与 `.env` 管理；主配置中只有 Qdrant 的 `api_key`，只来自环境变量与 `.env`。`config.yaml` 与 `config.local.yaml` 中出现非空 `api_key` 时启动失败，错误指出字段路径。

`GET /api/v1/config` 与 `GET /api/v1/config/defaults` 的响应中，`api_key` 只返回打码值（如 `sk-...abcd`），与 Provider 接口相同：这些 HTTP 接口对本机任何进程开放。

`GET /api/v1/config` 返回下次启动将装载的配置（`config`），以及两份字段路径清单：`immediate_fields` 是保存后立即生效的字段，`pending_restart` 是已经保存、要重启后才生效的字段。

`POST /api/v1/config` 会：

1. 把提交中的 `api_key` 一律换成当前的运行时值：原样提交读到的打码值不会覆盖真实密钥，提交新值也不会改变密钥；
2. 用 `HiveMemoryConfig` 校验提交内容；
3. 把与默认值（代码默认值、`config.yaml`、`.env` 与环境变量）不同的字段写进 `config.local.yaml`，写入的内容不含 `api_key`；`config.yaml` 不被写；
4. 把其中立即生效的字段写入运行中的服务，例如检索条数上限、相关度阈值、话题空闲结算时间、会话与附件上限（完整清单见 [System 配置](../system/configuration.md#21-个人覆盖与保存)）；
5. 返回与 `GET` 相同的结构。

其余字段（模型选择与采样参数、存储连接、嵌入与重排模型、调度任务、日志、默认语言等）在启动时就交给了长期存活的组件，保存后出现在 `pending_restart` 中，重启服务后生效。重启后再用 readiness 和实际请求确认。

恢复默认：`GET /api/v1/config/defaults` 返回不含 `config.local.yaml` 的默认值，把它原样保存即清空个人覆盖；也可以直接删掉 `config.local.yaml` 后重启。

ProviderRegistry 和 ModelRegistry 自身支持运行时 CRUD；Provider 凭证在后续动态解析时可立即被读取。但已经解析并交给长期存活组件的配置仍可能需要重启。

### 5.1 访问登记

与 Workspace 相关的 HTTP 请求（Chat、停止、对话导入、MCP 记忆访问、记忆、话题、Agent、附件、记忆任务）都先经统一认证网关。前端使用的路由由 server 以 `system.server_principal_id` 登记的 principal 接入；对话导入与 MCP 由外部 harness 以自己登记的 principal 与接入凭据接入（例如 Claude Code 为 `harness:claude-code`）；这些登记与凭据在设置页的“插件”页安装插件时产生，见各 harness 的接入手册（例如[接入 Claude Code](./claude-code-plugin.md#2-从插件页安装)）。随后按 `configs/workspace_actors.yaml` 判断请求的用户与 Agent 能否进入 Workspace、能执行哪些操作。Chat 的话题准备、Profile、附件与交互提交，以及对话导入，都由 HiveMemory 自己以登记的 `hivememory:system` 执行，不受 Agent 的操作许可限制。从模板生成的默认登记覆盖本地默认用户 `default` 在 `main_workspace` 中的所有具体 Agent（读取、检索、运行、写入、更新与调用其他 Agent），并单独登记保留的 `system`（管理操作，以及上面这些由 HiveMemory 执行的步骤），不接入任何外部 harness。插件页安装插件时，为插件的缺省 Agent（例如 `claude_code`）补一条获准 `memory.read`、`memory.search`、`memory.write` 与 `memory.update` 的记录（四个记忆工具；对话导入只要求准入，见[接入 Claude Code](./claude-code-plugin.md#5-访问登记)）。

- 使用其他用户（请求头 `x-user-id`）时，需要在 `workspace_actors.yaml` 中为该用户登记对应记录，并单独登记该用户的 `system`（不写 `agent_id` 的用户级记录不包含它），否则请求返回 403（`reason=actor_not_admitted`）；
- 外部 harness 的接入在设置页的“插件”页安装插件时登记、卸载时删除；“接入与权限”中可以启停外部来源、调整它能声明哪些 Agent，以及编辑各 Agent 的操作权限；修改立即生效，并写回本机的生效登记（不保留注释）。生效登记是本机的个人配置，不受版本控制：首次启动时从随仓库发布的 `*.example.yaml` 模板生成，此后升级不会改动它们；两个文件也可以分别用 `HIVEMEMORY_PRINCIPALS_PATH`、`HIVEMEMORY_WORKSPACE_ACTORS_PATH` 指向其他位置，指定的文件必须已经存在。直接修改文件后需要重启服务；运行中手工改过文件时，页面上的修改会被拒绝，重启后再改；
- 默认位置的登记文件被删掉时，下次启动从模板重新生成；路径变量指定的文件缺失时启动失败；
- `configs/system_principals.yaml` 中每个调用来源能否以 `system` 进入管理接口由 `allow_system_actor` 决定，默认不能；自定义过该文件时，server 的登记需要显式设为 `true`，否则管理接口返回 403（`reason=system_actor_not_allowed`）；
- 登记模板包含 HiveMemory 自己的来源 `hivememory:system`。更早生成或自定义过的生效登记缺少它时要补上这一条（`kind: local-process`、`adapters: [local]`、`allow_system_actor: true`，不写 `credential_env`），否则 Chat、对话导入与外部工具的记忆写入、更新都返回 403（`reason=unknown_principal`）；这个原因与外部工具自己的 principal 未登记相同，排查时两条登记都要核对；
- 外部 harness 的接入凭据由插件页安装时生成，保存在登记文件同目录的 `principals.secrets.yaml`（不受版本控制），同时写进 harness 一侧读取的 Windows 用户环境变量；更早手工配置的登记也可以沿用 `credential_env` 指向的环境变量，写在 `.env` 或环境变量中，它有值时优先于生成的凭据，修改后重启服务；
- 插件页登记的来源初始只能声明插件的缺省 Agent（`allowed_agent_ids`）；同一个 harness 可以以多个 Agent 运行，改用其他 Agent 时先在“接入与权限”中把它加入，否则请求返回 403（`reason=agent_not_allowed_for_principal`）；
- 前端使用的接口在经过认证网关之前，另要求后端首次启动时生成的管理凭据 `configs/management.token`；它不需要配置，前端自动带上，终端调用的写法见[安装与启动](./setup.md#43-启动后端)。桌面客户端不使用这个文件，凭据只在内存中（[使用桌面客户端](./desktop-client.md)）；
- 请求头中的用户身份不做证明，管理凭据也只防误用，两者都只适用于本地单用户部署。

字段说明与规则见 [System 配置](../system/configuration.md)第 1.1 节与 [认证与访问登记](../workspace/authentication.md#2-两类访问登记)第 2 节。

### 5.2 Alice 会话压缩

自动压缩由主配置的 `alice.compaction` 控制；完整默认值与单位见 [System 配置](../system/configuration.md#12-alice-会话预算与压缩)。在设置页保存时 `enabled` 与 `trigger_ratio` 立即生效，其余参数与直接修改文件一样，重启运行实例后生效。模型窗口可在 `configs/models.yaml` 或模型 API 登记，不把总窗口误填为最大输出长度；未登记时使用保守默认窗口，请求的 `max_tokens` 也会占用窗口预算。

关闭 `enabled` 会让 Alice 忽略已有检查点、重放原文，但不会关闭每次生成前的输入守卫。压缩不删除会话原文、不增加会话容量，也不提供跨后端重启保存；当前没有手动 `/compact` 入口。

### 5.3 对话导入（Import Bus）

Claude Code、DeepSeek Harness 等外部工具的对话经 `POST /api/v1/import/turns` 导入，安装、凭据与插件配置见[接入 Claude Code](./claude-code-plugin.md)与[接入 DeepSeek Harness](./deepseek-harness-plugin.md)。服务端的 `import_bus` 段只有两项，`configs/config.yaml` 中没有写出时按默认值运行：

| 键 | 默认值 | 说明 |
|:---|---:|:---|
| `import_bus.max_request_bytes` | `8388608`（8 MiB） | 单次导入请求体的上限；超出的那一轮被拒绝（413），连接器记录后跳过 |
| `import_bus.apply_wait_seconds` | `30` | 提交后等待记忆系统应用这一轮的秒数；超时返回 503，连接器稍后重发 |

连接器等待一次导入的上限是 60 秒，调大 `apply_wait_seconds` 或 Gateway 请求时限（`gateway.workflow.default_request_timeout_ms`，默认 8 秒）时，两者之和应保持在这个范围内。

升级注意：旧的 `passive_ingress` 段以及 `scheduler.tasks` 下的 `observer_idle_flush_interval_seconds`、`observer_idle_flush_timeout_seconds`、`enable_observer_idle_flush` 已删除。配置文件中仍有这些键时服务无法启动，需要删除它们；配置文件中出现任何未知的顶层段同样会让启动失败。旧的 `/api/v1/ingest` 与 `/api/v1/ingest/flush` 已由 `/api/v1/import/turns` 取代。

## 6. 当前 Settings 页面限制

Provider、Model Registry 与接入与权限使用独立、已对齐的 API。其余主配置表单仍采用旧的扁平前端类型，而后端已经使用 `shared/patchouli/alice/gateway/...` 嵌套结构；部分分类可能显示错误、读取空字段或提交不完整结构。当前不要把 Settings 中所有可见开关都视为可靠控制面。

需要修改主配置时，优先直接编辑 `configs/config.local.yaml`（只写要改的字段，层级与 `config.yaml` 相同，修改后重启）或通过经过核验的 API 客户端提交完整当前结构。`config.yaml` 是随版本更新的默认值，不在其中做个人改动。前端类型完成迁移后，本限制才能移除。

## 7. 安全检查

- 不提交 `.env`、`configs/providers.secrets.yaml`、`configs/management.token` 或任何真实 API key；
- 构建 Docker 镜像时，`configs` 中的三份本机密钥文件（`configs/.env`、`configs/providers.secrets.yaml`、`configs/management.token`）被 `.dockerignore` 排除，不进入镜像；容器经 Compose 挂载的 `configs` 目录读取它们，不经 Compose 运行镜像时需要自行挂载；
- 生产环境优先使用进程/编排环境注入，并限制配置 API 的网络可达性：后端默认只监听本机地址，改为对外监听前确认网络范围；当前服务没有登录系统，管理凭据只防误用，请求头中的用户身份不做证明，访问登记只适用于本地单用户部署；
- 修改 Provider/Model/Config 后检查实际进程使用的模型，而不只看页面保存 toast；
- 项目版本不是可覆盖配置；代码、Python 包、FastAPI 与 health 使用同一规范版本，是否已经发布仍以匹配的 Git tag 为准，完整口径见 [Project](../PROJECT.md)。
