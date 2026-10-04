---
title: HiveMemory Configuration Guide
status: current
owner: system
scope: operator-configuration-providers-and-models
code_paths:
  - configs/config.yaml
  - configs/models.yaml
  - configs/.env.example
  - configs/providers.secrets.example.yaml
  - configs/system_principals.yaml
  - configs/workspace_actors.yaml
  - src/hivememory/config/access.py
  - src/hivememory/config/
  - src/hivememory/system/provider_registry.py
  - src/hivememory/system/model_registry.py
  - src/hivememory/server/routers/config.py
related_contracts:
  - docs/architecture/boundaries.md
last_reviewed: 2026-10-04
---

# HiveMemory 配置指南

HiveMemory 把配置分为三类：主配置描述系统如何装配，Model Registry 描述可选择哪些模型，Provider Registry 提供模型访问凭证。分离这三类事实，是为了让受版本控制的模型与算法配置可以审查，而密钥保持在环境或 gitignored secret 文件中。

配置架构与所有权理由见 [System 配置与注册表](../system/configuration.md)；本文只说明当前怎样操作。

## 1. 文件与用途

| 文件/来源 | 用途 | 是否应提交真实值 |
|:---|:---|:---:|
| `configs/config.yaml` | System、Gateway、Patchouli、Alice、日志、调度等主配置 | 可以提交非敏感默认值 |
| `configs/models.yaml` | 模型 ID、展示名、LiteLLM model、Provider 引用和默认采样参数 | 可以提交；不要写密钥 |
| 根 `.env` 或 `configs/.env` | 环境覆盖与 Provider 密钥 | 不提交 |
| `configs/providers.secrets.yaml` | 由 UI/API 管理的 Provider 凭证 | 不提交，已 gitignore |
| `configs/system_principals.yaml` | 访问登记：允许接入的调用来源（server 自身作为 `hivememory:http-server` 登记） | 可以提交；不含凭据 |
| `configs/workspace_actors.yaml` | 访问登记：哪个用户/Agent 可以进入哪个 Workspace、能执行哪些操作 | 可以提交；不含凭据 |
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
  > config.yaml（或 HIVEMEMORY_CONFIG_PATH）
  > file secrets
```

嵌套环境变量以 `HIVEMEMORY__` 开头，用双下划线分段：

```env
HIVEMEMORY__PATCHOULI__STORAGE__HOST=127.0.0.1
HIVEMEMORY__PATCHOULI__STORAGE__PORT=6333
HIVEMEMORY__GATEWAY__WORKFLOW__DEFAULT_REQUEST_TIMEOUT_MS=8000
HIVEMEMORY__LOGGING__LEVEL=INFO
```

旧 `HIVEMEMORY__QDRANT__HOST`、`HIVEMEMORY__LLM__GATEWAY__...` 等形式仍由兼容 alias 接受，但新部署应使用当前配置树。显式指定的 `HIVEMEMORY_CONFIG_PATH` 若不存在会阻止配置加载；默认 `configs/config.yaml` 不存在时则回退到环境变量和代码默认值。

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

`configs/models.yaml` 是模型定义的当前单一注册表。每条记录主要包含：

- `id`：Agent Profile、Chat override 和组件配置引用的稳定 ID；
- `display_name`：前端展示名；
- `litellm_model`：交给 LiteLLM 的实际模型名；
- `provider`：用于解析凭证；
- temperature、top_p、max_tokens 默认值；
- `is_default`：系统默认模型标记。

Settings -> Model Registry 或 `/api/v1/models` 可以运行时 CRUD，并原子写回 YAML。注册表会在新增/更新默认模型时清除其他默认标记；如果没有显式默认，解析 `default` 会回退到第一条记录。

优先在 Provider Registry 保存密钥。Model 页面允许高级单模型 api key 覆盖，但 `models.yaml` 受版本控制，除非有明确的本地隔离措施，否则不要使用这一入口保存真实密钥。

## 5. 主配置与生效边界

`POST /api/v1/config` 会：

1. 用 `HiveMemoryConfig` 校验提交内容；
2. 原子写入当前 YAML 主配置；
3. 替换进程中的 `system.config`；
4. 返回校验后的配置。

这不保证所有组件热重载。已经在启动时创建的 Qdrant/Embedding/Reranker、Gateway/Librarian LLM config、scheduler 和部分 runtime 可能仍持有旧对象。涉及基础设施、模型装配、调度或子系统行为的变更，当前最安全做法是保存后重启服务，再用 readiness 和实际请求确认。

ProviderRegistry 和 ModelRegistry 自身支持运行时 CRUD；Provider 凭证在后续动态解析时可立即被读取。但已经解析并交给长期存活组件的配置仍可能需要重启。

### 5.1 访问登记

与 Workspace 相关的 HTTP 请求（Chat、停止、记忆、话题、Agent、附件、记忆任务）都先经统一认证网关：server 以 `system.server_principal_id` 登记的 principal 接入，再按 `configs/workspace_actors.yaml` 判断请求的用户与 Agent 能否进入 Workspace、能执行哪些操作。随仓库发布的默认登记覆盖本地默认用户 `default` 在 `main_workspace` 中的所有具体 Agent（Chat 所需的读取、检索、Profile、附件与交互提交），并为管理操作单独登记保留的 `system`（记忆、话题、附件与任务管理）。

- 使用其他用户（请求头 `x-user-id`）时，需要在 `workspace_actors.yaml` 中为该用户登记对应记录，否则请求返回 403（`reason=actor_not_admitted`）；
- 两个登记文件可分别用 `HIVEMEMORY_PRINCIPALS_PATH`、`HIVEMEMORY_WORKSPACE_ACTORS_PATH` 指向其他位置；修改后需要重启服务；
- 登记文件缺失时按空登记处理，所有 Workspace 请求都会被拒绝；
- 请求头中的用户身份不做证明，只适用于本地单用户部署。

字段说明与规则见 [System 配置](../system/configuration.md)第 1.1 节与 [Workspace 架构](../architecture/workspace.md)第 4.2 节。

## 6. 当前 Settings 页面限制

Provider 与 Model Registry 使用独立、已对齐的 API。其余主配置表单仍采用旧的扁平前端类型，而后端已经使用 `shared/patchouli/alice/gateway/...` 嵌套结构；部分分类可能显示错误、读取空字段或提交不完整结构。当前不要把 Settings 中所有可见开关都视为可靠控制面。

需要修改主配置时，优先直接编辑 `configs/config.yaml` 或通过经过核验的 API 客户端提交完整当前结构，并在修改前保留版本控制 diff。前端类型完成迁移后，本限制才能移除。

## 7. 安全检查

- 不提交 `.env`、`configs/providers.secrets.yaml` 或任何真实 API key；
- 不因为 API 响应会脱敏，就在 `models.yaml` 的高级覆盖中保存密钥；
- 生产环境优先使用进程/编排环境注入，并限制配置 API 的网络可达性；当前服务没有登录系统，请求头中的用户身份不做证明，访问登记只适用于本地单用户部署；
- 修改 Provider/Model/Config 后检查实际进程使用的模型，而不只看页面保存 toast；
- 项目版本不是可覆盖配置；代码、Python 包、FastAPI 与 health 使用同一规范版本，是否已经发布仍以匹配的 Git tag 为准，完整口径见 [Project](../PROJECT.md)。
