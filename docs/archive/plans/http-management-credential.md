---
title: HTTP 管理面的管理凭据
status: archived
archived_at: 2026-11-05
implemented_by: PR #182
superseded_by: docs/workspace/authentication.md
owner: system
target: 版本之外的工作（owner，2026-11-05：不绑定版本）
scope: management-credential-for-all-http-management-routes-and-local-listen-address
updates:
  - docs/workspace/authentication.md
  - docs/workspace/identity-and-authorization.md
  - docs/contracts/error-model.md
  - docs/frontend/state-and-transports.md
  - docs/help/configuration.md
  - docs/help/setup.md
  - docs/help/troubleshooting.md
  - docs/contracts/routes-and-events.md
  - docs/governance/security/identity-and-execution-safety.md
  - README.md
code_paths:
  - src/hivememory/server/management_credential.py
  - src/hivememory/server/app.py
  - src/hivememory/server/__main__.py
  - src/hivememory/server/error_handlers.py
  - frontend/vite.config.ts
  - docker/docker-compose.yml
  - .gitignore
  - .dockerignore
related_docs:
  - docs/ideas/identity-and-access-model.md
  - docs/archive/todo/local-http-management-access.md
  - docs/governance/baselines/identity-access-current-state-2026-11-05.md
  - docs/workspace/authentication.md
  - docs/contracts/error-model.md
---

# HTTP 管理面的管理凭据

> **状态**：Archived（2026-11-05，已实施，实施记录见第 10 节）。当前事实见[认证与访问登记](../../workspace/authentication.md#1-两阶段认证)第 1 节与[安装与启动](../../help/setup.md#43-启动后端)。以下为实施时的计划原文，链接已按归档位置改正。方向来自[身份与访问体系 Idea](../../ideas/identity-and-access-model.md#http-管理入口的调用方证明)第 4 节的 owner 决定（2026-11-05）：采用管理凭据，现在先做到防误用，防刻意随 v1.0 的桌面客户端完成。本计划承接[本机 HTTP 管理接口 Todo](../todo/local-http-management-access.md)，不绑定版本。

## 1. 背景和问题

HTTP 接口沿用本地单用户部署的信任假设，不证明调用方是谁（[身份与访问体系现状调查](../../governance/baselines/identity-access-current-state-2026-11-05.md)第 5 节）：

- 管理路由以 server 的 principal 和 `(请求头中的用户, system)` 声明经认证网关进入 root API。请求头只是文本，任何程序都能发出同样的请求。
- `config`、`providers`、`models`、运行时事件与日志 WebSocket 完全不经认证网关，可以读写主配置、提供商密钥与模型注册表，并读取含内容片段的日志与事件。
- `hivememory-server` 固定监听 `0.0.0.0`，Docker Compose 以 `8000:8000` 在宿主的全部地址上开放端口。WSL2 还会把 Windows 上的本机地址转发到 WSL 中的服务。

只有 `/mcp` 与 `/api/v1/import/turns` 要求外部 harness 出示各自的凭据。2026-11-01，dsh 中的 agent 经本机转发直接调用 `/api/v1/config`，取得了明文密钥（Todo 中的证据）。外部 harness 的 agent 因此可以绕过 Workspace 访问登记的白名单：管理记忆、改写配置、写入密钥。

## 2. 目标与非目标

### 2.1 目标

1. `/api/` 下除导入之外的全部路由都要求管理凭据，包括普通请求、chat 的流式响应、运行时事件的 SSE 与日志 WebSocket。缺少或不一致时拒绝，路由处理函数不会执行。
2. 前端在 Vite 开发模式与后端提供页面两种运行方式下照常工作，前端源码不处理凭据。
3. 正式入口默认只监听本机地址；Docker Compose 只在宿主的本机地址上开放端口。
4. 防护等级为防误用：agent 在工作中顺手调用管理接口时被拒，局域网中的其他机器连不进来。

### 2.2 非目标

- **防刻意**：以同一系统用户运行、专门在本机寻找凭据的程序仍能读到凭据文件。防刻意由 v1.0 的桌面客户端完成，主进程在内存中持有凭据，不落盘。
- **统一身份授权入口**：认证网关、两类访问登记、访问 context 与授权者都不改动。统一入口是 Idea 第 5 节的方向，另立计划。
- **请求头中的用户身份**：`x-user-id` 仍是声明。管理凭据证明的是“调用方是本机的前端或持有凭据的人”，不区分用户；当前只有一个用户。
- **harness 的接入凭据**：`/mcp` 与导入的凭据、principal 登记不变。
- 凭据的轮换、过期、多用户登录与 HTTPS。

## 3. 当前状态和缺口

| 入口 | 现在 | 本计划之后 |
|:---|:---|:---|
| 管理路由（agents、memories、memory-tasks、sessions、topics、workspace/assets） | 以 server 的 principal 经网关进入 root API，不证明调用方 | 先核对管理凭据，再照旧经网关 |
| `POST /api/v1/chat`、`POST /api/v1/chat/stop` | 同上，分别进入任务进程注册与进程控制 | 同上 |
| `config`、`providers`、`models` | 不经网关，可读写 | 核对管理凭据 |
| 运行时事件 SSE、日志 WebSocket | 不经网关，内容不过滤 | 核对管理凭据 |
| `/api/v1/import/turns`、`/mcp` | harness 自己的凭据 | 不变 |
| `/health`、`/health/ready`、前端静态文件 | 不需要身份 | 不变 |
| 监听地址 | `0.0.0.0:8769`；Compose `8000:8000` | `127.0.0.1:8769`，可显式配置；Compose `127.0.0.1:8000:8000` |

## 4. 目标设计

### 4.1 凭据的生成与保存

- 凭据保存在项目根目录的 `configs/management.token`。它与 `configs/providers.secrets.yaml` 同处一个目录：都是本机密钥，都被 `.gitignore` 排除，都随 Docker Compose 挂载的 `configs` 目录持久化。
- server 在应用启动时确保这个文件存在：
  - 不存在就生成一个随机凭据（`secrets.token_urlsafe(32)`），以同目录临时文件加原子替换的方式写入，文件权限设为只有当前用户可读写；
  - 存在就读取它，去掉首尾空白。
- 删除这个文件后重启后端，会生成新的凭据。不设环境变量覆盖，不设轮换。
- 凭据只存在于这个文件和后端进程内存中，不写入日志、配置接口的响应与错误信息。
- 凭据属于 server（入口适配层）：它保护的是 HTTP 管理面，不是 Workspace 准入，因此不进入接入登记，也不交给 System 或 workspace。

### 4.2 校验

- **位置**：真实应用上的一个 ASGI 中间件，同时处理 HTTP 与 WebSocket 两类连接。只按路径判定，默认拒绝：
  - 路径以 `/api/` 开头的连接都要求管理凭据，`/api/v1/import/turns` 除外；
  - 其余路径不要求，包括 `/health`、`/health/ready`、`/mcp` 与前端静态文件。

  以后在 `/api/` 下新增的路由自动受保护，不需要逐个路由声明依赖。
- **凭据的携带方式**：`Authorization: Bearer <管理凭据>` 请求头，或名为 `hivememory_management` 的 cookie，两者任一与后端持有的凭据一致即可。比较使用常量时间比较。
- **拒绝**：
  - HTTP 请求返回 401，响应体沿用统一的错误信封，错误码为 `request.management_credential_invalid`，说明文字不含凭据本身。中间件位于异常处理之外，直接写出响应。
  - WebSocket 在接受连接之前以策略违规关闭。
  - 启动完成之前凭据尚未读入时，同样按凭据无效拒绝。
- **为什么不放进认证网关**：
  - `config` 等五类接口不经网关，放在网关第 1 阶段覆盖不到它们；在中间件中只有一处检查，覆盖全部 HTTP 管理面。
  - 网关、接入登记与“进程内 adapter 不带凭据”的装载规则都保持不变，本计划不触及身份模型。
  - 统一身份授权入口落地时，通过这个检查的 HTTP 请求就是“经过证明”的调用方，由那份计划把它表达进工作 actor。

### 4.3 前端取得凭据

| 运行方式 | 方式 | 浏览器页面是否持有凭据 |
|:---|:---|:---|
| Vite 开发模式 | Vite 的 `/api` 代理在转发 HTTP 与 WebSocket 请求时附上 `Authorization` 请求头。凭据从 `../configs/management.token` 读取：每次转发时读文件，这样 Vite 先于后端启动、或凭据文件重新生成之后都不需要重启 Vite | 否 |
| 后端提供页面（`HIVEMEMORY_SERVE_FRONTEND=true`，含 Docker） | SPA 兜底路由返回页面时下发 cookie：`HttpOnly`、`SameSite=Strict`、`Path=/`。同源的 fetch、SSE 与 WebSocket 会自动携带它 | 否，脚本读不到 `HttpOnly` cookie |

- 两种方式下，前端源码都不读取、不保存凭据，现有的 fetch、`EventSource` 与 WebSocket 客户端不改。
- `SameSite=Strict` 使浏览器中其他站点发起的请求不会携带这个 cookie。
- 后端提供页面时，任何程序先请求页面也能取得 cookie，这只达到防误用，与第 2.2 节一致。

### 4.4 监听地址

- `hivememory-server` 默认监听 `127.0.0.1:8769`。需要对外监听时，以 `HIVEMEMORY_BACKEND_HOST` 环境变量显式指定，与开发脚本已用的变量同名。
- Docker 容器内仍监听 `0.0.0.0:8000`（容器网络需要），Compose 的端口映射改为 `127.0.0.1:8000:8000`。
- 开发脚本默认已是 `127.0.0.1`，不改。

### 4.5 手动调用

在终端中直接调用管理接口，需要带上凭据：

```bash
curl -H "Authorization: Bearer $(cat configs/management.token)" \
  http://127.0.0.1:8769/api/v1/memories
```

帮助文档相应说明。harness 插件只调用 `/mcp` 与导入，不受影响。

## 5. 受影响的系统、契约与数据

- **server**：
  - 新的凭据模块：负责生成、读取与比较；
  - `app.py`：注册中间件，启动时读入凭据，SPA 兜底路由下发 cookie；
  - `__main__.py`：默认监听地址；
  - `error_handlers.py`：登记新的错误码。
- **前端**：`frontend/vite.config.ts` 的代理配置。不改前端源码。
- **部署**：`docker/docker-compose.yml` 的端口映射；`.gitignore` 增加 `configs/management.token`。
- **契约**：[错误模型](../../contracts/error-model.md)新增 `request.management_credential_invalid`（401）。路由、请求体与响应体不变。
- **不变**：System、workspace、Patchouli、认证网关、两类访问登记、MCP 与导入入口、harness 插件。

## 6. 兼容与迁移

- 升级后首次启动后端会生成凭据文件，无需手动配置。
- 正在运行的 Vite 开发服务器需要重启一次，才能加载新的代理配置。
- 在终端中直接调用管理接口的脚本与习惯需要带上凭据（第 4.5 节）。
- 原来从局域网其他机器访问后端的部署，需要显式设置 `HIVEMEMORY_BACKEND_HOST`。在 Docker 中，需要自行改回端口映射。
- 不涉及数据迁移，不改动任何持久化格式。

## 7. 测试和验收标准

需要守住的不变量与归属层：

| 不变量 | 归属层 |
|:---|:---|
| 真实应用中，`/api/` 下除导入之外的每个 HTTP 与 WebSocket 路由，在缺少或携带错误凭据时被拒绝，且处理函数不执行；断言从应用的路由表枚举，新增路由自动纳入 | `tests/unit/server/` |
| 携带正确凭据（请求头或 cookie）时请求照常到达路由 | `tests/unit/server/` |
| `/health`、`/health/ready`、`/mcp`、`/api/v1/import/turns` 与静态页面不要求管理凭据 | `tests/unit/server/` |
| 凭据文件不存在时生成且只有当前用户可读写，存在时复用；凭据不出现在错误响应中 | `tests/unit/server/` |
| 后端提供页面时，页面响应下发带 `HttpOnly`、`SameSite=Strict` 的 cookie | `tests/unit/server/` |

- 现有路由测试大多自行组装 `FastAPI()` 并挂载路由，不经过真实应用的中间件，不需要改动。使用真实应用的测试（`test_error_handlers.py`、`test_app_lifespan.py`）按需调整。
- 监听地址与 Vite 代理不写自动化测试，以手动验收确认：
  1. 用开发脚本启动后，前端的记忆、话题、会话、chat、设置、运行时事件与日志页面都正常；
  2. 不带凭据 `curl` 管理接口得到 401，带凭据成功；
  3. 后端提供页面的模式下同样正常；
  4. dsh 与 Claude Code 的检索、写入与对话回流不受影响。

验证命令：

```bash
pytest tests/unit/server -q
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85
ruff check . && black --check .
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
cd frontend && npm run lint && npm run build
```

## 8. 风险与待决问题

- **Vite 代理对 WebSocket 附加请求头**：需要在实施时确认 Vite 代理能为 WebSocket 升级请求附上请求头。若不能，改为 Vite 为页面下发与后端相同的 cookie，前端源码仍然不改。
- **前端对 401 的呈现**：401 只在配置有误时出现，例如 Vite 读不到凭据文件。前端按现有的错误呈现显示说明文字，本计划不新增专门的界面。
- **凭据文件可被同一用户的程序读取**：这是防误用的边界（第 2.2 节），帮助文档写明。
- **待决**：无阻塞问题。错误码名称与 cookie 名称可在评审时调整。

## 9. 完成时必须更新的当前文档

- [认证与访问登记](../../workspace/authentication.md)：第 1 节“请求头中的用户身份不做证明”与第 8 节已知限制，改为说明 HTTP 管理面由管理凭据保护、请求头中的用户仍是声明。
- [身份与授权](../../workspace/identity-and-authorization.md)：第 8 节关于“能连到 server 的程序都能以 `system` 进入 root API”的限制。
- [错误模型](../../contracts/error-model.md)：新增错误码。
- [状态、持久化与传输](../../frontend/state-and-transports.md)：前端经 Vite 代理或 cookie 携带管理凭据。
- 帮助：[配置指南](../../help/configuration.md)第 5.1 节的信任假设、[安装与运行](../../help/setup.md)的监听地址与手动调用、[排障](../../help/troubleshooting.md)的 401。
- [身份治理](../../governance/security/identity-and-execution-safety.md)第 1 节“管理入口（root）”一行的当前风险。
- 收尾时：本计划归档；[本机 HTTP 管理接口 Todo](../todo/local-http-management-access.md) 归档；Idea 第 4 节、ROADMAP 第 4.2 节“本机 HTTP 接口的密钥与访问控制”一行与 [Plans 导航](../../plans/README.md)同步。

## 10. 实施记录

**实现**（2026-11-05，分支 `feat/http-management-credential`）：

- server：新模块 `server/management_credential.py` 负责凭据文件的生成与读取、按路径判定与中间件；`app.py` 在 lifespan 中把凭据读入 `app.state`，中间件先于 CORS 登记、位于 CORS 之内，跨域预检不要求凭据；`__main__.py` 默认监听 `127.0.0.1`，`HIVEMEMORY_BACKEND_HOST` 可改；`error_handlers.py` 登记 401 `request.management_credential_invalid`，中间件取用同一张表的状态码与说明文字。
- 与第 4 节的出入：
  - 401 响应带 `WWW-Authenticate: Bearer`；
  - 拒绝时记一条 warning 日志（连接类型与路径，不含凭据）；
  - 凭据文件存在但为空时启动失败，不重新生成；
  - SPA 兜底路由抽为 `mount_frontend(app, dist_dir)`，为了能在测试中装到最小应用上；cookie 只随页面（`index.html`）下发，直接请求的静态文件不带；
  - `.dockerignore` 排除 `configs/management.token`，否则 Dockerfile 的 `COPY configs/` 会把本机凭据打进镜像。
- 前端：`vite.config.ts` 的 `/api` 代理在 `proxyReq` 与 `proxyReqWs` 两个事件中附上请求头。第 8 节“Vite 代理对 WebSocket 附加请求头”的风险已排除：Vite 7.3 内置的 http-proxy-3 在 WebSocket 升级请求发出前触发 `proxyReqWs`，不需要退回 cookie。
- 部署：Compose 端口映射改为 `127.0.0.1:8000:8000`；`.gitignore` 排除凭据文件。

**测试**：不变量写入 `tests/unit/server/test_management_credential.py`，另改了两处现有测试：`test_app_lifespan.py` 断言凭据读入 `app.state`（加载函数打桩，不写仓库的 `configs/`），`test_main.py` 的默认监听地址改为 `127.0.0.1`。

- 真实应用的路由表枚举覆盖全部路由而不只是 `/api/`：除显式列出的公开路由（存活与就绪检查、`/mcp`、导入、接口文档）之外都必须被拒，以后在 `/api/` 之外新增的管理路由也会让测试失败。FastAPI 0.141 对 include 进来的 WebSocket 路由只在 `starlette_route` 上给出完整路径，测试对此有说明，并断言枚举到了配置接口与日志 WebSocket，避免枚举失效时静默通过。
- 已逐一核对：去掉中间件的登记时，路由表测试失败；去掉 cookie 的 `HttpOnly` 时，页面 cookie 测试失败。

**验证**：

- `pytest tests/unit/server -q`：232 通过；
- `pytest tests/unit tests/integration -m "not live_llm and not e2e and not slow" -n auto --cov=hivememory --cov-fail-under=85`：3220 通过，覆盖率 94.74%；
- `ruff check .`、`black --check .`、`mypy`（445 个文件）通过；
- `npm run lint`、`npm run build` 通过。

**手动验收**（第 7 节）：

- 已完成：以真实中间件加最小路由的应用监听临时端口，用本仓库的 Vite 配置把代理目标指向它。
  - 经 Vite 代理的 HTTP 与 WebSocket 都通过；直连后端时，不带凭据的 HTTP 得到 401，WebSocket 握手被拒（403），带凭据的 curl 成功；
  - 生成的凭据文件权限为 600；
  - 验收用的凭据文件已删除，后端首次启动时会重新生成。
- 未完成：第 1、3、4 项要用这个分支重启正在运行的后端（它从本仓库以 `serve` 模式运行，dsh 与 Claude Code 正在使用），在归档后的日常使用中进行。

**实施中发现的同类问题**：`.dockerignore` 同样没有排除 `configs/providers.secrets.yaml` 与配置加载器会读取的 `configs/.env`，`docker build` 会把本机密钥打进镜像。owner 要求一并修复（2026-11-05），两者已加入 `.dockerignore`；本机没有 Docker，未以实际构建验证。
