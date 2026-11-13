---
title: 桌面端安装插件
status: archived
archived_at: 2026-11-12
implemented_by: PR #218
superseded_by: docs/frontend/desktop-client.md
owner: system
target: v1.0 范围（桌面客户端与四个 harness 的插件模式）
scope: plugin-directory-in-the-installer-empty-access-on-first-start-and-system-performed-plugin-installation-for-the-four-supported-harnesses
updates:
  - docs/frontend/desktop-client.md
  - docs/frontend/management-views.md
  - docs/contracts/routes-and-events.md
  - docs/system/configuration.md
  - docs/workspace/authentication.md
  - docs/help/desktop-client.md
  - docs/help/setup.md
  - docs/help/claude-code-plugin.md
  - docs/help/codex-plugin.md
  - docs/help/deepseek-harness-plugin.md
  - docs/help/pi-plugin.md
code_paths:
  - desktop/scripts/package.mjs
  - configs/system_principals.example.yaml
  - configs/workspace_actors.example.yaml
  - src/hivememory/system/access/
  - src/hivememory/system/application/
  - src/hivememory/server/routers/
  - integrations/hivememory/
  - frontend/src/components/settings/
  - frontend/src/services/
related_docs:
  - docs/frontend/desktop-client.md
  - docs/workspace/authentication.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
---

# 桌面端安装插件

> **状态**：Archived（2026-11-12，已实施，实施记录见第 10 节）。当前事实见[桌面客户端](../../frontend/desktop-client.md)第 3、8 节、[管理页面](../../frontend/management-views.md#35-插件)第 3.5 节与四份插件帮助（例如[接入 Claude Code](../../help/claude-code-plugin.md)）。以下为实施时的计划原文，第 4.4、4.6 节已按实现改写，链接已按归档位置改正。第 7 节的手动验收由 owner 在日常使用中进行，不阻塞归档。owner 于 2026-11-12 决定先做这一项：桌面安装版没有安装插件的入口，桌面端因此失去意义。本计划的前提由 owner 同日决定：
>
> - 桌面端提供安装插件的入口；
> - 新用户安装之后没有任何连接，只有用户主动选择为某个 harness 安装插件之后才有连接；
> - 现在“登记外部 harness”的做法不对：登记出来的来源没有作用对象；
> - 由系统代为安装：调用 harness 自己的插件命令，找不到 harness 或命令失败时，改为显示要执行的命令；
> - Claude Code 与 Codex 的 hook 使用安装版自带的 Python；
> - skill 的更新随发布版本走，升级到新的发布版本之后要能自动更新（owner 在讨论 skill 时提出，本计划把它用到整个插件）。
>
> 其余内容是候选设计，不是 owner 决定，可在评审与样稿审阅时调整。

## 1. 背景和问题

- **安装包里没有插件**：安装版只带 Python 运行时、后端源码与前端页面、Qdrant（[package.mjs](../../../desktop/scripts/package.mjs)），不带 `integrations/`。桌面端与前端都没有安装插件的入口，四份插件帮助文档都从仓库 checkout 安装。
- **只装了安装版的新用户，要走开发者的步骤**：
  1. 克隆仓库，并切到与安装版相同的版本；插件与后端要同版本，但没有任何东西检查两者是否一致；
  2. 在设置页生成凭据，写进 harness 一侧的环境变量；
  3. 自己运行 harness 的插件命令；
  4. Claude Code 与 Codex 的 hook 用 PATH 上的 Python 运行，安装版自带的 Python 不在 PATH 上，所以还要另装一份 Python；
  5. Claude Code 插件的缺省地址是 `localhost:8000`（Docker 的端口），而安装版固定使用 8769，还要设置 `HIVEMEMORY_URL`。
- **没有装插件，也已经“接入”**：首次启动时，生效的访问登记从模板生成（[system_principals.example.yaml](../../../configs/system_principals.example.yaml)、[workspace_actors.example.yaml](../../../configs/workspace_actors.example.yaml)），模板已经为四个 harness 登记了来源与访问记录。没有安装任何插件，这四个来源也已经登记在册。
- **自由登记的来源没有作用对象**：设置页“接入与权限”的“登记外部 harness”接受任意来源标识，生成凭据后只提示写进哪个环境变量（[管理页面](../../frontend/management-views.md)第 3.3 节）。只有四个支持的 harness 有插件；凭据怎样交到 harness 一侧、插件怎样装上，都不在这个流程里。
- **升级**：安装版靠覆盖安装升级，而插件装在 checkout 或 harness 自己的副本里，不会随安装版一起升级。

## 2. 目标与非目标

### 2.1 目标

1. 安装包带上插件目录，插件与后端始终同版本。
2. 新的部署没有任何外部接入。
3. 设置页新增“插件”页，列出四个支持的 harness：Claude Code、Codex、pi、dsh。
   - 用户点“安装”后，系统依次完成：用 harness 自己的命令安装插件、登记来源与访问记录、生成凭据并交给 harness 一侧。
   - 找不到 harness 或命令失败时，页面显示要执行的命令。
   - “卸载”按相反的顺序撤销这些步骤。
4. 新用户不需要克隆仓库、设置环境变量，也不需要另装 Python。
5. 升级安装版之后，已安装的插件自动跟上新版本。
6. “接入与权限”页不再能自由登记外部来源。

### 2.2 非目标

- **四个 harness 以外的来源**：不提供通用的自定义接入。
- **替用户信任 Codex 的 hook**：Codex 要求用户在 Codex 中审阅并信任 hook，系统不绕过这一步。
- **代为安装命令行版 dsh**：命令行版按 profile 安装，系统不知道用户用哪个 profile。只代为安装 dsh 桌面端，命令行版只显示命令。
- **核对 harness 一侧的插件状态**：用户在 harness 中禁用或删除插件后，页面不会知道。页面状态以 HiveMemory 一侧的登记为准。
- **Docker 方式**：容器里调用不到宿主上的 harness，插件目录也不在镜像中。Docker 方式继续按帮助文档手动安装。
- **改动已有部署的生效登记**。
- **skill 的语言**：留给 D-12。它依赖本计划交付插件的方式，见[统一语言设置 Idea](../ideas/language-setting-unification.md)。

## 3. 当前状态和缺口

| 环节 | 现在 | 本计划之后 |
|:---|:---|:---|
| 安装包 | 不含插件目录 | 含 `integrations/`，与后端同版本 |
| 首次启动的登记 | 模板已登记四个 harness 的来源与访问记录 | 只有 HTTP 入口与 system 的登记，没有外部接入 |
| 安装插件 | 用户从 checkout 手工安装 | 插件页由系统代为安装，失败时给出命令 |
| 凭据交给 harness | 用户自己写环境变量 | 系统写入 Windows 用户环境变量 |
| hook 的 Python | PATH 上的 Python | 优先使用运行后端的那个解释器 |
| Claude Code 插件的缺省地址 | `localhost:8000` | `127.0.0.1:8769`，与其他三个插件一致 |
| 升级 | 插件不随安装版升级 | dsh 与 pi 直接读到新文件；Claude Code 与 Codex 在后端启动时自动重装 |
| “接入与权限”页 | 可以用任意来源标识登记外部 harness，并能轮换凭据、修改与删除 | 不能登记、轮换与删除，仍可启停和增减可声明的 Agent；安装、重装与卸载都在插件页 |

## 4. 目标设计

### 4.1 插件目录随安装包

- **打包**：`package.mjs` 把 `integrations/.claude-plugin/`（本地 marketplace）与 `integrations/hivememory/`（不含 `node/test` 与 `__pycache__`）复制到 `resources\backend\integrations\`。目录布局与仓库相同。
- **查找**：后端按源码位置找到插件目录，与现在查找 `configs/` 和 `frontend/dist` 的方式相同。开发方式下就是 checkout 里的目录。
- **升级**：覆盖安装时，插件目录随程序一起替换。

### 4.2 新的部署没有外部接入

- **模板**：两份模板删除四个 harness 的来源与访问记录，只保留 HTTP 入口与 `system` 两个来源，以及默认用户与 `system` 两条访问记录。
- **已有部署**：已经生成的生效登记不改。
- **测试**：连接器与插件测试中依赖模板登记的部分，改为依赖第 4.3 节的 harness 表。

### 4.3 支持的 harness

System 维护一张表，列出四个支持的 harness。每一项给出：
- 来源标识，与现在模板中的写法相同：`harness:claude-code`、`harness:codex`、`harness:pi`、`harness:dsh`；
- 插件的缺省 Agent：没有设置 `HIVEMEMORY_AGENT_ID` 时，插件声明的 Agent（`claude_code`、`codex`、`pi`、`dsh`）；
- 查找 harness 命令的方式；
- 安装与卸载用的插件命令；
- harness 一侧读取凭据的环境变量名；
- 安装之后的提示。

来源标识与现在相同，所以已有部署中的登记会被识别为“已安装”。

**来源不等于 Agent**：一个 harness 对应一个来源，但同一个 harness 可以以多个 Agent 运行。登记中的“可声明的 Agent”是一个集合，权限与可见性只跟随 Agent。安装插件管理的是来源；Agent 及其访问记录不随某个插件的安装和卸载而产生或消失。缺省 Agent 只用来让新用户装好就能用。

### 4.4 安装

管理接口 `POST /api/v1/plugins/{harness}/install` 由 System 的插件服务执行。接入登记属于 System；访问记录经 root API 建立，与 Import Bus 服务调用 root API 的方式相同。

1. **查找**：先确认插件目录存在，再找 harness 的命令：
   - `claude`、`codex`、`pi` 在 PATH 上查找；
   - dsh 查找桌面端自带的 `dsh.cmd`（`%LOCALAPPDATA%\Programs\DeepSeek Harness\resources\runtime\cli\bin\`）。

   找不到时什么都不登记，返回要执行的命令。用户照命令手动装好插件之后，可以在插件页点“已用命令装好，完成登记”：同一个接口以 `run_plugin_commands: false` 调用，跳过第 2 步，只做后面几步。
2. **安装插件**：运行 harness 的插件命令。命令有超时，不弹出控制台窗口。

   | harness | 命令 |
   |:---|:---|
   | Claude Code | `claude plugin marketplace add <integrations>`（已添加时改为更新），再 `claude plugin install hivememory-connector@hivememory-local` |
   | Codex | `codex plugin marketplace add <integrations>`，再 `codex plugin add hivememory-connector@hivememory-local` |
   | pi | `pi install <integrations\hivememory>` |
   | dsh 桌面端 | `dsh.cmd plugin --profile desktop add <integrations\hivememory>`；桌面端正在运行时 dsh 会拒绝，这时提示先从托盘退出 dsh |

   命令失败时同样不登记，返回失败原因、命令输出的最后几行和要执行的命令。
3. **登记**：
   - 登记来源时不声明 `credential_env`，只使用生成的凭据。后端一侧因此不读 harness 的环境变量，不会被它覆盖。
   - 来源可声明的 Agent，初始为缺省 Agent 一个；之后可以在“接入与权限”页增加。
   - 缺省 Agent 还没有访问记录时，为它建立一条，只开放检索、读取、写入与更新；已经有记录时不改动，用户调整过的权限保持不变。
   - 已经登记过时沿用原有登记：可声明的 Agent 与生成的凭据都保留，来自环境变量的旧登记不改。
4. **交付凭据**：把凭据写进 Windows 用户环境变量 `HIVEMEMORY_<HARNESS>_TOKEN`，并通知正在运行的程序环境变量已经改变。Claude Code 与 Codex 另外写入 `HIVEMEMORY_PYTHON`（第 4.6 节）。凭据来自环境变量的旧登记不写，用户原来的设置保持不变。
5. **返回下一步**：完全退出并重新打开 harness；Codex 还要在 Codex 中信任五个 hook。

先装插件、后登记，是为了在命令失败时不留下没有作用对象的登记。

### 4.5 卸载

管理接口 `POST /api/v1/plugins/{harness}/uninstall`：
1. 运行 harness 的卸载命令：
   - Claude Code：`claude plugin uninstall`；
   - Codex：`codex plugin remove`；
   - pi：`pi remove`；
   - dsh：`dsh plugin remove`。
2. 删除来源与生成的凭据。
3. 删除第 4.4 节写入的用户环境变量。

卸载命令失败或找不到 harness 时，第 2、3 步照常执行，用户要的就是断开连接。页面另外显示手动卸载插件的命令。

Agent 的访问记录不删除：它属于 Agent，不属于某个 harness，可能还有别的来源在声明这个 Agent。来源删除之后，经这个 harness 的请求以 `unknown_principal` 拒绝，连接就断了；重新安装时沿用原来的记录。

### 4.6 hook 的 Python

`HIVEMEMORY_PYTHON` 指向运行后端的那个解释器所在目录中的 `python.exe`：安装版是自带的 Python，开发方式是 `.venv` 中的 Python。连接器只用标准库，两者都能运行。
- **Claude Code**：hook 在 Git Bash 中执行，命令改为优先使用 `HIVEMEMORY_PYTHON`，没有设置时再找 `python3` 或 `python`。
- **Codex**：hook 命令由会话所用的 shell 执行，没有会话 shell 时回落到 cmd。PowerShell 和 cmd 引用环境变量的写法不同，所以 hook 命令改为 `cmd /c "${PLUGIN_ROOT}/hivememory_hook.cmd" codex hook <事件>`，由插件中的启动脚本选择解释器：设置了 `HIVEMEMORY_PYTHON` 时用它，否则用 PATH 上的 `python`。实施时在本机核对过：路径带空格时，经 PowerShell 与 cmd 调用都能运行，标准输入照常交给连接器。启动脚本只用 ASCII（cmd 按 OEM 代码页读批处理文件），以 CRLF 保存（`.gitattributes` 中的 `*.cmd`）。

### 4.7 插件的缺省地址

Claude Code 插件的 `.mcp.json` 与 Python 连接器，缺省地址由 `localhost:8000` 改为 `127.0.0.1:8769`。改后与 Codex、Node 连接器、安装版和开发脚本一致，新用户不需要设置 `HIVEMEMORY_URL`。Docker 方式（8000）改为在帮助文档中写明设置 `HIVEMEMORY_URL`。

### 4.8 升级后自动更新

- **链接安装的 dsh 与 pi**：覆盖安装后直接读到新的插件文件，重新打开 harness 即可。
- **复制安装的 Claude Code 与 Codex**：
  - 后端以桌面模式启动时，检查这两个已安装的 harness。如果记录的安装版本与当前版本不同，就在后台重新执行第 4.4 节第 2 步，然后记下新版本。
  - 版本记录在数据根的一个小文件中。失败时写日志，插件页把这个 harness 标为“需要更新”，并给出命令。
  - Codex 更新后 hook 内容变化，需要在 Codex 中重新信任，插件页给出提示。
- **开发方式不自动更新**：checkout 的更新节奏由开发者掌握，仍按帮助文档手动更新。

### 4.9 页面

- **插件页**：设置页“管理”一组新增“插件”，排在最前。四个 harness 各占一张卡片：
  - **状态**：未安装、已安装、找不到 harness、需要更新，或插件目录不可用（例如 Docker）；
  - **操作**：安装、重新安装、卸载（卸载要二次确认）；
  - **安装中**：显示进行中，结束后显示结果，不实时显示命令输出；
  - **成功后**：显示下一步提示；
  - **失败时**：显示原因、命令输出的最后几行，以及可以复制的命令。
- **“接入与权限”页**：
  - 去掉“登记 harness”和“轮换凭据”：凭据要交到 harness 一侧才有意义，交付现在由插件页的安装与重装完成；
  - 去掉外部来源的删除：只删登记而插件还装着，会留下没有作用对象的连接；删除改为在插件页卸载；
  - 修改保留，包括启停与可声明的 Agent：用户在这里让同一个 harness 以更多 Agent 运行；
  - 列表与 Agent 操作权限一节保留。
- **样稿先行**：先在运行页样稿画布上画出插件页与“接入与权限”页的改动，owner 定稿后再写前端代码。样稿已经画在画布的“系统设置 · 插件”一行，共三块画板。

### 4.10 实施顺序

1. 核对：本机 Codex 执行 hook 的 shell；四个 harness 的插件命令能否非交互运行，以及已经安装时的行为；
2. 样稿：插件页与“接入与权限”页，交 owner 审阅；
3. 后端：harness 表、插件服务与管理接口、两份模板、用户环境变量的读写；
4. 插件：hook 的 Python、缺省地址；
5. 打包：安装包带上插件目录；升级后的自动更新；
6. 前端：插件页与“接入与权限”页；
7. 收尾：帮助文档改为以插件页安装为主、手动命令为辅；更新当前文档，归档本计划。

## 5. 受影响的系统、契约与数据

| 位置 | 变化 |
|:---|:---|
| 访问登记模板 | 删除四个 harness 的来源与访问记录 |
| System | 新增支持的 harness 表与插件服务；运行外部命令、读写 Windows 用户环境变量 |
| server | 新增 `/api/v1/plugins` 的状态查询、安装与卸载接口，由管理凭据保护；`/api/v1/access/principals` 删除登记、轮换与删除三个接口（前端不再使用，插件服务直接调用注册表），保留列表与修改 |
| 插件目录 | Claude Code 与 Codex 的 hook 优先使用 `HIVEMEMORY_PYTHON`；Claude Code 的缺省地址改为 8769 |
| 桌面客户端 | 安装包带上 `integrations/`；桌面模式启动时检查并更新复制安装的插件 |
| 前端 | 新增插件页；“接入与权限”页去掉外部来源的登记、轮换与删除 |
| 数据 | 数据根中新增插件安装版本的记录文件；Windows 用户环境变量中写入凭据与 Python 路径 |

## 6. 兼容与迁移

- **已有部署**：生效登记不变。现有的四个 harness 登记与 harness 表中的来源标识相同，插件页显示为已安装。
  - 凭据来自环境变量（`.env` 或 `setx`）的旧登记，重新安装时不改动凭据和环境变量。
  - 这类部署之前手工安装的插件照常工作。
- **新的部署**：从模板生成的登记里没有外部接入，要在插件页安装之后才能连接。
- **缺省地址**：用 Docker 方式运行、又依赖 Claude Code 插件缺省地址 8000 的部署，需要设置 `HIVEMEMORY_URL`。
- **帮助文档**：四份插件帮助改为以插件页安装为主，手动命令保留，供 Docker、命令行版 dsh 和安装失败时使用。

## 7. 测试和验收标准

需要守住的不变量与归属层：

| 不变量 | 归属层 |
|:---|:---|
| 命令成功后才登记来源与访问记录，并写入用户环境变量；找不到 harness 或命令失败时什么都不登记，返回命令 | 插件服务的单元测试，以替身代替命令执行与环境变量写入 |
| 已登记时沿用原有登记，凭据来自环境变量的旧登记不改；缺省 Agent 已有访问记录时不改动 | 同上 |
| 卸载时即使命令失败，也删除来源、生成的凭据与写入的环境变量；Agent 的访问记录保留 | 同上 |
| 由插件服务登记的来源只用生成的凭据，后端不读 harness 的环境变量 | `tests/unit/system/access/test_registry.py` |
| 从模板生成的登记没有外部接入 | `tests/unit/system/test_access_registration_files.py` |
| 管理接口的状态、安装与卸载响应，以及没有管理凭据时被拒绝 | `tests/unit/server/routers/` 中插件接口的测试 |
| Claude Code 连接器的缺省地址为 8769 | `tests/unit/connectors/` |
| 插件页的状态与失败显示 | `frontend/tests/` 中插件接口的测试 |

harness 命令与 Windows 用户环境变量依赖本机环境，以手动方式验收：

- 用全新的数据根启动安装版：
  - “接入与权限”中没有外部接入；
  - 在插件页依次安装四个 harness，重新打开 harness 后，四个记忆工具都可用，对话能回流；
  - Codex 信任 hook 之后同样可用。
- 卸载一个 harness 之后，它的调用以 `unknown_principal` 被拒绝，用户环境变量已删除。
- 在没有安装 Python 的情况下，Claude Code 的 hook 正常运行。
- 覆盖安装更高版本的安装版之后，Claude Code 与 Codex 的插件自动更新，dsh 与 pi 重新打开后读到新文件。
- 已有部署（owner 的 checkout）启动后，插件页显示四个 harness 都已安装，原有连接不受影响。

验证命令：

```bash
pytest tests/unit/system tests/unit/server tests/unit/connectors -q
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85
ruff check . && black --check .
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
cd frontend && npm run lint && npm test && npm run build
cd integrations/hivememory && npm test
cd desktop && npm run package
```

## 8. 风险与待决问题

- **Codex 执行 hook 的 shell**：启动脚本的写法已在本机经 PowerShell 与 cmd 核对（第 4.6 节）；Codex 实际执行 hook 的情况在手动验收中确认。
- **用户环境变量的作用范围**：写入之后，这个用户此后启动的所有程序都能读到。已经打开的终端和 harness 读不到，必须完全退出再打开，页面会给出提示。凭据只防误用，与现在手工 `setx` 的做法相同。
- **harness 的命令会变**：四个 harness 都没有承诺插件命令稳定。harness 升级后命令失败时，页面给出命令，交由用户手动处理。
- **后端运行外部命令**：只运行 harness 表中固定的命令，参数只有插件目录的路径；只能经管理接口触发，受管理凭据保护。
- **dsh 桌面端正在运行**：安装与卸载会被拒绝，页面提示先退出 dsh。
- **新用户怎样找到入口**：本计划只在设置页提供入口。对话页或首次启动时是否也给出提示，在样稿审阅时决定。
- **待决**：无阻塞问题。

## 9. 完成时必须更新的当前文档

- [桌面客户端](../../frontend/desktop-client.md)：安装包内容、升级后自动更新插件；
- [管理页面](../../frontend/management-views.md)：插件页，以及“接入与权限”页的变化；
- [公开路由与事件](../../contracts/routes-and-events.md)第 6 节：插件管理接口；
- [System 配置](../../system/configuration.md)与[认证与访问登记](../../workspace/authentication.md)：模板中不再有外部接入，插件服务登记的来源只用生成的凭据；
- [使用桌面客户端](../../help/desktop-client.md)、[安装与启动](../../help/setup.md)第 9 节，以及 [Claude Code](../../help/claude-code-plugin.md)、[Codex](../../help/codex-plugin.md)、[DeepSeek Harness](../../help/deepseek-harness-plugin.md)、[pi](../../help/pi-plugin.md) 四份插件帮助；
- [ROADMAP](../../ROADMAP.md) 的对应行；
- 本计划移入 `archive/plans/`，从 [Plans](../../plans/README.md) 中删除，在[归档索引](./README.md)中登记。

## 10. 实施记录

2026-11-12 在分支 `feat/desktop-plugin-installation` 实施第 4 节，以 PR #218 合并。第 4.4 节的手动登记与第 4.6 节 Codex 的启动脚本在实施时定下，已写回正文。与计划不同，或计划没有写到的地方：

- **已经装过时的命令**：
  - Claude Code 与 Codex 的本地 marketplace 指向别处（例如 checkout）时，先删除，再指向当前的插件目录。已经指向这里时，Claude Code 执行 `marketplace update`，插件已安装则执行 `plugin update`；Codex 再执行一次 `plugin add`，把新版本复制进它的缓存。
  - pi 中指向别处的 HiveMemory 插件先移除，否则同名的记忆工具会注册两次。
  - dsh 按路径添加，指向别处的旧链接随之替换。
  - Claude Code 卸载时带 `--keep-data`，保留连接器的导入进度，重新安装后不必从头补发。
- **`HIVEMEMORY_PYTHON` 的删除**：卸载 Claude Code 或 Codex 时，另一个仍然安装着就保留这个变量。
- **不在 Windows 上**：没有用户环境变量可写，凭据交不到 harness 一侧，插件页按“插件目录不可用”处理，不安装也不登记。
- **安装记录**：新增配置项 `system.plugin_install_record`，缺省为数据根下的 `.hivememory/plugin_installs.json`。
- **新用户的入口**（第 8 节）：样稿审阅时没有提出其他入口，只在设置页提供。设置页“插件”一类旁只显示已安装的数量，按 owner 审阅样稿时的要求。
- **测试的归属**：
  - “插件服务登记的来源只用生成的凭据”由插件服务的单元测试断言；注册表没有改动，`test_registry.py` 不变。
  - 已经装过时的命令序列由 `tests/unit/system/plugins/test_harnesses.py` 守护。
  - Claude Code 连接器的缺省地址只是常量，没有单独加测试。
- **开发方式与安装版的冲突**：实施期间发现，安装版以 checkout 为数据根运行时占着 8769，开发方式的 Vite 连上它一律 401。两者分开运行的方向另起 Idea，见[日常实例与开发实例分开](../ideas/daily-and-dev-instance-separation.md)。

实测（2026-11-12）：

- 后端门槛 3381 项通过、9 项跳过，覆盖率 94.71%；mypy、ruff 与 black 通过；前端 lint、83 项测试与构建通过；插件的 Node 测试 72 项通过。
- `hivememory_hook.cmd`：路径带空格时，经 PowerShell 与 cmd 调用都能运行，标准输入照常交给连接器。
- 免安装目录版：`resources\backend\integrations` 带上了本地 marketplace 与插件目录，不含 `node/test` 与 `__pycache__`；后端以 `pythonw` 运行时，交给 hook 的仍是同一目录中的 `python.exe`。
- 完整安装包：`npm run package` 生成 `HiveMemory Setup 0.7.3.exe`（约 254 MB），其中的插件目录与免安装目录版相同。
- 插件页与“接入与权限”页在浏览器中以替身接口核对：未安装、安装成功与 dsh 安装失败三种状态，以及“接入与权限”页，都与样稿一致。

没有在本机实测的是：在插件页真实安装四个 harness、Codex 信任 hook 之后的运行、没有 Python 的新用户环境，以及覆盖安装之后的自动更新。它们属于第 7 节的手动验收，由 owner 在日常使用中进行。
