---
title: 日常实例与开发实例分开
status: active
owner: project
target: 版本之外的工作（开发与验收环境，服务 v1.0 的桌面客户端）
scope: run-the-installed-daily-instance-and-the-checkout-dev-instance-side-by-side-with-separate-ports-data-and-harness-bindings
updates:
  - docs/help/setup.md
  - docs/help/desktop-client.md
  - docs/frontend/desktop-client.md
  - docs/frontend/management-views.md
  - docs/system/configuration.md
code_paths:
  - scripts/hivememory-dev.ps1
  - scripts/hivememory-dev.env.example
  - frontend/vite.config.ts
  - desktop/src/layout.js
  - desktop/src/services.js
  - desktop/src/main.js
  - desktop/package.json
  - src/hivememory/server/__main__.py
  - src/hivememory/config/app.py
  - src/hivememory/system/assembler.py
  - frontend/src/utils/pluginStatus.ts
  - frontend/src/components/settings/PluginSettings.tsx
related_docs:
  - docs/ideas/daily-and-dev-instance-separation.md
  - docs/archive/plans/desktop-plugin-installation.md
  - docs/frontend/desktop-client.md
  - docs/help/setup.md
---

# 日常实例与开发实例分开

> **状态**：Active（2026-11-12 起草并开始实施；同日代码与文档完成，第 4.6 节的迁移完成：日常实例在安装版的数据根上运行，四个 harness 已在安装版的插件页重新安装，开发实例使用 8779 这组端口；等待合并后归档）。依据[日常实例与开发实例分开 Idea](../ideas/daily-and-dev-instance-separation.md)。owner 的决定：
>
> - 日常使用与开发分成两个互不相干的 HiveMemory 实例，每样资源只归其中一个（2026-11-12）；
> - [桌面端安装插件](../archive/plans/desktop-plugin-installation.md)合并之后，先做这一项（2026-11-12）；
> - Windows Sandbox 中的新用户验收改为可选项，不在本计划交付（2026-11-12）；
> - 开发实例以分开时日常 Qdrant 数据的一份副本作冷启动，方便开发（2026-11-12）。
>
> 其余内容是候选设计，不是 owner 决定，可在评审时调整。与 Idea 第 3 节不同的地方，在第 4 节逐条注明。第 4.6 节迁移日常数据的时间，动手前与 owner 约定。

## 1. 背景和问题

一台电脑上只有一套 HiveMemory 的资源，开发方式与桌面安装版轮流使用它们（详见 Idea 第 1 节）：

- **数据根**：安装版选了 checkout 作数据根，开发脚本的 Qdrant 存储也默认在 checkout，两者读写同一个记忆库。
- **端口**：Windows 开发脚本、Vite 代理、用 checkout 运行的桌面端与安装版都固定使用 8769、6333、6334，不能同时运行。开发时日常记忆不回流，正在开发 HiveMemory 的会话本身也在其中。
- **管理凭据**：安装版的凭据只在内存中，Vite 读 `configs/management.token`，连到安装版的后端一律 401。
- **harness 配置**：每个 Windows 用户只有一份。在哪个实例的插件页安装，harness 就指向那个实例的插件目录，用户环境变量中的凭据也换成那个实例的凭据。
- **新用户验收**：本机已经装了 Python 和 harness，也已经有连接，验证不了“新用户装完之后什么都没有”。

## 2. 目标与非目标

### 2.1 目标

1. 安装版与开发方式可以同时运行，两者的端口不冲突；开发时日常记忆照常回流。
2. 日常数据只在安装版的数据根中，checkout 只放开发数据。
3. 只有安装版代为安装插件；开发方式运行的后端不改动 harness 的配置和用户环境变量。
4. 开发时能让单个 harness 连到开发实例，不动日常的绑定。

### 2.2 非目标

- **WSL/Linux 脚本**（`hivememory-dev.sh`）不改。它已经从 `dev.env` 读端口，但没有把 Qdrant 端口交给后端（第 8 节）；不在 Windows 上时插件服务本来就不可用。
- **Docker 方式**不变。
- **共享模型缓存**：两个实例各有一份模型缓存，迁移时复制过去。开发实例想省下这 1.6 GB 时，可以在个人覆盖文件里把两个 `cache_dir` 指到日常实例的缓存，本计划不做。
- **迁移工具**：迁移只做一次，按第 4.6 节的步骤手动进行。
- **Windows Sandbox 中的新用户验收**：owner 决定改为可选项，本计划不实现，做法记在第 4.7 节。
- **插件的缺省地址**：仍是 8769，日常实例占用这组端口。

## 3. 当前状态和缺口

| 环节 | 现在 | 本计划之后 |
|:---|:---|:---|
| Windows 开发脚本的端口 | 固定 8769、6333、6334 | 从 `dev.env` 读，缺省不变；Qdrant 端口交给后端 |
| Vite 代理 | 固定转发到 8769 | 从 `dev.env` 读后端端口 |
| 用 checkout 运行的桌面端 | 固定 8769、6333、6334；端口上已有后端时附着 | 与开发脚本用同一组端口 |
| `python -m hivememory.server` 的端口 | 固定 8769 | 可由 `HIVEMEMORY_BACKEND_PORT` 指定，缺省 8769 |
| 插件代为安装 | 在 Windows 上、插件目录存在时就可用 | 只有安装版可用 |
| 日常数据 | checkout（安装版以它为数据根） | `%LOCALAPPDATA%\HiveMemory` |
| Windows 开发方式的帮助 | [安装与启动](../help/setup.md)只写了 WSL/Linux 脚本 | 增加 Windows 开发脚本与两个实例的说明 |

## 4. 目标设计

### 4.1 资源归属

| 资源 | 日常实例 | 开发实例 |
|:---|:---|:---|
| 代码 | 桌面安装版（打包时的快照） | checkout |
| 端口 | 8769、6333、6334（插件缺省连这组） | `dev.env` 中的另一组，例如 8779、6343、6344 |
| 数据根 | `%LOCALAPPDATA%\HiveMemory` | checkout（开发数据，可以随时清空） |
| 管理凭据 | 只在内存中，只有桌面窗口能管理 | `configs/management.token`，Vite 照常使用 |
| 插件代为安装 | 可用，harness 只绑定这里 | 不可用 |
| 外部接入的登记 | 插件页安装时登记 | checkout 的个人登记文件（第 4.4 节） |

### 4.2 开发实例的端口

- **唯一来源**：`.hivememory/dev.env` 中的三个键 `HIVEMEMORY_BACKEND_PORT`、`QDRANT_HTTP_PORT` 与 `QDRANT_GRPC_PORT`，键名与 WSL 脚本相同。
  - 缺省仍是 8769、6333、6334：没有设置这三个键的开发者，以及用 checkout 的 `serve` 日常运行的部署，都照旧。
  - 要与安装版同时运行时，在 `dev.env` 中写另一组。样例文件给出 8779、6343、6344。
  - 与 Idea 不同：Idea 写的是缺省就用开发的那一组。缺省一变，所有用开发脚本的部署都会离开插件连的 8769。
- **Windows 开发脚本**：
  - 按这三个键启动 Qdrant 与后端，健康检查与 `status` 用同一组端口；
  - 启动后端时设置 `HIVEMEMORY__PATCHOULI__STORAGE__PORT` 与 `HIVEMEMORY__PATCHOULI__STORAGE__GRPC_PORT`，把 Qdrant 端口交给后端。环境变量优先于 `config.yaml` 与个人覆盖文件。
- **Vite**：从 `../.hivememory/dev.env` 读 `HIVEMEMORY_BACKEND_PORT`，缺省 8769。前端端口仍是 5173，安装版不用它。
- **用 checkout 运行的桌面端**（`desktop` 目录的 `npm start`）：
  - `checkoutLayout` 从 `dev.env` 取三个端口，`installedLayout` 固定为缺省的一组；
  - `services.js` 的健康检查、端口检查、Qdrant 与后端的启动，以及 `main.js` 的页面地址与 cookie 地址，都改用布局给出的端口；
  - 自己启动后端时，以 `HIVEMEMORY_BACKEND_PORT` 和上面两个 Qdrant 端口的环境变量交给后端；
  - 端口上已有后端时照旧附着，附着的就是开发脚本启动的那个。
- **`python -m hivememory.server`**：读 `HIVEMEMORY_BACKEND_PORT`，缺省 8769，与已有的 `HIVEMEMORY_BACKEND_HOST` 写法相同。用 checkout 运行的桌面端以这种方式启动后端。
- **管理凭据**：保持文件方式，Vite 读 checkout 的 `configs/management.token`。

### 4.3 只有安装版代为安装插件

- **开关**：新增配置项 `system.plugin_installation_enabled`，缺省 `false`。安装版启动后端时，以环境变量 `HIVEMEMORY__SYSTEM__PLUGIN_INSTALLATION_ENABLED=true` 打开，与它设置日志路径的方式相同。
- **关闭时**：组合根不给插件服务用户环境变量的写入方，插件服务按“不可用”处理。插件服务本身不改：
  - 插件页显示“不能代为安装”与手动命令；
  - 安装与卸载接口返回 `unavailable`；
  - 桌面模式启动时也不自动更新插件。
- **理由**：开发实例代为安装时，harness 被改指到 checkout 的插件目录，用户环境变量中日常的凭据被开发实例的凭据覆盖，而插件仍然连 8769，日常的连接因此断开。
- **与 Idea 不同**：Idea 写的是在开发数据根的覆盖文件里关掉。改为只有安装版打开，是因为开发脚本、用 checkout 运行的桌面端、Docker 与 WSL 都不该代为安装，只打开一处比关掉四处简单。
- **插件页的文字**：“不能代为安装”的说明改为同时涵盖三种情况：不是安装版（开发方式、Docker）、不在 Windows 上、插件目录不可用。只改文字，不新增状态，不需要样稿，也不为文字加测试。
- **影响**：以 checkout 的 `serve` 作日常使用的 Windows 部署，插件页不再代为安装，改按帮助文档手动安装，或者改用安装版。

### 4.4 开发实例的接入登记

开发实例不代为安装，“接入与权限”页也不能登记外部来源。经 harness 测试开发实例时，来源登记写在 checkout 的个人登记文件里：

- `configs/system_principals.yaml` 以 `credential_env` 声明凭据所在的环境变量；
- 凭据的值写在 checkout 的 `.env`。

owner 的 checkout 已经有四个 harness 的这种登记，迁移后原样保留，不需要新做。

### 4.5 让单个 harness 连开发实例

- **测后端的改动**：
  - 用日常已经安装的插件，只在这个 harness 进程的环境里改 `HIVEMEMORY_URL` 与 `HIVEMEMORY_<HARNESS>_TOKEN`，例如从设置了这两个变量的终端启动 harness；
  - 不写用户环境变量。
- **测插件代码本身**：加载 checkout 中的插件，用 harness 自己的隔离方式。候选做法如下：
  - Claude Code：会话级加载插件目录（`--plugin-dir`）；
  - Codex：另设 `CODEX_HOME`；
  - pi：项目本地安装（`pi install -l`）或会话级加载；
  - dsh：独立的 profile。

  实施第 1 步逐个核对：能否与日常安装的同名插件并存，是否需要先在这个会话中停用日常的那一份。核对结果写进[安装与启动](../help/setup.md)。核对不通的 harness 写明临时做法，做完要在安装版的插件页重新安装。

### 4.6 日常实例与一次性迁移

- **日常实例**：只有桌面安装版，数据根是缺省的 `%LOCALAPPDATA%\HiveMemory`。
- **新代码进入日常**：打包，再覆盖安装。Claude Code 与 Codex 的插件随后自动更新，这一点已由桌面端安装插件实现。此前“checkout 就是日常部署，重启即上线”的做法到此结束。
- **迁移步骤**：与 owner 约定时间后进行，日常实例停一次。
  1. 打出带本计划改动的安装包并覆盖安装。安装版要带上第 4.3 节的开关。
  2. 从托盘退出 HiveMemory，确认 8769、6333、6334 都已空闲。
  3. 把 checkout 中的日常数据复制到 `%LOCALAPPDATA%\HiveMemory`，目录布局不变：
     - `.env`；
     - `configs` 中的个人文件：`models.yaml`、`providers.secrets.yaml`、`system_principals.yaml`、`workspace_actors.yaml`，以及存在时的 `config.local.yaml`。不复制随程序发布的 `config.yaml` 与 `management.token`；
     - `.hivememory` 中的 `qdrant`、`artifacts`、`pending_generation.json` 与 `preset_agent_profiles_seeded`。`evidence` 是调研笔记，不是运行数据，留在 checkout；
     - `data` 中的 `archived` 与 `model_cache`。
  4. 启动安装版，从托盘的“更改数据目录”指到新的数据根。
  5. 在插件页把四个 harness 各重新安装一次，让它们指向安装目录。这些旧登记的凭据来自环境变量，重新安装时不改动凭据和用户环境变量。
  6. 在 `dev.env` 写上开发的那一组端口。此后 checkout 只作开发实例。
- **开发实例的冷启动**（owner，2026-11-12）：开发实例的 Qdrant 用分开时日常记忆库的一份副本作冷启动。
  - 第 3 步在 Qdrant 停止时复制，checkout 中的 `.hivememory/qdrant` 原样留给开发实例，其余数据也留在 checkout；
  - `pending_generation.json` 复制给日常实例之后从 checkout 删除：保存的生成工作只在日常实例中继续，开发实例不重复调用模型；
  - 开发数据可以随时清空，需要时再从日常实例复制一份（复制时先停下日常实例的 Qdrant）。

### 4.7 可选：在 Windows Sandbox 中验收

owner 2026-11-12 决定改为可选项，本计划不实现。需要验收新用户流程时，按下面的做法另行实施。

- **前提**：本机还没有启用 Windows Sandbox（系统中没有 `WindowsSandbox.exe`）。启用要管理员权限并重启，由 owner 在“启用或关闭 Windows 功能”中打开。
- **启动方式**：`desktop` 新增 `npm run sandbox`。
  - 它在 `desktop/dist` 写一份 `.wsb`：把 `desktop/dist` 只读映射到沙盒中，内存 8 GB，然后启动 Sandbox。
  - 用脚本生成，是因为 `.wsb` 中的主机路径要写绝对路径。实施第 1 步核对：如果它接受相对路径，就改为仓库中的固定文件。
- **验收内容**：
  - 安装；
  - 首次启动选择数据根，下载模型；
  - “接入与权限”中没有外部接入；
  - 插件页四个 harness 都显示“找不到”，并给出命令。
- **在沙盒中安装 harness**：沙盒中装上 harness 的命令行之后，能否代为安装插件，取决于 harness 安装插件前是否要先登录，实施时核对后再定。
- **模型**：沙盒每次都要重新下载约 1.6 GB。可选做法是映射日常的模型缓存。

### 4.8 实施顺序

1. **核对**：第 4.5 节各 harness 的隔离方式。
2. **后端**：
   - `HIVEMEMORY_BACKEND_PORT`；
   - `system.plugin_installation_enabled` 与组合根中的判断。
3. **端口与开关的使用方**：
   - 开发脚本、样例 `dev.env`、Vite 与用 checkout 运行的桌面端读同一组端口；
   - 安装版打开插件代为安装。
4. **插件页**：改“不能代为安装”的说明。
5. **文档**：更新帮助与当前文档。
6. **迁移**：与 owner 约定时间后迁移日常数据（第 4.6 节），在本机验证两个实例同时运行。
7. **收尾**：归档本计划与 Idea。

## 5. 受影响的系统、契约与数据

| 位置 | 变化 |
|:---|:---|
| server | `python -m hivememory.server` 读 `HIVEMEMORY_BACKEND_PORT` |
| System 配置与组合根 | 新增 `system.plugin_installation_enabled`；关闭时插件服务不可用 |
| 开发脚本 | Windows 脚本从 `dev.env` 读三个端口，并把 Qdrant 端口交给后端；样例文件给出开发的一组端口 |
| 前端 | Vite 从 `dev.env` 读后端端口；插件页“不能代为安装”的说明 |
| 桌面客户端 | 用 checkout 运行时读 `dev.env` 的端口；安装版打开插件代为安装 |
| 数据 | owner 的日常数据从 checkout 迁到 `%LOCALAPPDATA%\HiveMemory` |
| 公共契约 | 路由、事件与插件接口不变；插件服务关闭时返回已有的 `unavailable` |

## 6. 兼容与迁移

- **没有在 `dev.env` 设置端口的开发者**：端口不变。
- **安装版**：升级后照常代为安装插件。
- **以 checkout 运行、靠插件页代为安装的 Windows 部署**：插件页不再代为安装，改为手动安装或改用安装版。
- **owner 的部署**：按第 4.6 节迁移一次。迁移前，安装版仍以 checkout 为数据根，现状不变。

## 7. 测试和验收标准

需要守住的不变量与归属层：

| 不变量 | 归属层 |
|:---|:---|
| 开关关闭时插件服务不可用：状态为不可用，安装与卸载都不运行命令、不登记，启动时也不更新 | 组合根的单元测试：开关映射到插件服务是否可用。服务自身的“不可用”已有测试 |
| `python -m hivememory.server` 按 `HIVEMEMORY_BACKEND_PORT` 监听，缺省 8769 | `tests/unit/server/test_main.py` 中已有的启动参数测试 |

开发脚本、Vite 与 Electron 没有自动测试，以手动方式验收：

- 安装版在 8769 运行时，用开发脚本以 8779、6343、6344 启动开发实例，两者都就绪；Vite 连开发实例的管理接口不再 401。
- 开发实例的插件页显示“不能代为安装”，安装版的插件页照常。
- 用 checkout 运行的桌面端附着到开发实例，不碰安装版。
- 迁移之后：
  - 安装版的数据根是 `%LOCALAPPDATA%\HiveMemory`，记忆数量与迁移前一致；
  - 四个 harness 重新安装后都连到安装版，对话能回流。
- 用第 4.5 节的方式让一个 harness 连开发实例，日常的连接不受影响。

验证命令：

```bash
pytest tests/unit/system tests/unit/server -q
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85
ruff check . && black --check .
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
cd frontend && npm run lint && npm test && npm run build
cd desktop && npm run package
```

## 8. 风险与待决问题

- **内存**：两个实例同时运行时，推理模型与 Qdrant 各占一份；开发实例靠已有的模型空闲卸载减轻。
- **迁移**：迁移期间日常实例停一次。迁移后 checkout 中的数据与日常数据分开演变，开发实例的检索效果不再代表日常。
- **harness 的隔离方式可能不成立**：核对不通的 harness 只能临时改绑，测完要在安装版的插件页重新安装。
- **WSL/Linux 脚本**：它从 `dev.env` 读 Qdrant 端口，但没有交给后端，改了端口后端仍连 6333。这是实施前就有的问题，本计划不处理，只记在这里。
- **待决**：无阻塞问题。迁移时间待与 owner 约定。

## 9. 完成时必须更新的当前文档

- [安装与启动](../help/setup.md)：
  - Windows 开发脚本；
  - 日常实例与开发实例，以及 `dev.env` 的端口；
  - 第 5 节的端口表；
  - 让单个 harness 连开发实例。
- [使用桌面客户端](../help/desktop-client.md)：日常数据在安装版的数据根中，新代码经打包与覆盖安装进入日常。
- [桌面客户端](../frontend/desktop-client.md)：
  - 第 1、2 节用 checkout 运行时的端口；
  - 第 3 节安装版打开插件代为安装。
- [管理页面](../frontend/management-views.md)第 3.5 节：插件代为安装只在安装版中可用。
- [System 配置](../system/configuration.md)：`system.plugin_installation_enabled`。
- [ROADMAP](../ROADMAP.md) 的对应行。
- 本计划与 Idea 移入 `archive/`，从 [Plans](./README.md) 与 [Ideas](../ideas/README.md) 中删除，在归档索引中登记。
