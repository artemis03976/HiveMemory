---
title: Electron 桌面客户端
status: archived
archived_at: 2026-11-10
implemented_by: PR #TBD
superseded_by: docs/frontend/desktop-client.md
owner: frontend
target: v1.0 范围（候选清单 D-01，含 A-02、A-05）
scope: installable-windows-desktop-client-with-tray-resident-backend-first-run-upgrade-and-data-deletion
updates:
  - docs/frontend/desktop-client.md
  - docs/frontend/README.md
  - docs/help/desktop-client.md
  - docs/help/setup.md
  - docs/help/configuration.md
  - docs/system/configuration.md
  - docs/governance/security/identity-and-execution-safety.md
  - docs/workspace/authentication.md
  - docs/PROJECT.md
code_paths:
  - desktop/
  - src/hivememory/server/__main__.py
  - src/hivememory/server/management_credential.py
  - src/hivememory/config/files.py
  - src/hivememory/system/
  - src/hivememory/patchouli/
  - scripts/inject_profiles.py
  - configs/system_principals.example.yaml
related_docs:
  - docs/archive/plans/shutdown-pending-generation-persistence.md
  - docs/ideas/identity-and-access-model.md
  - docs/frontend/state-and-transports.md
---

# Electron 桌面客户端

> **状态**：Archived（2026-11-10，已实施，实施记录见第 10 节）。当前事实见[桌面客户端](../../frontend/desktop-client.md)与[使用桌面客户端](../../help/desktop-client.md)。以下为实施时的计划原文，链接已按归档位置改正。owner 的决定见第 1.1 节，第 4 节注明“候选”的做法已由 owner 按候选定下。第 7 节的手动验收由 owner 在日常使用中进行，不阻塞归档。对应 [ROADMAP](../../ROADMAP.md) 第 5.1 节。

## 1. 背景和问题

v1.0 的目标之一是把 HiveMemory 做成桌面客户端（ROADMAP 第 5.1 节），验收是“可安装、启动、使用、关闭、重启和升级，用户知识资产持续可读；卸载/删除数据必须有明确语义”。

现在的运行方式是开发脚本：用户要自己装 Python 环境、下载 Qdrant、写 `.env` 与配置文件，再用 `ps1 serve` 启动。管理凭据写在 `configs/management.token` 里，只能防误用：harness 中的 agent 与后端以同一个系统用户运行，读得到这个文件（候选清单 A-02）。

2026-11-09 至 11-10 做了两版原型，验证了生命周期、凭据、数据目录与安装包的做法（第 3 节），也暴露出正式发布前还要补的几件事：

- 原型在端口已被占用时直接附着到已有的后端或 Qdrant。附着到开发用的 Qdrant 时，安装版换了数据目录也会读写 checkout 的记忆库。
- 后端启动失败时，错误框只有退出码。用户自己升级时最常见的失败是配置文件中残留了新版本删掉的键，这在错误框里看不出来。
- 模型下载失败时，`/health/ready` 只报告“预热中”，启动页要等 60 秒没有进展才打开一个用不了的界面。
- 预设 Agent（coder_doll、reviewer_doll）只能用开发脚本 `scripts/inject_profiles.py` 注入，安装版的新数据目录里没有它们（候选清单 A-05）。
- 卸载保留数据目录，但应用里没有删除数据的入口。

### 1.1 owner 决定

1. **平台与技术栈**：Electron，先做 Windows；后端在用户自己的电脑上运行。
2. **托盘常驻**（2026-11-09）：主进程启动并持有后端与 Qdrant；关窗只隐藏到托盘，从托盘退出时正常关闭；可以选择登录 Windows 时在后台启动。正式版不附着，端口 8769 被占用时报错。
3. **只有桌面窗口能管理**（2026-11-09）：桌面模式下后端不写凭据文件、不随页面下发凭据；凭据只在主进程与后端的内存中。开发模式照旧使用凭据文件。这是 A-02 的防刻意部分（2026-11-05 定为随桌面客户端完成）。
4. **数据目录**（2026-11-09）：默认 `%LOCALAPPDATA%\HiveMemory`，首次启动可以指向已有目录；数据目录之下沿用仓库根的布局；卸载只删程序、保留数据目录。
5. **安装包内容**（2026-11-09）：Python 运行时、依赖与 Qdrant 随包；推理模型在首次启动时下载，显示进度。
6. **跨重启状态**（2026-11-10）：任何退出都只结算话题、不做生成，未执行的生成工作保存到磁盘、下次启动后继续。已由 [PR #213](./shutdown-pending-generation-persistence.md) 完成。
7. **签名与升级**（2026-11-10）：不做代码签名；不做自动升级，由用户下载新版安装包自己升级。
8. **删除数据**（2026-11-10）：应用提供删除入口，删除前二次确认；先不做备份。
9. **与前端 UI 优化的先后**（2026-11-10）：桌面计划先行，打包后的桌面端参与前端 UI 优化的一部分。
10. **预设 Agent**（A-05，2026-10-20）：只在初次部署时注入，不经过权限系统；具体做法随正式计划给出候选设计（第 4.3 节）。
11. **候选设计按候选定**（2026-11-10）：第 4 节注明“候选”的做法全部采用，包括预设 Agent 的注入时机与标记、删除数据只删记忆、托盘中的更改数据目录与删除入口，以及第 4.7 节对前端 UI 优化的理解。

## 2. 目标与非目标

### 2.1 目标

1. 不装开发环境的用户能用一个安装包装上 HiveMemory，首次启动选好数据目录、等模型下载完就能使用。
2. 客户端满足 ROADMAP 的验收：安装、启动、使用、关闭、重启、升级，卸载与删除数据都有明确语义。
3. 四个 harness（Claude Code、Codex、pi、dsh）连接桌面端启动的后端，与连接 `ps1 serve` 时行为相同。
4. 失败时用户看得到原因：端口被占用、后端启动失败、模型下载失败都给出说明与日志位置。

### 2.2 非目标

- 代码签名、自动升级与备份（第 1.1 节第 7、8 条）。
- Windows 之外的平台。
- 崩溃或强制结束时不丢：话题与排队的生成工作仍只在正常退出时保存。
- harness 接入凭据的防刻意：A-02 只覆盖管理凭据，接入凭据仍只防误用。
- 前端 UI 优化本身（D-05、D-15 等），见第 4.7 节。
- Qdrant 跨小版本的升级策略：v1.0 随包的 Qdrant 与现有数据同为 1.19.x，第一次升级 Qdrant 时再定。

## 3. 当前状态

两版原型在分支 `feat/desktop-prototype` 上，以下都已实现、有测试或经过实测，但还没有写进当前文档。

| 部分 | 现状 |
|:---|:---|
| 后端桌面模式 | `python -m hivememory.server --desktop` 从 stdin 第一行读入管理凭据；不读写 `management.token`，返回页面不下发凭据 cookie；stdin 结束时与 Ctrl+C 走同一条正常关闭路径 |
| 数据目录 | 环境变量 `HIVEMEMORY_DATA_ROOT` 指定数据目录；个人配置、访问登记与管理凭据在 `<数据目录>/configs`，随包的 `config.yaml` 与模板从程序目录读取；`.env` 与相对路径按工作目录解析，桌面端以数据目录为工作目录启动后端；不设置时与仓库布局相同 |
| Electron 主进程（`desktop/src/`） | 单实例、托盘常驻、关窗隐藏；用 checkout 运行时附着已在运行的后端或 Qdrant；窗口用内存分区，凭据由主进程放进 cookie；退出时先关窗，再关 stdin 等后端退出，最后结束自己启动的 Qdrant；`query-session-end` 推迟 Windows 关机并走同一关闭路径；后端与 Qdrant 以 detached 子进程、后端以 `pythonw.exe` 启动 |
| 首次启动 | 选择数据目录（默认或已有目录，路径不超过 100 个字符，Qdrant 不支持长路径）；启动页显示模型下载进度，模型缓存 60 秒不增长时先打开界面 |
| 打包（`desktop/scripts/package.mjs`） | uv 管理的独立 CPython 3.12.10 加锁定的运行依赖，不建 venv；后端源码按仓库布局放在 `resources/backend`；`qdrant.exe` 取自开发环境；electron-builder 生成按用户安装的 NSIS 安装包（约 242 MB），版本取自 `_version.py` |

实测：安装、全新数据目录首次启动（下载模型 1636 MB）、托盘退出、静默卸载都通过，owner 手动安装验收通过（2026-11-10）。退出时不再等待生成（#213）之后，托盘退出通常几秒内完成。

## 4. 目标设计

### 4.1 生命周期

- **端口**：只在打包运行时，启动前检查 8769 与 6333。任一被占用时弹出说明（哪个端口、可能是 `ps1 serve` 或另一个 HiveMemory）并退出，不附着。用 checkout 运行（`npm start`）时保留附着，方便开发。
- **登录时启动**：托盘菜单增加“登录 Windows 时启动”勾选项，默认关闭，用 Electron 的登录项设置。登录时启动只进托盘，不弹出窗口。
- **后端启动失败或意外退出**：主进程保留后端 stderr 的最后若干行，错误框显示这些行与日志目录，而不只是退出码。用户升级后配置中残留已删除的键时，错误框里就能看到键名与文件。不自动改写用户的配置文件。
- **关机与注销**：沿用 `query-session-end` 推迟关机、走同一关闭路径。验收时实测关机与注销都能在 Windows 放行之前完成正常关闭。

### 4.2 首次启动与数据目录

- 首次启动选择数据目录，保持原型的做法。
- **候选**：托盘菜单增加“更改数据目录…”，选好后提示重启生效，由应用自己重启；不搬移原目录的数据。
- **模型下载失败**：后端的就绪检查在模型加载失败时报告失败与失败的模型名（`/health/ready` 的状态为 `model_load_failed`），不只报告“预热中”。启动页据此显示失败说明与日志目录，提示重开应用重试。模型缓存长时间不增长时先打开界面的做法保留，用于下载卡住而没有报错的情况。

### 4.3 预设 Agent（A-05，候选）

- 预设 Profile 的定义（coder_doll、reviewer_doll）从 `scripts/inject_profiles.py` 移进后端。
- 后端启动时，数据目录中没有注入标记文件就写入预设 Profile，再写下标记。写入由组合根直接调用 Patchouli 的 Profile 写入路径，以所属 Workspace 的 `system` 为发起者，不经认证网关与 root API，满足“只在初次部署、不经权限系统”。已存在的 alias 跳过。
- 标记文件让“初次部署”只发生一次：用户删掉预设 Agent 后重启不会再注入。
- 删除 `scripts/inject_profiles.py` 与访问登记模板中的 `hivememory:profile-injector`。

### 4.4 升级

- **发布**：每个版本在本机执行 `npm run package` 生成安装包，附在该版本的 GitHub Release 上。
- **安装**：用户下载新版安装包，覆盖安装。安装前要求退出正在运行的 HiveMemory（electron-builder 的 NSIS 安装包检测运行中的应用），验收时实测。覆盖安装保留数据目录与 userData 中的设置。
- **没有签名**：运行安装包时 Windows SmartScreen 会提示，需要点“更多信息”再选“仍要运行”。帮助文档写明这一步。
- **版本**：托盘提示与窗口标题带上版本号，用户能看出自己装的是哪一版。
- **跨版本的数据**：保存的生成工作在新版本中读不回时进入 rejected 文件（#213）；配置中残留已删除的键时启动失败，错误框给出原因（第 4.1 节）。

### 4.5 删除数据

- **入口**：托盘菜单“删除全部记忆数据…”。
- **二次确认**：第一个对话框列出数据目录路径、会删除什么、保留什么；第二个对话框说明不能撤销，默认按钮是“取消”。
- **候选的删除范围**：记忆库（Qdrant 存储）、Artifact、长期归档与保存的生成工作。保留配置、Provider 密钥、接入凭据、模型缓存与日志，删除后可以直接继续使用。另一种做法是全部重置（连配置一起删、回到首次启动），评审时可以选。
- **做法**：
  1. 主进程正常关闭后端与 Qdrant；
  2. 删除 Qdrant 存储目录，它由桌面端管理；
  3. 以一次性模式运行后端，按配置删除后端拥有的路径：Artifact 根目录、长期归档目录与保存文件；
  4. 重新启动服务，后端启动时重建空集合。
- **安全**：只删除上面列出的路径，不删除数据目录本身。数据目录指向 checkout 时，仓库文件、`configs/` 中随仓库发布的文件与 `.hivememory/dev.env` 都不受影响。

### 4.6 harness 接入

四个 harness 插件都写死 8769，正式版端口不变。桌面用户在设置页生成接入凭据，这个入口已经有了，本计划不改插件。验收时逐个确认检索、写入与对话导入。

### 4.7 与前端 UI 优化的关系

本计划先完成。之后的前端 UI 优化（D-05 的空入口接线、D-15 的设置页分层等）在打包后的桌面端中进行与验收，其中桌面端特有的部分（候选）随 UI 优化一起完善：启动页的样式、删除数据与更改数据目录的入口是否从托盘移进设置页。本计划中这些入口先用托盘菜单与原生对话框，不改前端页面。

### 4.8 仍沿用原型的部分

- 后端桌面模式与数据目录的实现不变；
- 打包脚本不变；
- 窗口只加载后端在 8769 提供的页面；renderer 只用公开 API，主进程只做进程管理与原生对话框，不执行页面或远程内容。

## 5. 受影响的系统、契约与数据

- **desktop/**：端口检查、登录时启动、更改数据目录、删除数据、错误框显示后端输出、启动页的模型失败状态、版本显示。
- **server**：新增删除记忆数据的一次性运行模式（第 4.5 节）。
- **System**：启动时注入预设 Profile 并写标记（第 4.3 节）；就绪检查报告模型加载失败。
- **Patchouli**：模型服务的加载失败状态经就绪路由交出（现有的失败状态已在模型服务中）。
- **配置与脚本**：删除 `scripts/inject_profiles.py` 与 `system_principals.example.yaml` 中的 `hivememory:profile-injector`。
- **公共接口**：`/health/ready` 增加状态 `model_load_failed` 与失败的模型名。
- **数据**：数据目录中新增预设 Profile 的注入标记文件。

## 6. 兼容与迁移

- **开发方式不变**：不设置 `HIVEMEMORY_DATA_ROOT`、不加 `--desktop` 时与现在相同。用 checkout 运行桌面端时仍可附着。
- **已有部署的预设 Agent**：没有标记文件的已有数据目录会在升级后第一次启动时注入一次，alias 已存在的跳过，结果与手动运行过注入脚本相同。
- **注入脚本的 principal**：个人的 `system_principals.yaml` 中仍有 `hivememory:profile-injector` 时只是一条不再使用的登记，不影响启动。
- **数据目录指向 checkout 的安装版**（owner 现在的用法）：与 `ps1 serve` 共用数据，两者不能同时运行。正式版遇到 `ps1 serve` 或开发用的 Qdrant 正在运行时报错退出，不再附着。

## 7. 测试和验收标准

| 不变量 | 归属层 |
|:---|:---|
| 没有标记文件时注入预设 Profile 并写标记；有标记时不注入；已存在的 alias 跳过 | System 的单元测试 |
| 模型加载失败时就绪检查报告 `model_load_failed` 与模型名；加载成功后恢复就绪 | 就绪服务或 Patchouli 运行时的现有单元测试 |
| 删除模式只删除配置给出的 Artifact、长期归档与保存文件，不碰其他文件 | server 入口的现有单元测试 |
| 桌面模式与数据目录的现有测试继续通过 | `tests/unit/server`、`tests/unit/system` |

桌面主进程没有自动化测试，以手动验收为准：

1. 全新安装：SmartScreen 放行、选择默认数据目录、首次启动下载模型并显示进度、预设 Agent 出现。
2. 使用：对话、记忆管理、设置页生成接入凭据；四个 harness 逐个检索、写入、导入。
3. 关闭与重启：托盘退出数秒内结束；重启后保存的生成工作在后台生成。
4. Windows 注销与关机时正常关闭。
5. 端口占用：`ps1 serve` 运行时启动安装版，看到说明并退出。
6. 升级：装上新版本号的安装包覆盖安装，数据保留、版本号更新；在配置中写入一个不存在的键，错误框显示键名。
7. 模型下载失败：断网首次启动，启动页显示失败说明。
8. 删除数据：两次确认后记忆清空、配置与模型保留，应用照常可用；数据目录指向 checkout 时仓库文件不受影响。
9. 登录时启动与更改数据目录。
10. 卸载：程序与快捷方式删除，数据目录保留。

验证命令：

```bash
pytest tests/unit/server tests/unit/system tests/unit/patchouli -q
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85
ruff check . && black --check .
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
cd desktop && node --check src/main.js && npm run package
```

## 8. 风险与待决问题

- **SmartScreen 与杀毒软件**：没有签名的安装包与 `pythonw.exe`、`qdrant.exe` 可能被拦截或延迟扫描；帮助文档写明放行方法，验收时记下实际表现。
- **路径长度**：数据目录限制 100 个字符；Windows 用户名很长时默认位置也可能接近上限，验收时检查。
- **模型下载源**：模型从 HuggingFace 下载，网络不通时首次启动无法完成；本计划只做失败提示，不做镜像或离线包。
- **候选设计**：第 4 节注明“候选”的做法已由 owner 按候选定下（第 1.1 节第 11 条），包括：
  - 预设 Agent 的注入时机与标记；
  - 删除数据的范围（只删记忆，还是全部重置）；
  - 更改数据目录的入口；
  - 删除、更改数据目录等入口先放托盘、之后随 UI 优化决定位置；
  - 对“打包后的桌面端参与前端 UI 优化的一部分”的理解（第 4.7 节）。
- **待决**：无阻塞问题。

## 9. 完成时必须更新的当前文档

- 新增 `docs/frontend/desktop-client.md`：桌面客户端的当前设计，包括生命周期、凭据、数据目录、打包、升级、删除数据与诊断；[Frontend](../../frontend/README.md)索引登记。
- 新增 `docs/help/desktop-client.md`：安装（含 SmartScreen）、首次启动、升级、删除数据、卸载与常见问题；[安装与启动](../../help/setup.md)加入口。
- [配置指南](../../help/configuration.md)与 [System 配置](../../system/configuration.md)：`HIVEMEMORY_DATA_ROOT` 与个人配置的位置。
- [身份与执行安全治理](../../governance/security/identity-and-execution-safety.md)与[认证与访问登记](../../workspace/authentication.md)：桌面模式下的管理凭据；`hivememory:profile-injector` 的删除。
- [PROJECT](../../PROJECT.md)：桌面客户端成为可用的运行方式。
- 收尾时：
  - 本计划归档；
  - [ROADMAP](../../ROADMAP.md) 第 4.2 节表格与第 5.1 节更新；
  - [身份与访问体系 Idea](../../ideas/identity-and-access-model.md) 中 A-02、A-05 的状态同步；
  - [Plans 导航](../../plans/README.md)同步。

## 10. 实施记录

2026-11-10 在分支 `feat/desktop-prototype` 实施第 4 节，做法按候选设计落地，以 PR #TBD 合并。与计划不同，或计划没有写到的地方：

- **模型加载失败的报告**：
  - 就绪链路新增一对路由：Patchouli 本地路由 `runtime.models.failed` 与公开路由 `patchouli.public.models.failed`，列出最近一次加载失败的模型名。原有的“是否就绪”路由不变。
  - `/health/ready` 未就绪时，有模型加载失败则状态为 `model_load_failed`，并带 `failed_models`，否则仍为 `warming_up`。
  - 启动页收到失败后停在启动页，显示失败的模型名与日志目录，不再打开界面。
- **预设 Agent 的写入方**：
  - 预设定义移到 `patchouli/application/preset_agent_profiles.py`。
  - Patchouli 新增公开路由 `patchouli.public.agent_profile.seed_presets`：以所属 Workspace 的 `system` 写入缺少的预设，不经授权。
  - System 门面 `seed_preset_agent_profiles` 负责标记文件，标记路径是新增的配置项 `system.preset_agent_profiles_marker`。
  - 写入由 server 的 lifespan 在模型预热之后发起，写进不带身份声明的请求所在的 Workspace：默认身份只能在 server 的身份解析中取得。
  - 模型没有就绪时不写入，否则写入会再次触发模型加载，就绪检查看不到“加载失败”。
- **删除记忆数据**：
  - 后端新增一次性运行模式 `--delete-memory-data`。
  - 删除范围除第 4.5 节列出的路径外，还包括读不回的 rejected 文件与预设 Agent 的标记。预设 Agent 存在记忆库中，删除后要随下次启动重新写入。
  - 删除完成后应用自己重新启动，与更改数据目录相同。
- **附着**：后端与 Qdrant 都由本应用启动时才能删除记忆数据；用 checkout 运行、附着到了已在运行的服务时，删除入口只给出说明。
- **端口检查**：安装版检查 8769、6333 与 6334，6334 是 Qdrant 的 gRPC 端口。
- **数据目录对话框**：第三个按钮由“退出”改为“取消”，在首次启动时表示退出，在更改数据目录时表示不改。
- **删除的测试**：注入脚本的集成测试随脚本删除，“已存在的 alias 跳过”改由 Patchouli 的单元测试守护。

实测（免安装目录版，独立的 userData 与新数据目录，2026-11-10）：

- 开发用 Qdrant 占着 6333 时启动：约 1 秒报错退出，没有启动后端，也没有接入那个 Qdrant。
- 端口空闲、带模型缓存的新数据目录：约 25 秒打开界面，窗口标题为“HiveMemory 0.7.3”，预设 Agent 在预热后写入；从主进程退出，4 秒内全部进程结束，后端 2.8 秒正常关闭。
- 删除记忆数据：用安装版的 Python 运行删除模式，只删了 Artifact、长期归档与标记，配置、日志与模型缓存保留；重新启动后记忆为空，两个预设 Agent 重新写入。
- 没有模型缓存且禁止下载时：约 14 秒启动页显示两个模型加载失败与日志目录，预设 Agent 的写入按设计跳过。

验证：后端单元与集成测试（不含 `live_llm`、`e2e`、`slow`）3350 通过、1 跳过；`ruff check .`、`black --check .`、`mypy` 通过；`node --check` 通过。

收尾（2026-11-10）：

- **第 7 节中需要手动操作或会改动本机设置的验收**，由 owner 在日常使用中进行，不阻塞归档：
  - 托盘菜单的两次确认对话框与删除流程；
  - 登录 Windows 时启动（会写注册表）；
  - 更改数据目录；
  - 安装包的 SmartScreen 放行；
  - 覆盖安装升级与配置残留键时的错误框；
  - Windows 注销与关机；
  - 四个 harness 的逐个验证。
- **当前文档**：第 9 节列出的文档已经更新，另外一并更新了：
  - [System 配置](../../system/configuration.md)、[认证与访问登记](../../workspace/authentication.md)与[能力层](../../workspace/capability.md)中关于注入脚本与 `hivememory:profile-injector` 的段落；
  - 提到 `/health/ready` 的[应用服务](../../system/application-services.md)、[安装与启动](../../help/setup.md)与[故障排查](../../help/troubleshooting.md)。
  - [路由与事件契约](../../contracts/routes-and-events.md)、[Patchouli](../../patchouli/README.md)、[状态与传输](../../frontend/state-and-transports.md)与[身份与访问体系 Idea](../../ideas/identity-and-access-model.md)。
