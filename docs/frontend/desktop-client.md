---
title: Desktop Client
status: current
owner: frontend
scope: electron-windows-desktop-client-lifecycle-credential-data-root-packaging-and-data-deletion
code_paths:
  - desktop/src/main.js
  - desktop/src/services.js
  - desktop/src/layout.js
  - desktop/src/starting.html
  - desktop/scripts/package.mjs
  - src/hivememory/server/__main__.py
  - src/hivememory/server/management_credential.py
  - src/hivememory/config/files.py
  - src/hivememory/system/system.py
  - src/hivememory/patchouli/application/preset_agent_profiles.py
related_docs:
  - docs/help/desktop-client.md
  - docs/workspace/authentication.md
  - docs/system/configuration.md
  - docs/patchouli/generation.md
last_reviewed: 2026-11-12
---

# 桌面客户端

HiveMemory 是个人软件，后端在用户自己的电脑上运行。桌面客户端把它做成一个不需要开发环境的 Windows 应用：一个安装包带上 Python 运行时与依赖、后端、前端页面与 Qdrant，首次启动时下载推理模型。它解决三件事：

- **安装与运行**：不需要用户自己准备 Python 环境、下载 Qdrant、写配置文件再用脚本启动。
- **常驻**：四个 harness（Claude Code、Codex、pi、dsh）随时会调用 `/mcp` 与对话导入，后端不在时它们的记忆工具都不可用。窗口关掉之后后端仍要运行。
- **管理凭据**：开发方式下管理凭据写在文件里，与 harness 中的 agent 同一个系统用户的程序都读得到。桌面客户端让凭据只留在内存中（第 4 节）。

客户端是 Electron 应用，只支持 Windows。主进程只做进程管理与原生对话框；窗口加载后端在 8769 提供的页面，页面只用公开的 HTTP 接口，主进程不执行页面或远程内容。renderer 开启 `sandbox`、`contextIsolation`，不开 `nodeIntegration`；它不能替代 Agent 的执行沙箱。

## 1. 两种运行方式

| | 安装版 | 用 checkout 运行（`desktop` 目录下 `npm start`） |
|:---|:---|:---|
| 数据根 | 用户选择，默认 `%LOCALAPPDATA%\HiveMemory` | 仓库根 |
| 后端 | 随包的 `resources\python\pythonw.exe`，以 `PYTHONPATH` 指向随包的后端源码 | 仓库 `.venv` 的 `pythonw.exe` 与仓库源码 |
| 前端页面 | 随包的 `resources\backend\frontend\dist` | 仓库 `frontend\dist`，改动前端后先 `npm run build` |
| Qdrant | 随包的 `resources\qdrant\qdrant.exe`，存储在 `<数据根>\.hivememory\qdrant` | `.hivememory\dev.env` 中的 `QDRANT_BIN` 与存储路径，与开发脚本相同 |
| 端口上已有服务 | 报错退出 | 直接使用（附着），不启动也不关闭它 |

两种方式的布局由 `desktop/src/layout.js` 给出，其余行为相同。安装版是打包那一刻的快照，不读 checkout 中的代码；改代码之后用 checkout 运行即可，只有验证安装版特有的行为或发布时才需要重新打包。

## 2. 生命周期

**启动**：

1. 单实例：第二次打开只显示已有窗口。
2. 安装版取得数据根（第 5 节）；以 `--hidden` 启动（登录 Windows 时）只进托盘、不显示窗口。
3. 安装版检查 8769、6333 与 6334（Qdrant 的 gRPC），任一被占用时说明原因并退出。端口固定：插件都写死 8769，换端口要改所有插件。被占用多半是开发用的 `ps1 serve`、Qdrant 或另一个 HiveMemory；安装版附着到开发用的 Qdrant 会读写 checkout 的记忆库，所以不附着。
4. 启动 Qdrant：工作目录是 `<数据根>\.hivememory\run`，pid 写在同目录的 `qdrant.pid`（与开发脚本共用，主进程意外退出后可以用 `hivememory-dev.ps1 stop-qdrant` 结束它）。
5. 生成管理凭据，以 `python -m hivememory.server --desktop` 启动后端：工作目录是数据根，环境变量带上数据根与日志路径，stdin 的第一行是凭据（第 4 节）。
6. 等后端能响应 `/health`，把凭据放进窗口的内存会话，再等推理模型就绪（第 7 节），然后加载后端页面。

后端与 Qdrant 以 detached 子进程启动：非 detached 的子进程会被放进随主进程结束而结束的 job，主进程被强制结束时，后端会在关闭中途被杀掉。后端用不带控制台的 `pythonw.exe`：venv 的 `python.exe` 只是转发器，detached 的转发器没有控制台，Windows 会给真正的解释器新开一个空的控制台窗口，用户关掉它时后端立即被结束。

**启动页与窗口**：加载后端页面之前，窗口显示本地的启动页 `desktop/src/starting.html`，由主进程改写其中的状态文字（启动 Qdrant 与后端、等待推理模型，第 7 节）。启动页用与前端相同的深浅两套配色，随 Windows 的深浅色设置切换，不读取页面中保存的主题选择；带品牌标记与进度条，减少动效时进度条停止移动。标题用 Manrope：安装版取打包时复制进程序的文件（第 8 节），用 checkout 运行时取 `frontend/node_modules` 中的同一个文件，两处都没有时回退到系统字体。窗口最小为 1024×640：再小时对话页放不下，右栏与会话列表都已收为抽屉，中间栏仍需 560px。

**关窗与退出**：关窗只隐藏到托盘。从托盘退出时：先销毁窗口（页面持有的事件流与 WebSocket 会让后端等待它们关闭），再关闭后端的 stdin，后端读到 stdin 结束时走与 Ctrl+C 相同的正常关闭；等后端退出后结束 Qdrant。后端退出不设时限、不强制结束。正常关闭只结算话题、不做生成，未执行的生成工作保存到磁盘，通常几秒内完成（[记忆生成](../patchouli/generation.md#62-提交阶段关闭时保存与启动时继续)）。Qdrant 先写 WAL 再确认写入，直接结束后下次启动能够恢复。

**Windows 关机与注销**：Windows 不会发出 `before-quit`；窗口收到 `query-session-end` 时推迟会话结束，走同一关闭路径。

**意外退出**：后端在运行中退出时，错误框显示退出代码与后端输出文件 `backend.stdout.log` 的最后 12 行，再关闭应用。后端启动失败时同样显示这些行：用户自己升级之后，配置中残留新版本删掉的键会让启动失败，这时从错误框就能看到键名与文件。

**托盘菜单**：打开 HiveMemory、打开数据目录、登录 Windows 时启动（安装版，默认关闭，写入 Windows 的登录项，带 `--hidden` 参数）、更改数据目录（安装版，第 5 节）、删除全部记忆数据（第 6 节）、退出。托盘提示与窗口标题带版本号；窗口标题不随页面标题变化。

## 3. 后端的桌面模式

`python -m hivememory.server --desktop`（`server/__main__.py`）：

- 从 stdin 第一行读入管理凭据，第一行为空时启动失败；
- 不读写 `configs/management.token`，返回页面时不下发凭据 cookie；
- 另起一个线程读 stdin，读到结束（主进程关闭 stdin，或主进程意外退出、管道随之关闭）时请求正常关闭。

主进程意外退出时，后端因此自己正常关闭，不会留下没有人持有凭据的孤儿后端。

桌面模式的后端启动后，在后台为插件页装过的 Claude Code 与 Codex 重新安装插件：两者保存的是插件的副本，覆盖安装新版本之后要重新安装才能读到。插件服务在数据根的 `.hivememory/plugin_installs.json`（`system.plugin_install_record`）中记下每个 harness 安装时的版本，只处理版本落后的；失败只写日志，插件页把它标为“需要更新”。pi 与 dsh 链接插件目录、原地读取，不需要处理。开发方式（checkout）不在启动时改动 harness。

## 4. 管理凭据

`/api/` 下除导入之外的接口都要求管理凭据（[认证与访问登记](../workspace/authentication.md#1-两阶段认证)）。开发方式下凭据保存在 `configs/management.token`，后端提供页面时随页面以 cookie 下发，只能防误用：同一个系统用户的程序读得到文件，也能请求页面拿到 cookie。

桌面客户端由主进程在每次启动时生成 32 字节的随机凭据，经 stdin 管道交给后端，不写文件、不放进环境变量或命令行参数。主进程把它设为窗口会话中 `HttpOnly`、`SameSite=Strict` 的 cookie；窗口用没有 `persist:` 前缀的分区，cookie 与缓存都不落盘，页面脚本读不到它。后端不随页面下发 cookie，所以在普通浏览器中打开 8769 只能得到 401，磁盘上留着的旧凭据文件同样被拒绝。只有桌面窗口能管理。

这达到的是“同一系统用户能读的文件与环境变量中都没有凭据”。用 checkout 运行、附着到已在运行的后端时，主进程没有凭据，页面由那个后端照旧下发 cookie。harness 的接入凭据不在这里，仍只防误用。

## 5. 数据根

环境变量 `HIVEMEMORY_DATA_ROOT` 指定数据根（[System 配置](../system/configuration.md#2-来源与优先级)）。设置后，个人配置在 `<数据根>\configs`：`config.local.yaml`、`models.yaml`、`providers.secrets.yaml`、两份访问登记与 `principals.secrets.yaml`；随包的 `config.yaml` 与各模板仍从程序目录读取。`.env` 与相对路径的数据目录按工作目录解析，桌面端以数据根为工作目录启动后端，所以数据根之下的布局与仓库根相同：`configs\`、`data\`（模型缓存、长期归档）、`.hivememory\`（记忆库、Artifact、日志、保存的生成工作）。不设置时与仓库布局相同。

安装版第一次启动时选择数据根：默认位置，或已有的目录（例如仓库的 checkout，不需要搬动文件）。选择记在 userData 的 `desktop-settings.json`（`%APPDATA%\HiveMemory`），之后沿用。数据根的路径不能超过 100 个字符：`qdrant.exe` 不支持超过 260 个字符的路径，记忆库的文件在数据根之下最深约 142 个字符，数据根更长时建集合失败、后端起不来。

托盘的“更改数据目录…”用同一个对话框重新选择，写入设置后重启应用，不搬移原目录的数据。

数据根指向 checkout 时，安装版与开发脚本共用一份数据；端口固定，两者不能同时运行。

## 6. 删除记忆数据

托盘的“删除全部记忆数据…”需要两次确认：第一个对话框列出数据根、会删除什么与会保留什么；第二个说明不能撤销、没有备份，默认按钮是“取消”。附着到不是本应用启动的后端或 Qdrant 时不能删除，只给出说明。

删除范围：

| 删除 | 保留 |
|:---|:---|
| 记忆库（Qdrant 存储，包括全部记忆与 Agent Profile）、Artifact、长期归档、关闭时保存的生成工作与读不回的 rejected 文件、预设 Agent 的写入标记 | 配置、Provider 密钥、harness 接入凭据、推理模型缓存、日志 |

步骤：主进程正常关闭后端与 Qdrant，删除 Qdrant 存储目录，再以一次性模式 `python -m hivememory.server --delete-memory-data` 运行后端，按配置删除后端拥有的路径（相对路径按数据根解析），然后重启应用。重启后记忆库为空，预设 Agent 重新写入。只删除上面列出的路径，不删除数据根本身：数据根指向 checkout 时，仓库文件、随仓库发布的 `configs\` 文件与 `.hivememory\dev.env` 都不受影响。

卸载只删除程序与快捷方式，保留数据根与 userData；删除数据只经这个入口。

## 7. 推理模型与首次部署

**模型就绪**：启动页轮询 `/health/ready`，期间按模型缓存目录（`<数据根>\data\model_cache`）的增长显示已下载的 MB 数；首次启动共需下载约 1.6 GB。

- 就绪时加载后端页面。
- 后端报告 `model_load_failed` 时停在启动页，显示失败的模型名与日志目录，提示检查网络后重开应用；没有推理模型时检索与记忆生成都不可用。
- 模型缓存 60 秒不增长、后端也没有报告失败时（下载卡住而没有报错），先打开界面。

就绪状态的含义见 [System 应用服务](../system/application-services.md)。

**预设 Agent**：数据根第一次启动时写入预设的 Agent Profile（coder_doll、reviewer_doll），预设内容在 `patchouli/application/preset_agent_profiles.py`。

- server 的 lifespan 在模型预热之后，调用门面的 `seed_preset_agent_profiles`，目标是不带身份声明的请求所在的 Workspace，也就是桌面窗口管理的 Workspace。
- 门面在数据根中没有标记文件（`system.preset_agent_profiles_marker`，默认 `.hivememory\preset_agent_profiles_seeded`）时，经 Patchouli 公开路由 `patchouli.public.agent_profile.seed_presets` 写入缺少的预设，成功后写下标记。
- Patchouli 以该 Workspace 的 `system` 为发起者直接写入，已存在的 alias 跳过，不经认证网关、root API 与授权：这时还没有任何调用方，写入的也只是随程序发布的内容。
- 标记让写入只发生一次，用户删掉预设 Agent 后重启不会写回；删除记忆数据会一并删除标记。
- 模型没有就绪时不写入：写入要计算嵌入，会再次触发模型加载，就绪检查就看不到“加载失败”。写入失败只记录日志、不留标记，下次启动重试。

## 8. 打包与升级

`desktop/scripts/package.mjs`（`npm run package`，`npm run package:dir` 只生成免安装目录）：

- **Python**：uv 管理的独立 CPython（与仓库 `.venv` 同版本），安装在 `desktop\.cache\python`，不写入用户的 uv 目录、PATH 与注册表。锁定的运行依赖由 `uv export --frozen --no-dev` 导出，直接装进这个解释器，不建 venv：venv 记录基础解释器的绝对路径，无法随安装位置移动。
- **后端与前端**：后端源码按仓库布局放在 `resources\backend`，经 `PYTHONPATH` 导入，不安装成包，按源码位置找到随程序发布的配置与前端；打包前构建前端。
- **插件**：`integrations/`（本地 marketplace 与插件目录，不含插件自己的测试与 Python 缓存）放在 `resources\backend\integrations`，与后端同版本；插件页从这里为 harness 安装插件（[管理页面](./management-views.md#35-插件)），覆盖安装时随程序换成新版本。
- **启动页字体**：electron-builder 的 `files` 把前端依赖 `@fontsource-variable/manrope` 中的字体文件复制到程序内启动页旁的 `fonts`，与前端打包进页面的是同一个文件，启动页不另带字体副本。
- **Qdrant**：取自开发环境 `.hivememory\dev.env` 的 `QDRANT_BIN`，与开发时同一个版本。
- **安装包**：electron-builder 生成按用户安装的 NSIS 安装包，默认装在 `%LOCALAPPDATA%\Programs\HiveMemory`，版本号取自 `_version.py`。

不做代码签名：运行安装包时 Windows SmartScreen 会提示，需要选“更多信息”再选“仍要运行”。不做自动升级：用户下载新版安装包覆盖安装，安装前退出正在运行的 HiveMemory；覆盖安装保留数据根与 userData 中的设置。关闭时保存的生成工作在新版本中读不回时进入 rejected 文件，配置中残留已删除的键时启动失败并在错误框中给出原因。

## 9. 日志与诊断

日志都在 `<数据根>\.hivememory\logs`：

- `desktop.log`：主进程，记录启动、数据根、关闭耗时与失败；
- `backend.log`、`runtime-events.jsonl`：后端日志与运行时事件；
- `backend.stdout.log`：后端在日志配置生效之前的输出，以及删除数据的一次性运行；
- `qdrant.log`、`qdrant.err.log`。

托盘的“打开数据目录”打开数据根。

## 10. 已知限制

- 只有正常退出（托盘退出、Windows 关机与注销）时保存话题与生成工作；崩溃、在任务管理器中结束进程或断电时，驻留话题与排队的生成工作丢失。
- 只支持 Windows；没有代码签名、自动升级与备份。
- 模型只从 HuggingFace 下载，没有镜像或离线包；网络不通时首次启动无法完成。
- Qdrant 跨小版本的升级策略还没有定，随包的 Qdrant 与已有数据同为 1.19.x。
- 管理凭据留在主进程与后端的内存中，同一系统用户读取进程内存仍能拿到；harness 接入凭据只防误用。
- 窗口的会话分区不落盘（第 4 节），页面保存在浏览器存储中的界面偏好（主题、Chat 两栏的收起、视图与筛选、当前会话与 Agent 的选择）只在一次运行内有效，重启桌面客户端后回到缺省值；保存在后端的数据不受影响。主题的持久化与自定义背景一起处理，见 [Todo](../todo/custom-background-and-preference-persistence.md)。
- 桌面主进程没有自动化测试，打包与安装行为以手动验收为准。

## 11. 代码与测试入口

- 主进程与服务管理：`desktop/src/main.js`、`desktop/src/services.js`、`desktop/src/layout.js`；启动页：`desktop/src/starting.html`；打包：`desktop/scripts/package.mjs`。
- 后端桌面模式、删除记忆数据的一次性模式：`src/hivememory/server/__main__.py`，测试 `tests/unit/server/test_main.py`；凭据：`server/management_credential.py`，测试 `tests/unit/server/test_management_credential.py`、`tests/unit/server/routers/test_app_lifespan.py`。
- 数据根：`src/hivememory/config/files.py`，测试 `tests/unit/system/test_access_registration_files.py`、`tests/unit/system/test_config_agent_runtime.py`。
- 预设 Agent：门面 `system/system.py`，Patchouli `application/agent_profile_management_service.py`，测试 `tests/unit/system/test_hivememory_system.py`、`tests/unit/patchouli/application/test_agent_profile_management_service.py`。
- 模型加载失败的报告：`system/application/readiness_service.py`，测试 `tests/unit/system/application/test_readiness_service.py`。
