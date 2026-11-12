---
title: 日常实例与开发实例分开
status: idea
horizon: candidate
owner: project
scope: separate-the-daily-desktop-instance-from-the-checkout-dev-instance-ports-data-root-credentials-and-harness-bindings
code_paths:
  - scripts/hivememory-dev.ps1
  - scripts/hivememory-dev.env.example
  - frontend/vite.config.ts
  - configs/config.yaml
  - desktop/src/layout.js
  - desktop/src/services.js
  - src/hivememory/system/plugins/
related_docs:
  - docs/frontend/desktop-client.md
  - docs/help/setup.md
  - docs/help/desktop-client.md
  - docs/archive/plans/desktop-plugin-installation.md
last_reviewed: 2026-11-12
---

# 日常实例与开发实例分开

**文档状态**：Idea（候选）。第 2 节是 owner 2026-11-12 的决定，其余是候选设计，可在计划评审时调整。[桌面端安装插件](../archive/plans/desktop-plugin-installation.md)合并之后，按本 Idea 编写下一份计划。

## 1. 问题

一台电脑上只有一套 HiveMemory 的资源，开发环境与桌面安装版轮流使用它们，每切换一次都牵动其他几样：

- **checkout 一身三职**：
  - 它是代码仓库；
  - 它是日常记忆的数据根：安装版选它作数据根，开发脚本的 Qdrant 存储也默认在这里，两者读写同一个记忆库；
  - 它是 harness 插件的来源。
- **端口**：Windows 开发脚本、Vite 代理、`config.yaml` 与桌面端都固定使用 8769、6333、6334（WSL 脚本已经能从 `dev.env` 读端口）。所以两者不能同时运行：开发时日常记忆不回流，正在开发 HiveMemory 的会话本身也在其中。
- **代码**：安装版是打包那一刻的快照，分支上的改动要重新打包，才能在安装版里看到。
- **管理凭据**：桌面模式的凭据只在内存中，只有桌面窗口能管理；Vite 读的是 `configs/management.token`，所以连到安装版的后端一律 401。
- **harness 配置**：每个 Windows 用户只有一份。插件页在哪个实例里安装，就产生两个后果：
  - Claude Code 与 Codex 的本地 marketplace、pi 与 dsh 的链接，都指向那个实例的插件目录；
  - 用户环境变量中的凭据，也换成那个实例的凭据。
- **新用户验收**：本机已经装了 Python 和 harness，也已经有连接，验证不了“新用户装完之后什么都没有”的场景。

## 2. owner 决定（2026-11-12）

日常使用与开发分成两个互不相干的 HiveMemory 实例，每样资源只归其中一个。

## 3. 候选设计

### 3.1 资源归属

| 资源 | 日常实例 | 开发实例 |
|:---|:---|:---|
| 代码 | 桌面安装版（发布版本的快照） | checkout |
| 端口 | 8769、6333、6334（插件默认连这组） | 另一组，例如 8779、6343、6344 |
| 数据根 | `%LOCALAPPDATA%\HiveMemory` | checkout（开发数据，可以随时清空） |
| 管理凭据 | 只在内存中，只有桌面窗口能管理 | `configs/management.token`，Vite 照常使用 |
| harness 插件 | 只绑定这里 | 不绑定 |

两个实例可以同时运行：开发时日常记忆照常回流。

### 3.2 日常实例

- **形态**：只有桌面安装版。数据根是缺省的 `%LOCALAPPDATA%\HiveMemory`，端口不变，harness 的插件只在这里安装。
- **新代码进入日常**：打包，再覆盖安装。升级后，Claude Code 与 Codex 的插件由桌面端安装插件计划自动重新安装。这改变了此前“checkout 就是日常部署，重启即上线”的工作方式，日常使用从此跟随打包的节奏。
- **一次性迁移**：
  1. 退出 HiveMemory；
  2. 把 checkout 中的日常数据复制到新的数据根，包括 `configs` 中的个人配置，以及 `.hivememory` 中的记忆库、Artifact 和保存的生成工作；
  3. 从托盘的“更改数据目录”指到新的数据根；
  4. 在安装版的插件页把各 harness 重新安装一次，让它们指向安装目录。

  迁移期间日常实例要停一次。

### 3.3 开发实例

- **端口**：只有 `.hivememory/dev.env` 一个来源，键为 `HIVEMEMORY_BACKEND_PORT`、`QDRANT_HTTP_PORT`、`QDRANT_GRPC_PORT`，缺省是开发的那一组。
  - Windows 开发脚本按这些键启动 Qdrant 与后端，并以环境变量把 Qdrant 端口交给后端（`patchouli.storage` 的 `port` 与 `grpc_port`），与 WSL 脚本相同；
  - Vite 代理从同一个文件取后端端口。
- **数据根**：checkout，用的是开发数据，可以随时清空。需要真实数据时，从日常实例复制一份快照过来。
- **模型缓存**：开发实例的模型缓存目录指向日常实例的缓存，不重复下载。两个进程仍各自把模型加载进内存。
- **管理凭据**：保持现在的文件方式。
- **插件服务**：开发实例不改动 harness，插件页显示“不能代为安装”并给出命令。在开发实例里安装会带来两个问题：harness 被改指到 checkout，用户环境变量中的日常凭据被覆盖。实现上可以在开发数据根的覆盖文件里关掉插件服务，具体写法在计划中确定。
- **checkout 方式的桌面端**（`desktop` 目录的 `npm start`）：属于开发实例，用开发端口与 checkout 数据根，附着到开发脚本启动的后端。

### 3.4 测插件与新用户验收

- **测插件代码本身**：让单个 harness 连开发实例，用 harness 自己的隔离方式，不动日常的绑定：
  - Claude Code：项目或本地范围的插件；
  - Codex：另设 `CODEX_HOME`；
  - pi：`pi install -l` 本地安装；
  - dsh：独立的 profile。

  地址与凭据写在这个 harness 进程的环境里，不写用户环境变量。各 harness 的具体做法在落地时逐个核对。
- **安装包与新用户验收**：在 Windows Sandbox 里做（Windows 11 Pro 自带，在“启用或关闭 Windows 功能”中打开）。每次启动都是一台干净的 Windows：没有 Python、没有 harness，也没有连接。仓库提供一份 `.wsb` 配置，把 `desktop/dist` 只读映射进去，在里面安装安装包；插件页的验收也在这里做。

## 4. 代价与开放问题

- **内存**：两个实例同时运行时，推理模型与 Qdrant 各占一份；开发实例靠已有的模型空闲卸载减轻。
- **开发实例的记忆库默认是空的**：检索效果要用复制来的快照或专门的测试数据来测。
- **Sandbox 的模型下载**：Sandbox 每次都是干净的，推理模型每次都要重新下载（约 1.6 GB），除非把日常的模型缓存映射进去。映射之后怎样让安装版使用它，在计划中确定。
- **Sandbox 里的 harness**：要在里面装 harness 的命令行；插件安装本身是否需要先登录，落地时核对。
- **开发实例里插件页不可用之后**：插件页自身的改动改在 Sandbox 的安装版中验证。

## 5. 升级为 Plan 的条件

- [桌面端安装插件](../archive/plans/desktop-plugin-installation.md)合并并归档；
- 动手之前与 owner 确认迁移日常数据的时间（第 3.2 节）。
