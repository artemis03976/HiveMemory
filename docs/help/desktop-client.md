---
title: Desktop Client
status: current
owner: frontend
scope: install-first-start-upgrade-data-deletion-and-uninstall-of-the-windows-desktop-client
related_docs:
  - docs/frontend/desktop-client.md
  - docs/help/setup.md
last_reviewed: 2026-11-12
---

# 使用桌面客户端

桌面客户端是 HiveMemory 的 Windows 应用：一个安装包带上后端、前端页面、Python 运行时与记忆库（Qdrant），不需要开发环境。它常驻托盘，关窗之后后端照常运行，外部 harness 的记忆工具随时可用。设计与行为见 [桌面客户端](../frontend/desktop-client.md)。

## 1. 安装

1. 运行安装包 `HiveMemory Setup <版本>.exe`。安装包没有代码签名，Windows 会显示“Windows 已保护你的电脑”：选择“更多信息”，再选择“仍要运行”。
2. 按向导完成安装，默认装在 `%LOCALAPPDATA%\Programs\HiveMemory`，按当前用户安装，不需要管理员权限。

安装包从源码生成：在 `desktop` 目录执行 `npm ci` 后执行 `npm run package`，产物在 `desktop\dist`。打包需要 uv、Node.js 与开发环境中可用的 `qdrant.exe`（`.hivememory\dev.env` 的 `QDRANT_BIN`）。

## 2. 首次启动

1. **选择数据目录**：配置、记忆库、模型缓存与日志都放在这里，卸载时保留。
   - 默认位置是 `%LOCALAPPDATA%\HiveMemory`。
   - 已经有 HiveMemory 的数据（例如仓库的 checkout）时，可以直接选择那个目录，不需要搬动文件。
   - 路径不能超过 100 个字符，否则会提示重新选择。
2. **等待推理模型下载**：启动页显示已下载的大小，首次共约 1.6 GB。下载完成后自动打开界面，并写入两个预设 Agent（coder_doll、reviewer_doll）。
3. **配置模型提供商**：在 Provider 页面保存 API key，与开发方式相同（[配置指南](./configuration.md)）。

## 3. 日常使用

- 关闭窗口只是隐藏到托盘；点击托盘图标重新打开。
- 托盘菜单：
  - **登录 Windows 时启动**：勾选后登录时在后台启动，只进托盘，不弹出窗口；
  - **打开数据目录**；
  - **更改数据目录…**：选择后应用重新启动，原目录的数据不会搬过去；
  - **删除全部记忆数据…**：见第 5 节；
  - **退出**：后端正常关闭后退出，通常几秒内完成。没有生成完的记忆在下次启动后继续生成。
- 管理只能在桌面窗口中进行：在浏览器中打开 `http://127.0.0.1:8769` 会被拒绝（401）。
- 新安装的 HiveMemory 不连接任何 harness。在设置页的“插件”页为 Claude Code、Codex、pi 或 dsh 桌面端安装插件：安装包带着插件，HiveMemory 代为运行 harness 的插件命令、登记接入，并把凭据写进 Windows 用户环境变量；装好之后完全退出该 harness 再打开。各 harness 的细节见[安装与启动](./setup.md#9-接入外部-harness)第 9 节列出的手册。

## 4. 升级

下载新版安装包，退出正在运行的 HiveMemory（托盘“退出”），再运行安装包覆盖安装。数据目录与设置保留，窗口标题显示新的版本号。没有自动升级。

插件随安装包一起换成新版本：pi 与 dsh 原地读取插件，重新打开即可；Claude Code 与 Codex 保存的是插件的副本，HiveMemory 启动后自动为它们重新安装，之后重新打开该 harness（Codex 还要重新信任 hook）。自动重新安装失败时，插件页把它标为“需要更新”。

升级之后后端起不来时，错误框会显示后端最后的输出：如果提示配置中有未知的键，按提示从数据目录下 `configs\config.local.yaml` 或 `.env` 中删掉那一项。

## 5. 删除记忆数据

托盘“删除全部记忆数据…”，确认两次后删除，没有备份，不能撤销。

- **删除**：全部记忆与 Agent（预设 Agent 会在重启后重新写入）、来源与版本记录、长期归档、还没生成完的记忆工作。
- **保留**：配置、Provider 密钥、harness 接入凭据与已安装的插件、推理模型与日志。

删除后应用自动重新启动，可以直接继续使用。数据目录指向仓库 checkout 时，删除的就是 checkout 中的记忆库，仓库文件不受影响。

## 6. 卸载

在 Windows“设置 → 应用”中卸载 HiveMemory。卸载删除程序与快捷方式，保留数据目录与 `%APPDATA%\HiveMemory` 中的设置；要连数据一起清除，先用第 5 节删除记忆数据，再手动删除数据目录。

## 7. 常见问题

| 现象 | 原因与处理 |
|:---|:---|
| 启动时提示端口 8769、6333 或 6334 已被占用 | 开发用的 `ps1 serve`、Qdrant 或另一个 HiveMemory 正在运行。先停止它们，例如执行 `powershell -File scripts\hivememory-dev.ps1 stop-qdrant`，再打开 HiveMemory |
| 启动页提示推理模型下载或加载失败 | 检查网络是否能访问 HuggingFace，从托盘退出后重新打开即重试；详细原因在日志目录的 `backend.log` |
| 启动时提示后端没有启动成功 | 错误框中是后端最后的输出；完整日志在数据目录的 `.hivememory\logs` |
| 选择数据目录时提示路径太长 | 记忆库的文件放得很深，换一个更短的目录 |
| 安装版与开发环境用同一份数据（数据目录指向 checkout） | 两者不能同时运行：用 `ps1 serve` 之前先从托盘退出 HiveMemory |

日志都在数据目录的 `.hivememory\logs`：`desktop.log` 是桌面端，`backend.log` 是后端，`qdrant.log` 是记忆库。更多排查见[故障排查](./troubleshooting.md)。
