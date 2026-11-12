---
title: Connect pi
status: current
owner: project
scope: pi-extension-installation-configuration-verification-and-troubleshooting
code_paths:
  - integrations/hivememory/package.json
  - integrations/hivememory/node/
  - src/hivememory/system/plugins/
related_contracts:
  - docs/contracts/mcp.md
related_docs:
  - docs/system/import-bus.md
  - docs/help/setup.md
  - docs/help/claude-code-plugin.md
last_reviewed: 2026-11-12
---

# 接入 pi

仓库中的插件目录 `integrations/hivememory/` 同时是一个 pi 包：pi 按其中的 `package.json` 加载 HiveMemory 扩展（`node/`）与使用手册 skill。在 pi 中它做三件事：

- **记忆工具**：扩展自己注册 `memory_search`（检索）、`memory_read`（按 alias 读取全文）、`memory_write`（写入）与 `memory_update`（修正已有记忆）四个工具。写入与更新需要这一轮的信息，HiveMemory 据此把写入落在这一轮所在的话题：扩展在调用时自动补上，模型看不到这个参数。工具只交给模型直接调用，codemode 中的脚本不能调用它们。
- **对话回流**：pi 中每一轮结束后，扩展在会话记录中写一条结束标记（不进入模型上下文），再把这一轮（用户的原始输入、回复、思考、工具调用及结果）导入 HiveMemory。网络提交在后台进行，HiveMemory 未运行时 pi 照常工作，未导入的轮次在下次打开 pi 时补发。工作原理见 [Import Bus](../system/import-bus.md#11-pi-扩展)第 11 节。
- **使用手册 skill**：`skills/hivememory/SKILL.md` 随包加载，引导 agent 何时检索、写入与修正记忆；也可以用 `/skill:hivememory` 展开它。

三部分都以 pi 自己的接入来源 `harness:pi` 认证，以 Agent `pi` 访问。新安装的 HiveMemory 不连接任何 harness：在设置页的“插件”页安装之后才有这个来源和它的凭据（第 2 节）。扩展只用 Node 标准库，不需要安装依赖。

当前只支持纯 Windows 环境：pi 与 HiveMemory 都在 Windows 上原生运行，按 pi 1.5.0 核对。

## 1. 前置条件与步骤一览

- HiveMemory 在 Windows 上运行：[桌面客户端](./desktop-client.md)，或按[安装与启动](./setup.md)第 4 节从仓库运行，地址为 `http://127.0.0.1:8769`；
- Windows 上的 pi（`@earendil-works/pi-coding-agent`，需要 Node 22.19 以上），PATH 上能找到 `pi` 命令，已经在 pi 中登录过模型账户。

| 步骤 | 在哪一侧 | 内容 |
|:---|:---|:---|
| 安装扩展 | HiveMemory 设置页 | 在“插件”页点 pi 的“安装”（第 2 节） |
| 重新打开并确认 | 两侧 | 重新打开终端再启动 pi，确认工具与导入（第 6 节） |

## 2. 从插件页安装

在设置页的“插件”页点 pi 的“安装”，HiveMemory 依次完成：

1. 用 pi 自己的命令安装插件目录：`pi install <插件目录>\hivememory`。插件目录在安装版中是程序目录中的 `resources\backend\integrations`，从仓库运行时是仓库的 `integrations`。pi 把这个本地路径写入个人设置（`%USERPROFILE%\.pi\agent\settings.json` 的 `packages`），以后启动时原地加载，不复制文件。pi 中已经装着另一处的 HiveMemory 插件（例如另一个 checkout）时先把它移除，否则同名工具会注册两次。
2. 登记接入来源 `harness:pi`，初始只能声明 Agent `pi`，凭据由 HiveMemory 生成。
3. `pi` 还没有访问记录时，为它补一条只开放检索、读取、写入与更新的记录（第 5 节）。
4. 把凭据写进 Windows 用户环境变量 `HIVEMEMORY_PI_TOKEN`。

第 1 步失败时什么都不登记，页面给出失败的命令、输出的最后几行和手动安装的命令。之后重新打开终端再启动 pi：Windows 程序只在启动时读取环境变量。

不要再在 pi 的 `mcp.json` 中配置 HiveMemory 的 MCP server，否则模型会同时看到两套记忆工具，经 MCP 的那一套也不会补上轮次信息。

## 3. 手动安装

插件页找不到 `pi`，或者插件命令失败时，可以在终端中手动安装（插件页给出的命令带着本机插件目录的实际位置）：

```powershell
pi install <插件目录>\hivememory
```

`pi list` 可以查看已安装的包。装好之后回到插件页，点“已用命令装好，完成登记”：HiveMemory 只完成第 2 节的第 2 至 4 步。

## 4. 配置

| 环境变量 | 默认值 | 说明 |
|:---|:---|:---|
| `HIVEMEMORY_PI_TOKEN` | 插件页写入 | 接入凭据；导入与记忆工具都使用它 |
| `HIVEMEMORY_URL` | `http://127.0.0.1:8769` | 后端地址；导入与记忆工具都使用它 |
| `HIVEMEMORY_USER_ID` | 不设置 | 以哪个用户导入与访问；不设置时使用服务端的默认用户 `default` |
| `HIVEMEMORY_AGENT_ID` | `pi` | 以哪个 Agent 导入与访问 |
| `HIVEMEMORY_CONNECTOR_STATE_DIR` | `%USERPROFILE%\.hivememory\pi-connector` | 扩展状态、日志与拒绝记录的目录 |

状态目录中的 `connector.log` 记录连接、确认与失败，只有标识与状态，不含对话内容。

## 5. 访问登记

插件页安装时登记 `harness:pi`（adapter `mcp` 与 `import`，初始只能声明 Agent `pi`），`pi` 还没有访问记录时为它补一条，获准检索、读取、写入与更新；已有的记录不改动，卸载插件也不删除它。同一个 pi 可以以多个 Agent 运行：经 `HIVEMEMORY_AGENT_ID` 换成其他 Agent 之前，先在设置页的“接入与权限”中把它加入 `harness:pi` 可声明的 Agent。改用别的用户时，按[接入 Claude Code](./claude-code-plugin.md#5-访问登记)第 5 节的写法补齐。

## 6. 确认

1. 重新打开 pi，让 agent 检索一次记忆，例如“检索一下 HiveMemory 里关于本项目的记忆”。工具返回检索结果（或“没有检索到相关记忆”）即说明连接与凭据可用。
2. 对话一轮之后，打开 `connector.log`（第 4 节），应出现 `connected`，以及这一轮的 `confirmed`。

## 7. 使用中的行为

- 写入与更新返回句柄（`draft_`、`rev_` 开头），表示已被接纳、正式记忆仍在后台生成，可以用 `memory_read` 读取。
- 一轮从发出消息开始，到 pi 不再自动继续为止：运行中追加的消息与排队的后续消息并入同一轮。被中断的一轮按取消导入，模型出错或输出达到上限的一轮按失败导入。
- 用户消息取你输入的原文：用 `/skill:...` 或提示模板时，导入的是你输入的命令，而不是展开后的正文。
- `/tree` 回到较早的条目继续时，新的一轮照常导入，离开的分支上已经导入的轮次不撤回；`/fork` 与 `/clone` 出的会话不重复导入复制来的轮次。
- 安装扩展之前的轮次不回填；`--no-session` 运行不保存会话，也不导入。
- `pi -p` 在这一轮导入得到确认之后才退出；HiveMemory 不可用时最多多等 5 秒，未导入的轮次在下次打开 pi 时补发。
- 进程崩溃或被强制结束时，进行中的一轮不导入。
- 社区的子 Agent 扩展不在支持范围内：它们另起的会话会被当作普通会话导入。

## 8. 排查

| 现象 | 原因与处理 |
|:---|:---|
| 记忆工具不出现 | 扩展没有安装（`pi list`）；HiveMemory 未运行或不在 `HIVEMEMORY_URL` 的地址：`connector.log` 中有 `connect failed`。后端恢复后，下一次发出消息时重新连接 |
| 工具返回 `invalid_credential` | pi 读到的凭据与 HiveMemory 登记的不一致：多半是终端在安装之前就打开了。重新打开终端与 pi；仍然不行时在插件页点“重新安装” |
| `connector.log` 中有 `stopped ... reason=unknown_principal` | 扩展还装着，但 HiveMemory 一侧没有 `harness:pi` 的登记（例如在插件页卸载过）：在插件页重新安装 |
| 写入返回“只能在一轮进行中调用” | 写入工具在一轮之外被调用，例如由其他扩展直接执行；在对话中由模型调用即可 |
| `connector.log` 中有 `stopped ... status=403` 或 `401` | 凭据或访问登记的问题，同上；扩展保留了未导入的轮次，修正后补发 |
| `connector.log` 中有 `stopped ... kind=retryable` | HiveMemory 暂时不可用；下次打开 pi 时补发 |

## 9. 更新与卸载

- **更新**：扩展原地读取插件目录：桌面客户端覆盖安装新版本、或从仓库运行时更新代码之后，在 pi 中执行 `/reload` 或重新打开 pi 即生效，不需要重新安装。扩展与 HiveMemory 需要一起升级。升级 pi 之后，若扩展行为异常，对照 [Import Bus](../system/import-bus.md#11-pi-扩展)第 11 节确认 pi 的扩展接口没有变化。
- **卸载**：在插件页点 pi 的“卸载”：从 pi 的个人设置中移除插件目录，删除 `harness:pi` 的登记、凭据与写入的环境变量；`pi` 的访问记录与已经写入的记忆保留。已写入会话记录的结束标记保留在原处，不影响 pi 的使用。
