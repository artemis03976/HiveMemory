---
title: Connect Codex
status: current
owner: project
scope: codex-plugin-installation-configuration-verification-and-troubleshooting
code_paths:
  - integrations/.claude-plugin/marketplace.json
  - integrations/hivememory/.codex-plugin/plugin.json
  - integrations/hivememory/codex.mcp.json
  - integrations/hivememory/hooks/codex.json
  - integrations/hivememory/connector/codex/
  - integrations/hivememory/hivememory_hook.cmd
  - src/hivememory/system/plugins/
related_contracts:
  - docs/contracts/mcp.md
related_docs:
  - docs/system/import-bus.md
  - docs/help/setup.md
  - docs/help/claude-code-plugin.md
last_reviewed: 2026-11-12
---

# 接入 Codex

仓库中的插件 `hivememory-connector`（`integrations/hivememory/`）同时是 Claude Code 与 Codex 的插件，两者共用连接器与使用手册 skill。在 Codex 中它做三件事：

- **记忆工具**：插件声明 HiveMemory 的 MCP server，Codex 中的 agent 经 `memory_search`（检索）、`memory_read`（按 alias 读取全文）、`memory_write`（写入）与 `memory_update`（修正已有记忆）四个工具访问记忆。写入与更新需要这一轮的信息，HiveMemory 据此把写入落在这一轮所在的话题：插件的 hook 在调用前自动补上，模型自己填写的会被覆盖。
- **对话回流**：Codex 中每一轮结束后，插件把这一轮（用户消息、回复、思考、命令与工具调用及结果、子 Agent 交回的消息）导入 HiveMemory。网络提交在后台进行，HiveMemory 未运行时 Codex 照常工作，未导入的轮次之后按顺序补发。工作原理见 [Import Bus](../system/import-bus.md#10-codex-插件)第 10 节。
- **使用手册 skill**：`skills/hivememory/SKILL.md` 随插件加载，引导 agent 何时检索、写入与修正记忆。

三部分都以 Codex 自己的接入来源 `harness:codex` 认证，以 Agent `codex` 访问。新安装的 HiveMemory 不连接任何 harness：在设置页的“插件”页安装之后才有这个来源和它的凭据（第 2 节）。

当前只支持纯 Windows 环境：Codex 与 HiveMemory 都在 Windows 上原生运行，按 Codex CLI 0.167.0 核对。

## 1. 前置条件与步骤一览

- HiveMemory 在 Windows 上运行：[桌面客户端](./desktop-client.md)，或按[安装与启动](./setup.md)第 4 节从仓库运行，地址为 `http://127.0.0.1:8769`；
- Windows 上的 Codex CLI 或桌面应用，两者共用 `CODEX_HOME`（默认 `%USERPROFILE%\.codex`）中的插件、hook 与配置；PATH 上能找到 `codex` 命令。

| 步骤 | 在哪一侧 | 内容 |
|:---|:---|:---|
| 安装插件 | HiveMemory 设置页 | 在“插件”页点 Codex 的“安装”（第 2 节） |
| 重新打开并信任 hook | Codex | 完全退出 Codex 再打开，审阅并信任插件的 hook（第 4 节） |
| 确认 | 两侧 | 记忆工具与导入（第 7 节） |

## 2. 从插件页安装

在设置页的“插件”页点 Codex 的“安装”，HiveMemory 依次完成：

1. 用 Codex 自己的命令安装插件：本地 marketplace `hivememory-local` 指向插件目录（安装版是程序目录中的 `resources\backend\integrations`，从仓库运行时是仓库的 `integrations`），再执行 `codex plugin add hivememory-connector@hivememory-local`，把插件复制到 `CODEX_HOME\plugins\cache\` 下。marketplace 已经指向别处时先移除再添加。
2. 登记接入来源 `harness:codex`，初始只能声明 Agent `codex`，凭据由 HiveMemory 生成。
3. `codex` 还没有访问记录时，为它补一条只开放检索、读取、写入与更新的记录（第 6 节）。
4. 把凭据写进 Windows 用户环境变量 `HIVEMEMORY_CODEX_TOKEN`，把运行 HiveMemory 的 Python 写进 `HIVEMEMORY_PYTHON`。插件的 hook 经 `hivememory_hook.cmd` 用它运行连接器，电脑上不需要另装 Python；没有这个变量时用 PATH 上的 `python`。

第 1 步失败时什么都不登记，页面给出失败的命令、输出的最后几行和手动安装的命令。之后完全退出 Codex（桌面应用从托盘退出）、终端与编辑器再打开：Windows 程序只在启动时读取环境变量。

Codex 按插件目录中的 `.codex-plugin/plugin.json` 加载：使用手册 skill、`codex.mcp.json` 中的 MCP server `hivememory`，以及 `hooks/codex.json` 中的五个 hook（`SessionStart`、`Stop`、`Interrupt`、`SessionEnd` 与只匹配写入、更新工具的 `PreToolUse`）。Claude Code 用的 `.mcp.json` 与 `hooks/hooks.json` 不会被 Codex 读取。

## 3. 手动安装

插件页找不到 `codex`，或者插件命令失败时，可以在终端中手动安装（插件页给出的命令带着本机插件目录的实际位置）：

```powershell
codex plugin marketplace add <插件目录>
codex plugin add hivememory-connector@hivememory-local
```

装好之后回到插件页，点“已用命令装好，完成登记”：HiveMemory 只完成第 2 节的第 2 至 4 步。

## 4. 信任 hook

插件的 hook 要经你审阅信任后才运行。安装后第一次打开 Codex 的交互界面时，它提示 “Hooks need review”，选择审阅并信任 `hivememory-connector` 的五个 hook；之后也可以用 `/hooks` 查看与管理。插件内容变化后（例如更新插件），hook 变为“已修改”，需要再次信任。

hook 没有信任时，对话不会回流；写入与更新缺少这一轮的信息，被 HiveMemory 以 `request.turn_required` 拒绝。

## 5. 配置

| 环境变量 | 默认值 | 说明 |
|:---|:---|:---|
| `HIVEMEMORY_CODEX_TOKEN` | 插件页写入 | 接入凭据；导入与 MCP 工具都使用它 |
| `HIVEMEMORY_PYTHON` | 插件页写入 | hook 运行连接器的 Python；没有设置时用 PATH 上的 `python` |
| `HIVEMEMORY_URL` | `http://127.0.0.1:8769` | 导入使用的后端地址 |
| `HIVEMEMORY_USER_ID` | 不设置 | 以哪个用户导入与访问；不设置时使用服务端的默认用户 `default` |
| `HIVEMEMORY_AGENT_ID` | `codex` | 以哪个 Agent 导入；MCP 工具的 Agent 固定为 `codex` |
| `HIVEMEMORY_CONNECTOR_STATE_DIR` | Codex 的插件数据目录 | 连接器状态、日志与拒绝记录的目录，默认 `CODEX_HOME\plugins\data\hivememory-connector-hivememory-local` |

MCP server 的地址固定为 `http://127.0.0.1:8769/mcp`（Codex 不展开插件 MCP 配置中的环境变量），四个记忆工具不逐次审批。后端不在这个地址时，在 `CODEX_HOME\config.toml` 中声明同名的 server；它会整条替换插件的定义，因此要连同凭据变量、请求头与审批模式一起写出：

```toml
[mcp_servers.hivememory]
url = "http://127.0.0.1:<端口>/mcp"
bearer_token_env_var = "HIVEMEMORY_CODEX_TOKEN"
http_headers = { "X-Principal-ID" = "harness:codex", "X-Agent-ID" = "codex" }
default_tools_approval_mode = "approve"
```

## 6. 访问登记

插件页安装时登记 `harness:codex`（adapter `mcp` 与 `import`，初始只能声明 Agent `codex`），`codex` 还没有访问记录时为它补一条，获准检索、读取、写入与更新；已有的记录不改动，卸载插件也不删除它。导入可以经 `HIVEMEMORY_AGENT_ID` 以其他 Agent 运行，先在设置页的“接入与权限”中把它加入 `harness:codex` 可声明的 Agent；MCP 工具的 Agent 固定为 `codex`。改用别的用户时，按 [接入 Claude Code](./claude-code-plugin.md#5-访问登记)第 5 节的写法补齐。

## 7. 确认

1. 重新打开 Codex，让 agent 检索一次记忆，例如“检索一下 HiveMemory 里关于本项目的记忆”。工具返回检索结果（或“没有检索到相关记忆”）即说明 MCP 与凭据可用。
2. 对话一轮之后，打开连接器日志 `connector.log`（状态目录见第 5 节），应出现 `registered`，以及这一轮的 `confirmed`。

## 8. 使用中的行为

- 写入与更新返回句柄（`draft_`、`rev_` 开头），表示已被接纳、正式记忆仍在后台生成，可以用 `memory_read` 读取。
- 正常结束的一轮在轮次结束后很快导入；失败的一轮没有结束通知，在下一轮结束、会话结束或下一次打开会话时导入；被中断的一轮按取消导入。
- 子 Agent 中的写入落在父会话中派生它的那一轮；子 Agent 自己的会话不导入，它交回给父 Agent 的消息随父会话的那一轮导入。
- 续会话（`codex resume` 或 `codex exec resume`）的新轮次接着导入，之前的轮次不重复导入。
- 插件安装之前就开始的会话不回填已经结束的轮次；首次见到时仍在进行的一轮（例如安装后第一个 `codex exec` 会话的第一轮）照常导入。

## 9. 排查

| 现象 | 原因与处理 |
|:---|:---|
| 记忆工具不出现 | 插件没有安装或已禁用（`codex plugin list`）；HiveMemory 未运行或不在 `127.0.0.1:8769`（第 5 节） |
| 工具返回 `invalid_credential` | Codex 读到的凭据与 HiveMemory 登记的不一致：多半是安装之后没有完全退出 Codex。完全退出再打开；仍然不行时在插件页点“重新安装” |
| 导入返回 403 `unknown_principal` | 插件还装着，但 HiveMemory 一侧没有 `harness:codex` 的登记（例如在插件页卸载过）：在插件页重新安装 |
| 写入或更新返回 `request.turn_required` | hook 没有运行：在 Codex 中用 `/hooks` 确认 `hivememory-connector` 的 hook 已信任且启用；更新插件后需要重新信任 |
| 写入要求逐次确认，或无头运行时写入失败 | 自己在 `config.toml` 中声明了 `[mcp_servers.hivememory]` 而没有写 `default_tools_approval_mode = "approve"`（第 5 节） |
| `connector.log` 中没有 `confirmed`，有 `stopped ... status=403` | 凭据或访问登记的问题，同上；连接器保留了进度，修正后补发 |
| `connector.log` 中什么都没有 | hook 没有运行：检查信任状态；`HIVEMEMORY_PYTHON` 没有设置时，检查命令行中能否运行 `python` |

## 10. 更新与卸载

- **更新**：Codex 保存的是插件的副本，插件目录更新后要重新安装才能读到。桌面客户端覆盖安装新版本之后，后端启动时自动为插件页装过的 Codex 重新安装插件，失败时插件页把它标为“需要更新”；从仓库运行时，更新代码后在插件页点“重新安装”，或重新执行 `codex plugin add hivememory-connector@hivememory-local`。之后在 Codex 中重新信任 hook。插件与 HiveMemory 需要一起升级。
- **卸载**：在插件页点 Codex 的“卸载”：卸载 Codex 中的插件、移除本地 marketplace，删除 `harness:codex` 的登记、凭据与写入的环境变量；`codex` 的访问记录与已经写入的记忆保留。
