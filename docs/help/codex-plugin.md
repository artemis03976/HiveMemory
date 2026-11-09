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
  - configs/system_principals.yaml
  - configs/workspace_actors.yaml
related_contracts:
  - docs/contracts/mcp.md
related_docs:
  - docs/system/import-bus.md
  - docs/help/setup.md
  - docs/help/claude-code-plugin.md
last_reviewed: 2026-11-08
---

# 接入 Codex

仓库中的插件 `hivememory-connector`（`integrations/hivememory/`）同时是 Claude Code 与 Codex 的插件，两者共用连接器与使用手册 skill。在 Codex 中它做三件事：

- **记忆工具**：插件声明 HiveMemory 的 MCP server，Codex 中的 agent 经 `memory_search`（检索）、`memory_read`（按 alias 读取全文）、`memory_write`（写入）与 `memory_update`（修正已有记忆）四个工具访问记忆。写入与更新需要这一轮的信息，HiveMemory 据此把写入落在这一轮所在的话题：插件的 hook 在调用前自动补上，模型自己填写的会被覆盖。
- **对话回流**：Codex 中每一轮结束后，插件把这一轮（用户消息、回复、思考、命令与工具调用及结果、子 Agent 交回的消息）导入 HiveMemory。网络提交在后台进行，HiveMemory 未运行时 Codex 照常工作，未导入的轮次之后按顺序补发。工作原理见 [Import Bus](../system/import-bus.md#10-codex-插件)第 10 节。
- **使用手册 skill**：`skills/hivememory/SKILL.md` 随插件加载，引导 agent 何时检索、写入与修正记忆。

两部分都以 Codex 自己的接入身份 `harness:codex` 认证，以 Agent `codex` 访问，需要在 HiveMemory 与 Codex 两侧配置同一份接入凭据（第 2 节）。插件只用 Python 标准库，不需要安装依赖。

当前只支持纯 Windows 环境：Codex 与 HiveMemory 都在 Windows 上原生运行，按 Codex CLI 0.167.0 核对。

## 1. 前置条件与步骤一览

- HiveMemory 后端已在 Windows 上原生运行，地址为 `http://127.0.0.1:8769`（[安装与启动](./setup.md)）；
- Windows 上的 Codex CLI 或桌面应用，两者共用 `CODEX_HOME`（默认 `%USERPROFILE%\.codex`）中的插件、hook 与配置；
- 命令行中可以运行 `python`（Windows 的 Python 安装包提供 `python`，插件的 hook 以它运行连接器）。

| 步骤 | 在哪一侧 | 内容 |
|:---|:---|:---|
| 生成并配置接入凭据 | 两侧 | HiveMemory 的 `.env` 与 Windows 用户环境变量写同值的 `HIVEMEMORY_CODEX_TOKEN`（第 2 节） |
| 安装插件 | Codex | 添加本地 marketplace 并安装 `hivememory-connector`（第 3 节） |
| 信任 hook | Codex | 在 Codex 中审阅并信任插件的 hook（第 4 节） |
| 配置 | Codex | 设置导入用的 `HIVEMEMORY_URL`（第 5 节） |
| 核对访问登记 | HiveMemory | 默认登记已包含 Agent `codex`（第 6 节） |
| 重启并确认 | 两侧 | 重启 HiveMemory、重新打开 Codex，确认工具与导入（第 7 节） |

## 2. 配置接入凭据

先生成一份随机凭据，例如：

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

把同一个值写入两侧，变量名都是 `HIVEMEMORY_CODEX_TOKEN`：

| 一侧 | 写在哪里 |
|:---|:---|
| HiveMemory | 仓库根 `.env`，或启动后端的环境 |
| Codex | Windows 用户环境变量（`setx HIVEMEMORY_CODEX_TOKEN <凭据>`） |

变量名带上 harness，多个 harness 共用一套 Windows 用户环境变量时各读各的凭据。HiveMemory 先读进程环境变量、没有时再读 `.env`，所以后端也从同一套用户环境启动时，设置一次即可同时供两侧使用。

修改后重启 HiveMemory。Windows 程序只在启动时读取环境变量：`setx` 之后要完全退出 Codex（桌面应用从托盘退出）、终端与编辑器再打开。凭据没有配置或两侧不一致时，导入与工具调用都以 `invalid_credential` 被拒绝，连接器保留进度等待补发（第 9 节）。

## 3. 安装插件

`integrations/.claude-plugin/marketplace.json` 声明了本地 marketplace `hivememory-local`，Codex 同样读取它。在终端中执行（把路径换成本机仓库的实际位置）：

```powershell
codex plugin marketplace add <HiveMemory 仓库路径>\integrations
codex plugin add hivememory-connector@hivememory-local
```

Codex 按插件目录中的 `.codex-plugin/plugin.json` 加载：使用手册 skill、`codex.mcp.json` 中的 MCP server `hivememory`，以及 `hooks/codex.json` 中的五个 hook（`SessionStart`、`Stop`、`Interrupt`、`SessionEnd` 与只匹配写入、更新工具的 `PreToolUse`）。Claude Code 用的 `.mcp.json` 与 `hooks/hooks.json` 不会被 Codex 读取。安装时 Codex 把插件复制到 `CODEX_HOME\plugins\cache\` 下，仓库中的插件更新后要重新安装（第 10 节）。

## 4. 信任 hook

插件的 hook 要经你审阅信任后才运行。安装后第一次打开 Codex 的交互界面时，它提示 “Hooks need review”，选择审阅并信任 `hivememory-connector` 的五个 hook；之后也可以用 `/hooks` 查看与管理。插件内容变化后（例如更新插件），hook 变为“已修改”，需要再次信任。

hook 没有信任时，对话不会回流；写入与更新缺少这一轮的信息，被 HiveMemory 以 `request.turn_required` 拒绝。

## 5. 配置

| 环境变量 | 默认值 | 说明 |
|:---|:---|:---|
| `HIVEMEMORY_CODEX_TOKEN` | 不设置 | 接入凭据，见第 2 节；导入与 MCP 工具都使用它 |
| `HIVEMEMORY_URL` | `http://localhost:8000` | 导入使用的后端地址，设为 `http://127.0.0.1:8769` |
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

随仓库发布的 `configs/system_principals.yaml` 登记了 `harness:codex`（adapter `mcp` 与 `import`，凭据变量 `HIVEMEMORY_CODEX_TOKEN`），`configs/workspace_actors.yaml` 为默认用户 `default` 登记了 Agent `codex`，获准检索、读取、写入与更新。使用默认配置时不需要修改；两份配置在后端启动时装载，更新到包含这些登记的版本之后要重启后端。自定义过登记或改用别的用户、Agent 时，按 [接入 Claude Code](./claude-code-plugin.md#5-访问登记)第 5 节的写法补齐。

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
| 工具返回 `invalid_credential` | 两侧凭据不一致或没有配置：两侧都是 `HIVEMEMORY_CODEX_TOKEN`。改正后重启 HiveMemory、完全退出并重新打开 Codex |
| 写入或更新返回 `request.turn_required` | hook 没有运行：在 Codex 中用 `/hooks` 确认 `hivememory-connector` 的 hook 已信任且启用；更新插件后需要重新信任 |
| 写入要求逐次确认，或无头运行时写入失败 | 自己在 `config.toml` 中声明了 `[mcp_servers.hivememory]` 而没有写 `default_tools_approval_mode = "approve"`（第 5 节） |
| `connector.log` 中没有 `confirmed`，有 `stopped ... status=403` | 凭据或访问登记的问题，同上；连接器保留了进度，修正后补发 |
| `connector.log` 中什么都没有 | hook 没有运行：检查信任状态，以及命令行中能否运行 `python` |

## 10. 更新与卸载

- **更新**：仓库中的插件更新后，重新执行 `codex plugin add hivememory-connector@hivememory-local` 把新版本复制进缓存，再在 Codex 中重新信任 hook。插件与 HiveMemory 需要一起升级。
- **卸载**：执行 `codex plugin remove hivememory-connector@hivememory-local`。
