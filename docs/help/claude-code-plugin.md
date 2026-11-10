---
title: Connect Claude Code
status: current
owner: project
scope: claude-code-plugin-installation-configuration-verification-and-troubleshooting
code_paths:
  - integrations/.claude-plugin/marketplace.json
  - integrations/hivememory/
  - configs/system_principals.example.yaml
  - configs/workspace_actors.example.yaml
related_contracts:
  - docs/contracts/mcp.md
related_docs:
  - docs/system/import-bus.md
  - docs/help/setup.md
  - docs/help/configuration.md
last_reviewed: 2026-11-09
---

# 接入 Claude Code

仓库提供 Claude Code 插件 `hivememory-connector`，它做三件事：

- **对话回流**：Claude Code 的每一轮对话结束后，插件把这一轮（用户消息、回复、Claude Code 本地记录的思考、工具调用与结果）导入 HiveMemory，记忆随日常使用积累。它的 hook 只做本地记录，网络提交在后台进程中完成，HiveMemory 未运行时 Claude Code 照常工作，未导入的轮次在之后被唤醒时按顺序补发。工作原理见 [Import Bus](../system/import-bus.md)。
- **记忆访问**：插件声明了 HiveMemory 的 MCP server，Claude Code 的 agent 可以调用 `memory_search` 检索记忆、`memory_read` 按 alias 读取全文、`memory_write` 写入值得长期保存的内容、`memory_update` 修正检索到的已有记忆。插件不主动向 Claude Code 注入记忆，何时检索与写入由 agent 自己决定；写入与更新前，插件的 hook 在本地为调用补上当前一轮的信息，HiveMemory 据此把写入落在这一轮所在的话题。契约见 [MCP 记忆访问](../contracts/mcp.md)。
- **使用手册 skill**：插件附带 skill `hivememory`（`skills/hivememory/SKILL.md`），在 Claude Code 中名为 `hivememory-connector:hivememory`。它说明何时检索、怎样采信记忆、什么值得写入、怎样写才能被以后找到、何时修正已有记忆，以及怎样与 Claude Code 自己的记忆文件共存。Claude Code 按 skill 的描述在需要时加载，也可以输入 `/hivememory-connector:hivememory` 手动加载。skill 的内容不区分 harness，DeepSeek Harness 等其他 harness 复制这一份使用。

两部分都以 Claude Code 自己的接入身份 `harness:claude-code` 认证，需要在 HiveMemory 与 Claude Code 两侧配置同一份接入凭据（第 2 节）。

## 1. 前置条件与步骤一览

- HiveMemory 后端已按[安装与启动](./setup.md)第 3 节或第 4 节运行；
- Claude Code 所在环境可以运行 `python3`。插件只用 Python 标准库，不需要安装依赖，也不需要安装 HiveMemory 包。

| 步骤 | 在哪一侧 | 内容 |
|:---|:---|:---|
| 生成并配置接入凭据 | 两侧 | HiveMemory 的 `.env` 与 Claude Code 的环境写同值的 `HIVEMEMORY_CLAUDE_CODE_TOKEN`（第 2 节） |
| 安装插件 | Claude Code | 添加本地 marketplace 并安装 `hivememory-connector`（第 3 节） |
| 配置连接器 | Claude Code | 至少设置 `HIVEMEMORY_URL`（第 4 节） |
| 核对访问登记 | HiveMemory | 默认登记已包含 Agent `claude_code`，自定义过登记时补齐（第 5 节） |
| 重启并确认 | 两侧 | 重启 HiveMemory、重新打开 Claude Code，确认工具与导入（第 6、7 节） |

## 2. 配置接入凭据

凭据可以在 HiveMemory 设置页的“接入与权限”中为 `harness:claude-code` 生成或轮换：HiveMemory 一侧立即生效，不需要重启后端，把页面显示的凭据写进 Claude Code 一侧的 `HIVEMEMORY_CLAUDE_CODE_TOKEN` 即可。HiveMemory 一侧的同名环境变量有值时优先于生成的凭据，页面上不能轮换。也可以按下面的方式自己生成一份，写入两侧的环境变量。

先生成一份随机凭据，例如：

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

把同一个值写入两侧，变量名都是 `HIVEMEMORY_CLAUDE_CODE_TOKEN`：

| 一侧 | 写在哪里 |
|:---|:---|
| HiveMemory | 仓库根 `.env`（与其他密钥放在一起，Docker 路径同样读取它），或启动后端的环境 |
| Claude Code | 启动 Claude Code 的环境，或 `~/.claude/settings.json` 的 `env` |

变量名带上 harness，是为了让多个 harness 共用一套环境变量时（例如 dsh 插件的 `HIVEMEMORY_DSH_TOKEN` 与本插件同在一个 Windows 用户环境中）各读各的凭据，不会拿别的 harness 的凭据冒充 `harness:claude-code`。HiveMemory 先读进程环境变量、没有时再读 `.env`，所以后端与 Claude Code 在同一套环境中运行时，设置一次即可同时供两侧使用。

修改后重启 HiveMemory，并重新打开 Claude Code。凭据没有配置或两侧不一致时，导入与工具调用都以 `invalid_credential` 被拒绝：HiveMemory 照常启动并在启动日志中提示凭据环境变量未设置，连接器保留进度等待补发，MCP 工具返回错误（第 9 节）。凭据只用于防止本机其他程序冒用 Claude Code 的身份，不进入日志与错误响应。

## 3. 安装插件

`integrations/.claude-plugin/marketplace.json` 声明了一个本地 marketplace `hivememory-local`，只供本机安装。在 Claude Code 中执行（把路径换成本机仓库的实际位置）：

```text
/plugin marketplace add <HiveMemory 仓库路径>/integrations
/plugin install hivememory-connector@hivememory-local
```

插件挂载 `Stop`、`StopFailure`、`SessionStart`、`SessionEnd` 与只匹配本插件写入、更新工具的 `PreToolUse` 五个 hook，在 `.mcp.json` 中声明 MCP server `hivememory`，并附带 skill `hivememory`。安装时 Claude Code 保存的是插件的副本，仓库中的插件更新后要重新更新插件（第 10 节）。

## 4. 配置连接器

连接器与 MCP server 从同一组环境变量读取配置。hook 与 MCP 连接都由 Claude Code 发起，因此在启动 Claude Code 的环境中设置（也可以写在 `~/.claude/settings.json` 的 `env` 中）：

| 环境变量 | 默认值 | 说明 |
|:---|:---|:---|
| `HIVEMEMORY_URL` | `http://localhost:8000` | HiveMemory 后端地址。默认值对应 Docker 路径；本地开发路径（含脚本的 `serve` 模式）的后端在 `8769`，需要设为 `http://localhost:8769`。MCP 只接受本机地址（`localhost`、`127.0.0.1` 或 `[::1]`，带端口） |
| `HIVEMEMORY_CLAUDE_CODE_TOKEN` | 不设置 | 接入凭据，见第 2 节 |
| `HIVEMEMORY_USER_ID` | 不设置 | 以哪个用户导入与访问；不设置时使用服务端的默认用户 `default` |
| `HIVEMEMORY_AGENT_ID` | `claude_code` | 以哪个 Agent 导入与访问；不能使用保留的 `system` |
| `HIVEMEMORY_CONNECTOR_STATE_DIR` | 见第 7 节 | 连接器状态、日志与拒绝记录的目录 |

例如本地开发路径：

```bash
export HIVEMEMORY_URL=http://localhost:8769
export HIVEMEMORY_CLAUDE_CODE_TOKEN=<第 2 节生成的凭据>
```

## 5. 访问登记

导入与 MCP 请求都经统一认证网关，请求的用户与 Agent 必须在 `configs/workspace_actors.yaml` 中获准进入 Workspace。从模板生成的登记已经为默认用户 `default` 登记了 `claude_code`，使用默认配置时不需要修改。这条记录获准 `memory.read`、`memory.search`、`memory.write` 与 `memory.update`，分别对应四个 MCP 工具；导入只要求准入、不需要任何 actor 操作；精确记录也使这个 Agent 不继承默认用户的 `memory.run` 与 `agent.call`。更早生成或自定义过的登记，需要手动为它加上 `memory.write` 与 `memory.update`，否则写入与更新返回 403（`workspace.operation_denied`）。

- `HIVEMEMORY_USER_ID` 设为 `default` 以外的用户时，需要为该用户登记记录，否则导入与工具调用返回 403，连接器停止提交并保留进度；
- 更换 `HIVEMEMORY_AGENT_ID` 时，先在设置页的“接入与权限”中把新的 Agent 加入 `harness:claude-code` 可声明的 Agent，否则请求以 403（`reason=agent_not_allowed_for_principal`）被拒；新的 Agent 由默认的用户级记录覆盖，如果希望它的权限与 `claude_code` 相同，在同一页为它增加记录；
- 在设置页修改登记立即生效；直接修改登记文件后重启 HiveMemory。登记规则见[配置指南](./configuration.md#51-访问登记)。

## 6. 确认记忆访问

配置完成并重新打开 Claude Code 后，在 Claude Code 中执行 `/mcp`：插件的 `hivememory` 应显示已连接，并提供 `memory_search`、`memory_read`、`memory_write` 与 `memory_update` 四个工具。

- 工具说明要求 agent 把检索结果当作带日期的历史证据：仓库中的代码与文档、Claude Code 自己的项目指令与记忆文件优先于检索结果。
- 写入与更新按 Claude Code 的权限设置确认：确认框中的参数带有 `turn`，是插件补上的当前一轮信息，不需要修改。结果是一个 pending alias（写入以 `draft_` 开头，更新以 `rev_` 开头），表示已被接纳、正式记忆仍在后台生成；可以用 `memory_read` 读取它，生成完成后读取会转到正式记忆。工具结果提示缺少当前一轮的信息（`request.turn_required`）时，说明插件的 hook 没有运行，检查插件是否已更新并重启了 Claude Code。
- HiveMemory 未运行时，Claude Code 重试后把 `hivememory` 标为连接失败，四个工具不出现，工作照常进行；HiveMemory 恢复后，在 Claude Code 中执行 `/mcp reconnect all` 或重新打开会话。
- 经 MCP 读取的正式记忆会记录一次引用；用过检索工具的轮次导入后，生成只保留一行检索摘要，不会把取回的记忆再提取一遍；写入与更新的调用不进入生成材料，显式写入的内容也不会被再提取一遍。

## 7. 确认导入

连接器的状态目录依次取 `HIVEMEMORY_CONNECTOR_STATE_DIR`、Claude Code 为插件提供的数据目录（`CLAUDE_PLUGIN_DATA`），都没有时使用 `~/.hivememory/claude-code-connector`。目录中：

- `connector.log`：每一轮的确认（`confirmed turn=... status=accepted`）、停止与拒绝，只记录标识与状态，不记录对话内容；
- `rejections.jsonl`：被放弃的轮次，记录会话与轮次标识、HTTP 状态码与错误码。

完成一轮对话后，`connector.log` 中出现这一轮的 `confirmed` 行，即表示导入成功。

## 8. 使用中的行为

- 只导入插件首次运行之后开始的会话；安装前已存在的会话不回填，恢复这类会话后只导入新的轮次。
- 一轮在结束后才导入：正常结束为 `completed`，因 API 错误结束为 `failed`，被用户中断、被下一条消息打断或随会话结束而结束的为 `cancelled`。
- agent 停下之后被后台命令的完成通知、Monitor 事件、后台子 Agent 交回的报告或桌面端注入的消息唤醒并继续工作时，这段续作单独作为一轮导入，落在上一轮的话题；子 Agent 交回的报告随这一轮进入记忆材料。由另一个会话派生出来的会话，以它收到的交接说明作为第一轮的用户消息导入。
- 连接器没有常驻进程，只在 hook 触发时被唤醒：一轮结束或会话结束时同步当前会话，会话开始（含恢复会话）时补发全部会话中积压的轮次。HiveMemory 停止期间积压的轮次，会在它恢复后的下一次唤醒时按顺序补发。
- HiveMemory 暂不可用（连接失败、503）或连接器配置有误（400、403、404）时，连接器保留进度、停止提交，修正后在下一次唤醒时补发；配置有误时日志中的 `stopped` 行带有状态码与错误码，凭据问题为 403 `invalid_credential`。
- 单轮内容超过服务端的请求体上限（413，默认 8 MiB）或格式被拒绝（422）时，这一轮写入 `rejections.jsonl` 后跳过，不阻塞后续轮次；同一轮连续三次得到 500 也会被跳过。

## 9. 排查

先在 Claude Code 中执行 `/mcp` 查看 `hivememory` 的状态，再看连接器的 `connector.log`（第 7 节）。

| 现象 | 原因与处理 |
|:---|:---|
| `/mcp` 显示 `hivememory` 连接失败，工具不出现 | HiveMemory 未运行或 `HIVEMEMORY_URL` 不对（本地开发与 `serve` 模式在 `8769`，Docker 在 `8000`）。启动后在 Claude Code 中执行 `/mcp reconnect all` 或重新打开会话 |
| 工具调用返回 `invalid_credential`，`connector.log` 的 `stopped` 行带 403 `invalid_credential` | 两侧凭据不一致或没有配置：两侧都是 `HIVEMEMORY_CLAUDE_CODE_TOKEN`，HiveMemory 一侧先取进程环境变量、再取 `.env`（后端启动日志会提示它未设置）。改正后重启 HiveMemory、重新打开 Claude Code；连接器保留了进度，下一次唤醒时补发未导入的轮次 |
| 导入返回 400 `request.principal_required` | 服务端已更新而插件仍是旧版本：更新插件（第 10 节）后重启 Claude Code，积压的轮次随后补发 |
| 工具调用返回 `operation_not_allowed` | `workspace_actors.yaml` 中该 Agent 的白名单缺少对应的 `memory.*` 操作（第 5 节）；修改后重启 HiveMemory |
| 写入或更新返回 `request.turn_required` | 插件的 `PreToolUse` hook 没有运行：确认插件已更新、已重启 Claude Code |
| 工具调用返回 `request.workspace_not_found` | `X-Workspace-ID` 只接受 `main_workspace` |
| MCP 请求返回 421 | MCP 端点只接受带端口的本机地址（`localhost`、`127.0.0.1`、`[::1]`）；`HIVEMEMORY_URL` 改用这些地址 |

## 10. 更新与卸载

- **更新**：仓库中的插件更新后，在 `/plugin` 中更新该插件，或执行 `claude plugin marketplace update hivememory-local` 与 `claude plugin update hivememory-connector@hivememory-local`，再重启 Claude Code。插件目录从 `integrations/claude-code/` 改名为 `integrations/hivememory/` 之后（与 Codex 合用一个插件目录），先执行这两条命令更新一次即可：插件名与 marketplace 名不变，插件数据目录与其中的游标保留。插件与 HiveMemory 需要一起升级：新插件导入的轮次可能带写入与更新的标记，旧版 HiveMemory 不认识它们，会整轮拒绝。
- **卸载**：执行 `claude plugin uninstall hivememory-connector@hivememory-local`，再重启 Claude Code。
