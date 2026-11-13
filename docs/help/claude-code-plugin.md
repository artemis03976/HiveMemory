---
title: Connect Claude Code
status: current
owner: project
scope: claude-code-plugin-installation-configuration-verification-and-troubleshooting
code_paths:
  - integrations/.claude-plugin/marketplace.json
  - integrations/hivememory/
  - src/hivememory/system/plugins/
related_contracts:
  - docs/contracts/mcp.md
related_docs:
  - docs/system/import-bus.md
  - docs/help/setup.md
  - docs/help/configuration.md
last_reviewed: 2026-11-12
---

# 接入 Claude Code

仓库提供 Claude Code 插件 `hivememory-connector`，它做三件事：

- **对话回流**：Claude Code 的每一轮对话结束后，插件把这一轮（用户消息、回复、Claude Code 本地记录的思考、工具调用与结果）导入 HiveMemory，记忆随日常使用积累。它的 hook 只做本地记录，网络提交在后台进程中完成，HiveMemory 未运行时 Claude Code 照常工作，未导入的轮次在之后被唤醒时按顺序补发。工作原理见 [Import Bus](../system/import-bus.md)。
- **记忆访问**：插件声明了 HiveMemory 的 MCP server，Claude Code 的 agent 可以调用 `memory_search` 检索记忆、`memory_read` 按 alias 读取全文、`memory_write` 写入值得长期保存的内容、`memory_update` 修正检索到的已有记忆。插件不主动向 Claude Code 注入记忆，何时检索与写入由 agent 自己决定；写入与更新前，插件的 hook 在本地为调用补上当前一轮的信息，HiveMemory 据此把写入落在这一轮所在的话题。契约见 [MCP 记忆访问](../contracts/mcp.md)。
- **使用手册 skill**：插件附带 skill `hivememory`（`skills/hivememory/SKILL.md`），在 Claude Code 中名为 `hivememory-connector:hivememory`。它说明何时检索、怎样采信记忆、什么值得写入、怎样写才能被以后找到、何时修正已有记忆，以及怎样与 Claude Code 自己的记忆文件共存。Claude Code 按 skill 的描述在需要时加载，也可以输入 `/hivememory-connector:hivememory` 手动加载。skill 的内容不区分 harness，DeepSeek Harness 等其他 harness 用的是同一份。skill 的语言随 HiveMemory 的语言设置：后端每次启动时从插件目录的 `skill-languages/<语言>/SKILL.md` 写入 `skills/hivememory/SKILL.md`（[配置指南](./configuration.md#54-语言)）。Claude Code 保存的是插件的副本，改语言后要在插件页重新安装一次。

三部分都以 Claude Code 自己的接入来源 `harness:claude-code` 认证。新安装的 HiveMemory 不连接任何 harness：在设置页的“插件”页安装之后才有这个来源和它的凭据（第 2 节）。

## 1. 前置条件与步骤一览

- HiveMemory 在 Windows 上运行：[桌面客户端](./desktop-client.md)，或按[安装与启动](./setup.md)第 4 节从仓库运行；
- 这台电脑上装有 Claude Code，PATH 上能找到 `claude` 命令。

| 步骤 | 在哪一侧 | 内容 |
|:---|:---|:---|
| 安装插件 | HiveMemory 设置页 | 在“插件”页点 Claude Code 的“安装”（第 2 节） |
| 重新打开 Claude Code | Claude Code | 完全退出（包括已经打开的终端）再打开，读到新写入的环境变量 |
| 确认 | 两侧 | 记忆工具与导入（第 6、7 节） |

## 2. 从插件页安装

在设置页的“插件”页点 Claude Code 的“安装”，HiveMemory 依次完成：

1. 用 Claude Code 自己的命令安装插件：本地 marketplace `hivememory-local` 指向插件目录（安装版是程序目录中的 `resources\backend\integrations`，从仓库运行时是仓库的 `integrations`），再安装 `hivememory-connector@hivememory-local`。marketplace 已经指向别处（例如另一个 checkout）时先移除再添加；插件已经装过时改为更新。随后在 Claude Code 的用户设置中放行写入与更新两个工具（见下文）。
2. 登记接入来源 `harness:claude-code`，初始只能声明 Agent `claude_code`，凭据由 HiveMemory 生成。
3. `claude_code` 还没有访问记录时，为它补一条只开放检索、读取、写入与更新的记录（第 5 节）。
4. 把凭据写进 Windows 用户环境变量 `HIVEMEMORY_CLAUDE_CODE_TOKEN`，把运行 HiveMemory 的 Python 写进 `HIVEMEMORY_PYTHON`。插件的 hook 用它运行连接器，电脑上不需要另装 Python。

**写入与更新的放行规则**：插件的 `PreToolUse` hook 在写入与更新之前给参数补上当前一轮的信息。auto 模式审查的是模型写的参数，hook 改写之后审查不覆盖实际执行的参数，Claude Code 以 “a hook changed this call's input after auto mode reviewed it” 拒绝执行，hook 自己给出放行也一样；用户设置中的 allow 规则匹配时不经这次审查。插件不能自带权限设置，所以第 1 步在用户设置 `~/.claude/settings.json`（设置了 `CLAUDE_CONFIG_DIR` 时是那个目录中的 `settings.json`）的 `permissions.allow` 中加入 `mcp__plugin_hivememory-connector_hivememory__memory_write` 与 `mcp__plugin_hivememory-connector_hivememory__memory_update`，其余设置与规则原样保留，卸载时删除这两条。检索与读取的参数不被改写，照常经 auto 模式审查。有了这两条规则，其他权限模式下写入与更新也不再逐次确认。设置文件不是有效的 JSON 时不改动它，安装以“设置文件读不懂”失败，页面给出文件位置，修好后重试。

第 1 步失败时什么都不登记，页面给出失败的命令、输出的最后几行和手动安装的命令。之后完全退出 Claude Code 再打开：已经在运行的程序与终端读不到新写入的环境变量。

插件挂载 `Stop`、`StopFailure`、`SessionStart`、`SessionEnd` 与只匹配本插件写入、更新工具的 `PreToolUse` 五个 hook，在 `.mcp.json` 中声明 MCP server `hivememory`，并附带 skill `hivememory`。安装时 Claude Code 保存的是插件的副本（第 10 节说明怎样更新）。

## 3. 手动安装

插件页找不到 `claude`，或者插件命令失败时，可以在 Claude Code 中手动安装（插件页给出的命令带着本机插件目录的实际位置）：

```text
/plugin marketplace add <插件目录>
/plugin install hivememory-connector@hivememory-local
```

装好之后回到插件页，点“已用命令装好，完成登记”：HiveMemory 只完成第 2 节的第 2 至 4 步，不写放行规则。在 auto 模式下使用时，自己把两条规则加进 `~/.claude/settings.json`（已有 `permissions.allow` 时并进去）：

```json
{
  "permissions": {
    "allow": [
      "mcp__plugin_hivememory-connector_hivememory__memory_write",
      "mcp__plugin_hivememory-connector_hivememory__memory_update"
    ]
  }
}
```

插件页不能代为安装时（HiveMemory 不在 Windows 上运行，例如 Docker），在生效的 `configs/system_principals.yaml` 中登记 `harness:claude-code`（adapter `mcp` 与 `import`，`allowed_agent_ids: ["claude_code"]`，`credential_env: "HIVEMEMORY_CLAUDE_CODE_TOKEN"`），在 `configs/workspace_actors.yaml` 中为 `claude_code` 登记四项记忆操作，再生成一份随机凭据（例如 `python -c "import secrets; print(secrets.token_urlsafe(32))"`），把同一个值写进 HiveMemory 的 `.env` 与 Claude Code 的环境（或 `~/.claude/settings.json` 的 `env`），重启 HiveMemory。登记规则见[配置指南](./configuration.md#51-访问登记)。

## 4. 连接器读取的环境变量

连接器与 MCP server 从同一组环境变量读取配置。hook 与 MCP 连接都由 Claude Code 发起，因此要在启动 Claude Code 的环境中能读到（插件页写在 Windows 用户环境变量中，也可以写在 `~/.claude/settings.json` 的 `env` 中）：

| 环境变量 | 默认值 | 说明 |
|:---|:---|:---|
| `HIVEMEMORY_URL` | `http://127.0.0.1:8769` | HiveMemory 后端地址。默认值对应桌面客户端与从仓库运行；Docker 路径的后端在 `8000`，需要设为 `http://localhost:8000`。MCP 只接受本机地址（`localhost`、`127.0.0.1` 或 `[::1]`，带端口） |
| `HIVEMEMORY_CLAUDE_CODE_TOKEN` | 插件页写入 | 接入凭据 |
| `HIVEMEMORY_PYTHON` | 插件页写入 | hook 运行连接器的 Python；没有设置时用 PATH 上的 `python3` 或 `python` |
| `HIVEMEMORY_USER_ID` | 不设置 | 以哪个用户导入与访问；不设置时使用服务端的默认用户 `default` |
| `HIVEMEMORY_AGENT_ID` | `claude_code` | 以哪个 Agent 导入与访问；不能使用保留的 `system` |
| `HIVEMEMORY_CONNECTOR_STATE_DIR` | 见第 7 节 | 连接器状态、日志与拒绝记录的目录 |

变量名带上 harness，是为了让多个 harness 共用一套 Windows 用户环境变量时各读各的凭据，不会拿别的 harness 的凭据冒充 `harness:claude-code`。凭据只用于防止本机其他程序冒用 Claude Code 的身份，不进入日志与错误响应。

## 5. 访问登记

导入与 MCP 请求都经统一认证网关，请求的用户与 Agent 必须在 Workspace 访问登记中获准进入。插件页安装时，`claude_code` 还没有记录就补一条，获准 `memory.read`、`memory.search`、`memory.write` 与 `memory.update`，分别对应四个 MCP 工具；导入只要求准入、不需要任何 actor 操作；精确记录也使这个 Agent 不继承默认用户的 `memory.run` 与 `agent.call`。已有的记录不改动，卸载插件也不删除它：权限属于 Agent，不属于 harness。

- `HIVEMEMORY_USER_ID` 设为 `default` 以外的用户时，需要为该用户登记记录，否则导入与工具调用返回 403，连接器停止提交并保留进度；
- 同一个 Claude Code 可以以多个 Agent 运行。更换 `HIVEMEMORY_AGENT_ID` 时，先在设置页的“接入与权限”中把新的 Agent 加入 `harness:claude-code` 可声明的 Agent，否则请求以 403（`reason=agent_not_allowed_for_principal`）被拒；新的 Agent 由默认的用户级记录覆盖，如果希望它的权限与 `claude_code` 相同，在同一页为它增加记录；
- 在设置页修改登记立即生效；直接修改登记文件后重启 HiveMemory。登记规则见[配置指南](./configuration.md#51-访问登记)。

## 6. 确认记忆访问

配置完成并重新打开 Claude Code 后，在 Claude Code 中执行 `/mcp`：插件的 `hivememory` 应显示已连接，并提供 `memory_search`、`memory_read`、`memory_write` 与 `memory_update` 四个工具。

- 工具说明要求 agent 把检索结果当作带日期的历史证据：仓库中的代码与文档、Claude Code 自己的项目指令与记忆文件优先于检索结果。
- 写入与更新由插件页写入的放行规则放行，不逐次确认（第 2 节）；没有这两条规则时按 Claude Code 的权限设置确认，确认框中的参数带有 `turn`，是插件补上的当前一轮信息，不需要修改，auto 模式下则会被拦下。结果是一个 pending alias（写入以 `draft_` 开头，更新以 `rev_` 开头），表示已被接纳、正式记忆仍在后台生成；可以用 `memory_read` 读取它，生成完成后读取会转到正式记忆。工具结果提示缺少当前一轮的信息（`request.turn_required`）时，说明插件的 hook 没有运行，检查插件是否已更新并重启了 Claude Code。
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
| `/mcp` 显示 `hivememory` 连接失败，工具不出现 | HiveMemory 未运行或 `HIVEMEMORY_URL` 不对（桌面客户端与从仓库运行在 `8769`，Docker 在 `8000`）。启动后在 Claude Code 中执行 `/mcp reconnect all` 或重新打开会话 |
| 工具调用返回 `invalid_credential`，`connector.log` 的 `stopped` 行带 403 `invalid_credential` | Claude Code 读到的凭据与 HiveMemory 登记的不一致：多半是安装之后没有完全退出 Claude Code，或者终端是安装之前打开的。完全退出再打开；仍然不行时在插件页点“重新安装”，凭据会重新写入。连接器保留了进度，下一次唤醒时补发未导入的轮次 |
| 导入返回 400 `request.principal_required` | 服务端已更新而插件仍是旧版本：更新插件（第 10 节）后重启 Claude Code，积压的轮次随后补发 |
| 导入返回 403 `unknown_principal` | 插件还装着，但 HiveMemory 一侧没有 `harness:claude-code` 的登记（例如在插件页卸载过，或换了一个新的数据目录）：在插件页重新安装 |
| 工具调用返回 `operation_not_allowed` | 该 Agent 的访问记录缺少对应的 `memory.*` 操作（第 5 节）：在设置页的“接入与权限”中勾选，立即生效 |
| 写入或更新返回 `request.turn_required` | 插件的 `PreToolUse` hook 没有运行：确认插件已更新、已重启 Claude Code |
| auto 模式下写入或更新被拦下，提示 `a hook changed this call's input after auto mode reviewed it` | 用户设置中没有两条放行规则：在插件页点“重新安装”写入，或按第 3 节手动加入 |
| 工具调用返回 `request.workspace_not_found` | `X-Workspace-ID` 只接受 `main_workspace` |
| MCP 请求返回 421 | MCP 端点只接受带端口的本机地址（`localhost`、`127.0.0.1`、`[::1]`）；`HIVEMEMORY_URL` 改用这些地址 |

## 10. 更新与卸载

- **更新**：Claude Code 保存的是插件的副本，插件目录更新后要重新安装才能读到。桌面客户端覆盖安装新版本之后，后端启动时自动为插件页装过的 Claude Code 重新安装插件，失败时插件页把它标为“需要更新”；从仓库运行时，更新代码后在插件页点“重新安装”，或执行 `claude plugin marketplace update hivememory-local` 与 `claude plugin update hivememory-connector@hivememory-local`。之后重启 Claude Code。插件与 HiveMemory 需要一起升级：新插件导入的轮次可能带旧版 HiveMemory 不认识的内容，会被整轮拒绝。
- **卸载**：在插件页点 Claude Code 的“卸载”：卸载 Claude Code 中的插件（保留插件的数据目录，重新安装后不必从头补发）、移除本地 marketplace，删除用户设置中两条放行规则，删除 `harness:claude-code` 的登记、凭据与写入的环境变量；`claude_code` 的访问记录与已经写入的记忆保留。再重启 Claude Code。
