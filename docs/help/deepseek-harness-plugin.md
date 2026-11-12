---
title: Connect DeepSeek Harness
status: current
owner: project
scope: deepseek-harness-plugin-installation-configuration-verification-and-troubleshooting
code_paths:
  - integrations/hivememory/package.json
  - integrations/hivememory/node/dsh/
  - integrations/hivememory/node/core/
  - integrations/hivememory/skills/hivememory/
  - src/hivememory/system/plugins/
related_contracts:
  - docs/contracts/mcp.md
related_docs:
  - docs/system/import-bus.md
  - docs/help/setup.md
  - docs/help/configuration.md
last_reviewed: 2026-11-12
---

# 接入 DeepSeek Harness

仓库提供 DeepSeek Harness（`dsh`）插件 `hivememory-connector`（`integrations/hivememory/`，dsh 部分在 `node/dsh/`，与 Claude Code、Codex 和 pi 合用这个插件目录），它是一个 dsh 原生插件，做两件事：

- **记忆工具**：插件启动时从 HiveMemory 取得工具列表与服务说明，在 dsh 中注册 `memory_search`（检索）、`memory_read`（按 alias 读取全文）、`memory_write`（写入）与 `memory_update`（修正已有记忆）四个工具，服务说明作为系统提示词的一节交给 Agent。何时检索与写入由 Agent 自己决定。写入与更新需要这一轮的信息，HiveMemory 据此把写入落在这一轮所在的话题：插件在调用时以 Agent 所在会话的当前一轮自动补上，Agent 看不到也不能填写这个参数。
- **对话回流**：dsh 中每一轮对话结束后，插件把这一轮（用户消息、回复、思考、工具调用与结果）导入 HiveMemory，记忆随日常使用积累。网络提交在后台进行，HiveMemory 未运行时 dsh 照常工作，未导入的轮次之后按顺序补发。工作原理见 [Import Bus](../system/import-bus.md)。

HiveMemory 的使用手册 skill 在同一个插件目录中维护（`integrations/hivememory/skills/hivememory/SKILL.md`），内容不区分 harness，安装插件后随插件提供给 dsh（第 4 节）。

两部分都以 dsh 自己的接入来源 `harness:dsh` 认证，以 Agent `dsh` 访问。新安装的 HiveMemory 不连接任何 harness：在设置页的“插件”页安装之后才有这个来源和它的凭据（第 2 节）。插件只用 Node 标准库，不需要安装依赖，也不需要安装 HiveMemory 包。

## 1. 前置条件与步骤一览

- HiveMemory 在 Windows 上运行：[桌面客户端](./desktop-client.md)，或按[安装与启动](./setup.md)第 4 节从仓库运行；
- 这台电脑上装有 dsh 桌面端，并且至少打开过一次（让它初始化自己的 profile）；
- dsh 版本在 `0.2.1-rc.5` 与 0.2.x 之间（插件的 `engines.dsh`，版本不满足时 dsh 拒绝安装与加载）。可以用桌面端自带的命令查看：

  ```powershell
  & "$env:LOCALAPPDATA\Programs\DeepSeek Harness\resources\runtime\cli\bin\dsh.cmd" --version
  ```

| 步骤 | 在哪一侧 | 内容 |
|:---|:---|:---|
| 退出 dsh 桌面端 | dsh | 从系统托盘菜单退出（只关闭窗口不够）：桌面端运行时独占自己的 profile，插件装不上 |
| 安装插件 | HiveMemory 设置页 | 在“插件”页点 DeepSeek Harness 的“安装”（第 2 节） |
| 重新打开并确认 | 两侧 | 重新打开 dsh 桌面端，确认工具与导入（第 6 节） |

## 2. 从插件页安装

在设置页的“插件”页点 DeepSeek Harness 的“安装”，HiveMemory 依次完成：

1. 用桌面端自带的命令（`%LOCALAPPDATA%\Programs\DeepSeek Harness\resources\runtime\cli\bin\dsh.cmd`）把插件链接安装到桌面端的 profile：`dsh.cmd plugin --profile desktop add <插件目录>\hivememory`。插件目录在安装版中是程序目录中的 `resources\backend\integrations`，从仓库运行时是仓库的 `integrations`；指向别处的旧链接随之替换。桌面端正在运行时 dsh 会拒绝，页面提示先从托盘退出 dsh 再重试。
2. 登记接入来源 `harness:dsh`，初始只能声明 Agent `dsh`，凭据由 HiveMemory 生成。
3. `dsh` 还没有访问记录时，为它补一条只开放检索、读取、写入与更新的记录（第 5 节）。
4. 把凭据写进 Windows 用户环境变量 `HIVEMEMORY_DSH_TOKEN`。

第 1 步失败时什么都不登记，页面给出失败的命令、输出的最后几行和手动安装的命令。之后重新打开 dsh 桌面端：它从 Windows 的用户环境变量启动，读到新写入的凭据。

链接安装的插件原地读取插件目录：桌面客户端覆盖安装新版本、或从仓库运行时更新代码之后，重新打开 dsh 就读到新版本。

## 3. 手动安装与命令行

插件页找不到 dsh 桌面端，或者插件命令失败时，可以在桌面端退出后，在 PowerShell 中手动安装（插件页给出的命令带着本机插件目录的实际位置）：

```powershell
$dsh = "$env:LOCALAPPDATA\Programs\DeepSeek Harness\resources\runtime\cli\bin\dsh.cmd"
& $dsh plugin --profile desktop add '<插件目录>\hivememory'
```

输出中出现 `+ hivememory-connector <- ...` 即安装成功；之后可以用 `& $dsh plugin --profile desktop list` 查看已安装的插件。装好之后回到插件页，点“已用命令装好，完成登记”：HiveMemory 只完成第 2 节的第 2 至 4 步。

命令行的 dsh 按 profile 安装，插件页不能代为安装。命令行的 `dsh plugin` 以 pnpm 安装插件，需要 PATH 上有 pnpm：

```bash
dsh plugin --profile <profile> add <插件目录>/hivememory
dsh --profile <profile> --dump-config   # 输出中出现 "# == hivememory-connector" 一层
```

安装后同样在插件页点“已用命令装好，完成登记”，再以 `dsh --profile <profile>` 加上要用的入口启动，例如 `dsh --profile <profile> headless "..."`。

插件在 dsh 启动时从环境变量读取配置，修改后重新打开 dsh 才生效：

| 环境变量 | 默认值 | 说明 |
|:---|:---|:---|
| `HIVEMEMORY_URL` | `http://127.0.0.1:8769` | HiveMemory 后端地址。默认值对应桌面客户端与从仓库运行；Docker 路径的后端在 `8000`，需要设为 `http://localhost:8000`。MCP 只接受本机地址（`localhost`、`127.0.0.1` 或 `[::1]`，带端口） |
| `HIVEMEMORY_DSH_TOKEN` | 插件页写入 | 接入凭据 |
| `HIVEMEMORY_USER_ID` | 不设置 | 以哪个用户导入与访问；不设置时使用服务端的默认用户 `default` |
| `HIVEMEMORY_AGENT_ID` | `dsh` | 以哪个 Agent 导入与访问；不能使用保留的 `system` |
| `HIVEMEMORY_CONNECTOR_STATE_DIR` | `~/.hivememory/dsh-connector` | 插件状态、日志与拒绝记录的目录；Windows 上默认是 `%USERPROFILE%\.hivememory\dsh-connector` |

HiveMemory 不在 Windows 上运行时（例如 Docker 或 WSL），插件页不能代为安装：按上面的命令手动安装插件，再按[接入 Claude Code](./claude-code-plugin.md#3-手动安装)第 3 节的写法，在登记文件中登记 `harness:dsh`（凭据变量 `HIVEMEMORY_DSH_TOKEN`、可声明的 Agent `dsh`），并把同一份凭据写进 dsh 一侧的环境变量。后端在 WSL2 中时，Windows 上的 `127.0.0.1` 会转发到 WSL 中监听本机地址的服务；不要改用 WSL 虚拟机的 IP：MCP 端点只接受本机地址，会以 421 拒绝。

## 4. 使用手册 skill

skill `hivememory` 随插件提供，不需要另外复制：插件的配置层另外插入一个 skill 提供方，只扫描插件目录下的 `skills/`，不改动用户与各模式自己的 skill 配置。在加载 skill 的模式中（例如桌面端的标准模式；最小模式不加载任何 skill），它出现在会话的 skill 列表里：任务与它的描述相符时，Agent 用 `skill` 工具加载正文；也可以在消息中输入 `/hivememory` 直接加载。链接安装时 skill 随插件目录更新。

此前把 skill 手动复制到 `<DSH_HOME>/skills/hivememory`（Windows 上是 `%USERPROFILE%\.dsh\skills\hivememory`）或 `~/.agents/skills/hivememory` 的，安装新版插件后删除这份副本：同名 skill 由离 Agent 更近的来源胜出，桌面端的各模式自己扫描这两个目录，会一直使用旧副本。

## 5. 访问登记

导入与记忆工具都经统一认证网关，请求的用户与 Agent 必须在 Workspace 访问登记中获准进入。插件页安装时登记 `harness:dsh`，开放 `mcp` 与 `import` 两个入口，不能声明保留的 `system`；`dsh` 还没有访问记录时为它补一条，获准 `memory.read`、`memory.search`、`memory.write` 与 `memory.update`，分别对应四个工具；导入只要求准入，不需要任何 actor 操作；这个 Agent 不继承默认用户的 `memory.run` 与 `agent.call`。已有的记录不改动，卸载插件也不删除它。

- `HIVEMEMORY_USER_ID` 设为 `default` 以外的用户时，需要为该用户登记记录，否则导入与工具调用返回 403，插件停止提交并保留进度；
- 同一个 dsh 可以以多个 Agent 运行。更换 `HIVEMEMORY_AGENT_ID` 时，先在设置页的“接入与权限”中把新的 Agent 加入 `harness:dsh` 可声明的 Agent，否则请求以 403（`reason=agent_not_allowed_for_principal`）被拒；新的 Agent 由默认的用户级记录覆盖，如果希望它的权限与 `dsh` 相同，在同一页为它增加记录；
- 在设置页修改登记立即生效；直接修改登记文件后重启 HiveMemory。登记规则见[配置指南](./configuration.md#51-访问登记)。

## 6. 确认

插件日志 `connector.log` 在状态目录中（第 3 节），只记录标识、轮号与状态，不记录对话内容。

1. **工具已注册**：dsh 启动后，日志中出现 `connected url=... tools=memory_search,memory_read,memory_write,memory_update`。
2. **记忆访问**：新建一个会话，让 Agent 用 `memory_search` 检索一个已知的话题，结果是带存档日期的记忆列表；让它用 `memory_write` 记下一件事，结果是一个以 `draft_` 开头的句柄 alias，表示写入已被接纳、正式记忆仍在后台生成，可以用 `memory_read` 读取它，生成完成后读取会转到正式记忆。
3. **对话回流**：这一轮结束后，日志中出现 `registered session=<会话> origin=new` 与 `confirmed session=<会话> turn=1 status=accepted`。
4. **使用手册 skill**：在消息框中输入 `/hive`，Skills 列表中出现 `hivememory`。

## 7. 使用中的行为

- **哪些会话会导入**：只导入插件首次运行之后创建的会话。安装前已存在的会话不回填，恢复这类会话后只导入之后开始的轮次。
- **子 Agent**：Agent 委托出去的子 Agent 的会话不导入，也不会建立新话题；子 Agent 结束时交回的输出（dsh 发给父 Agent 的结算通知）随父会话这一轮导入，子 Agent 读文件、调用工具的过程不进入记忆。子 Agent 认为值得长期保留的发现，应当自己用 `memory_write` 写入，写入落在父会话（顶层会话）最近的一轮所在的话题。子 Agent 运行中用 `send_message` 发给父 Agent 的消息不导入，要保留的结果应当出现在它最后的输出中。
- **一轮何时导入**：一轮在结束后才导入。正常结束为 `completed`；被停止、或 dsh 崩溃后在会话恢复时补上收尾的为 `cancelled`；出错、被拦截或输出达到上限的为 `failed`。崩溃时进行中的一轮，只有在之后恢复这个会话时才会导入。
- **何时同步与补发**：一轮结束、新建或恢复会话时同步该会话，dsh 启动时补发此前没有得到确认的轮次。HiveMemory 暂不可用（连接失败、超时、503）或插件配置有误（400、403、404）时，插件保留进度、停止提交，在下一次同步时补发。
- **被跳过的轮次**：单轮内容超过服务端的请求体上限（413，默认 8 MiB）或格式被拒绝（422）时，这一轮写入 `rejections.jsonl` 后跳过，不阻塞后续轮次；同一轮连续三次得到 500 也会被跳过。
- **记忆不会被再提取一遍**：用过检索与读取工具的轮次导入后，生成只保留一行检索或读取摘要；写入与更新的调用不进入生成材料。dsh 的 PTC 模式（默认关闭）下记忆工具在 `run_code` 内部调用，会话记录只有外层的 `run_code`，这一标记不生效。
- **HiveMemory 未运行时启动 dsh**：记忆工具不出现，日志中有 `connect failed`；HiveMemory 恢复后，下一次创建 Agent（例如新建会话）时插件重新连接并注册工具，重新打开 dsh 也可以。
- **多个 dsh 进程**：桌面端与命令行共用同一个状态目录同时运行时，同一轮可能被提交两次，HiveMemory 按交互标识去重。

## 8. 排查

| 现象 | 原因与处理 |
|:---|:---|
| 状态目录中没有 `connector.log` | 插件没有加载：确认安装命令成功、安装后重新打开了 dsh；dsh 版本不在支持范围内时安装与加载会被拒绝 |
| 日志中有 `connect failed`，记忆工具不出现 | HiveMemory 未运行，或 `HIVEMEMORY_URL` 不对（桌面客户端与从仓库运行在 `8769`，Docker 在 `8000`）；可以用 `curl.exe http://127.0.0.1:8769/health` 确认后端可达。恢复后新建一个会话，或重新打开 dsh |
| 工具返回 `HiveMemory 暂时无法访问` | 同上：调用时后端不可达 |
| 工具返回 `HiveMemory MCP 请求失败，HTTP 421` | `HIVEMEMORY_URL` 不是本机地址（例如填了 WSL 虚拟机的 IP），改用 `127.0.0.1` 或 `localhost` |
| 工具返回 `invalid_credential`，日志的 `stopped` 行带 `status=403 reason=invalid_credential` | dsh 读到的凭据与 HiveMemory 登记的不一致：多半是安装之后没有从托盘退出 dsh 再打开。退出再打开；仍然不行时在插件页点“重新安装”，凭据会重新写入。积压的轮次随后补发 |
| `stopped` 行带 `reason=unknown_principal` | 插件还装着，但 HiveMemory 一侧没有 `harness:dsh` 的登记（例如在插件页卸载过，或换了一个新的数据目录）：在插件页重新安装 |
| 工具返回 `operation_not_allowed` | 该 Agent 的访问记录缺少对应的 `memory.*` 操作（第 5 节）：在设置页的“接入与权限”中勾选，立即生效 |
| 工具返回 `当前没有进行中的一轮` | 写入与更新只能在一轮对话进行中调用 |
| 日志中有 `stopped ... kind=retryable` | HiveMemory 暂时不可用或繁忙，进度已保留，之后自动补发 |
| 日志中有 `rejected`，`rejections.jsonl` 有记录 | 这一轮被服务端拒绝（413、422，或连续三次 500）后跳过，记录中有会话、轮号、状态码与错误码 |

## 9. 更新与卸载

- **更新**：链接安装的插件与使用手册 skill 原地读取插件目录：桌面客户端覆盖安装新版本、或从仓库运行时更新代码之后，重新打开 dsh 即生效。插件与 HiveMemory 需要一起更新，并且先重启 HiveMemory、再重新打开 dsh：新版本插件可能发出旧版 HiveMemory 不认识的内容，旧服务端以 422 拒绝这样的一轮，插件会把它写入拒绝记录并跳过。dsh 升级到支持范围之外的版本时，插件不再加载。
- **卸载**：从托盘退出 dsh 桌面端，在插件页点 DeepSeek Harness 的“卸载”：移除桌面端 profile 中的插件，删除 `harness:dsh` 的登记、凭据与写入的环境变量；`dsh` 的访问记录与已经写入的记忆保留。使用手册 skill 随插件一起移除。命令行的 dsh 执行 `dsh plugin --profile <profile> remove hivememory-connector`。
