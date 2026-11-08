---
title: Connect DeepSeek Harness
status: current
owner: project
scope: deepseek-harness-plugin-installation-configuration-verification-and-troubleshooting
code_paths:
  - integrations/deepseek-harness/
  - integrations/claude-code/skills/hivememory/
  - configs/system_principals.yaml
  - configs/workspace_actors.yaml
related_contracts:
  - docs/contracts/mcp.md
related_docs:
  - docs/system/import-bus.md
  - docs/help/setup.md
  - docs/help/configuration.md
last_reviewed: 2026-11-08
---

# 接入 DeepSeek Harness

仓库提供 DeepSeek Harness（`dsh`）插件 `hivememory-connector`（`integrations/deepseek-harness/`），它是一个 dsh 原生插件，做两件事：

- **记忆工具**：插件启动时从 HiveMemory 取得工具列表与服务说明，在 dsh 中注册 `memory_search`（检索）、`memory_read`（按 alias 读取全文）、`memory_write`（写入）与 `memory_update`（修正已有记忆）四个工具，服务说明作为系统提示词的一节交给 Agent。何时检索与写入由 Agent 自己决定。写入与更新需要这一轮的信息，HiveMemory 据此把写入落在这一轮所在的话题：插件在调用时以 Agent 所在会话的当前一轮自动补上，Agent 看不到也不能填写这个参数。
- **对话回流**：dsh 中每一轮对话结束后，插件把这一轮（用户消息、回复、思考、工具调用与结果）导入 HiveMemory，记忆随日常使用积累。网络提交在后台进行，HiveMemory 未运行时 dsh 照常工作，未导入的轮次之后按顺序补发。工作原理见 [Import Bus](../system/import-bus.md)。

HiveMemory 的使用手册 skill 随 Claude Code 插件维护（`integrations/claude-code/skills/hivememory/SKILL.md`），内容不区分 harness。dsh 插件不会自动加载它，复制到 dsh 的 skill 目录即可使用（第 4.3 节）。

两部分都以 dsh 自己的接入身份 `harness:dsh` 认证，以 Agent `dsh` 访问，需要在 HiveMemory 与 dsh 两侧配置同一份接入凭据（第 2 节）。插件只用 Node 标准库，不需要安装依赖，也不需要安装 HiveMemory 包。

## 1. 前置条件与步骤一览

- HiveMemory 后端已按[安装与启动](./setup.md)第 3 节或第 4 节运行；
- dsh 版本在 `0.2.1-rc.5` 与 0.2.x 之间（插件的 `engines.dsh`，版本不满足时 dsh 拒绝安装与加载）。桌面端自带 dsh，两者版本相同，可以用桌面端自带的命令查看：

  ```powershell
  & "$env:LOCALAPPDATA\Programs\DeepSeek Harness\resources\runtime\cli\bin\dsh.cmd" --version
  ```

插件运行在 dsh 所在的环境，HiveMemory 可以在另一个环境，常见的两种组合：

| 组合 | dsh | HiveMemory | 插件的环境变量设在 |
|:---|:---|:---|:---|
| A | Windows 上的 dsh 桌面端 | WSL2 中运行 | Windows 用户环境变量 |
| B | 同一台 Linux（或同一个 WSL）中的 dsh 命令行 | 同一环境中运行 | 启动 dsh 的 shell 环境 |

| 步骤 | 在哪一侧 | 内容 |
|:---|:---|:---|
| 生成并配置接入凭据 | 两侧 | HiveMemory 的 `.env` 写 `HIVEMEMORY_DSH_TOKEN`，dsh 一侧写同值的 `HIVEMEMORY_TOKEN`（第 2 节） |
| 配置插件 | dsh | 设置 `HIVEMEMORY_URL` 与 `HIVEMEMORY_TOKEN`（第 3 节） |
| 安装插件 | dsh | 桌面端完全退出后用它自带的命令安装（第 4.1 节），命令行用 `dsh plugin add`（第 4.2 节） |
| 安装使用手册 skill（可选） | dsh | 把 skill 复制到 dsh 的 skill 目录（第 4.3 节） |
| 核对访问登记 | HiveMemory | 默认登记已包含 Agent `dsh`，自定义过登记时补齐（第 5 节） |
| 重启并确认 | 两侧 | 重启 HiveMemory、重新打开 dsh，确认工具与导入（第 6 节） |

## 2. 配置接入凭据

先生成一份随机凭据，例如：

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

把同一个值写入两侧，变量名不同：

| 一侧 | 环境变量 | 写在哪里 |
|:---|:---|:---|
| HiveMemory | `HIVEMEMORY_DSH_TOKEN` | 仓库根 `.env`（与其他密钥放在一起，Docker 路径同样读取它），或启动后端的环境 |
| dsh | `HIVEMEMORY_TOKEN` | 插件读取的环境（第 3 节） |

修改 `.env` 后重启 HiveMemory：接入登记与凭据都在后端启动时装载，更新到包含 `harness:dsh` 登记的版本之后同样需要重启。凭据没有配置或两侧不一致时，导入与工具调用都以 `invalid_credential` 被拒绝：HiveMemory 照常启动并在启动日志中提示凭据环境变量未设置，插件保留进度等待补发，记忆工具返回错误（第 8 节）。

Claude Code 插件的凭据变量是 `HIVEMEMORY_CLAUDE_CODE_TOKEN`，与本插件的 `HIVEMEMORY_TOKEN` 不同名。两个 harness 在同一套环境变量中运行时各读各的凭据，`HIVEMEMORY_DSH_TOKEN` 与 `HIVEMEMORY_CLAUDE_CODE_TOKEN` 各用一份凭据即可。

## 3. 配置插件

插件在 dsh 启动时从环境变量读取配置，修改后重新打开 dsh 才生效：

| 环境变量 | 默认值 | 说明 |
|:---|:---|:---|
| `HIVEMEMORY_URL` | `http://localhost:8000` | HiveMemory 后端地址。默认值对应 Docker 路径；本地开发路径（含脚本的 `serve` 模式）的后端在 `8769`，需要设为 `http://127.0.0.1:8769`。MCP 只接受本机地址（`localhost`、`127.0.0.1` 或 `[::1]`，带端口） |
| `HIVEMEMORY_TOKEN` | 不设置 | 接入凭据，见第 2 节 |
| `HIVEMEMORY_USER_ID` | 不设置 | 以哪个用户导入与访问；不设置时使用服务端的默认用户 `default` |
| `HIVEMEMORY_AGENT_ID` | `dsh` | 以哪个 Agent 导入与访问；不能使用保留的 `system` |
| `HIVEMEMORY_CONNECTOR_STATE_DIR` | `~/.hivememory/dsh-connector` | 插件状态、日志与拒绝记录的目录；Windows 上默认是 `%USERPROFILE%\.hivememory\dsh-connector` |

**组合 A（Windows 桌面端）**：桌面端从 Windows 的用户与系统环境变量启动，不读取 WSL 的 shell 配置。在 PowerShell 中设置：

```powershell
setx HIVEMEMORY_URL http://127.0.0.1:8769
setx HIVEMEMORY_TOKEN <第 2 节生成的凭据>
```

`setx` 只影响之后启动的程序。桌面端关闭窗口后仍驻留在系统托盘，要从托盘菜单选择退出，再重新打开。

WSL2 默认的网络模式会把 Windows 上的 `127.0.0.1`、`localhost` 转发到 WSL 中监听本机地址的服务，HiveMemory 在 WSL 中以默认方式监听即可从 Windows 访问。不要改用 WSL 虚拟机的 IP：MCP 端点只接受本机地址，会以 421 拒绝。在 Windows 中可以这样确认后端可达：

```powershell
curl.exe http://127.0.0.1:8769/health
```

**组合 B（命令行）**：在启动 dsh 的环境中设置：

```bash
export HIVEMEMORY_URL=http://127.0.0.1:8769
export HIVEMEMORY_TOKEN=<第 2 节生成的凭据>
```

## 4. 安装插件

### 4.1 Windows 桌面端

桌面端独占自己的 profile（`%USERPROFILE%\.dsh\profiles\desktop`），插件只能在桌面端退出时安装：

1. 至少打开过一次桌面端，让它初始化 profile；
2. 从系统托盘菜单退出桌面端（只关闭窗口不够）；
3. 在 PowerShell 中用桌面端自带的命令安装。插件直接链接 WSL 中的仓库目录，路径可以在 WSL 中用 `wslpath -w <仓库路径>/integrations/deepseek-harness` 得到：

   ```powershell
   $dsh = "$env:LOCALAPPDATA\Programs\DeepSeek Harness\resources\runtime\cli\bin\dsh.cmd"
   & $dsh plugin --profile desktop add '\\wsl.localhost\<发行版>\home\<用户>\<仓库路径>\integrations\deepseek-harness'
   ```

   输出中出现 `+ hivememory-connector <- \\wsl.localhost\...` 即安装成功；之后可以用 `& $dsh plugin --profile desktop list` 查看已安装的插件。
4. 重新打开桌面端。

桌面端自带 Node 与 pnpm，Windows 上不需要另外安装。链接安装的插件随仓库更新，桌面端启动时需要能访问 WSL（访问 `\\wsl.localhost` 时 WSL 会自动启动）。也可以先把 `integrations/deepseek-harness` 复制到 Windows 本地目录再安装，代价是仓库更新后要重新复制。

### 4.2 命令行

命令行的 `dsh plugin` 以 pnpm 安装插件，需要 PATH 上有 pnpm：

```bash
dsh plugin --profile <profile> add <HiveMemory 仓库路径>/integrations/deepseek-harness
dsh --profile <profile> --dump-config   # 输出中出现 "# == hivememory-connector" 一层
```

安装后以 `dsh --profile <profile>` 加上要用的入口启动，例如 `dsh --profile <profile> headless "..."`。

### 4.3 使用手册 skill

dsh 从用户级的 skill 目录 `<DSH_HOME>/skills` 发现 skill（`DSH_HOME` 默认是 `~/.dsh`，Windows 上是 `%USERPROFILE%\.dsh`），每个 skill 是其中的一个目录 `<名字>/SKILL.md`，只扫描一层。dsh 也扫描几个 harness 通用的 `~/.agents/skills`，放在那里同样可以。目录有变化时 dsh 自动刷新 skill 列表，不需要重新打开。

在普通终端中复制（dsh 会话的沙箱不允许 agent 写入 `.dsh` 目录）。Windows 桌面端在 PowerShell 中：

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.dsh\skills\hivememory" | Out-Null
Copy-Item '\\wsl.localhost\<发行版>\home\<用户>\<仓库路径>\integrations\claude-code\skills\hivememory\SKILL.md' "$env:USERPROFILE\.dsh\skills\hivememory\"
```

命令行：

```bash
mkdir -p ~/.dsh/skills/hivememory
cp <HiveMemory 仓库路径>/integrations/claude-code/skills/hivememory/SKILL.md ~/.dsh/skills/hivememory/
```

之后 skill `hivememory` 出现在会话的 skill 列表中：任务与它的描述相符时，Agent 用 `skill` 工具加载正文；也可以在消息中输入 `/hivememory` 直接加载。

## 5. 访问登记

导入与记忆工具都经统一认证网关，请求的用户与 Agent 必须在 `configs/workspace_actors.yaml` 中获准进入 Workspace。随仓库发布的登记已经为默认用户 `default` 登记了 `dsh`，获准 `memory.read`、`memory.search`、`memory.write` 与 `memory.update`，分别对应四个工具；导入只要求准入，不需要任何 actor 操作；这个 Agent 不继承默认用户的 `memory.run` 与 `agent.call`。`configs/system_principals.yaml` 中的 `harness:dsh` 开放 `mcp` 与 `import` 两个入口，不能声明保留的 `system`。

- `HIVEMEMORY_USER_ID` 设为 `default` 以外的用户时，需要为该用户登记记录，否则导入与工具调用返回 403，插件停止提交并保留进度；
- 更换 `HIVEMEMORY_AGENT_ID` 时，默认的用户级记录会覆盖新的 Agent；如果希望它的权限与 `dsh` 相同，按那条记录的写法为它增加精确记录；
- 修改登记后重启 HiveMemory。登记规则见[配置指南](./configuration.md#51-访问登记)。

## 6. 确认

插件日志 `connector.log` 在状态目录中（第 3 节），只记录标识、轮号与状态，不记录对话内容。

1. **工具已注册**：dsh 启动后，日志中出现 `connected url=... tools=memory_search,memory_read,memory_write,memory_update`。
2. **记忆访问**：新建一个会话，让 Agent 用 `memory_search` 检索一个已知的话题，结果是带存档日期的记忆列表；让它用 `memory_write` 记下一件事，结果是一个以 `draft_` 开头的句柄 alias，表示写入已被接纳、正式记忆仍在后台生成，可以用 `memory_read` 读取它，生成完成后读取会转到正式记忆。
3. **对话回流**：这一轮结束后，日志中出现 `registered session=<会话> origin=new` 与 `confirmed session=<会话> turn=1 status=accepted`。

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
| 日志中有 `connect failed`，记忆工具不出现 | HiveMemory 未运行，或 `HIVEMEMORY_URL` 不对（本地开发与 `serve` 模式在 `8769`，Docker 在 `8000`）；组合 A 先在 Windows 中用 `curl.exe` 确认后端可达。恢复后新建一个会话，或重新打开 dsh |
| 工具返回 `HiveMemory 暂时无法访问` | 同上：调用时后端不可达 |
| 工具返回 `HiveMemory MCP 请求失败，HTTP 421` | `HIVEMEMORY_URL` 不是本机地址（例如填了 WSL 虚拟机的 IP），改用 `127.0.0.1` 或 `localhost` |
| 工具返回 `invalid_credential`，日志的 `stopped` 行带 `status=403 reason=invalid_credential` | 两侧凭据不一致或没有配置：HiveMemory 一侧为 `.env` 中的 `HIVEMEMORY_DSH_TOKEN`（后端启动日志会提示它未设置），dsh 一侧为 `HIVEMEMORY_TOKEN`。改正后重启 HiveMemory、重新打开 dsh，积压的轮次随后补发 |
| `stopped` 行带 `reason=unknown_principal` | HiveMemory 的接入登记中没有 `harness:dsh`：更新仓库后需要重启后端 |
| 工具返回 `operation_not_allowed` | `workspace_actors.yaml` 中该 Agent 的白名单缺少对应的 `memory.*` 操作（第 5 节）；修改后重启 HiveMemory |
| 工具返回 `当前没有进行中的一轮` | 写入与更新只能在一轮对话进行中调用 |
| 日志中有 `stopped ... kind=retryable` | HiveMemory 暂时不可用或繁忙，进度已保留，之后自动补发 |
| 日志中有 `rejected`，`rejections.jsonl` 有记录 | 这一轮被服务端拒绝（413、422，或连续三次 500）后跳过，记录中有会话、轮号、状态码与错误码 |

## 9. 更新与卸载

- **更新**：链接安装（第 4.1 节的 `\\wsl.localhost` 路径或第 4.2 节的仓库路径）的插件随仓库更新，重新打开 dsh 即生效；复制安装的需要重新复制，使用手册 skill（第 4.3 节）同样要重新复制。插件与 HiveMemory 需要一起更新，并且先重启 HiveMemory、再重新打开 dsh：新版本插件可能发出旧版 HiveMemory 不认识的内容（例如子 Agent 交回的消息），旧服务端以 422 拒绝这样的一轮，插件会把它写入拒绝记录并跳过。dsh 升级到支持范围之外的版本时，插件不再加载。
- **卸载**：桌面端从托盘退出后执行 `& $dsh plugin --profile desktop remove hivememory-connector`，命令行执行 `dsh plugin --profile <profile> remove hivememory-connector`，再重新打开 dsh。使用手册 skill 删除 skill 目录中的 `hivememory` 目录即可。
