---
title: Codex 插件接入
status: archived
archived_at: 2026-11-08
implemented_by: PR #199（feat/codex-plugin）
superseded_by: docs/system/import-bus.md
owner: project
target: v1.0 范围的 plugin 模式接入（不绑定具体版本号，owner 2026-11-04）
scope: codex-plugin-unified-directory-turn-import-and-memory-tools
related_docs:
  - docs/research/codex-plugin-survey.md
  - docs/contracts/mcp.md
  - docs/system/import-bus.md
  - docs/system/configuration.md
  - docs/help/claude-code-plugin.md
code_paths:
  - integrations/.claude-plugin/marketplace.json
  - integrations/claude-code/
  - configs/system_principals.yaml
  - configs/workspace_actors.yaml
  - tests/unit/connectors/
updates:
  - docs/system/import-bus.md
  - docs/contracts/mcp.md
  - docs/system/configuration.md
  - docs/workspace/authentication.md
  - docs/architecture/overview.md
  - docs/help/claude-code-plugin.md
  - docs/help/deepseek-harness-plugin.md
  - docs/help/README.md
  - docs/help/setup.md
  - docs/help/troubleshooting.md
  - docs/PROJECT.md
  - docs/ROADMAP.md
  - README.md
  - README_EN.md
last_reviewed: 2026-11-08
---

# Codex 插件接入

> 状态：Archived（2026-11-08，插件、接入登记、测试与帮助文档完成，事实文档收尾后归档；以真实模型与本机后端的验收在日常使用中完成，见第 10 节）。当前事实见 [Import Bus](../../system/import-bus.md#10-codex-插件)第 10 节、[MCP 记忆访问](../../contracts/mcp.md#6-连接器调用名轮次信息与回声标记)第 6 节与[接入 Codex](../../help/codex-plugin.md)。以下为实施时的计划原文。原状态：Active（2026-11-08 建立并实施）。owner 决定（2026-11-08）：接入沿用 Claude Code 的形态（Codex 插件声明 HiveMemory 的 `/mcp`，`PreToolUse` hook 为写入与更新补上可信的轮次信息，会话记录读取器把轮次回流 Import Bus），不采用本地 MCP 包装服务；Codex 与 Claude Code 合用一个插件目录，共用 Python 连接器与同一份 `SKILL.md`，dsh 插件不并入。外部证据见[调研报告](../../research/codex-plugin-survey.md#7-调研结果)第 7 节，其中第 7.9 节是 2026-11-08 在 Windows 原生以 `codex-cli 0.167.0` 做的机制实测。本计划记录 HiveMemory 一侧的设计、验收与收尾清单；标为“候选设计”的内容不是 owner 决定，可在计划评审时调整。

## 1. 背景和问题

v1.0 的 plugin 模式覆盖 Claude Code、DeepSeek Harness、Codex 与 pi（owner，2026-11-04），Codex 先于 pi 接入（owner，2026-11-08）；接入先只支持纯 Windows 环境，harness 与 HiveMemory 都在 Windows 上原生运行（owner，2026-11-07）。

调研与 Windows 实测得出的约束：

- Codex 的 `PreToolUse` 可以改写 MCP 工具参数，但必须同时返回 `permissionDecision: "allow"`；现有 Claude Code 连接器的输出不给权限决定，不能原样复用。
- Codex 没有 `StopFailure`，`Stop` 早于一轮的结束记录写入，失败的一轮不触发 `Stop`；一轮的结局只能从会话记录的 `task_complete`（带或不带 `error`）与中断记录得知。现有连接器的结局来自 hook 记录或“下一轮出现、中断标记、30 分钟静止”的推断，没有“读取器从记录中读到结局”这一来源。
- 子 Agent 有自己的会话记录文件；子 Agent 内 hook 的 `session_id` 是根会话，`turn_id` 与 `transcript_path` 却是子 Agent 自己的。
- Codex 安装插件时复制整个目录，插件必须自成一体；两份清单并存时 Codex 用 `.codex-plugin/plugin.json`，清单写明自己的 hook 与 MCP 文件后不读取 Claude Code 默认的 `hooks/hooks.json` 与 `.mcp.json`（调研第 7.9 节）。
- Codex 的 hook 要经用户审阅信任后才运行，插件更新后需要重新信任；不设审批模式时，MCP 写入工具每次都要审批。

## 2. 目标与非目标

目标：

- Codex 中的 Agent 经插件声明的 MCP server 检索、读取、写入与更新 HiveMemory 记忆；写入与更新由 hook 补上这一轮的可信轮次信息，模型填写的被覆盖。
- Codex 会话中已结束的顶层轮次回流 Import Bus，确认之后才前进；完成、失败与中断分别得到对应结局；HiveMemory 不可用时不影响 Codex，恢复后补发；本插件记忆工具的调用回流时归一化为 `SEARCH`、`READ`、`WRITE`、`UPDATE`。
- 插件目录合并为 `integrations/hivememory/`，Claude Code 与 Codex 共用连接器公共骨架与 `skills/hivememory/SKILL.md`；Claude Code 的行为不变。
- HiveMemory 一侧登记 Codex 自己的接入身份与 Agent 白名单。

非目标：

- HiveMemory 服务端的 MCP 与 Import Bus 契约不变，不新增路由、参数或错误码。
- 引导 skill 的内容；Codex 原生 memories 的开关与分工说明（使用选择，不由插件改动）。
- 接入登记的运行时登记与配置热更新：Codex 先按现行做法在两份配置文件中登记，运行时登记之后另做（owner，2026-11-08）。
- Linux、macOS 与 WSL；dsh 插件；pi 接入；controller 模式。
- 代码执行模式中间接调用的回声标记（与 dsh 的同类限制相同）。

## 3. 当前状态和缺口

- 服务端：`/mcp` 与 Import Bus 已支持外部 harness，接入登记有 `harness:claude-code` 与 `harness:dsh`，凭据变量按 harness 命名（`HIVEMEMORY_<harness>_TOKEN`，与服务端 `credential_env` 同名，2026-11-08）。
- 连接器：`integrations/claude-code/connector/` 分为与 harness 无关的 `core/`（配置、状态与游标、提交与结果分类、同步循环、日志）与 `claude_code/`（读取器与 hook 处理）。`core/config.py` 中的 principal、默认 Agent、凭据变量与默认状态目录写死为 Claude Code；`TranscriptTurn` 没有读取器给出的结局。
- Codex 一侧的机制已在 Windows 原生实测（调研第 7.9 节）：插件安装与合并目录、hook 信任、hook 命令与环境（`${PLUGIN_ROOT}` 替换、`PLUGIN_DATA`）、HTTP MCP 的凭据与请求头、工具名与 matcher、审批、参数改写、hook 输入、会话记录结构、续会话与子 Agent。
- 未实测：用户中断、fork 与回退、上下文压缩、桌面版与 IDE 扩展、真实模型与真实 HiveMemory 后端的端到端调用。

## 4. 设计

### 4.1 合并后的插件目录

`integrations/claude-code/` 改名为 `integrations/hivememory/`，插件名仍为 `hivememory-connector`、marketplace 仍为 `hivememory-local`，marketplace 的来源改为 `./hivememory`。Claude Code 与 Codex 都从 `integrations/.claude-plugin/marketplace.json` 安装（Codex 读取这一路径的 marketplace，实测通过）。

| 文件 | 读取方 | 职责 |
|:---|:---|:---|
| `.claude-plugin/plugin.json`、`.mcp.json`、`hooks/hooks.json` | Claude Code | 与现在相同，hook 命令带上 harness 参数（第 4.2 节） |
| `.codex-plugin/plugin.json` | Codex | 写明 `skills`、`mcpServers: ./codex.mcp.json`、`hooks: ./hooks/codex.json` 与版本 |
| `codex.mcp.json`、`hooks/codex.json` | Codex | 第 4.4、4.5 节 |
| `hivememory_hook.py` | 两者 | 启动脚本，不变 |
| `connector/core/` | 两者 | 公共骨架，第 4.2、4.7 节的改动 |
| `connector/claude_code/` | Claude Code | 不变 |
| `connector/codex/` | Codex | 会话记录读取器与 hook 处理 |
| `skills/hivememory/SKILL.md` | 两者 | 同一份正文 |

插件名与 marketplace 名不变，Claude Code 的工具调用名（`mcp__plugin_hivememory-connector_hivememory__*`）与插件数据目录都不变，已有会话记录的回声识别与游标状态继续有效。

### 4.2 按 harness 区分的连接器配置

命令行入口第一个参数为 harness（`claude-code` 或 `codex`），例如 `hivememory_hook.py codex hook Stop`；后台同步的命令同样带上它。每个 harness 对应一组固定取值（候选设计）：

| 取值 | Claude Code | Codex |
|:---|:---|:---|
| 来源 `source` | `claude-code` | `codex` |
| principal | `harness:claude-code` | `harness:codex` |
| 默认 Agent | `claude_code` | `codex` |
| 凭据变量 | `HIVEMEMORY_CLAUDE_CODE_TOKEN` | `HIVEMEMORY_CODEX_TOKEN` |
| 读取器与 hook 处理 | `claude_code` | `codex` |

状态目录的取法不变（`HIVEMEMORY_CONNECTOR_STATE_DIR`，其次 `CLAUDE_PLUGIN_DATA`）：Codex 为 hook 设置的 `CLAUDE_PLUGIN_DATA` 指向它自己的插件数据目录，两个 harness 的状态因此天然分开。`HIVEMEMORY_URL`、`HIVEMEMORY_USER_ID`、`HIVEMEMORY_AGENT_ID` 与现在相同。

### 4.3 身份与登记

候选设计：`configs/system_principals.yaml` 新增 `harness:codex`（kind `harness`，adapter `mcp` 与 `import`，`credential_env: HIVEMEMORY_CODEX_TOKEN`，不能声明 `system`）；`configs/workspace_actors.yaml` 为 `(default, codex)` 登记检索、读取、写入与更新，不继承 `memory.run` 与 `agent.call`。两份配置仍在启动时装载，改完重启后端。

### 4.4 MCP 与审批

`codex.mcp.json` 声明 server `hivememory`：`type: "http"`，`bearer_token_env_var: "HIVEMEMORY_CODEX_TOKEN"`，`http_headers` 带 `X-Principal-ID: harness:codex` 与 `X-Agent-ID: codex`，`env_http_headers` 把 `HIVEMEMORY_USER_ID` 作为 `X-User-ID`（未设置时不发送）。

- `url` 不支持环境变量展开，候选设计写死为 `http://127.0.0.1:8769/mcp`（Windows 原生运行时后端的地址）；实施时核对用户配置能否覆盖插件 MCP server 的地址，能则写进接入手册，不能则作为已知限制。
- 候选设计：`default_tools_approval_mode: "approve"`，四个记忆工具不再逐次审批。依据：工具只作用于 HiveMemory，授权由 Workspace 访问登记逐次判定，写入可经更新修正，自主维护记忆是长期目标；不设时交互使用中每次写入都要确认、无头运行时写入失败。

### 4.5 hook

`hooks/codex.json` 注册的命令为 `python "${PLUGIN_ROOT}/hivememory_hook.py" codex hook <事件>`（Codex 在交给 shell 前替换 `${PLUGIN_ROOT}`，不依赖 shell 语法）：

| 事件 | 作用 |
|:---|:---|
| `SessionStart` | 登记会话记录，唤醒全部已登记会话记录的同步；来源为 `resume` 时清除会话结束标记 |
| `PreToolUse`（matcher `^mcp__hivememory__(memory_write\|memory_update)$`） | 第 4.6 节 |
| `Stop`、`Interrupt` | 唤醒这份会话记录的同步 |
| `SessionEnd` | 记下会话结束，唤醒同步 |

子 Agent 内触发的 hook（输入带 `agent_id`）不登记、不同步子 Agent 的会话记录：只导入顶层会话，与 dsh 的做法相同（owner，2026-11-02）。hook 读入按字节解码、输出 ASCII 转义的 JSON，网络提交在脱离 Codex 的后台进程中完成，任何失败只进日志、以 0 退出。

### 4.6 写入时补上轮次信息

`PreToolUse` 返回 `permissionDecision: "allow"` 与 `updatedInput`（原参数加上 `turn`，模型填写的被覆盖）。当前一轮按以下方式确定：

- 顶层会话：读取 hook 输入的 `transcript_path`，`turn_id` 对应的一轮就是当前一轮，以与导入相同的函数生成轮次信息（外部会话为线程 ID，外部轮次为 Codex 的 `turn_id`，用户消息与之前的轮次取自会话记录）。
- 子 Agent：读取子 Agent 的会话记录，取当前一轮 `task_started` 的 `root_turn_id`，在根会话（`session_id`）的会话记录中找到这一轮，按顶层的方式生成；写入因此归到派生它的那一轮。根会话记录的定位方式在实施时确定（候选：子 Agent 会话记录所在的 `sessions` 目录中按线程 ID 查找文件名）。
- 找不到当前一轮时返回空输出，工具调用照常进行，由 HiveMemory 以缺少轮次信息拒绝。

`allow` 只使改写生效；审批由第 4.4 节的审批模式决定。

### 4.7 会话记录读取器

Codex 的会话记录是 `CODEX_HOME\sessions\<年>\<月>\<日>\rollout-<时间>-<线程ID>.jsonl`，续会话在同一文件末尾追加，按字节偏移的游标照常适用。读取器只读 `event_msg` 中的结构化条目，不读 `response_item`（两者表示同一内容，后者保留的是模型改写前的参数）：

- 一轮为 `task_started` 到同一 `turn_id` 的 `task_complete` 或中断记录；轮次标识取 `turn_id`，会话标识取 `session_meta` 的线程 ID。之后出现下一轮而没有结束记录的一轮按被打断处理。
- 用户消息取这一轮 `UserMessage` 条目的正文，多条按顺序以空行拼接；环境上下文、技能说明等注入内容不在其中。
- 事件：`AgentMessage` → `assistant_message`；推理条目 → `thought`（只有加密内容时跳过）；`McpToolCall` 与命令执行等工具条目 → `tool_call`，结局取条目状态；父会话中 `wait` 的 `CollabAgentToolCall` 记录的子 Agent 最终消息 → `subagent_message`。条目类型的完整映射在实施时按 0.167.0 的记录核对。
- 回声标记：`McpToolCall` 的 `server` 为 `hivememory`、插件 ID 以 `hivememory-connector@` 开头、工具为四个记忆工具时写 `verb` 与作用对象，规则与 Claude Code 连接器相同（去掉 `turn`）。
- 结局由读取器从记录给出：`task_complete` 不带 `error` → `completed`，带 `error` → `failed`，中断记录 → `cancelled`。`core` 的 `TranscriptTurn` 为此新增“记录中的结局”，有值时优先于 hook 记录与推断；Claude Code 的读取器不填。
- `session_meta` 表明是子 Agent 的会话记录不导入（hook 已不登记，读取器再拒绝一次，防止 `sync --all` 误读）。
- 首次见到的会话不回填、确认后前进、提交结果分类、最近轮次与截断都沿用公共骨架。

### 4.8 回流与结束判定

`Stop` 早于 `task_complete` 写入：`Stop` 唤醒的同步沿用现有的有限轮询等待这一轮的结束记录写完，超时则留给下一次唤醒（下一轮的 `Stop`、`SessionEnd` 或下次 `SessionStart`）。失败的一轮没有 `Stop`，在下一次唤醒时导入。会话结束或长时间静止而一轮没有结束记录时，沿用公共骨架的处理记为 `cancelled`。

### 4.9 安装与更新

安装、信任 hook、配置、确认、排查与更新的步骤写入新的接入手册 `docs/help/codex-plugin.md`（候选：参照 dsh 的做法随实现写入）。要点：经 marketplace 安装后在 Codex 中审阅并信任插件的 hook；插件更新要重新安装并重新信任；未信任时写入会被拒绝。

## 5. 受影响的系统、契约和数据

- 配置：`system_principals.yaml` 新增 `harness:codex`，`workspace_actors.yaml` 新增 Agent `codex`。服务端代码、公开路由、事件与错误语义不变。
- 集成：`integrations/claude-code/` 改名为 `integrations/hivememory/`，新增 Codex 清单、hook、MCP 文件与 `connector/codex/`；`core/` 按 harness 取值、`TranscriptTurn` 新增记录中的结局；marketplace 来源更新。
- 测试：`tests/unit/connectors/conftest.py` 的导入路径随目录改名；新增 Codex 读取器与 hook 的单元测试。

## 6. 兼容与迁移

- Claude Code：目录改名后，在 Claude Code 中更新 marketplace 与插件一次（`claude plugin marketplace update hivememory-local`、`claude plugin update hivememory-connector@hivememory-local`）并重开会话。插件 ID 不变，插件数据目录与其中的游标保留；hook 命令改为带 harness 参数，旧缓存的 hook 在更新前仍按旧参数调用旧代码，不受影响。
- 服务端未设置 `HIVEMEMORY_CODEX_TOKEN` 时照常启动并告警，Codex 的请求以凭据无效拒绝，连接器保留游标，配置后补发。
- 不迁移数据。

## 7. 测试和验收

不变量与归属层：

| 不变量 | 归属层 |
|:---|:---|
| 按 harness 取得来源、principal、默认 Agent 与凭据变量；记录中的结局优先于 hook 记录与推断 | `tests/unit/connectors/test_sync.py` |
| Codex 会话记录的轮次切分、用户消息、事件映射、回声标记、结局、子 Agent 会话记录不导入 | 新增 `tests/unit/connectors/test_codex_reader.py` |
| `PreToolUse` 输出带 `allow` 与覆盖后的 `turn`，子 Agent 中的写入归到根轮；子 Agent 的 hook 不登记会话记录；各事件的登记与唤醒 | 新增 `tests/unit/connectors/test_codex_hooks.py` |
| Codex 清单写明的 hook 与 MCP 文件存在，matcher 匹配插件工具名，principal 与 Agent 在随仓库发布的登记中 | Codex 读取器测试中的声明一致性用例（与 Claude Code 的同类测试并列） |
| Claude Code 的现有行为 | 现有连接器测试随目录改名继续通过，不复制 |

```bash
pytest tests/unit/connectors -q
pytest tests/integration/workspace/test_published_registration_chain.py -q
ruff check . && black --check .
```

Windows 原生验收（使用 owner 的 Codex CLI、本机 HiveMemory 后端与 owner 的账户，执行前征得同意）：

- 经 marketplace 安装、信任 hook 后，真实模型写入记忆得到 `draft_` 句柄，这一轮回流落在写入时确定的话题，生成后读取句柄转到正式记忆；新会话中检索召回。
- 失败的一轮以 `failed` 导入；续会话的新一轮照常导入；子 Agent 中的写入归到父会话的那一轮，子 Agent 会话不单独导入。
- 补测并记录：用户中断（`Interrupt` 与中断记录）、fork 与回退对游标与重复导入的影响、桌面版中插件与 hook 是否可用。实测与第 4.7、4.8 节的假设不符时在实施记录中写明调整。
- Claude Code 回归：改名并更新插件后，检索、写入与回流照常。

## 8. 风险与已知限制

- fork 与回退未实测：fork 产生的新会话记录如果复制了父会话的轮次，会以新会话重复导入；回退不撤回已导入的轮次。实施时补测后确定处理，处理不了的写入已知限制。
- hook 未信任、被禁用或出错时写入缺少 `turn`，被服务端拒绝；模型自己填写的 `turn` 只在 hook 运行时才被覆盖。这是 MCP 契约中适配器信任边界的既有限制。
- 插件每次更新都要重新信任 hook；Codex 的插件与 hook 机制在 0.167.0 之后可能变化，升级后重新核对第 3 节依赖的能力。
- MCP 地址写死，后端不在 `127.0.0.1:8769` 时需要用户另行覆盖（第 4.4 节）。
- 一轮中追加的用户消息与助手事件的相对位置不保留；代码执行模式中的间接调用不做回声标记。
- 免审批（第 4.4 节）使模型无需确认即可写入与更新 HiveMemory。

## 9. 完成时必须更新的当前文档

- [Import Bus](../../system/import-bus.md)：第 8 节改为按 harness 区分的 Python 连接器并加入 Codex（目录、hook、读取器、结束判定、配置），组成表、设计不变量、已知限制与代码测试入口随之更新；
- [MCP 记忆访问](../../contracts/mcp.md)：外部 harness 列表与第 6 节的 Codex 调用名、轮次信息与回声标记；
- [配置](../../system/configuration.md)、[认证与访问登记](../../workspace/authentication.md)：`harness:codex`、`HIVEMEMORY_CODEX_TOKEN` 与 Agent `codex`；
- [架构概览](../../architecture/overview.md) 中的插件路径；
- 帮助：新增 `docs/help/codex-plugin.md`；[接入 Claude Code](../../help/claude-code-plugin.md)、[接入 DeepSeek Harness](../../help/deepseek-harness-plugin.md)（`SKILL.md` 路径）、[帮助索引](../../help/README.md)、[安装与启动](../../help/setup.md)、[故障排查](../../help/troubleshooting.md) 中的目录与入口；
- [PROJECT](../../PROJECT.md)、README 与 README_EN、[ROADMAP](../../ROADMAP.md)、Research 与 Plans 索引。

## 10. 实施记录

分支 `feat/codex-plugin`（2026-11-08）：插件目录改名为 `integrations/hivememory/`；`connector/core/` 按 harness 取值（`Harness`）、新增记录中的结局与共用的记忆工具回声标记规则（`memory_tools.py`）、按行读取的函数移入公共骨架；新增 `connector/codex/`（读取器与 hook 处理）、Codex 清单、`codex.mcp.json` 与 `hooks/codex.json`；接入登记新增 `harness:codex` 与 Agent `codex`；测试与第 9 节的事实文档、新的[接入 Codex](../../help/codex-plugin.md)。

**实施中对设计的调整**：

- `Stop`、`Interrupt` 不只唤醒同步，还按输入中的 `turn_id` 写下结束记录（`completed` / `cancelled`）：二者早于这一轮的结束记录写入会话记录，有了结束记录，同步才会沿用等待写完的轮询，而不是把这一轮当作仍在进行而留到下一次唤醒。结局的先后定为：已保存的结束记录、会话记录写明的结局、推断（第 4.7 节原写“会话记录的结局优先于 hook 记录”，改为保留“重发沿用原结局”的既有规则；Codex 中两者不会冲突）。
- 回声标记按 `server` 名识别，不看插件 ID：用户在 `config.toml` 中声明同名的 `[mcp_servers.hivememory]` 时，它整条替换插件的 server 定义（实测，不合并请求头与审批模式），Codex 记下的调用不再带插件 ID。用户配置因此可以覆盖插件的 MCP 地址，但要整条重写，写入接入手册。
- 续会话时 Codex 给出的会话记录路径带 Windows 扩展长度前缀 `\\?\`，会话开始时的不带；不处理时同一份会话记录有两份状态，轮次被重复导入且结局不一致（端到端实测发现）。Codex 的 hook 去掉盘符形式的这个前缀。
- 子 Agent 的会话记录只由 hook 不登记（输入带 `agent_id`），读取器不再重复拒绝：顶层会话之外的会话记录不会进入登记。
- Codex 清单不写版本号（可选，缺省时 Codex 以固定名字缓存），避免多出一个版本来源。
- `codex exec` 中 `SessionStart` 在第一轮开始写入之后才触发：接入前开始、此时才第一次见到的会话，游标落在这一轮之中，这一轮不导入。写入已知限制。

**验证**（2026-11-08，Windows 原生）：

- `pytest tests/unit/connectors tests/unit/architecture tests/integration/workspace/test_published_registration_chain.py`：143 passed；`ruff check .`、`black --check .`、`git diff --check` 通过；dsh 插件 `node --test`：42 passed。新的回归测试在修复前的代码上确认会失败（凭据变量、扩展长度前缀）。
- 端到端（隔离的 `CODEX_HOME`、`codex-cli 0.167.0`、本机模拟的模型与 MCP/导入服务，不调用真实模型与后端）：从仓库的 marketplace 安装插件，Codex 用 `.codex-plugin` 清单、不读 Claude Code 的默认文件；经 app-server 信任 hook 后，写入的 `turn` 由连接器从会话记录补上并覆盖模型填写的值；每一轮只导入一次：正常结束的为 `completed`（带 `WRITE`/`SEARCH` 回声标记），失败的一轮在会话结束时以 `failed` 导入，子 Agent 中的写入归到父会话中派生它的那一轮、父会话的一轮带子 Agent 交回的消息、子 Agent 会话不导入，续会话的新一轮带上一轮作为最近上文。

**待验收**（需要 owner 的 Codex 账户与本机 HiveMemory，执行前征得同意）：真实模型写入得到 `draft_` 句柄、这一轮回流落在写入时确定的话题、新会话检索召回；用户中断、fork 与回退；桌面应用中的插件与 hook；Claude Code 在目录改名并更新插件后的回归。
