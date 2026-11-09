---
title: Codex 插件接入调研需求与报告
status: baseline
owner: project
scope: codex-plugin-mode-integration-survey
code_paths:
  - integrations/claude-code/
  - integrations/deepseek-harness/
  - src/hivememory/mcp_server/
  - src/hivememory/system/services/import_bus/
related_docs:
  - docs/system/import-bus.md
  - docs/contracts/mcp.md
  - docs/system/configuration.md
  - docs/help/claude-code-plugin.md
  - docs/research/deepseek-harness-plugin-survey.md
  - docs/research/harness-interface-survey-snapshot.md
  - docs/research/pi-plugin-survey.md
last_reviewed: 2026-11-08
snapshot_at: 2026-11-07
source_version: rust-v0.167.0
source_commit: acbbe32e165dcb0bf450b454ad1b3547b690db8f
upstream_commit: 086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a
---

# Codex 插件接入调研需求与报告

## 0. 文档性质

第 1–6 节保留原调研需求；第 7 节是 2026-11-07 的资料调研快照，固定到 `rust-v0.167.0`。已完成能力清单、hook 对应、路径比较、引导与共存分析；第 7.7 节列出的 Windows 原生最小实测于 2026-11-08 以同一版本的 CLI 补做（模拟模型与模拟 MCP，不调用真实模型），结果与仍未实测的项见第 7.9 节。报告不代表 HiveMemory 已实现 Codex 接入；接入路径之后由 owner 决定（2026-11-08，沿用 Claude Code 的形态），决定不写入本报告。需求中的线索（第 2 节）与第 7 节不一致时，以第 7 节所列固定源码为准。

外部项目变化很快，调研结论只代表完成时的日期。Codex 发版频繁，调研时以固定的发行版本与提交为准。

## 1. 目的

owner 的方向（2026-11-04）：v1.0 的 plugin 模式覆盖 Claude Code、DeepSeek Harness、Codex 与 pi，使主流 harness 都能接入 HiveMemory；controller 模式（由 HiveMemory 驱动 harness 执行任务）推迟。2026-11-07 owner 决定开始 Codex 与 pi 的接入，本文与 [pi 插件接入调研需求](./pi-plugin-survey.md)分别为两者的接入计划提供依据。

plugin 模式的接入包括两件事：harness 的每一轮对话回流到 HiveMemory（Import Bus），以及 harness 中的 agent 经记忆工具检索、读取、写入与更新记忆（MCP）。记忆的写入与访问主要由执行者经记忆工具主动完成，话题结算的被动提取是兜底（owner，2026-10-31）。

**目标环境**（owner，2026-11-07）：Codex 与 pi 的接入先只支持纯 Windows 环境，harness 与 HiveMemory 都在 Windows 上原生运行，不经 WSL。调研只核对 Windows 上的行为，Linux、macOS 与 WSL 暂不核对。

这次调研要回答：

- **接入路径**：按第 2 节的线索，Codex 的插件可以打包 skill、MCP server 与 hook，形态与 Claude Code 相近；能否沿用 Claude Code 连接器的做法（插件、会话记录读取器、MCP 与改写参数的 hook），缺口在哪里；
- **HiveMemory 一侧**需要新增或改动什么；
- **引导 skill** 在 Codex 中以什么方式承载；
- Codex 若有**原生记忆**，与 HiveMemory 怎样共存。

### 1.1 HiveMemory 一侧已有的接入面

判断 Codex 的接口是否够用时，以下列现有事实为准，权威说明见所链接的文档：

- **对话回流**：`POST /api/v1/import/turns`，每个请求是一轮已经结束的交互：用户消息、按顺序的事件（助手正文、思考、子 Agent 交回的消息、工具调用及其结局）、结局（completed / cancelled / failed）、模型名、外部会话与外部轮次标识，以及之前几轮（只供话题路由）。服务端按推导出的交互标识去重，确认之后连接器才前进（[Import Bus](../system/import-bus.md)第 3、4 节与第 8.5 节）。
- **两种已有实现**：
  - Claude Code 连接器是 Python 标准库写的会话记录读取器加 hook。游标、确认后前进、重发去重、首次见到的旧会话不回填等逻辑在与 harness 无关的公共骨架中，接入新 harness 只需新增读取器（[Import Bus](../system/import-bus.md#8-claude-code-连接器)第 8 节）。
  - dsh 插件是 Node 写的原生插件：dsh 的工具前置事件不能改写参数，插件因此自己注册四个记忆工具，并经会话事件回流，不共用 Python 骨架（owner，2026-11-01；[Import Bus](../system/import-bus.md#9-deepseek-harness-插件)第 9 节）。
- **记忆访问**：HTTP 的 MCP 端点 `POST /mcp`，无状态的 streamable HTTP、JSON 响应，Host 检查只接受带端口的本机地址；请求头声明 principal、Bearer 凭据、用户与 Agent；工具为 `memory_search`、`memory_read`、`memory_write`、`memory_update`（[MCP 记忆访问](../contracts/mcp.md)第 1–3 节）。
- **写入时确定话题**：写入与更新必须带参数 `turn`（这一轮的来源、外部会话与轮次标识、用户消息与之前几轮），缺少时以 `request.turn_required` 拒绝。`turn` 必须由可信的适配器提供，不能由模型填写；它的外部标识与这一轮回流时的相同，服务端因此推导出同一个交互标识（[MCP 记忆访问](../contracts/mcp.md#6-连接器调用名轮次信息与回声标记)第 6 节）。Claude Code 由 `PreToolUse` hook 以 `updatedInput` 改写参数补上它；dsh 由插件自己的工具在执行时补上。
- **回声标记**：本插件四个记忆工具的调用回流时归一化为 `SEARCH`、`READ`、`WRITE`、`UPDATE` 能力调用，避免已有记忆被再提取一遍。这要求读取器能从会话记录中认出这些调用，识别依据是调用名（同上）。
- **子 Agent**：子 Agent 是父 Agent 临时委托的执行线程，它的过程不进入记忆生成，要保留的内容由它自己写入；父会话的一轮保留它交回的最终输出（`subagent_message` 事件）；子 Agent 中的写入以顶层会话的一轮确定话题（owner，2026-11-02；[Import Bus](../system/import-bus.md#93-轮次的切分与映射)第 9.3 节）。
- **接入登记**：每个 harness 有自己的 principal（`harness:claude-code`、`harness:dsh`）与凭据环境变量，登记在 [`system_principals.yaml`](../../configs/system_principals.yaml)；具体 Agent 与它的四项记忆操作白名单登记在 [`workspace_actors.yaml`](../../configs/workspace_actors.yaml)。两份配置在启动时装载（[配置](../system/configuration.md)）。
- **引导 skill**：使用手册 skill 只维护一份（[`SKILL.md`](../../integrations/hivememory/skills/hivememory/SKILL.md)），内容不区分 harness；Claude Code 随插件加载，dsh 由用户复制到 dsh 的 skill 目录（[接入 DeepSeek Harness](../help/deepseek-harness-plugin.md#43-使用手册-skill)第 4.3 节）。

Claude Code 连接器用到的 hook 是 `Stop`、`StopFailure`、`SessionStart`、`SessionEnd` 与 `PreToolUse`（[Import Bus](../system/import-bus.md#82-hooks)第 8.2 节），第 5 节要求逐项给出 Codex 中的对应物。

### 1.2 前两次接入暴露的问题

以下问题在 Claude Code 与 dsh 中实际出现过，调研须核对 Codex 中是否存在同类情形：

- **改写参数**：dsh 的工具前置事件不能改写参数，通用 MCP 的写入因此无法补上可信的 `turn`。
- **调用名**：回声标记与 hook 的 matcher 都依赖调用名格式，格式变化时识别静默失效，表现为重复记忆与写入被拒。
- **会话记录**：Claude Code 的记录格式不是公开承诺的接口；并行调用时有时只记下最后一个调用的结果；本地思考多数只有签名。dsh 的记录经压缩，不能按行读取。
- **结束判定**：Claude Code 被用户中断时不触发 `Stop`，连接器只能以中断标记、下一轮或 30 分钟静止判定。
- **子 Agent**：dsh 的子会话各自导入时挤占话题池，轨迹被切成粒度不一致的记忆，之后改为只导入顶层会话（owner，2026-11-02）。
- **间接调用**：dsh 的代码执行模式下记忆工具在 `run_code` 内部调用，会话记录只有外层调用，回声标记不生效。
- **一轮多条用户消息**：运行中追加的用户消息与助手事件的相对位置无法保留。
- **运行环境**：已有的两种接入都没有在纯 Windows 环境中验证过。Claude Code 连接器只在 Linux（WSL）上验证过，hook 命令以 `python3` 运行连接器，状态文件的锁在 Windows 上以持续重试取得；dsh 运行在 Windows 桌面端，访问的是 WSL 中的 HiveMemory 后端。

## 2. 调研对象

- **必选**：OpenAI Codex，官方仓库 [openai/codex](https://github.com/openai/codex)，Apache-2.0。以 Codex CLI（终端交互与 `codex exec`）为主；IDE 扩展、桌面应用与 app-server 是否共用同一套插件、hook、会话记录与配置，作为问题 A5 回答。2026-10-20 的[接口调研快照](./harness-interface-survey-snapshot.md)以 `rust-v0.163.0` 为参考发行，核对的是 controller 模式所需的程序化接口；本次核对 plugin 模式所需的部分，以调研时的最新稳定发行为准。
- **已知线索**（2026-11-07 经 Context7 检索 `openai/codex` 主分支源码得到，未核对发行版本，须在调研中核实）：
  - hook 配置（`hooks.json` 或 TOML）的事件名有 `PreModelRequest`、`PostModelRequest`、`PreToolUse`、`PermissionRequest`、`PostToolUse`、`PreCompact`、`PostCompact`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`SubagentStart`、`SubagentStop`、`Stop`、`Interrupt`；处理器有 `command`、`mcp_tool`、`prompt`、`agent` 四类，`command` 可设超时、异步与在 Windows 上使用的 `commandWindows`（[`codex-rs/protocol/src/hooks.rs`](https://github.com/openai/codex/blob/main/codex-rs/protocol/src/hooks.rs)）；
  - 插件清单可打包 skills、MCP servers、apps 与 hooks，路径相对插件根目录，可从本地文件系统或 git 的 marketplace 安装（[`codex-rs/core-plugins/src/manifest.rs`](https://github.com/openai/codex/blob/main/codex-rs/core-plugins/src/manifest.rs)、[`loader.rs`](https://github.com/openai/codex/blob/main/codex-rs/core-plugins/src/loader.rs)）。
- **可选**：社区已有的 Codex 记忆类插件或 MCP server，只用于了解其他记忆系统怎样接入 Codex、Codex 用户已经在用什么，不作为接入候选。

## 3. 调研问题

每个问题注明服务的方面：**路径**（选择接入路径）、**回流**、**访问**（经工具的记忆访问）、**落位**（写入时确定话题）、**回声**（回声标记）、**引导**（skill 的承载）、**共存**（与原生记忆的分工）、**使用**（owner 环境中的实际使用）。

### A. 插件与分发

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| A1 | 插件清单的格式与可包含的组件；插件怎样注册、启用、禁用与卸载？能否从本地目录或本地 marketplace 安装、不发布？ | 路径 |
| A2 | 插件声明的 MCP server 支持哪些传输？HTTP 请求能否带自定义请求头，能否从环境变量展开 Bearer 凭据？连接失败时的行为（重试、工具缺席、是否影响会话）？ | 访问 |
| A3 | 插件中的 hook 与用户级、项目级 hook 怎样合并？hook 命令能否引用插件根目录（类似 Claude Code 的插件根变量）？命令在什么工作目录与环境变量下运行，输入从哪里读？ | 路径、回流 |
| A4 | 插件与 hook 是否仍是实验功能、需要特性开关或特定账户？有没有版本号或兼容承诺，近期变更的频率如何？ | 路径 |
| A5 | CLI、IDE 扩展、桌面应用与 app-server 是否共用同一套插件、hook、会话记录与配置目录？哪些入口不执行 hook？ | 使用、回流 |

### B. hook

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| B1 | 第 2 节列出的事件各在什么时机触发？与 Claude Code 连接器的五个 hook 怎样对应？`Stop` 是否在每轮结束都触发（含出错），`Interrupt` 能否可靠地标出用户中断的一轮？ | 路径、回流 |
| B2 | `PreToolUse` 能否改写工具参数（对应 Claude Code 的 `updatedInput`）？对 MCP 工具是否生效？matcher 怎样写，能否只匹配本插件的写入与更新两个工具？改写后的参数是否仍经过审批，用户审批时看到的是不是改写后的参数？ | 落位 |
| B3 | 各 hook 的输入有哪些字段：会话标识、轮次标识、会话记录路径、工作目录、工具名与参数、模型名？能否据此定位"当前一轮"及其用户消息？ | 落位、回流 |
| B4 | hook 的同步与异步语义、超时与失败对 Codex 的影响：同步 hook 是否阻塞下一轮，失败或超时是否中断会话？hook 能否启动脱离 Codex 的后台进程？ | 回流 |
| B5 | `SubagentStart` 与 `SubagentStop` 提供什么信息？子 agent 中的工具调用触发的 `PreToolUse` 输入指向哪个会话与哪一轮？ | 回流、落位 |

### C. 会话记录

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| C1 | 会话记录保存在哪里、什么格式？是否有稳定的会话标识与轮次标识？格式是否是公开承诺的接口，有没有官方的读取工具或 schema？ | 路径、回流 |
| C2 | 一轮怎样在记录中表示：用户消息（含运行中追加或排队的消息）、助手正文、思考（是否保存正文，还是只有摘要或加密内容）、工具调用与结果（内置工具、MCP 工具、shell 命令的记录形态）、并行调用的结果是否都记下？ | 回流 |
| C3 | 一轮的结局怎样表达：完成、用户中断、出错、超出上下文、审批被拒？进程崩溃时记录会留下什么？ | 回流 |
| C4 | 恢复、fork、回退与上下文压缩对记录有什么影响：是否重写或截断已写入的内容，恢复或 fork 后轮次标识如何编号，压缩摘要是否写入记录？ | 回流 |
| C5 | 子 agent 的会话是否单独记录，与父会话怎样关联？子 agent 结束时交回给父 agent 的内容记在哪里？ | 回流、落位 |

### D. MCP

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| D1 | 插件 MCP 工具的调用名格式是什么（模型看到的名字、会话记录中的名字、hook matcher 中的名字是否相同）？命名规则是否文档化？ | 回声、落位 |
| D2 | MCP 工具调用默认是否需要审批？插件或配置能否为本插件的工具预设审批策略？ | 访问、使用 |
| D3 | MCP server 的 instructions 是否进入模型上下文，有没有长度限制？ | 引导 |
| D4 | 与 HiveMemory 的 `/mcp` 是否兼容：无状态的 streamable HTTP、JSON 响应、不提供订阅；客户端请求时的 Host 头是什么，能否通过本机地址带端口的检查？ | 访问 |
| D5 | 有没有代码执行或批量调用一类的模式，让 MCP 工具在另一个工具内部被调用？这种情况下会话记录与 `PreToolUse` 看到的是什么？ | 回声、落位 |

### E. 指令与 skill

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| E1 | skill 从哪些位置发现（插件内、用户级、项目级），怎样触发与加载？HiveMemory 的 `SKILL.md`（frontmatter 含 `name` 与 `description`）能否原样使用，随插件分发后怎样命名？ | 引导 |
| E2 | `AGENTS.md` 的读取范围与长度限制；插件能否向系统提示词或每轮上下文追加说明？ | 引导 |
| E3 | 上述说明与已加载的 skill 是否在上下文压缩后保留？ | 引导 |

### F. 原生记忆与上下文

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| F1 | Codex 是否内置跨会话的记忆或持久笔记？如有：存储位置、何时写入与读取、能否关闭，与外部记忆工具是否冲突？ | 共存、引导 |
| F2 | 上下文压缩何时发生，能否从 hook（`PreCompact`、`PostCompact`）或会话记录观察到？ | 回流、使用 |
| F3 | 社区已有的记忆类插件或 MCP server 怎样接入 Codex（工具、hook、存储），有哪些可以借鉴或需要避开的做法？ | 路径 |

### G. 运行、沙箱、认证与许可

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| G1 | 在 Windows 上原生运行（不经 WSL）是否受官方支持，有哪些已知限制？安装方式，以及配置、插件、hook、会话记录与 skill 的目录在 Windows 上的位置？ | 使用 |
| G2 | Windows 上 hook 命令由哪个 shell 执行，`commandWindows` 怎样使用？hook 调用 Python 或 Node 脚本时，插件根目录、路径中的空格与引号、环境变量怎样写？插件 MCP 配置中的环境变量展开在 Windows 上是否照常？ | 路径、回流、访问 |
| G3 | Windows 上的沙箱是否影响接入：hook 命令是否在沙箱内运行，沙箱的网络限制是否阻止 hook 或 MCP 访问本机 HTTP 服务（`localhost:<端口>`），hook 能否写 Codex 目录之外的状态目录？ | 路径、访问、回流 |
| G4 | 登录方式（ChatGPT 账户或 API key）对插件、hook 与 MCP 有没有影响，例如某些功能只对特定账户开放？ | 使用 |
| G5 | 能否无头运行（`codex exec` 等），无头运行时插件、hook 与会话记录是否照常？ | 使用、回流 |
| G6 | 插件、hook 与 Codex 本身的许可与使用条款中有没有限制？ | 路径 |

## 4. 取证要求

- 每项结论注明来源（官方文档、源码、发布说明）的链接、固定版本或提交与访问日期。文档与源码不一致时两者都记录，以所核对发行版本的源码为准；主分支上尚未发布的能力单独标出。
- 区分"文档声明""源码核对""实际运行"三种验证程度。
- **最小实测**：在 Windows 上原生安装 Codex（不经 WSL），使用隔离的配置目录（不改动 owner 已有的 Codex 配置），写一个本地插件：注册只记录输入的 `SessionStart`、`Stop`、`SessionEnd` 与 `PreToolUse` hook，声明一个本机 HTTP 的 MCP server（以 Windows 本机上的模拟 server 代替 HiveMemory），回答 A2、B1、B2、B3、C1、C2、D1、D4、G1、G2、G3。需要调用模型的实测使用 owner 确认的凭据；没有获得授权时只做不调用模型的机制核对并注明。凭据、日志与产物不提交到仓库。
- 无法确定的项标为"未知"并说明原因，不作推测。

## 5. 交付物

1. **能力清单**：第 3 节的每个问题，取值为"支持 / 部分支持 / 不支持 / 未知"，附来源与验证程度。
2. **hook 对应表**：Claude Code 连接器用到的五个 hook 与 `PreToolUse` 改写参数的能力，在 Codex 中的对应物或缺口。
3. **接入路径比较**：至少比较"沿用 Claude Code 的形态"（Codex 插件、会话记录读取器复用 Python 公共骨架、插件声明的 MCP server、hook 补上 `turn`）与"插件自有工具"（不把 `/mcp` 直接配置为 Codex 的 MCP server，由插件以可信方式补上 `turn` 后调用）两条路径；`PreToolUse` 不能改写参数时，写明前一条路径的缺口与可行的替代。每条写明复用 HiveMemory 的哪些部分、需要新增什么、写入时确定话题与回声标记怎样满足、子 Agent 怎样处理、在 Windows 上的运行时依赖（例如复用 Python 公共骨架需要 Windows 上可用的 Python）与主要风险。只列影响，不作选择。
4. **引导 skill 的承载方式**：Codex 中可以承载使用说明的机制及其限制。
5. **与原生记忆的共存**：F1 为"支持"时，Codex 原生记忆与 HiveMemory 的分工、冲突点与需要的说明；为"不支持"时写明依据。

## 6. 不在范围

- 编写插件、连接器或任何生产代码；
- 决定接入路径，或修改 HiveMemory 的接入契约；
- 接入登记的运行时登记与配置热更新（[ROADMAP](../ROADMAP.md) 的配置热更新一项），只写明 Codex 接入需要登记什么；
- 引导 skill 的内容；
- HiveMemory 后端在 Windows 上的原生运行与打包，随桌面客户端的工作进行；实测以模拟的 MCP server 代替 HiveMemory；
- Linux、macOS 与 WSL 上的接入行为；
- Codex 的模型质量、回复效果或性能对比；
- controller 模式所需的程序化接口（app-server、SDK、`exec` JSONL 的驱动方式），见 [2026-10-20 快照](./harness-interface-survey-snapshot.md)，只在发现与 plugin 模式明显相关时顺带记录。

## 7. 调研结果

### 7.1 结论与验证边界

Codex `rust-v0.167.0` 已有默认开启的 plugins 与 hooks，**MCP 工具经过 `PreToolUse` 时可以改写完整参数**，因此“插件 + MCP + hook 注入 turn + 会话读取器”在接口层可行。但不能照搬 Claude Code 的 hook 输出、工具名和 JSONL 读取逻辑：

- `updatedInput` 必须与 `permissionDecision: "allow"` 一起返回；多个改写 hook 并行运行，最后完成的改写胜出。未信任、禁用、出错或超时的 hook 不能保证阻断原调用，不能把 hook 当作强制授权边界。[C-PRE]、[C-HOOK-DOC]
- 没有 `StopFailure`；`Stop` 发生在正常结束前且可以要求继续，`Interrupt` 也可能早于终态落盘。文件补发必须查实际终态，不能只凭 hook 或最后一段助手文本。[C-TURN]、[C-TASKS]、[C-HOOK-RUNTIME]
- 新建本地会话默认采用 paginated history；回退可在同一 thread ID 下换新 rollout 文件，fork 可通过 `history_base` 引用父历史。不是“一个会话 ID 对应一个永远追加的文件”。[C-STORE]、[C-LINEAGE]、[C-RECORDER]
- code mode 的内层工具仍可经过 hook，但持久化内层调用元数据有界，不能替代完整结果；外层输出混入记忆的回声问题仍存在。[C-REGISTRY]、[C-NESTED]

取证日期统一为 **2026-11-07**。GitHub `releases/latest` 返回非预发行 `rust-v0.167.0`，发布时间为 2026-11-06；通过 tag 的 commits API 得到 `086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a`，下载该提交的源码核对。[C-RELEASE] Context7 仅用于发现入口，能力结论以固定发行源码为准。

验证程度：**文档声明**指官方说明；**源码核对**指固定提交的实现及已有测试，未运行上游测试；**实际运行**仅完成发行元数据、源码下载与本地文本检查，未运行 Codex、模拟模型或 MCP 探针。当前执行环境为 Linux，没有可用的 Windows 执行端，未用 Linux/WSL 接入结果替代目标环境验证，未读取或使用模型凭据。表中的“支持”表示机制存在，不表示 Windows 验收通过。

### 7.2 能力清单

#### A. 插件与分发

| 编号 | 判定 | 结论、边界与来源 | 验证程度 |
|:---|:---|:---|:---|
| A1 | 支持 | 支持根目录 `plugin.json` 的 Agent Plugins 格式及 `.codex-plugin/plugin.json` 兼容格式；可携带 skills、MCP、hooks、apps。内联 `extensions.com.openai` 优先于独立 overlay，不能假设逐字段合并。支持本地/git marketplace，`codex plugin add <plugin>@<marketplace>` 安装、list/remove 管理，配置中的 enabled 控制启停；无需公开发布。通用清单与旧 `.mcp.json` 的字段语义不同。[C-PLUGIN-DOC]、[C-MANIFEST]、[C-LOADER]、[C-CLI] | 文档声明、源码核对 |
| A2 | 部分支持 | stdio 与 Streamable HTTP 可用。原生/兼容 MCP 配置有 `http_headers`、`env_http_headers`、`bearer_token_env_var`；通用 `mcp.json` 的 `headers` 被转换为静态请求头，未见对其任意 `${TOKEN}` 的通用展开。不能原样复制 Claude 的 headers 配置。optional server 失败会报告状态、缺席工具，不必终止全部会话；required 配置可使启动失败。默认启动/调用超时为 10/60 秒；没有可承诺覆盖所有失败的透明写调用重试。[C-MCP-CONFIG]、[C-MCP-LEGACY]、[C-MCP-PORTABLE]、[C-MCP-DOC] | 文档声明、源码核对；Windows HTTP 未实测 |
| A3 | 支持 | 配置层由低到高收集 hook，再加启用插件的 hook，不是同事件整组覆盖；JSON 与 TOML 并存可能同时装载并警告。项目层需信任，非 managed hook 还需按内容 hash 审阅信任；插件安装不等于 hook 已获信任。命令从 stdin 读 JSON，cwd 为会话工作目录，环境来自会话快照并过滤受限变量；提供 PLUGIN_ROOT/PLUGIN_DATA 及 Claude 兼容别名。[C-DISCOVERY]、[C-COMMAND]、[C-HOOK-DOC] | 文档声明、源码核对 |
| A4 | 部分支持 | 本版 `plugins`、`hooks` 为 Stable 且默认 true；`plugin_hooks` 已是移除的旧开关。事件枚举与输出 schema 可查，但未找到跨版本不变承诺。所列四种 handler 中 command、mcp_tool 有执行实现，prompt、agent 会被 discovery 跳过，不能由枚举推断可用。与旧快照相比会话/插件仍有变化，须固定并复核版本。[C-FEATURES]、[C-HOOK-SCHEMA]、[C-DISCOVERY] | 源码核对 |
| A5 | 部分支持 | CLI、exec 与 app-server 复用 core 的配置、插件、hook 和本地存储，IDE/桌面经 app-server 使用这些能力；这不保证不同客户端所捆绑版本、UI、信任与账户策略完全相同。直接 `CodexThread.call_mcp_tool` 绕过工具 registry hook；不能把所有 app-server MCP 调用当作模型工具调用。官方插件网页将 hooks 限于手动安装的桌面插件、禁止进入公开目录，而固定 CLI 源码也有装载执行路径；分别记录分发声明与源码能力，未验证各 UI。[C-EXEC]、[C-DIRECT]、[C-PLUGIN-DOC]、[C-LOADER] | 文档声明、源码核对；跨客户端实测未知 |

#### B. hook

| 编号 | 判定 | 结论、边界与来源 | 验证程度 |
|:---|:---|:---|:---|
| B1 | 部分支持 | 14 个事件的触发点见 7.3。Stop 只在正常生成循环准备结束时运行，失败/取消不保证进入；没有 StopFailure。Interrupt 对顶层活动任务的用户中断有效，不覆盖所有关机、崩溃或子 Agent 路径；最终结局仍需读取 TurnComplete/TurnAborted。Stop 也能要求继续，不能在触发即认定成功。[C-TURN]、[C-TASKS]、[C-HOOK-RUNTIME] | 源码核对 |
| B2 | 支持 | MCP handler 支持把 updatedInput 序列化回调用参数；需返回 PreToolUse 的 hookSpecificOutput，且 permissionDecision=allow。工具名简单且无碰撞时 matcher 可用 `^mcp__hivememory__(memory_write\|memory_update)$`。改写后才进行需要的审批，审批处理读取改写后的 payload；allow 不自动豁免所有审批。多个改写结果按完成顺序取最后一份，deny 优先，不是顺序叠加。[C-MCP-HANDLER]、[C-PRE]、[C-REGISTRY] | 源码核对；Windows matcher/审批 UI 未实测 |
| B3 | 部分支持 | PreToolUse 有 session_id、turn_id、transcript_path、cwd、model、tool_name、tool_input、tool_use_id、permission_mode 与可选 agent 字段。Stop/Interrupt 也有 turn_id；SessionStart/SessionEnd 不提供同样的轮次字段。transcript_path 可为空，且 payload 不直接给整轮用户消息，必须读取相应历史、排除指令注入并处理追加输入，不能把 turn_id 当文本索引。[C-HOOK-SCHEMA]、[C-HOOK-RUNTIME] | 源码核对 |
| B4 | 部分支持 | 同步 handler 在对应处理点被等待；显式 deny/exit 2 可阻断支持阻断的事件，错误、无效 JSON、超时不等于阻断。command 默认多数事件 600 秒；SessionEnd/Interrupt 默认 1 秒、上限 3 秒。async command 不等结果、不参与改写/决策，每会话最多 8 个在途，结束时取消；SessionEnd 强制同步。可自行启动进程，但 Windows 脱离、句柄与退出后的持续运行未验证，async 不是持久补发服务。[C-HOOK-DOC]、[C-DISCOVERY]、[C-COMMAND]、[C-PRE] | 文档声明、源码核对；脱离进程未知 |
| B5 | 部分支持 | 子线程 hook 给 agent_id/type；PreToolUse 的 turn_id 与 transcript_path 指向子线程自身，session_id 实测为根会话（2026-11-08 更正，见 7.9）。SubagentStop 的 session_id/turn_id 仍来自子线程，而 transcript_path 指父、agent_transcript_path 指子；须按事件 schema 区分，不能从同名字段猜根会话。嵌套父链与父轮绑定仍需适配器解析，不能用父会话“最近一轮”。[C-HOOK-SCHEMA]、[C-HOOK-RUNTIME] | 源码核对 |

#### C. 会话记录

| 编号 | 判定 | 结论、边界与来源 | 验证程度 |
|:---|:---|:---|:---|
| C1 | 部分支持 | 本地 rollout 为 CODEX_HOME 下 sessions/日期目录中的 JSONL，也有 archived_sessions；按需物化，hook 请求路径会促使物化，非本地或 ephemeral 情形不能假定有文件。SessionMeta.id、turn_id/root_turn_id 提供身份。RolloutItem、TurnItem 与历史 schema 在 Rust 源码公开，官方 app-server 有历史读取接口，但原始文件没有外部读取器稳定承诺；新会话默认 Paginated，旧记录可为 Legacy。[C-RECORDER]、[C-STORE]、[C-PROTOCOL]、[C-POLICY]、[C-HOOK-RUNTIME] | 源码核对 |
| C2 | 部分支持 | 记录含 session_meta、turn_context、response_item、event_msg 等。Paginated 用 ItemCompleted 的 TurnItem，Legacy 还保存 UserMessage/AgentMessage/McpToolCallEnd 等旧事件；不可把不同表示重复导入。调用以 call_id 关联，支持多条并行结果，未见“只留最后一个”的设计；但 MCP/命令完成条目有 64 KiB 输出上限。思考可能只有摘要或加密字段；steering/追加输入可在同轮穿插，HiveMemory 单 user_message 无法无损表达其位置。[C-PROTOCOL]、[C-POLICY]、[C-TURN-ITEM] | 源码核对；并行/追加输入未实测 |
| C3 | 部分支持 | TurnComplete 带 error，可区分成功与失败；TurnAborted 有原因。上下文超限/模型错误应以最终任务结局判断，审批拒绝通常是工具级失败，模型仍可继续。流式 error 或单个工具失败不是整轮 failed。强杀可能留下缺终态、缺结果或未 flush 的尾部；没有可靠终态时结局未知，不能把文件静止自动标 completed。[C-TASKS]、[C-PROTOCOL]、[C-TURN] | 源码核对 |
| C4 | 部分支持 | resume 延续逻辑 thread；fork 创建新 thread，可引用父 rollout 的有界历史前缀。新式 revert 保持 thread ID 却切换到新的 rollout ID/文件，旧式 rollback 为投影事件；compacted 条目/替换上下文不等于删除所有历史。必须跟踪 history_mode、history_base 和文件更换，排除 fork 继承前缀重导，不能只用路径+字节偏移代表会话。新执行需用实际 turn ID，不沿用被回退执行的身份。[C-RECORDER]、[C-LINEAGE]、[C-PROTOCOL]、[C-POLICY] | 源码核对 |
| C5 | 部分支持 | 子 Agent 单独持有 thread/rollout，SessionSource::SubAgent 的 parent_thread_id 与相关通信记录可关联；父历史可有 inter_agent_communication 及工具的最终返回。应只导入顶层一轮与子 Agent 最终交回消息，不能把父侧聚合结果和同一通信重复导入。恢复后异步子任务与父轮的准确映射需保留派生时绑定，单独子文件不充分。[C-PROTOCOL]、[C-HOOK-RUNTIME]、[C-POLICY] | 源码核对、适配分析 |

#### D. MCP

| 编号 | 判定 | 结论、边界与来源 | 验证程度 |
|:---|:---|:---|:---|
| D1 | 部分支持 | 逻辑 server/tool 与模型 callable namespace/name、hook 扁平名字是不同表示。普通 server 的 namespace 来自 server 名，有清洗/长度处理；hook 用 `mcp__<server>__<tool>`，不是 Claude 的 plugin 前缀规则。模型可呈现 namespace.tool，原始响应与结构化完成事件也可能分别保存 namespace/name 或 server/tool。读取器应优先结构化身份，按固定版本处理退化字符串。[C-MCP-HANDLER]、[C-MCP-CLIENT]、[C-MCP-TOOLS]、[C-TURN-ITEM] | 源码核对；实际 Windows 记录待采集 |
| D2 | 支持 | server 可设 default_tools_approval_mode，单工具可设 `tools.<tool>.approval_mode`，取值有 auto/prompt/writes/approve；另有 server enabled 与 enabled_tools/disabled_tools。最终还受全局审批、信任与工具注解影响，不能断言所有 MCP 默认每次审批或默认全放行。可为 HiveMemory 明确配置策略，但 hook 的 allow 只使参数改写有效，不替代该策略。[C-MCP-CONFIG]、[C-MCP-DOC]、[C-REGISTRY] | 文档声明、源码核对 |
| D3 | 部分支持 | server instructions 用作 namespace 说明进入工具上下文；官方建议重要内容放在前 512 字符，这是可见性建议，不是完整 instructions 的协议上限。实现保留 namespace_description，未见这一层的专属硬长度限制；实际工具发现与上下文预算会影响可见性，不能代替 skill 与短引导。[C-MCP-DOC]、[C-MCP-CLIENT] | 文档声明、源码核对 |
| D4 | 部分支持 | Streamable HTTP 客户端有 JSON 响应、可选 session ID 路径，与 HiveMemory 无状态 tools 接口在机制上相容；不要求 HiveMemory 提供订阅才能调用工具。使用带端口的 127.0.0.1 URL 可表达所需目标，但实际 Host、代理/IPv6、协议协商及 Windows 回环尚未验证，不能报告端到端兼容已通过。[C-HTTP]、[C-MCP-DOC] | 源码核对、适配分析；实测未执行 |
| D5 | 部分支持 | code mode/工具内部执行可进入同一 registry，因此内层 MCP PreToolUse 仍可执行。ExecutedToolCalls 是有界元数据，含 complete 标志，不能保证完整原始结果；外层输出仍可能包含取回记忆。直接 app-server call_mcp_tool 不经该 hook，hook 自身的 mcp_tool handler 也不递归触发普通工具 hook/审批，不能据“是 MCP”推断受注入保护。[C-REGISTRY]、[C-NESTED]、[C-DIRECT]、[C-HOOK-DOC] | 文档声明、源码核对 |

#### E. 指令与 skill

| 编号 | 判定 | 结论、边界与来源 | 验证程度 |
|:---|:---|:---|:---|
| E1 | 支持 | 用户 ~/.agents/skills、项目从 cwd 到仓库根的 .agents/skills 与插件 skills 均有发现入口；元数据进入上下文，按显式调用或相关性加载正文。现有 name/description frontmatter 可用；插件 skill 带插件身份供消歧，不应照抄 Claude 的 slash 调用名。具体 CLI/桌面的显示与调用名仍需插件实装验证。[C-SKILL-DOC]、[C-SKILLS]、[C-PLUGIN-DOC] | 文档声明、源码核对 |
| E2 | 支持 | CODEX_HOME 的 AGENTS.override.md/AGENTS.md 与项目根到 cwd 的分层说明被读取，每层优先 override；合并预算默认 32 KiB，可配 project_doc_max_bytes。插件可携带 skill，并由支持的 hook 返回 additionalContext 等附加说明；不是任意替换宿主系统提示的扩展 API。[C-AGENTS-DOC]、[C-HOOK-DOC]、[C-SKILLS] | 文档声明、源码核对 |
| E3 | 部分支持 | 压缩后会重建运行指令、工具/skill 目录等；已展开 skill 正文与 hook 上下文不保证逐字永存，可能只在摘要中。SessionStart 的 compact 来源与 Pre/PostCompact 提供观察点。应让短引导可重建、长手册可重读，不靠某一次注入永久常驻。[C-HOOK-DOC]、[C-TURN]、[C-SKILLS] | 文档声明、源码核对、适配分析 |

#### F. 原生记忆与共存

| 编号 | 判定 | 结论、边界与来源 | 验证程度 |
|:---|:---|:---|:---|
| F1 | 支持 | 有跨会话 memories 子系统：从 rollout 提炼、汇总本地笔记并向后续上下文注入摘要，可选专用工具。features.memories 在本版源码默认 false，用户/产品配置可改变；memories.generate_memories 与 memories.use_memories 分别控制生成与使用。memories.version 决定 CODEX_HOME/memories 或 memories_v2，不能只认旧目录；摘要读取有 token 预算。这是独立于 HiveMemory 的本地记忆。[C-FEATURES]、[C-MEMORY-CONFIG]、[C-MEMORY-VERSION]、[C-MEMORY-WRITE]、[C-MEMORY-EXT]、[C-MEMORY-PROMPT] | 源码核对；账户实际配置未知 |
| F2 | 支持 | 自动 token 阈值、上下文溢出恢复与手动 compact 可触发压缩；PreCompact/PostCompact 及 compacted 记录可观察。压缩摘要是模型工作上下文，不是用户新知识；归档应从原始可用记录重建，不能只读压缩后的模型视图。[C-HOOK-DOC]、[C-TURN]、[C-PROTOCOL] | 文档声明、源码核对 |
| F3 | 未知 | 第三方记忆插件是可选样本，本次未扩展调查 Codex 社区插件的安装量、兼容性与存储机制；不把“未调查”解释为没有此类插件，也不以第三方行为支持接入路径。本体 memories 已足以回答必选的共存分析。[C-PLUGIN-DOC] | 未调查可选样本 |

#### G. Windows、认证与许可

| 编号 | 判定 | 结论、边界与来源 | 验证程度 |
|:---|:---|:---|:---|
| G1 | 支持 | 官方支持原生 Windows CLI/桌面/IDE；CLI 可用 npm 的 @openai/codex（本次固定 0.167.0）或官方 Windows 二进制。默认 CODEX_HOME 为用户主目录 .codex，其下 config.toml、hooks.json、plugins、sessions 等；项目 .codex 为项目配置，skill 另有 .agents 路径。可设隔离 CODEX_HOME；原生安装、长路径、终端行为未实测。[C-WINDOWS-DOC]、[C-CLI-README]、[C-HOOK-DOC]、[C-SKILL-DOC] | 文档声明、源码核对；Windows 未执行 |
| G2 | 部分支持 | commandWindows 只替换命令文本，不选择 shell。core 通常传入会话 shell 的 program/args；无显式 shell 时 Windows runner 回退 COMSPEC/cmd.exe /C。因此不能统一假定 PowerShell，也不能一律写 cmd 语法。PowerShell 的 $env:PLUGIN_ROOT 与 cmd 的 %PLUGIN_ROOT% 不通用；脚本路径含空格必须按实际 shell 引用，JSON 反斜线另需转义。Python 应确认 python.exe/py -3，不能沿用 python3 假设。MCP 环境解析由配置实现决定，见 A2。[C-COMMAND]、[C-SESSION]、[C-HOOK-DOC] | 源码核对；命令引用组合未实测 |
| G3 | 部分支持 | hook command 直接由宿主启动，不经普通模型 shell 工具的审批/沙箱执行链；仍受 Windows 账户、进程与网络策略约束。模型 sandbox 的网络开关不能直接证明 hook/MCP 无法访问 localhost，反之也不能证明所有部署允许。PLUGIN_DATA 可放状态，写外部目录依赖 OS 权限；实际回环访问、状态锁与进程退出需 Windows 验证。[C-COMMAND]、[C-SESSION]、[C-WINDOWS-SANDBOX] | 文档声明、源码核对；OS 策略结果未知 |
| G4 | 部分支持 | 本地插件/hook/MCP 与模型 ChatGPT/API key 认证是不同链路，固定实现未把本地 command hook 统一限定为某订阅；apps、目录分发与组织策略则有额外账户条件。HiveMemory 自己的 Bearer/身份头仍需单独配置；未做账户矩阵实测，不能承诺所有计划与管理策略完全等价。[C-FEATURES]、[C-LOADER]、[C-PLUGIN-DOC]、[C-MCP-DOC] | 文档声明、源码核对；账户矩阵未知 |
| G5 | 支持 | codex exec 使用共用 app-server/core，并处理 hook 开始/完成事件；无头运行不是天然禁用插件/hook。非交互时须事先解决 trust，不能等待 UI 审阅；--ephemeral 不保存会话文件，文件读取器路径因此不可用。正常退出也不保证所有异步回流已持久确认。[C-EXEC]、[C-HOOK-DOC]、[C-RECORDER] | 源码核对；Windows 无头实测未执行 |
| G6 | 部分支持 | Codex 源码 Apache-2.0，可按许可使用、修改和分发并保留相关通知；插件自身和依赖另计。公开插件目录不接收 hooks，不妨碍代码所支持的本地安装路径。模型/ChatGPT 服务条款不是源码许可授予的权利，本次未作法律审查，账户与分发限制需按实际场景确认。[C-LICENSE]、[C-PLUGIN-DOC] | 许可文本、文档声明；服务条款限制未知 |

### 7.3 事件时机与 Claude Code 对应

| Codex 事件 | 本版触发含义 |
|:---|:---|
| SessionStart / SessionEnd | 会话开始或恢复/压缩等来源；结束处理。SessionEnd 不代表每一轮，也不保证强杀执行。 |
| UserPromptSubmit | 用户提示提交、模型处理之前；不是持久的整轮终态。 |
| PreModelRequest / PostModelRequest | 每次模型请求尝试前/后；包括重试，不能作为一轮边界。 |
| PreToolUse / PostToolUse | 经工具 registry 的调用前/后；可改写/观察，非所有内部 RPC 都经过它。 |
| PermissionRequest | 即将请求审批时，可给决策；不需要审批的操作不会触发。 |
| PreCompact / PostCompact | 上下文压缩前/后，压缩可能发生在一轮中间。 |
| SubagentStart / SubagentStop | 子线程开始与停止处理；子终止不等于父轮结束。 |
| Stop | 正常生成循环尝试结束，可请求继续；早于最终完成标记。 |
| Interrupt | 顶层活动任务收到用户中断；还须等待实际取消终态。 |

以上时机来自 [hook 文档][C-HOOK-DOC]、[turn 执行][C-TURN]、[任务收尾][C-TASKS]与 [hook runtime][C-HOOK-RUNTIME]。当前枚举不是“每次都会执行”的保证。

| Claude Code 入口 | Codex 对应 | 复用限制 |
|:---|:---|:---|
| SessionStart | SessionStart | 可做启动补发；字段/来源值按 Codex schema 解析。 |
| Stop | Stop + 持久终态检查 | hook 可能要求继续，必须等实际 TurnComplete。 |
| StopFailure | 无直接对应 | 读取带 error 的 TurnComplete；下一次启动/提交输入/结束时补查，无法仅凭 hook 保证失败后一刻回流。 |
| SessionEnd | SessionEnd | 顶层结束、短超时；不适合依赖长 HTTP 重试，不能覆盖崩溃。 |
| PreToolUse | PreToolUse | 支持 MCP updatedInput，但需 allow、不同 matcher 与冲突处理；改写失败并非默认拒绝。 |

用户中断另由 Interrupt 提示扫描。hook runtime 会在若干节点 flush rollout，但 Interrupt/Stop 所需的最终任务标记可以在 hook 之后才生成。因此“hook 到达 → 读一次文件 → 确认结束”的实现存在竞态；应等待真实终态并保留后续补查机会。长期静止不能自行证明完成。[C-HOOK-RUNTIME]、[C-TASKS]

### 7.4 两条接入路径的影响比较

以下是适配分析，不是 owner 决定。两条路径均沿用当前 Import Bus 与 `/mcp`；本次没有证据要求先改变服务端契约。

| 路径 | 可复用与新增 | turn、回声与子 Agent | Windows 依赖与主要风险 |
|:---|:---|:---|:---|
| Codex 插件 + HTTP MCP + PreToolUse + Python 读取器 | 复用手册、Python 的 HTTP 提交/错误分类与确认后前进原则；新增 Codex 插件清单、hook 输出适配、rollout 投影与身份/终态处理。并非只换一个 reader。 | hook 读取实际当前轮、覆盖完整 turn，工具与导入共用 ID；直接调用按 server/tool 归一化。子调用沿派生绑定回到根轮，只导入父级最终交回；间接调用仍有回声限制。 | Codex + Windows Python；hook shell/引号、信任未完成、并行改写冲突、缺失败 hook、文件换代和锁均需验证。 |
| 插件自有工具：本地 MCP 包装服务 + 适配器上下文 | Codex 插件不是 dsh 的 JS registerTool 工厂；可由插件声明本地包装 MCP server，再由它调用 HiveMemory。新增包装工具/schema、可信上下文交接和客户端；回流仍需读取器/生命周期接线。 | 包装工具可不向模型暴露 turn，并在执行时注入，但 server 自身只收到工具参数，不能凭 stdio 进程或 HTTP 连接猜当前 thread/turn。需可信关联通道，尤其共享 server、并行调用和子 Agent；只包装四个工具不能自动解决此问题。直接工具按包装名称映射能力，嵌套回声同样存在。 | 包装服务可用 Python 或 Node，取决于实现；不额外承诺任选组合都兼容。增加运行时状态、并发与进程收尾；若仍由可跳过的 hook 提供关联，不能宣称已消除 hook 缺失风险。 |

**真实复用边界**（HiveMemory 基线 `acbbe32e165dcb0bf450b454ad1b3547b690db8f`）：Python [`TranscriptReader`](../../integrations/hivememory/connector/core/transcript.py) 与 [`sync.py`](../../integrations/hivememory/connector/core/sync.py) 的 turn 区间、字节水位及 closed_by_next_turn/looks_complete 反映 Claude 的文件模型；Codex 的历史前缀引用、同 thread 换文件、失败终态与追加输入需要审查这些假设。HTTP 客户端与“确认后推进、首次接入不回填旧历史”的策略可复用；不能把现有 30 分钟静止启发式直接当作 Codex 结局证明。已有 [`hooks.py`](../../integrations/hivememory/connector/claude_code/hooks.py) 的 JSON 输出与 matcher 也需要专门适配。

**可信 turn 的限制**：覆盖模型提供的 turn 是必要条件，但 MCP 当前 schema 并不证明字段一定来自适配器。若 hook 未加载/未信任/失败，原调用仍可能继续：缺 turn 会被现有服务端拒绝，模型伪造的非空 turn 则不能仅靠该机制鉴别。运行中的 hook 可在无法解析上下文时显式 deny；“确保 hook 必定运行”需要安装/运行约束或另一可信绑定方案。这是 [现有 MCP 契约](../contracts/mcp.md#6-连接器调用名轮次信息与回声标记) 的适配器信任边界，不是本次擅自新增认证协议的理由。[C-PRE]、[C-HOOK-DOC]

**身份与回流**：候选外部身份应锚定逻辑 thread ID 与实际根 turn ID，不以文件路径或时间戳编号；写入与回流都从同一绑定取得 source/conversation/turn。fork 前缀不能当新交互重导，revert 不会撤销已导入的旧交互。Import Bus 的有限去重窗口也不能替代本地持久确认状态，不能宣称无限期 exactly-once。steering/多输入的相对位置在当前导入结构中不能无损表达，具体支持范围留给接入计划。

**子 Agent**：顶层派生时保留根轮绑定，沿 parent_thread_id 向上解析；子自己的 turn_id 不能直接用于写入。父级只取最终交回消息，不导入子完整轨迹。跨父轮继续运行、恢复后的子进程或没有绑定的后台工作不能猜测根轮，应作为未支持情况显式处理。[C-HOOK-RUNTIME]、[C-PROTOCOL]

**回声**：直接记忆调用归一化 SEARCH/READ/WRITE/UPDATE，去掉注入的 turn 并按当前契约处理 aliases/alias。内层调用元数据即便齐全，外层 code mode 结果仍可能混合记忆与新输出；不宜全删外层，也不能声称已完全去回声。限制间接调用与补足输出归因是不同选项，需要后续路径决定。[C-NESTED]

### 7.5 引导 skill 的承载

现有 [`hivememory/SKILL.md`](../../integrations/hivememory/skills/hivememory/SKILL.md) 可作为插件 skill 分发，保持一份正文；目录发现和按需读取符合当前 frontmatter。安装时还需核对插件限定名在目标 CLI/桌面中的显示，不能把 Claude 的命令写法搬入说明。[C-SKILL-DOC]、[C-PLUGIN-DOC]

MCP instructions 适合简短说明工具用途，skill 承载完整操作规则；需要每轮可重建的短引导可由适当 hook 附加。AGENTS.md 属于用户/项目说明，不应由插件自动改写；任何提示文本都不能生成可信 turn 或代替 hook 接线。压缩后允许重新读取 skill，不要求其正文永久保留。[C-AGENTS-DOC]、[C-HOOK-DOC]

### 7.6 与 Codex 原生记忆共存

本版 memories 的生成与使用是独立开关，路径也按 memories.version 区分；读取阶段可将本地 memory_summary 注入，写入阶段从合适 rollout 生成和汇总笔记。可选专用 memory tools 与 HiveMemory 的四个 MCP 工具不是同一命名空间。[C-MEMORY-CONFIG]、[C-MEMORY-VERSION]、[C-MEMORY-WRITE]、[C-MEMORY-EXT]

适配说明可区分：Codex 本地记忆供本机工作延续，HiveMemory 管理其 Workspace 内的长期 Memory/Topic 与跨 harness 访问；项目 AGENTS.md/当前代码仍按既有手册优先于记忆。二者没有自动去重或冲突协调协议，不应把本地 memory 注入、压缩摘要或检索结果再次当作新用户事实回流。Codex 有 external context 对记忆生成的抑制配置，但不能据此承诺已经排除了 HiveMemory 的所有回声。是否关闭任一原生开关属于使用选择，本报告不替 owner 作决定。[C-MEMORY-CONFIG]

### 7.7 HiveMemory 侧影响与 Windows 验证缺口

若后续实施，需要独立 principal/凭据登记（例如 `harness:codex`）、具体 Agent 与四项操作白名单，稳定 source 名，以及工具调用与导入共用的身份。配置仍按现有启动装载，不引入热更新。本次未修改生产集成、后端、skill 或接入登记，也未创建实施 Plan。

| 要验证的机制 | 本次状态 | Windows 最小验证应取得的证据 |
|:---|:---|:---|
| 隔离安装/插件/信任（A1/A3/G1） | 未执行：无 Windows 执行端 | 独立临时 CODEX_HOME、本地 marketplace，安装固定 CLI；记录信任前跳过、信任后执行及禁用/卸载结果，不动 owner 配置。 |
| shell 与路径（G2/G3） | 仅源码核对 | 原生 PowerShell 与实际配置 shell 的 commandWindows 行为，含空格/中文路径、Python/Node 可执行路径、PLUGIN_ROOT/DATA、stdin JSON 与状态写入。 |
| HTTP MCP（A2/D1/D4） | 未执行 | 无真实凭据的 Windows 模拟 server 记录 initialize/list/call、Host、Bearer/身份头与 JSON 响应；不提供 session ID/订阅也能完成，覆盖拒连/恢复/超时。 |
| 参数改写与审批（B2/B3） | 仅源码核对 | 直接与 code mode 写入由模拟 server 收到可信 turn；审批看到改写值；测试无效输出、未信任、超时和双改写 hook，确认失败边界。 |
| 生命周期与记录（B1/C1/C2/C3/G5） | 仅源码核对 | 采集 SessionStart/Stop/Interrupt/SessionEnd、TurnComplete/Aborted 的真实先后；完成、失败、中断、强杀、exec 与 ephemeral 不误判。 |
| 历史与子线程（B5/C4/C5） | 仅源码核对 | resume/fork/revert/compact、并行调用与子 Agent 的根轮绑定；文件更换不漏不重，直接和嵌套结果的截断可被辨认。 |

可先用脚本化模型与模拟 MCP 服务完成无凭据机制验证；涉及真实模型时须使用 owner 确认的凭据。原生 Windows 上已有 Python 连接器的锁、HTTP 与原子写亦未验证，不因本次源码研究自动视为可用。

### 7.8 固定来源

下列来源访问日期均为 **2026-11-07**。GitHub 源码链接固定到发行提交；官方网站未提供与发行一一对应的不可变文档版本，标为当日文档声明，不能单独证明旧版具备能力。网页与源码范围不一致处已在 A5 注明；上游测试仅阅读，未运行。

[C-RELEASE]: https://github.com/openai/codex/releases/tag/rust-v0.167.0
[C-PLUGIN-DOC]: https://developers.openai.com/plugins/build/plugins
[C-HOOK-DOC]: https://learn.chatgpt.com/docs/hooks
[C-MCP-DOC]: https://learn.chatgpt.com/docs/extend/mcp
[C-SKILL-DOC]: https://learn.chatgpt.com/docs/build-skills
[C-AGENTS-DOC]: https://learn.chatgpt.com/docs/agent-configuration/agents-md
[C-WINDOWS-DOC]: https://learn.chatgpt.com/docs/windows/windows-app
[C-WINDOWS-SANDBOX]: https://learn.chatgpt.com/docs/windows/windows-sandbox
[C-MANIFEST]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/core-plugins/src/manifest.rs
[C-LOADER]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/core-plugins/src/loader.rs
[C-CLI]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/cli/src/plugin_cmd.rs
[C-FEATURES]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/features/src/lib.rs
[C-HOOK-SCHEMA]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/hooks/src/schema.rs
[C-DISCOVERY]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/hooks/src/engine/discovery.rs
[C-COMMAND]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/hooks/src/engine/command_runner.rs
[C-PRE]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/hooks/src/events/pre_tool_use.rs
[C-HOOK-RUNTIME]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/core/src/hook_runtime.rs
[C-SESSION]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/core/src/session/mod.rs
[C-TURN]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/core/src/session/turn.rs
[C-TASKS]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/core/src/tasks/mod.rs
[C-REGISTRY]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/core/src/tools/registry.rs
[C-DIRECT]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/core/src/codex_thread.rs
[C-MCP-HANDLER]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/core/src/tools/handlers/mcp.rs
[C-NESTED]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/core/src/tools/executed_tool_calls.rs
[C-MCP-CONFIG]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/config/src/mcp_types.rs
[C-MCP-LEGACY]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/codex-mcp/src/plugin_config.rs
[C-MCP-PORTABLE]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/codex-mcp/src/agent_plugin_config.rs
[C-MCP-CLIENT]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/codex-mcp/src/rmcp_client.rs
[C-HTTP]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/rmcp-client/src/http_client_adapter.rs
[C-MCP-TOOLS]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/codex-mcp/src/tools.rs
[C-RECORDER]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/rollout/src/recorder.rs
[C-STORE]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/thread-store/src/local/mod.rs
[C-LINEAGE]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/thread-store/src/local/rollout_lineage.rs
[C-POLICY]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/rollout/src/policy.rs
[C-PROTOCOL]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/protocol/src/protocol.rs
[C-TURN-ITEM]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/protocol/src/items.rs
[C-SKILLS]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/ext/skills/src/extension.rs
[C-MEMORY-CONFIG]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/config/src/types.rs
[C-MEMORY-VERSION]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/protocol/src/memory_version.rs
[C-MEMORY-WRITE]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/memories/write/src/lib.rs
[C-MEMORY-EXT]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/ext/memories/src/extension.rs
[C-MEMORY-PROMPT]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/ext/memories/src/prompts.rs
[C-EXEC]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/codex-rs/exec/src/lib.rs
[C-CLI-README]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/README.md
[C-LICENSE]: https://github.com/openai/codex/blob/086ccc833bc9cf01024ad2dd6af7dbd421b6bf9a/LICENSE

### 7.9 Windows 原生机制实测（2026-11-08）

**环境**：Windows 11 原生，Codex CLI `codex-cli 0.167.0`（npm 包，装在 fnm 管理的 Node v24.21.0 下），Python 3.12.10。使用独立的临时 `CODEX_HOME`，不动 owner 已有的配置；模型提供方指向本机模拟的 Responses API（`127.0.0.1:18080`，按提示词中的关键字返回固定的工具调用或回复），MCP 指向本机模拟 server（`127.0.0.1:18081`，只记录请求）。没有调用真实模型或 HiveMemory 后端，没有使用任何账户凭据。探针、模拟服务与日志都在会话临时目录，不进入仓库。

**探针插件**按"一个目录同时作为 Claude Code 与 Codex 插件"的布局准备：根目录同时放 `.codex-plugin/plugin.json` 与 `.claude-plugin/plugin.json`；Codex 清单显式指定 `hooks/codex.json`、`codex.mcp.json` 与 `skills/`；另放 Claude Code 默认读取的 `.mcp.json` 与 `hooks/hooks.json` 作为诱饵，检查 Codex 是否误读。marketplace 只有 `.claude-plugin/marketplace.json`。

以下均为**实际运行**的结果；源码定位见第 7.8 节的固定提交。

| 项目 | 结果 |
|:---|:---|
| marketplace 与安装（A1） | `codex plugin marketplace add <目录>` 读取 `.claude-plugin/marketplace.json`；`codex plugin add <插件>@<marketplace>` 把插件目录整份复制到 `CODEX_HOME\plugins\cache\<marketplace>\<插件>\<版本>`，版本取自 Codex 清单。改动插件文件后要重新执行 `add` 才进入缓存。 |
| 两份清单并存（A1/A3） | Codex 按根目录 `plugin.json` → `.codex-plugin/plugin.json` → `.claude-plugin/plugin.json` 的顺序取第一份（源码），实测用的是 `.codex-plugin`。清单写明 `mcpServers` 与 `hooks` 的路径后，不再读取默认的 `.mcp.json` 与 `hooks/hooks.json`：诱饵 MCP 没有收到任何请求，hook 列表里只有 `hooks/codex.json` 的 12 个 hook。 |
| hook 信任（A3） | 安装后的 hook 处于未信任状态，不运行。经 app-server 的 `hooks/list` 取得 hook 的当前 hash，再以 `config/batchWrite` 写入 `hooks.state`（TUI 的 hook 审阅走同一接口），之后才运行；键形如 `<插件ID>:hooks/codex.json:<事件>:0:0`，同时记录声明 hash 与插件内容 hash。重新安装改动过的插件后，hook 变为"已修改"，需要再次信任。另有命令行开关 `--dangerously-bypass-hook-trust` 跳过信任（源码），实测未使用。 |
| hook 命令与环境（G2/G3） | 命令 `python "${PLUGIN_ROOT}/probe_hook.py" …` 正常运行：Codex 在交给 shell 之前把命令文本中的 `${PLUGIN_ROOT}` 替换为插件缓存目录，`python` 解析到用户安装的 Python 3.12。环境中有 `PLUGIN_ROOT`、`PLUGIN_DATA`（`CODEX_HOME\plugins\data\<插件>-<marketplace>`）与 `CLAUDE_PLUGIN_ROOT`、`CLAUDE_PLUGIN_DATA` 两个别名，用户环境变量照常传入。hook 以管道读写标准输入输出，Python 的标准输出按 cp1252 编码：探针用 `ensure_ascii=False` 输出中文时抛出编码错误，hook 记为失败；改为 ASCII 转义的 JSON 后正常。`SessionEnd` 与 `Interrupt` 的超时被截到 3 秒并给出警告。 |
| HTTP MCP（A2/D4） | 插件 MCP 文件为 `{"mcpServers": {...}}`，`type: "http"`。`bearer_token_env_var` 从环境变量取得凭据并作为 `Authorization: Bearer` 发送；`http_headers` 原样发送；`env_http_headers` 在变量未设置时不发送该请求头。`Host` 为 `127.0.0.1:<端口>`，`Accept` 为 `text/event-stream, application/json`；按 `initialize`、`notifications/initialized`、`tools/list`、`tools/call` 的顺序调用，JSON 响应、无会话 ID 均可正常完成。`url` 中的环境变量是否展开未实测。 |
| 工具名与 matcher（D1） | 模型请求中的工具为 namespace `mcp__hivememory` 下的 `memory_*`；hook 中的 `tool_name` 为 `mcp__hivememory__memory_write`；matcher `^mcp__hivememory__(memory_write\|memory_update)$` 只对写入与更新触发，检索调用前不触发 `PreToolUse`。 |
| 审批（D2） | 不设审批模式时，MCP 写入工具需要审批：在审批策略为 `never` 的无头运行中直接失败（"MCP tool call requires approval, but approval policy is never"），并触发 `PermissionRequest` hook。在插件 MCP 文件中设 `default_tools_approval_mode: "approve"` 后不再审批，调用正常完成。 |
| 参数改写（B2/B3） | `PreToolUse` 返回 `permissionDecision: "allow"` 与 `updatedInput` 后，MCP server 收到的参数是改写后的：模型自己填的 `turn` 被 hook 补上的值覆盖，中文完整。会话记录中 `response_item` 的 `function_call` 保留模型原始参数，`item_completed` 的 `McpToolCall` 是改写后的参数。每次 `tools/call` 的 `_meta.x-codex-turn-metadata` 还带有 `session_id`、`thread_id`、`turn_id`、`thread_source`（`user` / `subagent`）等字段，`_meta.plugin_id` 为插件 ID；这些由 Codex 填写，不经过模型。 |
| hook 输入（B3） | 各事件带 `session_id`、`transcript_path`、`cwd`、`model`、`permission_mode`；`UserPromptSubmit`、`PreToolUse`、`Stop` 另带 `turn_id`，`UserPromptSubmit` 带用户原文 `prompt`，`Stop` 带 `last_assistant_message` 与 `stop_hook_active`，`SessionStart` 带来源（`startup` / `resume`）。 |
| 会话记录与结局（C1/C2/C3） | 会话记录为 `CODEX_HOME\sessions\<年>\<月>\<日>\rollout-<时间>-<线程ID>.jsonl`。一轮以 `event_msg` 的 `task_started`（带 `turn_id`、`root_turn_id` 与 `turn_attribution`）开始，以 `task_complete` 结束；中间的用户消息、助手正文、MCP 调用与子 Agent 调用各有结构化的 `item_completed`（`UserMessage`、`AgentMessage`、`McpToolCall` 含 `server`、`tool`、参数、状态与结果、`CollabAgentToolCall`），同时另有 `response_item` 形式的模型输入输出。同一次响应中的两个工具调用都完整记下了结果。模型流失败时 `task_complete` 带 `error`（消息与类别），不触发 `Stop`，`codex exec` 以退出码 1 结束。 |
| 续会话（C4、G5） | `codex exec resume <线程ID>` 沿用同一线程 ID，在同一个会话记录文件末尾追加新的一轮（新的 `turn_id`）；`SessionStart` 的来源为 `resume`，结束时触发 `SessionEnd`。无头运行时，标准输入不是终端且没有关闭，`codex exec` 会一直等待输入，需要显式关闭标准输入。 |
| 子 Agent（B5/C5） | 经 `spawn_agent` 派生的子 Agent 有自己的线程 ID 与会话记录文件。子 Agent 内的 `UserPromptSubmit`、`PreToolUse`、`PostToolUse` 中，`session_id` 是根会话，`turn_id` 与 `transcript_path` 是子 Agent 自己的，另带 `agent_id`；`SubagentStop` 的 `transcript_path` 指父会话、`agent_transcript_path` 指子会话。子 Agent 会话记录的 `session_meta` 带 `parent_thread_id`，它的 `task_started` 的 `root_turn_id` 与 `parent_turn_id` 指向父会话中派生它的那一轮。父会话中 `wait` 的 `CollabAgentToolCall` 记下子 Agent 最终交回的消息。 |

**仍未实测**：用户中断（`Interrupt` 与 `TurnAborted` 的先后，需要交互界面或 app-server 的中断请求）、fork 与回退（`codex exec` 没有入口）、上下文压缩、桌面版与 IDE 扩展中的插件和 hook、真实模型与真实 HiveMemory 后端的端到端调用、代码执行模式中的间接调用。

**对接入的影响**（适配分析，不是 owner 决定）：

- 一个目录同时作为两个 harness 的插件在机制上可行，前提是 Codex 清单写明自己的 hook 与 MCP 文件；Codex 侧的 hook 命令不能沿用现有 Claude Code hook 的 bash 写法（`$(command -v …)`），要单独一份。
- 插件更新后要重新安装并重新信任 hook，这是 Codex 的使用成本，需要写进接入手册；未信任时写入缺少 `turn`，会被服务端以 `request.turn_required` 拒绝。
- 不设审批模式时，交互使用中每次写入都要用户确认、无头运行时写入失败；是否由插件把四个记忆工具设为免审批，要在接入计划中定下。
- 连接器写给 hook 的输出须是 ASCII 转义的 JSON（或按 UTF-8 字节写出），读入时按字节解码。
- 写入时的用户消息可以取自 `UserPromptSubmit` 的 `prompt`，也可以从会话记录读取；子 Agent 中的写入可由 hook 读取自身会话记录的 `root_turn_id` 归到根轮，外部会话标识取 hook 输入中的 `session_id`（根会话）。
- 一轮的结局应以 `task_complete` 是否带 `error` 判定，不依赖 `Stop`；续会话在同一文件追加，按文件读取的游标仍然适用；fork 与回退对游标的影响仍待验证。

## 8. 完成标准

- 第 3 节的问题全部作答，或标为"未知"并说明原因；
- 第 4 节的最小实测完成，或注明未执行的部分与原因；
- 第 5 节的五项交付物齐全；
- 所有来源带固定版本或提交与访问日期，frontmatter 记录 `snapshot_at`，并在[调研目录](./README.md)更新验证状态。
