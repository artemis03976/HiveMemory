---
title: Codex 插件接入调研需求
status: current
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
last_reviewed: 2026-11-07
---

# Codex 插件接入调研需求

## 0. 文档性质

本文是调研需求：规定范围、问题、取证与交付要求。调研结果写入本文第 7 节（与 [DeepSeek Harness 插件接入调研](./deepseek-harness-plugin-survey.md)相同），届时 frontmatter 改为 `baseline` 并记录 `snapshot_at` 与所核对的固定版本。需求有效不表示调研已经完成，也不表示接入路径已经决定。

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
- **引导 skill**：使用手册 skill 只维护一份（[`SKILL.md`](../../integrations/claude-code/skills/hivememory/SKILL.md)），内容不区分 harness；Claude Code 随插件加载，dsh 由用户复制到 dsh 的 skill 目录（[接入 DeepSeek Harness](../help/deepseek-harness-plugin.md#43-使用手册-skill)第 4.3 节）。

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

尚未开始。调研完成后按第 5 节的交付物写入本节。

## 8. 完成标准

- 第 3 节的问题全部作答，或标为"未知"并说明原因；
- 第 4 节的最小实测完成，或注明未执行的部分与原因；
- 第 5 节的五项交付物齐全；
- 所有来源带固定版本或提交与访问日期，frontmatter 记录 `snapshot_at`，并在[调研目录](./README.md)更新验证状态。
