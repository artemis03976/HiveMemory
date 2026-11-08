---
title: pi 插件接入调研需求
status: current
owner: project
scope: pi-plugin-mode-integration-survey
code_paths:
  - integrations/claude-code/
  - integrations/deepseek-harness/
  - src/hivememory/mcp_server/
  - src/hivememory/system/services/import_bus/
related_docs:
  - docs/system/import-bus.md
  - docs/contracts/mcp.md
  - docs/system/configuration.md
  - docs/help/deepseek-harness-plugin.md
  - docs/research/deepseek-harness-plugin-survey.md
  - docs/research/harness-interface-survey-snapshot.md
  - docs/research/codex-plugin-survey.md
last_reviewed: 2026-11-07
---

# pi 插件接入调研需求

## 0. 文档性质

本文是调研需求：规定范围、问题、取证与交付要求。调研结果写入本文第 7 节（与 [DeepSeek Harness 插件接入调研](./deepseek-harness-plugin-survey.md)相同），届时 frontmatter 改为 `baseline` 并记录 `snapshot_at` 与所核对的固定版本。需求有效不表示调研已经完成，也不表示接入路径已经决定。

外部项目变化很快，调研结论只代表完成时的日期。pi 在 1.x 期间发版频繁且 changelog 中仍有不兼容变更，调研时以固定的发行版本与提交为准。

## 1. 目的

owner 的方向（2026-11-04）：v1.0 的 plugin 模式覆盖 Claude Code、DeepSeek Harness、Codex 与 pi，使主流 harness 都能接入 HiveMemory；controller 模式（由 HiveMemory 驱动 harness 执行任务）推迟。2026-11-07 owner 决定开始 Codex 与 pi 的接入，本文与 [Codex 插件接入调研需求](./codex-plugin-survey.md)分别为两者的接入计划提供依据。

plugin 模式的接入包括两件事：harness 的每一轮对话回流到 HiveMemory（Import Bus），以及 harness 中的 agent 经记忆工具检索、读取、写入与更新记忆（MCP）。记忆的写入与访问主要由执行者经记忆工具主动完成，话题结算的被动提取是兜底（owner，2026-10-31）。

**目标环境**（owner，2026-11-07）：Codex 与 pi 的接入先只支持纯 Windows 环境，harness 与 HiveMemory 都在 Windows 上原生运行，不经 WSL。调研只核对 Windows 上的行为，Linux、macOS 与 WSL 暂不核对。

这次调研要回答：

- **接入路径**：按第 2 节的线索，pi 以 TypeScript 扩展为主要扩展方式，形态接近 dsh 的原生插件；是写成 pi 扩展（自带记忆工具与回流），还是用 pi 的 MCP 扩展加会话文件读取器，或两者组合；dsh 插件的 Node 实现有多少可以复用；
- **HiveMemory 一侧**需要新增或改动什么；
- **引导 skill** 在 pi 中以什么方式承载。

### 1.1 HiveMemory 一侧已有的接入面

判断 pi 的接口是否够用时，以下列现有事实为准，权威说明见所链接的文档：

- **对话回流**：`POST /api/v1/import/turns`，每个请求是一轮已经结束的交互：用户消息、按顺序的事件（助手正文、思考、子 Agent 交回的消息、工具调用及其结局）、结局（completed / cancelled / failed）、模型名、外部会话与外部轮次标识，以及之前几轮（只供话题路由）。服务端按推导出的交互标识去重，确认之后连接器才前进（[Import Bus](../system/import-bus.md)第 3、4 节与第 8.5 节）。
- **两种已有实现**：
  - Claude Code 连接器是 Python 标准库写的会话记录读取器加 hook。游标、确认后前进、重发去重、首次见到的旧会话不回填等逻辑在与 harness 无关的公共骨架中，接入新 harness 只需新增读取器（[Import Bus](../system/import-bus.md#8-claude-code-连接器)第 8 节）。
  - dsh 插件是只用 Node 标准库的原生插件：插件自己注册四个记忆工具，经 dsh 的会话事件与会话查询回流，同步、水位、提交结果分类与 dsh 的投影分在不同模块（`integrations/deepseek-harness/lib/`），不共用 Python 骨架（owner，2026-11-01；[Import Bus](../system/import-bus.md#9-deepseek-harness-插件)第 9 节）。
- **记忆访问**：HTTP 的 MCP 端点 `POST /mcp`，无状态的 streamable HTTP、JSON 响应，Host 检查只接受带端口的本机地址；请求头声明 principal、Bearer 凭据、用户与 Agent；工具为 `memory_search`、`memory_read`、`memory_write`、`memory_update`（[MCP 记忆访问](../contracts/mcp.md)第 1–3 节）。
- **写入时确定话题**：写入与更新必须带参数 `turn`（这一轮的来源、外部会话与轮次标识、用户消息与之前几轮），缺少时以 `request.turn_required` 拒绝。`turn` 必须由可信的适配器提供，不能由模型填写；它的外部标识与这一轮回流时的相同，服务端因此推导出同一个交互标识（[MCP 记忆访问](../contracts/mcp.md#6-连接器调用名轮次信息与回声标记)第 6 节）。Claude Code 由 `PreToolUse` hook 改写参数补上它；dsh 由插件自己的工具在执行时补上，交给模型的 schema 中去掉 `turn`。
- **回声标记**：本插件四个记忆工具的调用回流时归一化为 `SEARCH`、`READ`、`WRITE`、`UPDATE` 能力调用，避免已有记忆被再提取一遍。这要求读取器能从会话记录中认出这些调用，识别依据是调用名（同上）。
- **子 Agent**：子 Agent 是父 Agent 临时委托的执行线程，它的过程不进入记忆生成，要保留的内容由它自己写入；父会话的一轮保留它交回的最终输出（`subagent_message` 事件）；子 Agent 中的写入以顶层会话的一轮确定话题（owner，2026-11-02；[Import Bus](../system/import-bus.md#93-轮次的切分与映射)第 9.3 节）。
- **接入登记**：每个 harness 有自己的 principal（`harness:claude-code`、`harness:dsh`）与凭据环境变量，登记在 [`system_principals.yaml`](../../configs/system_principals.yaml)；具体 Agent 与它的四项记忆操作白名单登记在 [`workspace_actors.yaml`](../../configs/workspace_actors.yaml)。两份配置在启动时装载（[配置](../system/configuration.md)）。
- **引导 skill**：使用手册 skill 只维护一份（[`SKILL.md`](../../integrations/claude-code/skills/hivememory/SKILL.md)），内容不区分 harness；Claude Code 随插件加载，dsh 由用户复制到 dsh 的 skill 目录（[接入 DeepSeek Harness](../help/deepseek-harness-plugin.md#43-使用手册-skill)第 4.3 节）。

Claude Code 连接器用到的 hook 是 `Stop`、`StopFailure`、`SessionStart`、`SessionEnd` 与 `PreToolUse`（[Import Bus](../system/import-bus.md#82-hooks)第 8.2 节），第 5 节要求逐项给出 pi 中的对应物。

### 1.2 前两次接入暴露的问题

以下问题在 Claude Code 与 dsh 中实际出现过，调研须核对 pi 中是否存在同类情形：

- **改写参数**：dsh 的工具前置事件不能改写参数，通用 MCP 的写入因此无法补上可信的 `turn`。
- **调用名**：回声标记与 hook 的 matcher 都依赖调用名格式，格式变化时识别静默失效，表现为重复记忆与写入被拒。
- **会话记录**：Claude Code 的记录格式不是公开承诺的接口，并行调用时有时只记下最后一个调用的结果；dsh 的记录经压缩，插件改为经会话查询服务读取逻辑事件，存活的会话先落盘再读。
- **结束判定与重复**：dsh 在读取持久记录时会为未结束的一轮补上收尾，而该会话可能正在另一个进程中运行这一轮，插件因此只提交实际看到结束的轮次；多个进程共用状态目录时同一轮可能提交两次。
- **子 Agent**：dsh 的子会话各自导入时挤占话题池，轨迹被切成粒度不一致的记忆，之后改为只导入顶层会话（owner，2026-11-02）。
- **间接调用**：dsh 的代码执行模式下记忆工具在 `run_code` 内部调用，会话记录只有外层调用，回声标记不生效。
- **一轮多条用户消息**：运行中追加的用户消息与助手事件的相对位置无法保留。
- **运行环境**：已有的两种接入都没有在纯 Windows 环境中验证过。Claude Code 连接器只在 Linux（WSL）上验证过，hook 命令以 `python3` 运行连接器，状态文件的锁在 Windows 上以持续重试取得；dsh 运行在 Windows 桌面端，访问的是 WSL 中的 HiveMemory 后端。

## 2. 调研对象

- **必选**：pi coding agent，官方仓库 [earendil-works/pi](https://github.com/earendil-works/pi)（原 `badlogic/pi-mono`），npm 包 `@earendil-works/pi-coding-agent`，MIT。不是 Inflection Pi、Oh My Pi 或其他移植版。2026-10-20 的[接口调研快照](./harness-interface-survey-snapshot.md)以 `v1.3.0` 为基线，核对的是 controller 模式所需的程序化接口；本次核对 plugin 模式所需的部分，以调研时的最新稳定发行为准。
- **已知线索**（须在调研中按固定版本核实）：
  - 2026-10-20 快照：会话保存为 `~/.pi/agent/sessions/` 下按工作目录分组的 JSONL 会话树；`v1.3.0` 有官方 MCP 扩展，支持 stdio 与 Streamable HTTP、不支持旧的 SSE，配置来自用户级与项目级 `mcp.json`；没有内建的逐工具审批，扩展可以拦截或询问；默认没有子 agent 与计划模式，由扩展提供；一轮的结束以 `agent_settled` 为准，`agent_end` 之后仍可能重试或压缩；
  - 2026-11-07 经 Context7 检索主分支文档：扩展的 `tool_call` 事件在工具执行前触发，可以阻止调用，`event.input` 可以原地改写、改写后不再校验，事件触发前 `ctx.sessionManager` 已更新到当前这条发起调用的助手消息；默认并行执行时，同一条助手消息的多个调用依次预检、并发执行（[`packages/coding-agent/docs/extensions.md`](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/extensions.md)、[`src/core/extensions/types.ts`](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/src/core/extensions/types.ts)）。
- **可选**：社区已有的 pi 记忆类扩展，例如 Honcho 的 pi 扩展（[plastic-labs/pi-honcho](https://github.com/plastic-labs/pi-honcho)，从对话中学习、每轮注入相关上下文），只用于了解其他记忆系统怎样接入 pi，不作为接入候选。

## 3. 调研问题

每个问题注明服务的方面：**路径**（选择接入路径）、**回流**、**访问**（经工具的记忆访问）、**落位**（写入时确定话题）、**回声**（回声标记）、**引导**（skill 的承载）、**共存**（与原生记忆或其他记忆扩展的分工）、**使用**（owner 环境中的实际使用）。

### A. 扩展与分发

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| A1 | 扩展的形式：TypeScript 模块怎样编写、加载与卸载？用户级与项目级扩展的位置，项目扩展是否需要信任确认？能否以本地路径或本地包安装、不发布？ | 路径 |
| A2 | 扩展能否注册供模型调用的工具：名称、说明、参数 schema 与结果怎样声明，模型看到的调用名与会话记录中的名字是什么？与 MCP 扩展提供的工具同名时怎样处理？ | 路径、访问、回声 |
| A3 | 扩展能否访问网络（调用本机 HTTP 服务）、读写 pi 目录之外的文件、启动后台任务？有没有沙箱或权限限制？ | 路径、回流 |
| A4 | 扩展 API 的稳定性：有没有版本号或兼容承诺，扩展怎样声明支持的 pi 版本？1.x 的不兼容变更中涉及扩展 API 的有哪些？ | 路径 |
| A5 | 扩展的生命周期：新建、切换、恢复与 fork 会话时扩展是否重新加载或收到事件？pi 退出时扩展能否完成清理（中止请求、写入状态）？ | 回流 |

### B. 事件

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| B1 | 扩展能订阅哪些事件：会话开始与结束、一轮开始与结束、工具调用前后、中断、出错、压缩？与 Claude Code 连接器的五个 hook 怎样对应？ | 路径、回流 |
| B2 | `tool_call` 改写参数是否对所有工具生效（内置工具、扩展注册的工具、MCP 扩展的工具）？能否在其中可靠地取得当前一轮（会话标识、轮次标识、用户消息、之前几轮）？并行调用、多个扩展同时改写时的顺序与结果如何？ | 落位 |
| B3 | 扩展能否实时取得一轮的完整内容：用户消息、助手正文、思考、工具调用与结果？顺序与完整性如何保证（并行调用、流式中断、重试）？ | 回流 |
| B4 | 一轮的结束能否可靠判定，结局（完成、取消、出错、输出达到上限、超出上下文）怎样表达？`agent_end` 与 `agent_settled` 的区别；进程崩溃时会丢失什么？ | 回流 |
| B5 | 运行中追加的消息（steering、follow-up 一类）怎样进入一轮，事件中能否区分用户输入与扩展注入的消息？ | 回流 |

### C. 会话记录

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| C1 | 会话 JSONL 的格式是否公开承诺（有没有文档或类型定义）？条目的标识与父子关系怎样表示，有没有稳定的会话标识与轮次标识？扩展能否经 API 读取会话，而不是直接解析文件？ | 路径、回流 |
| C2 | 会话是一棵树：在树中回到旧节点继续、分支与 fork 之后，同一会话中有多条路径。"一轮"的身份怎样定义才能在重读时不变？已回流的轮次所在的分支被放弃时，记录有什么变化？ | 回流 |
| C3 | 压缩、分支摘要与扩展写入的自定义条目怎样记录？哪些条目是 harness 注入的上下文（skill 正文、压缩摘要等），需要在回流时排除？ | 回流 |
| C4 | 思考内容是否保存正文？工具结果是否完整保存？图片与附件怎样记录？ | 回流 |
| C5 | 多个 pi 进程能否同时打开同一会话？同一会话被并发写入时记录会怎样？ | 回流 |

### D. MCP

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| D1 | 官方 MCP 扩展是否默认安装与启用？HTTP 传输能否带自定义请求头、从环境变量展开 Bearer 凭据？连接失败时的行为？ | 访问 |
| D2 | MCP 工具的调用名格式是什么（模型看到的、会话记录中的、`tool_call` 事件中的是否相同）？ | 回声、落位 |
| D3 | 与 HiveMemory 的 `/mcp` 是否兼容：无状态的 streamable HTTP、JSON 响应、不提供订阅；客户端请求时的 Host 头能否通过本机地址带端口的检查？ | 访问 |
| D4 | MCP server 的 instructions 是否进入模型上下文，有没有长度限制？ | 引导 |
| D5 | 有没有代码执行（codemode 一类）或批量调用的模式，让工具在另一个工具内部被调用？这种情况下会话记录与 `tool_call` 事件看到的是什么？ | 回声、落位 |

### E. 指令与 skill

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| E1 | 有没有 skill 机制：从哪些位置发现、怎样触发与加载，格式是否与 HiveMemory 的 `SKILL.md`（frontmatter 含 `name` 与 `description`）兼容？扩展或包能否附带 skill？ | 引导 |
| E2 | `AGENTS.md` 一类文件的读取范围与长度限制；扩展能否向系统提示词或每轮上下文追加说明？ | 引导 |
| E3 | 上述说明与已加载的 skill 是否在上下文压缩后保留？ | 引导 |

### F. 原生记忆与其他记忆扩展

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| F1 | pi 本身是否有跨会话的记忆或持久笔记？ | 共存、引导 |
| F2 | 社区已有的记忆扩展（例如 Honcho 的 pi 扩展）怎样接入：用了哪些事件、怎样取得一轮内容、怎样把记忆送回上下文、怎样处理会话树与压缩？有哪些可以借鉴或需要避开的做法？ | 路径、共存 |
| F3 | 上下文压缩何时发生，扩展能否观察到，压缩后原始条目是否仍可读取？ | 回流、使用 |

### G. 运行、模型与许可

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| G1 | 在 Windows 上原生运行（不经 WSL）是否受官方支持，有哪些已知限制（例如内置 shell 工具依赖的 shell）？Node 版本要求与安装方式，配置、扩展、会话记录与 skill 的目录在 Windows 上的位置？ | 使用 |
| G2 | 扩展在 Windows 上访问本机 HTTP 服务（`localhost:<端口>`）、读写 pi 目录之外的状态目录有没有限制？路径分隔符、路径中的空格与长路径是否影响扩展与会话记录？MCP 扩展配置中的环境变量展开在 Windows 上是否照常？ | 路径、访问、回流 |
| G3 | 模型提供方与凭据怎样配置（多提供方、订阅登录）？ | 使用 |
| G4 | print、JSON 与 RPC 等非交互模式下，扩展、事件与会话记录是否照常？能否用于长时间的调研任务？ | 使用、回流 |
| G5 | 扩展与 pi 本身的许可、使用条款中有没有限制？ | 路径 |

## 4. 取证要求

- 每项结论注明来源（官方文档、源码、发布说明）的链接、固定版本或提交与访问日期。文档与源码不一致时两者都记录，以所核对发行版本的源码为准；主分支上尚未发布的能力单独标出。
- 区分"文档声明""源码核对""实际运行"三种验证程度。
- **最小实测**：在 Windows 上原生安装 pi（不经 WSL），以隔离的配置目录运行（不改动 owner 已有的 pi 配置），写一个扩展：记录会话与轮次事件、注册一个工具、在 `tool_call` 中改写一个参数，并经 MCP 扩展连接一个本机 HTTP 的 MCP server（以 Windows 本机上的模拟 server 代替 HiveMemory），回答 A2、B1、B2、B3、B4、C1、C2、D1、D3、G1、G2。需要调用模型的实测使用 owner 确认的凭据；没有获得授权时只做不调用模型的机制核对并注明（dsh 调研以脚本化的模型适配器完成了同类探针）。凭据、日志与产物不提交到仓库。
- 无法确定的项标为"未知"并说明原因，不作推测。

## 5. 交付物

1. **能力清单**：第 3 节的每个问题，取值为"支持 / 部分支持 / 不支持 / 未知"，附来源与验证程度。
2. **hook 对应表**：Claude Code 连接器用到的五个 hook 与 `PreToolUse` 改写参数的能力，在 pi 中的对应物或缺口。
3. **接入路径比较**：至少比较三条路径——"pi 原生扩展"（与 dsh 相同：扩展注册四个记忆工具并经事件回流）、"MCP 扩展加会话文件读取器"（与 Claude Code 相同：经官方 MCP 扩展访问 `/mcp`，在 `tool_call` 中补上 `turn`，回流由复用 Python 公共骨架的读取器完成）与两者的组合。每条写明复用 HiveMemory 的哪些部分（包括 dsh 插件中与 dsh 无关的同步、水位与提交模块能否在 pi 扩展中复用）、需要新增什么、写入时确定话题与回声标记怎样满足、会话树与子 Agent 怎样处理、在 Windows 上的运行时依赖（pi 本身需要 Node，复用 Python 公共骨架还需要 Windows 上可用的 Python）与主要风险。只列影响，不作选择。
4. **引导 skill 的承载方式**：pi 中可以承载使用说明的机制及其限制。

## 6. 不在范围

- 编写扩展、连接器或任何生产代码；
- 决定接入路径，或修改 HiveMemory 的接入契约；
- 接入登记的运行时登记与配置热更新（[ROADMAP](../ROADMAP.md) 的配置热更新一项），只写明 pi 接入需要登记什么；
- 引导 skill 的内容；
- HiveMemory 后端在 Windows 上的原生运行与打包，随桌面客户端的工作进行；实测以模拟的 MCP server 代替 HiveMemory；
- Linux、macOS 与 WSL 上的接入行为；
- pi 的模型质量、回复效果或性能对比；
- controller 模式所需的程序化接口（SDK、RPC 的驱动方式），见 [2026-10-20 快照](./harness-interface-survey-snapshot.md)，只在发现与 plugin 模式明显相关时顺带记录。

## 7. 调研结果

尚未开始。调研完成后按第 5 节的交付物写入本节。

## 8. 完成标准

- 第 3 节的问题全部作答，或标为"未知"并说明原因；
- 第 4 节的最小实测完成，或注明未执行的部分与原因；
- 第 5 节的四项交付物齐全；
- 所有来源带固定版本或提交与访问日期，frontmatter 记录 `snapshot_at`，并在[调研目录](./README.md)更新验证状态。
