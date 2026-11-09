---
title: pi 插件接入调研需求与报告
status: baseline
owner: project
scope: pi-plugin-mode-integration-survey
snapshot_at: 2026-11-07
source_version: v1.5.0
source_commit: acbbe32e165dcb0bf450b454ad1b3547b690db8f
upstream_commit: f9a26365568127461c47a07a37dcb42c140a3ac5
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

# pi 插件接入调研需求与报告

## 0. 文档性质

第 1–6 节保留原调研需求，第 7 节是 2026-11-07 对 pi `v1.5.0` 的证据快照，第 8 节保留完成标准。`baseline` 冻结资料核查结果与未知项；Windows 原生实测未执行，接入路径未决定，HiveMemory 尚未因此获得 pi 接入能力。需求中的旧版本线索与第 7 节不一致时，以第 7 节所列固定源码为准。

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
- **引导 skill**：使用手册 skill 只维护一份（[`SKILL.md`](../../integrations/hivememory/skills/hivememory/SKILL.md)），内容不区分 harness；Claude Code 随插件加载，dsh 由用户复制到 dsh 的 skill 目录（[接入 DeepSeek Harness](../help/deepseek-harness-plugin.md#43-使用手册-skill)第 4.3 节）。

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

### 7.1 结论与验证边界

pi `v1.5.0` 同时具备原生工具注册、工具参数改写、会话树读取与最终空闲事件，能够承载三条候选路径。**不能直接复制 dsh 的递增轮号水位，也不能把 JSONL 文件按行追加等同于线性对话。** 对接需要先固定“用户一轮”的身份，并把写入时的 `turn` 与回流时的标识绑定起来。

本版已确认、直接影响路径比较的两点是：

- MCP 已作为随包加载的内建扩展提供，无须另装社区 MCP adapter；配置仍是 `mcp.json`。它默认将工具设为 `codemode` exposure，连接后可自动启用 codemode。[P-MCP]
- codemode 内部的调用也经过 `tool_call`、`tool_result` 与执行事件，因此可信参数可以注入；但内部调用没有独立的会话消息，持久化的 `nestedCalls` 只保存有界的参数与状态，不保存结果正文。**能拦截不等于能从会话文件完整还原，补记内部调用也不能自动消除外层输出中的记忆回声。**[P-EXT]、[P-SESSION]

取证日期统一为 **2026-11-07**；以下 `[P-*]` 引用除 Honcho 外均指向发行提交 `f9a26365568127461c47a07a37dcb42c140a3ac5`。GitHub `releases/latest` 返回非预发行 `v1.5.0`，发布时间为 2026-11-02；再通过该 tag 的 commits API 解析提交并下载源码核对。[P-RELEASE]

验证程度：**文档声明**指固定发行中的官方文档；**源码核对**指阅读该提交实现及已有测试，未运行上游测试；**实际运行**仅完成发行元数据、源码下载与本地文本检查，未运行 pi、模拟模型或 MCP 探针。执行环境为 Linux，没有可用的 Windows 执行端，因此未用 Linux 或 WSL 的结果替代目标环境验证；未读取或使用模型凭据。各表“支持”表示接口存在，不表示 Windows 接入已经验收。

### 7.2 能力清单

#### A. 扩展与分发

| 编号 | 判定 | 结论、边界与来源 | 验证程度 |
|:---|:---|:---|:---|
| A1 | 支持 | TS/JS 模块默认导出接收 `ExtensionAPI` 的工厂，支持异步工厂。自动发现用户 `~/.pi/agent/extensions/` 与项目 `.pi/extensions/`，项目资源须信任；可用 `pi -e <路径>` 临时加载，或 `pi install ./local-package` 安装本地包而不发布。`pi remove`、配置资源过滤和 `/reload` 管理移除/重载；工具本身无 unregister，撤下工具可设 hidden。[P-EXT]、[P-PKG]、[P-SEC] | 文档声明、源码核对 |
| A2 | 支持 | `registerTool` 声明 `name`、description、参数 JSON Schema/TypeBox、execute；结果为 content、details 等。直接调用的模型名、assistant `toolCall.name`、`toolResult.toolName` 与事件名一致。扩展间同名工具按加载顺序第一份注册生效；同一扩展重注册可更新，内置工具可被覆盖。MCP 命名另有清洗/哈希，不能把同名视作安全替换协议，应避免同时注册两套记忆工具。[P-EXT]、[P-RUNNER]、[P-MCP-SRC] | 文档声明、源码核对 |
| A3 | 支持 | 扩展在 pi 进程内以启动用户权限运行，可访问本机 HTTP、任意获操作系统许可的文件、进程与定时器。没有按扩展划分的内置沙箱。长连接、后台队列等应在 `session_start` 或使用时启动，并在 `session_shutdown` 清理；工厂可能在不创建会话的命令中运行。[P-EXT]、[P-SEC] | 文档声明、源码核对 |
| A4 | 部分支持 | 包与 API 随 pi 版本发布，未找到独立稳定 ABI 或“不破坏 1.x 扩展”的承诺。官方建议 host 包列为 `peerDependencies: "*"`、不打包宿主依赖，这不是兼容校验。1.2.0 改变 `before_agent_start` 的触发范围和 `messageType` 联合类型；1.5.0 改用 pi-schema 校验工具参数，TypeBox 非 JSON Schema 能力不再适用。接入需声明已验版本并在升级时复核。[P-PKG]、[P-CHANGE] | 文档声明、源码核对 |
| A5 | 支持 | `session_start.reason` 有 startup/reload/new/resume/fork；会话替换先 shutdown、再装载新扩展 runtime、再 start。`/tree` 在同一 runtime 内触发 tree 事件。shutdown 可 await 清理，但强杀/崩溃不保证回调；替换后的旧 ctx 不可继续使用。[P-EXT]、[P-RUNNER] | 文档声明、源码核对 |

#### B. 事件

| 编号 | 判定 | 结论、边界与来源 | 验证程度 |
|:---|:---|:---|:---|
| B1 | 支持 | 会话 start/shutdown、input、before_agent_start、agent_start/end/before_settle/settled、turn_start/end、message_*、tool_call/result/execution_*、session_before_compact/compact/compact_failed 均可订阅。没有与 Claude Code 同名的 StopFailure；错误从消息终态与 settled 组合判断，中断用 `agent_settled.aborted`。对应表见 7.3。[P-EXT]、[P-SESSION] | 文档声明、源码核对 |
| B2 | 支持 | 所有进入工具管线的内置、扩展、MCP 工具及 `ctx.executeTool` 嵌套调用都可原地修改 `event.input`；无效参数在此前校验失败时不会进入此事件，改写后不再校验。handler 按扩展加载顺序、注册顺序执行，后者看到前者修改；并行兄弟调用依次预检，再并发执行。此时 sessionManager 已包含当前发起调用的助手消息；能读 session ID、当前分支用户条目，但没有持久的用户轮次 ID，不能用 turnIndex 直接充当它。[P-EXT]、[P-SESSION]、[P-RUNNER] | 文档声明、源码核对 |
| B3 | 部分支持 | `message_end` 与 sessionManager 可取得最终正文、可见思考、工具调用和结果；流式 message_update 可观察中间过程。直接并行调用的最终 toolResult 按助手发起顺序入消息流，execution_end 则按完成顺序。嵌套调用结果只能从实时事件捕获；强杀前未 finalized 的流式内容不保证落盘，工具自身截断也无法还原。[P-EXT]、[P-SESSION]、[P-MSG] | 文档声明、源码核对 |
| B4 | 部分支持 | `turn_end` 是一次模型请求及工具批次结束；`agent_end` 后还可能重试、压缩恢复或消费 follow-up；`agent_before_settle` 仍能请求继续；`agent_settled` 才表示不会自动继续。它只给 aborted，不给完整成功/失败枚举，须结合最终助手 stopReason（stop/length/toolUse/error/aborted 等）。length 可触发恢复，不能在首次 length 时提前导入失败。settled 没有内置持久条目，崩溃后的完成证明需适配器补足。[P-EXT]、[P-SESSION]、[P-COMPACT] | 文档声明、源码核对 |
| B5 | 部分支持 | input 有 interactive/rpc/extension 来源及 streamingBehavior，可在 skill/template 展开前取原文；用户消息、steering、follow-up 会进入消息流，custom message 则可另辨认。原始来源没有作为统一字段保留在所有 user message 中；扩展也可发送 user message，纯文件重读不足以还原来源。一个 settled 区间可含多次追加输入，HiveMemory 的单 `user_message` 无法保留它们穿插助手事件的位置。[P-EXT]、[P-MSG]、[P-SESSION] | 文档声明、源码核对 |

#### C. 会话记录

| 编号 | 判定 | 结论、边界与来源 | 验证程度 |
|:---|:---|:---|:---|
| C1 | 支持 | 官方文档和类型公开 JSONL v3：header 有 UUID 会话 ID；条目有稳定 id、parentId、ISO timestamp。默认保存于 agent-dir 的 sessions 下，按 cwd 编码分组。ctx.sessionManager 提供 getEntries/getBranch/getTree/getEntry/getLeafId 等只读 API；不存在与 HiveMemory 一轮等价的持久 turn ID。版本迁移会改写文件，不是永久不变格式。[P-FORMAT]、[P-SM]、[P-EXT] | 文档声明、源码核对 |
| C2 | 部分支持 | `/tree` 在原文件形成分支，放弃的条目仍存在；fork/clone 创建新 session ID，并继承路径内容与 `parentSession`。用户条目 ID 可作为轮次锚点，但分支重用同一用户节点产生另一份回答时仍需区分执行实例；单用 `(sessionId,userEntryId)` 会撞。已导入分支不会由 HiveMemory 自动撤销；候选身份分析见 7.4。[P-FORMAT]、[P-SM] | 文档声明、源码核对、适配分析 |
| C3 | 支持 | compaction、branch_summary、context_edit、custom/custom_message 等均有独立类型；压缩与 context edit 改变模型投影，保留原条目。getEntries/getBranch 与 buildSessionProjection 的语义不同。回流应排除 system、摘要和自身注入的上下文，不能把展开后的 skill 正文当作用户新知识；需保留 raw input 与其对应条目的关联。[P-FORMAT]、[P-EXT]、[P-RESOURCE] | 文档声明、源码核对 |
| C4 | 部分支持 | thinking 正文在提供方实际返回时保存，签名/加密数据不等于可读正文。图片是 base64+mimeType，附件经各输入/工具投影；HiveMemory 当前导入不是附件导入。直接工具保存最终返回值，但 MCP 给模型的文本超过 20 KB 会截断并引用临时文件；嵌套结果不入 transcript。因此不保证“原始完整结果”。[P-MSG]、[P-MCP]、[P-SM] | 文档声明、源码核对 |
| C5 | 部分支持 | 可以按路径打开已有会话；所核对 SessionManager 写入路径用各进程自己的内存树与 appendFileSync，未见会话 writer 互斥/合并协议。两个进程同写的排序、父链与 Windows 文件共享结果没有保障；不能据此承诺安全并发。读文件不等于拥有该会话，不能把另一进程仍运行的尾部推断为已结束。[P-SM] | 源码核对；Windows 并发结果未知 |

#### D. MCP

| 编号 | 判定 | 结论、边界与来源 | 验证程度 |
|:---|:---|:---|:---|
| D1 | 支持 | 内建 MCP 默认随资源加载，可被 `--no-mcp` 禁用；注册 `/mcp` 的其他扩展可能替换它。支持 stdio、Streamable HTTP，不支持旧 SSE。headers/env 支持 `${NAME}` 与完整 `!command` 值；用户/项目信任配置的覆盖规则见官方文档。后台连接失败显示状态，工具连接后加入；HTTP 网络/临时状态连接请求重试两次，掉线后下次调用重连，工具调用不自动重试，以免重复副作用。[P-MCP]、[P-MCP-SRC]、[P-RESOURCE] | 文档声明、源码核对 |
| D2 | 支持 | 标准形式 `mcp__<server>__<tool>`，非字母数字下划线字符清洗；清洗碰撞加哈希。直接调用的模型、会话和 tool_call 名称一致；codemode 的 JS 标识符仍要经过规范化，实际事件名称是识别依据。无碰撞的 server `hivememory` 对应 `mcp__hivememory__memory_write` 等。[P-MCP]、[P-MCP-SRC] | 文档声明、源码核对 |
| D3 | 部分支持 | HTTP transport 接受 application/json 与 SSE 响应，不要求必须取得 Mcp-Session-Id；GET 返回 405 时按不提供通知流处理，临时网络/服务错误则另有重连。无资源/订阅不妨碍 tools/list、tools/call。配置 `http://127.0.0.1:<端口>/mcp` 与身份头在机制上匹配 HiveMemory。实际 Host、代理/IPv6、协议协商与 Windows 回环连接尚未抓包验证，故不宣称端到端兼容已通过。[P-HTTP]、[P-MCP] | 源码核对、适配分析；实测未执行 |
| D4 | 部分支持 | instructions 保存于工具 namespace；codemode 可用 describeNamespace 读取全文。系统提示的 mcp_servers 段对间接 exposure 只取配置 description 或 instructions 第一行，工具描述不自动附加全文。未发现专属 instructions 长度上限，仍受上下文预算限制；不能仅靠初始化 instructions 保证每轮知道 HiveMemory 用法。[P-MCP]、[P-MCP-SRC] | 文档声明、源码核对 |
| D5 | 部分支持 | codemode 与扩展 `ctx.executeTool` 支持内部调用，内层仍有 tool_call/result/execution_*、parentToolCallId 和 `<parent>/<n>` ID，可信 turn 注入可以生效。持久 nestedCalls 最多 256 条，参数单条 8 KiB、整组 32 KiB，丢失时 complete=false，永不保存内层结果。外层可混合、加工或打印记忆，单凭内层能力标记无法防止外层再次提取。[P-EXT]、[P-SESSION] | 文档声明、源码核对 |

#### E. 指令与 skill

| 编号 | 判定 | 结论、边界与来源 | 验证程度 |
|:---|:---|:---|:---|
| E1 | 支持 | 支持 Agent Skills 格式，name/description frontmatter 与现有手册兼容。发现用户/项目 `.pi` skills、`~/.agents/skills`、cwd 到仓库根的 `.agents/skills`，也可由包 `pi.skills` 附带。默认只暴露名称、说明、路径，模型按需读取；`/skill:hivememory` 强制展开。无需为 pi 改写手册正文。[P-SKILL]、[P-PKG] | 文档声明、源码核对 |
| E2 | 支持 | agent-dir 与 cwd 向上各层发现 AGENTS.override.md/AGENTS.md/CLAUDE.md 等，每目录取优先匹配；context files 与项目可执行资源的信任规则不同。加载实现读取完整文本，未见 Codex 式 32 KiB 总额配置。SYSTEM.md/APPEND_SYSTEM.md、before_agent_start 的 systemPromptOptions、context 等可追加说明。[P-CONFIG]、[P-RESOURCE]、[P-EXT] | 文档声明、源码核对 |
| E3 | 部分支持 | 当前系统说明与 skill 目录可重新组成 system message，compaction 条目保存系统快照；已读/展开的 skill 正文属于普通会话内容，可能被摘要替代，没有逐字保留保证。关键使用规则应由可重建的简短说明承载，完整操作步骤仍由单一 skill 按需读取。[P-COMPACT]、[P-FORMAT]、[P-SKILL] | 文档声明、源码核对、适配分析 |

#### F. 原生记忆与共存

| 编号 | 判定 | 结论、边界与来源 | 验证程度 |
|:---|:---|:---|:---|
| F1 | 不支持 | 在本次核对的默认资源与配置中，没有自动从跨会话历史学习、检索并注入的原生长期记忆子系统。已有持久会话、手写上下文文件、skill 与压缩摘要，均不等于 HiveMemory 的跨会话记忆。此结论限 pi 本体，安装的扩展可另行提供记忆。[P-CONFIG]、[P-RESOURCE]、[P-COMPACT] | 文档声明、源码核对 |
| F2 | 支持 | Honcho 1.0.0 的固定源码在 input 记原始输入、session_start 初始化、before_agent_start 加系统段和自定义上下文、agent_end 提取消息入上传队列，并在 before_compact/shutdown 等待队列。它不是整轮工具轨迹归档；默认按目录映射远端会话，也不是 pi 树节点身份。其 agent_end 边界与 HiveMemory 所需 settled 不同，且该入口未区分本版 before_agent_start 的 custom 分支，不能原样借用。[P-HONCHO]、[P-HONCHO-SRC] | 第三方文档声明、源码核对；运行兼容未知 |
| F3 | 支持 | 手动 compact、上下文阈值及溢出恢复均有入口；默认 reserveTokens=16384、keepRecentTokens=20000，可按模型覆盖。扩展可观察成功、失败与中断；压缩追加 summary/firstKeptEntryId，原始条目仍可从 sessionManager 读取。不要以仅含摘要的模型投影替代归档原文。[P-COMPACT]、[P-SM] | 文档声明、源码核对 |

#### G. Windows、模型与许可

| 编号 | 判定 | 结论、边界与来源 | 验证程度 |
|:---|:---|:---|:---|
| G1 | 支持 | 官方支持原生 Windows；npm 安装需 Node >=22.19.0，固定本版可用 `npm install -g --ignore-scripts @earendil-works/pi-coding-agent@1.5.0`，另有官方 PowerShell installer。bash/`!` 默认找 Git for Windows，可选模型工具 powershell；Windows Terminal 是官方主要验证终端，旧 conhost 有崩溃/输入问题。默认目录为 `%USERPROFILE%\.pi\agent`，其下 settings、extensions、sessions、skills；隔离变量是 PI_CODING_AGENT_DIR。[P-WIN]、[P-QUICK]、[P-CONFIG] | 文档声明；Windows 安装未执行 |
| G2 | 部分支持 | 扩展网络与文件访问使用 Node/OS 权限，没有限定只能写 pi 目录；`${NAME}` 由配置代码解析，不依赖 Bash。Windows 路径在 JSON 中需转义反斜线，Node path API 处理分隔符。带空格/中文目录、超长 cwd 编码、杀进程与并发 rename 的完整行为未实测，不能给“Windows 全部可用”结论。[P-WIN]、[P-SEC]、[P-MCP-CONFIG]、[P-SM] | 文档声明、源码核对；长路径等未知 |
| G3 | 支持 | `/login` 支持可用提供方的订阅/OAuth 或 API key，凭据保存在 agent-dir/auth.json；也支持环境变量、models.json 的兼容端点与模型覆盖。与 HiveMemory 的 Bearer 凭据是不同认证链，扩展接入不要求特定模型账户。实际账户授权、订阅可用性未核验。[P-QUICK]、[P-CONFIG] | 文档声明 |
| G4 | 支持 | print、JSON、RPC 共用 AgentSession 和扩展 runtime，print 退出在 finally 中 dispose；JSON 可输出事件，RPC 有 UI request 协议。非交互入口不能显示项目 trust 提示，未授权项目资源默认可能不加载；`--no-session` 不持久化，UI 依赖扩展须按模式处理。可运行长任务，但生命周期事件并不提供进程保活或崩溃后自动续传保证。[P-PRINT]、[P-RPC]、[P-SEC] | 文档声明、源码核对 |
| G5 | 部分支持 | pi 本体 MIT，允许按其条件使用、修改与分发，需保留版权/许可；扩展和依赖各有许可证。模型服务与订阅条款不是 MIT 所授予，本次未作服务条款的法律审查，也未发现本地 HiveMemory 扩展必须发布到 gallery 的要求。[P-LICENSE]、[P-PKG] | 许可文本、文档声明；服务条款限制未知 |

### 7.3 Claude Code hook 对应

| Claude Code 入口 | pi 对应 | 接入含义 |
|:---|:---|:---|
| SessionStart | session_start | 覆盖启动、恢复、新建、fork、reload；重建当前树路径的适配状态。 |
| Stop | agent_settled | 用作已结束提示；turn_end/agent_end 均不足以表示用户一轮最终结束。 |
| StopFailure | agent_settled + 最终助手 stopReason/errorMessage | 无单独失败 hook；中途 error 可能恢复，需等 settled。 |
| SessionEnd | session_shutdown | 正常退出/替换时清理和落状态；不能承诺强杀也执行。 |
| PreToolUse | tool_call | 原地覆盖 input.turn；匹配本工具实际全名，改写后不再校验。 |

事件顺序、参数和模式限制的依据为 [扩展文档][P-EXT] 与 [AgentSession 实现][P-SESSION]。网络回流不宜阻塞每条流式通知；但异步提交前必须冻结该轮身份和内容，不能让后续 session 切换改变在途请求的归属。

### 7.4 三条路径的影响比较

下表是适配分析，不是 owner 决定或实施计划。三条路径均可复用现有 `/mcp` 和 Import Bus 契约；没有证据要求先改服务端契约。

| 路径 | 可复用与新增 | turn、回声、会话树与子 Agent | Windows 依赖与主要风险 |
|:---|:---|:---|:---|
| 原生扩展，自有四个工具与事件回流 | 可复用 dsh 的 HTTP 请求/结果分类思路、手册和后端；新增 pi registerTool 接线、树投影、结束记录、补发状态。不能直接复用 dsh 的 session query/turns。 | execute 中移除模型 turn 并注入可信上下文，可从 schema 隐藏 turn；直接工具按自有名称归一化。可用 model-only exposure 阻止经 codemode 调用，代价是工具只能由模型直接调用；若开放嵌套，则需捕获实时事件并解决外层回声。树与子 Agent 规则见下文。 | pi 已需 Node；不额外依赖 Python。维护自己的 MCP schema/结果适配与队列；工厂、reload、shutdown 不能泄漏请求。 |
| 内建 MCP + tool_call 扩展 + Python 文件读取器 | 复用 Python HTTP 客户端、确认后前进/拒绝分类和手册；新增 MCP 配置、pi 事件扩展与 JSONL 树读取器。现有 Python 按字节线性推进的同步策略不能直接沿用。 | tool_call 覆盖 turn，MCP 的可选 turn schema 不必修改；名称按 mcp__hivememory__* 识别。文件中无 settled 与完整嵌套结果，仍需扩展落结束证据；只设 direct 不会禁止 codemode 调用。 | Node + Windows Python；两套运行时和状态交接。树分叉/重写、临时文件丢失、嵌套结果缺失、Python 锁语义均需验收。 |
| 内建 MCP + pi 扩展处理事件/回流 | 内建 MCP 负责连接与工具 schema；扩展负责可信 turn、最终结束记录和提交，复用 dsh HTTP 提交分类的可分离部分。无需 Python 读取进程。 | 可经 registerMcpServer 打包 server 声明；配置文件同名条目优先，必须防止被无意替换。通过 sessionManager 读树，实时补内层事件；仍需处理改写扩展顺序、外层回声、完成记录和补发。 | Node 即可；比自有工具少一套 MCP 客户端，但依赖内建 MCP 的 naming/exposure/loader 行为和其他扩展共存。 |

**已有代码的实际复用边界**（HiveMemory 基线 `acbbe32e165dcb0bf450b454ad1b3547b690db8f`）：

- [`client.js`](../../integrations/deepseek-harness/lib/client.js) 的请求、超时和 confirmed/retryable/misconfigured/server_error/rejected 分类不依赖 dsh 会话 API；但 principal 从 dsh config 导入，clientInfo 也写着 dsh，不能不改就直接引用。
- [`sync.js`](../../integrations/deepseek-harness/lib/sync.js) 直接导入插件中 dsh 专用投影 [`turns.js`](../../integrations/deepseek-harness/lib/turns.js) 的 sessionTurns/importBody，依赖 flush/read/isLive、递增 number 与 endedTurn；属于需要改写的同步编排，而非已有通用库。
- [`state.js`](../../integrations/deepseek-harness/lib/state.js) 的原子写和首次接入时刻逻辑可借鉴，但 lastTurn/endedTurn 是数值水位，且只保证同进程同步读改写。pi 的树分支与多进程不能沿用该保证。
- Python 的 [`TranscriptReader`](../../integrations/hivememory/connector/core/transcript.py) 和 [`sync.py`](../../integrations/hivememory/connector/core/sync.py) 假设文件线性追加、以用户消息切轮；pi 的 fork/迁移/分支与 settled 缺失让“只新增 reader”不足。此处只指出影响，不提前抽取跨 harness 新抽象。

**一轮身份的候选约束**：session UUID 与用户 entry ID 是可用锚点；turnIndex 是模型循环计数，不能作持久外部轮号。若一轮包含运行中追加输入，则起始身份必须在第一次写入之前确定并在整轮保持不变。回到同一用户节点重新执行会产生不同回答，因此还需要持久的执行实例边界；可由扩展在输入被接受时记录适配器轮次标识，结束时记录实际结束的叶节点与 outcome。该标识同时用于工具 turn 和 Import Bus。纯文件路径缺少这些信息，不能把“下一个用户条目出现”当作所有分支都已结束。

**会话树与 fork**：恢复时从当前分支重建上下文，补发以适配器实际记录的已结束执行实例为准。fork 的继承前缀不应作为新对话重新回填；原分支已导入的交互不能自动撤销。分支策略需要接入计划选定，不能依靠服务端去重弥补错误的外部标识。

**可信注入的边界**：已运行的 tool_call handler 抛错会阻止该工具，适配器可在无法解析当前轮时拒绝；这比 Codex 的 hook 出错放行更容易约束。但扩展未加载/被禁用、或后续扩展覆盖 turn 时，仍不能从 MCP 参数本身证明来源。自有工具可以把注入放在 execute 内并隐藏 turn；直接 MCP 两条路径需把扩展装载和改写顺序列为运行约束，不能仅靠提示词保证。[P-EXT]、[P-RUNNER]

**子 Agent**：pi 本体没有统一子 Agent 生命周期与父轮次契约；示例/社区扩展各自创建进程或 SDK 会话。`parentSession` 同时用于 fork/clone，不能据此把所有派生会话当作子 Agent 丢弃。三条路径都需与具体子 Agent 扩展建立父执行实例绑定，才能让子工具写入落在顶层轮次、仅导入父级最终交回消息。没有绑定时应明确不支持该子 Agent 场景，不能猜父会话最近一轮。[P-README]、[P-FORMAT]

**回声**：直接调用映射 SEARCH/READ/WRITE/UPDATE 时移除 turn，READ/UPDATE 提取 aliases/alias；无效参数不伪造成能力调用。嵌套调用即便补记成功，外层 codemode 输出仍可包含取回的记忆与其他正文；既不能把整段外层结果统统删掉，也不能声称四个内层 verb 已排除全部回声。限制间接调用和补足输出归因是不同选项，需由后续路径选择决定。

### 7.5 引导与共存

现有 [`hivememory/SKILL.md`](../../integrations/hivememory/skills/hivememory/SKILL.md) 的 frontmatter 可直接使用；由 pi 包的 `pi.skills` 分发，或复制到用户/项目 skill 目录。只维护同一份正文，名称仍为 `hivememory`。skill 是按需加载，不能取代固定事件接线，也不能承担可信 turn 的生成。[P-SKILL]、[P-PKG]

需要常驻的简短工具说明可以通过原生工具 promptGuidelines 或 before_agent_start 的系统段承载；MCP instructions 不会自动完整常驻。压缩后重新提供短说明、按需重新读 skill，比假设已加载正文永久保留更符合接口行为。[P-EXT]、[P-MCP]、[P-COMPACT]

与 Honcho 等扩展共存时，不应把其 custom_message、系统注入、检索结果再次当作用户新事实导入；四个 HiveMemory 工具的回声标记不会自动识别别的记忆扩展。可借鉴 Honcho 的“原始输入与展开输入分开”“上传队列在压缩/关闭时收尾”，不能借用其 agent_end 切轮和按目录合并会话规则。这是共存限制，不是要求安装或关闭 Honcho。[P-HONCHO-SRC]

### 7.6 HiveMemory 侧影响与 Windows 验证缺口

后续若实施，需要为 pi 增加独立 principal/凭据登记（例如 `harness:pi`）、具体 Agent 与四项操作白名单，source 使用稳定的 pi 来源名；导入与 MCP 使用同一身份和外部轮次标识。登记文件仍按现有启动装载机制处理，不引入热更新。新增内容限所选路径的扩展、树投影、确认状态及 Windows 安装配置；本次没有修改配置、skill、后端或生产集成。

| 要验证的机制 | 本次状态 | Windows 最小验证应取得的证据 |
|:---|:---|:---|
| 隔离安装、路径与信任（A1/G1/G2） | 未执行：无 Windows 执行端 | PI_CODING_AGENT_DIR 指向独立临时目录，记录 pi/Node 版本；含空格/中文路径、项目拒绝信任与允许信任两种加载结果。 |
| 工具注册、参数改写与并行（A2/B1/B2） | 仅源码核对 | 无真实凭据的脚本化 provider 发起直接和 codemode 记忆工具调用；模拟 server 收到扩展覆盖的 turn，模型提供的伪造值未被采用。 |
| 完成、取消、错误与追加输入（B3/B4/B5） | 仅源码核对 | 记录 agent_end、恢复、settled 次序，完成/中断/失败只结算一次；流式强杀不得凭尾部文本推断 completed。 |
| 会话树、fork、压缩、重放（C1/C2/C5） | 仅源码核对 | 同轮写入和回流 ID 一致；新分支不撞旧轮，继承前缀不重导；两个进程不误判对方活动轮次。 |
| HTTP MCP（D1/D3） | 未执行 | 模拟无状态 JSON server 记录 initialize、tools/list、tools/call、通知流行为及 Host/身份头；拒连、超时、服务恢复不会造成重复写入。 |
| 间接调用回声（D5） | 已确认持久记录缺口，未实测 | 区分内层调用与外层结果，验证所选策略不会重复提取记忆或丢弃外层非记忆输出。 |

可使用脚本化模型避免真实模型费用；真实模型验证仍需 owner 确认凭据。上述未实测项是后续接入验收的前提，尤其是树身份和间接调用策略；本报告没有替 owner 选择它们，也没有创建承诺实施的 Plan。

### 7.7 固定来源

下列引用访问日期均为 2026-11-07。官方源码和文档以 `v1.5.0` 提交固定；Honcho 为其单独的源码快照，不作为 pi 核心能力证据。所读上游测试仅作为实现意图的补充，未报告其通过。

[P-RELEASE]: https://github.com/earendil-works/pi/releases/tag/v1.5.0
[P-EXT]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/docs/extensions.md
[P-PKG]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/docs/packages.md
[P-SEC]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/docs/security.md
[P-CHANGE]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/CHANGELOG.md
[P-RUNNER]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/src/core/extensions/runner.ts
[P-SESSION]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/src/core/agent-session.ts
[P-FORMAT]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/docs/session-format.md
[P-SM]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/src/core/session-manager.ts
[P-MSG]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/docs/message-types.md
[P-MCP]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/docs/mcp.md
[P-MCP-SRC]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/src/extensions/mcp/index.ts
[P-MCP-CONFIG]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/src/extensions/mcp/config.ts
[P-HTTP]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/mcp/src/transports/streamable-http.ts
[P-SKILL]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/docs/skills.md
[P-CONFIG]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/docs/configuration.md
[P-RESOURCE]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/src/core/resource-loader.ts
[P-COMPACT]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/docs/compaction.md
[P-WIN]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/docs/windows.md
[P-QUICK]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/docs/quickstart.md
[P-PRINT]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/src/modes/print-mode.ts
[P-RPC]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/src/modes/rpc/rpc-mode.ts
[P-README]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/packages/coding-agent/README.md
[P-LICENSE]: https://github.com/earendil-works/pi/blob/f9a26365568127461c47a07a37dcb42c140a3ac5/LICENSE
[P-HONCHO]: https://github.com/plastic-labs/pi-honcho/blob/2644a08ddddc4fc47e71698763bc27e17c47a3fb/README.md
[P-HONCHO-SRC]: https://github.com/plastic-labs/pi-honcho/blob/2644a08ddddc4fc47e71698763bc27e17c47a3fb/extensions/index.ts

## 8. 完成标准

- 第 3 节的问题全部作答，或标为"未知"并说明原因；
- 第 4 节的最小实测完成，或注明未执行的部分与原因；
- 第 5 节的四项交付物齐全；
- 所有来源带固定版本或提交与访问日期，frontmatter 记录 `snapshot_at`，并在[调研目录](./README.md)更新验证状态。
