---
title: DeepSeek Harness 插件接入
status: active
owner: project
target: v0.7.3 之前（版本之外的工作，owner 2026-10-31）
scope: dsh-native-plugin-memory-tools-and-turn-import
related_docs:
  - docs/research/deepseek-harness-plugin-survey.md
  - docs/contracts/mcp.md
  - docs/system/import-bus.md
  - docs/system/configuration.md
  - docs/help/deepseek-harness-plugin.md
code_paths:
  - integrations/deepseek-harness/
  - configs/system_principals.yaml
  - configs/workspace_actors.yaml
  - tests/unit/connectors/test_deepseek_harness_plugin.py
  - .github/workflows/ci.yml
updates:
  - docs/system/import-bus.md
  - docs/contracts/mcp.md
  - docs/system/configuration.md
  - docs/workspace/authentication.md
  - docs/help/deepseek-harness-plugin.md
  - docs/help/claude-code-plugin.md
  - docs/help/setup.md
  - docs/help/configuration.md
  - docs/help/troubleshooting.md
  - docs/PROJECT.md
  - docs/ROADMAP.md
  - README.md
  - README_EN.md
last_reviewed: 2026-11-01
---

# DeepSeek Harness 插件接入

> 状态：Active（2026-11-01 建立并开始实施）。接入路径由 owner 决定（2026-11-01）：写成纯 dsh 原生插件，记忆工具与对话回流都在插件内完成，Python 连接器不改动。外部证据见[调研报告](../research/deepseek-harness-plugin-survey.md#7-调研结果)第 7 节；本计划记录 HiveMemory 一侧的设计、验收与收尾清单。

## 1. 背景和问题

owner 决定（2026-10-31）以 DeepSeek Harness（`dsh`）作为第二个接入的外部 harness，采用 plugin 模式：对话回流加经工具的记忆访问，版本之外完成，之后经它进行 agent memory 与 harness 项目的调研（[ROADMAP](../ROADMAP.md) 第 4.2 节）。owner 的方向是记忆的写入与访问主要由执行者经记忆工具主动完成，话题结算的被动提取作为兜底。

调研（2026-11-01 快照，dsh `0.2.2-alpha.2`）得出两点约束：

- dsh 的工具前置事件不能改写调用参数（调研 B2），通用 MCP 客户端配置无法为写入与更新补上可信的 `turn`；插件自己注册的原生工具可以在执行体中以调用方 Agent 的会话构造轮次信息。
- 默认会话文件以 Zstandard 压缩（调研 C1），现有 Python 连接器按行读取的骨架不能直接使用；dsh 推荐经 `sessionQuery.readSession` 读取逻辑事件流。

调研比较了三条路径但不作选择。owner 在 2026-11-01 选择纯原生插件：一个 dsh bundle 同时提供四个记忆工具与对话回流，代价是确认后前进、重试分类等同步规则要在 JavaScript 中再表达一份；另一条组合路径（插件写 outbox、Python 连接器提交）需要维护两种运行时并拆出 Python 公共骨架，未采用。

## 2. 目标与非目标

目标：

- dsh 中的 Agent 经四个原生工具检索、读取、写入与更新 HiveMemory 记忆；写入与更新由插件补上这一轮的可信轮次信息，模型无法填写或绕过。
- dsh 会话中已结束的轮次回流 Import Bus，确认之后才前进；HiveMemory 不可用时不影响 dsh，恢复后补发；本插件记忆工具的调用回流时归一化为 `SEARCH`、`READ`、`WRITE`、`UPDATE`（回声标记）。
- HiveMemory 一侧登记 dsh 自己的接入身份与 Agent 白名单。

非目标：

- HiveMemory 服务端的 MCP 与 Import Bus 契约不变，不新增路由、参数或错误码。
- 引导 skill 的内容（owner 之后编写）；MCP 服务说明中“对话本身会自动归档，不需要复述写入”与主动写入方向的冲突留给 skill 一并处理。
- PTC 模式下的回声标记、多个 dsh 进程共用状态目录时的跨进程互斥、远程部署、controller 模式。

## 3. 当前状态和缺口

- HiveMemory 的 `/mcp` 是无状态、JSON 响应的 streamable HTTP：`initialize`（带服务说明）与 `tools/list` 不需要会话即可调用，已在本机运行的 0.7.2 服务上只读核对（2026-11-01）。写入与更新缺少 `turn` 时以 `request.turn_required` 拒绝。
- Import Bus 按 `principal/user/agent/source/外部会话/外部轮次` 推导交互标识，写入时的落位与回流共用这一标识（[Import Bus](../system/import-bus.md#47-写入时落位)第 4.7 节）。
- 接入登记只有 `harness:claude-code`；Workspace 访问登记只有 Agent `claude_code` 的精确记录。

插件依赖的 dsh 能力均来自固定源码 `ce7073d5` 与实测：工具注册表接受原始 JSON Schema 参数（只对输出 schema 做子集校验）；`session/event` 在事件提交后同步通知；`agent/created` 在 Agent 创建与恢复时触发；`ctx.sessionQuery.readSession(id)` 优先读实时会话，冷读时为崩溃未结束的最后一轮补上 `interrupted` 收尾；`ctx.sessions.flush(session)` 把缓冲事件写入持久记录；`ctx.systemPrompt.section` 注册系统提示词的一节。owner 日常使用的 Windows 桌面端是 dsh `0.2.1-rc.5`（npm 的 `latest`，比调研快照旧），上述能力在这个版本上同样实测可用；调研中提到的 `ctx.agents.turnBoundaryOf` 在 0.2.1-rc.5 中不存在，插件因此不依赖它，当前一轮与水位都从会话记录本身得到。

## 4. 设计

### 4.1 组成

| 文件 | 职责 |
|:---|:---|
| `integrations/deepseek-harness/package.json`、`cordis.patch.yml` | dsh bundle 清单（`engines.dsh` 为 `>=0.2.1-rc.5 <0.3.0-0`：dsh 按包含预发布版本的规则判定，`<0.3.0-0` 才能排除 0.3.0 的预发布版本）与插入插件行 `hivememory` 的配置层 |
| `index.js` | 插件入口：装配、监听 `agent/created` 与 `session/event`、连接 HiveMemory 并注册工具 |
| `lib/config.js` | 环境变量配置、principal 与来源标识 |
| `lib/client.js` | MCP 工具调用与 Import Bus 提交，提交结果分类 |
| `lib/turns.js` | dsh 会话事件到导入轮次与在途轮次的投影、回声标记、结局映射 |
| `lib/tools.js` | 由服务端工具列表生成 dsh 工具定义，写入与更新补上轮次信息 |
| `lib/sync.js`、`lib/state.js`、`lib/log.js` | 同步循环、按会话保存的水位、日志 |
| `test/` | `node --test` 单元测试 |

插件只用 Node 标准库，不依赖 dsh 的包，经 `ctx` 取得 `tools`、`agents`、`sessions`、`sessionQuery`、`systemPrompt` 服务；与 Python 连接器的模块划分对应（`config`、`client`、`state`、`sync`、`log`，读取器对应 `turns`），便于对照两份同步规则。

### 4.2 身份与配置

以下名字是候选设计（不是 owner 决定，可在计划评审时调整）：principal `harness:dsh`，凭据环境变量（服务端）`HIVEMEMORY_DSH_TOKEN`，Agent `dsh`，来源标识 `dsh`。

- 接入登记开放 `mcp` 与 `import`，不能声明 `system`；Workspace 访问登记为 `(default, dsh)` 登记检索、读取、写入与更新，不继承 `memory.run` 与 `agent.call`。
- 插件一侧的环境变量与 Claude Code 连接器同名：`HIVEMEMORY_URL`（默认 `http://localhost:8000`）、`HIVEMEMORY_USER_ID`、`HIVEMEMORY_AGENT_ID`（默认 `dsh`）、`HIVEMEMORY_TOKEN`（去掉首尾空白）、`HIVEMEMORY_CONNECTOR_STATE_DIR`（默认 `~/.hivememory/dsh-connector`）。服务端 `HIVEMEMORY_DSH_TOKEN` 与插件一侧 `HIVEMEMORY_TOKEN` 配置同一份凭据。
- MCP 请求带 `X-Principal-ID`、`Authorization: Bearer`、`X-User-ID`（配置时）与 `X-Agent-ID`；导入请求带前三者，Agent 在请求体 `agent_id`。写入与回流因此得到同一组身份。

### 4.3 记忆工具

- 插件加载时并行请求 `initialize` 与 `tools/list`，把服务端列出的 `memory_search`、`memory_read`、`memory_write`、`memory_update` 注册为同名 dsh 原生工具，说明与参数 schema 取自服务端，不另写一份；服务说明注册为系统提示词的一节 `hivememory`（排在 MCP 服务说明的位置）。四个工具全部注册或全部不注册。
- 连接失败（HiveMemory 未启动、地址不对）时工具不出现，只写日志；之后每次创建 Agent 时重试，直到成功一次。dsh 照常工作。
- 交给模型的 schema 去掉参数 `turn`（及只承载它的 `$defs`）。写入与更新执行时以 `exec.agent.session` 构造轮次信息：`readSession` 读取会话的实时记录，工具在一轮之中执行，最后一轮就是当前一轮，以与导入相同的函数（第 4.4 节）生成 `source`、外部会话与外部轮次标识、用户消息与之前最多三轮；模型给出的 `turn` 被覆盖。最后一轮已经结束或会话还没有轮次时以工具错误拒绝。
- 子 Agent 中的调用以子 Agent 自己的会话与当前一轮落位（候选设计，见第 8 节）。
- HiveMemory 的工具结果正文原样交给模型；`isError` 结果与网络错误作为 dsh 工具错误交给模型。调用超时 60 秒，并随 dsh 的取消信号取消；写入派发开始后的取消不撤回写入，与 MCP 契约相同。

### 4.4 轮次投影

- 一轮为 `turn/start` 到同轮号的 `turn/end`；外部会话标识取会话标识，外部轮次标识取轮号（会话内唯一且递增，恢复与 fork 接着已有的最大轮号编号）。fork 会话只读序号不小于 `inheritedEventCount` 的事件。
- 用户消息：这一轮中来源为 `user` 的 `user/message`（提示、运行中追加的消息、父 Agent 委托的任务）按顺序以空行拼接，图片与文件写占位；注入的上下文（技能正文、目录说明、压缩摘要、子 Agent 结束通知等）不读取。没有这类消息的一轮（例如由子 Agent 结束通知唤醒）用户消息为空。追加消息与助手事件的相对位置不保留（第 8 节）。
- 事件：助手消息的 `text` → `assistant_message`，`reasoning` → `thought`（空正文跳过）；`tool/call` → `tool_call`（`action_id` 为调用标识），`tool/result` 按调用标识补全结局（`isError` → `error`，否则 `success`）；助手消息中的 `tool-call` 块与 `assistant/attempt` 不读取。`model_used` 取最后一条助手消息的模型。
- 回声标记：工具名为四个记忆工具时写 `verb` 与作用对象，规则与 Claude Code 连接器相同（`aliases`、`alias` 移为 `targets`，去掉 `turn`，作用对象形状不对时按普通工具写出）。dsh 工具注册表中同名工具只能有一个，工具名就是识别依据。
- 结局：`completed` → `completed`；`aborted`、`interrupted`、`forked` → `cancelled`；`error`、`blocked`、`max-tokens` 与插件扩展的未知原因 → `failed`（候选设计）。
- 最近轮次：该轮之前最近三个已结束轮次的用户消息、助手正文与外部轮次标识，正文按字符截断到 2000。

### 4.5 回流

- 触发：`session/event` 中的 `turn/end`（记下实际结束的轮号）与 `agent/created`（创建与恢复）同步该会话；插件加载时同步全部“实际结束的轮号大于水位”的会话。同一会话同时只有一个同步，运行期间再次触发时结束后重跑一次。监听器只做本地快速工作，任何失败只进日志，网络提交在后台进行。
- 读取：会话在本进程存活时先 `flush` 再读实时记录，确认之后 dsh 崩溃也不会让已确认的轮次从会话记录中消失、再被新的一轮以同一轮号取代。会话不在本进程时读持久记录，只提交实际看到结束的轮次：冷读补上的收尾可能对应另一个 dsh 进程中仍在运行的一轮。
- 水位只在确认后前进；首次见到的会话按创建时间与插件首次运行时间比较，更早创建的不回填，水位留空，第一次读取时放在当时最后一个已结束的轮次（之后开始的轮次照常提交）；会话被删除（`SESSION_QUERY_SESSION_NOT_FOUND`）时移除状态。
- 提交结果的分类与处理与 Claude Code 连接器相同（[Import Bus](../system/import-bus.md#85-提交结果的处理)第 8.5 节）：2xx 前进；网络错误、超时、429/502/503/504 保留水位并停止；413/422 写拒绝记录后前进；其他 4xx 保留水位并停止；其他 5xx 同一轮连续第三次时写拒绝记录并前进。
- 插件卸载时中止进行中的请求，水位不前进。

### 4.6 状态目录与日志

状态目录保存 `sessions/`（每个会话一份：会话标识、来源 `new`/`existing`、水位 `lastTurn`（接入前已有的会话在第一次读取前为空）、实际结束的最大轮号 `endedTurn`、各轮连续服务器错误次数）、`connector.json`（首次运行时间）、`rejections.jsonl`（会话标识、轮号、状态码、错误码与原因）与 `connector.log`（超过 1 MiB 轮转一份）。只保存标识、轮号与状态，不保存会话正文。dsh 的插件代码在一个线程中运行，状态以同步文件操作读改写、替换写入。

### 4.7 安装

安装、配置、确认、排查与更新的操作步骤（含 Windows 桌面端连接 WSL 中的 HiveMemory）只在[接入 DeepSeek Harness](../help/deepseek-harness-plugin.md)中维护，应 owner 要求（2026-11-01）随实现写入帮助文档，不等收尾。

## 5. 受影响的系统、契约和数据

- 配置：`configs/system_principals.yaml` 新增 `harness:dsh`，`configs/workspace_actors.yaml` 新增 Agent `dsh`。服务端代码、公开路由、事件与错误语义不变。
- 集成：新增 `integrations/deepseek-harness/`；Claude Code 插件不变。
- 测试与 CI：新增插件的 `node --test` 单元测试与 CI job `dsh-plugin`（Node 24）；新增跨语言一致性测试。

## 6. 兼容与迁移

- 不迁移数据。服务端未设置 `HIVEMEMORY_DSH_TOKEN` 时照常启动并告警，dsh 的请求以凭据无效拒绝，插件保留水位，配置后补发。
- 插件只在 `engines.dsh` 范围内加载；dsh 跨次版本升级时重新核对第 3 节依赖的能力并放宽范围。

## 7. 测试和验收

不变量与归属层：

| 不变量 | 归属层 |
|:---|:---|
| 轮次切分、用户消息、事件映射、回声标记、结局映射、fork 前缀、最近轮次与截断 | 插件 `test/turns.test.js` |
| 写入时的轮次信息与之后导入的同名字段一致；`turn` 对模型隐藏且覆盖模型填写；工具错误交给模型 | 插件 `test/tools.test.js` |
| 提交请求头与结果分类 | 插件 `test/client.test.js` |
| 确认后前进、各类结果的处理、不回填、冷读只提交实际结束的轮次、删除即遗忘、加载时补发、运行中触发的重跑 | 插件 `test/sync.test.js` |
| 插件的工具名、作用对象参数与 `turn` 参数与 MCP server 一致；principal 与 Agent 在随仓库发布的登记中 | `tests/unit/connectors/test_deepseek_harness_plugin.py` |

```bash
cd integrations/deepseek-harness && npm test
pytest tests/unit/connectors -q
pytest tests/integration/workspace/test_published_registration_chain.py -q
```

已完成的运行验证（2026-11-01，脚本在仓库之外）：

- 真实 dsh 运行时（Cordis、Session、jsonl 持久化、session-query、AgentLoop、原生执行器，脚本化模型）加载插件，对接以 HiveMemory 真实 MCP 工具列表应答、以真实 `InFlightTurn` 与 `ImportTurnRequest` 模型校验请求的替身服务：四个工具出现且没有 `turn`，服务说明进入每次模型请求；写入带上的轮次信息合法，模型伪造的 `turn` 被覆盖；两轮依次导入并通过校验，`SEARCH`/`WRITE` 回声标记、思考与结局正确，第二轮的最近轮次指向第一轮。
- 导入两次以 503 拒绝后进程退出，新进程加载插件时冷读 zstd 会话记录补发并确认。
- `dsh plugin --profile <隔离 profile> add` 安装 bundle，`--dump-config` 出现插件层；完整 base 组合下以 `dsh headless` 启动时插件加载并连接成功（没有模型凭据，模型请求未完成，与插件无关）。
- 以上两项在 dsh `0.2.1-rc.5` 与 `0.2.2-alpha.2` 上都通过（0.2.1 没有单独的原生执行器包，探针改由 `dsh-tools` 执行工具）。
- Windows 侧（桌面端自带的 `dsh.cmd`，Electron 内置 Node 24.21，隔离的 `DSH_HOME`）：从 Windows 本地目录与 `\\wsl.localhost` 路径安装都成功，两种 profile 以 headless 启动时插件都加载，并经 `127.0.0.1`、`localhost` 连上 WSL 中的替身服务、注册四个工具；Windows 侧经 `127.0.0.1:8769`、`localhost:8769` 调用本机运行的 HiveMemory `/mcp` 的 `tools/list` 通过 Host 检查。

待验收（需要 owner 的模型凭据与真实 HiveMemory）：

- 真实模型经 dsh 写入记忆，得到 `draft_` 句柄，这一轮回流落在写入派发的话题，生成后读取句柄转到正式记忆；新会话中检索召回。
- Web UI 中的日常使用：工具审批、子 Agent、取消与停止、会话恢复后的补发。

## 8. 风险与已知限制

- dsh 是开发者预览版，插件依赖的服务与事件可能不兼容变更；`engines.dsh` 限定在 0.2.x（从 0.2.1-rc.5 起）。桌面端升级会同时升级 dsh 版本，升级后需重新核对。
- 从 `\\wsl.localhost` 路径安装时，桌面端启动需要能访问 WSL（访问时 WSL 会自动启动）；发行版改名或仓库移动后要重新安装。
- 子 Agent 的会话作为独立的外部会话导入，其中的写入落在子 Agent 自己的一轮（候选设计：确定、可恢复，但子 Agent 的轨迹也进入记忆生成）。另一种做法与 Claude Code 相同：只导入顶层会话，子 Agent 中的写入落在父会话当时的一轮，后台子 Agent 会落到之后的一轮。
- 一轮中运行时追加的用户消息并入用户消息，与助手事件的相对位置丢失；没有人发出消息的一轮用户消息为空。
- PTC 模式（默认关闭）下记忆工具在 `run_code` 内调用，会话记录只有外层调用，回声标记不生效。
- 每次轮次结束与每次写入都读取整个会话记录，开销随会话长度增长。
- 崩溃时进行中的一轮只在会话恢复时由 dsh 补上收尾后回流；不再恢复的会话不回流这一轮。
- 多个 dsh 进程共用状态目录时没有跨进程互斥：可能重复提交（服务端按交互标识去重），水位可能回退后重发。
- 插件加载时 HiveMemory 不可用，工具要到下一次创建 Agent 时才出现。
- Web UI 中工具审批、权限预设与记忆工具的交互未实测。

## 9. 完成时必须更新的当前文档

已随实现更新（owner 要求，2026-11-01）：帮助文档。新增[接入 DeepSeek Harness](../help/deepseek-harness-plugin.md)；Claude Code 的接入步骤由安装与启动第 9 节移为[接入 Claude Code](../help/claude-code-plugin.md)，[安装与启动](../help/setup.md)第 9 节改为两份手册的入口；[配置指南](../help/configuration.md)与[故障排查](../help/troubleshooting.md)的相关段落与链接随之更新。

收尾时更新：

- [Import Bus](../system/import-bus.md)：外部 harness 连接器一节加入 dsh 插件（组成、触发、投影、水位与补发、状态目录、已知限制），与 Claude Code 连接器共用的规则只写一处。
- [MCP 记忆访问](../contracts/mcp.md)：外部 harness 列表、dsh 插件的工具注册与轮次信息、回声标记。
- [配置](../system/configuration.md)、[认证与访问登记](../workspace/authentication.md)：`harness:dsh`、`HIVEMEMORY_DSH_TOKEN` 与 Agent `dsh`。
- [PROJECT](../PROJECT.md)、README 与 README_EN、[ROADMAP](../ROADMAP.md)：第二个外部 harness 的状态。
- 归档本计划，从 [Plans 导航](./README.md)移除。
