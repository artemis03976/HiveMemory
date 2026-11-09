---
title: pi 插件接入
status: archived
archived_at: 2026-11-08
implemented_by: PR #201（feat/pi-plugin）
superseded_by: docs/system/import-bus.md
owner: project
target: v1.0 范围的 plugin 模式接入（不绑定具体版本号，owner 2026-11-04）
scope: pi-extension-memory-tools-and-turn-import
related_docs:
  - docs/research/pi-plugin-survey.md
  - docs/contracts/mcp.md
  - docs/system/import-bus.md
  - docs/system/configuration.md
  - docs/help/deepseek-harness-plugin.md
code_paths:
  - integrations/hivememory/
  - configs/system_principals.yaml
  - configs/workspace_actors.yaml
  - tests/unit/connectors/
  - .github/workflows/ci.yml
updates:
  - docs/system/import-bus.md
  - docs/contracts/mcp.md
  - docs/system/configuration.md
  - docs/workspace/authentication.md
  - docs/architecture/overview.md
  - docs/help/README.md
  - docs/help/setup.md
  - docs/help/troubleshooting.md
  - docs/help/deepseek-harness-plugin.md
  - docs/PROJECT.md
  - docs/ROADMAP.md
  - README.md
  - README_EN.md
last_reviewed: 2026-11-08
---

# pi 插件接入

> 状态：Archived（2026-11-08，扩展、接入登记、测试与帮助文档完成，事实文档收尾后归档；以真实模型与本机后端的验收在日常使用中完成，见第 10 节）。当前事实见 [Import Bus](../../system/import-bus.md#11-pi-扩展)第 11 节、[MCP 记忆访问](../../contracts/mcp.md#6-连接器调用名轮次信息与回声标记)第 6 节与[接入 pi](../../help/pi-plugin.md)。以下为实施时的计划原文。原状态：Active（2026-11-08 建立并实施，分支 `feat/pi-plugin`）。owner 决定（2026-11-08）：
>
> - 接入方式与 dsh 相同：pi 扩展自己注册四个记忆工具，在工具执行时补上这一轮的轮次信息；记忆工具只允许模型直接调用，不对 codemode 开放；回流由扩展按事件完成，只用 Node，不用 pi 的内建 MCP，也不用 Python 读取器。
> - 放进统一插件目录 `integrations/hivememory/`，以 pi 包清单附带同一份 `SKILL.md`。扩展的 JS 分为公共部分与 pi 专属部分；dsh 插件在 pi 合并之后再并入这个目录、共用公共部分，本计划不改动 dsh。
> - 社区子 Agent 扩展第一版不支持，写进已知限制。
> - 先在 Windows 原生做机制探针，真实验收在 owner 登录 pi 的模型账户之后进行。
>
> 外部证据见[调研报告](../../research/pi-plugin-survey.md#7-调研结果)第 7 节（pi v1.5.0，2026-11-07，只核对了文档与源码）。本计划另据 owner 本机安装的同版本包自带的扩展、会话格式、包与会话文档核对接口。标为“候选设计”的内容不是 owner 决定，可在计划评审或探针之后调整。

## 1. 背景和问题

v1.0 的 plugin 模式覆盖 Claude Code、DeepSeek Harness、Codex 与 pi（owner，2026-11-04），接入先只支持纯 Windows 环境，harness 与 HiveMemory 都在 Windows 上原生运行（owner，2026-11-07）。Codex 已于 2026-11-08 接入（[归档计划](./codex-plugin-integration.md)），pi 是四个中的最后一个。

pi 与已接入的 harness 有以下不同，它们决定了这次的主要工作：

- **没有持久的轮次标识**：会话记录的条目有稳定的 `id` 与 `parentId`，但没有与 HiveMemory 一轮对应的标识；`turnIndex` 只是模型循环的计数（调研 B2、C1）。dsh 有会话内递增的轮号与 `turn/end`，Codex 有 `turn_id` 与 `task_complete`，pi 都没有，轮次的标识与结束记录要由扩展自己写下。
- **会话是一棵树**：`/tree` 回到较早的条目继续，离开的分支仍留在同一个文件中；`/fork` 与 `/clone` 生成带 `parentSession` 的新会话，并复制当前路径上的条目（调研 C2）。按行追加、按字节游标推进的做法不适用。
- **只有 `agent_settled` 表示结束**：`agent_end` 之后还可能重试、压缩后重试或继续处理排队的消息；中断的运行跳过 `agent_before_settle`，直接到 `agent_settled`（`aborted` 为真）。成功与失败要结合最后一条助手消息的 `stopReason` 判断（调研 B4）。
- **能改写工具参数，但内建 MCP 默认走 codemode**：`tool_call` 能原地改写参数，但扩展没有加载、或被后加载的扩展覆盖时，服务端无法分辨模型自己填写的 `turn`；内建 MCP 默认把工具交给 codemode，经 codemode 的内层调用不进入会话记录，回声标记失效（调研 B2、D5、7.4）。owner 因此选择扩展自带工具。
- **没有内置的子 Agent**：子 Agent 由社区扩展提供，各自另起进程或 SDK 会话；`parentSession` 同时用于 fork 与 clone，无法据此识别子 Agent（调研 7.4）。
- **非交互模式跑完即退出**：`pi -p` 处理完一轮就退出进程，网络提交不能留到进程退出之后；`session_shutdown` 可以等待清理完成（调研 A5、G4）。

## 2. 目标与非目标

目标：

- pi 中的 Agent 经扩展注册的四个记忆工具检索、读取、写入与更新 HiveMemory 记忆；写入与更新由扩展在执行时补上这一轮的轮次信息，模型看不到 `turn`，也无法填写它。
- pi 会话中已结束的一轮回流 Import Bus，确认之后才记为完成；完成、失败与中断分别得到对应的结局；HiveMemory 不可用时不影响 pi，恢复后补发；本扩展记忆工具的调用回流时归一化为 `SEARCH`、`READ`、`WRITE`、`UPDATE`。
- fork 与 clone 继承来的轮次不重复导入；`/tree` 回到较早条目之后的新一轮照常导入，离开的分支上已经导入的轮次不撤回。
- `integrations/hivememory/` 同时是 pi 包，pi 加载同一份 `skills/hivememory/SKILL.md`；Claude Code 与 Codex 的行为不变。
- HiveMemory 一侧登记 pi 自己的接入身份与 Agent 白名单。

非目标：

- HiveMemory 服务端的 MCP 与 Import Bus 契约不变，不新增路由、参数或错误码。
- 经 pi 的内建 MCP 连接 HiveMemory；经 codemode 或其他工具间接调用记忆工具（记忆工具不对它们开放）。
- 社区子 Agent 扩展（owner，2026-11-08）。
- 崩溃或强制结束时进行中的一轮：它没有结束标记，不导入。
- dsh 插件并入统一目录、与 pi 共用公共部分：pi 合并之后单独做（owner，2026-11-08）。
- 接入登记的运行时登记：沿用配置文件，之后另做（owner，2026-11-08）。
- Linux、macOS 与 WSL；controller 模式；与 Honcho 等其他记忆扩展共存时的专门处理。

## 3. 当前状态和缺口

- **服务端**：`/mcp` 与 Import Bus 已支持外部 harness；接入登记有 `harness:claude-code`、`harness:dsh` 与 `harness:codex`，凭据变量按 harness 命名（`HIVEMEMORY_<harness>_TOKEN`，与服务端 `credential_env` 同名）。交互标识由 principal、用户、Agent、来源与两个外部标识推导，不含用户消息（[Import Bus](../../system/import-bus.md#43-交互标识与串行)第 4.3 节）。
- **统一插件目录**：`integrations/hivememory/` 是 Claude Code 与 Codex 共用的插件，连接器是 Python（`connector/`），使用手册 skill 在 `skills/hivememory/`；目录中没有 JS 代码，也没有 `package.json`。
- **dsh 插件**（`integrations/deepseek-harness/`）是只用 Node 标准库的 JS。`lib/client.js`（MCP 调用、导入提交与结果分类）、`lib/state.js`（原子写入的状态）、`lib/log.js` 与 `lib/config.js` 中与 harness 无关的部分可以照着改写，但 principal、来源与凭据变量写死为 dsh；`lib/turns.js` 与 `lib/sync.js` 依赖 dsh 的会话查询与递增轮号，不能用于 pi。
- **pi**：owner 本机装有 `@earendil-works/pi-coding-agent` 1.5.0，与调研固定的版本相同，装在 fnm 管理的 Node v24.21.0 下，尚未运行或登录。调研只核对了文档与源码，Windows 上的安装、事件顺序、工具注册与会话记录都没有实测。
- **本计划另行核对的接口**（同版本包自带的文档）：
  - 扩展可以在会话开始后注册工具，并在同一会话中立即生效；
  - 工具的 `exposure: "model-only"` 只交给模型，不能被 codemode 等其他工具调用；
  - `pi.appendEntry` 写入的 `custom` 条目不进入模型上下文，挂在当前叶子条目之下；
  - 以本地路径安装的包原地加载，不复制；
  - 在 `/tree` 中选中用户消息，会把它的文本放回编辑器，提交后成为新的用户条目。

## 4. 设计

### 4.1 目录与包

`integrations/hivememory/` 加一份 `package.json`，以 `pi` 字段声明扩展入口与 skill 目录；pi 以本地路径安装这个目录，仓库更新即生效。Claude Code 与 Codex 只读各自的清单，不读 `package.json`。

| 位置 | 读取方 | 职责 |
|:---|:---|:---|
| `package.json` | pi | 包名 `hivememory-connector`；`pi.extensions` 指向 `node/pi/index.js`，`pi.skills` 指向 `./skills`；不依赖任何 npm 包 |
| `node/core/` | pi（之后还有 dsh） | 与 harness 无关的部分：按 harness 取值的配置、MCP 调用与导入提交及结果分类、原子写入的状态、日志、记忆工具的回声规则与最近轮次的截断；照着 dsh 的 `lib/` 改写 |
| `node/pi/` | pi | 扩展入口、轮次标记、会话条目到导入轮次的投影、记忆工具 |
| `node/test/` | CI | `node --test` 单元测试 |
| `skills/hivememory/SKILL.md` | 三者 | 同一份正文，不改 |

候选设计：

- 目录名 `node/` 与 Python 的 `connector/` 并列。
- 扩展用 ESM JavaScript 写，不用 TypeScript，不依赖 pi 的包与 typebox。工具参数直接用服务端工具列表给出的 JSON Schema：pi 1.5.0 改用 pi-schema 校验工具参数，TypeBox 中 JSON Schema 之外的能力不再适用（调研 A4）；服务端的 schema 能否直接使用，由探针核对。

### 4.2 按 harness 区分的取值

`node/core/` 的配置由一组 harness 取值构造，与 Python 连接器的 `Harness` 是同一种做法；dsh 并入时只需新增一组取值。pi 的取值：

| 取值 | pi |
|:---|:---|
| 来源 `source` | `pi` |
| principal | `harness:pi` |
| 默认 Agent | `pi`（`HIVEMEMORY_AGENT_ID` 可以覆盖） |
| 凭据变量 | `HIVEMEMORY_PI_TOKEN` |
| 状态目录 | `HIVEMEMORY_CONNECTOR_STATE_DIR`，未设置时为 `~/.hivememory/pi-connector` |
| 服务地址 | `HIVEMEMORY_URL`；候选默认值 `http://127.0.0.1:8769`，即 Windows 原生运行时后端的地址 |

### 4.3 身份与登记

候选设计：

- `configs/system_principals.yaml` 新增 `harness:pi`：kind `harness`，adapter 为 `mcp` 与 `import`，`credential_env: HIVEMEMORY_PI_TOKEN`，不能声明 `system`。
- `configs/workspace_actors.yaml` 为 `(default, pi)` 登记检索、读取、写入与更新，不继承 `memory.run` 与 `agent.call`。

两份配置仍在启动时装载，改完后重启后端。

### 4.4 记忆工具

- **连接**：会话开始（`session_start`）后，在后台向 `/mcp` 取服务说明与工具列表，不阻塞 pi 启动；失败时在下一次开始运行（`before_agent_start`）时重试，直到成功一次。HiveMemory 不可用时工具不出现，pi 照常工作（与 dsh 相同）。按 pi 的约定，后台连接不在扩展的工厂函数中启动，因为工厂也会在不建立会话的命令中运行。
- **注册**：四个工具以服务端给出的名字、说明与参数 schema 注册，schema 中去掉 `turn`，四个工具要么全部注册、要么都不注册。注册时设 `exposure: "model-only"`：codemode 与其他工具不能调用它们，也就不会出现不进会话记录的内层调用，回声标记只需识别会话记录中的直接调用。
- **执行**：调用经 `/mcp` 转交 HiveMemory，结果正文原样交给模型，HiveMemory 的工具错误作为工具错误返回；写入与更新先补上 `turn`（第 4.6 节）。
- **服务说明**：每次开始运行时，经 `before_agent_start` 的 `systemPromptOptions.sections` 加入一节（候选节名 `hivememory`）。它属于系统提示词，上下文压缩后仍在；内容没有变化时，pi 不会重复补发这一节。

### 4.5 一轮的边界与轮次标记

HiveMemory 的一轮对应 pi 从开始运行到 `agent_settled` 的一段。运行中追加的消息（steer）与排队的后续消息（follow-up）都在这一段之内，并入同一轮（与 dsh 相同）。

候选设计：扩展为每一轮在会话记录中写一条结束标记，它是这一轮的标识与结局在会话记录中的唯一来源。

- 开始运行时（`before_agent_start`），扩展生成轮次标识（UUID），记下当时的叶子条目，这一轮的条目都追加在它之后；同时开始收集这一轮的原始输入。
- `input` 事件中，来源为 `interactive` 或 `rpc` 的原始文本（技能与模板展开之前）按顺序收集。来源为 `extension` 的输入（其他扩展经 `sendUserMessage` 发来）不收集（候选）。
- `agent_settled` 时，以 `pi.appendEntry` 追加一条 `custom` 条目（候选 `customType: "hivememory-turn"`），数据为轮次标识、会话标识、开始时的叶子条目、结局与原始输入。`custom` 条目不进入模型上下文。
- 这一轮的条目，是从结束标记沿 `parentId` 回溯、直到开始时叶子条目之前的那一段，与 `/tree` 产生的其他分支无关。
- fork 与 clone 复制来的结束标记带着来源会话的标识，与新会话的标识不同，不导入。
- 接入之前的轮次没有结束标记，自然不回填，不需要比较首次运行时间。
- 没有结束标记的一轮（进程崩溃、强制结束）不导入。

结束标记写在 pi 的会话记录中、而不是插件的状态目录中，原因是轮次的内容本来就在会话记录里：续会话、补发与另一个进程都只需读同一份记录，fork 带走的标记也能按会话标识认出来。代价是用户的会话文件中多了一类 `custom` 条目；它不进入模型上下文，交互界面中是否显示由探针核对。

### 4.6 写入时补上轮次信息

记忆写入与更新只在一轮进行中由模型调用。扩展在工具执行时，从当前一轮构造 `turn`：

- 外部会话为会话标识，外部轮次为这一轮的轮次标识；
- 用户消息为至今收集到的原始输入；
- 之前几轮取当前分支上、这一轮之前最近三条结束标记对应的轮次。

结束标记写下之后，导入用同一个函数从同样的条目构造这些字段，服务端推导出同一个交互标识。

不在一轮之中（没有轮次标识）时，工具返回错误，与 dsh 相同。运行中追加的输入会使导入时的用户消息比写入时多，但交互标识只由来源与外部标识推导，不受影响（与 dsh 相同）。

### 4.7 一轮的投影

候选设计：在 `node/pi/` 中由这一轮的条目构造导入请求。

- **用户消息**：取结束标记中的原始输入，以空行拼接；没有原始输入时为空，例如由其他扩展的 custom 消息开始的运行。不取会话记录中的用户消息，因为技能与模板展开后的正文不是用户的新内容（调研 C3）。
- **事件**：
  - 助手消息的 `text` 块写为 `assistant_message`，有正文的 `thinking` 块写为 `thought`；
  - `toolCall` 块写为 `tool_call`，结局按 `toolCallId` 取对应的 `toolResult` 消息，`isError` 为真时为 `error`；
  - 图片写占位；
  - `custom_message`、系统消息、压缩与分支摘要、`context_edit`、模型切换等条目不读取；
  - `model_used` 取最后一条助手消息的模型。
- **回声标记**：本扩展四个记忆工具的调用归一化为 `SEARCH`、`READ`、`WRITE`、`UPDATE`，规则与 dsh 相同；工具由扩展注册，工具名就是识别依据。
- **结局**：`agent_settled` 的 `aborted` 为真时为 `cancelled`；否则按最后一条助手消息的 `stopReason` 判断，`error` 与 `length` 为 `failed`（此时 pi 已经做过恢复），其余为 `completed`。这一映射在探针中核对。
- **最近轮次**：当前分支上之前最近三条结束标记对应的轮次，正文截断到 2000 字符（与 dsh 相同）。

### 4.8 回流、状态与补发

- **状态**：按会话保存会话文件路径、已确认的轮次标识，以及各轮连续得到服务器内部错误的次数，原子写入。状态只保存标识、路径与状态码，不保存会话正文（与 dsh 相同）；拒绝记录写入 `rejections.jsonl`。
- **触发**：写下结束标记后，在后台提交这个会话中尚未确认的轮次，按结束标记在文件中的顺序提交；会话开始时，补发全部已登记会话中尚未确认的轮次。当前会话从 pi 的会话管理读取条目；其他会话从会话文件逐行读取（候选：只读 JSONL，不经 pi 的会话管理，以免触发格式迁移）。
- **确认后才记为完成**：提交结果的分类与 dsh 相同（确认、可重试、配置错误、服务器错误、拒绝）。可重试与配置错误时停止处理这个会话，等下次触发；同一轮连续三次服务器内部错误后按拒绝处理。同一会话同时只有一个同步在运行，运行期间再次触发时，结束后再同步一次。
- **退出前等待**：`session_shutdown` 时等待进行中的提交，最多等一段固定时间（候选 5 秒），超时的留到下次启动补发。这样 `pi -p` 在这一轮的导入得到确认之后才退出；HiveMemory 不可用时，退出最多多等这段时间。

### 4.9 安装与更新

接入手册新写 `docs/help/pi-plugin.md`，结构参照 Codex 与 dsh 的手册（候选）。要点：

- 以本地路径安装：`pi install <仓库>\integrations\hivememory`，写入 pi 的个人设置。包原地加载，仓库更新后执行 `/reload` 或重开 pi 即生效。
- `HIVEMEMORY_URL` 与 `HIVEMEMORY_PI_TOKEN`（与服务端 `.env` 同名同值）设为 Windows 用户环境变量。
- 不要再在 pi 的 `mcp.json` 中配置 HiveMemory 的 MCP server，否则模型会看到两套记忆工具。
- 确认接入：会话中能看到四个记忆工具，一轮结束后状态目录的日志记下确认。

## 5. 受影响的系统、契约和数据

- **配置**：`system_principals.yaml` 新增 `harness:pi`，`workspace_actors.yaml` 新增 Agent `pi`；服务端凭据变量的说明中加入 `HIVEMEMORY_PI_TOKEN`。服务端代码、公开路由、事件与错误语义不变。
- **集成**：`integrations/hivememory/` 新增 `package.json` 与 `node/`；Python 连接器、两份插件清单与 skill 不变。
- **测试与 CI**：新增 pi 扩展的 `node --test` 任务，与 dsh 插件的任务并列；`tests/unit/connectors/` 新增 pi 扩展与 HiveMemory 一侧声明的一致性测试。
- **用户数据**：pi 的会话文件中，每一轮多一条 `custom` 条目。

## 6. 兼容与迁移

- **Claude Code 与 Codex**：插件目录多出 `package.json` 与 `node/`，两者的清单都不引用；Codex 安装时复制整个目录，多出的文件不会运行。不需要重新安装。
- **凭据未配置**：服务端没有设置 `HIVEMEMORY_PI_TOKEN` 时照常启动并告警，pi 的请求以凭据无效被拒绝，扩展保留未确认的轮次，配置好之后补发。
- 不迁移数据。

## 7. 测试和验收

### 7.1 Windows 机制探针（实施的第一步）

用 owner 本机的 pi 1.5.0（以完整路径调用），`PI_CODING_AGENT_DIR` 指向会话临时目录。模型、MCP 与导入服务都用本机模拟（模型为 OpenAI 兼容端点，按脚本返回工具调用），不调用真实模型与后端。核对：

- 以本地路径安装包后，扩展与 skill 都能加载；Windows 路径（含空格与中文目录）。
- 会话开始后注册的工具立即可用；JSON Schema 参数被接受；`model-only` 的工具不出现在 codemode 中。
- 事件顺序与结束标记：
  - 正常结束、运行中追加输入、排队的后续消息、中断、模型出错与输出达到上限时，`input`、`before_agent_start`、`agent_settled` 的次序，以及 `aborted` 与 `stopReason` 的取值；
  - `agent_settled` 中经 `appendEntry` 写下的条目挂在这一轮最后一个条目之下，从它回溯能到开始时的叶子条目。
- `/tree` 回到较早条目后的新一轮；`/fork` 与 `/clone` 生成的新会话的会话头，以及复制来的结束标记。
- 技能与模板展开后，会话记录中的用户消息是什么。
- `pi -p` 退出时，`session_shutdown` 是否等到提交完成。
- `custom` 条目在交互界面与 `/tree` 中是否显示。

探针结果写入调研报告新增的一节，与 Codex 调研第 7.9 节的做法相同；结果与第 4 节的假设不符时，在实施记录中写明调整。

### 7.2 不变量与归属层

| 不变量 | 归属层 |
|:---|:---|
| 提交结果的分类；按 harness 取得来源、principal、默认 Agent 与凭据变量 | `node/test/` 的公共部分测试（照 dsh 的同类测试改写） |
| 一轮的条目只取结束标记回溯到开始叶子之间的一段；fork 继承来的标记不导入；用户消息取原始输入；事件映射、回声标记与结局 | `node/test/` 的投影测试 |
| 写入时的 `turn` 与导入时的同名字段相同；不在一轮之中时工具报错；交给模型的 schema 中没有 `turn`；工具为 `model-only` | `node/test/` 的工具测试 |
| 确认之后才记为完成；可重试与配置错误时保留、下次补发；同一轮连续三次服务器错误按拒绝处理；退出前的等待有上限 | `node/test/` 的同步测试 |
| 扩展认识的记忆工具与参数就是 MCP server 的；事件种类是导入接受的；principal 与默认 Agent 在随仓库发布的登记中；包清单指向的扩展与 skill 存在 | 新增 `tests/unit/connectors/test_pi_extension.py`，与 dsh 插件的同类测试并列 |

```bash
cd integrations/hivememory && npm test
pytest tests/unit/connectors -q
ruff check . && black --check .
```

### 7.3 真实验收

在 owner 于 pi 中登录模型账户之后进行，使用本机 HiveMemory 后端，执行前征得同意：

- 安装后，真实模型写入记忆得到 `draft_` 句柄，这一轮回流落在写入时确定的话题，生成后读取句柄转到正式记忆；新会话中检索能召回。
- 中断的一轮以 `cancelled` 导入；`/tree` 回到较早条目后的新一轮照常导入；fork 出的会话不重复导入继承来的轮次。
- Claude Code 与 Codex 回归：检索与回流照常。

## 8. 风险与已知限制

- pi 1.x 的扩展接口没有稳定承诺，1.2 与 1.5 都有不兼容的变更（调研 A4）。手册写明核对过的版本，升级 pi 后重新核对第 4 节依赖的事件与字段。
- 同一会话在两个 pi 进程中同时运行时，pi 本身不保证写入顺序（调研 C5）。多个 pi 进程共用状态目录时没有跨进程互斥，同一轮可能提交两次，由服务端按交互标识去重（与 dsh 相同）。
- 崩溃或强制结束时进行中的一轮没有结束标记，不导入。
- 一轮中追加的输入与助手事件的相对位置不保留（与 dsh 相同）。
- 记忆工具不对 codemode 开放：由 codemode 编排的任务不能在脚本中访问记忆，只能由模型直接调用。
- 社区子 Agent 扩展的会话被当作普通会话导入，各占一个新话题；子 Agent 中的写入以它自己会话的轮次确定话题。
- 每次同步都读取整个会话文件，开销随会话长度增长（与 dsh 相同）。
- 与 Honcho 等记忆扩展共存时，它们注入的上下文以 custom 消息出现、不读取；但它们自己的检索工具调用作为普通工具调用导入，可能形成回声。
- 状态目录丢失时，会话文件中已有的结束标记会全部重新提交；超出服务端查重的保留窗口时，会形成重复的交互。

## 9. 完成时必须更新的当前文档

- [Import Bus](../../system/import-bus.md)：新增 pi 扩展一节（组成、触发、轮次标记与投影、状态与补发、配置），组成表、设计不变量、已知限制与代码测试入口随之更新。
- [MCP 记忆访问](../../contracts/mcp.md)：外部 harness 列表，以及第 6 节 pi 的调用名、轮次信息与回声标记。
- [配置](../../system/configuration.md)、[认证与访问登记](../../workspace/authentication.md)：`harness:pi`、`HIVEMEMORY_PI_TOKEN` 与 Agent `pi`。
- [架构概览](../../architecture/overview.md)中的插件列表。
- 帮助：
  - 新增 `docs/help/pi-plugin.md`；
  - 更新[帮助索引](../../help/README.md)、[安装与启动](../../help/setup.md)与[故障排查](../../help/troubleshooting.md)；
  - 修正[接入 DeepSeek Harness](../../help/deepseek-harness-plugin.md) 中复制 skill 的路径，它仍是改名前的 `integrations/claude-code/`。
- [PROJECT](../../PROJECT.md)、README 与 README_EN、[ROADMAP](../../ROADMAP.md)、Research 与 Plans 索引。

## 10. 实施记录

分支 `feat/pi-plugin`（2026-11-08）：`integrations/hivememory/` 新增 pi 包清单 `package.json` 与 `node/`（`core/`：按 harness 取值的配置、客户端、状态、日志与记忆工具的回声规则；`pi/`：扩展入口、轮次投影、同步与记忆工具；`test/`：`node --test`）；接入登记新增 `harness:pi` 与 Agent `pi`；CI 新增 `pi-extension` 任务；`tests/unit/connectors/test_pi_extension.py`；第 9 节的事实文档、新的[接入 pi](../../help/pi-plugin.md)，以及调研报告第 7.8 节的 Windows 实测。

**实施中对设计的调整**：

- 记忆工具的注册：探针发现会话开始时在后台注册的工具赶不上 `pi -p` 的第一次模型请求，改为每次开始运行（`before_agent_start`）时最多等 3 秒让注册完成；在这里注册的工具进入同一轮的第一次请求。
- 待提交的轮次：状态按会话保存会话文件路径与待提交的轮次标识，一轮结束时先记入待提交、再写结束标记，按待提交列表提交，清空后移除状态。第 4.8 节原写“已确认的轮次标识”；按待提交列表提交后，状态目录丢失时不会把文件中的全部标记重提（第 8 节最后一项因此不再成立），接入之前的轮次也自然不回填。
- 用户消息的收集：空闲时只保留最近一条用户输入，开始运行且由用户消息开始时才作为这一轮的第一条输入，被其他扩展处理掉的输入不会混进下一轮。
- 不保存会话的运行（`--no-session`）没有会话文件，不写结束标记、不导入。
- 从 `/tree` 回到一轮中间的条目继续时，那一轮的结束标记不在新分支上，新一轮的最近轮次不含它：写入已知限制。

**验证**（2026-11-08，Windows 原生）：

- `npm test`（`integrations/hivememory`）：38 passed；`pytest tests/unit tests/integration -m "not live_llm and not e2e and not slow" -n auto --cov=hivememory --cov-fail-under=85`：3270 passed，1 skipped，覆盖率 94.85%；`ruff check .`、`black --check .`、`git diff --check` 通过。
- 端到端（隔离的 `PI_CODING_AGENT_DIR` 与状态目录，pi 1.5.0 从仓库的插件目录以本地路径安装，本机模拟的模型与 MCP/导入服务，不调用真实模型与后端）：
  - 写入的 `turn` 由扩展补上并覆盖模型填写的值，与这一轮导入的外部标识相同，带 `WRITE` 与 `SEARCH` 回声标记；续会话的新一轮带上一轮作为最近上文。
  - 导入端点返回 503 时这一轮留在待提交，`pi -p` 不多等；下一次启动先补发它，再导入新的一轮。
  - RPC 模式下，运行中追加的输入与排队的后续消息并入同一轮；中断的一轮以 `cancelled` 导入；`/tree` 回到较早条目后的新一轮照常导入；fork 与 clone 出的会话只导入自己的轮次。
  - 模型出错与输出达到上限的一轮以 `failed` 导入，思考以 `thought` 导入，`/skill:` 输入以原文作为用户消息；包附带的 skill 出现在系统提示词中。

**待验收**（需要 owner 在 pi 中登录模型账户，使用本机 HiveMemory，执行前征得同意）：真实模型写入得到 `draft_` 句柄、这一轮回流落在写入时确定的话题、新会话检索召回；交互界面中的中断、`/tree`、fork，以及结束标记是否显示；Claude Code 与 Codex 的回归。
