---
title: 统一语言设置与外部 Agent 文字的英文版
status: idea
horizon: candidate
owner: system
scope: one-language-setting-for-hivememory-prompts-mcp-text-http-error-text-and-the-plugin-skill
code_paths:
  - src/hivememory/config/app.py
  - src/hivememory/i18n/
  - src/hivememory/system/assembler.py
  - src/hivememory/prompts/compilation.py
  - src/hivememory/alice/orchestration/call/coordinator.py
  - src/hivememory/core/models/agent.py
  - src/hivememory/patchouli/application/preset_agent_profiles.py
  - src/hivememory/mcp_server/server.py
  - src/hivememory/server/app.py
  - src/hivememory/server/error_handlers.py
  - integrations/hivememory/
  - desktop/src/
  - frontend/src/constants/settingsSchema.ts
  - frontend/src/types/agent.ts
related_docs:
  - docs/system/i18n.md
  - docs/contracts/mcp.md
  - docs/system/configuration.md
  - docs/archive/plans/desktop-plugin-installation.md
last_reviewed: 2026-11-12
---

# 统一语言设置与外部 Agent 文字的英文版

**文档状态**：Idea（候选）。方向来自候选清单 D-12（[Epic #42](https://github.com/artemis03976/HiveMemory/issues/42) 的剩余部分）。2026-11-12 先起草为计划，同日 owner 决定先做[桌面端安装插件](../archive/plans/desktop-plugin-installation.md)。skill 怎样交到 harness 手中取决于插件随安装包交付的方式，前提因此改变，计划退回本 Idea（[文档治理规范](../DOCUMENTATION.md)第 8.4 节）。第 2 节是 owner 决定，其余是候选设计。

## 1. 问题

Epic #42 要求用全局配置统一 Agent 提示词的语言。配置项 `i18n.default_language` 与中英两套提示词已经存在（[全局 i18n](../system/i18n.md)），但语言实际上有好几个来源，外部 harness 读到的文字也只有中文：

- **Profile 语言把全局设置挡住了**：解析优先级是“显式值 > Profile 语言 > 组件语言 > 进程默认值”。
  - Agent Profile 的 `language` 字段缺省为 `"zh"`；内置的 omni_doll、两个预设 Agent，以及前端新建 Agent 的初始图纸都写的是 `zh`；管理页又不能修改这个字段（[管理页面](../frontend/management-views.md)）。所以它总有值。
  - 设置页说“默认语言”是“提示词与系统消息使用的语言”。但改为 `en` 之后，只有 Gateway、记忆生成、会话压缩和记忆编译的缺省语言换成了英文。对话 Agent 主帧与 CALL 子帧的 MTP 教学、人设标题、会话摘要和执行期回填仍是中文（[compilation.py](../../src/hivememory/prompts/compilation.py)、[coordinator.py](../../src/hivememory/alice/orchestration/call/coordinator.py)）。
- **没有调用方的优先级**：“组件语言”一档没有任何调用方传入；`FALLBACK_LANGUAGE` 只被导出，没有使用方。
- **配置值宽松**：`default_language` 是自由字符串，经别名归一化后，未知值（例如 `fr`）会静默按 `zh` 运行。
- **外部 harness 读到的文字只有中文**：
  - MCP 的服务说明、四个工具描述、参数说明与结果文本都写死为中文（[server.py](../../src/hivememory/mcp_server/server.py)）。dsh 与 pi 的插件直接沿用服务给出的描述；
  - MCP 工具出错时复用 HTTP 出口的错误投影，说明文字来自 [error_handlers.py](../../src/hivememory/server/error_handlers.py) 中只有中文的错误表；
  - 插件 skill（`integrations/hivememory/skills/hivememory/SKILL.md`）只有中文，description 中的触发词也是中文；
  - dsh 插件在没有进行中的一轮时，会报一条中文错误。
- **两个结构约束**：
  - MCP 服务在 `server/app.py` 导入时就已构造，早于组合根设置语言，描述在构造时就注册了；
  - skill 是静态文件。四个 harness 都只读取插件目录下的 `skills/`，读取时不经过 HiveMemory。

根源在于：语言被当作各个组件、各个 Agent 的“提示词属性”，各处各有一份带缺省值的设置，再用一条优先级链合并。HiveMemory 是一个人的记忆库，这些文字真正应当跟随的只有一件事：用户工作所用的语言。

## 2. owner 决定（2026-11-12）

- 只做进入外部 Agent 上下文的文字：MCP 的服务说明、工具与参数描述、结果文本与错误说明，以及插件 skill。前端界面、桌面客户端与 Gateway 命令回复之后再做。
- HiveMemory 只有一个语言设置，配置键为 `i18n.language`，不再有“默认语言”的概念；Agent Profile 不再携带语言。
- MCP 文字在运行时跟随这个设置；skill 在安装时二选一，同样由这个设置决定。
- skill 的更新随发布版本走：升级到新的发布版本之后，skill 要能自动更新。两次发布之间的开发更新不要求自动跟上。

## 3. 目标与非目标

目标：

1. 只有一个语言设置 `i18n.language`，取值为 `zh` 或 `en`。以下文字都由它决定：HiveMemory 自己的提示词（Gateway、记忆生成、会话压缩、记忆编译、对话 Agent 的主帧与 CALL 子帧）、MCP 文字和 HTTP 错误说明。
2. 设置为 `en` 时，MCP 的服务说明、工具与参数描述、结果文本和错误说明都是英文。
3. skill 有英文版。安装的 skill 由语言设置决定，与 MCP 文字使用同一种语言。
4. 桌面客户端首次部署时，按 Windows 的界面语言写入初始值。
5. 未知的语言值在启动时被拒绝；配置中残留旧键时，错误信息指向新键。

非目标：

- **界面文字**：前端界面，桌面客户端的托盘、对话框与启动页，Gateway 命令回复，以及 SSE 中的安全说明。插件清单中给人看的说明与帮助文档也不翻译。
- **数据**：记忆、话题与 Agent 人设都不翻译，已写入的预设 Agent 也不改。
- **记忆正文的语言**：生成提示词不规定记忆正文用什么语言，这里不改（第 7 节）。
- **按 Agent、按 harness 或按请求选择语言**。
- **运行中切换语言**：仍然重启后生效。
- **输入清单、执行帧与记忆编译选项中的 `language` 字段**：保留（第 7 节）。

## 4. 现状与缺口

| 环节 | 现在 | 候选设计之后 |
|:---|:---|:---|
| 配置 | `i18n.default_language`，自由字符串，未知值按 `zh` | `i18n.language`，只接受 `zh`、`en`；遇到旧键时启动失败，错误指出新键 |
| 语言解析 | 显式值 > Profile > 组件 > 进程默认值 > 回落参数 | 显式值 > 语言设置 |
| 对话 Agent 主帧与 CALL 子帧 | 按 Profile 语言（总是 `zh`） | 按语言设置 |
| Agent Profile | `language` 缺省为 `zh`，管理页不能修改 | 删除字段；已存图纸中的 `language` 键在读取时忽略 |
| MCP 文字 | 写死中文；服务在模块导入时构造 | 中英两份；服务在读取配置之后按语言设置构造 |
| HTTP 错误说明 | 只有中文 | 中英两份，按语言设置选择 |
| skill | `skills/hivememory/SKILL.md` 只有中文，受版本控制 | 中英两份正文受版本控制；`skills/hivememory/SKILL.md` 在后端启动时按语言设置写入，不受版本控制 |
| 桌面客户端 | 不写语言 | 首次部署时按系统语言写入初始值 |
| 设置页 | 字段 `i18n.default_language`，标签“默认语言” | 字段 `i18n.language`，标签“语言” |

## 5. 候选设计

### 5.1 一个语言设置

- **配置**：`I18nConfig.language` 的类型为 `Language`，只接受 `zh`、`en`。代码中的缺省值与 `config.yaml` 一样为 `zh`。
- **旧键**：配置段遇到旧键 `default_language` 时，报错指出“改为 `i18n.language`”，做法与 `shared.llm` 移走时相同。环境变量 `HIVEMEMORY__I18N__DEFAULT_LANGUAGE` 也经过同一个配置段，同样被拒绝。
- **设置一次**：组合根装配时调用一次 `set_language(config.i18n.language)`，位置不变。
- **改名与简化**：
  - `set_default_language`、`get_default_language` 改名为 `set_language`、`get_language`。
  - `resolve_language` 只保留 `explicit` 一个参数：给出显式值时用它，否则用语言设置。
  - `profile_language`、`component_language`、`fallback` 三个参数和 `FALLBACK_LANGUAGE` 删除。

### 5.2 删除 Agent Profile 的语言

- **删除字段**：删除 `AgentProfile.language`。任务进程的中立编译入口（`compile_prompt_input`）与 `CallCoordinator` 都改用语言设置，主帧与子帧的语言因此总是相同。
- **随之删除**：omni_doll、两个预设 Agent 的图纸，以及图纸字段说明中的 `language`。
- **已存的图纸**：`agent_config` 中的 `language` 键在解析为 Profile 时被忽略，存储不迁移。前端保存时不再写这个键。
- **它原本支持的行为**是“某个 Agent 使用另一种语言”，在生产中没有使用方：页面不能修改这个字段，所有来源都写 `zh`。

### 5.3 MCP 文字

- **文字表**：以下文字放进 i18n 包新的 MCP 文本域，按 key 和语言取文本，未知 key 抛 `KeyError`：
  - 服务说明；
  - 四个工具的描述；
  - 参数说明（含 `turn`）；
  - 结果文本。

  工具名、参数名、`draft_`/`rev_` 前缀等机器标识保持不变。
- **构造时机**：`create_mcp_server` 增加语言参数。server 的 lifespan 先读取配置、再进入 MCP 会话管理器，服务改为在这两步之间构造。`/mcp` 路由仍在模块导入时注册，收到的请求转交给 lifespan 中构造的应用。
- **SDK 的占位 handler**：构造挪到日志配置之后，现在为 SDK 的 `basicConfig` 挂的占位 handler 可能不再需要，实施时核对。

### 5.4 HTTP 错误说明

- **双语**：错误表中按错误码给出的安全说明、422 校验说明、上传格式的原因说明，以及固定的兜底说明，都改为中英两份，按语言设置选择。错误码、HTTP 状态与 `reason` 机器值不变。
- **MCP**：MCP 工具的错误结果复用同一个投影，随之切换。
- **副作用**：语言设为 `en` 时，前端显示英文的错误说明，界面其余部分仍是中文。

### 5.5 skill 由语言设置决定

- **两份正文**：现有 skill 移到 `integrations/hivememory/skill-languages/zh/SKILL.md`，新增英文版 `skill-languages/en/SKILL.md`，两份都受版本控制。英文版的措辞决定 Agent 何时检索、写入与修正记忆，需要 owner 审阅。
- **后端启动时写入**：四个 harness 读取的 `skills/hivememory/SKILL.md` 改为不受版本控制。
  - 后端每次启动、读取配置之后，如果插件目录与后端在同一个布局中，就把语言设置对应的那一份复制过去。同一布局指仓库 checkout，或[桌面端安装插件](../archive/plans/desktop-plugin-installation.md)之后的安装目录。
  - 不需要单独的准备步骤，`/health` 也不需要公开语言。
- **为什么放在后端启动时**：升级发布版本和改语言都必须重启后端，而且重启在 harness 重新加载或更新插件之前。所以在重启时写入，就满足第 2 节“随发布版本自动更新”的要求。
  - 链接安装的 dsh 与 pi，重新打开就会读到新文件；
  - 复制安装的 Claude Code 与 Codex，要在插件更新时复制新文件。安装版的插件更新由桌面端安装插件计划负责。
- **覆盖不到的情况**：后端不在插件目录所在的布局中运行时（Docker）不写入。这时 harness 中没有这个 skill，四个工具照常可用。
- **dsh 插件的那条错误**改为英文。连接器不知道语言设置，而这条错误只在连接器状态异常时出现。

### 5.6 桌面客户端写入初始值

- **写入**：安装版第一次选择数据根之后、启动后端之前，如果数据根中没有 `configs\config.local.yaml`，就看系统首选语言：
  - 以 `zh` 开头时不写；
  - 否则写入一个只含 `i18n.language: en` 的覆盖文件。
- **不写**：数据根中已有覆盖文件时不改动；开发方式不写。

### 5.7 设置页

- 字段路径改为 `i18n.language`，标签改为“语言”。
- 说明写明影响范围，以及需要重启后端；Claude Code 与 Codex 的插件还要再更新一次。
- 只改字段路径和文案，不新增组件。

## 6. 影响与兼容

| 位置 | 变化 |
|:---|:---|
| 配置 | `i18n.default_language` 改为 `i18n.language`，类型收紧；旧键被拒绝 |
| i18n | 接口改名，解析链简化；新增 MCP 与 HTTP 错误两个文本域 |
| 提示词编译与 Alice | 主帧与 CALL 子帧按语言设置解析 |
| 核心模型与预设 | `AgentProfile.language` 删除 |
| server | 错误说明按语言选择；lifespan 构造 MCP 服务，`/mcp` 转交请求；启动时写入 skill |
| 插件目录 | 两份 skill 正文；`skills/` 不再受版本控制 |
| 桌面客户端 | 首次部署时写入初始值 |
| 前端 | 设置页字段与文案；Agent 类型与保存请求不再带语言 |

- **残留旧键**：个人覆盖文件或环境变量中残留 `i18n.default_language` 时，启动失败，错误指出新键。
- **语言为 `en` 的部署**：对话 Agent 的提示词由中文变为英文（此前被 Profile 的 `zh` 挡住了）。
- **skill 文件**：升级之后，仓库中的 `skills/hivememory/SKILL.md` 不再受版本控制。其他 checkout 在拉取时会删除这个文件，下次启动后端时重新写入。

## 7. 风险与开放问题

- **记忆正文的语言会随提示词语言变化**：生成提示词不规定输出语言。设为 `en` 之后，新记忆大概率是英文。这属于记忆核心，不在这里处理。
- **预设 Agent 的人设只有中文**。
- **首次部署的判定**：只要数据根中没有覆盖文件就写入。如果用户选用的已有目录恰好没有覆盖文件，同样会写入。
- **重复表达**：输入清单、执行帧与记忆编译选项中的 `language` 字段，与语言设置重复。删除它们会改动 CPU 端口契约。
- **后端写插件目录**：后端从此要知道插件目录在布局中的位置，与现在找 `configs/` 和 `frontend/dist` 的方式相同。

## 8. 升级为 Plan 的条件

- [桌面端安装插件](../archive/plans/desktop-plugin-installation.md)完成，插件目录随安装包交付成为事实（2026-11-12 已完成）；
- 第 5.5 节按那份计划落地后的事实核对一遍，尤其是安装版的插件更新方式。
