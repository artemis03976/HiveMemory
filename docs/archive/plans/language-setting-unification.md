---
title: 统一语言设置与外部 Agent 文字的英文版
status: archived
archived_at: 2026-11-12
implemented_by: PR #220
superseded_by: docs/system/i18n.md
owner: system
target: v1.0 范围（外部 Agent 读到的文字有英文版）
scope: one-language-setting-for-hivememory-prompts-mcp-text-http-error-text-and-the-plugin-skill
updates:
  - docs/system/i18n.md
  - docs/system/configuration.md
  - docs/help/configuration.md
  - docs/contracts/mcp.md
  - docs/contracts/subsystem-contracts.md
  - docs/alice/agent-runtime.md
  - docs/workspace/task-process.md
  - docs/frontend/management-views.md
  - docs/frontend/desktop-client.md
  - docs/help/claude-code-plugin.md
  - docs/help/codex-plugin.md
  - docs/help/deepseek-harness-plugin.md
  - docs/help/pi-plugin.md
code_paths:
  - src/hivememory/config/app.py
  - src/hivememory/i18n/
  - src/hivememory/system/assembler.py
  - src/hivememory/system/system.py
  - src/hivememory/system/plugins/
  - src/hivememory/prompts/compilation.py
  - src/hivememory/alice/orchestration/call/coordinator.py
  - src/hivememory/core/models/agent.py
  - src/hivememory/patchouli/application/preset_agent_profiles.py
  - src/hivememory/mcp_server/server.py
  - src/hivememory/server/app.py
  - src/hivememory/server/error_handlers.py
  - integrations/hivememory/
  - desktop/src/
  - desktop/scripts/package.mjs
  - frontend/src/constants/settingsSchema.ts
  - frontend/src/types/agent.ts
related_docs:
  - docs/ideas/language-setting-unification.md
  - docs/archive/plans/desktop-plugin-installation.md
  - docs/system/i18n.md
  - docs/contracts/mcp.md
---

# 统一语言设置与外部 Agent 文字的英文版

> **状态**：Archived（2026-11-12，已实施，实施记录见第 10 节）。当前事实见[全局 i18n](../../system/i18n.md)、[MCP 记忆访问](../../contracts/mcp.md)与[配置指南](../../help/configuration.md#54-语言)第 5.4 节。以下为实施时的计划原文，链接已按归档位置改正。方向来自候选清单 D-12（[Epic #42](https://github.com/artemis03976/HiveMemory/issues/42) 的剩余部分），依据[统一语言设置 Idea](../ideas/language-setting-unification.md)。owner 的决定（2026-11-12）：
>
> - 只做进入外部 Agent 上下文的文字：MCP 的服务说明、工具与参数描述、结果文本与错误说明，以及插件 skill。前端界面、桌面客户端与 Gateway 命令回复以后再做。
> - HiveMemory 只有一个语言设置，配置键为 `i18n.language`，不再有“默认语言”的概念；Agent Profile 不再携带语言。
> - MCP 文字在运行时跟随这个设置；skill 在安装时二选一，同样由这个设置决定。
> - skill 的更新随发布版本走：升级到新的发布版本之后要能自动更新；两次发布之间的开发更新不要求自动跟上。
> - [桌面端安装插件](./desktop-plugin-installation.md)与[日常实例与开发实例分开](./daily-and-dev-instance-separation.md)完成之后推进这一项。
> - CALL 子帧沿用调用方帧的语言，CPU 内部不读进程的语言设置（第 4.2 节）。
>
> 其余内容是候选设计，不是 owner 决定，可在评审时调整。英文 skill 的措辞要 owner 审阅。

## 1. 背景和问题

Epic #42 要求用全局配置统一 Agent 提示词的语言。配置项 `i18n.default_language` 与中英两套提示词已经存在（[全局 i18n](../../system/i18n.md)），但语言实际上有好几个来源，外部 harness 读到的文字也只有中文：

- **Profile 语言挡住了全局设置**：
  - 解析优先级是“显式值 > Profile 语言 > 组件语言 > 进程默认值”。
  - Agent Profile 的 `language` 字段缺省为 `"zh"`。内置的 omni_doll、两个预设 Agent 和前端新建 Agent 的初始图纸都写 `zh`，管理页又不能修改它，所以这个字段总有值。
  - 设置页把“默认语言”说成“提示词与系统消息使用的语言”。但改成 `en` 之后，只有 Gateway、记忆生成、会话压缩和记忆编译的缺省语言变成英文。对话 Agent 主帧与 CALL 子帧的 MTP 教学、人设标题、会话摘要和执行期回填仍是中文（[compilation.py](../../../src/hivememory/prompts/compilation.py)、[coordinator.py](../../../src/hivememory/alice/orchestration/call/coordinator.py)）。
- **没有调用方的优先级**：“组件语言”一档没有任何调用方传入；`FALLBACK_LANGUAGE` 只被导出，没有使用方。
- **配置值宽松**：`default_language` 是自由字符串，未知值（例如 `fr`）经别名归一化后静默按 `zh` 运行。
- **外部 harness 读到的文字只有中文**：
  - MCP 的服务说明、四个工具描述、参数说明与结果文本都写死为中文（[server.py](../../../src/hivememory/mcp_server/server.py)）。dsh 与 pi 的插件直接沿用服务给出的描述；
  - MCP 工具出错时复用 HTTP 出口的错误投影，说明文字来自 [error_handlers.py](../../../src/hivememory/server/error_handlers.py) 中只有中文的错误表；
  - 插件 skill（`integrations/hivememory/skills/hivememory/SKILL.md`）只有中文，description 中的触发词也是中文；
  - dsh 插件在没有进行中的一轮时，报一条中文错误。
- **两个结构约束**：
  - MCP 服务在 `server/app.py` 导入时就构造好，早于读取配置，描述在构造时注册；
  - skill 是静态文件。四个 harness 都只读取插件目录下的 `skills/`，读取时不经过 HiveMemory。

## 2. 目标与非目标

### 2.1 目标

1. 只有一个语言设置 `i18n.language`，取值为 `zh` 或 `en`。HiveMemory 自己的提示词（Gateway、记忆生成、会话压缩、记忆编译、对话 Agent 的主帧与 CALL 子帧）、MCP 文字与 HTTP 错误说明都由它决定。
2. 设置为 `en` 时，MCP 的服务说明、工具与参数描述、结果文本与错误说明都是英文。
3. skill 有英文版，harness 读到的 skill 与 MCP 文字使用同一种语言。
4. 升级到新的发布版本后，skill 随插件自动更新。
5. 桌面客户端首次部署时，按 Windows 的界面语言写入初始值。
6. 未知的语言值在启动时被拒绝；配置中残留旧键时，错误信息指向新键。

### 2.2 非目标

- **界面文字**：前端界面，桌面客户端的托盘、对话框与启动页，Gateway 命令回复，以及 SSE 中的安全说明。插件清单中给人看的说明与帮助文档也不翻译。
- **数据**：记忆、话题与 Agent 人设都不翻译，已写入的预设 Agent 也不改。
- **记忆正文的语言**：生成提示词不规定记忆正文用什么语言，这里不改（第 8 节）。
- **按 Agent、按 harness 或按请求选择语言**。
- **运行中切换语言**：仍然重启后生效。
- **改语言之后自动更新复制安装的插件**：Claude Code 与 Codex 要在插件页重新安装一次（第 4.5 节）。
- **输入清单、执行帧与记忆编译选项中的 `language` 字段**：保留，删除会改动 CPU 端口契约（第 8 节）。

## 3. 当前状态和缺口

| 环节 | 现在 | 本计划之后 |
|:---|:---|:---|
| 配置 | `i18n.default_language`，自由字符串，未知值按 `zh` | `i18n.language`，只接受 `zh`、`en`；遇到旧键时启动失败，错误指出新键 |
| 语言解析 | 显式值 > Profile > 组件 > 进程默认值 > 回落参数 | 显式值 > 语言设置 |
| 对话 Agent 主帧与 CALL 子帧 | 按 Profile 语言（总是 `zh`） | 主帧按语言设置，子帧沿用调用方帧 |
| Agent Profile | `language` 缺省为 `zh`，管理页不能修改 | 删除字段；已存图纸中的 `language` 键在读取时忽略 |
| MCP 文字 | 写死中文；服务在模块导入时构造 | 中英两份；读取配置之后按语言设置构造 |
| HTTP 错误说明 | 只有中文 | 中英两份，按语言设置选择 |
| skill | `skills/hivememory/SKILL.md` 只有中文，受版本控制 | 中英两份正文受版本控制；`skills/hivememory/SKILL.md` 在后端启动时按语言设置写入，不受版本控制，不进安装包 |
| 桌面客户端 | 不写语言 | 首次部署时按系统语言写入初始值 |
| 设置页 | 字段 `i18n.default_language`，标签“默认语言” | 字段 `i18n.language`，标签“语言” |

## 4. 目标设计

### 4.1 一个语言设置

- **配置**：`I18nConfig.language` 的类型为 `Language`，只接受 `zh`、`en`。代码中的缺省值与 `config.yaml` 一样为 `zh`。
- **旧键**：配置段遇到旧键 `default_language` 时报错，指出“改为 `i18n.language`”，做法与 `shared.llm` 移走时相同。环境变量 `HIVEMEMORY__I18N__DEFAULT_LANGUAGE` 也经过同一个配置段，同样被拒绝。
- **设置一次**：组合根装配时调用一次 `set_language(config.i18n.language)`，位置不变。
- **改名与简化**：
  - `set_default_language`、`get_default_language` 改名为 `set_language`、`get_language`；
  - `resolve_language` 只保留 `explicit` 一个参数：给出显式值时用它，否则用语言设置；
  - `profile_language`、`component_language`、`fallback` 三个参数和 `FALLBACK_LANGUAGE` 删除。

### 4.2 删除 Agent Profile 的语言

- **删除字段**：删除 `AgentProfile.language`。任务进程的中立编译入口（`compile_prompt_input`）改用语言设置；`CallCoordinator` 让子帧沿用调用方帧的语言（owner，2026-11-12），CPU 内部的语言因此只来自输入清单，主帧与子帧总是同一种语言。
- **随之删除**：omni_doll、两个预设 Agent 的图纸，以及图纸字段说明中的 `language`。
- **已存的图纸**：`agent_config` 中的 `language` 键在解析为 Profile 时忽略，存储不迁移。前端保存时不再写这个键。
- **它原本支持的行为**是“某个 Agent 使用另一种语言”，在生产中没有使用方：页面不能修改这个字段，所有来源都写 `zh`。

### 4.3 MCP 文字

- **文字表**：服务说明、四个工具的描述、参数说明（含 `turn`）与结果文本放进 i18n 包新的 MCP 文本域，按 key 和语言取文本，未知 key 抛 `KeyError`。工具名、参数名、`draft_`/`rev_` 前缀等机器标识不变。
- **构造时机**：
  - `create_mcp_server` 增加语言参数；
  - server 的 lifespan 先读取配置，再进入 MCP 会话管理器，服务改在这两步之间构造；
  - `/mcp` 路由仍在模块导入时注册，收到的请求转交给 lifespan 中构造的应用。
- **SDK 的占位 handler**：构造挪到日志配置之后，现在为 SDK 的 `basicConfig` 挂的占位 handler 可能不再需要，实施时核对。

### 4.4 HTTP 错误说明

- **双语**：错误表中按错误码给出的安全说明、422 校验说明、上传格式的原因说明，以及固定的兜底说明，都改为中英两份，按语言设置选择。错误码、HTTP 状态与 `reason` 机器值不变。
- **MCP**：MCP 工具的错误结果复用同一个投影，随之切换。
- **副作用**：语言设为 `en` 时，前端显示英文的错误说明，界面其余部分仍是中文。

### 4.5 skill 由语言设置决定

- **两份正文**：现有 skill 移到 `integrations/hivememory/skill-languages/zh/SKILL.md`，新增英文版 `skill-languages/en/SKILL.md`，两份都受版本控制。英文版的措辞决定 Agent 何时检索、写入与修正记忆，要 owner 审阅。
- **后端启动时写入**：
  - 四个 harness 读取的 `skills/hivememory/SKILL.md` 改为不受版本控制（`.gitignore`），打包时也不复制它。
  - 由 System 的插件模块写入：`HiveMemorySystem` 启动时，如果插件目录存在（与插件服务使用同一个插件目录，仓库 checkout 与安装目录布局相同），就把语言设置对应的那一份写过去；内容相同时不改写。
  - 写入在桌面模式自动更新复制安装插件之前完成：Claude Code 与 Codex 在升级后重新安装时，复制进去的就是新的 skill。
  - 写入不依赖插件代为安装是否打开：开发实例也写 checkout 中的那一份，供 `--plugin-dir` 等方式测试插件时使用。
- **为什么放在后端启动时**：升级发布版本和改语言都必须重启后端，而且重启在 harness 重新加载或更新插件之前，所以在重启时写入就满足“随发布版本自动更新”。
  - 链接安装的 dsh 与 pi，重新打开就读到新文件；
  - 复制安装的 Claude Code 与 Codex，升级后由桌面端已有的自动更新重新复制；只改语言时版本没变，不会自动更新，要在插件页重新安装一次（第 4.7 节的说明）。
- **覆盖不到的情况**：插件目录不存在时（Docker）不写入。这时 harness 中没有这个 skill，四个工具照常可用。
- **dsh 插件的那条错误**改为英文：连接器不知道语言设置，而这条错误只在连接器状态异常时出现。

### 4.6 桌面客户端写入初始值

- **写入**：安装版第一次选择数据根之后、启动后端之前，如果数据根中没有 `configs\config.local.yaml`，就看系统首选语言：以 `zh` 开头时不写，否则写入一个只含 `i18n.language: en` 的覆盖文件。
- **不写**：数据根中已有覆盖文件时不改动；用 checkout 运行时不写。

### 4.7 设置页

- 字段路径改为 `i18n.language`，标签改为“语言”。
- 说明写明影响范围与需要重启后端，并提示 Claude Code 与 Codex 要在插件页重新安装一次，才能读到另一种语言的 skill。
- 只改字段路径和文案，不新增组件，不需要样稿。

### 4.8 实施顺序

1. 配置与 i18n：`i18n.language`、旧键报错、接口改名与解析链简化。
2. 删除 `AgentProfile.language`，主帧改用语言设置，CALL 子帧沿用调用方帧的语言。
3. MCP 文字表与构造时机。
4. HTTP 错误说明的英文版。
5. skill：两份正文、启动时写入、`.gitignore` 与打包；英文 skill 的草稿交 owner 审阅。
6. 桌面客户端的初始值；设置页。
7. 帮助与当前文档；归档本计划与 Idea。

## 5. 受影响的系统、契约与数据

| 位置 | 变化 |
|:---|:---|
| 配置 | `i18n.default_language` 改为 `i18n.language`，类型收紧；旧键被拒绝 |
| i18n | 接口改名，解析链简化；新增 MCP 与 HTTP 错误两个文本域 |
| 提示词编译与 Alice | 主帧按语言设置解析，CALL 子帧沿用调用方帧；Alice 不读进程的语言设置 |
| 核心模型与预设 | `AgentProfile.language` 删除 |
| System | 启动时把对应语言的 skill 写入插件目录 |
| server | 错误说明按语言选择；lifespan 构造 MCP 服务，`/mcp` 转交请求 |
| MCP 契约 | 工具名、参数名与结果结构不变，说明文字随语言设置 |
| 插件目录 | 两份 skill 正文；`skills/hivememory/SKILL.md` 不再受版本控制 |
| 桌面客户端 | 首次部署时写入初始值；打包时不带生成的 skill |
| 前端 | 设置页字段与文案；Agent 类型与保存请求不再带语言 |

## 6. 兼容与迁移

- **残留旧键**：个人覆盖文件或环境变量中残留 `i18n.default_language` 时启动失败，错误指出新键。owner 的日常实例与开发实例都没有设置过这个键。
- **语言为 `en` 的部署**：对话 Agent 的提示词由中文变为英文（此前被 Profile 的 `zh` 挡住了）。
- **已存的 Agent 图纸**：其中的 `language` 键保留在存储中，读取时忽略。
- **skill 文件**：升级之后，仓库中的 `skills/hivememory/SKILL.md` 不再受版本控制。其他 checkout 在拉取时会删除这个文件，下次启动后端时重新写入。
- **已经安装的插件**：语言不变时没有差别。安装版升级后，Claude Code 与 Codex 由已有的自动更新带上新的 skill。

## 7. 测试和验收标准

需要守住的不变量与归属层：

| 不变量 | 归属层 |
|:---|:---|
| 只接受 `zh`、`en`；旧键 `default_language` 被拒绝并指出新键 | `tests/unit/system` 中配置段的单元测试 |
| 解析只看显式值与语言设置 | i18n 的单元测试 |
| 主帧按语言设置编译，CALL 子帧沿用调用方帧的语言；Profile 不再有语言，已存图纸中的 `language` 键被忽略 | 提示词编译、CallCoordinator 与 Profile 解析的现有测试 |
| 语言为 `en` 时 MCP 的服务说明、工具与参数描述、结果文本与错误说明都是英文 | MCP 入口的单元测试 |
| HTTP 错误说明按语言设置选择，错误码与状态不变 | server 错误处理的单元测试 |
| 启动时写入对应语言的 skill，内容相同时不改写；插件目录不存在时不写 | System 插件模块的单元测试 |
| 写入发生在复制安装插件的自动更新之前 | 由启动顺序保证：写入在 `HiveMemorySystem.start` 中，自动更新在其后由 lifespan 启动 |

手动验收：

- 把开发实例的语言设为 `en` 并重启：checkout 中的 `skills/hivememory/SKILL.md` 是英文；用 `--plugin-dir` 加载 checkout 插件的 Claude Code 会话中，MCP 工具描述、结果与错误都是英文。
- 新的数据根、系统语言为英文时，桌面客户端写入 `i18n.language: en`；中文系统不写。
- 安装版覆盖安装后，Claude Code 与 Codex 的插件随自动更新带上当前语言的 skill。

验证命令：

```bash
pytest tests/unit/i18n tests/unit/system tests/unit/server tests/unit/mcp_server tests/unit/prompts tests/unit/alice -q
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85
ruff check . && black --check .
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
uv run --no-sync mypy --warn-unused-ignores --platform linux src scripts/typecheck_bus_routes.py
cd frontend && npm run lint && npm test && npm run build
cd integrations/hivememory && npm test
```

## 8. 风险与待决问题

- **记忆正文的语言会随提示词语言变化**：生成提示词不规定输出语言，设为 `en` 之后新记忆大概率是英文。这属于记忆核心，不在本计划处理。
- **预设 Agent 的人设只有中文**。
- **首次部署的判定**：只要数据根中没有覆盖文件就写入；用户选用的已有目录恰好没有覆盖文件时，同样会写入。
- **重复表达**：输入清单、执行帧与记忆编译选项中的 `language` 字段与语言设置重复，删除会改动 CPU 端口契约，本计划保留。
- **后端写插件目录**：安装版的插件目录在按用户安装的程序目录中，后端可以写；如果以后改为按机器安装，需要重新考虑写入位置。
- **待决**：无阻塞问题。英文 skill 的措辞在实施第 5 步交 owner 审阅。

## 9. 完成时必须更新的当前文档

- [全局 i18n](../../system/i18n.md)：一个语言设置、解析链、MCP 与 HTTP 错误两个文本域；
- [System 配置](../../system/configuration.md)与[配置指南](../../help/configuration.md)：`i18n.language` 与旧键；
- [MCP 记忆访问](../../contracts/mcp.md)：说明文字随语言设置；
- [子系统公共契约](../../contracts/subsystem-contracts.md)、[Agent 运行时](../../alice/agent-runtime.md)与[任务进程](../../workspace/task-process.md)中关于 Profile 语言的描述；
- [管理页面](../../frontend/management-views.md)：设置页的语言字段，Agent 不再有语言；
- [桌面客户端](../../frontend/desktop-client.md)：首次部署写入初始值，打包时不带生成的 skill；
- 四份插件帮助：skill 的语言随设置、改语言后重新安装；
- [ROADMAP](../../ROADMAP.md) 的对应行；
- 本计划与 Idea 移入 `archive/`，从 [Plans](../../plans/README.md) 与 [Ideas](../../ideas/README.md) 中删除，在归档索引中登记。

## 10. 实施记录

2026-11-12 在分支 `feat/language-setting-unification` 实施第 4 节，以 PR #220 合并。英文 skill 的措辞由 owner 同日审阅通过。与计划不同，或计划没有写到的地方：

- **记忆正文的语言**（owner，2026-11-12）：语言改为 `en` 之后，新生成的记忆随之使用英文，这是要求的行为，不是风险。第 2.2 节与第 8 节原把它列为非目标与风险，以此为准。
- **CALL 子帧的语言**（owner，2026-11-12）：子帧沿用调用方帧的语言，不读进程的语言设置，已写回第 4.2 节。
- **skill 的写入位置**：计划写在 `HiveMemorySystem` 启动时，实施改为插件服务的 `write_skill`，由 server 的 lifespan 在系统启动之后、复制安装插件的自动更新之前调用。插件服务持有插件目录，测试中是临时目录，单元测试因此不会往仓库写文件。
- **MCP 服务的构造**：`/mcp` 路由在导入时登记一个端点对象，lifespan 构造 MCP 应用后交给它。SDK 构造时为 `basicConfig` 挂的占位 handler 已删除：服务在日志配置之后构造，根 logger 总有 handler。
- **连接器的错误**：除 dsh“没有进行中的一轮”那条之外，Node 连接器“后端不可达”的错误也改为英文；中文 skill 的错误表随之引用英文原文。
- **测试中的配置替身**：`set_language` 拒绝 `zh`、`en` 以外的值，组合根测试中替身配置的语言改为真实的 `I18nConfig`。
- **设置页**：只改字段路径与说明，没有样稿。

实测（2026-11-12）：

- 后端门槛 3389 项通过、1 项跳过，覆盖率 94.71%；ruff 与 black 通过；mypy 按 Windows 与 Linux 两个平台检查均通过；前端 lint、83 项测试与构建通过；插件的 Node 测试 72 项通过。
- 开发实例启动时按语言设置写入了中文 skill，与正文一致；`/mcp` 的 initialize 与 tools/list 正常返回；设置页显示“语言”字段。

没有实测的是：把语言切到英文跑完整一轮对话，以及非中文系统上桌面端首次部署写入初始值。
