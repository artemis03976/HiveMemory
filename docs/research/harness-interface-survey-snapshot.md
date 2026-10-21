---
title: 外部 harness 程序化接口调研快照
status: baseline
owner: project
scope: external-harness-programmatic-interface-survey-findings
snapshot_at: 2026-10-20
source_commit: 0e79678d4776a1da4e9c5b1f79115096f6591971
last_reviewed: 2026-10-20
code_paths:
  - src/hivememory/workspace/contracts/cpu.py
  - src/hivememory/workspace/contracts/process.py
related_docs:
  - docs/research/harness-interface-survey.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/ideas/execution-environment.md
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/fifth-generation-architecture.md
  - docs/contracts/subsystem-contracts.md
---

# 外部 harness 程序化接口调研快照

资料层面，**三个候选均有可用的程序化接入路径，首个接入的差异主要落在历史重建、逐轮指令、工具控制与取消回收**。Codex 已有官方 Python 异步 SDK 和结构化历史预填接口；Claude 的工具/权限入口较完整，但外部历史种入与提示词快照需要另外处理；pi 可由 Python 直接驱动 JSONL RPC，但标准 RPC 的会话并发和逐轮配置能力与其 SDK 不同。ACP 能复用基础通信，尚不能抹平这些差异。下文给出版本、来源和限制，不作最终选型。[C-SDK]、[C-INJECT]、[L-PERMISSIONS]、[L-SESSIONS]、[L-PROMPT]、[P-RPC]、[P-RPCTYPES]、[A-schema]

## 1. 范围与阅读方式

本文交付[调研需求](./harness-interface-survey.md)规定的接口对比、候选说明、待决问题的发现汇总和首个 harness 的选择依据。调研日为 **2026-10-20**，服务于 v0.7.1 首个外部驱动的纯文本闭环；作为 Research 证据快照保存，**不决定首个 harness、驱动协议、会话粒度或权限策略**，也不描述 HiveMemory 已有外部接入能力。

本次以官方文档、发行记录和公开接口定义取证；Context7 用于定位，再核对官方原文。用户于调研过程中明确选择“仅完成资料调研，实测列为待验证”。因此，本次没有使用登录凭据、调用模型或进行费用、质量、性能比较，调研需求第 8 节的完整实测验收仍未完成。

证据分为三类：**文档声明**表示官方说明支持；**接口核对**表示在版本化 SDK 类型或协议 schema 中确认字段与语义；**本地检查**仅指本机版本、帮助和离线 schema 生成。三者都不能替代真实模型、多会话隔离、取消和进程回收实测。下文“支持”仅代表该路径有资料依据。

矩阵保留需求的 A1–H2 编号，每格以“支持 / 部分支持 / 不支持 / 未知”开头，并附来源。部分支持表示一题中的能力只覆盖一部分，或需要明确的接口、版本和部署条件；未知会写出未能确认的内容。引用编号对应第 8 节的官方链接、版本和访问日期。

HiveMemory 本地核对基线为 commit `0e79678d4776a1da4e9c5b1f79115096f6591971`。CPU 的实际输入、输出和原生取消语义以 [CPU 端口](../../src/hivememory/workspace/contracts/cpu.py)、[输入清单](../../src/hivememory/workspace/contracts/process.py)及[子系统契约](../contracts/subsystem-contracts.md#4-cpu-端口与-alice-实现)为准。外部 harness 会话是执行侧缓存，恢复不能取代 HiveMemory 原文；此方向的约束见[任务进程 Idea Q-8](../ideas/task-process-table-and-registration-entry.md#q-8-外部执行单元的进程)。

## 2. 产品与版本基线

| 对象 | 本次确认的产品与官方来源 | 版本基线 | 本地检查 |
|:---|:---|:---|:---|
| Codex | OpenAI 的 Codex CLI / app-server / Codex SDK，[openai/codex][C-RELEASE]；不混入 OpenAI Agents SDK | 参考发行 `rust-v0.163.0`；官方 Python/TS SDK 已存在，但本次未核实已发布包版本，不能拿源码的 `0.0.0-dev` 作发行号。[C-SDK]、[C-PYMANIFEST] | CLI `0.156.1`，成功生成该版本的 stable / experimental schema；未验证 0.163.0 运行行为 |
| Claude Code | Anthropic Claude Code + Claude Agent SDK；不混入 `anthropic` Messages SDK 或 Managed Agents | Python SDK `v0.2.168`（2026-10-15，`a39c96e`，捆绑 CLI `2.1.301`）；TS SDK `v0.3.304`（2026-10-19，`b8060fa`，CLI `2.1.304`）。[L-RELEASE]、[L-TSRELEASE] | CLI `2.1.278`；当前 python3 环境未安装 Python SDK |
| pi | Mario Zechner 原 `badlogic/pi-mono` 的 pi coding agent；现官方仓库 `earendil-works/pi`，不是 Inflection Pi、Oh My Pi 或其他移植版 | 正式版 `v1.3.0`（2026-10-15），commit `71023aec1546d6d677a65669c5665c0927ce0dd0`；包为 `@earendil-works/pi-coding-agent`，Node `>=22.19.0`。[P-RELEASE]、[P-PACKAGE] | PATH 无 pi；Node `v22.23.2`、npm `10.9.8` |

这三项均有程序化接口和 API-key 自动化路径，**在资料层面通过初筛**。H1 的这一判断不覆盖共用个人订阅、代用户持有 OAuth token 或任意多用户转售服务；实际账户与部署用途仍需分别确认。没有候选在本次获得“真实纯文本闭环已通过”的结论。[C-AUTH]、[L-LEGAL]、[P-PROVIDERS]

## 3. 候选 × 问题对比矩阵

### 3.1 A：驱动方式

| 编号 / 问题 | Codex | Claude Code | pi |
|:---|:---|:---|:---|
| A1 程序化接口 | **支持**：原生 app-server JSON-RPC、exec JSONL、Python/TS SDK；ACP 用适配器。CLI/SDK Apache-2.0。[C-SDK]、[C-EXEC]、[C-LICENSE] | **支持**：Python/TS Agent SDK，`claude -p` 的 text/json/stream-json；ACP 用适配器。Python wrapper MIT，CLI/TS 另受商业条款约束。[L-OVERVIEW]、[L-HEADLESS]、[L-LICENSE]、[L-LEGAL] | **支持**：TS SDK、print/text、JSON 事件和常驻双向 JSONL RPC；无官方 Python SDK，ACP 用社区适配器。CLI/SDK MIT。[P-INTEGRATION]、[P-SDK]、[P-LICENSE] |
| A2 Python asyncio 代价 | **支持**：`AsyncCodex`（Python ≥3.10）驱动本地 app-server 子进程，无需 Node 旁路；也可自写 stdio client。TS SDK 另需 Node。[C-PY]、[C-TS] | **支持**：Python ≥3.10 异步 wrapper，wheel 捆绑原生 CLI，但每会话仍有 CLI 子进程；client 连接与关闭必须留在同一 async runtime context。[L-README]、[L-CLIENT]、[L-HOSTING] | **支持**：asyncio 启动 `pi --mode rpc`，读写 LF 分隔 JSONL；CLI 本身需 Node，可免另写 SDK host。response id 关联命令，事件通常无命令 id。[P-RPC]、[P-RPCTYPES] |
| A3 稳定性 / 版本协商 | **部分支持**：软件版本明确，可生成 schema；app-server 命令及部分传输仍标 experimental。有 initialize/能力门控，未确认整数协议版本协商。[C-APP]、[C-RELEASE] | **部分支持**：Python 0.2.x、TS 0.3.x，发布频繁；有 CLI 最低版本检查与 initialize，但未确认对外独立 stdio 协议兼容承诺。[L-RELEASE]、[L-TSRELEASE]、[L-TRANSPORT] | **部分支持**：正式软件版本及公共 RPC 类型明确；未见协议版本协商。1.2.0→1.3.0 相隔三天，changelog 仍有 Breaking Changes。[P-CHANGELOG]、[P-RPCTYPES] |
| A4 安装 / Linux / WSL / 冷启动 | **部分支持**：独立二进制、安装器或 npm；Linux/WSL 有官方路径。app-server 可常驻，多 thread；exec 单任务。冷启动、RSS、并发上限未知。[C-CLI]、[C-WSL]、[C-APP] | **部分支持**：pip SDK / 原生 CLI；Linux/WSL 支持，shell sandbox 要 WSL2。一个活动 session 一个 CLI 子进程，可跨轮常驻；冷启动/资源未知。[L-SETUP]、[L-HOSTING] | **部分支持**：npm 固定包或官方安装器，支持 Linux/WSL；RPC 常驻、print/json 单次。冷启动/资源未知。[P-README]、[P-WINDOWS]、[P-INTEGRATION] |

### 3.2 B：会话与轮次

| 编号 / 问题 | Codex | Claude Code | pi |
|:---|:---|:---|:---|
| B1 多轮 / 重启恢复 / 存储 | **支持**：thread/start + 多次 turn/start，thread/resume 按 id 恢复；默认 `~/.codex/sessions`，可隔离 CODEX_HOME，ephemeral 不持久化。崩溃恢复可靠性未测。[C-APP]、[C-RESUME]、[C-EXEC] | **支持**：同一 ClaudeSDKClient 多次 query；将 ResultMessage.session_id 作为 resume 值可在重启后恢复。默认 `~/.claude/projects/<encoded-cwd>/` JSONL，可改 CLAUDE_CONFIG_DIR；不含文件系统快照。[L-SESSIONS]、[L-CLIENT] | **支持**：重复 prompt，`--session` / switch_session 重启恢复。默认 `~/.pi/agent/sessions/` 下按 cwd 分组的 JSONL 会话树；no-session 只存内存。[P-SESSIONS]、[P-COMMANDS] |
| B2 既有历史注入 / 重放限制 | **支持**（原生 RPC）：thread/inject_items 可预填 Responses user/assistant items，持久化并供后续请求使用；角色映射/窗口限待验证。resume.history 明示内部用途、DO NOT USE；高层 SDK 封装覆盖另核对。[C-INJECT]、[C-RESUME]、[C-APP] | **部分支持**：可恢复/镜像原生 transcript；未确认任意外部角色历史导入。AsyncIterable 输入是新请求流。序列化原文为材料会改变角色语义；总长度受模型窗口/压缩约束。[L-SESSIONS]、[L-INPUT]、[L-CLIENT]、[L-TYPES] | **部分支持**：SDK 可通过 SessionManager / importFromJsonl 导入公共会话格式；RPC 无专用历史注入命令，可尝试预建合法 session 文件再加载，尚未验证；长度受窗口/压缩影响。[P-SDK]、[P-FORMAT]、[P-RPCTYPES] |
| B3 多会话并发 / 资源 | **支持**：AsyncCodex 按 turnId 路由多个活动 turn，一个 app-server 可承载多个 thread；同 thread steering 不算独立并行轮次。资源/凭据隔离未测。[C-PY]、[C-TURNSTART] | **支持**：同后端多个 client，对应多个 CLI 进程；不是单 CLI 多路复用。资源数值未知，同 cwd 仍共享文件系统。[L-HOSTING] | **部分支持**：一个 RPC 进程只有一个 active session，new/switch 会替换；并发需多进程或 SDK host 创建多个 AgentSession。后者隔离与资源未测。[P-RPCTYPES]、[P-SDK] |
| B4 自动压缩 / 观测 | **支持**：有自动阈值、contextCompaction item 和 thread/compact/start；不等价于 HiveMemory 原文检查点。[C-CONFIG]、[C-ITEM] | **支持**：自动压缩，compact_boundary / PreCompact 可观测；保留内容由 harness 自管。[L-LOOP] | **支持**：可开关自动压缩，compaction_start/end 报 reason、结果、取消/错误及是否重试，原始 entries 保留。[P-JSON]、[P-SETTINGS] |

### 3.3 C：流式输出与终态

| 编号 / 问题 | Codex | Claude Code | pi |
|:---|:---|:---|:---|
| C1 正文增量 | **支持**：app-server 的 item/agentMessage/delta，带 thread/turn/item 标识；不保证逐 token。exec JSONL 的 item lifecycle 不等于逐片段正文流。[C-DELTA]、[C-EXEC] | **支持**：include_partial_messages=True 后读取 StreamEvent 的 content_block_delta/text_delta；默认 AssistantMessage 并非增量。CLI 还需 stream-json / verbose / include-partial-messages。[L-STREAM]、[L-HEADLESS] | **支持**：message_update 内 text_delta；按 contentIndex 重建，用 message_end 最终块校准，不能重复累计快照。[P-JSON] |
| C2 完成 / 取消 / 错误 / 最终正文 | **支持**：等 turn/completed，检查 completed/interrupted/failed；正文在 agentMessage，Python TurnResult.final_response 可能为 None。上下文超限为 failed+错误。[C-TURN]、[C-PY]、[C-ERROR] | **部分支持**：ResultMessage 为轮末；success 才保证 result。错误/预算/轮数终止有 subtype，stop_reason 可见输出长度；terminal_reason 的 aborted_streaming/aborted_tools 表示取消。部分文本自行保存。[L-LOOP]、[L-TYPES] | **支持**：等 agent_settled，aborted 表示取消；prompt response 只表示接纳，agent_end 后仍可能重试/压缩。get_last_assistant_text 取正文，length 不等于上下文超限。[P-RPC]、[P-COMMANDS]、[P-MESSAGES] |
| C3 思考 / 计划 / 工具 / 子 agent | **支持**：reasoning、plan、commandExecution、fileChange、MCP/动态工具、collabAgentToolCall 等；MCP 有 server/tool/arguments。[C-ITEM] | **支持**：thinking、tool_use/result、system/压缩、任务与额度事件；工具有 id/name/input，子 agent 有关联。子 agent 正文是否转发取决配置，未确认统一独立 plan 事件。[L-STREAM]、[L-TYPES] | **支持**：thinking/toolcall 增量、执行 start/update/end、重试/压缩；tool_execution_start 含 toolName/args。默认无 plan mode/subagents，扩展另计。[P-JSON]、[P-README] |
| C4 认证 / 额度 / 限流 / 模型 / 超窗分类 | **部分支持**：wire 枚举含 unauthorized、usageLimitExceeded、rateLimitExceeded、contextWindowExceeded；模型不可用无专门稳定枚举，保留错误上下文。[C-ERROR] | **部分支持**：authentication_failed、billing_error、rate_limit、invalid_request 等；未确认模型不可用/超窗分别有独立稳定枚举。[L-TYPES] | **部分支持**：command error 与 assistant errorMessage、overflow 压缩可区分；未见跨 provider 统一 auth/quota/rate/model/context 错误码，不能承诺纯结构化分类。[P-RPC]、[P-MESSAGES]、[P-JSON] |

### 3.4 D：取消与回收

| 编号 / 问题 | Codex | Claude Code | pi |
|:---|:---|:---|:---|
| D1 取消 / 部分输出 / 工具 | **部分支持**：turn/interrupt 或 handle.interrupt，等 interrupted 终态；驱动保留 delta。无立即停止或全部工具副作用回滚保证，Task.cancel 不自动等于协议中断。[C-INTERRUPT]、[C-PY]、[C-RELEASE] | **部分支持**：client.interrupt + terminal_reason；需自行保存部分正文。停止时延未知；固定版本说明 mcp 1.x 的进程内工具被 Claude 放弃后仍跑完，不能认定全部工具停止。[L-CLIENT]、[L-TYPES] | **部分支持**：abort 等 idle，但不清队列；交互停止语义须考虑先 clear_queue 再 abort。保留已收正文；工具取消依实现，副作用不回滚、延迟未知。[P-COMMANDS]、[P-AI] |
| D2 强杀后残留 / 会话损坏 | **未知**：有恢复/清理接口，未取得 SIGKILL 后无子进程残留和日志完整的保证，也未实测。[C-RESUME]、[C-APP] | **未知**：正常 close 有 terminate/kill，但固定 transport 明示 anyio shield 不屏蔽裸 Task.cancel/asyncio.timeout，回收 await 可能被打断；强杀后的进程树/JSONL 未验证。[L-TRANSPORT] | **未知**：关 stdin 可正常 shutdown，但强杀后的后代进程、半行 JSONL 和恢复一致性未测；正常退出不构成强杀保证。[P-RPC]、[P-RELEASE] |
| D3 超时 / 无响应 | **部分支持**：thread 状态/read/loaded-list 可查询；stdio 未确认周期心跳，WS health/ready 不证明推理推进。需驱动 deadline/exit 监测。[C-APP] | **部分支持**：API 等待/stream idle/stall watchdog 有文档，但非轮次总时限；未确认独立心跳。context usage 查询会请求 API，不能当无成本探活。[L-PYTHON] | **部分支持**：get_state 可看 isStreaming/isCompacting/队列数；未见原生 heartbeat，状态返回不保证模型推进。需驱动总时限、无进展时限与 EOF 处理。[P-COMMANDS]、[P-RPC] |

### 3.5 E：输入注入与 Profile

| 编号 / 问题 | Codex | Claude Code | pi |
|:---|:---|:---|:---|
| E1 系统指令 / 追加替换 / 上限 | **部分支持**：thread/start/resume 有 baseInstructions/developerInstructions，配置可替换内置基础指令；不能据此宣称清空所有环境/权限提示。独立指令长度上限未知，按窗口预算核验。[C-START]、[C-RESUME]、[C-CONFIG] | **支持**：system_prompt 字符串替换，preset+append 追加；SDK 默认 minimal，与 CLI 默认不同。resume 默认重用首轮快照；逐轮重建需 snapshot=False 及对应版本。独立长度限未知。[L-PROMPT]、[L-TYPES] | **部分支持**：CLI system-prompt 替换/append；SDK resource loader 和 extension 可配置。标准 RPC 无逐轮 set_system_prompt，可能需重启/SDK/扩展；独立长度限未知。[P-CLI]、[P-PROMPT]、[P-RPCTYPES] |
| E2 模型 / temperature / top_p / max_tokens / 语言 | **部分支持**：model/list，thread/turn 选模型及 effort；未见通用 temperature/top_p/max_tokens 字段。上下文或工具 token 限额不能替代生成上限；语言靠指令。[C-APP]、[C-TURNSTART]、[C-CONFIG] | **部分支持**：model/set_model，受账户/provider约束；完整模型枚举未确认。未见 temperature/top_p；启动 env 有 CLAUDE_CODE_MAX_OUTPUT_TOKENS，但非逐轮参数。语言靠指令。[L-PYTHON]、[L-MODEL]、[L-ENV] | **部分支持**：get_available_models/set_model，thinking 可调；标准 RPC 无逐轮 temperature/top_p/max_tokens setter，底层 pi-ai 选项不能算 RPC 支持。语言靠指令。[P-COMMANDS]、[P-RPCTYPES]、[P-AITYPES] |
| E3 大段记忆 / 附件文本 / 长度 | **部分支持**：turn 的 text 块或 exec stdin；材料受上下文/压缩约束，统一字节上限未知；不等于接受任意文件对象。[C-TURNSTART]、[C-EXEC] | **部分支持**：user text/prompt/stdin；verbatim_prompts 可避免 @path/命令解析，也会跳过部分 turn-start 附件/提醒流程。统一材料上限未知。[L-INPUT]、[L-TYPES] | **部分支持**：RPC prompt.message 直接传文本；print/json 支持 @file/stdin，RPC 启动拒绝 @file。单条硬上限未知，仍受模型窗口/压缩影响。[P-RPC]、[P-COMMANDS]、[P-CLI] |
| E4 实际模型 / 配置回执 | **部分支持**：ThreadStartResponse 报 model/provider/权限/cwd 等，有 rerouted 通知；config/read 不是全部采样项实际采用的回执。[C-STARTRESP]、[C-APP] | **部分支持**：AssistantMessage.model、init tools/MCP/权限、model_usage 可观测；未确认完整 Profile 应用回执。[L-TYPES]、[L-SETTINGS] | **部分支持**：get_state/model，assistant provider/model/responseModel（上游有返回时）；SDK 可读有效 systemPrompt/工具集，RPC 无完整参数应用回执。[P-COMMANDS]、[P-MESSAGES]、[P-SDK] |

### 3.6 F：工具、权限与工作目录

| 编号 / 问题 | Codex | Claude Code | pi |
|:---|:---|:---|:---|
| F1 完全关闭 / 限制内置工具 | **部分支持**：可逐项禁 shell/web/MCP 等；未确认公共 Python/CLI/RPC 有统一空工具集开关。read-only 与 never 均不等于禁全部工具；Rust extension ToolPolicy 不能算公共驱动字段。[C-CONFIG]、[C-START]、[C-TOOLPOLICY] | **支持**（built-ins）：tools=[] 映射 `--tools ""`；allowed_tools=[] 只清空自动批准表。MCP/插件/技能加载需另控，tools 本身不禁 MCP。[L-TRANSPORT]、[L-CLIFLAGS]、[L-TYPES] | **支持**：v1.3.0 `--no-tools` 覆盖内置/扩展/custom/MCP 工具；no-builtin-tools 只禁内置。还可禁扩展/MCP 自动加载；不提供进程隔离。[P-CLI]、[P-SDKOPTIONS] |
| F2 权限模式 / 应答 / 无人回应 | **部分支持**：untrusted/on-request/never/granular，另有 reviewer；server-initiated approval 请求由驱动回应。无应答自动拒绝的统一时限未知；高层 Python 默认 auto_review 不能当 deny。[C-APP]、[C-PY] | **支持**：default/acceptEdits/plan/dontAsk/auto/bypass 等与 allow/deny/ask、can_use_tool。回调可无限等待；dontAsk / permission-prompts none 拒绝需询问的请求，但不撤销原已允许工具；所有调用审查要考虑 PreToolUse。[L-PERMISSIONS]、[L-USERINPUT]、[L-CLIFLAGS] | **部分支持**：无内建逐工具 permission system，默认直接执行；project trust approve 不等于工具审批。扩展可 block/confirm，RPC 有 extension_ui；无 timeout 的询问会等待。[P-SECURITY]、[P-EXTENSIONS]、[P-UI] |
| F3 沙箱 / 只读 / 网络 | **支持**（有范围）：read-only/workspace-write 等与文件/网络策略；不能把工具网络限制当模型、MCP、Apps 流量全部断网。[C-CONFIG] | **部分支持**：默认关闭的 shell sandbox，文件/域名可配；MCP、hooks、文件工具不全在该边界。plan 非 OS 只读，restricted 也非完整容器隔离。[L-SANDBOX]、[L-SECURE]、[L-CLIFLAGS] | **不支持**（整个 harness 的内建隔离）：官方要求按需外加容器等；只读工具集/cwd/offline 不是统一文件或网络沙箱，codemode 脚本沙箱不包围整个 harness。[P-SECURITY]、[P-README]、[P-CLI] |
| F4 cwd / env / 目录约束 | **部分支持**：thread/turn cwd，子进程 env、shell 继承策略及写 roots；cwd 不等于读取边界，配置及平台效果待验证。[C-START]、[C-TURNSTART]、[C-CONFIG] | **部分支持**：cwd/env/add_dirs；默认仍继承父进程环境。cwd 非硬边界，文件工具 restricted、shell sandbox、MCP/hook 访问分开核验。[L-TRANSPORT]、[L-SANDBOX] | **部分支持**：子进程 cwd/env，SDK cwd/agentDir；控制资源发现、相对路径和会话分组，不能阻止越目录访问，硬边界要另提供。[P-SDK]、[P-SECURITY]、[P-ENV] |

### 3.7 G–H：MCP、认证与许可

| 编号 / 问题 | Codex | Claude Code | pi |
|:---|:---|:---|:---|
| G1 创建会话注入 MCP / 传输 | **支持**：thread/start.config 可注入 mcp_servers；stdio/Streamable HTTP 有官方说明。旧 SSE 独立传输兼容性未知；ACP 组合另核验。[C-START]、[C-MCP] | **支持**：mcp_servers 支持 stdio/HTTP/SSE，另可提供 Python 进程内 SDK MCP server；后者不代表 harness 在 Python 内运行。[L-MCP]、[L-TYPES] | **部分支持**：v1.3.0 已有官方 MCP extension，stdio/Streamable HTTP，明确无 legacy SSE；SDK 默认不加载，须显式装配 createMcpExtension / bindExtensions 才可会话注册；RPC new_session 不直接收 servers。[P-MCP]、[P-SDK]、[P-EXTENSIONS]、[P-RPCTYPES] |
| G2 作用域 / 凭据 / 线程绑定 | **部分支持**：thread 配置、HTTP headers/bearer env、stdio env 可承载凭据；同 app-server 的 env/CODEX_HOME/OAuth 状态可能共享，跨 thread 凭据隔离未测。[C-MCP]、[C-CONFIG] | **部分支持**：client 级 server、strict_mcp_config、HTTP/SSE headers / stdio env 可承载会话凭据。跨轮保持的连接不会自动重绑新线程或撤销 token，需我方授权点保证。[L-MCP]、[L-TYPES] | **部分支持**：user/project mcp.json 与 session 动态注册并存，可带 headers/env/OAuth；同名文件配置优先于动态注册，RPC/ACP路径能力可能更少。[P-MCP]、[P-EXTENSIONS] |
| G3 MCP 工具是否经权限请求 | **部分支持**：有按 server/tool 策略和 elicitation；并非每次调用必经审批，sandbox 不替代 MCP 资源授权。[C-MCP]、[C-CONFIG] | **支持**（同一权限体系）：mcp__server__tool 可 allow/deny；不保证每次 callback，需交互注解还影响 dontAsk 行为。[L-MCP]、[L-PERMISSIONS] | **部分支持**：扩展可在 tool_call 拦截/询问，默认无审批；工具 annotations 是提示，不是可信授权结论。[P-EXTENSIONS]、[P-SECURITY] |
| H1 凭据 / 无人值守许可 | **部分支持**：官方 API key CLI/CI 路径；ChatGPT/设备码登录也存在，凭据由用户/部署方提供，存 auth.json 或凭据库。个人订阅共用给长期多用户服务的适用性未知。[C-AUTH] | **部分支持**：用户/部署方提供 ANTHROPIC_API_KEY 或云 provider credentials，经 env 等交给 CLI，支持 scripts/CI；个人订阅兼容 SDK/-p，但产品凭据规则另核，不能收集/中介 claude.ai token。订阅凭据的具体物理存储本次未核实。[L-HEADLESS]、[L-SUBSCRIPTION]、[L-LEGAL] | **部分支持**：用户/部署方提供 API key，经 env/auth.json/SDK 注入，官方给 CI 用法；OAuth 先人工登录、可刷新。pi 支持某订阅登录不证明 provider 授权第三方自动化，具体订阅用途未知。[P-PROVIDERS]、[P-AI] |
| H2 软件许可 / 服务条款 / 限流 | **部分支持**：CLI/SDK/本次 ACP adapter Apache-2.0；模型服务、账户共享及部署条款另算。可查询 rate limits，实际账户额度未知。[C-LICENSE]、[C-ACPMANIFEST]、[C-AUTH]、[C-APP] | **部分支持**：Python wrapper MIT；TS SDK/CLI 为商业条款。官方允许产品运行未改 binary，但终端用户凭据/费用归属、不得代售/中介等限制需逐部署核对；限流随组织/方案，未查账户。[L-LICENSE]、[L-TSLICENSE]、[L-CLILICENSE]、[L-LEGAL]、[L-RATELIMITS] | **部分支持**：CLI/SDK MIT，无禁止程序驱动的附加软件许可限制；模型条款、限额由选定 provider/账户决定，无统一 pi 额度。本次不推断所有订阅可用。[P-LICENSE]、[P-PROVIDERS] |

## 4. ACP 的共性与边界

### 4.1 规范版本与 Python 接入

ACP 指 **Agent Client Protocol**，协议官方仓库为 `agentclientprotocol/agent-client-protocol`，不是其他同名 Agent 协议。本次核对的稳定 wire 协议是 **`protocolVersion: 1`**；v2 仍是 Draft。Rust crate 的 `v1.11.0`（2026-10-09）与 JSON Schema artifact 版本各自独立，不能用它们代替 wire 版本。初始化须交换版本和能力，未声明的可选能力不能视为支持。[版本说明][A-version]、[初始化][A-init]、[v1.11.0 发布][A-release]

稳定 stdio 传输是换行分隔的 JSON-RPC 2.0；Python 可以用官方 `agent-client-protocol` 的异步 client 和进程辅助设施，或自行维护 stdio。该库本次观察到主线包版本 `0.12.1`、Python `>=3.10,<3.16`、Apache-2.0，但未安装运行，也未确认它与全部候选适配器的组合兼容。协议层的 Streamable HTTP 仍是草案，不能与“harness 能连接 HTTP MCP server”混为一项能力。[传输][A-transport]、[Python SDK][A-python]、[包声明][A-python-package]

### 4.2 对 2026-10-06 旧核对的复核

| 旧记录 | 本次核对 | 对接入的影响 |
|:---|:---|:---|
| `session/new` 没有系统提示词字段 | 仍成立：当前稳定请求列出 `cwd`、`mcpServers`、可选 `additionalDirectories`、`_meta`，没有通用 system/persona/history 参数。[A-schema] | 私有扩展或启动参数需要单独声明；把人格写进首轮用户正文不等同于系统指令。 |
| 模型只能从 harness 声明的选项选择 | 标准路径仍是读取 `configOptions` 并用 `session/set_config_option` 设置。现有类别包括 `model`、`model_config`、`thought_level` 等，类别只是语义元数据，不保证某个参数存在。[A-config] | 不能由“支持模型选择”推出 temperature、top_p、max_tokens 均可透传。 |
| `tool_call` 不标识调用的工具 | **需要更正**：当前有可选的程序化 `name`，并保留 `rawInput` / `rawOutput`；缺失或 null 仍合法。发布记录显示 `name` 在 2026-09-17 已稳定，不能将其描述成 10 月 6 日之后新增。[A-tools]、[A-changelog] | 可改善观测，但不能依赖每个适配器一定发送工具名，更不能把名字当授权依据。 |
| `session/request_permission` 为 MAY | 仍成立。客户端可按设置自动应答；取消轮次时必须以 cancelled 回答待决权限请求。[A-tools] | “统一拒绝收到的权限请求”不等于“所有工具都被禁用”；agent 可自行执行未发起询问的工具。 |

### 4.3 一个驱动能复用到哪里

ACP 可共用初始化、会话标识、文本分块、权限请求与取消收尾的传输逻辑。`session/prompt` 返回 `stopReason`，正文来自 `agent_message_chunk`，驱动需要自行累积；不要把 thought、工具输出、会话加载时的历史重放纳入本轮正文。[轮次][A-turn]

标准提供 `session/load`，当前还描述可选的 `session/resume`、`session/close`：前者加载时向客户端重放历史，resume 恢复而不重放，close 取消并释放活动资源；均须核对实际能力。**这些接口不等于允许客户端导入任意角色历史**，也没有统一的会话文件位置、内存开销或心跳保证。[会话][A-session]

`session/cancel` 是通知，不能以写入 stdin 成功作为取消完成；应继续接收更新并等待原 `session/prompt` 的 cancelled 响应。协议允许终态前继续发送更新，没有统一的取消时延上限；通用 `$/cancel_request` 的支持及对子操作的传播也是可选。强杀后的工具子进程与磁盘恢复要逐 harness 验证。[轮次][A-turn]、[请求取消][A-cancel]

MCP server 可在 new/load/resume 中提供，stdio 为基线，HTTP/SSE 取决于能力；HTTP/SSE 支持请求头，stdio 可传环境变量。它提供绑定凭据的承载方式，不能证明凭据会按 HiveMemory 执行线程隔离、轮末轮换或即时撤销；后者仍需接入侧设计和实测。[MCP 会话配置][A-session]

据此，ACP 具有复用**通信和基础轮次控制**的依据；仅新增一条登记就能完整接入新 harness，仍取决于系统指令、历史重建、工具关闭、配置应用和适配器覆盖。第 3 节的原生能力不能自动计入 ACP 路径。

### 4.4 三个候选的 ACP 实现

| 候选 | 维护方 / 版本 / 接入形态 | 已核对的覆盖与成熟度边界 |
|:---|:---|:---|
| Codex | 当前是 [agentclientprotocol/codex-acp][C-ACP]，旧 Zed 仓库已归档；固定 `v2.2.2`（2026-10-10，`202e66e`），Apache-2.0。TS stdio adapter 启动 app-server，依赖 `@openai/codex ^0.160.1`、ACP SDK `^1.5.0`。[C-ACPRELEASE]、[C-ACPMANIFEST] | README 声明模型/effort/权限、嵌入上下文、stdio/HTTP MCP；部分增强依赖专有协商，子 agent 涉 draft。已发布不等于本次组合已验证；不能把 2.2.2 解释成 ACP wire v2，也不能将本地 CLI 0.156.1 直接算作此适配器已验证运行时。 |
| Claude Code | Zed 发起，现为 [agentclientprotocol/claude-agent-acp][L-ACP]；固定 `v0.94.0`（2026-10-19，`7bcbfc7`），Node ≥22，Apache-2.0；TS SDK `0.3.303`、ACP SDK `2.0.0`，再启动 Claude CLI。[L-ACPRELEASE]、[L-ACPPACKAGE] | 默认 v1，v2 须显式启用且为 draft。已有 systemPrompt 能力/不稳定类型及 `_meta.systemPrompt`，`_meta.claudeCode.options.tools=[]` 可禁内置工具；MCP 参数已接线，但不是所有 SDK 选项都透明透传。0.x 活跃维护，未验证具体组合；README 明示取消后 background commands 可能继续。[L-ACPSOURCE] |
| pi | 本次只核对社区维护者 Sergii Kozak 的 [svkozak/pi-acp][P-ACP]；固定 `v0.0.34`（2026-09-24，`b0581c9`），MIT，桥接到 `pi --mode rpc`；要求 pi ≥0.81.0 / Node22，依赖 ACP SDK `^0.26.0`。[P-ACPRELEASE]、[P-ACPPACKAGE] | README 标 MVP、预期 minor breaking changes；会话映射存 `~/.pi/pi-acp/session-map.json`。**接收 mcpServers 但未转给 pi**，无 fs/terminal delegation、无单独 thought stream。pi 1.3.0 自带 MCP 不会自动修复桥接缺口；0.0.34 × 1.3.0 未实测。其他同名 adapter 的能力不能混入这一行。 |

适配器版本、其依赖的 ACP SDK 包版本、协商得到的 wire 协议版本是三项不同数据。以上三项均为适配器支持，不能写成候选原生 ACP。Claude 适配器新增的 systemPrompt 能力不改变第 4.2 节对稳定 v1 通用字段的判断。

## 5. 候选接入路径与风险

**Codex** 可以由 Python AsyncCodex 管理本地 app-server，也可直接接 JSON-RPC；exec 适合单任务调用，正文增量和审批控制应按所选接口单独验收。当前 Python 入口使 Node 不再是原生接入的必需旁路；thread/inject_items 给出了从 HiveMemory 原文重建角色历史的可行接口。主要缺口是统一空工具集没有公共证据、采样参数无法等价映射、SDK/CLI 版本并不对齐，以及共享 app-server 时的状态和凭据隔离。SDK 源码 manifest 的开发版本与配套 CLI pin 不能当作发行版本。仅有 CLI 0.156.1 帮助与 schema 的本地检查，没有模型运行记录。[C-SDK]、[C-PY]、[C-INJECT]、[C-PYMANIFEST]、[C-CONFIG]

**Claude Code** 的 Python SDK 路径有官方输入、流式、interrupt、工具控制和权限回调。部署仍需每活动会话的 CLI 子进程；同一 client 的 async context 约束要求在后续生命周期设计中明确持有者。任意外部角色历史直接导入未确认，恢复 transcript 不能替代这项能力；逐轮 instructions 又受 system prompt snapshot 影响。固定版本的原生 asyncio 取消回收 caveat、MCP 工具取消边界以及 tools 与 allowed_tools 的不同语义，都会改变最小闭环的验收方式。ACP 已覆盖系统提示词、tools 和 MCP 的多项接线，但部分依赖候选专用扩展。仅检查本机 CLI 2.1.278 的版本和 help，未运行 SDK 会话。[L-CLIENT]、[L-SESSIONS]、[L-PROMPT]、[L-TRANSPORT]、[L-TYPES]、[L-ACPSOURCE]

Claude 的 `snapshot=False` 不是无代价的逐轮更新方案：官方提示它会改变提示缓存复用，并可能影响保留的 thinking，建议生产保持 recording。该入口证明有调整途径，具体如何承载每轮环境变化仍需要结合清单语义选择和验证。[L-PROMPT]

**pi** 的标准 RPC 可直接由 Python subprocess 驱动，无需先写 Node SDK host；但一个 RPC 进程只有一个活动会话。原生 SDK 的历史导入和资源加载能力比 RPC 丰富，若为逐轮人格/采样参数引入扩展或 SDK host，应将这部分工作计入驱动成本。v1.3.0 有覆盖所有工具来源的 no-tools 和官方 MCP extension；同时没有默认逐工具审批或整个 harness 的内建沙箱，不能把 project trust 当工具权限。agent_settled、clear_queue+abort 是终态/停止对接中的关键语义，社区 ACP 的 MCP 缺口另算。本机没有 pi，仅核对 Node/npm 环境，未安装或运行该候选。[P-RPC]、[P-COMMANDS]、[P-SDK]、[P-CLI]、[P-MCP]、[P-SECURITY]、[P-ACP]

## 6. 按待决问题汇总的发现与影响

以下均为基于接口证据的**接入影响分析**，不选择实现位置、策略或最终协议。

| 待决问题 | 调研发现与影响 |
|:---|:---|
| **E-2a 驱动方式** | 三家都有非交互/常驻接口；Codex/Claude 有官方 Python 异步入口，pi 可直接 JSONL RPC。ACP 能统一基础流，但 systemPrompt/禁工具/历史/MCP 覆盖仍随 adapter 与版本变化。选择通用协议时，仍须给每个登记核对能力、扩展和运行时版本，不能仅由“兼容 ACP”推导零代码新增 harness。[C-PY]、[L-CLIENT]、[P-RPC]、第 4 节 |
| **Q-8 外部执行单元的进程** | OS 子进程、harness 会话/turn、HiveMemory 任务进程是不同生命周期。Codex 支持多 thread 共用 server，Claude 每会话一个 CLI，pi RPC 一个 active session。缓存丢失后，Codex inject、pi SDK import 与 Claude 未确认的外部历史种入形成不同重建成本。取消通知/正常 close 都不足以证明强杀时无残留；闲置回收、deadline 和缓存对应关系仍待决定。[C-PY]、[C-INJECT]、[L-HOSTING]、[L-SESSIONS]、[P-SDK]、[P-RPC] |
| **T-7 组合有效性** | 模型可选不等于 Profile 所有字段能应用。三家对 temperature/top_p/max_tokens 均没有统一完整的逐轮映射，语言主要靠指令；指令的层级、持久化快照和逐轮更新也不同。组合有效性的判断需要覆盖模型可用性、指令注入方式、材料预算、工具控制与 cwd，而不是只检查 harness 可执行文件存在。何时拒绝或报告降级仍为待决。[C-TURNSTART]、[L-PROMPT]、[L-TYPES]、[P-RPCTYPES] |
| **T-8 环境选择与层级控制** | 一个环境登记至少需要明确实际接口及运行时版本，才能解释同一模型目录、权限与配置的含义；全局配置、会话状态、进程 env 的作用域各不相同。若会话中更换 Agent/环境/模型或权限，旧 prompt 与 MCP 凭据是否可复用必须另外判定。资料不能确定选择应放会话还是请求，也不决定控制 API 的位置。[C-START]、[L-PROMPT]、[L-MCP]、[P-MCP] |
| **T-11 工具权限与 cwd** | 工具注册/暴露、自动批准、询问回调、OS 沙箱是独立控制面。Codex 未确认统一禁工具；Claude 可禁 built-ins 但 MCP 另算；pi no-tools 可全关而默认无审批系统。拒绝收到的 ACP 请求不能覆盖未请求批准的执行。三家 cwd 都不能单独证明目录隔离，无应答处理也不能靠默认假定。权限应答与工作目录策略仍需 owner 决定。[C-CONFIG]、[L-PERMISSIONS]、[P-SECURITY]、[A-tools] |
| **E-2 运行时访问传输** | 三家原生路径都存在 stdio/HTTP MCP 接入方式，但会话创建参数、扩展与继承设置不同，pi 的本次 ACP adapter 仍未接通 MCP。MCP 工具调用可为后续回调提供承载，不能替代执行结束信号或自行确定 HTTP/JSON 与 MCP 的取舍。[C-MCP]、[L-MCP]、[P-MCP]、[P-ACP] |
| **E-3b 操作适配器位置** | 子进程/SDK host 需要把外部工具调用归一化并进入已有能力层；协议不规定该适配器在 HiveMemory 包层级的位置。Python 进程内 MCP 的可用性也不改变公共契约或资源所有权，不能因实现方便就把外部请求直接接到 Patchouli。[L-MCP]、[CPU 端口](../../src/hivememory/workspace/contracts/cpu.py) |
| **P-1b 请求与进程归属** | HTTP header / stdio env 能携带凭据，但 harness session 可能跨越多个任务进程，而执行线程在每轮结束时撤销。复用连接不能复用已过期的线程授权；需要将当前请求与当前存活线程/进程核对，不能用外部 sessionId/threadId 或观测标签代替访问身份。[C-MCP]、[L-MCP]、[P-MCP]、[执行线程](../workspace/execution-threads.md) |
| **P-1c 身份证明位置** | 三家可以带 token/OAuth 信息，只证明具备凭据承载；token 的签发、绑定、撤销与身份验证由接入侧决定。共享 app-server、配置目录、环境及 OAuth 缓存的边界未实测，不得假设由 harness 自动完成 HiveMemory principal 认证。[C-CONFIG]、[L-TYPES]、[P-MCP] |
| **T-10 输出词汇** | 原生事件通常给工具名/参数、思考、压缩与子 agent 关联，ACP 的 tool name 却允许缺失；相同 JSONL 字样也不代表同样的流粒度。正文、重放历史、工具结果、压缩摘要和部分终态必须分别识别；pi 等 agent_settled，Claude 避免 delta+完整块重复，Codex 区分 item 与 turn 结束。如何投影外部事件仍待决定。[C-ITEM]、[L-STREAM]、[P-JSON]、[P-RPC]、[A-tools] |
| **P-10a Profile 应用结果** | 可观测模型及部分配置，但本次未确认三家提供完整的每字段实际应用回执。提交参数、harness 接受参数与上游实际采用参数不是同一事实。后续需要保留“支持 / 不支持 / 未知 / 观测到何值”的区别，不能用请求值填充生效结果；具体承载由后续讨论决定。[C-STARTRESP]、[L-TYPES]、[P-MESSAGES] |

## 7. 首个 harness 的选择依据与待验证项

### 7.1 纯文本闭环的必要条件

下面列出形成首份计划所需的证据，而非评分或最终排名。“部分”不自动淘汰候选：应先决定哪些缺口可由驱动补足、哪些组合应拒绝，并对所选路径做实际验证。

| 条件 | Codex | Claude Code | pi |
|:---|:---|:---|:---|
| Python asyncio 可驱动，不阻塞多个会话 | 文档支持 Python SDK / RPC 多 thread | 文档支持多 client/CLI；async context 所有者需明确 | 文档支持 RPC；多会话需多进程或 SDK host |
| 同会话多轮，缓存丢失后可从我方原文重建 | 多轮/resume/inject 有接口；映射与去重待验 | 多轮/resume 有接口；任意角色历史重建未确认 | 多轮/resume 有接口；SDK 导入可用，RPC 路径有额外工作 |
| 指令、记忆/附件文本和原始用户消息可按清单传入 | 具备指令与文本入口；层级/逐轮更新待验 | 具备入口；snapshot/verbatim 组合待验 | 具备入口；逐轮 system prompt 需补足 |
| 正文增量、确定轮末、独立最终正文 | app-server 文档支持 | 文档支持；失败/取消正文自行保存 | 文档支持；必须等 agent_settled |
| Task.cancel 后中断外部轮次、返回已有正文且回收资源 | interrupt 有接口；桥接/回收待验 | interrupt 有接口；裸 asyncio cleanup 风险待验 | clear_queue/abort 有接口；桥接/回收待验 |
| 工具行为可限制，权限无人应答时不无限挂起 | 部分工具开关+审批；统一零工具集未知 | built-ins 空集、MCP另控；拒绝/回调有接口 | no-tools 全工具空集；启用工具后审批需另提供 |
| 必需模型与逐轮参数的应用或未应用情况可明确 | 模型可选；采样项存在缺口 | 模型可选；采样项与上限作用域有缺口 | 模型可选；RPC 采样项有缺口 |
| Linux/WSL、可控 cwd/env、可用认证与许可 | 资料有路径；具体部署策略待核 | 资料有路径；版本/凭据场景待核 | 资料有路径；选定 provider 条款待核 |

各格证据沿用第 3 节对应的 A2、B1–B3、C1–C2、D1–D2、E1–E4、F1–F4、H1–H2，限制同样生效。首步不回调 HiveMemory 时，MCP 的成熟度是后续扩展成本，不能单独冒充纯文本闭环已完成的证据。若把纯文本模式进一步定义成“模型可见的工具集必须为空”，Codex 的公共路径仍有未确认项；若允许受限工具，应明确 T-11 的实际要求，不能把 read-only 当作空工具集。

资料已足以比较四类成本：**原生 Python SDK 的生命周期、ACP 的能力差异、历史重建、工具与权限控制**。尚不足以给出三家启动/内存/取消时延排名，也不足以宣称任一候选已满足全部运行条件。本文不作最终决定。

### 7.2 实际执行过的本地检查

| 范围 | 命令 / 操作 | 实际结果与可证明范围 |
|:---|:---|:---|
| 仓库基线 | git status --short；git rev-parse HEAD | 开始时工作区干净；commit 见第 1 节 |
| Codex 版本与帮助 | codex --version；codex app-server --help；codex exec --help | 均成功；本机 0.156.1，确认命令/flag 存在，不证明推理可用 |
| Codex 离线 schema | codex app-server generate-json-schema --out 临时目录；另加 --experimental 再生成 | 均退出 0，生成 310 / 436 个 JSON 文件；stable 可见 inject_items、指令字段，未见 TurnStart 的三项采样字段；仅证明该二进制公开的序列化表面 |
| Claude 版本与帮助 | claude --version；claude --help | 均成功；CLI 2.1.278，可见 tools、stream-json、permission-prompts 等；未启动会话 |
| Claude Python 包 | python3 importlib.metadata 查询 claude-agent-sdk | PackageNotFoundError，仅表示当前 python3 环境未安装 |
| pi 环境 | PATH 查询 pi；node --version；npm --version | pi 未找到；Node v22.23.2 / npm 10.9.8；没有安装 pi |
| 公共网络抓取 | python3 urllib 请求公共 GitHub | shell DNS 失败；官方资料改经 Context7 与浏览工具核对，无须凭据，不影响资料层交付 |

临时检查产物位于仓库外 `/tmp`，不作为长期依赖，不提交完整 schema 或日志。没有读取认证状态、登录文件或密钥；没有把 shell 联网失败解释成候选产品故障。

### 7.3 留给后续的最小实测记录

**以下全部未执行**，符合用户本次仅资料调研的范围；保留原需求的八项验证，不将版本/help/schema 检查计为通过。后续应固定 harness、SDK、adapter、模型/provider 与实际驱动路径，使用用户确认的凭据，在仓库外临时目录记录命令、终态、脱敏事件和资源回收结果。

| 需求项 | 最小观察与通过依据 | 三个候选的本次状态 |
|:---|:---|:---|
| A2 异步接入 | 启动→输入→流读取→正常关闭；后端可同时推进其他会话，无协议输出污染 | 均待验证 |
| B1 多轮与恢复 | 两轮连续性；重启 harness 后按标识继续；记录缓存丢失时从 HiveMemory 原文重建的结果，不能只验证磁盘 resume | 均待验证 |
| C1 流式正文 | 在轮末前收到至少一个可识别正文片段；最终文本与事件重建一致且不重复 | 均待验证 |
| C2 终态 | 完成、取消和具体失败分开记录；接纳响应/单条消息结束不会提前封口 | 均待验证 |
| D1 取消 | 收到部分正文后触发原生 asyncio Task.cancel；确认外部中断、已有正文返回、无后续排队生成；另核对权限等待和工具子进程 | 均待验证 |
| E1 指令 | 同一会话两轮改变已定稿 instructions，确认对应字段/快照行为；语言和材料布局按实际路径记录 | 均待验证 |
| F1 工具限制 | 检查模型可见工具及实际工具事件；区分禁 built-ins、禁所有工具、只读和拒绝审批，并排除继承 MCP/扩展 | 均待验证 |
| F2 无人审批 | 实际触发请求后显式拒绝；无应答时按后续选定 deadline 终结，取消不会留下 pending 回调 | 均待验证 |

冷启动/RSS/并发资源、D2 强杀恢复和每提供方错误分类也仍未知；本次不虚构数值。调研需求四项资料交付已经具备，完整实测完成标准保留待验证。

## 8. 来源登记

以下来源的访问日期均为 **2026-10-20**。滚动产品文档只表示当日声明，不能证明本机旧版本具备全部能力；固定 tag / commit 的链接按其版本阅读。本文不复制凭据、原始日志或临时生成的完整 schema。

### 8.1 ACP

| 来源 | 官方页面 | 对应版本或范围 |
|:---|:---|:---|
| A-version | [官方仓库的版本说明][A-version] | 稳定 wire v1；crate/schema 独立版本 |
| A-init | [Initialization][A-init] | 协议 v1 滚动规范 |
| A-transport | [Transports][A-transport] | 协议 v1；远程传输草案另列 |
| A-schema | [Schema][A-schema] | 协议 v1 稳定字段，非 v2 Draft |
| A-session | [Session Setup][A-session] | 协议 v1 滚动规范 |
| A-turn | [Prompt Turn][A-turn] | 协议 v1 滚动规范 |
| A-tools | [Tool Calls][A-tools] | 协议 v1，name 可选 |
| A-config | [Session Config Options][A-config] | 协议 v1，按声明设置 |
| A-cancel | [Cancellation][A-cancel] | 协议 v1，区分请求与轮次取消 |
| A-python | [官方 Python SDK 文档][A-python] | 未实测；与包声明分开取证 |
| A-python-package | [Python SDK pyproject][A-python-package] | 主线观察值 0.12.1，不宣称为已验证发行组合 |
| A-release | [Rust Crate v1.11.0][A-release] | 2026-10-09；稳定 compaction/notices |
| A-changelog | [官方 CHANGELOG][A-changelog] | 1.8.0：2026-09-17 稳定 tool call name |

### 8.2 Codex

产品网页为当日滚动快照；下表公共类型和 SDK 源文件固定 `rust-v0.163.0`，本地 schema 检查另为 `0.156.1`。SDK 已发布包版本未核实，保留未知；Python manifest 的配套 CLI pin 为 `0.153.4`，不能用它或主仓库发行号冒充 SDK 版本。

| 来源 | 官方页面 | 对应版本或范围 |
|:---|:---|:---|
| C-RELEASE | [发行页][C-RELEASE] | rust-v0.163.0 |
| C-SDK / C-PY / C-TS | [产品 SDK 入口][C-SDK]、[Python API][C-PY]、[TS SDK][C-TS] | 网页滚动；API/README 固定 tag |
| C-APP / C-EXEC | [app-server][C-APP]、[非交互模式][C-EXEC] | 滚动文档 |
| C-CONFIG / C-MCP | [配置参考][C-CONFIG]、[MCP][C-MCP] | 滚动文档 |
| C-AUTH / C-CLI / C-WSL | [认证][C-AUTH]、[CLI][C-CLI]、[WSL][C-WSL] | 滚动文档；不代表特定用户条款与额度已验证 |
| C-START / C-RESUME / C-INJECT | [ThreadStartParams][C-START]、[ThreadResumeParams][C-RESUME]、[ThreadInjectItemsParams][C-INJECT] | 固定 tag 的公共请求类型 |
| C-TURNSTART / C-STARTRESP | [TurnStartParams][C-TURNSTART]、[ThreadStartResponse][C-STARTRESP] | 固定 tag 的公共类型 |
| C-DELTA / C-TURN / C-ITEM / C-ERROR | [正文 delta][C-DELTA]、[Turn][C-TURN]、[ThreadItem][C-ITEM]、[CodexErrorInfo][C-ERROR] | 固定 tag 的事件/终态/错误契约 |
| C-INTERRUPT / C-TOOLPOLICY | [TurnInterruptParams][C-INTERRUPT]、[extension ToolPolicy][C-TOOLPOLICY] | 固定 tag；后者不算公共 Python/RPC 开关 |
| C-LICENSE / C-PYMANIFEST / C-TSMANIFEST | [许可证][C-LICENSE]、[Python manifest][C-PYMANIFEST]、[TS manifest][C-TSMANIFEST] | 固定 tag；manifest 开发版不算包发行号 |
| C-ACP / C-ACPMANIFEST / C-ACPRELEASE | [adapter README][C-ACP]、[package][C-ACPMANIFEST]、[发布页][C-ACPRELEASE] | adapter v2.2.2；独立于 Codex/ACP wire 版本 |

### 8.3 Claude Code

公共产品/SDK 文档为当日滚动快照。Python README、类型、client、transport、LICENSE 固定 `v0.2.168`；TS 和 CLI 的许可证页是当日 main 快照，不能由 Python wrapper 的 MIT 推断 binary 的许可证。

| 来源 | 官方页面 | 对应版本或范围 |
|:---|:---|:---|
| L-RELEASE / L-TSRELEASE | [Python 发布][L-RELEASE]、[TS 发布][L-TSRELEASE] | Python 0.2.168 / TS 0.3.304，各自配套 CLI |
| L-OVERVIEW / L-README / L-PYTHON | [总览][L-OVERVIEW]、[Python README][L-README]、[Python API 文档][L-PYTHON] | 总览/API滚动；README 固定 tag |
| L-CLIENT / L-TYPES / L-TRANSPORT | [client][L-CLIENT]、[公共类型][L-TYPES]、[subprocess transport][L-TRANSPORT] | Python v0.2.168；生命周期/字段接线证据 |
| L-HEADLESS / L-CLIFLAGS / L-SETUP / L-HOSTING | [无头模式][L-HEADLESS]、[CLI flags][L-CLIFLAGS]、[安装][L-SETUP]、[托管进程模型][L-HOSTING] | 滚动文档；并发/平台/加载策略 |
| L-SESSIONS / L-INPUT / L-STREAM / L-LOOP | [会话][L-SESSIONS]、[输入流][L-INPUT]、[输出流][L-STREAM]、[Agent loop][L-LOOP] | 滚动文档 |
| L-PROMPT / L-MODEL / L-ENV / L-SETTINGS | [系统提示词][L-PROMPT]、[模型][L-MODEL]、[环境变量][L-ENV]、[设置][L-SETTINGS] | 滚动文档；最低 CLI 版本须另核 |
| L-PERMISSIONS / L-USERINPUT / L-MCP | [权限][L-PERMISSIONS]、[等待用户输入][L-USERINPUT]、[MCP][L-MCP] | 滚动文档 |
| L-SANDBOX / L-SECURE | [沙箱][L-SANDBOX]、[安全部署][L-SECURE] | 滚动文档；隔离范围非整个 harness |
| L-LICENSE / L-TSLICENSE / L-CLILICENSE | [Python MIT][L-LICENSE]、[TS license][L-TSLICENSE]、[CLI license][L-CLILICENSE] | Python固定tag；TS/CLI当日main |
| L-LEGAL / L-SUBSCRIPTION / L-RATELIMITS | [法律与凭据使用][L-LEGAL]、[订阅与 SDK][L-SUBSCRIPTION]、[API 限流][L-RATELIMITS] | 当日官方说明；具体部署与账户适用性另核 |
| L-ACP / L-ACPRELEASE / L-ACPPACKAGE / L-ACPSOURCE | [adapter README][L-ACP]、[发布][L-ACPRELEASE]、[package][L-ACPPACKAGE]、[公开 ACP 接线][L-ACPSOURCE] | 固定 adapter v0.94.0，非 Python v0.2.168 |

### 8.4 pi

以下 pi 本体文档与类型均固定 **v1.3.0**；adapter 另固定 **svkozak/pi-acp v0.0.34**，没有把其他同名 adapter、主线新增能力或旧包名混入。

| 来源 | 官方页面 | 对应版本或范围 |
|:---|:---|:---|
| P-RELEASE / P-PACKAGE / P-LICENSE / P-CHANGELOG | [发布][P-RELEASE]、[package][P-PACKAGE]、[MIT][P-LICENSE]、[changelog][P-CHANGELOG] | pi v1.3.0 |
| P-README / P-INTEGRATION / P-CLI / P-WINDOWS | [README][P-README]、[CLI integration][P-INTEGRATION]、[CLI][P-CLI]、[Windows/WSL][P-WINDOWS] | pi v1.3.0 |
| P-RPC / P-RPCTYPES / P-COMMANDS / P-UI | [RPC][P-RPC]、[RPC types][P-RPCTYPES]、[RPC commands][P-COMMANDS]、[extension UI][P-UI] | pi v1.3.0 |
| P-SDK / P-SDKOPTIONS / P-PROMPT | [SDK][P-SDK]、[session options][P-SDKOPTIONS]、[自定义 prompt 示例][P-PROMPT] | pi v1.3.0 |
| P-SESSIONS / P-FORMAT / P-MESSAGES | [会话][P-SESSIONS]、[会话格式][P-FORMAT]、[消息类型][P-MESSAGES] | pi v1.3.0 |
| P-JSON / P-SETTINGS | [JSON 事件][P-JSON]、[设置][P-SETTINGS] | pi v1.3.0 |
| P-SECURITY / P-EXTENSIONS / P-ENV / P-MCP | [安全][P-SECURITY]、[扩展][P-EXTENSIONS]、[环境变量][P-ENV]、[MCP][P-MCP] | pi v1.3.0；SDK 与 CLI 默认装配不同 |
| P-PROVIDERS / P-AI / P-AITYPES | [providers][P-PROVIDERS]、[pi-ai][P-AI]、[StreamOptions][P-AITYPES] | pi v1.3.0；底层选项不算 RPC setter |
| P-ACP / P-ACPRELEASE / P-ACPPACKAGE | [adapter README][P-ACP]、[发布][P-ACPRELEASE]、[package][P-ACPPACKAGE] | 社区 adapter v0.0.34；与 pi1.3.0组合未测 |

[A-version]: https://github.com/agentclientprotocol/agent-client-protocol#versioning
[A-init]: https://agentclientprotocol.com/protocol/v1/initialization
[A-transport]: https://agentclientprotocol.com/protocol/v1/transports
[A-schema]: https://agentclientprotocol.com/protocol/v1/schema
[A-session]: https://agentclientprotocol.com/protocol/v1/session-setup
[A-turn]: https://agentclientprotocol.com/protocol/v1/prompt-turn
[A-tools]: https://agentclientprotocol.com/protocol/v1/tool-calls
[A-config]: https://agentclientprotocol.com/protocol/v1/session-config-options
[A-cancel]: https://agentclientprotocol.com/protocol/v1/cancellation
[A-python]: https://agentclientprotocol.github.io/python-sdk/
[A-python-package]: https://raw.githubusercontent.com/agentclientprotocol/python-sdk/main/pyproject.toml
[A-release]: https://github.com/agentclientprotocol/agent-client-protocol/releases/tag/v1.11.0
[A-changelog]: https://github.com/agentclientprotocol/agent-client-protocol/blob/main/CHANGELOG.md

[C-ACP]: https://github.com/agentclientprotocol/codex-acp/blob/v2.2.2/README.md
[C-ACPMANIFEST]: https://github.com/agentclientprotocol/codex-acp/blob/v2.2.2/package.json
[C-ACPRELEASE]: https://github.com/agentclientprotocol/codex-acp/releases/tag/v2.2.2
[C-APP]: https://learn.chatgpt.com/docs/app-server
[C-AUTH]: https://learn.chatgpt.com/docs/auth
[C-CLI]: https://learn.chatgpt.com/docs/codex/cli
[C-CONFIG]: https://learn.chatgpt.com/docs/config-file/config-reference
[C-DELTA]: https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/app-server-protocol/schema/typescript/v2/AgentMessageDeltaNotification.ts
[C-ERROR]: https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/app-server-protocol/schema/typescript/v2/CodexErrorInfo.ts
[C-EXEC]: https://learn.chatgpt.com/docs/non-interactive-mode
[C-INJECT]: https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/app-server-protocol/schema/typescript/v2/ThreadInjectItemsParams.ts
[C-INTERRUPT]: https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/app-server-protocol/schema/typescript/v2/TurnInterruptParams.ts
[C-ITEM]: https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/app-server-protocol/schema/typescript/v2/ThreadItem.ts
[C-LICENSE]: https://github.com/openai/codex/blob/rust-v0.163.0/LICENSE
[C-MCP]: https://learn.chatgpt.com/docs/extend/mcp
[C-PY]: https://github.com/openai/codex/blob/rust-v0.163.0/sdk/python/docs/api-reference.md
[C-PYMANIFEST]: https://github.com/openai/codex/blob/rust-v0.163.0/sdk/python/pyproject.toml
[C-RELEASE]: https://github.com/openai/codex/releases/tag/rust-v0.163.0
[C-RESUME]: https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/app-server-protocol/schema/typescript/v2/ThreadResumeParams.ts
[C-SDK]: https://learn.chatgpt.com/docs/codex-sdk
[C-START]: https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/app-server-protocol/schema/typescript/v2/ThreadStartParams.ts
[C-STARTRESP]: https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/app-server-protocol/schema/typescript/v2/ThreadStartResponse.ts
[C-TOOLPOLICY]: https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/ext/extension-api/src/tool_policy.rs
[C-TS]: https://github.com/openai/codex/blob/rust-v0.163.0/sdk/typescript/README.md
[C-TSMANIFEST]: https://github.com/openai/codex/blob/rust-v0.163.0/sdk/typescript/package.json
[C-TURN]: https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/app-server-protocol/schema/typescript/v2/Turn.ts
[C-TURNSTART]: https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/app-server-protocol/schema/typescript/v2/TurnStartParams.ts
[C-WSL]: https://learn.chatgpt.com/docs/windows/wsl
[L-ACP]: https://github.com/agentclientprotocol/claude-agent-acp/blob/v0.94.0/README.md
[L-ACPPACKAGE]: https://github.com/agentclientprotocol/claude-agent-acp/blob/v0.94.0/package.json
[L-ACPRELEASE]: https://github.com/agentclientprotocol/claude-agent-acp/releases/tag/v0.94.0
[L-ACPSOURCE]: https://github.com/agentclientprotocol/claude-agent-acp/blob/v0.94.0/src/acp-agent.ts
[L-CLIENT]: https://github.com/anthropics/claude-agent-sdk-python/blob/v0.2.168/src/claude_agent_sdk/client.py
[L-CLIFLAGS]: https://code.claude.com/docs/en/cli-reference
[L-CLILICENSE]: https://github.com/anthropics/claude-code/blob/main/LICENSE.md
[L-ENV]: https://code.claude.com/docs/en/env-vars
[L-HEADLESS]: https://code.claude.com/docs/en/headless
[L-HOSTING]: https://code.claude.com/docs/en/agent-sdk/hosting
[L-INPUT]: https://code.claude.com/docs/en/agent-sdk/streaming-vs-single-mode
[L-LEGAL]: https://code.claude.com/docs/en/legal-and-compliance
[L-LICENSE]: https://github.com/anthropics/claude-agent-sdk-python/blob/v0.2.168/LICENSE
[L-LOOP]: https://code.claude.com/docs/en/agent-sdk/agent-loop
[L-MCP]: https://code.claude.com/docs/en/agent-sdk/mcp
[L-MODEL]: https://code.claude.com/docs/en/model-config
[L-OVERVIEW]: https://code.claude.com/docs/en/agent-sdk/overview
[L-PERMISSIONS]: https://code.claude.com/docs/en/agent-sdk/permissions
[L-PROMPT]: https://code.claude.com/docs/en/agent-sdk/modifying-system-prompts
[L-PYTHON]: https://code.claude.com/docs/en/agent-sdk/python
[L-RATELIMITS]: https://platform.claude.com/docs/en/api/rate-limits
[L-README]: https://github.com/anthropics/claude-agent-sdk-python/blob/v0.2.168/README.md
[L-RELEASE]: https://github.com/anthropics/claude-agent-sdk-python/releases/tag/v0.2.168
[L-SANDBOX]: https://code.claude.com/docs/en/sandboxing
[L-SECURE]: https://code.claude.com/docs/en/agent-sdk/secure-deployment
[L-SESSIONS]: https://code.claude.com/docs/en/agent-sdk/sessions
[L-SETTINGS]: https://code.claude.com/docs/en/settings
[L-SETUP]: https://code.claude.com/docs/en/setup
[L-STREAM]: https://code.claude.com/docs/en/agent-sdk/streaming-output
[L-SUBSCRIPTION]: https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan
[L-TRANSPORT]: https://github.com/anthropics/claude-agent-sdk-python/blob/v0.2.168/src/claude_agent_sdk/_internal/transport/subprocess_cli.py
[L-TSLICENSE]: https://github.com/anthropics/claude-agent-sdk-typescript/blob/main/LICENSE.md
[L-TSRELEASE]: https://github.com/anthropics/claude-agent-sdk-typescript/releases/tag/v0.3.304
[L-TYPES]: https://github.com/anthropics/claude-agent-sdk-python/blob/v0.2.168/src/claude_agent_sdk/types.py
[L-USERINPUT]: https://code.claude.com/docs/en/agent-sdk/user-input
[P-ACP]: https://github.com/svkozak/pi-acp/blob/v0.0.34/README.md
[P-ACPPACKAGE]: https://github.com/svkozak/pi-acp/blob/v0.0.34/package.json
[P-ACPRELEASE]: https://github.com/svkozak/pi-acp/releases/tag/v0.0.34
[P-AI]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/ai/README.md
[P-AITYPES]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/ai/src/types.ts
[P-CHANGELOG]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/CHANGELOG.md
[P-CLI]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/cli.md
[P-COMMANDS]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/rpc-commands.md
[P-ENV]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/environment-variables.md
[P-EXTENSIONS]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/extensions.md
[P-FORMAT]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/session-format.md
[P-INTEGRATION]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/cli-integration.md
[P-JSON]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/json.md
[P-LICENSE]: https://github.com/earendil-works/pi/blob/v1.3.0/LICENSE
[P-MCP]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/mcp.md
[P-MESSAGES]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/message-types.md
[P-PACKAGE]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/package.json
[P-PROMPT]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/examples/sdk/03-custom-prompt.ts
[P-PROVIDERS]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/providers.md
[P-README]: https://github.com/earendil-works/pi/blob/v1.3.0/README.md
[P-RELEASE]: https://github.com/earendil-works/pi/releases/tag/v1.3.0
[P-RPC]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/rpc.md
[P-RPCTYPES]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/src/modes/rpc/rpc-types.ts
[P-SDK]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/sdk.md
[P-SDKOPTIONS]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/src/core/sdk.ts
[P-SECURITY]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/security.md
[P-SESSIONS]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/sessions.md
[P-SETTINGS]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/settings.md
[P-UI]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/rpc-extension-ui.md
[P-WINDOWS]: https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/windows.md
