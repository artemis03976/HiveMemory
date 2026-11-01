---
title: DeepSeek Harness 插件接入调研需求与报告
status: baseline
owner: project
scope: deepseek-harness-plugin-mode-integration-survey
snapshot_at: 2026-11-01
source_commit: 23288e33d69f394a1882525b2a7ff23baa6aeb4f
dsh_source_commit: ce7073d51254c26cd63b22a60835b8f948deb447
dsh_version: 0.2.2-alpha.2
code_paths:
  - integrations/claude-code/
  - src/hivememory/mcp_server/
  - src/hivememory/system/application/
related_docs:
  - docs/system/import-bus.md
  - docs/contracts/mcp.md
  - docs/system/configuration.md
  - docs/research/harness-interface-survey.md
  - docs/research/harness-interface-survey-snapshot.md
  - docs/ROADMAP.md
last_reviewed: 2026-11-01
---

# DeepSeek Harness 插件接入调研需求与报告

## 0. 文档性质

本文保留原调研需求（第 1–6、8 节），并按 owner 本次要求将调研报告直接写回第 7 节。报告冻结已取得的外部证据、适配分析与未知项，不代表接入路径已经决定，也不证明 HiveMemory 已完成 dsh 接入。

外部项目变化很快，调研结论只代表完成时的日期。DeepSeek Harness 目前是开发者预览版，形成报告快照时在 frontmatter 记录 `snapshot_at` 与所核对的固定提交。

## 1. 目的

owner 决定（2026-10-31）：以 DeepSeek Harness（`dsh`）作为第二个接入的外部 harness，采用 plugin 模式（对话回流与经工具的记忆访问），作为版本之外的一项工作，在 v0.7.3 之前完成。选择它的理由：

- Claude Code 有自己的原生记忆，HiveMemory 作为插件接入后难以显出作用，只能积累开发过程的记忆；
- dsh 对插件友好，而且轻量。

接入之后，owner 将经 dsh 进行 agent memory 与 harness 项目的广泛调研，同时作为 HiveMemory 的压测、记忆与交互轨迹的积累。owner 的方向（2026-10-31）是：记忆的写入与访问主要由执行者经记忆工具主动完成，因为只有执行者知道什么是一手且重要的；话题结算的被动提取作为兜底。owner 之后将为没有原生记忆的 harness 编写一份 skill，引导 agent 主动使用记忆工具。

这次调研为接入计划提供依据：

- 接入路径：沿用 Claude Code 的做法（会话记录读取器加 MCP），还是写成 dsh 的原生插件，或两者组合；
- HiveMemory 一侧需要新增或改动什么；
- 引导 skill 在 dsh 中以什么方式承载。

### 1.1 HiveMemory 一侧已有的接入面

判断 dsh 的接口是否够用时，以下列现有事实为准：

- **对话回流**：`POST /api/v1/import/turns`，每个请求是一轮已经结束的交互：用户消息、按顺序的事件（助手正文、思考、工具调用及其结局）、结局（completed / cancelled / failed）、模型名，以及外部会话与外部轮次标识。服务端按推导出的交互标识去重（[Import Bus](../system/import-bus.md)第 3、4 节）。
- **连接器**：游标、确认后前进、重发去重、首次见到的旧会话不回填等逻辑与 harness 无关，位于连接器的公共骨架；每个 harness 只需要一个读取器（[Import Bus](../system/import-bus.md#81-组成)第 8.1 节）。现有连接器只用 Python 标准库，由 harness 所在环境的 `python3` 运行。
- **记忆访问**：HTTP 的 MCP 端点 `/mcp`，请求头声明 principal、接入凭据、用户与 Agent；工具为 `memory_search`、`memory_read`、`memory_write`、`memory_update`（[MCP 记忆访问](../contracts/mcp.md)第 1–3 节）。
- **写入时落位**：写入与更新必须带参数 `turn`（这一轮的外部标识、用户消息与之前最多三轮），否则以 `request.turn_required` 拒绝。Claude Code 由 `PreToolUse` hook 在调用前改写工具参数补上它（[MCP 记忆访问](../contracts/mcp.md#6-连接器调用名轮次信息与回声标记)第 6 节）。
- **回声标记**：本插件的记忆工具调用随回流导入时归一化为 `SEARCH`、`READ`、`WRITE`、`UPDATE` 能力调用，避免已有记忆被再提取一遍；这需要读取器能认出这些调用。
- **接入登记**：每个外部 harness 有自己的 principal（Claude Code 为 `harness:claude-code`）与凭据环境变量，登记在 `system_principals.yaml`（[配置](../system/configuration.md)）。

Claude Code 连接器用到的 hook 是 `Stop`、`StopFailure`、`SessionStart`、`SessionEnd` 与 `PreToolUse`（[Import Bus](../system/import-bus.md#82-hooks)第 8.2 节），第 5 节要求逐项给出 dsh 中的对应物。

## 2. 调研对象

- **必选**：DeepSeek Harness，官方仓库 [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)，npm 包 `@deepseek-ai/dsh`。2026-10-31 访问仓库首页时：MIT 许可，标为开发者预览并警告会有不兼容变更，Releases 为空；首页没有说明插件 API、hook、MCP、会话存储与内置记忆，这些都是本次要回答的问题。调研时以固定提交为准。
- **可选**：社区已有的 dsh 记忆类插件。只用于了解其他记忆系统怎样接入 dsh，以及 dsh 用户已经在用什么，不作为接入候选。社区目录与 star 数等信息只作线索，结论以官方仓库与实际运行为准。

## 3. 调研问题

每个问题注明服务的方面：**路径**（选择接入路径）、**回流**、**访问**（经工具的记忆访问）、**落位**（写入时落位）、**引导**（skill 的承载）、**使用**（经 dsh 进行调研时的实际使用）。

### A. 插件机制

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| A1 | 插件以什么语言与形式编写（Cordis 的插件、服务与上下文概念如何对应）？怎样注册、加载与卸载？ | 路径 |
| A2 | 插件能否注册供模型调用的工具？工具的名称、说明与参数 schema 怎样声明，模型看到的调用名是什么？ | 路径、访问、引导 |
| A3 | 插件能否访问网络（调用本机 HTTP 服务）、读写文件、启动子进程？有没有沙箱或权限限制？ | 路径、回流 |
| A4 | 插件 API 的稳定性：是否有版本号或兼容承诺，开发者预览期间变更的频率如何？ | 路径 |
| A5 | 插件怎样安装与分发：本地路径、`dsh plugin add`、插件市场？能否只在本机安装、不发布？ | 路径 |

### B. 事件与 hook

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| B1 | 插件能订阅哪些事件：会话开始与结束、一轮开始与结束、工具调用前后、中断与出错？与 Claude Code 的五个 hook 怎样对应？ | 路径、回流 |
| B2 | 工具调用前的事件能否改写调用参数（对应 `PreToolUse` 的 `updatedInput`），改写后的参数是否还经过权限确认？ | 落位 |
| B3 | 插件能否实时取得一轮的完整内容：用户消息、助手正文、思考、工具调用与结果？顺序与完整性如何保证（并行调用、流式中断）？ | 回流 |
| B4 | 一轮的结束能否可靠判定，结局（完成、取消、出错、超出上下文）怎样表达？进程崩溃时会丢失什么？ | 回流 |
| B5 | 子 agent 或并行任务怎样表示：是否有单独的会话，其中的工具调用归属于哪一轮？ | 回流、落位 |

### C. 会话记录

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| C1 | 会话保存在哪里、什么格式？是否有稳定的会话标识与轮次标识？格式是否是公开承诺的接口？ | 路径、回流 |
| C2 | 恢复、分支、回退与上下文压缩对会话记录有什么影响（是否重写或截断已写入的内容）？ | 回流 |
| C3 | 思考内容是否保存正文？工具结果是否完整保存？ | 回流 |

### D. MCP

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| D1 | 是否支持 MCP 客户端？支持哪些传输（stdio、HTTP、SSE），能否给 HTTP 请求带自定义请求头与环境变量展开？ | 路径、访问 |
| D2 | MCP server 的配置作用范围：插件声明、项目级、全局？连接失败时的行为？ | 访问 |
| D3 | 调用 MCP 工具是否经过权限确认？MCP 工具的调用名格式是什么？ | 访问、落位 |
| D4 | MCP server 提供的 instructions 是否进入模型上下文？ | 引导 |

### E. 指令与 skill

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| E1 | 有没有 skill 或同类机制（按需加载的说明文档）？怎样发现、触发与加载？ | 引导 |
| E2 | 插件能否向系统提示词或每轮上下文追加说明？是否读取项目中的 `AGENTS.md` 一类文件？ | 引导 |
| E3 | 上述说明有没有长度限制，是否在上下文压缩后保留？ | 引导 |

### F. 原生记忆与上下文

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| F1 | dsh 是否内置记忆或持久笔记（跨会话保存的事实、偏好、项目说明）？如有，与外部记忆工具是否冲突？ | 使用、引导 |
| F2 | 上下文压缩何时发生、是否可以观察，压缩后的摘要是否写入会话记录？ | 回流、使用 |
| F3 | 社区已有的记忆插件怎样接入（工具、事件、存储），有哪些可以借鉴或需要避开的做法？ | 路径 |

### G. 运行、模型与许可

| 编号 | 问题 | 服务 |
|:---|:---|:---|
| G1 | 运行环境要求（Node 版本、操作系统），在 Linux 与 WSL 上能否正常运行？启动方式有哪些（Web UI、终端、无头）？ | 使用 |
| G2 | 模型提供方怎样配置：只支持 DeepSeek，还是可以接其他提供方？凭据放在哪里？ | 使用 |
| G3 | 能否无头或批量运行（用于长时间的调研任务）？ | 使用 |
| G4 | 插件与 harness 本身的许可、使用条款中有没有限制？ | 路径 |

## 4. 取证要求

- 每项结论注明来源（官方文档、源码、发布说明）的链接、固定提交与访问日期。开发者预览阶段的文档可能落后于源码，文档与源码不一致时两者都记录，以源码为准。
- 区分“文档声明”“源码核对”“实际运行”三种验证程度。
- 最小实测：在仓库之外的临时目录安装 dsh，写一个只记录事件并注册一个工具的插件，连接一个 MCP server，回答 A2、B1、B2、B3、D1。实测使用 owner 确认的模型凭据，不把凭据、日志或产物提交到仓库。
- 无法确定的项标为“未知”并说明原因，不作推测。

## 5. 交付物

1. **能力清单**：第 3 节的每个问题，取值为“支持 / 部分支持 / 不支持 / 未知”，附来源与验证程度。
2. **hook 对应表**：Claude Code 连接器用到的五个 hook 与 `PreToolUse` 改写参数的能力，在 dsh 中的对应物或缺口。
3. **接入路径比较**：至少比较“会话记录读取器加 MCP”（与 Claude Code 相同）和“dsh 原生插件”两条路径；每条写明复用 HiveMemory 的哪些部分、需要新增什么、落位与回声标记怎样满足、主要风险。只列影响，不作选择。
4. **引导 skill 的承载方式**：dsh 中可以承载使用说明的机制及其限制。

结果按本次要求直接写入本文第 7 节，在目录索引登记同一文档及其验证状态，不另建重复报告。

## 6. 不在范围

- 编写插件、连接器或任何生产代码；
- 决定接入路径或修改 HiveMemory 的接入契约；
- 引导 skill 的内容；
- dsh 的模型质量、回复效果或性能对比；
- controller 模式（由 HiveMemory 驱动 dsh 执行任务）的问题，只在发现明显相关的接口时顺带记录。

## 7. 调研结果

### 7.1 结论与证据边界

**dsh 提供原生插件、MCP 客户端、完整逻辑会话事件、skill 与无头运行能力，但不能原样复制 Claude Code 插件接入。** 关键缺口是：工具前置事件不允许改写参数，无法用它给通用 MCP WRITE/UPDATE 自动补充可信 `turn`；默认会话文件采用 Zstandard 压缩，现有按行读取骨架也不能直接消费。完整接入需要 harness 侧适配；本次核对未发现必须修改 HiveMemory 四工具或 Import Bus 业务契约的依据。路径比较见第 7.4 节，要求见第 7.6 节。

证据快照与取证方法：

| 项目 | 本次边界 |
|:---|:---|
| 访问与证据日期 | 2026-11-01；下文所有外部来源均在该日期核对。 |
| 官方源码 | [deepseek-ai/deepseek-harness 固定提交][dsh-commit]：`ce7073d51254c26cd63b22a60835b8f948deb447`；提交日期 2026-10-30。下文源码及仓库文档链接均固定到这个提交，不以浮动 `master` 作为证据。 |
| 版本 | 源码与实测包均为 `0.2.2-alpha.2`；实测 Cordis 为 `4.0.6-alpha.1`。npm 查询时 `latest` 指向 `0.2.1-rc.5`，所以安装时显式固定 alpha 版本，不能将 `latest` 当作源码快照。 |
| 实测包校验 | [`@deepseek-ai/dsh@0.2.2-alpha.2` 官方 registry 元数据][dsh-npm]；tarball SHA-1 `c149de1694cc156a79765faca2e90af8f237c40a`。版本名相同不证明包逐字对应此 Git 提交，因此以下分别标记源码核对与包实测。 |
| HiveMemory 证据 | 本地代码基线 `23288e33d69f394a1882525b2a7ff23baa6aeb4f`，结合现有 [MCP 契约](../contracts/mcp.md)、[Import Bus](../system/import-bus.md)和实现核对；保留任务开始前的未提交文档修改。 |
| 检索方法 | 先通过 Context7 解析 `/deepseek-ai/deepseek-harness`，分别检索插件/事件、MCP、持久化；再读取官方仓库固定源码及相邻文档。Context7 用于发现入口，结论以固定来源及实测为准。 |
| 验证标记 | **文档**＝官方明确声明；**源码**＝核对相应实现；**实测 P1**＝第 7.7 节本地脚本化模型探针。没有标记实测的能力不视为运行验证。 |

调研完成的是资料核对及无真实凭据的机制探针。第 4 节要求的 owner 确认模型凭据实测尚未完成；没有运行真实模型、真实 HiveMemory MCP/Import Bus 或 Qdrant，也没有比较模型质量。

### 7.2 能力清单

#### A. 插件机制

| 编号 | 判定 | 结论、来源与验证程度 |
|:---|:---|:---|
| A1 | 支持 | JS/TS Cordis 插件可为函数、带 `apply(ctx)` 的模块/对象或 Service 类；`inject` 声明服务依赖，Context 管理作用域与 effect 生命周期。加载配置/`ctx.plugin` 启动，卸载清理由该插件登记的工具、监听器与 effects；自建连接和任务仍须登记清理。[插件开发][plugin-basic]、[Cordis 生命周期][cordis-life]；文档＋源码，P1 验证程序化加载与清理。 |
| A2 | 支持 | `ctx.tools.register(defineTool({...}))` 声明 `name`、`description`、`parameters`、输出 schema/渲染和执行体。参数转 JSON Schema，并在执行体入口校验；原生模式模型看到登记名，例如 P1 的 `survey_echo`，没有自动 MCP 前缀。PTC 模式改以 `run_code` 调度底层能力，不能把两种表面混为一谈。[工具开发][tool-guide]、[schema 实现][tool-schema]、[工具执行器][native-executor]；文档＋源码＋P1。 |
| A3 | 支持 | Host 插件可使用 Node 网络、文件和子进程；MCP transport 本身也直接创建子进程/HTTP 连接。工具执行的 sandbox/approval 不是第三方插件代码的隔离沙箱。[安全说明][dsh-safety]、[transport 源码][mcp-transport]；文档＋源码，P1 验证本机 HTTP；文件/子进程能力未单独实测。 |
| A4 | 部分支持 | 有包版本、bundle `engines.dsh`/peer dependency 兼容范围检查与升级指南；官方仍明确预告开发者预览期间的不兼容变更，没有长期插件 API 兼容承诺。本次浅克隆不能量化历史变更频率，频率记为未知，不由 star 数或预览标签推算。[README][dsh-readme]、[分发与兼容范围][plugin-publish]、[升级指南目录][upgrade-guides]；文档＋源码。 |
| A5 | 支持 | 开发可用 `--patch` 配置引用本地插件路径；bundle 以 `dsh.bundle.patch` 声明组合入口。支持本地目录、tarball、npm、Git，例 `dsh plugin --profile demo add ./hello-plugin`，移除用 `remove`，无须公开发布。插件目录 Discover 是可选实验 bundle，官方不内置索引源。[分发说明][plugin-publish]、[插件目录][plugin-market]；文档＋源码，P1 未验证 CLI bundle 安装。 |

#### B. 事件与 hook

| 编号 | 判定 | 结论、来源与验证程度 |
|:---|:---|:---|
| B1 | 支持 | 会话有 `session/created`、`session/disposed`、提交后的 `session/event`；Agent 有 created/disposed/status、pre-step、request-error、assistant-stream、turn-stopping/error；工具有 pre-execute/execute/post-execute/result。持久轮次边界在 `session/event` 的 `turn/start`、`turn/end`，生命周期事件不等于一轮结束。[Session 事件][session-impl]、[Agent 事件][agent-runtime]、[工具事件][tool-events]；文档＋源码，P1 只验证正常轮次及工具前置事件子集。 |
| B2 | 不支持 | **不支持 Claude `updatedInput` 式参数改写。** `PreToolDecision` 只有 allow/deny/cancel/ask；参数在 policy 前被 snapshot 并深冻结。权限/ask/final guards 后才进入工具体及其 schema 校验；`tools/execute` 包装器只可调整 signal，也不是改参后重新审批入口。[决策契约][tool-contracts]、[执行流水线][tool-pipeline]；源码＋P1（两种工具均改写失败并执行原参数）。 |
| B3 | 部分支持 | 可取得原始用户输入、助手正文、provider 返回的思考、工具调用与结果，已提交事件有顺序 `seq`；实时 delta 另走 assistant-stream。并行工具只并发 body，pre/post 和结果按模型顺序提交。一 turn 可有多个 user/message，包括 steering 和合成上下文，不能见到 user-role 就切新轮。`session/event` 是提交后的 fire-and-forget 通知，监听器失败不回滚记录，必须另有补扫/确认。[事件模型][session-types]、[Session 通知][session-impl]、[批执行][tool-batch]；源码＋P1 的普通三 step 轮次，未实测并行或中断。 |
| B4 | 部分支持 | `turn/end.data.reason.kind` 包含 completed/aborted/blocked/error/max-tokens/interrupted/forked；aborted 进一步区分 user/parent/hook/disposed，历史迁移还可能保留 legacy 原因。max-tokens 是输出上限，不能当上下文超限；上下文压力先尝试压缩恢复。turn/end 不等待持久化 flush；硬崩溃可能丢未 settlement 的 delta 和未刷盘尾部。恢复可追加 interrupted 等合成收尾，不伪造 completed。[终态模型][session-types]、[Agent 收尾][loop-agent]、[恢复修复][session-repair]；源码＋P1 的 completed，其他终态未实测。 |
| B5 | 支持 | 进程内 spawn 子 Agent 有独立会话，进程内 fork 子 Agent 继承父会话完整轮次前缀后独立运行。header 保存 parentSession/origin/delegationDepth，工具 context 带调用 Agent；原始轨迹中父 delegation 工具和子会话工具各有自己的轮次。fork 继承前缀须按 inheritedEventCount 排除，避免重复回流。这些日志结论不泛化到外部 harness 子 Agent backend。[spawn][subagent-spawn]、[fork 子 Agent][subagent-fork]、[会话 header][session-types]；文档＋源码，未实测子 Agent。 |

#### C. 会话记录

| 编号 | 判定 | 结论、来源与验证程度 |
|:---|:---|:---|
| C1 | 部分支持 | 逻辑记录是 append-only SessionEvent，连续 seq、稳定 session header id、会话内数字 turn；当前格式版本为 4。默认 base root 是 `$DSH_HOME/sessions`，路径按 project/session 分目录，文件形如 `session.v4.jsonl.zstd`；显式 `compression: none` 才是 `.jsonl` 文本。官方建议通过 persistence handle 读取压缩/迁移后的逻辑流；公开格式有版本，不等于预览阶段跨版本文件接口承诺。[base 配置][base-config]、[持久化说明][persistence-jsonl]、[路径实现][persistence-format]；文档＋源码，P1 只验证进程内事件标识。 |
| C2 | 部分支持 | resume 沿同一 session 追加；fork 新建 session、复制前缀并标记 inherited cut，原 log 不变；压缩追加 summary/surface replacement，保留早期事件。未发现用户级原地 rewind 契约，不能将 fork 称为原地回退。格式升级另写 generation、保留历史文件；崩溃损坏的物理尾部可在恢复写入前截断，不应误认成正常历史回退。[fork 命令][fork-command]、[fork 构造][session-fork]、[压缩写入][compaction-region]、[持久化迁移][persistence-jsonl]；文档＋源码，未实测恢复/分支/压缩。 |
| C3 | 部分支持 | 保存 provider 实际返回的 text/reasoning 及精确 stream，取消可保存已交付前缀；没有模型返回的隐藏思考当然无法取得。工具记录是最终 model-facing result/可选 meta，不保证任意工具的无限原始输出或 canonical value。应读原始逻辑事件，不用压缩后的 derived surface 替代原轨迹。[事件数据][session-types]、[助手 settlement][loop-agent]、[工具输出契约][tool-contracts]；源码＋P1 的正文、思考和两个工具结果。 |

#### D. MCP

| 编号 | 判定 | 结论、来源与验证程度 |
|:---|:---|:---|
| D1 | 部分支持 | 支持 stdio、Streamable HTTP，后者允许 `headers`；Loader 配置用 `!!js process.env.X` 或模板表达式取环境变量，不照搬 Claude `${VAR}` 配置。未提供独立 legacy HTTP+SSE transport；Streamable HTTP 内的 SSE 不等于旧 SSE endpoint。[客户端配置][mcp-guide]、[transport][mcp-transport]；文档＋源码，P1 验证 Streamable HTTP JSON 响应及自定义请求头，未实测 stdio/环境展开。 |
| D2 | 支持 | 每个 server 是一条 Cordis MCP client 配置，可由 bundle、home patch、profile patch、`--patch` 与 Agent scope 组合；不自动读取 Claude `.mcp.json`。默认未配置 server。初连失败默认记录错误且不提供该 server 工具；`failOnStartupError: true` 拒绝插件激活，但 optional entry 仍可能容许整个 harness 启动。断线有重连及预算。[MCP 说明][mcp-guide]、[分发配置层][plugin-publish]；文档＋源码，P1 验证程序化连接成功，未验证断线。 |
| D3 | 支持 | MCP 工具经过普通工具 policy/approval，是否询问取决于部署策略，不是每次调用必弹确认；ask 无 approval service/可用 Agent 时拒绝。公开名为 `mcp__<serverName>__<tool>`，非法/过长名称另规范化加 hash；wire 仍用原始工具名。[工具名与桥接][mcp-tools]、[权限流水线][tool-pipeline]；源码＋P1 验证前置 gate、公开名与 wire `echo`，未验证交互确认。 |
| D4 | 支持 | server instructions 以 `mcp:<server>` section 注入 system prompt，字面保留、不开插值；默认 attributed 文本限 32768 UTF-8 bytes，超限拒绝连接。它与 MCP prompt templates 是两种机制。[section 发布][mcp-context]、[连接与长度限制][mcp-connection]；文档＋源码＋P1（三次模型请求均含模拟 server instruction）。 |

#### E. 指令与 skill

| 编号 | 判定 | 结论、来源与验证程度 |
|:---|:---|:---|
| E1 | 支持 | 有 skill registry、filesystem provider、`skill` 工具；可放 `<root>/<name>/SKILL.md` 或 root 平铺 `<name>.md`，需 name/description。依次发现项目 `.dsh/skills`、`.agents/skills`、custom、用户 `$DSH_HOME/skills`/`$DSH_AGENTS_HOME/skills` 与 bundled；只扫一层。模型按目录摘要主动装载，用户显式 `/name` 可由 pre-step 解析。frontmatter 可分别控制模型/用户调用。[发现说明][skill-filesystem]、[装载实现][skill-tool]；文档＋源码，未实测。 |
| E2 | 支持 | 插件可注册 system prompt section、动态 runtime context 或在 pre-step 增加输入；默认 instructions provider 读取全局及项目 `AGENTS.md`/`CLAUDE.md` 和 local 变体，按 root→cwd 链及后续触达目录提供说明。[提示词机制][system-prompt]、[instructions provider][agent-instructions]；文档＋源码；P1 只验证 MCP section，未验证文件发现。 |
| E3 | 部分支持 | 载体有不同预算：skill 目录 description 默认 500 字符，正文没有统一长度上限但作为普通历史结果可被压缩；AGENTS 每条 global/project chain 使用独立 maxBytes，base 配置 65536 bytes，单源默认 1 MiB、过大可忽略；MCP instructions 另有 32 KiB 限制。持续 system section 每 step 重组，runtime context 可重建，但不承诺 skill 正文压缩后自动重新装载。[skill 工具][skill-tool]、[instructions 限制][instructions-config]、[base 配置][base-config]、[system prompt][system-prompt]；文档＋源码，未实测压缩后引导。 |

#### F. 原生记忆与上下文

| 编号 | 判定 | 结论、来源与验证程度 |
|:---|:---|:---|
| F1 | 部分支持 | 默认组合有持久会话、AGENTS/skills 和压缩摘要，未发现默认跨会话事实/偏好记忆服务。官方三个外部 memory MCP 参考配置默认关闭。与 HiveMemory 无固定工具命名冲突；若并开其他 memory server，重复存储和使用哪套记忆需部署/引导约定，dsh 不自动协调。[第三方记忆指南][memory-guide]、[base 配置][base-config]；文档＋源码，未做跨会话召回实测。 |
| F2 | 支持 | compaction-basic 支持 pre-step 压力触发、模型容量检查、上下文超限后的恢复与 `/compact`；可先减重工具结果。Session 记录 compaction/start、summary、end 和 surface replacement，保留原事件；replacement user/message 不是用户新发言。[压缩说明][compaction-guide]、[summary 写入][compaction-region]；文档＋源码，未实测。 |
| F3 | 部分支持 | 官方已收录 Memorix、MCP Reference Memory、Engram 三种默认关闭的 MCP 互操作参考，标注各自版本与上游提交；均以工具接入，数据库、模型/embedding、许可归上游，dsh 不初始化或迁移其存储，也不监督外部 HTTP 服务。可借鉴配置隔离与“新会话写入→召回→使用”验证；不能据此推断已支持整轮回流或可信 turn 注入。本次未独立核验社区插件实现、活跃度或上游运行效果。[官方参考与上游固定版本][memory-guide]；文档声明＋dsh 配置源码，非上游实测。 |

#### G. 运行、模型与许可

| 编号 | 判定 | 结论、来源与验证程度 |
|:---|:---|:---|
| G1 | 支持 | 此快照要求 Node **>=24.9.0**，源码 pnpm 固定 11.28.5；CLI/Web/SDK 等支持 Linux/Windows/macOS，官方明确说明 WSL 2 工作流，同一环境安装依赖、WSL 项目放 Linux 文件系统。入口有 `web`、`headless`、`sdk`/`sdk-minimal`、`acp`；Desktop 发行目标仅 Windows/macOS。不能因帮助示例出现 `tui` 就断言默认发行带完整交互终端 UI，终端自动化本次确认的是 headless。[包与 engine][dsh-package]、[开发环境][dsh-development]、[CLI 帮助][cli-help]、[桌面发行目标][desktop-packaging]；文档＋源码，P1 在 Linux 用 Node 24.9.0 运行，未实测 WSL/Windows/macOS/Web UI。 |
| G2 | 支持 | 不限 DeepSeek；内置 provider catalog 与自定义 API 支持 OpenAI Chat Completions/Responses、Anthropic Messages 等适配。Web Settings→Models 管理 provider/model，密钥存 `$DSH_HOME/.credentials.yaml`、设置只存引用；也支持环境凭据。headless 默认关闭 GUI managed credential store，按启动环境→cwd `.env`→`$DSH_HOME/.env` 取 API key，不能以为 Web 保存的 key 默认适用于 headless。[模型配置][provider-guide]、[headless 凭据规则][headless-guide]；文档＋源码，P1 只使用进程内脚本化 adapter，没有访问真实凭据。 |
| G3 | 支持 | `dsh headless "任务"` 或 stdin 单任务运行，`--json` 输出 NDJSON，`--session-id` 续会话；退出码区分完成和失败。没有交互 follow-up/实时审批回应通道，长期批任务需调用方管理进程、续跑与权限。headless NDJSON 对非 final 字符串/事件有截断，不能当完整回流源。[headless 契约][headless-guide]、[CLI][cli-help]；文档＋源码，P1 使用同类底层 AgentLoop，未跑 CLI 批任务。 |
| G4 | 支持 | harness 及核对的 Cordis/dsh 插件包使用 MIT，可本地编写、分发插件；再分发需保留相应版权与许可声明，并逐项核对第三方依赖许可。未发现强制公开插件或独家记忆后端条款。模型 API/账号、第三方服务另受其条款约束，本次未审阅这些服务条款，MIT 不代表服务使用许可。[MIT][dsh-license]、[CLI 包许可][dsh-package]、[Cordis 包许可][cordis-package]、[第三方 notices][third-party-notices]；许可文本核对，无额外法律结论。 |

### 7.3 Claude Code hook 对应表

| 现有 hook/职责 | dsh 对应物 | 接入影响 |
|:---|:---|:---|
| Stop：触发已结束轮次回流 | `session/event` 的 `turn/end` | 用已提交终态切轮；`agent/turn-stopping` 尚可被 steering 继续，不能当 Stop。reason 需投影到 HiveMemory 三种 outcome。 |
| StopFailure：异常/失败后的回流 | error/aborted 等 `turn/end`，辅以 `agent/error`/`agent/request-error` | request-error 可能触发重试，不能每个错误都提前封口。先等整轮终态，恢复 interrupted 另处理。 |
| SessionStart：恢复时补扫 | `agent/created` 的 fresh/resume 来源；`session/created` | 恢复与新会话区分，补扫尚未确认的 closed turns；新建一个 live Session 本身不证明有新用户轮次。 |
| SessionEnd：最后补扫/清理 | `agent/disposed`、`session/disposed` 和插件 effect 清理 | 正常退出可以触发收尾，硬崩溃不能保证触发；持久记录/下次恢复补扫不可省略。 |
| PreToolUse：权限与参数补全 | 权限对应 `tools/pre-execute`；**updatedInput 无对应能力** | 自有工具的 execute 可以另构造后端请求，不能篡改已审批模型参数。没有 dsh 可信上下文的外置 MCP bridge 也不能自动知道当前轮。 |

来源：[Agent 事件][agent-runtime]、[Session 事件][session-impl]、[轮次模型][session-types]、[工具决策契约][tool-contracts]。表中是职责对应，不表示 Claude hook 的 stdin/stdout JSON 协议或原连接器脚本可直接在 dsh 运行。

### 7.4 接入路径比较

下列均是基于证据的适配分析，只列影响，不选择路径，也不构成实施计划。

| 路径 | 可复用部分 | 需要新增/适配 | 落位与回声 | 主要限制 |
|:---|:---|:---|:---|:---|
| 会话记录读取器＋通用 MCP 配置 | HiveMemory `/mcp`、Import Bus、确认后推进和重发语义；dsh MCP client | dsh 逻辑日志读取/轮次归一化、结束触发/恢复补扫；将现有连接器来源身份/入口改为可配置。默认 zstd 不适用 Python 标准库文本 reader，可读官方 handle，或在新独立 root 显式用 none 并锁格式。 | SEARCH/READ 可直接走 MCP；已知 server namespace 可识别回声。**WRITE/UPDATE 尚缺可信 turn 自动注入**，单靠此路径不能满足全部需求。 | 预览文件格式/generation、压缩、fork 继承与多用户轮次；不能让模型补 ID 以绕过缺口。切 compression 不是原目录就地转码。 |
| dsh 原生插件 | 同一 HiveMemory 后端契约；dsh tool registry、事件、projection/persistence 服务 | 插件拥有四个模型工具及 HTTP MCP client；从调用 Agent/公开 turn boundary 和真实用户事件构造后端请求；归一化、回流提交与可恢复水位。 | 在**自有工具体**中合成新 payload 并补 turn，无需修改被冻结的模型 args；自有工具名与 callId 必须列入回声识别。 | 依赖 dsh API/插件版本；事件通知本身不耐久，仍需持久来源和确认补扫；连接、超时、取消及卸载由插件负责。 |
| 组合：原生访问包装＋既有连接器提交循环 | 保留 Python 同步/HTTP 提交语义，原生工具解决可信上下文 | 原生插件输出适配记录或提供可消费的逻辑轮次，外部 reader/提交入口接入；明确适配记录与原会话谁负责恢复。 | 包装工具在调用时落位，回流沿同一身份/轮次；回声映射同时覆盖包装工具及可能的 MCP 内层调用。 | 同时维护 Node/Python 两种运行时；需避免两边各导入一次、在结果回写时重复计数。是否值得这一分工需后续决定。 |

不论选择哪一条，外置 bridge 只有在获得可信且隔离的 Agent/Session/turn 映射时才能补 turn；MCP 协议和静态 headers 不自动携带“当前 dsh 轮次”。原生插件若使用 SDK Client，只依赖公开 transport/callTool，不读取通用 MCP 插件的私有连接状态。[公开 MCP 工具适配入口][mcp-tools]提供模型工具与自有 callback 的桥接能力，但注册、上下文补全和连接生命周期仍是插件工作。

**现有连接器的实际复用边界**：需求第 1.1 节“每个 harness 只需要读取器”是语义层概括。代码的 `core/` 仍在 [`integrations/claude-code/connector/`](../../integrations/claude-code/connector/) 内；[`core/config.py`](../../integrations/claude-code/connector/core/config.py) 与 [`core/client.py`](../../integrations/claude-code/connector/core/client.py) 固定 Claude principal/default agent，[`__main__.py`](../../integrations/claude-code/connector/__main__.py) 固定 reader/hook；[`core/transcript.py`](../../integrations/claude-code/connector/core/transcript.py) 假设按行追加文本与字节游标。需先适配最少的来源、入口与存储边界，不能在接入计划中视为现成独立共享包。

### 7.5 引导 skill 的承载方式

| 载体 | 适用范围 | 限制 |
|:---|:---|:---|
| 本地 `.dsh/skills/<name>/SKILL.md`、`.agents/skills/` 或用户 skill root | 按任务/显式调用加载详细工具使用说明；可只本地安装，也可随 bundle 配置自定义 root | 需有效 name/description，遵守发现优先级和单层扫描；目录可见不等于正文已经加载，压缩后可能需要重载。 |
| 项目/用户 AGENTS 指令 | 提供持续可见的简短入口和何时加载 skill 的提示 | 受 root→cwd 范围和字节预算影响，进入其他项目要重新核对可见性；不承担动态 turn 或凭据。 |
| 原生插件 system section / runtime context | 插件提供持续的简短操作边界，动态部分通过 context 更新 | system section 每 step 组装；动态上下文压缩后重建，不应把完整历史记忆常驻。 |
| MCP server instructions | 使用通用 MCP client 时提供 server 级说明 | 需成功连接及字节预算；不等于完整 skill。若改为插件自有 Client，须显式处理说明传递，不能假设通用 client 的 section 自动存在。 |

来源：[skill provider][skill-filesystem]、[skill 装载与历史行为][skill-tool]、[instructions provider][agent-instructions]、[system prompt][system-prompt]、[MCP section][mcp-context]。本次只核对承载机制，不编写 skill 内容；skill、Agent Profile 和 dsh 的提示词均不能授予 HiveMemory actor 权限。

### 7.6 HiveMemory 插件接入要求

这些是后续接入方案必须满足或明确限制的条件，不新增公共模型、配置名或已决定的插件结构。现有行为的权威说明仍是 [MCP 契约](../contracts/mcp.md)与 [Import Bus](../system/import-bus.md)，本节仅说明 dsh 适配处的要求。

**身份与访问**：

- 在 [`system_principals.yaml`](../../configs/system_principals.yaml) 登记独立 harness principal/凭据，只开放所需 mcp/import 入口；不得借用 Claude principal 或保留 `system`。在 [`workspace_actors.yaml`](../../configs/workspace_actors.yaml) 登记具体 Agent 及四项记忆操作白名单。principal、Agent、环境变量具体名字尚未决定。
- MCP 传 `X-Principal-ID`、Bearer、`X-Agent-ID` 及用户/Workspace 选择；Import Bus 用同一身份，其中 Agent 在请求体 `agent_id`。凭据由配置/进程环境提供，不写入 skill、模型参数或交互轨迹。
- 继续经 `/mcp` 的请求级 actor 操作与 `/api/v1/import/turns`，不经管理 API、不建立 HiveMemory chat 任务进程、不直接访问 Patchouli。HiveMemory MCP 保留 SDK 默认 Host/Origin 校验，只允许本机名称；dsh/WSL 需配置可达且符合校验的端点，不能把这项检查解释成按远端来源 IP 隔离的防火墙。

**写入时落位**：

- WRITE/UPDATE 必须由可信适配器提供 `turn.source`、`external_conversation_id`、`external_turn_id`、`user_message`，可附 recent_turns；缺少 turn 会得到 `request.turn_required`。不能让模型填造或通过取消校验补缺口。若采用原生包装工具，不同时暴露可绕过可信 turn 注入的通用 MCP 写工具。最近三轮是现有连接器策略，不是服务端 schema 的三轮硬上限。
- 插件可通过声明 `agents` 依赖，调用 `ctx.agents.turnBoundaryOf(agent.session)` 取得 openTurnStartSeq/lastTurn，再结合 `source.kind === 'user'` 的已提交输入构造上下文。恢复依赖公开 session projection 或 `sessionQuery.readSession`；新实现不使用已被标记 deprecated 的 snapshotEvents/eventAt/ownEvents。[Agent 公共入口][agent-registry]、[来源标记][llm-message]、[Session 读取边界][session-impl]、[query 入口][session-query]。
- 写入与回流必须保持 **principal/user/agent/source/conversation/turn 六元组一致**；服务端交互 ID 的推导包含全部六项。可用 session id＋own turn/start seq 作为稳定候选映射，但最终编码与 fork/子 Agent 策略留给接入方案；不得用每次重读会变化的文件偏移、当前时间或模型生成 ID。
- WRITE/UPDATE ACK 只表示意图与生成队列接纳；dsh 随后的 cancelled/ABORTED、断开或卸载不撤回已派发写入，不因未看到成功结果就盲目重试写入。证据：[`MCP server`](../../src/hivememory/mcp_server/server.py)、[`导入标识模型`](../../src/hivememory/system/services/import_bus/models.py)。

**回流投影与回声**：

- 按原始逻辑事件的 closed turn 回流；用 seq 和 callId 关联调用及结果，避免 assistant/message 中的 tool-call block 与独立 tool/call 重复登记。读取 request/model 信息，保留实际正文与 reasoning；不使用会截断的 headless NDJSON 或压缩后的可见 messages 作为完整原轨迹。
- 必须识别合成 user/message、压缩 replacement、goal continuation 和 fork inherited prefix。dsh 一 turn 可有多条真实用户/steering，HiveMemory 只有单个 user_message；后续方案需定义保持来源次序且写入/回流一致的投影，或对该形态明确拒绝，不能静默只取最后一条。子会话须定义可信且可恢复的归属映射；独立导入子轮或关联父轮均须保证写入/回流一致，不得用调用时偶然的父会话当前轮替代确定归属。
- outcome 映射需覆盖 completed、aborted、blocked、error、max-tokens、interrupted、forked；只有 completed 可直接映射完成。其他 reason 先按原因与输出判断 cancelled/failed 或排除继承的合成收尾，不将 max-tokens/压缩事件一律当成功。此处要求显式映射，不预先替 owner 决定边界取舍。
- HiveMemory 导入事件从 sequence 1 起严格递增；工具调用和结果合为一个事件，以 action_id 对齐。仅本插件四工具归一化为 SEARCH/READ/WRITE/UPDATE，并按现有模型将 READ aliases/UPDATE alias 放入 targets、去除自动补入的 turn；普通同名工具不误标。原始工具结果仍保留，防回声不是删除轨迹。参见 [`ImportedTurn 模型`](../../src/hivememory/system/services/import_bus/models.py)与 [`Claude reader 现有归一化`](../../integrations/claude-code/connector/claude_code/reader.py)。

**确认与恢复**：

- `session/event` 仅触发处理，不能代替耐久来源；需补扫已持久化的 closed turns、确认后推进水位，并处理 turn/end 与 flush 之间的窗口。自建 outbox 或直接补扫是路径选择，不在本调研中新增设施。
- 复用 accepted/duplicate/skipped 后确认与网络/服务繁忙重发语义，区分拒绝轮次与身份配置错误；新来源首次旧会话不回填、未结束轮次、损坏尾部、日志 generation 和游标重定位都要明确策略。不能把 Claude 的“静止 30 分钟按取消”自动视为 dsh 的轮次保证。
- 服务端去重记录有进程内保留边界，重启/超窗口后不能承诺 exactly once。插件卸载需要停止自建提交任务、关闭 MCP client；正常卸载不是强崩溃下绝不丢数据的证据。现有实现入口：[`core/sync.py`](../../integrations/claude-code/connector/core/sync.py)、[`core/client.py`](../../integrations/claude-code/connector/core/client.py)。

### 7.7 已执行的最小探针与未完成验证

**P1：无真实凭据的机制实测，已通过。** 在仓库外的 `/tmp/hivememory-dsh-probe` 安装 Node 24.9.0 与 dsh 0.2.2-alpha.2；官方源码另存 `/tmp/hivememory-dsh-survey`。没有安装到用户全局环境，没有读取既有模型密钥，也没有把探针、依赖或日志写入本仓库。

最终环境检查与执行命令（均退出 0）：

```bash
/tmp/hivememory-dsh-probe/node_modules/node/bin/node --version
/tmp/hivememory-dsh-probe/node_modules/node/bin/node /tmp/hivememory-dsh-probe/node_modules/@deepseek-ai/dsh/lib/bin.js --version
/tmp/hivememory-dsh-probe/node_modules/node/bin/node /tmp/hivememory-dsh-probe/node_modules/@deepseek-ai/dsh/lib/bin.js --help
/tmp/hivememory-dsh-probe/node_modules/node/bin/node /tmp/hivememory-dsh-probe/probe.mjs
```

探针方法：程序化挂载真实 Cordis、LLM registry、Session、projection、system prompt、tools/native executor、Agent registry/AgentLoop；加载一个事件观察＋原生工具插件。插件显式声明 tools/agents 依赖，注册 `survey_echo`，监听 session/event 和 tools/pre-execute。仅在 `127.0.0.1` 随机端口启动模拟 MCP server，经 dsh MCP client 连接；脚本化 LlmAdapter 驱动“原生工具→MCP echo→最终正文”三个 step，不调用真实模型。

| 检查 | 实际观察 |
|:---|:---|
| 工具对模型可见（A2/D3） | 三次模型请求均包含 `survey_echo` 与 `mcp__survey__echo`；MCP wire 使用原名 `echo`。 |
| 提交事件与轮次（B1/B3） | 事件 seq 连续 0–20；turn/start 为 seq 1，两个 tool/call→tool/result 依次为 9→10、14→15，turn/end 为 20，reason 为 `{kind: completed}`。 |
| 可信轮次信息 | 自有工具通过调用 Agent 得到 `survey-session`、turn 1/openTurnStartSeq 1，并从真实 user 来源事件取得 `Run local survey`。证明公开入口可用；尚未验证跨恢复重建。 |
| 前置改参（B2） | 原生/MCP 两次 gate 中 Object.isFrozen(arguments) 均为 true；赋值抛错，两工具仍分别收到 `native-original`/`mcp-original`。 |
| 正文/思考/结果（B3/C3） | 观察到 reasoning `probe reasoning`、最终 text `probe complete` 及两个工具结果；断言通过。 |
| Streamable HTTP/headers（D1） | 完成 initialize→initialized→tools/list→tools/call；模拟服务收到配置的 `X-Survey: local-only`。未使用真实 Bearer，未验证 SSE/stdio。 |
| MCP instructions（D4） | 模拟 server instruction 在三次模型请求中均存在，断言通过。 |
| 生命周期 | 使用当前 Cordis `ctx.fiber.dispose()` 清理，关闭模拟 server 后进程正常退出。 |

探针只覆盖上述机制。**第 4 节规定的“owner 确认模型凭据”最小实测尚未完成**：本次未取得这一授权凭据；没有借用环境中的未知凭据。真实模型的工具调用、权限询问和实际上下文仍不能据此判定已经验证。

后续接入验证仍需覆盖：真实模型＋HiveMemory 四工具和 Import Bus 的整轮链路；写入/回流六元组相同且沿用落点；新会话主动召回；skill 发现及压缩后重载；并行工具与子会话；取消/服务断开后已接纳写入不撤回；崩溃/恢复/重发、fork 继承排除与多用户输入。WSL 与 Web UI 的实际使用也留待目标环境验证。这些是验收缺口，不是本次已完成结果。

### 7.8 交付核对

- A–G 全部 **27 项**已给出判定、来源与验证程度；细分未知保留在各项中。
- hook 对应、路径比较、skill 承载、HiveMemory 适配要求均已写入本文；没有选择接入路径或修改生产契约。
- 官方源码固定提交与包版本已记录；完成无真实凭据 P1，真实模型/后端端到端实测明确待验证。
- 本文作为 Research baseline 冻结本次证据；后续持续演进的 HiveMemory 目标接口与实施步骤应进入对应 Idea/Plan，不回写为本次已确认外部事实。

## 8. 完成标准

- 第 3 节的问题全部作答，或标为“未知”并说明原因；
- 第 4 节的最小实测完成，或注明未执行的原因；
- 第 5 节的四项交付物齐全；
- 所有来源带固定提交与访问日期，报告 frontmatter 记录 `snapshot_at`。

[dsh-commit]: https://github.com/deepseek-ai/deepseek-harness/commit/ce7073d51254c26cd63b22a60835b8f948deb447
[dsh-npm]: https://registry.npmjs.org/@deepseek-ai/dsh/0.2.2-alpha.2
[dsh-readme]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/README.md
[dsh-package]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/apps/cli/package.json
[dsh-development]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/docs/development.md
[desktop-packaging]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/apps/desktop/packaging/README.md
[cordis-package]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/vendor/cordis/package.json
[dsh-safety]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/SAFETY.md
[dsh-license]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/LICENSE
[third-party-notices]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/THIRD_PARTY_NOTICES.md
[cli-help]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/docs/cli-help.md
[plugin-basic]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/docs/user/develop/basic/index.md
[cordis-life]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/vendor/cordis/src/fiber.ts
[plugin-publish]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/docs/user/develop/basic/publish.md
[plugin-market]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/docs/user/guide/plugin-market.md
[upgrade-guides]: https://github.com/deepseek-ai/deepseek-harness/tree/ce7073d51254c26cd63b22a60835b8f948deb447/docs/upgrade-guide
[tool-guide]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/docs/user/develop/basic/tool.md
[tool-schema]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/core/tools/src/schema.ts
[tool-contracts]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/core/tools/src/contracts.ts
[tool-events]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/core/tools/src/events.ts
[tool-pipeline]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/core/tools/src/pipeline.ts
[tool-batch]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/core/tools/src/batch.ts
[native-executor]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/core/tool-executor-native/src/index.ts
[session-impl]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/core/session/src/index.ts
[session-types]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/core/session/src/types.ts
[session-repair]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/core/session/src/repair.ts
[session-fork]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/core/session/src/fork.ts
[agent-runtime]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/core/agent/src/runtime-types.ts
[agent-registry]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/core/agent/src/index.ts
[llm-message]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/llm/llm/src/message.ts
[loop-agent]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/core/agent-loop/src/agent.ts
[session-query]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/session-query/session-query/src/index.ts
[subagent-spawn]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/subagent/subagent-spawn-in-process/src/index.ts
[subagent-fork]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/subagent/subagent-fork-in-process/src/index.ts
[base-config]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/bundle/base/cordis.patch.yml
[persistence-jsonl]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/session/session-persistence-jsonl/README.md
[persistence-format]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/session/session-persistence-jsonl/src/format.ts
[fork-command]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/api/session-controller/src/commands.ts
[mcp-guide]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/mcp/mcp-client/README.md
[mcp-transport]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/mcp/mcp-client/src/transport.ts
[mcp-tools]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/mcp/mcp-client/src/tools.ts
[mcp-context]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/mcp/mcp-client/src/server-context.ts
[mcp-connection]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/mcp/mcp-client/src/connection.ts
[skill-filesystem]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/skill/skill-filesystem/README.md
[skill-tool]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/skill/tool-skill/src/index.ts
[system-prompt]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/docs/subsystems/system-prompt.md
[agent-instructions]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/context/agent-instructions/README.md
[instructions-config]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/context/agent-instructions/src/config.ts
[memory-guide]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/docs/user/guide/mcp-memory.md
[compaction-guide]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/compaction/compaction-basic/README.md
[compaction-region]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/compaction/compaction-basic/src/region.ts
[provider-guide]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/docs/user/guide/providers.md
[headless-guide]: https://github.com/deepseek-ai/deepseek-harness/blob/ce7073d51254c26cd63b22a60835b8f948deb447/packages/bundle/headless/README.md
