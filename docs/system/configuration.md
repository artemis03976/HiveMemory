---
title: System Configuration and Registries
status: current
owner: system
scope: configuration-loading-and-global-registries
code_paths:
  - src/hivememory/config/
  - src/hivememory/system/provider_registry.py
  - src/hivememory/system/model_registry.py
  - configs/config.yaml
  - configs/models.yaml
  - configs/system_principals.yaml
  - configs/workspace_actors.yaml
related_contracts:
  - docs/architecture/boundaries.md
  - docs/contracts/subsystem-contracts.md
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-10-18
---

# System 配置与注册表

配置是 System 的装配输入，不是另一套运行时控制 API。它需要同时满足两个现实：开发者希望通过 YAML 和环境变量管理整套服务，子系统又必须拥有自己那部分语义和默认值。当前做法是由 `HiveMemoryConfig` 统一加载和校验，再把各配置段注入对应宿主；System 不在请求路径里重新解释 Gateway、Patchouli 或 Alice 的内部配置。

配置模型集中在顶层 `config` 包（依赖层级最低，只依赖 pydantic、core 常量与 i18n），按子系统和高聚合组件分段：`config.shared`、`config.patchouli`、`config.gateway`、`config.alice`、`config.memory_compiler`、`config.attachments`、`config.workspace`、`config.runtime`、`config.passive`、`config.access`；根配置 `HiveMemoryConfig` 与加载函数位于 `config.app`。使用规则：

- `config.app` 只供组合根（`system`）与入口（`server`、脚本）使用，由分层测试守护；
- 组件只导入并接收自己的配置段，不持有根配置，也不自行调用 `load_app_config()`；子系统宿主的构造参数是对应配置段（例如 `PatchouliSystem(config=PatchouliConfig, shared_config=..., ...)`），由组合根从根配置中取出注入；
- 基础设施工厂函数（Embedding、Rerank 等）要求调用方显式传入配置，缺失时抛 `ValueError`，不回退读取全局配置；LLM 服务不接收配置，组件 LLM 配置由组合根转换为每次调用的请求参数（第 3 节）。

## 1. 配置树

`HiveMemoryConfig` 当前包含：

| 区域 | 模块 | 主要内容 | 责任边界 |
|:---|:---|:---|:---|
| `system` / `logging` / `i18n` | `config.app` | 名称、调试标志、server 自身的 principal 标识 `server_principal_id`；日志级别、格式与输出（尚未生效，见下文第 5 节）及 WebSocket 日志广播；默认语言 | System / 全局文本解析 |
| `scheduler` / `runtime_events` | `config.runtime` | tick、关闭等待、observer/perception/GC 任务开关与间隔；事件 ring buffer 与订阅队列大小 | 共享运行时设施 |
| `shared` | `config.shared` | LLM、embedding、provider credentials | Registry 与共享模型能力 |
| `gateway` | `config.gateway` | interceptor、commands、workflow、topic router、query analysis | Gateway |
| `passive_ingress` | `config.passive` | dedup、turn accumulator 上限 | System passive ingress |
| `memory_compiler` | `config.memory_compiler` | 编译策略 | MemoryCompiler 所有者；由组合根注入 workspace 任务进程（检索结果编译）与 Alice（MTP 输出编译） |
| `attachment_parser` / `attachment_compiler` | `config.attachments` | 附件解析资源限制；附件编译预算 | 附件解析器 / AttachmentCompiler；`attachment_compiler` 由组合根注入 workspace 任务进程 |
| `workspace` | `config.workspace` | 读取视图缓存容量与会话容量 | workspace |
| `patchouli` / `alice` | `config.patchouli` / `config.alice` | 各自运行时和存储配置 | 对应子系统 |

访问登记不在 `HiveMemoryConfig` 中，见下文“访问登记文件”。System 只直接拥有顶层设施、接入登记和 passive ingress 配置；Gateway 的 workflow timeout、Patchouli 的 retrieval 和 Alice 的 MTP 权限仍由各自所有者解释。当前没有用于创建、切换或复制 Workspace 的配置项；默认 `main_workspace` 由入口按用户身份解析，Workspace 资源边界和 AssetStore 生命周期见 [Workspace 架构](../architecture/workspace.md)。

主动写入的背景轮数由 `patchouli.generation.active_recent_blocks` 控制，默认 5、最小 0；即话题过去 N 个 block 与 `state_summary`，不含当前轮。该项与 `patchouli.perception.fold_retain_recent_blocks` 的折叠保留轮数独立，见[Patchouli 生成](../patchouli/generation.md)。

升级注意（写入意图实时派发）：WRITE/UPDATE 的 ACK 现在等待生成队列接纳，可能在 ACK 前返回明确拒绝或未知结果；后续运行失败或取消不撤回已提交写入。SSE done 与 `chat.run.completed` 删除 `memory_task_ids`，消费者改用任务事件与 list/get。该次迁移删除了 payload 的 `materialize_tasks`（当前 codec 版本见下方会话升级说明）；当前队列为进程内队列，不提供旧版本重放。

会话配置为 `workspace.sessions.max_sessions_per_workspace=128` 与 `max_turns_per_session=512`，均要求至少 1。达到上限时显式拒绝创建或进程认领，不清理历史。`gateway.context_preparation.session_turn_limit=3` 控制 Gateway 的近期会话轮数（至少 1）；Alice 始终接收全量原文，自行按 `alice.compaction` 构建预算受限的视图，不受该项与 Topic 的 `context_block_limit` 影响。`workspace.process.shutdown_wait_seconds=10`（至少 0）是 System 关闭时等待在途任务进程由交付方自行收口的上限，超时后兜底关闭。

会话升级后 `session_id` 实际绑定资源；缺省仍自动新建。SSE 首个事件回传会话 ID，done 以 `topic_id` 标注本轮材料归属；`topic_info` 与 `pool_topics` 已删除。Interaction codec 5 增加 `outcome`，旧版本不重放；会话仅在内存中，后端重启后前端需重新新建。

升级注意（配置未知键）：各配置段改为拒绝未知键，存量 YAML 或 `HIVEMEMORY__*` 环境变量中写错的键会让启动失败。同时删除了从未生效的键：`i18n.fallback_language`、`i18n.supported_languages`、`shared.embedding.*.batch_size`、`normalize_embeddings`、`dimension`（集合向量维度由 `patchouli.storage.vector_dimension` 决定，须与嵌入模型的输出一致）、`patchouli.storage.deployment`、`data_dir`、`binary_path`（及旧环境变量 `QDRANT__DEPLOYMENT`、`QDRANT__DATA_DIR`、`QDRANT__BINARY_PATH`）、`patchouli.perception.scan_interval_seconds`（扫描间隔由 `scheduler.tasks.perception_idle_flush_interval_seconds` 决定）、`patchouli.generation.deduplicator.enable_vitality_tracking`、`patchouli.lifecycle.vitality_calculator.max_access_boost`、`points_per_access`、`alice.koakuma.execution_timeout_seconds`、`tool_cache_size` 与 `gateway.commands.unknown_command_policy`，需要从配置中移除。示例配置中位置写错的 `patchouli.perception.engine.idle_timeout_seconds` 已移到 `patchouli.perception.idle_timeout_seconds`（此前该值被忽略、按默认 900 秒运行），残留的 `gateway.analyzer` 段已删除。

### 1.1 访问登记文件

两类访问登记各用一个 YAML 文件，对应各自的配置所有者，由 `config/access.py` 的 `load_access_registration()` 在组合根装配时装载，不经过 `HiveMemoryConfig`：

| 文件 | 顶层键 | 配置所有者 | 路径覆盖 |
|:---|:---|:---|:---|
| `configs/system_principals.yaml` | `principals`：调用来源的接入登记 | System | `HIVEMEMORY_PRINCIPALS_PATH` |
| `configs/workspace_actors.yaml` | `workspace_actors`：Workspace Actor 的准入与行为白名单 | workspace | `HIVEMEMORY_WORKSPACE_ACTORS_PATH` |

- 两个文件都拒绝未知字段；默认路径的文件缺失时按空登记装载并告警，网关随后拒绝一切认证（fail closed）；显式指定的路径缺失或内容非法时装载失败。
- 登记在运行实例内不可变，修改经重启生效；登记中没有 context 有效期。
- `system.server_principal_id`（默认 `hivememory:http-server`）是 server 经统一认证网关认证时使用的 principal，必须与 `system_principals.yaml` 中的登记一致。

`allowed_operations` 只接受六项 actor 操作：`memory.read`、`memory.search`、`memory.run`、`memory.write`、`memory.update`、`agent.call`，分别对应 MTP 的 READ、SEARCH、RUN（记忆中的用户工具）、WRITE、UPDATE 与 CALL，互不隐含。默认用户级登记包含全部六项；内置 CALL Profile 同样要求 `agent.call`。chat 的阶段调用（Gateway、预检索、Profile 解析、附件租借、finalize）不检查 actor 操作，白名单为空的 Agent 也能完成对话，只是不能执行任何 MTP 资源操作。

访问登记的白名单是 Agent 操作权限的唯一来源：Agent Profile 的 `agent_config` 不再承载权限，`sys_*` 系统工具不设权限。修改白名单需改本文件并重启，运行时编辑属于尚未排期的配置热更新。

保留的 `system` 记录只登记准入，不得配置 `allowed_operations`：管理入口经 root API 授权，只要求 actor 是 `system`，不读取白名单。默认接入登记另有本地来源 `hivememory:profile-injector`（kind `local-process`，adapter `local`），供 `scripts/inject_profiles.py` 经认证网关以 `system` 取得 context 后调用 root API 注入 Profile；脚本可用 `--principal-id` 指定其他已登记的本地来源。

升级注意（能力层分为 root API 与 actor API）：旧名称 `resource.read`、`resource.search`、`profile.read`、`asset.acquire`、`interaction.submit`、`memory_intent.submit`、`task.observe` 与 `management.*` 已删除，配置中出现时装载失败并指出值与记录坐标；`system` 记录配置任何 operation 时同样装载失败。自定义过该文件的部署应改用新名称，并删除 `system` 记录上的 `allowed_operations`。旧 `memory_intent.submit` 拆为 `memory.write` 与 `memory.update`；RUN 的记忆目标改为需要 `memory.run`（不再借用 READ）。缺少对应操作时，MTP 权限拒绝沿用 `mtp.permission.verb_denied` 文案键，缺少 `agent.call` 时 CALL 以准备失败回填。

升级注意（Agent 操作白名单）：Agent Profile 的 `allowed_mtp_verbs` 与 `allowed_sys_tools` 已删除，存量图纸中的这两个键在解析时被忽略、不需要迁移，前端也不再提供对应开关。曾依赖它们收窄动词的 Agent，现在按访问登记执行（默认用户级记录为全部六项）；需要收窄时，在本文件为该 Agent 增加精确记录。系统工具无法再按 Agent 限制。CALL 子 Agent 不再按被调用方 Profile 收窄，与父 Agent 的动词集合一致（去掉 CALL）。

升级注意（执行线程层）：CALL 派生的子 Agent 以目标 Agent 的身份运行，需要目标在同一 Workspace 有准入记录，并按目标自己的白名单授权。随仓库发布的默认登记以用户级记录覆盖该用户的全部 Agent，行为不受影响；只为个别 Agent 配置精确记录的登记，需要为 CALL 目标补充记录，否则这些 CALL 以“agent 不可用”失败。内置图纸（未指定目标、`default` 或 `omni_doll`）以 `omni_doll` 身份运行。

登记的字段语义、用户级记录规则、随仓库发布的默认登记与认证授权模型见 [Workspace 架构](../architecture/workspace.md)第 4.2 节。

### 1.2 Alice 会话预算与压缩

`ModelDefinition.context_window` 与解析后的 `LLMConfig.context_window` 是可选的正整数，单位为总窗口 token；模型登记在 `configs/models.yaml`，模型管理 API 支持读写，显式 null 清除登记。它不作为生成参数透传给提供商。Alice 按实际解析模型与请求覆盖后的 `max_tokens` 计算输入上限：总窗口减去生成上限和安全余量，未登记窗口时使用下表默认值。

输入预算由 `AgentRuntimeConfig`（`alice.runtime`）解释，独立于历史压缩：

| 字段 | 默认值 | 作用 |
|:---|---:|:---|
| `default_context_window` | `65536` | 未登记模型的保守总窗口，必须大于 0 |
| `safety_margin` | `2048` | 估算误差预留 token，允许 0 |

这两个字段只从 `alice.runtime` 读取，不提供 `alice.compaction` 位置的别名。`alice.compaction` 的配置模型是 `CompactionConfig`，只控制视图压缩，默认值如下：

| 字段 | 默认值 | 作用 |
|:---|---:|:---|
| `enabled` | `true` | 自动减重与摘要开关；关闭时忽略已有检查点，历史超出输入上限时以 `compaction_failed` 结束，输入守卫仍有效 |
| `trigger_ratio` | `0.75` | 完整视图超过输入上限的此比例时压缩 |
| `target_ratio` | `0.5` | 压缩后目标占输入上限的比例 |
| `recent_raw_ratio` | `0.25` | 最近完整轮次原文预算占输入上限的比例，允许 0 |
| `tool_result_excerpt_chars` | `2000` | RUN 摘录与其他保留型工具结果正文限长 |
| `summary_max_tokens` | `1024` | 摘要正文估算 token 上限；生成另受模型输出上限约束 |
| `summary_input_tool_chars` | `2000` | 摘要输入中每条工具消息字符上限 |
| `summary_timeout_seconds` | `60` | 每次摘要调用超时秒数 |
| `max_references` | `16` | 引用条数上限，允许 0 |
| `reference_max_chars` | `256` | 单条引用字符上限，超长引用忽略，不截成错误地址 |

比例上限均为 1，触发与目标比例必须大于 0；其余长度与超时必须大于 0。默认比例按固定样本做过预算校准，不代表对真实模型质量的保证。算法、回退与已知限制见 [Agent Runtime](../alice/agent-runtime.md#31-会话上下文视图)。摘要复用本轮模型，没有独立的辅助模型配置；这些参数修改后重启运行实例生效，不影响会话容量上限或原文保存规则。

## 2. 来源与优先级

当前 `BaseSettings` 的来源顺序为：

```text
显式构造参数
  > HIVEMEMORY__* 环境变量
  > .env / configs/.env
  > 旧环境变量别名映射
  > provider 动态凭证扫描
  > configs/config.yaml（或 HIVEMEMORY_CONFIG_PATH 指定文件）
  > file secrets
```

嵌套环境变量使用 `HIVEMEMORY__` 前缀和 `__` 分隔，例如：

```text
HIVEMEMORY__SCHEDULER__TICK_SECONDS=1
HIVEMEMORY__GATEWAY__WORKFLOW__DEFAULT_REQUEST_TIMEOUT_MS=8000
HIVEMEMORY__PROVIDERS__DEEPSEEK__API_KEY=...
```

旧的 `LLM__...`、`QDRANT__...` 等形式仍通过显式 alias 映射兼容。动态 provider 名不能由静态 alias 枚举，因此由 `provider_credentials_settings_source()` 单独扫描并归入 `shared.providers`。

默认配置文件不存在时使用默认值和环境变量；显式指定的 `HIVEMEMORY_CONFIG_PATH` 不存在会抛 `FileNotFoundError`。YAML 解析失败当前记录 error 并返回空配置源，最终由 Pydantic 默认值和其他来源继续构造。

## 3. Registry 解析

`SystemAssembler` 在装配时：

1. 用共享 provider 配置创建 `ProviderRegistry`；
2. 创建引用该 registry 的 `ModelRegistry`；
3. 解析 Gateway 和 Librarian 的 LLM config，把 `model_id` 对应的 model/provider/api key/api base 补齐；
4. 把两份解析结果各自转换为 LLM 请求参数（模型、凭证、temperature 与 max_tokens；组件配置中的 `top_p` 不发送），与唯一的 LLM 服务一同注入 Gateway 与 Patchouli；未配置模型的组件得到空参数并按无 LLM 运行。

这一步把凭证解析从业务请求热路径移开，也避免 Gateway 和 Patchouli 对同一模型引用各自得出不同结果；两个组件的模型、凭证与采样参数只来自各自的配置段，共享的 LLM 服务不持有任何一方的配置（[System 组合根](./composition.md#13-llm-bundle)）。Alice 的模型参数不经组件配置，由 Agent Profile、会话覆盖与注册表逐帧解析。注册表提供模型和 provider 元数据，但不拥有某次请求的执行状态。

## 4. 配置所有权与兼容边界

- Gateway timeout 由 `gateway.workflow.default_request_timeout_ms` 控制；Chat 应用只能传入更小的 request timeout，不能扩大系统默认 deadline；
- Passive idle interval/timeout 由 `scheduler.tasks` 单一持有，`passive_ingress` 不重复定义同一事实；
- RuntimeEvent 的 buffer/queue 配置只影响观测容量，不改变业务状态；
- 项目版本属于构建事实，不是运行配置；`system.version` 已从配置模型和示例 YAML 中移除，版本唯一来源是 `src/hivememory/_version.py`；
- 配置模型只声明已有消费者的运行时控制；
- 各配置段拒绝未知键（`extra="forbid"`）：写错或已删除的键让配置校验失败，启动与 `POST /config` 都显式报错，不会被静默丢弃后让人误以为生效；根配置 `HiveMemoryConfig` 例外，忽略未知的顶层键，因为它同时接收环境变量与 `.env`，其中有不属于配置树的变量（如 `HIVEMEMORY_CONFIG_PATH`）；
- `i18n.default_language` 在配置校验后同步到进程级 resolver，但请求级显式语言仍由调用方或 Profile 传递。

配置扩展必须先判断它属于哪一个所有者；不能为了方便在 `HiveMemoryConfig` 添加一个字段，然后让多个子系统各自解释不同含义。

## 5. 当前限制

- 部分历史环境变量仍保留兼容映射，清理前不能假设只有嵌套新格式；
- registry 只在进程装配时形成解析结果，运行中 provider/model 配置变更不会自动热重载；
- `logging` 段的 `level`、`format`、`file_path`、`console_output` 没有读取方，日志级别与格式由入口写死或沿用 uvicorn 的日志配置，见 [Todo](../todo/logging-config-not-applied.md)；只有 `websocket_*` 生效。

## 6. 配置变更检查

1. 新字段是否有唯一所有者和唯一真相源？
2. 是否改变了来源优先级或旧 alias 的含义？
3. 凭证是否仍只通过环境变量/secret 进入，而不会写入受版本控制的 YAML？
4. 该字段是装配期事实还是请求期控制？是否被错误地在两处同时读取？
5. 配置失败是应该阻止装配，还是允许明确的保守默认？是否有测试证明？

## 7. 验证入口

- `tests/unit/system/test_config_agent_runtime.py`
- `tests/unit/system/test_model_registry.py`
- `tests/unit/system/test_provider_registry.py`
- `src/hivememory/config/`（`app.py` 为根配置与加载）
- `tests/unit/architecture/test_package_layers.py`（`config.app` 的使用约束）
