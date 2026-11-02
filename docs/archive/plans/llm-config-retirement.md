---
title: 模型选择统一为 ModelSelection，组件模型只经注册表解析
status: archived
archived_at: 2026-11-01
implemented_by: PR #167
superseded_by: docs/system/configuration.md
owner: system
target: v0.7.3 之前（版本之外的工作）
scope: unified-model-selection-for-components-and-sessions-through-registries
related_docs:
  - docs/system/configuration.md
  - docs/system/composition.md
  - docs/alice/agent-runtime.md
  - docs/alice/orchestration.md
  - docs/contracts/subsystem-contracts.md
  - docs/workspace/task-process.md
  - docs/gateway/analysis.md
  - docs/patchouli/generation.md
  - docs/patchouli/perception.md
  - docs/help/configuration.md
  - docs/frontend/management-views.md
  - docs/frontend/chat-view.md
  - docs/frontend/state-and-transports.md
  - docs/archive/todo/config-secrets-written-to-yaml.md
code_paths:
  - src/hivememory/core/models/model_definition.py
  - src/hivememory/config/shared.py
  - src/hivememory/config/app.py
  - src/hivememory/config/gateway.py
  - src/hivememory/config/patchouli.py
  - src/hivememory/system/model_registry.py
  - src/hivememory/system/assembler.py
  - src/hivememory/agent_runtime/model/resolution.py
  - src/hivememory/agent_runtime/frame.py
  - src/hivememory/alice/context/
  - src/hivememory/workspace/contracts/cpu.py
  - src/hivememory/server/models/chat.py
  - src/hivememory/server/models/config.py
  - src/hivememory/engines/gateway/llm.py
  - configs/config.yaml
  - frontend/src/components/settings/EngineSettings.tsx
  - frontend/src/components/SettingsPanel.tsx
  - frontend/src/stores/chat/runtimeConfigStore.ts
updates:
  - docs/system/configuration.md
  - docs/system/composition.md
  - docs/alice/agent-runtime.md
  - docs/contracts/subsystem-contracts.md
  - docs/help/configuration.md
  - docs/frontend/management-views.md
last_reviewed: 2026-11-01
---

# 模型选择统一为 ModelSelection，组件模型只经注册表解析

> 状态：Archived（2026-11-01，实现、测试与事实文档完成；设置页在浏览器中的核对未进行，见第 10 节）。当前事实见 [System 配置与注册表](../../system/configuration.md#3-registry-解析)第 3 节与 [Agent Runtime](../../alice/agent-runtime.md#4-模型选择与-mtp-生成器)第 4 节。以下为实施时的计划原文，相对链接已随归档改正。原状态：Active（2026-11-01 由 Todo“Gateway 与 Librarian 的旧式直配路径”升级而来，删除前的最后版本见 commit `190accd2`）。
>
> owner 决定（2026-11-01，按决定的先后）：
>
> - Gateway 两个分析器的 `model_override` 删除；
> - 两段组件配置从 `shared.llm` 挪到 Gateway 与 Patchouli 各自的配置下；
> - 维持启动时解析，热重载留待全部配置的热重载一并处理；前端修复引擎设置页；
> - 模型选择只用一个类型：组件配置、会话请求与注册表的解析入口共用 `ModelSelection`，`LLMConfig` 与会话请求的 `GenerationOptions` 删除；组件与 Alice 有同等的配置能力，temperature、top_p、max_tokens 都可以覆盖；
> - 注册表的端口不同时返回请求参数与模型定义，`ResolvedFrameModel` 删除；
> - 组件不使用 LLM 写作 `null`。
>
> 中间曾按“保留三字段的 `LLMConfig`、注册表返回 `(LLMRequestParams, ModelDefinition)`”实施过一版（commit `4395520b`），owner 审视数据模型后改为本计划的设计。第 4 节标为“候选”的做法由 owner 确认，或是实施中的默认做法，见各项说明。

## 1. 背景和问题

Gateway 与 Librarian 的模型配置原先在 `shared.llm.gateway` / `librarian`（类型 `LLMConfig`），有两种写法：以 `model_id` 引用模型注册表，或者不写 `model_id`，直接给出 `model`、`api_key`、`api_base`。后一种写法完全绕过两个注册表。2026-11-01 主配置的密钥移出 `config.yaml` 时（[归档 Todo](../todo/config-secrets-written-to-yaml.md)），这条路径未纳入当时的范围，登记为 Todo。

审视数据模型时看到三层数据各自的角色：

1. **模型定义 `ModelDefinition`**：哪个模型、经哪个 Provider 连接、上下文窗口、这个模型的默认生成参数；
2. **使用方的选择与覆盖**：选哪个模型，以及要不要改写它的默认生成参数；
3. **请求参数 `LLMRequestParams`**：一次调用实际发送的参数。

第 1、3 层各只有一个类型，第 2 层却有四种互不一致的形状：Agent Profile（`model_name` 加可选的 temperature、top_p，没有 max_tokens）、会话请求的 `GenerationOptions`（全部可选，经 HTTP 进入后以 dict 一路传递）、组件的 `LLMConfig`（覆盖值必填，混着 api_key 与无人读取的字段），以及注册表 `resolve` 的三个零散参数。注册表的三个解析方法都返回 `LLMConfig`，组合根与 Alice 再各自把它翻译为请求参数；Alice 又把请求参数、输入上限与展示名打包为 `ResolvedFrameModel`。

## 2. 目标与非目标

目标：

- 第 2 层只有一个类型 `ModelSelection`，组件配置、会话请求与注册表的解析入口共用；Gateway 与 Librarian 与 Alice 会话有同等的配置能力。
- 组件的模型只能以 `model_id` 引用模型注册表，凭证只来自 Provider 注册表；配置文件与环境变量中不再有直接指定模型、密钥或地址的入口，`model_override` 一并删除。
- 注册表回答两个问题，各返回一个已有类型：按 ID 取模型定义，按选择取请求参数；`LLMConfig`、`GenerationOptions` 与 `ResolvedFrameModel` 删除。
- 两段组件配置位于 Gateway 与 Patchouli 各自的配置下；主配置中不再有 LLM 的 `api_key`。
- 前端引擎设置页能读到并保存两个组件的模型选择。

非目标：

- 热重载：维持启动时解析，修改后重启生效。
- 前端主配置其余分类与后端配置结构的对齐。本计划只让它们在真实配置加载后不再渲染出错（第 4.7 节）。
- Agent Profile 的持久化结构不变：它的 `model_name`、temperature、top_p 在 Alice 内部拼入 `ModelSelection`；让 Profile 直接持有 `ModelSelection`（顺带补上 max_tokens）要改持久化结构与 Agent 编辑页，另行处理（[Todo](../../todo/agent-profile-model-selection.md)）。
- Qdrant 的 `api_key` 与主配置接口的密钥机制不变。

## 3. 当前状态和缺口

以下按 2026-11-01 的代码核对。

1. **`LLMConfig` 的两种角色**：既是组件配置，也是注册表的解析结果；`provider` 没有读取方，`context_window`、`top_p` 不被组件使用。
2. **注册表的三个解析方法都返回 `LLMConfig`**：`resolve_for_llm_config` 服务组件，没有 `model_id` 时原样返回配置；`resolve` 经 `agent_runtime.ModelResolver` 服务 Alice；`to_llm_config` 只有测试调用。
3. **会话请求的覆盖**：`ChatRequest.generation_options`（`GenerationOptions`）在路由处转为 dict，经任务进程、CPU 端口一路以 `dict[str, Any]` 传给 Alice。
4. **`ResolvedFrameModel`**：请求参数、输入上限与展示名打成一个包随帧保存；展示名只在解析时写进 `progress.model_used` 一次，输入上限是 Alice 自己算出的整数。
5. **旧式环境变量别名**：`HIVEMEMORY__LLM__GATEWAY__` 与 `__LIBRARIAN__` 下的 `PROVIDER`、`MODEL`、`API_KEY`、`API_BASE` 共 8 条只服务直配写法。
6. **`model_override`**：Gateway 两个分析器可直接写 LiteLLM 模型名，只替换模型、沿用 Gateway 凭证，不经过注册表；仓库的配置没有设置它。
7. **只在启动时解析**：组件的请求参数在装配时解析一次，模型页面与 Provider 页面的修改要重启才对组件生效。
8. **前端主配置页实际显示 mock 数据**：前端类型仍是旧的顶层结构，加载校验读取后端不存在的段时抛出，页面退回 `MOCK_CONFIG`，此时保存提交 mock 数据并被后端拒绝；主配置接口的响应模型也不返回 `gateway` 段。

## 4. 目标设计

### 4.1 `ModelSelection`

位于 `core.models.model_definition`，与 `ModelDefinition` 放在一起：

| 字段 | 含义 |
|:---|:---|
| `model_id: str \| None` | 模型注册表 ID，`'default'` 表示默认模型；None 表示这一层不选模型 |
| `temperature: float \| None` | 0 到 2；None 表示不覆盖 |
| `top_p: float \| None` | 0 到 1；None 表示不覆盖 |
| `max_tokens: int \| None` | 至少 1；None 表示不覆盖 |

每一项为 None 时沿用上一层：会话请求沿用 Agent Profile，Profile 与组件配置沿用模型定义；`model_id` 一路都没有给出时取注册表的默认模型。取值校验沿用原 `GenerationOptions`（原 `max_tokens` 的上限 32768 没有依据，未保留）。生成参数跟着使用方而不是模型定义，因为它们常常是使用方任务的要求，放进模型定义会随模型记录被其他使用方连带修改。

### 4.2 注册表的两个问题

- `select_model(model_id) -> ModelDefinition`：按 ID 选出模型定义，`'default'` 选出默认模型，找不到时抛出 `ModelNotFoundError`；
- `resolve(selection) -> LLMRequestParams`：模型取选中定义的 `litellm_model`（`model_id` 为 None 时取默认模型），凭证取它的 provider 凭证，生成参数以选择中的覆盖值优先。

`agent_runtime.ModelResolver` 端口就是这两个方法。`resolve_for_llm_config`、`to_llm_config` 删除。

### 4.3 Alice

- `resolve_frame_model` 按会话优先于 Profile 的顺序拼出一份 `ModelSelection`，先 `select_model` 取上下文窗口与展示名，再 `resolve` 取请求参数；两次查询之间没有 `await`。返回 `(请求参数, 输入上限, 展示名)`。
- `ResolvedFrameModel` 删除。`ExecutionFrame` 直接保存 `llm_params` 与 `input_token_limit`，在同一处一起赋值；展示名写入 `progress.model_used`。CALL 恢复时沿用已保存的两项。
- 上下文构建器把两项交给运行服务挂到根帧上；会话摘要器改为接收请求参数与输入上限。
- 未注入注册表时，会话选择的 `model_id` 直接作为可执行的模型标识符，行为与原先相同。

### 4.4 组件配置与组合根

- `SystemGatewayConfig.llm` 与 `PatchouliConfig.llm` 的类型是 `ModelSelection | None`，默认值分别为 temperature 0.0、max_tokens 512 与 temperature 0.3、max_tokens 32768，`model_id` 不写（默认模型）；配置为 None 的组件不使用 LLM。
- 组合根直接以 `registry.resolve(selection)` 得到两个组件的请求参数，`component_llm_params` 删除；`shared.llm` 与 `LLMGlobalConfig` 删除，`shared` 只剩 `embedding`。
- 组件请求从此带上模型定义的 `top_p`（未覆盖时），与 Alice 一致。

### 4.5 会话请求与 CPU 端口

- `ChatRequest.generation_options` 改为 `model_selection: ModelSelection | None`，其中的 `model` 改为 `model_id`；`GenerationOptions` 删除。
- 任务进程与 CPU 端口的 `generation_options: dict[str, Any]` 改为 `model_selection: ModelSelection | None`，进程原样传递；未给出的项如何沿用由执行者决定。
- CALL 失败响应中报告的模型名取会话选择的 `model_id`，没有时取被调方 Profile 的 `model_name`，与原先相同。

### 4.6 配置加载与环境变量

- `shared.llm` 迁到了别处，按 `retrieval.retriever.reranker` 的先例显式拒绝：`SharedConfig` 在校验前检查到 `llm` 时加载失败，错误指出两段的新位置。
- 组件配置中残留的 `model`、`api_base`、`provider`、`context_window`，以及 `model_override`，由配置段拒绝未知键而加载失败；配置文件中非空的 `api_key` 仍先由已有的密钥检查拒绝。
- 删除 8 条直配的环境变量别名；`TEMPERATURE`、`MAX_TOKENS` 的 4 条别名改为指向 `gateway.llm` 与 `patchouli.llm`。删除无生产调用方的 `get_*_llm_config`。

### 4.7 前端

- 引擎设置页读写 `gateway.llm`、`patchouli.llm` 与 `shared.embedding.default.model_name`：模型从注册表中选择（另有“注册表默认模型”），temperature、top_p、max_tokens 可以留空表示沿用模型默认值（新增可留空的数字输入）；配置为 `null` 时只显示提示，因为页面不知道各组件的默认生成参数，从页面重新启用会丢掉它们（实施中的默认做法）。
- 加载校验只检查与后端一致的字段；仍按旧结构读写的分类显示“尚未与后端配置结构对齐”的提示，不渲染表单（owner 确认）。主配置接口的响应加入 `gateway` 段（实施中发现，原先不返回）。
- 聊天侧：请求负载改为 `model_selection`（`ChatModelSelection`，字段 `model_id`）；侧边栏 store 的持久化字段改为 `modelSelection`，版本升到 3，旧的 `generationOptions` 在加载时迁移，原值保留。

## 5. 受影响的系统、契约和数据

- **配置契约**：`shared.llm` 删除，两段移到 `gateway.llm` 与 `patchouli.llm`；`model_override` 删除；环境变量别名删除 8 条、改向 4 条。
- **HTTP 契约**：`POST /api/v1/chat` 的 `generation_options` 改为 `model_selection`，`model` 改为 `model_id`；`GET /api/v1/config` 与 `/config/defaults` 的响应加入 `gateway`。消费方只有前端。
- **子系统契约**：`workspace.contracts.CPUPort.execute` 的 `generation_options` 改为 `model_selection`；`agent_runtime.ModelResolver` 改为 `select_model` 与 `resolve` 两个方法。
- **数据**：不迁移持久化数据；浏览器中保存的侧边栏设置随 store 版本迁移。

## 6. 兼容与迁移

- 随仓库发布的 `configs/config.yaml` 在同一改动中移动两段。部署的配置需要同样移动，写了 `model`、`api_base` 等的先在注册表登记模型与 Provider 凭证再改为 `model_id`；迁移说明写在 [System 配置](../../system/configuration.md)的升级注意中。
- 旧式环境变量名删除别名之后被静默忽略；嵌套形式的 `HIVEMEMORY__SHARED__LLM__...` 以迁移错误失败。
- 组件开始发送模型定义的 `top_p`；仓库中唯一的模型使用默认值 1.0。
- 直接调用 `/api/v1/chat` 的外部客户端需要改用 `model_selection` 与 `model_id`；当前没有前端以外的调用方。

## 7. 测试和验收

| 不变量 | 归属层 |
|:---|:---|
| `select_model`：ID 与 `default` 选模型，找不到时抛出；`resolve`：覆盖值优先、未选模型与 `default` 都取默认模型、凭证只来自 Provider | `tests/unit/system/test_model_registry.py` |
| 组件请求参数分别来自 `gateway.llm` 与 `patchouli.llm`；配置为 None 时没有请求参数；请求参数不写回配置对象 | `tests/unit/system/test_assembler.py` |
| `shared.llm` 残留时以指出新位置的错误失败；旧式别名映射到新位置且嵌套变量优先 | `tests/unit/system/test_config_agent_runtime.py` |
| 会话选择优先于 Profile、两者都没给的项留给注册表；输入上限与展示名来自模型定义；CALL 恢复沿用已保存的请求参数与上限 | `tests/unit/agent_runtime/test_runtime.py` |
| 会话请求的 `model_selection` 原样转交注册入口 | `tests/unit/server/routers/test_chat.py` |
| 主配置接口的密钥机制（只剩 Qdrant 的密钥） | `tests/unit/server/routers/test_config.py` |

随契约删除或改写的旧测试：没有 `model_id` 时原样返回、`to_llm_config`、新旧 LLM 环境变量的优先级、`model_override` 只替换模型、`component_llm_params`、注册表返回元组、`ResolvedFrameModel` 打包的“合并配置转为请求参数”（改为端口直接返回请求参数后只剩回读替身的值）。

```bash
pytest tests/unit tests/integration -m "not live_llm and not e2e and not slow" -n auto --cov=hivememory --cov-fail-under=85
pytest tests/e2e -m "not live_llm and not real_infra and not slow"
ruff check . && black --check . && uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
cd frontend && npm run lint && npm run build && npm run test:session
```

## 8. 风险与已知限制

- 设置页未在浏览器中渲染核对（第 10 节）。
- 旧式环境变量名被静默忽略。
- Agent Profile 仍以三个平铺字段表达模型选择，且没有 max_tokens；会话与组件之外的第 2 层仍有这一种形状（第 2 节非目标，[Todo](../../todo/agent-profile-model-selection.md)）。
- `configApi.ts` 的 `fetchFoldTokenThreshold` 读取后端不存在的 `config.perception`，且没有调用方；General 页读取的 `system.version` 已从后端配置删除，显示为空。

## 9. 完成时必须更新的当前文档

全部已更新：[System 配置与注册表](../../system/configuration.md)（配置树、来源与别名、解析步骤、密钥说明与升级注意）、[System 组合根](../../system/composition.md)第 1.3 节、[Agent Runtime](../../alice/agent-runtime.md)第 4 节、[Alice 编排](../../alice/orchestration.md)、[子系统公共契约](../../contracts/subsystem-contracts.md)的 CPU 端口、[任务进程](../../workspace/task-process.md)第 7 节、[Gateway 话题与查询分析](../../gateway/analysis.md)、[Patchouli 生成](../../patchouli/generation.md)与[感知与短期话题](../../patchouli/perception.md)、[配置指南](../../help/configuration.md)、[管理页面](../../frontend/management-views.md)第 3.3 节、[Chat 视图](../../frontend/chat-view.md)、[状态与传输](../../frontend/state-and-transports.md)、[PROJECT](../../PROJECT.md)，以及 ROADMAP、Plans 导航与归档索引。

## 10. 实施记录

**提交**（PR #167，分支 `refactor/llm-config-retirement`）：计划（`18469be7`）；第一版实现（`4395520b`，三字段的 `LLMConfig`、注册表返回元组）；按本计划改写的实现、测试与文档（本次提交）。

**验证**：

- 后端快速测试集与覆盖率门槛、确定性 e2e、ruff、black、mypy 通过；前端 lint、build 与 `test:session` 通过。`test_generation_topic_extraction.py`（`live_llm`）只核对了收集，未调用真实模型。
- 设置页的数据路径以命令行核对（脚本在仓库之外）：以本分支的配置构造配置对象，经主配置接口读出的 JSON 交给编译后的 `validateConfig` 没有错误，引擎页读取的字段都有值；把 Gateway 的 `model_id` 清空、设置 `top_p`、清空 Librarian 的 temperature 后提交，写出的文件随之改变。
- 侧边栏 store 的迁移以命令行核对：版本 2 的 `generationOptions` 迁移为 `modelSelection` 且模型原样保留，版本 1 的遗留 litellm 字符串被清空。
- 未在浏览器中渲染设置页与聊天侧边栏：本机没有浏览器自动化工具，正在运行的后端仍是旧代码，重启会中断 owner 正在使用的 Claude Code 与 dsh 会话。
