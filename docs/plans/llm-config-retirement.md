---
title: 组件模型只经注册表解析，LLMConfig 退出中间翻译
status: active
owner: system
target: v0.7.3 之前（版本之外的工作）
scope: component-llm-selection-through-registries-and-llmconfig-retirement
related_docs:
  - docs/system/configuration.md
  - docs/system/composition.md
  - docs/alice/agent-runtime.md
  - docs/gateway/analysis.md
  - docs/patchouli/generation.md
  - docs/patchouli/perception.md
  - docs/help/configuration.md
  - docs/frontend/management-views.md
  - docs/archive/todo/config-secrets-written-to-yaml.md
code_paths:
  - src/hivememory/config/shared.py
  - src/hivememory/config/app.py
  - src/hivememory/config/gateway.py
  - src/hivememory/config/patchouli.py
  - src/hivememory/system/model_registry.py
  - src/hivememory/system/assembler.py
  - src/hivememory/agent_runtime/model/resolution.py
  - src/hivememory/engines/gateway/llm.py
  - src/hivememory/server/routers/config.py
  - configs/config.yaml
  - frontend/src/components/settings/EngineSettings.tsx
  - frontend/src/components/SettingsPanel.tsx
  - frontend/src/utils/configValidation.ts
  - frontend/src/types/config.ts
updates:
  - docs/system/configuration.md
  - docs/system/composition.md
  - docs/alice/agent-runtime.md
  - docs/gateway/analysis.md
  - docs/patchouli/generation.md
  - docs/patchouli/perception.md
  - docs/help/configuration.md
  - docs/frontend/management-views.md
last_reviewed: 2026-11-01
---

# 组件模型只经注册表解析，LLMConfig 退出中间翻译

> 状态：Active（2026-11-01 建立，owner 同日审阅后开始实施）。由 Todo“Gateway 与 Librarian 的旧式直配路径”升级而来，删除前的最后版本见 commit `190accd2`。owner 决定（2026-11-01）：
>
> - Gateway 两个分析器的 `model_override` 删除；
> - 解析结果由 `LLMRequestParams` 与 `ModelDefinition` 表达，`LLMConfig` 不再充当配置与请求参数之间的中间翻译；
> - 组件配置只保留模型选择与组件任务要求的生成参数（`model_id`、`temperature`、`max_tokens`），沿用 `LLMConfig` 这个名字，不改名；
> - 两段配置从 `shared.llm` 挪到 Gateway 与 Patchouli 各自的配置下；
> - 维持启动时解析，热重载留待全部配置的热重载一并处理；
> - 前端修复引擎设置页；
> - 影响较大，先形成计划审阅。
>
> 第 4 节中标为“候选”的做法由 owner 在审阅时确认（2026-11-01）：两处键名都用 `llm`、组件发送模型定义的 `top_p`、未对齐的前端分类显示提示。

## 1. 背景和问题

Alice 的模型由 Agent Profile 选择、经模型注册表与 Provider 注册表逐帧解析。Gateway 与 Librarian 的模型配置在 `shared.llm.gateway` / `librarian`（类型 `LLMConfig`），有两种写法：以 `model_id` 引用模型注册表，或者不写 `model_id`，直接给出 `model`、`api_key`、`api_base`。后一种写法完全绕过两个注册表，经环境变量注入时，模型与密钥也不出现在 Provider 页面与模型页面中。2026-11-01 主配置的密钥移出 `config.yaml` 时（[归档 Todo](../archive/todo/config-secrets-written-to-yaml.md)），这条路径未纳入当时的范围，登记为 Todo。

owner 的方向是：Gateway 与 Librarian 与 Alice 一样，只经项目级的 Provider 与模型注册表选择模型。核对代码时还看到 `LLMConfig` 同时承担两种角色：配置文件中的组件配置，以及注册表解析后的运行时结果（Alice 的解析也返回它）；每个使用方再把它翻译为请求参数 `LLMRequestParams`。解析结果需要的模型、凭证与采样参数已由 `LLMRequestParams` 表达，上下文窗口与展示名属于 `ModelDefinition`，中间这一层翻译没有独立的含义。

组件配置仍有不能省掉的内容：每个组件用哪个模型，以及组件任务本身要求的生成参数。Gateway 的话题路由与查询分析需要稳定的短 JSON 输出（temperature 0、max_tokens 512）；Librarian 一次结算输出多条记忆，32768 的输出上限是多条提取的正确性要求，输出被截断时任务失败。这些参数是组件任务的要求，不是模型的属性；放进模型定义会让它们随模型记录被其他使用方连带修改，因此留在组件配置中。这两段放在 `shared` 下，却分别只属于 Gateway 与 Patchouli，因此挪到各自的配置下。

## 2. 目标与非目标

目标：

- Gateway 与 Librarian 的模型只能以 `model_id` 引用模型注册表，凭证只来自 Provider 注册表。配置文件与环境变量中不再有直接指定模型、密钥或地址的入口，Gateway 两个分析器的 `model_override` 一并删除。
- 模型注册表的解析直接产出请求参数与所选的模型定义，组件与 Alice 共用同一个解析入口；`LLMConfig` 只作组件配置，不再作为解析结果出现。
- 组件配置只有 `model_id`、`temperature`、`max_tokens`，位于 Gateway 与 Patchouli 各自的配置下；主配置模型中不再有 LLM 的 `api_key`。
- 前端引擎设置页能读到并保存这两段的模型选择与生成参数。

非目标：

- 热重载：维持启动时解析，修改后重启生效。全部配置的热重载由 owner 之后另行处理。
- 前端主配置其余分类与后端配置结构的对齐。本计划只让它们在真实配置加载后不再渲染出错（第 4.7 节）。
- Qdrant 的 `api_key` 与主配置接口的密钥机制（响应打码、保存时沿用、写入文件时剔除）不变。
- `model_id: null` 表示组件不使用 LLM，此时组件的降级行为不变：Gateway 走无 LLM 路径，记忆抽取器以 No-Op 运行，Page Folding 回退为规则摘要。

## 3. 当前状态和缺口

以下按 2026-11-01 的代码核对。

1. **`LLMConfig` 的两种角色**（[`config/shared.py`](../../src/hivememory/config/shared.py)）：字段有 `provider`、`model_id`、`model`、`api_key`、`api_base`、`temperature`、`max_tokens`、`context_window`、`top_p`。作为组件配置时，只有 `model_id`、`temperature`、`max_tokens` 由写入者设置；`provider` 没有任何读取方，`context_window` 与 `top_p` 不被组件使用（[`component_llm_params`](../../src/hivememory/system/assembler.py) 只发送模型、凭证、temperature 与 max_tokens）。
2. **注册表的三个解析方法都返回 `LLMConfig`**（[`ModelRegistry`](../../src/hivememory/system/model_registry.py)）：`resolve_for_llm_config` 服务组件，没有 `model_id` 时原样返回配置；`resolve` 经 `agent_runtime.ModelResolver` 端口服务 Alice；`to_llm_config` 只有测试调用。组合根与 [Alice 的帧模型解析](../../src/hivememory/agent_runtime/model/resolution.py)各自再把结果翻译为 `LLMRequestParams`。
3. **配置位置**：两段位于 `shared.llm`（`LLMGlobalConfig`），由组合根读取后把请求参数分别注入 Gateway 与 Patchouli；`shared` 中另一项是嵌入模型配置 `shared.embedding`。
4. **旧式环境变量别名**（[`LEGACY_ENV_ALIASES`](../../src/hivememory/config/app.py)）：`HIVEMEMORY__LLM__GATEWAY__` 与 `__LIBRARIAN__` 下的 `PROVIDER`、`MODEL`、`API_KEY`、`API_BASE` 共 8 条，只服务直配写法；同一前缀下的 `TEMPERATURE`、`MAX_TOKENS` 对应保留的字段。
5. **`model_override`**（[`config/gateway.py`](../../src/hivememory/config/gateway.py) 的 `TopicRouterConfig` 与 `UserQueryAnalysisConfig`）：直接写 LiteLLM 模型名，[`json_analysis_params`](../../src/hivememory/engines/gateway/llm.py) 只替换模型、沿用 Gateway 的凭证，不经过注册表。仓库的 `config.yaml` 没有设置它。
6. **只在启动时解析**：组合根装配时把两段解析为请求参数注入 Gateway 与 Patchouli，之后不再变化；模型页面与 Provider 页面的修改，Alice 下一帧生效，Gateway 与 Librarian 要重启。
7. **前端主配置页实际显示 mock 数据**：前端的 `HiveMemoryConfig` 仍是旧的顶层结构（`llm`、`embedding`、`qdrant`、`retrieval`、`generation`、`lifecycle`、`perception`、`koakuma`），后端的顶层段中只有 `system`、`logging`、`gateway` 与之同名，`gateway` 的子结构也不同。设置页加载后先执行 [`validateConfig`](../../frontend/src/utils/configValidation.ts)，它读取 `config.llm.gateway.temperature` 时抛出 TypeError，页面退回 `MOCK_CONFIG` 并显示错误；此时保存提交的是 mock 数据，后端以校验错误拒绝。[`EngineSettings`](../../frontend/src/components/settings/EngineSettings.tsx) 虽已提供按 `model_id` 选择的下拉框，读写的却是 `config.llm.*`。
8. **`get_librarian_llm_config` / `get_gateway_llm_config`**（`config/app.py` 中的方法与同名模块函数）没有生产调用方，只有一个集成测试使用。

仓库的 `configs/config.yaml` 与当前运行的部署，两段都以 `model_id` 引用注册表，不使用上述直配写法与 `model_override`。

## 4. 目标设计

### 4.1 组件配置

`LLMConfig` 只保留三项：

| 字段 | 含义 |
|:---|:---|
| `model_id` | 模型注册表 ID，`"default"` 表示注册表的默认模型；`null` 表示组件不使用 LLM |
| `temperature` | 组件任务要求的生成参数，覆盖模型定义的默认值 |
| `max_tokens` | 组件任务要求的生成参数，覆盖模型定义的默认值 |

两段的位置：

| 原位置 | 新位置（候选键名） | 默认值 |
|:---|:---|:---|
| `shared.llm.gateway` | `gateway.llm`（`SystemGatewayConfig.llm`） | `model_id: "default"`、0.0、512 |
| `shared.llm.librarian` | `patchouli.llm`（`PatchouliConfig.llm`） | `model_id: "default"`、0.3、32768 |

- 候选：两处都用 `llm` 作键名，读作“Gateway 的模型”“Patchouli 的模型”。另一种写法是 `patchouli.librarian`，保留现有的 `librarian` 一词，迁移时只需整块移动；代价是两处键名不对称。
- `LLMGlobalConfig` 删除，`shared` 只剩 `embedding`。`LLMConfig` 仍定义在 `config/shared.py`，作为两个配置段共用的类型。
- 配置段继续拒绝未知键（第 4.6 节）。

### 4.2 注册表的解析入口

`ModelRegistry` 只保留一个解析方法：

```python
def resolve(
    self,
    model_name: str,
    temperature_override: float | None = None,
    max_tokens_override: int | None = None,
    top_p_override: float | None = None,
) -> tuple[LLMRequestParams, ModelDefinition]: ...
```

- `model_name` 为 `"default"` 时取默认模型，找不到时抛出 `ModelNotFoundError`，与现状相同；
- 请求参数的模型取 `litellm_model`，凭证取模型 `provider` 对应的 Provider 凭证，采样参数以覆盖值优先、否则取模型定义的默认值；
- 第二项是所选的模型定义，调用方从中读取 `context_window` 与 `display_name`。

`resolve_for_llm_config` 与 `to_llm_config` 删除。

### 4.3 Alice 的模型解析

`agent_runtime.ModelResolver` 的签名与上一节相同。`resolve_frame_model` 直接使用返回的请求参数，输入上限由 `ModelDefinition.context_window` 计算，展示名取 `display_name`。“没有可执行的模型标识符”这项检查删除：`LLMRequestParams.model` 要求非空，注册表构造请求参数时已经保证。未注入注册表时的分支不变。

### 4.4 组件的请求参数

组合根的 `component_llm_params` 接收组件配置与注册表，组合根分别传入 `config.gateway.llm` 与 `config.patchouli.llm`：`model_id` 为 `null` 时返回 `None`；否则以组件配置的 `temperature`、`max_tokens` 为覆盖值调用 `resolve`，返回其中的请求参数。解析结果不写回配置对象，与现状相同。Gateway 与 Patchouli 继续只接收注入的请求参数，不读取 `llm` 段，也不持有注册表。

候选：组件请求从此带上模型定义的 `top_p`，与 Alice 一致，不再特别处理。这是一处行为变化：现在组件不发送 `top_p`，由提供商取默认值。仓库中唯一的模型使用默认值 1.0，对现有部署没有可见影响。另一种做法是组件继续不发送 `top_p`，代价是在这里多一步改写请求参数。

### 4.5 Gateway 分析器

`TopicRouterConfig` 与 `UserQueryAnalysisConfig` 删除 `model_override`。`json_analysis_params` 只为 Gateway 的请求参数追加 JSON 输出要求，两个分析器使用同一个 Gateway 模型。

### 4.6 配置加载与环境变量

- `shared.llm` 迁到了别处，按 `retrieval.retriever.reranker` 的先例显式拒绝：`SharedConfig` 在校验前检查到 `llm` 时加载失败，错误指出两段的新位置。
- 组件配置中残留的 `model`、`api_base`、`provider`、`context_window`、`top_p`，以及 `gateway.topic_router` / `gateway.user_query_analysis` 的 `model_override`，都是删除而不是迁移，由配置段拒绝未知键而加载失败，错误带字段路径；迁移方式写在 [System 配置](../system/configuration.md)的升级注意中。配置文件中非空的 `api_key` 仍先由已有的密钥检查拒绝，错误指出密钥应在的位置。
- 删除第 3 节第 4 条中的 8 条环境变量别名；`HIVEMEMORY__LLM__GATEWAY__TEMPERATURE` 等 4 条改为指向新位置（`gateway.llm` 与 `patchouli.llm`）。嵌套形式的旧变量（`HIVEMEMORY__SHARED__LLM__...`）落到 `shared.llm`，同样以迁移错误失败。
- 删除 `get_librarian_llm_config` / `get_gateway_llm_config`。
- 主配置接口的密钥机制不变，说明中的“LLM 与 Qdrant”改为只有 Qdrant。
- 随仓库发布的 `configs/config.yaml` 把两段移到 `gateway.llm` 与 `patchouli.llm`，注释随之改写。

### 4.7 前端引擎设置页

- 前端的 `HiveMemoryConfig` 按后端结构描述引擎页用到的字段：`gateway.llm` 与 `patchouli.llm`（`model_id`、`temperature`、`max_tokens`），`shared.embedding.default`（`model_name` 等后端字段）；顶层的 `llm` 与 `embedding` 删除。
- `EngineSettings` 读写 `gateway.llm.*`、`patchouli.llm.*` 与 `shared.embedding.default.model_name`；模型下拉框的说明改为“保存后重启生效”。
- `validateConfig` 只校验后端存在的字段：两段 LLM 的 temperature 改用新路径；针对后端不存在的段的校验（`qdrant`、`gateway.analyzer` 等）删除，否则真实配置仍会让校验抛出。
- 候选：真实配置能够加载之后，仍按旧结构读取的分类一打开就会在渲染时报错。因此 `SettingsPanel` 对这些分类（Infrastructure、Retrieval、Generation、Lifecycle、Perception、Koakuma，以及子结构不同的 Gateway）显示“该分类尚未与后端配置结构对齐”的提示，不渲染表单；General、引擎、Provider 与模型页照常渲染。现在这些分类显示的是 mock 数据，看起来正常、保存却会失败；改为提示之后表达的是真实状态。另一种做法是一次对齐全部分类，范围明显更大，宜另立计划。
- `MOCK_CONFIG` 中引擎页使用的字段按新结构更新。

## 5. 受影响的系统、契约和数据

- **配置契约**：`shared.llm` 删除，两段移到 `gateway.llm` 与 `patchouli.llm`，且只剩三项；`gateway.topic_router` / `user_query_analysis` 的 `model_override` 删除；环境变量别名删除 8 条、改向 4 条。`GET /api/v1/config` 与 `GET /api/v1/config/defaults` 的响应随配置模型变化，消费方只有前端设置页。
- **System**：模型注册表、组合根的组件参数、配置加载与主配置接口的说明。
- **agent_runtime / Alice**：`ModelResolver` 端口的返回类型。System 的模型注册表实现该端口，由组合根注入。
- **Gateway / Patchouli**：配置段各多一个 `llm`；Gateway 两个分析器与 `json_analysis_params`。两个子系统的构造参数不变。
- **前端**：设置页的类型、校验、引擎页与分类渲染。
- **数据**：不迁移持久化数据；公开路由、事件与错误码不变。

## 6. 兼容与迁移

- 随仓库发布的 `configs/config.yaml` 在同一改动中移动两段。
- 部署的 `config.yaml`：把 `shared.llm.gateway` 移到 `gateway.llm`、`shared.llm.librarian` 移到 `patchouli.llm`，否则启动时以迁移错误失败。两段中若写了 `model`、`api_base` 等键，先在 `models.yaml` 登记模型、在 Provider 页面或 `.env` 保存凭证，再改为 `model_id`。写了 `model_override` 的删除即可；需要不同的分析模型时，改 `gateway.llm.model_id`。
- 旧式环境变量 `HIVEMEMORY__LLM__GATEWAY__MODEL` 等删除别名之后被静默忽略：根配置对环境变量宽松，其中本来就有不属于配置树的变量。当前没有部署使用这些变量。
- 第 4.4 节的 `top_p` 行为变化。
- 前端：主配置页从显示 mock 数据，变为加载真实配置；未对齐的分类显示提示。

## 7. 测试和验收

| 不变量 | 归属层 |
|:---|:---|
| 注册表解析：`default` 与 ID 选模型、覆盖值优先于模型默认值、凭证只来自 Provider、找不到时 `ModelNotFoundError`、返回所选的模型定义 | `tests/unit/system/test_model_registry.py` |
| 组件请求参数：`model_id` 为 `null` 时没有请求参数；组件的 temperature 与 max_tokens 覆盖模型默认值；两段分别取自 `gateway.llm` 与 `patchouli.llm`；解析不写回配置对象 | `tests/unit/system/test_assembler.py` |
| `shared.llm` 残留时以指出新位置的错误加载失败；组件配置拒绝已删除的键（一个代表值）；配置文件中非空的 `api_key` 仍由密钥检查拒绝 | `tests/unit/system/test_config_agent_runtime.py` |
| 旧式别名与嵌套环境变量的优先级：原 LLM 用例改用保留的别名与新位置 | `tests/unit/system/test_config_agent_runtime.py` |
| Alice 帧模型的请求参数、输入上限与展示名来自解析结果 | `tests/unit/agent_runtime/test_runtime.py`（随端口改写现有用例与 `tests/helpers/alice.py` 的替身） |
| Gateway 分析请求追加 JSON 输出，沿用 Gateway 的模型与凭证 | `tests/unit/engines/gateway/test_query_understanding.py` |
| 主配置接口的响应打码、保存时沿用密钥、写入文件时剔除密钥（只剩 Qdrant 的密钥） | `tests/unit/server/routers/test_config.py`（改写现有用例） |

随契约删除的旧测试：没有 `model_id` 时原样返回、`to_llm_config`、新旧 LLM 环境变量的优先级、`model_override` 只替换模型、`component_llm_params(LLMConfig(model=None))`，以及配置文件中 gateway 段写 `api_key: ''` 照常加载的那部分断言。

```bash
pytest tests/unit/system tests/unit/agent_runtime tests/unit/engines/gateway tests/unit/server/routers -q
pytest tests/integration/patchouli/test_generation_topic_extraction.py --collect-only -q
pytest tests/unit tests/integration -m "not live_llm and not e2e and not slow" -n auto --cov=hivememory --cov-fail-under=85
ruff check . && black --check . && uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
cd frontend && npm run lint && npm run build
```

前端没有设置页的自动化测试。实施后以 `scripts/hivememory-dev.sh` 启动后端与 Vite，在浏览器中核对：设置页加载的是真实配置；引擎页显示两段当前的模型与生成参数，修改后保存，`config.yaml` 中 `gateway.llm` 与 `patchouli.llm` 的对应字段改变，重启后生效；未对齐的分类显示提示而不报错。

## 8. 风险与待决问题

- `ModelResolver` 是 System 与 agent_runtime 之间的端口，返回类型变化需要同时改写 Alice 的替身与现有测试；端口的参数不变，调用方只有 `resolve_frame_model`。
- 本计划不处理的已知问题：
  - 旧式环境变量名被静默忽略（第 6 节）；
  - `configApi.ts` 的 `fetchFoldTokenThreshold` 读取后端不存在的 `config.perception`，且没有调用方；
  - General 页读取的 `system.version` 已从后端配置删除，加载真实配置后显示为空。

## 9. 完成时必须更新的当前文档

- [System 配置与注册表](../system/configuration.md)：第 1 节配置树中 `shared`、`gateway`、`patchouli` 的说明，第 2 节的兼容别名说明，第 3 节的解析步骤，第 4 节的密钥与组件配置说明，并加入升级注意；
- [System 组合根](../system/composition.md)第 1.3 节：组件请求参数的来源与 `model_id: null` 的含义；
- [Agent Runtime](../alice/agent-runtime.md)第 4 节：`ModelResolver` 返回请求参数与模型定义；
- [Gateway 话题与查询分析](../gateway/analysis.md)：请求参数改由 `gateway.llm` 转换，删除 `model_override` 的说明；
- [Patchouli 生成](../patchouli/generation.md)与[感知与短期话题](../patchouli/perception.md)：Librarian 的请求参数改由 `patchouli.llm` 转换；
- [配置指南](../help/configuration.md)：环境变量别名与组件模型选择；
- [管理页面](../frontend/management-views.md)第 3.3 节：引擎页已对齐，其余分类的提示；
- [ROADMAP](../ROADMAP.md)、[Plans 导航](./README.md)与归档索引中本计划的状态。
