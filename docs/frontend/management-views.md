---
title: Frontend Management Views
status: current
owner: frontend
scope: memory-agent-and-settings-management
code_paths:
  - frontend/src/components/MemoryLibrary.tsx
  - frontend/src/components/memory/
  - frontend/src/components/AgentManagement.tsx
  - frontend/src/components/agent/
  - frontend/src/components/SettingsPanel.tsx
  - frontend/src/components/settings/
  - frontend/src/hooks/
  - frontend/src/services/
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/mtp.md
last_reviewed: 2026-11-01
---

# 前端管理页面

管理页面是人对 HiveMemory 的纠错和配置入口。它延续“记忆是资产”的设计：用户不仅能够看到最终回答，还应能检查正式记忆、修正错误内容、管理 Agent 图纸，并了解系统实际使用的 Provider 和模型。与此同时，人工干预必须尊重后端所有权；一个看起来像管理按钮的控件，若没有可验证的后端语义，就不能被描述为已完成能力。

## 1. Memory Library

### 1.1 当前能力

Memory Library 首次加载调用 `GET /api/v1/memories?limit=100`，随后在浏览器中完成搜索、类型过滤、排序和 grid/list 切换。卡片和列表展示 title、alias、类型、摘要、标签、confidence、vitality、访问次数和时间；详情弹层可查看正文并修改 title、summary、content、alias 与 tags。

用户还可以：

- 通过 `POST /api/v1/memories` 手动创建普通 MemoryAtom；
- 通过 `PATCH /api/v1/memories/{id}` 修改上述可编辑字段；
- 通过 `DELETE /api/v1/memories/{id}` 删除记忆；
- 从 Chat 的引用记忆面板提交 feedback。

创建菜单不提供 `AGENT_PROFILE` 与 `TOOL`，避免把专业 Profile 和工具资产当成普通文本记忆随意播种。Agent Profile 由独立页面管理。

### 1.2 花园隐喻的当前落点

卡片视图强调记忆作为可生长、可检查的独立资产，列表视图服务于高密度浏览；alias 提供可寻址身份，confidence/vitality 与访问次数帮助读者理解一条记忆的可信度和生命力。这个“花园”不是统计大屏的宣传概念，而是把自动生成的知识重新交给人检查和修正的界面原则。

### 1.3 当前偏差

- 所谓 semantic 模式只是对已加载 100 条记录执行 title/summary/content/tag substring matching，没有调用后端 embedding retrieval；
- 没有分页，页面的 `total` 是客户端过滤后的数量，不是完整库总数；
- `statusFilter` 已存在于 store，但过滤计算没有使用它；`selectedTags` 也没有对应 CommandBar UI；
- Pin 按钮只显示“施工中”，没有免疫 GC 或锁定语义；没有 archive/revive 操作和统计大屏；
- delete 先乐观移除，API 失败后不回滚，可能让页面暂时偏离后端；
- 读取失败会静默改用 `MOCK_MEMORIES` 并清除页面错误，用户可能把演示数据误认为真实记忆；在 mock 状态下继续编辑、创建或删除仍会请求后端并失败。

因此旧 Memory Garden 设计中的真正语义检索、Pin/Lock、归档筛选和完整统计仍是设想，不属于当前事实。

## 2. Agent Management

Agent Profile 是存储在 Patchouli 中的 `AGENT_PROFILE` 记忆，也是 Alice 的“人偶图纸”。前端管理页负责把结构化 Profile 投影为可编辑表单，而不重新定义 Agent Runtime 的权限语义。

### 2.1 当前能力

- 读取、搜索 Agent Profile；
- 创建本地草稿，再通过 `POST /api/v1/agents` 创建 Profile；
- 编辑 title、alias、summary、tags、persona/system prompt；
- 编辑 model name、temperature、language；
- 通过通用 memory `PATCH` / `DELETE` 更新和删除已存在 Profile。

后端 agents router 当前只提供 create/list；更新和删除依赖 generic memories API。这一组合成立的前提是 Agent Profile 继续保持 MemoryAtom 的一种，而不是另建平行实体。

### 2.2 权限与状态限制

- 管理页不编辑权限：Agent Profile 不承载权限，Agent 的操作权限只由 Workspace 访问登记的 actor 操作白名单表达，`sys_*` 系统工具不设权限；
- Active/Inactive 只是前端字段，`toPayload()` 不发送 status，启停按钮没有后端运行语义；
- model 是自由文本，而不是 Model Registry 选择器，可能保存不存在的 model ID；
- Profile 获取失败会回退到 `MOCK_AGENT_CONFIGS`，这些条目看似可编辑，但保存仍需要真实后端；
- Chat 消息中的 Agent 名称和图标有一部分仍依赖静态 `MOCK_AGENTS`，不完全反映 Profile 元数据。

权限的后端契约与 runtime 闸门以 [MTP 契约](../contracts/mtp.md)第 4 节和 [System 配置](../system/configuration.md)的访问登记为准。

## 3. Settings

Settings 当前包含通用、Provider、Model Registry、内部引擎、基础设施、Gateway、Perception、Generation、Retrieval、Lifecycle 与 Koakuma 分类。它们实际分成三种不同的管理路径。

### 3.1 Provider Registry

Provider 页面通过独立 `/api/v1/providers` API 管理 `configs/providers.secrets.yaml`。API key 只以脱敏形式返回；环境变量注入的 Provider 优先级最高，并在 UI 中标为只读，不能通过页面删除或覆盖。YAML 层写入采用原子替换。

### 3.2 Model Registry

Model 页面通过 `/api/v1/models` 管理 `configs/models.yaml`，支持创建、查询、更新、删除和默认模型标记。模型的密钥与 API 地址只来自所选 Provider 的凭证，页面与 API 都不读写模型级的 api key 或 api base；单个模型需要不同的密钥或地址时，在 Provider 页面另登记一个 Provider 再选择它。

### 3.3 主配置表单

主配置页面通过 `GET/POST /api/v1/config` 读取和原子写入 YAML，后端使用 Pydantic 重新校验，并替换 `system.config`。主配置接口不读写密钥：响应中的 `api_key` 只有打码值，保存时沿用当前密钥，写入的 YAML 不含 `api_key`；Provider 凭证由 Provider 页面管理。但“配置对象已替换”不等于所有已装配组件已经热重载；Qdrant 连接、模型服务和部分 runtime 仍需要重启。

与后端配置结构一致的只有 General 与“内部引擎”两个分类。内部引擎读写 `gateway.llm`、`patchouli.llm` 与 `shared.embedding.default.model_name`：两个组件各有一份模型选择（`ModelSelection`），模型从模型注册表中选择（另有“注册表默认模型”一项），temperature、top_p 与 max_tokens 可以留空表示沿用模型默认值，保存后重启生效；配置为 `null`（不使用 LLM）时只显示提示，需要在 `configs/config.yaml` 中改回，因为页面不知道各组件的默认生成参数，从页面重新启用会丢掉它们。 General 读取的 `system.version` 已从后端配置删除，显示为空。

其余分类（基础设施、全局网关、记忆感知、记忆生成、记忆检索、生命周期、MTP 引擎）的表单仍按旧的前端配置结构读写，后端配置中没有对应的段或子结构不同，渲染真实配置会报错；页面对这些分类只显示“尚未与后端配置结构对齐”的提示，不渲染表单，需要修改时直接编辑 `configs/config.yaml` 并重启。加载时的校验同样只检查与后端一致的字段，读取不存在的段会让页面退回 mock 数据。`ConfigResponse` 返回的段也不是完整配置：`workspace`、`memory_compiler`、`import_bus` 与附件配置不在响应中，保存时由当前的 `config.yaml` 补齐。Provider 与 Model Registry 使用独立 API，不受这一具体类型偏差影响。

后端不可达时，主配置页会加载 `MOCK_CONFIG` 并保留 error；页面仍可能显示完整表单，但保存会失败。Settings 草稿只存在内存中，不会跨刷新恢复。

## 4. 人工干预不变量

- 页面展示的 Memory、Agent、Provider、Model 与 Config 必须能追溯到明确后端来源或显式标记的 mock；
- 乐观更新失败时必须恢复或显著提示失配，不能让“界面消失”冒充“后端删除成功”；
- Agent persona 与权限字段分离，权限以 runtime 校验为准；
- API key 不在列表响应中返回明文，不把密钥写进受版本控制的配置或模型文件；
- 配置保存与配置生效是两个阶段，只有被重新装配或动态读取的组件才能宣称即时生效；
- 设计稿中的 Pin、Memory Library 的 Archive、统计、状态切换或 semantic search，只有后端契约和失败语义同时落地后才可进入当前能力表；Topic 生命周期使用 `settle`，不在此处复用 `archive`。

## 5. 代码与测试入口

| 页面 | 前端入口 | 后端验证 |
|:---|:---|:---|
| Memory Library | `hooks/useMemories.ts`、`components/memory/`、`services/memoryApi.ts` | `tests/unit/server/routers/test_memories.py` |
| Agent Management | `hooks/useAgents.ts`、`useAgentManagement.ts`、`components/agent/` | `test_agents.py`、Patchouli agent profile tests |
| Settings | `hooks/useSettings.ts`、`components/settings/`、`services/configApi.ts` | `test_config.py` |
| Provider Registry | `ProviderSettings.tsx`、`providerRegistryApi.ts` | `test_providers_router.py`、`test_provider_registry.py` |
| Model Registry | `ModelRegistrySettings.tsx`、`modelRegistryApi.ts` | `test_models_router.py`、`test_model_registry.py` |

配置文件、优先级和安全操作步骤见 [Help 配置指南](../help/configuration.md)。
