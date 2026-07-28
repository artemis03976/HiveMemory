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
last_reviewed: 2026-07-28
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
- 编辑 MTP verb 和五个 kernel syscall 白名单；
- 通过通用 memory `PATCH` / `DELETE` 更新和删除已存在 Profile。

后端 agents router 当前只提供 create/list；更新和删除依赖 generic memories API。这一组合成立的前提是 Agent Profile 继续保持 MemoryAtom 的一种，而不是另建平行实体。

### 2.2 权限与状态限制

- 前端 `MTPVerb` 列表没有 CALL，无法从 UI 为 Profile 配置已经由后端实现的 CALL；
- `null` 白名单表示全部允许。UI 从全权限关闭某一项时，会展开为静态“全部已知项减一”，因此静态清单遗漏会变成权限表达偏差；
- Active/Inactive 只是前端字段，`toPayload()` 不发送 status，启停按钮没有后端运行语义；
- model 是自由文本，而不是 Model Registry 选择器，可能保存不存在的 model ID；
- Profile 获取失败会回退到 `MOCK_AGENT_CONFIGS`，这些条目看似可编辑，但保存仍需要真实后端；
- Chat 消息中的 Agent 名称和图标有一部分仍依赖静态 `MOCK_AGENTS`，不完全反映 Profile 元数据。

权限的后端契约与 runtime 闸门以 [Alice 当前设计](../alice/README.md)和 [MTP Runtime](../alice/mtp-runtime.md)为准，不能由前端复选框反向定义。

## 3. Settings

Settings 当前包含通用、Provider、Model Registry、内部引擎、基础设施、Gateway、Perception、Generation、Retrieval、Lifecycle 与 Koakuma 分类。它们实际分成三种不同的管理路径。

### 3.1 Provider Registry

Provider 页面通过独立 `/api/v1/providers` API 管理 `configs/providers.secrets.yaml`。API key 只以脱敏形式返回；环境变量注入的 Provider 优先级最高，并在 UI 中标为只读，不能通过页面删除或覆盖。YAML 层写入采用原子替换。

### 3.2 Model Registry

Model 页面通过 `/api/v1/models` 管理 `configs/models.yaml`，支持创建、查询、更新、删除和默认模型标记。模型优先引用 Provider 凭证，只有展开高级覆盖时才把单模型 api key/api base 写入受版本控制的模型文件；实际运维应优先使用 Provider Registry，避免密钥进入 `models.yaml`。

### 3.3 主配置表单

主配置页面通过 `GET/POST /api/v1/config` 读取和原子写入 YAML，后端使用 Pydantic 重新校验，并替换 `system.config`。但“配置对象已替换”不等于所有已装配组件已经热重载；Qdrant 连接、模型服务和部分 runtime 仍需要重启。

更重要的是，当前前端 `HiveMemoryConfig` 仍采用旧的顶层 `llm/embedding/qdrant/perception/...` 结构，而后端已经使用 `shared/patchouli/alice/gateway/...` 嵌套树；`ConfigResponse` 也没有完整返回前端各分类所假设的字段。这意味着 General 以外的多组旧表单可能读取 undefined、显示错误或提交不完整配置。Provider 与 Model Registry 使用独立 API，不受这一具体类型偏差影响。

后端不可达时，主配置页会加载 `MOCK_CONFIG` 并保留 error；页面仍可能显示完整表单，但保存会失败。Settings 草稿只存在内存中，不会跨刷新恢复。

## 4. 人工干预不变量

- 页面展示的 Memory、Agent、Provider、Model 与 Config 必须能追溯到明确后端来源或显式标记的 mock；
- 乐观更新失败时必须恢复或显著提示失配，不能让“界面消失”冒充“后端删除成功”；
- Agent persona 与权限字段分离，权限以 runtime 校验为准；
- API key 不在列表响应中返回明文，不把密钥写进受版本控制的配置或模型文件；
- 配置保存与配置生效是两个阶段，只有被重新装配或动态读取的组件才能宣称即时生效；
- 设计稿中的 Pin、Archive、统计、状态切换或 semantic search，只有后端契约和失败语义同时落地后才可进入当前能力表。

## 5. 代码与测试入口

| 页面 | 前端入口 | 后端验证 |
|:---|:---|:---|
| Memory Library | `hooks/useMemories.ts`、`components/memory/`、`services/memoryApi.ts` | `tests/unit/server/routers/test_memories.py` |
| Agent Management | `hooks/useAgents.ts`、`useAgentManagement.ts`、`components/agent/` | `test_agents.py`、Patchouli agent profile tests |
| Settings | `hooks/useSettings.ts`、`components/settings/`、`services/configApi.ts` | `test_config.py` |
| Provider Registry | `ProviderSettings.tsx`、`providerRegistryApi.ts` | `test_providers_router.py`、`test_provider_registry.py` |
| Model Registry | `ModelRegistrySettings.tsx`、`modelRegistryApi.ts` | `test_models_router.py`、`test_model_registry.py` |

配置文件、优先级和安全操作步骤见 [Help 配置指南](../help/configuration.md)。
