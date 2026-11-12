---
title: Frontend Management Views
status: current
owner: frontend
scope: memory-library-topic-pool-agent-and-settings-management
code_paths:
  - frontend/src/components/MemoryLibrary.tsx
  - frontend/src/components/memory/
  - frontend/src/stores/memory/viewStore.ts
  - frontend/src/stores/topic/topicStore.ts
  - frontend/src/components/AgentManagement.tsx
  - frontend/src/components/agent/
  - frontend/src/components/SettingsPanel.tsx
  - frontend/src/components/settings/
  - frontend/src/constants/settingsSchema.ts
  - frontend/src/hooks/
  - frontend/src/services/
  - frontend/src/utils/memoryVisibility.ts
  - src/hivememory/server/models/memory.py
  - src/hivememory/server/models/agent.py
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/mtp.md
last_reviewed: 2026-11-11
---

# 前端管理页面

管理页面是人对 HiveMemory 的纠错和配置入口。它延续“记忆是资产”的设计：用户不仅能够看到最终回答，还应能检查正式记忆与尚未结算的短期话题、修正错误内容、管理 Agent 图纸，并了解系统实际使用的 Provider 和模型。与此同时，人工干预必须尊重后端所有权；一个看起来像管理按钮的控件，若没有可验证的后端语义，就不能被描述为已完成能力。

## 1. Memory Library

记忆图书馆以“长期记忆”与“短期话题池”两个标签切换，标签上带条数（库内记忆总数与驻留话题数）。选中的标签与类型筛选、排序、视图模式一起作为视图偏好持久化在 `memory-view-store` 中。页头在两个标签下相同：库内记忆总数（已加载的条数少于总数时一并提示）与“注入记忆”按钮。

### 1.1 长期记忆

长期记忆加载时逐页调用 `GET /api/v1/memories?limit=100`，跟随响应的 `next_cursor` 取完整个记忆库（不含 Agent Profile）。后端按存储顺序翻页，顺序与写入时间无关，只取一页会得到与时间无关的子集，所以必须取全后由浏览器排序。页头的总数取自响应的 `total`。

搜索分两种模式：

- **语义检索**：停止输入 400 毫秒后，以 `query` 调用同一接口，在整个记忆库中返回按相关度排列的至多 50 条。结果保持相关度顺序，排序选项对它不生效；页面分别显示检索中、检索失败与没有结果。
- **本地过滤**：只匹配已加载记忆的标题、摘要、正文与标签，`alias:`、`tag:` 前缀把匹配限定在对应字段。

类型过滤、排序和 grid/list 切换都在浏览器中完成。卡片与列表项用类型图标块（`MemoryTypeTile`）标出类型：类型色只上在图标与它的浅底上，类型名是中性文字，卡片背景与边框不带类型色；状态在前端总是圆点加文字，两者靠形状区分，类型色不会被读成状态。图标块旁是类型名与别名，没有别名时显示 `mem_` 加 ID 的前 8 位。卡片高度随内容，标题与摘要限制行数，最多显示三个标签；底部的“生命力”“置信度”带文字标签、悬停说明含义，与时间一起始终可见。列表视图在一行内放同样的信息，只显示两个标签。只对一个 Agent 可见的记忆，卡片在标题行右侧、列表项在标题之后显示锁形图标与这个 Agent 的 agent_id（不加“只有”之类的前缀），悬停说明含义；卡片上的标记与悬停时出现的操作按钮共用右上角，悬停或聚焦时让位；指向的 Agent 已不在候选列表中时标记改用警示色（第 1.1.1 节）。时间取最近更新：一周以内显示相对时间（“3 分钟前”），更早的按统一格式（今天只显示时分，今年显示月日与时分，更早的显示完整日期），悬停显示完整时间（`utils/dateTime.ts`）。

详情弹层分两栏。头部是类型图标块、标题、可复制的别名与摘要，右上角依次是编辑、Pin、删除与关闭，都有悬停提示，删除与其余按钮以分隔线隔开。左栏是 Markdown 渲染的正文（可复制）；右栏依次是元数据（可复制完整 ID，置信度与生命力带短条与悬停说明，被引用次数）、时间线（创建与最近更新）、标签与可见范围。编辑在原位修改 title、summary、content、alias、tags 与可见范围，编辑中按 Esc 等同“取消”。

卡片与列表项可用 Tab 到达，Enter 或空格打开详情；卡片上的编辑与删除按钮在悬停或键盘聚焦时出现。类型筛选与排序菜单在悬停或键盘聚焦时展开。详情与“注入记忆”对话框打开时焦点留在其中，Esc 关闭。

用户还可以：

- 通过 `POST /api/v1/memories` 手动创建普通 MemoryAtom；
- 通过 `PATCH /api/v1/memories/{id}` 修改上述可编辑字段；
- 通过 `DELETE /api/v1/memories/{id}` 删除记忆：删除需要二次确认，页面先移除，后端删除失败时放回并提示；
- 在对话页的引用记忆面板与消息的反馈按钮上提交 feedback（[Chat 视图](./chat-view.md)）。

创建菜单不提供 `AGENT_PROFILE` 与 `TOOL`，避免把专业 Profile 和工具资产当成普通文本记忆随意播种。Agent Profile 由独立页面管理。

#### 1.1.1 可见范围

可见范围是后端 `MemoryAccessPolicy` 在管理页的编辑入口，也是用户修改记忆可见性的唯一入口：被动提取、Agent 的写入与 Agent Profile 都生成所有 Agent 可见的记忆（[Patchouli 生成](../patchouli/generation.md)）。页面只呈现两种：所有 Agent（`PUBLIC`）与只有一个 Agent（`PRIVATE`，目标是 agent_id）；管理接口不接受 `TEAM`，生产中也没有 Team 归属的来源。

- **含义**：可见范围只约束 Agent 的检索与读取（chat 的预检索、MTP 与 MCP），不授予任何操作，管理页始终能看到全部记忆。它也不是保密：被动结算的查重看不到只对某个 Agent 可见的记忆，之后的对话里再提到同一件事，仍可能提取出一条所有 Agent 可见的新记忆，两条的策略不同也不会合并。这段说明在“注入记忆”中直接显示，在详情中放在悬停提示里。
- **控件**：创建与详情编辑共用一个下拉框（`MemoryVisibilitySelect`）：第一项“所有 Agent”，其后按“外部接入”“对话 Agent”两组直接列出 agent_id。选中即确定目标，没有选了“只有一个 Agent”却没有目标的中间状态，不需要额外的必填校验。详情的可见范围是标签下面单独一张卡片，查看时显示值与一句解释，编辑时换成下拉框。
- **候选 Agent**：取当前 Workspace 访问记录中的具体 agent_id（不含用户级记录与保留的 `system`）和 Agent 列表（含内置的 omni_doll），有 Profile 的归入对话 Agent（`hooks/useVisibilityAgents.ts`、`utils/memoryVisibility.ts`）。两个来源都是已有接口；后端不核对目标是否存在。
- **目标不在候选中**：Agent 已删除或改名时，下拉框把原目标单独列在最前，用户不改动即原样保留。只有两个来源都读取成功时才判定为“已不在列表中”：详情只显示警示，说明现在没有 Agent 能检索到这条记忆；卡片与列表的标记改用警示色。任一来源读取失败时不作判断，避免误报。
- **提交**：创建默认所有 Agent，请求总是带 `access_policy`。详情保存时只在可见范围改变时带上 `access_policy`；后端在内容提交之后单独替换策略，不产生新版本、不改更新时间（[MemoryLibrary](../patchouli/memory-library.md#12-中期当前可检索书库)）。策略写入失败时内容已经保存，请求按失败提示，重新保存会补上策略。Agent Profile 的策略不经这里修改，后端拒绝对 Profile 提交 `access_policy`。

### 1.2 短期话题池

短期话题是 Patchouli 中尚未结算的话题：对话与导入中相关的几轮聚成一个话题暂存，话题结算时由记忆生成从中提炼长期记忆，话题随之离开话题池。话题的建立、折叠、空闲结算与 LRU 驱逐都由 Patchouli 负责（[感知与短期话题](../patchouli/perception.md)）；话题池只读取快照、发起手动结算与丢弃，并把自动结算的规则写给人看，不自行推断话题的生命周期。

**读取**：进入记忆图书馆与点工具栏的刷新时，以 `GET /api/v1/topics` 经 root API 读取当前 Workspace 的驻留话题，其他时候不自动刷新。每个话题带标题、话题摘要（`topic_summary`）、进展摘要（`state_summary`）、最近一轮的用户消息与回复（`last_turn`）、片段数、tokens、所用模型与最近一次写入的时间。读取失败时显示原因，连接类的原因由页面顶部的连接状态条说明。

**规则**：容量、空闲时长与折叠阈值经 `GET /api/v1/config` 从主配置读取，分别是 `patchouli.perception.engine.max_resident_topics`、`patchouli.perception.idle_timeout_seconds`、`patchouli.perception.engine.fold_token_threshold` 与 `fold_retain_recent_blocks`。四项中任一读不到时，页面只显示话题本身，工具栏、上下文与自动结算都不写具体数值。

**布局**：

- **工具栏**：标签旁显示驻留数与容量（容量不超过 10 时画成格子），写明自动结算的规则（“话题空闲 N 分钟，或池满需要腾出位置时，会自动结算为长期记忆”），另有刷新按钮。
- **列表**：每个话题一行，写标题（没有时为“未命名话题”）、进展摘要（没有时用话题摘要）、片段数、tokens 与最近写入的相对时间；预计十分钟内自动结算的话题以警示色标出。
- **详情**：选中的话题（默认第一个）的标题、可复制的话题 ID、片段数、tokens 与模型。正文是“在聊什么”（话题摘要）、“目前进展”（进展摘要）与“最近一轮”（用户消息与回复），没有空格的长串也会换行。侧栏是上下文用量（对比折叠阈值，超过后较早的片段折叠成摘要，只保留最近几个片段的原文，[Page Folding](../patchouli/perception.md#5-page-folding)）、自动结算的估算与基本信息。
- **自动结算的估算**：以最近一次写入加空闲时长估算剩余分钟，每分钟刷新。后端以话题最近一次被确认或写入的时间计时并由周期扫描结算，实际结算可能稍晚，话题再次写入后重新计时（[维护与关闭](../patchouli/perception.md#7-维护与关闭)）。
- **窄窗口**：宽度在 1280px 以下时列表由 400px 收窄到 300px，信息卡排到正文之后、与正文一起滚动。
- **空池**：说明话题从哪里来、何时离开，并给出“去对话”。

**操作**：

- **结算为长期记忆**：`POST /api/v1/topics/{id}/settle`，即 Patchouli 的手动结算：冻结话题的全部材料交给记忆生成，被接纳后删除话题（[具名用例](../patchouli/perception.md#4-触发原因与具名用例)）。成功提示写明生成进度在对话页右栏的“记忆任务”中查看。
- **丢弃**：二次确认写明话题中的片段会被删除、不会提炼长期记忆、无法恢复，对话中已经写入的记忆不受影响；确认后 `DELETE /api/v1/topics/{id}`，只驱逐、不结算。
- 结算与丢弃都先从列表中移除，失败时放回并以 toast 提示原因。
- **在对话中引用**：把这个话题设为对话状态中的指定话题（chat store 的 `designatedTopic`）并切到对话页，输入框上方出现这个话题的 `#` 引用，下一轮请求带上它的 `topic_id`（[Chat 视图](./chat-view.md)）。
- **从对话页跳入**：对话页消息底部的话题仍在池中时可以点击，记忆图书馆切到短期话题池并选中它。

### 1.3 花园隐喻的当前落点

卡片视图强调记忆作为可生长、可检查的独立资产，列表视图服务于高密度浏览；alias 提供可寻址身份，生命力、置信度与被引用次数帮助读者理解一条记忆的可信度和生命力。短期话题池把尚未成为资产的上下文也交给人看：可以提前结算、丢弃不值得保留的话题，或在对话中指定继续某个话题。这个“花园”不是统计大屏的宣传概念，而是把自动生成的知识重新交给人检查和修正的界面原则。

### 1.4 当前偏差

- 加载时取全整个记忆库，记忆数到数千条时首次加载会变慢；页面只在进入和创建记忆后重新加载，后台生成的新记忆要重新进入页面才会出现（响应速度的后续处理见 [Todo](../todo/frontend-ui-polish.md)）；
- `selectedTags` 参与过滤，但没有对应的 CommandBar UI；
- 卡片、列表与详情中的 Pin 按钮一律禁用，提示“Pin 尚未提供”，没有免疫 GC 或锁定语义；没有 archive/revive 操作和统计大屏；页头的“冲突警告”计数固定为 0，没有数据来源；
- 读取失败时清空列表并显示错误（连不上后端或管理凭据失效时只写“数据暂不可用”），只有 debug 模式改用 `MOCK_MEMORIES`（[状态与传输](./state-and-transports.md#6-mock-降级边界)第 6 节）；debug 模式下继续编辑、创建或删除仍会请求后端并失败；
- 话题池读到的是配置接口返回的期望配置：容量与折叠阈值保存后要重启才按新值运行，在此之前页面显示的已是新值；话题池不随后台的结算与新建话题自动刷新。

因此旧 Memory Garden 设计中的 Pin/Lock、归档筛选和完整统计仍是设想，不属于当前事实。

## 2. Agent Management

Agent Profile 是存储在 Patchouli 中的 `AGENT_PROFILE` 记忆，也是 Alice 的“人偶图纸”。前端管理页负责把结构化 Profile 投影为可编辑表单，而不重新定义 Agent Runtime 的权限语义。

### 2.1 页面与当前能力

左栏（300px）是标题与说明、搜索与新建，搜索匹配名称、描述、别名与标签；列表每行写图标、名称、别名与状态：启用、已停用或草稿（尚未保存）。右侧按卡片编辑选中的 Agent：

- **身份**：图标、名称、别名与一句话描述；右上是启用开关、“撤销更改”与“保存”。停用时下方出现说明条。
- **模型**：模型从模型注册表中选择，另有“模型注册表的默认模型”一项（保存为 `default`），保存的仍是 `agent_config.model_name`。当前值不在注册表中时仍列出并标注“不在注册表中”，提示对话时会解析失败；读不到注册表时不做这个判断。温度用滑块调整。Top P 可以留空，留空即沿用模型定义，拖动滑块即为这个 Agent 单独设置，也可以改回沿用；保存为 `agent_config.top_p`，留空为 `null`。
- **系统指令**：人设与行为准则，带字数。
- **标签**。
- **操作权限**：写明权限不在这里设置，按钮跳到设置页的“接入与权限”（第 3.3 节）。
- **底部**：创建与最近保存的时间，取自 Agent 列表接口返回的 `created_at`、`updated_at`（Profile 记忆的元数据）；保存成功后以保存的时刻显示“最近保存”，尚未保存的草稿写“尚未保存”。删除 Agent 需要二次确认。

提示词语言（`agent_config.language`）不在页面上编辑，保存时沿用图纸中的值，缺省为 `zh`。

新建在本地生成一个启用的草稿，保存时通过 `POST /api/v1/agents` 创建 Profile；已存在的 Profile 通过通用 memory `PATCH` / `DELETE` 更新和删除，`agent_config` 原样透传。后端 agents router 当前只提供 create/list，这一组合成立的前提是 Agent Profile 继续保持 MemoryAtom 的一种，而不是另建平行实体。内置的默认 Agent 由后端代码提供，没有 Profile 记忆，不出现在 Agent 页，对话页仍可以选择它。

### 2.2 启用与停用

`enabled` 写在 Profile 记忆的 `payload.agent_config` 中，缺省为启用，没有这个字段的已存 Profile 读作启用。它只表示这个 Profile 能否用于对话：停用的 Agent 不出现在对话页的 Agent 选择与 `@` 提及中，已有消息里仍显示它的名称；以它发起对话，或经 CALL 调用它时，后端的 Profile 解析以 `profile_disabled` 拒绝，CALL 的调用方看到的是“当前不可用”（[Patchouli 检索](../patchouli/retrieval.md)、[Chat 视图](./chat-view.md)）。

`enabled` 不是操作权限，也不改变访问登记；plugin 模式的 harness 不使用 Profile，不受它影响。

### 2.3 权限与状态限制

- 管理页不编辑权限：Agent Profile 不承载权限，Agent 的操作权限只由 Workspace 访问登记的 actor 操作白名单表达，在 Settings 的“接入与权限”中编辑（第 3.3 节），`sys_*` 系统工具不设权限；
- 别名只在尚未保存的草稿中可以修改，保存后只读：对话、访问登记、会话记录与 CALL 都以别名指代这个 Agent，改名需要同时更新这些引用，见 [Todo](../todo/agent-alias-rename.md)；
- Profile 获取失败时列表为空，右侧显示原因：后端拒绝时是后端给出的原因，连不上后端或管理凭据失效时只写“数据暂不可用”；只有 debug 模式回退到 `MOCK_AGENT_CONFIGS`，这些条目看似可编辑，但保存仍需要真实后端。

权限的后端契约与 runtime 闸门以 [MTP 契约](../contracts/mtp.md)第 4 节和 [System 配置](../system/configuration.md)的访问登记为准。

## 3. Settings

设置页左侧导航分为“常用”“高级”两层与“管理”一组（提供商凭证、模型注册表、接入与权限）。常用与高级共用主配置接口，管理下的三类各有独立的 API。选中的分类作为视图偏好持久化，旧版本停留在已删除分类上的迁移为常用。

### 3.1 Provider Registry

Provider 页面（导航中的“提供商凭证”）通过独立 `/api/v1/providers` API 管理 `configs/providers.secrets.yaml`。API key 只以脱敏形式返回；环境变量注入的 Provider 优先级最高，并在 UI 中标为只读，不能通过页面删除或覆盖。YAML 层写入采用原子替换。

### 3.2 Model Registry

Model 页面通过 `/api/v1/models` 管理本机的生效文件 `configs/models.yaml`（从模板 `models.example.yaml` 生成，不受版本控制），支持创建、查询、更新、删除和默认模型标记。模型的密钥与 API 地址只来自所选 Provider 的凭证，页面与 API 都不读写模型级的 api key 或 api base；单个模型需要不同的密钥或地址时，在 Provider 页面另登记一个 Provider 再选择它。

### 3.3 接入与权限

接入与权限页面读写 `/api/v1/access` 的两组接口（[公开路由与事件](../contracts/routes-and-events.md#6-http-入口的身份交接与管理读取边界)第 6 节），修改立即生效并写回登记文件：

- 外部接入列出全部接入登记，进程内来源只读。可以登记外部 harness：登记后弹出只显示一次的凭据，并提示写进 harness 一侧的哪个环境变量；同一流程中可以为它可声明的 Agent 建立访问记录，这一步失败时来源已登记，页面提示补建。也可以启停、修改接入方式与可声明的 Agent、轮换凭据与删除；凭据来自环境变量的来源不能轮换。
- Agent 操作权限列出当前 Workspace 的访问记录（不含 `system`），用户级记录显示为“其余所有 Agent”。每条以准入开关与六项操作的勾选编辑，改动立即保存；关闭准入时一并清空操作，因为禁用的记录不能带白名单。可以从已有 Profile 中选择或直接填写 Agent 标识，新增精确记录。
- 修改被拒时以 toast 显示原因：登记冲突按 `reason` 换成可操作的提示，例如登记文件在运行中被手工修改过时提示重启后端。

### 3.4 常用与高级

常用与高级两页通过 `GET/POST /api/v1/config` 读取与保存，两者都返回 `config`（下次启动将装载的完整配置）、`immediate_fields` 与 `pending_restart`。后端用 Pydantic 重新校验，只把与默认值不同的字段写进个人覆盖文件 `config.local.yaml`，其中立即生效的字段（检索条数上限、相关度阈值、会话与附件上限等）同时写入运行中的服务，其余字段重启后生效（[System 配置](../system/configuration.md#21-个人覆盖与保存)）。主配置接口不读写密钥：响应中的 `api_key` 只有打码值，保存时沿用当前密钥，覆盖文件不含 `api_key`；Provider 凭证由 Provider 页面管理。

**字段表**：两页的设置项列在 `constants/settingsSchema.ts` 中，每一项绑定主配置的一个字段路径，有直白的名称与一行说明，部分小节标题旁带风格化的名称（如“帕秋莉”）。页面按字段路径读写配置，前端的配置类型不展开后端的配置结构，配置结构只在后端维护。

- **常用**：模型（Gateway 与帕秋莉各自的模型，从模型注册表中选择）、记忆、对话、附件与系统五节，另加外观一节。
- **高级**：模型（采样参数与本地模型的驻留）、检索、生成、话题与生命周期、对话与附件、运行时与存储六节；说明下方写出字段路径，便于对照 `config.local.yaml`；存储地址与集合名只读展示。

**生效标记**：每一项旁写明“立即生效”或“重启后生效”，取自配置接口的 `immediate_fields`，不在前端写死。保存后仍在 `pending_restart` 中的项标“等待重启”，页面顶部汇总“N 项设置等待重启”并列出名称，重启前服务仍按原值运行；改动过、尚未保存的项标“已修改”。模型字段的当前值不在注册表中时仍列出并标注“不在注册表中”；采样参数可以留空，表示沿用模型定义。

**保存**：常用或高级有改动时，底部出现保存栏，写明有几项未保存的更改及其名称，可以放弃或保存；有校验错误时不能保存。在设置页内切换分类时草稿保留；离开设置页前先确认，浏览器中刷新或关闭页面前也会提醒。“恢复本节默认值”在每个小节上，只把这一节的字段改回默认配置（`/api/v1/config/defaults`，不含个人覆盖），需要二次确认，恢复后仍需保存。

**外观**：常用页的外观一节选择深色、浅色或跟随系统的主题，只保存在这台电脑上、选择即生效，不进入主配置的保存（[状态与传输](./state-and-transports.md)）。

**失败**：读取失败时两页只显示“设置暂不可用”，连接类的原因由页面顶部的连接状态条说明。只有 debug 模式加载 `MOCK_CONFIG`（按后端的真实结构写，只含前端读取的字段）并保留 error，页面显示完整表单，但保存会失败。Settings 草稿只存在内存中，不会跨刷新恢复。“恢复本节默认值”读取默认配置失败时，以错误提示说明这一节没有恢复。

## 4. 人工干预不变量

- 页面展示的 Memory、Topic、Agent、Provider、Model 与 Config 必须能追溯到明确后端来源，mock 数据只在 debug 模式下出现；
- 乐观更新失败时必须恢复或显著提示失配，不能让“界面消失”冒充“后端删除成功”；
- Agent persona 与权限字段分离，权限以 runtime 校验为准；Agent 的启用只决定 Profile 能否用于对话，不是权限；
- 记忆的可见范围只编辑后端的 `MemoryAccessPolicy`：页面不从来源字段推导可见性，不把保留的 `system` 当作目标，管理读取也不受可见范围约束；
- API key 与接入凭据不在列表响应中返回明文，不把密钥写进受版本控制的配置或模型文件；
- 配置保存与配置生效是两个阶段，只有被重新装配或动态读取的组件才能宣称即时生效；页面的生效标记只取后端给出的 `immediate_fields` 与 `pending_restart`；
- 短期话题的结算与丢弃只经 root API 的手动用例，话题池显示的自动结算时间只是估算，不代表后端的结算计划；
- 设计稿中的 Pin、Memory Library 的 Archive、统计或状态切换，只有后端契约和失败语义同时落地后才可进入当前能力表；Topic 生命周期使用 `settle`，不在此处复用 `archive`。

## 5. 代码与测试入口

| 页面 | 前端入口 | 后端验证 |
|:---|:---|:---|
| Memory Library | `hooks/useMemories.ts`、`components/memory/`、`stores/memory/viewStore.ts`、`services/memoryApi.ts`；翻页取全由 `frontend/tests/memory-api.test.mjs`（`node --test`）验证，统一的时间格式由 `date-time.test.mjs` 验证；可见范围的候选合并与取值互换由 `memory-visibility.test.mjs` 验证 | `tests/unit/server/routers/test_memories.py`（含可见范围的透传与 TEAM 的拒绝）；可见范围的创建与修改由 `tests/unit/patchouli/services/test_external_edit_validation.py` 验证，经真实路由改策略后其他 Agent 读不到由 `tests/integration/patchouli/test_memory_management_edit_route.py` 验证；翻页、类型排除与计数经真实 Qdrant 内存模式的 `tests/integration/patchouli/test_memory_management_listing.py` |
| 短期话题池 | `components/memory/TopicPool.tsx`、`stores/topic/topicStore.ts`、`services/topicApi.ts`、`utils/topicPool.ts`；自动结算的估算与 tokens 的显示由 `frontend/tests/topic-pool.test.mjs` 验证 | `tests/unit/server/routers/test_topics.py` |
| Agent Management | `hooks/useAgents.ts`、`useAgentManagement.ts`、`components/agent/`、`services/agentApi.ts` | `test_agents.py`（含创建与保存时间）、停用的 Profile 由 `tests/unit/patchouli/services/test_profile_resolution.py` 验证、Patchouli agent profile tests |
| Settings | `hooks/useSettings.ts`、`components/settings/TieredSettings.tsx`、`constants/settingsSchema.ts`、`services/configApi.ts`；debug 模式的模拟配置为每个字段给出与控件相符的值，由 `frontend/tests/settings-mock-config.test.mjs` 验证 | `test_config.py`，其中校验字段表中的每条路径都存在于主配置 |
| Provider Registry | `ProviderSettings.tsx`、`providerRegistryApi.ts` | `test_providers_router.py`、`test_provider_registry.py` |
| Model Registry | `ModelRegistrySettings.tsx`、`modelRegistryApi.ts` | `test_models_router.py`、`test_model_registry.py` |
| 接入与权限 | `AccessSettings.tsx`、`accessApi.ts`；冲突提示与请求路径由 `frontend/tests/access-api.test.mjs` 验证 | `tests/unit/server/routers/test_access.py`、`tests/unit/system/access/test_registry.py`、`tests/unit/workspace/capability/root/test_access.py` |

配置文件、优先级和安全操作步骤见 [Help 配置指南](../help/configuration.md)。
