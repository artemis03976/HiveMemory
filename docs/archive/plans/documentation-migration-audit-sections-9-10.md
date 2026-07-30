---
title: 文档迁移第 9～10 节逐篇审计：Gateway、Applications 与 Frontend
status: archived
owner: project
scope: documentation-migration-audit-sections-9-10
archived_at: 2026-07-29
superseded_by:
  - docs/archive/plans/documentation-migration-inventory.md
  - docs/DOCUMENTATION.md
source_inventory: docs/archive/plans/documentation-migration-inventory.md
---

# 第 9～10 节逐篇审计：Gateway、Applications 与 Frontend

本记录是第 9～10 节旧文档的审计与物理迁移门禁，不是新的 Gateway 或 Frontend 设计入口。审计逐篇回答四个问题：旧文档中的事实是否已经进入当前文档、哪些设计理念值得保留、哪些历史承诺必须明确拒绝、以及归档后读者应从哪里继续阅读。本批范围严格限定为清单第 9～10 节，不处理 `docs/mod/`、Ideas 或其他源码 README。

## 1. 审计口径

本批同时核对了当前代码、测试、跨子系统契约和现行文档：

1. **事实**以 `src/hivememory/gateway/`、`src/hivememory/patchouli/`、`src/hivememory/alice/`、`src/hivememory/system/application/`、`frontend/src/` 及其测试为准；
2. **理念**保留能够解释所有权、边界和取舍的背景，而不是只留下类名与文件清单；
3. **产品计划**必须与当前能力分开。`MealAssistantProductSpec.md` 继续保持 `planned`，它的验收项不是已完成证据；
4. **旧承诺**若没有代码、契约或测试支撑，必须在当前文档中标为历史设想或实现缺口，不能因为旧稿写得完整就继承为规范。

本批复核后的当前入口为 [Gateway](../../gateway/README.md)、[Gateway 固定工作流](../../gateway/workflow.md)、[话题与查询分析](../../gateway/analysis.md)、[全局命令](../../gateway/commands.md)、[Frontend 总览](../../frontend/README.md)、[应用壳与视觉系统](../../frontend/application-shell.md)、[Chat 工作区](../../frontend/chat-workspace.md)、[管理页面](../../frontend/management-views.md)、[状态与传输](../../frontend/state-and-transports.md) 以及 [Applications 索引](../../applications/README.md)。

## 2. 第 9 节：`docs/engines/gateway.md`

### 2.1 已被当前文档承接的事实与理念

- **The Eye 的入口隐喻**保留为 Gateway 形成统一入口视角的设计背景，但当前文档同时说明它不是系统中枢神经，也不拥有下游控制权；
- **漏斗式两级入口**已经拆分为固定 workflow 中的 L1 确定性拦截、命令 registry/parser/dispatcher 和 L2 话题/查询分析；
- **一次分析、受限复用**已经补入 Gateway 总览。一次请求形成冻结 `GatewayDecision`，下游共享重写查询、关键词、检索计划、话题目标和写入信号的入口投影，以避免重复解释；复用不意味着 Gateway 拥有检索结果、记忆价值最终裁决或持久化真相；
- **主动/被动入口分离**已经进入 Gateway 总览、workflow 与 System Passive Ingress。主动入口可以短路命令，被动入口不接受系统命令；被动 buffer、outbox、去重和提交由 System 所有；
- **fallback、deadline、取消和终态投影**已经由 `GatewayExecutionState`、固定 Step 提交和 `GatewayCommandOutcome`/`GatewayDecisionOutcome` 记录，公共结果是冻结的依赖中立协议。

### 2.2 明确拒绝继承的历史口径

- Gateway 是“系统中枢神经”、驱动所有下游并拥有全局调度权；
- 一个同时控制话题、检索、感知、生成和记忆保存的 `GatewayResult`；
- 旧 `worth_saving: bool` 作为最终记忆决定；当前只有 `memory_write_signal` 预判，最终写回由 Patchouli/Generation 决定；
- 所有 `CHAT` 默认进入 RAG 的“乐观检索”作为统一真相；当前由 `RetrievalPlan` 显式表达，模式可为 `SKIP`；
- `ObserverSessionBuffer`、`MessageTurnBufferManager`、被动模式的完整 ContextRenderer 和外部 Agent 的观察状态归 Gateway 所有；
- 动态 DAG、通用工作流、持久化 workflow/job 状态，或旧 Engine 文件布局、模型字段和配置作为当前 API。

### 2.3 物理迁移

旧稿已标记 `superseded`，现移动至 [`archive/legacy-docs/engines/gateway.md`](../legacy-docs/engines/gateway.md)。归档文件保留历史设计与未落地设想，顶部替代入口改为指向当前 Gateway 文档；原路径不再作为有效文档入口。

## 3. 第 10 节：Applications 与 Frontend

### 3.1 `docs/applications/MealAssistantProductSpec.md`

本文件不物理移动，继续留在 Applications 目录并保持 `planned`。它保留的核心理念是：用一个单用户、单 Agent、最小权限、三天 MVP 的真实应用验证“用户是否感到系统记得我”，失败后先区分检索、读取、写入、更新、时间与回复决策层，再决定是否改 Prompt 或排期，而不是直接重构架构。

本次复核补正了四项边界：

- MVP 流程改为 `Gateway decision -> Patchouli prepare/retrieval -> Alice run/MTP -> Patchouli finalize`，与当前组合根和 `ChatService` 一致；
- 七种 `MemoryType` 被说明为通用类型映射，不新增餐食专用枚举；`QueryFilters.time_range` 目前只是未接入 MTP parser 与 Retrieval 的模型字段，不能引用为端到端时间过滤能力，也没有自动的 `recent_meals` 专用索引；
- “新会话”改写为清空工作区后发起新的独立 chat run 的验收语义。当前前端没有 session/message history 恢复，产品规格不能暗示已有会话恢复；
- `CALL`、`CODE_SNIPPET`、MaaT 和多 Agent 仍是本应用明确不依赖的能力；`RUN` 仅为 `sys_clock` 提供入口，当前权限粒度的 `RUN`/记忆执行缺口继续作为已知限制记录。

产品文档中的用户场景、system prompt 和验收清单属于未来试验设计，不是当前后端或前端能力的证明。

### 3.2 `docs/frontend/FrontendDesign.md`

已承接的内容包括“透明、沉浸、可观测”的产品方向、人类态与机器态分离、同屏工作台、日月交辉与五行状态色、水晶层次，以及把协议事实翻译成结构化卡片的理由。它们分别进入 [Frontend 总览](../../frontend/README.md)、[应用壳](../../frontend/application-shell.md) 和 [Chat 工作区](../../frontend/chat-workspace.md)。本次复核进一步在应用壳中明确：动效是后端状态迁移的视觉投影，真实事件到达前不得用动画制造成功感。

明确不继承：Shadcn/Radix 作为当前组件基础、四栏可拖拽布局已经交付、Terminal/账户/附件/`#` 引用等空壳入口已经完成、前端正则解析原始 MTP XML 是主路径，以及恒速 token 队列可以掩盖停顿或断流。当前组件、布局和事件事实以 Frontend 当前文档为准。

旧稿已移动至 [`archive/legacy-docs/frontend/FrontendDesign.md`](../legacy-docs/frontend/FrontendDesign.md)。

### 3.3 `docs/frontend/frontend-state-persistence-research.md`

已承接的内容包括“业务真相归后端、UI 偏好可本地恢复”的状态所有权原则、Topic/Memory/Config/Logs 与消息流的持久化边界、HTTP/SSE/WebSocket 的传输选择、多窗口日志广播和 mock 降级风险，均进入 [状态、持久化与传输](../../frontend/state-and-transports.md)。

本次复核将“localStorage 是本地契约”补入当前文档：持久化 store 必须用 `name`、`version`、`migrate`、`partialize` 维护结构升级，优先保存小型、可重新解释的偏好，不能让过期的 Agent、Topic 或消息结构在升级后复活。研究稿中的 store 拆分建议仍属于演进建议；当前实现的具体 `partialize`、mock 与迁移行为才是事实。

旧稿已移动至 [`archive/legacy-docs/frontend/frontend-state-persistence-research.md`](../legacy-docs/frontend/frontend-state-persistence-research.md)。

### 3.4 `docs/frontend/MemoryGardenUI.md`

已承接的内容包括“记忆是可检查、可修正的资产”这一 Memory Garden 隐喻、卡片/列表两种密度、alias 可寻址身份、confidence/vitality/访问次数的人工观察价值，以及详情、编辑和删除作为 human-in-the-loop 的意义；它们进入 [管理页面](../../frontend/management-views.md)。

明确不继承：真正 embedding semantic search 已完成、`Pin/Lock` 免疫 GC、人工编辑自动把 confidence 锁为 1.0、Archive/统计大屏和 `TOOL` 记忆类型已交付。当前页面只对已加载记录做客户端匹配，Pin 仍是施工中，后端 generic memory API 才是写入和删除的事实来源。

旧稿已移动至 [`archive/legacy-docs/frontend/MemoryGardenUI.md`](../legacy-docs/frontend/MemoryGardenUI.md)。

## 4. 物理迁移结果

审计通过后，本批只移动四篇已标记 `superseded` 的旧稿：

```text
docs/engines/gateway.md
  -> docs/archive/legacy-docs/engines/gateway.md

docs/frontend/FrontendDesign.md
docs/frontend/frontend-state-persistence-research.md
docs/frontend/MemoryGardenUI.md
  -> docs/archive/legacy-docs/frontend/
```

`MealAssistantProductSpec.md` 保持原位。归档后的 Frontend 旧稿统一把当前文档链接改为 `../../../frontend/...`，源码链接改为从 Archive 根出发的四级相对路径；Gateway 旧稿改为 `../../../gateway/...`，避免物理移动后留下失效的同目录链接。

## 5. 验证门禁

- [x] Gateway 旧稿的事实、理念和拒绝继承项均有当前承接位置；
- [x] 三篇 Frontend 旧稿的设计理念与当前实现偏差均有承接或拒绝说明；
- [x] Meal Assistant 保持 `planned`，并明确当前没有独立应用和正式 session/history 恢复；
- [x] 当前 Gateway、Frontend、Applications 相关文档统一复核日期为 2026-07-29；
- [x] 四篇旧稿移动前完成源/目标绝对路径和 `docs/` 边界检查；
- [x] 移动后旧路径消失、归档链接与反向引用完成检查；
- [x] 严格 UTF-8、无尾随空白、`git diff --check` 及 Gateway/Frontend 定向验证通过。

验证结果：本批 20 篇相关 Markdown 的严格 UTF-8 与相对链接检查通过；Gateway、System application、Passive Gateway 与 Patchouli Decision 共 48 项定向测试通过；前端 `npm run lint` 和 `npm run build` 均通过。首次构建因沙箱禁止 esbuild 读取工作区父目录而失败，按同一命令在获准的非沙箱环境重试后成功；仅保留现有的大 chunk 提示，不影响本批文档结论。
