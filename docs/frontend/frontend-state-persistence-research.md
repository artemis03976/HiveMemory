---
title: Legacy Frontend State Persistence Research
status: superseded
owner: frontend
scope: completed-state-persistence-research
archived_at: 2026-07-28
superseded_by:
  - docs/frontend/state-and-transports.md
---

> 本文保留前端状态所有权调研过程，已停止维护。已经接受并落地的持久化边界、各 store 的当前 `partialize` 行为、传输和 mock 降级以[状态、持久化与传输](./state-and-transports.md)为准。

# 前端状态持久化调研报告

## 背景

当前项目的前端尚未系统完成状态恢复与持久化设计。考虑到 HiveMemory 的产品形态，前端状态不能一概而论，而应拆分为两类分别处理：

- **UI 状态**：用于恢复用户界面、交互偏好和操作上下文，适合由前端本地持久化。
- **业务核心数据状态**：对应系统真实数据、后端计算结果或服务端权威状态，应以后端为准，前端仅适合做短期缓存、草稿恢复或乐观更新。

本文基于当前仓库中的前端与部分关联后端实现，对需要持久化的状态进行归类总结。

---

## 调研范围

### 前端核心文件

- [frontend/src/App.tsx](../../frontend/src/App.tsx)
- [frontend/src/components/ChatLayout.tsx](../../frontend/src/components/ChatLayout.tsx)
- [frontend/src/components/chat/ContextSidebar.tsx](../../frontend/src/components/chat/ContextSidebar.tsx)
- [frontend/src/components/chat/KernelVision.tsx](../../frontend/src/components/chat/KernelVision.tsx)
- [frontend/src/components/chat/OmniInput.tsx](../../frontend/src/components/chat/OmniInput.tsx)
- [frontend/src/components/SettingsPanel.tsx](../../frontend/src/components/SettingsPanel.tsx)
- [frontend/src/hooks/useMemories.ts](../../frontend/src/hooks/useMemories.ts)
- [frontend/src/hooks/useSettings.ts](../../frontend/src/hooks/useSettings.ts)
- [frontend/src/stores/chatStore.ts](../../frontend/src/stores/chat/chatStore.ts)
- [frontend/src/stores/topicStore.ts](../../frontend/src/stores/topic/topicStore.ts)
- [frontend/src/stores/kernelStore.ts](../../frontend/src/stores/kernel/kernelStore.ts)
- [frontend/src/stores/chatRuntimeConfigStore.ts](../../frontend/src/stores/chat/runtimeConfigStore.ts)

### 关联后端文件

- [src/hivememory/server/routers/chat.py](../../src/hivememory/server/routers/chat.py)
- [src/hivememory/server/routers/topics.py](../../src/hivememory/server/routers/topics.py)
- [src/hivememory/server/routers/memories.py](../../src/hivememory/server/routers/memories.py)
- [src/hivememory/server/routers/config.py](../../src/hivememory/server/routers/config.py)

---

## 总体结论

当前项目前端依赖的后端实体主要包括：

1. **Chat 对话流**
2. **Topic 活跃话题池**
3. **Memory 记忆实体**
4. **Config 系统配置**
5. **Kernel Logs 实时日志流**

在这些能力中：

- **界面偏好、面板布局、筛选排序、视图模式、输入草稿**等，属于 UI 状态，适合在前端本地持久化。
- **topics、messages、retrieved memories、memory 实体、系统配置、日志内容**等，属于业务核心数据状态，应以后端为准，不能简单落在前端本地作为真相来源。

---

## 一、前端状态分层现状

当前前端主要通过 React `useState` 和 Zustand store 管理状态。

### 1. 全局 Zustand store

#### Chat Store
文件：[
frontend/src/stores/chatStore.ts](../../frontend/src/stores/chat/chatStore.ts)

主要状态：

- `messages`
- `connection`
- `isStreaming`
- `currentTopicId`
- `retrievedMemories`

其中：

- `messages`、`currentTopicId`、`retrievedMemories` 属于业务核心数据或业务执行结果。
- `connection`、`isStreaming` 属于业务过程状态。

#### Topic Store
文件：[frontend/src/stores/topicStore.ts](../../frontend/src/stores/topic/topicStore.ts)

主要状态：

- `topics`
- `isLoading`
- `error`

其中 `topics` 明确是后端话题池数据的前端映射。

#### Kernel Store
文件：[frontend/src/stores/kernelStore.ts](../../frontend/src/stores/kernel/kernelStore.ts)

主要状态：

- `logs`
- `traceGroups`
- `connection`
- `filters`
- `ui`
- `stats`

其中：

- `logs`、`traceGroups`、`connection`、`stats` 属于日志流业务数据和运行状态。
- `filters`、`ui` 属于典型 UI 状态。

#### Chat Runtime Config Store
文件：[frontend/src/stores/chatRuntimeConfigStore.ts](../../frontend/src/stores/chat/runtimeConfigStore.ts)

主要状态：

- `generationOptions`

这组状态更接近“用户运行时偏好配置”，不是后端权威业务实体。

---

### 2. Hook 内的局部状态

#### useMemories
文件：[frontend/src/hooks/useMemories.ts](../../frontend/src/hooks/useMemories.ts)

主要状态：

- 业务数据：`rawMemories`、`memories`、`total`
- 请求状态：`loading`、`error`
- UI 状态：`searchQuery`、`searchMode`、`selectedType`、`selectedTags`、`statusFilter`、`sortBy`、`viewMode`

#### useSettings
文件：[frontend/src/hooks/useSettings.ts](../../frontend/src/hooks/useSettings.ts)

主要状态：

- 业务数据：`config`、`originalConfig`
- 请求状态：`loading`、`error`
- UI/编辑状态：`validationErrors`、`isDirty`

---

### 3. 页面和组件局部状态

#### App
文件：[frontend/src/App.tsx](../../frontend/src/App.tsx)

- `activeNavTab`

#### ChatLayout
文件：[frontend/src/components/ChatLayout.tsx](../../frontend/src/components/ChatLayout.tsx)

- `activeTopicId`
- `isContextSidebarCollapsed`
- `isKernelVisionCollapsed`

#### ContextSidebar
文件：[frontend/src/components/chat/ContextSidebar.tsx](../../frontend/src/components/chat/ContextSidebar.tsx)

- `activeTab`

#### KernelVision
文件：[frontend/src/components/chat/KernelVision.tsx](../../frontend/src/components/chat/KernelVision.tsx)

- `activeTab`

#### OmniInput
文件：[frontend/src/components/chat/OmniInput.tsx](../../frontend/src/components/chat/OmniInput.tsx)

- `message`
- `enableMemory`

#### SettingsPanel
文件：[frontend/src/components/SettingsPanel.tsx](../../frontend/src/components/SettingsPanel.tsx)

- `activeCategory`

---

## 二、当前项目中已存在的本地持久化实现

### 1. 已使用 Zustand persist

#### Chat Store
文件：[frontend/src/stores/chatStore.ts](../../frontend/src/stores/chat/chatStore.ts)

使用了 `persist`，但通过 `partialize` 实际持久化为：

- `messages: []`
- `currentTopicId: null`

这意味着当前实现**有持久化壳子，但刻意不保留聊天核心数据**。

#### Kernel Store
文件：[frontend/src/stores/kernelStore.ts](../../frontend/src/stores/kernel/kernelStore.ts)

实际持久化：

- `filters`
- `ui`

这是典型且合理的 UI 状态持久化。

#### Chat Runtime Config Store
文件：[frontend/src/stores/chatRuntimeConfigStore.ts](../../frontend/src/stores/chat/runtimeConfigStore.ts)

实际持久化：

- `generationOptions`

这属于用户运行参数偏好持久化。

---

### 2. 已直接使用 localStorage

#### Kernel 主窗口选举
文件：[frontend/src/stores/kernelStore.ts](../../frontend/src/stores/kernel/kernelStore.ts)

通过 `localStorage` 维护：

- `PRIMARY_WINDOW_KEY`

用途是多窗口下决定哪个窗口持有 WebSocket 主连接。这属于运行时协调状态。

---

### 3. 未发现的本地持久化方式

当前未看到以下机制：

- `sessionStorage`
- `indexedDB`
- cookie
- URL query / route state 持久化

---

## 三、哪些状态应归类为 UI 状态持久化

这类状态的共同特征是：

- 不构成系统业务真相
- 与后端实体一致性要求弱
- 刷新后恢复能显著提升用户体验

### 1. 对话页面布局与导航状态

#### 建议本地持久化

- 主导航当前页签 `activeNavTab`
  文件：[frontend/src/App.tsx](../../frontend/src/App.tsx)
- 左侧边栏展开/收起 `isContextSidebarCollapsed`
  文件：[frontend/src/components/ChatLayout.tsx](../../frontend/src/components/ChatLayout.tsx)
- 右侧边栏展开/收起 `isKernelVisionCollapsed`
  文件：[frontend/src/components/ChatLayout.tsx](../../frontend/src/components/ChatLayout.tsx)
- 左侧边栏当前 tab `topics | config`
  文件：[frontend/src/components/chat/ContextSidebar.tsx](../../frontend/src/components/chat/ContextSidebar.tsx)
- 右侧边栏当前 tab `context | terminal`
  文件：[frontend/src/components/chat/KernelVision.tsx](../../frontend/src/components/chat/KernelVision.tsx)

#### 结论
这些都属于纯界面状态，不会破坏业务一致性，建议优先本地持久化。

---

### 2. Memory Garden 的搜索/筛选/排序/视图状态

#### 建议本地持久化

- `searchQuery`
- `searchMode`
- `selectedType`
- `selectedTags`
- `statusFilter`
- `sortBy`
- `viewMode`

文件：[frontend/src/hooks/useMemories.ts](../../frontend/src/hooks/useMemories.ts)

#### 结论
这些状态仅影响“用户如何看数据”，不影响“数据本身是什么”，非常适合本地持久化。

---

### 3. Settings 页面导航状态

#### 建议本地持久化

- `activeCategory`

文件：[frontend/src/components/SettingsPanel.tsx](../../frontend/src/components/SettingsPanel.tsx)

#### 结论
这属于设置页面的界面上下文恢复，适合本地持久化。

---

### 4. Chat 输入侧偏好状态

#### 建议本地持久化

- `enableMemory`
  文件：[frontend/src/components/chat/OmniInput.tsx](../../frontend/src/components/chat/OmniInput.tsx)
- `generationOptions`
  文件：[frontend/src/stores/chatRuntimeConfigStore.ts](../../frontend/src/stores/chat/runtimeConfigStore.ts)

#### 可选本地持久化

- 输入框草稿 `message`

#### 结论
这类状态属于用户偏好或短期交互上下文，适合本地恢复，但不应被视为后端权威配置。

---

### 5. Kernel 面板的过滤器与展示偏好

#### 已有正确实现

- `filters`
- `ui`

文件：[frontend/src/stores/kernelStore.ts](../../frontend/src/stores/kernel/kernelStore.ts)

#### 结论
这是当前项目中最典型、最合理的 UI 状态持久化案例，应作为后续设计参考。

---

### 6. 可选但非优先的瞬时 UI 状态

以下状态可根据产品需求决定是否持久化：

- Memory 编辑弹窗草稿
- 卡片展开/折叠状态
- hover 状态
- toast 队列

这类状态通常不是高优先级持久化对象，除非明确要求“刷新后恢复现场”。

---

## 四、哪些状态应归类为业务核心数据持久化

这类状态的共同特征是：

- 对应真实业务实体或服务端计算结果
- 会随服务端状态变化而变化
- 本地缓存不能替代后端权威源

### 1. Topic 列表与 Topic 状态

前端文件：

- [frontend/src/stores/topicStore.ts](../../frontend/src/stores/topic/topicStore.ts)
- [frontend/src/components/ChatLayout.tsx](../../frontend/src/components/ChatLayout.tsx)

后端文件：

- [src/hivememory/server/routers/topics.py](../../src/hivememory/server/routers/topics.py)

#### 必须以后端为准

- `topics`
- topic 的标题、摘要、token 统计、活跃状态

#### 可作为本地 UI 恢复线索

- `activeTopicId`

#### 结论
Topic 列表属于后端活跃池快照，必须服务端同步。前端最多只应保存“上次选中了哪个 topic”这种 UI 恢复信息，且恢复后仍需校验该 topic 是否存在。

---

### 2. Chat 消息、会话执行结果、检索记忆

前端文件：

- [frontend/src/stores/chatStore.ts](../../frontend/src/stores/chat/chatStore.ts)

后端文件：

- [src/hivememory/server/routers/chat.py](../../src/hivememory/server/routers/chat.py)

#### 必须以后端为准

- `messages`
- `currentTopicId`
- `retrievedMemories`
- SSE 流式事件结果：`token`、`topic_info`、`memory_refs`、`done.final_text`

#### 结论
聊天相关状态本质上由后端 SSE 驱动生成，不能简单通过本地持久化恢复为真实会话记录。当前项目已经通过 `chatStore` 的 `persist` 配置显式避免持久化聊天核心数据，这一设计方向是合理的。

---

### 3. Memory 实体数据

前端文件：

- [frontend/src/hooks/useMemories.ts](../../frontend/src/hooks/useMemories.ts)

后端文件：

- [src/hivememory/server/routers/memories.py](../../src/hivememory/server/routers/memories.py)

#### 必须以后端为准

- `rawMemories`
- `memories`
- memory 详情字段
- memory 更新/删除结果

#### 可本地持久化的仅是展示偏好

- 搜索、筛选、排序、视图模式

#### 结论
Memory 是后端 CRUD 实体，前端不应把列表或详情缓存当作长期真相来源。

---

### 4. 系统配置 Config

前端文件：

- [frontend/src/hooks/useSettings.ts](../../frontend/src/hooks/useSettings.ts)
- [frontend/src/components/SettingsPanel.tsx](../../frontend/src/components/SettingsPanel.tsx)

后端文件：

- [src/hivememory/server/routers/config.py](../../src/hivememory/server/routers/config.py)

#### 必须以后端为准

- `config`
- `originalConfig`
- 所有真正生效的系统参数

#### 可本地持久化的仅是 UI 编辑态

- 当前分类
- 草稿输入
- `isDirty`
- `validationErrors`（如有必要）

#### 结论
系统配置是明确的后端权威数据。后端会校验、写盘并更新运行时配置，因此前端本地只能保存编辑草稿，不能把本地副本当作系统真实配置。

---

### 5. Kernel 日志内容

前端文件：

- [frontend/src/stores/kernelStore.ts](../../frontend/src/stores/kernel/kernelStore.ts)

#### 必须以后端为准

- `logs`
- `traceGroups`
- 实时连接状态衍生的日志流内容

#### 适合本地持久化的仅是 UI 偏好

- `filters`
- `ui`

#### 结论
日志内容属于实时业务流，前端展示层不应将其本地持久化为业务真相。

---

## 五、建议的状态持久化归类表

| 状态项 | 所在位置 | 归类 | 建议持久化方式 | 说明 |
|---|---|---|---|---|
| 主导航页签 | `App.tsx` | UI 状态 | 前端本地持久化 | 纯界面上下文 |
| 左右侧边栏折叠状态 | `ChatLayout.tsx` | UI 状态 | 前端本地持久化 | 恢复阅读布局 |
| 左右面板当前 tab | `ContextSidebar.tsx` / `KernelVision.tsx` | UI 状态 | 前端本地持久化 | 恢复操作上下文 |
| Memory 搜索/筛选/排序/视图 | `useMemories.ts` | UI 状态 | 前端本地持久化 | 仅影响展示 |
| Settings 当前分类 | `SettingsPanel.tsx` | UI 状态 | 前端本地持久化 | 仅影响导航 |
| Chat generation options | `chatRuntimeConfigStore.ts` | UI/运行偏好 | 前端本地持久化 | 已有实现 |
| `enableMemory` 开关 | `OmniInput.tsx` | UI/运行偏好 | 前端本地持久化 | 用户请求偏好 |
| 输入框草稿 | `OmniInput.tsx` | UI 草稿 | 可选本地持久化 | 不应作为会话真相 |
| 当前选中的 topic id | `ChatLayout.tsx` | UI 恢复线索 | 可选本地持久化 | 恢复后需校验 |
| Topic 列表 | `topicStore.ts` | 业务核心数据 | 后端同步 | 不应本地长期持久化 |
| Chat messages | `chatStore.ts` | 业务核心数据 | 后端持久化/后端会话能力 | 当前不应本地持久化 |
| retrieved memories | `chatStore.ts` | 业务核心数据 | 后端同步 | 来自 SSE 检索结果 |
| Memory 列表与详情 | `useMemories.ts` | 业务核心数据 | 后端同步 | 前端仅短期缓存 |
| Config 实体 | `useSettings.ts` | 业务核心数据 | 后端同步 | 后端校验并写盘 |
| Kernel logs 内容 | `kernelStore.ts` | 业务核心数据 | 后端实时流 | 仅 UI 偏好适合本地持久化 |

---

## 六、实施优先级建议

### 第一优先级：直接提升体验且风险低

建议优先本地持久化以下状态：

1. 主导航页签
2. 左右侧边栏展开/收起
3. 左右面板当前 tab
4. Memory 页面搜索/筛选/排序/视图模式
5. Settings 当前分类
6. `enableMemory`

这些状态都是纯 UI 偏好，实现成本低，且不会引入业务一致性风险。

---

### 第二优先级：可恢复草稿，但要与业务状态解耦

1. 输入框草稿
2. 当前选中的 topic id
3. Settings 未保存配置草稿
4. Memory 编辑草稿

这些状态恢复时必须明确是“前端草稿/偏好”，不能等同于后端真实状态。

---

### 第三优先级：需要后端能力配合后才能实现完整恢复

1. 聊天历史消息刷新恢复
2. 当前会话恢复
3. topic 上下文与消息链恢复

这类需求本质上已经超出单纯前端本地持久化范畴，必须以后端提供明确的 session/message 持久化与查询接口为前提。

---

## 七、结论

当前项目在状态恢复设计上应坚持以下原则：

- **凡是界面偏好、布局、筛选排序、输入草稿，都应优先考虑前端本地持久化。**
- **凡是 Topic、Chat、Memory、Config、Logs 等系统真实数据，都必须以后端为权威源。**
- **前端对业务核心数据最多只做短期缓存、乐观更新或草稿恢复，不能把本地存储当作最终真相。**

从现有代码来看，项目已经在部分地方体现了这一边界，例如：

- `kernelStore` 只持久化 UI 偏好，不持久化日志内容。
- `chatStore` 虽使用 `persist`，但有意清空消息数据，避免将聊天内容错误地当成本地可恢复会话。

这说明当前代码结构已经具备清晰拆分 UI 状态与业务状态的基础，后续只需要围绕这条边界继续补全即可。

---

## 附录 A：推荐落地方案

本附录偏实施视角，目标不是一次性重构全部状态，而是在尽量小的改动下，把“应本地持久化的状态”先稳定落地。

### A.1 推荐的总体原则

建议采用以下四层策略：

1. **纯 UI 偏好**：直接使用前端本地持久化。
2. **前端草稿态**：允许本地持久化，但恢复时必须标注为草稿，不直接覆盖后端状态。
3. **服务端实体快照**：只做内存缓存或短期缓存，页面初始化后必须重新拉取。
4. **会话/消息类核心数据**：不做前端长期持久化，等后端补齐 session/message 能力后再接入真正恢复。

---

### A.2 推荐优先落地的状态

#### 第一批：可立即落地

建议优先补齐以下状态的持久化：

- `activeNavTab`
- `isContextSidebarCollapsed`
- `isKernelVisionCollapsed`
- `ContextSidebar.activeTab`
- `KernelVision.activeTab`
- `SettingsPanel.activeCategory`
- `useMemories` 中的搜索/筛选/排序/视图模式
- `OmniInput.enableMemory`

这批状态都有共同特点：

- 都是纯 UI 状态
- 与后端数据一致性耦合极低
- 刷新恢复后体验提升明显
- 即使出错，影响范围也只在展示层

#### 第二批：谨慎落地

- `OmniInput.message` 输入草稿
- `ChatLayout.activeTopicId`
- `useSettings` 未保存草稿
- Memory 编辑弹窗草稿

这批状态可以持久化，但要明确“只是恢复用户现场”，而不是恢复系统真实状态。

#### 第三批：暂不建议仅前端落地

- 聊天消息历史恢复
- 当前 session 恢复
- MTP 执行过程恢复
- 当前 topic 的完整上下文恢复

这批能力需要后端提供正式的会话持久化支持，否则前端只能恢复出不完整甚至错误的状态。

---

### A.3 推荐的数据流模式

#### 模式 1：UI 偏好型

适用于：tab、折叠状态、筛选器、视图模式。

建议流程：

- 首次加载时从本地持久化读取默认值
- 用户操作后立即写回本地
- 不需要后端参与

#### 模式 2：草稿恢复型

适用于：输入框草稿、设置页未提交编辑内容。

建议流程：

- 本地保存草稿副本
- 页面进入时恢复草稿
- 同时重新拉取后端权威数据
- 如果后端数据已变化，草稿应以“未提交修改”身份叠加，而不是直接替换服务端结果

#### 模式 3：服务端同步型

适用于：topics、memories、config、logs。

建议流程：

- 初始化先渲染 loading 或内存快照
- 页面加载后立即请求后端
- 用后端返回值覆盖前端快照
- 本地不保存实体真值，只保存展示偏好或最后一次选择线索

---

## 附录 B：建议拆分成哪些 store

当前项目已经有 Zustand store，但仍有一部分状态散落在组件 `useState` 和 hook 内。为了让持久化边界更清晰，建议按“状态性质”而不是“页面文件位置”拆分。

### B.1 Chat UI Store

建议新增或抽离一个专门的 chat-ui-store，用于存放纯聊天界面状态，例如：

- 主导航页签（如果后续仍保留在 chat 入口范围）
- 左侧边栏折叠状态
- 右侧边栏折叠状态
- 左侧边栏当前 tab
- 右侧边栏当前 tab
- 当前选中的 topic id（仅作为 UI 恢复线索）
- 输入框草稿
- `enableMemory`

推荐原因：

- 这些状态天然适合统一 `persist`
- 能避免把 UI 状态混进 `chatStore` 这种业务 store
- 能减少组件之间通过 props 层层传递折叠状态和 tab 状态

---

### B.2 Memory View Store

建议将 `useMemories` 里的“展示控制状态”抽离为 memory-view-store：

- `searchQuery`
- `searchMode`
- `selectedType`
- `selectedTags`
- `statusFilter`
- `sortBy`
- `viewMode`

而 `rawMemories`、`memories`、`loading`、`error` 继续保留为数据请求层状态。

推荐原因：

- 这样可以把“后端实体数据”和“展示偏好”彻底分开
- Memory 页面未来若出现多个入口，也可以共享同一套视图偏好
- 本地持久化时只需 persist 一个干净的小 store

---

### B.3 Settings Draft Store

建议将 Settings 的本地编辑态拆成两层：

1. **settings-server-state**
   - `config`
   - `originalConfig`
   - `loading`
   - `error`

2. **settings-draft-state**
   - 当前分类
   - 草稿修改记录
   - `isDirty`
   - 可选的本地校验结果缓存

推荐原因：

- 现在 `useSettings` 同时承担“拉取后端配置”和“本地编辑草稿”两种责任
- 后续若要支持“刷新后继续编辑未保存配置”，拆层后更容易实现
- 也能避免把后端配置对象整棵写进 localStorage

---

### B.4 保持现状即可的 store

以下 store 当前边界已经比较合理：

#### `kernelStore`

建议保留现在的结构：

- `logs` / `traceGroups` / `connection` 不持久化
- `filters` / `ui` 持久化

#### `chatRuntimeConfigStore`

当前已经很接近理想形态：

- 只保存 generation options
- 不混入聊天消息或话题数据

---

## 附录 C：哪些状态适合 `zustand persist`，哪些必须改成后端能力

### C.1 适合直接使用 `zustand persist`

以下状态适合直接放入独立 store 并使用 `persist`：

- 主导航页签
- 左右侧边栏折叠状态
- 左右面板当前 tab
- Memory 搜索/筛选/排序/视图状态
- Settings 当前分类
- `enableMemory`
- Chat generation options
- 输入框草稿
- 当前选中的 topic id（仅作恢复线索）

这类状态适合 `zustand persist` 的原因是：

- 数据结构小
- 不需要服务端校验才能成立
- 覆盖写回的副作用低
- 用户体验提升明显

---

### C.2 适合“内存缓存 + 每次重新拉取”，不适合长期 persist

以下状态不建议直接持久化到 localStorage：

- `topics`
- `rawMemories`
- `memories`
- `logs`
- `traceGroups`
- 聊天中的 `retrievedMemories`

更合适的策略是：

- store 里保留运行时内存态
- 页面重新进入时重新请求后端
- 仅在必要时使用短期缓存作为占位显示
- 后端结果返回后立即覆盖

---

### C.3 必须依赖后端新增能力才能真正恢复的状态

以下状态如果未来想支持“刷新继续/跨端同步/可追溯恢复”，必须由后端提供正式支持：

- chat message history
- conversation/session 列表
- topic 对应消息链
- 某次 MTP 执行的历史记录
- 当前会话恢复入口

建议的后端能力包括：

1. **session 列表接口**
   - 查询用户可恢复会话
2. **message 历史接口**
   - 按 session/topic 拉取消息链
3. **topic 与 message 的稳定映射**
   - 避免前端只能拿到 topic snapshot 却拿不到完整上下文
4. **草稿/偏好与实体状态分离**
   - 避免前端把 session 元数据混进 UI 偏好存储

---

## 附录 D：建议的实施顺序

### D.1 第一阶段：只做纯 UI 状态持久化

目标：快速提升刷新恢复体验，且不引入业务一致性风险。

建议实施项：

- 新增 chat-ui-store 或 layout-ui-store
- 接管 chat 页面的折叠、tab、activeTopicId、enableMemory
- 为 `App.tsx` 的 `activeNavTab` 增加本地持久化
- 为 Memory 页面增加独立的 view-store
- 为 Settings 页面持久化 `activeCategory`

这一阶段不需要改后端接口。

---

### D.2 第二阶段：补齐草稿恢复

目标：恢复用户未提交输入，但不污染业务真相。

建议实施项：

- 输入框草稿恢复
- Settings 草稿恢复
- Memory 编辑草稿恢复

这一阶段应特别注意：

- 草稿必须和服务端实体分层
- 要提供“放弃草稿/清空草稿”入口
- 当后端实体发生变化时，不能无提示覆盖

---

### D.3 第三阶段：评估后端会话持久化方案

目标：支持真正的聊天恢复，而不是仅恢复 UI 外壳。

建议实施项：

- 设计 session/message 数据模型
- 增加会话列表接口
- 增加消息历史接口
- 明确 topic 与 session 的关系
- 评估是否需要消息分页与增量加载

这一阶段才适合重新讨论：

- 刷新后恢复聊天记录
- 切回历史 topic 时显示消息链
- 多端恢复同一会话

---

## 附录 E：风险与注意事项

### E.1 不要把业务实体直接写入 localStorage

特别是以下对象：

- 全量 `config`
- 全量 `messages`
- 全量 `topics`
- 全量 `memories`
- 全量 `logs`

原因：

- 结构可能较大
- 容易过期
- 和后端不一致时难以判定谁是正确值
- 版本升级后兼容成本高

---

### E.2 本地持久化需要版本化

当前 `chatStore` 已经在 `persist` 中使用了 `version` 与 `migrate`，这是一个好的模式。

建议后续所有新增的持久化 store 都考虑：

- `name`
- `version`
- `migrate`
- `partialize`

这样在 UI 结构调整后，可以主动清理旧状态，避免刷新即崩溃或恢复到错误布局。

---

### E.3 优先持久化“小状态”，避免持久化“大对象”\n
推荐持久化：

- 布尔值
- 当前 tab
- 枚举值
- 小型筛选器配置
- 简短草稿

不推荐优先持久化：

- 大型数组
- 深层嵌套对象
- 服务端完整响应对象
- 高频变化的实时流数据

---

## 附录 F：一句话实施建议

如果只给一个落地建议，可以概括为：

> 先把 chat、memory、settings 三个页面中的“布局、tab、筛选、开关、草稿”抽成独立 UI store，并统一用 `zustand persist` 管理；同时继续坚持 topics、messages、memories、config、logs 以后端为准，不在前端本地保存为业务真相。
