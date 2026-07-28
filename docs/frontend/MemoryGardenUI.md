---
title: Legacy Memory Garden UI Concept
status: superseded
owner: frontend
scope: legacy-memory-garden-concept
archived_at: 2026-07-28
superseded_by:
  - docs/frontend/management-views.md
---

> 本文保留 Memory Garden 的产品隐喻和早期功能设想，已停止维护。当前 Memory Library 的真实读写能力、客户端搜索、mock fallback 和未实现项以[管理页面](./management-views.md)为准；本文的真正语义搜索、Pin/Lock、归档筛选与统计大屏均不能作为当前能力引用。

“记忆花园（Memory Garden）”是 HiveMemory 系统中最重要的管理后台，它是你（作为人类上帝视角）与帕秋莉（系统 Librarian）直接交互的界面。由于“记忆原子”包含了丰富的元数据（类型、标签、置信度、时间戳），这个页面的功能设计需要兼顾**“高密度的信息检索”**与**“便捷的人工干预”**。

关于布局，推荐 **默认采用卡片式瀑布流（Grid/Masonry），并提供一个切换到紧凑列表（Table/List）的按钮**。卡片式更符合“花园”隐喻和区块化阅读，而列表式适合批量管理。

---

### 一、 顶部控制台 (The Command Bar)
这是进入花园后的第一视觉区域，用于精准定位和筛选记忆。

1. **双模搜索框 (Hybrid Search Bar)**
   * **语义搜索 (Semantic)**：默认模式，输入自然语言，调用后端的 Embedding 模型去 Qdrant 里做相似度匹配（测试 RAG 效果的最佳入口）。
   * **精确搜索 (Keyword/Alias)**：通过前缀（如 `alias:` 或 `tag:`）直接匹配具体的真名或标签。
2. **多维过滤器 (Filters)**
   * **按类型 (Type)**：下拉单选（`CODE_SNIPPET`, `FACT`, `TOOL`, `URL_RESOURCE` 等）。
   * **按标签 (Tags)**：多选（如 `#python`, `#config`）。
   * **按状态 (Status)**：筛选 `Active`（活跃）、`Archived`（已归档/休眠）。
3. **排序控制 (Sort By)**
   * 创建时间（最新/最旧）
   * 最后访问时间（最近使用）
   * 访问频次（最常使用，反映记忆的热度）
   * 置信度（Confidence Score）

### 二、 记忆原子展示区 (The Atom Cards)
每个记忆原子（Memory Atom）在卡片上应该透出最核心的元数据，让用户一眼看懂它存了什么。

在卡片上需要展示的要素：
1. **身份标识**：
   * `Title`（大标题）：记忆的主旨。
   * `Alias`（副标题/真名）：如 `@tool_write_file`，支持一键复制，方便用户在对话中直接用 MTP 引用。
2. **核心属性徽章 (Badges)**：
   * **Type Badge**：用不同颜色区分，如 `[代码]`、`[事实]`、`[工具]`。
   * **Tags**：显示 2-3 个核心标签，超出的折叠。
3. **数据摘要 (Summary)**：
   * `Index.Summary` 的前 2-3 行文本，不展示完整的 Payload。
4. **生命周期指标 (Lifecycle Metrics)**：
   * **置信度 (Confidence)**：用一个进度条或星级表示（如 0.9/1.0）。如果低于 0.5，卡片标红警告（可能是幻觉）。
   * **引用次数 (Hits)**：一个火焰图标 🔥 配合数字，表示它被 Agent 成功调用过多少次。

### 三、 人工干预与操作 (Human-in-the-loop Operations)
鼠标悬浮在卡片上（Hover），或点击卡片右上角的 `...` 菜单时，出现的操作项：

1. **查看详情 (View Detail Modal)**：
   * 点击卡片，弹出一个大的 Dialog/Sheet。
   * 左侧渲染完整的 Markdown Payload（代码、文本）。
   * 右侧显示完整的元数据（创建时间、来源 Agent、完整标签流）。
2. **强制修改 (Edit/Correct)**：
   * 允许人类直接修改 Summary、Payload 或增删 Tags。
   * *重要逻辑*：人类修改后，该记忆的置信度 (Confidence) 强行锁定为 `1.0`，系统不再自动遗忘它。
3. **固化/加星 (Pin/Lock)**：
   * 一个图钉 📌 图标。锁定后，免疫感知层的 `Evict` (驱逐) 和垃圾回收机制，变成永久核心法则。
4. **删除/归档 (Delete/Archive)**：
   * 发现帕秋莉总结错了，或者 Agent 产生幻觉记录了垃圾数据，人类一键删除。

### 四、 开发者专属要素 (可选，但非常实用)

1. **手动播种 (Manual Inject)**：
   * 一个悬浮的 `[+ 新建记忆]` 按钮。允许用户不通过对话，直接向系统里录入一段配置代码或系统级 Prompt（比如公司最新的 API 规范）。
2. **统计大屏 (Garden Stats)**：
   * 在顶部展示简略的系统健康度：`总记忆数: 1,240` | `向量库占用: 45MB` | `幻觉警告: 3 条`。

