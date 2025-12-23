# HiveMemory 系统开发路线图

## 阶段 0: 基础设施准备 (Infrastructure Setup)
> **目标**：搭建好本地开发环境，跑通基础的 LLM 和 数据库连接。

1.  **环境配置**：
    *   安装 Python 3.12+。
    *   部署 Docker 版 **Qdrant**（本地运行）。
    *   配置 **LiteLLM**，打通 API (OpenAI/Claude/DeepSeek) 的调用测试。
2.  **数据结构定义**：
    *   使用 `Pydantic` 定义 3.1 节中的 `MemoryAtom` Schema（Meta, Index, Payload）。
    *   编写一个简单的脚本，测试将 Python Object 存入 Qdrant 并读取。

## 阶段 I: 帕秋莉的原型 (MVP Ingestion)
> **目标**：实现“单向通道”，即对话日志 -> 结构化记忆入库。不考虑检索，只看入库质量。

**核心任务：**
1.  **Librarian Agent (Patchouli) 开发**：
    *   使用 LangChain 构建一个 Chain。
    *   **Prompt 调试**：编写 4.2 节的 System Prompt，测试她能否从一段乱糟糟的对话中提取出干净的 JSON。
    *   **输出解析**：配置 LangChain 的 `PydanticOutputParser`，确保 LLM 输出合法的 JSON。
2.  **简单写入逻辑**：
    *   实现 `save_memory(transcript)` 函数。
    *   暂时跳过复杂的去重逻辑，直接将生成的 Atom `upsert` 到 Qdrant 中。
3.  **缓冲触发器 (Trigger)**：
    *   写一个简单的 `Buffer` 类，模拟积累 5 句话后自动调用 `save_memory`。

*   **🏆 交付物**：一个 Python 脚本。你输入一段对话文本，运行脚本后，能在 Qdrant Dashboard 中看到生成的结构化向量数据。

## 阶段 II: 记忆闭环 (Retrieval & Injection)
> **目标**：实现“双向交互”，让 Worker Agent 变聪明。这是系统产生价值的第一步。

**核心任务：**
1.  **混合检索器 (Hybrid Retriever)**：
    *   实现 5.1 节的检索逻辑。
    *   先实现纯 Vector Search (Dense)。
    *   再加入 Metadata Filter (如 `type="CODE_SNIPPET"` 或 `tags` 包含某词)。
2.  **上下文渲染器 (Renderer)**：
    *   实现 5.2 节的 `render_context(atoms)` 函数。
    *   将检索到的 JSON 列表转换为带有 XML 标签的 Markdown 文本。
3.  **Worker Agent 集成**：
    *   构建一个简单的 Chat Loop。
    *   在 System Prompt 中加入 `{memory_context}` 占位符。
    *   流程：User Input -> 检索 -> 渲染 -> 填充 Prompt -> LLM 回答。

*   **🏆 交付物**：一个 CLI (命令行) 聊天机器人。当你告诉它“我的 API Key 是 123”，重启程序后问它“我的 Key 是多少”，它能准确回答。

## 阶段 III: 智力进化 (Lifecycle & Evolution)
> **目标**：引入“时间”和“演化”的概念，解决冲突，保证数据质量。**这是帕秋莉从记录员变成管理员的关键。**

**核心任务：**
1.  **严谨写入逻辑 (Strict Ingestion)**：
    *   实现 4.2 节的 **"Search-before-Write"** 流程。
    *   开发 `deduplicate_and_merge` 函数：对比新旧记忆，决定是 `Insert` 还是 `Update`。
    *   实现版本历史堆栈逻辑（Git-like history）。
2.  **记忆评分系统**：
    *   在 Schema 中加入 `last_accessed_at`, `access_count`, `vitality_score`。
    *   实现 6.1 节的评分公式。
3.  **垃圾回收 (GC) 原型**：
    *   写一个定时脚本（或在每次写入后触发），查找低分记忆并将其标记为 `archived`（软删除）。

*   **🏆 交付物**：进行多轮冲突对话测试（如先说 Python 3.8，后改口 3.10）。系统应保留 3.10 为主版本，并将 3.8 压入历史，且检索时优先返回 3.10。

## 阶段 IV: 可视化与接口 (Frontend & API)
> **目标**：从黑盒变成白盒，提供用户友好的操作界面。

**核心任务：**
1.  **API 封装**：
    *   使用 **FastAPI** 将上述 Python 逻辑封装为 RESTful API (`/chat`, `/memories`, `/upload`)。
2.  **Streamlit Dashboard**：
    *   **Chat Tab**：左侧聊天，右侧实时显示“当前检索到的记忆（Retrieved Context）”和“Librarian 的思考过程”。
    *   **Garden Tab**：按时间轴展示记忆卡片，提供“编辑”和“删除”按钮。
3.  **人工干预接口**：
    *   实现 UI 上的 CRUD 操作，允许用户手动修正帕秋莉提取错误的 Tag。

*   **🏆 交付物**：一个 Web 网页。你可以像使用 ChatGPT 一样使用它，并且能直观地看到右侧的“记忆侧边栏”在实时跳动。

## 阶段 V: 多 Agent 扩展 (Multi-Agent & Scale)
> **目标**：打破孤岛，实现群组记忆。

**核心任务：**
1.  **Namespace 隔离**：
    *   修改检索逻辑，加入 5.3 节的 `Visibility` 过滤（Public/Private）。
2.  **多 Agent 模拟**：
    *   创建两个不同的 Agent Profile（例如：FrontendCoder 和 BackendCoder）。
    *   设置它们的 System Prompt 和 User ID。
3.  **知识晋升逻辑**：
    *   编写规则：当 Private 记忆被引用超过 N 次，自动升级为 Public。

*   **🏆 交付物**：一个模拟场景。BackendCoder 定义了一个 API 接口（存入记忆），FrontendCoder 在随后的对话中无需再次询问，直接通过共享记忆获取了该接口定义。
