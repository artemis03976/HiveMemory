---
title: Legacy Generation Engine Design
status: superseded
owner: patchouli
scope: legacy-generation-engine-description
archived_at: 2026-07-28
superseded_by:
  - docs/patchouli/generation.md
  - docs/patchouli/artifacts.md
---

> 本文保留三模式生成与早期认知链背景，已停止维护。当前控制面/数据面、去重、持久化、artifact 与任务终态分别以[记忆生成](../../../patchouli/generation.md)和[Artifacts](../../../patchouli/artifacts.md)为准。

# 6 核心功能 III：记忆生成 (The Generation Layer)

> **\[归属分身：大图书馆本体 (Librarian Core)]**
>
> 对应实现：`src/hivememory/engines/generation/`

本章定义 Patchouli 在接收到感知层提交的 Block 后的处理逻辑。即如何将混沌的对话流转化为有序的记忆蜂巢。

---

## 6.0 模块概览 (Module Overview)

### 6.0.1 目录结构

```text
src/hivememory/engines/generation/
│  __init__.py          # 模块入口，导出所有公共接口
│  engine.py            # MemoryGenerationEngine — 编排器
│  extractor.py         # LLMMemoryExtractor — LLM 提取器
│  deduplicator.py      # MemoryDeduplicator — 查重与演化管理器
│  interfaces.py        # 抽象接口层 (BaseMemoryExtractor, BaseDeduplicator)
│  models.py            # 数据模型 (ExtractedMemoryDraft, GenerationRequest, ...)
│
└─ prompts/
       patchouli.py     # 三套 Prompt 模板 (Mode A/B/C)
```

### 6.0.2 三种工作模式

Generation 层并非只有一种工作方式。根据触发来源的不同，引擎支持三种模式，由 `GenerationRequest` 的字段决定路由：

| 模式 | 触发方式 | 核心输入 | 典型场景 |
| :--- | :--- | :--- | :--- |
| **Mode A (被动观察)** | 感知层 Flush | `context_messages` | 普通对话结束后自动归档 |
| **Mode B (主动响应)** | MTP `WRITE` 指令 | `write_focus` | Agent 明确要求保存某段内容 |
| **Mode C (合并更新)** | MTP `UPDATE` 指令 | `update_focus` | Agent 请求修改已有记忆 |

---

## 6.1 认知流程：帕秋莉的思考链 (Patchouli's Cognitive Chain)

本节定义 Patchouli 接收到对话片段（Transcript Segment）后的内部处理逻辑。
**设计原则**：采用 **"Search-before-Write" (写前查重)** 的严谨模式，确保记忆库的唯一性和时序演化能力，宁可消耗更多的推理 Token，也要保证知识库的纯净度。

### Step 1: 价值校验 (Signal Check)

- **输入**：对话片段（Block）及其附带的 `memory_signal`。
- **动作**：快速布尔判断。
- **逻辑**：
  - 直接读取 Gateway 传入的 `memory_signal.worth_saving` 字段。
  - 若为 `false`，直接丢弃（Drop），不进行任何 LLM 调用。
  - 若为 `true`，进入 Step 2。
- **优势**：相较于旧版设计中在此处重新进行 Gating 判断，新流程完全避免了重复计算。

### Step 2: 记忆原子提取与精炼 (Extraction & Refinement)

**目标**：将非结构化对话转化为结构化的 `MemoryAtom` 草稿。
**核心机制**：Prompt Engineering + 特殊类型处理。

#### 精炼提示词工程 (The Refinement Prompt)

Patchouli 接收特定的 System Prompt，要求她扮演"结构化强迫症"的记录员。

> **\[System Prompt Template — Mode A]**
> 你是 Patchouli，HiveMemory 系统的记忆管理员。
>
> 1. **去噪**：忽略寒暄，提取核心事实、代码逻辑或结论。
> 2. **原子化**：将内容拆解为独立的知识点（Atom）。
> 3. **冰山构建**：
>    - **Index**: 生成精准的 Title, Summary 和 3-5 个动态 Tags（Folksonomy）。
>    - **Payload**: 将核心内容重写为清晰的 Markdown。
> 4. **置信度评估**：根据信息来源（用户指令 vs 模型推理）打分 (0.0-1.0)。
>
> **输出格式**：JSON (ExtractedMemoryDraft Schema).

实际实现中，Prompt 还要求 LLM 为每条记忆生成一个 **`alias_suffix`**（别名后缀），用于 MTP 协议的语义化寻址（详见 6.3 节）。

#### 外源信息快照策略 (External Snapshot Strategy)

当 Patchouli 检测到 Worker Agent 进行了 Web Search 或引用了 URL 时，需执行特殊的**"快照清洗"**流程，以解决版权与链接失效问题。

- **识别**：检测到 `ToolOutput` 包含 URL 或 `Source` 引用。
- **清洗逻辑**：
  - **不存储**：Raw HTML DOM（太脏，且有版权风险）。
  - **存储 (Payload)**：
    1. `source_url`: 原始链接。
    2. `access_date`: 抓取时间。
    3. `clean_markdown`: 提取正文内容的摘要或关键段落（引用形式）。
    4. `copyright_notice`: 自动标记"仅用于内部检索索引"。
- **标记**：设置 `type = "URL_RESOURCE"`，并添加标签 `immutable` (不可变)，防止后续因为内容过时而被错误修改（历史文档应保持原样）。

### Step 3: 查重、合并与演化 (Deduplication, Merge & Evolution)

**目标**：决定是"创建新记忆"还是"更新旧记忆"。
**核心机制**：向量检索 + 决策矩阵。

#### 检索 (Recall)

Patchouli 使用 Step 2 生成的 Draft Atom 的 `Index` 部分（Title + Summary）在向量数据库中执行 **Top-1 检索**。

```python
query_text = f"{draft.title} {draft.summary}"
results = storage.search_memories(query_text=query_text, top_k=1, score_threshold=threshold)
```

#### 决策逻辑矩阵 (Decision Matrix)

Patchouli 根据检索结果的 **相似度分数 (Similarity Score)** 和 **内容差异 (Content Diff)** 执行以下逻辑：

| 场景 | 判别条件 | 逻辑动作 (Action) | 解释 |
| :--- | :--- | :--- | :--- |
| **A. 全新知识** | Score < 0.75 | **CREATE (Insert)** | 库中无相关记录，直接新建。 |
| **B. 完全冗余** | Score > 0.95 **AND 内容几乎一致** | **TOUCH (Skip)** | 知识点已存在。仅更新旧记忆的 `last_accessed_at` 和 `access_count` (权重强化)。 |
| **C. 知识演化** | 0.75 < Score < 0.95 **OR** Score > 0.95 AND 内容有实质冲突 | **EVOLVE (Update)** | **(重点)** 判定为同一实体的状态变更。执行演化流程。 |
| **D. 幻觉/噪音** | Score 高 **BUT Draft 置信度低** AND 旧记忆置信度高 | **DISCARD (Drop)** | **(防污)** 新生成的记忆是 Agent 的推测，而库里存的是用户确定的事实。帕秋莉选择信任旧记忆，丢弃新草稿。 |

内容一致性判断采用 **Jaccard 相似度**（词集合交并比），阈值默认为 0.9。

（**防污染 (Anti-Pollution)**：利用置信度体系，防止 Agent 的胡说八道覆盖了用户设定的真理）

#### 演化执行流程 (Evolution Execution)

当触发 **场景 C (知识演化)** 时，执行 **Git-like Versioning**：

1. **加载旧记忆**：读取库中 Hit 的完整 JSON。
2. **生成 Diff**：对比 `old.payload` 和 `draft.payload`（Jaccard 相似度 < 0.9 时追加，否则直接替换）。
3. **压栈历史**：
   - 将 `old.payload` 移动到 `old.artifacts.history` 列表中。
   - 记录 `timestamp` 和 `change_reason`。
4. **更新 Head**：
   - 将 `draft.payload` 写入 `old.payload`。
   - 更新 `old.index` (Title/Summary/Tags) 以匹配新状态。
5. **继承元数据**：
   - 保留 `old.meta.created_at`。
   - 累加 `access_count`。
   - 新置信度 = 旧置信度 × 0.6 + 新草稿置信度 × 0.4（加权平均）。

### Step 4: 持久化 (Commit)

- **动作**：执行具体的数据库写操作（`storage.upsert_memory(memory)`）。
- **事务性**：确保 Vector DB (索引) 和 Document DB (Payload) 的原子性写入。如果写入失败，回滚操作并记录 Error Log。

---

## 6.2 三模式详解 (Three-Mode Processing)

### Mode A：被动观察 (Passive Observation)

这是最常见的工作模式。感知层在检测到话题切换或 Buffer 超时后，将积累的 `LogicalBlock` 序列 Flush 给生成层。

```
感知层 Flush → GenerationRequest(context_messages=[...]) → Mode A 流程
```

**流程**：
1. 将消息列表格式化为对话文本（`[User]: ... \n[Assistant]: ...`）。
2. 调用 LLM（Mode A Prompt），判断是否有价值（`has_value`）。
3. 若无价值，直接丢弃；若有价值，进入查重 → 构建 → 持久化流程。

### Mode B：主动响应 (Active Response / WRITE)

当 Worker Agent 通过 MTP 协议发出 `WRITE` 指令时触发。Agent 明确提交了一份记忆草稿，Patchouli 的任务从"发现价值"变为"验证并结构化"。

```
MTP WRITE 指令 → WriteFocus(content, reason, title) → Mode B 流程
```

**关键差异**：
- 使用专用的 **Mode B Prompt**，以 `write_content` 为核心，对话历史为背景参考。
- LLM 被要求**强制入库**（`has_value=true, confidence_score=1.0`），除非内容完全错误。
- 包含 **Fallback 机制**：若 LLM 调用失败，直接从 `WriteFocus` 字段构建草稿，保证 WRITE 内容不丢失。

```python
# Fallback 草稿构建（LLM 失败时的保底策略）
draft = ExtractedMemoryDraft(
    title=focus.title or focus.content[:50],
    content=focus.content,
    confidence_score=1.0,
    has_value=True,
    ...
)
```

### Mode C：合并更新 (Merge Update / UPDATE)

当 Worker Agent 通过 MTP 协议发出 `UPDATE` 指令时触发。Agent 请求修改一条已有记忆，Patchouli 执行智能合并。

```
MTP UPDATE 指令 → UpdateFocus(instruction, content, base_alias, base_uuid) → Mode C 流程
```

**流程**：
1. 由 Koakuma 解析 `base_alias`，生成携带 `base_alias/base_uuid` 的 `UpdateFocus`；LibrarianCore 在进入 Mode C 前根据 `base_uuid` 加载目标 `MemoryAtom`，并注入 `GenerationRequest.existing_memory`。
2. 调用 `extractor.merge()`，使用 **Mode C Prompt** 执行 LLM 驱动的智能合并。
3. LLM 输出 `MergeResult { new_content, changelog }`。
4. 执行版本历史追踪：旧内容压入 `artifacts.full_history`，新内容覆盖 `payload.content`，版本号 +1。
5. 持久化（重新生成向量）。

**三种合并模式**（由 LLM 根据指令自动判断）：
- **Replacement（替换）**：用新内容完全替换旧内容。
- **Refinement（精修）**：仅修改特定行或段落，保留其他细节。
- **Append（追加）**：在末尾追加新内容。

同样包含 **Fallback 机制**：LLM 合并失败时，若有 `content` 则直接追加，否则保留旧内容并在 changelog 中记录指令。

---

## 6.3 MTP 别名系统 (MTP Alias System)

为了让 Agent 能够通过语义化名称（而非 UUID）精准定位记忆，Generation 层在创建每条记忆时会自动生成一个 **MTP 别名 (Alias)**。

### 6.3.1 别名构建规则

别名由 **类型前缀** + **`_`** + **语义后缀** 组成：

| 记忆类型 | 前缀 | 示例别名 |
| :--- | :--- | :--- |
| `CODE_SNIPPET` | `code` | `code_quicksort_impl` |
| `FACT` | `fact` | `fact_project_env` |
| `URL_RESOURCE` | `url` | `url_python_datetime_docs` |
| `REFLECTION` | `ref` | `ref_avoid_global_state` |
| `USER_PROFILE` | `user` | `user_prefers_typescript` |
| `WORK_IN_PROGRESS` | `wip` | `wip_refactor_auth_module` |

**后缀生成策略**：
1. 优先使用 LLM 在提取时生成的 `alias_suffix`（snake_case，2-4 个单词，≤30 字符）。
2. LLM 未提供时，从 `title` 派生：转小写 → 去除非字母数字字符 → 空格替换为 `_`。
3. 最终清洗：确保 snake_case 合规，截断至 40 字符。

### 6.3.2 别名的用途

别名主要服务于 MTP 协议中的 `UPDATE` 和 `READ` 指令，允许 Agent 用自然语言风格的名称操作记忆：

```
# Agent 发出 UPDATE 指令
UPDATE code_quicksort_impl
instruction: "将时间复杂度注释从 O(n log n) 改为 O(n²) worst case"
```

---

## 6.4 数据模型参考 (Data Models)

### ExtractedMemoryDraft

LLM 提取阶段的输出草稿，是 Generation 层的核心中间产物。

```python
class ExtractedMemoryDraft(BaseModel):
    title: str              # 简洁明确的标题 (≤100字)
    summary: str            # 一句话摘要 (≤200字)
    tags: List[str]         # 3-5个语义标签 (Folksonomy)
    memory_type: str        # CODE_SNIPPET / FACT / URL_RESOURCE / REFLECTION / USER_PROFILE / WORK_IN_PROGRESS
    content: str            # 清洗后的 Markdown 内容
    confidence_score: float # 置信度 (0.0-1.0)
    has_value: bool         # 是否有长期价值
    alias_suffix: str       # 别名后缀 (snake_case, 由 LLM 生成)
```

### GenerationRequest

Generation Engine 的统一输入协议，封装三种模式的输入。

```python
class GenerationRequest(BaseModel):
    context_messages: List[StreamMessage]  # 感知层 flush 的上下文消息
    write_focus: Optional[WriteFocus]      # Mode B: WRITE 指令聚焦内容
    update_focus: Optional[UpdateFocus]    # Mode C: UPDATE 指令聚焦内容

    @property
    def is_write(self) -> bool: ...  # 是否为 Mode B
    @property
    def is_update(self) -> bool: ...   # 是否为 Mode C
```

### MergeResult

Mode C 中 LLM 执行合并后的结构化输出。

```python
class MergeResult(BaseModel):
    new_content: str  # 合并后的完整内容 (Markdown)
    changelog: str    # 一句话变更日志
```

---

## 6.5 组件接口与扩展 (Interfaces & Extension)

Generation 层遵循**依赖倒置原则**，所有核心组件均通过抽象接口注入，便于替换和测试。

```python
class BaseMemoryExtractor(ABC):
    @abstractmethod
    def extract(self, transcript: str, metadata: Dict) -> Optional[ExtractedMemoryDraft]: ...
    def merge(self, old_content: str, metadata: Dict) -> Optional[MergeResult]: ...  # 默认 NoOp

class BaseDeduplicator(ABC):
    @abstractmethod
    def check_duplicate(self, draft: ExtractedMemoryDraft, threshold: float) -> DuplicateDecision: ...
    @abstractmethod
    def merge_memory(self, existing: MemoryAtom, new_draft: ExtractedMemoryDraft) -> MemoryAtom: ...
```

每个接口均提供 **No-Op 实现**（`NoOpMemoryExtractor`、`NoOpDeduplicator`），可通过配置禁用对应功能，方便在测试或资源受限场景下使用。

---

## 6.6 配置参考 (Configuration)

Generation 模块的行为通过 `ExtractorConfig` 和 `DeduplicatorConfig` 控制，均挂载在 `patchouli.config` 下。

| 配置项 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `extractor.enabled` | `true` | 是否启用 LLM 提取（禁用时使用 NoOp） |
| `deduplicator.enabled` | `true` | 是否启用查重（禁用时总是 CREATE） |
| `deduplicator.high_similarity_threshold` | `0.95` | TOUCH 判定阈值 |
| `deduplicator.low_similarity_threshold` | `0.75` | CREATE/UPDATE 分界阈值 |
| `deduplicator.content_similarity_threshold` | `0.9` | 内容一致性判定阈值（Jaccard） |

LLM 模型通过环境变量配置：

```bash
LIBRARIAN_LLM_MODEL=deepseek/deepseek-chat
LIBRARIAN_LLM_API_KEY=sk-xxxxx
```
