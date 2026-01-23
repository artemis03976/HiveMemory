# MemoryGeneration - 记忆生成模块

## 📖 概述

MemoryGeneration 模块是 HiveMemory 系统的核心组件之一，负责从对话流中自动提取、精炼和存储结构化的记忆原子。

该模块实现了 **PROJECT.md 第 4 章** 定义的完整记忆生成流程。

---

## 🎯 核心职责

1. **对话监听与缓冲** - 累积对话消息，智能判断处理时机
2. **价值评估 (Gating)** - 过滤无长期价值的闲聊和噪音
3. **LLM 驱动的提取** - 将自然对话转换为结构化记忆原子
4. **查重与演化** - 检测重复记忆，支持知识更新与合并
5. **持久化存储** - 生成向量并存储到 Qdrant

---

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                   ConversationBuffer                        │
│         (对话缓冲器 - 累积消息，触发处理)                        │
└────────────────────┬────────────────────────────────────────┘
                     │ 触发处理
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  MemoryOrchestrator                         │
│  (编排器 - 协调整个生成流程)                                 │
│                                                             │
│   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐    │
│   │ ValueGater  │──>│  Extractor  │──>│ Deduplicator │    │
│   │ (价值评估)  │   │ (LLM 提取)  │   │  (查重演化)  │    │
│   └─────────────┘   └─────────────┘   └──────┬───────┘    │
│                                               │             │
│                                               ▼             │
│                                        ┌──────────────┐     │
│                                        │VectorStore   │     │
│                                        │(Qdrant 存储) │     │
│                                        └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 核心组件

### 1. `interfaces.py` - 接口抽象层

定义了模块内所有组件的抽象接口，便于扩展和测试。

```python
from hivememory.generation.interfaces import (
    ValueGater,           # 价值评估器接口
    MemoryExtractor,      # 提取器接口
    Deduplicator,         # 查重器接口
    TriggerStrategy,      # 触发策略接口
)
```

### 2. `gating.py` - 价值评估器

**职责**: 判断对话是否有长期记忆价值

**策略**:
- **规则引擎**: 过滤寒暄、简单确认 ("你好"、"谢谢" 等)
- **LLM 辅助**: 可选的 GPT-4o-mini 辅助判断
- **白名单**: 强制保留特定类型 (代码片段、配置等)

**用法**:
```python
from hivememory.generation.gating import RuleBasedGater

gater = RuleBasedGater()
has_value = gater.evaluate(messages)
```

### 3. `extractor.py` - LLM 记忆提取器

**职责**: 调用 LLM 将对话转换为结构化记忆草稿

**特性**:
- 使用 LiteLLM 统一接口
- Pydantic 输出解析
- JSON 容错与重试机制
- 支持自定义 Prompt

**用法**:
```python
from hivememory.generation.extractor import LLMMemoryExtractor

extractor = LLMMemoryExtractor(llm_config)
draft = extractor.extract(transcript, metadata)
```

### 4. `deduplicator.py` - 查重与演化管理器 ⭐

**职责**: 检测重复记忆，支持知识更新

**决策矩阵**:

| 相似度 | 内容一致 | 决策 | 操作 |
|--------|---------|------|------|
| > 0.95 | 是 | **TOUCH** | 仅更新访问时间 |
| 0.75-0.95 | - | **UPDATE** | 知识演化合并 |
| < 0.75 | - | **CREATE** | 创建新记忆 |

**用法**:
```python
from hivememory.generation.deduplicator import MemoryDeduplicator

dedup = MemoryDeduplicator(storage)
decision = dedup.check_duplicate(draft)

if decision == DuplicateDecision.UPDATE:
    merged = dedup.merge_memory(existing, draft)
```

### 5. `triggers.py` - 触发策略管理

**职责**: 决定何时触发记忆处理

**支持的触发器**:
- `MessageCountTrigger`: 消息数阈值 (默认 5 条)
- `IdleTimeoutTrigger`: 超时触发 (默认 15 分钟)
- `SemanticBoundaryTrigger`: 语义边界 (话题切换、工具调用结束)

**用法**:
```python
from hivememory.generation.triggers import TriggerManager

trigger_mgr = TriggerManager(
    strategies=[
        MessageCountTrigger(threshold=5),
        IdleTimeoutTrigger(timeout=900),
    ]
)

if trigger_mgr.should_trigger(buffer):
    buffer.flush()
```

### 6. `buffer.py` - 对话缓冲器

**职责**: 累积对话消息，管理刷新逻辑

**特性**:
- 线程安全 (threading.Lock)
- 支持手动刷新
- 回调函数机制

**用法**:
```python
from hivememory.generation.buffer import ConversationBuffer

buffer = ConversationBuffer(
    orchestrator=orchestrator,
    user_id="user123",
    agent_id="agent456",
    on_flush_callback=lambda msgs, mems: print(f"提取了 {len(mems)} 条记忆")
)

buffer.add_message("user", "如何解析 ISO8601 日期?")
buffer.add_message("assistant", "使用 datetime.fromisoformat()...")
```

### 7. `orchestrator.py` - 编排器

**职责**: 协调所有组件，执行完整的生成流程

**流程**:
```python
Step 1: 价值评估 (Gating) → Pass/Drop
Step 2: LLM 提取 → ExtractedMemoryDraft
Step 3: 查重检测 → CREATE/UPDATE/TOUCH
Step 4: 记忆原子构建 → MemoryAtom
Step 5: 持久化 → Qdrant
```

**用法**:
```python
from hivememory.generation.orchestrator import MemoryOrchestrator

orchestrator = MemoryOrchestrator(storage=storage)
memories = orchestrator.process(messages, user_id, agent_id)
```

---

## 🚀 快速开始

### 基本用法

```python
from hivememory.generation import MemoryOrchestrator, ConversationBuffer
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.core.models import ConversationMessage

# 1. 初始化存储
storage = QdrantMemoryStore()

# 2. 创建编排器
orchestrator = MemoryOrchestrator(storage=storage)

# 3. 创建缓冲器
buffer = ConversationBuffer(
    orchestrator=orchestrator,
    user_id="user_123",
    agent_id="worker_agent",
)

# 4. 添加对话
buffer.add_message("user", "帮我写个快排算法")
buffer.add_message("assistant", "好的，这是 Python 实现...")

# 5. 触发处理 (自动或手动)
buffer.flush()
```

### 高级用法 - 自定义组件

```python
from hivememory.generation.interfaces import ValueGater
from hivememory.generation.orchestrator import MemoryOrchestrator

class CustomGater(ValueGater):
    """自定义价值评估器"""
    def evaluate(self, messages):
        # 自定义逻辑
        return any("代码" in msg.content for msg in messages)

orchestrator = MemoryOrchestrator(
    storage=storage,
    gater=CustomGater(),  # 注入自定义组件
)
```

---

## 🧪 测试

运行单元测试:
```bash
pytest tests/generation/
```

运行特定测试:
```bash
pytest tests/generation/test_deduplicator.py -v
```

---

## 📊 性能指标

| 指标 | 目标值 | 实际值 |
|------|--------|--------|
| 价值评估准确率 | > 90% | TBD |
| LLM 提取成功率 | > 95% | TBD |
| 查重准确率 | > 85% | TBD |
| 平均处理时间 | < 3s | TBD |

---

## 🔧 配置

### 通过 config.yaml 配置

```yaml
memory:
  buffer:
    max_messages: 5           # 消息数触发阈值
    timeout_seconds: 900      # 超时时间 (15 分钟)

  extraction:
    min_confidence: 0.4       # 最低置信度
    max_tags: 5               # 最多标签数

  deduplication:
    similarity_threshold: 0.75  # 查重相似度阈值
    enable_evolution: true      # 启用知识演化
```

### 通过环境变量配置

```bash
# Librarian LLM 配置
LIBRARIAN_LLM_MODEL=deepseek/deepseek-chat
LIBRARIAN_LLM_API_KEY=sk-xxxxx
```

---

## 📝 设计决策与权衡

### 为什么使用 LLM 而非规则引擎?

- **灵活性**: 自然语言理解优于固定模式匹配
- **泛化能力**: 适应多样化的对话风格
- **可演化**: Prompt 更新即可改进能力

**权衡**: LLM 调用增加延迟和成本

### 为什么需要查重?

- **避免冗余**: 同一知识点反复提及时合并
- **知识演化**: 支持信息更新和修正
- **存储优化**: 减少向量数据库体积

---

## 🛣️ 未来路线图

- [ ] 支持批量提取 (处理长对话)
- [ ] 实现增量更新 (Git-like diff)
- [ ] 添加 A/B 测试框架 (对比不同 Prompt)
- [ ] 实现分布式处理 (Celery 异步队列)

---

## 📚 相关文档

- [PROJECT.md 第 4 章](../../docs/PROJECT.md) - 完整设计文档
- [ROADMAP.md Stage 1](../../docs/ROADMAP.md) - 开发路线图
- [API 文档](../../docs/API.md) - 接口说明 (TBD)

---

## 🤝 贡献指南

1. 所有新增组件必须实现对应的抽象接口
2. 添加完整的类型注解和 Docstring
3. 编写单元测试 (覆盖率 > 80%)
4. 更新本 README 和示例代码

---

**维护者**: HiveMemory Team
**最后更新**: 2025-12-23
**版本**: 0.1.0
