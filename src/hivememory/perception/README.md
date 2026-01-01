# MemoryPerception - 记忆感知模块

## 📖 概述

MemoryPerception 模块是 HiveMemory 系统的 "感官" 入口，负责实时监听、解析和组织来自不同来源（如 LangChain, OpenAI API）的原始对话流。

该模块实现了 **PROJECT.md 2.3.1 节** 定义的 **语义流感知层 (Semantic Flow Perception Layer)**，能够智能地识别话题边界，将碎片化的消息流组织成连贯的 **逻辑块 (LogicalBlock)**，并在语义漂移时自动触发记忆生成流程。

---

## 🎯 核心职责

1.  **流式解析与归一化** - 抹平不同 Agent 框架（LangChain, OpenAI）的消息格式差异
2.  **逻辑块构建** - 将 User Query 及其后续的思维链、工具调用、最终响应组织为原子单元
3.  **语义吸附 (Adsorption)** - 基于 Embedding 相似度判断上下文连贯性，自动识别话题切换
4.  **上下文接力 (Relay)** - 在 Token 溢出时生成中间态摘要，维持长对话的记忆连贯性
5.  **多策略触发** - 支持基于消息数、时间、语义边界的灵活触发机制

---

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     Raw Message Stream                      │
│            (LangChain / OpenAI / Plain Text)                │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  SemanticFlowPerceptionLayer                │
│                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌─────────────┐   │
│   │ StreamParser │───>│ LogicalBlock │───>│   Buffer    │   │
│   │ (解析/归一)  │    │ (逻辑块构建) │    │ (语义缓冲)  │   │
│   └──────────────┘    └──────────────┘    └──────┬──────┘   │
│                                                  │          │
│                                                  ▼          │
│   ┌──────────────┐                       ┌──────────────┐   │
│   │RelayControl  │<──────────────────────│   Adsorber   │   │
│   │(接力/摘要)   │      (Token溢出)      │  (语义吸附)  │   │
│   └──────┬───────┘                       └──────┬───────┘   │
│          │                                      │           │
│          │ (Summary)                            │ (Flush)   │
│          └───────────────────┐                  ▼           │
│                              │        ┌──────────────────┐  │
│                              └───────>│Generation Module │  │
│                                       │   (记忆生成)     │  │
│                                       └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 核心组件

### 1. `interfaces.py` - 接口抽象层

定义了感知层所有组件的抽象基类，遵循依赖倒置原则。

```python
from hivememory.perception.interfaces import (
    StreamParser,      # 流式解析器接口
    SemanticAdsorber,  # 语义吸附器接口
    RelayController,   # 接力控制器接口
    BasePerceptionLayer, # 感知层基类
)
```

### 2. `semantic_flow_perception_layer.py` - 语义流感知层

**职责**: 模块的主入口，协调解析、吸附和接力过程，管理会话 Buffer。

**用法**:
```python
from hivememory.perception import SemanticFlowPerceptionLayer

def on_flush(messages, reason):
    print(f"触发记忆生成: {reason}, 消息数: {len(messages)}")

layer = SemanticFlowPerceptionLayer(on_flush_callback=on_flush)

# 添加消息 (支持多会话隔离)
layer.add_message("user", "Python 里的 GIL 是什么？", "user1", "agent1", "session1")
layer.add_message("assistant", "GIL 是全局解释器锁...", "user1", "agent1", "session1")
```

### 3. `stream_parser.py` - 统一流式解析器

**职责**: 将异构的原始消息转换为标准化的 `StreamMessage`，并识别 `LogicalBlock` 边界。

**支持格式**:
- LangChain (`AIMessage`, `HumanMessage`, `ToolMessage`)
- OpenAI API (`{"role": "...", "content": "..."}`)
- 纯文本字符串

**用法**:
```python
from hivememory.perception import UnifiedStreamParser

parser = UnifiedStreamParser()
msg = parser.parse_message({"role": "user", "content": "hello"})
# 输出: StreamMessage(type=USER_QUERY, content="hello")
```

### 4. `semantic_adsorber.py` - 语义边界吸附器 ⭐

**职责**: 决定新的逻辑块是"吸附"到当前话题，还是因"语义漂移"触发刷新。

**核心逻辑**:
1.  **短文本强吸附**: 避免因简短回复（"好的"）导致切分
2.  **Token 溢出检测**: 预判 Token 是否超限
3.  **空闲超时**: 长时间无交互自动切分
4.  **语义相似度**: 计算新 Block 与当前话题核心向量的 Cosine 相似度

**用法**:
```python
from hivememory.perception import SemanticBoundaryAdsorber

adsorber = SemanticBoundaryAdsorber(
    semantic_threshold=0.6,
    short_text_threshold=50
)
should_adsorb, reason = adsorber.should_adsorb(new_block, buffer)
```

### 5. `relay_controller.py` - 接力控制器

**职责**: 处理长对话导致的 Token 溢出，生成摘要以便在下一个 Buffer 中通过 Context Injection 维持连贯性。

**用法**:
```python
from hivememory.perception import TokenOverflowRelayController

controller = TokenOverflowRelayController(max_processing_tokens=8192)
if controller.should_trigger_relay(buffer, new_block):
    summary = controller.generate_summary(buffer.blocks)
    # 将 summary 注入下一个 buffer
```

### 6. `trigger_strategies.py` - 基础触发策略

**职责**: 提供基础的触发判断逻辑（主要用于 `SimplePerceptionLayer` 或作为辅助策略）。

**支持**:
- `MessageCountTrigger`: 消息计数
- `IdleTimeoutTrigger`: 时间阈值
- `SemanticBoundaryTrigger`: 关键词/正则匹配结束语

---

## 🚀 快速开始

### 集成到 Agent 循环中

```python
from hivememory.perception import SemanticFlowPerceptionLayer
from hivememory.generation import MemoryOrchestrator

# 1. 初始化
orchestrator = MemoryOrchestrator(...)
perception = SemanticFlowPerceptionLayer(
    on_flush_callback=orchestrator.process  # 连接到生成模块
)

# 2. 在 Agent 循环中调用
def chat_loop(user_input):
    # 用户输入
    perception.add_message("user", user_input, "u1", "a1", "s1")
    
    # ... Agent 执行逻辑 ...
    response = agent.run(user_input)
    
    # Agent 响应
    perception.add_message("assistant", response, "u1", "a1", "s1")

# 3. 手动刷新 (可选)
perception.flush_buffer("u1", "a1", "s1")
```

### 自定义配置

```python
from hivememory.perception import (
    SemanticFlowPerceptionLayer,
    SemanticBoundaryAdsorber,
    UnifiedStreamParser
)

# 自定义吸附策略
adsorber = SemanticBoundaryAdsorber(
    semantic_threshold=0.75,  # 更严格的语义匹配
    idle_timeout_seconds=300  # 5分钟超时
)

# 启用 Claude 思考过程提取
parser = UnifiedStreamParser(enable_thought_extraction=True)

layer = SemanticFlowPerceptionLayer(
    parser=parser,
    adsorber=adsorber
)
```

---

## 📊 性能指标

| 组件 | 指标 | 目标值 | 说明 |
|------|------|--------|------|
| StreamParser | 解析延迟 | < 5ms | 单条消息解析 |
| SemanticAdsorber | 判定延迟 | < 50ms | 包含 Embedding 计算 |
| Adsorption | 准确率 | > 90% | 话题边界识别准确性 |

---

## 📝 设计决策

### 为什么引入 LogicalBlock？
传统的基于消息（Message-based）的处理容易切断 User Query 与 Tool Call 之间的逻辑联系。LogicalBlock 将一次完整的交互（Query -> Tools -> Response）视为原子单元，确保记忆提取时上下文的完整性。

### 为什么需要语义吸附？
固定消息数（如每 10 条）或固定时间切分往往会打断正在进行的话题。语义吸附通过 Embedding 实时计算话题相似度，实现"话题结束即切分"的动态边界，提高记忆生成的质量。

---

## 📚 相关文档

- [PROJECT.md 2.3.1 感知层](../../docs/PROJECT.md)
- [MemoryGeneration README](../generation/README.md)

---

**维护者**: HiveMemory Team
**最后更新**: 2026-01-01
**版本**: 0.1.0
