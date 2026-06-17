# MemoryPerception - 记忆感知模块

## 概述

MemoryPerception 模块是 HiveMemory 系统的 "感官" 入口，负责实时监听、解析和组织来自不同来源（如 LangChain, OpenAI API）的原始对话流。

该模块实现了 **PROJECT.md 2.3.1 节** 定义的 **语义流感知层 (Semantic Flow Perception Layer)**，能够智能地识别话题边界，将碎片化的消息流组织成连贯的 **逻辑块 (LogicalBlock)**，并在语义漂移时自动触发记忆生成流程。

---

## 核心职责

1.  **流式解析与归一化** - 抹平不同 Agent 框架（LangChain, OpenAI）的消息格式差异
2.  **逻辑块构建** - 将 User Query 及其后续的思维链、工具调用、最终响应组织为原子单元
3.  **语义吸附 (Adsorption)** - 基于 Embedding 相似度判断上下文连贯性，自动识别话题切换
4.  **上下文接力 (Relay)** - 在 Token 溢出时生成中间态摘要，维持长对话的记忆连贯性
5.  **异步空闲监控** - 后台监控 Buffer 空闲状态，超时自动触发 Flush
6.  **多话题并发管理** - 支持多话题并发生命周期管理，LRU 驱逐策略

---

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     Raw Message Stream                      │
│            (LangChain / OpenAI / Plain Text)                │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  SemanticFlowPerceptionLayer                │
│                         (MMU / 内存管理单元)                  │
│                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌─────────────┐   │
│   │ StreamParser │───>│ LogicalBlock │───>│   Buffer    │   │
│   │ (解析/归一)   │    │ (逻辑块构建)   │    │ (语义缓冲)   │   │
│   └──────────────┘    └──────────────┘    └──────┬──────┘   │
│                                                  │          │
│   ┌──────────────────┐                           │          │
│   │IdleTimeoutMonitor│◄──────────────────────────┤          │
│   │  (异步超时监控)    │       (后台扫描)            │          │
│   └────────┬─────────┘                           ▼          │
│            │                                                │
│            │                                                │
│            │                                                │
│            │                                                │
│            │                                                │
│   ┌────────┴───────┐                                        │
│   │ RelayController│                                        │
│   │ (接力/摘要)    │     (TOKEN_OVERFLOW)        │           │
│   └────────┬───────┘                            │           │
│            │                                    │ (Flush)   │
│            │ (Summary)                          ▼           │
│            │                          ┌──────────────────┐  │
│            └─────────────────────────>│Generation Module │  │
│                                       │   (记忆生成)      │  │
│                                       └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心组件

### 1. `interfaces.py` - 接口抽象层

定义了感知层所有组件的抽象基类，遵循依赖倒置原则。

```python
from hivememory.perception.interfaces import (
    BasePerceptionLayer,    # 感知层基类
)
```

### 2. `semantic_flow_perception_layer.py` - 语义流感知层 / MMU

**职责**: 模块的主入口，协调解析、吸附、接力和空闲监控过程，管理会话 Buffer。作为短期记忆的内存管理单元 (MMU)，管理多话题的并发生命周期。

**特性**:
- 话题路由：根据 TheEye 的 target_topic 将载荷路由到正确话题
- LRU 驱逐：活跃话题池满时驱逐最久未访问的话题
- URGENT 信号：write_focus/update_focus 触发立即 flush
- 空闲超时：长期不活跃的话题自动换出

**用法**:
```python
from hivememory.perception import SemanticFlowPerceptionLayer

def on_flush(messages, reason):
    print(f"触发记忆生成: {reason}, 消息数: {len(messages)}")

layer = SemanticFlowPerceptionLayer(
    on_flush_callback=on_flush,
    idle_timeout_seconds=900,   # 15分钟空闲超时
    scan_interval_seconds=30,   # 30秒扫描间隔
)

# 启动异步空闲监控（可选）
layer.start_idle_monitor()

# 摄入载荷
from hivememory.perception import InteractionPayload
payload = InteractionPayload(
    user_message="Hello",
    assistant_message="Hi there!",
    identity=identity,
)
layer.ingest_payload(payload)

# 或使用路由模式
layer.route_and_ingest("NEW_TOPIC", payload)

# 停止监控（程序退出前）
layer.stop_idle_monitor()
```

### 3. `relay_controller.py` - 接力控制器 / Page Folding 摘要生成器

**职责**: 检测 Token 溢出，生成摘要以便在下一个 Buffer 中通过 Context Injection 维持连贯性。

**用法**:
```python
from hivememory.perception import LLMRelayController

controller = LLMRelayController()
if controller.should_trigger_relay(buffer, new_block):
    summary = controller.generate_summary(buffer.blocks)
    # 将 summary 注入下一个 buffer
```

### 5. `buffer_manager.py` - 话题管理器 / MMU

**职责**: 管理活跃话题池的生命周期，提供 CRUD 操作、LRU 驱逐判定和话题路由。

**用法**:
```python
from hivememory.perception import SemanticBufferManager

manager = SemanticBufferManager(max_resident_topics=5)
buffer = manager.create_topic_buffer(identity)
buffers = manager.get_all_buffers()
```

---

## 快速开始

### 集成到 Agent 循环中

```python
from hivememory.perception import SemanticFlowPerceptionLayer, InteractionPayload
from hivememory.core.models import Identity

# 1. 初始化
perception = SemanticFlowPerceptionLayer(
    on_flush_callback=on_flush,
    idle_timeout_seconds=900,  # 15分钟超时
)

# 2. 启动异步空闲监控
perception.start_idle_monitor()

# 3. 在 Agent 循环中调用
identity = Identity(user_id="u1", agent_id="a1", session_id="s1")

def chat_loop(user_input, response):
    payload = InteractionPayload(
        user_message=user_input,
        assistant_message=response,
        identity=identity,
    )
    perception.ingest_payload(payload)

# 4. 手动触发结算 (可选)
await perception.manual_trigger()

# 5. 程序退出前停止监控
perception.stop_idle_monitor()
```

---

## Flush 触发机制

| 触发类型 | 触发条件 | 触发时机 | 负责组件 |
|----------|----------|----------|----------|
| `TOKEN_OVERFLOW` | Token 数超过阈值 | 新 Block 加入前 | RelayController |
| `IDLE_TIMEOUT` | Buffer 空闲超时 | 后台异步扫描 | IdleTimeoutMonitor |
| `MANUAL` | 用户手动调用 | 调用 manual_trigger() | 用户代码 |
| `MTP_WRITE` | MTP WRITE 指令 | 载荷摄入时 | SemanticFlowPerceptionLayer |
| `MTP_UPDATE` | MTP UPDATE 指令 | 载荷摄入时 | SemanticFlowPerceptionLayer |
| `LRU_EVICTION` | 活跃话题池满 | 新话题创建时 | SemanticBufferManager |

---

## 性能指标

| 组件 | 指标 | 目标值 | 说明 |
|------|------|--------|------|
| StreamParser | 解析延迟 | < 5ms | 单条消息解析 |
| IdleTimeoutMonitor | 扫描延迟 | < 100ms | 全 Buffer 池扫描 |

---

## 设计决策

### 为什么引入 LogicalBlock？
传统的基于消息（Message-based）的处理容易切断 User Query 与 Tool Call 之间的逻辑联系。LogicalBlock 将一次完整的交互（Query -> Tools -> Response）视为原子单元，确保记忆提取时上下文的完整性。

### 为什么需要语义吸附？
固定消息数（如每 10 条）或固定时间切分往往会打断正在进行的话题。语义吸附通过 Embedding 实时计算话题相似度，实现"话题结束即切分"的动态边界，提高记忆生成的质量。

### 为什么空闲超时要异步监控？
原有设计中，空闲超时只在添加新消息时检测，导致如果用户长时间不活动，旧 Buffer 不会被 Flush。使用 `IdleTimeoutMonitor` 后台定时扫描，确保即使没有新消息，超时的 Buffer 也能被及时处理。

---

## 相关文档

- [PROJECT.md 2.3.1 感知层](../../docs/PROJECT.md)
- [MemoryGeneration README](../generation/README.md)

---

**维护者**: HiveMemory Team
**最后更新**: 2026-03-02
**版本**: 4.5.0
