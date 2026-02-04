# 冷链路 (Cold Path) 端到端测试设计文档

> **版本**: 1.0.0
> **作者**: HiveMemory Team
> **最后更新**: 2026-02-01

---

## 1. 测试范围与目标

### 1.1 被测链路

```
Worker Agent Message -> Gateway -> Observation -> Perception (StreamParser + LogicalBlockBuilder + SemanticAdsorber + RelayController) -> Generation (LLMExtractor + Deduplicator) -> Storage (QdrantMemoryStore)
```

**链路说明**: 冷链路负责将用户与 LLM 的对话异步处理为长期记忆，包括话题切分、记忆提取、去重演化和持久化存储。

### 1.2 核心模块

| 模块 | 文件路径 | 核心职责 |
|------|----------|----------|
| SemanticFlowPerceptionLayer | `src/hivememory/engines/perception/semantic_flow_perception_layer.py` | 编排 flush 逻辑，协调 BufferManager、Adsorber 和 Relay |
| UnifiedStreamParser | `src/hivememory/engines/perception/stream_parser.py` | 解析原始消息为 StreamMessage |
| LogicalBlockBuilder | `src/hivememory/engines/perception/models.py` | 构建 LogicalBlock，管理状态机 |
| SemanticBoundaryAdsorber | `src/hivememory/engines/perception/semantic_adsorber.py` | 语义漂移检测，三阶段处理管道 |
| RelayController | `src/hivememory/engines/perception/relay_controller.py` | Token 溢出检测与接力摘要生成 |
| MemoryGenerationEngine | `src/hivememory/engines/generation/engine.py` | 协调 LLM 提取、查重、存储 |
| LLMMemoryExtractor | `src/hivememory/engines/generation/extractor.py` | 调用 LLM 提取记忆草稿 |
| SemanticDeduplicator | `src/hivememory/engines/generation/deduplicator.py` | 查重决策 (CREATE/UPDATE/TOUCH/DISCARD) |
| QdrantMemoryStore | `src/hivememory/infrastructure/storage/vector_store.py` | 向量存储与检索 |

### 1.3 测试目标与验收标准

| 目标类型 | 具体指标 | 验收标准 |
|----------|----------|----------|
| 功能正确性 | 语义漂移检测准确率 | >= 90% |
| 功能正确性 | Token 溢出触发准确率 | 100% |
| 功能正确性 | 记忆提取召回率 | >= 85% |
| 功能正确性 | 去重决策准确率 | >= 95% |
| 性能指标 | 单 Block 处理延迟 (P95) | < 100ms |
| 性能指标 | 记忆生成延迟 (P95) | < 3000ms (含 LLM 调用) |
| 稳定性 | 异常恢复成功率 | 100% |
| 稳定性 | 内存泄漏 | 0 |

---

## 2. 测试策略

### 2.1 Mock 策略

| 组件 | Mock 策略 | 理由 |
|------|-----------|------|
| EmbeddingService | **真实调用** (本地 BGE-M3) | 语义计算是Perception模块的核心逻辑，必须真实验证 |
| LLMService | **真实调用** | LLM服务是Gateway模块与Extractor组件的核心，驱动整个冷链路 |
| QdrantMemoryStore | **真实调用** (Docker 容器) | 存储是最终出口，必须验证实际写入 |
| Arbiter | **可配置** | P0 测试 Mock RerankService，P1 测试真实调用 |

### 2.2 数据隔离方案

| 隔离维度 | 策略 | 实现方式 |
|----------|------|----------|
| 用户隔离 | 每个测试用例使用唯一 user_id | `user_id = f"test_user_{uuid4()}"` |
| 会话隔离 | 每个测试用例使用唯一 session_id | `session_id = f"test_session_{uuid4()}"` |
| 存储隔离 | 使用独立的 Qdrant Collection | `collection_name = f"test_cold_{timestamp}"` |
| 测试后清理 | fixture 自动清理 | `@pytest.fixture(scope="function", autouse=True)` |

**Fixture 示例**:
```python
@pytest.fixture(scope="function")
def isolated_storage(qdrant_client):
    collection_name = f"test_cold_{int(time.time() * 1000)}"
    qdrant_client.create_collection(collection_name, ...)
    yield QdrantMemoryStore(collection_name=collection_name)
    qdrant_client.delete_collection(collection_name)
```

### 2.3 并发与性能边界

| 场景 | 并发级别 | 预期行为 | 测试方法 |
|------|----------|----------|----------|
| 单用户多消息 | 串行处理 | Buffer 状态一致性保证 | 顺序发送消息，验证 Buffer 状态 |
| 多用户并发 | 并行处理 | 各 Buffer 独立，无竞争 | 多线程模拟，验证无数据串扰 |
| Buffer 空闲超时 | 异步监控 | 定时扫描，触发 flush | Mock 时间，验证超时触发 |
| 高负载压测 | 100 QPS | 无消息丢失，延迟可控 | locust 压测脚本 |

---

## 3. 测试点拆解

### 3.1 Perception 层测试点 (7 个)

#### CP-PER-001: 语义漂移检测 - 高阈值吸附

| 属性 | 值 |
|------|-----|
| **ID** | CP-PER-001 |
| **名称** | 语义漂移检测 - 高阈值吸附 |
| **优先级** | P0 |
| **描述** | 验证相似度 >= 0.55 时，新 Block 被吸附到当前 Buffer |
| **前置条件** | Buffer 已有 1 个 Block (数据科学话题) |
| **触发动作** | 添加同话题的新 Block |
| **预期结果** | FlushEvent 为 None，Buffer.blocks 数量 +1 |

**输入-处理-输出-断言**:
```
输入:
  - base_block: "Python数据可视化库Matplotlib的使用方法"
  - new_block: "Matplotlib绑图的基本语法和参数设置"

处理:
  1. 计算 new_block.anchor_text 与 buffer.topic_kernel_vector 的余弦相似度
  2. similarity >= 0.55 → 进入吸附逻辑

输出:
  - FlushEvent: None
  - buffer.blocks: [base_block, new_block]

断言:
  - assert flush_event is None
  - assert len(buffer.blocks) == 2
  - assert buffer.total_tokens == base_tokens + new_tokens
```

**测试数据组**:
| 数据ID | 基础话题 | 新查询 | 预期相似度 |
|--------|----------|--------|------------|
| CP-PER-001-A | Python数据可视化库Matplotlib的使用方法 | Matplotlib绑图的基本语法和参数设置 | >= 0.55 |
| CP-PER-001-B | Docker容器的基本概念和架构 | Docker镜像和容器的区别是什么 | >= 0.55 |
| CP-PER-001-C | Python快速排序算法的实现 | 这个排序算法的时间复杂度是多少 | >= 0.55 |

---

#### CP-PER-002: 语义漂移检测 - 低阈值切分

| 属性 | 值 |
|------|-----|
| **ID** | CP-PER-002 |
| **名称** | 语义漂移检测 - 低阈值切分 |
| **优先级** | P0 |
| **描述** | 验证相似度 < 0.45 时，触发 SEMANTIC_DRIFT flush |
| **前置条件** | Buffer 已有 2 个 Block (技术话题) |
| **触发动作** | 添加完全不相关话题的 Block |
| **预期结果** | FlushEvent.flush_reason == SEMANTIC_DRIFT |

**输入-处理-输出-断言**:
```
输入:
  - buffer_topic: "Python机器学习库scikit-learn的分类算法"
  - new_block: "红烧茄子的做法和调料配比"

处理:
  1. 计算相似度 → similarity < 0.45
  2. 触发 SEMANTIC_DRIFT flush

输出:
  - FlushEvent(flush_reason=SEMANTIC_DRIFT, blocks_to_flush=[...])

断言:
  - assert flush_event is not None
  - assert flush_event.flush_reason == FlushReason.SEMANTIC_DRIFT
  - assert len(flush_event.blocks_to_flush) >= 1
```

**测试数据组**:
| 数据ID | 基础话题 | 新查询 | 预期相似度 |
|--------|----------|--------|------------|
| CP-PER-002-A | Python机器学习库scikit-learn的分类算法 | 红烧茄子的做法和调料配比 | < 0.45 |
| CP-PER-002-B | React前端框架的组件生命周期 | 北京故宫的门票价格和开放时间 | < 0.45 |
| CP-PER-002-C | MySQL索引优化和查询性能调优 | 健身房增肌训练计划和饮食建议 | < 0.45 |

---

#### CP-PER-003: 灰度区间仲裁

| 属性 | 值 |
|------|-----|
| **ID** | CP-PER-003 |
| **名称** | 灰度区间仲裁 |
| **优先级** | P1 |
| **描述** | 验证 0.45 <= 相似度 < 0.55 时，Arbiter 介入决策 |
| **前置条件** | Buffer 已有 1 个 Block，Arbiter 已启用 |
| **触发动作** | 添加相关但不同领域的 Block |
| **预期结果** | Arbiter 被调用，返回明确的吸附/切分决策 |

**测试数据组**:
| 数据ID | 基础话题 | 新查询 | 仲裁上下文 |
|--------|----------|--------|------------|
| CP-PER-003-A | Python数据可视化库Matplotlib | JavaScript前端图表库D3.js的使用 | 两者都是可视化库，但语言不同 |
| CP-PER-003-B | Vue.js组件通信和状态管理 | Spring Boot的RESTful API设计 | 都是Web开发，但前后端不同 |

---

#### CP-PER-004: 短文本强吸附

| 属性 | 值 |
|------|-----|
| **ID** | CP-PER-004 |
| **名称** | 短文本强吸附 |
| **优先级** | P1 |
| **描述** | 验证停用词短文本绕过向量计算，直接吸附 |
| **前置条件** | Buffer 已有 1 个 Block |
| **触发动作** | 添加 "好的"、"继续"、"ok" 等短文本 |
| **预期结果** | 不触发 Embedding 计算，直接吸附 |

**测试数据组**:
| 数据ID | 短文本 | 预期行为 |
|--------|--------|----------|
| CP-PER-004-A | 好的 | 直接吸附，不计算向量 |
| CP-PER-004-B | 继续 | 直接吸附，不计算向量 |
| CP-PER-004-C | ok | 直接吸附，不计算向量 |
| CP-PER-004-D | 然后呢 | 直接吸附，不计算向量 |
| CP-PER-004-E | 明白了 | 直接吸附，不计算向量 |

---

#### CP-PER-005: Token 溢出触发接力

| 属性 | 值 |
|------|-----|
| **ID** | CP-PER-005 |
| **名称** | Token 溢出触发接力 |
| **优先级** | P0 |
| **描述** | 验证 Buffer Token 超过 8192 时触发 TOKEN_OVERFLOW flush |
| **前置条件** | Buffer 累计 Token 接近阈值 |
| **触发动作** | 添加使总 Token 超过阈值的 Block |
| **预期结果** | FlushEvent.flush_reason == TOKEN_OVERFLOW，relay_summary 非空 |

**输入-处理-输出-断言**:
```
输入:
  - buffer.total_tokens: 7500
  - new_block.total_tokens: 1000
  - max_processing_tokens: 8192

处理:
  1. projected_tokens = 7500 + 1000 = 8500 > 8192
  2. 触发 TOKEN_OVERFLOW flush
  3. 生成 relay_summary

输出:
  - FlushEvent(flush_reason=TOKEN_OVERFLOW, relay_summary="处理了 N 个用户请求...")

断言:
  - assert flush_event.flush_reason == FlushReason.TOKEN_OVERFLOW
  - assert flush_event.relay_summary is not None
  - assert len(flush_event.relay_summary) > 0
```

**测试数据组**:
| 数据ID | 初始 Tokens | 新增 Tokens | 预期行为 |
|--------|-------------|-------------|----------|
| CP-PER-005-A | 7500 | 1000 | 触发 TOKEN_OVERFLOW |
| CP-PER-005-B | 4000 | 5000 | 触发 TOKEN_OVERFLOW |
| CP-PER-005-C | 8000 | 100 | 不触发（边界测试） |

---

#### CP-PER-006: 空闲超时触发

| 属性 | 值 |
|------|-----|
| **ID** | CP-PER-006 |
| **名称** | 空闲超时触发 |
| **优先级** | P1 |
| **描述** | 验证 Buffer 空闲超过 15 分钟时触发 IDLE_TIMEOUT flush |
| **前置条件** | Buffer 已有 Block，启动空闲监控 |
| **触发动作** | 模拟时间流逝超过超时阈值 |
| **预期结果** | FlushEvent.flush_reason == IDLE_TIMEOUT |

**测试数据组**:
| 数据ID | 空闲时间 (秒) | 预期行为 |
|--------|---------------|----------|
| CP-PER-006-A | 901 | 触发 IDLE_TIMEOUT |
| CP-PER-006-B | 899 | 不触发 |
| CP-PER-006-C | 1800 | 触发 IDLE_TIMEOUT |

---

#### CP-PER-007: Agent 工具调用解析

| 属性 | 值 |
|------|-----|
| **ID** | CP-PER-007 |
| **名称** | Agent 工具调用解析 |
| **优先级** | P1 |
| **描述** | 验证 Triplet (Thought -> Tool Call -> Observation) 正确解析 |
| **前置条件** | 无 |
| **触发动作** | 输入包含工具调用的消息序列 |
| **预期结果** | LogicalBlock.execution_chain 包含完整 Triplet |

**测试数据组**:
| 数据ID | 场景 | 工具数量 | 预期 Triplet 数 |
|--------|------|----------|-----------------|
| CP-PER-007-A | 单工具调用 (get_weather) | 1 | 1 |
| CP-PER-007-B | 多工具调用 (analyze + optimize) | 2 | 2 |
| CP-PER-007-C | 工具调用失败重试 | 2 | 2 |

---

### 3.2 Generation 层测试点 (5 个)

#### CP-GEN-001: 有价值记忆提取

| 属性 | 值 |
|------|-----|
| **ID** | CP-GEN-001 |
| **名称** | 有价值记忆提取 |
| **优先级** | P0 |
| **描述** | 验证包含事实性信息的对话被正确提取为 MemoryAtom |
| **前置条件** | LLM Mock 返回有效的 ExtractedMemoryDraft |
| **触发动作** | 调用 MemoryGenerationEngine.process(messages) |
| **预期结果** | 返回非空 List[MemoryAtom]，字段完整 |

**输入-处理-输出-断言**:
```
输入:
  - messages: [用户询问 API Key 配置, Assistant 回复配置方法]

处理:
  1. LLMExtractor 提取 → ExtractedMemoryDraft(has_value=True)
  2. Deduplicator 检查 → CREATE
  3. 构建 MemoryAtom → 写入 Storage

输出:
  - List[MemoryAtom] 长度 >= 1

断言:
  - assert len(memories) >= 1
  - assert memories[0].index.title is not None
  - assert memories[0].index.memory_type == MemoryType.FACT
  - assert memories[0].payload.content is not None
```

**测试数据组**:
| 数据ID | 对话场景 | 预期 memory_type | 预期 title 包含 |
|--------|----------|------------------|-----------------|
| CP-GEN-001-A | API Key 配置信息 | FACT | API, Key |
| CP-GEN-001-B | Python 快速排序代码 | CODE_SNIPPET | 排序, Python |
| CP-GEN-001-C | 用户偏好 VSCode + Vim | USER_PROFILE | 偏好, VSCode |

---

#### CP-GEN-002: 噪音过滤

| 属性 | 值 |
|------|-----|
| **ID** | CP-GEN-002 |
| **名称** | 噪音过滤 |
| **优先级** | P0 |
| **描述** | 验证无营养的闲聊对话被判定为无价值，不生成记忆 |
| **前置条件** | LLM Mock 返回 has_value=False |
| **触发动作** | 调用 MemoryGenerationEngine.process(messages) |
| **预期结果** | 返回空列表 |

**测试数据组**:
| 数据ID | 对话内容 | 预期 has_value |
|--------|----------|----------------|
| CP-GEN-002-A | "好的，谢谢你的帮助" / "不客气" | False |
| CP-GEN-002-B | "你好" / "你好！有什么可以帮助你的吗？" | False |
| CP-GEN-002-C | "明白了" / "好的，如果还有问题随时问我" | False |

---

#### CP-GEN-003: 去重决策 - CREATE

| 属性 | 值 |
|------|-----|
| **ID** | CP-GEN-003 |
| **名称** | 去重决策 - CREATE |
| **优先级** | P0 |
| **描述** | 验证与现有记忆相似度 < 0.75 时，决策为 CREATE |
| **前置条件** | Qdrant 中已有 PyTorch 安装指南记忆 |
| **触发动作** | 提取 Rust 语言入门教程 |
| **预期结果** | DuplicateDecision == CREATE，新记忆被写入 |

**测试数据组**:
| 数据ID | 现有记忆 | 新记忆 | 预期决策 |
|--------|----------|--------|----------|
| CP-GEN-003-A | PyTorch 安装指南 | Rust 语言入门 | CREATE |
| CP-GEN-003-B | React 组件开发 | Django REST API | CREATE |
| CP-GEN-003-C | MySQL 索引优化 | Redis 缓存策略 | CREATE |

---

#### CP-GEN-004: 去重决策 - UPDATE (知识演化)

| 属性 | 值 |
|------|-----|
| **ID** | CP-GEN-004 |
| **名称** | 去重决策 - UPDATE (知识演化) |
| **优先级** | P0 |
| **描述** | 验证相似度 0.75-0.95 且内容有实质变化时，决策为 UPDATE |
| **前置条件** | Qdrant 中已有项目周会时间记忆 |
| **触发动作** | 提取周会时间调整信息 |
| **预期结果** | DuplicateDecision == UPDATE，旧记忆被合并更新 |

**输入-处理-输出-断言**:
```
输入:
  - existing_memory: "项目周会时间：每周三下午2点"
  - new_draft: "项目周会时间调整为每周四下午3点"

处理:
  1. 向量检索 → similarity = 0.85
  2. 内容对比 → 有实质变化
  3. 决策 → UPDATE
  4. 合并 → 追加版本历史

输出:
  - updated_memory.payload.content 包含新旧信息
  - updated_memory.payload.history_summary 有记录

断言:
  - assert decision == DuplicateDecision.UPDATE
  - assert "周四" in updated_memory.payload.content
  - assert updated_memory.meta.version > existing_memory.meta.version
```

**测试数据组**:
| 数据ID | 现有记忆 | 新信息 | 预期决策 |
|--------|----------|--------|----------|
| CP-GEN-004-A | 周会时间：周三下午2点 | 周会调整为周四下午3点 | UPDATE |
| CP-GEN-004-B | API 端点：/api/v1/users | API 升级为 /api/v2/users | UPDATE |
| CP-GEN-004-C | 项目代号：Phoenix | 项目代号改为 Prometheus | UPDATE |

---

#### CP-GEN-005: 去重决策 - TOUCH

| 属性 | 值 |
|------|-----|
| **ID** | CP-GEN-005 |
| **名称** | 去重决策 - TOUCH |
| **优先级** | P1 |
| **描述** | 验证相似度 > 0.95 且内容一致时，仅更新访问时间 |
| **前置条件** | Qdrant 中已有 PyTorch 安装指南记忆 |
| **触发动作** | 提取几乎相同的 PyTorch 安装信息 |
| **预期结果** | DuplicateDecision == TOUCH，access_count +1 |

**测试数据组**:
| 数据ID | 现有记忆 | 新信息 | 预期决策 |
|--------|----------|--------|----------|
| CP-GEN-005-A | PyTorch 安装：pip install torch | PyTorch 安装方法：pip install torch | TOUCH |
| CP-GEN-005-B | 用户偏好：VSCode + Vim | 用户喜欢用 VSCode 配合 Vim 插件 | TOUCH |

---

## 4. 数据设计规范

### 4.1 命名规则

**测试用例 ID 规范**:
```
{链路缩写}-{模块缩写}-{序号}-{变体}

示例:
CP-PER-001-A  # Cold Path - Perception - 001 - 变体A
CP-GEN-003-B  # Cold Path - Generation - 003 - 变体B
```

**测试数据文件命名**:
```
{模块}_{场景}_{版本}.json

示例:
perception_semantic_drift_v1.json
generation_deduplication_v1.json
```

### 4.2 版本管理

| 字段 | 说明 | 示例 |
|------|------|------|
| version | 数据集版本号 | "1.0.0" |
| created_at | 创建时间 | "2026-02-01T00:00:00Z" |
| updated_at | 最后更新时间 | "2026-02-01T00:00:00Z" |
| author | 作者 | "HiveMemory Team" |
| description | 数据集描述 | "冷链路 Perception 层测试数据" |

### 4.3 依赖脚本

**数据生成脚本**:
```bash
# 生成测试数据
python scripts/generate_test_data.py --module cold_path --output tests/fixtures/

# 验证数据格式
python scripts/validate_test_data.py --input tests/fixtures/cold_path_test_data.json
```

**数据加载脚本**:
```python
# tests/fixtures/loader.py
def load_cold_path_test_data() -> Dict[str, Any]:
    with open("tests/fixtures/cold_path_test_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

def get_test_case_by_id(test_id: str) -> Dict[str, Any]:
    data = load_cold_path_test_data()
    for case in data["test_cases"]:
        if case["id"] == test_id:
            return case
    raise ValueError(f"Test case not found: {test_id}")
```

### 4.4 清理策略

| 清理时机 | 清理内容 | 实现方式 |
|----------|----------|----------|
| 测试前 | 旧测试数据 | `setup_method` 中清理 |
| 测试后 | 本次测试数据 | `teardown_method` 中清理 |
| 测试失败 | 保留现场 | 配置 `--keep-failed-data` |
| CI 流水线 | 全量清理 | `pytest --clean-all` |

**清理 Fixture 示例**:
```python
@pytest.fixture(scope="function")
def clean_test_data(qdrant_client, test_identity):
    yield
    # 清理本次测试产生的数据
    qdrant_client.delete(
        collection_name="hivememory_test",
        points_selector=Filter(
            must=[FieldCondition(key="meta.user_id", match=MatchValue(value=test_identity.user_id))]
        )
    )
```

---

## 5. 断言与验收标准

### 5.1 功能断言

**Perception 层断言函数**:
```python
def assert_semantic_drift_detection(buffer, new_block, expected_flush: bool):
    """语义漂移检测断言"""
    flush_event = adsorber.should_adsorb(buffer, new_block)

    if expected_flush:
        assert flush_event is not None, "应触发 flush"
        assert flush_event.flush_reason == FlushReason.SEMANTIC_DRIFT
        assert len(flush_event.blocks_to_flush) > 0
    else:
        assert flush_event is None, "不应触发 flush"

def assert_token_overflow(buffer, new_block, max_tokens: int = 8192):
    """Token 溢出断言"""
    flush_event = relay_controller.should_relay(buffer, new_block)

    total_tokens = buffer.total_tokens + new_block.total_tokens
    if total_tokens > max_tokens:
        assert flush_event is not None
        assert flush_event.flush_reason == FlushReason.TOKEN_OVERFLOW
        assert flush_event.relay_summary is not None
    else:
        assert flush_event is None

def assert_triplet_parsing(block: LogicalBlock, expected_triplet_count: int):
    """Triplet 解析断言"""
    assert len(block.execution_chain) == expected_triplet_count
    for triplet in block.execution_chain:
        assert triplet.is_complete, "Triplet 应完整"
        assert triplet.tool_name is not None
        assert triplet.observation is not None
```

**Generation 层断言函数**:
```python
def assert_memory_extraction(memories: List[MemoryAtom], expected: Dict):
    """记忆提取断言"""
    if expected["has_value"]:
        assert len(memories) > 0, "应提取出记忆"
        memory = memories[0]

        # Schema 完整性
        assert memory.id is not None
        assert memory.meta.user_id is not None
        assert memory.index.title is not None
        assert memory.payload.content is not None

        # 类型匹配
        if "memory_type" in expected:
            assert memory.index.memory_type == expected["memory_type"]

        # 内容包含
        if "title_contains" in expected:
            for keyword in expected["title_contains"]:
                assert keyword in memory.index.title
    else:
        assert len(memories) == 0, "不应提取出记忆"

def assert_deduplication_decision(decision, expected_decision: DuplicateDecision):
    """去重决策断言"""
    assert decision == expected_decision, f"期望 {expected_decision}，实际 {decision}"
```

### 5.2 性能指标

| 指标 | 测量方法 | 采集点 |
|------|----------|--------|
| 单 Block 处理延迟 | `time.perf_counter()` | `perceive()` 入口到出口 |
| 记忆生成延迟 | `time.perf_counter()` | `process()` 入口到出口 |
| 向量计算延迟 | `time.perf_counter()` | `encode()` 调用 |
| 存储写入延迟 | `time.perf_counter()` | `upsert_memory()` 调用 |

**性能断言示例**:
```python
def assert_latency(start_time: float, max_latency_ms: float):
    """延迟断言"""
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    assert elapsed_ms < max_latency_ms, f"延迟 {elapsed_ms:.2f}ms 超过阈值 {max_latency_ms}ms"
```

### 5.3 量化验收标准

| 指标 | 计算方式 | 阈值 | 测试用例数 |
|------|----------|------|------------|
| 语义漂移检测准确率 | 正确判定数 / 总测试数 | >= 90% | >= 20 |
| Token 溢出触发准确率 | 正确触发数 / 应触发数 | 100% | >= 10 |
| 记忆提取召回率 | 正确提取数 / 应提取数 | >= 85% | >= 15 |
| 去重决策准确率 | 正确决策数 / 总决策数 | >= 95% | >= 20 |
| 端到端延迟 (P95) | 95 分位延迟 | < 5000ms | >= 100 |
| 内存泄漏 | 测试前后内存差 | 0 | 全量测试 |

---

## 6. 风险与回滚

### 6.1 风险识别

| 风险类型 | 风险描述 | 影响级别 | 发生概率 |
|----------|----------|----------|----------|
| 依赖服务不可用 | LLM API 超时或不可用 | 高 | 中 |
| 依赖服务不可用 | Qdrant 连接失败 | 高 | 低 |
| 依赖服务不可用 | Embedding 模型加载失败 | 高 | 低 |
| 数据污染 | 测试数据未清理干净 | 中 | 中 |
| 数据污染 | 并发测试数据串扰 | 中 | 低 |
| 性能衰退 | 延迟超过阈值 | 中 | 中 |
| 性能衰退 | 内存泄漏 | 高 | 低 |

### 6.2 监控阈值

| 监控项 | 警告阈值 | 严重阈值 | 采集频率 |
|--------|----------|----------|----------|
| 端到端延迟 (P95) | 4000ms | 5000ms | 每次测试 |
| 记忆提取成功率 | 90% | 85% | 每批次 |
| Qdrant 连接状态 | 重试 1 次 | 重试 3 次 | 每次操作 |
| 内存使用量 | 80% | 95% | 每分钟 |

### 6.3 告警通道

| 告警级别 | 通知方式 | 响应时间 |
|----------|----------|----------|
| 严重 (Critical) | 企业微信 + 邮件 | 5 分钟 |
| 警告 (Warning) | 企业微信 | 30 分钟 |
| 信息 (Info) | 日志记录 | 无 |

### 6.4 回滚步骤

**测试失败回滚**:
```bash
# 1. 停止当前测试
pytest --stop-on-first-failure

# 2. 清理测试数据
python scripts/cleanup_test_data.py --collection hivememory_test

# 3. 重置 Qdrant 集合
python scripts/reset_qdrant_collection.py --collection hivememory_test

# 4. 重新运行测试
pytest tests/pipelines/test_cold_path.py -v
```

**数据污染回滚**:
```python
def rollback_test_data(qdrant_client, test_session_id: str):
    """回滚测试数据"""
    # 删除本次测试产生的所有数据
    qdrant_client.delete(
        collection_name="hivememory_test",
        points_selector=Filter(
            must=[FieldCondition(key="meta.session_id", match=MatchValue(value=test_session_id))]
        )
    )
    logger.info(f"已回滚测试数据: session_id={test_session_id}")
```

---

## 7. 交付物清单

| 交付物 | 路径 | 说明 |
|--------|------|------|
| 测试设计文档 | `docs/tests/cold_path_e2e_test.md` | 本文档 |
| JSON 测试数据集 | `tests/fixtures/cold_path_test_data.json` | 结构化测试数据 |
| 测试代码 | `tests/pipelines/test_cold_path.py` | pytest 测试用例 |
| Mock 服务 | `tests/fixtures/mock_services.py` | LLM Mock 实现 |
| 数据加载器 | `tests/fixtures/loader.py` | 测试数据加载工具 |
| 清理脚本 | `scripts/cleanup_test_data.py` | 测试数据清理 |

---

## 附录 A: 测试数据 JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "version": { "type": "string" },
    "created_at": { "type": "string", "format": "date-time" },
    "test_cases": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string", "pattern": "^CP-(PER|GEN)-\\d{3}-[A-Z]$" },
          "name": { "type": "string" },
          "priority": { "type": "string", "enum": ["P0", "P1", "P2"] },
          "input": { "type": "object" },
          "expected": { "type": "object" }
        },
        "required": ["id", "name", "priority", "input", "expected"]
      }
    }
  },
  "required": ["version", "test_cases"]
}
```

---

## 附录 B: 相关文档链接

- [测试设计总览](test_design.md)
- [热链路测试文档](hot_path_e2e_test.md)
- [组件测试文档](../components/)
- [系统测试文档](../system/)
