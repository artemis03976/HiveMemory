# 热链路 (Hot Path) 端到端测试设计文档

> **版本**: 1.0.0
> **作者**: HiveMemory Team
> **最后更新**: 2026-02-01

---

## 1. 测试范围与目标

### 1.1 被测链路

```
User Query -> Gateway (L1 Interceptor + L2 LLMAnalyzer)
           -> Retrieval (HybridRetriever + RRF Fusion + Reranker + Renderer)
           -> Context Output
```

**链路说明**: 热链路负责实时处理用户查询，通过意图识别、查询重写、混合检索和上下文渲染，为 LLM 提供相关的历史记忆上下文。

### 1.2 核心模块

| 模块 | 文件路径 | 核心职责 |
|------|----------|----------|
| GatewayEngine | `src/hivememory/engines/gateway/engine.py` | 协调 L1/L2 处理流程 |
| RuleInterceptor | `src/hivememory/engines/gateway/interceptors.py` | L1 正则拦截（系统指令、闲聊） |
| LLMAnalyzer | `src/hivememory/engines/gateway/semantic_analyzer.py` | L2 语义分析（意图、重写、关键词） |
| HybridRetriever | `src/hivememory/engines/retrieval/retriever.py` | 混合检索（稠密+稀疏） |
| ReciprocalRankFusion | `src/hivememory/engines/retrieval/fusion.py` | RRF 融合算法 |
| CrossEncoderReranker | `src/hivememory/engines/retrieval/reranker.py` | Cross-Encoder 重排序 |
| ContextRenderer | `src/hivememory/engines/retrieval/renderer.py` | 上下文渲染（XML/Markdown） |
| QdrantMemoryStore | `src/hivememory/infrastructure/storage/vector_store.py` | 向量存储与检索 |

### 1.3 测试目标与验收标准

| 目标类型 | 具体指标 | 验收标准 |
|----------|----------|----------|
| 功能正确性 | L1 拦截准确率 | 100% |
| 功能正确性 | 意图识别准确率 | >= 90% |
| 功能正确性 | 指代消解准确率 | >= 85% |
| 功能正确性 | 检索召回率 | >= 80% |
| 功能正确性 | Rerank Top-1 准确率 | >= 90% |
| 性能指标 | 端到端延迟 (P99) | < 500ms |
| 性能指标 | Gateway 延迟 (P95) | < 200ms |
| 性能指标 | Retrieval 延迟 (P95) | < 300ms |
| 稳定性 | 异常恢复成功率 | 100% |

---

## 2. 测试策略

### 2.1 Mock 策略

| 组件 | Mock 策略 | 理由 |
|------|-----------|------|
| Gateway LLM | **Mock** (预设响应) | 避免 API 成本，保证测试可重复性 |
| EmbeddingService | **真实调用** (本地 BGE-M3) | 向量检索是核心逻辑，必须真实验证 |
| QdrantMemoryStore | **真实调用** (Docker 容器) | 检索是最终出口，必须验证实际查询 |
| RerankService | **可配置** | P0 测试真实调用，P1 测试可 Mock |

**Mock 实现参考**:
```python
# 位置: tests/fixtures/mock_services.py
class MockGatewayLLM:
    def __init__(self, preset_responses: Dict[str, GatewayResult]):
        self.preset_responses = preset_responses

    def analyze(self, query: str, context: List) -> SemanticAnalysisResult:
        for key, response in self.preset_responses.items():
            if key in query:
                return response
        return self._default_rag_response(query)
```

### 2.2 数据隔离方案

| 隔离维度 | 策略 | 实现方式 |
|----------|------|----------|
| 用户隔离 | 每个测试用例使用唯一 user_id | `user_id = f"test_user_{uuid4()}"` |
| 存储隔离 | 使用独立的 Qdrant Collection | `collection_name = f"test_hot_{timestamp}"` |
| Golden Data | 预注入标准记忆库 | `GOLDEN_MEMORIES` 数据集 |
| 测试后清理 | fixture 自动清理 | `@pytest.fixture(scope="function")` |

**Golden Memories 预注入**:
```python
@pytest.fixture(scope="module")
def golden_memories(qdrant_store):
    """预注入标准记忆库"""
    from tests.fixtures.retrieval_test_data import GOLDEN_MEMORIES
    for memory_data in GOLDEN_MEMORIES:
        memory = build_memory_atom(memory_data)
        qdrant_store.upsert_memory(memory)
    yield
    # 清理
    for memory_data in GOLDEN_MEMORIES:
        qdrant_store.delete_memory(UUID(memory_data["id"]))
```

### 2.3 并发与性能边界

| 场景 | 并发级别 | 预期行为 | 测试方法 |
|------|----------|----------|----------|
| 单用户查询 | 串行处理 | 正常响应 | 顺序发送查询 |
| 多用户并发 | 并行处理 | 各查询独立，无竞争 | 多线程模拟 |
| 高负载压测 | 500 QPS | 延迟可控，无超时 | locust 压测脚本 |
| 服务降级 | LLM 超时 | Fallback 响应 | Mock 超时场景 |

---

## 3. 测试点拆解

### 3.1 Gateway 层测试点 (6 个)

#### HP-GW-001: L1 系统指令拦截

| 属性 | 值 |
|------|-----|
| **ID** | HP-GW-001 |
| **名称** | L1 系统指令拦截 |
| **优先级** | P0 |
| **描述** | 验证系统指令 (/clear, /reset) 被 L1 正则拦截，不调用 LLM |
| **前置条件** | 无 |
| **触发动作** | 发送系统指令查询 |
| **预期结果** | intent = SYSTEM, L1 命中, LLM 未调用 |

**输入-处理-输出-断言**:
```
输入:
  - query: "/clear"

处理:
  1. L1 RuleInterceptor 正则匹配 → 命中 SYSTEM_PATTERNS
  2. 直接返回 InterceptorResult(intent=SYSTEM)
  3. 不进入 L2 LLMAnalyzer

输出:
  - GatewayResult(intent=SYSTEM, l1_result.hit=True)

断言:
  - assert result.intent == GatewayIntent.SYSTEM
  - assert result.is_l1_intercepted == True
  - assert llm_mock.call_count == 0
```

**测试数据组**:
| 数据ID | 查询 | 预期 Intent | L1 命中 |
|--------|------|-------------|---------|
| HP-GW-001-A | /clear | SYSTEM | True |
| HP-GW-001-B | /reset | SYSTEM | True |
| HP-GW-001-C | /help | SYSTEM | True |

---

#### HP-GW-002: L1 闲聊拦截

| 属性 | 值 |
|------|-----|
| **ID** | HP-GW-002 |
| **名称** | L1 闲聊拦截 |
| **优先级** | P1 |
| **描述** | 验证简单问候语被 L1 拦截，识别为 CHAT 意图 |
| **前置条件** | 无 |
| **触发动作** | 发送问候语查询 |
| **预期结果** | intent = CHAT, L1 命中 |

**测试数据组**:
| 数据ID | 查询 | 预期 Intent | L1 命中 |
|--------|------|-------------|---------|
| HP-GW-002-A | 你好 | CHAT | True |
| HP-GW-002-B | hello | CHAT | True |
| HP-GW-002-C | 谢谢 | CHAT | True |
| HP-GW-002-D | hi | CHAT | True |

---

#### HP-GW-003: L2 RAG 意图识别

| 属性 | 值 |
|------|-----|
| **ID** | HP-GW-003 |
| **名称** | L2 RAG 意图识别 |
| **优先级** | P0 |
| **描述** | 验证技术问题被 L2 识别为 RAG 意图 |
| **前置条件** | L1 未拦截 |
| **触发动作** | 发送技术问题查询 |
| **预期结果** | intent = RAG, rewritten_query 包含关键实体 |

**输入-处理-输出-断言**:
```
输入:
  - query: "Rust 的所有权机制是什么？"

处理:
  1. L1 RuleInterceptor → 未命中
  2. L2 LLMAnalyzer → Function Calling
  3. 返回 SemanticAnalysisResult(intent=RAG, rewritten_query=...)

输出:
  - GatewayResult(intent=RAG, rewritten_query="Rust 所有权机制...")

断言:
  - assert result.intent == GatewayIntent.RAG
  - assert "Rust" in result.rewritten_query
  - assert "所有权" in result.rewritten_query
```

**测试数据组**:
| 数据ID | 查询 | 预期 Intent | rewritten 包含 |
|--------|------|-------------|----------------|
| HP-GW-003-A | Rust 的所有权机制是什么？ | RAG | Rust, 所有权 |
| HP-GW-003-B | 如何在 Docker 中配置网络？ | RAG | Docker, 网络 |
| HP-GW-003-C | Python 装饰器怎么用？ | RAG | Python, 装饰器 |

---

#### HP-GW-004: 指代消解

| 属性 | 值 |
|------|-----|
| **ID** | HP-GW-004 |
| **名称** | 指代消解 |
| **优先级** | P0 |
| **描述** | 验证代词"它"、"这个"被正确消解为上下文中的实体 |
| **前置条件** | 提供对话上下文 |
| **触发动作** | 发送包含代词的查询 |
| **预期结果** | rewritten_query 包含消解后的实体 |

**输入-处理-输出-断言**:
```
输入:
  - context: [{"role": "user", "content": "介绍下 Docker"}, ...]
  - query: "它怎么安装？"

处理:
  1. L2 LLMAnalyzer 分析上下文
  2. 识别"它"指代 Docker
  3. 重写查询

输出:
  - rewritten_query: "Docker 怎么安装？" 或 "如何安装 Docker？"

断言:
  - assert "Docker" in result.rewritten_query
  - assert "安装" in result.rewritten_query
```

**测试数据组**:
| 数据ID | 上下文主题 | 查询 | 预期消解实体 |
|--------|------------|------|--------------|
| HP-GW-004-A | Docker 介绍 | 它怎么安装？ | Docker |
| HP-GW-004-B | Python 装饰器 | 能给个例子吗？ | 装饰器 |
| HP-GW-004-C | 贪吃蛇游戏 | 怎么部署它？ | 贪吃蛇 |

---

#### HP-GW-005: 关键词提取

| 属性 | 值 |
|------|-----|
| **ID** | HP-GW-005 |
| **名称** | 关键词提取 |
| **优先级** | P1 |
| **描述** | 验证技术名词被正确提取为稀疏检索关键词 |
| **前置条件** | 无 |
| **触发动作** | 发送包含技术名词的查询 |
| **预期结果** | search_keywords 包含关键技术名词 |

**测试数据组**:
| 数据ID | 查询 | 预期关键词 (任一) |
|--------|------|-------------------|
| HP-GW-005-A | 如何在 FastAPI 中使用 Pydantic？ | FastAPI, Pydantic |
| HP-GW-005-B | 比较 React、Vue 和 Angular | React, Vue, Angular |
| HP-GW-005-C | TensorFlow 实现卷积神经网络 | TensorFlow, CNN |

---

#### HP-GW-006: Fallback 处理

| 属性 | 值 |
|------|-----|
| **ID** | HP-GW-006 |
| **名称** | Fallback 处理 |
| **优先级** | P1 |
| **描述** | 验证 LLM 解析失败时返回保守的默认值 |
| **前置条件** | Mock LLM 返回无效响应 |
| **触发动作** | 发送任意查询 |
| **预期结果** | 返回 Fallback 结果，gateway_parse_failed = True |

**测试数据组**:
| 数据ID | 场景 | 预期 Intent | parse_failed |
|--------|------|-------------|--------------|
| HP-GW-006-A | LLM 返回空响应 | CHAT | True |
| HP-GW-006-B | LLM 返回畸形 JSON | CHAT | True |
| HP-GW-006-C | LLM 超时 | CHAT | True |

---

### 3.2 Retrieval 层测试点 (6 个)

#### HP-RET-001: 纯语义召回

| 属性 | 值 |
|------|-----|
| **ID** | HP-RET-001 |
| **名称** | 纯语义召回 |
| **优先级** | P0 |
| **描述** | 验证语义相关但无关键词重叠的 Query 能召回相关记忆 |
| **前置条件** | Golden Memories 已注入（水果组） |
| **触发动作** | 发送语义相关查询 |
| **预期结果** | 召回语义相关的记忆，Top-K 包含预期 ID |

**输入-处理-输出-断言**:
```
输入:
  - query: "水果"
  - golden_memories: [苹果营养, 香蕉功效, 橙子维生素C]

处理:
  1. 向量编码 query
  2. 稠密检索 → 召回语义相关记忆
  3. RRF 融合 → 排序

输出:
  - SearchResults 包含水果相关记忆

断言:
  - assert len(results.results) >= 2
  - assert "550e8400-e29b-41d4-a716-446655440101" in recalled_ids  # 苹果
  - assert "550e8400-e29b-41d4-a716-446655440102" in recalled_ids  # 香蕉
```

**测试数据组**:
| 数据ID | 查询 | 预期召回 ID | 最小召回数 |
|--------|------|-------------|------------|
| HP-RET-001-A | 水果 | golden-fruit-001, 002, 003 | 2 |
| HP-RET-001-B | 健康饮食 | golden-fruit-001, 002 | 1 |
| HP-RET-001-C | 维生素补充 | golden-fruit-003 | 1 |

---

#### HP-RET-002: 纯关键词召回

| 属性 | 值 |
|------|-----|
| **ID** | HP-RET-002 |
| **名称** | 纯关键词召回 |
| **优先级** | P0 |
| **描述** | 验证包含特定专有名词的 Query 能精确匹配召回 |
| **前置条件** | Golden Memories 已注入（配置组） |
| **触发动作** | 发送包含专有名词的查询 |
| **预期结果** | 精确匹配的记忆排在 Top-1 |

**输入-处理-输出-断言**:
```
输入:
  - query: "X-1024 参数"
  - keywords: ["X-1024"]

处理:
  1. 稀疏检索 → 关键词匹配
  2. RRF 融合 → 精确匹配加权

输出:
  - Top-1 为 X-1024 服务器配置

断言:
  - assert results.results[0].memory.id == "550e8400-e29b-41d4-a716-446655440201"
```

**测试数据组**:
| 数据ID | 查询 | 关键词 | 预期 Top-1 ID |
|--------|------|--------|---------------|
| HP-RET-002-A | X-1024 参数 | X-1024 | golden-config-001 |
| HP-RET-002-B | X-1025 测试环境 | X-1025 | golden-config-002 |
| HP-RET-002-C | Python 快速排序 | Python, 排序 | golden-code-001 |

---

#### HP-RET-003: 混合冲突处理

| 属性 | 值 |
|------|-----|
| **ID** | HP-RET-003 |
| **名称** | 混合冲突处理 |
| **优先级** | P1 |
| **描述** | 验证歧义 Query（如"苹果"）通过 RRF 正确融合语义和关键词信号 |
| **前置条件** | Golden Memories 已注入（苹果公司 + 水果苹果） |
| **触发动作** | 发送歧义查询 |
| **预期结果** | 语义+关键词双重匹配的记忆排名更高 |

**输入-处理-输出-断言**:
```
输入:
  - query: "苹果公司的股价"

处理:
  1. 稠密检索 → 召回苹果相关记忆
  2. 稀疏检索 → 关键词"苹果公司"、"股价"匹配
  3. RRF 融合 → Apple Stock 双重命中，排名更高

输出:
  - Top-1: Apple Inc. 股票分析
  - Top-2: 苹果品种分类（仅关键词"苹果"命中）

断言:
  - assert results.results[0].memory.id == "550e8400-e29b-41d4-a716-446655440301"
  - assert rank_of(apple_stock) < rank_of(apple_fruit)
```

**测试数据组**:
| 数据ID | 查询 | 预期 Top-1 | 预期排序 |
|--------|------|------------|----------|
| HP-RET-003-A | 苹果公司的股价 | Apple Stock | Stock > Fruit |
| HP-RET-003-B | 苹果的营养价值 | 苹果营养 | Fruit > Stock |

---

#### HP-RET-004: Rerank 精排优化

| 属性 | 值 |
|------|-----|
| **ID** | HP-RET-004 |
| **名称** | Rerank 精排优化 |
| **优先级** | P0 |
| **描述** | 验证粗排 Top-1 并非最优时，Rerank 后正确重排序 |
| **前置条件** | Golden Memories 已注入 |
| **触发动作** | 发送需要精排的查询 |
| **预期结果** | Rerank 后最相关记忆排在 Top-1 |

**输入-处理-输出-断言**:
```
输入:
  - query: "Python 排序算法实现"
  - 粗排结果: [苹果营养(弱相关), Python快速排序(强相关), 地球结构(无关)]

处理:
  1. CrossEncoderReranker 计算 [query, passage] 对分数
  2. 重新排序

输出:
  - Rerank 后 Top-1: Python 快速排序

断言:
  - assert results.results[0].memory.id == "550e8400-e29b-41d4-a716-446655440501"
```

**测试数据组**:
| 数据ID | 查询 | 粗排 Top-1 | Rerank 后 Top-1 |
|--------|------|------------|-----------------|
| HP-RET-004-A | Python 排序算法实现 | 任意 | golden-code-001 |
| HP-RET-004-B | 用户开发环境配置 | 任意 | golden-pref-001 |

---

#### HP-RET-005: 阈值过滤

| 属性 | 值 |
|------|-----|
| **ID** | HP-RET-005 |
| **名称** | 阈值过滤 |
| **优先级** | P1 |
| **描述** | 验证无关 Query 经 Rerank 后返回空列表或低分结果 |
| **前置条件** | Golden Memories 已注入 |
| **触发动作** | 发送与所有记忆无关的查询 |
| **预期结果** | 所有结果分数低于阈值，或返回空列表 |

**测试数据组**:
| 数据ID | 查询 | 分数阈值 | 预期结果 |
|--------|------|----------|----------|
| HP-RET-005-A | 外星人入侵地球的电影推荐 | 0.5 | 空或全部低于阈值 |
| HP-RET-005-B | 量子纠缠的哲学意义 | 0.5 | 空或全部低于阈值 |

---

#### HP-RET-006: 渲染格式验证

| 属性 | 值 |
|------|-----|
| **ID** | HP-RET-006 |
| **名称** | 渲染格式验证 |
| **优先级** | P1 |
| **描述** | 验证 XML 和 Markdown 格式输出包含正确的标签结构 |
| **前置条件** | 检索返回非空结果 |
| **触发动作** | 指定渲染格式 |
| **预期结果** | 输出包含正确的格式标签 |

**测试数据组**:
| 数据ID | 格式 | 预期包含 | 预期不包含 |
|--------|------|----------|------------|
| HP-RET-006-A | XML | `<system_memory_context>`, `<memory_block>` | `## 相关记忆` |
| HP-RET-006-B | Markdown | `## 相关记忆上下文`, `###` | `<system_memory_context>` |
| HP-RET-006-C | Cascade XML | `<memory_block>`, 摘要视图 | - |

---

## 4. 数据设计规范

### 4.1 命名规则

**测试用例 ID 规范**:
```
{链路缩写}-{模块缩写}-{序号}-{变体}

示例:
HP-GW-001-A   # Hot Path - Gateway - 001 - 变体A
HP-RET-003-B  # Hot Path - Retrieval - 003 - 变体B
```

**Golden Memory ID 规范**:
```
550e8400-e29b-41d4-a716-446655440[XXX]

分类:
- 440101-440103: 水果组
- 440201-440202: 配置组
- 440301-440302: 混合冲突组
- 440401-440402: 无关数据组
- 440501: 代码片段组
- 440601: 用户偏好组
```

### 4.2 版本管理

| 字段 | 说明 | 示例 |
|------|------|------|
| version | 数据集版本号 | "1.0.0" |
| created_at | 创建时间 | "2026-02-01T00:00:00Z" |
| updated_at | 最后更新时间 | "2026-02-01T00:00:00Z" |
| golden_memories_version | Golden Memories 版本 | "1.0.0" |

### 4.3 依赖脚本

**Golden Memories 注入脚本**:
```bash
# 注入 Golden Memories
python scripts/inject_golden_memories.py --collection hivememory_test

# 验证注入结果
python scripts/verify_golden_memories.py --collection hivememory_test
```

**数据加载脚本**:
```python
# tests/fixtures/loader.py
def load_hot_path_test_data() -> Dict[str, Any]:
    with open("tests/fixtures/hot_path_test_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

def get_golden_memory_by_id(memory_id: str) -> Dict[str, Any]:
    from tests.fixtures.retrieval_test_data import GOLDEN_MEMORIES
    for memory in GOLDEN_MEMORIES:
        if memory["id"] == memory_id:
            return memory
    raise ValueError(f"Golden memory not found: {memory_id}")
```

### 4.4 清理策略

| 清理时机 | 清理内容 | 实现方式 |
|----------|----------|----------|
| 测试前 | 重置 Golden Memories | `setup_module` 中重新注入 |
| 测试后 | 无需清理 | Golden Memories 保留 |
| CI 流水线 | 全量重建 | 删除并重建 Collection |

---

## 5. 断言与验收标准

### 5.1 功能断言

**Gateway 层断言函数**:
```python
def assert_l1_interception(result: GatewayResult, expected_intent: GatewayIntent):
    """L1 拦截断言"""
    assert result.is_l1_intercepted == True, "应被 L1 拦截"
    assert result.intent == expected_intent
    assert result.l1_result is not None
    assert result.l1_result.hit == True

def assert_intent_classification(result: GatewayResult, expected_intent: GatewayIntent):
    """意图分类断言"""
    assert result.intent == expected_intent, f"期望 {expected_intent}，实际 {result.intent}"

def assert_coreference_resolution(result: GatewayResult, expected_entities: List[str]):
    """指代消解断言"""
    for entity in expected_entities:
        assert entity in result.rewritten_query, f"重写查询应包含 {entity}"

def assert_keyword_extraction(result: GatewayResult, expected_keywords: List[str]):
    """关键词提取断言"""
    extracted = set(result.search_keywords)
    expected = set(expected_keywords)
    assert len(extracted & expected) > 0, f"应提取出关键词: {expected_keywords}"
```

**Retrieval 层断言函数**:
```python
def assert_recall(results: SearchResults, expected_ids: List[str], min_count: int = 1):
    """召回断言"""
    recalled_ids = [str(r.memory.id) for r in results.results]
    matched = [eid for eid in expected_ids if eid in recalled_ids]
    assert len(matched) >= min_count, f"召回数 {len(matched)} < 最小要求 {min_count}"

def assert_top1(results: SearchResults, expected_id: str):
    """Top-1 断言"""
    assert len(results.results) > 0, "结果不应为空"
    assert str(results.results[0].memory.id) == expected_id, f"Top-1 应为 {expected_id}"

def assert_ranking_order(results: SearchResults, higher_id: str, lower_id: str):
    """排序顺序断言"""
    ids = [str(r.memory.id) for r in results.results]
    assert higher_id in ids and lower_id in ids, "两个 ID 都应在结果中"
    assert ids.index(higher_id) < ids.index(lower_id), f"{higher_id} 应排在 {lower_id} 之前"

def assert_render_format(rendered: str, format_type: str):
    """渲染格式断言"""
    if format_type == "xml":
        assert "<system_memory_context>" in rendered
        assert "</system_memory_context>" in rendered
    elif format_type == "markdown":
        assert "## 相关记忆上下文" in rendered
```

### 5.2 性能指标

| 指标 | 测量方法 | 采集点 |
|------|----------|--------|
| Gateway 延迟 | `time.perf_counter()` | `process()` 入口到出口 |
| Retrieval 延迟 | `time.perf_counter()` | `retrieve()` 入口到出口 |
| 端到端延迟 | `time.perf_counter()` | 完整链路 |
| 向量检索延迟 | Qdrant 返回的 `latency_ms` | 检索结果 |

**性能断言示例**:
```python
def assert_latency(start_time: float, max_latency_ms: float, component: str):
    """延迟断言"""
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    assert elapsed_ms < max_latency_ms, f"{component} 延迟 {elapsed_ms:.2f}ms 超过阈值 {max_latency_ms}ms"
```

### 5.3 量化验收标准

| 指标 | 计算方式 | 阈值 | 测试用例数 |
|------|----------|------|------------|
| L1 拦截准确率 | 正确拦截数 / 应拦截数 | 100% | >= 10 |
| 意图识别准确率 | 正确识别数 / 总测试数 | >= 90% | >= 20 |
| 指代消解准确率 | 正确消解数 / 总测试数 | >= 85% | >= 10 |
| 检索召回率 | 召回预期数 / 应召回数 | >= 80% | >= 15 |
| Rerank Top-1 准确率 | Top-1 正确数 / 总测试数 | >= 90% | >= 10 |
| 端到端延迟 (P99) | 99 分位延迟 | < 500ms | >= 100 |

---

## 6. 风险与回滚

### 6.1 风险识别

| 风险类型 | 风险描述 | 影响级别 | 发生概率 |
|----------|----------|----------|----------|
| 依赖服务不可用 | Gateway LLM 超时或不可用 | 高 | 中 |
| 依赖服务不可用 | Qdrant 连接失败 | 高 | 低 |
| 依赖服务不可用 | Reranker 模型加载失败 | 中 | 低 |
| 数据污染 | Golden Memories 被意外修改 | 高 | 低 |
| 性能衰退 | 检索延迟超过阈值 | 中 | 中 |
| 性能衰退 | 并发下响应超时 | 高 | 低 |

### 6.2 监控阈值

| 监控项 | 警告阈值 | 严重阈值 | 采集频率 |
|--------|----------|----------|----------|
| 端到端延迟 (P99) | 400ms | 500ms | 每次测试 |
| Gateway 延迟 (P95) | 150ms | 200ms | 每次测试 |
| Retrieval 延迟 (P95) | 250ms | 300ms | 每次测试 |
| 检索召回率 | 85% | 80% | 每批次 |
| Qdrant 连接状态 | 重试 1 次 | 重试 3 次 | 每次操作 |

### 6.3 告警通道

| 告警级别 | 通知方式 | 响应时间 |
|----------|----------|----------|
| 严重 (Critical) | 企业微信 + 邮件 | 5 分钟 |
| 警告 (Warning) | 企业微信 | 30 分钟 |
| 信息 (Info) | 日志记录 | 无 |

### 6.4 回滚步骤

**Golden Memories 损坏回滚**:
```bash
# 1. 删除当前 Collection
python scripts/delete_collection.py --collection hivememory_test

# 2. 重建 Collection
python scripts/create_collection.py --collection hivememory_test

# 3. 重新注入 Golden Memories
python scripts/inject_golden_memories.py --collection hivememory_test

# 4. 验证注入结果
python scripts/verify_golden_memories.py --collection hivememory_test
```

**服务降级处理**:
```python
def fallback_gateway_result(query: str) -> GatewayResult:
    """Gateway 服务降级 Fallback"""
    return GatewayResult(
        intent=GatewayIntent.CHAT,
        rewritten_query=query,
        search_keywords=[],
        target_filters=QueryFilters(),
        worth_saving=False,
        reason="Fallback due to service unavailability",
        gateway_parse_failed=True,
    )
```

---

## 7. 交付物清单

| 交付物 | 路径 | 说明 |
|--------|------|------|
| 测试设计文档 | `docs/tests/hot_path_e2e_test.md` | 本文档 |
| JSON 测试数据集 | `tests/fixtures/hot_path_test_data.json` | 结构化测试数据 |
| Golden Memories | `tests/fixtures/retrieval_test_data.py` | 预注入记忆库 |
| 测试代码 | `tests/pipelines/test_hot_path.py` | pytest 测试用例 |
| Mock 服务 | `tests/fixtures/mock_services.py` | Gateway LLM Mock |
| 注入脚本 | `scripts/inject_golden_memories.py` | Golden Memories 注入 |

---

## 附录 A: 测试数据 JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "version": { "type": "string" },
    "created_at": { "type": "string", "format": "date-time" },
    "golden_memories_version": { "type": "string" },
    "test_cases": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string", "pattern": "^HP-(GW|RET)-\\d{3}-[A-Z]$" },
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
- [冷链路测试文档](cold_path_e2e_test.md)
- [组件测试文档](../components/)
- [系统测试文档](../system/)
