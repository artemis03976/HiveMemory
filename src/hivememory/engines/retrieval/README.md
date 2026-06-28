# MemoryRetrieval - 记忆检索模块

## 📖 概述

MemoryRetrieval 模块负责智能检索相关记忆并注入到对话上下文中。
检索模块只返回记忆原子与检索元信息；Agent 可读上下文由 MemoryCompiler 负责编译。

对应设计文档: **PROJECT.md 第 5 章**

---

## ✅ 当前状态

**🎉 Stage 2 实现完成**

本模块已完成核心功能开发，包括：
- 查询预处理（时间解析、类型识别）
- 检索路由（规则 + LLM 两种模式）
- 混合检索（向量 + 元数据过滤）
- 检索结果产出（记忆原子 + 元信息）
- 统一检索引擎门面

---

## 🎯 核心组件

### 1. 查询预处理 (`query.py`)
- `QueryProcessor` - 查询预处理器
- `ProcessedQuery` - 结构化查询对象
- `TimeExpressionParser` - 时间表达式解析
- `MemoryTypeDetector` - 记忆类型识别

### 2. 检索路由 (`router.py`)
- `SimpleRouter` - 基于规则的路由器
- `LLMRouter` - 基于 LLM 的智能路由器

### 3. 混合检索 (`searcher.py`)
- `HybridSearcher` - 混合检索引擎
- `SearchResult` / `SearchResults` - 检索结果封装
- `CachedSearcher` - 带缓存的检索器

### 4. 统一引擎 (`engine.py`)
- `RetrievalEngine` - 统一检索入口
- `RetrievalResult` - 检索结果封装

---

## 🚀 快速使用

```python
from hivememory.engines.memory_compiler import MemoryCompiler, MemoryEnvelopeTarget
from hivememory.engines.retrieval import RetrievalEngine, RetrievalQuery, create_retriever
from hivememory.infrastructure.storage import QdrantMemoryStore

# 创建检索引擎
storage = QdrantMemoryStore()
retriever = create_retriever(mid_term=mid_term_store, config=retriever_config)
engine = RetrievalEngine(retriever=retriever)

# 检索记忆
result = await engine.retrieve(
    RetrievalQuery(semantic_query="我之前设置的 API Key 是什么？")
)

# 编译 Agent 可读上下文
if not result.is_empty():
    context = MemoryCompiler().compile(
        result.memories,
        MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
    )
    print(context.text)
```

---

## 📚 相关文档

- [PROJECT.md 第 5 章](../../docs/PROJECT.md) - 完整设计文档
- [ROADMAP.md Stage 2](../../docs/ROADMAP.md) - 开发路线图

---

**维护者**: HiveMemory Team  
**最后更新**: 2025-12-25  
**版本**: 0.2.0
