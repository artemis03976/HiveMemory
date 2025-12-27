# MemoryRetrieval - 记忆检索模块

## 📖 概述

MemoryRetrieval 模块负责智能检索相关记忆并注入到对话上下文中。

对应设计文档: **PROJECT.md 第 5 章**

---

## ✅ 当前状态

**🎉 Stage 2 实现完成**

本模块已完成核心功能开发，包括：
- 查询预处理（时间解析、类型识别）
- 检索路由（规则 + LLM 两种模式）
- 混合检索（向量 + 元数据过滤）
- 上下文渲染（XML / Markdown 格式）
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

### 4. 上下文渲染 (`renderer.py`)
- `ContextRenderer` - 上下文渲染器（XML/Markdown）
- `MinimalRenderer` - 极简渲染器

### 5. 统一引擎 (`engine.py`)
- `RetrievalEngine` - 统一检索入口
- `RetrievalResult` - 检索结果封装

---

## 🚀 快速使用

```python
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.retrieval import create_retrieval_engine

# 创建检索引擎
storage = QdrantMemoryStore()
engine = create_retrieval_engine(storage)

# 检索记忆
result = engine.retrieve_context(
    query="我之前设置的 API Key 是什么？",
    user_id="user_123"
)

# 获取渲染后的上下文
if not result.is_empty():
    print(result.rendered_context)
```

---

## 📚 相关文档

- [PROJECT.md 第 5 章](../../docs/PROJECT.md) - 完整设计文档
- [ROADMAP.md Stage 2](../../docs/ROADMAP.md) - 开发路线图

---

**维护者**: HiveMemory Team  
**最后更新**: 2025-12-25  
**版本**: 0.2.0
