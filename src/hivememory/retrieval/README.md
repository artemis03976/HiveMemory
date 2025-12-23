# MemoryRetrieval - 记忆检索模块

## 📖 概述

MemoryRetrieval 模块负责智能检索相关记忆并注入到对话上下文中。

对应设计文档: **PROJECT.md 第 5 章**

---

## ⚠️ 当前状态

**🚧 骨架接口 - 待 Stage 2 实现**

本模块目前仅包含接口定义，核心功能将在 Stage 2 开发中完成。

---

## 🎯 核心职责 (计划)

1. **检索路由 (Router)** - 判断查询是否需要记忆
2. **查询预处理** - Query Rewriting，补全上下文
3. **混合检索** - 向量 + BM25 + 结构化过滤
4. **重排序 (Reranking)** - Cross-Encoder 精排
5. **上下文注入** - 渲染为 Markdown 供 LLM 使用
6. **权限控制** - 基于 Visibility 过滤

---

## 📦 预定义接口

### `interfaces.py`

```python
from abc import ABC, abstractmethod

class RetrievalRouter(ABC):
    """检索路由器 - 判断是否需要记忆"""
    @abstractmethod
    def should_retrieve(self, query: str, context: List[Message]) -> bool:
        pass

class QueryProcessor(ABC):
    """查询预处理器 - 重写和扩展查询"""
    @abstractmethod
    def process(self, query: str) -> ProcessedQuery:
        pass

class MemorySearcher(ABC):
    """记忆检索器 - 混合检索"""
    @abstractmethod
    def search(self, query: ProcessedQuery) -> List[MemoryAtom]:
        pass
```

---

## 🛣️ 开发计划

**Stage 2 任务清单**:
- [ ] 实现 RetrievalRouter (轻量级分类器)
- [ ] 实现 QueryProcessor (Query Rewriting)
- [ ] 实现 HybridSearcher (Vector + BM25 + Filters)
- [ ] 实现 Reranker (Cross-Encoder)
- [ ] 实现 ContextRenderer (Markdown 渲染)
- [ ] 集成权限控制 (Visibility Scopes)

---

## 📚 相关文档

- [PROJECT.md 第 5 章](../../docs/PROJECT.md) - 完整设计文档
- [ROADMAP.md Stage 2](../../docs/ROADMAP.md) - 开发路线图

---

**维护者**: HiveMemory Team
**最后更新**: 2025-12-23
**版本**: 0.1.0 (骨架)
